"""
Sector Scout Base Class
=======================
Uses Ollama (local, free) for continuous pattern detection.
TA-Lib for technical analysis, Ollama for interpretation.
"""

import asyncio
import json
import re
import httpx
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional
from collections import deque
import logging

# TA-Lib for pattern detection
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SectorScout")


class SectorScout:
    """
    Base class for sector-scanning scouts.
    Uses Ollama (local, free) for continuous pattern detection.
    """
    
    def __init__(
        self,
        name: str,
        sector: str,
        watchlist: List[str],
        ollama_model: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434",
        confidence_threshold: float = 75.0,
        scan_interval: int = 300,  # 5 minutes
        long_only: bool = False,
    ):
        self.name = name
        self.sector = sector
        self.watchlist = watchlist
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.confidence_threshold = confidence_threshold
        self.scan_interval = scan_interval
        self.long_only = long_only
        
        # State
        self.running = False
        self.alerts: deque = deque(maxlen=100)
        self.last_alerts: Dict[str, datetime] = {}  # Cooldown tracking
        self.alert_cooldown = timedelta(hours=4)  # Don't re-alert same symbol
        
        self.stats = {
            'scans': 0,
            'symbols_checked': 0,
            'patterns_found': 0,
            'alerts_sent': 0,
            'last_scan': None,
            'errors': 0,
        }
        
        # Market hours (EST) - Generous window
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.premarket_open = time(4, 0)
        self.afterhours_close = time(20, 0)
    
    def is_market_hours(self, include_extended: bool = True) -> bool:
        """Check if we're in market hours."""
        now = datetime.now()
        current_time = now.time()
        
        # Skip weekends
        if now.weekday() >= 5:
            return False
        
        if include_extended:
            return self.premarket_open <= current_time <= self.afterhours_close
        return self.market_open <= current_time <= self.market_close
    
    async def start(self):
        """Start the scout - runs during market hours."""
        
        self.running = True
        logger.info(f"[{self.name}] Scout activated for {self.sector}")
        logger.info(f"[{self.name}] Watching: {', '.join(self.watchlist[:5])}...")
        
        while self.running:
            try:
                if self.is_market_hours():
                    await self.scan_sector()
                else:
                    # Check every 5 minutes if market opened
                    logger.debug(f"[{self.name}] Market closed. Waiting...")
                
                await asyncio.sleep(self.scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.name}] Error in main loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(60)
    
    async def scan_sector(self):
        """Scan all symbols in watchlist."""
        
        self.stats['scans'] += 1
        self.stats['last_scan'] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Starting sector scan ({len(self.watchlist)} symbols)...")
        
        for symbol in self.watchlist:
            try:
                self.stats['symbols_checked'] += 1
                
                # Check cooldown
                if not self.can_alert(symbol):
                    continue
                
                result = await self.analyze_symbol(symbol)
                
                if result and result.get('confidence', 0) >= self.confidence_threshold:
                    # Apply long_only filter
                    if self.long_only and result.get('direction') != 'LONG':
                        logger.debug(f"[{self.name}] {symbol}: Skipping {result.get('direction')} (long_only mode)")
                        continue
                    
                    self.stats['patterns_found'] += 1
                    await self.send_alert(symbol, result)
                    
            except Exception as e:
                logger.error(f"[{self.name}] Error analyzing {symbol}: {e}")
                self.stats['errors'] += 1
            
            # Small delay between symbols to be nice to yfinance
            await asyncio.sleep(2)
        
        logger.info(f"[{self.name}] Scan complete. Patterns: {self.stats['patterns_found']}, Alerts: {self.stats['alerts_sent']}")
    
    def can_alert(self, symbol: str) -> bool:
        """Check if we can alert on this symbol (cooldown check)."""
        last = self.last_alerts.get(symbol)
        if last is None:
            return True
        return datetime.now() - last > self.alert_cooldown
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyze a single symbol for patterns.
        Uses TA-Lib for detection + Ollama for interpretation.
        """
        
        # Fetch data
        df = await self.fetch_data(symbol)
        if df is None or len(df) < 50:
            return None
        
        # Technical analysis with TA-Lib
        analysis = self.technical_analysis(df)
        
        # Only proceed if we found something interesting
        if not analysis['has_signal']:
            return None
        
        logger.info(f"[{self.name}] {symbol}: Signal detected - {len(analysis['patterns'])} patterns, {len(analysis['divergences'])} divergences")
        
        # Get Ollama interpretation
        interpretation = await self.ollama_interpret(symbol, df, analysis)
        
        return interpretation
    
    async def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent price data."""
        
        try:
            ticker = yf.Ticker(symbol)
            # Get 5m data for intraday analysis
            df = ticker.history(period="5d", interval="5m")
            
            if df.empty:
                # Try daily as fallback
                df = ticker.history(period="3mo", interval="1d")
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            logger.debug(f"[{self.name}] Fetch error for {symbol}: {e}")
            return None
    
    def technical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run TA-Lib indicators and pattern detection.
        Returns signals and context for Ollama.
        """
        
        open_p = df['open'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        analysis = {
            'has_signal': False,
            'patterns': [],
            'indicators': {},
            'divergences': [],
            'support_resistance': {},
        }
        
        if not TALIB_AVAILABLE:
            # Fallback: basic analysis
            analysis['indicators']['price'] = float(close[-1])
            analysis['indicators']['change_5'] = float((close[-1] / close[-5] - 1) * 100)
            analysis['has_signal'] = True  # Always pass to Ollama
            return analysis
        
        # === CANDLESTICK PATTERNS ===
        pattern_funcs = [
            ('CDL3WHITESOLDIERS', 'Three White Soldiers', 'BULLISH', 'CRITICAL'),
            ('CDL3BLACKCROWS', 'Three Black Crows', 'BEARISH', 'CRITICAL'),
            ('CDLHAMMER', 'Hammer', 'BULLISH', 'HIGH'),
            ('CDLSHOOTINGSTAR', 'Shooting Star', 'BEARISH', 'HIGH'),
            ('CDLENGULFING', 'Engulfing', 'BOTH', 'HIGH'),
            ('CDLMORNINGSTAR', 'Morning Star', 'BULLISH', 'HIGH'),
            ('CDLEVENINGSTAR', 'Evening Star', 'BEARISH', 'HIGH'),
            ('CDLPIERCING', 'Piercing Line', 'BULLISH', 'HIGH'),
            ('CDLDARKCLOUDCOVER', 'Dark Cloud Cover', 'BEARISH', 'HIGH'),
            ('CDLHARAMI', 'Harami', 'BOTH', 'MEDIUM'),
            ('CDLDOJI', 'Doji', 'NEUTRAL', 'LOW'),
            ('CDLDRAGONFLYDOJI', 'Dragonfly Doji', 'BULLISH', 'MEDIUM'),
            ('CDLGRAVESTONEDOJI', 'Gravestone Doji', 'BEARISH', 'MEDIUM'),
        ]
        
        for func_name, pattern_name, direction, severity in pattern_funcs:
            try:
                func = getattr(talib, func_name)
                result = func(open_p, high, low, close)
                if result[-1] != 0:
                    actual_dir = direction
                    if direction == 'BOTH':
                        actual_dir = 'BULLISH' if result[-1] > 0 else 'BEARISH'
                    analysis['patterns'].append({
                        'name': pattern_name,
                        'direction': actual_dir,
                        'severity': severity,
                        'signal': int(result[-1]),
                    })
            except Exception:
                pass
        
        # === INDICATORS ===
        try:
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            analysis['indicators']['rsi'] = round(float(rsi[-1]), 1)
            analysis['indicators']['rsi_5_ago'] = round(float(rsi[-5]), 1) if len(rsi) > 5 else None
            
            # MACD
            macd, signal, hist = talib.MACD(close)
            analysis['indicators']['macd'] = round(float(macd[-1]), 4)
            analysis['indicators']['macd_signal'] = round(float(signal[-1]), 4)
            analysis['indicators']['macd_hist'] = round(float(hist[-1]), 4)
            analysis['indicators']['macd_hist_5_ago'] = round(float(hist[-5]), 4) if len(hist) > 5 else None
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close)
            analysis['indicators']['bb_upper'] = round(float(upper[-1]), 2)
            analysis['indicators']['bb_middle'] = round(float(middle[-1]), 2)
            analysis['indicators']['bb_lower'] = round(float(lower[-1]), 2)
            analysis['indicators']['price'] = round(float(close[-1]), 2)
            
            # Price position relative to bands
            bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) * 100
            analysis['indicators']['bb_position'] = round(float(bb_position), 1)
            
            # Volume analysis
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            analysis['indicators']['volume_ratio'] = round(float(current_volume / avg_volume), 2) if avg_volume > 0 else 1.0
            
            # ATR for volatility
            atr = talib.ATR(high, low, close, timeperiod=14)
            analysis['indicators']['atr'] = round(float(atr[-1]), 2)
            analysis['indicators']['atr_pct'] = round(float(atr[-1] / close[-1] * 100), 2)
            
            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            analysis['indicators']['sma_20'] = round(float(sma_20[-1]), 2)
            analysis['indicators']['sma_50'] = round(float(sma_50[-1]), 2) if len(sma_50) > 50 else None
            analysis['indicators']['above_sma_20'] = close[-1] > sma_20[-1]
            
        except Exception as e:
            logger.debug(f"Indicator error: {e}")
        
        # === DIVERGENCE DETECTION ===
        try:
            if len(close) >= 10 and len(rsi) >= 10:
                # Price making lower lows but RSI making higher lows = BULLISH divergence
                price_ll = close[-1] < close[-5] and close[-5] < close[-10]
                rsi_hl = rsi[-1] > rsi[-5] and rsi[-5] > rsi[-10]
                
                if price_ll and rsi_hl:
                    analysis['divergences'].append({
                        'type': 'BULLISH_RSI_DIVERGENCE',
                        'description': 'Price lower lows, RSI higher lows',
                        'direction': 'BULLISH',
                    })
                
                # Price making higher highs but RSI making lower highs = BEARISH divergence
                price_hh = close[-1] > close[-5] and close[-5] > close[-10]
                rsi_lh = rsi[-1] < rsi[-5] and rsi[-5] < rsi[-10]
                
                if price_hh and rsi_lh:
                    analysis['divergences'].append({
                        'type': 'BEARISH_RSI_DIVERGENCE',
                        'description': 'Price higher highs, RSI lower highs',
                        'direction': 'BEARISH',
                    })
                
                # MACD divergence
                if len(hist) >= 10:
                    macd_bullish_div = close[-1] < close[-10] and hist[-1] > hist[-10]
                    macd_bearish_div = close[-1] > close[-10] and hist[-1] < hist[-10]
                    
                    if macd_bullish_div:
                        analysis['divergences'].append({
                            'type': 'BULLISH_MACD_DIVERGENCE',
                            'description': 'Price down, MACD histogram rising',
                            'direction': 'BULLISH',
                        })
                    
                    if macd_bearish_div:
                        analysis['divergences'].append({
                            'type': 'BEARISH_MACD_DIVERGENCE',
                            'description': 'Price up, MACD histogram falling',
                            'direction': 'BEARISH',
                        })
                        
        except Exception:
            pass
        
        # === DETERMINE IF WE HAVE A SIGNAL ===
        rsi_val = analysis['indicators'].get('rsi', 50)
        vol_ratio = analysis['indicators'].get('volume_ratio', 1)
        bb_pos = analysis['indicators'].get('bb_position', 50)
        
        analysis['has_signal'] = (
            len(analysis['patterns']) > 0 or
            len(analysis['divergences']) > 0 or
            rsi_val < 30 or rsi_val > 70 or  # Oversold/overbought
            vol_ratio > 2.5 or  # Volume spike
            bb_pos < 5 or bb_pos > 95  # Near band extremes
        )
        
        return analysis
    
    async def ollama_interpret(
        self,
        symbol: str,
        df: pd.DataFrame,
        analysis: Dict
    ) -> Optional[Dict]:
        """
        Use Ollama to interpret the technical analysis.
        Returns confidence score and recommendation.
        """
        
        price = df['close'].iloc[-1]
        
        # Calculate price changes
        price_1d = ((price - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100 if len(df) > 1 else 0
        price_5d = ((price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100 if len(df) > 5 else 0
        
        # Format patterns
        patterns_str = "\n".join([
            f"  - {p['name']} ({p['direction']}, {p['severity']})"
            for p in analysis['patterns']
        ]) if analysis['patterns'] else "  None detected"
        
        # Format divergences
        divergences_str = "\n".join([
            f"  - {d['type']}: {d['description']}"
            for d in analysis['divergences']
        ]) if analysis['divergences'] else "  None detected"
        
        # Format indicators
        ind = analysis['indicators']
        
        prompt = f"""You are a technical analysis expert for {self.sector} stocks. Analyze this setup.

SYMBOL: {symbol}
SECTOR: {self.sector}
PRICE: ${price:.2f}
5-DAY CHANGE: {price_5d:.1f}%

=== CANDLESTICK PATTERNS ===
{patterns_str}

=== DIVERGENCES ===
{divergences_str}

=== INDICATORS ===
RSI(14): {ind.get('rsi', 'N/A')}
MACD Histogram: {ind.get('macd_hist', 'N/A')}
Bollinger Position: {ind.get('bb_position', 'N/A')}% (0=lower band, 100=upper band)
Volume Ratio: {ind.get('volume_ratio', 'N/A')}x average
ATR%: {ind.get('atr_pct', 'N/A')}%
Above SMA20: {ind.get('above_sma_20', 'N/A')}

=== YOUR TASK ===
1. Is this a high-probability setup? (Give confidence 0-100)
2. Direction: LONG, SHORT, or WAIT
3. Entry price (current or better level)
4. Stop loss level (based on ATR/support)
5. Two take-profit targets
6. Brief reasoning (1-2 sentences)

IMPORTANT: Only give 75%+ confidence if MULTIPLE signals align (pattern + divergence + indicator).
Be conservative. Wrong signals cost money.

Respond ONLY in this JSON format:
{{"confidence": 82, "direction": "LONG", "entry": {price:.2f}, "stop": {price*0.95:.2f}, "target1": {price*1.10:.2f}, "target2": {price*1.20:.2f}, "reasoning": "Hammer at support + RSI divergence + volume spike"}}"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # More deterministic
                            "num_predict": 200,
                        }
                    },
                    timeout=45,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('response', '')
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            interpretation = json.loads(json_match.group())
                            
                            # Add metadata
                            interpretation['symbol'] = symbol
                            interpretation['sector'] = self.sector
                            interpretation['scout'] = self.name
                            interpretation['price'] = round(price, 2)
                            interpretation['analysis'] = {
                                'patterns': analysis['patterns'],
                                'divergences': analysis['divergences'],
                                'rsi': ind.get('rsi'),
                                'volume_ratio': ind.get('volume_ratio'),
                            }
                            interpretation['timestamp'] = datetime.now().isoformat()
                            
                            return interpretation
                        except json.JSONDecodeError:
                            logger.warning(f"[{self.name}] Could not parse Ollama JSON for {symbol}")
                else:
                    logger.warning(f"[{self.name}] Ollama returned {response.status_code}")
                        
        except httpx.TimeoutException:
            logger.warning(f"[{self.name}] Ollama timeout for {symbol}")
        except Exception as e:
            logger.error(f"[{self.name}] Ollama error: {e}")
        
        return None
    
    async def send_alert(self, symbol: str, result: Dict):
        """Send alert to War Room, log, and AUTO-EXECUTE on IB."""
        
        self.stats['alerts_sent'] += 1
        self.alerts.append(result)
        self.last_alerts[symbol] = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.name}] 🎯 ALERT: {symbol}")
        logger.info(f"  Confidence: {result.get('confidence', 0)}%")
        logger.info(f"  Direction: {result.get('direction', 'N/A')}")
        logger.info(f"  Entry: ${result.get('entry', 0):.2f}")
        logger.info(f"  Stop: ${result.get('stop', 0):.2f}")
        logger.info(f"  Targets: ${result.get('target1', 0):.2f} / ${result.get('target2', 0):.2f}")
        logger.info(f"  Reason: {result.get('reasoning', 'N/A')}")
        logger.info(f"{'='*60}\n")
        
        async with httpx.AsyncClient() as client:
            # Send to API for storage and War Room
            try:
                await client.post(
                    "http://localhost:8890/scouts/alert",
                    json=result,
                    timeout=10,
                )
            except Exception as e:
                logger.debug(f"Could not send to scouts API: {e}")
            
            # AUTO-EXECUTE on IB (Paper Mode)
            try:
                exec_response = await client.post(
                    "http://localhost:8890/ib/execute/scout-alert",
                    json=result,
                    timeout=15,
                )
                
                if exec_response.status_code == 200:
                    exec_result = exec_response.json()
                    status = exec_result.get('status', 'unknown')
                    
                    if status == 'submitted':
                        logger.info(f"[{self.name}] 🎯 TRADE EXECUTED: {exec_result.get('action')} {exec_result.get('quantity')} {symbol}")
                    elif status == 'not_connected':
                        logger.debug(f"[{self.name}] IB not connected - trade not executed")
                    else:
                        logger.info(f"[{self.name}] IB response: {status} - {exec_result.get('message', '')}")
                        
            except Exception as e:
                logger.debug(f"[{self.name}] IB execution skipped: {e}")
    
    def stop(self):
        """Stop the scout."""
        self.running = False
        logger.info(f"[{self.name}] Scout deactivated")
    
    def get_stats(self) -> Dict:
        """Get scout statistics."""
        return {
            'name': self.name,
            'sector': self.sector,
            'running': self.running,
            'symbols': len(self.watchlist),
            'long_only': self.long_only,
            **self.stats,
        }

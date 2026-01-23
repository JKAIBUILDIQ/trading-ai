#!/usr/bin/env python3
"""
NEO Market Feed - Real Price Data + Technical Analysis
NO RANDOM DATA - All prices from real sources.

Data Sources:
- MT5 API (localhost:8085) for forex + OHLCV history
- CoinGecko for crypto
- Frankfurter for forex rates
- Twelve Data for historical OHLCV (backup)

Technical Indicators Calculated:
- RSI(2), RSI(14)
- SMA(50), SMA(200)
- Recent High/Low (20 candles)
- Trend Direction
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import json
import os

from config import (
    MT5_API_URL, COINGECKO_URL, FRANKFURTER_URL,
    FOREX_PAIRS, CRYPTO_PAIRS, USER_AGENT, PROVEN_PARAMETERS
)

# Twelve Data API for OHLCV history (free tier: 800 calls/day)
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "a2542c3955c5417d99226668f7709301")
TWELVE_DATA_URL = "https://api.twelvedata.com"


@dataclass
class OHLCV:
    """Single candlestick - REAL data only."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


@dataclass 
class TechnicalIndicators:
    """Technical analysis for a symbol - calculated from REAL OHLCV."""
    symbol: str
    
    # RSI Values
    rsi_2: Optional[float] = None   # Connors RSI(2) - key for strategy!
    rsi_14: Optional[float] = None  # Standard RSI
    
    # Moving Averages
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma200: Optional[str] = None  # "ABOVE" or "BELOW"
    
    # Recent Range
    high_20: Optional[float] = None  # 20-period high
    low_20: Optional[float] = None   # 20-period low
    
    # Trend
    trend: Optional[str] = None  # "BULLISH", "BEARISH", "RANGING"
    trend_strength: Optional[float] = None  # 0-100
    
    # Key Levels
    resistance: Optional[float] = None
    support: Optional[float] = None
    
    # Data quality
    candles_used: int = 0
    data_source: str = "UNKNOWN"


@dataclass
class PriceData:
    """Real price data point - NO random values."""
    symbol: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    timestamp: str
    source: str  # MT5, COINGECKO, FRANKFURTER
    change_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    timestamp: str
    forex: Dict[str, PriceData]
    crypto: Dict[str, PriceData]
    technicals: Dict[str, TechnicalIndicators] = field(default_factory=dict)
    mt5_connected: bool = False
    data_sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MarketFeed:
    """
    Fetches REAL market data from multiple sources.
    Falls back gracefully if sources unavailable.
    NEVER generates random/fake prices.
    
    Now includes TECHNICAL ANALYSIS:
    - RSI(2), RSI(14)
    - SMA(50), SMA(200)
    - Support/Resistance levels
    - Trend direction
    """
    
    def __init__(self):
        self.headers = {"User-Agent": USER_AGENT}
        self.last_prices: Dict[str, PriceData] = {}
        self.ohlcv_cache: Dict[str, List[OHLCV]] = {}  # Symbol -> candles
        self.technicals_cache: Dict[str, TechnicalIndicators] = {}
        self.errors: List[str] = []
        self.last_ohlcv_fetch: Dict[str, datetime] = {}
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI from close prices. NO RANDOM DATA."""
        if len(closes) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smoothed averages
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def _calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return round(sum(prices[-period:]) / period, 5)
    
    def _fetch_ohlcv_yahoo(self, symbol: str, period: str = "5d", interval: str = "1h") -> List[OHLCV]:
        """Fetch OHLCV from Yahoo Finance (FREE, no API key needed)."""
        candles = []
        
        # Map forex symbols to Yahoo format
        yahoo_map = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "XAUUSD": "GC=F",  # Gold futures
            "NZDUSD": "NZDUSD=X"
        }
        
        yahoo_symbol = yahoo_map.get(symbol)
        if not yahoo_symbol:
            return candles
        
        try:
            # Use yfinance library if available
            import yfinance as yf
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)
            
            for idx, row in df.iterrows():
                candles.append(OHLCV(
                    timestamp=idx.isoformat(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row.get('Volume', 0))
                ))
        except ImportError:
            # yfinance not installed, try direct API
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
                params = {"interval": interval, "range": period}
                response = requests.get(url, params=params, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    result = data.get("chart", {}).get("result", [{}])[0]
                    timestamps = result.get("timestamp", [])
                    quote = result.get("indicators", {}).get("quote", [{}])[0]
                    
                    opens = quote.get("open", [])
                    highs = quote.get("high", [])
                    lows = quote.get("low", [])
                    closes = quote.get("close", [])
                    volumes = quote.get("volume", [])
                    
                    for i, ts in enumerate(timestamps):
                        if opens[i] is not None and closes[i] is not None:
                            candles.append(OHLCV(
                                timestamp=datetime.utcfromtimestamp(ts).isoformat(),
                                open=float(opens[i]),
                                high=float(highs[i]) if highs[i] else float(opens[i]),
                                low=float(lows[i]) if lows[i] else float(opens[i]),
                                close=float(closes[i]),
                                volume=float(volumes[i]) if volumes[i] else 0
                            ))
            except Exception as e:
                self.errors.append(f"Yahoo Finance error for {symbol}: {str(e)}")
        except Exception as e:
            self.errors.append(f"yfinance error for {symbol}: {str(e)}")
        
        return candles
    
    def _fetch_ohlcv_twelve_data(self, symbol: str, interval: str = "1h", outputsize: int = 100) -> List[OHLCV]:
        """Fetch OHLCV from Twelve Data API (backup, has daily limits)."""
        candles = []
        
        # Map our symbols to Twelve Data format
        symbol_map = {
            "EURUSD": "EUR/USD",
            "GBPUSD": "GBP/USD",
            "USDJPY": "USD/JPY",
            "AUDUSD": "AUD/USD",
            "USDCAD": "USD/CAD",
            "USDCHF": "USD/CHF",
            "XAUUSD": "XAU/USD",
            "NZDUSD": "NZD/USD"
        }
        
        td_symbol = symbol_map.get(symbol, symbol)
        
        try:
            response = requests.get(
                f"{TWELVE_DATA_URL}/time_series",
                params={
                    "symbol": td_symbol,
                    "interval": interval,
                    "outputsize": outputsize,
                    "apikey": TWELVE_DATA_API_KEY
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "values" in data:
                    for candle in reversed(data["values"]):  # Oldest first
                        candles.append(OHLCV(
                            timestamp=candle.get("datetime", ""),
                            open=float(candle.get("open", 0)),
                            high=float(candle.get("high", 0)),
                            low=float(candle.get("low", 0)),
                            close=float(candle.get("close", 0)),
                            volume=float(candle.get("volume", 0)) if candle.get("volume") else 0
                        ))
                elif "message" in data and "API credits" not in data.get("message", ""):
                    self.errors.append(f"Twelve Data: {data['message']}")
        except Exception as e:
            pass  # Silently fail, we have Yahoo as primary
        
        return candles
    
    def _fetch_ohlcv_mt5(self, symbol: str, timeframe: str = "H1", count: int = 100) -> List[OHLCV]:
        """Fetch OHLCV from MT5 API if available."""
        candles = []
        try:
            response = requests.get(
                f"{MT5_API_URL}/history/{symbol}",
                params={"timeframe": timeframe, "count": count},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for candle in data.get("candles", []):
                    candles.append(OHLCV(
                        timestamp=candle.get("time", ""),
                        open=float(candle.get("open", 0)),
                        high=float(candle.get("high", 0)),
                        low=float(candle.get("low", 0)),
                        close=float(candle.get("close", 0)),
                        volume=float(candle.get("volume", 0))
                    ))
        except Exception as e:
            pass  # MT5 history might not be available
        
        return candles
    
    def _get_ohlcv(self, symbol: str, force_refresh: bool = False) -> List[OHLCV]:
        """Get OHLCV data for a symbol, using cache if recent."""
        now = datetime.utcnow()
        
        # Check cache (refresh every 5 minutes)
        if not force_refresh and symbol in self.ohlcv_cache:
            last_fetch = self.last_ohlcv_fetch.get(symbol)
            if last_fetch and (now - last_fetch).total_seconds() < 300:
                return self.ohlcv_cache[symbol]
        
        candles = []
        source = "UNKNOWN"
        
        # 1. Try MT5 first (best for real-time)
        candles = self._fetch_ohlcv_mt5(symbol)
        if candles:
            source = "MT5"
        
        # 2. Try Yahoo Finance (FREE, no limits)
        if not candles:
            candles = self._fetch_ohlcv_yahoo(symbol)
            if candles:
                source = "YAHOO"
        
        # 3. Fallback to Twelve Data (has daily limits)
        if not candles:
            candles = self._fetch_ohlcv_twelve_data(symbol)
            if candles:
                source = "TWELVE_DATA"
        
        if candles:
            self.ohlcv_cache[symbol] = candles
            self.last_ohlcv_fetch[symbol] = now
        
        return candles
    
    def calculate_technicals(self, symbol: str, current_price: float = None) -> TechnicalIndicators:
        """Calculate all technical indicators for a symbol from REAL data."""
        candles = self._get_ohlcv(symbol)
        
        if not candles or len(candles) < 20:
            return TechnicalIndicators(
                symbol=symbol,
                candles_used=len(candles) if candles else 0,
                data_source="INSUFFICIENT_DATA"
            )
        
        # Extract close prices
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Use current price if provided
        if current_price:
            closes.append(current_price)
        
        # Calculate indicators
        rsi_2 = self._calculate_rsi(closes, period=2)
        rsi_14 = self._calculate_rsi(closes, period=14)
        sma_50 = self._calculate_sma(closes, period=50)
        sma_200 = self._calculate_sma(closes, period=200)
        
        # Recent high/low
        high_20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        low_20 = min(lows[-20:]) if len(lows) >= 20 else min(lows)
        
        # Price vs SMA200
        current = closes[-1]
        price_vs_sma200 = None
        if sma_200:
            price_vs_sma200 = "ABOVE" if current > sma_200 else "BELOW"
        
        # Trend detection
        trend = "RANGING"
        trend_strength = 50.0
        if sma_50 and sma_200:
            if sma_50 > sma_200 and current > sma_50:
                trend = "BULLISH"
                trend_strength = min(100, 50 + (current - sma_200) / sma_200 * 1000)
            elif sma_50 < sma_200 and current < sma_50:
                trend = "BEARISH"
                trend_strength = min(100, 50 + (sma_200 - current) / sma_200 * 1000)
        
        # Simple support/resistance (recent swing points)
        resistance = high_20
        support = low_20
        
        return TechnicalIndicators(
            symbol=symbol,
            rsi_2=rsi_2,
            rsi_14=rsi_14,
            sma_50=sma_50,
            sma_200=sma_200,
            price_vs_sma200=price_vs_sma200,
            high_20=high_20,
            low_20=low_20,
            trend=trend,
            trend_strength=round(trend_strength, 1),
            resistance=resistance,
            support=support,
            candles_used=len(candles),
            data_source="TWELVE_DATA" if candles else "MT5"
        )
    
    def _fetch_mt5_prices(self) -> Dict[str, PriceData]:
        """Fetch real prices from MT5 API."""
        prices = {}
        try:
            # Try to get quotes from MT5
            response = requests.get(
                f"{MT5_API_URL}/quotes",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for quote in data.get("quotes", []):
                    symbol = quote.get("symbol", "").replace("/", "")
                    if symbol in FOREX_PAIRS:
                        prices[symbol] = PriceData(
                            symbol=symbol,
                            price=float(quote.get("bid", 0)),
                            bid=float(quote.get("bid", 0)),
                            ask=float(quote.get("ask", 0)),
                            timestamp=datetime.utcnow().isoformat(),
                            source="MT5_REAL"
                        )
        except requests.exceptions.ConnectionError:
            self.errors.append("MT5 API not available")
        except Exception as e:
            self.errors.append(f"MT5 error: {str(e)}")
        
        return prices
    
    def _fetch_forex_rates(self) -> Dict[str, PriceData]:
        """Fetch forex rates from Frankfurter (free, no API key)."""
        prices = {}
        try:
            # Get USD-based rates
            response = requests.get(
                f"{FRANKFURTER_URL}/latest",
                params={"from": "USD", "to": "EUR,GBP,JPY,AUD,CAD,CHF"},
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                rates = data.get("rates", {})
                timestamp = datetime.utcnow().isoformat()
                
                # Convert to forex pair format
                for currency, rate in rates.items():
                    # USD/XXX pairs
                    symbol = f"USD{currency}"
                    if symbol in FOREX_PAIRS or f"{currency}USD" in FOREX_PAIRS:
                        # Frankfurter returns USD -> XXX, so rate is correct for USDXXX
                        prices[symbol] = PriceData(
                            symbol=symbol,
                            price=rate,
                            bid=rate,
                            ask=rate,  # No spread data from this API
                            timestamp=timestamp,
                            source="FRANKFURTER_REAL"
                        )
                        # Also create inverse pair
                        inverse_symbol = f"{currency}USD"
                        if inverse_symbol in FOREX_PAIRS:
                            prices[inverse_symbol] = PriceData(
                                symbol=inverse_symbol,
                                price=round(1/rate, 5),
                                bid=round(1/rate, 5),
                                ask=round(1/rate, 5),
                                timestamp=timestamp,
                                source="FRANKFURTER_REAL"
                            )
        except Exception as e:
            self.errors.append(f"Frankfurter error: {str(e)}")
        
        return prices
    
    def _fetch_crypto_prices(self) -> Dict[str, PriceData]:
        """Fetch crypto prices from CoinGecko (free, no API key)."""
        prices = {}
        try:
            coin_ids = {
                "BTC": "bitcoin",
                "ETH": "ethereum", 
                "SOL": "solana"
            }
            
            response = requests.get(
                f"{COINGECKO_URL}/simple/price",
                params={
                    "ids": ",".join(coin_ids.values()),
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_24hr_high": "true",
                    "include_24hr_low": "true"
                },
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                timestamp = datetime.utcnow().isoformat()
                
                for symbol, coin_id in coin_ids.items():
                    if coin_id in data:
                        coin_data = data[coin_id]
                        prices[symbol] = PriceData(
                            symbol=symbol,
                            price=coin_data.get("usd", 0),
                            bid=None,
                            ask=None,
                            timestamp=timestamp,
                            source="COINGECKO_REAL",
                            change_24h=coin_data.get("usd_24h_change"),
                            high_24h=coin_data.get("usd_24h_high"),
                            low_24h=coin_data.get("usd_24h_low")
                        )
            elif response.status_code == 429:
                self.errors.append("CoinGecko rate limit - waiting")
        except Exception as e:
            self.errors.append(f"CoinGecko error: {str(e)}")
        
        return prices
    
    def get_snapshot(self, include_technicals: bool = True) -> MarketSnapshot:
        """Get complete market snapshot from all sources."""
        self.errors = []  # Reset errors
        
        # Fetch from all sources
        mt5_prices = self._fetch_mt5_prices()
        forex_prices = self._fetch_forex_rates()
        crypto_prices = self._fetch_crypto_prices()
        
        # Merge forex (prefer MT5 if available)
        merged_forex = forex_prices.copy()
        merged_forex.update(mt5_prices)  # MT5 overwrites Frankfurter
        
        # Track which sources we got data from
        sources = []
        if mt5_prices:
            sources.append("MT5")
        if forex_prices:
            sources.append("FRANKFURTER")
        if crypto_prices:
            sources.append("COINGECKO")
        
        # Calculate technicals for forex pairs
        technicals = {}
        if include_technicals:
            for symbol, price_data in merged_forex.items():
                if symbol in FOREX_PAIRS:
                    tech = self.calculate_technicals(symbol, price_data.price)
                    technicals[symbol] = tech
            sources.append("TWELVE_DATA")
        
        # Update cache
        self.last_prices.update(merged_forex)
        self.last_prices.update(crypto_prices)
        self.technicals_cache = technicals
        
        return MarketSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            forex=merged_forex,
            crypto=crypto_prices,
            technicals=technicals,
            mt5_connected=bool(mt5_prices),
            data_sources=sources,
            errors=self.errors
        )
    
    def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get latest price for a specific symbol."""
        if symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # Try to fetch fresh
        snapshot = self.get_snapshot()
        all_prices = {**snapshot.forex, **snapshot.crypto}
        return all_prices.get(symbol)
    
    def to_llm_context(self, snapshot: MarketSnapshot) -> str:
        """Format market data for LLM consumption - NOW WITH TECHNICAL ANALYSIS!"""
        lines = [
            f"=== MARKET DATA (REAL) ===",
            f"Timestamp: {snapshot.timestamp}",
            f"Sources: {', '.join(snapshot.data_sources)}",
            f"MT5 Connected: {snapshot.mt5_connected}",
            ""
        ]
        
        if snapshot.forex:
            lines.append("FOREX PRICES:")
            for symbol, data in sorted(snapshot.forex.items()):
                spread = ""
                if data.bid and data.ask:
                    spread_pips = abs(data.ask - data.bid) * 10000
                    spread = f" (spread: {spread_pips:.1f} pips)"
                lines.append(f"  {symbol}: {data.price:.5f}{spread} [{data.source}]")
        
        # ===== NEW: TECHNICAL ANALYSIS SECTION =====
        if snapshot.technicals:
            lines.append("")
            lines.append("=" * 50)
            lines.append("=== TECHNICAL ANALYSIS (REAL OHLCV DATA) ===")
            lines.append("=" * 50)
            
            for symbol, tech in sorted(snapshot.technicals.items()):
                if tech.candles_used > 0:
                    lines.append(f"\n📊 {symbol} ({tech.candles_used} candles from {tech.data_source}):")
                    
                    # RSI - CRITICAL for strategy!
                    rsi_signal = ""
                    if tech.rsi_2 is not None:
                        if tech.rsi_2 < 10:
                            rsi_signal = "⚡ EXTREME OVERSOLD - BUY SIGNAL!"
                        elif tech.rsi_2 < 20:
                            rsi_signal = "📉 Oversold"
                        elif tech.rsi_2 > 90:
                            rsi_signal = "⚡ EXTREME OVERBOUGHT - SELL SIGNAL!"
                        elif tech.rsi_2 > 80:
                            rsi_signal = "📈 Overbought"
                        lines.append(f"   RSI(2): {tech.rsi_2:.1f} {rsi_signal}")
                    
                    if tech.rsi_14 is not None:
                        lines.append(f"   RSI(14): {tech.rsi_14:.1f}")
                    
                    # Trend & MAs
                    if tech.trend:
                        trend_emoji = "🟢" if tech.trend == "BULLISH" else "🔴" if tech.trend == "BEARISH" else "⚪"
                        lines.append(f"   Trend: {trend_emoji} {tech.trend} (strength: {tech.trend_strength}%)")
                    
                    if tech.sma_50:
                        lines.append(f"   SMA(50): {tech.sma_50:.5f}")
                    if tech.sma_200:
                        lines.append(f"   SMA(200): {tech.sma_200:.5f} (Price {tech.price_vs_sma200})")
                    
                    # Support/Resistance
                    if tech.support and tech.resistance:
                        lines.append(f"   Support: {tech.support:.5f} | Resistance: {tech.resistance:.5f}")
                    
                    # 20-period range
                    if tech.high_20 and tech.low_20:
                        range_pct = ((tech.high_20 - tech.low_20) / tech.low_20) * 100
                        lines.append(f"   20-Period Range: {tech.low_20:.5f} - {tech.high_20:.5f} ({range_pct:.2f}%)")
                else:
                    lines.append(f"\n📊 {symbol}: ⚠️ Insufficient data for analysis")
            
            lines.append("")
            lines.append("=" * 50)
            lines.append("TRADING SIGNALS FROM TECHNICALS:")
            
            # Summarize trading opportunities
            buy_signals = []
            sell_signals = []
            for symbol, tech in snapshot.technicals.items():
                if tech.rsi_2 is not None:
                    if tech.rsi_2 < 10 and tech.price_vs_sma200 == "ABOVE":
                        buy_signals.append(f"{symbol} (RSI2={tech.rsi_2:.1f}, above 200SMA)")
                    elif tech.rsi_2 > 90 and tech.price_vs_sma200 == "BELOW":
                        sell_signals.append(f"{symbol} (RSI2={tech.rsi_2:.1f}, below 200SMA)")
            
            if buy_signals:
                lines.append(f"   🟢 BUY CANDIDATES: {', '.join(buy_signals)}")
            if sell_signals:
                lines.append(f"   🔴 SELL CANDIDATES: {', '.join(sell_signals)}")
            if not buy_signals and not sell_signals:
                lines.append("   ⚪ No extreme RSI(2) signals - consider WAIT or look at other factors")
            
            lines.append("=" * 50)
        
        if snapshot.crypto:
            lines.append("")
            lines.append("CRYPTO PRICES (for correlation):")
            for symbol, data in snapshot.crypto.items():
                change = ""
                if data.change_24h:
                    change = f" ({data.change_24h:+.2f}% 24h)"
                lines.append(f"  {symbol}: ${data.price:,.2f}{change} [{data.source}]")
        
        if snapshot.errors:
            lines.append("")
            lines.append("WARNINGS:")
            for err in snapshot.errors:
                lines.append(f"  ⚠️ {err}")
        
        return "\n".join(lines)


def test_market_feed():
    """Test the market feed."""
    print("=" * 60)
    print("NEO MARKET FEED TEST")
    print("=" * 60)
    
    feed = MarketFeed()
    snapshot = feed.get_snapshot()
    
    print(feed.to_llm_context(snapshot))
    print("")
    print("=" * 60)
    print(f"Total forex pairs: {len(snapshot.forex)}")
    print(f"Total crypto pairs: {len(snapshot.crypto)}")
    print(f"Errors: {len(snapshot.errors)}")
    print("=" * 60)


if __name__ == "__main__":
    test_market_feed()

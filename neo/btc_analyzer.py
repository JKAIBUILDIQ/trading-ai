"""
NEO-BTC Module - Bitcoin Intelligence for IREN Correlation Trading

This module provides:
1. Real-time BTC price and market data
2. BTC technical analysis (RSI, MACD, EMAs)
3. BTC-IREN correlation and beta calculations
4. BTC trading signals
5. IREN implied signals based on BTC movements

Key Insight: IREN has ~1.5x beta to BTC
- BTC +10% → IREN likely +15%
- Better BTC signals = Better IREN trades!
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import data libraries
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed - pip install yfinance")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class BTCAnalyzer:
    """
    Bitcoin analysis engine for NEO trading system.
    
    Provides BTC price data, technicals, and correlation with IREN.
    """
    
    def __init__(self):
        self.data_sources = ['yahoo', 'coingecko', 'binance']
        self._cache = {}
        self._cache_expiry = {}
        self.cache_duration = 60  # Cache for 60 seconds
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache or key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]
    
    def _set_cache(self, key: str, data: dict):
        """Set cache with expiry"""
        self._cache[key] = data
        self._cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_duration)
    
    # ==================== PRICE DATA ====================
    
    def get_btc_price(self) -> Dict:
        """
        Get current BTC price and market data from multiple sources.
        
        Returns:
            {
                'price': 89692.15,
                'change_24h': 2.5,
                'change_7d': 8.3,
                'volume_24h': 45000000000,
                'market_cap': 1800000000000,
                'source': 'yahoo',
                'timestamp': '2026-01-24T12:00:00'
            }
        """
        cache_key = 'btc_price'
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        result = {
            'price': 0,
            'change_24h': 0,
            'change_7d': 0,
            'volume_24h': 0,
            'market_cap': 0,
            'source': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Try Yahoo Finance first
        if HAS_YFINANCE:
            try:
                btc = yf.Ticker('BTC-USD')
                hist = btc.history(period='7d', interval='1h')
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    price_24h_ago = float(hist['Close'].iloc[-24]) if len(hist) >= 24 else current_price
                    price_7d_ago = float(hist['Close'].iloc[0])
                    
                    result['price'] = current_price
                    result['change_24h'] = ((current_price - price_24h_ago) / price_24h_ago) * 100
                    result['change_7d'] = ((current_price - price_7d_ago) / price_7d_ago) * 100
                    result['volume_24h'] = float(hist['Volume'].iloc[-24:].sum()) if len(hist) >= 24 else float(hist['Volume'].sum())
                    result['source'] = 'yahoo'
                    
                    # Get market cap from info
                    try:
                        info = btc.info
                        result['market_cap'] = info.get('marketCap', 0)
                    except:
                        # Estimate market cap (21M max supply * price)
                        result['market_cap'] = current_price * 19700000  # ~19.7M BTC in circulation
                    
                    logger.info(f"BTC price from Yahoo: ${current_price:,.2f}")
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.warning(f"Yahoo Finance BTC error: {e}")
        
        # Try CoinGecko
        if HAS_REQUESTS:
            try:
                url = 'https://api.coingecko.com/api/v3/simple/price'
                params = {
                    'ids': 'bitcoin',
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_7d_change': 'true',
                    'include_24hr_vol': 'true',
                    'include_market_cap': 'true'
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json().get('bitcoin', {})
                    result['price'] = data.get('usd', 0)
                    result['change_24h'] = data.get('usd_24h_change', 0)
                    result['change_7d'] = data.get('usd_7d_change', 0)
                    result['volume_24h'] = data.get('usd_24h_vol', 0)
                    result['market_cap'] = data.get('usd_market_cap', 0)
                    result['source'] = 'coingecko'
                    
                    logger.info(f"BTC price from CoinGecko: ${result['price']:,.2f}")
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.warning(f"CoinGecko BTC error: {e}")
        
        # Try Binance
        if HAS_REQUESTS:
            try:
                url = 'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT'
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    result['price'] = float(data.get('lastPrice', 0))
                    result['change_24h'] = float(data.get('priceChangePercent', 0))
                    result['volume_24h'] = float(data.get('quoteVolume', 0))
                    result['source'] = 'binance'
                    
                    logger.info(f"BTC price from Binance: ${result['price']:,.2f}")
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.warning(f"Binance BTC error: {e}")
        
        return result
    
    # ==================== TECHNICAL ANALYSIS ====================
    
    def get_btc_technicals(self, timeframe: str = 'H4') -> Dict:
        """
        Calculate BTC technical indicators.
        
        Args:
            timeframe: 'H1', 'H4', 'D1'
        
        Returns:
            {
                'rsi': 65.4,
                'macd': {'value': 1250, 'signal': 1100, 'histogram': 150, 'trend': 'BULLISH'},
                'ema_20': 88500,
                'ema_50': 85000,
                'ema_200': 75000,
                'trend': 'BULLISH',
                'strength': 85,
                'support': [85000, 82000, 78000],
                'resistance': [90000, 95000, 100000]
            }
        """
        cache_key = f'btc_technicals_{timeframe}'
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        result = {
            'rsi': 50,
            'macd': {'value': 0, 'signal': 0, 'histogram': 0, 'trend': 'NEUTRAL'},
            'ema_20': 0,
            'ema_50': 0,
            'ema_200': 0,
            'sma_20': 0,
            'trend': 'NEUTRAL',
            'strength': 50,
            'support': [],
            'resistance': [],
            'atr': 0,
            'bollinger': {'upper': 0, 'middle': 0, 'lower': 0}
        }
        
        if not HAS_YFINANCE:
            return result
        
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                'H1': '1h',
                'H4': '1h',  # We'll resample
                'D1': '1d'
            }
            period_map = {
                'H1': '30d',
                'H4': '60d',
                'D1': '1y'
            }
            
            interval = interval_map.get(timeframe, '1h')
            period = period_map.get(timeframe, '60d')
            
            btc = yf.Ticker('BTC-USD')
            hist = btc.history(period=period, interval=interval)
            
            if hist.empty or len(hist) < 50:
                return result
            
            closes = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            
            # RSI (14)
            result['rsi'] = self._calculate_rsi(closes, 14)
            
            # MACD (12, 26, 9)
            macd_data = self._calculate_macd(closes)
            result['macd'] = macd_data
            
            # EMAs
            result['ema_20'] = self._calculate_ema(closes, 20)
            result['ema_50'] = self._calculate_ema(closes, 50)
            result['ema_200'] = self._calculate_ema(closes, 200) if len(closes) >= 200 else result['ema_50']
            result['sma_20'] = np.mean(closes[-20:])
            
            # ATR (14)
            result['atr'] = self._calculate_atr(highs, lows, closes, 14)
            
            # Bollinger Bands (20, 2)
            result['bollinger'] = self._calculate_bollinger(closes, 20, 2)
            
            # Determine trend
            current_price = closes[-1]
            
            if current_price > result['ema_20'] > result['ema_50']:
                result['trend'] = 'BULLISH'
                result['strength'] = min(100, 60 + (result['rsi'] - 50))
            elif current_price < result['ema_20'] < result['ema_50']:
                result['trend'] = 'BEARISH'
                result['strength'] = min(100, 60 + (50 - result['rsi']))
            else:
                result['trend'] = 'NEUTRAL'
                result['strength'] = 50
            
            # Calculate support and resistance levels
            result['support'], result['resistance'] = self._calculate_sr_levels(highs, lows, closes)
            
            self._set_cache(cache_key, result)
            logger.info(f"BTC technicals: RSI={result['rsi']:.1f}, Trend={result['trend']}")
            
        except Exception as e:
            logger.error(f"Error calculating BTC technicals: {e}")
        
        return result
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict:
        """Calculate MACD"""
        if len(prices) < 26:
            return {'value': 0, 'signal': 0, 'histogram': 0, 'trend': 'NEUTRAL'}
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        # For signal line, we need historical MACD values
        macd_history = []
        for i in range(26, len(prices)):
            e12 = self._calculate_ema(prices[:i+1], 12)
            e26 = self._calculate_ema(prices[:i+1], 26)
            macd_history.append(e12 - e26)
        
        if len(macd_history) >= 9:
            signal_line = np.mean(macd_history[-9:])
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        trend = 'BULLISH' if histogram > 0 and macd_line > signal_line else 'BEARISH' if histogram < 0 else 'NEUTRAL'
        
        return {
            'value': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(histogram),
            'trend': trend
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period+1:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        if len(closes) < period + 1:
            return float(np.mean(highs - lows))
        
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        return float(np.mean(tr[-period:]))
    
    def _calculate_bollinger(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}
        
        middle = float(np.mean(prices[-period:]))
        std = float(np.std(prices[-period:]))
        
        return {
            'upper': middle + (std * std_dev),
            'middle': middle,
            'lower': middle - (std * std_dev)
        }
    
    def _calculate_sr_levels(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[List, List]:
        """Calculate support and resistance levels"""
        current_price = closes[-1]
        
        # Find pivot points
        recent_highs = sorted(highs[-50:], reverse=True)[:10]
        recent_lows = sorted(lows[-50:])[:10]
        
        # Cluster similar levels
        supports = []
        resistances = []
        
        for low in recent_lows:
            if low < current_price:
                supports.append(float(low))
        
        for high in recent_highs:
            if high > current_price:
                resistances.append(float(high))
        
        # Remove duplicates within 1%
        supports = self._cluster_levels(supports)[:3]
        resistances = self._cluster_levels(resistances)[:3]
        
        return supports, resistances
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
        
        return clustered
    
    # ==================== CORRELATION ====================
    
    def get_btc_iren_correlation(self, lookback_days: int = 30) -> Dict:
        """
        Calculate rolling correlation between BTC and IREN.
        
        Returns:
            {
                'correlation_30d': 0.75,
                'correlation_90d': 0.70,
                'beta': 1.50,
                'r_squared': 0.56,
                'implied_iren_move': {
                    'if_btc_up_5': 7.5,
                    'if_btc_down_5': -7.5,
                    'if_btc_up_10': 15.0,
                    'if_btc_down_10': -15.0
                }
            }
        """
        cache_key = f'btc_iren_corr_{lookback_days}'
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        result = {
            'correlation_30d': 0.75,  # Default based on historical data
            'correlation_90d': 0.70,
            'beta': 1.50,
            'r_squared': 0.56,
            'implied_iren_move': {
                'if_btc_up_5': 7.5,
                'if_btc_down_5': -7.5,
                'if_btc_up_10': 15.0,
                'if_btc_down_10': -15.0
            }
        }
        
        if not HAS_YFINANCE:
            return result
        
        try:
            # Get BTC and IREN historical data
            btc = yf.Ticker('BTC-USD')
            iren = yf.Ticker('IREN')
            
            btc_hist = btc.history(period='6mo', interval='1d')
            iren_hist = iren.history(period='6mo', interval='1d')
            
            if btc_hist.empty or iren_hist.empty:
                return result
            
            # Align dates
            common_dates = btc_hist.index.intersection(iren_hist.index)
            
            if len(common_dates) < 30:
                return result
            
            btc_prices = btc_hist.loc[common_dates, 'Close'].values
            iren_prices = iren_hist.loc[common_dates, 'Close'].values
            
            # Calculate returns
            btc_returns = np.diff(btc_prices) / btc_prices[:-1]
            iren_returns = np.diff(iren_prices) / iren_prices[:-1]
            
            # 30-day correlation
            if len(btc_returns) >= 30:
                corr_30d = np.corrcoef(btc_returns[-30:], iren_returns[-30:])[0, 1]
                result['correlation_30d'] = float(corr_30d) if not np.isnan(corr_30d) else 0.75
            
            # 90-day correlation
            if len(btc_returns) >= 90:
                corr_90d = np.corrcoef(btc_returns[-90:], iren_returns[-90:])[0, 1]
                result['correlation_90d'] = float(corr_90d) if not np.isnan(corr_90d) else 0.70
            
            # Beta calculation: Cov(IREN, BTC) / Var(BTC)
            window = min(30, len(btc_returns))
            covariance = np.cov(iren_returns[-window:], btc_returns[-window:])[0, 1]
            btc_variance = np.var(btc_returns[-window:])
            
            if btc_variance > 0:
                beta = covariance / btc_variance
                result['beta'] = float(beta) if not np.isnan(beta) else 1.50
            
            # R-squared
            result['r_squared'] = result['correlation_30d'] ** 2
            
            # Implied IREN moves
            beta = result['beta']
            result['implied_iren_move'] = {
                'if_btc_up_5': round(5 * beta, 2),
                'if_btc_down_5': round(-5 * beta, 2),
                'if_btc_up_10': round(10 * beta, 2),
                'if_btc_down_10': round(-10 * beta, 2)
            }
            
            self._set_cache(cache_key, result)
            logger.info(f"BTC-IREN correlation: {result['correlation_30d']:.2f}, Beta: {result['beta']:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating BTC-IREN correlation: {e}")
        
        return result
    
    # ==================== SIGNALS ====================
    
    def get_btc_signal(self) -> Dict:
        """
        Generate BTC trading signal based on technicals.
        
        Returns:
            {
                'signal': 'BUY',
                'confidence': 85,
                'entry': 89500,
                'stop_loss': 85000,
                'take_profit': [95000, 100000],
                'reasoning': [...]
            }
        """
        price_data = self.get_btc_price()
        technicals = self.get_btc_technicals('H4')
        
        signal = 'HOLD'
        confidence = 50
        reasoning = []
        
        current_price = price_data['price']
        rsi = technicals['rsi']
        macd = technicals['macd']
        trend = technicals['trend']
        ema_20 = technicals['ema_20']
        ema_50 = technicals['ema_50']
        atr = technicals['atr']
        
        # Scoring system
        score = 0
        
        # Trend analysis
        if trend == 'BULLISH':
            score += 25
            reasoning.append(f"Trend is BULLISH")
        elif trend == 'BEARISH':
            score -= 25
            reasoning.append(f"Trend is BEARISH")
        
        # RSI analysis
        if 30 < rsi < 45:
            score += 20
            reasoning.append(f"RSI at {rsi:.1f} - potential oversold bounce")
        elif 55 < rsi < 70:
            score += 15
            reasoning.append(f"RSI at {rsi:.1f} - bullish momentum")
        elif rsi >= 70:
            score -= 15
            reasoning.append(f"RSI at {rsi:.1f} - overbought warning")
        elif rsi <= 30:
            score += 25
            reasoning.append(f"RSI at {rsi:.1f} - oversold, potential reversal")
        
        # MACD analysis
        if macd['trend'] == 'BULLISH':
            score += 20
            reasoning.append("MACD bullish crossover")
        elif macd['trend'] == 'BEARISH':
            score -= 20
            reasoning.append("MACD bearish crossover")
        
        # Price vs EMAs
        if current_price > ema_20 > ema_50:
            score += 20
            reasoning.append("Price above all EMAs - strong uptrend")
        elif current_price < ema_20 < ema_50:
            score -= 20
            reasoning.append("Price below all EMAs - downtrend")
        elif ema_20 > ema_50 and current_price > ema_50:
            score += 10
            reasoning.append("Golden cross pattern - bullish")
        
        # 24h momentum
        if price_data['change_24h'] > 3:
            score += 10
            reasoning.append(f"Strong 24h momentum: +{price_data['change_24h']:.1f}%")
        elif price_data['change_24h'] < -3:
            score -= 10
            reasoning.append(f"Negative 24h momentum: {price_data['change_24h']:.1f}%")
        
        # Convert score to signal
        if score >= 40:
            signal = 'BUY'
            confidence = min(95, 60 + score)
        elif score <= -40:
            signal = 'SELL'
            confidence = min(95, 60 + abs(score))
        else:
            signal = 'HOLD'
            confidence = 50 + abs(score) // 2
        
        # Calculate levels
        if atr == 0:
            atr = current_price * 0.02  # Default 2% ATR
        
        if signal == 'BUY':
            stop_loss = current_price - (atr * 2)
            take_profits = [
                round(current_price + (atr * 2), 2),
                round(current_price + (atr * 4), 2)
            ]
        elif signal == 'SELL':
            stop_loss = current_price + (atr * 2)
            take_profits = [
                round(current_price - (atr * 2), 2),
                round(current_price - (atr * 4), 2)
            ]
        else:
            stop_loss = current_price - (atr * 1.5)
            take_profits = [current_price]
        
        return {
            'signal': signal,
            'confidence': int(confidence),
            'price': current_price,
            'entry': current_price,
            'stop_loss': round(stop_loss, 2),
            'take_profit': take_profits,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_iren_implied_signal(self) -> Dict:
        """
        Generate IREN signal based on BTC analysis.
        
        Uses BTC signal + correlation to generate IREN trade recommendation.
        """
        btc_signal = self.get_btc_signal()
        correlation = self.get_btc_iren_correlation()
        
        # Get IREN current price
        iren_price = 0
        if HAS_YFINANCE:
            try:
                iren = yf.Ticker('IREN')
                iren_price = float(iren.history(period='1d')['Close'].iloc[-1])
            except:
                pass
        
        # Adjust confidence based on correlation strength
        corr_strength = abs(correlation['correlation_30d'])
        adjusted_confidence = btc_signal['confidence'] * corr_strength
        
        # Determine signal
        if btc_signal['signal'] == 'BUY' and corr_strength > 0.5:
            signal = 'BUY'
            # Boost confidence for high correlation
            adjusted_confidence = min(95, adjusted_confidence * 1.1)
        elif btc_signal['signal'] == 'SELL' and corr_strength > 0.5:
            signal = 'SELL'
            adjusted_confidence = min(95, adjusted_confidence * 1.1)
        else:
            signal = 'HOLD'
            adjusted_confidence = 50
        
        # Calculate implied IREN targets
        beta = correlation['beta']
        btc_pct_to_tp1 = ((btc_signal['take_profit'][0] - btc_signal['price']) / btc_signal['price']) * 100
        
        iren_implied_tp = iren_price * (1 + (btc_pct_to_tp1 * beta / 100))
        iren_implied_sl = iren_price * (1 - (2 * beta / 100))  # ~2% BTC move = stop
        
        reasoning = f"BTC {btc_signal['signal']} ({btc_signal['confidence']}% confidence) × {correlation['correlation_30d']:.2f} correlation × {beta:.2f}x beta"
        
        return {
            'signal': signal,
            'confidence': int(adjusted_confidence),
            'iren_price': round(iren_price, 2),
            'iren_target': round(iren_implied_tp, 2),
            'iren_stop': round(iren_implied_sl, 2),
            'reasoning': reasoning,
            'btc_signal': btc_signal,
            'correlation': correlation['correlation_30d'],
            'beta': beta,
            'timestamp': datetime.now().isoformat()
        }
    
    # ==================== FULL ANALYSIS ====================
    
    def get_full_analysis(self) -> Dict:
        """
        Get complete BTC analysis for NEO integration.
        """
        price = self.get_btc_price()
        technicals = self.get_btc_technicals('H4')
        correlation = self.get_btc_iren_correlation()
        btc_signal = self.get_btc_signal()
        iren_implied = self.get_iren_implied_signal()
        
        return {
            'price': price,
            'technicals': technicals,
            'correlation': correlation,
            'btc_signal': btc_signal,
            'iren_implied': iren_implied,
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance for easy access
_btc_analyzer = None

def get_btc_analyzer() -> BTCAnalyzer:
    """Get or create BTCAnalyzer singleton"""
    global _btc_analyzer
    if _btc_analyzer is None:
        _btc_analyzer = BTCAnalyzer()
    return _btc_analyzer


def get_btc_analysis() -> Dict:
    """Quick function to get full BTC analysis"""
    return get_btc_analyzer().get_full_analysis()


# CLI testing
if __name__ == '__main__':
    import json
    
    print("=" * 60)
    print("🔶 NEO-BTC Analyzer Test")
    print("=" * 60)
    
    analyzer = BTCAnalyzer()
    
    print("\n📊 BTC Price:")
    price = analyzer.get_btc_price()
    print(f"  Price: ${price['price']:,.2f}")
    print(f"  24h Change: {price['change_24h']:+.2f}%")
    print(f"  Source: {price['source']}")
    
    print("\n📈 BTC Technicals:")
    tech = analyzer.get_btc_technicals('H4')
    print(f"  RSI: {tech['rsi']:.1f}")
    print(f"  MACD: {tech['macd']['trend']}")
    print(f"  Trend: {tech['trend']} (Strength: {tech['strength']})")
    
    print("\n🔗 BTC-IREN Correlation:")
    corr = analyzer.get_btc_iren_correlation()
    print(f"  30D Correlation: {corr['correlation_30d']:.2f}")
    print(f"  Beta: {corr['beta']:.2f}x")
    print(f"  If BTC +10%: IREN {corr['implied_iren_move']['if_btc_up_10']:+.1f}%")
    
    print("\n🎯 BTC Signal:")
    sig = analyzer.get_btc_signal()
    print(f"  Signal: {sig['signal']}")
    print(f"  Confidence: {sig['confidence']}%")
    print(f"  Entry: ${sig['entry']:,.2f}")
    print(f"  Stop Loss: ${sig['stop_loss']:,.2f}")
    print("  Reasoning:")
    for r in sig['reasoning']:
        print(f"    - {r}")
    
    print("\n💎 IREN Implied Signal:")
    iren = analyzer.get_iren_implied_signal()
    print(f"  Signal: {iren['signal']}")
    print(f"  Confidence: {iren['confidence']}%")
    print(f"  IREN Target: ${iren['iren_target']:.2f}")
    print(f"  Reasoning: {iren['reasoning']}")
    
    print("\n" + "=" * 60)
    print("✅ BTC Analyzer Ready!")

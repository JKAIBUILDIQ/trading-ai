"""
Volume Analyzer for Crellastein Meta Bot
Analyzes volume patterns for trade confirmation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VolumeAnalyzer")


class VolumeAnalyzer:
    """
    Analyzes volume patterns to confirm signals:
    - Volume spikes (breakout confirmation)
    - Volume divergences (potential reversals)
    - Volume profile (support/resistance)
    - Accumulation/Distribution
    """
    
    ASSET_CONFIG = {
        "XAUUSD": {"ticker": "GC=F", "name": "Gold"},
        "IREN": {"ticker": "IREN", "name": "IREN"},
    }
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.config = self.ASSET_CONFIG.get(symbol, self.ASSET_CONFIG["XAUUSD"])
        self._df = None
        self._last_fetch = None
    
    def _get_data(self, period: str = "30d", interval: str = "1h") -> pd.DataFrame:
        """Fetch market data with caching"""
        now = datetime.now()
        if self._df is not None and self._last_fetch and (now - self._last_fetch).seconds < 300:
            return self._df
        
        try:
            ticker = yf.Ticker(self.config["ticker"])
            self._df = ticker.history(period=period, interval=interval)
            self._last_fetch = now
            return self._df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def analyze_volume(self) -> Dict:
        """Complete volume analysis"""
        df = self._get_data()
        if df.empty or len(df) < 20:
            return self._default_analysis()
        
        try:
            # Basic volume stats
            current_volume = float(df['Volume'].iloc[-1])
            avg_volume_20 = float(df['Volume'].tail(20).mean())
            avg_volume_50 = float(df['Volume'].tail(50).mean()) if len(df) >= 50 else avg_volume_20
            
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # Volume spike detection
            volume_spike = volume_ratio > 1.5
            volume_surge = volume_ratio > 2.0
            volume_dry = volume_ratio < 0.5
            
            # Price-Volume relationship
            price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
            prev_volume = float(df['Volume'].iloc[-2])
            
            # Bullish: price up on increasing volume
            # Bearish: price down on increasing volume
            bullish_volume = price_change > 0 and current_volume > prev_volume
            bearish_volume = price_change < 0 and current_volume > prev_volume
            
            # Volume divergence
            # Price making new highs but volume declining = bearish divergence
            # Price making new lows but volume declining = bullish divergence
            high_5 = df['High'].tail(5)
            vol_5 = df['Volume'].tail(5)
            
            price_trend_up = high_5.iloc[-1] > high_5.iloc[0]
            volume_trend_down = vol_5.iloc[-1] < vol_5.iloc[0]
            
            bearish_divergence = price_trend_up and volume_trend_down
            
            low_5 = df['Low'].tail(5)
            price_trend_down = low_5.iloc[-1] < low_5.iloc[0]
            bullish_divergence = price_trend_down and volume_trend_down
            
            # Volume profile - find high volume nodes (support/resistance)
            volume_profile = self._calc_volume_profile(df)
            
            # Accumulation/Distribution
            ad_signal = self._calc_accumulation_distribution(df)
            
            # On-Balance Volume trend
            obv_signal = self._calc_obv_signal(df)
            
            # Overall volume signal
            volume_signal = self._determine_signal(
                volume_spike, volume_surge, volume_dry,
                bullish_volume, bearish_volume,
                bullish_divergence, bearish_divergence,
                ad_signal, obv_signal
            )
            
            return {
                "symbol": self.symbol,
                "timestamp": datetime.utcnow().isoformat(),
                
                # Current state
                "current_volume": int(current_volume),
                "avg_volume_20": int(avg_volume_20),
                "avg_volume_50": int(avg_volume_50),
                "volume_ratio": round(volume_ratio, 2),
                
                # Volume signals
                "volume_spike": volume_spike,
                "volume_surge": volume_surge,
                "volume_dry": volume_dry,
                
                # Price-Volume relationship
                "bullish_volume": bullish_volume,
                "bearish_volume": bearish_volume,
                
                # Divergences
                "bullish_divergence": bullish_divergence,
                "bearish_divergence": bearish_divergence,
                
                # Advanced signals
                "accumulation_distribution": ad_signal,
                "obv_signal": obv_signal,
                "volume_profile_levels": volume_profile,
                
                # Overall
                "volume_signal": volume_signal["signal"],
                "volume_confidence": volume_signal["confidence"],
                "volume_reasoning": volume_signal["reasoning"],
            }
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return self._default_analysis()
    
    def _calc_volume_profile(self, df: pd.DataFrame, bins: int = 10) -> List[Dict]:
        """Calculate volume profile - price levels with high volume"""
        try:
            # Create price bins
            price_range = df['Close'].max() - df['Close'].min()
            bin_size = price_range / bins
            
            levels = []
            for i in range(bins):
                low = df['Close'].min() + (i * bin_size)
                high = low + bin_size
                
                # Sum volume at this price level
                mask = (df['Close'] >= low) & (df['Close'] < high)
                vol_at_level = df.loc[mask, 'Volume'].sum()
                
                levels.append({
                    "price_low": round(float(low), 2),
                    "price_high": round(float(high), 2),
                    "volume": int(vol_at_level),
                })
            
            # Sort by volume and return top 3 high-volume nodes
            levels.sort(key=lambda x: x["volume"], reverse=True)
            return levels[:3]
        except:
            return []
    
    def _calc_accumulation_distribution(self, df: pd.DataFrame) -> str:
        """Calculate A/D line trend"""
        try:
            # A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0)
            ad = (clv * df['Volume']).cumsum()
            
            # Check trend of last 10 periods
            ad_10 = ad.tail(10)
            if ad_10.iloc[-1] > ad_10.iloc[0] * 1.05:
                return "ACCUMULATION"
            elif ad_10.iloc[-1] < ad_10.iloc[0] * 0.95:
                return "DISTRIBUTION"
            else:
                return "NEUTRAL"
        except:
            return "NEUTRAL"
    
    def _calc_obv_signal(self, df: pd.DataFrame) -> str:
        """Calculate On-Balance Volume signal"""
        try:
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            obv_series = pd.Series(obv)
            obv_10 = obv_series.tail(10)
            
            if obv_10.iloc[-1] > obv_10.iloc[0]:
                return "BULLISH"
            elif obv_10.iloc[-1] < obv_10.iloc[0]:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except:
            return "NEUTRAL"
    
    def _determine_signal(
        self, spike: bool, surge: bool, dry: bool,
        bullish_vol: bool, bearish_vol: bool,
        bull_div: bool, bear_div: bool,
        ad_signal: str, obv_signal: str
    ) -> Dict:
        """Determine overall volume signal"""
        
        bullish_points = 0
        bearish_points = 0
        reasons = []
        
        # Volume spike on bullish move = strong confirmation
        if surge and bullish_vol:
            bullish_points += 3
            reasons.append("Volume surge on bullish candle")
        elif spike and bullish_vol:
            bullish_points += 2
            reasons.append("Volume spike on bullish candle")
        elif bullish_vol:
            bullish_points += 1
            reasons.append("Increasing volume on up move")
        
        # Volume spike on bearish move = bearish confirmation
        if surge and bearish_vol:
            bearish_points += 3
            reasons.append("Volume surge on bearish candle")
        elif spike and bearish_vol:
            bearish_points += 2
            reasons.append("Volume spike on bearish candle")
        elif bearish_vol:
            bearish_points += 1
            reasons.append("Increasing volume on down move")
        
        # Divergences
        if bull_div:
            bullish_points += 2
            reasons.append("Bullish volume divergence")
        if bear_div:
            bearish_points += 2
            reasons.append("Bearish volume divergence")
        
        # A/D signal
        if ad_signal == "ACCUMULATION":
            bullish_points += 1
            reasons.append("Accumulation detected")
        elif ad_signal == "DISTRIBUTION":
            bearish_points += 1
            reasons.append("Distribution detected")
        
        # OBV signal
        if obv_signal == "BULLISH":
            bullish_points += 1
            reasons.append("OBV bullish")
        elif obv_signal == "BEARISH":
            bearish_points += 1
            reasons.append("OBV bearish")
        
        # Dry volume = uncertainty
        if dry:
            reasons.append("Low volume - weak conviction")
        
        # Determine signal
        total = bullish_points + bearish_points
        if total == 0:
            return {"signal": "NEUTRAL", "confidence": 50, "reasoning": "No clear volume signal"}
        
        if bullish_points > bearish_points:
            confidence = min(85, 50 + (bullish_points - bearish_points) * 10)
            return {"signal": "BULLISH", "confidence": confidence, "reasoning": "; ".join(reasons)}
        elif bearish_points > bullish_points:
            confidence = min(85, 50 + (bearish_points - bullish_points) * 10)
            return {"signal": "BEARISH", "confidence": confidence, "reasoning": "; ".join(reasons)}
        else:
            return {"signal": "NEUTRAL", "confidence": 50, "reasoning": "Mixed volume signals"}
    
    def _default_analysis(self) -> Dict:
        """Default when data unavailable"""
        return {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "current_volume": 0,
            "avg_volume_20": 0,
            "avg_volume_50": 0,
            "volume_ratio": 1.0,
            "volume_spike": False,
            "volume_surge": False,
            "volume_dry": False,
            "bullish_volume": False,
            "bearish_volume": False,
            "bullish_divergence": False,
            "bearish_divergence": False,
            "accumulation_distribution": "NEUTRAL",
            "obv_signal": "NEUTRAL",
            "volume_profile_levels": [],
            "volume_signal": "NEUTRAL",
            "volume_confidence": 50,
            "volume_reasoning": "Data unavailable",
        }


def get_xauusd_volume() -> Dict:
    analyzer = VolumeAnalyzer("XAUUSD")
    return analyzer.analyze_volume()


def get_iren_volume() -> Dict:
    analyzer = VolumeAnalyzer("IREN")
    return analyzer.analyze_volume()


if __name__ == "__main__":
    print("=" * 60)
    print("📊 VOLUME ANALYZER TEST")
    print("=" * 60)
    
    for symbol in ["XAUUSD", "IREN"]:
        print(f"\n{symbol}:")
        analyzer = VolumeAnalyzer(symbol)
        data = analyzer.analyze_volume()
        print(f"  Volume Ratio: {data['volume_ratio']}x average")
        print(f"  Spike: {data['volume_spike']} | Surge: {data['volume_surge']}")
        print(f"  A/D: {data['accumulation_distribution']} | OBV: {data['obv_signal']}")
        print(f"  Signal: {data['volume_signal']} ({data['volume_confidence']}%)")
        print(f"  Reason: {data['volume_reasoning']}")

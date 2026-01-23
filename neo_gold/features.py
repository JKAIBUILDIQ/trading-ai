"""
NEO-GOLD Feature Engineering
Gold-specific features for XAUUSD trading
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
from .config import (
    SESSIONS, MAJOR_ROUND_LEVELS, MINOR_ROUND_LEVELS,
    ROUND_NUMBER_PROXIMITY_PIPS, logger
)


class GoldFeatureExtractor:
    """
    Extracts Gold-specific features for trading decisions.
    
    Features:
    - Session timing (Asia/London/NY/Overlap)
    - Distance to round numbers ($50/$100 levels)
    - Asian session range
    - Volatility regime
    - DXY correlation
    - Time since last news
    """
    
    def __init__(self):
        self.current_price: float = 0
        self.asian_high: float = 0
        self.asian_low: float = 0
        self.dxy_price: float = 0
        self.last_news_time: Optional[datetime] = None
        self.recent_candles: List[Dict] = []
        
    def extract_all(self, price_data: Dict) -> Dict:
        """Extract all Gold-specific features from price data."""
        
        self.current_price = price_data.get("price", 0)
        self.recent_candles = price_data.get("candles", [])
        self.dxy_price = price_data.get("dxy", 0)
        
        features = {
            "timestamp": datetime.utcnow().isoformat(),
            "price": self.current_price,
            "session": self.get_current_session(),
            "session_details": self.get_session_details(),
            "round_number": self.get_round_number_features(),
            "asian_range": self.get_asian_range(),
            "volatility": self.get_volatility_regime(),
            "dxy_correlation": self.get_dxy_correlation(),
            "time_since_news": self.get_time_since_news(),
            "atr": self.calculate_atr(),
            "momentum": self.get_momentum_features()
        }
        
        logger.info(f"📊 Features extracted: Session={features['session']}, "
                   f"Near Round={features['round_number']['is_near']}, "
                   f"Volatility={features['volatility']['regime']}")
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════
    # SESSION FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_current_session(self) -> str:
        """Determine current trading session."""
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute
        current_time = hour * 60 + minute  # Minutes since midnight
        
        def time_to_minutes(t: str) -> int:
            h, m = map(int, t.split(":"))
            return h * 60 + m
        
        # Check overlap first (most important for Gold)
        overlap_start = time_to_minutes(SESSIONS["OVERLAP_LONDON_NY"]["start"])
        overlap_end = time_to_minutes(SESSIONS["OVERLAP_LONDON_NY"]["end"])
        if overlap_start <= current_time < overlap_end:
            return "OVERLAP_LONDON_NY"
        
        # Check dead zone
        dead_start = time_to_minutes(SESSIONS["DEAD_ZONE"]["start"])
        if current_time >= dead_start or current_time < time_to_minutes(SESSIONS["ASIA"]["start"]):
            return "DEAD_ZONE"
        
        # Check individual sessions
        for session_name, times in SESSIONS.items():
            if session_name in ["OVERLAP_LONDON_NY", "DEAD_ZONE"]:
                continue
            start = time_to_minutes(times["start"])
            end = time_to_minutes(times["end"])
            if start <= current_time < end:
                return session_name
        
        return "UNKNOWN"
    
    def get_session_details(self) -> Dict:
        """Get detailed session information."""
        now = datetime.utcnow()
        session = self.get_current_session()
        
        session_info = SESSIONS.get(session, {})
        
        # Calculate time into session
        if session_info:
            start_parts = session_info["start"].split(":")
            session_start = now.replace(hour=int(start_parts[0]), minute=int(start_parts[1]), second=0)
            if session_start > now:
                session_start -= timedelta(days=1)
            minutes_into_session = (now - session_start).seconds // 60
        else:
            minutes_into_session = 0
        
        return {
            "session": session,
            "minutes_into_session": minutes_into_session,
            "is_overlap": session == "OVERLAP_LONDON_NY",
            "is_dead_zone": session == "DEAD_ZONE",
            "is_london_open": session == "LONDON" and minutes_into_session < 60,
            "is_ny_open": session == "NEW_YORK" and minutes_into_session < 60,
            "expected_volatility": self._get_session_volatility(session)
        }
    
    def _get_session_volatility(self, session: str) -> str:
        """Expected volatility by session."""
        volatility_map = {
            "ASIA": "LOW",
            "LONDON": "HIGH",
            "NEW_YORK": "HIGH", 
            "OVERLAP_LONDON_NY": "VERY_HIGH",
            "DEAD_ZONE": "VERY_LOW"
        }
        return volatility_map.get(session, "MEDIUM")
    
    # ═══════════════════════════════════════════════════════════════════
    # ROUND NUMBER FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_round_number_features(self) -> Dict:
        """Analyze proximity to psychological round numbers."""
        
        if self.current_price == 0:
            return {"is_near": False, "nearest": 0, "distance": 0, "type": "none"}
        
        # Find nearest major round ($100 levels)
        nearest_major = min(MAJOR_ROUND_LEVELS, 
                          key=lambda x: abs(x - self.current_price))
        major_distance = abs(self.current_price - nearest_major)
        
        # Find nearest minor round ($50 levels)
        all_levels = MAJOR_ROUND_LEVELS + MINOR_ROUND_LEVELS
        nearest_any = min(all_levels, key=lambda x: abs(x - self.current_price))
        any_distance = abs(self.current_price - nearest_any)
        
        # Distance in pips (1 pip = $0.10 for Gold)
        distance_pips = any_distance * 10
        
        is_near = distance_pips <= ROUND_NUMBER_PROXIMITY_PIPS
        level_type = "MAJOR" if nearest_any in MAJOR_ROUND_LEVELS else "MINOR"
        
        # Determine if price is above or below the level
        position = "ABOVE" if self.current_price > nearest_any else "BELOW"
        
        return {
            "is_near": is_near,
            "nearest": nearest_any,
            "distance_dollars": round(any_distance, 2),
            "distance_pips": round(distance_pips, 1),
            "type": level_type,
            "position": position,
            "nearest_major": nearest_major,
            "major_distance_dollars": round(major_distance, 2),
            "magnet_strength": self._calculate_magnet_strength(distance_pips, level_type)
        }
    
    def _calculate_magnet_strength(self, distance_pips: float, level_type: str) -> str:
        """Calculate how strongly price is attracted to round number."""
        base_strength = 100 - min(100, distance_pips * 2)
        if level_type == "MAJOR":
            base_strength *= 1.5
        
        if base_strength >= 80:
            return "STRONG"
        elif base_strength >= 50:
            return "MODERATE"
        else:
            return "WEAK"
    
    # ═══════════════════════════════════════════════════════════════════
    # ASIAN RANGE FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_asian_range(self) -> Dict:
        """Calculate Asian session range for breakout prediction."""
        
        # Get candles from Asian session (00:00 - 07:00 UTC)
        asian_candles = self._get_session_candles("ASIA")
        
        if not asian_candles:
            return {
                "high": 0, "low": 0, "range_pips": 0,
                "breakout_expected": False, "range_type": "UNKNOWN"
            }
        
        asian_high = max(c.get("high", 0) for c in asian_candles)
        asian_low = min(c.get("low", float("inf")) for c in asian_candles)
        
        range_dollars = asian_high - asian_low
        range_pips = range_dollars * 10  # $1 = 10 pips
        
        # Tight Asian range = expect London breakout
        breakout_expected = range_pips < 100  # Less than $10 range
        
        range_type = "TIGHT" if range_pips < 80 else "NORMAL" if range_pips < 150 else "WIDE"
        
        return {
            "high": round(asian_high, 2),
            "low": round(asian_low, 2),
            "range_dollars": round(range_dollars, 2),
            "range_pips": round(range_pips, 1),
            "breakout_expected": breakout_expected,
            "range_type": range_type,
            "price_vs_range": self._get_price_position_in_range(asian_high, asian_low)
        }
    
    def _get_session_candles(self, session: str) -> List[Dict]:
        """Filter candles to specific session."""
        if not self.recent_candles:
            return []
        
        session_times = SESSIONS.get(session, {})
        if not session_times:
            return []
        
        start_hour = int(session_times["start"].split(":")[0])
        end_hour = int(session_times["end"].split(":")[0])
        
        return [
            c for c in self.recent_candles
            if start_hour <= c.get("hour", 0) < end_hour
        ]
    
    def _get_price_position_in_range(self, high: float, low: float) -> str:
        """Where is current price relative to Asian range?"""
        if high == 0 or low == 0 or high == low:
            return "UNKNOWN"
        
        if self.current_price > high:
            return "ABOVE_RANGE"
        elif self.current_price < low:
            return "BELOW_RANGE"
        else:
            mid = (high + low) / 2
            if self.current_price > mid:
                return "UPPER_HALF"
            else:
                return "LOWER_HALF"
    
    # ═══════════════════════════════════════════════════════════════════
    # VOLATILITY FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_volatility_regime(self) -> Dict:
        """Determine current volatility regime."""
        
        atr = self.calculate_atr()
        
        # Gold typical ATR ranges (14-period)
        if atr < 15:
            regime = "QUIET"
        elif atr < 30:
            regime = "NORMAL"
        elif atr < 50:
            regime = "VOLATILE"
        else:
            regime = "CHAOS"
        
        return {
            "regime": regime,
            "atr": round(atr, 2),
            "position_size_adjustment": self._get_size_adjustment(regime)
        }
    
    def calculate_atr(self, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(self.recent_candles) < period + 1:
            return 25.0  # Default for Gold
        
        true_ranges = []
        for i in range(1, min(period + 1, len(self.recent_candles))):
            candle = self.recent_candles[i]
            prev_candle = self.recent_candles[i - 1]
            
            high = candle.get("high", 0)
            low = candle.get("low", 0)
            prev_close = prev_candle.get("close", 0)
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 25.0
    
    def _get_size_adjustment(self, regime: str) -> float:
        """Adjust position size based on volatility."""
        adjustments = {
            "QUIET": 1.2,    # Can size up slightly
            "NORMAL": 1.0,   # Normal size
            "VOLATILE": 0.7, # Reduce size
            "CHAOS": 0.5     # Half size
        }
        return adjustments.get(regime, 1.0)
    
    # ═══════════════════════════════════════════════════════════════════
    # DXY CORRELATION FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_dxy_correlation(self) -> Dict:
        """Analyze Gold's correlation with US Dollar Index."""
        
        if self.dxy_price == 0:
            return {
                "dxy_price": 0,
                "correlation": "UNKNOWN",
                "divergence": False,
                "trade_bias": "NEUTRAL"
            }
        
        # Gold should move inverse to DXY
        # If DXY up, Gold should be down (and vice versa)
        
        # TODO: Calculate actual correlation from historical data
        # For now, use simple logic
        
        return {
            "dxy_price": self.dxy_price,
            "correlation": "INVERSE",  # Expected
            "divergence": False,  # Flag if correlation breaks
            "trade_bias": "NEUTRAL"  # Will be set based on DXY movement
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # NEWS TIMING FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_time_since_news(self) -> Dict:
        """Get time since last major news event."""
        
        if self.last_news_time is None:
            return {
                "minutes_since_news": -1,
                "news_active": False,
                "fade_ready": False
            }
        
        now = datetime.utcnow()
        delta = now - self.last_news_time
        minutes = delta.seconds // 60
        
        return {
            "minutes_since_news": minutes,
            "news_active": minutes < 5,
            "fade_ready": 5 <= minutes < 30,  # Good window to fade news spike
            "news_impact_fading": minutes >= 30
        }
    
    def set_last_news_time(self, news_time: datetime):
        """Update last news event time."""
        self.last_news_time = news_time
    
    # ═══════════════════════════════════════════════════════════════════
    # MOMENTUM FEATURES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_momentum_features(self) -> Dict:
        """Calculate momentum indicators."""
        
        if len(self.recent_candles) < 14:
            return {
                "rsi_2": 50, "rsi_14": 50,
                "momentum_bias": "NEUTRAL"
            }
        
        closes = [c.get("close", 0) for c in self.recent_candles[-14:]]
        
        rsi_14 = self._calculate_rsi(closes, 14)
        rsi_2 = self._calculate_rsi(closes[-2:], 2) if len(closes) >= 2 else 50
        
        # Determine bias
        if rsi_2 < 10:
            bias = "EXTREMELY_OVERSOLD"
        elif rsi_2 < 20:
            bias = "OVERSOLD"
        elif rsi_2 > 90:
            bias = "EXTREMELY_OVERBOUGHT"
        elif rsi_2 > 80:
            bias = "OVERBOUGHT"
        else:
            bias = "NEUTRAL"
        
        return {
            "rsi_2": round(rsi_2, 1),
            "rsi_14": round(rsi_14, 1),
            "momentum_bias": bias
        }
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI."""
        if len(prices) < period:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period if gains else 0
        avg_loss = sum(losses[-period:]) / period if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

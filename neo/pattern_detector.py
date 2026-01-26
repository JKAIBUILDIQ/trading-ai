#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO PATTERN DETECTOR
═══════════════════════════════════════════════════════════════════════════════

Pattern recognition for XAUUSD and IREN using:
- Candlestick patterns (engulfing, pin bars, doji, hammer, etc.)
- Chart patterns (flags, double top/bottom, breakouts)
- 90-day historical pattern learning

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatternDetector")


class PatternType(Enum):
    # Candlestick Patterns
    BULL_ENGULFING = "bull_engulfing"
    BEAR_ENGULFING = "bear_engulfing"
    BULL_PIN_BAR = "bull_pin_bar"
    BEAR_PIN_BAR = "bear_pin_bar"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    
    # Chart Patterns
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    
    # Trend Patterns
    HIGHER_HIGHS = "higher_highs"
    LOWER_LOWS = "lower_lows"
    TREND_REVERSAL_UP = "trend_reversal_up"
    TREND_REVERSAL_DOWN = "trend_reversal_down"


@dataclass
class PatternResult:
    pattern_type: PatternType
    confidence: float  # 0-100
    direction: str     # "BUY" or "SELL"
    description: str
    timeframe: str = "H4"
    historical_accuracy: float = 0.0  # From learning


class PatternDetector:
    """
    Detects chart patterns from OHLC data with historical learning.
    Works for both XAUUSD and IREN.
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.min_confidence = 55.0
        self.data_dir = Path("/home/jbot/trading_ai/neo/pattern_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load historical pattern performance
        self.pattern_stats = self._load_pattern_stats()
        
    def _load_pattern_stats(self) -> Dict:
        """Load historical pattern performance"""
        stats_file = self.data_dir / f"{self.symbol.lower()}_pattern_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                return json.load(f)
        return {}
    
    def _save_pattern_stats(self):
        """Save pattern performance stats"""
        stats_file = self.data_dir / f"{self.symbol.lower()}_pattern_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.pattern_stats, f, indent=2)
    
    def analyze(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Analyze OHLC DataFrame for patterns.
        
        Args:
            df: DataFrame with columns: open, high, low, close, time/datetime
                Most recent candle at index -1
        
        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []
        
        if len(df) < 30:
            logger.warning(f"Insufficient data: {len(df)} candles (need 30+)")
            return patterns
        
        # Standardize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Convert to numpy for speed
        opens = df['open'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        
        # Detect all pattern types
        patterns.extend(self._detect_engulfing(opens, highs, lows, closes))
        patterns.extend(self._detect_pin_bars(opens, highs, lows, closes))
        patterns.extend(self._detect_doji(opens, highs, lows, closes))
        patterns.extend(self._detect_hammer_shooting_star(opens, highs, lows, closes))
        patterns.extend(self._detect_three_candle_patterns(opens, highs, lows, closes))
        patterns.extend(self._detect_flag_patterns(opens, highs, lows, closes))
        patterns.extend(self._detect_double_patterns(opens, highs, lows, closes))
        patterns.extend(self._detect_breakouts(opens, highs, lows, closes))
        patterns.extend(self._detect_support_resistance(opens, highs, lows, closes))
        patterns.extend(self._detect_trend_patterns(opens, highs, lows, closes))
        
        # Apply historical accuracy adjustment
        for p in patterns:
            p.historical_accuracy = self._get_historical_accuracy(p.pattern_type.value)
            # Boost confidence if pattern historically accurate
            if p.historical_accuracy > 60:
                p.confidence = min(95, p.confidence * (1 + (p.historical_accuracy - 50) / 100))
            elif p.historical_accuracy > 0 and p.historical_accuracy < 40:
                p.confidence *= 0.8  # Reduce confidence for poor performers
        
        # Filter by minimum confidence
        return [p for p in patterns if p.confidence >= self.min_confidence]
    
    def _get_historical_accuracy(self, pattern_name: str) -> float:
        """Get historical accuracy for a pattern"""
        stats = self.pattern_stats.get(pattern_name, {})
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        if total >= 5:  # Need minimum samples
            return (correct / total) * 100
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CANDLESTICK PATTERNS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_engulfing(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect bullish and bearish engulfing patterns"""
        patterns = []
        
        if len(closes) < 3:
            return patterns
        
        # Previous candle
        prev_open = opens[-2]
        prev_close = closes[-2]
        prev_body = abs(prev_close - prev_open)
        
        # Current candle
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_body = abs(curr_close - curr_open)
        
        # Skip tiny candles
        if prev_body < 0.0001 or curr_body < 0.0001:
            return patterns
        
        # Bullish engulfing
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        engulfs = curr_open <= prev_close and curr_close >= prev_open
        
        if prev_bearish and curr_bullish and engulfs and curr_body > prev_body * 1.1:
            ratio = curr_body / prev_body
            confidence = min(85, 55 + ratio * 10)
            patterns.append(PatternResult(
                pattern_type=PatternType.BULL_ENGULFING,
                confidence=confidence,
                direction="BUY",
                description=f"Bullish engulfing - body {ratio:.1f}x larger"
            ))
        
        # Bearish engulfing
        prev_bullish = prev_close > prev_open
        curr_bearish = curr_close < curr_open
        engulfs = curr_open >= prev_close and curr_close <= prev_open
        
        if prev_bullish and curr_bearish and engulfs and curr_body > prev_body * 1.1:
            ratio = curr_body / prev_body
            confidence = min(85, 55 + ratio * 10)
            patterns.append(PatternResult(
                pattern_type=PatternType.BEAR_ENGULFING,
                confidence=confidence,
                direction="SELL",
                description=f"Bearish engulfing - body {ratio:.1f}x larger"
            ))
        
        return patterns
    
    def _detect_pin_bars(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect pin bar / rejection patterns"""
        patterns = []
        
        if len(closes) < 2:
            return patterns
        
        # Analyze completed candle (-2)
        o, h, l, c = opens[-2], highs[-2], lows[-2], closes[-2]
        
        body = abs(c - o)
        total_range = h - l
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        if body < 0.0001 or total_range < 0.0001:
            return patterns
        
        body_ratio = body / total_range
        
        # Bullish pin bar: long lower wick, small body at top
        if lower_wick >= body * 2.5 and upper_wick < body * 0.5 and body_ratio < 0.35:
            wick_ratio = lower_wick / body
            confidence = min(82, 55 + wick_ratio * 3)
            patterns.append(PatternResult(
                pattern_type=PatternType.BULL_PIN_BAR,
                confidence=confidence,
                direction="BUY",
                description=f"Bullish pin bar - {wick_ratio:.1f}x wick/body ratio"
            ))
        
        # Bearish pin bar: long upper wick, small body at bottom
        if upper_wick >= body * 2.5 and lower_wick < body * 0.5 and body_ratio < 0.35:
            wick_ratio = upper_wick / body
            confidence = min(82, 55 + wick_ratio * 3)
            patterns.append(PatternResult(
                pattern_type=PatternType.BEAR_PIN_BAR,
                confidence=confidence,
                direction="SELL",
                description=f"Bearish pin bar - {wick_ratio:.1f}x wick/body ratio"
            ))
        
        return patterns
    
    def _detect_doji(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect doji patterns (indecision)"""
        patterns = []
        
        if len(closes) < 5:
            return patterns
        
        o, h, l, c = opens[-2], highs[-2], lows[-2], closes[-2]
        
        body = abs(c - o)
        total_range = h - l
        
        if total_range < 0.0001:
            return patterns
        
        body_ratio = body / total_range
        
        # Doji: very small body relative to range
        if body_ratio < 0.1:
            # Check trend context
            prev_closes = closes[-7:-2]
            trend_up = all(prev_closes[i] < prev_closes[i+1] for i in range(len(prev_closes)-1))
            trend_down = all(prev_closes[i] > prev_closes[i+1] for i in range(len(prev_closes)-1))
            
            if trend_up:
                patterns.append(PatternResult(
                    pattern_type=PatternType.DOJI,
                    confidence=65,
                    direction="SELL",
                    description="Doji after uptrend - potential reversal"
                ))
            elif trend_down:
                patterns.append(PatternResult(
                    pattern_type=PatternType.DOJI,
                    confidence=65,
                    direction="BUY",
                    description="Doji after downtrend - potential reversal"
                ))
        
        return patterns
    
    def _detect_hammer_shooting_star(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect hammer and shooting star patterns"""
        patterns = []
        
        if len(closes) < 5:
            return patterns
        
        o, h, l, c = opens[-2], highs[-2], lows[-2], closes[-2]
        
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l
        
        if body < 0.0001 or total_range < 0.0001:
            return patterns
        
        # Check recent trend
        recent_closes = closes[-10:-2]
        trend_down = closes[-7] > closes[-2]  # Downtrend into the candle
        trend_up = closes[-7] < closes[-2]    # Uptrend into the candle
        
        # Hammer: small body, long lower wick, appears after downtrend
        if lower_wick >= body * 2 and upper_wick < body * 0.5 and trend_down:
            patterns.append(PatternResult(
                pattern_type=PatternType.HAMMER,
                confidence=70,
                direction="BUY",
                description="Hammer after downtrend - bullish reversal signal"
            ))
        
        # Shooting star: small body, long upper wick, appears after uptrend
        if upper_wick >= body * 2 and lower_wick < body * 0.5 and trend_up:
            patterns.append(PatternResult(
                pattern_type=PatternType.SHOOTING_STAR,
                confidence=70,
                direction="SELL",
                description="Shooting star after uptrend - bearish reversal signal"
            ))
        
        return patterns
    
    def _detect_three_candle_patterns(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect three white soldiers, three black crows, morning/evening star"""
        patterns = []
        
        if len(closes) < 5:
            return patterns
        
        # Last 3 completed candles
        o1, o2, o3 = opens[-4], opens[-3], opens[-2]
        c1, c2, c3 = closes[-4], closes[-3], closes[-2]
        h1, h2, h3 = highs[-4], highs[-3], highs[-2]
        l1, l2, l3 = lows[-4], lows[-3], lows[-2]
        
        body1 = c1 - o1
        body2 = c2 - o2
        body3 = c3 - o3
        
        # Three white soldiers: 3 consecutive bullish candles with higher closes
        if body1 > 0 and body2 > 0 and body3 > 0:
            if c2 > c1 and c3 > c2 and o2 > o1 and o3 > o2:
                patterns.append(PatternResult(
                    pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                    confidence=75,
                    direction="BUY",
                    description="Three white soldiers - strong bullish continuation"
                ))
        
        # Three black crows: 3 consecutive bearish candles with lower closes
        if body1 < 0 and body2 < 0 and body3 < 0:
            if c2 < c1 and c3 < c2 and o2 < o1 and o3 < o2:
                patterns.append(PatternResult(
                    pattern_type=PatternType.THREE_BLACK_CROWS,
                    confidence=75,
                    direction="SELL",
                    description="Three black crows - strong bearish continuation"
                ))
        
        # Morning star: bearish, small body (doji), bullish
        if body1 < 0 and abs(body2) < abs(body1) * 0.3 and body3 > 0:
            if c3 > (o1 + c1) / 2:  # Third candle closes above midpoint of first
                patterns.append(PatternResult(
                    pattern_type=PatternType.MORNING_STAR,
                    confidence=72,
                    direction="BUY",
                    description="Morning star - bullish reversal pattern"
                ))
        
        # Evening star: bullish, small body (doji), bearish
        if body1 > 0 and abs(body2) < abs(body1) * 0.3 and body3 < 0:
            if c3 < (o1 + c1) / 2:  # Third candle closes below midpoint of first
                patterns.append(PatternResult(
                    pattern_type=PatternType.EVENING_STAR,
                    confidence=72,
                    direction="SELL",
                    description="Evening star - bearish reversal pattern"
                ))
        
        return patterns
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHART PATTERNS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_flag_patterns(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect bull and bear flag patterns"""
        patterns = []
        
        if len(closes) < 15:
            return patterns
        
        # Look at last 15 bars
        recent = closes[-15:]
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        # Pole detection (first 7 bars)
        pole_start_low = min(recent_lows[:7])
        pole_end_high = max(recent_highs[:7])
        pole_start_high = max(recent_highs[:7])
        pole_end_low = min(recent_lows[:7])
        
        bull_pole = pole_end_high - pole_start_low
        bear_pole = pole_start_high - pole_end_low
        
        # Flag detection (last 8 bars)
        flag_highs = recent_highs[7:]
        flag_lows = recent_lows[7:]
        flag_range = max(flag_highs) - min(flag_lows)
        
        # Bull flag: strong up move, tight consolidation, breakout up
        if bull_pole > 0 and flag_range < bull_pole * 0.5:
            if closes[-1] > max(flag_highs[:-1]):  # Breakout
                confidence = min(80, 55 + (bull_pole / flag_range) * 3)
                patterns.append(PatternResult(
                    pattern_type=PatternType.BULL_FLAG,
                    confidence=confidence,
                    direction="BUY",
                    description=f"Bull flag breakout - pole {bull_pole:.1f}, flag {flag_range:.1f}"
                ))
        
        # Bear flag: strong down move, tight consolidation, breakout down
        if bear_pole > 0 and flag_range < bear_pole * 0.5:
            if closes[-1] < min(flag_lows[:-1]):  # Breakdown
                confidence = min(80, 55 + (bear_pole / flag_range) * 3)
                patterns.append(PatternResult(
                    pattern_type=PatternType.BEAR_FLAG,
                    confidence=confidence,
                    direction="SELL",
                    description=f"Bear flag breakdown - pole {bear_pole:.1f}, flag {flag_range:.1f}"
                ))
        
        return patterns
    
    def _detect_double_patterns(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(closes) < 30:
            return patterns
        
        recent_highs = highs[-30:]
        recent_lows = lows[-30:]
        
        # Find peaks and troughs
        high1_idx = np.argmax(recent_highs[:15])
        high2_idx = 15 + np.argmax(recent_highs[15:])
        low1_idx = np.argmin(recent_lows[:15])
        low2_idx = 15 + np.argmin(recent_lows[15:])
        
        high1 = recent_highs[high1_idx]
        high2 = recent_highs[high2_idx]
        low1 = recent_lows[low1_idx]
        low2 = recent_lows[low2_idx]
        
        # Double bottom: two similar lows
        if abs(low1 - low2) / low1 < 0.02:  # Within 2%
            # Find neckline (peak between lows)
            if low1_idx < low2_idx:
                neckline = max(recent_highs[low1_idx:low2_idx])
                if closes[-1] > neckline:
                    patterns.append(PatternResult(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=72,
                        direction="BUY",
                        description=f"Double bottom at {low1:.2f}/{low2:.2f}, neckline broken"
                    ))
        
        # Double top: two similar highs
        if abs(high1 - high2) / high1 < 0.02:  # Within 2%
            if high1_idx < high2_idx:
                neckline = min(recent_lows[high1_idx:high2_idx])
                if closes[-1] < neckline:
                    patterns.append(PatternResult(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=72,
                        direction="SELL",
                        description=f"Double top at {high1:.2f}/{high2:.2f}, neckline broken"
                    ))
        
        return patterns
    
    def _detect_breakouts(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect range breakouts"""
        patterns = []
        
        if len(closes) < 25:
            return patterns
        
        # Define range (bars -25 to -5, excluding recent)
        range_highs = highs[-25:-3]
        range_lows = lows[-25:-3]
        
        range_high = max(range_highs)
        range_low = min(range_lows)
        range_size = range_high - range_low
        
        if range_size < 1:  # Minimum range
            return patterns
        
        current = closes[-1]
        
        # Bullish breakout
        if current > range_high:
            breakout_pct = (current - range_high) / range_size * 100
            if breakout_pct > 5:  # Meaningful breakout
                confidence = min(78, 55 + breakout_pct)
                patterns.append(PatternResult(
                    pattern_type=PatternType.BREAKOUT_UP,
                    confidence=confidence,
                    direction="BUY",
                    description=f"Breakout {breakout_pct:.1f}% above range high {range_high:.2f}"
                ))
        
        # Bearish breakout
        if current < range_low:
            breakout_pct = (range_low - current) / range_size * 100
            if breakout_pct > 5:
                confidence = min(78, 55 + breakout_pct)
                patterns.append(PatternResult(
                    pattern_type=PatternType.BREAKOUT_DOWN,
                    confidence=confidence,
                    direction="SELL",
                    description=f"Breakdown {breakout_pct:.1f}% below range low {range_low:.2f}"
                ))
        
        return patterns
    
    def _detect_support_resistance(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect support bounces and resistance rejections"""
        patterns = []
        
        if len(closes) < 50:
            return patterns
        
        # Find key levels from last 50 bars
        all_highs = highs[-50:]
        all_lows = lows[-50:]
        
        # Simple support/resistance: areas with multiple touches
        price_levels = np.concatenate([all_highs, all_lows])
        
        # Current price
        current = closes[-1]
        prev_low = lows[-2]
        prev_high = highs[-2]
        
        # Check for bounce off support (recent low near historical lows)
        historical_lows = sorted(all_lows)[:10]  # Bottom 10 lows
        support_zone = np.mean(historical_lows)
        
        if prev_low <= support_zone * 1.01 and current > prev_low:
            patterns.append(PatternResult(
                pattern_type=PatternType.SUPPORT_BOUNCE,
                confidence=68,
                direction="BUY",
                description=f"Support bounce near {support_zone:.2f}"
            ))
        
        # Check for rejection at resistance
        historical_highs = sorted(all_highs, reverse=True)[:10]
        resistance_zone = np.mean(historical_highs)
        
        if prev_high >= resistance_zone * 0.99 and current < prev_high:
            patterns.append(PatternResult(
                pattern_type=PatternType.RESISTANCE_REJECTION,
                confidence=68,
                direction="SELL",
                description=f"Resistance rejection near {resistance_zone:.2f}"
            ))
        
        return patterns
    
    def _detect_trend_patterns(self, opens, highs, lows, closes) -> List[PatternResult]:
        """Detect trend patterns (higher highs, lower lows, reversals)"""
        patterns = []
        
        if len(closes) < 20:
            return patterns
        
        # Check for higher highs / higher lows (uptrend)
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Find swing points
        hh_count = sum(1 for i in range(2, len(recent_highs)) 
                       if recent_highs[i] > recent_highs[i-2])
        ll_count = sum(1 for i in range(2, len(recent_lows)) 
                       if recent_lows[i] < recent_lows[i-2])
        
        if hh_count >= 3:
            patterns.append(PatternResult(
                pattern_type=PatternType.HIGHER_HIGHS,
                confidence=65,
                direction="BUY",
                description=f"Higher highs pattern - {hh_count} consecutive"
            ))
        
        if ll_count >= 3:
            patterns.append(PatternResult(
                pattern_type=PatternType.LOWER_LOWS,
                confidence=65,
                direction="SELL",
                description=f"Lower lows pattern - {ll_count} consecutive"
            ))
        
        # Trend reversal detection
        older_trend = closes[-20] - closes[-10]  # Older segment
        recent_trend = closes[-5] - closes[-1]   # Recent segment
        
        if older_trend < 0 and recent_trend > 0 and abs(recent_trend) > abs(older_trend) * 0.5:
            patterns.append(PatternResult(
                pattern_type=PatternType.TREND_REVERSAL_UP,
                confidence=62,
                direction="BUY",
                description="Potential trend reversal to upside"
            ))
        
        if older_trend > 0 and recent_trend < 0 and abs(recent_trend) > abs(older_trend) * 0.5:
            patterns.append(PatternResult(
                pattern_type=PatternType.TREND_REVERSAL_DOWN,
                confidence=62,
                direction="SELL",
                description="Potential trend reversal to downside"
            ))
        
        return patterns
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEARNING & STATS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_pattern_outcome(self, pattern_type: str, was_correct: bool):
        """Record whether a pattern prediction was correct"""
        if pattern_type not in self.pattern_stats:
            self.pattern_stats[pattern_type] = {"total": 0, "correct": 0}
        
        self.pattern_stats[pattern_type]["total"] += 1
        if was_correct:
            self.pattern_stats[pattern_type]["correct"] += 1
        
        self._save_pattern_stats()
        
        accuracy = self.pattern_stats[pattern_type]["correct"] / self.pattern_stats[pattern_type]["total"] * 100
        logger.info(f"📊 {pattern_type} accuracy: {accuracy:.1f}% ({self.pattern_stats[pattern_type]['total']} samples)")
    
    def get_combined_signal(self, patterns: List[PatternResult]) -> Dict:
        """Combine multiple patterns into single signal"""
        if not patterns:
            return {
                "direction": "NEUTRAL",
                "confidence": 0,
                "patterns": [],
                "pattern_count": 0,
                "reasoning": "No patterns detected"
            }
        
        buy_score = 0
        sell_score = 0
        buy_patterns = []
        sell_patterns = []
        
        for p in patterns:
            weight = 1.0
            if p.historical_accuracy > 60:
                weight = 1.2
            elif p.historical_accuracy > 0 and p.historical_accuracy < 40:
                weight = 0.7
            
            if p.direction == "BUY":
                buy_score += p.confidence * weight
                buy_patterns.append(f"{p.pattern_type.value} ({p.confidence:.0f}%)")
            else:
                sell_score += p.confidence * weight
                sell_patterns.append(f"{p.pattern_type.value} ({p.confidence:.0f}%)")
        
        if buy_score > sell_score * 1.2:  # Need 20% stronger for clear signal
            avg_conf = buy_score / len([p for p in patterns if p.direction == "BUY"])
            return {
                "direction": "BUY",
                "confidence": min(90, avg_conf),
                "patterns": buy_patterns,
                "pattern_count": len(buy_patterns),
                "reasoning": f"Bullish: {', '.join(buy_patterns)}"
            }
        elif sell_score > buy_score * 1.2:
            avg_conf = sell_score / len([p for p in patterns if p.direction == "SELL"])
            return {
                "direction": "SELL",
                "confidence": min(90, avg_conf),
                "patterns": sell_patterns,
                "pattern_count": len(sell_patterns),
                "reasoning": f"Bearish: {', '.join(sell_patterns)}"
            }
        else:
            return {
                "direction": "NEUTRAL",
                "confidence": 0,
                "patterns": buy_patterns + sell_patterns,
                "pattern_count": len(patterns),
                "reasoning": "Mixed signals - no clear direction"
            }
    
    def get_stats(self) -> Dict:
        """Get pattern performance statistics"""
        stats = {}
        for pattern, data in self.pattern_stats.items():
            if data["total"] >= 3:
                stats[pattern] = {
                    "total": data["total"],
                    "correct": data["correct"],
                    "accuracy": data["correct"] / data["total"] * 100
                }
        return dict(sorted(stats.items(), key=lambda x: x[1]["accuracy"], reverse=True))


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_xauusd(df: pd.DataFrame) -> Dict:
    """Quick analysis for Gold"""
    detector = PatternDetector("XAUUSD")
    patterns = detector.analyze(df)
    return detector.get_combined_signal(patterns)

def analyze_iren(df: pd.DataFrame) -> Dict:
    """Quick analysis for IREN"""
    detector = PatternDetector("IREN")
    patterns = detector.analyze(df)
    return detector.get_combined_signal(patterns)


if __name__ == "__main__":
    # Test with sample data
    print("🔍 Pattern Detector Test")
    print("=" * 50)
    
    # Create sample data
    import random
    random.seed(42)
    
    base_price = 5000
    data = []
    for i in range(50):
        o = base_price + random.uniform(-20, 20)
        c = o + random.uniform(-15, 15)
        h = max(o, c) + random.uniform(0, 10)
        l = min(o, c) - random.uniform(0, 10)
        data.append({"open": o, "high": h, "low": l, "close": c})
        base_price = c
    
    df = pd.DataFrame(data)
    
    detector = PatternDetector("XAUUSD")
    patterns = detector.analyze(df)
    
    print(f"\nDetected {len(patterns)} patterns:")
    for p in patterns:
        print(f"  • {p.pattern_type.value}: {p.direction} ({p.confidence:.0f}%)")
        print(f"    {p.description}")
    
    signal = detector.get_combined_signal(patterns)
    print(f"\n📊 Combined Signal: {signal['direction']} ({signal['confidence']:.0f}%)")
    print(f"   {signal['reasoning']}")

"""
NEO-GOLD Pattern Detection
Gold-specific chart patterns and market structure
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .config import (
    ASIAN_RANGE_BREAKOUT_THRESHOLD, SWEEP_REVERSAL_CANDLES,
    TRIPLE_TAP_TOLERANCE_PIPS, logger
)


class PatternType(Enum):
    """Gold-specific pattern types."""
    ASIAN_CONSOLIDATION = "asian_consolidation"
    LONDON_BREAKOUT = "london_breakout"
    ROUND_NUMBER_SWEEP = "round_number_sweep"
    NEWS_SPIKE_FADE = "news_spike_fade"
    USD_DIVERGENCE = "usd_divergence"
    STOP_HUNT = "stop_hunt"
    TRIPLE_TAP = "triple_tap"
    SESSION_OVERLAP_VOLATILITY = "session_overlap_volatility"
    NONE = "none"


@dataclass
class DetectedPattern:
    """A detected trading pattern."""
    type: PatternType
    confidence: int  # 0-100
    direction: str  # "BUY" or "SELL"
    description: str
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit: float
    timestamp: str


class GoldPatternDetector:
    """
    Detects Gold-specific trading patterns.
    
    Patterns:
    1. ASIAN_CONSOLIDATION: Price ranging during Asia session
    2. LONDON_BREAKOUT: First decisive move after London open
    3. ROUND_NUMBER_SWEEP: Price touches $XX00/$XX50 and reverses
    4. NEWS_SPIKE_FADE: Large candle on news → reversal opportunity
    5. USD_DIVERGENCE: DXY moves but Gold doesn't follow
    6. STOP_HUNT: Quick wick below support/above resistance
    7. TRIPLE_TAP: 3 tests of same level → breakout imminent
    8. SESSION_OVERLAP_VOLATILITY: London+NY overlap big moves
    """
    
    def __init__(self):
        self.recent_candles: List[Dict] = []
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.detected_patterns: List[DetectedPattern] = []
        
    def detect_all(self, candles: List[Dict], features: Dict) -> List[DetectedPattern]:
        """Run all pattern detection algorithms."""
        
        self.recent_candles = candles
        self.detected_patterns = []
        
        current_price = features.get("price", 0)
        session = features.get("session", "")
        asian_range = features.get("asian_range", {})
        round_numbers = features.get("round_number", {})
        
        # Calculate support/resistance from recent price action
        self._calculate_levels()
        
        # Run each pattern detector
        self._detect_asian_consolidation(asian_range)
        self._detect_london_breakout(session, asian_range, current_price)
        self._detect_round_number_sweep(current_price, round_numbers)
        self._detect_stop_hunt(current_price)
        self._detect_triple_tap(current_price)
        self._detect_session_overlap(session, features.get("volatility", {}))
        
        # Sort by confidence
        self.detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"🔍 Patterns detected: {len(self.detected_patterns)}")
        for p in self.detected_patterns[:3]:
            logger.info(f"   • {p.type.value}: {p.direction} ({p.confidence}%)")
        
        return self.detected_patterns
    
    def _calculate_levels(self):
        """Calculate support and resistance levels from price action."""
        if len(self.recent_candles) < 20:
            return
        
        highs = [c.get("high", 0) for c in self.recent_candles[-50:]]
        lows = [c.get("low", 0) for c in self.recent_candles[-50:]]
        
        # Simple pivot-based S/R
        self.resistance_levels = []
        self.support_levels = []
        
        for i in range(2, len(highs) - 2):
            # Resistance: local maximum
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                self.resistance_levels.append(highs[i])
            
            # Support: local minimum
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                self.support_levels.append(lows[i])
        
        # Keep most recent/relevant
        self.resistance_levels = sorted(self.resistance_levels, reverse=True)[:5]
        self.support_levels = sorted(self.support_levels)[:5]
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 1: ASIAN CONSOLIDATION
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_asian_consolidation(self, asian_range: Dict):
        """Detect tight Asian range consolidation."""
        
        range_pips = asian_range.get("range_pips", 0)
        
        if range_pips == 0:
            return
        
        if range_pips < ASIAN_RANGE_BREAKOUT_THRESHOLD:
            high = asian_range.get("high", 0)
            low = asian_range.get("low", 0)
            
            pattern = DetectedPattern(
                type=PatternType.ASIAN_CONSOLIDATION,
                confidence=min(90, int(100 - range_pips)),  # Tighter = higher confidence
                direction="PENDING",  # Wait for breakout direction
                description=f"Tight Asian range ({range_pips:.0f} pips). "
                           f"Expect London breakout. Range: ${low:.2f} - ${high:.2f}",
                entry_zone=(low, high),
                stop_loss=low - 5 if True else high + 5,  # Depends on direction
                take_profit=0,  # Set after breakout
                timestamp=datetime.utcnow().isoformat()
            )
            self.detected_patterns.append(pattern)
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 2: LONDON BREAKOUT
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_london_breakout(self, session: str, asian_range: Dict, current_price: float):
        """Detect London session breakout of Asian range."""
        
        if session != "LONDON":
            return
        
        high = asian_range.get("high", 0)
        low = asian_range.get("low", 0)
        
        if high == 0 or low == 0:
            return
        
        # Check if price has broken Asian range
        breakout_margin = 3.0  # $3 clear break
        
        if current_price > high + breakout_margin:
            pattern = DetectedPattern(
                type=PatternType.LONDON_BREAKOUT,
                confidence=75,
                direction="BUY",
                description=f"London breakout ABOVE Asian range ${high:.2f}. "
                           f"Bullish momentum confirmed.",
                entry_zone=(high, high + 5),
                stop_loss=high - 5,  # Below breakout level
                take_profit=high + 20,  # ~$20 target
                timestamp=datetime.utcnow().isoformat()
            )
            self.detected_patterns.append(pattern)
            
        elif current_price < low - breakout_margin:
            pattern = DetectedPattern(
                type=PatternType.LONDON_BREAKOUT,
                confidence=75,
                direction="SELL",
                description=f"London breakout BELOW Asian range ${low:.2f}. "
                           f"Bearish momentum confirmed.",
                entry_zone=(low - 5, low),
                stop_loss=low + 5,
                take_profit=low - 20,
                timestamp=datetime.utcnow().isoformat()
            )
            self.detected_patterns.append(pattern)
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 3: ROUND NUMBER SWEEP
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_round_number_sweep(self, current_price: float, round_numbers: Dict):
        """Detect sweep of round number level and reversal."""
        
        if not round_numbers.get("is_near"):
            return
        
        nearest = round_numbers.get("nearest", 0)
        position = round_numbers.get("position", "")
        distance_pips = round_numbers.get("distance_pips", 0)
        
        if distance_pips > 30:  # Too far from level
            return
        
        # Look for sweep pattern in recent candles
        if len(self.recent_candles) < SWEEP_REVERSAL_CANDLES:
            return
        
        recent = self.recent_candles[-SWEEP_REVERSAL_CANDLES:]
        
        # Check if any candle wicked through the round number
        for candle in recent:
            low = candle.get("low", 0)
            high = candle.get("high", 0)
            close = candle.get("close", 0)
            
            # Sweep below round number and close above
            if low < nearest and close > nearest:
                pattern = DetectedPattern(
                    type=PatternType.ROUND_NUMBER_SWEEP,
                    confidence=80,
                    direction="BUY",
                    description=f"Round number sweep at ${nearest:.0f}. "
                               f"Price swept below and closed above. Bullish.",
                    entry_zone=(nearest, nearest + 3),
                    stop_loss=low - 2,
                    take_profit=nearest + 15,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.detected_patterns.append(pattern)
                return
            
            # Sweep above round number and close below
            if high > nearest and close < nearest:
                pattern = DetectedPattern(
                    type=PatternType.ROUND_NUMBER_SWEEP,
                    confidence=80,
                    direction="SELL",
                    description=f"Round number sweep at ${nearest:.0f}. "
                               f"Price swept above and closed below. Bearish.",
                    entry_zone=(nearest - 3, nearest),
                    stop_loss=high + 2,
                    take_profit=nearest - 15,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.detected_patterns.append(pattern)
                return
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 4: STOP HUNT
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_stop_hunt(self, current_price: float):
        """Detect stop hunt patterns (quick wick and reversal)."""
        
        if len(self.recent_candles) < 5:
            return
        
        # Look at last few candles for long wicks
        for i, candle in enumerate(self.recent_candles[-5:]):
            high = candle.get("high", 0)
            low = candle.get("low", 0)
            open_price = candle.get("open", 0)
            close = candle.get("close", 0)
            
            body = abs(close - open_price)
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low
            
            # Long lower wick (stop hunt below)
            if lower_wick > body * 2 and lower_wick > 3:  # >$3 wick
                # Check if near support
                for support in self.support_levels:
                    if abs(low - support) < 3:  # Swept support
                        pattern = DetectedPattern(
                            type=PatternType.STOP_HUNT,
                            confidence=78,
                            direction="BUY",
                            description=f"Stop hunt below support ${support:.2f}. "
                                       f"Long wick shows rejection. Expect reversal up.",
                            entry_zone=(close, close + 2),
                            stop_loss=low - 1,
                            take_profit=close + 12,
                            timestamp=datetime.utcnow().isoformat()
                        )
                        self.detected_patterns.append(pattern)
                        return
            
            # Long upper wick (stop hunt above)
            if upper_wick > body * 2 and upper_wick > 3:
                for resistance in self.resistance_levels:
                    if abs(high - resistance) < 3:
                        pattern = DetectedPattern(
                            type=PatternType.STOP_HUNT,
                            confidence=78,
                            direction="SELL",
                            description=f"Stop hunt above resistance ${resistance:.2f}. "
                                       f"Long wick shows rejection. Expect reversal down.",
                            entry_zone=(close - 2, close),
                            stop_loss=high + 1,
                            take_profit=close - 12,
                            timestamp=datetime.utcnow().isoformat()
                        )
                        self.detected_patterns.append(pattern)
                        return
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 5: TRIPLE TAP
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_triple_tap(self, current_price: float):
        """Detect triple test of same level (breakout setup)."""
        
        if len(self.recent_candles) < 20:
            return
        
        highs = [c.get("high", 0) for c in self.recent_candles[-30:]]
        lows = [c.get("low", 0) for c in self.recent_candles[-30:]]
        
        tolerance = TRIPLE_TAP_TOLERANCE_PIPS / 10  # Convert to dollars
        
        # Check for triple top (resistance)
        for resistance in self.resistance_levels:
            touches = sum(1 for h in highs if abs(h - resistance) < tolerance)
            if touches >= 3:
                pattern = DetectedPattern(
                    type=PatternType.TRIPLE_TAP,
                    confidence=72,
                    direction="SELL",  # Expect rejection or wait for breakout
                    description=f"Triple tap at resistance ${resistance:.2f}. "
                               f"{touches} tests. Expect strong reaction.",
                    entry_zone=(resistance - 3, resistance),
                    stop_loss=resistance + 5,
                    take_profit=resistance - 15,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.detected_patterns.append(pattern)
                break
        
        # Check for triple bottom (support)
        for support in self.support_levels:
            touches = sum(1 for l in lows if abs(l - support) < tolerance)
            if touches >= 3:
                pattern = DetectedPattern(
                    type=PatternType.TRIPLE_TAP,
                    confidence=72,
                    direction="BUY",
                    description=f"Triple tap at support ${support:.2f}. "
                               f"{touches} tests. Expect strong bounce or break.",
                    entry_zone=(support, support + 3),
                    stop_loss=support - 5,
                    take_profit=support + 15,
                    timestamp=datetime.utcnow().isoformat()
                )
                self.detected_patterns.append(pattern)
                break
    
    # ═══════════════════════════════════════════════════════════════════
    # PATTERN 6: SESSION OVERLAP VOLATILITY
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_session_overlap(self, session: str, volatility: Dict):
        """Detect London/NY overlap setup for big moves."""
        
        if session != "OVERLAP_LONDON_NY":
            return
        
        regime = volatility.get("regime", "NORMAL")
        atr = volatility.get("atr", 25)
        
        pattern = DetectedPattern(
            type=PatternType.SESSION_OVERLAP_VOLATILITY,
            confidence=70,
            direction="PENDING",  # Wait for signal
            description=f"London/NY overlap active. Volatility: {regime}. "
                       f"ATR: ${atr:.2f}. Expect big moves. Size appropriately.",
            entry_zone=(0, 0),  # Use other patterns for entry
            stop_loss=0,
            take_profit=0,
            timestamp=datetime.utcnow().isoformat()
        )
        self.detected_patterns.append(pattern)
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_highest_confidence_pattern(self) -> Optional[DetectedPattern]:
        """Get the most confident pattern."""
        if not self.detected_patterns:
            return None
        return self.detected_patterns[0]
    
    def get_actionable_patterns(self, min_confidence: int = 70) -> List[DetectedPattern]:
        """Get patterns that meet confidence threshold."""
        return [p for p in self.detected_patterns 
                if p.confidence >= min_confidence and p.direction != "PENDING"]

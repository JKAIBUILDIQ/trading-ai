"""
NEO Herd Detector - Think Like the Herd!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHT: Most retail traders BUY the same bots from:
- MQL5 Market (top sellers)
- ForexFactory (popular free EAs)
- YouTube "best EA" recommendations
- GitHub (starred repos)

Result: THOUSANDS of accounts running IDENTICAL logic!

This module predicts:
- Where the herd will enter
- Where their stops are clustered
- When they're exhausted (fade opportunity)
- When to front-run them

"When everyone is on one side of the boat, be ready to swim."

Created: 2026-01-23
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# ============================================================================
# COMMON RETAIL BOT SETTINGS (Reverse-Engineered)
# ============================================================================

COMMON_INDICATORS = {
    # RSI - Used by ~80% of retail bots
    'RSI': {
        'period': 14,
        'oversold': 30,
        'overbought': 70,
        'popularity': 0.80
    },
    'RSI2': {
        'period': 2,
        'oversold': 10,
        'overbought': 90,
        'popularity': 0.30
    },
    
    # EMA Cross - Used by ~70% of bots
    'EMA_CROSS': {
        'fast': 20,
        'slow': 50,
        'popularity': 0.70
    },
    'EMA_CROSS_FAST': {
        'fast': 9,
        'slow': 21,
        'popularity': 0.50
    },
    
    # Bollinger Bands - Used by ~60%
    'BOLLINGER': {
        'period': 20,
        'std': 2.0,
        'popularity': 0.60
    },
    
    # MACD - Used by ~65%
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'popularity': 0.65
    },
    
    # Stochastic - Used by ~50%
    'STOCHASTIC': {
        'k_period': 14,
        'd_period': 3,
        'slowing': 3,
        'oversold': 20,
        'overbought': 80,
        'popularity': 0.50
    },
    
    # ATR for stops - Used by ~75%
    'ATR': {
        'period': 14,
        'sl_multiplier': [1.5, 2.0, 3.0],
        'popularity': 0.75
    }
}

# Round numbers that all bots watch
GOLD_ROUND_NUMBERS = [
    4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200
]

# Common fixed pip stop distances for Gold
COMMON_STOP_DISTANCES = [15, 20, 25, 30, 50, 100]  # Points

# Fibonacci levels everyone uses
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]


@dataclass
class HerdSignal:
    """Single herd indicator signal"""
    indicator: str
    action: str  # 'BUY', 'SELL', 'WATCH', 'NONE'
    strength: float  # 0-1 (how many bots use this)
    reason: str
    value: float = 0.0


@dataclass
class HerdAnalysis:
    """Complete herd analysis result"""
    
    # Individual signals
    signals: List[HerdSignal] = field(default_factory=list)
    
    # Aggregated metrics
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0
    net_direction: str = 'NEUTRAL'
    herd_strength: float = 0.0  # 0-100
    
    # Stop clusters
    stop_clusters_long: List[float] = field(default_factory=list)
    stop_clusters_short: List[float] = field(default_factory=list)
    
    # Exhaustion
    is_exhausted: bool = False
    exhaustion_type: str = ''  # 'BULLISH_EXHAUSTION', 'BEARISH_EXHAUSTION'
    
    # Recommendations
    strategy: str = ''  # 'FADE', 'FRONT_RUN', 'RIDE', 'WAIT'
    confidence: float = 0.0
    reasoning: str = ''
    
    # Summary
    summary: str = ''


class HerdDetector:
    """
    Predict where retail algo bots will trade
    
    Most retail traders use the SAME indicators with SAME settings
    because they all bought the SAME bots!
    
    Usage:
        detector = HerdDetector()
        analysis = detector.analyze(ohlcv_data)
        
        if analysis.herd_strength > 80 and analysis.is_exhausted:
            # FADE THE HERD!
            pass
    """
    
    def __init__(self, symbol: str = 'XAUUSD'):
        """Initialize with symbol-specific settings"""
        self.symbol = symbol
        
        # Gold-specific round numbers
        if 'XAU' in symbol or 'GOLD' in symbol:
            self.round_numbers = GOLD_ROUND_NUMBERS
            self.point_value = 1.0  # Gold uses points
        else:
            # Forex pairs
            self.round_numbers = []  # Calculate dynamically
            self.point_value = 0.0001  # Standard forex pip
    
    def analyze(self, ohlcv: pd.DataFrame, volume: pd.Series = None) -> HerdAnalysis:
        """
        Perform complete herd analysis
        
        Args:
            ohlcv: DataFrame with OHLCV data
            volume: Optional separate volume series
            
        Returns:
            HerdAnalysis with predictions and recommendations
        """
        # Ensure lowercase columns
        ohlcv.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in ohlcv.columns]
        
        if volume is None and 'volume' in ohlcv.columns:
            volume = ohlcv['volume']
        
        current_price = ohlcv['close'].iloc[-1]
        
        # Step 1: Get individual indicator signals
        signals = self._get_all_signals(ohlcv)
        
        # Step 2: Calculate aggregate pressure
        buy_pressure = sum(s.strength for s in signals if s.action == 'BUY')
        sell_pressure = sum(s.strength for s in signals if s.action == 'SELL')
        
        total_pressure = buy_pressure + sell_pressure
        if total_pressure > 0:
            herd_strength = abs(buy_pressure - sell_pressure) / total_pressure * 100
        else:
            herd_strength = 0
        
        net_direction = 'BUY' if buy_pressure > sell_pressure else 'SELL' if sell_pressure > buy_pressure else 'NEUTRAL'
        
        # Step 3: Find stop clusters
        stop_clusters_long, stop_clusters_short = self._find_stop_clusters(ohlcv, current_price)
        
        # Step 4: Check for exhaustion
        is_exhausted, exhaustion_type = self._detect_exhaustion(ohlcv, volume, signals)
        
        # Step 5: Generate recommendation
        strategy, confidence, reasoning = self._generate_strategy(
            signals, buy_pressure, sell_pressure, 
            herd_strength, is_exhausted, exhaustion_type, volume
        )
        
        # Build summary
        summary = self._build_summary(
            net_direction, herd_strength, len(signals), 
            is_exhausted, strategy, current_price
        )
        
        return HerdAnalysis(
            signals=signals,
            buy_pressure=round(buy_pressure, 2),
            sell_pressure=round(sell_pressure, 2),
            net_direction=net_direction,
            herd_strength=round(herd_strength, 1),
            stop_clusters_long=stop_clusters_long,
            stop_clusters_short=stop_clusters_short,
            is_exhausted=is_exhausted,
            exhaustion_type=exhaustion_type,
            strategy=strategy,
            confidence=round(confidence, 1),
            reasoning=reasoning,
            summary=summary
        )
    
    def _get_all_signals(self, ohlcv: pd.DataFrame) -> List[HerdSignal]:
        """Get signals from all common retail indicators"""
        signals = []
        
        # RSI(14) - 80% of bots
        signals.append(self._check_rsi(ohlcv, 14, 30, 70, 0.80))
        
        # RSI(2) - 30% of bots (Connors RSI)
        signals.append(self._check_rsi(ohlcv, 2, 10, 90, 0.30))
        
        # EMA Cross 20/50 - 70% of bots
        signals.append(self._check_ema_cross(ohlcv, 20, 50, 0.70))
        
        # EMA Cross 9/21 - 50% of bots (scalpers)
        signals.append(self._check_ema_cross(ohlcv, 9, 21, 0.50))
        
        # Bollinger Bands - 60% of bots
        signals.append(self._check_bollinger(ohlcv, 20, 2.0, 0.60))
        
        # MACD - 65% of bots
        signals.append(self._check_macd(ohlcv, 12, 26, 9, 0.65))
        
        # Stochastic - 50% of bots
        signals.append(self._check_stochastic(ohlcv, 14, 3, 3, 0.50))
        
        # Round Numbers - 90% of bots watch these
        signals.append(self._check_round_numbers(ohlcv, 0.90))
        
        # Fibonacci Retracements - 40% of bots
        signals.append(self._check_fibonacci(ohlcv, 0.40))
        
        # Filter out NONE signals
        return [s for s in signals if s.action != 'NONE']
    
    def _check_rsi(self, ohlcv: pd.DataFrame, period: int, 
                   oversold: float, overbought: float, popularity: float) -> HerdSignal:
        """Check RSI signal"""
        close = ohlcv['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        
        if np.isnan(rsi_value):
            return HerdSignal(f'RSI({period})', 'NONE', 0, '', 0)
        
        if rsi_value < oversold:
            return HerdSignal(
                f'RSI({period})', 'BUY', popularity,
                f'RSI({period}) at {rsi_value:.1f} < {oversold} (oversold)',
                rsi_value
            )
        elif rsi_value > overbought:
            return HerdSignal(
                f'RSI({period})', 'SELL', popularity,
                f'RSI({period}) at {rsi_value:.1f} > {overbought} (overbought)',
                rsi_value
            )
        
        return HerdSignal(f'RSI({period})', 'NONE', 0, '', rsi_value)
    
    def _check_ema_cross(self, ohlcv: pd.DataFrame, fast: int, slow: int, 
                         popularity: float) -> HerdSignal:
        """Check EMA crossover signal"""
        close = ohlcv['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]
        
        # Golden cross (fast crosses above slow)
        if prev_fast <= prev_slow and current_fast > current_slow:
            return HerdSignal(
                f'EMA({fast}/{slow})', 'BUY', popularity,
                f'Golden cross: EMA{fast} crossed above EMA{slow}',
                current_fast - current_slow
            )
        
        # Death cross (fast crosses below slow)
        if prev_fast >= prev_slow and current_fast < current_slow:
            return HerdSignal(
                f'EMA({fast}/{slow})', 'SELL', popularity,
                f'Death cross: EMA{fast} crossed below EMA{slow}',
                current_fast - current_slow
            )
        
        # Trend continuation
        if current_fast > current_slow:
            return HerdSignal(
                f'EMA({fast}/{slow})', 'WATCH', popularity * 0.3,
                f'Bullish trend: EMA{fast} > EMA{slow}',
                current_fast - current_slow
            )
        elif current_fast < current_slow:
            return HerdSignal(
                f'EMA({fast}/{slow})', 'WATCH', popularity * 0.3,
                f'Bearish trend: EMA{fast} < EMA{slow}',
                current_fast - current_slow
            )
        
        return HerdSignal(f'EMA({fast}/{slow})', 'NONE', 0, '', 0)
    
    def _check_bollinger(self, ohlcv: pd.DataFrame, period: int, 
                         std: float, popularity: float) -> HerdSignal:
        """Check Bollinger Band signal"""
        close = ohlcv['close']
        
        sma = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        # Price below lower band = oversold
        if current_price < current_lower:
            return HerdSignal(
                f'BB({period},{std})', 'BUY', popularity,
                f'Price ${current_price:.2f} below lower BB ${current_lower:.2f}',
                (current_price - current_lower)
            )
        
        # Price above upper band = overbought
        if current_price > current_upper:
            return HerdSignal(
                f'BB({period},{std})', 'SELL', popularity,
                f'Price ${current_price:.2f} above upper BB ${current_upper:.2f}',
                (current_price - current_upper)
            )
        
        return HerdSignal(f'BB({period},{std})', 'NONE', 0, '', 0)
    
    def _check_macd(self, ohlcv: pd.DataFrame, fast: int, slow: int, 
                    signal: int, popularity: float) -> HerdSignal:
        """Check MACD signal"""
        close = ohlcv['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        # MACD cross above signal
        if prev_macd <= prev_signal and current_macd > current_signal:
            return HerdSignal(
                'MACD', 'BUY', popularity,
                'MACD crossed above signal line',
                current_macd - current_signal
            )
        
        # MACD cross below signal
        if prev_macd >= prev_signal and current_macd < current_signal:
            return HerdSignal(
                'MACD', 'SELL', popularity,
                'MACD crossed below signal line',
                current_macd - current_signal
            )
        
        return HerdSignal('MACD', 'NONE', 0, '', current_macd - current_signal)
    
    def _check_stochastic(self, ohlcv: pd.DataFrame, k_period: int, 
                          d_period: int, slowing: int, popularity: float) -> HerdSignal:
        """Check Stochastic signal"""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        raw_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k = raw_k.rolling(slowing).mean()
        d = k.rolling(d_period).mean()
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        
        if np.isnan(current_k):
            return HerdSignal('Stochastic', 'NONE', 0, '', 0)
        
        # Oversold
        if current_k < 20 and current_d < 20:
            return HerdSignal(
                'Stochastic', 'BUY', popularity,
                f'Stochastic oversold: K={current_k:.1f}, D={current_d:.1f}',
                current_k
            )
        
        # Overbought
        if current_k > 80 and current_d > 80:
            return HerdSignal(
                'Stochastic', 'SELL', popularity,
                f'Stochastic overbought: K={current_k:.1f}, D={current_d:.1f}',
                current_k
            )
        
        return HerdSignal('Stochastic', 'NONE', 0, '', current_k)
    
    def _check_round_numbers(self, ohlcv: pd.DataFrame, popularity: float) -> HerdSignal:
        """Check if price is near round number (EVERYONE watches these)"""
        current_price = ohlcv['close'].iloc[-1]
        
        # Dynamic round numbers if not set
        if not self.round_numbers:
            base = int(current_price / 100) * 100
            self.round_numbers = [base + i * 50 for i in range(-2, 5)]
        
        for level in self.round_numbers:
            distance = abs(current_price - level)
            if distance < 10:  # Within 10 points
                return HerdSignal(
                    'ROUND_NUMBER', 'WATCH', popularity,
                    f'Price ${current_price:.2f} near round number ${level}',
                    current_price - level
                )
        
        return HerdSignal('ROUND_NUMBER', 'NONE', 0, '', 0)
    
    def _check_fibonacci(self, ohlcv: pd.DataFrame, popularity: float) -> HerdSignal:
        """Check Fibonacci retracement levels"""
        # Use last 50 candles for swing high/low
        recent = ohlcv.tail(50)
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        current_price = ohlcv['close'].iloc[-1]
        
        # Calculate fib levels
        fib_range = swing_high - swing_low
        
        for level in FIB_LEVELS:
            fib_price = swing_high - (fib_range * level)
            
            # Check if price is near fib level
            if abs(current_price - fib_price) < fib_range * 0.02:  # Within 2%
                # 0.618 is the golden ratio - strong level
                strength = popularity * (1.5 if level == 0.618 else 1.0)
                return HerdSignal(
                    f'FIB_{level}', 'WATCH', min(strength, 1.0),
                    f'Price ${current_price:.2f} at {level*100:.1f}% Fib (${fib_price:.2f})',
                    current_price - fib_price
                )
        
        return HerdSignal('FIB', 'NONE', 0, '', 0)
    
    def _find_stop_clusters(self, ohlcv: pd.DataFrame, 
                            current_price: float) -> Tuple[List[float], List[float]]:
        """Find where the herd's stop losses are likely clustered"""
        
        stop_clusters_long = []  # Stops for long positions (below price)
        stop_clusters_short = []  # Stops for short positions (above price)
        
        # 1. Fixed pip stops (very common)
        for distance in COMMON_STOP_DISTANCES:
            stop_clusters_long.append(round(current_price - distance, 2))
            stop_clusters_short.append(round(current_price + distance, 2))
        
        # 2. ATR-based stops
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        if not np.isnan(atr):
            for mult in [1.5, 2.0, 2.5, 3.0]:
                stop_clusters_long.append(round(current_price - atr * mult, 2))
                stop_clusters_short.append(round(current_price + atr * mult, 2))
        
        # 3. Below recent swing lows / Above recent swing highs
        recent = ohlcv.tail(20)
        recent_low = recent['low'].min()
        recent_high = recent['high'].max()
        
        stop_clusters_long.append(round(recent_low - 10, 2))
        stop_clusters_long.append(round(recent_low - 20, 2))
        stop_clusters_short.append(round(recent_high + 10, 2))
        stop_clusters_short.append(round(recent_high + 20, 2))
        
        # 4. Round numbers (very common stop placement)
        for level in self.round_numbers:
            if level < current_price:
                stop_clusters_long.append(level - 5)
            else:
                stop_clusters_short.append(level + 5)
        
        # Remove duplicates and sort
        stop_clusters_long = sorted(list(set(stop_clusters_long)), reverse=True)
        stop_clusters_short = sorted(list(set(stop_clusters_short)))
        
        return stop_clusters_long[:10], stop_clusters_short[:10]  # Top 10 each
    
    def _detect_exhaustion(self, ohlcv: pd.DataFrame, volume: pd.Series,
                           signals: List[HerdSignal]) -> Tuple[bool, str]:
        """
        Detect if the herd is running out of steam
        
        Key Insight: When ALL indicators say BUY but volume is dying
        = Everyone who wants to buy has bought
        = FADE THE HERD!
        """
        
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        
        # Check volume trend
        volume_declining = False
        if volume is not None and len(volume) >= 10:
            vol_sma_5 = volume.tail(5).mean()
            vol_sma_20 = volume.tail(20).mean()
            volume_declining = vol_sma_5 < vol_sma_20 * 0.8
        
        # Bullish exhaustion: Many buy signals but volume dying
        if len(buy_signals) >= 3 and volume_declining:
            return True, 'BULLISH_EXHAUSTION'
        
        # Bearish exhaustion: Many sell signals but volume dying
        if len(sell_signals) >= 3 and volume_declining:
            return True, 'BEARISH_EXHAUSTION'
        
        # Price exhaustion (extended from MA)
        close = ohlcv['close']
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        current = close.iloc[-1]
        
        deviation = (current - ema_50) / ema_50 * 100
        
        if deviation > 5 and len(buy_signals) >= 2:  # 5%+ above EMA50
            return True, 'BULLISH_EXHAUSTION'
        if deviation < -5 and len(sell_signals) >= 2:  # 5%+ below EMA50
            return True, 'BEARISH_EXHAUSTION'
        
        return False, ''
    
    def _generate_strategy(self, signals: List[HerdSignal], 
                           buy_pressure: float, sell_pressure: float,
                           herd_strength: float, is_exhausted: bool,
                           exhaustion_type: str, volume: pd.Series) -> Tuple[str, float, str]:
        """Generate trading strategy recommendation"""
        
        # Check volume trend
        vol_rising = False
        if volume is not None and len(volume) >= 10:
            vol_sma_5 = volume.tail(5).mean()
            vol_sma_20 = volume.tail(20).mean()
            vol_rising = vol_sma_5 > vol_sma_20 * 1.2
        
        # Strategy 1: FADE the exhausted herd
        if is_exhausted and herd_strength > 60:
            if exhaustion_type == 'BULLISH_EXHAUSTION':
                return (
                    'FADE_SELL', 
                    min(90, herd_strength), 
                    f"Herd is bullish ({buy_pressure:.1f} strength) but EXHAUSTED - FADE with SELL! "
                    "All the buyers have bought, reversal imminent."
                )
            else:
                return (
                    'FADE_BUY',
                    min(90, herd_strength),
                    f"Herd is bearish ({sell_pressure:.1f} strength) but EXHAUSTED - FADE with BUY! "
                    "All the sellers have sold, bounce imminent."
                )
        
        # Strategy 2: RIDE with strong herd + volume
        if herd_strength > 70 and vol_rising and not is_exhausted:
            direction = 'BUY' if buy_pressure > sell_pressure else 'SELL'
            return (
                f'RIDE_{direction}',
                min(85, herd_strength),
                f"Herd is strongly {direction.lower()} ({herd_strength:.0f}% aligned) "
                f"and volume RISING - RIDE with tight trailing stop!"
            )
        
        # Strategy 3: Moderate herd signal
        if herd_strength > 50:
            direction = 'BUY' if buy_pressure > sell_pressure else 'SELL'
            return (
                f'CAUTIOUS_{direction}',
                min(70, herd_strength),
                f"Moderate herd {direction.lower()} pressure ({herd_strength:.0f}%) - "
                "Join cautiously with reduced size."
            )
        
        # Strategy 4: No clear herd direction
        return (
            'WAIT',
            30,
            f"Herd is confused (strength only {herd_strength:.0f}%) - "
            "Wait for clearer signal."
        )
    
    def _build_summary(self, net_direction: str, herd_strength: float,
                       num_signals: int, is_exhausted: bool,
                       strategy: str, current_price: float) -> str:
        """Build human-readable summary"""
        lines = []
        lines.append("🐑 HERD DETECTOR ANALYSIS")
        lines.append(f"   Price: ${current_price:.2f}")
        lines.append(f"   Herd Direction: {net_direction}")
        lines.append(f"   Herd Strength: {herd_strength:.0f}%")
        lines.append(f"   Active Signals: {num_signals}")
        
        if is_exhausted:
            lines.append(f"   ⚠️ EXHAUSTION DETECTED!")
        
        lines.append(f"   Strategy: {strategy}")
        
        return "\n".join(lines)
    
    def get_stop_hunt_probability(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Predict probability of stop hunt
        
        MMs love to sweep stops before reversing!
        """
        current_price = ohlcv['close'].iloc[-1]
        stop_long, stop_short = self._find_stop_clusters(ohlcv, current_price)
        
        # Find nearest stop cluster
        nearest_long = max(stop_long) if stop_long else 0
        nearest_short = min(stop_short) if stop_short else 0
        
        dist_to_long_stops = current_price - nearest_long if nearest_long else float('inf')
        dist_to_short_stops = nearest_short - current_price if nearest_short else float('inf')
        
        # Probability based on distance
        prob_long_hunt = max(0, 100 - dist_to_long_stops * 2) if dist_to_long_stops != float('inf') else 0
        prob_short_hunt = max(0, 100 - dist_to_short_stops * 2) if dist_to_short_stops != float('inf') else 0
        
        return {
            'long_stops_at': nearest_long,
            'short_stops_at': nearest_short,
            'distance_to_long_stops': round(dist_to_long_stops, 2),
            'distance_to_short_stops': round(dist_to_short_stops, 2),
            'prob_long_hunt': round(prob_long_hunt, 1),
            'prob_short_hunt': round(prob_short_hunt, 1),
            'hunt_direction': 'DOWN' if prob_long_hunt > prob_short_hunt else 'UP',
            'warning': f"⚠️ Stop hunt likely {'below' if prob_long_hunt > prob_short_hunt else 'above'}!" 
                      if max(prob_long_hunt, prob_short_hunt) > 50 else ""
        }


def get_herd_analysis(ohlcv: pd.DataFrame, symbol: str = 'XAUUSD') -> Dict:
    """
    Quick function for NEO integration
    """
    detector = HerdDetector(symbol)
    analysis = detector.analyze(ohlcv)
    
    return {
        'net_direction': analysis.net_direction,
        'herd_strength': analysis.herd_strength,
        'buy_pressure': analysis.buy_pressure,
        'sell_pressure': analysis.sell_pressure,
        'is_exhausted': analysis.is_exhausted,
        'exhaustion_type': analysis.exhaustion_type,
        'strategy': analysis.strategy,
        'confidence': analysis.confidence,
        'reasoning': analysis.reasoning,
        'stop_clusters_long': analysis.stop_clusters_long[:5],
        'stop_clusters_short': analysis.stop_clusters_short[:5],
        'signals': [{'indicator': s.indicator, 'action': s.action, 
                    'strength': s.strength, 'reason': s.reason} 
                   for s in analysis.signals],
        'summary': analysis.summary
    }


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("🐑 HERD DETECTOR TEST")
    print("="*60)
    
    # Get Gold data
    gold = yf.download('GC=F', period='3mo', interval='1d', progress=False)
    
    # Handle MultiIndex columns
    if hasattr(gold.columns, 'levels'):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]
    
    detector = HerdDetector('XAUUSD')
    analysis = detector.analyze(gold)
    
    print(f"\n{analysis.summary}")
    
    print(f"\n📊 INDIVIDUAL SIGNALS:")
    for sig in analysis.signals:
        emoji = "📈" if sig.action == "BUY" else "📉" if sig.action == "SELL" else "👀"
        print(f"  {emoji} {sig.indicator}: {sig.action} (strength: {sig.strength:.0%})")
        print(f"      {sig.reason}")
    
    print(f"\n🎯 STRATEGY: {analysis.strategy}")
    print(f"   Confidence: {analysis.confidence}%")
    print(f"   {analysis.reasoning}")
    
    print(f"\n🛑 STOP CLUSTERS:")
    print(f"   Longs' stops at: {analysis.stop_clusters_long[:5]}")
    print(f"   Shorts' stops at: {analysis.stop_clusters_short[:5]}")
    
    # Stop hunt analysis
    print("\n" + "="*60)
    print("🎯 STOP HUNT PROBABILITY")
    print("="*60)
    hunt = detector.get_stop_hunt_probability(gold)
    print(f"   Nearest long stops: ${hunt['long_stops_at']} ({hunt['distance_to_long_stops']} pts away)")
    print(f"   Nearest short stops: ${hunt['short_stops_at']} ({hunt['distance_to_short_stops']} pts away)")
    print(f"   Probability of hunt DOWN: {hunt['prob_long_hunt']}%")
    print(f"   Probability of hunt UP: {hunt['prob_short_hunt']}%")
    if hunt['warning']:
        print(f"   {hunt['warning']}")

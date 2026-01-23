#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
MM DETECTOR - Market Maker Hunting Pattern Detection
═══════════════════════════════════════════════════════════════════════════════

Detects when Market Makers have completed hunting retail algos:
- Stop Hunt Detection (V-bottom/inverse patterns)
- False Breakout Detection (liquidity grabs)
- Session Trap Detection (Asian→London reversals)
- Liquidity Pool Analysis (where stops cluster)

Philosophy: Don't trade WITH the herd. Trade AFTER the herd gets trapped.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("MM-DETECTOR")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StopHuntSignal:
    """Signal generated after stop hunt detection"""
    signal_type: str          # LONG_HUNT_COMPLETE, SHORT_HUNT_COMPLETE
    direction: str            # BUY, SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    hunt_level: float         # Level where stops were hunted
    reclaim_level: float      # Level where price reclaimed
    volume_spike: float       # Volume ratio vs average
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class FalseBreakoutSignal:
    """Signal generated after false breakout detection"""
    signal_type: str          # FALSE_BREAK_RESISTANCE, FALSE_BREAK_SUPPORT
    direction: str            # BUY, SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    breakout_level: float     # Level that was "broken"
    failure_candles: int      # How many candles before failure
    trapped_direction: str    # LONGS_TRAPPED, SHORTS_TRAPPED
    timestamp: datetime = None


@dataclass
class LiquidityPool:
    """Area where retail stops likely cluster"""
    level: float
    pool_type: str            # LONG_STOPS, SHORT_STOPS
    strength: float           # 0-100 based on confluence
    reasons: List[str]        # Why stops cluster here
    last_hunted: datetime = None


# ═══════════════════════════════════════════════════════════════════════════════
# MM DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MMDetector:
    """
    Detects Market Maker hunting patterns and generates contrarian signals.
    
    Core Principle: Enter AFTER the trap, not INTO the trap.
    """
    
    def __init__(self, lookback: int = 50, volume_spike_threshold: float = 1.5):
        self.lookback = lookback
        self.volume_spike_threshold = volume_spike_threshold
        self.recent_hunts: List[StopHuntSignal] = []
        self.liquidity_pools: List[LiquidityPool] = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STOP HUNT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_stop_hunt(self, df: pd.DataFrame) -> Optional[StopHuntSignal]:
        """
        Detect completed stop hunt patterns.
        
        Stop Hunt Anatomy:
        1. Price spikes through obvious level (hunting stops)
        2. Volume spikes (stops being triggered)
        3. Immediate reversal (MMs taking profits)
        4. Price reclaims the level (trap complete)
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
        
        Returns:
            StopHuntSignal if hunt detected, None otherwise
        """
        if len(df) < self.lookback:
            return None
        
        recent = df.tail(self.lookback).copy()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate key levels
        swing_low = recent['low'].min()
        swing_high = recent['high'].max()
        avg_volume = recent['volume'].mean()
        avg_range = (recent['high'] - recent['low']).mean()
        
        # Check for LONG STOP HUNT (Bullish Signal)
        # Pattern: Spike below swing low, then strong close back above
        long_hunt_detected = (
            # Current candle spiked below recent swing low
            current['low'] < swing_low and
            # But closed back above the swing low (reclaim)
            current['close'] > swing_low and
            # Strong bullish body (buyers stepping in)
            current['close'] > current['open'] and
            # Body is significant (not just a wick)
            (current['close'] - current['open']) > avg_range * 0.3 and
            # Volume spike (stops triggered)
            current['volume'] > avg_volume * self.volume_spike_threshold
        )
        
        if long_hunt_detected:
            # Calculate entry, SL, TP
            entry = current['close']
            stop_loss = current['low'] - (avg_range * 0.5)  # Below the hunt level
            take_profit = entry + (entry - stop_loss) * 2.5  # 1:2.5 R:R
            
            return StopHuntSignal(
                signal_type="LONG_HUNT_COMPLETE",
                direction="BUY",
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(85, 50 + (current['volume'] / avg_volume) * 10),
                reason=f"Stop hunt below ${swing_low:.2f}, price reclaimed with {current['volume']/avg_volume:.1f}x volume",
                hunt_level=swing_low,
                reclaim_level=current['close'],
                volume_spike=current['volume'] / avg_volume
            )
        
        # Check for SHORT STOP HUNT (Bearish Signal)
        # Pattern: Spike above swing high, then strong close back below
        short_hunt_detected = (
            # Current candle spiked above recent swing high
            current['high'] > swing_high and
            # But closed back below the swing high (rejection)
            current['close'] < swing_high and
            # Strong bearish body (sellers stepping in)
            current['close'] < current['open'] and
            # Body is significant
            (current['open'] - current['close']) > avg_range * 0.3 and
            # Volume spike
            current['volume'] > avg_volume * self.volume_spike_threshold
        )
        
        if short_hunt_detected:
            entry = current['close']
            stop_loss = current['high'] + (avg_range * 0.5)  # Above the hunt level
            take_profit = entry - (stop_loss - entry) * 2.5  # 1:2.5 R:R
            
            return StopHuntSignal(
                signal_type="SHORT_HUNT_COMPLETE",
                direction="SELL",
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=min(85, 50 + (current['volume'] / avg_volume) * 10),
                reason=f"Stop hunt above ${swing_high:.2f}, price rejected with {current['volume']/avg_volume:.1f}x volume",
                hunt_level=swing_high,
                reclaim_level=current['close'],
                volume_spike=current['volume'] / avg_volume
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FALSE BREAKOUT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_false_breakout(
        self, 
        df: pd.DataFrame, 
        level: float, 
        level_type: str = "resistance"
    ) -> Optional[FalseBreakoutSignal]:
        """
        Detect false breakout (liquidity grab) patterns.
        
        False Breakout Anatomy:
        1. Price breaks through key level (triggers breakout bots)
        2. Holds for 1-3 candles (retail enters)
        3. Fails to continue (no follow-through)
        4. Reverses back through level (trapping breakout traders)
        
        Args:
            df: OHLCV DataFrame
            level: The support/resistance level
            level_type: "resistance" or "support"
        
        Returns:
            FalseBreakoutSignal if detected, None otherwise
        """
        if len(df) < 10:
            return None
        
        last_5 = df.tail(5).copy()
        current = df.iloc[-1]
        avg_range = (df.tail(20)['high'] - df.tail(20)['low']).mean()
        
        if level_type == "resistance":
            # Check for false breakout above resistance
            # Pattern: Broke above, then failed back below
            
            # Did we break above in recent candles?
            broke_above = any(last_5.iloc[i]['high'] > level for i in range(len(last_5) - 1))
            
            # Are we now back below?
            failed_back = current['close'] < level
            
            # Volume declining (no follow-through buying)
            volume_declining = current['volume'] < last_5['volume'].mean() * 0.8
            
            # Bearish candle on failure
            bearish_close = current['close'] < current['open']
            
            if broke_above and failed_back and (volume_declining or bearish_close):
                # Count how many candles held above
                candles_above = sum(1 for i in range(len(last_5)) if last_5.iloc[i]['close'] > level)
                
                entry = current['close']
                stop_loss = last_5['high'].max() + (avg_range * 0.3)
                take_profit = entry - (stop_loss - entry) * 2.0
                
                return FalseBreakoutSignal(
                    signal_type="FALSE_BREAK_RESISTANCE",
                    direction="SELL",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=min(80, 55 + (5 - candles_above) * 5),
                    reason=f"False breakout above ${level:.2f}, longs trapped, {candles_above} candles before failure",
                    breakout_level=level,
                    failure_candles=candles_above,
                    trapped_direction="LONGS_TRAPPED"
                )
        
        elif level_type == "support":
            # Check for false breakdown below support
            # Pattern: Broke below, then reclaimed
            
            broke_below = any(last_5.iloc[i]['low'] < level for i in range(len(last_5) - 1))
            reclaimed = current['close'] > level
            volume_spike = current['volume'] > last_5['volume'].mean() * 1.2
            bullish_close = current['close'] > current['open']
            
            if broke_below and reclaimed and (volume_spike or bullish_close):
                candles_below = sum(1 for i in range(len(last_5)) if last_5.iloc[i]['close'] < level)
                
                entry = current['close']
                stop_loss = last_5['low'].min() - (avg_range * 0.3)
                take_profit = entry + (entry - stop_loss) * 2.0
                
                return FalseBreakoutSignal(
                    signal_type="FALSE_BREAK_SUPPORT",
                    direction="BUY",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=min(80, 55 + (5 - candles_below) * 5),
                    reason=f"False breakdown below ${level:.2f}, shorts trapped, {candles_below} candles before reclaim",
                    breakout_level=level,
                    failure_candles=candles_below,
                    trapped_direction="SHORTS_TRAPPED"
                )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDITY POOL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_liquidity_pools(self, df: pd.DataFrame, current_price: float) -> List[LiquidityPool]:
        """
        Identify areas where retail stops likely cluster.
        
        Common Stop Placement Patterns:
        1. Below/above recent swing lows/highs
        2. Round numbers ($X000, $X50)
        3. ATR-based distances from entry
        4. Previous day high/low
        
        Returns list of liquidity pools sorted by proximity to current price.
        """
        pools = []
        recent = df.tail(self.lookback)
        
        # 1. Swing Low Pool (Long stops below here)
        swing_low = recent['low'].min()
        swing_low_reasons = [
            "Recent swing low",
            "Standard stop placement for longs",
            "Breakout trader stops"
        ]
        pools.append(LiquidityPool(
            level=swing_low - 5,  # Just below the swing
            pool_type="LONG_STOPS",
            strength=85,
            reasons=swing_low_reasons
        ))
        
        # 2. Swing High Pool (Short stops above here)
        swing_high = recent['high'].max()
        swing_high_reasons = [
            "Recent swing high",
            "Standard stop placement for shorts",
            "Breakout short stops"
        ]
        pools.append(LiquidityPool(
            level=swing_high + 5,
            pool_type="SHORT_STOPS",
            strength=85,
            reasons=swing_high_reasons
        ))
        
        # 3. Round Number Pools
        base_100 = int(current_price / 100) * 100
        for level in [base_100 - 50, base_100, base_100 + 50, base_100 + 100]:
            if level < current_price:
                pools.append(LiquidityPool(
                    level=level,
                    pool_type="LONG_STOPS",
                    strength=70,
                    reasons=["Round psychological level", "Common stop placement"]
                ))
            else:
                pools.append(LiquidityPool(
                    level=level,
                    pool_type="SHORT_STOPS",
                    strength=70,
                    reasons=["Round psychological level", "Common stop placement"]
                ))
        
        # 4. ATR-based pools (1.5x and 2x ATR from swing)
        atr = (recent['high'] - recent['low']).mean()
        atr_long_stop = swing_low - (atr * 1.5)
        atr_short_stop = swing_high + (atr * 1.5)
        
        pools.append(LiquidityPool(
            level=atr_long_stop,
            pool_type="LONG_STOPS",
            strength=60,
            reasons=["1.5x ATR stop level", "Algorithmic bot standard"]
        ))
        pools.append(LiquidityPool(
            level=atr_short_stop,
            pool_type="SHORT_STOPS",
            strength=60,
            reasons=["1.5x ATR stop level", "Algorithmic bot standard"]
        ))
        
        # Sort by proximity to current price
        pools.sort(key=lambda x: abs(x.level - current_price))
        
        return pools
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION TRAP DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_session_trap(
        self, 
        df: pd.DataFrame,
        current_session: str,
        prev_session_direction: str
    ) -> Optional[Dict]:
        """
        Detect when Asian session traders are about to be trapped by London.
        
        Pattern:
        - Asian session establishes a direction (low volume)
        - London opens with volume against Asian direction
        - Asian session longs/shorts get trapped
        
        Args:
            df: OHLCV data
            current_session: "ASIAN", "LONDON", "OVERLAP", "NY"
            prev_session_direction: "BULLISH" or "BEARISH"
        
        Returns:
            Trap signal if London is reversing Asian, None otherwise
        """
        if current_session != "LONDON":
            return None
        
        # Get last 4 hours of data (approximate London open)
        recent_4h = df.tail(4)
        
        if len(recent_4h) < 4:
            return None
        
        # Check volume spike at London open
        london_volume = recent_4h['volume'].mean()
        pre_london_volume = df.tail(8).head(4)['volume'].mean()
        volume_spike = london_volume / pre_london_volume if pre_london_volume > 0 else 1
        
        # Check if London is reversing Asian direction
        london_direction = "BULLISH" if recent_4h.iloc[-1]['close'] > recent_4h.iloc[0]['open'] else "BEARISH"
        
        if london_direction != prev_session_direction and volume_spike > 1.5:
            # Trap detected!
            trapped = "LONGS" if prev_session_direction == "BULLISH" else "SHORTS"
            
            return {
                "trap_type": "SESSION_TRAP",
                "trapped": trapped,
                "direction": "SELL" if trapped == "LONGS" else "BUY",
                "confidence": min(75, 50 + volume_spike * 10),
                "reason": f"London reversing Asian {prev_session_direction}, {trapped} trapped, {volume_spike:.1f}x volume spike",
                "asian_direction": prev_session_direction,
                "london_direction": london_direction,
                "volume_spike": volume_spike
            }
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPREHENSIVE MM ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_mm_analysis(
        self, 
        df: pd.DataFrame,
        current_price: float,
        current_session: str = None,
        key_levels: List[float] = None
    ) -> Dict:
        """
        Run full MM detection suite and return comprehensive analysis.
        
        Args:
            df: OHLCV DataFrame (at least 50 candles)
            current_price: Current price
            current_session: Current trading session
            key_levels: List of key support/resistance levels
        
        Returns:
            Dictionary with all MM analysis results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": current_price,
            "stop_hunt": None,
            "false_breakouts": [],
            "liquidity_pools": [],
            "session_trap": None,
            "contrarian_signal": None,
            "warnings": [],
            "opportunities": []
        }
        
        # 1. Check for stop hunt
        stop_hunt = self.detect_stop_hunt(df)
        if stop_hunt:
            results["stop_hunt"] = {
                "type": stop_hunt.signal_type,
                "direction": stop_hunt.direction,
                "entry": stop_hunt.entry_price,
                "stop_loss": stop_hunt.stop_loss,
                "take_profit": stop_hunt.take_profit,
                "confidence": stop_hunt.confidence,
                "reason": stop_hunt.reason
            }
            results["contrarian_signal"] = results["stop_hunt"]
            results["opportunities"].append(f"🎯 {stop_hunt.reason}")
        
        # 2. Check for false breakouts at key levels
        if key_levels:
            for level in key_levels:
                level_type = "resistance" if level > current_price else "support"
                fb = self.detect_false_breakout(df, level, level_type)
                if fb:
                    fb_dict = {
                        "type": fb.signal_type,
                        "direction": fb.direction,
                        "level": fb.breakout_level,
                        "entry": fb.entry_price,
                        "stop_loss": fb.stop_loss,
                        "take_profit": fb.take_profit,
                        "confidence": fb.confidence,
                        "reason": fb.reason
                    }
                    results["false_breakouts"].append(fb_dict)
                    if not results["contrarian_signal"]:
                        results["contrarian_signal"] = fb_dict
                    results["opportunities"].append(f"🎯 {fb.reason}")
        
        # 3. Find liquidity pools
        pools = self.find_liquidity_pools(df, current_price)
        for pool in pools[:5]:  # Top 5 nearest
            results["liquidity_pools"].append({
                "level": pool.level,
                "type": pool.pool_type,
                "strength": pool.strength,
                "reasons": pool.reasons,
                "distance": abs(pool.level - current_price)
            })
        
        # Add warnings for nearby pools
        for pool in pools[:3]:
            distance = abs(pool.level - current_price)
            if distance < 30:  # Within 30 points
                direction = "below" if pool.level < current_price else "above"
                results["warnings"].append(
                    f"⚠️ {pool.pool_type} pool {direction} at ${pool.level:.2f} ({distance:.0f} pts away)"
                )
        
        # 4. Session trap (if session info provided)
        if current_session:
            # Would need previous session direction from caller
            # results["session_trap"] = self.detect_session_trap(...)
            pass
        
        # 5. Generate summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary of MM analysis."""
        lines = [
            "═══════════════════════════════════════════════════════════════════",
            "🦊 MARKET MAKER ANALYSIS (Contrarian Intelligence)",
            "═══════════════════════════════════════════════════════════════════",
            ""
        ]
        
        if results.get("stop_hunt"):
            sh = results["stop_hunt"]
            lines.append(f"🎯 STOP HUNT DETECTED:")
            lines.append(f"   Signal: {sh['direction']} at ${sh['entry']:.2f}")
            lines.append(f"   Confidence: {sh['confidence']:.0f}%")
            lines.append(f"   Reason: {sh['reason']}")
            lines.append("")
        
        if results.get("false_breakouts"):
            lines.append(f"🪤 FALSE BREAKOUTS:")
            for fb in results["false_breakouts"]:
                lines.append(f"   • {fb['direction']} signal at ${fb['entry']:.2f}")
                lines.append(f"     {fb['reason']}")
            lines.append("")
        
        if results.get("liquidity_pools"):
            lines.append("💧 NEARBY LIQUIDITY POOLS:")
            for pool in results["liquidity_pools"][:3]:
                lines.append(f"   • ${pool['level']:.2f} ({pool['type']}) - {pool['distance']:.0f} pts away")
            lines.append("")
        
        if results.get("warnings"):
            lines.append("⚠️ WARNINGS:")
            for w in results["warnings"]:
                lines.append(f"   {w}")
            lines.append("")
        
        if results.get("opportunities"):
            lines.append("✅ OPPORTUNITIES:")
            for o in results["opportunities"]:
                lines.append(f"   {o}")
        
        if results.get("contrarian_signal"):
            cs = results["contrarian_signal"]
            lines.append("")
            lines.append("═══════════════════════════════════════════════════════════════════")
            lines.append(f"🦊 CONTRARIAN SIGNAL: {cs['direction']} at ${cs['entry']:.2f}")
            lines.append(f"   SL: ${cs['stop_loss']:.2f} | TP: ${cs['take_profit']:.2f}")
            lines.append(f"   Confidence: {cs['confidence']:.0f}%")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_detector = None

def get_detector() -> MMDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = MMDetector()
    return _detector


def get_mm_analysis(
    df: pd.DataFrame,
    current_price: float,
    current_session: str = None,
    key_levels: List[float] = None
) -> Dict:
    """
    Convenience function to get full MM analysis.
    
    Usage:
        from mm_detector import get_mm_analysis
        analysis = get_mm_analysis(df, current_price=4950)
    """
    detector = get_detector()
    return detector.get_mm_analysis(df, current_price, current_session, key_levels)


def detect_stop_hunt(df: pd.DataFrame) -> Optional[StopHuntSignal]:
    """Convenience function to detect stop hunts."""
    return get_detector().detect_stop_hunt(df)


def find_liquidity_pools(df: pd.DataFrame, current_price: float) -> List[LiquidityPool]:
    """Convenience function to find liquidity pools."""
    return get_detector().find_liquidity_pools(df, current_price)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🦊 MM DETECTOR - Market Maker Hunting Pattern Detection")
    print("=" * 60)
    
    # Create sample data for testing
    import yfinance as yf
    
    print("\n📊 Fetching XAUUSD data...")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="1mo", interval="1h")
    
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
        current_price = df['close'].iloc[-1]
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Data points: {len(df)}")
        
        # Run analysis
        analysis = get_mm_analysis(
            df=df,
            current_price=current_price,
            key_levels=[current_price - 50, current_price + 50]
        )
        
        print("\n" + analysis["summary"])
    else:
        print("❌ Could not fetch data")

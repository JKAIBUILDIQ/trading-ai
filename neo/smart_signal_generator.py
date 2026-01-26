#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO SMART SIGNAL GENERATOR - Anti-Retail Logic
═══════════════════════════════════════════════════════════════════════════════

THIS IS THE MISSING PIECE!

The old NEO logic:
    if rsi > 70: SELL  ← Every bot does this!
    if rsi < 30: BUY   ← Every bot does this!

The NEW logic:
    1. Check if trend is STRONG (don't fight it!)
    2. Check if MM hunt is complete (enter AFTER the trap)
    3. Check if herd is exhausted (fade when everyone agrees)
    4. Multi-timeframe confirmation (H1+H4+D1 must align)
    5. THEN and ONLY THEN generate signal

"If you trade like the herd, you get hunted like the herd."

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Import the smart modules that NEO WASN'T USING!
try:
    from mm_detector import MMDetector, get_mm_analysis
    MM_DETECTOR_AVAILABLE = True
except ImportError:
    MM_DETECTOR_AVAILABLE = False

try:
    from crowd_psychology import analyze_crowd_psychology, CrowdPsychology, adjust_signal_for_crowd
    CROWD_PSYCHOLOGY_AVAILABLE = True
except ImportError:
    CROWD_PSYCHOLOGY_AVAILABLE = False

try:
    from herd_detector import HerdDetector, get_herd_analysis
    HERD_DETECTOR_AVAILABLE = True
except ImportError:
    HERD_DETECTOR_AVAILABLE = False

try:
    from algo_hype_index import get_algo_hype_index
    ALGO_HYPE_AVAILABLE = True
except ImportError:
    ALGO_HYPE_AVAILABLE = False

# Signal learner for historical learning
try:
    from signal_learner import (
        SignalLearner, SignalOutcome, get_learner,
        get_pattern_confidence, detect_breakout, record_signal_outcome
    )
    LEARNER_AVAILABLE = True
except ImportError:
    LEARNER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartSignal")


@dataclass
class SmartSignal:
    """A signal that's smarter than retail"""
    
    # Basic info
    symbol: str
    direction: str  # BUY, SELL, WAIT
    confidence: float  # 0-100
    
    # Entry parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_pct: float = 2.0  # % of account
    
    # Intelligence layers
    trend_aligned: bool = False
    mm_hunt_complete: bool = False
    herd_exhausted: bool = False
    mtf_confirmed: bool = False
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # Reasoning
    reasoning: str = ""
    
    # Meta
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class SmartSignalGenerator:
    """
    Generate signals that are DIFFERENT from retail bots.
    
    Key Rules:
    1. NEVER fight a strong trend (RSI overbought in uptrend = WAIT, not SELL!)
    2. WAIT for MM hunt to complete (enter after the trap, not into it)
    3. FADE the herd only when exhausted (everyone bullish + volume dying = SELL)
    4. Multi-timeframe confirmation (all timeframes must agree)
    5. Avoid crowded trades (Algo Hype Index)
    6. LEARN from mistakes (historical pattern performance)
    7. Handle breakouts intelligently (real vs fake)
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        
        # Initialize detectors
        self.mm_detector = MMDetector() if MM_DETECTOR_AVAILABLE else None
        self.herd_detector = HerdDetector(symbol) if HERD_DETECTOR_AVAILABLE else None
        
        # Initialize learner for historical pattern learning
        self.learner = get_learner() if LEARNER_AVAILABLE else None
        
        # State tracking
        self.last_signal_time = None
        self.signal_cooldown_minutes = 60  # Don't spam signals
        
        # Performance tracking
        self.signals_generated = 0
        self.correct_signals = 0
        
        logger.info("=" * 60)
        logger.info("🧠 SMART SIGNAL GENERATOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   MM Detector: {'✅ ACTIVE' if self.mm_detector else '❌ UNAVAILABLE'}")
        logger.info(f"   Herd Detector: {'✅ ACTIVE' if self.herd_detector else '❌ UNAVAILABLE'}")
        logger.info(f"   Crowd Psychology: {'✅ ACTIVE' if CROWD_PSYCHOLOGY_AVAILABLE else '❌ UNAVAILABLE'}")
        logger.info(f"   Algo Hype Index: {'✅ ACTIVE' if ALGO_HYPE_AVAILABLE else '❌ UNAVAILABLE'}")
        logger.info(f"   Signal Learner: {'✅ ACTIVE' if self.learner else '❌ UNAVAILABLE'}")
        logger.info("=" * 60)
    
    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _get_trend_strength(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Determine trend direction and strength.
        
        Returns: (direction, strength_0_100)
        
        CRITICAL: Don't fight a strong trend!
        - ADX > 25 = trending
        - ADX > 40 = strong trend (DON'T FIGHT IT!)
        """
        close = df['close']
        
        # EMA alignment
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()
        
        current = close.iloc[-1]
        ema20 = ema_20.iloc[-1]
        ema50 = ema_50.iloc[-1]
        ema200 = ema_200.iloc[-1] if len(ema_200) >= 200 else ema50
        
        # Count bullish alignment
        bullish_score = 0
        if current > ema20: bullish_score += 1
        if current > ema50: bullish_score += 1
        if current > ema200: bullish_score += 1
        if ema20 > ema50: bullish_score += 1
        if ema50 > ema200: bullish_score += 1
        
        # Simple ADX proxy using price momentum
        # (Real ADX would need +DI/-DI calculation)
        price_change_20 = (current - close.iloc[-20]) / close.iloc[-20] * 100 if len(close) >= 20 else 0
        momentum_strength = min(100, abs(price_change_20) * 10)
        
        if bullish_score >= 4:
            return "STRONG_UP", momentum_strength
        elif bullish_score >= 3:
            return "UP", momentum_strength * 0.7
        elif bullish_score <= 1:
            return "STRONG_DOWN", momentum_strength
        elif bullish_score <= 2:
            return "DOWN", momentum_strength * 0.7
        else:
            return "RANGING", momentum_strength * 0.3
    
    def _check_trend_override(self, df: pd.DataFrame, rsi: float, proposed_direction: str) -> Tuple[bool, str]:
        """
        CRITICAL RULE: Don't fight strong trends!
        
        Even if RSI is overbought, if trend is STRONG_UP, don't sell!
        Wait for trend to weaken OR RSI to hit extreme (>90).
        
        Returns: (should_override, reason)
        """
        trend, strength = self._get_trend_strength(df)
        
        # Rule 1: Don't SELL in strong uptrend (unless RSI > 90)
        if proposed_direction == "SELL" and trend in ["STRONG_UP", "UP"]:
            if rsi < 90:  # Only allow sell if RSI is EXTREME
                return True, f"OVERRIDE: Don't sell into {trend} trend (RSI only {rsi:.0f}, need >90)"
            
        # Rule 2: Don't BUY in strong downtrend (unless RSI < 10)
        if proposed_direction == "BUY" and trend in ["STRONG_DOWN", "DOWN"]:
            if rsi > 10:  # Only allow buy if RSI is EXTREME
                return True, f"OVERRIDE: Don't buy into {trend} trend (RSI only {rsi:.0f}, need <10)"
        
        return False, ""
    
    def _check_mm_hunt(self, df: pd.DataFrame, current_price: float) -> Tuple[bool, str]:
        """
        Check if MM stop hunt has completed.
        
        KEY INSIGHT: MMs spike price to hunt stops, THEN reverse.
        We want to enter AFTER the hunt, not INTO it!
        
        Returns: (hunt_complete, reason)
        """
        if not self.mm_detector:
            return False, "MM Detector unavailable"
        
        try:
            analysis = self.mm_detector.get_mm_analysis(df, current_price)
            
            if analysis.get("stop_hunt"):
                hunt = analysis["stop_hunt"]
                return True, f"Stop hunt complete: {hunt['reason']}"
            
            # Check liquidity pools - are we near one?
            pools = analysis.get("liquidity_pools", [])
            for pool in pools[:3]:  # Check nearest 3
                distance = abs(pool["level"] - current_price)
                if distance < 20:  # Within 20 points
                    return False, f"Near liquidity pool at ${pool['level']:.2f} - wait for hunt!"
            
            return False, "No stop hunt detected"
            
        except Exception as e:
            logger.error(f"MM hunt check error: {e}")
            return False, f"Error: {e}"
    
    def _check_herd_exhaustion(self, df: pd.DataFrame) -> Tuple[bool, str, str]:
        """
        Check if herd is exhausted (everyone agrees but volume dying).
        
        KEY INSIGHT: When ALL indicators say BUY but nobody is buying anymore,
        it's time to FADE!
        
        Returns: (is_exhausted, exhaustion_type, reason)
        """
        if not self.herd_detector:
            return False, "", "Herd Detector unavailable"
        
        try:
            analysis = self.herd_detector.analyze(df)
            
            if analysis.is_exhausted:
                fade_direction = "SELL" if analysis.exhaustion_type == "BULLISH_EXHAUSTION" else "BUY"
                return True, analysis.exhaustion_type, f"Herd is {analysis.net_direction} but EXHAUSTED - FADE with {fade_direction}!"
            
            # Also check herd strength - if too crowded, avoid
            if analysis.herd_strength > 80:
                return False, "", f"Herd very crowded ({analysis.herd_strength:.0f}%) - dangerous to join"
            
            return False, "", f"Herd direction: {analysis.net_direction} ({analysis.herd_strength:.0f}% strength)"
            
        except Exception as e:
            logger.error(f"Herd check error: {e}")
            return False, "", f"Error: {e}"
    
    def _check_mtf_confirmation(
        self, 
        df_h1: pd.DataFrame, 
        df_h4: pd.DataFrame = None, 
        df_d1: pd.DataFrame = None
    ) -> Tuple[bool, str, str]:
        """
        Multi-timeframe confirmation.
        
        KEY INSIGHT: If H1 says SELL but D1 is in strong uptrend,
        the H1 signal is probably a trap!
        
        Returns: (confirmed, direction, reason)
        """
        signals = {}
        
        # H1 signal
        rsi_h1 = self._calculate_rsi(df_h1['close']).iloc[-1]
        trend_h1, _ = self._get_trend_strength(df_h1)
        
        if rsi_h1 < 30:
            signals['H1'] = 'BUY'
        elif rsi_h1 > 70:
            signals['H1'] = 'SELL'
        else:
            signals['H1'] = 'NEUTRAL'
        
        # H4 signal
        if df_h4 is not None and len(df_h4) >= 20:
            rsi_h4 = self._calculate_rsi(df_h4['close']).iloc[-1]
            trend_h4, _ = self._get_trend_strength(df_h4)
            
            if rsi_h4 < 35:
                signals['H4'] = 'BUY'
            elif rsi_h4 > 65:
                signals['H4'] = 'SELL'
            else:
                signals['H4'] = 'NEUTRAL'
        
        # D1 signal (most important for trend)
        if df_d1 is not None and len(df_d1) >= 20:
            rsi_d1 = self._calculate_rsi(df_d1['close']).iloc[-1]
            trend_d1, _ = self._get_trend_strength(df_d1)
            
            # D1 is about TREND not RSI
            if trend_d1 in ['STRONG_UP', 'UP']:
                signals['D1'] = 'BUY'  # D1 bias is bullish
            elif trend_d1 in ['STRONG_DOWN', 'DOWN']:
                signals['D1'] = 'SELL'  # D1 bias is bearish
            else:
                signals['D1'] = 'NEUTRAL'
        
        # Check alignment
        buy_count = sum(1 for s in signals.values() if s == 'BUY')
        sell_count = sum(1 for s in signals.values() if s == 'SELL')
        
        if len(signals) >= 2:
            if buy_count >= len(signals) - 1 and sell_count == 0:
                return True, 'BUY', f"MTF confirmed BUY: {signals}"
            elif sell_count >= len(signals) - 1 and buy_count == 0:
                return True, 'SELL', f"MTF confirmed SELL: {signals}"
        
        return False, 'WAIT', f"MTF NOT confirmed: {signals}"
    
    def _check_algo_hype(self, technical_data: Dict = None) -> Tuple[bool, str]:
        """
        Check if trade is too crowded (Algo Hype Index).
        
        KEY INSIGHT: When AHI > 75, everyone is in the same trade.
        That's when MMs will reverse it!
        
        Returns: (is_safe, reason)
        """
        if not ALGO_HYPE_AVAILABLE:
            return True, "Algo Hype Index unavailable"
        
        try:
            ahi = get_algo_hype_index(technical_data or {})
            
            if ahi.get('ahi_score', 0) > 75:
                return False, f"AHI at {ahi['ahi_score']:.0f} - EXTREMELY CROWDED! Avoid new positions."
            elif ahi.get('ahi_score', 0) > 60:
                return True, f"AHI at {ahi['ahi_score']:.0f} - Crowded, reduce size"
            else:
                return True, f"AHI at {ahi['ahi_score']:.0f} - Trade not crowded"
                
        except Exception as e:
            logger.error(f"AHI check error: {e}")
            return True, f"Error: {e}"
    
    def generate_signal(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame = None,
        df_d1: pd.DataFrame = None,
        current_price: float = None
    ) -> SmartSignal:
        """
        Generate a SMART signal that avoids retail traps.
        
        Process:
        1. Check trend (don't fight it!)
        2. Check MM hunt status (enter after trap, not into it)
        3. Check herd exhaustion (fade when everyone agrees)
        4. Multi-timeframe confirmation
        5. Algo hype check (avoid crowded trades)
        6. THEN generate signal
        
        Returns: SmartSignal
        """
        logger.info("\n" + "=" * 60)
        logger.info("🧠 SMART SIGNAL ANALYSIS")
        logger.info("=" * 60)
        
        # Ensure lowercase columns
        df_h1.columns = [c.lower() if isinstance(c, str) else c for c in df_h1.columns]
        if df_h4 is not None:
            df_h4.columns = [c.lower() if isinstance(c, str) else c for c in df_h4.columns]
        if df_d1 is not None:
            df_d1.columns = [c.lower() if isinstance(c, str) else c for c in df_d1.columns]
        
        current_price = current_price or df_h1['close'].iloc[-1]
        
        signal = SmartSignal(
            symbol=self.symbol,
            direction="WAIT",
            confidence=0,
            entry_price=current_price
        )
        
        reasoning_parts = []
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: GET BASIC INDICATORS
        # ═══════════════════════════════════════════════════════════
        rsi_h1 = self._calculate_rsi(df_h1['close']).iloc[-1]
        rsi_14_h1 = self._calculate_rsi(df_h1['close'], 14).iloc[-1]
        atr = self._calculate_atr(df_h1).iloc[-1]
        
        logger.info(f"📊 H1 RSI(2): {rsi_h1:.1f}, RSI(14): {rsi_14_h1:.1f}")
        logger.info(f"📊 Current Price: ${current_price:.2f}, ATR: {atr:.2f}")
        
        # Determine initial direction from RSI
        initial_direction = "WAIT"
        if rsi_h1 < 30:
            initial_direction = "BUY"
        elif rsi_h1 > 70:
            initial_direction = "SELL"
        
        logger.info(f"📊 Initial direction from RSI: {initial_direction}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: TREND OVERRIDE CHECK
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking trend override...")
        
        trend, trend_strength = self._get_trend_strength(df_h1)
        logger.info(f"   Trend: {trend} (strength: {trend_strength:.0f}%)")
        
        if initial_direction != "WAIT":
            should_override, override_reason = self._check_trend_override(df_h1, rsi_h1, initial_direction)
            
            if should_override:
                logger.warning(f"   ⚠️ {override_reason}")
                signal.warnings.append(override_reason)
                reasoning_parts.append(override_reason)
                signal.trend_aligned = False
                
                # DON'T generate signal if fighting trend!
                signal.direction = "WAIT"
                signal.confidence = 0
                signal.reasoning = " | ".join(reasoning_parts)
                logger.info(f"\n❌ SIGNAL BLOCKED: Fighting strong trend!")
                return signal
            else:
                signal.trend_aligned = True
                reasoning_parts.append(f"Trend aligned: {trend}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: MM HUNT CHECK
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking MM stop hunt...")
        
        hunt_complete, hunt_reason = self._check_mm_hunt(df_h1, current_price)
        logger.info(f"   {hunt_reason}")
        
        if hunt_complete:
            signal.mm_hunt_complete = True
            reasoning_parts.append(f"MM hunt complete: {hunt_reason}")
        else:
            # If near liquidity pool, WAIT!
            if "wait for hunt" in hunt_reason.lower():
                signal.warnings.append(hunt_reason)
                logger.warning(f"   ⚠️ Near liquidity pool - waiting for hunt!")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: HERD EXHAUSTION CHECK
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking herd exhaustion...")
        
        is_exhausted, exhaustion_type, herd_reason = self._check_herd_exhaustion(df_h1)
        logger.info(f"   {herd_reason}")
        
        if is_exhausted:
            signal.herd_exhausted = True
            reasoning_parts.append(f"Herd exhausted: {exhaustion_type}")
            
            # Override direction to FADE the herd!
            if exhaustion_type == "BULLISH_EXHAUSTION":
                initial_direction = "SELL"
                logger.info("   🔄 Fading bullish herd → SELL")
            elif exhaustion_type == "BEARISH_EXHAUSTION":
                initial_direction = "BUY"
                logger.info("   🔄 Fading bearish herd → BUY")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: MULTI-TIMEFRAME CONFIRMATION
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking multi-timeframe confirmation...")
        
        mtf_confirmed, mtf_direction, mtf_reason = self._check_mtf_confirmation(df_h1, df_h4, df_d1)
        logger.info(f"   {mtf_reason}")
        
        if mtf_confirmed:
            signal.mtf_confirmed = True
            reasoning_parts.append(f"MTF confirmed: {mtf_direction}")
            
            # Use MTF direction if it differs from H1
            if mtf_direction != initial_direction and initial_direction != "WAIT":
                logger.warning(f"   ⚠️ MTF says {mtf_direction} but H1 says {initial_direction} - CONFLICT!")
                signal.warnings.append(f"MTF conflict: {mtf_reason}")
        else:
            signal.warnings.append(mtf_reason)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: ALGO HYPE CHECK
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking algo hype (crowding)...")
        
        is_safe, ahi_reason = self._check_algo_hype()
        logger.info(f"   {ahi_reason}")
        
        if not is_safe:
            signal.warnings.append(ahi_reason)
            reasoning_parts.append("Trade crowded - AVOID!")
            signal.direction = "WAIT"
            signal.confidence = 0
            signal.reasoning = " | ".join(reasoning_parts)
            logger.info(f"\n❌ SIGNAL BLOCKED: Trade too crowded!")
            return signal
        
        # ═══════════════════════════════════════════════════════════
        # STEP 7: BREAKOUT DETECTION
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking for breakout patterns...")
        
        breakout_info = None
        if self.learner:
            breakout_info = self.learner.detect_breakout(df_h1)
            
            if breakout_info:
                logger.info(f"   ⚡ {breakout_info['type']} detected at ${breakout_info['level']:.2f}")
                
                if 'probability_real' in breakout_info:
                    prob = breakout_info['probability_real']
                    logger.info(f"   📊 Probability real: {prob:.0f}%")
                    logger.info(f"   🎯 Suggested action: {breakout_info['action']}")
                    
                    # If it's a real breakout, RIDE IT!
                    if prob > 65 and breakout_info['type'] == 'RESISTANCE_BREAK':
                        initial_direction = "BUY"
                        reasoning_parts.append(f"Breakout BUY: {prob:.0f}% probability real")
                        logger.info("   🚀 HIGH probability breakout - RIDE IT!")
                    elif prob > 65 and breakout_info['type'] == 'SUPPORT_BREAK':
                        initial_direction = "SELL"
                        reasoning_parts.append(f"Breakdown SELL: {prob:.0f}% probability real")
                        logger.info("   📉 HIGH probability breakdown - RIDE IT!")
                    
                    # If it's likely a FAKE breakout, FADE IT!
                    elif prob < 35 and breakout_info['type'] == 'RESISTANCE_BREAK':
                        initial_direction = "SELL"
                        reasoning_parts.append(f"Fake breakout FADE: Only {prob:.0f}% real")
                        logger.info("   🎯 LOW probability breakout - FADE IT!")
                    elif prob < 35 and breakout_info['type'] == 'SUPPORT_BREAK':
                        initial_direction = "BUY"
                        reasoning_parts.append(f"Fake breakdown FADE: Only {prob:.0f}% real")
                        logger.info("   🎯 LOW probability breakdown - FADE IT!")
                    
                    # If uncertain (35-65%), WAIT for confirmation
                    elif 35 <= prob <= 65:
                        signal.warnings.append(f"Breakout uncertain ({prob:.0f}%) - need confirmation")
                        logger.info(f"   ⏳ Uncertain breakout - waiting for confirmation")
                
                elif breakout_info['type'] in ['NEAR_RESISTANCE', 'NEAR_SUPPORT']:
                    signal.warnings.append(f"Near {breakout_info['type'].split('_')[1].lower()} - breakout imminent!")
                    logger.info(f"   ⏳ {breakout_info['warning']}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 8: HISTORICAL LEARNING CHECK
        # ═══════════════════════════════════════════════════════════
        logger.info("\n🔍 Checking historical pattern performance...")
        
        confidence_multiplier = 1.0
        if self.learner and initial_direction != "WAIT":
            mult, learn_reason = self.learner.get_pattern_confidence(
                self.symbol, initial_direction, trend, rsi_h1
            )
            confidence_multiplier = mult
            
            if mult < 0.8:
                logger.warning(f"   ⚠️ {learn_reason}")
                signal.warnings.append(learn_reason)
                reasoning_parts.append(f"Historical: {mult:.0%} confidence")
            elif mult > 1.1:
                logger.info(f"   ✅ {learn_reason}")
                reasoning_parts.append(f"Historical: {mult:.0%} confidence boost")
            else:
                logger.info(f"   📊 {learn_reason}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 9: GENERATE FINAL SIGNAL
        # ═══════════════════════════════════════════════════════════
        logger.info("\n📊 Generating final signal...")
        
        # Calculate confidence based on confirmations
        confidence = 50  # Base
        
        if signal.trend_aligned:
            confidence += 15
        if signal.mm_hunt_complete:
            confidence += 15
        if signal.herd_exhausted:
            confidence += 10
        if signal.mtf_confirmed:
            confidence += 10
        
        # Reduce confidence for warnings
        confidence -= len(signal.warnings) * 5
        
        # Apply historical learning multiplier
        confidence = confidence * confidence_multiplier
        
        confidence = max(0, min(95, confidence))
        
        # Only signal if confidence > 65
        if confidence >= 65 and initial_direction != "WAIT":
            signal.direction = initial_direction
            signal.confidence = confidence
            
            # Calculate SL/TP
            if initial_direction == "BUY":
                signal.stop_loss = current_price - (atr * 2)
                signal.take_profit = current_price + (atr * 4)  # 1:2 R:R
            else:
                signal.stop_loss = current_price + (atr * 2)
                signal.take_profit = current_price - (atr * 4)
            
            reasoning_parts.append(f"Final: {initial_direction} @ {confidence}% confidence")
        else:
            signal.direction = "WAIT"
            signal.confidence = confidence
            reasoning_parts.append(f"Confidence too low ({confidence}%) - WAIT")
        
        signal.reasoning = " | ".join(reasoning_parts)
        
        # Log final result
        logger.info("\n" + "=" * 60)
        if signal.direction != "WAIT":
            logger.info(f"✅ SMART SIGNAL: {signal.direction} {self.symbol}")
            logger.info(f"   Entry: ${signal.entry_price:.2f}")
            logger.info(f"   SL: ${signal.stop_loss:.2f}")
            logger.info(f"   TP: ${signal.take_profit:.2f}")
            logger.info(f"   Confidence: {signal.confidence}%")
        else:
            logger.info(f"⏳ NO SIGNAL - Waiting for better setup")
            logger.info(f"   Confidence: {signal.confidence}%")
        
        if signal.warnings:
            logger.info("   Warnings:")
            for w in signal.warnings:
                logger.info(f"      ⚠️ {w}")
        
        logger.info(f"   Reasoning: {signal.reasoning}")
        logger.info("=" * 60)
        
        # Record signal for learning (if not WAIT)
        if signal.direction != "WAIT" and self.learner:
            outcome = SignalOutcome(
                signal_id=f"NEO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                symbol=self.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                entry_time=signal.timestamp,
                rsi_at_signal=rsi_h1,
                trend_at_signal=trend,
                was_breakout=breakout_info is not None if breakout_info else False,
                breakout_type=breakout_info.get('type', '') if breakout_info else ''
            )
            self.learner.record_signal(outcome)
            logger.info(f"   📝 Signal recorded for learning: {outcome.signal_id}")
        
        return signal
    
    def record_outcome(
        self,
        signal_id: str,
        exit_price: float,
        outcome: str,  # WIN, LOSS, BREAKEVEN
        pnl_pips: float,
        mistake_type: str = "",
        lesson: str = ""
    ):
        """
        Record the outcome of a signal for learning.
        
        Call this when a trade closes to teach NEO what worked!
        
        Common mistake_types:
        - FOUGHT_TREND: Sold in uptrend or bought in downtrend
        - EARLY_ENTRY: Entered before confirmation
        - LATE_EXIT: Held too long, gave back profits
        - BREAKOUT_FADE: Faded a real breakout
        - FALSE_BREAKOUT_RIDE: Rode a fake breakout
        """
        if self.learner:
            self.learner.update_outcome(
                signal_id, exit_price, outcome, pnl_pips,
                mistake_type=mistake_type, lesson=lesson
            )
            logger.info(f"📚 Outcome recorded: {signal_id} → {outcome} ({pnl_pips:+.1f} pips)")
            
            if mistake_type:
                logger.warning(f"   ⚠️ Mistake: {mistake_type}")
                logger.info(f"   📚 Lesson: {lesson}")
    
    def get_learning_summary(self) -> str:
        """Get summary of what NEO has learned"""
        if self.learner:
            return self.learner.get_learning_summary()
        return "Learning not available"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION FOR NEO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_smart_signal(
    symbol: str,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame = None,
    df_d1: pd.DataFrame = None,
    current_price: float = None
) -> Dict:
    """
    Quick function to generate a smart signal.
    
    Use this in NEO instead of basic RSI checks!
    """
    generator = SmartSignalGenerator(symbol)
    signal = generator.generate_signal(df_h1, df_h4, df_d1, current_price)
    
    return asdict(signal)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_smart_signal():
    """Test the smart signal generator"""
    import yfinance as yf
    
    print("\n" + "=" * 70)
    print("🧪 TESTING SMART SIGNAL GENERATOR")
    print("=" * 70)
    
    # Fetch Gold data
    print("\n📊 Fetching XAUUSD data...")
    gold = yf.download('GC=F', period='3mo', interval='1h', progress=False)
    
    if gold.empty:
        print("❌ Could not fetch data")
        return
    
    # Handle MultiIndex columns
    if hasattr(gold.columns, 'levels'):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]
    
    # Also get daily data
    gold_d1 = yf.download('GC=F', period='1y', interval='1d', progress=False)
    if hasattr(gold_d1.columns, 'levels'):
        gold_d1.columns = [col[0].lower() for col in gold_d1.columns]
    else:
        gold_d1.columns = [c.lower() for c in gold_d1.columns]
    
    # Generate signal
    generator = SmartSignalGenerator("XAUUSD")
    signal = generator.generate_signal(
        df_h1=gold,
        df_d1=gold_d1,
        current_price=gold['close'].iloc[-1]
    )
    
    print("\n" + "=" * 70)
    print("📝 FINAL SIGNAL:")
    print("=" * 70)
    print(f"   Direction: {signal.direction}")
    print(f"   Confidence: {signal.confidence}%")
    print(f"   Entry: ${signal.entry_price:.2f}")
    print(f"   SL: ${signal.stop_loss:.2f}")
    print(f"   TP: ${signal.take_profit:.2f}")
    print(f"\n   Trend Aligned: {'✅' if signal.trend_aligned else '❌'}")
    print(f"   MM Hunt Complete: {'✅' if signal.mm_hunt_complete else '❌'}")
    print(f"   Herd Exhausted: {'✅' if signal.herd_exhausted else '❌'}")
    print(f"   MTF Confirmed: {'✅' if signal.mtf_confirmed else '❌'}")
    print(f"\n   Reasoning: {signal.reasoning}")
    
    if signal.warnings:
        print("\n   Warnings:")
        for w in signal.warnings:
            print(f"      ⚠️ {w}")
    
    print("=" * 70)
    
    return signal


if __name__ == "__main__":
    test_smart_signal()

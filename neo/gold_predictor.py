#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO GOLD 4-HOUR PREDICTOR
═══════════════════════════════════════════════════════════════════════════════

"If we can predict Gold's direction 60% of the time with proper risk management,
 we make money. We don't need 100%, we just need an edge and consistency."

This module:
1. Makes 4-hour predictions for Gold price movement
2. Uses multiple features (technical, sentiment, market structure)
3. Tracks predictions vs outcomes
4. LEARNS from mistakes → Improves over time

TARGET: 60%+ direction accuracy

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

# Pattern recognition
try:
    from pattern_detector import PatternDetector
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False

# Gold trading rules (learned from live success 2026-01-28)
try:
    from learning.training_rules_gold import GoldTradingRules, get_gold_rules
    GOLD_RULES_AVAILABLE = True
except ImportError:
    GOLD_RULES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoldPredictor")

# Default feature weights (will be learned over time)
DEFAULT_WEIGHTS = {
    # Technical indicators
    "rsi_oversold": 0.15,       # RSI < 30 → UP
    "rsi_overbought": 0.10,     # RSI > 70 → DOWN (weak because of trend!)
    "ema_trend": 0.20,          # EMA 20 > 50 = bullish
    "momentum_h4": 0.15,        # Last 4h momentum
    "macd_signal": 0.10,        # MACD crossover
    "atr_spike": 0.05,          # High volatility warning
    
    # Market structure
    "session_london": 0.10,     # London session bias
    "session_ny": 0.08,         # NY session bias
    "session_asia": 0.03,       # Asia session (low volatility)
    "day_of_week": 0.05,        # Friday = caution
    
    # Support/Resistance
    "near_support": 0.12,       # Price near support → UP
    "near_resistance": 0.10,    # Price near resistance → DOWN (weak!)
    
    # External
    "btc_correlation": 0.05,    # BTC leading indicator
    "usd_inverse": 0.08,        # USD inverse correlation
}


@dataclass
class Prediction:
    """A single 4-hour Gold prediction"""
    prediction_id: str
    timestamp: str
    target_time: str
    current_price: float
    predicted_direction: str  # UP, DOWN, FLAT
    predicted_change_pips: float
    predicted_price: float
    confidence: float
    features: Dict = field(default_factory=dict)
    feature_contributions: Dict = field(default_factory=dict)
    reasoning: str = ""
    status: str = "PENDING"  # PENDING, EVALUATED
    
    # Filled after evaluation
    actual_price: float = 0.0
    actual_change_pips: float = 0.0
    actual_direction: str = ""
    direction_correct: bool = False
    magnitude_accuracy: float = 0.0
    score: float = 0.0
    evaluated_at: str = ""


class GoldPredictor:
    """
    4-Hour Gold Price Predictor with Learning
    
    Makes predictions using weighted features, then learns from outcomes.
    """
    
    def __init__(self, weights_file: Optional[str] = None):
        self.data_dir = Path(__file__).parent / "prediction_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.weights_file = weights_file or str(self.data_dir / "feature_weights.json")
        self.history_file = str(self.data_dir / "prediction_history.json")
        
        # Load or initialize weights
        self.weights = self._load_weights()
        
        # Cache for features
        self._feature_cache = {}
        self._cache_time = None
        
        logger.info("=" * 60)
        logger.info("🔮 GOLD PREDICTOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Weights file: {self.weights_file}")
        logger.info(f"   History file: {self.history_file}")
        logger.info(f"   Features: {len(self.weights)} active")
        logger.info("=" * 60)
    
    def _load_weights(self) -> Dict[str, float]:
        """Load feature weights from file or use defaults"""
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                logger.info(f"📂 Loaded weights from {self.weights_file}")
                return data.get('weights', DEFAULT_WEIGHTS.copy())
        except FileNotFoundError:
            logger.info("📂 Using default weights (first run)")
            return DEFAULT_WEIGHTS.copy()
    
    def _save_weights(self):
        """Save current weights to file"""
        with open(self.weights_file, 'w') as f:
            json.dump({
                'weights': self.weights,
                'updated_at': datetime.utcnow().isoformat()
            }, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_gold_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch Gold price data"""
        try:
            df = yf.download('GC=F', period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning("No Gold data received, trying backup symbol")
                df = yf.download('XAUUSD=X', period=period, interval=interval, progress=False)
            
            # Handle multi-level columns
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Gold data: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        if len(df) < period + 1:
            return 50.0
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        return float(rsi) if not np.isnan(rsi) else 50.0
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate EMA"""
        return float(df['close'].ewm(span=period, adjust=False).mean().iloc[-1])
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """Calculate MACD and signal"""
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        
        if macd_val > signal_val:
            state = "BULLISH"
        elif macd_val < signal_val:
            state = "BEARISH"
        else:
            state = "NEUTRAL"
        
        return macd_val, signal_val, state
    
    def _get_session(self) -> str:
        """Get current trading session"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Sessions in UTC
        if 7 <= hour < 15:  # London 7-15 UTC
            return "LONDON"
        elif 13 <= hour < 21:  # NY 13-21 UTC (overlap with London 13-15)
            return "NEW_YORK"
        else:  # Asia
            return "ASIA"
    
    def _get_day_bias(self) -> Tuple[str, float]:
        """Get day-of-week bias"""
        day = datetime.now(timezone.utc).weekday()
        
        biases = {
            0: ("MONDAY", 0.55),      # Monday often continues Friday trend
            1: ("TUESDAY", 0.52),     # Tuesday average
            2: ("WEDNESDAY", 0.50),   # Wednesday neutral
            3: ("THURSDAY", 0.48),    # Thursday mixed
            4: ("FRIDAY", 0.45),      # Friday often reversals, caution!
            5: ("SATURDAY", 0.50),    # Weekend
            6: ("SUNDAY", 0.50),      # Weekend
        }
        
        return biases.get(day, ("UNKNOWN", 0.50))
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support/resistance levels"""
        if len(df) < 20:
            return {"support": 0, "resistance": 0, "near_support": False, "near_resistance": False}
        
        high = df['high'].rolling(20).max().iloc[-1]
        low = df['low'].rolling(20).min().iloc[-1]
        current = df['close'].iloc[-1]
        
        # Define "near" as within 0.3% of level
        near_threshold = current * 0.003
        
        return {
            "support": float(low),
            "resistance": float(high),
            "near_support": (current - low) < near_threshold,
            "near_resistance": (high - current) < near_threshold
        }
    
    def extract_features(self) -> Dict:
        """
        Extract all prediction features from current market state.
        
        Returns dict with feature values and their directional signals.
        """
        logger.info("📊 Extracting features...")
        
        # Get data for different timeframes
        df_h1 = self._get_gold_data(period="7d", interval="1h")
        df_h4 = self._get_gold_data(period="30d", interval="1h")  # Aggregate to H4
        
        if df_h1.empty:
            logger.error("No data available!")
            return {}
        
        current_price = float(df_h1['close'].iloc[-1])
        
        # ═══ TECHNICAL FEATURES ═══
        
        # RSI
        rsi_h1 = self._calculate_rsi(df_h1, 14)
        rsi_h4 = self._calculate_rsi(df_h4.iloc[::4], 14) if len(df_h4) >= 60 else rsi_h1
        
        # RSI signals
        rsi_oversold = rsi_h4 < 30
        rsi_overbought = rsi_h4 > 70
        rsi_extreme_oversold = rsi_h4 < 20
        rsi_extreme_overbought = rsi_h4 > 80
        
        # EMA Trend
        ema20 = self._calculate_ema(df_h1, 20)
        ema50 = self._calculate_ema(df_h1, 50)
        ema_trend = "BULLISH" if ema20 > ema50 else "BEARISH"
        ema_strength = abs(ema20 - ema50) / ema50 * 100  # Trend strength %
        
        # MACD
        macd_val, macd_signal, macd_state = self._calculate_macd(df_h1)
        
        # Momentum (last 4 candles = 4 hours on H1)
        momentum_4h = (current_price - df_h1['close'].iloc[-5]) / df_h1['close'].iloc[-5] * 100
        
        # ATR (volatility)
        atr = (df_h1['high'] - df_h1['low']).rolling(14).mean().iloc[-1]
        avg_atr = (df_h1['high'] - df_h1['low']).rolling(50).mean().iloc[-1]
        atr_spike = atr > avg_atr * 1.5  # Volatility spike
        
        # ═══ MARKET STRUCTURE ═══
        
        session = self._get_session()
        day_name, day_bias = self._get_day_bias()
        sr_levels = self._calculate_support_resistance(df_h1)
        
        # ═══ BUILD FEATURES DICT ═══
        
        features = {
            # Prices
            "current_price": current_price,
            "ema20": ema20,
            "ema50": ema50,
            
            # Technical values
            "rsi_h1": rsi_h1,
            "rsi_h4": rsi_h4,
            "macd_value": macd_val,
            "macd_signal": macd_signal,
            "macd_state": macd_state,
            "momentum_4h": momentum_4h,
            "atr": float(atr),
            
            # Technical signals (boolean or categorical)
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "rsi_extreme_oversold": rsi_extreme_oversold,
            "rsi_extreme_overbought": rsi_extreme_overbought,
            "ema_trend": ema_trend,
            "ema_strength": ema_strength,
            "atr_spike": atr_spike,
            
            # Market structure
            "session": session,
            "session_london": session == "LONDON",
            "session_ny": session == "NEW_YORK",
            "session_asia": session == "ASIA",
            "day_name": day_name,
            "day_bias": day_bias,
            
            # Support/Resistance
            "support": sr_levels["support"],
            "resistance": sr_levels["resistance"],
            "near_support": sr_levels["near_support"],
            "near_resistance": sr_levels["near_resistance"],
            
            # Timestamps
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"   ✅ Extracted {len(features)} features")
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTION GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def predict_4h(self) -> Prediction:
        """
        Make a 4-hour prediction for Gold.
        
        Uses weighted features to determine direction, magnitude, and confidence.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔮 GENERATING 4-HOUR GOLD PREDICTION")
        logger.info("=" * 60)
        
        # Extract features
        features = self.extract_features()
        
        if not features:
            return self._create_no_signal_prediction()
        
        current_price = features["current_price"]
        
        # Calculate weighted score (-1 to +1, negative = bearish, positive = bullish)
        bullish_score = 0.0
        bearish_score = 0.0
        contributions = {}
        reasoning_parts = []
        
        # ═══ TECHNICAL SIGNALS ═══
        
        # RSI oversold → BULLISH
        if features["rsi_oversold"]:
            weight = self.weights.get("rsi_oversold", 0.15)
            if features["rsi_extreme_oversold"]:
                weight *= 1.5  # Boost for extreme
            bullish_score += weight
            contributions["rsi_oversold"] = weight
            reasoning_parts.append(f"RSI {features['rsi_h4']:.0f} oversold (+{weight:.0%})")
        
        # RSI overbought → BEARISH (but weak in uptrend!)
        if features["rsi_overbought"]:
            weight = self.weights.get("rsi_overbought", 0.10)
            # REDUCE weight if in uptrend (learned from mistakes!)
            if features["ema_trend"] == "BULLISH":
                weight *= 0.3  # Only 30% effect in uptrend
                reasoning_parts.append(f"RSI {features['rsi_h4']:.0f} overbought but UPTREND (reduced)")
            else:
                bearish_score += weight
                contributions["rsi_overbought"] = -weight
                reasoning_parts.append(f"RSI {features['rsi_h4']:.0f} overbought (-{weight:.0%})")
        
        # EMA Trend
        weight = self.weights.get("ema_trend", 0.20)
        if features["ema_trend"] == "BULLISH":
            bullish_score += weight
            contributions["ema_trend"] = weight
            reasoning_parts.append(f"EMA trend BULLISH (+{weight:.0%})")
        else:
            bearish_score += weight
            contributions["ema_trend"] = -weight
            reasoning_parts.append(f"EMA trend BEARISH (-{weight:.0%})")
        
        # Momentum
        weight = self.weights.get("momentum_h4", 0.15)
        mom = features["momentum_4h"]
        if mom > 0.3:  # Strong bullish momentum
            bullish_score += weight
            contributions["momentum_h4"] = weight
            reasoning_parts.append(f"Momentum +{mom:.2f}% (+{weight:.0%})")
        elif mom < -0.3:  # Strong bearish momentum
            bearish_score += weight
            contributions["momentum_h4"] = -weight
            reasoning_parts.append(f"Momentum {mom:.2f}% (-{weight:.0%})")
        
        # MACD
        weight = self.weights.get("macd_signal", 0.10)
        if features["macd_state"] == "BULLISH":
            bullish_score += weight
            contributions["macd_signal"] = weight
            reasoning_parts.append(f"MACD bullish (+{weight:.0%})")
        elif features["macd_state"] == "BEARISH":
            bearish_score += weight
            contributions["macd_signal"] = -weight
            reasoning_parts.append(f"MACD bearish (-{weight:.0%})")
        
        # ═══ MARKET STRUCTURE ═══
        
        # Session bias
        if features["session_london"]:
            weight = self.weights.get("session_london", 0.10)
            bullish_score += weight * 0.6  # London often bullish
            contributions["session_london"] = weight * 0.6
            reasoning_parts.append(f"London session (+{weight*0.6:.0%})")
        elif features["session_asia"]:
            # Asia = low conviction, reduce overall confidence
            contributions["session_asia"] = 0
            reasoning_parts.append("Asia session (low conviction)")
        
        # Support/Resistance
        if features["near_support"]:
            weight = self.weights.get("near_support", 0.12)
            bullish_score += weight
            contributions["near_support"] = weight
            reasoning_parts.append(f"Near support ${features['support']:.0f} (+{weight:.0%})")
        
        if features["near_resistance"]:
            weight = self.weights.get("near_resistance", 0.10)
            # REDUCE if in uptrend (resistance often breaks!)
            if features["ema_trend"] == "BULLISH":
                weight *= 0.4
            bearish_score += weight
            contributions["near_resistance"] = -weight
            reasoning_parts.append(f"Near resistance ${features['resistance']:.0f} (-{weight:.0%})")
        
        # ATR spike warning
        if features["atr_spike"]:
            contributions["atr_spike"] = 0
            reasoning_parts.append("⚠️ High volatility - reduce size!")
        
        # ═══ PATTERN RECOGNITION ═══
        
        if PATTERNS_AVAILABLE:
            try:
                df = self._get_gold_data(period="10d", interval="1h")
                if not df.empty:
                    detector = PatternDetector("XAUUSD")
                    patterns = detector.analyze(df)
                    pattern_signal = detector.get_combined_signal(patterns)
                    
                    if pattern_signal["direction"] != "NEUTRAL":
                        pattern_weight = 0.15 * (pattern_signal["confidence"] / 100)
                        
                        if pattern_signal["direction"] == "BUY":
                            bullish_score += pattern_weight
                            contributions["patterns"] = pattern_weight
                        else:
                            bearish_score += pattern_weight
                            contributions["patterns"] = -pattern_weight
                        
                        reasoning_parts.append(
                            f"Patterns: {pattern_signal['direction']} "
                            f"({pattern_signal['pattern_count']} detected, {pattern_signal['confidence']:.0f}%)"
                        )
                        
                        logger.info(f"   🔍 Patterns detected: {pattern_signal['patterns']}")
            except Exception as e:
                logger.warning(f"Pattern detection failed: {e}")
        
        # ═══ CALCULATE FINAL PREDICTION ═══
        
        net_score = bullish_score - bearish_score
        
        # Direction
        if net_score > 0.05:
            direction = "UP"
        elif net_score < -0.05:
            direction = "DOWN"
        else:
            direction = "FLAT"
        
        # ═══ APPLY LEARNED GOLD TRADING RULES (from 2026-01-28 success) ═══
        if GOLD_RULES_AVAILABLE:
            try:
                gold_rules = get_gold_rules()
                
                # Check if we're trying to go SHORT in an uptrend (BLOCKED!)
                supertrend = "UP" if features["ema_trend"] == "BULLISH" else "DOWN"
                trend_check = gold_rules.rule_1_never_short_uptrend(supertrend)
                
                if direction == "DOWN" and not trend_check["can_short"]:
                    # BLOCK the bearish signal - change to HOLD
                    logger.info(f"   ⚠️ RULE 1 OVERRIDE: {trend_check['reason']}")
                    direction = "FLAT"
                    reasoning_parts.append(f"⚠️ RULE: {trend_check['reason']}")
                    net_score = 0  # Neutralize
                
                # Check exhaustion (don't buy at tops)
                if direction == "UP":
                    exhaustion = gold_rules.rule_2_exhaustion_detection(
                        current_price, 
                        features.get("ema20", current_price),
                        features.get("rsi_h4", 50),
                        features.get("recent_high", current_price)
                    )
                    if exhaustion["is_exhausted"]:
                        logger.info(f"   ⚠️ RULE 2 OVERRIDE: Exhausted - {exhaustion['reasons']}")
                        direction = "FLAT"  # Don't buy, but don't sell either
                        reasoning_parts.append(f"⚠️ EXHAUSTED: {', '.join(exhaustion['reasons'][:2])}")
                        net_score *= 0.3  # Reduce conviction
                
            except Exception as e:
                logger.warning(f"Gold rules check failed: {e}")
        
        # Magnitude (based on ATR and score strength)
        atr = features["atr"]
        score_strength = abs(net_score)
        
        # Expected move = ATR * 4 hours * score strength
        base_move = atr * 4 * score_strength
        predicted_change = max(5, min(100, base_move))  # Clamp 5-100 pips
        
        if direction == "DOWN":
            predicted_change = -predicted_change
        elif direction == "FLAT":
            predicted_change = 0
        
        predicted_price = current_price + predicted_change
        
        # Confidence (based on agreement of signals)
        total_signals = len(contributions)
        agreeing_signals = sum(1 for v in contributions.values() if (v > 0) == (net_score > 0))
        
        confidence = 50 + (agreeing_signals / max(total_signals, 1) * 30)
        confidence += abs(net_score) * 20  # Boost for strong conviction
        
        # Reduce confidence for Asia session or Friday
        if features["session_asia"]:
            confidence *= 0.8
        if features["day_name"] == "FRIDAY":
            confidence *= 0.9
        
        confidence = max(30, min(90, confidence))
        
        # Create prediction
        now = datetime.now(timezone.utc)
        target_time = now + timedelta(hours=4)
        
        prediction = Prediction(
            prediction_id=f"PRED_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            timestamp=now.isoformat(),
            target_time=target_time.isoformat(),
            current_price=current_price,
            predicted_direction=direction,
            predicted_change_pips=predicted_change,
            predicted_price=predicted_price,
            confidence=confidence,
            features=features,
            feature_contributions=contributions,
            reasoning=" | ".join(reasoning_parts)
        )
        
        # Log prediction
        logger.info(f"\n📊 PREDICTION GENERATED:")
        logger.info(f"   Current Price: ${current_price:.2f}")
        logger.info(f"   Direction: {direction}")
        logger.info(f"   Predicted Change: {predicted_change:+.1f} pips")
        logger.info(f"   Target Price: ${predicted_price:.2f}")
        logger.info(f"   Confidence: {confidence:.0f}%")
        logger.info(f"   Target Time: {target_time.strftime('%Y-%m-%d %H:%M')} UTC")
        logger.info(f"\n   Reasoning: {prediction.reasoning}")
        
        return prediction
    
    def _create_no_signal_prediction(self) -> Prediction:
        """Create a no-data prediction"""
        now = datetime.now(timezone.utc)
        return Prediction(
            prediction_id=f"PRED_{now.strftime('%Y%m%d_%H%M%S')}_NODATA",
            timestamp=now.isoformat(),
            target_time=(now + timedelta(hours=4)).isoformat(),
            current_price=0,
            predicted_direction="FLAT",
            predicted_change_pips=0,
            predicted_price=0,
            confidence=0,
            reasoning="No market data available"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTION EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def evaluate_prediction(self, prediction: Prediction) -> Prediction:
        """
        Evaluate a prediction against actual outcome.
        
        Called 4 hours after prediction was made.
        """
        logger.info(f"\n📊 EVALUATING PREDICTION: {prediction.prediction_id}")
        
        # Get current price
        df = self._get_gold_data(period="1d", interval="1h")
        if df.empty:
            logger.error("Cannot evaluate - no data")
            return prediction
        
        actual_price = float(df['close'].iloc[-1])
        actual_change = actual_price - prediction.current_price
        
        # Determine actual direction
        if actual_change > 5:
            actual_direction = "UP"
        elif actual_change < -5:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"
        
        # Score direction
        direction_correct = (
            (prediction.predicted_direction == "UP" and actual_change > 0) or
            (prediction.predicted_direction == "DOWN" and actual_change < 0) or
            (prediction.predicted_direction == "FLAT" and abs(actual_change) < 10)
        )
        
        # Magnitude accuracy (1.0 = perfect, 0 = way off)
        if prediction.predicted_change_pips != 0:
            magnitude_accuracy = max(0, 1 - abs(actual_change - prediction.predicted_change_pips) / abs(prediction.predicted_change_pips))
        else:
            magnitude_accuracy = 1.0 if abs(actual_change) < 10 else 0.5
        
        # Overall score
        score = 1.0 if direction_correct else 0.0
        
        # Update prediction with results
        prediction.actual_price = actual_price
        prediction.actual_change_pips = actual_change
        prediction.actual_direction = actual_direction
        prediction.direction_correct = direction_correct
        prediction.magnitude_accuracy = magnitude_accuracy
        prediction.score = score
        prediction.status = "EVALUATED"
        prediction.evaluated_at = datetime.utcnow().isoformat()
        
        # Log result
        emoji = "✅" if direction_correct else "❌"
        logger.info(f"\n{emoji} EVALUATION RESULT:")
        logger.info(f"   Predicted: {prediction.predicted_direction} {prediction.predicted_change_pips:+.1f} → ${prediction.predicted_price:.2f}")
        logger.info(f"   Actual:    {actual_direction} {actual_change:+.1f} → ${actual_price:.2f}")
        logger.info(f"   Direction Correct: {direction_correct}")
        logger.info(f"   Magnitude Accuracy: {magnitude_accuracy:.0%}")
        logger.info(f"   Score: {score:.1f}")
        
        return prediction


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_predictor = None

def get_predictor() -> GoldPredictor:
    """Get singleton predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = GoldPredictor()
    return _predictor


def predict_gold_4h() -> Dict:
    """Quick function to get a 4-hour Gold prediction"""
    predictor = get_predictor()
    prediction = predictor.predict_4h()
    return asdict(prediction)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_predictor():
    """Test the Gold predictor"""
    print("\n" + "=" * 70)
    print("🧪 TESTING GOLD PREDICTOR")
    print("=" * 70)
    
    predictor = GoldPredictor()
    
    # Make prediction
    prediction = predictor.predict_4h()
    
    print("\n" + "=" * 70)
    print("📋 PREDICTION SUMMARY")
    print("=" * 70)
    print(f"ID: {prediction.prediction_id}")
    print(f"Direction: {prediction.predicted_direction}")
    print(f"Change: {prediction.predicted_change_pips:+.1f} pips")
    print(f"Target: ${prediction.predicted_price:.2f}")
    print(f"Confidence: {prediction.confidence:.0f}%")
    print(f"Target Time: {prediction.target_time}")
    print(f"\nReasoning: {prediction.reasoning}")
    print(f"\nFeature Contributions:")
    for feat, contrib in prediction.feature_contributions.items():
        arrow = "↑" if contrib > 0 else "↓" if contrib < 0 else "→"
        print(f"   {arrow} {feat}: {contrib:+.2f}")


if __name__ == "__main__":
    test_predictor()

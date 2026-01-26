#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO PREDICTION LEARNER - Adaptive Weight Adjustment
═══════════════════════════════════════════════════════════════════════════════

This is where NEO gets SMARTER over time!

Learning Algorithm:
1. Track which features predict correctly vs incorrectly
2. Increase weights for accurate features
3. Decrease weights for inaccurate features
4. Optionally INVERT features that predict wrong consistently

Target: 60%+ accuracy

"The definition of insanity is doing the same thing over and over
 and expecting different results."
 
Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionLearner")

# Learning parameters
LEARNING_RATE = 0.1  # How much to adjust weights per outcome
MIN_WEIGHT = 0.01    # Minimum weight (don't go to zero)
MAX_WEIGHT = 0.40    # Maximum weight (don't over-rely on one feature)
MIN_SAMPLES = 10     # Minimum samples before adjusting
INVERT_THRESHOLD = 35  # If accuracy < 35%, consider inverting


class PredictionLearner:
    """
    Learns from prediction outcomes to improve future accuracy.
    
    Key Capabilities:
    1. Track feature accuracy
    2. Adjust weights based on performance
    3. Detect features that need inverting
    4. Maintain learning history
    """
    
    def __init__(self, weights_file: Optional[str] = None):
        self.data_dir = Path(__file__).parent / "prediction_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.weights_file = weights_file or str(self.data_dir / "feature_weights.json")
        self.learning_file = str(self.data_dir / "learning_history.json")
        
        self.weights = self._load_weights()
        self.feature_stats = self._load_feature_stats()
        self.learning_history = self._load_learning_history()
        
        logger.info("=" * 60)
        logger.info("🧠 PREDICTION LEARNER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Learning rate: {LEARNING_RATE}")
        logger.info(f"   Weight range: [{MIN_WEIGHT}, {MAX_WEIGHT}]")
        logger.info(f"   Features tracked: {len(self.feature_stats)}")
        logger.info("=" * 60)
    
    def _load_weights(self) -> Dict[str, float]:
        """Load current feature weights"""
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                return data.get('weights', {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_weights(self):
        """Save updated weights"""
        with open(self.weights_file, 'w') as f:
            json.dump({
                'weights': self.weights,
                'updated_at': datetime.utcnow().isoformat()
            }, f, indent=2)
    
    def _load_feature_stats(self) -> Dict:
        """Load feature statistics"""
        try:
            with open(self.learning_file, 'r') as f:
                data = json.load(f)
                return data.get('feature_stats', {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_learning_history(self) -> List:
        """Load learning history"""
        try:
            with open(self.learning_file, 'r') as f:
                data = json.load(f)
                return data.get('learning_history', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_learning_data(self):
        """Save feature stats and learning history"""
        with open(self.learning_file, 'w') as f:
            json.dump({
                'feature_stats': self.feature_stats,
                'learning_history': self.learning_history[-100:],  # Keep last 100
                'updated_at': datetime.utcnow().isoformat()
            }, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEARNING FROM OUTCOMES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_outcome(
        self,
        prediction_id: str,
        feature_contributions: Dict[str, float],
        predicted_direction: str,
        actual_direction: str,
        direction_correct: bool,
        confidence: float
    ):
        """
        Record the outcome of a prediction and update feature stats.
        
        Args:
            prediction_id: Unique prediction ID
            feature_contributions: Dict of feature -> contribution value
            predicted_direction: UP, DOWN, or FLAT
            actual_direction: UP, DOWN, or FLAT
            direction_correct: Whether direction was correct
            confidence: Prediction confidence
        """
        logger.info(f"\n📚 RECORDING OUTCOME: {prediction_id}")
        logger.info(f"   Predicted: {predicted_direction}, Actual: {actual_direction}")
        logger.info(f"   Correct: {direction_correct}")
        
        # Update feature statistics
        for feature, contribution in feature_contributions.items():
            if feature not in self.feature_stats:
                self.feature_stats[feature] = {
                    "total_uses": 0,
                    "correct_predictions": 0,
                    "bullish_signals": 0,
                    "bearish_signals": 0,
                    "bullish_correct": 0,
                    "bearish_correct": 0,
                    "avg_contribution": 0.0,
                    "accuracy": 50.0,
                    "inverted": False
                }
            
            fs = self.feature_stats[feature]
            fs["total_uses"] += 1
            
            # Track directional accuracy
            feature_bullish = contribution > 0
            
            if feature_bullish:
                fs["bullish_signals"] += 1
                if direction_correct and predicted_direction == "UP":
                    fs["bullish_correct"] += 1
                    fs["correct_predictions"] += 1
            else:
                fs["bearish_signals"] += 1
                if direction_correct and predicted_direction == "DOWN":
                    fs["bearish_correct"] += 1
                    fs["correct_predictions"] += 1
            
            # Update average contribution
            old_avg = fs["avg_contribution"]
            fs["avg_contribution"] = old_avg + (abs(contribution) - old_avg) / fs["total_uses"]
            
            # Calculate accuracy
            if fs["total_uses"] > 0:
                fs["accuracy"] = fs["correct_predictions"] / fs["total_uses"] * 100
        
        self._save_learning_data()
    
    def learn(self) -> Dict:
        """
        Run learning algorithm to adjust weights.
        
        Returns dict of adjustments made.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🧠 RUNNING LEARNING ALGORITHM")
        logger.info("=" * 60)
        
        adjustments = {}
        
        for feature, stats in self.feature_stats.items():
            if stats["total_uses"] < MIN_SAMPLES:
                logger.info(f"   {feature}: Skipping (only {stats['total_uses']} samples)")
                continue
            
            accuracy = stats["accuracy"]
            current_weight = self.weights.get(feature, 0.1)
            
            # ═══ DETERMINE ADJUSTMENT ═══
            
            if accuracy >= 65:
                # GOOD feature - increase weight
                adjustment = LEARNING_RATE * (accuracy - 50) / 50
                new_weight = min(MAX_WEIGHT, current_weight * (1 + adjustment))
                reason = f"HIGH accuracy ({accuracy:.0f}%)"
                
            elif accuracy >= 55:
                # OKAY feature - slight increase
                adjustment = LEARNING_RATE * 0.5 * (accuracy - 50) / 50
                new_weight = current_weight * (1 + adjustment)
                reason = f"GOOD accuracy ({accuracy:.0f}%)"
                
            elif accuracy >= 45:
                # NEUTRAL feature - no change
                new_weight = current_weight
                reason = f"NEUTRAL accuracy ({accuracy:.0f}%)"
                adjustment = 0
                
            elif accuracy >= INVERT_THRESHOLD:
                # WEAK feature - decrease weight
                adjustment = LEARNING_RATE * (50 - accuracy) / 50
                new_weight = max(MIN_WEIGHT, current_weight * (1 - adjustment))
                reason = f"LOW accuracy ({accuracy:.0f}%)"
                
            else:
                # VERY WEAK feature - consider inverting!
                if not stats.get("inverted", False):
                    logger.warning(f"   ⚠️ {feature}: Accuracy {accuracy:.0f}% < {INVERT_THRESHOLD}% - INVERTING!")
                    stats["inverted"] = True
                    new_weight = current_weight * 0.5  # Also reduce weight
                    reason = f"INVERTED (was {accuracy:.0f}%)"
                else:
                    # Already inverted - just reduce
                    new_weight = max(MIN_WEIGHT, current_weight * 0.8)
                    reason = f"Still weak after inversion ({accuracy:.0f}%)"
            
            # Apply adjustment
            if new_weight != current_weight:
                self.weights[feature] = new_weight
                adjustments[feature] = {
                    "old_weight": current_weight,
                    "new_weight": new_weight,
                    "accuracy": accuracy,
                    "samples": stats["total_uses"],
                    "reason": reason
                }
                
                arrow = "↑" if new_weight > current_weight else "↓"
                logger.info(f"   {arrow} {feature}: {current_weight:.3f} → {new_weight:.3f} ({reason})")
            else:
                logger.info(f"   → {feature}: {current_weight:.3f} (unchanged, {reason})")
        
        # Save updated weights
        self._save_weights()
        
        # Record learning event
        self.learning_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "adjustments": adjustments,
            "total_features": len(self.feature_stats),
            "adjusted_features": len(adjustments)
        })
        self._save_learning_data()
        
        logger.info(f"\n✅ Learning complete: {len(adjustments)} features adjusted")
        
        return adjustments
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_feature_report(self) -> str:
        """Get detailed feature performance report"""
        lines = []
        lines.append("=" * 70)
        lines.append("📊 FEATURE PERFORMANCE REPORT")
        lines.append("=" * 70)
        
        if not self.feature_stats:
            lines.append("No feature data yet. Make some predictions first!")
            return "\n".join(lines)
        
        # Sort by accuracy
        sorted_features = sorted(
            self.feature_stats.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )
        
        lines.append("\n🏆 TOP PERFORMERS:")
        for feature, stats in sorted_features[:5]:
            if stats["total_uses"] >= 5:
                weight = self.weights.get(feature, 0)
                emoji = "🟢" if stats["accuracy"] >= 60 else "🟡"
                lines.append(f"   {emoji} {feature}")
                lines.append(f"      Accuracy: {stats['accuracy']:.1f}% ({stats['total_uses']} samples)")
                lines.append(f"      Weight: {weight:.3f}")
        
        lines.append("\n⚠️ WEAK FEATURES:")
        for feature, stats in sorted_features[-3:]:
            if stats["total_uses"] >= 5 and stats["accuracy"] < 50:
                weight = self.weights.get(feature, 0)
                inverted = "🔄 INVERTED" if stats.get("inverted") else ""
                lines.append(f"   🔴 {feature} {inverted}")
                lines.append(f"      Accuracy: {stats['accuracy']:.1f}% ({stats['total_uses']} samples)")
                lines.append(f"      Weight: {weight:.3f}")
        
        lines.append("\n📈 WEIGHT DISTRIBUTION:")
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        total_weight = sum(self.weights.values())
        for feature, weight in sorted_weights[:5]:
            pct = weight / total_weight * 100 if total_weight > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"   {feature}: {bar} {weight:.3f} ({pct:.0f}%)")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on current learning state"""
        actions = []
        
        # Check for features that need attention
        for feature, stats in self.feature_stats.items():
            if stats["total_uses"] >= MIN_SAMPLES:
                if stats["accuracy"] < 40:
                    actions.append(f"🔴 Consider REMOVING or INVERTING '{feature}' (accuracy: {stats['accuracy']:.0f}%)")
                elif stats["accuracy"] > 70:
                    actions.append(f"🟢 BOOST weight for '{feature}' (accuracy: {stats['accuracy']:.0f}%)")
        
        # Check overall accuracy
        total_correct = sum(s["correct_predictions"] for s in self.feature_stats.values())
        total_uses = sum(s["total_uses"] for s in self.feature_stats.values())
        if total_uses > 0:
            overall_accuracy = total_correct / total_uses * 100
            if overall_accuracy < 50:
                actions.append(f"⚠️ Overall accuracy is LOW ({overall_accuracy:.0f}%) - consider fundamental strategy changes")
            elif overall_accuracy >= 60:
                actions.append(f"✅ Overall accuracy meets target ({overall_accuracy:.0f}%)!")
        
        return actions
    
    def suggest_new_features(self) -> List[str]:
        """Suggest new features to add based on learning gaps"""
        suggestions = []
        
        # If we have few features or low accuracy, suggest new ones
        if len(self.feature_stats) < 5:
            suggestions.append("Add more technical features (RSI, MACD, Bollinger Bands)")
        
        # Check if we have sentiment features
        has_sentiment = any("sentiment" in f.lower() for f in self.feature_stats.keys())
        if not has_sentiment:
            suggestions.append("Consider adding sentiment features (news, social media)")
        
        # Check if we have multi-timeframe
        has_mtf = any("h4" in f.lower() or "d1" in f.lower() for f in self.feature_stats.keys())
        if not has_mtf:
            suggestions.append("Consider adding multi-timeframe analysis (H4, D1)")
        
        return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_learner = None

def get_learner() -> PredictionLearner:
    """Get singleton learner instance"""
    global _learner
    if _learner is None:
        _learner = PredictionLearner()
    return _learner


def learn_from_outcome(
    prediction_id: str,
    feature_contributions: Dict,
    predicted_direction: str,
    actual_direction: str,
    direction_correct: bool,
    confidence: float
):
    """Quick function to record outcome and optionally trigger learning"""
    learner = get_learner()
    learner.record_outcome(
        prediction_id, feature_contributions, predicted_direction,
        actual_direction, direction_correct, confidence
    )


def run_learning():
    """Quick function to run learning algorithm"""
    return get_learner().learn()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_learner():
    """Test the prediction learner"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PREDICTION LEARNER")
    print("=" * 70)
    
    learner = PredictionLearner()
    
    # Simulate some outcomes
    test_outcomes = [
        # EMA trend is usually good
        {"id": "T001", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        {"id": "T002", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        {"id": "T003", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "DOWN", "correct": False},
        {"id": "T004", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        {"id": "T005", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        {"id": "T006", "features": {"ema_trend": 0.2}, "pred": "DOWN", "actual": "DOWN", "correct": True},
        {"id": "T007", "features": {"ema_trend": -0.2}, "pred": "DOWN", "actual": "DOWN", "correct": True},
        {"id": "T008", "features": {"ema_trend": -0.2}, "pred": "DOWN", "actual": "DOWN", "correct": True},
        {"id": "T009", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        {"id": "T010", "features": {"ema_trend": 0.2}, "pred": "UP", "actual": "UP", "correct": True},
        
        # RSI overbought is often WRONG in uptrends
        {"id": "T011", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T012", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T013", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T014", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "DOWN", "correct": True},
        {"id": "T015", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T016", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T017", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T018", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "DOWN", "correct": True},
        {"id": "T019", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
        {"id": "T020", "features": {"rsi_overbought": -0.1}, "pred": "DOWN", "actual": "UP", "correct": False},
    ]
    
    for outcome in test_outcomes:
        learner.record_outcome(
            outcome["id"],
            outcome["features"],
            outcome["pred"],
            outcome["actual"],
            outcome["correct"],
            70  # confidence
        )
    
    # Run learning
    print("\n" + "=" * 70)
    print("Running learning algorithm...")
    print("=" * 70)
    adjustments = learner.learn()
    
    # Print report
    print(learner.get_feature_report())
    
    # Print recommendations
    print("\n📋 RECOMMENDATIONS:")
    for action in learner.get_recommended_actions():
        print(f"   {action}")


if __name__ == "__main__":
    test_learner()

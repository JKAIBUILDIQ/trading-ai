#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO PREDICTION STORE - Persistent Storage for Predictions
═══════════════════════════════════════════════════════════════════════════════

Stores all predictions and outcomes for:
1. Historical tracking
2. Accuracy calculation
3. Learning from patterns

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionStore")


class PredictionStore:
    """
    Persistent storage for Gold predictions.
    
    Maintains:
    - Full prediction history
    - Running accuracy statistics
    - Feature performance tracking
    """
    
    def __init__(self, store_file: Optional[str] = None):
        self.data_dir = Path(__file__).parent / "prediction_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.store_file = store_file or str(self.data_dir / "prediction_history.json")
        self.data = self._load_store()
        
        logger.info(f"📂 Prediction Store loaded: {len(self.data.get('predictions', []))} predictions")
    
    def _load_store(self) -> Dict:
        """Load prediction store from file"""
        try:
            with open(self.store_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._create_empty_store()
    
    def _create_empty_store(self) -> Dict:
        """Create empty store structure"""
        return {
            "predictions": [],
            "stats": {
                "total_predictions": 0,
                "evaluated_predictions": 0,
                "correct_direction": 0,
                "accuracy": 0.0,
                "avg_confidence_when_correct": 0.0,
                "avg_confidence_when_wrong": 0.0,
                "current_streak": 0,
                "best_streak": 0,
                "worst_streak": 0,
                "feature_performance": {},
                "last_updated": datetime.utcnow().isoformat()
            },
            "meta": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
        }
    
    def _save_store(self):
        """Save store to file"""
        self.data["stats"]["last_updated"] = datetime.utcnow().isoformat()
        with open(self.store_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_prediction(self, prediction) -> str:
        """
        Save a new prediction.
        
        Args:
            prediction: Prediction object or dict
            
        Returns:
            prediction_id
        """
        # Convert to dict if needed
        if hasattr(prediction, '__dict__'):
            pred_dict = asdict(prediction) if hasattr(prediction, '__dataclass_fields__') else prediction.__dict__
        else:
            pred_dict = prediction
        
        # Add to history
        self.data["predictions"].append(pred_dict)
        self.data["stats"]["total_predictions"] += 1
        
        # Keep only last 1000 predictions to avoid huge files
        if len(self.data["predictions"]) > 1000:
            self.data["predictions"] = self.data["predictions"][-1000:]
        
        self._save_store()
        logger.info(f"📝 Saved prediction: {pred_dict.get('prediction_id')}")
        
        return pred_dict.get('prediction_id')
    
    def update_prediction(self, prediction_id: str, updates: Dict):
        """Update an existing prediction with evaluation results"""
        for pred in self.data["predictions"]:
            if pred.get("prediction_id") == prediction_id:
                pred.update(updates)
                self._update_stats_after_evaluation(pred)
                self._save_store()
                logger.info(f"📝 Updated prediction: {prediction_id}")
                return True
        
        logger.warning(f"⚠️ Prediction not found: {prediction_id}")
        return False
    
    def get_prediction(self, prediction_id: str) -> Optional[Dict]:
        """Get a specific prediction by ID"""
        for pred in self.data["predictions"]:
            if pred.get("prediction_id") == prediction_id:
                return pred
        return None
    
    def get_pending_predictions(self) -> List[Dict]:
        """Get all predictions awaiting evaluation"""
        now = datetime.utcnow()
        pending = []
        
        for pred in self.data["predictions"]:
            if pred.get("status") == "PENDING":
                # Check if target time has passed
                target_time = datetime.fromisoformat(pred["target_time"].replace('Z', '+00:00').replace('+00:00', ''))
                if now > target_time:
                    pending.append(pred)
        
        return pending
    
    def get_last_prediction(self) -> Optional[Dict]:
        """Get the most recent prediction"""
        if self.data["predictions"]:
            return self.data["predictions"][-1]
        return None
    
    def get_recent_predictions(self, n: int = 20) -> List[Dict]:
        """Get the N most recent predictions"""
        return self.data["predictions"][-n:]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_stats_after_evaluation(self, pred: Dict):
        """Update running statistics after a prediction is evaluated"""
        stats = self.data["stats"]
        
        if pred.get("status") != "EVALUATED":
            return
        
        stats["evaluated_predictions"] += 1
        
        if pred.get("direction_correct"):
            stats["correct_direction"] += 1
            stats["current_streak"] = stats.get("current_streak", 0) + 1
            stats["best_streak"] = max(stats["best_streak"], stats["current_streak"])
        else:
            stats["worst_streak"] = min(stats.get("worst_streak", 0), -stats.get("current_streak", 0))
            stats["current_streak"] = -1  # Start negative streak
        
        # Calculate accuracy
        if stats["evaluated_predictions"] > 0:
            stats["accuracy"] = stats["correct_direction"] / stats["evaluated_predictions"] * 100
        
        # Update confidence stats
        self._update_confidence_stats(pred)
        
        # Update feature performance
        self._update_feature_performance(pred)
    
    def _update_confidence_stats(self, pred: Dict):
        """Update average confidence when correct/wrong"""
        stats = self.data["stats"]
        evaluated = self.get_evaluated_predictions()
        
        correct_confs = [p.get("confidence", 50) for p in evaluated if p.get("direction_correct")]
        wrong_confs = [p.get("confidence", 50) for p in evaluated if not p.get("direction_correct")]
        
        if correct_confs:
            stats["avg_confidence_when_correct"] = sum(correct_confs) / len(correct_confs)
        if wrong_confs:
            stats["avg_confidence_when_wrong"] = sum(wrong_confs) / len(wrong_confs)
    
    def _update_feature_performance(self, pred: Dict):
        """Track which features contributed to correct/incorrect predictions"""
        stats = self.data["stats"]
        
        if "feature_performance" not in stats:
            stats["feature_performance"] = {}
        
        contributions = pred.get("feature_contributions", {})
        direction_correct = pred.get("direction_correct", False)
        predicted_direction = pred.get("predicted_direction", "FLAT")
        
        for feature, contribution in contributions.items():
            if feature not in stats["feature_performance"]:
                stats["feature_performance"][feature] = {
                    "total_uses": 0,
                    "correct_predictions": 0,
                    "bullish_signals": 0,
                    "bearish_signals": 0,
                    "bullish_correct": 0,
                    "bearish_correct": 0,
                    "accuracy": 0.0
                }
            
            fp = stats["feature_performance"][feature]
            fp["total_uses"] += 1
            
            # Track if this feature's signal was in the direction of the prediction
            feature_bullish = contribution > 0
            
            if feature_bullish:
                fp["bullish_signals"] += 1
                if direction_correct and predicted_direction == "UP":
                    fp["bullish_correct"] += 1
            else:
                fp["bearish_signals"] += 1
                if direction_correct and predicted_direction == "DOWN":
                    fp["bearish_correct"] += 1
            
            if direction_correct:
                fp["correct_predictions"] += 1
            
            # Calculate accuracy
            if fp["total_uses"] > 0:
                fp["accuracy"] = fp["correct_predictions"] / fp["total_uses"] * 100
    
    def get_evaluated_predictions(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get evaluated predictions"""
        evaluated = [p for p in self.data["predictions"] if p.get("status") == "EVALUATED"]
        if last_n:
            return evaluated[-last_n:]
        return evaluated
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.data["stats"]
    
    def get_accuracy(self, last_n: int = 100) -> float:
        """Get accuracy for last N predictions"""
        evaluated = self.get_evaluated_predictions(last_n)
        if not evaluated:
            return 0.0
        
        correct = sum(1 for p in evaluated if p.get("direction_correct"))
        return correct / len(evaluated) * 100
    
    def get_feature_leaderboard(self) -> List[Dict]:
        """Get features sorted by accuracy"""
        fp = self.data["stats"].get("feature_performance", {})
        
        leaderboard = []
        for feature, stats in fp.items():
            if stats["total_uses"] >= 5:  # Minimum 5 uses
                leaderboard.append({
                    "feature": feature,
                    "accuracy": stats["accuracy"],
                    "total_uses": stats["total_uses"],
                    "correct": stats["correct_predictions"]
                })
        
        return sorted(leaderboard, key=lambda x: x["accuracy"], reverse=True)
    
    def get_summary(self) -> str:
        """Get a text summary of prediction performance"""
        stats = self.data["stats"]
        
        lines = []
        lines.append("=" * 60)
        lines.append("📊 PREDICTION PERFORMANCE SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"\n📈 OVERALL:")
        lines.append(f"   Total Predictions: {stats['total_predictions']}")
        lines.append(f"   Evaluated: {stats['evaluated_predictions']}")
        lines.append(f"   Correct Direction: {stats['correct_direction']}")
        lines.append(f"   Accuracy: {stats['accuracy']:.1f}%")
        
        target_emoji = "✅" if stats['accuracy'] >= 60 else "🔄" if stats['accuracy'] >= 50 else "❌"
        lines.append(f"   Target (60%): {target_emoji}")
        
        lines.append(f"\n📊 CONFIDENCE:")
        lines.append(f"   Avg when CORRECT: {stats.get('avg_confidence_when_correct', 0):.0f}%")
        lines.append(f"   Avg when WRONG: {stats.get('avg_confidence_when_wrong', 0):.0f}%")
        
        lines.append(f"\n🔥 STREAKS:")
        lines.append(f"   Current: {stats.get('current_streak', 0)}")
        lines.append(f"   Best: {stats.get('best_streak', 0)}")
        
        # Feature leaderboard
        leaderboard = self.get_feature_leaderboard()
        if leaderboard:
            lines.append(f"\n🏆 FEATURE LEADERBOARD (min 5 uses):")
            for i, feat in enumerate(leaderboard[:5]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                lines.append(f"   {emoji} {feat['feature']}: {feat['accuracy']:.0f}% ({feat['total_uses']} uses)")
            
            if len(leaderboard) > 5:
                lines.append(f"\n⚠️ WORST FEATURES:")
                for feat in leaderboard[-3:]:
                    lines.append(f"   ❌ {feat['feature']}: {feat['accuracy']:.0f}% ({feat['total_uses']} uses)")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_store = None

def get_store() -> PredictionStore:
    """Get singleton store instance"""
    global _store
    if _store is None:
        _store = PredictionStore()
    return _store


def save_prediction(prediction) -> str:
    """Quick function to save a prediction"""
    return get_store().save_prediction(prediction)


def get_accuracy(last_n: int = 100) -> float:
    """Quick function to get accuracy"""
    return get_store().get_accuracy(last_n)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_store():
    """Test the prediction store"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PREDICTION STORE")
    print("=" * 70)
    
    store = PredictionStore()
    
    # Add some test predictions
    test_preds = [
        {"prediction_id": "TEST_001", "predicted_direction": "UP", "confidence": 70, 
         "status": "EVALUATED", "direction_correct": True, 
         "feature_contributions": {"ema_trend": 0.2, "rsi_oversold": 0.15}},
        {"prediction_id": "TEST_002", "predicted_direction": "UP", "confidence": 65, 
         "status": "EVALUATED", "direction_correct": True,
         "feature_contributions": {"ema_trend": 0.2, "momentum_h4": 0.15}},
        {"prediction_id": "TEST_003", "predicted_direction": "DOWN", "confidence": 55, 
         "status": "EVALUATED", "direction_correct": False,
         "feature_contributions": {"rsi_overbought": -0.1}},
    ]
    
    for pred in test_preds:
        store.save_prediction(pred)
        if pred.get("status") == "EVALUATED":
            store._update_stats_after_evaluation(pred)
    
    # Print summary
    print(store.get_summary())
    
    # Print leaderboard
    print("\n🏆 Feature Leaderboard:")
    for feat in store.get_feature_leaderboard():
        print(f"   {feat['feature']}: {feat['accuracy']:.0f}%")


if __name__ == "__main__":
    test_store()

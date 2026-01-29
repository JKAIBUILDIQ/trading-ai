"""
ANALYST PERFORMANCE TRACKER
Tracks predictions, grades performance, updates confidence weights.
"Ride the hot hand" - trust specialists in their domain.
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

SCORECARD_PATH = "/home/jbot/trading_ai/playbook/analyst_scorecard.json"
HISTORY_PATH = "/home/jbot/trading_ai/playbook/prediction_history.json"


@dataclass
class Prediction:
    date: str
    analyst: str
    asset: str
    direction: str  # BULLISH, BEARISH, NEUTRAL
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_price: Optional[float]
    confidence: str  # HIGH, MEDIUM, LOW
    thesis: str
    
    # Filled after market close
    actual_direction: Optional[str] = None
    actual_move_pct: Optional[float] = None
    hit_target: Optional[bool] = None
    hit_stop: Optional[bool] = None
    grade: Optional[str] = None
    pnl: Optional[float] = None
    lesson: Optional[str] = None


def load_scorecard() -> Dict:
    """Load current scorecard."""
    if os.path.exists(SCORECARD_PATH):
        with open(SCORECARD_PATH, 'r') as f:
            return json.load(f)
    return {"analysts": {}, "last_updated": None}


def save_scorecard(scorecard: Dict):
    """Save updated scorecard."""
    scorecard["last_updated"] = date.today().isoformat()
    with open(SCORECARD_PATH, 'w') as f:
        json.dump(scorecard, f, indent=2)


def load_history() -> List[Dict]:
    """Load prediction history."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    return []


def save_history(history: List[Dict]):
    """Save prediction history."""
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def calculate_grade(prediction: Prediction) -> str:
    """Calculate grade based on prediction accuracy."""
    score = 0
    
    # Direction correct: 40 points
    if prediction.actual_direction == prediction.direction:
        score += 40
    elif prediction.actual_direction == "NEUTRAL":
        score += 20  # Partial credit
    
    # Target hit: 30 points
    if prediction.hit_target:
        score += 30
    
    # Stop not hit: 20 points
    if not prediction.hit_stop:
        score += 20
    
    # Confidence calibration: 10 points
    # High confidence + correct = +10
    # High confidence + wrong = -10
    if prediction.confidence == "HIGH":
        if prediction.actual_direction == prediction.direction:
            score += 10
        else:
            score -= 10
    
    # Convert to grade
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def update_analyst_accuracy(scorecard: Dict, analyst: str, asset: str, grade: str) -> Dict:
    """Update analyst's accuracy for a specific asset."""
    if analyst not in scorecard["analysts"]:
        scorecard["analysts"][analyst] = {"accuracy_by_asset": {}}
    
    if asset not in scorecard["analysts"][analyst]["accuracy_by_asset"]:
        scorecard["analysts"][analyst]["accuracy_by_asset"][asset] = {
            "calls": 0,
            "correct": 0,
            "accuracy": 0.0,
            "last_5": [],
            "confidence_weight": 0.50
        }
    
    asset_data = scorecard["analysts"][analyst]["accuracy_by_asset"][asset]
    
    # Update calls count
    asset_data["calls"] += 1
    
    # Update correct count (B or better = correct)
    if grade in ["A+", "A", "B"]:
        asset_data["correct"] += 1
    
    # Update accuracy
    asset_data["accuracy"] = asset_data["correct"] / asset_data["calls"]
    
    # Update last_5
    asset_data["last_5"].append(grade)
    if len(asset_data["last_5"]) > 5:
        asset_data["last_5"] = asset_data["last_5"][-5:]
    
    # Update confidence weight (accuracy * recency factor)
    recent_grades = asset_data["last_5"]
    recent_correct = sum(1 for g in recent_grades if g in ["A+", "A", "B"])
    recency_factor = recent_correct / len(recent_grades) if recent_grades else 0.5
    
    # Weight is blend of overall accuracy and recent performance
    asset_data["confidence_weight"] = round(
        (asset_data["accuracy"] * 0.6) + (recency_factor * 0.4), 2
    )
    
    # Calculate average grade
    grade_values = {"A+": 4.3, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
    avg_value = sum(grade_values.get(g, 2.0) for g in recent_grades) / len(recent_grades)
    if avg_value >= 4.0:
        asset_data["avg_grade"] = "A"
    elif avg_value >= 3.5:
        asset_data["avg_grade"] = "A-"
    elif avg_value >= 3.0:
        asset_data["avg_grade"] = "B+"
    elif avg_value >= 2.5:
        asset_data["avg_grade"] = "B"
    elif avg_value >= 2.0:
        asset_data["avg_grade"] = "B-"
    elif avg_value >= 1.5:
        asset_data["avg_grade"] = "C"
    elif avg_value >= 1.0:
        asset_data["avg_grade"] = "D"
    else:
        asset_data["avg_grade"] = "F"
    
    return scorecard


def log_prediction(prediction: Prediction):
    """Log a new prediction."""
    history = load_history()
    history.append(asdict(prediction))
    save_history(history)
    print(f"📝 Logged prediction: {prediction.analyst} → {prediction.asset} ({prediction.direction})")


def grade_prediction(analyst: str, asset: str, date_str: str, 
                     actual_direction: str, actual_move_pct: float,
                     hit_target: bool, hit_stop: bool, 
                     pnl: float, lesson: str):
    """Grade a prediction after market close."""
    history = load_history()
    scorecard = load_scorecard()
    
    # Find the prediction
    for pred in history:
        if (pred["analyst"] == analyst and 
            pred["asset"] == asset and 
            pred["date"] == date_str and
            pred["grade"] is None):
            
            # Update with results
            pred["actual_direction"] = actual_direction
            pred["actual_move_pct"] = actual_move_pct
            pred["hit_target"] = hit_target
            pred["hit_stop"] = hit_stop
            pred["pnl"] = pnl
            pred["lesson"] = lesson
            
            # Calculate grade
            temp_pred = Prediction(**{k: v for k, v in pred.items() if k in Prediction.__dataclass_fields__})
            grade = calculate_grade(temp_pred)
            pred["grade"] = grade
            
            # Update scorecard
            scorecard = update_analyst_accuracy(scorecard, analyst, asset, grade)
            
            print(f"📊 Graded: {analyst} → {asset} = {grade}")
            print(f"   Move: {actual_move_pct:+.1f}% | P&L: ${pnl:+.2f}")
            print(f"   Lesson: {lesson}")
            
            break
    
    save_history(history)
    save_scorecard(scorecard)


def get_analyst_weight(analyst: str, asset: str) -> float:
    """Get the confidence weight for an analyst on a specific asset."""
    scorecard = load_scorecard()
    
    if analyst in scorecard.get("analysts", {}):
        analyst_data = scorecard["analysts"][analyst]
        if asset in analyst_data.get("accuracy_by_asset", {}):
            return analyst_data["accuracy_by_asset"][asset].get("confidence_weight", 0.50)
    
    return 0.50  # Default weight


def get_best_analyst_for_asset(asset: str) -> tuple:
    """Find the analyst with highest confidence for this asset."""
    scorecard = load_scorecard()
    
    best_analyst = None
    best_weight = 0.0
    
    for analyst, data in scorecard.get("analysts", {}).items():
        if asset in data.get("accuracy_by_asset", {}):
            weight = data["accuracy_by_asset"][asset].get("confidence_weight", 0)
            if weight > best_weight:
                best_weight = weight
                best_analyst = analyst
    
    return best_analyst, best_weight


def calculate_position_size(base_size: int, analyst: str, asset: str) -> int:
    """Calculate position size based on analyst confidence."""
    weight = get_analyst_weight(analyst, asset)
    adjusted_size = int(base_size * weight)
    return max(1, adjusted_size)  # Minimum 1


def print_daily_weights():
    """Print current analyst weights for all assets."""
    scorecard = load_scorecard()
    
    print("\n" + "="*60)
    print("📊 ANALYST CONFIDENCE WEIGHTS")
    print("="*60)
    
    # Collect all assets
    all_assets = set()
    for analyst, data in scorecard.get("analysts", {}).items():
        all_assets.update(data.get("accuracy_by_asset", {}).keys())
    
    for asset in sorted(all_assets):
        print(f"\n{asset}:")
        best, best_weight = get_best_analyst_for_asset(asset)
        
        for analyst, data in scorecard.get("analysts", {}).items():
            if asset in data.get("accuracy_by_asset", {}):
                asset_data = data["accuracy_by_asset"][asset]
                weight = asset_data.get("confidence_weight", 0)
                accuracy = asset_data.get("accuracy", 0)
                grade = asset_data.get("avg_grade", "N/A")
                
                star = "⭐" if analyst == best else "  "
                print(f"  {star} {analyst}: {weight:.2f} (acc: {accuracy:.0%}, grade: {grade})")
    
    print("\n" + "="*60)


def get_weighted_consensus(asset: str, opinions: Dict[str, str]) -> Dict:
    """
    Calculate weighted consensus from multiple analysts.
    
    opinions = {"NEO": "BULLISH", "CLAUDIA": "BULLISH", "META": "BEARISH"}
    """
    scorecard = load_scorecard()
    
    weighted_bullish = 0.0
    weighted_bearish = 0.0
    total_weight = 0.0
    
    details = []
    
    for analyst, opinion in opinions.items():
        weight = get_analyst_weight(analyst, asset)
        total_weight += weight
        
        if opinion == "BULLISH":
            weighted_bullish += weight
        elif opinion == "BEARISH":
            weighted_bearish += weight
        
        details.append({
            "analyst": analyst,
            "opinion": opinion,
            "weight": weight
        })
    
    if total_weight == 0:
        return {"consensus": "NEUTRAL", "confidence": 0.0, "details": details}
    
    bullish_pct = weighted_bullish / total_weight
    bearish_pct = weighted_bearish / total_weight
    
    if bullish_pct > 0.6:
        consensus = "BULLISH"
        confidence = bullish_pct
    elif bearish_pct > 0.6:
        consensus = "BEARISH"
        confidence = bearish_pct
    else:
        consensus = "NEUTRAL"
        confidence = max(bullish_pct, bearish_pct)
    
    return {
        "consensus": consensus,
        "confidence": round(confidence, 2),
        "bullish_weight": round(bullish_pct, 2),
        "bearish_weight": round(bearish_pct, 2),
        "details": details
    }


# Example usage / daily grading
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "weights":
            print_daily_weights()
        
        elif sys.argv[1] == "grade":
            # Example: python analyst_tracker.py grade NEO XAUUSD 2026-01-29 BEARISH -3.3 True False 189 "Good prediction"
            if len(sys.argv) >= 10:
                grade_prediction(
                    analyst=sys.argv[2],
                    asset=sys.argv[3],
                    date_str=sys.argv[4],
                    actual_direction=sys.argv[5],
                    actual_move_pct=float(sys.argv[6]),
                    hit_target=sys.argv[7].lower() == "true",
                    hit_stop=sys.argv[8].lower() == "true",
                    pnl=float(sys.argv[9]),
                    lesson=" ".join(sys.argv[10:]) if len(sys.argv) > 10 else ""
                )
        
        elif sys.argv[1] == "consensus":
            # Example: python analyst_tracker.py consensus IREN NEO:BULLISH CLAUDIA:BULLISH META:BEARISH
            asset = sys.argv[2]
            opinions = {}
            for arg in sys.argv[3:]:
                analyst, opinion = arg.split(":")
                opinions[analyst] = opinion
            
            result = get_weighted_consensus(asset, opinions)
            print(f"\n📊 Weighted Consensus for {asset}:")
            print(f"   Consensus: {result['consensus']}")
            print(f"   Confidence: {result['confidence']:.0%}")
            print(f"   Bullish Weight: {result['bullish_weight']:.0%}")
            print(f"   Bearish Weight: {result['bearish_weight']:.0%}")
            print("\n   Details:")
            for d in result['details']:
                print(f"     {d['analyst']}: {d['opinion']} (weight: {d['weight']:.2f})")
    
    else:
        print("Usage:")
        print("  python analyst_tracker.py weights              - Show current weights")
        print("  python analyst_tracker.py grade <args>         - Grade a prediction")
        print("  python analyst_tracker.py consensus <args>     - Calculate weighted consensus")

"""
NEO LEARNING SYSTEM
===================
Tracks predictions vs outcomes, learns from mistakes, adjusts weights.

Key Learning Points:
1. SELL signals during strong uptrends = BAD (penalize)
2. Missing rallies due to "overbought" RSI = BAD (reduce RSI weight)
3. Fighting trends = BAD (increase trend-following weight)

The system:
1. Records every signal NEO generates
2. Checks actual outcome 4 hours later
3. Grades the signal (A-F)
4. Adjusts feature weights based on performance
5. Logs misinterpretations for review
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO_Trainer")

# Directories
DATA_DIR = Path("/home/jbot/trading_ai/neo/learning")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_LOG = DATA_DIR / "signals_log.json"
OUTCOMES_LOG = DATA_DIR / "outcomes_log.json"
MISTAKES_LOG = DATA_DIR / "mistakes_log.json"
WEIGHTS_FILE = DATA_DIR / "feature_weights.json"
LEARNING_STATS = DATA_DIR / "learning_stats.json"


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT FEATURE WEIGHTS (adjusted based on learning)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    # Trend indicators (should be HIGH - don't fight the trend!)
    "ema_trend": 1.5,          # EMA 20 > EMA 50 = bullish
    "price_vs_ema200": 1.3,    # Price above 200 EMA = bullish
    "higher_highs": 1.2,       # Making new highs = bullish
    "supertrend": 1.4,         # SuperTrend direction
    
    # Momentum (medium weight)
    "macd": 1.0,
    "momentum_h1": 0.9,
    
    # Overbought/Oversold (REDUCED - these cause bad SELL signals!)
    "rsi_overbought": 0.3,     # RSI > 70 - REDUCED from 1.0
    "rsi_oversold": 0.8,       # RSI < 30 - still useful for buying dips
    
    # Pattern recognition
    "double_top": 0.7,
    "double_bottom": 0.9,
    "bull_flag": 1.1,
    "bear_flag": 0.6,          # Reduced - often wrong in uptrends
    
    # Volume
    "volume_spike": 0.8,
    "accumulation": 1.0,
    "distribution": 0.5,       # Reduced - often wrong signal
    
    # Smart Money Concepts
    "order_block_bullish": 1.1,
    "order_block_bearish": 0.6,
    "fair_value_gap": 0.9,
    
    # Gold-specific fundamentals (STRONG LONG BIAS)
    "gold_fundamental_bullish": 1.5,  # BRICS, central banks, etc.
    "btc_correlation": 0.4,           # Reduced - not always correlated
}


def load_weights() -> Dict[str, float]:
    """Load current feature weights"""
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: Dict[str, float]):
    """Save updated weights"""
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=2)


def load_json_log(file_path: Path) -> List[Dict]:
    """Load a JSON log file"""
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return []


def save_json_log(file_path: Path, data: List[Dict]):
    """Save a JSON log file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_signal(signal: Dict):
    """
    Log a signal NEO generates for later evaluation.
    
    Signal should include:
    - symbol: XAUUSD, IREN, etc.
    - action: BUY, SELL, HOLD
    - confidence: 0-100
    - price_at_signal: current price when signal generated
    - features_used: dict of features that contributed to decision
    - reasoning: why this signal was generated
    """
    signals = load_json_log(SIGNALS_LOG)
    
    signal_record = {
        "id": f"SIG_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat(),
        "check_time": (datetime.utcnow() + timedelta(hours=4)).isoformat(),
        "symbol": signal.get("symbol"),
        "action": signal.get("action"),
        "confidence": signal.get("confidence"),
        "price_at_signal": signal.get("price_at_signal"),
        "features_used": signal.get("features_used", {}),
        "reasoning": signal.get("reasoning"),
        "strategy": signal.get("strategy"),
        "status": "PENDING"  # PENDING, EVALUATED
    }
    
    signals.append(signal_record)
    save_json_log(SIGNALS_LOG, signals)
    
    logger.info(f"📝 Logged signal: {signal_record['id']} - {signal.get('action')} @ {signal.get('price_at_signal')}")
    return signal_record["id"]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTCOME EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_signal(signal_id: str, actual_price: float) -> Dict:
    """
    Evaluate a signal against actual outcome.
    
    Grading:
    - A: Perfect direction + magnitude
    - B: Correct direction, okay magnitude
    - C: Correct direction, poor magnitude
    - D: Neutral outcome
    - F: Wrong direction (TERRIBLE - especially SELL during rally!)
    """
    signals = load_json_log(SIGNALS_LOG)
    
    # Find the signal
    signal = None
    for s in signals:
        if s["id"] == signal_id:
            signal = s
            break
    
    if not signal:
        return {"error": f"Signal {signal_id} not found"}
    
    price_at_signal = signal["price_at_signal"]
    action = signal["action"]
    symbol = signal["symbol"]
    
    # Calculate actual move
    price_change = actual_price - price_at_signal
    price_change_pct = (price_change / price_at_signal) * 100
    
    # Determine actual direction
    if price_change > 0:
        actual_direction = "UP"
    elif price_change < 0:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"
    
    # Grade the signal
    grade = "D"
    grade_reason = ""
    is_mistake = False
    mistake_type = None
    
    if action == "BUY":
        if actual_direction == "UP":
            if price_change_pct >= 1.0:
                grade = "A"
                grade_reason = f"BUY correct! Price up {price_change_pct:.2f}%"
            elif price_change_pct >= 0.3:
                grade = "B"
                grade_reason = f"BUY correct, modest gain {price_change_pct:.2f}%"
            else:
                grade = "C"
                grade_reason = f"BUY correct but small move {price_change_pct:.2f}%"
        else:
            grade = "F"
            grade_reason = f"BUY WRONG! Price dropped {price_change_pct:.2f}%"
            is_mistake = True
            mistake_type = "FALSE_BUY"
    
    elif action == "SELL":
        if actual_direction == "DOWN":
            if price_change_pct <= -1.0:
                grade = "A"
                grade_reason = f"SELL correct! Price down {price_change_pct:.2f}%"
            elif price_change_pct <= -0.3:
                grade = "B"
                grade_reason = f"SELL correct, modest drop {price_change_pct:.2f}%"
            else:
                grade = "C"
                grade_reason = f"SELL correct but small move {price_change_pct:.2f}%"
        else:
            grade = "F"
            grade_reason = f"SELL WRONG! Price RALLIED {price_change_pct:.2f}%! 🔥"
            is_mistake = True
            mistake_type = "FALSE_SELL_DURING_RALLY"
            
            # This is the critical mistake NEO keeps making!
            if price_change_pct >= 0.5:
                mistake_type = "CATASTROPHIC_SELL_DURING_RALLY"
    
    elif action == "HOLD":
        grade = "D"
        grade_reason = f"HOLD - price moved {price_change_pct:.2f}%"
        if abs(price_change_pct) >= 1.0:
            is_mistake = True
            mistake_type = "MISSED_MOVE"
    
    # Create outcome record
    outcome = {
        "signal_id": signal_id,
        "evaluated_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "action": action,
        "price_at_signal": price_at_signal,
        "actual_price": actual_price,
        "price_change": price_change,
        "price_change_pct": round(price_change_pct, 2),
        "actual_direction": actual_direction,
        "grade": grade,
        "grade_reason": grade_reason,
        "is_mistake": is_mistake,
        "mistake_type": mistake_type,
        "features_used": signal.get("features_used", {}),
        "confidence": signal.get("confidence")
    }
    
    # Save outcome
    outcomes = load_json_log(OUTCOMES_LOG)
    outcomes.append(outcome)
    save_json_log(OUTCOMES_LOG, outcomes)
    
    # Update signal status
    for s in signals:
        if s["id"] == signal_id:
            s["status"] = "EVALUATED"
            s["grade"] = grade
            break
    save_json_log(SIGNALS_LOG, signals)
    
    # Log mistakes for learning
    if is_mistake:
        log_mistake(outcome)
    
    # Adjust weights based on outcome
    adjust_weights_from_outcome(outcome)
    
    logger.info(f"📊 Evaluated {signal_id}: Grade={grade} | {grade_reason}")
    return outcome


def log_mistake(outcome: Dict):
    """Log a mistake for detailed analysis"""
    mistakes = load_json_log(MISTAKES_LOG)
    
    mistake = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal_id": outcome["signal_id"],
        "symbol": outcome["symbol"],
        "mistake_type": outcome["mistake_type"],
        "action_taken": outcome["action"],
        "price_at_signal": outcome["price_at_signal"],
        "actual_price": outcome["actual_price"],
        "missed_move_pct": outcome["price_change_pct"],
        "features_that_failed": outcome.get("features_used", {}),
        "confidence_at_time": outcome.get("confidence"),
        "lesson": generate_lesson(outcome)
    }
    
    mistakes.append(mistake)
    save_json_log(MISTAKES_LOG, mistakes)
    
    logger.warning(f"❌ MISTAKE LOGGED: {outcome['mistake_type']} - {outcome['grade_reason']}")


def generate_lesson(outcome: Dict) -> str:
    """Generate a learning lesson from a mistake"""
    mistake_type = outcome.get("mistake_type")
    features = outcome.get("features_used", {})
    
    if mistake_type == "CATASTROPHIC_SELL_DURING_RALLY":
        return (
            f"CRITICAL LESSON: Sold at ${outcome['price_at_signal']:.2f} but price rallied "
            f"to ${outcome['actual_price']:.2f} (+{outcome['price_change_pct']:.1f}%). "
            f"Features that caused bad signal: {list(features.keys())}. "
            f"DO NOT SELL INTO STRONG UPTRENDS even if RSI is 'overbought'!"
        )
    
    elif mistake_type == "FALSE_SELL_DURING_RALLY":
        return (
            f"LESSON: SELL signal was wrong. Price went UP {outcome['price_change_pct']:.1f}%. "
            f"Check if RSI overbought or bearish patterns triggered this. "
            f"In strong Gold trends, overbought can stay overbought for weeks!"
        )
    
    elif mistake_type == "FALSE_BUY":
        return (
            f"LESSON: BUY signal was wrong. Price dropped {outcome['price_change_pct']:.1f}%. "
            f"Check for overhead resistance or distribution patterns missed."
        )
    
    elif mistake_type == "MISSED_MOVE":
        return (
            f"LESSON: HOLD while price moved {outcome['price_change_pct']:.1f}%. "
            f"Should have detected the directional move. Check momentum indicators."
        )
    
    return f"General mistake: {mistake_type}"


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT ADJUSTMENT (LEARNING!)
# ═══════════════════════════════════════════════════════════════════════════════

def adjust_weights_from_outcome(outcome: Dict):
    """
    Adjust feature weights based on outcome.
    
    If a feature contributed to a BAD signal (grade F):
    - Reduce its weight
    
    If a feature contributed to a GOOD signal (grade A/B):
    - Increase its weight
    """
    weights = load_weights()
    features = outcome.get("features_used", {})
    grade = outcome.get("grade", "D")
    mistake_type = outcome.get("mistake_type")
    
    adjustment = 0
    if grade == "A":
        adjustment = 0.05  # Increase weight
    elif grade == "B":
        adjustment = 0.02
    elif grade == "F":
        adjustment = -0.1  # Decrease weight significantly
        
        # Extra penalty for catastrophic mistakes
        if mistake_type == "CATASTROPHIC_SELL_DURING_RALLY":
            adjustment = -0.2  # Heavy penalty!
    
    if adjustment != 0:
        for feature, used in features.items():
            if used and feature in weights:
                old_weight = weights[feature]
                new_weight = max(0.1, min(2.0, old_weight + adjustment))  # Keep between 0.1 and 2.0
                weights[feature] = round(new_weight, 2)
                
                if adjustment < 0:
                    logger.info(f"⬇️ Reduced weight for '{feature}': {old_weight:.2f} → {new_weight:.2f}")
                else:
                    logger.info(f"⬆️ Increased weight for '{feature}': {old_weight:.2f} → {new_weight:.2f}")
        
        save_weights(weights)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION (run periodically)
# ═══════════════════════════════════════════════════════════════════════════════

async def evaluate_pending_signals(get_price_func):
    """
    Evaluate all pending signals that are past their check_time.
    
    Args:
        get_price_func: Async function to get current price for a symbol
    """
    signals = load_json_log(SIGNALS_LOG)
    now = datetime.utcnow()
    
    evaluated_count = 0
    
    for signal in signals:
        if signal["status"] != "PENDING":
            continue
        
        check_time = datetime.fromisoformat(signal["check_time"])
        if now >= check_time:
            # Time to evaluate!
            symbol = signal["symbol"]
            actual_price = await get_price_func(symbol)
            
            if actual_price:
                evaluate_signal(signal["id"], actual_price)
                evaluated_count += 1
    
    logger.info(f"✅ Evaluated {evaluated_count} pending signals")
    return evaluated_count


# ═══════════════════════════════════════════════════════════════════════════════
# STATS AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_learning_stats() -> Dict:
    """Get overall learning statistics"""
    outcomes = load_json_log(OUTCOMES_LOG)
    mistakes = load_json_log(MISTAKES_LOG)
    weights = load_weights()
    
    if not outcomes:
        return {
            "total_signals": 0,
            "message": "No signals evaluated yet"
        }
    
    # Grade distribution
    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for o in outcomes:
        grades[o.get("grade", "D")] += 1
    
    # Accuracy
    total = len(outcomes)
    correct = grades["A"] + grades["B"] + grades["C"]
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Mistake analysis
    mistake_types = {}
    for m in mistakes:
        mt = m.get("mistake_type", "UNKNOWN")
        mistake_types[mt] = mistake_types.get(mt, 0) + 1
    
    # Recent performance (last 20)
    recent = outcomes[-20:] if len(outcomes) >= 20 else outcomes
    recent_correct = sum(1 for o in recent if o.get("grade") in ["A", "B", "C"])
    recent_accuracy = (recent_correct / len(recent) * 100) if recent else 0
    
    return {
        "total_signals_evaluated": total,
        "accuracy_overall": round(accuracy, 1),
        "accuracy_recent_20": round(recent_accuracy, 1),
        "grade_distribution": grades,
        "total_mistakes": len(mistakes),
        "mistake_breakdown": mistake_types,
        "biggest_problem": max(mistake_types.items(), key=lambda x: x[1])[0] if mistake_types else None,
        "current_weights": weights,
        "lesson_count": len(mistakes),
        "last_updated": datetime.utcnow().isoformat()
    }


def get_recent_mistakes(limit: int = 10) -> List[Dict]:
    """Get recent mistakes for review"""
    mistakes = load_json_log(MISTAKES_LOG)
    return mistakes[-limit:] if mistakes else []


def force_learn_from_current_state(symbol: str, price_when_sold: float, current_price: float, action: str = "SELL"):
    """
    Manually teach NEO about a bad signal.
    
    Example: NEO said SELL at 5100, now Gold is 5300
    Call: force_learn_from_current_state("XAUUSD", 5100, 5300, "SELL")
    """
    price_change = current_price - price_when_sold
    price_change_pct = (price_change / price_when_sold) * 100
    
    outcome = {
        "signal_id": f"MANUAL_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "evaluated_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "action": action,
        "price_at_signal": price_when_sold,
        "actual_price": current_price,
        "price_change": price_change,
        "price_change_pct": round(price_change_pct, 2),
        "actual_direction": "UP" if price_change > 0 else "DOWN",
        "grade": "F",
        "grade_reason": f"MANUAL CORRECTION: {action} at ${price_when_sold} was WRONG. Price is now ${current_price} ({price_change_pct:+.1f}%)",
        "is_mistake": True,
        "mistake_type": "CATASTROPHIC_SELL_DURING_RALLY" if action == "SELL" and price_change_pct > 1 else "FALSE_SELL",
        "features_used": {
            "rsi_overbought": True,  # Likely culprit
            "bear_flag": True,       # Pattern misread
        },
        "confidence": 70  # Assumed
    }
    
    # Log the mistake
    log_mistake(outcome)
    
    # Save outcome
    outcomes = load_json_log(OUTCOMES_LOG)
    outcomes.append(outcome)
    save_json_log(OUTCOMES_LOG, outcomes)
    
    # Heavily penalize the features
    adjust_weights_from_outcome(outcome)
    
    logger.warning(f"🎓 MANUAL LESSON LEARNED: {action} at ${price_when_sold} was catastrophically wrong!")
    return outcome


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Teach NEO about the recent bad SELL signals during Gold rally
    print("Teaching NEO about bad SELL signals during Gold rally...")
    
    # Gold rallied from ~5080 to 5300
    force_learn_from_current_state("XAUUSD", 5080, 5300, "SELL")
    force_learn_from_current_state("XAUUSD", 5100, 5300, "SELL")
    force_learn_from_current_state("XAUUSD", 5150, 5300, "SELL")
    
    print("\n📊 Learning Stats:")
    stats = get_learning_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n📝 Recent Mistakes:")
    for m in get_recent_mistakes(5):
        print(f"  - {m['mistake_type']}: {m['lesson'][:100]}...")

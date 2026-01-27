#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IREN SCALE-IN MONITOR
═══════════════════════════════════════════════════════════════════════════════

Monitors IREN for scale-in opportunities based on:
- 5% price drops from reference levels
- Volume confirmation (capitulation signals)
- Day of week patterns (Mondays preferred)
- Research-backed entry signals

Runs every 15 minutes during market hours.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import requests
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("IrenScaleIn")

# Configuration
CONFIG = {
    "symbol": "IREN",
    "scale_in_pct": 5.0,  # Buy on every 5% drop
    "min_volume_ratio": 1.0,  # Prefer high volume (1.2x for strong signal)
    "preferred_days": ["Monday", "Wednesday", "Friday"],  # Best days from research
    "avoid_days": ["Tuesday"],  # Worst day from research
    "paper_trading_url": "http://localhost:8500",
    "research_api_url": "http://localhost:8025",
    "neo_api_url": "http://localhost:8021",
    "contracts_per_scale_in": 3,  # Buy 3 contracts each time
    "preferred_strike": 55,
    "preferred_expiry": "2026-02-20",
    "max_option_contracts": 25,  # Don't exceed this total
    "max_shares": 500,  # Don't exceed this total
}

# State file to track scale-in levels
STATE_FILE = Path("/home/jbot/trading_ai/neo/research/iren_scale_in_state.json")


def load_state() -> Dict:
    """Load scale-in state from file"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    
    # Initial state
    return {
        "reference_price": None,
        "last_scale_in_price": None,
        "scale_in_levels": [],
        "executed_scale_ins": [],
        "alerts": [],
        "created_at": datetime.utcnow().isoformat()
    }


def save_state(state: Dict):
    """Save scale-in state to file"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_current_price() -> Optional[float]:
    """Get current IREN price"""
    try:
        ticker = yf.Ticker(CONFIG["symbol"])
        hist = ticker.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
    return None


def get_volume_ratio() -> float:
    """Get today's volume vs 20-day average"""
    try:
        ticker = yf.Ticker(CONFIG["symbol"])
        hist = ticker.history(period='25d')
        if len(hist) >= 20:
            today_vol = hist['Volume'].iloc[-1]
            avg_vol = hist['Volume'].iloc[-21:-1].mean()
            return today_vol / avg_vol if avg_vol > 0 else 1.0
    except Exception as e:
        logger.error(f"Volume fetch error: {e}")
    return 1.0


def get_current_positions() -> Dict:
    """Get current paper trading positions"""
    try:
        response = requests.get(f"{CONFIG['paper_trading_url']}/paper-trading/positions", timeout=5)
        if response.ok:
            return response.json()
    except Exception as e:
        logger.error(f"Positions fetch error: {e}")
    return {"positions": []}


def count_option_contracts() -> int:
    """Count total IREN option contracts"""
    positions = get_current_positions()
    total = 0
    for p in positions.get("positions", []):
        if p["symbol"] == "IREN" and p["is_option"] and p["status"] == "OPEN":
            total += int(p["size"])
    return total


def count_shares() -> int:
    """Count total IREN shares"""
    positions = get_current_positions()
    total = 0
    for p in positions.get("positions", []):
        if p["symbol"] == "IREN" and not p["is_option"] and p["status"] == "OPEN":
            total += int(p["size"])
    return total


def get_neo_signal() -> Dict:
    """Get NEO's current prediction"""
    try:
        response = requests.get(f"{CONFIG['neo_api_url']}/api/iren/prediction/summary", timeout=5)
        if response.ok:
            return response.json()
    except Exception as e:
        logger.error(f"NEO API error: {e}")
    return {}


def get_research_data() -> Dict:
    """Get research data for entry rules"""
    try:
        response = requests.get(f"{CONFIG['research_api_url']}/api/iren/research/summary", timeout=5)
        if response.ok:
            return response.json()
    except Exception as e:
        logger.error(f"Research API error: {e}")
    return {}


def check_buy_conditions(current_price: float, state: Dict) -> Dict:
    """
    Check if scale-in conditions are met
    
    Returns:
        {
            "should_buy": bool,
            "reason": str,
            "confidence": int,
            "signals": list
        }
    """
    signals = []
    confidence = 50  # Base confidence
    
    # 1. Check if price dropped 5% from last scale-in or reference
    reference = state.get("last_scale_in_price") or state.get("reference_price")
    if reference:
        drop_pct = (reference - current_price) / reference * 100
        if drop_pct >= CONFIG["scale_in_pct"]:
            signals.append(f"✅ Price dropped {drop_pct:.1f}% from ${reference:.2f}")
            confidence += 15
        else:
            signals.append(f"⚠️ Only {drop_pct:.1f}% drop (need {CONFIG['scale_in_pct']}%)")
            return {
                "should_buy": False,
                "reason": f"Waiting for {CONFIG['scale_in_pct']}% drop",
                "confidence": 0,
                "signals": signals
            }
    
    # 2. Check volume (capitulation signal)
    volume_ratio = get_volume_ratio()
    if volume_ratio >= 1.2:
        signals.append(f"✅ HIGH VOLUME: {volume_ratio:.2f}x (capitulation)")
        confidence += 20
    elif volume_ratio >= 1.0:
        signals.append(f"✓ Volume OK: {volume_ratio:.2f}x")
        confidence += 5
    else:
        signals.append(f"⚠️ Low volume: {volume_ratio:.2f}x")
        confidence -= 10
    
    # 3. Check day of week
    today = datetime.now().strftime("%A")
    if today in CONFIG["preferred_days"]:
        signals.append(f"✅ Good day: {today}")
        confidence += 10
        if today == "Monday":
            confidence += 10  # Monday is best (69% win rate!)
    elif today in CONFIG["avoid_days"]:
        signals.append(f"⚠️ Bad day: {today} (42% win rate)")
        confidence -= 15
    
    # 4. Check NEO signal
    neo = get_neo_signal()
    if neo:
        neo_signal = neo.get("signal", "UNKNOWN")
        neo_direction = neo.get("predicted_direction", "UNKNOWN")
        patterns = neo.get("patterns_detected", [])
        
        if neo_direction == "UP" or neo_signal == "BUY":
            signals.append(f"✅ NEO says: {neo_signal} ({neo_direction})")
            confidence += 15
        elif neo_signal == "HOLD":
            signals.append(f"✓ NEO says: HOLD")
        else:
            signals.append(f"⚠️ NEO says: {neo_signal}")
            confidence -= 10
        
        # Check for buy patterns
        if patterns:
            buy_patterns = [p for p in patterns if "BUY" in str(p).upper() or "BULL" in str(p).upper()]
            if buy_patterns:
                signals.append(f"✅ {len(buy_patterns)} BUY patterns detected")
                confidence += 10
    
    # 5. Check position limits
    current_contracts = count_option_contracts()
    if current_contracts >= CONFIG["max_option_contracts"]:
        signals.append(f"❌ Max contracts reached ({current_contracts}/{CONFIG['max_option_contracts']})")
        return {
            "should_buy": False,
            "reason": "Position limit reached",
            "confidence": 0,
            "signals": signals
        }
    
    # 6. Check support levels
    research = get_research_data()
    if research:
        key_support = research.get("key_levels", {}).get("strong_support", 50)
        if current_price <= key_support * 1.02:  # Within 2% of support
            signals.append(f"✅ Near support: ${key_support:.2f}")
            confidence += 15
    
    # Final decision
    should_buy = confidence >= 60
    
    return {
        "should_buy": should_buy,
        "reason": "All conditions met" if should_buy else "Insufficient confidence",
        "confidence": min(100, confidence),
        "signals": signals
    }


def execute_scale_in(current_price: float, check_result: Dict, state: Dict) -> bool:
    """Execute a scale-in buy on paper trading"""
    try:
        # Get current option price (approximate)
        option_price = max(2.0, current_price - CONFIG["preferred_strike"] + 3)  # Rough estimate
        
        # Try to get real option price from Yahoo
        try:
            ticker = yf.Ticker(CONFIG["symbol"])
            options = ticker.option_chain(CONFIG["preferred_expiry"])
            calls = options.calls
            strike_row = calls[calls['strike'] == CONFIG["preferred_strike"]]
            if not strike_row.empty:
                option_price = float(strike_row['ask'].iloc[0])
        except:
            pass
        
        payload = {
            "symbol": CONFIG["symbol"],
            "type": "BUY_CALL",
            "size": CONFIG["contracts_per_scale_in"],
            "entry_price": option_price,
            "strike": CONFIG["preferred_strike"],
            "expiry": CONFIG["preferred_expiry"],
            "is_option": True,
            "option_type": "call",
            "source": "AUTO_SCALE_IN",
            "confidence": check_result["confidence"],
            "notes": f"Auto scale-in at ${current_price:.2f} (-5% drop). Signals: {len([s for s in check_result['signals'] if '✅' in s])} bullish"
        }
        
        response = requests.post(
            f"{CONFIG['paper_trading_url']}/paper-trading/open",
            json=payload,
            timeout=10
        )
        
        if response.ok:
            result = response.json()
            logger.info(f"🎯 SCALE-IN EXECUTED: {CONFIG['contracts_per_scale_in']} contracts @ ${option_price:.2f}")
            
            # Update state
            state["executed_scale_ins"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "price": current_price,
                "option_price": option_price,
                "contracts": CONFIG["contracts_per_scale_in"],
                "confidence": check_result["confidence"],
                "signals": check_result["signals"]
            })
            state["last_scale_in_price"] = current_price
            save_state(state)
            
            return True
        else:
            logger.error(f"Scale-in failed: {response.text}")
            
    except Exception as e:
        logger.error(f"Scale-in execution error: {e}")
    
    return False


def generate_scale_in_levels(reference_price: float, num_levels: int = 5) -> List[Dict]:
    """Generate scale-in price levels"""
    levels = []
    price = reference_price
    
    for i in range(num_levels):
        price = price * (1 - CONFIG["scale_in_pct"] / 100)
        levels.append({
            "level": i + 1,
            "price": round(price, 2),
            "drop_from_ref": round((reference_price - price) / reference_price * 100, 1),
            "contracts": CONFIG["contracts_per_scale_in"],
            "status": "PENDING"
        })
    
    return levels


def run_monitor():
    """Main monitoring loop iteration"""
    logger.info("=" * 60)
    logger.info("🔍 IREN SCALE-IN MONITOR CHECK")
    logger.info("=" * 60)
    
    # Load state
    state = load_state()
    
    # Get current price
    current_price = get_current_price()
    if not current_price:
        logger.error("Could not get current price")
        return
    
    logger.info(f"📊 Current IREN: ${current_price:.2f}")
    
    # Initialize reference price if not set
    if not state.get("reference_price"):
        state["reference_price"] = current_price
        state["scale_in_levels"] = generate_scale_in_levels(current_price)
        save_state(state)
        logger.info(f"📌 Set reference price: ${current_price:.2f}")
        logger.info("📋 Scale-in levels generated:")
        for level in state["scale_in_levels"]:
            logger.info(f"   Level {level['level']}: ${level['price']:.2f} (-{level['drop_from_ref']}%)")
        return
    
    # Check buy conditions
    check = check_buy_conditions(current_price, state)
    
    logger.info(f"\n📋 Buy Signal Check (Confidence: {check['confidence']}%):")
    for signal in check["signals"]:
        logger.info(f"   {signal}")
    
    if check["should_buy"]:
        logger.info(f"\n🎯 SCALE-IN TRIGGERED!")
        success = execute_scale_in(current_price, check, state)
        if success:
            logger.info("✅ Scale-in order executed!")
        else:
            logger.warning("❌ Scale-in execution failed")
    else:
        logger.info(f"\n⏳ No scale-in: {check['reason']}")
        
        # Show next scale-in level
        reference = state.get("last_scale_in_price") or state.get("reference_price")
        next_level = reference * (1 - CONFIG["scale_in_pct"] / 100)
        logger.info(f"📍 Next scale-in at: ${next_level:.2f} ({CONFIG['scale_in_pct']}% below ${reference:.2f})")
    
    # Show current position summary
    contracts = count_option_contracts()
    shares = count_shares()
    logger.info(f"\n📊 Current Positions: {shares} shares, {contracts} option contracts")


def get_scale_in_status() -> Dict:
    """Get current scale-in status for API"""
    state = load_state()
    current_price = get_current_price()
    
    if not current_price:
        return {"error": "Could not get price"}
    
    reference = state.get("last_scale_in_price") or state.get("reference_price") or current_price
    next_level = reference * (1 - CONFIG["scale_in_pct"] / 100)
    drop_needed = (reference - next_level) / reference * 100
    current_drop = (reference - current_price) / reference * 100 if reference > 0 else 0
    
    check = check_buy_conditions(current_price, state)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "current_price": current_price,
        "reference_price": state.get("reference_price"),
        "last_scale_in_price": state.get("last_scale_in_price"),
        "next_scale_in_level": round(next_level, 2),
        "drop_to_next_level": round(next_level - current_price, 2),
        "drop_to_next_level_pct": round(drop_needed - current_drop, 2),
        "current_drop_pct": round(current_drop, 2),
        "buy_signal": check,
        "total_scale_ins": len(state.get("executed_scale_ins", [])),
        "contracts_per_scale_in": CONFIG["contracts_per_scale_in"],
        "scale_in_levels": state.get("scale_in_levels", []),
        "executed_scale_ins": state.get("executed_scale_ins", [])[-5:],  # Last 5
        "current_contracts": count_option_contracts(),
        "max_contracts": CONFIG["max_option_contracts"],
        "config": {
            "scale_in_pct": CONFIG["scale_in_pct"],
            "preferred_strike": CONFIG["preferred_strike"],
            "preferred_expiry": CONFIG["preferred_expiry"]
        }
    }


if __name__ == "__main__":
    run_monitor()

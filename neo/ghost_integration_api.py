#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO → GHOST INTEGRATION API
═══════════════════════════════════════════════════════════════════════════════

Provides API endpoints for Ghost Commander to:
1. Fetch daily trading plan
2. Get real-time signal updates
3. Report trade results back for learning

Flow:
  NEO Pre-Market → JSON API → Ghost reads → Auto trading → Results back → NEO learns

Endpoints:
  GET  /api/neo/xauusd/daily-plan    - Get today's XAUUSD trading plan
  GET  /api/neo/iren/daily-plan      - Get today's IREN trading plan
  POST /api/neo/trade-result         - Report trade result for learning
  GET  /api/neo/learning-stats       - Get accuracy statistics

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent paths
sys.path.insert(0, '/home/jbot/trading_ai/neo')
sys.path.insert(0, '/home/jbot/trading_ai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("GhostIntegration")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("/home/jbot/trading_ai/neo/daily_data")
DATA_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("/home/jbot/trading_ai/neo/trade_results")
RESULTS_DIR.mkdir(exist_ok=True)

LEARNING_FILE = DATA_DIR / "learning_stats.json"

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NEO Ghost Integration API",
    description="API for Ghost Commander to fetch trading plans and report results",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_today_prediction(symbol: str) -> Optional[Dict]:
    """Load today's prediction file"""
    date_str = datetime.utcnow().strftime("%Y%m%d")
    file_path = DATA_DIR / f"{symbol}_{date_str}.json"
    
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return None


def get_learning_stats() -> Dict:
    """Get accumulated learning statistics"""
    if LEARNING_FILE.exists():
        with open(LEARNING_FILE) as f:
            return json.load(f)
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "win_rate": 0.0,
        "total_pips": 0.0,
        "avg_pips_per_trade": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "hunt_accuracy": 0.0,
        "direction_accuracy": 0.0,
        "by_strategy": {},
        "by_day_of_week": {},
        "recent_trades": []
    }


def update_learning_stats(result: Dict):
    """Update learning statistics with new trade result"""
    stats = get_learning_stats()
    
    pnl = result.get("pnl_pips", 0)
    
    stats["total_trades"] += 1
    stats["total_pips"] += pnl
    
    if pnl > 5:
        stats["wins"] += 1
    elif pnl < -5:
        stats["losses"] += 1
    else:
        stats["breakeven"] += 1
    
    if stats["total_trades"] > 0:
        stats["win_rate"] = (stats["wins"] / stats["total_trades"]) * 100
        stats["avg_pips_per_trade"] = stats["total_pips"] / stats["total_trades"]
    
    if pnl > stats.get("best_trade", 0):
        stats["best_trade"] = pnl
    if pnl < stats.get("worst_trade", 0):
        stats["worst_trade"] = pnl
    
    # Track by strategy
    strategy = result.get("strategy", "UNKNOWN")
    if strategy not in stats["by_strategy"]:
        stats["by_strategy"][strategy] = {"trades": 0, "wins": 0, "pips": 0}
    stats["by_strategy"][strategy]["trades"] += 1
    if pnl > 5:
        stats["by_strategy"][strategy]["wins"] += 1
    stats["by_strategy"][strategy]["pips"] += pnl
    
    # Track by day
    day = datetime.utcnow().strftime("%A")
    if day not in stats["by_day_of_week"]:
        stats["by_day_of_week"][day] = {"trades": 0, "wins": 0, "pips": 0}
    stats["by_day_of_week"][day]["trades"] += 1
    if pnl > 5:
        stats["by_day_of_week"][day]["wins"] += 1
    stats["by_day_of_week"][day]["pips"] += pnl
    
    # Track hunt/direction accuracy
    if result.get("hunt_hit"):
        stats["hunt_accuracy"] = ((stats.get("hunt_accuracy", 0) * (stats["total_trades"] - 1) + 100) / stats["total_trades"])
    else:
        stats["hunt_accuracy"] = ((stats.get("hunt_accuracy", 0) * (stats["total_trades"] - 1)) / stats["total_trades"])
    
    if result.get("direction_correct"):
        stats["direction_accuracy"] = ((stats.get("direction_accuracy", 0) * (stats["total_trades"] - 1) + 100) / stats["total_trades"])
    else:
        stats["direction_accuracy"] = ((stats.get("direction_accuracy", 0) * (stats["total_trades"] - 1)) / stats["total_trades"])
    
    # Keep last 20 trades
    stats["recent_trades"] = stats.get("recent_trades", [])[-19:] + [{
        "date": datetime.utcnow().isoformat(),
        "symbol": result.get("symbol"),
        "strategy": strategy,
        "pnl": pnl,
        "direction_correct": result.get("direction_correct", False)
    }]
    
    # Save
    with open(LEARNING_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "NEO Ghost Integration API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "xauusd_plan": "/api/neo/xauusd/daily-plan",
            "iren_plan": "/api/neo/iren/daily-plan",
            "trade_result": "/api/neo/trade-result",
            "learning_stats": "/api/neo/learning-stats"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════
# XAUUSD DAILY PLAN
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/neo/xauusd/daily-plan")
async def get_xauusd_daily_plan():
    """
    Get today's XAUUSD trading plan for Ghost Commander.
    
    Returns a structured plan that Ghost can execute:
    - Strategy type (FADE_THE_SQUEEZE, BUY_THE_DIP, etc.)
    - Entry conditions
    - Stop loss and take profit levels
    - Risk parameters
    """
    try:
        # Try to load today's prediction
        pred = get_today_prediction("XAUUSD")
        
        if not pred:
            # Generate fresh prediction if none exists
            logger.info("No prediction found, generating fresh...")
            from daily_reports import XAUUSDReports
            reporter = XAUUSDReports()
            reporter.generate_premarket()  # This saves the prediction
            pred = get_today_prediction("XAUUSD")
        
        if not pred:
            raise HTTPException(status_code=404, detail="Could not generate daily plan")
        
        # Determine strategy type
        expected_hunt = pred.get("expected_direction", "NONE")
        
        if expected_hunt == "SHORTS":
            strategy = "FADE_THE_SQUEEZE"
            direction = "SELL"
            entry_condition = f"Price breaks above {pred.get('hunt_target', 0):.0f} then fails"
            wait_for = f"squeeze_above_{pred.get('hunt_target', 0):.0f}"
        elif expected_hunt == "LONGS":
            strategy = "BUY_THE_DIP"
            direction = "BUY"
            entry_condition = f"Price drops to {pred.get('hunt_target', 0):.0f} and reverses"
            wait_for = f"hunt_at_{pred.get('hunt_target', 0):.0f}"
        else:
            strategy = "WAIT"
            direction = "NONE"
            entry_condition = "No clear setup - wait for better opportunity"
            wait_for = "none"
        
        # Calculate position size based on risk
        risk_pips = abs(pred.get("entry_level", 0) - pred.get("stop_loss", 0)) or 20
        risk_pct = 1.0  # 1% account risk
        
        # Get current price for reference
        import yfinance as yf
        try:
            data = yf.Ticker("GC=F").history(period="1d", interval="1m")
            current_price = float(data['Close'].iloc[-1]) if not data.empty else pred.get("pre_market_price", 0)
        except:
            current_price = pred.get("pre_market_price", 0)
        
        plan = {
            "signal_id": f"NEO_DAILY_{datetime.utcnow().strftime('%Y%m%d')}",
            "timestamp": datetime.utcnow().isoformat(),
            "date": pred.get("date"),
            "symbol": "XAUUSD",
            "current_price": current_price,
            
            # Strategy
            "strategy": strategy,
            "direction": direction,
            "confidence": pred.get("primary_probability", 50),
            
            # Entry conditions
            "entry": {
                "type": "LIMIT" if strategy != "WAIT" else "NONE",
                "condition": entry_condition,
                "wait_for": wait_for,
                "level": pred.get("entry_level", 0),
                "hunt_target": pred.get("hunt_target", 0),
                "valid_from": "08:00 UTC",  # London open
                "valid_until": "20:00 UTC",  # Before Asia
            },
            
            # Risk management
            "risk": {
                "stop_loss": pred.get("stop_loss", 0),
                "take_profit_1": pred.get("take_profit", 0),
                "take_profit_2": pred.get("take_profit", 0) + (30 if direction == "BUY" else -30),
                "risk_pips": risk_pips,
                "risk_percent": risk_pct,
                "max_position_usd": 5000,
                "trail_stop_after": 20,  # Trail after 20 pips profit
            },
            
            # Execution rules
            "execution": {
                "wait_for_reversal_candle": True,
                "min_reversal_size": 5,  # pips
                "confirmation_timeframe": "M15",
                "avoid_times": ["12:00-13:00 UTC"],  # London lunch
                "best_times": ["08:00-09:00 UTC", "13:00-14:00 UTC"],
            },
            
            # Context
            "context": {
                "overnight_bias": pred.get("overnight_bias", "UNKNOWN"),
                "overnight_high": pred.get("overnight_high", 0),
                "overnight_low": pred.get("overnight_low", 0),
                "scenario": pred.get("primary_scenario", "N/A"),
                "scenario_probability": pred.get("primary_probability", 0),
            },
            
            # Learning metadata
            "learning": {
                "prediction_id": f"XAUUSD_{datetime.utcnow().strftime('%Y%m%d')}",
                "report_result_to": "/api/neo/trade-result",
            }
        }
        
        logger.info(f"XAUUSD daily plan: {strategy} @ {pred.get('entry_level', 0):.0f}")
        return plan
        
    except Exception as e:
        logger.error(f"Error getting XAUUSD plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# IREN DAILY PLAN
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/neo/iren/daily-plan")
async def get_iren_daily_plan():
    """
    Get today's IREN trading plan for Ghost Commander.
    
    Includes both share and options strategies.
    """
    try:
        pred = get_today_prediction("IREN")
        
        if not pred:
            logger.info("No IREN prediction found, generating fresh...")
            from daily_reports import IRENReports
            reporter = IRENReports()
            reporter.generate_premarket()
            pred = get_today_prediction("IREN")
        
        if not pred:
            raise HTTPException(status_code=404, detail="Could not generate IREN plan")
        
        # Get BTC correlation
        import yfinance as yf
        try:
            btc = yf.Ticker("BTC-USD").history(period="2d", interval="1h")
            btc_price = float(btc['Close'].iloc[-1]) if not btc.empty else 0
            btc_prev = float(btc['Close'].iloc[-24]) if len(btc) >= 24 else btc_price
            btc_change = ((btc_price - btc_prev) / btc_prev) * 100
        except:
            btc_price = 0
            btc_change = 0
        
        # Get current IREN price
        try:
            iren = yf.Ticker("IREN").history(period="1d", interval="1m")
            current_price = float(iren['Close'].iloc[-1]) if not iren.empty else pred.get("pre_market_price", 0)
        except:
            current_price = pred.get("pre_market_price", 0)
        
        # Determine strategy
        overnight_change = pred.get("overnight_bias", "UNKNOWN")
        
        if overnight_change == "BEARISH" or (current_price < pred.get("pre_market_price", 100) * 0.95):
            strategy = "BUY_THE_DIP"
            direction = "BUY"
            entry_condition = "Gap down = accumulation opportunity"
        elif btc_change > 3:
            strategy = "MOMENTUM_LONG"
            direction = "BUY"
            entry_condition = f"BTC +{btc_change:.1f}% → IREN follows"
        elif btc_change < -3:
            strategy = "WAIT_FOR_BOTTOM"
            direction = "WAIT"
            entry_condition = f"BTC {btc_change:.1f}% → avoid longs"
        else:
            strategy = "RANGE_SCALP"
            direction = "NEUTRAL"
            entry_condition = "Range-bound, scalp or wait"
        
        plan = {
            "signal_id": f"NEO_IREN_{datetime.utcnow().strftime('%Y%m%d')}",
            "timestamp": datetime.utcnow().isoformat(),
            "date": pred.get("date"),
            "symbol": "IREN",
            "current_price": current_price,
            
            # Strategy
            "strategy": strategy,
            "direction": direction,
            "confidence": pred.get("primary_probability", 50),
            
            # BTC correlation
            "btc_correlation": {
                "btc_price": btc_price,
                "btc_change_24h": btc_change,
                "iren_expected": "UP" if btc_change > 2 else "DOWN" if btc_change < -2 else "FLAT"
            },
            
            # Share strategy
            "shares": {
                "action": direction,
                "entry_level": pred.get("entry_level", current_price),
                "stop_loss": pred.get("stop_loss", current_price * 0.92),
                "take_profit": pred.get("take_profit", current_price * 1.15),
                "position_size": "Based on account size",
                "dca_levels": [
                    current_price * 0.95,  # -5%
                    current_price * 0.90,  # -10%
                ]
            },
            
            # Options strategy
            "options": {
                "recommendation": "BUY_CALLS" if direction == "BUY" else "WAIT" if direction == "WAIT" else "SELL_CALLS",
                "strike": int(current_price) + 5,
                "expiry": "2-3 weeks out",
                "size": "2-5 contracts",
                "max_risk": "$500",
            },
            
            # Execution rules
            "execution": {
                "best_times": ["9:30-10:00 EST", "15:00-16:00 EST"],
                "avoid_times": ["12:00-14:00 EST"],
                "watch_btc": True,
            },
            
            # Context
            "context": {
                "overnight_high": pred.get("overnight_high", 0),
                "overnight_low": pred.get("overnight_low", 0),
                "overnight_change": ((current_price - pred.get("pre_market_price", 100)) / pred.get("pre_market_price", 100)) * 100,
                "hunt_target": pred.get("hunt_target", 0),
            },
            
            # Learning
            "learning": {
                "prediction_id": f"IREN_{datetime.utcnow().strftime('%Y%m%d')}",
                "report_result_to": "/api/neo/trade-result",
            }
        }
        
        logger.info(f"IREN daily plan: {strategy} @ ${current_price:.2f}")
        return plan
        
    except Exception as e:
        logger.error(f"Error getting IREN plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE RESULT REPORTING (Learning Loop)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/neo/trade-result")
async def report_trade_result(result: Dict):
    """
    Report trade result back to NEO for learning.
    
    Ghost should call this after each trade completes with:
    - prediction_id: From the daily plan
    - symbol: XAUUSD or IREN
    - strategy: The strategy used
    - direction: BUY or SELL
    - entry_price: Actual entry
    - exit_price: Actual exit
    - pnl_pips: Profit/loss in pips
    - hunt_hit: Did price hit the hunt target?
    - direction_correct: Was the direction prediction correct?
    - notes: Any additional notes
    """
    try:
        logger.info(f"Trade result received: {result}")
        
        # Validate required fields
        required = ["prediction_id", "symbol", "pnl_pips"]
        for field in required:
            if field not in result:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Add timestamp
        result["reported_at"] = datetime.utcnow().isoformat()
        
        # Save individual trade result
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{result['symbol']}_{date_str}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update learning statistics
        stats = update_learning_stats(result)
        
        # Update the daily prediction with actual results
        pred_file = DATA_DIR / f"{result['symbol']}_{datetime.utcnow().strftime('%Y%m%d')}.json"
        if pred_file.exists():
            with open(pred_file) as f:
                pred = json.load(f)
            
            pred["actual_pnl"] = result.get("pnl_pips", 0)
            pred["trade_taken"] = True
            pred["trade_result"] = result
            
            with open(pred_file, 'w') as f:
                json.dump(pred, f, indent=2)
        
        return {
            "status": "received",
            "message": "Trade result recorded for learning",
            "updated_stats": {
                "total_trades": stats["total_trades"],
                "win_rate": f"{stats['win_rate']:.1f}%",
                "total_pips": stats["total_pips"],
                "hunt_accuracy": f"{stats['hunt_accuracy']:.1f}%",
                "direction_accuracy": f"{stats['direction_accuracy']:.1f}%"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording trade result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/neo/learning-stats")
async def get_learning_statistics():
    """
    Get NEO's accumulated learning statistics.
    
    Shows:
    - Overall win rate
    - Total P&L
    - Accuracy by strategy type
    - Accuracy by day of week
    - Recent trades
    """
    stats = get_learning_stats()
    
    # Calculate additional metrics
    if stats["total_trades"] > 0:
        stats["profit_factor"] = (stats["wins"] / stats["losses"]) if stats["losses"] > 0 else float('inf')
    else:
        stats["profit_factor"] = 0
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "breakeven": stats["breakeven"],
            "win_rate": f"{stats['win_rate']:.1f}%",
            "profit_factor": f"{stats.get('profit_factor', 0):.2f}",
        },
        "performance": {
            "total_pips": stats["total_pips"],
            "avg_pips_per_trade": f"{stats['avg_pips_per_trade']:.1f}",
            "best_trade": stats["best_trade"],
            "worst_trade": stats["worst_trade"],
        },
        "accuracy": {
            "hunt_target": f"{stats['hunt_accuracy']:.1f}%",
            "direction": f"{stats['direction_accuracy']:.1f}%",
        },
        "by_strategy": stats.get("by_strategy", {}),
        "by_day_of_week": stats.get("by_day_of_week", {}),
        "recent_trades": stats.get("recent_trades", [])[-10:]
    }


@app.get("/api/neo/learning-stats/reset")
async def reset_learning_stats():
    """Reset learning statistics (use with caution)"""
    if LEARNING_FILE.exists():
        # Backup first
        backup = LEARNING_FILE.with_suffix('.backup.json')
        LEARNING_FILE.rename(backup)
    
    return {"status": "reset", "message": "Learning stats reset. Backup saved."}


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SIGNAL ENDPOINTS (For Ghost polling with INVALIDATION support)
# ═══════════════════════════════════════════════════════════════════════════════

def get_realtime_price(symbol: str) -> float:
    """Get real-time price for invalidation checks"""
    try:
        import yfinance as yf
        ticker_map = {"XAUUSD": "GC=F", "IREN": "IREN"}
        ticker = ticker_map.get(symbol, symbol)
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch real-time price for {symbol}: {e}")
    return 0.0


def check_invalidation_xauusd(plan: Dict, current_price: float) -> Dict:
    """
    Check if XAUUSD signal should be invalidated.
    
    Returns invalidation status with:
    - valid: bool
    - level: Price that invalidates
    - reason: Why invalidated (if applicable)
    """
    entry = plan["entry"]["level"]
    stop_loss = plan["risk"]["stop_loss"]
    direction = plan["direction"]
    strategy = plan["strategy"]
    confidence = plan["confidence"]
    
    # Default: signal is valid
    result = {
        "valid": True,
        "level": stop_loss,
        "reason": None,
        "confidence_check": "PASS"
    }
    
    # Rule 1: Confidence too low
    if confidence < 50:
        result["valid"] = False
        result["reason"] = f"LOW_CONFIDENCE: {confidence}% < 50%"
        result["confidence_check"] = "FAIL"
        return result
    
    # Rule 2: Strategy is WAIT
    if strategy == "WAIT":
        result["valid"] = False
        result["reason"] = "NO_SETUP: Strategy is WAIT"
        return result
    
    # Rule 3: Price moved too far from entry (missed the trade)
    if direction == "SELL" and current_price < entry - 50:
        result["valid"] = False
        result["reason"] = f"MISSED_ENTRY: Price {current_price:.0f} too far below entry {entry:.0f}"
        result["level"] = entry - 50
        return result
    
    if direction == "BUY" and current_price > entry + 50:
        result["valid"] = False
        result["reason"] = f"MISSED_ENTRY: Price {current_price:.0f} too far above entry {entry:.0f}"
        result["level"] = entry + 50
        return result
    
    # Rule 4: Stop loss would be hit (dangerous zone)
    if direction == "SELL" and current_price > stop_loss:
        result["valid"] = False
        result["reason"] = f"SL_BREACH: Price {current_price:.0f} above SL {stop_loss:.0f}"
        return result
    
    if direction == "BUY" and current_price < stop_loss:
        result["valid"] = False
        result["reason"] = f"SL_BREACH: Price {current_price:.0f} below SL {stop_loss:.0f}"
        return result
    
    # Rule 5: Time-based validity (check if still in valid window)
    current_hour = datetime.utcnow().hour
    if current_hour < 6 or current_hour > 20:
        result["valid"] = False
        result["reason"] = f"OUTSIDE_HOURS: Current hour {current_hour} UTC not in 06:00-20:00"
        return result
    
    return result


def check_invalidation_iren(plan: Dict, current_price: float) -> Dict:
    """Check if IREN signal should be invalidated"""
    entry = plan["shares"]["entry_level"]
    stop_loss = plan["shares"]["stop_loss"]
    direction = plan["direction"]
    strategy = plan["strategy"]
    confidence = plan["confidence"]
    btc_change = plan["btc_correlation"]["btc_change_24h"]
    
    result = {
        "valid": True,
        "level": stop_loss,
        "reason": None,
        "confidence_check": "PASS"
    }
    
    # Rule 1: Confidence too low
    if confidence < 50:
        result["valid"] = False
        result["reason"] = f"LOW_CONFIDENCE: {confidence}% < 50%"
        result["confidence_check"] = "FAIL"
        return result
    
    # Rule 2: Strategy is WAIT
    if strategy == "WAIT" or direction == "WAIT":
        result["valid"] = False
        result["reason"] = "NO_SETUP: Strategy is WAIT"
        return result
    
    # Rule 3: BTC crash invalidates long
    if direction == "BUY" and btc_change < -5:
        result["valid"] = False
        result["reason"] = f"BTC_CRASH: BTC {btc_change:.1f}% invalidates long"
        return result
    
    # Rule 4: Stop loss breach
    if direction == "BUY" and current_price < stop_loss:
        result["valid"] = False
        result["reason"] = f"SL_BREACH: Price ${current_price:.2f} below SL ${stop_loss:.2f}"
        return result
    
    # Rule 5: Missed entry (price ran away)
    if direction == "BUY" and current_price > entry * 1.10:
        result["valid"] = False
        result["reason"] = f"MISSED_ENTRY: Price ${current_price:.2f} > 10% above entry ${entry:.2f}"
        result["level"] = entry * 1.10
        return result
    
    # Rule 6: Market hours check (US market)
    current_hour = datetime.utcnow().hour
    # US market: 14:30 - 21:00 UTC
    if current_hour < 13 or current_hour > 21:
        result["valid"] = False
        result["reason"] = f"OUTSIDE_HOURS: Current hour {current_hour} UTC not in US market hours"
        return result
    
    return result


@app.get("/api/neo/xauusd/signal")
async def get_xauusd_quick_signal():
    """
    Quick signal endpoint for Ghost - with INVALIDATION support.
    
    Ghost should poll this every 20 minutes and check:
    - valid: true/false
    - invalidation.level: Price that breaks the trade
    - invalidation.reason: Why invalidated
    - last_updated: When signal was last checked
    """
    plan = await get_xauusd_daily_plan()
    current_price = get_realtime_price("XAUUSD") or plan["current_price"]
    
    # Check invalidation rules
    invalidation = check_invalidation_xauusd(plan, current_price)
    
    return {
        "symbol": "XAUUSD",
        "action": plan["direction"],
        "strategy": plan["strategy"],
        "entry": plan["entry"]["level"],
        "stop_loss": plan["risk"]["stop_loss"],
        "take_profit": plan["risk"]["take_profit_1"],
        "take_profit_2": plan["risk"]["take_profit_2"],
        "confidence": plan["confidence"],
        "current_price": current_price,
        
        # INVALIDATION FIELDS (for Ghost's 20-min checks)
        "valid": invalidation["valid"],
        "invalidation": {
            "level": invalidation["level"],
            "reason": invalidation["reason"],
            "confidence_check": invalidation["confidence_check"]
        },
        
        "last_updated": datetime.utcnow().isoformat(),
        "timestamp": plan["timestamp"],
        "signal_id": plan.get("signal_id", f"NEO_XAUUSD_{datetime.utcnow().strftime('%Y%m%d')}")
    }


@app.get("/api/neo/iren/signal")
async def get_iren_quick_signal():
    """
    Quick signal endpoint for Ghost - with INVALIDATION support.
    """
    plan = await get_iren_daily_plan()
    current_price = get_realtime_price("IREN") or plan["current_price"]
    
    # Check invalidation rules
    invalidation = check_invalidation_iren(plan, current_price)
    
    return {
        "symbol": "IREN",
        "action": plan["direction"],
        "strategy": plan["strategy"],
        "entry": plan["shares"]["entry_level"],
        "stop_loss": plan["shares"]["stop_loss"],
        "take_profit": plan["shares"]["take_profit"],
        "confidence": plan["confidence"],
        "current_price": current_price,
        "btc_change": plan["btc_correlation"]["btc_change_24h"],
        
        # INVALIDATION FIELDS
        "valid": invalidation["valid"],
        "invalidation": {
            "level": invalidation["level"],
            "reason": invalidation["reason"],
            "confidence_check": invalidation["confidence_check"]
        },
        
        "last_updated": datetime.utcnow().isoformat(),
        "timestamp": plan["timestamp"],
        "signal_id": plan.get("signal_id", f"NEO_IREN_{datetime.utcnow().strftime('%Y%m%d')}")
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERTREND COMMAND SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# Store last SuperTrend state for change detection
_supertrend_state = {"XAUUSD": None, "IREN": None}


def calculate_supertrend_command(symbol: str) -> Dict:
    """
    Calculate SuperTrend direction based on:
    - EMA 20/50 crossovers (Golden/Death cross)
    - Price vs EMA 200
    - Higher highs/Lower lows pattern
    - Volume confirmation
    
    Returns: LONG, SHORT, or NEUTRAL with confidence and reasoning
    """
    import yfinance as yf
    import pandas as pd
    
    try:
        ticker_map = {"XAUUSD": "GC=F", "IREN": "IREN"}
        ticker = ticker_map.get(symbol, symbol)
        
        # Fetch data
        df = yf.Ticker(ticker).history(period="30d", interval="1h")
        if df.empty or len(df) < 200:
            return {"supertrend": "NEUTRAL", "supertrend_confidence": 0, "supertrend_reason": "Insufficient data"}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Calculate EMAs
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean() if len(close) >= 200 else close.ewm(span=len(close)).mean()
        
        current_price = float(close.iloc[-1])
        current_ema20 = float(ema20.iloc[-1])
        current_ema50 = float(ema50.iloc[-1])
        current_ema200 = float(ema200.iloc[-1])
        
        prev_ema20 = float(ema20.iloc[-2])
        prev_ema50 = float(ema50.iloc[-2])
        
        # Scoring system
        score = 0
        reasons = []
        
        # 1. EMA 20/50 Cross (Golden Cross = +30, Death Cross = -30)
        golden_cross = prev_ema20 < prev_ema50 and current_ema20 > current_ema50
        death_cross = prev_ema20 > prev_ema50 and current_ema20 < current_ema50
        ema_bullish = current_ema20 > current_ema50
        ema_bearish = current_ema20 < current_ema50
        
        if golden_cross:
            score += 35
            reasons.append("🌟 Golden cross (EMA20 crossed above EMA50)")
        elif death_cross:
            score -= 35
            reasons.append("💀 Death cross (EMA20 crossed below EMA50)")
        elif ema_bullish:
            score += 15
            reasons.append("EMA20 > EMA50 (bullish)")
        elif ema_bearish:
            score -= 15
            reasons.append("EMA20 < EMA50 (bearish)")
        
        # 2. Price vs EMA200 (+25 or -25)
        if current_price > current_ema200 * 1.01:  # 1% above
            score += 25
            reasons.append("Price above EMA200 (bullish trend)")
        elif current_price < current_ema200 * 0.99:  # 1% below
            score -= 25
            reasons.append("Price below EMA200 (bearish trend)")
        
        # 3. Higher Highs / Lower Lows (+20 or -20)
        recent_highs = high.iloc[-20:]
        recent_lows = low.iloc[-20:]
        
        hh = recent_highs.iloc[-5:].max() > recent_highs.iloc[-10:-5].max()
        hl = recent_lows.iloc[-5:].min() > recent_lows.iloc[-10:-5].min()
        lh = recent_highs.iloc[-5:].max() < recent_highs.iloc[-10:-5].max()
        ll = recent_lows.iloc[-5:].min() < recent_lows.iloc[-10:-5].min()
        
        if hh and hl:
            score += 20
            reasons.append("Higher highs & higher lows (uptrend)")
        elif lh and ll:
            score -= 20
            reasons.append("Lower highs & lower lows (downtrend)")
        
        # 4. Volume confirmation (+10 or -10)
        avg_volume = volume.iloc[-20:].mean()
        recent_volume = volume.iloc[-5:].mean()
        volume_increasing = recent_volume > avg_volume * 1.2
        
        if volume_increasing:
            if score > 0:
                score += 10
                reasons.append("Volume confirming bullish move")
            elif score < 0:
                score -= 10
                reasons.append("Volume confirming bearish move")
        
        # Determine SuperTrend direction
        if score >= 30:
            supertrend = "LONG"
            confidence = min(90, 50 + score)
        elif score <= -30:
            supertrend = "SHORT"
            confidence = min(90, 50 + abs(score))
        else:
            supertrend = "NEUTRAL"
            confidence = 50
        
        # Check if changed from last state
        global _supertrend_state
        last_state = _supertrend_state.get(symbol)
        changed = last_state is not None and last_state != supertrend
        _supertrend_state[symbol] = supertrend
        
        return {
            "supertrend": supertrend,
            "supertrend_confidence": confidence,
            "supertrend_changed": changed,
            "supertrend_reason": " | ".join(reasons) if reasons else "No clear trend",
            "supertrend_score": score,
            "indicators": {
                "ema20": round(current_ema20, 2),
                "ema50": round(current_ema50, 2),
                "ema200": round(current_ema200, 2),
                "price": round(current_price, 2),
                "golden_cross": golden_cross,
                "death_cross": death_cross
            }
        }
        
    except Exception as e:
        logger.error(f"SuperTrend calculation error for {symbol}: {e}")
        return {
            "supertrend": "NEUTRAL",
            "supertrend_confidence": 0,
            "supertrend_changed": False,
            "supertrend_reason": f"Error: {str(e)}"
        }


def get_consensus_signal(symbol: str) -> Dict:
    """Get unified consensus signal combining NEO + Meta Bot"""
    neo_action = "HOLD"
    neo_conf = 0
    meta_action = "HOLD"
    meta_conf = 0
    st = {"supertrend": "NEUTRAL", "supertrend_confidence": 0, "supertrend_changed": False, "supertrend_reason": ""}
    conflict_warning = None
    
    try:
        # Get Meta Bot signal
        try:
            meta_resp = requests.get(f"http://localhost:8035/api/meta/ghost/{symbol.lower()}", timeout=5)
            if meta_resp.ok:
                meta = meta_resp.json()
                meta_action = meta.get("action", "HOLD").upper()
                meta_conf = meta.get("confidence", 0) or 0
        except Exception as e:
            logger.warning(f"Could not fetch Meta Bot: {e}")
        
        # Get NEO prediction
        try:
            pred = get_today_prediction(symbol.upper())
            if pred:
                exp_dir = pred.get("expected_direction", "")
                neo_action = "SELL" if exp_dir == "SHORTS" else "BUY" if exp_dir == "LONGS" else "HOLD"
                neo_conf = pred.get("primary_probability", 0) or 0
        except Exception as e:
            logger.warning(f"Could not load NEO prediction: {e}")
        
        # Calculate SuperTrend command
        try:
            st = calculate_supertrend_command(symbol.upper())
        except Exception as e:
            logger.warning(f"Could not calculate SuperTrend: {e}")
        
        # Consensus logic
        if neo_action == meta_action and neo_action != "HOLD":
            action = neo_action
            confidence = int((neo_conf + meta_conf) / 2 * 1.2)
            consensus_level = "STRONG"
        elif neo_action == "HOLD" and meta_action == "HOLD":
            action = "HOLD"
            confidence = 50
            consensus_level = "NEUTRAL"
        elif (neo_action == "BUY" and meta_action == "SELL") or (neo_action == "SELL" and meta_action == "BUY"):
            action = "HOLD"
            confidence = 30
            consensus_level = "CONFLICT"
            conflict_warning = f"⚠️ CONFLICT: NEO says {neo_action}, Meta says {meta_action}. WAITING for alignment."
        elif neo_action in ["BUY", "SELL"] and meta_action == "HOLD":
            action = neo_action
            confidence = int(neo_conf * 0.7) if neo_conf else 50
            consensus_level = "MEDIUM"
        elif meta_action in ["BUY", "SELL"] and neo_action == "HOLD":
            action = meta_action
            confidence = int(meta_conf * 0.7) if meta_conf else 50
            consensus_level = "MEDIUM"
        else:
            action = "HOLD"
            confidence = 40
            consensus_level = "WEAK"
        
        return {
            "action": action,
            "confidence": min(confidence, 95),
            "consensus_level": consensus_level,
            "neo": {"action": neo_action, "confidence": neo_conf},
            "meta": {"action": meta_action, "confidence": meta_conf},
            "supertrend": st,
            "conflict_warning": conflict_warning
        }
    except Exception as e:
        logger.error(f"Consensus error: {e}")
        return {
            "action": "HOLD",
            "confidence": 0,
            "consensus_level": "ERROR",
            "neo": {"action": neo_action, "confidence": neo_conf},
            "meta": {"action": meta_action, "confidence": meta_conf},
            "supertrend": st,
            "conflict_warning": f"Error: {str(e)}"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SIGNAL ENDPOINT (Ghost should use THIS!)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/neo/xauusd/unified")
@app.get("/api/consensus/xauusd")
async def get_xauusd_unified():
    """
    UNIFIED signal for Ghost - combines NEO + Meta + SuperTrend command.
    Ghost should use THIS endpoint for all decisions!
    """
    plan = await get_xauusd_daily_plan()
    current_price = get_realtime_price("XAUUSD") or plan["current_price"]
    
    # Get consensus (NEO + Meta)
    consensus = get_consensus_signal("XAUUSD")
    
    # Get SuperTrend command
    st = consensus.get("supertrend", {})
    
    # Check invalidation
    invalidation = check_invalidation_xauusd(plan, current_price)
    
    return {
        "symbol": "XAUUSD",
        "action": consensus["action"],
        "confidence": consensus["confidence"],
        "consensus_level": consensus["consensus_level"],
        
        # SuperTrend Command (NEW - Ghost uses this!)
        "supertrend": st.get("supertrend", "NEUTRAL"),
        "supertrend_confidence": st.get("supertrend_confidence", 0),
        "supertrend_changed": st.get("supertrend_changed", False),
        "supertrend_reason": st.get("supertrend_reason", ""),
        
        # Individual signals
        "neo": consensus["neo"],
        "meta": consensus["meta"],
        
        # Trading levels
        "entry": plan["entry"]["level"],
        "stop_loss": plan["risk"]["stop_loss"],
        "take_profit": plan["risk"]["take_profit_1"],
        "take_profit_2": plan["risk"]["take_profit_2"],
        "current_price": current_price,
        
        # Validation
        "valid": invalidation["valid"] and consensus["action"] != "HOLD",
        "invalidation": invalidation,
        "conflict_warning": consensus.get("conflict_warning"),
        
        "last_updated": datetime.utcnow().isoformat(),
        "timestamp": plan["timestamp"]
    }


@app.get("/api/neo/iren/unified")
@app.get("/api/consensus/iren")
async def get_iren_unified():
    """UNIFIED signal for IREN - combines NEO + Meta + SuperTrend command."""
    plan = await get_iren_daily_plan()
    current_price = get_realtime_price("IREN") or plan["current_price"]
    
    # Get consensus (NEO + Meta)
    consensus = get_consensus_signal("IREN")
    
    # Get SuperTrend command
    st = consensus.get("supertrend", {})
    
    # Check invalidation
    invalidation = check_invalidation_iren(plan, current_price)
    
    return {
        "symbol": "IREN",
        "action": consensus["action"],
        "confidence": consensus["confidence"],
        "consensus_level": consensus["consensus_level"],
        
        # SuperTrend Command
        "supertrend": st.get("supertrend", "NEUTRAL"),
        "supertrend_confidence": st.get("supertrend_confidence", 0),
        "supertrend_changed": st.get("supertrend_changed", False),
        "supertrend_reason": st.get("supertrend_reason", ""),
        
        # Individual signals
        "neo": consensus["neo"],
        "meta": consensus["meta"],
        
        # Trading levels
        "entry": plan["shares"]["entry_level"],
        "stop_loss": plan["shares"]["stop_loss"],
        "take_profit": plan["shares"]["take_profit"],
        "current_price": current_price,
        "btc_change": plan["btc_correlation"]["btc_change_24h"],
        
        # Validation
        "valid": invalidation["valid"] and consensus["action"] != "HOLD",
        "invalidation": invalidation,
        "conflict_warning": consensus.get("conflict_warning"),
        
        "last_updated": datetime.utcnow().isoformat(),
        "timestamp": plan["timestamp"]
    }


@app.get("/api/neo/supertrend/{symbol}")
async def get_supertrend_only(symbol: str):
    """Get just the SuperTrend command for a symbol"""
    st = calculate_supertrend_command(symbol.upper())
    return {
        "symbol": symbol.upper(),
        **st,
        "timestamp": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "ghost_integration_api:app",
        host="0.0.0.0",
        port=8036,
        reload=False,
        log_level="info"
    )

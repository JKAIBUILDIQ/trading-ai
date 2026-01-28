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
        ticker_map = {"XAUUSD": "GC=F", "IREN": "IREN", "CLSK": "CLSK", "CIFR": "CIFR"}
        ticker = ticker_map.get(symbol, symbol)
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch real-time price for {symbol}: {e}")
    return 0.0


def generate_fresh_xauusd_signal(current_price: float) -> Dict:
    """
    Generate a FRESH intraday signal based on current market conditions.
    
    Called when the pre-market signal becomes stale (price moved significantly).
    This ensures Ghost always has actionable, up-to-date signals.
    
    🎓 LEARNING INTEGRATED:
    - Uses learned feature weights from /home/jbot/trading_ai/neo/learning/
    - Avoids SELL signals during strong uptrends (learned from mistakes)
    - RSI overbought weight reduced to 0.1 (was causing bad SELL signals)
    """
    import yfinance as yf
    import numpy as np
    from pathlib import Path
    
    logger.info(f"🔄 GENERATING FRESH XAUUSD SIGNAL @ ${current_price:.2f}")
    
    # Load learned weights
    weights_file = Path("/home/jbot/trading_ai/neo/learning/feature_weights.json")
    weights = {}
    if weights_file.exists():
        import json
        with open(weights_file) as f:
            weights = json.load(f)
        logger.info(f"📚 Loaded learned weights: RSI_overbought={weights.get('rsi_overbought', 'N/A')}")
    
    try:
        # Fetch recent data for analysis
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="5d", interval="1h")
        
        if df.empty:
            return _fallback_signal("XAUUSD", current_price, "NO_DATA")
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Calculate EMAs
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        
        # Calculate ATR for stop loss
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift(1)), 
                                  abs(low - close.shift(1))))
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        # Support/Resistance from recent swing points
        recent_high = float(high.tail(20).max())
        recent_low = float(low.tail(20).min())
        
        # Determine trend and direction
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        
        # DETERMINE SIGNAL BASED ON CURRENT CONDITIONS
        if current_price > ema20_val > ema50_val:
            # BULLISH: Price above rising EMAs
            direction = "BUY"
            strategy = "BUY_THE_DIP"
            
            # Entry zone: near EMA20 or recent support
            entry_zone_low = max(ema20_val - atr * 0.5, recent_low)
            entry_zone_high = ema20_val + atr * 0.3
            optimal_entry = (entry_zone_low + entry_zone_high) / 2
            
            # Stop below recent swing low
            stop_loss = min(ema50_val, recent_low) - atr * 0.5
            
            # Targets
            take_profit_1 = current_price + atr * 1.5
            take_profit_2 = current_price + atr * 3.0
            
            reasoning = f"BULLISH: Price ${current_price:.0f} > EMA20 ${ema20_val:.0f} > EMA50 ${ema50_val:.0f}"
            confidence = 70 if current_rsi < 70 else 55  # Lower conf if overbought
            
        elif current_price < ema20_val < ema50_val:
            # BEARISH: Price below falling EMAs
            # 🎓 LEARNED: Check if we're in a macro uptrend before SELL
            macro_bullish = current_price > close.rolling(50).mean().iloc[-1]
            
            if macro_bullish:
                # Don't SELL in macro uptrend - learned from mistakes!
                direction = "HOLD"
                strategy = "MACRO_UPTREND_PROTECTION"
                reasoning = f"🎓 LEARNED: Avoiding SELL in macro uptrend. Price > 50-period avg."
                confidence = 40
                logger.warning(f"🎓 LEARNING APPLIED: Blocked SELL signal in macro uptrend")
            else:
                direction = "SELL"
                strategy = "SELL_THE_RALLY"
                reasoning = f"BEARISH: Price ${current_price:.0f} < EMA20 ${ema20_val:.0f} < EMA50 ${ema50_val:.0f}"
                confidence = 70 if current_rsi > 30 else 55
            
            # Entry zone: near EMA20 or recent resistance
            entry_zone_low = ema20_val - atr * 0.3
            entry_zone_high = min(ema20_val + atr * 0.5, recent_high)
            optimal_entry = (entry_zone_low + entry_zone_high) / 2
            
            # Stop above recent swing high
            stop_loss = max(ema50_val, recent_high) + atr * 0.5
            
            # Targets
            take_profit_1 = current_price - atr * 1.5
            take_profit_2 = current_price - atr * 3.0
            
        else:
            # MIXED/CONSOLIDATION - Wait for breakout or buy dip
            # 🎓 LEARNED: In Gold, prefer BUY on dips (fundamental thesis)
            rsi_overbought_weight = weights.get('rsi_overbought', 0.3)  # Default 0.3, learned may be 0.1
            
            if current_rsi < 40:
                direction = "BUY"
                strategy = "OVERSOLD_BOUNCE"
                optimal_entry = current_price
                stop_loss = recent_low - atr * 0.5
                take_profit_1 = ema20_val
                take_profit_2 = ema50_val
                confidence = 65  # Increased from 60 - oversold bounces work better
                reasoning = f"OVERSOLD: RSI {current_rsi:.0f} < 40, looking for bounce (Gold dip = opportunity)"
            elif current_rsi > 60 and rsi_overbought_weight > 0.2:
                # 🎓 LEARNED: Only SELL on overbought if weight is still significant
                # Weight was reduced to 0.1 due to bad SELL signals, so this won't trigger!
                direction = "SELL"
                strategy = "OVERBOUGHT_FADE"
                optimal_entry = current_price
                stop_loss = recent_high + atr * 0.5
                take_profit_1 = ema20_val
                take_profit_2 = ema50_val
                confidence = int(55 * rsi_overbought_weight)  # Reduced by learned weight!
                reasoning = f"OVERBOUGHT: RSI {current_rsi:.0f} > 60 (weight={rsi_overbought_weight})"
            elif current_rsi > 60 and rsi_overbought_weight <= 0.2:
                # 🎓 LEARNED: RSI overbought weight too low - prefer HOLD/BUY
                direction = "HOLD"
                strategy = "RSI_OVERBOUGHT_IGNORED"
                optimal_entry = current_price
                stop_loss = recent_low - atr * 0.5
                take_profit_1 = current_price + atr
                take_profit_2 = current_price + atr * 2
                confidence = 45
                reasoning = f"🎓 LEARNED: RSI overbought ignored (weight={rsi_overbought_weight}). Gold can stay overbought in uptrends!"
                logger.info(f"🎓 LEARNING APPLIED: RSI overbought signal ignored due to low weight")
            else:
                direction = "WAIT"
                strategy = "NO_CLEAR_SETUP"
                optimal_entry = current_price
                stop_loss = current_price - atr
                take_profit_1 = current_price + atr
                take_profit_2 = current_price + atr * 2
                confidence = 40
                reasoning = f"CONSOLIDATION: EMAs mixed, RSI neutral at {current_rsi:.0f}"
        
        # Distance from optimal entry
        distance_from_entry = abs(current_price - optimal_entry)
        in_entry_zone = distance_from_entry < atr * 0.5
        
        signal = {
            "symbol": "XAUUSD",
            "signal_type": "FRESH_INTRADAY",
            "generated_at": datetime.utcnow().isoformat(),
            "action": direction,
            "direction": direction,
            "strategy": strategy,
            
            "current_price": round(current_price, 2),
            "entry": round(optimal_entry, 2),
            "entry_zone_low": round(entry_zone_low if direction != "WAIT" else current_price - atr * 0.5, 2),
            "entry_zone_high": round(entry_zone_high if direction != "WAIT" else current_price + atr * 0.5, 2),
            "in_entry_zone": in_entry_zone,
            "distance_to_entry": round(distance_from_entry, 2),
            
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            
            "confidence": confidence,
            "reasoning": reasoning,
            
            "technicals": {
                "ema20": round(ema20_val, 2),
                "ema50": round(ema50_val, 2),
                "rsi": round(current_rsi, 1),
                "atr": round(atr, 2),
                "recent_high": round(recent_high, 2),
                "recent_low": round(recent_low, 2),
            },
            
            "valid": direction != "WAIT" and confidence >= 50,
            "invalidation": {
                "level": round(stop_loss, 2),
                "reason": None if direction != "WAIT" else "NO_CLEAR_SETUP",
                "confidence_check": "PASS" if confidence >= 50 else "FAIL"
            },
            
            "last_updated": datetime.utcnow().isoformat(),
            "signal_id": f"NEO_XAUUSD_FRESH_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        }
        
        logger.info(f"✅ FRESH SIGNAL: {direction} @ ${optimal_entry:.0f} (conf: {confidence}%)")
        return signal
        
    except Exception as e:
        logger.error(f"Error generating fresh signal: {e}")
        return _fallback_signal("XAUUSD", current_price, str(e))


def _fallback_signal(symbol: str, current_price: float, error: str) -> Dict:
    """Return a safe fallback signal when generation fails"""
    return {
        "symbol": symbol,
        "signal_type": "FALLBACK",
        "generated_at": datetime.utcnow().isoformat(),
        "action": "WAIT",
        "direction": "WAIT",
        "strategy": "ERROR_FALLBACK",
        "current_price": round(current_price, 2),
        "entry": round(current_price, 2),
        "stop_loss": round(current_price - 50, 2),
        "take_profit_1": round(current_price + 50, 2),
        "take_profit_2": round(current_price + 100, 2),
        "confidence": 0,
        "reasoning": f"Fallback due to: {error}",
        "valid": False,
        "invalidation": {
            "level": round(current_price - 50, 2),
            "reason": f"FALLBACK: {error}",
            "confidence_check": "FAIL"
        },
        "last_updated": datetime.utcnow().isoformat(),
        "signal_id": f"NEO_{symbol}_FALLBACK_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    }


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
    Quick signal endpoint for Ghost - with INVALIDATION support + FRESH SIGNAL GENERATION.
    
    Ghost should poll this every 20 minutes and check:
    - valid: true/false
    - invalidation.level: Price that breaks the trade
    - invalidation.reason: Why invalidated
    - last_updated: When signal was last checked
    
    🆕 NEW: When pre-market signal becomes stale, this endpoint generates a FRESH
    intraday signal based on current market conditions instead of returning invalid data.
    """
    plan = await get_xauusd_daily_plan()
    current_price = get_realtime_price("XAUUSD") or plan["current_price"]
    
    # Check invalidation rules against the pre-market plan
    invalidation = check_invalidation_xauusd(plan, current_price)
    
    # 🆕 NEW: If signal is invalidated due to MISSED_ENTRY, generate FRESH signal!
    if not invalidation["valid"]:
        reason = invalidation.get("reason", "")
        
        # These invalidation types mean we should generate a fresh signal
        should_regenerate = any([
            "MISSED_ENTRY" in reason,  # Price moved significantly
            "SL_BREACH" in reason,     # Original setup failed
        ])
        
        if should_regenerate:
            logger.info(f"⚠️ Pre-market signal stale ({reason}). Generating FRESH signal...")
            fresh_signal = generate_fresh_xauusd_signal(current_price)
            fresh_signal["original_plan_invalidated"] = True
            fresh_signal["invalidation_reason"] = reason
            return fresh_signal
    
    # Original pre-market signal still valid - return it
    return {
        "symbol": "XAUUSD",
        "signal_type": "PREMARKET_PLAN",
        "action": plan["direction"],
        "strategy": plan["strategy"],
        "entry": plan["entry"]["level"],
        "entry_zone_low": plan["entry"]["level"] - 15,
        "entry_zone_high": plan["entry"]["level"] + 15,
        "stop_loss": plan["risk"]["stop_loss"],
        "take_profit": plan["risk"]["take_profit_1"],
        "take_profit_1": plan["risk"]["take_profit_1"],
        "take_profit_2": plan["risk"]["take_profit_2"],
        "confidence": plan["confidence"],
        "current_price": current_price,
        "in_entry_zone": abs(current_price - plan["entry"]["level"]) < 20,
        "distance_to_entry": abs(current_price - plan["entry"]["level"]),
        
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

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING ENDPOINTS - Track NEO's mistakes and improvements
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, '/home/jbot/trading_ai/neo/learning')

@app.get("/api/neo/learning/stats")
async def get_neo_learning_stats():
    """Get NEO's learning statistics - what it has learned from mistakes"""
    try:
        from neo_trainer import get_learning_stats, get_recent_mistakes
        stats = get_learning_stats()
        recent_mistakes = get_recent_mistakes(5)
        
        return {
            "status": "success",
            "learning_stats": stats,
            "recent_mistakes": recent_mistakes,
            "key_learnings": [
                {
                    "lesson": "RSI overbought does NOT mean SELL in strong Gold uptrends",
                    "weight_before": 0.3,
                    "weight_after": stats.get("current_weights", {}).get("rsi_overbought", 0.1),
                    "outcome": "Now ignoring RSI > 70 as SELL trigger"
                },
                {
                    "lesson": "Bear flag patterns often fail in Gold uptrends",
                    "weight_before": 0.6,
                    "weight_after": stats.get("current_weights", {}).get("bear_flag", 0.1),
                    "outcome": "Reduced bearish pattern weight significantly"
                }
            ],
            "biggest_problem": stats.get("biggest_problem"),
            "accuracy_trend": f"{stats.get('accuracy_overall', 0):.1f}% overall → {stats.get('accuracy_recent_20', 0):.1f}% recent"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/neo/learning/teach")
async def teach_neo_manually(lesson: Dict):
    """
    Manually teach NEO about a mistake.
    
    Body:
    {
        "symbol": "XAUUSD",
        "action_taken": "SELL",
        "price_at_signal": 5100,
        "actual_price": 5300,
        "reason": "NEO sold during rally, price went up 4%"
    }
    """
    try:
        from neo_trainer import force_learn_from_current_state
        
        result = force_learn_from_current_state(
            symbol=lesson.get("symbol", "XAUUSD"),
            price_when_sold=lesson.get("price_at_signal"),
            current_price=lesson.get("actual_price"),
            action=lesson.get("action_taken", "SELL")
        )
        
        return {
            "status": "success",
            "message": "NEO has learned from this mistake",
            "grade": result["grade"],
            "lesson": result.get("lesson", ""),
            "weights_adjusted": True
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/neo/learning/weights")
async def get_neo_feature_weights():
    """Get current feature weights used in signal generation"""
    try:
        from neo_trainer import load_weights
        weights = load_weights()
        
        # Categorize weights
        high_weight = {k: v for k, v in weights.items() if v >= 1.0}
        medium_weight = {k: v for k, v in weights.items() if 0.5 <= v < 1.0}
        low_weight = {k: v for k, v in weights.items() if v < 0.5}
        
        return {
            "status": "success",
            "weights": weights,
            "by_priority": {
                "high_impact": high_weight,
                "medium_impact": medium_weight,
                "low_impact_or_penalized": low_weight
            },
            "explanation": {
                "rsi_overbought": "REDUCED to 0.1 - was causing bad SELL signals during Gold rallies",
                "bear_flag": "REDUCED to 0.1 - patterns often wrong in strong uptrends",
                "ema_trend": "HIGH at 1.5 - trend following is key",
                "supertrend": "HIGH at 1.4 - reliable trend indicator"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SIGNAL ENDPOINT - NEO + META BOT COMBINED
# ═══════════════════════════════════════════════════════════════════════════════

import httpx

async def get_meta_bot_signal(symbol: str) -> Optional[Dict]:
    """Fetch signal from Meta Bot API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:8035/api/meta/{symbol.lower()}/signal",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Meta Bot not available: {e}")
    return None


@app.get("/api/unified/xauusd/signal")
async def get_unified_xauusd_signal():
    """
    UNIFIED SIGNAL - Combines NEO + Meta Bot for consensus.
    
    This is the PRIMARY endpoint Ghost should use!
    
    Decision Matrix:
    - NEO BUY + Meta BUY → STRONG BUY (confidence avg)
    - NEO BUY + Meta HOLD → BUY (reduced confidence)
    - NEO BUY + Meta SELL → HOLD (conflict - wait)
    - NEO SELL + Meta SELL → STRONG SELL (but Gold bias overrides)
    - NEO HOLD + Meta HOLD → HOLD
    
    Gold Bias Rule: For XAUUSD, prefer LONG due to fundamental thesis.
    """
    # Get NEO signal
    neo_signal = await get_xauusd_quick_signal()
    
    # Get Meta Bot signal
    meta_signal = await get_meta_bot_signal("xauusd")
    
    # Default values
    neo_action = neo_signal.get("action", "HOLD")
    neo_confidence = neo_signal.get("confidence", 50)
    neo_direction = neo_signal.get("direction", neo_action)
    
    meta_action = "HOLD"
    meta_confidence = 50
    meta_reasoning = "Meta Bot unavailable"
    
    if meta_signal:
        meta_action = meta_signal.get("action", "HOLD")
        meta_confidence = meta_signal.get("confidence", 50)
        meta_reasoning = meta_signal.get("reasoning", "")
    
    # Consensus logic
    consensus_action = "HOLD"
    consensus_confidence = 50
    consensus_reason = ""
    
    # Both agree on BUY
    if neo_action == "BUY" and meta_action == "BUY":
        consensus_action = "BUY"
        consensus_confidence = (neo_confidence + meta_confidence) / 2
        consensus_reason = "CONSENSUS: NEO + Meta both BUY"
    
    # NEO BUY, Meta HOLD - lean BUY with reduced confidence
    elif neo_action == "BUY" and meta_action == "HOLD":
        consensus_action = "BUY"
        consensus_confidence = neo_confidence * 0.8  # 20% reduction
        consensus_reason = "NEO BUY, Meta HOLD - cautious BUY"
    
    # Both agree on SELL (but Gold bias favors waiting)
    elif neo_action == "SELL" and meta_action == "SELL":
        # Gold fundamental thesis: Don't short, wait for dip to buy
        consensus_action = "HOLD"
        consensus_confidence = 60
        consensus_reason = "GOLD BIAS: Both say SELL but prefer to HOLD for dip buying opportunity"
    
    # Conflict - NEO and Meta disagree
    elif (neo_action == "BUY" and meta_action == "SELL") or (neo_action == "SELL" and meta_action == "BUY"):
        consensus_action = "HOLD"
        consensus_confidence = 40
        consensus_reason = f"CONFLICT: NEO={neo_action}, Meta={meta_action} - waiting for agreement"
    
    # Default: Follow NEO with reduced confidence
    else:
        consensus_action = neo_action if neo_action != "HOLD" else "HOLD"
        consensus_confidence = neo_confidence * 0.75
        consensus_reason = f"Following NEO ({neo_action}) with caution"
    
    # Build unified response
    return {
        "symbol": "XAUUSD",
        "timestamp": datetime.utcnow().isoformat(),
        "signal_type": "UNIFIED_CONSENSUS",
        
        # UNIFIED SIGNAL (use this!)
        "action": consensus_action,
        "direction": consensus_action if consensus_action in ["BUY", "SELL"] else neo_direction,
        "confidence": round(consensus_confidence, 1),
        "reasoning": consensus_reason,
        
        # Trade parameters (from NEO)
        "current_price": neo_signal.get("current_price"),
        "entry": neo_signal.get("entry"),
        "entry_zone_low": neo_signal.get("entry_zone_low"),
        "entry_zone_high": neo_signal.get("entry_zone_high"),
        "in_entry_zone": neo_signal.get("in_entry_zone"),
        "stop_loss": neo_signal.get("stop_loss"),
        "take_profit_1": neo_signal.get("take_profit_1"),
        "take_profit_2": neo_signal.get("take_profit_2"),
        "strategy": neo_signal.get("strategy"),
        
        # Validity (from NEO)
        "valid": neo_signal.get("valid", True),
        "invalidation": neo_signal.get("invalidation"),
        
        # Individual signals (for debugging)
        "neo": {
            "action": neo_action,
            "confidence": neo_confidence,
            "strategy": neo_signal.get("strategy"),
            "signal_type": neo_signal.get("signal_type")
        },
        "meta": {
            "action": meta_action,
            "confidence": meta_confidence,
            "reasoning": meta_reasoning,
            "bullish_count": meta_signal.get("indicator_count", {}).get("bullish", 0) if meta_signal else 0,
            "bearish_count": meta_signal.get("indicator_count", {}).get("bearish", 0) if meta_signal else 0
        },
        
        # SUPERTREND (from NEO)
        "supertrend": neo_signal.get("supertrend", {}).get("direction", "UNKNOWN") if isinstance(neo_signal.get("supertrend"), dict) else "BULLISH",
        "supertrend_confidence": neo_signal.get("supertrend", {}).get("confidence", 50) if isinstance(neo_signal.get("supertrend"), dict) else 50,
        
        "last_updated": datetime.utcnow().isoformat(),
        "signal_id": f"UNIFIED_XAUUSD_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    }


@app.get("/api/unified/iren/signal")
async def get_unified_iren_signal():
    """UNIFIED IREN signal - NEO + Meta Bot combined"""
    # Get NEO signal
    neo_signal = await get_iren_quick_signal()
    
    # Get Meta Bot signal
    meta_signal = await get_meta_bot_signal("iren")
    
    neo_action = neo_signal.get("action", "HOLD")
    neo_confidence = neo_signal.get("confidence", 50)
    
    meta_action = "HOLD"
    meta_confidence = 50
    
    if meta_signal:
        meta_action = meta_signal.get("action", "HOLD")
        meta_confidence = meta_signal.get("confidence", 50)
    
    # Simple consensus for IREN (similar logic)
    if neo_action == "BUY" and meta_action == "BUY":
        action = "BUY"
        confidence = (neo_confidence + meta_confidence) / 2
        reason = "CONSENSUS: NEO + Meta both BUY"
    elif neo_action == "BUY" and meta_action == "HOLD":
        action = "BUY"
        confidence = neo_confidence * 0.8
        reason = "NEO BUY, Meta HOLD"
    elif neo_action == "SELL" and meta_action == "SELL":
        action = "SELL"
        confidence = (neo_confidence + meta_confidence) / 2
        reason = "CONSENSUS: NEO + Meta both SELL"
    else:
        action = "HOLD"
        confidence = 50
        reason = f"No consensus: NEO={neo_action}, Meta={meta_action}"
    
    return {
        "symbol": "IREN",
        "timestamp": datetime.utcnow().isoformat(),
        "signal_type": "UNIFIED_CONSENSUS",
        "action": action,
        "confidence": round(confidence, 1),
        "reasoning": reason,
        "current_price": neo_signal.get("current_price"),
        "entry": neo_signal.get("entry"),
        "stop_loss": neo_signal.get("stop_loss"),
        "take_profit": neo_signal.get("take_profit"),
        "valid": neo_signal.get("valid", True),
        "neo": {"action": neo_action, "confidence": neo_confidence},
        "meta": {"action": meta_action, "confidence": meta_confidence},
        "last_updated": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL DETECTOR ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# Import institutional detector
try:
    from institutional_detector import InstitutionalDetector
    institutional_detector = InstitutionalDetector()
    INSTITUTIONAL_DETECTOR_AVAILABLE = True
    logger.info("✅ Institutional Detector loaded successfully")
except ImportError as e:
    INSTITUTIONAL_DETECTOR_AVAILABLE = False
    institutional_detector = None
    logger.warning(f"⚠️ Institutional Detector not available: {e}")


@app.get("/api/neo/institutional/{symbol}")
async def get_institutional_analysis(symbol: str):
    """
    Get institutional analysis for a symbol.
    Includes: options flow, sentiment, stop hunt risk, cascade protection.
    
    This helps NEO see what Citadel sees - trade the TRADER, not just the chart.
    """
    if not INSTITUTIONAL_DETECTOR_AVAILABLE:
        return {
            "error": "Institutional detector not available",
            "symbol": symbol,
            "fallback": True,
            "signal": "NEUTRAL",
            "action": "FOLLOW_TREND"
        }
    
    # Get current price (would be from market feed in production)
    current_prices = {
        "XAUUSD": 5270.0,
        "GOLD": 5270.0,
        "GLD": 265.0,
        "IREN": 59.94,
        "CIFR": 12.50,
        "CLSK": 15.75
    }
    
    current_price = current_prices.get(symbol.upper(), 100.0)
    
    # Recent price levels (would be from market data in production)
    recent_lows = {
        "XAUUSD": [5200, 5180, 5150, 5100, 5050],
        "IREN": [52, 48, 45, 42, 40],
        "CIFR": [11, 10.5, 10, 9.5, 9],
        "CLSK": [14, 13.5, 13, 12.5, 12]
    }
    
    recent_highs = {
        "XAUUSD": [5280, 5300, 5320],
        "IREN": [60, 62, 65],
        "CIFR": [13, 13.5, 14],
        "CLSK": [16, 16.5, 17]
    }
    
    try:
        analysis = await institutional_detector.get_full_analysis(
            symbol=symbol.upper(),
            current_price=current_price,
            recent_lows=recent_lows.get(symbol.upper(), []),
            recent_highs=recent_highs.get(symbol.upper(), [])
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Institutional analysis error for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "fallback": True,
            "signal": "NEUTRAL",
            "action": "FOLLOW_TREND"
        }


@app.get("/api/neo/options-flow/{symbol}")
async def get_options_flow(symbol: str):
    """
    Get options flow analysis (put/call ratio, unusual activity).
    
    Key signals:
    - Put/Call > 2.0 = Retail euphoria (DANGER - institutions may dump)
    - Put/Call < 0.5 = Retail panic (OPPORTUNITY - institutions may buy)
    """
    if not INSTITUTIONAL_DETECTOR_AVAILABLE:
        return {"error": "Institutional detector not available", "fallback": True}
    
    try:
        return await institutional_detector.options_detector.get_put_call_ratio(symbol)
    except Exception as e:
        return {"error": str(e), "fallback": True}


@app.get("/api/neo/sentiment/{symbol}")
async def get_social_sentiment(symbol: str):
    """
    Get social media sentiment analysis.
    
    Key signals:
    - Score > 80 = EUPHORIA (everyone bullish = potential top)
    - Score < 20 = PANIC (everyone bearish = potential bottom)
    """
    if not INSTITUTIONAL_DETECTOR_AVAILABLE:
        return {"error": "Institutional detector not available", "fallback": True}
    
    try:
        return await institutional_detector.sentiment_detector.get_sentiment_score(symbol)
    except Exception as e:
        return {"error": str(e), "fallback": True}


@app.get("/api/neo/cascade-protection")
async def get_cascade_protection_status():
    """
    Get current cascade protection status.
    
    Unlike other algos that freeze at fixed 92% volatility,
    NEO uses RANDOMIZED thresholds (85-95%) to avoid coordinated hunts.
    """
    if not INSTITUTIONAL_DETECTOR_AVAILABLE:
        return {"error": "Institutional detector not available", "fallback": True}
    
    try:
        return institutional_detector.cascade_protection.get_current_threshold()
    except Exception as e:
        return {"error": str(e), "fallback": True}


@app.get("/api/neo/gold-thesis")
async def get_gold_thesis():
    """
    Get NEO's Gold macro thesis ($10K-$50K target).
    This is the fundamental backdrop for all Gold trading decisions.
    """
    thesis_file = DATA_DIR.parent / "learning" / "gold_macro_thesis.json"
    
    if thesis_file.exists():
        with open(thesis_file) as f:
            return json.load(f)
    
    return {
        "core_thesis": "Gold in multi-decade bull market",
        "2030_target": 15000,
        "2035_target": 50000,
        "bias": "EXTREME_LONG",
        "file_missing": True
    }


if __name__ == "__main__":
    uvicorn.run(
        "ghost_integration_api:app",
        host="0.0.0.0",
        port=8036,
        reload=False,
        log_level="info"
    )

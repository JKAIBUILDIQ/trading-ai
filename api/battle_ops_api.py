"""
BATTLE OPS API - Control daily operations via REST API
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops.daily_battle_ops import (
    DailyOps,
    run_pre_market,
    run_market_open,
    run_mid_day,
    run_power_hour,
    run_after_hours,
    run_all_phases
)
from ops.learning_engine import LearningEngine

router = APIRouter(prefix="/ops", tags=["Battle Operations"])


# ============ MODELS ============

class PhaseRequest(BaseModel):
    phase: str  # pre_market, market_open, mid_day, power_hour, after_hours, all

class TradeRecord(BaseModel):
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    patterns: Optional[List[str]] = []
    agents_involved: Optional[List[str]] = []
    notes: Optional[str] = None

class PredictionRecord(BaseModel):
    agent: str
    symbol: str
    direction: str  # BULLISH or BEARISH
    conviction: str  # LOW, MEDIUM, HIGH
    target_price: Optional[float] = None
    timeframe: Optional[str] = "1d"
    reasoning: Optional[str] = None

class VerifyPrediction(BaseModel):
    prediction_index: int
    actual_direction: str  # UP or DOWN
    actual_price: Optional[float] = None


# ============ ENDPOINTS ============

@router.get("/status")
async def get_ops_status():
    """Get current day's operations status"""
    ops = DailyOps()
    
    # Calculate completion stats
    total_tasks = 0
    completed_tasks = 0
    
    for phase in ops.state["phases"].values():
        for task in phase["tasks"].values():
            total_tasks += 1
            if task["status"] == "COMPLETED":
                completed_tasks += 1
    
    return {
        "date": ops.today,
        "defcon_level": ops.state["defcon_level"],
        "completion": f"{completed_tasks}/{total_tasks}",
        "completion_pct": round(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        "phases": {
            name: {
                "status": phase["status"],
                "started": phase.get("started_at"),
                "completed": phase.get("completed_at")
            }
            for name, phase in ops.state["phases"].items()
        },
        "performance": ops.state["performance"],
        "telegram_reports_sent": len(ops.state["telegram_reports_sent"])
    }


@router.get("/checklist")
async def get_checklist():
    """Get detailed checklist of all tasks"""
    ops = DailyOps()
    return {
        "date": ops.today,
        "phases": ops.state["phases"]
    }


@router.get("/checklist/telegram")
async def get_checklist_telegram():
    """Get checklist formatted for Telegram"""
    ops = DailyOps()
    return {
        "message": ops.generate_checklist_message()
    }


@router.post("/run-phase")
async def run_phase(request: PhaseRequest, background_tasks: BackgroundTasks):
    """Manually trigger a phase to run"""
    phase = request.phase.lower()
    
    valid_phases = ["pre_market", "market_open", "mid_day", "power_hour", "after_hours", "all"]
    if phase not in valid_phases:
        raise HTTPException(status_code=400, detail=f"Invalid phase. Must be one of: {valid_phases}")
    
    # Run in background
    async def execute_phase():
        if phase == "pre_market":
            await run_pre_market()
        elif phase == "market_open":
            await run_market_open()
        elif phase == "mid_day":
            await run_mid_day()
        elif phase == "power_hour":
            await run_power_hour()
        elif phase == "after_hours":
            await run_after_hours()
        elif phase == "all":
            await run_all_phases()
    
    background_tasks.add_task(lambda: asyncio.run(execute_phase()))
    
    return {
        "status": "started",
        "phase": phase,
        "message": f"Phase '{phase}' started in background"
    }


@router.post("/send-report")
async def send_report(phase: Optional[str] = None):
    """Send Telegram report for current status"""
    ops = DailyOps()
    
    if phase:
        sent = await ops.send_phase_report(phase)
    else:
        sent = await ops.send_daily_summary()
    
    return {
        "sent": sent,
        "phase": phase or "full_summary"
    }


@router.post("/send-morning-brief")
async def send_morning_brief():
    """Send morning brief with yesterday's review"""
    ops = DailyOps()
    await ops.send_morning_brief()
    return {"status": "sent"}


@router.get("/yesterday-review")
async def get_yesterday_review():
    """Get yesterday's performance review"""
    ops = DailyOps()
    return {
        "review": ops.generate_yesterday_review()
    }


@router.post("/record-trade")
async def record_trade(trade: TradeRecord):
    """Record a trade for performance tracking"""
    ops = DailyOps()
    
    trade_data = trade.dict()
    trade_data["timestamp"] = datetime.now().isoformat()
    
    if trade.exit_price and trade.entry_price:
        if trade.direction.upper() == "LONG":
            trade_data["pnl"] = trade.exit_price - trade.entry_price
        else:
            trade_data["pnl"] = trade.entry_price - trade.exit_price
    
    ops.record_trade(trade_data)
    
    return {
        "status": "recorded",
        "trade": trade_data,
        "today_performance": ops.state["performance"]
    }


@router.post("/record-prediction")
async def record_prediction(prediction: PredictionRecord):
    """Record an agent prediction for accuracy tracking"""
    ops = DailyOps()
    
    ops.record_prediction(
        agent=prediction.agent,
        symbol=prediction.symbol,
        prediction={
            "direction": prediction.direction,
            "conviction": prediction.conviction,
            "target_price": prediction.target_price,
            "timeframe": prediction.timeframe,
            "reasoning": prediction.reasoning
        }
    )
    
    return {
        "status": "recorded",
        "total_predictions_today": len(ops.state["agent_predictions"])
    }


@router.post("/verify-prediction")
async def verify_prediction(data: VerifyPrediction):
    """Verify a prediction outcome for learning"""
    ops = DailyOps()
    
    if data.prediction_index >= len(ops.state["agent_predictions"]):
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    pred = ops.state["agent_predictions"][data.prediction_index]
    pred["verified"] = True
    pred["outcome"] = {
        "direction": data.actual_direction,
        "actual_price": data.actual_price,
        "verified_at": datetime.now().isoformat()
    }
    
    ops.save_state()
    
    return {
        "status": "verified",
        "prediction": pred
    }


@router.get("/predictions")
async def get_predictions():
    """Get today's predictions"""
    ops = DailyOps()
    return {
        "date": ops.today,
        "predictions": ops.state["agent_predictions"]
    }


# ============ LEARNING ENDPOINTS ============

@router.get("/learning/weights")
async def get_agent_weights():
    """Get current agent weights based on learning"""
    engine = LearningEngine()
    return engine.load_weights()


@router.post("/learning/update")
async def update_learning():
    """Manually trigger learning update"""
    engine = LearningEngine()
    weights = engine.update_agent_weights()
    return {
        "status": "updated",
        "weights": weights
    }


@router.get("/learning/recommendations")
async def get_recommendations():
    """Get agent recommendations based on learning"""
    engine = LearningEngine()
    return engine.get_agent_recommendations()


@router.get("/learning/report")
async def get_learning_report():
    """Get full learning report"""
    engine = LearningEngine()
    return {
        "report": engine.generate_training_report()
    }


@router.get("/learning/patterns")
async def get_pattern_performance():
    """Get pattern success rates"""
    engine = LearningEngine()
    return engine.analyze_pattern_success()


# ============ DEFCON ENDPOINTS ============

@router.get("/defcon")
async def get_defcon():
    """Get current DEFCON level"""
    ops = DailyOps()
    
    defcon_descriptions = {
        1: "MAXIMUM OFFENSE - Full position sizes, aggressive entries",
        2: "STRONG OPPORTUNITY - 75% sizes, normal operations",
        3: "NEUTRAL - 50% sizes, selective entries only",
        4: "DEFENSIVE - 25% sizes, tighten stops",
        5: "CASH MODE - No new positions, preserve capital"
    }
    
    level = ops.state["defcon_level"]
    
    return {
        "level": level,
        "description": defcon_descriptions.get(level, "Unknown"),
        "position_size_modifier": {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0}[level]
    }


@router.post("/defcon/{level}")
async def set_defcon(level: int):
    """Manually set DEFCON level (1-5)"""
    if level < 1 or level > 5:
        raise HTTPException(status_code=400, detail="DEFCON level must be 1-5")
    
    ops = DailyOps()
    ops.state["defcon_level"] = level
    ops.save_state()
    
    # Send alert
    await ops.send_telegram(f"🚨 <b>DEFCON LEVEL CHANGED TO {level}</b>")
    
    return {
        "status": "updated",
        "defcon_level": level
    }

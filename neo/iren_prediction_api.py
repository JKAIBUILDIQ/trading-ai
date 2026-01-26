#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IREN 4-HOUR PREDICTION API
═══════════════════════════════════════════════════════════════════════════════

FastAPI endpoints for IREN predictions with options recommendations.

Endpoints:
- GET /api/iren/prediction          - Get current 4-hour prediction
- GET /api/iren/prediction/options  - Get options recommendations
- GET /api/iren/prediction/summary  - Get quick summary

Port: 8021

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from dataclasses import asdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from iren_predictor import IrenPredictor, IrenPrediction, get_iren_predictor
from iren_ibkr_bridge import IrenIBKRBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IrenPredictionAPI")

# Initialize
predictor = get_iren_predictor()
ibkr_bridge: Optional[IrenIBKRBridge] = None

# FastAPI app
app = FastAPI(
    title="IREN 4-Hour Prediction API",
    description="IREN price predictions with options recommendations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
current_prediction: Optional[IrenPrediction] = None
last_prediction_time: Optional[datetime] = None


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "IREN 4-Hour Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/api/iren/prediction",
            "options": "/api/iren/prediction/options",
            "summary": "/api/iren/prediction/summary",
            "refresh": "/api/iren/prediction/refresh (POST)"
        }
    }


@app.get("/api/iren/prediction")
async def get_prediction():
    """
    Get current 4-hour IREN prediction with options.
    
    Includes:
    - Price prediction (direction, confidence)
    - Technical analysis
    - BTC correlation
    - Options recommendations
    - Best option pick
    """
    global current_prediction, last_prediction_time
    
    now = datetime.now(timezone.utc)
    
    # Check if we need a new prediction (every 4 hours or first time)
    need_new = False
    if current_prediction is None:
        need_new = True
    elif last_prediction_time:
        time_since = (now - last_prediction_time).total_seconds() / 3600
        if time_since >= 4:
            need_new = True
    
    if need_new:
        current_prediction = predictor.predict_4h()
        last_prediction_time = now
    
    # Calculate time remaining
    if current_prediction:
        target_time = datetime.fromisoformat(current_prediction.target_time.replace('Z', '+00:00'))
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        
        time_remaining = int((target_time - now).total_seconds() / 60)
        current_prediction.time_remaining_minutes = max(0, time_remaining)
        current_prediction.time_remaining_display = f"{time_remaining // 60}h {time_remaining % 60}m" if time_remaining > 0 else "Expired"
    
    return asdict(current_prediction) if current_prediction else {"error": "No prediction available"}


@app.get("/api/iren/prediction/options")
async def get_options():
    """Get just the options recommendations"""
    global current_prediction
    
    if current_prediction is None:
        current_prediction = predictor.predict_4h()
    
    return {
        "current_price": current_prediction.current_price,
        "predicted_direction": current_prediction.predicted_direction,
        "signal": current_prediction.signal,
        "best_option": current_prediction.best_option,
        "all_options": current_prediction.options,
        "earnings": current_prediction.earnings,
        "timestamp": current_prediction.timestamp
    }


@app.get("/api/iren/prediction/summary")
async def get_summary():
    """Get quick summary for display"""
    global current_prediction, last_prediction_time
    
    now = datetime.now(timezone.utc)
    
    if current_prediction is None or (last_prediction_time and (now - last_prediction_time).total_seconds() / 3600 >= 4):
        current_prediction = predictor.predict_4h()
        last_prediction_time = now
    
    # Calculate time remaining
    target_time = datetime.fromisoformat(current_prediction.target_time.replace('Z', '+00:00'))
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    time_remaining = int((target_time - now).total_seconds() / 60)
    
    # Build summary
    best_opt = current_prediction.best_option
    
    return {
        "prediction_id": current_prediction.prediction_id,
        "timestamp": current_prediction.timestamp,
        "target_time": current_prediction.target_time,
        
        # Price
        "current_price": current_prediction.current_price,
        "predicted_direction": current_prediction.predicted_direction,
        "predicted_change_pct": current_prediction.predicted_change_pct,
        "predicted_price": current_prediction.predicted_price,
        "confidence": current_prediction.confidence,
        
        # Signal
        "signal": current_prediction.signal,
        "reasoning": current_prediction.reasoning,
        
        # Key technicals
        "rsi": current_prediction.technicals.get("rsi"),
        "trend": current_prediction.technicals.get("trend"),
        "macd_state": current_prediction.technicals.get("macd_state"),
        "momentum_4h": current_prediction.technicals.get("momentum_4h"),
        
        # BTC
        "btc_correlation": current_prediction.btc_analysis.get("correlation"),
        "btc_status": current_prediction.btc_analysis.get("status"),
        
        # Best option
        "best_option": {
            "strike": best_opt.get("strike") if best_opt else None,
            "expiration": best_opt.get("expiration") if best_opt else None,
            "dte": best_opt.get("dte") if best_opt else None,
            "price": best_opt.get("last_price") if best_opt else None,
            "is_pauls_pick": best_opt.get("is_pauls_pick") if best_opt else False
        } if best_opt else None,
        
        # Earnings
        "earnings_date": current_prediction.earnings.get("date"),
        "earnings_days_away": current_prediction.earnings.get("days_away"),
        
        # Warnings
        "warnings": current_prediction.warnings,
        
        # Status
        "status": current_prediction.status,
        "time_remaining_minutes": max(0, time_remaining),
        "time_remaining_display": f"{time_remaining // 60}h {time_remaining % 60}m" if time_remaining > 0 else "Expired"
    }


@app.post("/api/iren/prediction/refresh")
async def refresh_prediction():
    """Force a new prediction"""
    global current_prediction, last_prediction_time
    
    current_prediction = predictor.predict_4h()
    last_prediction_time = datetime.now(timezone.utc)
    
    return {
        "message": "Prediction refreshed",
        "prediction_id": current_prediction.prediction_id,
        "predicted_direction": current_prediction.predicted_direction,
        "confidence": current_prediction.confidence,
        "signal": current_prediction.signal
    }


@app.on_event("startup")
async def startup():
    """Generate initial prediction on startup"""
    global current_prediction, last_prediction_time
    current_prediction = predictor.predict_4h()
    last_prediction_time = datetime.now(timezone.utc)
    logger.info(f"Initial prediction: {current_prediction.predicted_direction} ({current_prediction.confidence}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR BRIDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/iren/ibkr/status")
async def get_ibkr_status():
    """Get IBKR bridge status"""
    global ibkr_bridge, current_prediction
    
    if ibkr_bridge is None:
        ibkr_bridge = IrenIBKRBridge(paper_trading=True, auto_execute=False)
    
    status = ibkr_bridge.get_status()
    # Inject current prediction from local predictor
    if current_prediction:
        status['current_prediction'] = {
            'prediction_id': current_prediction.prediction_id,
            'signal': current_prediction.signal,
            'confidence': current_prediction.confidence,
            'direction': current_prediction.predicted_direction,
            'best_option': current_prediction.best_option
        }
        # Check if signal is valid
        if current_prediction.signal == 'BUY_CALLS' and current_prediction.confidence >= 60:
            status['pending_signals'] = 1
    
    return status


@app.post("/api/iren/ibkr/connect")
async def connect_ibkr():
    """Connect to IBKR"""
    global ibkr_bridge
    
    if ibkr_bridge is None:
        ibkr_bridge = IrenIBKRBridge(paper_trading=True, auto_execute=False)
    
    success = ibkr_bridge.connect_ibkr()
    
    return {
        "success": success,
        "message": "Connected to IBKR" if success else "Failed to connect",
        "paper_trading": ibkr_bridge.paper_trading
    }


@app.get("/api/iren/ibkr/signal")
async def get_ibkr_signal():
    """Get current signal formatted for IBKR execution"""
    global ibkr_bridge, current_prediction
    
    if ibkr_bridge is None:
        ibkr_bridge = IrenIBKRBridge(paper_trading=True, auto_execute=False)
    
    if current_prediction is None:
        return {"success": False, "error": "No prediction available"}
    
    # Validate signal
    pred_dict = {
        'signal': current_prediction.signal,
        'confidence': current_prediction.confidence,
        'best_option': current_prediction.best_option,
        'earnings_days_away': current_prediction.earnings.get('days_away', 100),
        'prediction_id': current_prediction.prediction_id
    }
    
    valid, reason = ibkr_bridge.validate_signal(pred_dict)
    
    if not valid:
        return {
            "success": False,
            "valid": False,
            "reason": reason,
            "prediction": {
                "signal": current_prediction.signal,
                "confidence": current_prediction.confidence,
                "direction": current_prediction.predicted_direction,
                "best_option": current_prediction.best_option
            }
        }
    
    # Format order
    order = ibkr_bridge.format_order(pred_dict)
    
    return {
        "success": True,
        "valid": True,
        "order": order,
        "executed": False,
        "ibkr_connected": ibkr_bridge.connected,
        "prediction": {
            "signal": current_prediction.signal,
            "confidence": current_prediction.confidence,
            "direction": current_prediction.predicted_direction,
            "price": current_prediction.current_price,
            "best_option": current_prediction.best_option
        }
    }


@app.post("/api/iren/ibkr/execute")
async def execute_ibkr_order(quantity: int = 5):
    """Execute current signal on IBKR (USE WITH CAUTION!)"""
    global ibkr_bridge, current_prediction
    
    if ibkr_bridge is None:
        ibkr_bridge = IrenIBKRBridge(paper_trading=True, auto_execute=False)
    
    if current_prediction is None:
        return {"success": False, "error": "No prediction available"}
    
    # Validate signal first
    pred_dict = {
        'signal': current_prediction.signal,
        'confidence': current_prediction.confidence,
        'best_option': current_prediction.best_option,
        'earnings_days_away': current_prediction.earnings.get('days_away', 100),
        'prediction_id': current_prediction.prediction_id
    }
    
    valid, reason = ibkr_bridge.validate_signal(pred_dict)
    
    if not valid:
        return {
            "success": False,
            "valid": False,
            "reason": reason,
            "executed": False
        }
    
    # Connect if not connected
    if not ibkr_bridge.connected:
        connected = ibkr_bridge.connect_ibkr()
        if not connected:
            return {
                "success": False,
                "error": "Could not connect to IBKR. Make sure TWS is running with API enabled.",
                "executed": False
            }
    
    # Format and execute order
    order = ibkr_bridge.format_order(pred_dict, quantity)
    result = ibkr_bridge.execute_order(order)
    
    return {
        "success": result.get('success', False),
        "valid": True,
        "order": order,
        "executed": True,
        "result": result
    }


@app.get("/api/iren/ibkr/history")
async def get_ibkr_history():
    """Get signal execution history"""
    global ibkr_bridge
    
    if ibkr_bridge is None:
        ibkr_bridge = IrenIBKRBridge(paper_trading=True, auto_execute=False)
    
    return {
        "total_signals": len(ibkr_bridge.signal_history),
        "recent": ibkr_bridge.signal_history[-10:] if ibkr_bridge.signal_history else []
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021, log_level="info")

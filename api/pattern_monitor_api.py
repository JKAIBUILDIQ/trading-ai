"""
Pattern Monitor API - Endpoints for pattern alerts.

Provides:
- GET /alerts/recent - Get recent alerts
- GET /alerts/check/{symbol} - Trigger immediate check
- POST /alerts/acknowledge - Acknowledge an alert
- GET /alerts/defcon-recommendations - Get pending DEFCON recommendations
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.pattern_detector import PatternDetector
from api.alert_system import AlertSystem

import yfinance as yf
from motor.motor_asyncio import AsyncIOMotorClient

router = APIRouter(prefix="/alerts", tags=["alerts"])

# Initialize components
detector = PatternDetector()
alert_system = AlertSystem()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.trading_ai


class AcknowledgeRequest(BaseModel):
    alert_id: str


@router.get("/recent")
async def get_recent_alerts(symbol: Optional[str] = None, limit: int = 20):
    """Get recent pattern alerts."""
    try:
        query = {"symbol": symbol} if symbol else {}
        cursor = db.pattern_alerts.find(query).sort("created_at", -1).limit(limit)
        alerts = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
            if "created_at" in alert:
                alert["created_at"] = alert["created_at"].isoformat()
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "symbol_filter": symbol,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/{symbol}")
async def check_symbol_now(symbol: str):
    """Trigger immediate pattern check for a symbol."""
    try:
        # Fetch candles
        yf_symbol = symbol if symbol != "XAUUSD" else "GC=F"
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="5d", interval="1h")
        
        if hist.empty:
            return {"symbol": symbol, "patterns": [], "message": "No data available"}
        
        candles = []
        for idx, row in hist.iterrows():
            candles.append({
                "time": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        
        # Detect patterns
        patterns = detector.detect_all_patterns(candles, symbol)
        
        # Send alerts for detected patterns
        alerts_sent = []
        for pattern in patterns:
            sent = await alert_system.send_alert(pattern, symbol)
            if sent:
                alerts_sent.append(pattern["pattern"])
        
        return {
            "symbol": symbol,
            "patterns": patterns,
            "alerts_sent": alerts_sent,
            "candle_count": len(candles),
            "latest_price": candles[-1]["close"] if candles else None,
            "checked_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acknowledge")
async def acknowledge_alert(request: AcknowledgeRequest):
    """Acknowledge an alert."""
    try:
        from bson import ObjectId
        
        result = await db.pattern_alerts.update_one(
            {"_id": ObjectId(request.alert_id)},
            {"$set": {"acknowledged": True, "acknowledged_at": datetime.now()}}
        )
        
        if result.modified_count > 0:
            return {"status": "acknowledged", "alert_id": request.alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/defcon-recommendations")
async def get_defcon_recommendations(pending_only: bool = True):
    """Get pending DEFCON change recommendations."""
    try:
        query = {"applied": False} if pending_only else {}
        cursor = db.defcon_recommendations.find(query).sort("created_at", -1).limit(10)
        recommendations = await cursor.to_list(length=10)
        
        for rec in recommendations:
            rec["_id"] = str(rec["_id"])
            if "created_at" in rec:
                rec["created_at"] = rec["created_at"].isoformat()
        
        return {
            "recommendations": recommendations,
            "pending_only": pending_only,
            "count": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/defcon-recommendations/{rec_id}/apply")
async def apply_defcon_recommendation(rec_id: str):
    """Apply a DEFCON recommendation."""
    try:
        from bson import ObjectId
        
        # Get recommendation
        rec = await db.defcon_recommendations.find_one({"_id": ObjectId(rec_id)})
        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        new_defcon = rec["recommended_defcon"]
        
        # Import and update DEFCON
        from defcon_playbooks import ACTIVE_STATE, PLAYBOOKS, save_active_state
        
        ACTIVE_STATE["defcon"] = new_defcon
        ACTIVE_STATE["playbook"] = PLAYBOOKS.get(new_defcon, PLAYBOOKS[3])
        ACTIVE_STATE["scenario"] = f"Pattern: {rec.get('trigger_pattern', 'Unknown')}"
        ACTIVE_STATE["updated_at"] = datetime.now().isoformat()
        
        save_active_state()
        
        # Mark as applied
        await db.defcon_recommendations.update_one(
            {"_id": ObjectId(rec_id)},
            {"$set": {"applied": True, "applied_at": datetime.now()}}
        )
        
        return {
            "status": "applied",
            "new_defcon": new_defcon,
            "trigger": rec.get("trigger_pattern"),
            "rec_id": rec_id,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_monitor_status():
    """Get pattern monitor status."""
    try:
        # Get latest alert
        latest = await db.pattern_alerts.find_one(sort=[("created_at", -1)])
        
        # Get pending recommendations
        pending = await db.defcon_recommendations.count_documents({"applied": False})
        
        # Get alert counts
        total_24h = await db.pattern_alerts.count_documents({
            "created_at": {"$gte": datetime.now() - __import__("datetime").timedelta(hours=24)}
        })
        
        return {
            "status": "running",
            "latest_alert": {
                "pattern": latest.get("pattern") if latest else None,
                "symbol": latest.get("symbol") if latest else None,
                "time": latest.get("created_at").isoformat() if latest and latest.get("created_at") else None,
            } if latest else None,
            "alerts_24h": total_24h,
            "pending_recommendations": pending,
            "checked_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/signal-file")
async def get_signal_file():
    """Get contents of the pattern alerts signal file."""
    import json
    
    signal_file = "/home/jbot/trading_ai/neo/signals/pattern_alerts.json"
    
    try:
        with open(signal_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"alerts": [], "last_updated": None, "message": "No alerts yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

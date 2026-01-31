"""
Bot Signals API - Endpoints for Ghost/Casper to receive pattern alerts.

Ghost polls this endpoint every 60 seconds to check for new alerts
and receive specific trading instructions.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import APIRouter, Form
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BotSignals")

router = APIRouter(prefix="/signals", tags=["signals"])

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.trading_ai

# Signal file paths
SIGNAL_DIR = "/home/jbot/trading_ai/neo/signals"
BOT_ALERTS_FILE = os.path.join(SIGNAL_DIR, "bot_alerts.json")
os.makedirs(SIGNAL_DIR, exist_ok=True)


def get_ghost_orders(alert: Dict) -> Dict:
    """Convert alert to specific Ghost trading orders."""
    
    severity = alert.get("severity", "MEDIUM")
    direction = alert.get("direction", "BEARISH")
    pattern = alert.get("pattern", "UNKNOWN")
    
    if direction == "BEARISH":
        if severity == "CRITICAL":
            return {
                "action": "CLOSE_PARTIAL",
                "close_percent": 75,
                "tighten_stops": True,
                "new_entries": False,
                "lot_multiplier": 0.25,
                "message": f"CRITICAL: {pattern} - Close 75%, no new entries",
                "priority": "URGENT",
            }
        elif severity == "HIGH":
            return {
                "action": "REDUCE_AND_PROTECT",
                "close_percent": 50,
                "tighten_stops": True,
                "new_entries": False,
                "lot_multiplier": 0.5,
                "message": f"HIGH: {pattern} - Close 50%, tighten stops",
                "priority": "HIGH",
            }
        else:  # MEDIUM, LOW
            return {
                "action": "CAUTION",
                "close_percent": 0,
                "tighten_stops": True,
                "new_entries": True,
                "lot_multiplier": 0.75,
                "message": f"CAUTION: {pattern} - Tighten stops, reduce size",
                "priority": "NORMAL",
            }
    
    else:  # BULLISH
        if severity in ["CRITICAL", "HIGH"]:
            return {
                "action": "OPPORTUNITY",
                "close_percent": 0,
                "tighten_stops": False,
                "new_entries": True,
                "lot_multiplier": 1.0,
                "message": f"BULLISH: {pattern} - Entry opportunity",
                "priority": "NORMAL",
                "entry_signal": True,
            }
        else:
            return {
                "action": "WATCH",
                "close_percent": 0,
                "tighten_stops": False,
                "new_entries": True,
                "lot_multiplier": 1.0,
                "message": f"BULLISH: {pattern} - Watch for confirmation",
                "priority": "LOW",
            }


def save_bot_signal(signal: Dict):
    """Save current signal to file for bots without API access."""
    try:
        with open(BOT_ALERTS_FILE, "w") as f:
            json.dump(signal, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save bot signal file: {e}")


@router.get("/bot-alerts")
async def get_bot_alerts():
    """
    Endpoint for Ghost/Casper to poll.
    Returns current active alert and trading instructions.
    
    Ghost should poll this every 60 seconds.
    """
    try:
        # Get most recent unacknowledged alert from last hour
        alert = await db.pattern_alerts.find_one(
            {
                "acknowledged": {"$ne": True},
                "created_at": {"$gte": datetime.now() - timedelta(hours=1)}
            },
            sort=[("created_at", -1)]
        )
        
        if alert:
            # Convert alert to ghost orders
            ghost_orders = get_ghost_orders(alert)
            
            # Get recommended DEFCON
            defcon_rec = await db.defcon_recommendations.find_one(
                {"applied": False},
                sort=[("created_at", -1)]
            )
            
            recommended_defcon = defcon_rec.get("recommended_defcon", 3) if defcon_rec else None
            
            response = {
                "has_alert": True,
                "alert_id": str(alert["_id"]),
                "pattern": alert.get("pattern"),
                "severity": alert.get("severity"),
                "direction": alert.get("direction"),
                "defcon_recommended": recommended_defcon,
                "price_at_alert": alert.get("price"),
                "timestamp": alert.get("created_at").isoformat() if alert.get("created_at") else None,
                "message": alert.get("message"),
                "ghost_orders": ghost_orders,
            }
            
            # Save to file for bots without API access
            save_bot_signal(response)
            
            return response
        
        # No active alert - return current DEFCON status
        active_defcon = await db.active_defcon.find_one({"_id": "current"})
        
        response = {
            "has_alert": False,
            "defcon": active_defcon.get("defcon", 5) if active_defcon else 5,
            "playbook": active_defcon.get("playbook", {}).get("name", "LONG_AND_STRONG") if active_defcon else "LONG_AND_STRONG",
            "ghost_orders": {
                "action": "NORMAL",
                "close_percent": 0,
                "tighten_stops": False,
                "new_entries": True,
                "lot_multiplier": 1.0,
                "message": "No active alerts - normal operation",
                "priority": "NONE",
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save to file
        save_bot_signal(response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting bot alerts: {e}")
        return {
            "has_alert": False,
            "error": str(e),
            "ghost_orders": {
                "action": "NORMAL",
                "message": "Error fetching alerts - continue normal operation",
            }
        }


@router.post("/acknowledge-alert")
async def acknowledge_alert(alert_id: str = Form(...)):
    """
    Ghost acknowledges it received and acted on alert.
    This prevents the same alert from triggering multiple times.
    """
    try:
        from bson import ObjectId
        
        result = await db.pattern_alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "acknowledged": True,
                    "acknowledged_at": datetime.now(),
                    "acknowledged_by": "ghost_bot",
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Alert {alert_id} acknowledged by Ghost")
            return {"status": "acknowledged", "alert_id": alert_id}
        else:
            return {"status": "not_found", "alert_id": alert_id}
            
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/report-action")
async def report_action_taken(
    alert_id: str = Form(...),
    action_taken: str = Form(...),
    positions_closed: int = Form(0),
    stops_tightened: int = Form(0),
    notes: str = Form(""),
):
    """
    Ghost reports what action it took on an alert.
    This helps track the effectiveness of alerts.
    """
    try:
        from bson import ObjectId
        
        await db.pattern_alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "action_taken": action_taken,
                    "positions_closed": positions_closed,
                    "stops_tightened": stops_tightened,
                    "action_notes": notes,
                    "action_timestamp": datetime.now(),
                }
            }
        )
        
        # Also log to action history
        await db.bot_actions.insert_one({
            "alert_id": alert_id,
            "bot": "ghost",
            "action": action_taken,
            "positions_closed": positions_closed,
            "stops_tightened": stops_tightened,
            "notes": notes,
            "timestamp": datetime.now(),
        })
        
        logger.info(f"Ghost action recorded: {action_taken} on alert {alert_id}")
        
        return {
            "status": "recorded",
            "alert_id": alert_id,
            "action": action_taken,
        }
        
    except Exception as e:
        logger.error(f"Error recording action: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/defcon-status")
async def get_defcon_status():
    """Get current DEFCON status for bots."""
    try:
        active = await db.active_defcon.find_one({"_id": "current"})
        
        if active:
            playbook = active.get("playbook", {})
            return {
                "defcon": active.get("defcon", 5),
                "playbook_name": playbook.get("name", "LONG_AND_STRONG"),
                "position_size_pct": playbook.get("position_size_pct", 100),
                "new_entries_allowed": playbook.get("new_entries_allowed", True),
                "ghost_status": playbook.get("ghost_status", "NORMAL"),
                "spy_status": playbook.get("spy_status", "STANDBY"),
                "updated_at": active.get("updated_at"),
            }
        
        return {
            "defcon": 5,
            "playbook_name": "LONG_AND_STRONG",
            "position_size_pct": 100,
            "new_entries_allowed": True,
        }
        
    except Exception as e:
        return {"error": str(e), "defcon": 5}


@router.get("/action-history")
async def get_action_history(limit: int = 20):
    """Get history of bot actions."""
    try:
        cursor = db.bot_actions.find().sort("timestamp", -1).limit(limit)
        actions = await cursor.to_list(length=limit)
        
        for action in actions:
            action["_id"] = str(action["_id"])
            if "timestamp" in action:
                action["timestamp"] = action["timestamp"].isoformat()
        
        return {"actions": actions, "count": len(actions)}
        
    except Exception as e:
        return {"error": str(e), "actions": []}

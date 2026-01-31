"""
Scout Swarm API Routes
======================
Control and monitor the Sector Scout Swarm.
"""

import asyncio
import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScoutAPI")

router = APIRouter(prefix="/scouts", tags=["scouts"])

# Global coordinator reference
_coordinator = None
_coordinator_task = None


async def _run_swarm_background():
    """Background task to run the swarm."""
    global _coordinator
    
    import sys
    sys.path.insert(0, '/home/jbot/trading_ai')
    
    from services.sector_scouts.swarm_coordinator import ScoutSwarmCoordinator, set_coordinator
    
    _coordinator = ScoutSwarmCoordinator()
    set_coordinator(_coordinator)
    
    await _coordinator.start_swarm()


@router.post("/start")
async def start_scouts(background_tasks: BackgroundTasks):
    """
    Start the Scout Swarm.
    5 scouts will begin scanning their sectors during market hours.
    """
    global _coordinator
    
    if _coordinator and _coordinator.running:
        return {
            "status": "already_running",
            "stats": _coordinator.get_stats(),
        }
    
    # Start swarm in background
    background_tasks.add_task(_run_swarm_background)
    
    return {
        "status": "starting",
        "message": "Scout Swarm activated. 5 scouts scanning sectors.",
        "scouts": [
            {"name": "TECH_TITAN", "sector": "Technology", "symbols": 16},
            {"name": "ENERGY_EAGLE", "sector": "Energy", "symbols": 15},
            {"name": "MINER_HAWK", "sector": "Crypto Miners", "symbols": 11, "mode": "LONG_ONLY"},
            {"name": "GROWTH_HUNTER", "sector": "Growth Stocks", "symbols": 15},
            {"name": "DEFENSE_FORTRESS", "sector": "Defense", "symbols": 11},
        ],
    }


@router.post("/stop")
async def stop_scouts():
    """Stop the Scout Swarm."""
    global _coordinator
    
    if _coordinator:
        final_stats = _coordinator.get_stats()
        _coordinator.stop_swarm()
        _coordinator = None
        return {
            "status": "stopped",
            "final_stats": final_stats,
        }
    
    return {"status": "not_running"}


@router.get("/status")
async def scout_status():
    """Get swarm status and statistics."""
    global _coordinator
    
    if _coordinator:
        return {
            "status": "running" if _coordinator.running else "stopped",
            "stats": _coordinator.get_stats(),
        }
    
    return {
        "status": "not_initialized",
        "message": "Use POST /scouts/start to activate the swarm",
    }


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    """Get recent alerts from all scouts."""
    global _coordinator
    
    alerts = []
    
    # Try to get from coordinator memory
    if _coordinator:
        alerts = _coordinator.get_recent_alerts(limit)
    
    # Also fetch from MongoDB
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client.trading_ai
        
        cursor = db.scout_alerts.find().sort("timestamp", -1).limit(limit)
        db_alerts = await cursor.to_list(length=limit)
        
        for alert in db_alerts:
            alert["_id"] = str(alert["_id"])
        
        # Merge and dedupe by timestamp+symbol
        seen = {(a.get('timestamp'), a.get('symbol')) for a in alerts}
        for a in db_alerts:
            key = (a.get('timestamp'), a.get('symbol'))
            if key not in seen:
                alerts.append(a)
                seen.add(key)
        
        # Sort by timestamp
        alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        alerts = alerts[:limit]
        
    except Exception as e:
        logger.debug(f"MongoDB fetch error: {e}")
    
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


@router.post("/alert")
async def receive_alert(alert: dict):
    """
    Receive alert from a scout (internal endpoint).
    Stores in MongoDB and could push to Telegram/War Room.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client.trading_ai
        
        # Add received timestamp
        alert['received_at'] = datetime.now().isoformat()
        
        await db.scout_alerts.insert_one(alert)
        
        logger.info(f"Scout alert saved: {alert.get('symbol')} - {alert.get('direction')} ({alert.get('confidence')}%)")
        
        # Push to War Room WebSocket (if available)
        try:
            import httpx
            async with httpx.AsyncClient() as http:
                await http.post(
                    "http://localhost:8889/war-room/alert",
                    json={
                        "type": "SCOUT_ALERT",
                        "data": alert,
                    },
                    timeout=5,
                )
        except Exception:
            pass
        
        return {"status": "received", "symbol": alert.get('symbol')}
        
    except Exception as e:
        logger.error(f"Error saving alert: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/sectors")
async def list_sectors():
    """List all sectors and their watchlists."""
    
    return {
        "total_symbols": 68,
        "sectors": [
            {
                "scout": "TECH_TITAN",
                "sector": "Technology",
                "count": 16,
                "symbols": ["AAPL", "NVDA", "MSFT", "META", "GOOGL", "AMZN", "AMD", "INTC", "AVGO", "TXN", "QCOM", "MU", "CRM", "ORCL", "ADBE", "NOW"],
            },
            {
                "scout": "ENERGY_EAGLE",
                "sector": "Energy",
                "count": 15,
                "symbols": ["XOM", "CVX", "COP", "OXY", "EOG", "PXD", "SLB", "HAL", "BKR", "MPC", "VLO", "PSX", "KMI", "WMB", "ET"],
            },
            {
                "scout": "MINER_HAWK",
                "sector": "Crypto Miners",
                "count": 11,
                "mode": "LONG_ONLY",
                "note": "Your protected longs - only LONG signals",
                "symbols": ["IREN", "CLSK", "CIFR", "MARA", "RIOT", "HUT", "BITF", "BTBT", "CORZ", "WULF", "HIVE"],
            },
            {
                "scout": "GROWTH_HUNTER",
                "sector": "Growth Stocks",
                "count": 15,
                "symbols": ["COIN", "SQ", "PYPL", "HOOD", "SNOW", "DDOG", "NET", "ZS", "CRWD", "PLTR", "MDB", "TEAM", "SHOP", "ETSY", "ARKK"],
            },
            {
                "scout": "DEFENSE_FORTRESS",
                "sector": "Defense & Aerospace",
                "count": 11,
                "symbols": ["LMT", "RTX", "GD", "NOC", "BA", "HII", "LHX", "TDG", "HWM", "TXT", "SPR"],
            },
        ],
    }


@router.get("/scout/{scout_name}")
async def get_scout_details(scout_name: str):
    """Get details for a specific scout."""
    global _coordinator
    
    scout_name = scout_name.upper()
    
    if _coordinator:
        scout = _coordinator.get_scout_by_name(scout_name)
        if scout:
            return {
                "found": True,
                "scout": scout.get_stats(),
                "recent_alerts": list(scout.alerts)[-10:],
            }
    
    # Return static info if swarm not running
    sectors = {
        "TECH_TITAN": {"sector": "Technology", "symbols": 16},
        "ENERGY_EAGLE": {"sector": "Energy", "symbols": 15},
        "MINER_HAWK": {"sector": "Crypto Miners", "symbols": 11, "mode": "LONG_ONLY"},
        "GROWTH_HUNTER": {"sector": "Growth Stocks", "symbols": 15},
        "DEFENSE_FORTRESS": {"sector": "Defense & Aerospace", "symbols": 11},
    }
    
    if scout_name in sectors:
        return {
            "found": True,
            "scout": {
                "name": scout_name,
                "running": False,
                **sectors[scout_name],
            },
            "message": "Swarm not running - start with POST /scouts/start",
        }
    
    return {"found": False, "message": f"Scout '{scout_name}' not found"}


@router.get("/health")
async def scout_health():
    """Health check for scout system."""
    global _coordinator
    
    # Check Ollama availability
    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=5)
            ollama_ok = resp.status_code == 200
    except Exception:
        pass
    
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama_available": ollama_ok,
        "swarm_running": _coordinator is not None and _coordinator.running,
        "timestamp": datetime.now().isoformat(),
    }

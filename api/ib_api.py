"""
IB Auto-Execution API Routes
============================
Control and monitor IB trade execution.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException

# Add path for imports
sys.path.insert(0, '/home/jbot/trading_ai')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IB_API")

router = APIRouter(prefix="/ib", tags=["interactive-brokers"])

# Import service (will be initialized on first use)
_ib_service = None


def get_service():
    """Get or create the IB service instance."""
    global _ib_service
    if _ib_service is None:
        from services.ib_executor.service import IBExecutionService
        _ib_service = IBExecutionService()
    return _ib_service


@router.post("/connect")
async def connect_ib(paper: bool = True):
    """
    Connect to Interactive Brokers TWS/Gateway.
    
    - paper=True: Connect to paper trading (port 7497)
    - paper=False: Connect to live trading (port 7496)
    
    Requirements:
    - TWS or IB Gateway must be running
    - API connections must be enabled in TWS settings
    """
    
    service = get_service()
    
    try:
        success = await service.start(paper=paper)
        
        if success:
            return {
                "status": "connected",
                "mode": "PAPER" if paper else "LIVE",
                "message": f"Connected to IB {'Paper' if paper else 'Live'} Trading",
                "account_info": service.connection.account_info,
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to connect to IB. Is TWS/Gateway running with API enabled?"
            )
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
async def disconnect_ib():
    """Disconnect from Interactive Brokers."""
    
    service = get_service()
    service.stop()
    return {"status": "disconnected"}


@router.get("/status")
async def ib_status():
    """Get IB connection status and execution stats."""
    
    service = get_service()
    return service.get_status()


@router.post("/execute/scout-alert")
async def execute_scout_alert(alert: dict):
    """
    Execute a trade based on scout alert.
    
    Called automatically by Scout Swarm when confidence >= 75%.
    
    Expected alert format:
    {
        "symbol": "IREN",
        "confidence": 82,
        "direction": "LONG",
        "entry": 8.50,
        "stop": 8.00,
        "target1": 9.50,
        "target2": 10.50
    }
    """
    
    service = get_service()
    
    if not service.running:
        return {
            "status": "not_connected",
            "message": "IB not connected. Use POST /ib/connect first.",
        }
    
    result = service.process_scout_alert(alert)
    
    # Log to MongoDB
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client.trading_ai
        
        await db.ib_executions.insert_one({
            'type': 'SCOUT_ALERT',
            'alert': alert,
            'result': result,
            'timestamp': datetime.now(),
        })
    except Exception as e:
        logger.debug(f"MongoDB log error: {e}")
    
    return result


@router.post("/execute/sentinel-alert")
async def execute_sentinel_alert(alert: dict):
    """
    Execute based on Sentinel pattern alert.
    
    Called automatically by Live Sentinel on HIGH/CRITICAL patterns.
    
    Expected alert format:
    {
        "symbol": "GC=F",
        "patterns": ["Shooting Star"],
        "ghost_action": {
            "action": "CLOSE_PARTIAL",
            "close_percent": 50
        }
    }
    """
    
    service = get_service()
    
    if not service.running:
        return {
            "status": "not_connected",
            "message": "IB not connected. Use POST /ib/connect first.",
        }
    
    result = service.process_sentinel_alert(alert)
    
    # Log to MongoDB
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client.trading_ai
        
        await db.ib_executions.insert_one({
            'type': 'SENTINEL_ALERT',
            'alert': alert,
            'result': result,
            'timestamp': datetime.now(),
        })
    except Exception as e:
        logger.debug(f"MongoDB log error: {e}")
    
    return result


@router.get("/orders/pending")
async def get_pending_orders():
    """Get orders pending confirmation (live mode only)."""
    
    service = get_service()
    
    if service.executor:
        return {"orders": service.executor.get_pending_orders()}
    return {"orders": []}


@router.get("/orders/executed")
async def get_executed_orders(limit: int = 50):
    """Get recently executed orders."""
    
    service = get_service()
    
    if service.executor:
        return {"orders": service.executor.get_executed_orders(limit)}
    return {"orders": []}


@router.get("/orders/failed")
async def get_failed_orders():
    """Get failed orders."""
    
    service = get_service()
    
    if service.executor:
        return {"orders": service.executor.get_failed_orders()}
    return {"orders": []}


@router.post("/orders/confirm/{index}")
async def confirm_order(index: int):
    """Confirm a pending order (live mode only)."""
    
    service = get_service()
    
    if service.executor:
        result = service.executor.confirm_order(index)
        return result
    raise HTTPException(status_code=400, detail="Executor not running")


@router.post("/orders/cancel/{index}")
async def cancel_pending_order(index: int):
    """Cancel a pending order."""
    
    service = get_service()
    
    if service.executor:
        result = service.executor.cancel_pending_order(index)
        return result
    raise HTTPException(status_code=400, detail="Executor not running")


@router.get("/positions")
async def get_positions():
    """Get current IB positions."""
    
    service = get_service()
    
    if service.connection.connected:
        positions = service.get_positions()
        return {
            "count": len(positions),
            "positions": positions,
        }
    return {"count": 0, "positions": [], "message": "Not connected"}


@router.get("/account")
async def get_account():
    """Get IB account summary."""
    
    service = get_service()
    
    if service.connection.connected:
        summary = service.get_account()
        return {
            "mode": "PAPER" if service.connection.is_paper else "LIVE",
            "account": summary,
        }
    return {"account": {}, "message": "Not connected"}


@router.post("/settings/auto-execute")
async def set_auto_execute(enabled: bool):
    """
    Enable/disable auto-execution in live mode.
    
    WARNING: Enabling this in LIVE mode will execute trades automatically!
    """
    
    service = get_service()
    
    if service.executor:
        service.executor.live_auto_execute = enabled
        return {
            "live_auto_execute": enabled,
            "mode": "PAPER" if service.connection.is_paper else "LIVE",
            "message": f"Live auto-execute {'ENABLED - trades will execute automatically!' if enabled else 'disabled'}",
        }
    
    return {"error": "Executor not running"}


@router.post("/settings/risk")
async def set_risk_settings(
    risk_per_trade: float = 0.01,
    max_position_size: int = 500,
    max_position_value: int = 10000,
):
    """Update risk management settings."""
    
    service = get_service()
    
    if service.executor:
        service.executor.risk_per_trade = risk_per_trade
        service.executor.max_position_size = max_position_size
        service.executor.max_position_value = max_position_value
        
        return {
            "risk_per_trade": risk_per_trade,
            "max_position_size": max_position_size,
            "max_position_value": max_position_value,
            "message": "Risk settings updated",
        }
    
    return {"error": "Executor not running"}


@router.get("/health")
async def ib_health():
    """Health check for IB connection."""
    
    service = get_service()
    
    return {
        "status": "healthy" if service.running else "disconnected",
        "connected": service.connection.connected,
        "mode": "PAPER" if service.connection.is_paper else "LIVE" if service.running else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

"""
Live Sentinel API Routes
========================
Control and monitor the Live Sentinel Agent.
"""

import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
import logging
import yfinance as yf
import pandas as pd

from talib_patterns import TALibPatternDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelAPI")

router = APIRouter(prefix="/sentinel", tags=["sentinel"])

# Global sentinel reference
_sentinel = None
_sentinel_task = None

# Pattern detector for on-demand scans
detector = TALibPatternDetector()


async def _run_sentinel_background(symbols: List[str], check_interval: int, auto_execute: bool):
    """Background task to run sentinel."""
    global _sentinel
    
    from services.live_sentinel import LiveSentinel, set_sentinel
    
    _sentinel = LiveSentinel(
        symbols=symbols,
        check_interval=check_interval,
        auto_execute=auto_execute,
    )
    set_sentinel(_sentinel)
    
    await _sentinel.start()


@router.post("/start")
async def start_sentinel(
    background_tasks: BackgroundTasks,
    symbols: List[str] = ["GC=F"],
    check_interval: int = 60,
    auto_execute: bool = False,
):
    """
    Start the Live Sentinel Agent.
    
    - symbols: List of yfinance symbols to watch (e.g., ["GC=F", "SI=F"])
    - check_interval: Seconds between pattern checks (default: 60)
    - auto_execute: Auto-execute Ghost actions (default: False)
    """
    global _sentinel, _sentinel_task
    
    if _sentinel and _sentinel.watching:
        return {
            "status": "already_running",
            "stats": _sentinel.get_stats(),
        }
    
    # Start sentinel in background
    background_tasks.add_task(
        _run_sentinel_background, 
        symbols, 
        check_interval, 
        auto_execute
    )
    
    return {
        "status": "starting",
        "symbols": symbols,
        "check_interval": check_interval,
        "auto_execute": auto_execute,
        "message": "Sentinel is starting... Check /sentinel/status in a few seconds.",
    }


@router.post("/stop")
async def stop_sentinel():
    """Stop the Live Sentinel Agent."""
    global _sentinel
    
    if _sentinel:
        _sentinel.stop()
        stats = _sentinel.get_stats()
        _sentinel = None
        return {
            "status": "stopped",
            "final_stats": stats,
        }
    
    return {"status": "not_running"}


@router.get("/status")
async def sentinel_status():
    """Get current sentinel status and statistics."""
    global _sentinel
    
    if _sentinel:
        return {
            "status": "running" if _sentinel.watching else "stopped",
            "stats": _sentinel.get_stats(),
        }
    
    return {
        "status": "not_initialized",
        "message": "Use POST /sentinel/start to begin monitoring",
    }


@router.get("/scan/{symbol}")
async def scan_symbol(
    symbol: str = "GC=F",
    priority_only: bool = False,
    period: str = "5d",
    interval: str = "1h",
):
    """
    Manually scan a symbol for patterns (one-time).
    
    - symbol: yfinance symbol (e.g., GC=F, XAUUSD=X, IREN)
    - priority_only: Only check priority patterns (faster)
    - period: Data period (1d, 5d, 1mo, etc.)
    - interval: Candle interval (1h, 4h, 1d, etc.)
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        df.columns = df.columns.str.lower()
        
        patterns = detector.detect_patterns(df, priority_only=priority_only)
        ghost_action = detector.get_ghost_action(patterns)
        defcon_impact = detector.get_defcon_impact(patterns)
        
        return {
            "symbol": symbol,
            "current_price": round(float(df['close'].iloc[-1]), 2),
            "period": period,
            "interval": interval,
            "candles_analyzed": len(df),
            "patterns_detected": len(patterns),
            "patterns": patterns,
            "ghost_action": ghost_action,
            "defcon_impact": defcon_impact,
            "summary": detector.get_pattern_summary(patterns),
            "scanned_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Scan error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan-multi")
async def scan_multiple_symbols(
    symbols: str = "GC=F,SI=F,IREN",
    priority_only: bool = True,
):
    """
    Scan multiple symbols at once.
    
    - symbols: Comma-separated list of symbols
    """
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    results = []
    for symbol in symbol_list:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1h")
            
            if df.empty:
                results.append({"symbol": symbol, "error": "No data"})
                continue
            
            df.columns = df.columns.str.lower()
            patterns = detector.detect_patterns(df, priority_only=priority_only)
            
            results.append({
                "symbol": symbol,
                "price": round(float(df['close'].iloc[-1]), 2),
                "patterns_count": len(patterns),
                "patterns": [p['pattern'] for p in patterns],
                "directions": list(set(p['direction'] for p in patterns)),
                "max_severity": max([p['severity'] for p in patterns], default='NONE'),
            })
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    
    # Sort by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'NONE': 4}
    results.sort(key=lambda x: severity_order.get(x.get('max_severity', 'NONE'), 4))
    
    return {
        "scanned": len(results),
        "with_patterns": sum(1 for r in results if r.get('patterns_count', 0) > 0),
        "results": results,
        "scanned_at": datetime.now().isoformat(),
    }


@router.get("/patterns")
async def list_all_patterns():
    """List all available TA-Lib candlestick patterns."""
    
    patterns = []
    for func, name, direction, severity in detector.ALL_PATTERNS:
        patterns.append({
            "function": func,
            "name": name,
            "direction": direction,
            "severity": severity,
        })
    
    # Group by severity
    by_severity = {
        'CRITICAL': [p for p in patterns if p['severity'] == 'CRITICAL'],
        'HIGH': [p for p in patterns if p['severity'] == 'HIGH'],
        'MEDIUM': [p for p in patterns if p['severity'] == 'MEDIUM'],
        'LOW': [p for p in patterns if p['severity'] == 'LOW'],
    }
    
    return {
        "total_patterns": len(patterns),
        "talib_available": detector.talib_available,
        "by_severity": by_severity,
        "priority_bearish": detector.PRIORITY_BEARISH,
        "priority_bullish": detector.PRIORITY_BULLISH,
        "all_patterns": patterns,
    }


@router.post("/toggle-auto-execute")
async def toggle_auto_execute(enabled: bool):
    """Toggle auto-execution mode."""
    global _sentinel
    
    if _sentinel:
        _sentinel.auto_execute = enabled
        return {
            "auto_execute": enabled,
            "message": f"Auto-execute {'ENABLED - Ghost will act on alerts!' if enabled else 'disabled'}",
        }
    
    return {"error": "Sentinel not running"}


@router.post("/set-check-interval")
async def set_check_interval(seconds: int):
    """Change the check interval (in seconds)."""
    global _sentinel
    
    if _sentinel:
        if seconds < 10:
            return {"error": "Minimum interval is 10 seconds"}
        
        _sentinel.check_interval = seconds
        return {
            "check_interval": seconds,
            "message": f"Now checking every {seconds} seconds",
        }
    
    return {"error": "Sentinel not running"}


@router.get("/recent-alerts")
async def get_recent_alerts(limit: int = 20):
    """Get recent sentinel alerts from MongoDB."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        import os
        
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client.trading_ai
        
        cursor = db.sentinel_alerts.find().sort("created_at", -1).limit(limit)
        alerts = await cursor.to_list(length=limit)
        
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
            if "created_at" in alert:
                alert["created_at"] = alert["created_at"].isoformat()
        
        return {
            "alerts": alerts,
            "count": len(alerts),
        }
    except Exception as e:
        return {"error": str(e), "alerts": []}


@router.get("/health")
async def sentinel_health():
    """Health check endpoint."""
    global _sentinel
    
    return {
        "status": "healthy",
        "sentinel_running": _sentinel is not None and _sentinel.watching,
        "talib_available": detector.talib_available,
        "timestamp": datetime.now().isoformat(),
    }

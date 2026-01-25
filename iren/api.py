"""
IREN Trading API
FastAPI endpoints for real-time trading dashboard
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iren.realtime_engine import IRENRealTimeEngine, get_full_dashboard

app = FastAPI(title="IREN Trading API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = IRENRealTimeEngine()


class TradeRequest(BaseModel):
    action: str  # BUY or SELL
    strike: float
    expiry: str
    contracts: int
    price: float


@app.get("/")
async def root():
    return {"status": "IREN Trading API Online", "version": "1.0.0"}


@app.get("/api/iren/dashboard")
async def get_dashboard():
    """Get complete trading dashboard data"""
    try:
        return get_full_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/metrics")
async def get_metrics():
    """Get real-time IREN metrics"""
    try:
        return engine.get_realtime_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/options")
async def get_options():
    """Get IREN options chain for Paul's strikes"""
    try:
        return engine.get_options_chain()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/portfolio")
async def get_portfolio():
    """Get paper trading portfolio status"""
    try:
        return engine.get_portfolio_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/earnings")
async def get_earnings_strategy():
    """Get earnings play strategy"""
    try:
        return engine.get_earnings_strategy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iren/trade")
async def execute_trade(trade: TradeRequest):
    """Execute a paper trade"""
    try:
        result = engine.execute_paper_trade(
            action=trade.action,
            strike=trade.strike,
            expiry=trade.expiry,
            contracts=trade.contracts,
            price=trade.price
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/signal")
async def get_signal():
    """Get current trading signal"""
    try:
        metrics = engine.get_realtime_metrics()
        return {
            "signal": metrics.get("signal", {}),
            "price": metrics.get("price", {}),
            "earnings": metrics.get("earnings", {}),
            "timestamp": metrics.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600)

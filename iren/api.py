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


# ============================================================
# STEADY CLIMB OPTIONS ENDPOINTS
# ============================================================

from iren.steady_climb_options import IrenSteadyClimbOptions

# Initialize Steady Climb
iren_climber = IrenSteadyClimbOptions()

@app.get("/api/iren/climb/status")
async def get_climb_status():
    """Get Steady Climb status"""
    return iren_climber.get_status()

@app.get("/api/iren/climb/signal")
async def get_climb_signal():
    """Generate a Steady Climb signal"""
    return iren_climber.generate_signal()

@app.post("/api/iren/climb/open")
async def open_climb_position(entry_price: float, contracts: int = None):
    """Open a position at the current climb level"""
    signal = iren_climber.generate_signal()
    if signal['action'] != 'BUY_CALL':
        return {"error": signal.get('reason', 'Cannot open position')}
    
    option = signal['option']
    contracts = contracts or signal['contracts']
    return iren_climber.open_position(option, entry_price, contracts)

@app.post("/api/iren/climb/close")
async def close_climb_position(exit_price: float, reason: str = "Manual"):
    """Close the current position"""
    return iren_climber.close_position(exit_price, reason)

@app.post("/api/iren/climb/reset")
async def reset_climb():
    """Reset the progression to position 1"""
    iren_climber.reset_progression("API reset")
    return {"message": "Progression reset to position 1", "status": iren_climber.get_status()}

@app.post("/api/iren/climb/win")
async def record_climb_win(profit: float):
    """Manually record a win (for testing)"""
    result = iren_climber.record_win(profit)
    return {"result": result, "status": iren_climber.get_status()}

@app.post("/api/iren/climb/loss")
async def record_climb_loss(loss: float):
    """Manually record a loss (for testing)"""
    result = iren_climber.record_loss(abs(loss) * -1)
    return {"result": result, "status": iren_climber.get_status()}

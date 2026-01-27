"""
NEO Portfolio Manager API
FastAPI endpoints for portfolio management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import logging
import json
from pathlib import Path

from .portfolio_manager import NEOPortfolioManager, run_check, get_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO_PortfolioAPI")

app = FastAPI(
    title="NEO Portfolio Manager API",
    description="Autonomous portfolio management powered by NEO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global manager instance
manager = NEOPortfolioManager()


@app.get("/")
async def root():
    return {
        "service": "NEO Portfolio Manager",
        "status": "ACTIVE",
        "version": "1.0.0",
        "endpoints": [
            "/api/portfolio/status",
            "/api/portfolio/check",
            "/api/portfolio/positions",
            "/api/portfolio/decisions",
            "/api/portfolio/rules"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/portfolio/status")
async def portfolio_status():
    """Get current portfolio status and metrics"""
    try:
        status = manager.get_portfolio_status()
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/check")
async def run_portfolio_check(background_tasks: BackgroundTasks):
    """Trigger a portfolio check and execute any needed trades"""
    try:
        # Run check (could be backgrounded for large portfolios)
        result = manager.run_portfolio_check()
        return {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error running check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/positions")
async def get_positions():
    """Get all open positions with current analysis"""
    try:
        manager._load_state()
        
        positions = []
        for p in manager.positions:
            analysis = manager.analyze_position(p)
            positions.append({
                "id": p['id'],
                "symbol": p['symbol'],
                "type": p['type'],
                "size": p['size'],
                "entry_price": p['entry_price'],
                "current_price": p.get('current_price'),
                "pnl": p.get('pnl', 0),
                "pnl_percent": p.get('pnl_percent', 0),
                "is_option": p.get('is_option', False),
                "strike": p.get('strike'),
                "expiry": p.get('expiry'),
                "option_type": p.get('option_type'),
                "source": p.get('signal_source'),
                "neo_action": analysis['action'],
                "neo_reason": analysis['reason']
            })
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(positions),
            "positions": positions
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/decisions")
async def get_decisions(limit: int = 50):
    """Get recent trading decisions"""
    try:
        decisions_file = Path("/home/jbot/trading_ai/neo/portfolio_decisions.json")
        if decisions_file.exists():
            with open(decisions_file) as f:
                decisions = json.load(f)
            return {
                "count": len(decisions[-limit:]),
                "decisions": decisions[-limit:]
            }
        return {"count": 0, "decisions": []}
    except Exception as e:
        logger.error(f"Error getting decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/rules")
async def get_rules():
    """Get current trading rules"""
    return {
        "rules": manager.rules,
        "description": {
            "shares_tp_percent": "Take profit on shares at this % gain",
            "calls_tp_percent": "Take profit on calls at this % gain (100% = 2x)",
            "scalp_tp_percent": "Take profit on scalp positions",
            "shares_sl_percent": "Stop loss on shares",
            "calls_sl_percent": "Stop loss on calls",
            "hedge_sl_percent": "Stop loss on hedges (let them ride)",
            "dca_trigger_percent": "DCA when position drops this %",
            "dca_max_entries": "Maximum DCA entries per position",
            "hedge_roll_days": "Roll hedges this many days before expiry",
            "hedge_target_ratio": "Target hedge ratio (% of portfolio)"
        }
    }


@app.put("/api/portfolio/rules")
async def update_rules(updates: dict):
    """Update trading rules"""
    try:
        for key, value in updates.items():
            if key in manager.rules:
                manager.rules[key] = value
                logger.info(f"Updated rule {key} to {value}")
        
        return {"status": "updated", "rules": manager.rules}
    except Exception as e:
        logger.error(f"Error updating rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/close/{position_id}")
async def close_position(position_id: int, reason: str = "Manual close"):
    """Manually close a position"""
    try:
        manager._load_state()
        
        # Find position
        position = None
        for p in manager.positions:
            if p['id'] == position_id:
                position = p
                break
        
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Get current price
        current_price = manager._get_market_price(
            position['symbol'],
            position.get('is_option', False),
            position.get('strike'),
            position.get('expiry'),
            position.get('option_type')
        )
        
        if current_price is None:
            current_price = position.get('current_price', position['entry_price'])
        
        # Close position
        success = manager._close_position_direct(position_id, current_price, reason)
        
        if success:
            return {"status": "closed", "position_id": position_id, "exit_price": current_price}
        else:
            raise HTTPException(status_code=500, detail="Failed to close position")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)

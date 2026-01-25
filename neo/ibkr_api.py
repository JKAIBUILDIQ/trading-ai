"""
Interactive Brokers API Endpoints
FastAPI endpoints for IBKR options trading

Endpoints:
- GET /api/ibkr/status - Connection and account status
- GET /api/ibkr/positions - Current positions
- GET /api/ibkr/options-chain/{symbol} - Options chain
- GET /api/ibkr/quote/{symbol}/{expiry}/{strike} - Option quote
- POST /api/ibkr/execute - Execute order
- POST /api/ibkr/signal - Process NEO signal
- GET /api/ibkr/orders - Open orders
- DELETE /api/ibkr/order/{order_id} - Cancel order
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo.ibkr_connector import IBKRConnector
from neo.neo_ibkr_bridge import NeoIBKRBridge

app = FastAPI(
    title="NEO-IBKR Options Trading API",
    version="1.0.0",
    description="Interactive Brokers integration for NEO's IREN signal system"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connector (initialized on first use)
_connector: Optional[IBKRConnector] = None
_bridge: Optional[NeoIBKRBridge] = None


def get_connector(paper_trading: bool = True) -> IBKRConnector:
    """Get or create IBKR connector"""
    global _connector
    if _connector is None:
        _connector = IBKRConnector(paper_trading=paper_trading)
    return _connector


def get_bridge(paper_trading: bool = True, auto_execute: bool = False) -> NeoIBKRBridge:
    """Get or create NEO-IBKR bridge"""
    global _bridge
    if _bridge is None:
        _bridge = NeoIBKRBridge(paper_trading=paper_trading, auto_execute=auto_execute)
    return _bridge


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ExecuteOrderRequest(BaseModel):
    action: str = "BUY_CALL"  # BUY_CALL, SELL_CALL, BUY_PUT
    symbol: str = "IREN"
    expiry: str  # "20260220" format
    strike: float
    quantity: int
    limit_price: Optional[float] = None


class SignalRequest(BaseModel):
    auto_execute: bool = False
    min_confidence: int = 70


class ConnectRequest(BaseModel):
    paper_trading: bool = True
    client_id: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "NEO-IBKR Options Trading API Online",
        "version": "1.0.0",
        "warning": "⚠️ Ensure TWS/IB Gateway is running before using trading endpoints",
        "endpoints": {
            "status": "/api/ibkr/status",
            "connect": "POST /api/ibkr/connect",
            "disconnect": "POST /api/ibkr/disconnect",
            "positions": "/api/ibkr/positions",
            "options_chain": "/api/ibkr/options-chain/IREN",
            "quote": "/api/ibkr/quote/IREN/{expiry}/{strike}",
            "execute": "POST /api/ibkr/execute",
            "signal": "POST /api/ibkr/signal",
            "orders": "/api/ibkr/orders",
            "cancel": "DELETE /api/ibkr/order/{order_id}"
        }
    }


@app.get("/api/ibkr/status")
async def get_status():
    """
    Get IBKR connection and account status
    
    Returns:
        Connection status, account summary, positions overview
    """
    connector = get_connector()
    bridge = get_bridge()
    
    return {
        "connected": connector.is_connected(),
        "mode": "PAPER" if connector.paper_trading else "LIVE",
        "port": connector.port,
        "bridge": {
            "auto_execute": bridge.auto_execute,
            "min_confidence": bridge.min_confidence,
            "pending_signals": len(bridge.pending_signals)
        },
        "paul_rules": IBKRConnector.PAUL_RULES,
        "message": "Connected to IBKR" if connector.is_connected() else "Not connected - start TWS/IB Gateway first",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/ibkr/connect")
async def connect_to_ibkr(request: ConnectRequest):
    """
    Connect to TWS/IB Gateway
    
    Prerequisites:
    1. TWS or IB Gateway running
    2. API enabled on port 7497 (paper) or 7496 (live)
    """
    global _connector, _bridge
    
    try:
        _connector = IBKRConnector(paper_trading=request.paper_trading, client_id=request.client_id)
        _bridge = NeoIBKRBridge(paper_trading=request.paper_trading, auto_execute=False)
        
        success = _connector.connect()
        
        if success:
            return {
                "success": True,
                "message": f"Connected to IBKR {'PAPER' if request.paper_trading else 'LIVE'} on port {_connector.port}",
                "account": _connector.get_account_summary()
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Failed to connect. Ensure TWS/IB Gateway is running with API enabled."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ibkr/disconnect")
async def disconnect_from_ibkr():
    """Disconnect from IBKR"""
    connector = get_connector()
    connector.disconnect()
    return {"success": True, "message": "Disconnected from IBKR"}


@app.get("/api/ibkr/account")
async def get_account():
    """Get account summary"""
    connector = get_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to IBKR")
    
    return connector.get_account_summary()


@app.get("/api/ibkr/positions")
async def get_positions(symbol: Optional[str] = None):
    """
    Get current positions
    
    Args:
        symbol: Filter by symbol (e.g., "IREN")
    """
    connector = get_connector()
    
    if not connector.is_connected():
        # Return cached/simulated data if not connected
        return {
            "connected": False,
            "message": "Not connected to IBKR - showing demo data",
            "positions": [
                {
                    "symbol": "IREN",
                    "sec_type": "OPT",
                    "expiry": "20260220",
                    "strike": 60,
                    "right": "C",
                    "quantity": 5,
                    "avg_cost": 2.35,
                    "current_price": 2.80,
                    "market_value": 1400,
                    "pnl": 225,
                    "pnl_pct": 19.15
                }
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    positions = connector.get_positions()
    
    if symbol:
        positions = [p for p in positions if p['symbol'] == symbol.upper()]
    
    return {
        "connected": True,
        "positions": positions,
        "total_contracts": sum(p.get('quantity', 0) for p in positions),
        "total_pnl": sum(p.get('pnl', 0) for p in positions),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ibkr/positions/iren")
async def get_iren_positions():
    """Get IREN positions only"""
    return await get_positions(symbol="IREN")


@app.get("/api/ibkr/options-chain/{symbol}")
async def get_options_chain(symbol: str, min_dte: int = 14, max_dte: int = 90):
    """
    Get options chain for a symbol
    
    Args:
        symbol: Stock symbol (e.g., "IREN")
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
    """
    connector = get_connector()
    
    if symbol.upper() != "IREN":
        raise HTTPException(status_code=400, detail="Currently only IREN is supported")
    
    if not connector.is_connected():
        # Return demo data
        return {
            "connected": False,
            "message": "Not connected to IBKR - showing demo data",
            "symbol": "IREN",
            "current_price": 55.50,
            "expirations": [
                {"expiry": "20260130", "dte": 5, "is_paul_pick": False, "warning": "Too close to earnings"},
                {"expiry": "20260205", "dte": 11, "is_paul_pick": False, "warning": "Earnings date"},
                {"expiry": "20260220", "dte": 26, "is_paul_pick": True},
                {"expiry": "20260227", "dte": 33, "is_paul_pick": True},
                {"expiry": "20260306", "dte": 40, "is_paul_pick": False}
            ],
            "strikes": [45, 50, 55, 60, 65, 70, 75, 80],
            "recommended": {
                "expirations": ["20260220", "20260227"],
                "strikes": [55, 60, 65, 70],
                "reason": "Paul prefers Feb 20 & Feb 27, avoid Feb 5 earnings"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    chain = connector.get_iren_options_chain(min_dte=min_dte, max_dte=max_dte)
    chain['connected'] = True
    return chain


@app.get("/api/ibkr/quote/{symbol}/{expiry}/{strike}")
async def get_option_quote(symbol: str, expiry: str, strike: float, right: str = "C"):
    """
    Get quote for specific option
    
    Args:
        symbol: Stock symbol
        expiry: Expiration date (YYYYMMDD)
        strike: Strike price
        right: "C" for Call, "P" for Put
    """
    connector = get_connector()
    
    if symbol.upper() != "IREN":
        raise HTTPException(status_code=400, detail="Currently only IREN is supported")
    
    if not connector.is_connected():
        # Return demo data
        return {
            "connected": False,
            "message": "Not connected to IBKR - showing demo data",
            "symbol": symbol.upper(),
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "bid": 2.30,
            "ask": 2.40,
            "last": 2.35,
            "volume": 1250,
            "open_interest": 5430,
            "iv": 0.85,
            "delta": 0.45,
            "gamma": 0.08,
            "theta": -0.05,
            "vega": 0.12,
            "timestamp": datetime.now().isoformat()
        }
    
    quote = connector.get_option_quote(expiry, strike, right)
    quote['connected'] = True
    return quote


@app.post("/api/ibkr/execute")
async def execute_order(request: ExecuteOrderRequest):
    """
    Execute an options order
    
    ⚠️ WARNING: This will place a real order on your IBKR account!
    
    Actions:
    - BUY_CALL: Buy call option
    - SELL_CALL: Sell call option (close position)
    - BUY_PUT: Buy put option (hedge)
    """
    connector = get_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to IBKR")
    
    if request.symbol.upper() != "IREN":
        raise HTTPException(status_code=400, detail="Currently only IREN is supported")
    
    # Execute based on action
    if request.action == "BUY_CALL":
        result = connector.buy_iren_call(
            expiry=request.expiry,
            strike=request.strike,
            quantity=request.quantity,
            limit_price=request.limit_price
        )
    elif request.action == "SELL_CALL":
        result = connector.sell_iren_call(
            expiry=request.expiry,
            strike=request.strike,
            quantity=request.quantity,
            limit_price=request.limit_price
        )
    elif request.action == "BUY_PUT":
        result = connector.buy_iren_put(
            expiry=request.expiry,
            strike=request.strike,
            quantity=request.quantity,
            limit_price=request.limit_price
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error', 'Order failed'))
    
    return result


@app.post("/api/ibkr/signal")
async def process_neo_signal(request: SignalRequest):
    """
    Process NEO's latest IREN signal
    
    Args:
        auto_execute: If True, automatically execute valid signals
        min_confidence: Minimum confidence to act on signal
    """
    bridge = get_bridge()
    bridge.auto_execute = request.auto_execute
    bridge.min_confidence = request.min_confidence
    
    result = bridge.process_signal_once()
    
    return {
        "signal": result.get('signal'),
        "validation": result.get('validation'),
        "executed": result.get('executed', False),
        "result": result.get('result'),
        "message": result.get('message'),
        "auto_execute_enabled": request.auto_execute,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ibkr/signal/current")
async def get_current_signal():
    """Get current NEO signal without processing"""
    bridge = get_bridge()
    signal = bridge.get_neo_iren_signal()
    validation = bridge.validate_signal(signal)
    
    return {
        "signal": signal,
        "validation": validation,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ibkr/signal/pending")
async def get_pending_signals():
    """Get pending signals (not yet executed)"""
    bridge = get_bridge()
    return {
        "pending": bridge.pending_signals,
        "count": len(bridge.pending_signals),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/ibkr/signal/execute-pending")
async def execute_pending_signals():
    """Execute all pending signals"""
    bridge = get_bridge()
    
    if not bridge.ibkr.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to IBKR")
    
    results = bridge.execute_pending()
    
    return {
        "executed": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ibkr/orders")
async def get_open_orders():
    """Get open orders"""
    connector = get_connector()
    
    if not connector.is_connected():
        return {
            "connected": False,
            "orders": [],
            "message": "Not connected to IBKR"
        }
    
    return {
        "connected": True,
        "orders": connector.get_open_orders(),
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/api/ibkr/order/{order_id}")
async def cancel_order(order_id: int):
    """Cancel an open order"""
    connector = get_connector()
    
    if not connector.is_connected():
        raise HTTPException(status_code=503, detail="Not connected to IBKR")
    
    result = connector.cancel_order(order_id)
    
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error', 'Cancel failed'))
    
    return result


@app.get("/api/ibkr/history/executions")
async def get_execution_history():
    """Get execution history"""
    bridge = get_bridge()
    
    return {
        "executed": bridge.executed_signals,
        "rejected": bridge.rejected_signals,
        "count": {
            "executed": len(bridge.executed_signals),
            "rejected": len(bridge.rejected_signals)
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ibkr/rules")
async def get_paul_rules():
    """Get Paul's trading rules"""
    return {
        "rules": IBKRConnector.PAUL_RULES,
        "description": {
            "allowed_actions": "Only BUY_CALL allowed (Paul is LONG ONLY)",
            "blocked_expirations": "Jan 30 and Feb 5 blocked (near earnings)",
            "preferred_expirations": "Feb 20 and Feb 27 preferred",
            "min_dte": "Minimum 14 days to expiration",
            "max_contracts_per_trade": "Maximum 50 contracts per trade",
            "max_total_contracts": "Maximum 200 total contracts",
            "min_confidence": "Minimum 70% confidence to execute"
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)

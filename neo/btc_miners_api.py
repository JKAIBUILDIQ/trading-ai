"""
BTC Miners Trading API
======================
API endpoints for the BTC Miners autonomous trader
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from btc_miners_trader import BTCMinersTrader, BTC_MINERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BTC_Miners_API")

app = FastAPI(
    title="BTC Miners Trading API",
    description="NEO/Meta Bot autonomous trading for IREN, CLSK, CIFR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize trader
trader = BTCMinersTrader()

DECISIONS_LOG = Path("/home/jbot/trading_ai/neo/btc_miners_decisions.json")


@app.get("/")
async def root():
    return {
        "service": "BTC Miners Trading API",
        "status": "online",
        "symbols": list(BTC_MINERS.keys()),
        "endpoints": [
            "/status - Current portfolio status",
            "/analyze/{symbol} - Analyze single symbol",
            "/analyze - Analyze all symbols",
            "/run-check - Run full trading check (may execute trades)",
            "/decisions - Recent trading decisions",
            "/config - Trading configuration"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status")
async def get_status():
    """Get current portfolio status for all BTC miners"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": {}
    }
    
    for symbol in BTC_MINERS.keys():
        config = BTC_MINERS[symbol]
        price = trader.get_current_price(symbol)
        positions = trader.get_paper_positions(symbol)
        meta_signal = trader.get_meta_signal(symbol)
        
        shares = [p for p in positions if not p.get('is_option')]
        options = [p for p in positions if p.get('is_option')]
        
        total_shares = sum(p.get('size', 0) for p in shares)
        total_cost = sum(p.get('entry_price', 0) * p.get('size', 0) for p in shares)
        avg_cost = total_cost / total_shares if total_shares > 0 else 0
        
        current_value = total_shares * price
        unrealized_pnl = current_value - total_cost
        pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
        
        status['symbols'][symbol] = {
            "name": config['name'],
            "thesis": config['thesis'],
            "current_price": round(price, 2),
            "target_price": config['target_price'],
            "upside_pct": round((config['target_price'] - price) / price * 100, 1) if price > 0 else 0,
            "shares_held": total_shares,
            "shares_target": config['accumulation_target'],
            "avg_cost": round(avg_cost, 2),
            "current_value": round(current_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "options_positions": len(options),
            "meta_signal": meta_signal.get('action', 'UNKNOWN') if meta_signal else 'NO_SIGNAL',
            "meta_confidence": meta_signal.get('confidence', 0) if meta_signal else 0,
            "progress": round(total_shares / config['accumulation_target'] * 100, 1) if config['accumulation_target'] > 0 else 0
        }
    
    # Calculate totals
    total_value = sum(s['current_value'] for s in status['symbols'].values())
    total_pnl = sum(s['unrealized_pnl'] for s in status['symbols'].values())
    
    status['portfolio'] = {
        "total_value": round(total_value, 2),
        "total_unrealized_pnl": round(total_pnl, 2),
        "symbols_count": len(BTC_MINERS)
    }
    
    return status


@app.get("/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Analyze a single symbol"""
    symbol = symbol.upper()
    
    if symbol not in BTC_MINERS:
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}. Valid: {list(BTC_MINERS.keys())}")
    
    analysis = trader.analyze_symbol(symbol)
    return analysis


@app.get("/analyze")
async def analyze_all():
    """Analyze all BTC miner symbols"""
    results = {}
    for symbol in BTC_MINERS.keys():
        results[symbol] = trader.analyze_symbol(symbol)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "analyses": results
    }


@app.post("/run-check")
async def run_trading_check():
    """
    Run full trading check - THIS MAY EXECUTE TRADES!
    Call this to trigger the daily trading logic.
    """
    results = trader.run_daily_check()
    return results


@app.get("/decisions")
async def get_decisions(limit: int = 50):
    """Get recent trading decisions"""
    try:
        if DECISIONS_LOG.exists():
            with open(DECISIONS_LOG) as f:
                decisions = json.load(f)
            return {
                "total": len(decisions),
                "showing": min(limit, len(decisions)),
                "decisions": decisions[-limit:][::-1]  # Most recent first
            }
    except Exception as e:
        logger.error(f"Failed to load decisions: {e}")
    
    return {"total": 0, "showing": 0, "decisions": []}


@app.get("/config")
async def get_config():
    """Get trading configuration for all symbols"""
    return {
        "symbols": BTC_MINERS,
        "rules": trader.rules if hasattr(trader, 'rules') else {
            "dca_trigger": -5.0,
            "tp_shares": 50.0,
            "tp_options": 100.0,
            "sl_options": -50.0
        }
    }


@app.get("/signals")
async def get_all_signals():
    """Get Meta Bot signals for all BTC miners"""
    signals = {}
    for symbol in BTC_MINERS.keys():
        signal = trader.get_meta_signal(symbol)
        if signal:
            signals[symbol] = {
                "action": signal.get('action'),
                "confidence": signal.get('confidence'),
                "composite_score": signal.get('composite_score'),
                "patterns": signal.get('patterns_detected', []),
                "reasoning": signal.get('reasoning', '')[:200]
            }
        else:
            signals[symbol] = {"action": "NO_SIGNAL", "confidence": 0}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "signals": signals
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8650)

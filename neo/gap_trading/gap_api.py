#!/usr/bin/env python3
"""
Gap Trading API
FastAPI endpoint for gap detection and signals
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn

from gap_detector import GapDetector

app = FastAPI(
    title="NEO Gap Trading API",
    description="Gap detection and fill probability for forex and gold",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = GapDetector()


@app.get("/")
async def root():
    return {
        "service": "NEO Gap Trading API",
        "version": "1.0.0",
        "endpoints": [
            "/api/gaps",
            "/api/gaps/{symbol}",
            "/api/gaps/tradeable",
            "/api/gaps/best"
        ]
    }


@app.get("/api/gaps/tradeable")
async def get_tradeable_gaps():
    """Get only tradeable gaps sorted by confidence"""
    try:
        gaps = detector.get_tradeable_gaps()
        return {
            "success": True,
            "count": len(gaps),
            "data": gaps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gaps/best")
async def get_best_gap_trade():
    """Get the single best gap trade opportunity"""
    try:
        gaps = detector.get_tradeable_gaps()
        if not gaps:
            return {
                "success": True,
                "has_trade": False,
                "message": "No tradeable gaps at this time"
            }
        
        best = gaps[0]
        return {
            "success": True,
            "has_trade": True,
            "trade": {
                "symbol": best['symbol'],
                "action": best['trade_action'],
                "direction": best['direction'],
                "gap_type": best['gap_type'],
                "gap_percent": f"{best['gap_percent']:.2f}%",
                "fill_probability": f"{best['fill_probability']:.1f}%",
                "confidence": best['confidence'],
                "entry": best['entry'],
                "stop_loss": best['stop_loss'],
                "take_profit": best['take_profit'],
                "risk_reward": f"{best['risk_reward']:.2f}",
                "fill_target": best['fill_target'],
                "reasoning": f"Gap fill trade: {best['direction']} gap detected. "
                           f"Historical fill rate for {best['symbol']} {best['direction']} gaps is "
                           f"{best['fill_probability']:.0f}%. Fade the gap with {best['trade_action']}."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gaps")
async def get_all_gaps():
    """Get all gap statuses"""
    try:
        status = detector.get_gap_status()
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gaps/{symbol}")
async def get_gap(symbol: str):
    """Get gap status for specific symbol"""
    symbol = symbol.upper()
    if symbol not in detector.YAHOO_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not supported")
    
    try:
        gap = detector.detect_gap(symbol)
        return {
            "success": True,
            "data": gap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gaps/research")
async def get_research_data():
    """Get the research data behind gap fill rates"""
    return {
        "success": True,
        "research": {
            "methodology": "365-day historical analysis of daily gaps",
            "min_gap_thresholds": detector.MIN_GAP_SIZE,
            "fill_rates": detector.FILL_RATES,
            "key_findings": [
                "DOWN gaps fill more reliably than UP gaps across all pairs",
                "USDJPY has highest overall fill rate (80.6%)",
                "Average fill time is 1.4-2.1 days",
                "Gaps between 0.3%-1.0% are optimal for trading",
                "Large gaps (>1.5%) may be breakaway gaps - avoid fading"
            ],
            "recommendations": [
                "Favor BUY signals on DOWN gaps (higher fill rate)",
                "Wait for 10% fill confirmation before entry",
                "Use 80% of gap as TP, 50% extension as SL",
                "Avoid gaps older than 24 hours"
            ]
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8750)

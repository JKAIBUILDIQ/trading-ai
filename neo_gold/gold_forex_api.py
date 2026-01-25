"""
Gold-Forex Correlation API
FastAPI endpoints for Gold-Forex correlation trading signals

Endpoints:
- GET /api/neo/gold-forex - Full correlation analysis
- GET /api/neo/gold-forex/signals - Forex signals based on Gold
- GET /api/neo/gold-forex/heatmap - Correlation heatmap
- GET /api/neo/gold-forex/hedge - Hedge recommendations
- GET /api/neo/gold-forex/divergence - Divergence detection
- GET /api/neo/gold-forex/regime - Market regime analysis
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo.gold_forex_correlator import GoldForexCorrelator
from neo.correlation_monitor import CorrelationMonitor

app = FastAPI(
    title="NEO Gold-Forex Correlation API",
    version="1.0.0",
    description="Generate forex signals based on Gold (XAUUSD) price action and correlations"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
correlator = GoldForexCorrelator()
monitor = CorrelationMonitor()


class HedgeRequest(BaseModel):
    direction: str = "LONG"  # LONG or SHORT


@app.get("/")
async def root():
    return {
        "status": "NEO Gold-Forex Correlation API Online",
        "version": "1.0.0",
        "endpoints": [
            "/api/neo/gold-forex",
            "/api/neo/gold-forex/signals",
            "/api/neo/gold-forex/heatmap",
            "/api/neo/gold-forex/hedge",
            "/api/neo/gold-forex/divergence",
            "/api/neo/gold-forex/regime",
            "/api/neo/gold-forex/best-trade"
        ]
    }


@app.get("/api/neo/gold-forex")
async def get_full_analysis():
    """
    Get complete Gold-Forex correlation analysis
    
    Returns:
        - Gold status (price, direction, strength, volatility)
        - Forex signals for all correlated pairs
        - Best single trade recommendation
        - Hedge recommendations
        - Divergence alerts
        - Correlation heatmap
    """
    try:
        analysis = correlator.get_full_analysis()
        
        # Add regime analysis from monitor
        regime = monitor.get_regime_analysis()
        analysis['market_regime'] = {
            'regime': regime['regime'],
            'description': regime['description'],
            'recommendation': regime['recommendation']
        }
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/signals")
async def get_forex_signals(min_confidence: int = Query(default=50, ge=0, le=100)):
    """
    Get forex signals based on current Gold momentum
    
    Args:
        min_confidence: Minimum confidence threshold (0-100)
        
    Returns:
        List of forex signals sorted by confidence
    """
    try:
        correlator.signal_confidence_threshold = min_confidence
        gold_momentum = correlator.get_gold_momentum()
        signals = correlator.generate_forex_signals(gold_momentum)
        
        return {
            "gold_status": {
                "price": gold_momentum['price'],
                "direction": gold_momentum['direction'],
                "strength": gold_momentum['strength'],
                "volatility": gold_momentum['volatility']
            },
            "signals": signals,
            "signal_count": len([s for s in signals if s.get('pair') != 'NONE']),
            "min_confidence": min_confidence,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/heatmap")
async def get_correlation_heatmap():
    """
    Get correlation heatmap for all monitored pairs
    
    Returns:
        Dictionary of pair correlations suitable for visualization
    """
    try:
        heatmap_data = monitor.get_correlation_heatmap()
        
        return {
            "heatmap": heatmap_data['heatmap'],
            "details": heatmap_data['details'],
            "monitored_pairs": list(monitor.monitored_pairs.keys()),
            "timestamp": heatmap_data['timestamp']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/hedge")
async def get_hedge_recommendations(direction: str = Query(default="LONG")):
    """
    Get hedge recommendations for Gold position
    
    Args:
        direction: Current Gold position direction (LONG or SHORT)
        
    Returns:
        List of recommended hedges with sizing
    """
    try:
        hedges = correlator.get_hedge_recommendations({"direction": direction.upper()})
        
        gold_momentum = correlator.get_gold_momentum()
        
        return {
            "gold_position": direction.upper(),
            "gold_price": gold_momentum['price'],
            "gold_direction": gold_momentum['direction'],
            "hedge_recommendations": hedges,
            "recommendation_count": len(hedges),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/divergence")
async def get_divergence_detection():
    """
    Detect divergences between Gold and correlated pairs
    
    Divergence = potential reversal signal
    Example: Gold up but AUD/USD down = bearish divergence for Gold
    
    Returns:
        Divergence alerts with trading implications
    """
    try:
        divergence = correlator.detect_divergence()
        
        return {
            "divergence_detected": divergence['divergence_detected'],
            "divergence_count": divergence['count'],
            "divergences": divergence['divergences'],
            "gold_momentum": {
                "direction": divergence['gold_momentum']['direction'],
                "strength": divergence['gold_momentum']['strength'],
                "change_4h": divergence['gold_momentum']['price_change_4h']
            },
            "timestamp": divergence['timestamp']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/regime")
async def get_market_regime():
    """
    Analyze current market regime based on correlation patterns
    
    Regimes:
        - RISK_OFF: Strong safe-haven flows, correlations reliable
        - RISK_ON: Risk appetite high, correlations weakening
        - TRANSITIONAL: Market regime changing
        - NORMAL: Standard patterns
        
    Returns:
        Current regime with trading recommendations
    """
    try:
        regime = monitor.get_regime_analysis()
        
        # Add breakdown alerts
        alerts = monitor.detect_correlation_breakdown()
        
        return {
            "regime": regime['regime'],
            "description": regime['description'],
            "recommendation": regime['recommendation'],
            "avg_positive_correlation": regime['avg_positive_correlation'],
            "avg_negative_correlation": regime['avg_negative_correlation'],
            "current_correlations": regime['correlations'],
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": regime['timestamp']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/best-trade")
async def get_best_trade():
    """
    Get the single best forex trade based on current Gold direction
    
    Returns:
        Best pair to trade with entry, SL, TP, and reasoning
    """
    try:
        gold_momentum = correlator.get_gold_momentum()
        best = correlator.get_best_pair_for_gold_move(gold_momentum['direction'])
        
        # Get levels if we have a valid trade
        if best['pair'] != 'NONE':
            levels = correlator._calculate_sl_tp(
                best['pair'], 
                gold_momentum, 
                best['action']
            )
            best['entry'] = levels['entry']
            best['stop_loss'] = levels['sl_price']
            best['take_profit'] = levels['tp_price']
            best['sl_pips'] = levels['sl_pips']
            best['tp_pips'] = levels['tp_pips']
            best['risk_reward'] = levels['risk_reward']
        
        return {
            "gold_status": {
                "price": gold_momentum['price'],
                "direction": gold_momentum['direction'],
                "strength": gold_momentum['strength'],
                "volatility": gold_momentum['volatility']
            },
            "best_trade": best,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/pair/{pair}")
async def get_pair_analysis(pair: str):
    """
    Get detailed correlation analysis for a specific pair
    
    Args:
        pair: Forex pair code (AUDUSD, NZDUSD, EURUSD, USDCHF, USDJPY, USDCAD)
        
    Returns:
        Detailed correlation data and trading signal for the pair
    """
    pair = pair.upper()
    
    if pair not in correlator.correlations:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid pair. Valid pairs: {list(correlator.correlations.keys())}"
        )
    
    try:
        # Get correlation analysis
        corr_analysis = monitor.calculate_rolling_correlation(pair)
        
        # Get signal for this pair
        gold_momentum = correlator.get_gold_momentum()
        signals = correlator.generate_forex_signals(gold_momentum)
        pair_signal = next((s for s in signals if s.get('pair') == pair), None)
        
        return {
            "pair": pair,
            "correlation_analysis": corr_analysis,
            "signal": pair_signal,
            "gold_momentum": {
                "direction": gold_momentum['direction'],
                "strength": gold_momentum['strength']
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo/gold-forex/monitor/update")
async def update_correlation_history():
    """
    Update correlation history (call periodically to track changes)
    """
    try:
        record = monitor.update_history()
        return {
            "status": "updated",
            "record": record
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700)

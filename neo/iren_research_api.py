#!/usr/bin/env python3
"""
IREN Research API
Serves IREN trading research and analysis data
Port: 8025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
from datetime import datetime
import logging

# Run research on startup
from neo.iren_research import run_full_research

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IRENResearchAPI")

app = FastAPI(
    title="IREN Trading Research API",
    description="Research-based trading signals and analysis for IREN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESEARCH_FILE = Path("/home/jbot/trading_ai/neo/research/iren_research.json")


def load_research():
    """Load research data from JSON file"""
    if RESEARCH_FILE.exists():
        with open(RESEARCH_FILE, 'r') as f:
            return json.load(f)
    return None


@app.get("/")
async def root():
    return {"service": "IREN Research API", "status": "online"}


@app.get("/api/iren/research")
async def get_full_research():
    """Get full IREN research data"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    return data


@app.get("/api/iren/research/summary")
async def get_research_summary():
    """Get quick trading summary for IREN"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    strategy = data.get("trading_strategy", {})
    summary = strategy.get("summary", {})
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "IREN",
        "current_price": summary.get("current_price"),
        "take_profit": {
            "pct": summary.get("recommended_tp_pct"),
            "price": summary.get("take_profit_target")
        },
        "stop_loss": {
            "pct": summary.get("recommended_sl_pct"),
            "price": summary.get("stop_loss_target")
        },
        "key_levels": strategy.get("key_levels", {}),
        "entry_rules": strategy.get("entry_rules", []),
        "exit_rules": strategy.get("exit_rules", []),
        "risk_management": strategy.get("risk_management", {})
    }


@app.get("/api/iren/research/paul")
async def get_paul_brief():
    """Get Paul-ready brief in text format"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    return {
        "format": "text",
        "brief": data.get("trading_strategy", {}).get("paul_summary", ""),
        "generated_at": data.get("trading_strategy", {}).get("generated")
    }


@app.get("/api/iren/research/volume-patterns")
async def get_volume_patterns():
    """Get volume analysis by day of week"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "volume_by_day": data.get("time_based_volume", {}).get("volume_by_day", {}),
        "best_volume_day": data.get("time_based_volume", {}).get("best_volume_day"),
        "best_return_day": data.get("time_based_volume", {}).get("best_return_day"),
        "insights": data.get("time_based_volume", {}).get("insights", [])
    }


@app.get("/api/iren/research/post-rally")
async def get_post_rally_analysis():
    """Analyze behavior after 5%+ rallies"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    post_rally = data.get("post_rally", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "threshold_pct": post_rally.get("threshold"),
        "total_occurrences": post_rally.get("total_rallies"),
        "recent_rallies": post_rally.get("rally_dates", [])[-10:],
        "next_day": post_rally.get("next_day_stats", {}),
        "next_3_days": post_rally.get("next_3day_stats", {}),
        "next_5_days": post_rally.get("next_5day_stats", {}),
        "avg_take_profit_days": post_rally.get("avg_take_profit_days"),
        "insights": post_rally.get("insights", [])
    }


@app.get("/api/iren/research/post-drop")
async def get_post_drop_analysis():
    """Analyze behavior after 3%+ drops"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    post_drop = data.get("post_drop", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "threshold_pct": post_drop.get("threshold"),
        "total_occurrences": post_drop.get("total_drops"),
        "recent_drops": post_drop.get("drop_dates", [])[-10:],
        "next_day": post_drop.get("next_day_stats", {}),
        "bounce_rate": post_drop.get("bounce_rate"),
        "insights": post_drop.get("insights", [])
    }


@app.get("/api/iren/research/divergences")
async def get_divergences():
    """Get volume/price divergence analysis"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    divs = data.get("divergences", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "bearish_divergences": {
            "recent": divs.get("bearish_divergences", [])[-5:],
            "accuracy_pct": divs.get("bearish_div_accuracy"),
            "description": "Price UP on LOW volume = likely drop"
        },
        "bullish_divergences": {
            "recent": divs.get("bullish_divergences", [])[-5:],
            "accuracy_pct": divs.get("bullish_div_accuracy"),
            "description": "Price DOWN on HIGH volume = capitulation bounce"
        },
        "insights": divs.get("insights", [])
    }


@app.get("/api/iren/research/support-resistance")
async def get_support_resistance():
    """Get key support and resistance levels"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    sr = data.get("support_resistance", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "current_price": sr.get("current_price"),
        "range_60_day": {
            "high": sr.get("60_day_high"),
            "low": sr.get("60_day_low")
        },
        "fibonacci_levels": sr.get("fibonacci_levels", {}),
        "key_supports": sr.get("key_supports", []),
        "key_resistances": sr.get("key_resistances", []),
        "insights": sr.get("insights", [])
    }


@app.get("/api/iren/research/optimal-tp-sl")
async def get_optimal_tp_sl():
    """Get optimal take-profit and stop-loss levels"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    tp_sl = data.get("optimal_tp_sl", {})
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "current_price": tp_sl.get("current_price"),
        "atr_14": tp_sl.get("atr_14"),
        "atr_pct": tp_sl.get("atr_pct"),
        "recommended": tp_sl.get("recommendations", {}),
        "all_scenarios": tp_sl.get("scenarios", []),
        "insights": tp_sl.get("insights", [])
    }


@app.post("/api/iren/research/refresh")
async def refresh_research():
    """Re-run the full research analysis"""
    try:
        results = run_full_research()
        return {
            "status": "success",
            "message": "Research refreshed",
            "timestamp": datetime.utcnow().isoformat(),
            "summary": results.get("trading_strategy", {}).get("summary", {})
        }
    except Exception as e:
        logger.error(f"Research refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/iren/signals/paper-trade")
async def get_paper_trade_signals():
    """Get signals formatted for paper trading with TP/SL"""
    data = load_research()
    if not data:
        raise HTTPException(status_code=404, detail="Research data not found")
    
    strategy = data.get("trading_strategy", {})
    summary = strategy.get("summary", {})
    sr = data.get("support_resistance", {})
    
    current_price = summary.get("current_price", 53.0)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "IREN",
        "current_price": current_price,
        "direction": "LONG",  # Default - could be dynamic based on signals
        "entry_range": {
            "low": round(current_price * 0.97, 2),
            "high": round(current_price * 1.01, 2)
        },
        "take_profit": {
            "price": summary.get("take_profit_target"),
            "pct": summary.get("recommended_tp_pct")
        },
        "stop_loss": {
            "price": summary.get("stop_loss_target"),
            "pct": summary.get("recommended_sl_pct")
        },
        "position_sizing": {
            "shares": {
                "recommended": "200 shares core position",
                "add_on_dip": "Add 50-100 shares on 5% pullback"
            },
            "options": {
                "recommended": "5-10 contracts, 3-4 weeks DTE",
                "strike": "ATM or 1 strike OTM",
                "strategy": "Buy calls on support, sell covered calls on rally"
            }
        },
        "alerts": {
            "buy_more_below": round(sr.get("key_supports", [{}])[0].get("price", current_price * 0.94), 2),
            "take_profit_above": summary.get("take_profit_target"),
            "stop_loss_below": summary.get("stop_loss_target")
        },
        "day_of_week_bias": data.get("time_based_volume", {}).get("volume_by_day", {}),
        "risk_reward": round(summary.get("recommended_tp_pct", 10) / summary.get("recommended_sl_pct", 5), 1)
    }


@app.get("/api/iren/scale-in/status")
async def get_scale_in_status():
    """Get current scale-in monitor status"""
    try:
        from neo.iren_scale_in_monitor import get_scale_in_status
        return get_scale_in_status()
    except Exception as e:
        logger.error(f"Scale-in status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iren/scale-in/check")
async def check_scale_in():
    """Manually trigger a scale-in check"""
    try:
        from neo.iren_scale_in_monitor import run_monitor, get_scale_in_status
        run_monitor()
        return get_scale_in_status()
    except Exception as e:
        logger.error(f"Scale-in check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iren/scale-in/reset")
async def reset_scale_in(new_reference: float = None):
    """Reset scale-in levels with new reference price"""
    import json
    from pathlib import Path
    from datetime import datetime
    
    state_file = Path("/home/jbot/trading_ai/neo/research/iren_scale_in_state.json")
    
    if new_reference is None:
        # Use current price
        import yfinance as yf
        ticker = yf.Ticker("IREN")
        hist = ticker.history(period='1d')
        new_reference = float(hist['Close'].iloc[-1])
    
    # Generate new levels
    levels = []
    price = new_reference
    for i in range(5):
        price = price * 0.95  # 5% drop
        levels.append({
            "level": i + 1,
            "price": round(price, 2),
            "drop_from_ref": round((new_reference - price) / new_reference * 100, 1),
            "contracts": 3,
            "status": "PENDING"
        })
    
    state = {
        "reference_price": new_reference,
        "last_scale_in_price": None,
        "scale_in_levels": levels,
        "executed_scale_ins": [],
        "alerts": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    return {
        "success": True,
        "new_reference": new_reference,
        "scale_in_levels": levels
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)

"""
DEFCON Playbook System - Market Regime + Trading Strategy
Each DEFCON level = Complete trading playbook with entry/exit rules.

DEFCON 5 = LONG_AND_STRONG  → Ghost accumulates aggressively
DEFCON 4 = BULLISH_CAUTION  → Ghost selective, smaller size
DEFCON 3 = NEUTRAL_RANGE    → Ghost & SPY both cautious
DEFCON 2 = BEARISH_ALERT    → SPY hunting, Ghost defensive
DEFCON 1 = SHORT_MODE       → SPY aggressive, Ghost closed
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Form, Body
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DefconPlaybooks")

router = APIRouter(prefix="/defcon", tags=["defcon"])

# Signal file paths
SIGNAL_DIR = "/home/jbot/trading_ai/neo/signals"
ACTIVE_DEFCON_FILE = os.path.join(SIGNAL_DIR, "active_defcon.json")

# Ensure signal directory exists
os.makedirs(SIGNAL_DIR, exist_ok=True)


# ============================================================================
# PLAYBOOK DEFINITIONS
# ============================================================================

PLAYBOOKS = {
    5: {
        "name": "LONG_AND_STRONG",
        "display_name": "Long & Strong",
        "regime": "STRONG_BULL",
        "scenario": "Clear uptrend, buying everything, no fear",
        "bias": "BULLISH",
        "color": "#22c55e",  # Green
        
        # Position rules
        "position_size_pct": 100,
        "max_positions": 5,
        "new_entries_allowed": True,
        "entry_mode": "AGGRESSIVE",
        
        # Ghost orders
        "ghost_status": "MAXIMUM_AGGRESSION",
        "ghost_orders": [
            "BUY any dip to 20 EMA",
            "ADD on breakouts with volume",
            "TRAIL stops, let winners run",
            "FULL position size allowed",
            "DCA on all pullbacks",
        ],
        
        # SPY orders
        "spy_status": "STANDBY",
        "spy_orders": [
            "NO shorts in strong bull",
            "Watch for FOMO signals only",
            "Prepare watchlist for reversal",
        ],
        
        # Entry/Exit rules
        "entry_rules": [
            "Buy any dip to 20 EMA",
            "Add on breakouts above resistance",
            "Scale in on strength",
        ],
        "stop_loss_rule": "Below recent swing low or 2% trailing",
        "take_profit_rule": "Trail stops, no hard targets",
        
        # Transition triggers
        "upgrade_trigger": None,  # Already best case
        "downgrade_trigger": [
            "FOMO score > 70",
            "RSI > 75 on daily",
            "Extended > 5% from 20 EMA",
            "Distribution candles appear",
        ],
        "downgrade_to": 4,
    },
    
    4: {
        "name": "BULLISH_CAUTION",
        "display_name": "Bullish Caution",
        "regime": "CAUTIOUS_BULL",
        "scenario": "Uptrend intact but extended, be selective",
        "bias": "CAUTIOUSLY_BULLISH",
        "color": "#84cc16",  # Lime
        
        "position_size_pct": 75,
        "max_positions": 3,
        "new_entries_allowed": True,
        "entry_mode": "SELECTIVE",
        
        "ghost_status": "SELECTIVE_BUYING",
        "ghost_orders": [
            "BUY at support levels ONLY",
            "REQUIRE confirmation candle",
            "NO chasing breakouts",
            "75% position size",
            "Tighten existing stops",
        ],
        
        "spy_status": "WATCHING",
        "spy_orders": [
            "SCAN for setups",
            "DO NOT enter yet",
            "Build target list",
        ],
        
        "entry_rules": [
            "Buy only at proven support",
            "Require bullish confirmation candle",
            "No chasing - if missed, wait",
        ],
        "stop_loss_rule": "Below support being tested",
        "take_profit_rule": "Scale out at resistance, trail remainder",
        
        "upgrade_trigger": [
            "Break above resistance with volume",
            "FOMO normalizes < 50",
            "Healthy consolidation completes",
        ],
        "upgrade_to": 5,
        "downgrade_trigger": [
            "Support test begins",
            "FOMO > 80",
            "Lower high forms",
        ],
        "downgrade_to": 3,
    },
    
    3: {
        "name": "NEUTRAL_RANGE",
        "display_name": "Neutral / Range",
        "regime": "NEUTRAL",
        "scenario": "Range-bound, unclear direction, wait for breakout",
        "bias": "NEUTRAL",
        "color": "#eab308",  # Yellow
        
        "position_size_pct": 50,
        "max_positions": 2,
        "new_entries_allowed": True,  # With strict confirmation
        "entry_mode": "RESTRICTED",
        
        "ghost_status": "MINIMAL",
        "ghost_orders": [
            "BUY only with STRONG confirmation",
            "ENTRY ZONE STRICT: support only",
            "TIGHT stops mandatory",
            "50% max position size",
            "If uncertain, STAY FLAT",
        ],
        
        "spy_status": "PREPARING",
        "spy_orders": [
            "PREPARE target list",
            "Run nightly scans",
            "Ready to activate on breakdown",
            "Do NOT short yet",
        ],
        
        "entry_rules": [
            "Entry zone only (strict)",
            "Require reversal candle pattern",
            "Confirmation on higher timeframe",
        ],
        "stop_loss_rule": "Below support level, no exceptions",
        "take_profit_rule": "TP1 at mid-range, TP2 at resistance",
        
        "upgrade_trigger": [
            "Break above range with volume",
            "Strong bounce from support",
            "Higher low confirmed",
        ],
        "upgrade_to": 4,
        "downgrade_trigger": [
            "Break below support",
            "Volume on downside",
            "Failed bounce pattern",
        ],
        "downgrade_to": 2,
    },
    
    2: {
        "name": "BEARISH_ALERT",
        "display_name": "Bearish Alert",
        "regime": "CAUTIOUS_BEAR",
        "scenario": "Breakdown starting, protect longs, SPY begins hunting",
        "bias": "BEARISH",
        "color": "#f97316",  # Orange
        
        "position_size_pct": 25,
        "max_positions": 1,
        "new_entries_allowed": False,
        "entry_mode": "FORBIDDEN",
        
        "ghost_status": "DEFENSIVE",
        "ghost_orders": [
            "NO NEW LONGS",
            "CLOSE 50% of positions",
            "TIGHTEN stops to breakeven",
            "Prepare to close all",
            "25% max remaining exposure",
        ],
        
        "spy_status": "HUNTING",
        "spy_orders": [
            "HUNTING MODE ACTIVE",
            "SHORT weak targets from nightly scan",
            "Entry on breakdown confirmation",
            "50% position size",
        ],
        
        "entry_rules": [
            "NO NEW LONG ENTRIES",
            "Manage existing positions only",
        ],
        "stop_loss_rule": "Tighten to breakeven or small profit",
        "take_profit_rule": "Close on any bounce to resistance",
        
        "upgrade_trigger": [
            "Reclaim broken support with volume",
            "V-recovery pattern forms",
            "Bullish divergence on RSI",
        ],
        "upgrade_to": 3,
        "downgrade_trigger": [
            "Accelerating breakdown",
            "Panic selling / capitulation",
            "Major support broken",
        ],
        "downgrade_to": 1,
    },
    
    1: {
        "name": "SHORT_MODE",
        "display_name": "Short Mode",
        "regime": "STRONG_BEAR",
        "scenario": "Full breakdown / capitulation, SPY aggressive, Ghost closed",
        "bias": "EXTREMELY_BEARISH",
        "color": "#ef4444",  # Red
        
        "position_size_pct": 0,
        "max_positions": 0,
        "new_entries_allowed": False,
        "entry_mode": "ZERO_LONGS",
        
        "ghost_status": "CLOSED",
        "ghost_orders": [
            "CLOSE ALL POSITIONS",
            "ZERO LONG EXPOSURE",
            "Wait for DEFCON upgrade signal",
            "Cash is king",
        ],
        
        "spy_status": "MAXIMUM_AGGRESSION",
        "spy_orders": [
            "MAXIMUM HUNTING",
            "SHORT all A-grade targets",
            "LEVERAGE allowed on best setups",
            "100% short allocation",
            "Target: NUGT, weak miners, overleveraged",
        ],
        
        "entry_rules": [
            "ZERO LONG ENTRIES",
            "ALL positions closed",
        ],
        "stop_loss_rule": "N/A - No positions",
        "take_profit_rule": "N/A - No positions",
        
        "upgrade_trigger": [
            "Capitulation candle (hammer on volume)",
            "Extreme oversold RSI < 25",
            "Volume climax / exhaustion",
            "Bullish divergence",
        ],
        "upgrade_to": 2,
        "downgrade_trigger": None,  # Already worst case
        "downgrade_to": None,
    },
}


# ============================================================================
# SCENARIO TO DEFCON MAPPING
# ============================================================================

SCENARIO_DEFCON_MAP = {
    # Bullish scenarios
    "V-Recovery": 4,
    "V-Recovery Continuation": 4,
    "Bull Flag Breakout": 5,
    "Support Hold Recovery": 3,
    "Bounce Play": 4,
    "New Highs": 5,
    "Breakout Continuation": 5,
    
    # Bearish scenarios
    "Continuation Breakdown": 2,
    "Breakdown": 2,
    "Dead Cat Bounce": 2,
    "Bear Flag Breakdown": 1,
    "Capitulation": 1,
    "Distribution": 2,
    "Lower High Rejection": 2,
    
    # Neutral scenarios
    "Sideways Consolidation": 3,
    "Range Consolidation": 3,
    "Range Bound": 3,
    "Decision Point": 3,
    "Uncertain": 3,
}


# ============================================================================
# ACTIVE STATE MANAGEMENT
# ============================================================================

# In-memory active state
ACTIVE_STATE = {
    "defcon": 3,
    "playbook": PLAYBOOKS[3],
    "scenario": "Default",
    "key_levels": None,
    "updated_at": datetime.now().isoformat(),
}


def save_active_state():
    """Save active state to signal file for Ghost/MT5."""
    try:
        signal_data = {
            "defcon": ACTIVE_STATE["defcon"],
            "playbook_name": ACTIVE_STATE["playbook"]["name"],
            "display_name": ACTIVE_STATE["playbook"]["display_name"],
            "scenario": ACTIVE_STATE.get("scenario"),
            "bias": ACTIVE_STATE["playbook"]["bias"],
            "ghost_status": ACTIVE_STATE["playbook"]["ghost_status"],
            "ghost_orders": ACTIVE_STATE["playbook"]["ghost_orders"],
            "spy_status": ACTIVE_STATE["playbook"]["spy_status"],
            "position_size_pct": ACTIVE_STATE["playbook"]["position_size_pct"],
            "new_entries_allowed": ACTIVE_STATE["playbook"]["new_entries_allowed"],
            "entry_mode": ACTIVE_STATE["playbook"]["entry_mode"],
            "max_positions": ACTIVE_STATE["playbook"]["max_positions"],
            "updated_at": datetime.now().isoformat(),
        }
        
        # Add key levels if provided
        if ACTIVE_STATE.get("key_levels"):
            signal_data["entry_zone"] = ACTIVE_STATE["key_levels"].get("entry_zone")
            signal_data["stop_loss"] = ACTIVE_STATE["key_levels"].get("stop_loss")
            signal_data["take_profit_1"] = ACTIVE_STATE["key_levels"].get("take_profit_1")
            signal_data["take_profit_2"] = ACTIVE_STATE["key_levels"].get("take_profit_2")
        
        with open(ACTIVE_DEFCON_FILE, "w") as f:
            json.dump(signal_data, f, indent=2)
        
        logger.info(f"Saved active DEFCON {ACTIVE_STATE['defcon']} to {ACTIVE_DEFCON_FILE}")
        
    except Exception as e:
        logger.error(f"Failed to save active state: {e}")


def load_active_state():
    """Load active state from signal file."""
    global ACTIVE_STATE
    
    try:
        if os.path.exists(ACTIVE_DEFCON_FILE):
            with open(ACTIVE_DEFCON_FILE, "r") as f:
                data = json.load(f)
            
            defcon = data.get("defcon", 3)
            ACTIVE_STATE = {
                "defcon": defcon,
                "playbook": PLAYBOOKS.get(defcon, PLAYBOOKS[3]),
                "scenario": data.get("scenario", "Loaded from file"),
                "key_levels": {
                    "entry_zone": data.get("entry_zone"),
                    "stop_loss": data.get("stop_loss"),
                    "take_profit_1": data.get("take_profit_1"),
                    "take_profit_2": data.get("take_profit_2"),
                } if data.get("entry_zone") else None,
                "updated_at": data.get("updated_at"),
            }
            logger.info(f"Loaded active DEFCON {defcon} from file")
    except Exception as e:
        logger.warning(f"Could not load active state: {e}")


# Load on startup
load_active_state()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/playbook/{level}")
async def get_playbook(level: int):
    """Get the playbook for a specific DEFCON level."""
    if level not in PLAYBOOKS:
        return {"error": f"Invalid DEFCON level: {level}. Must be 1-5."}
    return {
        "defcon": level,
        "playbook": PLAYBOOKS[level]
    }


@router.get("/playbooks")
async def get_all_playbooks():
    """Get all DEFCON playbooks."""
    return {
        "playbooks": PLAYBOOKS,
        "scenario_map": SCENARIO_DEFCON_MAP,
    }


@router.get("/active")
async def get_active_defcon():
    """Get the currently active DEFCON and playbook."""
    return {
        "defcon": ACTIVE_STATE["defcon"],
        "playbook": ACTIVE_STATE["playbook"],
        "scenario": ACTIVE_STATE.get("scenario"),
        "key_levels": ACTIVE_STATE.get("key_levels"),
        "updated_at": ACTIVE_STATE.get("updated_at"),
    }


class SetDefconRequest(BaseModel):
    level: int
    scenario: Optional[str] = None
    key_levels: Optional[Dict[str, Any]] = None


@router.post("/set-active")
async def set_active_defcon(request: SetDefconRequest):
    """
    Set the active DEFCON with scenario context.
    This updates what Ghost uses for trading decisions.
    """
    global ACTIVE_STATE
    
    level = request.level
    scenario = request.scenario
    key_levels = request.key_levels
    
    if level not in PLAYBOOKS:
        return {"error": f"Invalid DEFCON level: {level}. Must be 1-5."}
    
    playbook = PLAYBOOKS[level].copy()
    
    # Update active state
    ACTIVE_STATE = {
        "defcon": level,
        "playbook": playbook,
        "scenario": scenario,
        "key_levels": key_levels,
        "updated_at": datetime.now().isoformat(),
    }
    
    # Save to signal file
    save_active_state()
    
    logger.info(f"DEFCON set to {level} ({playbook['name']}) - Scenario: {scenario}")
    
    return {
        "status": "active",
        "defcon": level,
        "playbook": playbook,
        "scenario": scenario,
        "key_levels": key_levels,
        "ghost_orders": playbook["ghost_orders"],
        "spy_orders": playbook["spy_orders"],
    }


@router.post("/set-level/{level}")
async def set_defcon_level(level: int, scenario: str = None):
    """Quick endpoint to set DEFCON level."""
    return await set_active_defcon(SetDefconRequest(level=level, scenario=scenario))


@router.get("/map-scenario/{scenario_name}")
async def map_scenario_to_defcon(scenario_name: str):
    """Get recommended DEFCON for a scenario name."""
    
    # Try exact match first
    if scenario_name in SCENARIO_DEFCON_MAP:
        defcon = SCENARIO_DEFCON_MAP[scenario_name]
        return {
            "scenario": scenario_name,
            "recommended_defcon": defcon,
            "playbook": PLAYBOOKS[defcon],
        }
    
    # Try partial match
    scenario_lower = scenario_name.lower()
    for key, defcon in SCENARIO_DEFCON_MAP.items():
        if key.lower() in scenario_lower or scenario_lower in key.lower():
            return {
                "scenario": scenario_name,
                "matched_to": key,
                "recommended_defcon": defcon,
                "playbook": PLAYBOOKS[defcon],
            }
    
    # Default to DEFCON 3 for unknown
    return {
        "scenario": scenario_name,
        "recommended_defcon": 3,
        "playbook": PLAYBOOKS[3],
        "note": "Unknown scenario - defaulting to NEUTRAL",
    }


@router.get("/ghost-signal")
async def get_ghost_signal():
    """
    Get signal in format ready for Ghost/MT5.
    Returns simplified instruction set.
    """
    playbook = ACTIVE_STATE["playbook"]
    
    return {
        "defcon": ACTIVE_STATE["defcon"],
        "mode": playbook["name"],
        "bias": playbook["bias"],
        "position_size_pct": playbook["position_size_pct"],
        "new_entries": playbook["new_entries_allowed"],
        "entry_mode": playbook["entry_mode"],
        "orders": playbook["ghost_orders"],
        "entry_zone": ACTIVE_STATE.get("key_levels", {}).get("entry_zone") if ACTIVE_STATE.get("key_levels") else None,
        "stop_loss": ACTIVE_STATE.get("key_levels", {}).get("stop_loss") if ACTIVE_STATE.get("key_levels") else None,
        "take_profit_1": ACTIVE_STATE.get("key_levels", {}).get("take_profit_1") if ACTIVE_STATE.get("key_levels") else None,
    }


@router.get("/spy-signal")
async def get_spy_signal():
    """Get signal for SPY (short hunter)."""
    playbook = ACTIVE_STATE["playbook"]
    
    return {
        "defcon": ACTIVE_STATE["defcon"],
        "spy_status": playbook["spy_status"],
        "orders": playbook["spy_orders"],
        "active": playbook["spy_status"] in ["HUNTING", "MAXIMUM_AGGRESSION"],
        "aggression": "HIGH" if ACTIVE_STATE["defcon"] <= 2 else "LOW",
    }


@router.get("/transition-check")
async def check_transitions():
    """
    Check if current conditions suggest a DEFCON transition.
    Returns upgrade/downgrade suggestions.
    """
    playbook = ACTIVE_STATE["playbook"]
    current = ACTIVE_STATE["defcon"]
    
    return {
        "current_defcon": current,
        "current_name": playbook["name"],
        "upgrade_possible": playbook.get("upgrade_to") is not None,
        "upgrade_to": playbook.get("upgrade_to"),
        "upgrade_triggers": playbook.get("upgrade_trigger", []),
        "downgrade_possible": playbook.get("downgrade_to") is not None,
        "downgrade_to": playbook.get("downgrade_to"),
        "downgrade_triggers": playbook.get("downgrade_trigger", []),
    }


@router.get("/summary")
async def get_defcon_summary():
    """Get a quick summary of all DEFCON levels for display."""
    
    summary = []
    for level in [5, 4, 3, 2, 1]:
        pb = PLAYBOOKS[level]
        summary.append({
            "level": level,
            "name": pb["display_name"],
            "bias": pb["bias"],
            "color": pb["color"],
            "ghost": pb["ghost_status"],
            "spy": pb["spy_status"],
            "size": f"{pb['position_size_pct']}%",
            "active": level == ACTIVE_STATE["defcon"],
        })
    
    return {
        "summary": summary,
        "active_defcon": ACTIVE_STATE["defcon"],
    }

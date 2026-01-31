"""
Battle Intelligence API - Real-time pattern recognition with actionable guidance.

Combines:
1. Live pattern detection (what's happening NOW)
2. Historical backtest data (how reliable is this pattern?)
3. Actionable orders (exactly what to do)

This is the "field commander" that tells Ghost/you what to do when patterns appear.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BattleIntel")

router = APIRouter(prefix="/battle", tags=["battle"])

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.trading_ai


# === BATTLE CARDS ===
# Pre-defined actions based on pattern + confidence level

BATTLE_CARDS = {
    # BEARISH PATTERNS
    "SHOOTING_STAR": {
        "direction": "BEARISH",
        "description": "Long upper wick rejection at highs - sellers stepping in",
        "actions": {
            "HIGH_CONFIDENCE": {  # >70% success
                "ghost": "CLOSE 50%, tighten stops to entry",
                "manual": "Take partial profits, move stops to breakeven",
                "new_entries": False,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {  # 50-70%
                "ghost": "Tighten stops, reduce size",
                "manual": "Watch for confirmation, tighten stops",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {  # <50%
                "ghost": "Monitor only",
                "manual": "Note pattern, wait for more signals",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
    "BEARISH_ENGULFING": {
        "direction": "BEARISH",
        "description": "Large red candle swallows prior green - momentum shift",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "CLOSE 75%, no new longs",
                "manual": "Exit longs, consider short",
                "new_entries": False,
                "urgency": "CRITICAL",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "CLOSE 50%, tighten stops",
                "manual": "Reduce exposure, set tight stops",
                "new_entries": False,
                "urgency": "HIGH",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Tighten stops only",
                "manual": "Watch for follow-through",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
        },
    },
    "EVENING_STAR": {
        "direction": "BEARISH",
        "description": "3-candle reversal pattern - trend exhaustion",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "CLOSE 75%, set trailing stop",
                "manual": "Major reversal signal - exit longs",
                "new_entries": False,
                "urgency": "CRITICAL",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "CLOSE 50%, tighten stops",
                "manual": "Reduce exposure significantly",
                "new_entries": False,
                "urgency": "HIGH",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Tighten stops",
                "manual": "Caution - wait for confirmation",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
        },
    },
    "VOLUME_CLIMAX_REJECTION": {
        "direction": "BEARISH",
        "description": "Huge volume with rejection wick - institutional selling",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "CLOSE 100% longs, consider short",
                "manual": "EXIT NOW - smart money selling",
                "new_entries": False,
                "urgency": "CRITICAL",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "CLOSE 75%, no new entries",
                "manual": "Strong reversal signal - reduce heavily",
                "new_entries": False,
                "urgency": "CRITICAL",
            },
            "LOW_CONFIDENCE": {
                "ghost": "CLOSE 50%, tighten stops",
                "manual": "Significant warning - reduce exposure",
                "new_entries": False,
                "urgency": "HIGH",
            },
        },
    },
    "DISTRIBUTION_VOLUME": {
        "direction": "BEARISH",
        "description": "Multiple red candles with rising volume - institutions exiting",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "CLOSE 75%, no new longs",
                "manual": "Distribution in progress - exit positions",
                "new_entries": False,
                "urgency": "CRITICAL",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "CLOSE 50%, tighten stops",
                "manual": "Selling pressure building - reduce",
                "new_entries": False,
                "urgency": "HIGH",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Tighten stops",
                "manual": "Monitor volume for confirmation",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
        },
    },
    "HANGING_MAN": {
        "direction": "BEARISH",
        "description": "Long lower wick at top - warning of reversal",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "CLOSE 50%, tighten stops",
                "manual": "Reversal warning - take profits",
                "new_entries": False,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "Tighten stops",
                "manual": "Watch for bearish follow-through",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Monitor",
                "manual": "Note pattern, wait for confirmation",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
    
    # BULLISH PATTERNS
    "HAMMER": {
        "direction": "BULLISH",
        "description": "Long lower wick at lows - buyers stepping in",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "ADD 50% position, set stop below low",
                "manual": "Strong buy signal - scale in",
                "new_entries": True,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "ADD 25% position, tight stop",
                "manual": "Good entry point - small position",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Monitor for confirmation",
                "manual": "Wait for bullish follow-through",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
    "BULLISH_ENGULFING": {
        "direction": "BULLISH",
        "description": "Large green candle swallows prior red - momentum reversal",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "ADD 75% position, stop below pattern",
                "manual": "Strong reversal - buy aggressively",
                "new_entries": True,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "ADD 50% position",
                "manual": "Good buy signal - enter with stop",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {
                "ghost": "ADD 25% position",
                "manual": "Potential reversal - small entry",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
    "MORNING_STAR": {
        "direction": "BULLISH",
        "description": "3-candle reversal at lows - trend change",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "ADD 75% position, stop below star",
                "manual": "Major reversal - buy zone",
                "new_entries": True,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "ADD 50% position",
                "manual": "Good reversal setup - enter",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {
                "ghost": "ADD 25% position",
                "manual": "Watch for confirmation close",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
    "DRAGONFLY_DOJI": {
        "direction": "BULLISH",
        "description": "Long lower wick, no body - strong rejection of lows",
        "actions": {
            "HIGH_CONFIDENCE": {
                "ghost": "ADD 50% position",
                "manual": "Strong support found - buy",
                "new_entries": True,
                "urgency": "HIGH",
            },
            "MEDIUM_CONFIDENCE": {
                "ghost": "ADD 25% position",
                "manual": "Potential bottom - small entry",
                "new_entries": True,
                "urgency": "MEDIUM",
            },
            "LOW_CONFIDENCE": {
                "ghost": "Monitor",
                "manual": "Wait for bullish candle confirmation",
                "new_entries": True,
                "urgency": "LOW",
            },
        },
    },
}


def get_confidence_level(success_rate: float) -> str:
    """Convert success rate to confidence level."""
    if success_rate >= 70:
        return "HIGH_CONFIDENCE"
    elif success_rate >= 50:
        return "MEDIUM_CONFIDENCE"
    else:
        return "LOW_CONFIDENCE"


async def get_pattern_stats(pattern_name: str) -> Optional[Dict]:
    """Get historical stats for a pattern from most recent backtest."""
    try:
        # Get most recent backtest result
        latest = await db.backtest_results.find_one(
            {f"statistics.{pattern_name}": {"$exists": True}},
            sort=[("created_at", -1)]
        )
        
        if latest and "statistics" in latest:
            return latest["statistics"].get(pattern_name)
        return None
    except Exception as e:
        logger.error(f"Error fetching pattern stats: {e}")
        return None


@router.get("/card/{pattern_name}")
async def get_battle_card(pattern_name: str):
    """
    Get a complete battle card for a pattern.
    
    Combines:
    - Pattern description
    - Historical success rate
    - Confidence-adjusted actions
    - Ghost orders
    """
    pattern_upper = pattern_name.upper()
    
    if pattern_upper not in BATTLE_CARDS:
        return {"error": f"Unknown pattern: {pattern_name}"}
    
    card = BATTLE_CARDS[pattern_upper]
    
    # Get historical stats
    stats = await get_pattern_stats(pattern_upper)
    
    if stats:
        success_rate = stats.get("success_rate", 50)
        confidence = get_confidence_level(success_rate)
        actions = card["actions"][confidence]
        
        return {
            "pattern": pattern_upper,
            "direction": card["direction"],
            "description": card["description"],
            "historical": {
                "success_rate": success_rate,
                "total_occurrences": stats.get("total_occurrences", 0),
                "avg_move_favorable": stats.get("avg_move_favorable", 0),
                "avg_risk_reward": stats.get("avg_risk_reward", 0),
            },
            "confidence": confidence.replace("_", " "),
            "action": {
                "ghost_order": actions["ghost"],
                "manual_action": actions["manual"],
                "new_entries_allowed": actions["new_entries"],
                "urgency": actions["urgency"],
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    # No historical data - use medium confidence
    actions = card["actions"]["MEDIUM_CONFIDENCE"]
    return {
        "pattern": pattern_upper,
        "direction": card["direction"],
        "description": card["description"],
        "historical": None,
        "confidence": "MEDIUM (no data)",
        "action": {
            "ghost_order": actions["ghost"],
            "manual_action": actions["manual"],
            "new_entries_allowed": actions["new_entries"],
            "urgency": actions["urgency"],
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/active")
async def get_active_battle_situation():
    """
    Get the current battle situation - any active alerts with full battle cards.
    This is what you check when you sit down to trade.
    """
    try:
        # Get recent unacknowledged alerts
        alerts = await db.pattern_alerts.find(
            {
                "acknowledged": {"$ne": True},
                "created_at": {"$gte": datetime.now() - timedelta(hours=4)}
            }
        ).sort("created_at", -1).to_list(length=10)
        
        battle_cards = []
        
        for alert in alerts:
            pattern = alert.get("pattern", "UNKNOWN")
            
            # Get battle card
            if pattern.upper() in BATTLE_CARDS:
                card = BATTLE_CARDS[pattern.upper()]
                stats = await get_pattern_stats(pattern.upper())
                
                if stats:
                    success_rate = stats.get("success_rate", 50)
                    confidence = get_confidence_level(success_rate)
                else:
                    success_rate = 50
                    confidence = "MEDIUM_CONFIDENCE"
                
                actions = card["actions"][confidence]
                
                battle_cards.append({
                    "alert_id": str(alert["_id"]),
                    "pattern": pattern,
                    "direction": card["direction"],
                    "price_at_alert": alert.get("price"),
                    "time": alert.get("created_at").isoformat() if alert.get("created_at") else None,
                    "description": card["description"],
                    "success_rate": success_rate,
                    "confidence": confidence.replace("_", " "),
                    "urgency": actions["urgency"],
                    "action": {
                        "ghost": actions["ghost"],
                        "you": actions["manual"],
                        "new_entries": actions["new_entries"],
                    },
                })
        
        # Get current DEFCON
        defcon = await db.active_defcon.find_one({"_id": "current"})
        current_defcon = defcon.get("defcon", 5) if defcon else 5
        
        return {
            "status": "ACTIVE" if battle_cards else "ALL_CLEAR",
            "defcon": current_defcon,
            "active_patterns": len(battle_cards),
            "battle_cards": battle_cards,
            "checked_at": datetime.now().isoformat(),
            "message": f"{len(battle_cards)} patterns require attention" if battle_cards else "No active patterns - proceed normally",
        }
        
    except Exception as e:
        logger.error(f"Error getting battle situation: {e}")
        return {"error": str(e)}


@router.get("/cheatsheet")
async def get_battle_cheatsheet():
    """
    Get a quick reference cheatsheet of all patterns with their current confidence levels.
    Print this out or keep it visible while trading.
    """
    cheatsheet = {
        "bearish_patterns": [],
        "bullish_patterns": [],
        "generated_at": datetime.now().isoformat(),
    }
    
    for pattern_name, card in BATTLE_CARDS.items():
        stats = await get_pattern_stats(pattern_name)
        
        if stats:
            success_rate = stats.get("success_rate", 50)
            confidence = get_confidence_level(success_rate)
        else:
            success_rate = 50
            confidence = "MEDIUM_CONFIDENCE"
        
        actions = card["actions"][confidence]
        
        entry = {
            "pattern": pattern_name.replace("_", " "),
            "success_rate": f"{success_rate}%" if stats else "No data",
            "confidence": confidence.replace("_CONFIDENCE", ""),
            "action": actions["manual"],
            "urgency": actions["urgency"],
        }
        
        if card["direction"] == "BEARISH":
            cheatsheet["bearish_patterns"].append(entry)
        else:
            cheatsheet["bullish_patterns"].append(entry)
    
    # Sort by success rate (highest first)
    cheatsheet["bearish_patterns"].sort(
        key=lambda x: float(x["success_rate"].replace("%", "").replace("No data", "50")), 
        reverse=True
    )
    cheatsheet["bullish_patterns"].sort(
        key=lambda x: float(x["success_rate"].replace("%", "").replace("No data", "50")), 
        reverse=True
    )
    
    return cheatsheet


@router.get("/ghost-orders")
async def get_ghost_orders():
    """
    Get current orders for Ghost based on active patterns.
    Ghost calls this to know exactly what to do.
    """
    situation = await get_active_battle_situation()
    
    if situation.get("status") == "ALL_CLEAR":
        return {
            "has_orders": False,
            "action": "NORMAL_OPERATION",
            "message": "No active patterns - continue normal trading",
            "new_entries_allowed": True,
            "close_percent": 0,
            "tighten_stops": False,
        }
    
    # Find most urgent pattern
    cards = situation.get("battle_cards", [])
    if not cards:
        return {
            "has_orders": False,
            "action": "NORMAL_OPERATION",
        }
    
    # Sort by urgency
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    cards.sort(key=lambda x: urgency_order.get(x.get("urgency", "LOW"), 3))
    
    most_urgent = cards[0]
    
    # Parse ghost order
    ghost_order = most_urgent["action"]["ghost"]
    
    close_percent = 0
    if "CLOSE 100%" in ghost_order:
        close_percent = 100
    elif "CLOSE 75%" in ghost_order:
        close_percent = 75
    elif "CLOSE 50%" in ghost_order:
        close_percent = 50
    elif "CLOSE 25%" in ghost_order:
        close_percent = 25
    
    return {
        "has_orders": True,
        "pattern": most_urgent["pattern"],
        "direction": most_urgent["direction"],
        "confidence": most_urgent["confidence"],
        "urgency": most_urgent["urgency"],
        "action": ghost_order,
        "close_percent": close_percent,
        "tighten_stops": "tighten" in ghost_order.lower(),
        "new_entries_allowed": most_urgent["action"]["new_entries"],
        "price_at_signal": most_urgent.get("price_at_alert"),
        "timestamp": datetime.now().isoformat(),
    }

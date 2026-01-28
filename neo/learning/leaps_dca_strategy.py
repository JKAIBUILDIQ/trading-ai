"""
LEAPS + DCA + TP STRATEGY FOR PARADIGM SHIFT STOCKS
====================================================
When you believe in a thesis until countered, use this framework:

1. LEAPS (Long-Dated Options) - Handle the Greeks
2. DCA (Dollar Cost Average) - Build position over time  
3. TP Targets - Lock in profits, don't be greedy

Thesis Holds Until:
- Better solution emerges
- Becomes cost prohibitive
- Window of opportunity closes
- Execution fails
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# THE GREEKS PROBLEM WITH SHORT-TERM OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

GREEKS_EXPLAINED = {
    "THETA": {
        "what_it_is": "Time decay - options lose value every day",
        "problem_with_weeklies": "Lose 5-10% value PER DAY in last week",
        "problem_with_monthlies": "Lose 2-3% value per day in last 2 weeks",
        "solution": "LEAPS (6-12+ months) - theta decay is minimal",
        "example": {
            "weekly_option": "Buy Monday, lose 30% by Friday even if stock flat",
            "leap": "Buy today, lose only 0.1% per day for first 6 months"
        }
    },
    
    "VEGA": {
        "what_it_is": "Sensitivity to implied volatility",
        "problem": "After big moves, IV drops ('IV crush') destroying option value",
        "example": "IREN moves +10%, your call only up 5% due to IV crush",
        "solution": "LEAPS are less sensitive to IV changes",
        "tactic": "Buy when IV is low (before the move, not after)"
    },
    
    "DELTA": {
        "what_it_is": "How much option moves per $1 stock move",
        "atm_delta": "0.50 = option moves $0.50 per $1 stock move",
        "itm_delta": "0.70-0.90 = moves more like stock",
        "otm_delta": "0.20-0.40 = cheaper but needs big move",
        "strategy": "For thesis plays, buy ATM or slightly ITM for higher delta"
    },
    
    "GAMMA": {
        "what_it_is": "Rate of change of delta",
        "risk": "High gamma near expiration = wild swings",
        "solution": "LEAPS have lower gamma = smoother ride"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# LEAPS STRATEGY FOR THESIS-DRIVEN INVESTING
# ═══════════════════════════════════════════════════════════════════════════════

LEAPS_STRATEGY = {
    "what_are_leaps": "Options with 9-24+ months until expiration",
    
    "why_leaps_for_thesis": {
        "time_for_thesis": "Thesis plays out over 12-24 months, not 30 days",
        "survive_volatility": "Can handle 20-30% drawdowns without expiring",
        "reduced_theta": "Lose ~0.05-0.1% per day vs 2-5% for monthlies",
        "lower_stress": "Don't need to be right THIS WEEK",
        "leverage_without_margin": "Control more shares with less capital"
    },
    
    "optimal_expiry": {
        "minimum": "6 months out",
        "sweet_spot": "9-12 months out",
        "moonshot": "18-24 months out (Jan 2027, Jan 2028)",
        "avoid": "Less than 3 months (theta accelerates)"
    },
    
    "optimal_strike": {
        "conservative": "ATM (At The Money) or slightly ITM",
        "moderate": "5-10% OTM",
        "aggressive": "15-20% OTM (cheaper, needs big move)",
        "for_iren_thesis": {
            "current_price": 60,
            "conservative_strike": 60,  # ATM
            "moderate_strike": 70,      # 17% OTM
            "moonshot_strike": 100,     # 67% OTM (cheap lottery)
            "target": 150
        }
    },
    
    "example_iren_leaps": {
        "jan_2027_60_call": {
            "strike": 60,
            "expiry": "Jan 2027",
            "type": "ATM LEAP",
            "expected_premium": "~$15-20",
            "breakeven": "~$75-80 at expiry",
            "if_iren_hits_150": "Worth $90 (350%+ gain)",
            "theta_per_day": "~$0.02-0.03 (negligible)"
        },
        "jan_2027_80_call": {
            "strike": 80,
            "expiry": "Jan 2027", 
            "type": "OTM LEAP",
            "expected_premium": "~$8-12",
            "breakeven": "~$88-92 at expiry",
            "if_iren_hits_150": "Worth $70 (500%+ gain)",
            "theta_per_day": "~$0.01-0.02"
        },
        "jan_2027_100_call": {
            "strike": 100,
            "expiry": "Jan 2027",
            "type": "Deep OTM LEAP (moonshot)",
            "expected_premium": "~$4-7",
            "breakeven": "~$104-107 at expiry",
            "if_iren_hits_150": "Worth $50 (700%+ gain)",
            "risk": "Lose all if thesis fails",
            "reward": "Massive leverage if thesis succeeds"
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# DCA (DOLLAR COST AVERAGING) INTO LEAPS
# ═══════════════════════════════════════════════════════════════════════════════

DCA_STRATEGY = {
    "why_dca_leaps": {
        "reduces_timing_risk": "Don't need to pick the perfect entry",
        "averages_iv": "Buy some when IV high, some when low",
        "builds_conviction": "Add as thesis is validated",
        "manages_cash_flow": "Don't need all capital upfront"
    },
    
    "dca_schedule": {
        "initial_position": "25-33% of intended allocation",
        "add_on_dips": {
            "5%_dip": "Add 15% more",
            "10%_dip": "Add 25% more", 
            "15%_dip": "Add remaining allocation",
            "key": "Dips in paradigm shifts are BUYING OPPORTUNITIES"
        },
        "add_on_thesis_validation": {
            "new_contract_announced": "Add 10-15%",
            "earnings_beat": "Add 10%",
            "analyst_upgrade": "Add 5-10%"
        }
    },
    
    "example_dca_plan": {
        "total_allocation": "$10,000 for IREN LEAPS",
        "tranches": [
            {"timing": "Now", "amount": "$3,000", "reason": "Initial position"},
            {"timing": "On 5-10% dip", "amount": "$2,500", "reason": "Lower cost basis"},
            {"timing": "On thesis validation", "amount": "$2,500", "reason": "Conviction increase"},
            {"timing": "On 15%+ dip", "amount": "$2,000", "reason": "Aggressive accumulation"}
        ],
        "never_do": "All $10k at once on one day"
    },
    
    "rolling_strategy": {
        "when_to_roll": "When LEAP has 3-4 months left AND still believe thesis",
        "how_to_roll": "Sell current LEAP, buy new one 9-12 months out",
        "why_roll": "Avoid theta acceleration in final months",
        "example": "Sell Jan 2027 60C in Oct 2026 → Buy Jan 2028 80C"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# TAKE PROFIT TARGETS FOR LEAPS
# ═══════════════════════════════════════════════════════════════════════════════

TP_TARGETS = {
    "why_tp_matters": {
        "reason_1": "Options can give back gains FAST",
        "reason_2": "Realized gains > paper gains",
        "reason_3": "Frees up capital for next opportunity",
        "reason_4": "Removes emotional decision making"
    },
    
    "tp_ladder_for_leaps": {
        "tp1": {
            "trigger": "+50% gain",
            "action": "Sell 25% of position",
            "reason": "Lock in some profit, let rest run"
        },
        "tp2": {
            "trigger": "+100% gain (double)",
            "action": "Sell 25% more (now have 50% remaining)",
            "reason": "Now playing with house money"
        },
        "tp3": {
            "trigger": "+200% gain (3x)",
            "action": "Sell 25% more (now have 25% remaining)",
            "reason": "Massive win, protect it"
        },
        "final": {
            "trigger": "Thesis target reached OR expiry approaching",
            "action": "Close remaining position",
            "reason": "Don't let winners become losers"
        }
    },
    
    "example_iren_leap_tp": {
        "buy": "Jan 2027 $70 Call @ $10",
        "tp1_50%": "Sell 25% @ $15 → Lock $125 profit per contract",
        "tp2_100%": "Sell 25% @ $20 → Lock $250 profit per contract", 
        "tp3_200%": "Sell 25% @ $30 → Lock $500 profit per contract",
        "remaining": "Let 25% ride toward $150 target",
        "total_strategy": "Can't lose money after TP2, rest is free ride"
    },
    
    "trailing_stop_for_runners": {
        "after_tp2": "Set trailing stop at 25% below peak",
        "reason": "Let winners run but protect gains",
        "example": "LEAP hits $30, trail stop at $22.50"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# THESIS INVALIDATION TRIGGERS (When to EXIT)
# ═══════════════════════════════════════════════════════════════════════════════

THESIS_INVALIDATION = {
    "exit_triggers": {
        "better_solution": {
            "what": "New technology makes IREN's infrastructure obsolete",
            "example": "Breakthrough in chip efficiency reducing power needs 90%",
            "action": "EXIT - thesis broken",
            "probability": "LOW - physics limits efficiency gains"
        },
        "cost_prohibitive": {
            "what": "Economics no longer work for hyperscalers",
            "example": "AI compute costs drop so much that on-prem is cheaper",
            "action": "REDUCE position significantly",
            "probability": "LOW - demand still exponential"
        },
        "window_closes": {
            "what": "Competition catches up, moat erodes",
            "example": "New power permits approved everywhere, no scarcity",
            "action": "Tighten stops, reduce on rallies",
            "probability": "MEDIUM (3-5 year risk)"
        },
        "execution_failure": {
            "what": "IREN fails to deliver on contracts",
            "example": "Microsoft cancels, earnings miss badly",
            "action": "EXIT - management lost credibility",
            "probability": "LOW but monitor closely"
        }
    },
    
    "NOT_invalidation": {
        "price_drops_20%": "Volatility, not thesis break - BUY MORE",
        "btc_crashes": "Short-term correlation, thesis intact - HOLD",
        "market_correction": "All stocks down, not IREN specific - HOLD",
        "rsi_overbought": "Momentum, not fundamentals - IGNORE",
        "analysts_downgrade": "Check reasoning - usually wrong in paradigm shifts"
    },
    
    "thesis_health_check": {
        "frequency": "Monthly",
        "questions": [
            "Is AI compute demand still growing?",
            "Does IREN still have power/land advantage?",
            "Are contracts being executed successfully?",
            "Is management credible?",
            "Any better alternatives emerging?"
        ],
        "if_all_yes": "THESIS INTACT - hold and add",
        "if_any_no": "INVESTIGATE - may need to reduce"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE STRATEGY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

COMPLETE_STRATEGY = {
    "name": "LEAPS + DCA + TP for Paradigm Shift Thesis",
    
    "core_belief": "IREN thesis holds until countered",
    
    "instruments": {
        "core_shares": "Long-term hold, no stop (thesis-based exit only)",
        "leaps": "9-12 month calls, handle Greeks, leverage thesis",
        "scalp_options": "Monthly options for income (smaller size)"
    },
    
    "entry_rules": {
        "leaps": "DCA in over 2-4 weeks, add on 5-10% dips",
        "shares": "Accumulate on any red day",
        "never": "All-in on one day at market price"
    },
    
    "position_management": {
        "leaps": "Roll at 3-4 months remaining",
        "shares": "Hold unless thesis breaks",
        "stop_loss": "THESIS-BASED, not price-based"
    },
    
    "exit_rules": {
        "leaps_tp": "+50% → sell 25%, +100% → sell 25%, +200% → sell 25%",
        "shares_tp": "$150 target or thesis break",
        "thesis_break": "Exit everything"
    },
    
    "hedging": {
        "position_sizing": "No more than 30% portfolio in one thesis",
        "cash": "Keep 20% dry powder",
        "time_diversification": "LEAPS at different expiries"
    },
    
    "monitoring": {
        "daily": "Price action, news",
        "weekly": "Technical levels, options Greeks",
        "monthly": "Thesis health check",
        "quarterly": "Earnings, contract updates"
    }
}


def get_recommended_leaps(current_price: float, target_price: float, symbol: str = "IREN") -> Dict:
    """Generate recommended LEAPS based on current price and target"""
    return {
        "symbol": symbol,
        "current_price": current_price,
        "target_price": target_price,
        "upside": f"{((target_price/current_price)-1)*100:.0f}%",
        
        "recommended_leaps": {
            "conservative": {
                "strike": round(current_price, 0),
                "expiry": "Jan 2027 (12 months)",
                "type": "ATM",
                "risk": "LOW",
                "reward": "3-5x if target hit"
            },
            "moderate": {
                "strike": round(current_price * 1.15, 0),
                "expiry": "Jan 2027 (12 months)",
                "type": "15% OTM",
                "risk": "MEDIUM",
                "reward": "5-7x if target hit"
            },
            "aggressive": {
                "strike": round(current_price * 1.5, 0),
                "expiry": "Jan 2028 (24 months)",
                "type": "50% OTM",
                "risk": "HIGH",
                "reward": "10x+ if target hit"
            }
        },
        
        "dca_plan": {
            "tranche_1": "33% now",
            "tranche_2": "33% on next 5-10% dip",
            "tranche_3": "34% on thesis validation or 15% dip"
        },
        
        "tp_targets": {
            "tp1": "+50% → sell 25%",
            "tp2": "+100% → sell 25%",
            "tp3": "+200% → sell 25%",
            "runner": "Let 25% ride to target"
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("LEAPS + DCA + TP STRATEGY FOR PARADIGM SHIFTS")
    print("=" * 70)
    
    print("\n📊 RECOMMENDED LEAPS FOR IREN ($60 → $150 thesis):")
    recs = get_recommended_leaps(60, 150, "IREN")
    print(json.dumps(recs, indent=2))
    
    print("\n" + "=" * 70)
    print("📈 COMPLETE STRATEGY SUMMARY")
    print("=" * 70)
    print(json.dumps(COMPLETE_STRATEGY, indent=2))
    
    print("\n" + "=" * 70)
    print("⚠️ THESIS INVALIDATION TRIGGERS")
    print("=" * 70)
    print(json.dumps(THESIS_INVALIDATION["exit_triggers"], indent=2))

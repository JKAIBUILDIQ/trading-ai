"""
BTC MINERS STRATEGY ANALYSIS
============================
Social Bullish Trends, What Works, What to Avoid, and Hedging

Strategy: Buy Low → Sell at TP Targets → Don't Be Greedy
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# SOCIAL BULLISH TRENDS - What to Watch
# ═══════════════════════════════════════════════════════════════════════════════

SOCIAL_BULLISH_SIGNALS = {
    "EARLY_STAGE_BULLISH": {
        "description": "Good time to accumulate",
        "signals": [
            "Quiet accumulation by smart money (low volume, steady price)",
            "Negative sentiment but price holding support",
            "Insider buying or whale wallet activity",
            "Technical breakout with low social buzz",
            "BTC miners decoupling from BTC (AI thesis gaining traction)"
        ],
        "action": "BUY - Best risk/reward",
        "confidence": "HIGH"
    },
    
    "MID_STAGE_BULLISH": {
        "description": "Trend confirmed, still good entries on dips",
        "signals": [
            "Breaking out of consolidation with volume",
            "Analyst upgrades and price target increases",
            "Moderate social media buzz, not euphoric",
            "Higher lows forming on pullbacks",
            "Contract announcements (like IREN + Microsoft)"
        ],
        "action": "BUY DIPS - Scale in on pullbacks",
        "confidence": "MEDIUM-HIGH"
    },
    
    "LATE_STAGE_BULLISH": {
        "description": "⚠️ CAUTION - Take profits, don't chase",
        "signals": [
            "Parabolic price action (3+ green days in a row)",
            "Social media euphoria - 'TO THE MOON' posts everywhere",
            "FOMO buying - relatives asking about the stock",
            "Volume spikes on up days (distribution possible)",
            "RSI > 80 with bearish divergence",
            "Gap ups on open that fade",
            "Options IV crush after big moves"
        ],
        "action": "SELL INTO STRENGTH - Take profits at TP targets",
        "confidence": "Take profits, prepare for pullback"
    },
    
    "BLOW_OFF_TOP": {
        "description": "🚨 EXIT - Likely reversal incoming",
        "signals": [
            "+20% day with massive volume",
            "Social media ALL talking about it (even non-traders)",
            "News headlines: 'Stock SOARS as...'",
            "Options premiums extremely high",
            "Shorts getting squeezed (temporary)",
            "Price way above all moving averages"
        ],
        "action": "SELL - Don't be greedy, protect gains",
        "confidence": "HIGH probability of pullback"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT HAS WORKED (Based on BTC Miner Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

WHAT_WORKS = {
    "BUY_THE_DIP_SUPERTREND_BULLISH": {
        "strategy": "Buy when SuperTrend is bullish and price pulls back to EMA20/50",
        "win_rate": "~65%",
        "avg_gain": "+12-18% on options",
        "example": "IREN dip to $52 while SuperTrend bullish → rally to $60"
    },
    
    "SCALE_IN_ON_RED_DAYS": {
        "strategy": "DCA into positions on 3%+ down days",
        "win_rate": "~70% (buying fear)",
        "avg_gain": "Better average entry price",
        "example": "Add to IREN on -7% days, lower avg cost"
    },
    
    "TP_TARGETS_NOT_GREED": {
        "strategy": "Set TP at +15% stocks, +30% options - STICK TO IT",
        "win_rate": "Consistent profits",
        "key_rule": "A realized +20% is better than an unrealized +50% that becomes -10%",
        "example": "IREN calls: took +30% instead of holding for +50% which later dipped"
    },
    
    "HYPERSCALING_THESIS": {
        "strategy": "Hold core positions in AI/hyperscaling thesis",
        "rationale": "IREN, CIFR, CLSK have moats (land, power, infrastructure)",
        "horizon": "12-24 months for thesis to play out",
        "action": "Core positions = wider stops, scalp positions = tight TP"
    },
    
    "FOLLOW_VOLUME": {
        "strategy": "Buy accumulation, sell distribution",
        "green_candle_high_volume": "Bullish if price holds",
        "red_candle_high_volume": "⚠️ Possible distribution, tighten stops"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT TO BE CAUTIOUS OF
# ═══════════════════════════════════════════════════════════════════════════════

CAUTION_SIGNALS = {
    "FOMO_CHASING": {
        "description": "Buying AFTER a big move",
        "risk": "HIGH - Usually buying someone else's exit",
        "example": "Buying IREN at $60 after +15% day",
        "better_play": "Wait for pullback to EMA or support"
    },
    
    "IGNORING_BTC_CORRELATION": {
        "description": "BTC miners still partially correlated to BTC",
        "risk": "BTC dump can drag miners down short-term",
        "hedge": "Monitor BTC, tighten stops if BTC breaks key levels",
        "note": "Correlation decreasing as AI thesis strengthens"
    },
    
    "EARNINGS_VOLATILITY": {
        "description": "Holding options through earnings",
        "risk": "IV crush can destroy option value even if stock is flat",
        "better_play": "Close options before earnings OR buy further OTM for lottery"
    },
    
    "SOCIAL_MEDIA_EUPHORIA": {
        "description": "Everyone bullish = contrarian signal",
        "risk": "Smart money sells into retail FOMO",
        "indicator": "When your non-trading friends ask about IREN",
        "action": "Start taking profits, don't add new positions"
    },
    
    "OVERCONCENTRATION": {
        "description": "Too much portfolio in one sector",
        "risk": "Sector rotation can hurt all positions at once",
        "max_allocation": "BTC miners should be < 30% of total portfolio",
        "hedge_required": "YES - see hedging strategies"
    },
    
    "DIAMOND_HANDS_DELUSION": {
        "description": "Holding through -30% drawdowns hoping for recovery",
        "risk": "Opportunity cost, emotional damage",
        "rule": "If thesis breaks, exit. If thesis intact, buy the dip.",
        "thesis_break_examples": [
            "Major contract cancellation",
            "Regulatory issues",
            "Earnings miss + lowered guidance",
            "Management changes (negative)"
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# HEDGING STRATEGIES FOR BTC MINERS PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

HEDGING_STRATEGIES = {
    "QQQ_PUTS": {
        "instrument": "QQQ Put Options",
        "rationale": "Tech/growth correlation - if market dumps, miners likely follow",
        "allocation": "1-2% of portfolio",
        "strike": "5-10% OTM",
        "expiry": "60-90 days out",
        "when_to_add": "When VIX < 15 (cheap insurance)",
        "effectiveness": "Protects against broad market selloff"
    },
    
    "SBIT_OR_BITI": {
        "instrument": "ProShares Short Bitcoin ETF (SBIT) or Inverse BTC",
        "rationale": "Direct BTC hedge since miners still correlate",
        "allocation": "2-3% of portfolio when BTC extended",
        "when_to_add": "BTC RSI > 75, funding rates extremely positive",
        "effectiveness": "Protects against BTC-specific crash"
    },
    
    "CASH_POSITION": {
        "instrument": "Cash (dry powder)",
        "rationale": "Best hedge AND opportunity fund",
        "allocation": "20-30% cash during euphoric markets",
        "when_to_deploy": "3%+ down days, VIX spikes",
        "effectiveness": "Reduces volatility, enables buying dips"
    },
    
    "SECTOR_ROTATION": {
        "instrument": "Defensive sectors (utilities, staples, healthcare)",
        "rationale": "Uncorrelated to growth/tech",
        "allocation": "10-15% in defensive names",
        "when_to_rotate": "When growth extended, rotation signals",
        "effectiveness": "Smooths portfolio returns"
    },
    
    "GOLD_XAUUSD": {
        "instrument": "Gold (XAUUSD, GLD, physical)",
        "rationale": "Classic risk-off hedge, de-dollarization thesis",
        "allocation": "5-10% of portfolio",
        "correlation": "Often inverse to risk assets during panic",
        "effectiveness": "Long-term store of value, crisis hedge"
    },
    
    "COVERED_CALLS": {
        "instrument": "Sell calls against stock positions",
        "rationale": "Generate income, reduce cost basis",
        "allocation": "30-50% of stock position",
        "strike": "10-15% OTM, 30-45 days",
        "risk": "Capped upside if stock moons",
        "when_to_use": "Sideways/slightly bullish outlook"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# THE GOLDEN RULES - Buy Low, Sell at TP, Don't Be Greedy
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_RULES = {
    "RULE_1": {
        "name": "BUY LOW",
        "description": "Enter on pullbacks, not breakouts",
        "implementation": [
            "Set limit orders at support levels",
            "Buy when RSI < 40",
            "Add on 3%+ red days if thesis intact",
            "Scale in - don't all-in at once"
        ]
    },
    
    "RULE_2": {
        "name": "SELL AT TP TARGETS",
        "description": "Take profits at predetermined levels",
        "implementation": [
            "Options: +25-30% TP",
            "Stocks: +15% TP or key resistance",
            "Partial exits: Sell 50% at TP1, trail rest",
            "NEVER move TP higher 'just because'"
        ]
    },
    
    "RULE_3": {
        "name": "DON'T BE GREEDY",
        "description": "Pigs get slaughtered",
        "implementation": [
            "Realized profit > paper profit",
            "It's OK to miss the top",
            "No one went broke taking profits",
            "If up +50% and no TP, that's greed talking"
        ]
    },
    
    "RULE_4": {
        "name": "PROTECT CAPITAL",
        "description": "Stop losses are insurance, not failure",
        "implementation": [
            "Options: -30% SL (hard stop)",
            "Stocks: -8% SL or below key support",
            "NEVER remove stop losses",
            "Smaller positions = wider stops OK"
        ]
    },
    
    "RULE_5": {
        "name": "STAY HEDGED",
        "description": "Always have some protection",
        "implementation": [
            "Keep 20-30% cash",
            "Own some puts (QQQ or SBIT)",
            "Diversify across sectors",
            "Don't YOLO entire account on one thesis"
        ]
    }
}


def get_current_market_stage(social_buzz: str, price_action: str, volume: str) -> Dict:
    """
    Determine what stage of the bullish cycle we're in.
    
    Args:
        social_buzz: "quiet", "moderate", "loud", "euphoric"
        price_action: "accumulating", "breaking_out", "parabolic", "blow_off"
        volume: "low", "normal", "high", "extreme"
    """
    if social_buzz == "quiet" and price_action == "accumulating":
        return {
            "stage": "EARLY_STAGE_BULLISH",
            "action": "BUY - Best risk/reward",
            "details": SOCIAL_BULLISH_SIGNALS["EARLY_STAGE_BULLISH"]
        }
    elif social_buzz == "moderate" and price_action == "breaking_out":
        return {
            "stage": "MID_STAGE_BULLISH",
            "action": "BUY DIPS",
            "details": SOCIAL_BULLISH_SIGNALS["MID_STAGE_BULLISH"]
        }
    elif social_buzz in ["loud", "euphoric"] or price_action == "parabolic":
        return {
            "stage": "LATE_STAGE_BULLISH",
            "action": "TAKE PROFITS",
            "details": SOCIAL_BULLISH_SIGNALS["LATE_STAGE_BULLISH"]
        }
    elif price_action == "blow_off" or (social_buzz == "euphoric" and volume == "extreme"):
        return {
            "stage": "BLOW_OFF_TOP",
            "action": "EXIT NOW",
            "details": SOCIAL_BULLISH_SIGNALS["BLOW_OFF_TOP"]
        }
    else:
        return {
            "stage": "UNCERTAIN",
            "action": "HOLD - Wait for clarity",
            "details": {}
        }


def get_strategy_summary() -> Dict:
    """Get a complete strategy summary for BTC miners"""
    return {
        "core_strategy": "Buy Low → Sell at TP → Don't Be Greedy",
        
        "current_thesis": {
            "bullish_on": ["IREN", "CIFR", "CLSK"],
            "thesis": "Hyperscaling/AI pivot creates moat, decoupling from BTC",
            "time_horizon": "12-24 months",
            "conviction": "HIGH"
        },
        
        "entry_rules": {
            "buy_on_dips": "3%+ red days, RSI < 40",
            "scale_in": "Don't all-in, build position over time",
            "support_levels": "Set limit orders at key support"
        },
        
        "exit_rules": {
            "options_tp": "+25-30%",
            "stocks_tp": "+15% or key resistance",
            "stop_loss": "Options -30%, Stocks -8%",
            "partial_exit": "Sell 50% at TP1, trail rest"
        },
        
        "hedges": {
            "cash": "20-30% dry powder",
            "puts": "QQQ or SBIT puts, 1-2% allocation",
            "gold": "5-10% XAUUSD/GLD"
        },
        
        "danger_signals": [
            "Social media euphoria",
            "FOMO urge to chase",
            "Parabolic 3+ green days",
            "Friends asking about the stock",
            "Ignoring your own TP targets"
        ],
        
        "golden_rules": list(GOLDEN_RULES.keys()),
        
        "timestamp": datetime.utcnow().isoformat()
    }


def assess_current_iren_stage() -> Dict:
    """Assess where IREN is in the bullish cycle"""
    # Based on recent +14.5% day and current price near ATH
    return {
        "symbol": "IREN",
        "current_price": 59.94,
        "recent_move": "+14.5% (Jan 27)",
        
        "stage_assessment": "LATE_STAGE_BULLISH / BLOW_OFF_TOP",
        
        "social_signals": [
            "Big move attracting attention",
            "Microsoft contract news still fresh",
            "BTC at highs helping sentiment"
        ],
        
        "recommended_action": {
            "new_positions": "WAIT for pullback",
            "existing_positions": "TAKE PROFITS at TP targets",
            "options": "Consider taking +25%+ gains",
            "core_shares": "HOLD with trailing stop"
        },
        
        "key_levels": {
            "support_1": 55.00,
            "support_2": 52.00,
            "resistance_1": 60.00,
            "resistance_2": 65.00,
            "buy_zone": "52-55 (pullback entry)"
        },
        
        "hedge_recommendation": {
            "action": "Add QQQ puts if not already hedged",
            "rationale": "After +14.5% day, protection is cheap insurance"
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BTC MINERS STRATEGY ANALYSIS")
    print("=" * 60)
    
    summary = get_strategy_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 60)
    print("CURRENT IREN ASSESSMENT")
    print("=" * 60)
    
    iren = assess_current_iren_stage()
    print(json.dumps(iren, indent=2))
    
    print("\n" + "=" * 60)
    print("GOLDEN RULES")
    print("=" * 60)
    
    for rule_key, rule in GOLDEN_RULES.items():
        print(f"\n{rule['name']}: {rule['description']}")
        for impl in rule['implementation']:
            print(f"  • {impl}")

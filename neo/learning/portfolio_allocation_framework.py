"""
PORTFOLIO ALLOCATION FRAMEWORK
==============================
For Bullish Thesis Plays (IREN, CIFR, CLSK)

Allocation:
- 20% Short-term Options (Scalps) - Sell the news, momentum plays
- 50% Long-term Options (LEAPS) - Core thesis exposure
- 10% Hedges - Protection (puts, inverse ETFs)
- 20% Cash/Long Holdings - Dry powder + core shares

This balances:
- Income generation (scalps)
- Thesis conviction (LEAPS)
- Protection (hedges)
- Flexibility (cash)
"""

import json
from datetime import datetime
from typing import Dict

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ALLOCATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

ALLOCATION_MODEL = {
    "name": "Bullish Thesis Portfolio",
    "thesis": "BTC Miners → AI Hyperscaling Pivot",
    "symbols": ["IREN", "CIFR", "CLSK"],
    
    "allocation": {
        "short_term_scalps": {
            "percentage": 20,
            "description": "3-4 week call options for momentum/news plays",
            "purpose": "Income generation, sell-the-news profits",
            "risk_level": "HIGH (but small allocation)",
            "expected_return": "+20-50% per trade",
            "trade_frequency": "2-4 trades per month"
        },
        "long_term_leaps": {
            "percentage": 50,
            "description": "9-24 month call options",
            "purpose": "Core thesis exposure with leverage",
            "risk_level": "MEDIUM (time handles Greeks)",
            "expected_return": "3-10x if thesis plays out",
            "trade_frequency": "Roll every 6-9 months"
        },
        "hedges": {
            "percentage": 10,
            "description": "Puts, inverse ETFs, protection",
            "purpose": "Downside protection, sleep at night",
            "risk_level": "LOW (insurance cost)",
            "expected_return": "Lose premium if thesis right (that's OK)",
            "instruments": ["QQQ puts", "SBIT", "VIX calls"]
        },
        "cash_and_shares": {
            "percentage": 20,
            "description": "Cash for DCA + core share holdings",
            "purpose": "Dry powder for dips, long-term compounding",
            "risk_level": "LOW-MEDIUM",
            "expected_return": "Shares: match stock gains, Cash: optionality",
            "split": "10% cash, 10% shares (adjust as needed)"
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM SCALP STRATEGY (20%)
# ═══════════════════════════════════════════════════════════════════════════════

SCALP_STRATEGY = {
    "allocation": "20% of thesis portfolio",
    "timeframe": "3-4 weeks to expiration",
    "strike_selection": "ATM or slightly OTM (5-10%)",
    
    "entry_triggers": {
        "sell_the_news_setup": {
            "description": "Buy BEFORE expected catalyst, sell INTO the news",
            "examples": [
                "Earnings coming up (buy 2 weeks before)",
                "Contract announcement expected",
                "Analyst day / investor presentation",
                "Sector catalyst (BTC halving, AI conference)"
            ],
            "entry": "1-2 weeks before catalyst",
            "exit": "On day of news or day after (sell the pop)"
        },
        "technical_momentum": {
            "description": "Buy breakouts, ride momentum",
            "triggers": [
                "Break above key resistance with volume",
                "RSI divergence turning bullish",
                "EMA crossover (20 > 50)",
                "Sector rotation into AI/tech"
            ],
            "entry": "On breakout confirmation",
            "exit": "+20-30% or resistance reached"
        },
        "oversold_bounce": {
            "description": "Buy fear, sell relief",
            "triggers": [
                "RSI < 30",
                "3+ red days in a row",
                "VIX spike (fear)",
                "Support bounce"
            ],
            "entry": "When RSI < 35 and support holds",
            "exit": "+15-25% or RSI > 50"
        }
    },
    
    "exit_rules": {
        "take_profit": "+20-50% (don't be greedy on short-term)",
        "stop_loss": "-30% (hard stop, no exceptions)",
        "time_stop": "Close if < 1 week to expiry and not profitable",
        "news_exit": "ALWAYS sell into the news, not after"
    },
    
    "position_sizing": {
        "max_per_trade": "25% of scalp allocation (5% of total)",
        "example": "$100k portfolio → $20k scalps → $5k max per trade",
        "reason": "Scalps are high risk, size accordingly"
    },
    
    "sell_the_news_playbook": {
        "why_it_works": "Markets price in expectations BEFORE news",
        "pattern": [
            "Catalyst expected (earnings, contract)",
            "Stock runs up in anticipation",
            "News drops → brief spike",
            "Profit-taking → pullback",
            "Those who bought the news are bagholders"
        ],
        "your_play": [
            "Buy calls 1-2 weeks before catalyst",
            "Ride anticipation rally",
            "Sell INTO the news (before/at announcement)",
            "Let others hold the bag",
            "Repeat"
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# LONG-TERM LEAPS STRATEGY (50%)
# ═══════════════════════════════════════════════════════════════════════════════

LEAPS_STRATEGY = {
    "allocation": "50% of thesis portfolio",
    "timeframe": "9-24 months to expiration",
    "strike_selection": "ATM to 20% OTM",
    
    "purpose": "Core thesis exposure with leverage",
    
    "entry_rules": {
        "dca_approach": "Build position over 2-4 weeks",
        "tranches": ["33% now", "33% on dip", "34% on validation"],
        "add_triggers": ["5-10% pullback", "New contract announced", "Earnings beat"]
    },
    
    "management": {
        "roll_timing": "3-4 months before expiry",
        "tp_ladder": ["+50% sell 25%", "+100% sell 25%", "+200% sell 25%", "runner"],
        "stop_loss": "THESIS-BASED only (not price)"
    },
    
    "diversification": {
        "across_strikes": "Mix of ATM and OTM",
        "across_expiries": "Jan 2027 + Jan 2028",
        "across_symbols": "IREN 50%, CIFR 25%, CLSK 25%"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# HEDGE STRATEGY (10%)
# ═══════════════════════════════════════════════════════════════════════════════

HEDGE_STRATEGY = {
    "allocation": "10% of thesis portfolio",
    "purpose": "Insurance against thesis failure or market crash",
    
    "instruments": {
        "qqq_puts": {
            "allocation": "5% of hedge (0.5% total)",
            "strike": "5-10% OTM",
            "expiry": "60-90 days",
            "when_to_buy": "VIX < 15 (cheap insurance)",
            "purpose": "Broad market crash protection"
        },
        "sbit_shares": {
            "allocation": "3% of hedge (0.3% total)",
            "what": "ProShares Short Bitcoin ETF",
            "when_to_hold": "BTC RSI > 75 or euphoria",
            "purpose": "BTC correlation hedge"
        },
        "cash_in_hedge": {
            "allocation": "2% of hedge (0.2% total)",
            "purpose": "Dry powder for panic buying opportunities"
        }
    },
    
    "rebalancing": {
        "frequency": "Monthly",
        "if_puts_expire_worthless": "Good - means thesis working, rebuy",
        "if_puts_gain_value": "Sell for profit, reassess thesis"
    },
    
    "mental_model": "Hedge is insurance premium. Expect to lose it. That's the cost of sleeping well."
}

# ═══════════════════════════════════════════════════════════════════════════════
# CASH & SHARES STRATEGY (20%)
# ═══════════════════════════════════════════════════════════════════════════════

CASH_SHARES_STRATEGY = {
    "allocation": "20% of thesis portfolio",
    
    "split": {
        "cash": {
            "percentage": "10% of total (half of this bucket)",
            "purpose": "DCA ammunition, panic buy opportunities",
            "deployment": [
                "5-10% dip → deploy 25% of cash",
                "15%+ dip → deploy 50% of cash",
                "20%+ crash → deploy remaining cash"
            ],
            "never": "Deploy all cash at once"
        },
        "shares": {
            "percentage": "10% of total (half of this bucket)",
            "purpose": "Long-term core holding, no expiry risk",
            "management": "Hold until $150 target or thesis breaks",
            "income": "Can sell covered calls for income"
        }
    },
    
    "covered_call_strategy": {
        "on_shares": "Sell 30-45 DTE calls 15-20% OTM",
        "purpose": "Generate 2-4% monthly income",
        "risk": "Shares called away if stock moons",
        "mitigation": "Only cover 50% of shares"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE PORTFOLIO ($100,000)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_example_portfolio(total_capital: float = 100000) -> Dict:
    """Generate example portfolio allocation"""
    
    scalps = total_capital * 0.20
    leaps = total_capital * 0.50
    hedges = total_capital * 0.10
    cash_shares = total_capital * 0.20
    
    return {
        "total_capital": f"${total_capital:,.0f}",
        
        "allocations": {
            "short_term_scalps": {
                "amount": f"${scalps:,.0f}",
                "percentage": "20%",
                "positions": [
                    {"symbol": "IREN", "type": "Feb 2026 $65 Call", "amount": f"${scalps*0.5:,.0f}", "thesis": "Earnings run-up"},
                    {"symbol": "CIFR", "type": "Feb 2026 $20 Call", "amount": f"${scalps*0.25:,.0f}", "thesis": "Sector momentum"},
                    {"symbol": "CLSK", "type": "Feb 2026 $15 Call", "amount": f"${scalps*0.25:,.0f}", "thesis": "BTC correlation play"}
                ]
            },
            
            "long_term_leaps": {
                "amount": f"${leaps:,.0f}",
                "percentage": "50%",
                "positions": [
                    {"symbol": "IREN", "type": "Jan 2027 $70 Call", "amount": f"${leaps*0.50:,.0f}", "thesis": "Core thesis"},
                    {"symbol": "IREN", "type": "Jan 2028 $100 Call", "amount": f"${leaps*0.15:,.0f}", "thesis": "Moonshot"},
                    {"symbol": "CIFR", "type": "Jan 2027 $25 Call", "amount": f"${leaps*0.20:,.0f}", "thesis": "Secondary play"},
                    {"symbol": "CLSK", "type": "Jan 2027 $18 Call", "amount": f"${leaps*0.15:,.0f}", "thesis": "Diversification"}
                ]
            },
            
            "hedges": {
                "amount": f"${hedges:,.0f}",
                "percentage": "10%",
                "positions": [
                    {"symbol": "QQQ", "type": "Apr 2026 $480 Put", "amount": f"${hedges*0.50:,.0f}", "thesis": "Market crash protection"},
                    {"symbol": "SBIT", "type": "Shares", "amount": f"${hedges*0.30:,.0f}", "thesis": "BTC hedge"},
                    {"symbol": "CASH", "type": "Reserve", "amount": f"${hedges*0.20:,.0f}", "thesis": "Panic buy fund"}
                ]
            },
            
            "cash_and_shares": {
                "amount": f"${cash_shares:,.0f}",
                "percentage": "20%",
                "positions": [
                    {"symbol": "IREN", "type": "Shares", "amount": f"${cash_shares*0.40:,.0f}", "thesis": "Core long-term hold"},
                    {"symbol": "CIFR", "type": "Shares", "amount": f"${cash_shares*0.10:,.0f}", "thesis": "Small position"},
                    {"symbol": "CASH", "type": "Dry Powder", "amount": f"${cash_shares*0.50:,.0f}", "thesis": "DCA on dips"}
                ]
            }
        },
        
        "risk_profile": {
            "max_loss_if_all_options_expire": f"${scalps + leaps + (hedges*0.8):,.0f} (70% of portfolio)",
            "likely_scenario": "LEAPs gain, scalps +/-, hedges expire = net positive",
            "best_case": f"LEAPs 5x, scalps +50% = ${leaps*5 + scalps*1.5:,.0f}",
            "protection": "10% hedges + 10% cash = 20% protected"
        },
        
        "rebalancing_rules": [
            "Monthly: Review scalp positions, close winners/losers",
            "Quarterly: Assess LEAP positions, roll if needed",
            "On big moves: Take profits on scalps, redeploy to LEAPs",
            "Thesis check: Monthly review of invalidation triggers"
        ]
    }

# ═══════════════════════════════════════════════════════════════════════════════
# BALANCE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assess_allocation_balance() -> Dict:
    """Assess if the 20/50/10/20 allocation is balanced"""
    return {
        "allocation": {
            "scalps": "20%",
            "leaps": "50%",
            "hedges": "10%",
            "cash_shares": "20%"
        },
        
        "assessment": "WELL BALANCED ✅",
        
        "strengths": [
            "50% in LEAPs = strong thesis conviction with time buffer",
            "20% scalps = income generation without overexposure",
            "10% hedges = sleep-at-night protection",
            "20% cash/shares = flexibility and stability"
        ],
        
        "risk_analysis": {
            "aggressive_exposure": "70% (scalps + LEAPs)",
            "defensive_exposure": "30% (hedges + cash/shares)",
            "verdict": "Appropriately aggressive for high-conviction thesis"
        },
        
        "comparison_to_alternatives": {
            "too_conservative": "50% cash would miss the move",
            "too_aggressive": "80% options would be overleveraged",
            "your_model": "70/30 is optimal for validated thesis"
        },
        
        "adjustments_for_scenarios": {
            "if_less_conviction": "Reduce LEAPs to 40%, increase cash to 30%",
            "if_more_conviction": "Reduce cash to 10%, increase LEAPs to 60%",
            "if_hedging_more": "Reduce scalps to 15%, increase hedges to 15%"
        },
        
        "verdict": """
        ✅ This is a GOOD balance for a bullish thesis.
        
        - You have conviction (70% options)
        - You have protection (10% hedges)
        - You have flexibility (20% cash/shares)
        - You generate income (20% scalps)
        - You ride the thesis (50% LEAPs)
        
        The key is EXECUTION:
        - Scalps: Sell INTO the news, not after
        - LEAPs: Use TP ladder, don't be greedy
        - Hedges: Accept them as insurance cost
        - Cash: Deploy on dips, not FOMO
        """
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PORTFOLIO ALLOCATION FRAMEWORK")
    print("20% Scalps | 50% LEAPs | 10% Hedges | 20% Cash/Shares")
    print("=" * 70)
    
    print("\n📊 ALLOCATION ASSESSMENT:")
    assessment = assess_allocation_balance()
    print(json.dumps(assessment, indent=2))
    
    print("\n" + "=" * 70)
    print("💰 EXAMPLE $100K PORTFOLIO")
    print("=" * 70)
    portfolio = generate_example_portfolio(100000)
    print(json.dumps(portfolio, indent=2))
    
    print("\n" + "=" * 70)
    print("📈 SCALP STRATEGY (SELL THE NEWS)")
    print("=" * 70)
    print(json.dumps(SCALP_STRATEGY["sell_the_news_playbook"], indent=2))

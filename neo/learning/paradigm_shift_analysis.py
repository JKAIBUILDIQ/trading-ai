"""
PARADIGM SHIFT ANALYSIS
=======================
When a company pivots from one sector to another with exponential demand,
traditional "buy the dip" strategies can cause you to MISS THE ENTIRE MOVE.

Key Case Study: NVIDIA
- Before: Gaming GPUs, crypto mining
- Pivot Point: AI/ML compute demand exploded
- Result: Those waiting for "pullbacks" missed 6000%+ gains

Question: Is IREN following the same pattern?
"""

import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL PARADIGM SHIFTS - Companies That Pivoted and Exploded
# ═══════════════════════════════════════════════════════════════════════════════

PARADIGM_SHIFTS = {
    "NVIDIA": {
        "original_business": "Gaming GPUs, Crypto Mining",
        "pivot_to": "AI/ML Training Infrastructure",
        "pivot_year": "2022-2023",
        "pre_pivot_price": "$15 (split-adjusted, 2022)",
        "post_pivot_price": "$140+ (2024), $950+ (2025)",
        "gain": "6000%+",
        "thesis": "AI compute demand is exponential, NVIDIA has no competition",
        "key_insight": "Those waiting for 'pullbacks' from $20 to $15 missed the move to $950",
        "pattern": [
            "Initial skepticism ('just gaming chips')",
            "Early adopters accumulate quietly",
            "Major contract announcements (Microsoft, Google, Meta)",
            "Analysts scramble to upgrade",
            "FOMO phase BUT fundamentals support it",
            "Becomes 'obvious' in hindsight"
        ],
        "what_bears_said": [
            "'RSI overbought, due for correction'",
            "'P/E too high'",
            "'Just wait for the pullback'",
            "'This is a bubble'"
        ],
        "what_actually_happened": "Every 'pullback' was a buying opportunity. Dips were 5-10%, not 30%."
    },
    
    "AMAZON_AWS": {
        "original_business": "E-commerce retailer",
        "pivot_to": "Cloud Infrastructure (AWS)",
        "pivot_year": "2006-2015",
        "pre_pivot_price": "$5 (2008)",
        "post_pivot_price": "$180+ (2024)",
        "gain": "3500%+",
        "thesis": "Cloud computing is the future, AWS has first-mover advantage",
        "key_insight": "E-commerce was the loss leader, AWS became the profit engine",
        "pattern": [
            "Retail investors focused on e-commerce losses",
            "Smart money saw AWS potential early",
            "Enterprise adoption accelerated",
            "Became 'too expensive' at every price point",
            "Now worth $2T+"
        ]
    },
    
    "TESLA": {
        "original_business": "Electric car startup",
        "pivot_to": "Energy/AI/Robotics Platform",
        "pivot_year": "2020-2021",
        "pre_pivot_price": "$25 (split-adjusted, 2019)",
        "post_pivot_price": "$400+ (2021), volatile since",
        "gain": "1500%+ from lows",
        "thesis": "Not a car company - it's an energy/AI company",
        "key_insight": "Those who saw only 'car company' missed the thesis",
        "volatility_note": "High volatility due to Musk factor, but thesis intact"
    },
    
    "MICROSOFT_CLOUD": {
        "original_business": "Software (Windows, Office)",
        "pivot_to": "Cloud (Azure) + AI (OpenAI partnership)",
        "pivot_year": "2014-2020 (Satya Nadella era)",
        "pre_pivot_price": "$40 (2014)",
        "post_pivot_price": "$420+ (2024)",
        "gain": "1000%+",
        "thesis": "Cloud + AI integration creates unstoppable moat",
        "key_insight": "Ballmer era: dead money. Nadella era: 10x"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# IREN PARADIGM SHIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

IREN_PARADIGM_ANALYSIS = {
    "company": "IREN (Iris Energy)",
    "original_business": "Bitcoin Mining",
    "pivot_to": "AI Hyperscaling Infrastructure",
    
    "nvidia_parallels": {
        "similarity_score": "HIGH",
        "parallels": [
            "Both had existing infrastructure for one purpose (gaming/mining)",
            "Both pivoted to serve AI compute demand",
            "Both have PHYSICAL MOAT (chips for NVIDIA, land/power for IREN)",
            "Both benefit from exponential AI demand",
            "Both had skeptics saying 'just a mining company'"
        ],
        "differences": [
            "NVIDIA makes chips, IREN provides infrastructure",
            "NVIDIA is established, IREN is earlier stage",
            "IREN's moat is harder to replicate (power permits take 3-5 years)",
            "IREN still has BTC mining revenue as hedge"
        ]
    },
    
    "the_moat": {
        "description": "Why waiting might mean missing the move",
        "factors": [
            "Land rights in Texas - locked in",
            "Power agreements - grandfathered rates",
            "New competitors need 3-5 YEARS to get permits",
            "AI demand growing exponentially NOW",
            "Microsoft contract validates the pivot",
            "More contracts likely coming (Google, Meta, etc.)"
        ]
    },
    
    "bull_case": {
        "target": "$150 (150%+ from current)",
        "timeline": "12-24 months",
        "catalysts": [
            "Additional hyperscaler contracts",
            "Revenue ramp from Microsoft deal",
            "Analyst upgrades as thesis is validated",
            "Sector re-rating (AI infra, not 'crypto miner')",
            "Possible acquisition target"
        ],
        "key_insight": "If thesis is right, $60 is CHEAP, not expensive"
    },
    
    "bear_case": {
        "concerns": [
            "Execution risk on buildout",
            "BTC price crash could hurt sentiment",
            "Competition from other miners pivoting",
            "Macro recession reduces AI spend"
        ],
        "counter_to_bears": "Even if these happen, the infrastructure VALUE remains"
    },
    
    "why_traditional_ta_fails_here": {
        "reason": "Paradigm shifts break technical patterns",
        "examples": [
            "RSI 'overbought' at $40 → went to $60",
            "'Resistance' at $50 → blew through it",
            "'Wait for pullback to $45' → never came",
            "Every dip was shallow (5-10%, not 30%)"
        ],
        "lesson": "In paradigm shifts, the 'expensive' price becomes 'cheap' in hindsight"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# INVESTMENT FRAMEWORK: Thesis-Driven vs Technical-Driven
# ═══════════════════════════════════════════════════════════════════════════════

INVESTMENT_FRAMEWORKS = {
    "TECHNICAL_DRIVEN": {
        "description": "Traditional: Buy dips, sell resistance",
        "best_for": "Range-bound stocks, no catalyst",
        "works_when": "No paradigm shift, normal market",
        "fails_when": "Paradigm shift - you'll wait forever for 'the dip'",
        "example": "Waiting for NVDA pullback from $20 to $15... it went to $900"
    },
    
    "THESIS_DRIVEN": {
        "description": "Buy when thesis is validated, hold until thesis breaks",
        "best_for": "Paradigm shift stocks (NVDA, IREN type)",
        "works_when": "Clear catalyst + moat + exponential demand",
        "key_rules": [
            "Size position for conviction (larger if high conviction)",
            "Don't wait for 'perfect' entry - thesis matters more",
            "Add on SMALL dips, don't wait for big ones",
            "Hold until THESIS breaks, not until price drops",
            "Ignore RSI overbought signals"
        ],
        "thesis_break_triggers": [
            "Contract cancellation",
            "Failed execution",
            "Better technology emerges",
            "Management loses credibility"
        ]
    },
    
    "HYBRID_APPROACH_FOR_IREN": {
        "description": "Best of both for IREN specifically",
        "core_position": {
            "size": "60-70% of intended allocation",
            "entry": "NOW - thesis is validated",
            "hold_until": "Thesis breaks OR $150 target",
            "stop": "Only if thesis breaks (e.g., Microsoft cancels)"
        },
        "scalp_position": {
            "size": "30-40% for trading",
            "entry": "On 5-10% dips (not 30%)",
            "exit": "+20-30% gains",
            "purpose": "Generate income while holding core"
        },
        "key_insight": "Don't miss $60 → $150 waiting for $52"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# THE MINI-NVIDIA THESIS
# ═══════════════════════════════════════════════════════════════════════════════

MINI_NVIDIA_THESIS = {
    "title": "Is IREN the Next Mini-NVIDIA?",
    
    "similarities": {
        "infrastructure_pivot": "Both pivoted from one use case to AI",
        "physical_moat": "Chips (NVDA) vs Land/Power (IREN)",
        "exponential_demand": "AI compute needs growing 10x per year",
        "skeptic_phase": "Both dismissed as 'just gaming/mining'",
        "validation_catalyst": "Major contracts with hyperscalers"
    },
    
    "scale_difference": {
        "nvidia_tam": "$1 Trillion+ AI chip market",
        "iren_tam": "$100B+ AI infrastructure market",
        "implication": "IREN won't be $3T company, but $10-30B is realistic"
    },
    
    "price_trajectory_comparison": {
        "nvidia_pattern": "$15 → $150 (10x) → $950 (60x) over 3 years",
        "iren_potential": "$10 → $60 (6x) → $150 (15x) → ???",
        "current_stage": "Early-mid innings (Microsoft contract = NVIDIA's ChatGPT moment)"
    },
    
    "what_this_means_for_strategy": {
        "traditional_advice": "Wait for pullback to $52",
        "paradigm_shift_advice": "Accumulate NOW, add on any dip",
        "reason": "If IREN follows NVDA pattern, pullbacks will be shallow",
        "risk_of_waiting": "Miss the entire move waiting for 'the perfect entry'"
    },
    
    "conviction_levels": {
        "high_conviction": "Full position now, add on any dip",
        "medium_conviction": "50% now, 50% on 10% dip",
        "low_conviction": "25% now, scale in slowly"
    },
    
    "conclusion": """
    Your insight is correct. In paradigm shifts:
    
    1. Traditional 'buy the dip' can mean missing the train
    2. NVIDIA taught us that 'expensive' becomes 'cheap'
    3. The moat (land/power) is REAL and takes years to replicate
    4. Microsoft contract is validation, not hype
    
    If you believe $150 is coming, $60 is not expensive - it's early.
    
    The question isn't 'will it dip to $52?'
    The question is 'will it reach $150?'
    
    If YES → $60 is a buy
    If NO → Don't buy at any price
    """
}


def get_paradigm_shift_summary():
    """Get a summary of paradigm shift analysis"""
    return {
        "core_insight": "In paradigm shifts, waiting for 'the dip' means missing the move",
        "nvidia_lesson": "Those waiting for NVDA pullback from $20 to $15 missed $20 → $900",
        "iren_thesis": "Hyperscaling pivot = NVIDIA pattern at smaller scale",
        "moat": "Land + Power permits take 3-5 years - this is REAL",
        "recommendation": {
            "if_high_conviction": "Accumulate NOW, add on any 5-10% dip",
            "if_medium_conviction": "50% now, 50% on 10% dip",
            "stop_loss": "Only if THESIS breaks, not price dips"
        },
        "key_question": "Do you believe $150? If yes, $60 is cheap.",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PARADIGM SHIFT ANALYSIS: IREN vs NVIDIA")
    print("=" * 70)
    
    print("\n📊 NVIDIA PATTERN:")
    print(json.dumps(PARADIGM_SHIFTS["NVIDIA"], indent=2))
    
    print("\n" + "=" * 70)
    print("🔥 IS IREN THE MINI-NVIDIA?")
    print("=" * 70)
    print(json.dumps(MINI_NVIDIA_THESIS, indent=2))
    
    print("\n" + "=" * 70)
    print("📈 RECOMMENDATION")
    print("=" * 70)
    print(json.dumps(get_paradigm_shift_summary(), indent=2))

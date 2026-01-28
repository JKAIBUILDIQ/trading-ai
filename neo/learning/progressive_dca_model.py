"""
PROGRESSIVE DCA MODEL
=====================
"Dip the toes to test the temperature"

Scaling Pattern: 1, 1, 2, 2, 4 (Total = 10 units)
- Start small, add as conviction builds
- CAP at 4 to avoid huge downswings
- Unit size weighted by bankroll

Philosophy:
- Don't all-in on first entry
- Test the water, then scale
- Cut off before overleveraging
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE DCA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DCA_PATTERN = [1, 1, 2, 2, 4]  # Total = 10 units
MAX_TRANCHES = 5
TOTAL_UNITS = sum(DCA_PATTERN)  # 10

@dataclass
class DCAConfig:
    """Configuration for progressive DCA"""
    pattern: List[int] = None
    max_tranches: int = 5
    bankroll: float = 100000
    allocation_percent: float = 20  # % of bankroll for this position
    
    def __post_init__(self):
        if self.pattern is None:
            self.pattern = [1, 1, 2, 2, 4]
    
    @property
    def total_units(self) -> int:
        return sum(self.pattern)
    
    @property
    def position_budget(self) -> float:
        """Total budget for this position"""
        return self.bankroll * (self.allocation_percent / 100)
    
    @property
    def unit_size(self) -> float:
        """Dollar value of 1 unit"""
        return self.position_budget / self.total_units
    
    def get_tranche_size(self, tranche_number: int) -> float:
        """Get dollar amount for specific tranche (1-indexed)"""
        if tranche_number < 1 or tranche_number > len(self.pattern):
            return 0
        return self.unit_size * self.pattern[tranche_number - 1]
    
    def get_tranche_units(self, tranche_number: int) -> int:
        """Get unit count for specific tranche"""
        if tranche_number < 1 or tranche_number > len(self.pattern):
            return 0
        return self.pattern[tranche_number - 1]


# ═══════════════════════════════════════════════════════════════════════════════
# THE PROGRESSIVE DCA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

PROGRESSIVE_DCA_MODEL = {
    "name": "Progressive DCA (1-1-2-2-4)",
    "philosophy": "Dip the toes to test the temperature",
    
    "pattern": {
        "tranche_1": {"units": 1, "percent": "10%", "purpose": "Test the water"},
        "tranche_2": {"units": 1, "percent": "10%", "purpose": "Confirm entry"},
        "tranche_3": {"units": 2, "percent": "20%", "purpose": "Build position"},
        "tranche_4": {"units": 2, "percent": "20%", "purpose": "Add on strength"},
        "tranche_5": {"units": 4, "percent": "40%", "purpose": "Max conviction - CAP HERE"},
    },
    
    "total_units": 10,
    
    "why_this_pattern": {
        "starts_small": "1 unit = minimal risk to test thesis",
        "confirms_before_adding": "2nd unit confirms direction",
        "scales_with_conviction": "Larger tranches as confidence builds",
        "caps_at_4": "Prevents overleveraging on 5th entry",
        "total_10_units": "Easy math for position sizing"
    },
    
    "entry_triggers": {
        "tranche_1": {
            "trigger": "Initial thesis validation",
            "examples": ["Technical breakout", "Positive news", "Support bounce"],
            "confidence_required": "50%+ (just testing)"
        },
        "tranche_2": {
            "trigger": "Entry confirmed, price holds",
            "examples": ["Didn't get stopped out", "Price above entry", "Volume confirms"],
            "confidence_required": "60%+"
        },
        "tranche_3": {
            "trigger": "Thesis strengthening OR pullback to support",
            "examples": ["5-10% dip", "New catalyst", "Sector rotation in"],
            "confidence_required": "70%+"
        },
        "tranche_4": {
            "trigger": "Strong conviction, adding on strength or dip",
            "examples": ["Breaking resistance", "10-15% dip", "Major contract"],
            "confidence_required": "75%+"
        },
        "tranche_5": {
            "trigger": "Maximum conviction - ONLY on best setups",
            "examples": ["Major thesis validation", "15%+ dip in uptrend", "All signals align"],
            "confidence_required": "80%+",
            "warning": "⚠️ CAP HERE - No more adding after this"
        }
    },
    
    "anti_patterns": {
        "dont_do": [
            "All 10 units on first entry",
            "Tranche 5 before testing with Tranche 1",
            "Adding (Tranche 6+) after cap",
            "Averaging down into a broken thesis"
        ],
        "the_cap_rule": "After Tranche 5, STOP. No more adding regardless of price."
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# BANKROLL-WEIGHTED UNIT SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_unit_sizes(bankroll: float, allocation_pct: float = 20) -> Dict:
    """
    Calculate unit sizes based on bankroll.
    
    Args:
        bankroll: Total portfolio value
        allocation_pct: % of bankroll for this position (default 20%)
    
    Example: $100K bankroll, 20% allocation = $20K position budget
             $20K / 10 units = $2K per unit
    """
    position_budget = bankroll * (allocation_pct / 100)
    unit_size = position_budget / TOTAL_UNITS
    
    tranches = []
    cumulative = 0
    for i, units in enumerate(DCA_PATTERN, 1):
        tranche_amount = unit_size * units
        cumulative += tranche_amount
        tranches.append({
            "tranche": i,
            "units": units,
            "amount": round(tranche_amount, 2),
            "cumulative": round(cumulative, 2),
            "percent_of_position": f"{(units/TOTAL_UNITS)*100:.0f}%",
            "percent_of_bankroll": f"{(tranche_amount/bankroll)*100:.1f}%"
        })
    
    return {
        "bankroll": f"${bankroll:,.0f}",
        "allocation_percent": f"{allocation_pct}%",
        "position_budget": f"${position_budget:,.0f}",
        "unit_size": f"${unit_size:,.0f}",
        "pattern": "1-1-2-2-4",
        "total_units": TOTAL_UNITS,
        "tranches": tranches,
        "max_position_risk": f"${position_budget:,.0f} ({allocation_pct}% of bankroll)"
    }


def get_recommended_tranche(
    current_tranches: int,
    current_pnl_pct: float,
    thesis_confidence: int,
    is_dip: bool = False,
    dip_pct: float = 0
) -> Dict:
    """
    Determine if should add next tranche.
    
    Args:
        current_tranches: How many tranches already deployed (0-5)
        current_pnl_pct: Current P&L percentage
        thesis_confidence: 0-100 confidence in thesis
        is_dip: Is price currently in a dip?
        dip_pct: If dip, how much % down from recent high
    """
    if current_tranches >= 5:
        return {
            "action": "HOLD",
            "reason": "⚠️ CAP REACHED - No more adding",
            "add_tranche": False,
            "next_tranche": None
        }
    
    next_tranche = current_tranches + 1
    required_confidence = [50, 60, 70, 75, 80][next_tranche - 1]
    
    # Decision logic
    should_add = False
    reason = ""
    
    if current_tranches == 0:
        # First entry - just need basic thesis
        if thesis_confidence >= 50:
            should_add = True
            reason = "Test the water - thesis looks promising"
        else:
            reason = f"Wait - confidence {thesis_confidence}% < 50% required"
    
    elif current_tranches == 1:
        # Second entry - confirm first wasn't stopped
        if current_pnl_pct >= -5 and thesis_confidence >= 60:
            should_add = True
            reason = "Entry confirmed - price holding, add confirmation tranche"
        elif is_dip and dip_pct >= 5 and thesis_confidence >= 60:
            should_add = True
            reason = f"Buying the dip ({dip_pct}% down) - lower average"
        else:
            reason = "Wait for confirmation or better entry"
    
    elif current_tranches == 2:
        # Third entry - need stronger signal
        if is_dip and dip_pct >= 5 and thesis_confidence >= 70:
            should_add = True
            reason = f"Adding on {dip_pct}% dip - building position"
        elif current_pnl_pct >= 10 and thesis_confidence >= 70:
            should_add = True
            reason = "Adding on strength - thesis playing out"
        else:
            reason = "Wait for dip or strength confirmation"
    
    elif current_tranches == 3:
        # Fourth entry - high conviction only
        if is_dip and dip_pct >= 10 and thesis_confidence >= 75:
            should_add = True
            reason = f"Significant dip ({dip_pct}%) - high conviction add"
        elif thesis_confidence >= 80:
            should_add = True
            reason = "Major catalyst - high conviction add"
        else:
            reason = "Wait for significant dip or major catalyst"
    
    elif current_tranches == 4:
        # Fifth entry - MAXIMUM CONVICTION ONLY
        if is_dip and dip_pct >= 15 and thesis_confidence >= 80:
            should_add = True
            reason = f"🎯 Major dip ({dip_pct}%) + high conviction - MAX position"
        elif thesis_confidence >= 85:
            should_add = True
            reason = "🎯 Maximum conviction - completing position"
        else:
            reason = "⚠️ Final tranche - only on exceptional setups"
    
    return {
        "action": "ADD" if should_add else "WAIT",
        "add_tranche": should_add,
        "next_tranche": next_tranche if should_add else None,
        "units_to_add": DCA_PATTERN[next_tranche - 1] if should_add else 0,
        "reason": reason,
        "confidence_required": required_confidence,
        "current_confidence": thesis_confidence,
        "tranches_deployed": current_tranches,
        "tranches_remaining": 5 - current_tranches,
        "warning": "⚠️ CAP at 5 tranches - no more adding after this" if next_tranche == 5 and should_add else None
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressiveDCATracker:
    """Track progressive DCA positions"""
    
    def __init__(self, symbol: str, bankroll: float, allocation_pct: float = 20):
        self.symbol = symbol
        self.bankroll = bankroll
        self.allocation_pct = allocation_pct
        self.config = DCAConfig(bankroll=bankroll, allocation_percent=allocation_pct)
        
        self.tranches = []
        self.total_invested = 0
        self.total_units = 0
        self.avg_entry = 0
    
    def add_tranche(self, tranche_number: int, price: float, timestamp: str = None) -> Dict:
        """Record a new tranche entry"""
        if tranche_number > 5:
            return {"error": "CAP REACHED - Cannot add tranche > 5"}
        
        if len(self.tranches) >= 5:
            return {"error": "Already at maximum 5 tranches"}
        
        units = self.config.get_tranche_units(tranche_number)
        amount = self.config.get_tranche_size(tranche_number)
        
        tranche_record = {
            "tranche": tranche_number,
            "units": units,
            "amount": amount,
            "price": price,
            "timestamp": timestamp or datetime.utcnow().isoformat()
        }
        
        self.tranches.append(tranche_record)
        self.total_invested += amount
        self.total_units += units
        
        # Calculate new average entry
        weighted_sum = sum(t["amount"] for t in self.tranches)
        shares_equivalent = sum(t["amount"] / t["price"] for t in self.tranches)
        self.avg_entry = weighted_sum / shares_equivalent if shares_equivalent > 0 else price
        
        return {
            "status": "success",
            "tranche_added": tranche_record,
            "total_invested": self.total_invested,
            "total_units": self.total_units,
            "avg_entry": round(self.avg_entry, 2),
            "tranches_remaining": 5 - len(self.tranches),
            "at_cap": len(self.tranches) >= 5
        }
    
    def get_status(self, current_price: float) -> Dict:
        """Get current position status"""
        if not self.tranches:
            return {
                "symbol": self.symbol,
                "status": "NO POSITION",
                "tranches_deployed": 0,
                "next_action": "Consider Tranche 1 (test the water)"
            }
        
        pnl_dollars = (current_price - self.avg_entry) * (self.total_invested / self.avg_entry)
        pnl_pct = ((current_price - self.avg_entry) / self.avg_entry) * 100
        
        return {
            "symbol": self.symbol,
            "tranches_deployed": len(self.tranches),
            "tranches_remaining": 5 - len(self.tranches),
            "at_cap": len(self.tranches) >= 5,
            "total_invested": f"${self.total_invested:,.0f}",
            "total_units": self.total_units,
            "avg_entry": f"${self.avg_entry:.2f}",
            "current_price": f"${current_price:.2f}",
            "pnl_dollars": f"${pnl_dollars:,.0f}",
            "pnl_percent": f"{pnl_pct:+.1f}%",
            "tranche_history": self.tranches
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def example_iren_dca(bankroll: float = 100000) -> Dict:
    """Example IREN progressive DCA scenario"""
    
    sizes = calculate_unit_sizes(bankroll, allocation_pct=20)
    
    return {
        "scenario": "IREN Progressive DCA",
        "bankroll": f"${bankroll:,.0f}",
        "allocation": "20% ($20,000)",
        "unit_size": sizes["unit_size"],
        
        "execution_plan": [
            {
                "tranche": 1,
                "units": 1,
                "amount": "$2,000",
                "trigger": "Initial entry at $60 - testing the water",
                "confidence": "50%+",
                "what_youre_risking": "2% of bankroll"
            },
            {
                "tranche": 2,
                "units": 1,
                "amount": "$2,000",
                "trigger": "Price holds above $58, add confirmation",
                "confidence": "60%+",
                "cumulative": "$4,000 (4% of bankroll)"
            },
            {
                "tranche": 3,
                "units": 2,
                "amount": "$4,000",
                "trigger": "5-10% dip to $54-57 OR new contract news",
                "confidence": "70%+",
                "cumulative": "$8,000 (8% of bankroll)"
            },
            {
                "tranche": 4,
                "units": 2,
                "amount": "$4,000",
                "trigger": "10-15% dip OR major catalyst",
                "confidence": "75%+",
                "cumulative": "$12,000 (12% of bankroll)"
            },
            {
                "tranche": 5,
                "units": 4,
                "amount": "$8,000",
                "trigger": "15%+ dip with thesis intact OR max conviction",
                "confidence": "80%+",
                "cumulative": "$20,000 (20% of bankroll)",
                "note": "⚠️ CAP - No more adding after this"
            }
        ],
        
        "key_rules": [
            "Start with Tranche 1 to TEST - don't skip to Tranche 5",
            "Each tranche requires higher confidence",
            "CAP at Tranche 5 - no exceptions",
            "If thesis breaks at any point - STOP adding, consider exit"
        ]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PROGRESSIVE DCA MODEL: 1-1-2-2-4")
    print("'Dip the toes to test the temperature'")
    print("=" * 70)
    
    # Show unit sizes for different bankrolls
    for bankroll in [50000, 100000, 250000]:
        print(f"\n📊 BANKROLL: ${bankroll:,}")
        sizes = calculate_unit_sizes(bankroll, 20)
        print(f"   Position Budget: {sizes['position_budget']}")
        print(f"   Unit Size: {sizes['unit_size']}")
        print("   Tranches:")
        for t in sizes['tranches']:
            print(f"      #{t['tranche']}: {t['units']} units = ${t['amount']:,.0f} ({t['percent_of_bankroll']} of bankroll)")
    
    print("\n" + "=" * 70)
    print("📈 EXAMPLE: IREN DCA PLAN ($100K BANKROLL)")
    print("=" * 70)
    plan = example_iren_dca(100000)
    print(json.dumps(plan, indent=2))
    
    print("\n" + "=" * 70)
    print("🎯 TRANCHE RECOMMENDATION LOGIC")
    print("=" * 70)
    
    # Example scenarios
    scenarios = [
        (0, 0, 55, False, 0, "New position, thesis looks good"),
        (1, 2, 65, False, 0, "First tranche profitable"),
        (2, -5, 70, True, 8, "Position down, but buying dip"),
        (3, 5, 75, True, 12, "Adding on significant dip"),
        (4, 15, 85, False, 0, "Max conviction reached"),
        (5, 20, 90, True, 15, "At cap - can't add more"),
    ]
    
    for tranches, pnl, conf, is_dip, dip_pct, desc in scenarios:
        rec = get_recommended_tranche(tranches, pnl, conf, is_dip, dip_pct)
        print(f"\n{desc}:")
        print(f"   Tranches: {tranches}, PnL: {pnl}%, Confidence: {conf}%")
        print(f"   → {rec['action']}: {rec['reason']}")

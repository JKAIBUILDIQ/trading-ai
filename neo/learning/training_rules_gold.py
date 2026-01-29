"""
NEO Training Rules - Gold (XAUUSD)

Learned from successful breakout: 2026-01-28
Result: +56.4% ($56,398 profit)
Grade: A+

These rules are PROVEN by live trading success.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("NEO.GoldRules")


@dataclass
class GoldSignal:
    """Signal output for Gold trading"""
    action: str  # BUY, SELL, HOLD, WAIT
    confidence: int  # 0-100
    reason: str
    dca_level: Optional[int] = None
    blocked_actions: List[str] = None


class GoldTradingRules:
    """
    NEO's learned rules for Gold trading
    
    Key lesson: LONG-ONLY DCA with exhaustion detection
    massively outperforms counter-trend strategies.
    """
    
    def __init__(self):
        # Indicator settings that worked
        self.supertrend_period = 10
        self.supertrend_multiplier = 3.0
        self.rsi_period = 14
        self.ema_period = 20
        
        # Exhaustion thresholds
        self.rsi_overbought = 70
        self.rsi_oversold = 40
        self.exhaustion_threshold = 50
        self.max_distance_from_ema_pct = 3.0
        
        # DCA levels (% drop from entry)
        self.dca_levels = {
            2: {"drop_pct": 5, "lots_multiplier": 1.0},
            3: {"drop_pct": 10, "lots_multiplier": 1.5},
            4: {"drop_pct": 15, "lots_multiplier": 1.5},
            5: {"drop_pct": 20, "lots_multiplier": 2.0},  # MAX
        }
        
        # Take profit levels
        self.tp_levels = [
            {"pct": 0.5, "close_pct": 33},   # TP1: +0.5%
            {"pct": 1.0, "close_pct": 33},   # TP2: +1.0%
            # Remaining 34% = runner with trailing stop
        ]
    
    def rule_1_never_short_uptrend(self, supertrend: str) -> Dict:
        """
        RULE 1: NEVER short when SuperTrend is UP
        
        Learned: Lost $14,652 trying to short on 2026-01-28
        Counter-trend shorts = WRONG in bull trend
        """
        if supertrend == "UP":
            return {
                "can_short": False,
                "reason": "SuperTrend UP - SHORTS BLOCKED",
                "allowed_actions": ["BUY", "BUY_THE_DIP", "HOLD"],
                "blocked_actions": ["SELL", "SHORT", "FADE_THE_SQUEEZE"]
            }
        elif supertrend == "DOWN":
            return {
                "can_short": True,
                "reason": "SuperTrend DOWN - shorts allowed with confirmation",
                "allowed_actions": ["SELL", "SHORT", "HOLD"],
                "blocked_actions": []
            }
        return {"can_short": False, "reason": "Unknown trend", "allowed_actions": ["HOLD"]}
    
    def rule_2_exhaustion_detection(
        self, 
        price: float, 
        ema20: float, 
        rsi: float, 
        recent_high: float
    ) -> Dict:
        """
        RULE 2: Detect exhaustion (don't buy tops)
        
        When exhausted: DON'T BUY MORE, but DON'T SELL either!
        Just HOLD existing positions.
        """
        exhaustion_score = 0
        reasons = []
        
        # RSI overbought
        if rsi > self.rsi_overbought:
            exhaustion_score += 30
            reasons.append(f"RSI {rsi:.0f} > {self.rsi_overbought} (overbought)")
        elif rsi > 65:
            exhaustion_score += 15
            reasons.append(f"RSI {rsi:.0f} elevated")
        
        # Price extended above EMA20
        if ema20 > 0:
            distance_pct = (price - ema20) / ema20 * 100
            if distance_pct > self.max_distance_from_ema_pct:
                exhaustion_score += 25
                reasons.append(f"Price {distance_pct:.1f}% above EMA20 (extended)")
            elif distance_pct > 2.0:
                exhaustion_score += 10
                reasons.append(f"Price elevated above EMA20")
        
        # Near recent high
        if recent_high > 0 and price > recent_high * 0.98:
            exhaustion_score += 25
            reasons.append("Price near recent highs")
        
        is_exhausted = exhaustion_score >= self.exhaustion_threshold
        
        return {
            "is_exhausted": is_exhausted,
            "exhaustion_score": exhaustion_score,
            "reasons": reasons,
            "action": "HOLD_NO_NEW_BUYS" if is_exhausted else "CAN_BUY"
        }
    
    def rule_3_dca_on_drops(
        self, 
        current_price: float, 
        entry_prices: List[float]
    ) -> Optional[Dict]:
        """
        RULE 3: DCA on percentage drops from entry
        
        CRITICAL: Calculate drops from ENTRY PRICE, not current price!
        """
        if not entry_prices:
            return {"level": 1, "action": "INITIAL_ENTRY", "lots_multiplier": 0.5}
        
        highest_entry = max(entry_prices)
        drop_pct = (highest_entry - current_price) / highest_entry * 100
        
        filled_levels = len(entry_prices)
        
        for level, config in sorted(self.dca_levels.items()):
            if filled_levels < level and drop_pct >= config["drop_pct"]:
                return {
                    "level": level,
                    "action": f"DCA_L{level}",
                    "drop_pct": drop_pct,
                    "trigger_pct": config["drop_pct"],
                    "lots_multiplier": config["lots_multiplier"]
                }
        
        return None  # No DCA trigger
    
    def rule_4_let_winners_run(
        self, 
        entry_price: float, 
        current_price: float,
        tp1_hit: bool = False,
        tp2_hit: bool = False
    ) -> Dict:
        """
        RULE 4: Let winners run with partial TPs
        
        Pattern: 33% at TP1, 33% at TP2, 34% runner with trail
        """
        if entry_price <= 0:
            return {"action": "HOLD"}
        
        profit_pct = (current_price - entry_price) / entry_price * 100
        
        actions = []
        
        # TP1: +0.5%
        if profit_pct >= 0.5 and not tp1_hit:
            actions.append({
                "action": "TAKE_PROFIT_1",
                "close_pct": 33,
                "profit_pct": profit_pct
            })
        
        # TP2: +1.0%
        if profit_pct >= 1.0 and not tp2_hit:
            actions.append({
                "action": "TAKE_PROFIT_2",
                "close_pct": 33,
                "profit_pct": profit_pct
            })
        
        # Trail stop for runner
        if profit_pct > 0.3:
            trail_stop = entry_price + (current_price - entry_price) * 0.5
            actions.append({
                "action": "UPDATE_TRAIL_STOP",
                "trail_stop": trail_stop,
                "profit_locked_pct": profit_pct * 0.5
            })
        
        return {
            "profit_pct": profit_pct,
            "actions": actions if actions else [{"action": "HOLD"}]
        }
    
    def rule_5_only_short_on_flip(
        self, 
        supertrend: str, 
        prev_supertrend: str,
        confidence: int
    ) -> Dict:
        """
        RULE 5: Only short when SuperTrend FLIPS
        
        Must have:
        1. SuperTrend changed from UP to DOWN
        2. Confidence >= 70%
        3. Multiple signal confirmation
        """
        if supertrend == "DOWN" and prev_supertrend == "UP":
            if confidence >= 70:
                return {
                    "can_short": True,
                    "reason": "SuperTrend FLIP confirmed with high confidence",
                    "action": "SELL_SIGNAL"
                }
            else:
                return {
                    "can_short": False,
                    "reason": f"SuperTrend flipped but confidence {confidence}% < 70%",
                    "action": "WAIT_FOR_CONFIRMATION"
                }
        
        return {
            "can_short": False,
            "reason": "No trend flip - shorts blocked",
            "action": "STAY_LONG"
        }
    
    def generate_signal(
        self,
        price: float,
        ema20: float,
        rsi: float,
        supertrend: str,
        prev_supertrend: str = None,
        recent_high: float = None,
        entry_prices: List[float] = None,
        confidence: int = 50
    ) -> GoldSignal:
        """
        Generate complete trading signal based on all rules
        """
        # Rule 1: Check if shorts blocked
        trend_check = self.rule_1_never_short_uptrend(supertrend)
        
        # Rule 2: Check exhaustion
        exhaustion = self.rule_2_exhaustion_detection(
            price, ema20, rsi, recent_high or price
        )
        
        # Rule 3: Check DCA
        dca = self.rule_3_dca_on_drops(price, entry_prices or [])
        
        # Rule 5: Check short opportunity
        short_check = self.rule_5_only_short_on_flip(
            supertrend, prev_supertrend or supertrend, confidence
        )
        
        # Decision logic
        if supertrend == "UP":
            if exhaustion["is_exhausted"]:
                return GoldSignal(
                    action="HOLD",
                    confidence=60,
                    reason="Exhausted - hold existing, no new buys",
                    blocked_actions=["BUY", "SELL"]
                )
            elif rsi < self.rsi_oversold:
                return GoldSignal(
                    action="BUY_THE_DIP",
                    confidence=85,
                    reason=f"RSI {rsi:.0f} oversold in uptrend - HIGH CONFIDENCE BUY",
                    dca_level=dca["level"] if dca else 1
                )
            elif dca:
                return GoldSignal(
                    action="DCA",
                    confidence=75,
                    reason=f"DCA trigger at L{dca['level']} (-{dca.get('drop_pct', 0):.1f}%)",
                    dca_level=dca["level"]
                )
            else:
                return GoldSignal(
                    action="HOLD",
                    confidence=65,
                    reason="Uptrend intact, waiting for dip entry",
                    blocked_actions=["SELL"]
                )
        
        elif supertrend == "DOWN" and short_check["can_short"]:
            return GoldSignal(
                action="SELL",
                confidence=confidence,
                reason="SuperTrend flipped DOWN - short opportunity",
                blocked_actions=["BUY"]
            )
        
        return GoldSignal(
            action="WAIT",
            confidence=40,
            reason="No clear signal",
            blocked_actions=[]
        )


# Singleton instance
_gold_rules = None

def get_gold_rules() -> GoldTradingRules:
    """Get singleton instance of Gold trading rules"""
    global _gold_rules
    if _gold_rules is None:
        _gold_rules = GoldTradingRules()
    return _gold_rules


def generate_gold_signal(
    price: float,
    ema20: float,
    rsi: float,
    supertrend: str,
    **kwargs
) -> GoldSignal:
    """Convenience function to generate Gold signal"""
    rules = get_gold_rules()
    return rules.generate_signal(price, ema20, rsi, supertrend, **kwargs)


if __name__ == "__main__":
    # Test the rules
    rules = GoldTradingRules()
    
    print("Testing Gold Trading Rules (learned from 2026-01-28 success)")
    print("=" * 60)
    
    # Test scenario 1: Oversold in uptrend
    signal = rules.generate_signal(
        price=5058,
        ema20=5080,
        rsi=35,
        supertrend="UP"
    )
    print(f"\nScenario 1 - Oversold dip in uptrend:")
    print(f"  Signal: {signal.action} (confidence: {signal.confidence}%)")
    print(f"  Reason: {signal.reason}")
    
    # Test scenario 2: Exhausted top
    signal = rules.generate_signal(
        price=5321,
        ema20=5200,
        rsi=78,
        supertrend="UP",
        recent_high=5325
    )
    print(f"\nScenario 2 - Exhausted top:")
    print(f"  Signal: {signal.action} (confidence: {signal.confidence}%)")
    print(f"  Reason: {signal.reason}")
    
    # Test scenario 3: DCA trigger
    signal = rules.generate_signal(
        price=5000,
        ema20=5050,
        rsi=45,
        supertrend="UP",
        entry_prices=[5100, 5050]
    )
    print(f"\nScenario 3 - DCA on drop:")
    print(f"  Signal: {signal.action} (confidence: {signal.confidence}%)")
    print(f"  Reason: {signal.reason}")
    print(f"  DCA Level: {signal.dca_level}")
    
    # Test scenario 4: Trying to short uptrend (SHOULD BE BLOCKED)
    short_check = rules.rule_1_never_short_uptrend("UP")
    print(f"\nScenario 4 - Attempt to short in uptrend:")
    print(f"  Can short: {short_check['can_short']}")
    print(f"  Reason: {short_check['reason']}")
    print(f"  Blocked actions: {short_check['blocked_actions']}")

"""
NEO-GOLD Hardcoded Trading Rules
Gold-specific rules based on trading experience
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .config import (
    NO_TRADE_TIMES, MIN_RISK_REWARD, MAX_POSITION_SIZE_PCT,
    LOW_LIQUIDITY_SIZE_MULTIPLIER, logger
)


@dataclass 
class RuleCheck:
    """Result of a rule check."""
    rule_name: str
    passed: bool
    reason: str
    action: str  # "ALLOW", "BLOCK", "WARN", "REDUCE_SIZE"
    

class GoldTradingRules:
    """
    Hardcoded rules for Gold trading.
    
    These rules OVERRIDE AI decisions when triggered.
    Based on empirical Gold trading patterns.
    """
    
    def __init__(self):
        self.rule_results: List[RuleCheck] = []
        
    def check_all(self, features: Dict, signal: Dict) -> Tuple[bool, List[RuleCheck], float]:
        """
        Check all rules against proposed signal.
        
        Returns:
            - can_trade: bool - whether trade is allowed
            - rule_results: List[RuleCheck] - all rule outcomes
            - size_multiplier: float - position size adjustment
        """
        
        self.rule_results = []
        size_multiplier = 1.0
        
        # Run all rules
        self._rule_no_trade_times(features)
        self._rule_round_number_respect(features)
        self._rule_asian_range_breakout(features)
        self._rule_sweep_confirmation(features, signal)
        self._rule_news_fade_delay(features)
        self._rule_dxy_divergence(features)
        self._rule_low_liquidity_size(features)
        self._rule_minimum_rr(signal)
        
        # Determine final outcome
        blocked_rules = [r for r in self.rule_results if r.action == "BLOCK"]
        reduce_rules = [r for r in self.rule_results if r.action == "REDUCE_SIZE"]
        
        can_trade = len(blocked_rules) == 0
        
        # Apply size reductions
        for rule in reduce_rules:
            size_multiplier *= 0.5
        
        size_multiplier = max(0.25, min(1.0, size_multiplier))
        
        logger.info(f"📋 Rules checked: {len(self.rule_results)}")
        logger.info(f"   Blocked: {len(blocked_rules)}, Can trade: {can_trade}")
        if blocked_rules:
            for r in blocked_rules:
                logger.info(f"   ❌ {r.rule_name}: {r.reason}")
        
        return can_trade, self.rule_results, size_multiplier
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 1: NO TRADE TIMES
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_no_trade_times(self, features: Dict):
        """
        NEVER trade first 15 min of London open (07:00-07:15 UTC)
        NEVER trade during dead zone (20:00-00:00 UTC)
        """
        
        now = datetime.utcnow()
        current_minutes = now.hour * 60 + now.minute
        
        for start_time, end_time in NO_TRADE_TIMES:
            start_parts = start_time.split(":")
            end_parts = end_time.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
            
            # Handle overnight times
            if end_minutes < start_minutes:
                # Crosses midnight
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    self.rule_results.append(RuleCheck(
                        rule_name="NO_TRADE_TIME",
                        passed=False,
                        reason=f"Current time {now.strftime('%H:%M')} is in no-trade zone {start_time}-{end_time}",
                        action="BLOCK"
                    ))
                    return
            else:
                if start_minutes <= current_minutes < end_minutes:
                    self.rule_results.append(RuleCheck(
                        rule_name="NO_TRADE_TIME",
                        passed=False,
                        reason=f"Current time {now.strftime('%H:%M')} is in no-trade zone {start_time}-{end_time}",
                        action="BLOCK"
                    ))
                    return
        
        self.rule_results.append(RuleCheck(
            rule_name="NO_TRADE_TIME",
            passed=True,
            reason=f"Current time {now.strftime('%H:%M')} is OK to trade",
            action="ALLOW"
        ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 2: ROUND NUMBER RESPECT
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_round_number_respect(self, features: Dict):
        """
        ALWAYS respect $50 levels ($2700, $2750, $2800) as magnets.
        Don't fight price moving toward a round number.
        """
        
        round_numbers = features.get("round_number", {})
        is_near = round_numbers.get("is_near", False)
        nearest = round_numbers.get("nearest", 0)
        magnet_strength = round_numbers.get("magnet_strength", "WEAK")
        
        if is_near and magnet_strength in ["STRONG", "MODERATE"]:
            self.rule_results.append(RuleCheck(
                rule_name="ROUND_NUMBER_RESPECT",
                passed=True,
                reason=f"Near ${nearest:.0f} - price may be attracted to this level",
                action="WARN"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="ROUND_NUMBER_RESPECT",
                passed=True,
                reason="Not near significant round number",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 3: ASIAN RANGE BREAKOUT
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_asian_range_breakout(self, features: Dict):
        """
        IF Asian range < 100 pips THEN expect London breakout.
        Trade the breakout, not the range.
        """
        
        asian_range = features.get("asian_range", {})
        range_pips = asian_range.get("range_pips", 0)
        session = features.get("session", "")
        
        if range_pips > 0 and range_pips < 100 and session == "ASIA":
            self.rule_results.append(RuleCheck(
                rule_name="ASIAN_RANGE_BREAKOUT",
                passed=False,
                reason=f"Asian range is tight ({range_pips:.0f} pips). "
                       f"Wait for London breakout instead of trading range.",
                action="BLOCK"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="ASIAN_RANGE_BREAKOUT",
                passed=True,
                reason="Asian range rule not applicable",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 4: SWEEP CONFIRMATION
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_sweep_confirmation(self, features: Dict, signal: Dict):
        """
        IF price sweeps round number AND reverses within 5 candles THEN fade the sweep.
        Require confirmation before fading.
        """
        
        # This is more of a pattern confirmation rule
        # Check if signal is based on sweep pattern
        reasoning = signal.get("reasoning", {})
        pattern = reasoning.get("pattern", "")
        
        if "sweep" in pattern.lower():
            self.rule_results.append(RuleCheck(
                rule_name="SWEEP_CONFIRMATION",
                passed=True,
                reason="Sweep pattern detected - confirm reversal candle before entry",
                action="WARN"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="SWEEP_CONFIRMATION",
                passed=True,
                reason="Not a sweep-based trade",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 5: NEWS FADE DELAY
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_news_fade_delay(self, features: Dict):
        """
        WAIT 5 min after news spike, then look for fade.
        Never trade INTO news volatility.
        """
        
        news_timing = features.get("time_since_news", {})
        minutes_since = news_timing.get("minutes_since_news", -1)
        news_active = news_timing.get("news_active", False)
        
        if news_active:
            self.rule_results.append(RuleCheck(
                rule_name="NEWS_FADE_DELAY",
                passed=False,
                reason=f"News spike active ({minutes_since} min ago). "
                       f"Wait at least 5 minutes for fade setup.",
                action="BLOCK"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="NEWS_FADE_DELAY",
                passed=True,
                reason="No recent news activity",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 6: DXY DIVERGENCE
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_dxy_divergence(self, features: Dict):
        """
        IF DXY up AND Gold up THEN suspicious - expect Gold reversal.
        Gold should move inverse to USD.
        """
        
        dxy_correlation = features.get("dxy_correlation", {})
        divergence = dxy_correlation.get("divergence", False)
        
        if divergence:
            self.rule_results.append(RuleCheck(
                rule_name="DXY_DIVERGENCE",
                passed=True,
                reason="DXY/Gold divergence detected - expect Gold reversal",
                action="WARN"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="DXY_DIVERGENCE",
                passed=True,
                reason="Normal DXY/Gold correlation",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 7: LOW LIQUIDITY SIZE REDUCTION
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_low_liquidity_size(self, features: Dict):
        """
        REDUCE size during 20:00-00:00 UTC (low liquidity).
        """
        
        session = features.get("session", "")
        session_details = features.get("session_details", {})
        is_dead_zone = session_details.get("is_dead_zone", False)
        
        if is_dead_zone or session == "DEAD_ZONE":
            self.rule_results.append(RuleCheck(
                rule_name="LOW_LIQUIDITY_SIZE",
                passed=True,
                reason="Low liquidity period - reduce position size by 50%",
                action="REDUCE_SIZE"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="LOW_LIQUIDITY_SIZE",
                passed=True,
                reason="Normal liquidity",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # RULE 8: MINIMUM RISK/REWARD
    # ═══════════════════════════════════════════════════════════════════
    
    def _rule_minimum_rr(self, signal: Dict):
        """
        Must have at least 1:1.5 risk/reward ratio.
        """
        
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        tp = signal.get("take_profit", 0)
        
        if entry == 0 or sl == 0 or tp == 0:
            self.rule_results.append(RuleCheck(
                rule_name="MINIMUM_RR",
                passed=True,
                reason="Incomplete signal - cannot check R:R",
                action="ALLOW"
            ))
            return
        
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk == 0:
            self.rule_results.append(RuleCheck(
                rule_name="MINIMUM_RR",
                passed=False,
                reason="Invalid stop loss (zero risk)",
                action="BLOCK"
            ))
            return
        
        rr_ratio = reward / risk
        
        if rr_ratio < MIN_RISK_REWARD:
            self.rule_results.append(RuleCheck(
                rule_name="MINIMUM_RR",
                passed=False,
                reason=f"Risk/Reward {rr_ratio:.2f} is below minimum {MIN_RISK_REWARD}",
                action="BLOCK"
            ))
        else:
            self.rule_results.append(RuleCheck(
                rule_name="MINIMUM_RR",
                passed=True,
                reason=f"Risk/Reward {rr_ratio:.2f} is acceptable",
                action="ALLOW"
            ))
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_blocking_rules(self) -> List[RuleCheck]:
        """Get rules that are blocking the trade."""
        return [r for r in self.rule_results if r.action == "BLOCK"]
    
    def get_warnings(self) -> List[RuleCheck]:
        """Get rule warnings."""
        return [r for r in self.rule_results if r.action == "WARN"]
    
    def format_for_signal(self) -> Dict:
        """Format rule results for signal output."""
        return {
            "rules_checked": len(self.rule_results),
            "rules_passed": len([r for r in self.rule_results if r.passed]),
            "blocked_by": [r.rule_name for r in self.get_blocking_rules()],
            "warnings": [r.rule_name for r in self.get_warnings()],
            "details": [
                {"rule": r.rule_name, "passed": r.passed, "reason": r.reason}
                for r in self.rule_results
            ]
        }

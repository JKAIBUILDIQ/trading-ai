#!/usr/bin/env python3
"""
GEOPOLITICAL RESOLUTION RISK INDICATOR
═══════════════════════════════════════════════════════════════════════

Measures risk of geopolitical tensions EASING (which is bearish for Gold).

Gold's current rally is largely driven by:
- BRICS de-dollarization narrative
- US-China tensions
- Middle East conflict
- Ukraine war
- Central bank diversification

When these tensions RESOLVE, Gold loses its safe-haven bid.

Score 0-100 (Resolution Risk):
- 0-30:  LOW - Tensions remain high (bullish Gold)
- 30-50: MODERATE - Some easing possible
- 50-70: HIGH - Resolution signals emerging
- 70-100: VERY HIGH - Major de-escalation likely (bearish Gold)
"""

import logging
from datetime import datetime
from typing import Dict, List
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeopoliticalIndicator")


class GeopoliticalResolutionIndicator:
    """
    Monitors geopolitical tensions that drive Gold demand.
    RISING tensions = Bullish Gold
    FALLING tensions (resolution) = Bearish Gold
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Key geopolitical factors driving Gold
        self.factors = {
            'brics_dedollarization': {
                'weight': 0.25,
                'current_status': 'ACTIVE',  # Ongoing narrative
                'resolution_risk': 30  # Low risk of resolution
            },
            'us_china_tensions': {
                'weight': 0.20,
                'current_status': 'ELEVATED',
                'resolution_risk': 25  # Tensions unlikely to resolve soon
            },
            'middle_east_conflict': {
                'weight': 0.20,
                'current_status': 'HOT',
                'resolution_risk': 40  # Ceasefire talks ongoing
            },
            'ukraine_war': {
                'weight': 0.15,
                'current_status': 'ONGOING',
                'resolution_risk': 35  # Peace talks mentioned
            },
            'central_bank_buying': {
                'weight': 0.10,
                'current_status': 'STRONG',
                'resolution_risk': 20  # Structural, won't change quickly
            },
            'us_fiscal_concerns': {
                'weight': 0.10,
                'current_status': 'ELEVATED',
                'resolution_risk': 30  # Debt ceiling, deficit concerns
            }
        }
        
        logger.info("🌍 Geopolitical Resolution Indicator initialized")
    
    def calculate(self) -> Dict:
        """
        Calculate overall geopolitical resolution risk.
        """
        logger.info("🌍 Calculating geopolitical resolution risk...")
        
        # Calculate weighted resolution risk
        total_risk = 0
        factor_details = {}
        
        for factor, data in self.factors.items():
            risk = data['resolution_risk']
            weight = data['weight']
            total_risk += risk * weight
            
            factor_details[factor] = {
                'status': data['current_status'],
                'resolution_risk': int(risk),
                'weight': float(weight),
                'impact': self._get_impact_level(risk)
            }
        
        total_risk = float(round(total_risk, 1))
        
        # Get recent events that could change risk
        recent_events = self._get_recent_events()
        
        # Calculate Gold impact
        gold_impact = self._calc_gold_impact(total_risk)
        
        result = {
            'resolution_risk': total_risk,
            'level': self._get_level(total_risk),
            'timestamp': datetime.utcnow().isoformat(),
            
            'factors': factor_details,
            'recent_events': recent_events,
            
            # Gold implications
            'gold_sentiment': gold_impact['sentiment'],
            'gold_impact': gold_impact['impact'],
            
            # For Ghost Commander
            'is_high_resolution_risk': bool(total_risk >= 50),
            'reduce_safe_haven_longs': bool(total_risk >= 60),
            'suggested_action': self._get_suggested_action(total_risk)
        }
        
        logger.info(f"   Resolution Risk: {total_risk} ({result['level']})")
        return result
    
    def _get_level(self, score: float) -> str:
        if score >= 70:
            return 'VERY_HIGH'
        elif score >= 50:
            return 'HIGH'
        elif score >= 30:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _get_impact_level(self, risk: float) -> str:
        if risk >= 60:
            return 'BEARISH_GOLD'
        elif risk >= 40:
            return 'NEUTRAL'
        else:
            return 'BULLISH_GOLD'
    
    def _get_recent_events(self) -> List[Dict]:
        """
        Return recent geopolitical events affecting Gold.
        In production, would scrape news APIs.
        """
        return [
            {
                'date': '2026-01-22',
                'event': 'BRICS summit scheduled for March',
                'impact': 'Maintains de-dollarization narrative',
                'gold_effect': 'BULLISH'
            },
            {
                'date': '2026-01-20',
                'event': 'Middle East ceasefire talks resume',
                'impact': 'Possible tension reduction',
                'gold_effect': 'BEARISH_IF_SUCCESS'
            },
            {
                'date': '2026-01-18',
                'event': 'China central bank adds to Gold reserves',
                'impact': 'Continued structural demand',
                'gold_effect': 'BULLISH'
            }
        ]
    
    def _calc_gold_impact(self, resolution_risk: float) -> Dict:
        """
        Calculate impact on Gold based on resolution risk.
        """
        if resolution_risk >= 70:
            return {
                'sentiment': 'BEARISH',
                'impact': 'HIGH - Geopolitical premium may unwind',
                'price_risk': '-10% to -15%'
            }
        elif resolution_risk >= 50:
            return {
                'sentiment': 'CAUTIOUS',
                'impact': 'MODERATE - Some safe-haven unwind possible',
                'price_risk': '-5% to -10%'
            }
        elif resolution_risk >= 30:
            return {
                'sentiment': 'NEUTRAL',
                'impact': 'LOW - Tensions persist',
                'price_risk': 'Sideways to slightly lower'
            }
        else:
            return {
                'sentiment': 'BULLISH',
                'impact': 'SUPPORTIVE - Tensions high, Gold demand strong',
                'price_risk': 'Upside bias'
            }
    
    def _get_suggested_action(self, resolution_risk: float) -> str:
        """
        Trading suggestion based on resolution risk.
        """
        if resolution_risk >= 70:
            return "REDUCE_LONGS - Major de-escalation signals, safe-haven unwind likely"
        elif resolution_risk >= 50:
            return "HEDGE - Consider reducing Gold exposure, add USDCHF"
        elif resolution_risk >= 30:
            return "MONITOR - Watch for peace talks, summits"
        else:
            return "HOLD_LONGS - Tensions persist, Gold supported"
    
    def update_factor(self, factor: str, resolution_risk: int, status: str = None):
        """
        Update a specific factor's resolution risk.
        Called when news events change the landscape.
        """
        if factor in self.factors:
            self.factors[factor]['resolution_risk'] = resolution_risk
            if status:
                self.factors[factor]['current_status'] = status
            logger.info(f"Updated {factor}: risk={resolution_risk}, status={status}")


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

_indicator = None

def get_geopolitical_risk() -> Dict:
    """Main entry point."""
    global _indicator
    if _indicator is None:
        _indicator = GeopoliticalResolutionIndicator()
    return _indicator.calculate()


if __name__ == "__main__":
    print("=" * 60)
    print("🌍 GEOPOLITICAL RESOLUTION RISK - Test")
    print("=" * 60)
    
    result = get_geopolitical_risk()
    
    print(f"\n📊 Resolution Risk: {result['resolution_risk']}/100")
    print(f"   Level: {result['level']}")
    print(f"   Gold Sentiment: {result['gold_sentiment']}")
    print(f"\n   Key Factors:")
    for factor, data in result['factors'].items():
        print(f"      {factor}: {data['resolution_risk']}% ({data['status']})")
    print(f"\n   Recent Events:")
    for event in result['recent_events'][:3]:
        print(f"      • {event['event']} → {event['gold_effect']}")
    print(f"\n   Suggestion: {result['suggested_action']}")

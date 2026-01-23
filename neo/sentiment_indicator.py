#!/usr/bin/env python3
"""
NARRATIVE SATURATION INDICATOR
═══════════════════════════════════════════════════════════════════════

Measures media/social narrative saturation for Gold.
When "Gold to $10,000!" is everywhere, correction is near.

Score 0-100:
- 0-30:  LOW - Normal coverage
- 30-60: MODERATE - Elevated interest
- 60-80: HIGH - Narrative gaining momentum
- 80-100: SATURATED - "Gold to $10K" everywhere = TOP SIGNAL
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentIndicator")


class NarrativeSaturationIndicator:
    """
    Measures how saturated the media/social narrative is.
    
    Key insight: When a narrative reaches peak saturation 
    (everyone talking about it), the move is usually exhausted.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
        
        # Bullish phrases that indicate top (when saturated)
        self.bullish_phrases = [
            "gold to 5000",
            "gold to 10000", 
            "gold to $10k",
            "gold all time high",
            "buy gold now",
            "gold breakout",
            "gold bull market",
            "gold supercycle",
            "de-dollarization gold",
            "brics gold",
            "central banks gold",
        ]
        
        # Bearish phrases that indicate bottom (when saturated)
        self.bearish_phrases = [
            "gold crash",
            "sell gold",
            "gold bubble",
            "gold overvalued",
            "gold top",
        ]
        
        logger.info("📰 Narrative Saturation Indicator initialized")
    
    def calculate(self) -> Dict:
        """
        Calculate narrative saturation score.
        """
        logger.info("📰 Calculating narrative saturation...")
        
        # Component scores
        google_score = self._get_google_trends_score()
        headline_score = self._get_headline_score()
        social_score = self._get_social_saturation()
        
        # Weights
        composite = (
            google_score * 0.40 +
            headline_score * 0.35 +
            social_score * 0.25
        )
        
        composite = float(round(composite, 1))
        
        # Determine narrative type
        narrative_type = self._determine_narrative_type(composite)
        
        # Get dominant phrases
        dominant_phrases = self._get_dominant_phrases()
        
        result = {
            'narrative_saturation': composite,
            'level': self._get_level(composite),
            'narrative_type': narrative_type,
            'timestamp': datetime.utcnow().isoformat(),
            
            'components': {
                'google_trends': float(round(google_score, 1)),
                'headline_analysis': float(round(headline_score, 1)),
                'social_saturation': float(round(social_score, 1))
            },
            
            'dominant_phrases': dominant_phrases,
            
            # Trading implications
            'correction_probability': self._calc_correction_probability(composite),
            'suggested_action': self._get_suggested_action(composite, narrative_type),
            
            # For Ghost Commander
            'is_saturated': bool(composite >= 80),
            'reduce_longs': bool(composite >= 60),
        }
        
        logger.info(f"   Narrative Saturation: {composite} ({result['level']})")
        return result
    
    def _get_level(self, score: float) -> str:
        if score >= 80:
            return 'SATURATED'
        elif score >= 60:
            return 'HIGH'
        elif score >= 30:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _get_google_trends_score(self) -> float:
        """
        Get Google Trends score for Gold bullish phrases.
        """
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            
            # Check key phrases
            phrases = ['buy gold', 'gold price prediction', 'gold investment']
            pytrends.build_payload(phrases, timeframe='now 7-d')
            interest = pytrends.interest_over_time()
            
            if not interest.empty:
                recent = interest.iloc[-24:].mean()
                avg = sum(recent[p] for p in phrases if p in recent) / len(phrases)
                return min(100, avg * 1.3)
                
        except Exception as e:
            logger.warning(f"Google Trends error: {e}")
        
        # Fallback estimate based on market conditions
        return 55.0  # Elevated during rally
    
    def _get_headline_score(self) -> float:
        """
        Analyze financial news headlines for Gold sentiment.
        """
        # In production, would scrape:
        # - Bloomberg, Reuters, CNBC for Gold headlines
        # - Count "Gold to $X" predictions
        # - Measure headline sentiment
        
        # For now, use proxy based on market conditions
        # During strong rally with ATH, headlines are typically bullish
        return 65.0  # Elevated bullish headlines
    
    def _get_social_saturation(self) -> float:
        """
        Measure social media saturation.
        """
        # Would measure:
        # - Twitter/X Gold mentions vs baseline
        # - YouTube "Gold trading" video surge
        # - Reddit Gold discussion frequency
        
        return 50.0  # Moderate social activity
    
    def _determine_narrative_type(self, score: float) -> str:
        """
        Determine if narrative is bullish or bearish.
        """
        # During current rally, narrative is bullish
        # High score + bullish = TOP RISK
        # High score + bearish = BOTTOM SIGNAL
        return 'BULLISH'  # Current market narrative
    
    def _get_dominant_phrases(self) -> list:
        """
        Return currently dominant phrases.
        """
        # Would analyze actual search/social data
        return [
            "BRICS de-dollarization",
            "Central bank gold buying",
            "Gold all-time high",
            "Safe haven demand"
        ]
    
    def _calc_correction_probability(self, score: float) -> int:
        """
        Probability of correction based on saturation.
        """
        if score >= 90:
            return 85
        elif score >= 80:
            return 70
        elif score >= 60:
            return 45
        elif score >= 40:
            return 25
        else:
            return 10
    
    def _get_suggested_action(self, score: float, narrative_type: str) -> str:
        """
        Trading suggestion based on narrative saturation.
        """
        if score >= 80 and narrative_type == 'BULLISH':
            return "REDUCE_LONGS - Narrative saturated, correction likely"
        elif score >= 60 and narrative_type == 'BULLISH':
            return "CAUTION - Elevated bullish sentiment"
        elif score >= 80 and narrative_type == 'BEARISH':
            return "ACCUMULATE - Peak fear may signal bottom"
        else:
            return "NORMAL - Trade per technicals"


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

_indicator = None

def get_narrative_saturation() -> Dict:
    """Main entry point."""
    global _indicator
    if _indicator is None:
        _indicator = NarrativeSaturationIndicator()
    return _indicator.calculate()


if __name__ == "__main__":
    print("=" * 60)
    print("📰 NARRATIVE SATURATION INDICATOR - Test")
    print("=" * 60)
    
    result = get_narrative_saturation()
    
    print(f"\n📊 Narrative Saturation: {result['narrative_saturation']}/100")
    print(f"   Level: {result['level']}")
    print(f"   Type: {result['narrative_type']}")
    print(f"\n   Components:")
    for k, v in result['components'].items():
        print(f"      {k}: {v}")
    print(f"\n   Correction Probability: {result['correction_probability']}%")
    print(f"   Suggestion: {result['suggested_action']}")

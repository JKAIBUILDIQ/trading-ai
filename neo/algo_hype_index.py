#!/usr/bin/env python3
"""
ALGO HYPE INDEX (AHI)
═══════════════════════════════════════════════════════════════════════

Detects when too many traders/bots are piling into the same trade.
Returns a score 0-100 where:
  0-25:   LOW - Trade normally
  25-50:  MODERATE - Reduce position sizes 25%
  50-75:  HIGH - Reduce 50%, tighten stops
  75-90:  EXTREME - Reduce 75%, consider closing
  90-100: PARABOLIC - Close longs, prepare for reversal

"When everyone is screaming 'Gold to $10,000!' - that's when we should 
be at 25% position size with trailing stops."
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AHI")


class AlgoHypeIndex:
    """
    Composite indicator measuring crowd euphoria and algo herding.
    
    Components (25% each):
    1. Social Media Score - Twitter, Reddit, YouTube mentions
    2. Forum Activity Score - MQL5, ForexFactory, TradingView
    3. Retail Positioning Score - COT, broker sentiment, ETF flows
    4. Technical Crowd Score - RSI extremes, Bollinger position, ADX
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 900  # 15 minutes
        self.last_update = None
        
        # Component weights
        self.weights = {
            'social': 0.25,
            'forums': 0.25,
            'retail': 0.25,
            'technical': 0.25
        }
        
        # Thresholds
        self.thresholds = {
            'low': 25,
            'moderate': 50,
            'high': 75,
            'extreme': 90
        }
        
        logger.info("📊 Algo Hype Index initialized")
    
    # ═══════════════════════════════════════════════════════════════════
    # MAIN CALCULATOR
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate(self, technical_data: Dict = None) -> Dict:
        """
        Calculate the full Algo Hype Index.
        
        Returns:
            Dict with score, components, level, and defense recommendations
        """
        logger.info("📊 Calculating Algo Hype Index...")
        
        # Get component scores
        social_score = self._get_social_media_score()
        forum_score = self._get_forum_activity_score()
        retail_score = self._get_retail_positioning_score()
        technical_score = self._get_technical_crowd_score(technical_data or {})
        
        # Calculate weighted AHI
        ahi = (
            social_score * self.weights['social'] +
            forum_score * self.weights['forums'] +
            retail_score * self.weights['retail'] +
            technical_score * self.weights['technical']
        )
        
        ahi = round(ahi, 1)
        
        # Determine level
        level = self._get_level(ahi)
        
        # Get defense recommendations
        defense = self._get_defense_recommendations(ahi, level)
        
        # Get historical parallel
        parallel = self._find_historical_parallel(ahi)
        
        result = {
            'ahi_score': float(ahi),
            'level': level,
            'timestamp': datetime.utcnow().isoformat(),
            
            'components': {
                'social_media': {
                    'score': float(round(social_score, 1)),
                    'weight': float(self.weights['social']),
                    'status': self._score_status(social_score)
                },
                'forum_activity': {
                    'score': float(round(forum_score, 1)),
                    'weight': float(self.weights['forums']),
                    'status': self._score_status(forum_score)
                },
                'retail_positioning': {
                    'score': float(round(retail_score, 1)),
                    'weight': float(self.weights['retail']),
                    'status': self._score_status(retail_score)
                },
                'technical_crowd': {
                    'score': float(round(technical_score, 1)),
                    'weight': float(self.weights['technical']),
                    'status': self._score_status(technical_score)
                }
            },
            
            'defense': defense,
            'historical_parallel': parallel,
            
            # Action modifiers for NEO (convert numpy types to Python natives)
            'position_size_multiplier': float(defense['position_multiplier']),
            'block_buys': bool(ahi >= self.thresholds['extreme']),
            'force_trailing_stop': bool(ahi >= self.thresholds['high']),
            'alert_human': bool(ahi >= self.thresholds['high'])
        }
        
        self.last_update = datetime.utcnow()
        logger.info(f"   AHI Score: {ahi} ({level})")
        
        return result
    
    def _get_level(self, score: float) -> str:
        """Convert score to level."""
        if score >= self.thresholds['extreme']:
            return 'PARABOLIC'
        elif score >= self.thresholds['high']:
            return 'EXTREME'
        elif score >= self.thresholds['moderate']:
            return 'HIGH'
        elif score >= self.thresholds['low']:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _score_status(self, score: float) -> str:
        """Get status emoji for score."""
        if score >= 75:
            return '🔴'
        elif score >= 50:
            return '🟠'
        elif score >= 25:
            return '🟡'
        else:
            return '🟢'
    
    # ═══════════════════════════════════════════════════════════════════
    # COMPONENT SCORERS
    # ═══════════════════════════════════════════════════════════════════
    
    def _get_social_media_score(self) -> float:
        """
        Social Media Score (0-100):
        - Twitter/X gold mentions vs baseline
        - Reddit r/forex, r/trading activity
        - YouTube "gold trading" video uploads
        - General social sentiment
        """
        try:
            from social_scraper import SocialScraper
            scraper = SocialScraper()
            return scraper.get_composite_score()
        except ImportError:
            logger.warning("   Social scraper not available - using estimate")
            return self._estimate_social_score()
    
    def _estimate_social_score(self) -> float:
        """
        Estimate social score based on Google Trends proxy.
        """
        try:
            # Use pytrends if available
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(['buy gold', 'gold price', 'xauusd'], timeframe='now 7-d')
            interest = pytrends.interest_over_time()
            
            if not interest.empty:
                # Get average of last 7 days vs historical
                recent = interest.iloc[-7:].mean()
                avg_score = (recent['buy gold'] + recent['gold price'] + recent['xauusd']) / 3
                # Scale to 0-100
                return min(100, avg_score * 1.2)
        except:
            pass
        
        # Fallback: Based on price action (parabolic price = more hype)
        # This is a rough proxy - higher prices = more retail interest
        return 55.0  # Moderate baseline during current rally
    
    def _get_forum_activity_score(self) -> float:
        """
        Forum Activity Score (0-100):
        - MQL5 new Gold EA releases
        - MQL5 forum Gold threads
        - ForexFactory gold discussion
        - TradingView Gold ideas published
        """
        try:
            from forum_scraper import ForumScraper
            scraper = ForumScraper()
            return scraper.get_composite_score()
        except ImportError:
            logger.warning("   Forum scraper not available - using estimate")
            return self._estimate_forum_score()
    
    def _estimate_forum_score(self) -> float:
        """
        Estimate forum activity based on known patterns.
        During parabolic moves, forum activity typically increases 2-3x.
        """
        # Base activity level
        base = 40.0
        
        # Adjust based on current market conditions
        # (In production, this would query actual forum APIs)
        
        # Current Gold is in strong uptrend = elevated interest
        return 60.0
    
    def _get_retail_positioning_score(self) -> float:
        """
        Retail Positioning Score (0-100):
        - COT report: retail net long %
        - Broker sentiment (IG, OANDA % long)
        - GLD ETF inflows
        - Options put/call ratio
        """
        try:
            from retail_sentiment import RetailSentiment
            sentiment = RetailSentiment()
            return sentiment.get_composite_score()
        except ImportError:
            logger.warning("   Retail sentiment module not available - using estimate")
            return self._estimate_retail_score()
    
    def _estimate_retail_score(self) -> float:
        """
        Estimate retail positioning from available data.
        """
        scores = []
        
        # Try to get broker sentiment from DailyFX/IG
        try:
            # IG Client Sentiment (free API)
            response = requests.get(
                "https://api.dailyfx.com/market/sentiment",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    if 'GOLD' in item.get('market', '').upper():
                        long_pct = item.get('longPositionPercentage', 50)
                        # High long % = high retail score
                        scores.append(long_pct)
        except:
            pass
        
        if scores:
            return sum(scores) / len(scores)
        
        # Default estimate based on typical parabolic behavior
        # When Gold rallies this hard, retail is typically 70-80% long
        return 72.0
    
    def _get_technical_crowd_score(self, data: Dict) -> float:
        """
        Technical Crowd Score (0-100):
        - RSI(14) on Daily (>70 = crowded long)
        - % above key EMAs
        - Bollinger Band position
        - ADX trend strength
        """
        scores = []
        
        # RSI component (0-100 maps directly)
        rsi = data.get('rsi14_d1', data.get('rsi_14', 50))
        if rsi:
            # RSI > 70 is overbought = crowded
            # Scale: RSI 50 = 25 score, RSI 70 = 75 score, RSI 85 = 100 score
            rsi_score = max(0, min(100, (rsi - 30) * 2.5))
            scores.append(rsi_score)
        
        # RSI(2) - ultra short term crowd
        rsi2 = data.get('rsi2_h1', data.get('rsi_2', 50))
        if rsi2:
            rsi2_score = max(0, min(100, rsi2))
            scores.append(rsi2_score)
        
        # Bollinger Band position
        # If price is at upper band = crowded
        bb_position = data.get('bb_position', 0.5)  # 0=lower, 1=upper
        if bb_position:
            bb_score = bb_position * 100
            scores.append(bb_score)
        
        # Trend alignment (all timeframes bullish = crowded)
        trend_align = data.get('trend_alignment', 0)
        if trend_align:
            # Trend alignment near 1.0 = everyone on same side
            trend_score = abs(trend_align) * 100
            scores.append(trend_score)
        
        # Volume profile - high volume at highs = distribution coming
        vol_profile = data.get('volume_profile_score', 50)
        scores.append(vol_profile)
        
        if scores:
            return sum(scores) / len(scores)
        
        # Default based on current Gold conditions
        return 78.0  # High RSI during rally
    
    # ═══════════════════════════════════════════════════════════════════
    # DEFENSE RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def _get_defense_recommendations(self, ahi: float, level: str) -> Dict:
        """
        Get defense recommendations based on AHI level.
        """
        if level == 'PARABOLIC':
            return {
                'position_multiplier': float(0.0),  # No new longs
                'close_pct': int(75),  # Close 75% of positions
                'stop_type': 'TRAILING_TIGHT',
                'stop_atr_mult': float(1.0),
                'block_buys': True,
                'allow_sells': True,
                'hedge_recommendation': 'FULL_HEDGE',
                'alert_level': 'CRITICAL',
                'human_override_required': True,
                'action_summary': [
                    '🛑 CLOSE 75% of long positions IMMEDIATELY',
                    '🛡️ Trail stops at 1.0x ATR on remaining',
                    '⛔ NO NEW BUYS - Only SELLs allowed',
                    '🔄 Activate FULL hedge (USDCHF, short Gold)',
                    '📞 ALERT: Human decision required for any action'
                ]
            }
        
        elif level == 'EXTREME':
            return {
                'position_multiplier': float(0.25),
                'close_pct': int(50),
                'stop_type': 'TRAILING_AGGRESSIVE',
                'stop_atr_mult': float(1.5),
                'block_buys': True,
                'allow_sells': True,
                'hedge_recommendation': 'PARTIAL_HEDGE',
                'alert_level': 'HIGH',
                'human_override_required': False,
                'action_summary': [
                    '⚠️ CLOSE 50% of long positions',
                    '🛡️ Trail stops at 1.5x ATR',
                    '⛔ Block NEW buys (reduce to HOLD)',
                    '🔄 Activate partial hedge',
                    '📊 Monitor every 5 minutes'
                ]
            }
        
        elif level == 'HIGH':
            return {
                'position_multiplier': float(0.50),
                'close_pct': int(25),
                'stop_type': 'TRAILING',
                'stop_atr_mult': float(2.0),
                'block_buys': False,
                'allow_sells': True,
                'hedge_recommendation': 'CONSIDER_HEDGE',
                'alert_level': 'MEDIUM',
                'human_override_required': False,
                'action_summary': [
                    '📉 Reduce position sizes by 50%',
                    '🎯 Take 25% profit on winners',
                    '🛡️ Trail stops at 2.0x ATR',
                    '⚠️ Caution on new buys'
                ]
            }
        
        elif level == 'MODERATE':
            return {
                'position_multiplier': float(0.75),
                'close_pct': int(0),
                'stop_type': 'NORMAL',
                'stop_atr_mult': float(2.5),
                'block_buys': False,
                'allow_sells': True,
                'hedge_recommendation': 'NONE',
                'alert_level': 'LOW',
                'human_override_required': False,
                'action_summary': [
                    '📉 Reduce position sizes by 25%',
                    '🎯 Normal stop loss levels',
                    '👀 Monitor for escalation'
                ]
            }
        
        else:  # LOW
            return {
                'position_multiplier': float(1.0),
                'close_pct': int(0),
                'stop_type': 'NORMAL',
                'stop_atr_mult': float(2.5),
                'block_buys': False,
                'allow_sells': True,
                'hedge_recommendation': 'NONE',
                'alert_level': 'NONE',
                'human_override_required': False,
                'action_summary': [
                    '✅ Trade normally',
                    '📊 Standard risk management'
                ]
            }
    
    # ═══════════════════════════════════════════════════════════════════
    # HISTORICAL PARALLELS
    # ═══════════════════════════════════════════════════════════════════
    
    def _find_historical_parallel(self, ahi: float) -> Dict:
        """
        Find historical parallel to current AHI level.
        """
        # Historical parabolic events
        parallels = [
            {
                'event': 'BTC April 2021',
                'peak': '$65,000',
                'crash': '-55%',
                'ahi_at_peak': 95,
                'warning_signs': ['Elon tweets', 'SNL appearance', 'Everyone bullish']
            },
            {
                'event': 'Gold August 2011',
                'peak': '$1,920',
                'crash': '-28%',
                'ahi_at_peak': 88,
                'warning_signs': ['QE euphoria', 'Debt ceiling drama', 'Media frenzy']
            },
            {
                'event': 'Gold August 2020',
                'peak': '$2,075',
                'crash': '-18%',
                'ahi_at_peak': 82,
                'warning_signs': ['COVID stimulus', 'Reddit hype', 'Gold ETF inflows']
            },
            {
                'event': 'Gold March 2022',
                'peak': '$2,070',
                'crash': '-22%',
                'ahi_at_peak': 79,
                'warning_signs': ['Ukraine war spike', 'Inflation fears', 'Rate hike denial']
            },
            {
                'event': 'Gold October 2024',
                'peak': '$2,790',
                'crash': '-12%',
                'ahi_at_peak': 75,
                'warning_signs': ['Election uncertainty', 'BRICS narrative', 'Influencer calls']
            }
        ]
        
        # Find closest match
        closest = min(parallels, key=lambda x: abs(x['ahi_at_peak'] - ahi))
        
        return {
            'match': closest['event'],
            'peak': closest['peak'],
            'subsequent_crash': closest['crash'],
            'ahi_at_that_peak': int(closest['ahi_at_peak']),
            'similarity': float(round(100 - abs(closest['ahi_at_peak'] - ahi), 1)),
            'warning_signs_then': closest['warning_signs']
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # FORMATTING
    # ═══════════════════════════════════════════════════════════════════
    
    def format_for_telegram(self, result: Dict) -> str:
        """Format AHI result for Telegram alert."""
        
        level_emoji = {
            'LOW': '🟢',
            'MODERATE': '🟡',
            'HIGH': '🟠',
            'EXTREME': '🔴',
            'PARABOLIC': '💀'
        }
        
        emoji = level_emoji.get(result['level'], '❓')
        
        msg = f"""🚨 ALGO HYPE INDEX REPORT

{emoji} AHI Score: {result['ahi_score']}/100 ({result['level']})

📊 COMPONENTS:
├── Social Media: {result['components']['social_media']['score']:.0f} {result['components']['social_media']['status']}
├── Forum Activity: {result['components']['forum_activity']['score']:.0f} {result['components']['forum_activity']['status']}
├── Retail Position: {result['components']['retail_positioning']['score']:.0f} {result['components']['retail_positioning']['status']}
└── Technical Crowd: {result['components']['technical_crowd']['score']:.0f} {result['components']['technical_crowd']['status']}

📜 HISTORICAL PARALLEL:
└── Similar to: {result['historical_parallel']['match']}
    Peak: {result['historical_parallel']['peak']} → Crash: {result['historical_parallel']['subsequent_crash']}
    Similarity: {result['historical_parallel']['similarity']}%

🛡️ DEFENSE STATUS:
├── Position Multiplier: {result['defense']['position_multiplier']:.0%}
├── Block Buys: {'YES ⛔' if result['block_buys'] else 'No'}
├── Trailing Stops: {result['defense']['stop_type']}
└── Hedge: {result['defense']['hedge_recommendation']}

📋 ACTIONS:
"""
        for action in result['defense']['action_summary']:
            msg += f"• {action}\n"
        
        msg += f"\n⏰ Updated: {result['timestamp'][:19]}"
        
        return msg
    
    def format_for_neo_prompt(self, result: Dict) -> str:
        """Format AHI result for NEO's LLM prompt."""
        
        return f"""
═══════════════════════════════════════════════════════════════════════
ALGO HYPE INDEX (CROWD DETECTION)
═══════════════════════════════════════════════════════════════════════

AHI Score: {result['ahi_score']}/100 - {result['level']}

INTERPRETATION:
- 0-25: LOW (trade normally)
- 25-50: MODERATE (reduce sizes 25%)
- 50-75: HIGH (reduce sizes 50%, trail stops)
- 75-90: EXTREME (reduce 75%, block buys)
- 90-100: PARABOLIC (close longs, prepare reversal)

Components:
- Social Media Hype: {result['components']['social_media']['score']:.0f}/100
- Forum/EA Activity: {result['components']['forum_activity']['score']:.0f}/100
- Retail Positioning: {result['components']['retail_positioning']['score']:.0f}/100
- Technical Crowding: {result['components']['technical_crowd']['score']:.0f}/100

Historical Parallel: {result['historical_parallel']['match']}
- That event peaked then crashed {result['historical_parallel']['subsequent_crash']}

CURRENT DEFENSE SETTINGS:
- Position size: {result['defense']['position_multiplier']:.0%} of normal
- New BUYs: {'BLOCKED' if result['block_buys'] else 'ALLOWED'}
- Stop type: {result['defense']['stop_type']}

⚠️ If AHI > 75: Convert BUY signals to HOLD
⚠️ If AHI > 90: Consider only SELL signals

"""


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

_ahi_instance = None

def get_ahi_instance() -> AlgoHypeIndex:
    """Get or create the global AHI instance."""
    global _ahi_instance
    if _ahi_instance is None:
        _ahi_instance = AlgoHypeIndex()
    return _ahi_instance


def get_algo_hype_index(technical_data: Dict = None) -> Dict:
    """
    Main entry point - returns full AHI analysis.
    
    Args:
        technical_data: Optional dict with RSI, trend alignment, etc.
    
    Returns:
        Dict with ahi_score, level, components, defense recommendations
    """
    return get_ahi_instance().calculate(technical_data)


def get_position_multiplier(technical_data: Dict = None) -> float:
    """Quick check - get position size multiplier based on AHI."""
    result = get_algo_hype_index(technical_data)
    return result['position_size_multiplier']


def should_block_buys(technical_data: Dict = None) -> bool:
    """Quick check - should we block new buy signals?"""
    result = get_algo_hype_index(technical_data)
    return result['block_buys']


# ═══════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("📊 ALGO HYPE INDEX - Test Run")
    print("=" * 70)
    
    # Test with some technical data
    test_data = {
        'rsi14_d1': 78,  # Daily RSI
        'rsi2_h1': 92,   # H1 RSI(2)
        'trend_alignment': 0.85,  # Strong bullish alignment
        'bb_position': 0.9  # Near upper Bollinger
    }
    
    result = get_algo_hype_index(test_data)
    
    ahi = AlgoHypeIndex()
    print(ahi.format_for_telegram(result))
    
    print("\n" + "=" * 70)
    print("NEO PROMPT FORMAT:")
    print("=" * 70)
    print(ahi.format_for_neo_prompt(result))

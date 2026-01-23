"""
Indian Sentiment Aggregator - Combines All Sources
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Aggregates sentiment from:
- Telegram groups (highest weight - real-time retail chatter)
- YouTube live chats (during market hours)
- Twitter/X (trending topics, influencers)
- Reddit r/IndianStreetBets (FOMO detection)

Outputs actionable intelligence for NEO:
- Overall retail bias
- Stop loss clusters
- FOMO/Panic levels
- Counter-trade opportunities

Created: 2026-01-24
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

log = logging.getLogger(__name__)


@dataclass
class IndianSentiment:
    """Complete aggregated Indian sentiment"""
    
    # Overall
    overall_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0-1
    fear_greed_index: int  # 0-100 (50 = neutral, 100 = extreme greed)
    
    # Source-specific
    telegram_sentiment: Dict
    youtube_sentiment: Dict
    twitter_sentiment: Dict
    reddit_sentiment: Dict
    
    # Key intelligence
    key_levels_crowd_watching: List[float]
    stop_loss_clusters: List[float]
    take_profit_targets: List[float]
    
    # Actionable
    crowd_position: str  # 'MAX_LONG', 'LONG', 'NEUTRAL', 'SHORT', 'MAX_SHORT'
    fade_opportunity: bool
    stop_hunt_target: Optional[float]
    fomo_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    panic_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    
    # Meta
    data_quality: str  # 'GOOD', 'PARTIAL', 'SIMULATED'
    sources_active: int
    timestamp: datetime


class IndianSentimentAggregator:
    """
    Aggregates sentiment from all Indian trading sources
    
    Usage:
        aggregator = IndianSentimentAggregator()
        sentiment = aggregator.get_live_sentiment()
        
        if sentiment.fade_opportunity:
            # Retail is overextended, consider fading
            pass
    """
    
    # Weight for each source (Telegram highest - most real-time)
    SOURCE_WEIGHTS = {
        'telegram': 0.40,
        'youtube': 0.25,
        'twitter': 0.20,
        'reddit': 0.15
    }
    
    def __init__(self, 
                 telegram_api_id: str = None,
                 telegram_api_hash: str = None,
                 twitter_bearer: str = None,
                 reddit_client_id: str = None,
                 reddit_client_secret: str = None,
                 youtube_api_key: str = None):
        """
        Initialize aggregator with API credentials
        """
        try:
            from .telegram_scraper import IndianTelegramScraper
            from .youtube_scraper import IndianYouTubeScraper
            from .social_scraper import IndianTwitterScraper, IndianRedditScraper
        except ImportError:
            # Fallback for direct execution
            from telegram_scraper import IndianTelegramScraper
            from youtube_scraper import IndianYouTubeScraper
            from social_scraper import IndianTwitterScraper, IndianRedditScraper
        
        self.telegram = IndianTelegramScraper(telegram_api_id, telegram_api_hash)
        self.youtube = IndianYouTubeScraper(youtube_api_key)
        self.twitter = IndianTwitterScraper(twitter_bearer)
        self.reddit = IndianRedditScraper(reddit_client_id, reddit_client_secret)
        
        self.last_sentiment = None
    
    def get_live_sentiment(self, use_simulation: bool = True) -> IndianSentiment:
        """
        Get aggregated sentiment from all sources
        
        Args:
            use_simulation: Use simulated data if real data unavailable
        """
        # Collect from all sources
        telegram_data = self._get_telegram_sentiment(use_simulation)
        youtube_data = self._get_youtube_sentiment(use_simulation)
        twitter_data = self._get_twitter_sentiment(use_simulation)
        reddit_data = self._get_reddit_sentiment(use_simulation)
        
        # Calculate overall sentiment
        overall_bias, confidence = self._calculate_overall_bias(
            telegram_data, youtube_data, twitter_data, reddit_data
        )
        
        # Calculate Fear/Greed Index (0-100)
        fear_greed = self._calculate_fear_greed(
            telegram_data, youtube_data, twitter_data, reddit_data
        )
        
        # Extract key levels
        key_levels = self._extract_key_levels(
            telegram_data, youtube_data, twitter_data, reddit_data
        )
        
        # Extract stop loss clusters
        stop_clusters = self._extract_stop_clusters(telegram_data)
        
        # Extract TP targets
        tp_targets = telegram_data.get('top_tp_levels', [])
        
        # Determine crowd position
        crowd_position = self._determine_crowd_position(overall_bias, confidence, fear_greed)
        
        # Detect fade opportunity
        fade_opportunity = self._detect_fade_opportunity(
            overall_bias, confidence, fear_greed, 
            telegram_data, reddit_data
        )
        
        # Identify stop hunt target
        stop_hunt_target = stop_clusters[0] if stop_clusters else None
        
        # FOMO/Panic levels
        fomo_level = self._calculate_fomo_level(fear_greed, reddit_data, telegram_data)
        panic_level = self._calculate_panic_level(fear_greed, telegram_data)
        
        # Data quality
        sources_active = sum([
            1 if telegram_data.get('message_count', 0) > 0 else 0,
            1 if youtube_data.get('message_count', 0) > 0 else 0,
            1 if twitter_data.get('tweets_last_hour', 0) > 0 else 0,
            1 if reddit_data.get('posts_today', 0) > 0 else 0,
        ])
        
        data_quality = 'SIMULATED' if use_simulation else ('GOOD' if sources_active >= 3 else 'PARTIAL')
        
        sentiment = IndianSentiment(
            overall_bias=overall_bias,
            confidence=round(confidence, 2),
            fear_greed_index=fear_greed,
            telegram_sentiment=telegram_data,
            youtube_sentiment=youtube_data,
            twitter_sentiment=twitter_data,
            reddit_sentiment=reddit_data,
            key_levels_crowd_watching=key_levels,
            stop_loss_clusters=stop_clusters,
            take_profit_targets=tp_targets,
            crowd_position=crowd_position,
            fade_opportunity=fade_opportunity,
            stop_hunt_target=stop_hunt_target,
            fomo_level=fomo_level,
            panic_level=panic_level,
            data_quality=data_quality,
            sources_active=sources_active,
            timestamp=datetime.utcnow()
        )
        
        self.last_sentiment = sentiment
        return sentiment
    
    def _get_telegram_sentiment(self, simulate: bool) -> Dict:
        """Get Telegram sentiment"""
        if simulate:
            self.telegram.simulate_market_session('BULLISH')
        return self.telegram.to_dict()
    
    def _get_youtube_sentiment(self, simulate: bool) -> Dict:
        """Get YouTube sentiment"""
        if simulate:
            self.youtube.add_simulated_messages('BULLISH')
        return self.youtube.to_dict()
    
    def _get_twitter_sentiment(self, simulate: bool) -> Dict:
        """Get Twitter sentiment"""
        if simulate:
            self.twitter.add_simulated_tweets('BULLISH')
        return self.twitter.to_dict()
    
    def _get_reddit_sentiment(self, simulate: bool) -> Dict:
        """Get Reddit sentiment"""
        if simulate:
            self.reddit.add_simulated_posts('BULLISH')
        return self.reddit.to_dict()
    
    def _calculate_overall_bias(self, telegram: Dict, youtube: Dict,
                                twitter: Dict, reddit: Dict) -> tuple:
        """Calculate weighted overall bias"""
        
        direction_map = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}
        
        scores = []
        weights = []
        
        # Telegram
        if telegram.get('direction'):
            scores.append(direction_map.get(telegram['direction'], 0) * telegram.get('strength', 0.5))
            weights.append(self.SOURCE_WEIGHTS['telegram'])
        
        # YouTube
        if youtube.get('direction'):
            scores.append(direction_map.get(youtube['direction'], 0) * youtube.get('strength', 0.5))
            weights.append(self.SOURCE_WEIGHTS['youtube'])
        
        # Twitter
        if twitter.get('direction'):
            scores.append(direction_map.get(twitter['direction'], 0) * twitter.get('strength', 0.5))
            weights.append(self.SOURCE_WEIGHTS['twitter'])
        
        # Reddit
        if reddit.get('direction'):
            scores.append(direction_map.get(reddit['direction'], 0) * reddit.get('strength', 0.5))
            weights.append(self.SOURCE_WEIGHTS['reddit'])
        
        if not scores:
            return 'NEUTRAL', 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0
        
        if weighted_score > 0.2:
            bias = 'BULLISH'
        elif weighted_score < -0.2:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
        
        confidence = abs(weighted_score)
        
        return bias, min(confidence, 1.0)
    
    def _calculate_fear_greed(self, telegram: Dict, youtube: Dict,
                              twitter: Dict, reddit: Dict) -> int:
        """
        Calculate Fear/Greed Index (0-100)
        
        0 = Extreme Fear
        50 = Neutral
        100 = Extreme Greed
        """
        ratios = []
        
        # From Telegram
        if telegram.get('fear_greed_ratio'):
            ratios.append(telegram['fear_greed_ratio'])
        
        # From YouTube
        if youtube.get('fear_greed_ratio'):
            ratios.append(youtube['fear_greed_ratio'])
        
        # From Twitter
        if twitter.get('fear_greed_score'):
            ratios.append(twitter['fear_greed_score'])
        
        # Reddit FOMO factor
        if reddit.get('wsb_style_fomo'):
            ratios.append(2.5)  # High greed
        
        if not ratios:
            return 50  # Neutral
        
        avg_ratio = sum(ratios) / len(ratios)
        
        # Convert ratio to 0-100 scale
        # ratio > 1 = greedy, ratio < 1 = fearful
        # Map: 0.2 -> 0, 1.0 -> 50, 3.0 -> 100
        if avg_ratio <= 0.2:
            return 0
        elif avg_ratio >= 3.0:
            return 100
        elif avg_ratio < 1.0:
            return int(50 * (avg_ratio - 0.2) / 0.8)
        else:
            return int(50 + 50 * (avg_ratio - 1.0) / 2.0)
    
    def _extract_key_levels(self, telegram: Dict, youtube: Dict,
                            twitter: Dict, reddit: Dict) -> List[float]:
        """Extract key price levels being discussed"""
        levels = []
        
        if telegram.get('key_price_levels'):
            levels.extend(telegram['key_price_levels'])
        
        if youtube.get('top_discussed_levels'):
            levels.extend(youtube['top_discussed_levels'])
        
        # Count and return most common
        if not levels:
            return [5000, 4950, 4900]  # Default Gold levels
        
        counter = Counter([round(l / 10) * 10 for l in levels])
        return [level for level, _ in counter.most_common(5)]
    
    def _extract_stop_clusters(self, telegram: Dict) -> List[float]:
        """Extract stop loss clusters from Telegram"""
        return telegram.get('top_sl_levels', [])
    
    def _determine_crowd_position(self, bias: str, confidence: float, 
                                  fear_greed: int) -> str:
        """Determine crowd position intensity"""
        if bias == 'BULLISH':
            if confidence > 0.7 and fear_greed > 75:
                return 'MAX_LONG'
            elif confidence > 0.5:
                return 'LONG'
        elif bias == 'BEARISH':
            if confidence > 0.7 and fear_greed < 25:
                return 'MAX_SHORT'
            elif confidence > 0.5:
                return 'SHORT'
        
        return 'NEUTRAL'
    
    def _detect_fade_opportunity(self, bias: str, confidence: float,
                                 fear_greed: int, telegram: Dict,
                                 reddit: Dict) -> bool:
        """Detect if this is a good time to fade retail"""
        
        # High confidence + extreme greed = fade opportunity
        if confidence > 0.7 and fear_greed > 80:
            return True
        
        # FOMO detected + crowded position
        if telegram.get('fomo_detected') and confidence > 0.6:
            return True
        
        # Reddit WSB-style fomo
        if reddit.get('wsb_style_fomo') and fear_greed > 70:
            return True
        
        # Extreme fear = potential bottom (fade the panic)
        if fear_greed < 20 and confidence > 0.6:
            return True
        
        return False
    
    def _calculate_fomo_level(self, fear_greed: int, reddit: Dict,
                              telegram: Dict) -> str:
        """Calculate FOMO level"""
        if fear_greed > 85 or reddit.get('wsb_style_fomo'):
            return 'EXTREME'
        elif fear_greed > 70 or telegram.get('fomo_detected'):
            return 'HIGH'
        elif fear_greed > 55:
            return 'MEDIUM'
        return 'LOW'
    
    def _calculate_panic_level(self, fear_greed: int, telegram: Dict) -> str:
        """Calculate panic level"""
        if fear_greed < 15 or telegram.get('panic_detected'):
            return 'EXTREME'
        elif fear_greed < 30:
            return 'HIGH'
        elif fear_greed < 45:
            return 'MEDIUM'
        return 'LOW'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API"""
        sentiment = self.last_sentiment or self.get_live_sentiment()
        
        return {
            'overall_bias': sentiment.overall_bias,
            'confidence': sentiment.confidence,
            'fear_greed_index': sentiment.fear_greed_index,
            
            'telegram_sentiment': sentiment.telegram_sentiment,
            'youtube_sentiment': sentiment.youtube_sentiment,
            'twitter_sentiment': sentiment.twitter_sentiment,
            'reddit_sentiment': sentiment.reddit_sentiment,
            
            'key_levels_crowd_watching': sentiment.key_levels_crowd_watching,
            'stop_loss_clusters': sentiment.stop_loss_clusters,
            'take_profit_targets': sentiment.take_profit_targets,
            
            'actionable_intel': {
                'crowd_position': sentiment.crowd_position,
                'fade_opportunity': sentiment.fade_opportunity,
                'stop_hunt_target': sentiment.stop_hunt_target,
                'fomo_level': sentiment.fomo_level,
                'panic_level': sentiment.panic_level
            },
            
            'data_quality': sentiment.data_quality,
            'sources_active': sentiment.sources_active,
            'timestamp': sentiment.timestamp.isoformat()
        }


def get_indian_sentiment(use_simulation: bool = True) -> Dict:
    """
    Quick function for NEO integration
    
    Returns aggregated Indian retail sentiment
    """
    aggregator = IndianSentimentAggregator()
    sentiment = aggregator.get_live_sentiment(use_simulation)
    return aggregator.to_dict()


# Test
if __name__ == "__main__":
    print("="*70)
    print("🇮🇳 INDIAN SENTIMENT AGGREGATOR TEST")
    print("="*70)
    
    aggregator = IndianSentimentAggregator()
    sentiment = aggregator.get_live_sentiment(use_simulation=True)
    
    print(f"\n📊 OVERALL SENTIMENT:")
    print(f"   Bias: {sentiment.overall_bias}")
    print(f"   Confidence: {sentiment.confidence:.0%}")
    print(f"   Fear/Greed Index: {sentiment.fear_greed_index}/100")
    
    print(f"\n🎯 CROWD POSITION: {sentiment.crowd_position}")
    print(f"   FOMO Level: {sentiment.fomo_level}")
    print(f"   Panic Level: {sentiment.panic_level}")
    
    print(f"\n💡 ACTIONABLE INTEL:")
    print(f"   Fade Opportunity: {'✅ YES!' if sentiment.fade_opportunity else '❌ No'}")
    if sentiment.stop_hunt_target:
        print(f"   Stop Hunt Target: ${sentiment.stop_hunt_target}")
    
    print(f"\n📍 KEY LEVELS CROWD WATCHING:")
    for level in sentiment.key_levels_crowd_watching[:5]:
        print(f"   ${level}")
    
    if sentiment.stop_loss_clusters:
        print(f"\n🛑 STOP LOSS CLUSTERS:")
        for sl in sentiment.stop_loss_clusters[:5]:
            print(f"   ${sl}")
    
    print(f"\n📡 SOURCE STATUS:")
    print(f"   Telegram: {sentiment.telegram_sentiment.get('direction', 'N/A')}")
    print(f"   YouTube: {sentiment.youtube_sentiment.get('direction', 'N/A')}")
    print(f"   Twitter: {sentiment.twitter_sentiment.get('direction', 'N/A')}")
    print(f"   Reddit: {sentiment.reddit_sentiment.get('direction', 'N/A')}")
    print(f"\n   Data Quality: {sentiment.data_quality}")
    print(f"   Sources Active: {sentiment.sources_active}/4")

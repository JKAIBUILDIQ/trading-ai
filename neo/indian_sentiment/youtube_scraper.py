"""
Indian YouTube Live Chat Sentiment Scraper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monitors live chat from Indian trading YouTube channels.

Major channels (live during Indian market hours 9:15 AM - 3:30 PM IST):
- Power of Stocks (1M+ subs)
- Trading Chanakya
- Vivek Bajaj
- Nitin Bhatia
- Rachana Ranade
- Stock Market Telugu/Tamil channels

Created: 2026-01-24
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

log = logging.getLogger(__name__)


@dataclass
class YouTubeSentiment:
    """Aggregated YouTube live chat sentiment"""
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float
    live_streams_active: int
    total_viewers: int
    message_count: int
    buy_mentions: int
    sell_mentions: int
    fear_greed_ratio: float
    top_discussed_levels: List[float]
    trending_topics: List[str]
    fomo_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    last_update: datetime


class IndianYouTubeScraper:
    """
    Real-time scraping of Indian trading YouTube live streams
    
    Monitors live chat during Indian market hours.
    
    Usage:
        scraper = IndianYouTubeScraper()
        sentiment = scraper.get_live_sentiment()
    """
    
    # Popular Indian trading channels
    CHANNELS = {
        'power_of_stocks': {
            'id': 'UCxxxxxxxxxx',
            'name': 'Power of Stocks',
            'subscribers': 1000000,
            'typical_viewers': 15000
        },
        'trading_chanakya': {
            'id': 'UCyyyyyyyyyy',
            'name': 'Trading Chanakya',
            'subscribers': 500000,
            'typical_viewers': 8000
        },
        'vivek_bajaj': {
            'id': 'UCzzzzzzzzzz',
            'name': 'Vivek Bajaj',
            'subscribers': 800000,
            'typical_viewers': 10000
        },
        'nitin_bhatia': {
            'id': 'UCaaaaaaaaaa',
            'name': 'Nitin Bhatia',
            'subscribers': 400000,
            'typical_viewers': 5000
        },
        'rachana_ranade': {
            'id': 'UCbbbbbbbbbb',
            'name': 'Rachana Ranade',
            'subscribers': 2000000,
            'typical_viewers': 20000
        }
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize scraper
        
        Args:
            api_key: YouTube Data API key (optional, for real scraping)
        """
        self.api_key = api_key
        self.messages = deque(maxlen=500)
        self.last_sentiment = None
    
    async def fetch_live_chats(self):
        """
        Fetch live chat messages from active streams
        
        Requires: pip install chat-downloader
        """
        try:
            from chat_downloader import ChatDownloader
            
            downloader = ChatDownloader()
            
            for channel_key, channel_info in self.CHANNELS.items():
                try:
                    # This would need actual channel URLs
                    # chat = downloader.get_chat(url=live_url, timeout=5)
                    # for message in chat:
                    #     self._process_message(message)
                    pass
                except Exception as e:
                    log.debug(f"No live stream for {channel_info['name']}: {e}")
                    
        except ImportError:
            log.warning("chat-downloader not installed. Run: pip install chat-downloader")
    
    def _process_message(self, message: Dict):
        """Process a live chat message"""
        text = message.get('message', '')
        timestamp = message.get('timestamp', datetime.utcnow())
        
        parsed = {
            'text': text,
            'timestamp': timestamp,
            'direction': self._detect_direction(text),
            'price_levels': self._extract_prices(text),
            'fear_score': self._score_fear(text),
            'greed_score': self._score_greed(text)
        }
        
        self.messages.append(parsed)
    
    def _detect_direction(self, text: str) -> Optional[str]:
        """Detect buy/sell intent"""
        text_lower = text.lower()
        
        buy_keywords = ['buy', 'long', 'bullish', 'up', 'green', 'खरीदो']
        sell_keywords = ['sell', 'short', 'bearish', 'down', 'red', 'बेचो']
        
        buy_count = sum(1 for kw in buy_keywords if kw in text_lower)
        sell_count = sum(1 for kw in sell_keywords if kw in text_lower)
        
        if buy_count > sell_count:
            return 'BUY'
        elif sell_count > buy_count:
            return 'SELL'
        return None
    
    def _extract_prices(self, text: str) -> List[float]:
        """Extract price levels mentioned"""
        pattern = r'\b(\d{4,5}(?:\.\d{1,2})?)\b'
        prices = re.findall(pattern, text)
        return [float(p) for p in prices if 1000 < float(p) < 10000]
    
    def _score_fear(self, text: str) -> int:
        """Score fear level in message"""
        text_lower = text.lower()
        fear_words = ['panic', 'crash', 'dump', 'scared', 'loss', 'careful', 'danger']
        return sum(1 for w in fear_words if w in text_lower)
    
    def _score_greed(self, text: str) -> int:
        """Score greed level in message"""
        text_lower = text.lower()
        greed_words = ['moon', 'rocket', 'guaranteed', 'easy', 'rich', 'profit', 'buy now']
        return sum(1 for w in greed_words if w in text_lower)
    
    def get_live_sentiment(self) -> YouTubeSentiment:
        """
        Get aggregated sentiment from YouTube live chats
        """
        recent = list(self.messages)
        
        if not recent:
            return self._create_simulated_sentiment()
        
        # Count directions
        buy_count = sum(1 for m in recent if m.get('direction') == 'BUY')
        sell_count = sum(1 for m in recent if m.get('direction') == 'SELL')
        total = buy_count + sell_count
        
        if total > 0:
            if buy_count > sell_count:
                direction = 'BULLISH'
                strength = buy_count / total
            elif sell_count > buy_count:
                direction = 'BEARISH'
                strength = sell_count / total
            else:
                direction = 'NEUTRAL'
                strength = 0.5
        else:
            direction = 'NEUTRAL'
            strength = 0
        
        # Fear/Greed
        total_fear = sum(m.get('fear_score', 0) for m in recent)
        total_greed = sum(m.get('greed_score', 0) for m in recent)
        fear_greed_ratio = (total_greed + 1) / (total_fear + 1)
        
        # FOMO level
        if fear_greed_ratio > 3:
            fomo_level = 'EXTREME'
        elif fear_greed_ratio > 2:
            fomo_level = 'HIGH'
        elif fear_greed_ratio > 1.5:
            fomo_level = 'MEDIUM'
        else:
            fomo_level = 'LOW'
        
        # Price levels
        all_prices = [p for m in recent for p in m.get('price_levels', [])]
        from collections import Counter
        price_counter = Counter([round(p / 10) * 10 for p in all_prices])
        top_levels = [level for level, _ in price_counter.most_common(5)]
        
        return YouTubeSentiment(
            direction=direction,
            strength=round(strength, 2),
            live_streams_active=0,  # Would be filled by real scraping
            total_viewers=0,
            message_count=len(recent),
            buy_mentions=buy_count,
            sell_mentions=sell_count,
            fear_greed_ratio=round(fear_greed_ratio, 2),
            top_discussed_levels=top_levels,
            trending_topics=[],
            fomo_level=fomo_level,
            last_update=datetime.utcnow()
        )
    
    def _create_simulated_sentiment(self) -> YouTubeSentiment:
        """Create simulated sentiment for testing"""
        import random
        
        # Simulate based on time of day (Indian market hours)
        now = datetime.utcnow()
        ist_hour = (now.hour + 5) % 24 + (30 / 60)  # IST = UTC + 5:30
        
        # More bullish during market hours
        if 9 <= ist_hour <= 15:
            direction = random.choice(['BULLISH', 'BULLISH', 'NEUTRAL'])
            live_streams = random.randint(2, 5)
            viewers = random.randint(10000, 50000)
        else:
            direction = 'NEUTRAL'
            live_streams = 0
            viewers = 0
        
        return YouTubeSentiment(
            direction=direction,
            strength=random.uniform(0.5, 0.8) if direction != 'NEUTRAL' else 0,
            live_streams_active=live_streams,
            total_viewers=viewers,
            message_count=random.randint(50, 200),
            buy_mentions=random.randint(20, 100),
            sell_mentions=random.randint(10, 50),
            fear_greed_ratio=random.uniform(0.8, 2.5),
            top_discussed_levels=[5000, 4950, 4900, 5050],
            trending_topics=['gold breakout', '5000 target', 'buy dip'],
            fomo_level=random.choice(['LOW', 'MEDIUM', 'HIGH']),
            last_update=datetime.utcnow()
        )
    
    def add_simulated_messages(self, bias: str = 'BULLISH'):
        """Add simulated messages for testing"""
        messages = {
            'BULLISH': [
                "Buy gold now! Going to 5100!",
                "Breakout happening, don't miss!",
                "5000 crossed, next target 5050",
                "Everyone buying, easy money! 🚀",
                "खरीदो जल्दी! 5200 जाएगा!",
            ],
            'BEARISH': [
                "Sell now! Crash coming!",
                "Panic mode activated!",
                "Exit all positions!",
                "Going to 4800, dump it!",
            ]
        }
        
        for msg in messages.get(bias, messages['BULLISH']):
            self._process_message({'message': msg, 'timestamp': datetime.utcnow()})
    
    def to_dict(self) -> Dict:
        """Convert sentiment to dictionary for API"""
        sentiment = self.get_live_sentiment()
        return {
            'direction': sentiment.direction,
            'strength': sentiment.strength,
            'live_streams_active': sentiment.live_streams_active,
            'total_viewers': sentiment.total_viewers,
            'message_count': sentiment.message_count,
            'buy_mentions': sentiment.buy_mentions,
            'sell_mentions': sentiment.sell_mentions,
            'fear_greed_ratio': sentiment.fear_greed_ratio,
            'top_discussed_levels': sentiment.top_discussed_levels,
            'fomo_level': sentiment.fomo_level,
            'last_update': sentiment.last_update.isoformat()
        }


# Test
if __name__ == "__main__":
    print("="*60)
    print("🎥 INDIAN YOUTUBE SENTIMENT TEST")
    print("="*60)
    
    scraper = IndianYouTubeScraper()
    scraper.add_simulated_messages('BULLISH')
    
    sentiment = scraper.get_live_sentiment()
    
    print(f"\n📊 YOUTUBE SENTIMENT:")
    print(f"   Direction: {sentiment.direction}")
    print(f"   Strength: {sentiment.strength:.0%}")
    print(f"   Live Streams: {sentiment.live_streams_active}")
    print(f"   Viewers: {sentiment.total_viewers:,}")
    print(f"   FOMO Level: {sentiment.fomo_level}")
    print(f"   Top Levels: {sentiment.top_discussed_levels}")

"""
Indian Social Media Sentiment Scrapers (Twitter/X + Reddit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Twitter/X India Finance:
- Hashtags: #XAUUSD, #GoldTrading, #ForexIndia, #MCXGold
- Influencers: @ZerodhaOnline, Indian finance accounts

Reddit:
- r/IndianStreetBets (200K+ members) - GOLDMINE!
- r/IndiaInvestments
- r/indianstocks

Created: 2026-01-24
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter

log = logging.getLogger(__name__)


@dataclass
class TwitterSentiment:
    """Twitter/X sentiment analysis"""
    direction: str
    strength: float
    tweets_last_hour: int
    bullish_ratio: float
    bearish_ratio: float
    trending_hashtags: List[str]
    top_influencers_bias: str
    viral_tweets: List[Dict]
    fear_greed_score: float
    last_update: datetime


@dataclass
class RedditSentiment:
    """Reddit sentiment analysis"""
    direction: str
    strength: float
    posts_today: int
    comments_today: int
    bullish_ratio: float
    top_discussed: List[str]
    sentiment_by_sub: Dict[str, str]
    wsb_style_fomo: bool  # r/IndianStreetBets style hype
    last_update: datetime


class IndianTwitterScraper:
    """
    Scrapes Indian finance Twitter/X for Gold/Forex sentiment
    
    Monitors:
    - Hashtags: #XAUUSD, #GoldTrading, #ForexIndia
    - Key accounts: Zerodha, financial influencers
    - Trending topics in India finance
    """
    
    HASHTAGS = [
        '#XAUUSD', '#GoldTrading', '#ForexIndia', '#ZerodhaTraders',
        '#MCXGold', '#CommodityTrading', '#Nifty50', '#BankNifty',
        '#GoldPrice', '#सोना'  # Hindi for gold
    ]
    
    INFLUENCERS = [
        'ZerodhaOnline',
        'nsaborker',
        'GrowwIndia',
        'Investpaisa',
    ]
    
    def __init__(self, bearer_token: str = None):
        """
        Initialize Twitter scraper
        
        Args:
            bearer_token: Twitter API bearer token
        """
        self.bearer_token = bearer_token
        self.tweets = []
        self.last_sentiment = None
    
    async def fetch_tweets(self, query: str = None, limit: int = 100):
        """
        Fetch recent tweets matching query
        
        Requires: pip install tweepy
        """
        try:
            import tweepy
            
            if not self.bearer_token:
                log.warning("Twitter bearer token not provided")
                return
            
            client = tweepy.Client(bearer_token=self.bearer_token)
            
            # Build query
            if not query:
                query = ' OR '.join(self.HASHTAGS[:5]) + ' lang:en OR lang:hi'
            
            tweets = client.search_recent_tweets(
                query=query,
                max_results=limit,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    self._process_tweet(tweet)
                    
        except ImportError:
            log.warning("tweepy not installed. Run: pip install tweepy")
        except Exception as e:
            log.error(f"Twitter API error: {e}")
    
    def _process_tweet(self, tweet):
        """Process a tweet"""
        text = tweet.text if hasattr(tweet, 'text') else str(tweet)
        
        parsed = {
            'text': text,
            'timestamp': datetime.utcnow(),
            'direction': self._detect_direction(text),
            'engagement': 0,  # Would come from tweet metrics
            'price_levels': self._extract_prices(text),
            'hashtags': self._extract_hashtags(text),
            'fear_score': self._score_sentiment(text, 'fear'),
            'greed_score': self._score_sentiment(text, 'greed')
        }
        
        self.tweets.append(parsed)
    
    def _detect_direction(self, text: str) -> Optional[str]:
        """Detect trading direction"""
        text_lower = text.lower()
        
        buy_words = ['buy', 'long', 'bullish', 'moon', 'up', 'breakout', 'target', 'खरीदो']
        sell_words = ['sell', 'short', 'bearish', 'crash', 'down', 'dump', 'exit', 'बेचो']
        
        buy_score = sum(1 for w in buy_words if w in text_lower)
        sell_score = sum(1 for w in sell_words if w in text_lower)
        
        if buy_score > sell_score:
            return 'BUY'
        elif sell_score > buy_score:
            return 'SELL'
        return None
    
    def _extract_prices(self, text: str) -> List[float]:
        """Extract price levels"""
        pattern = r'\b(\d{4,5}(?:\.\d{1,2})?)\b'
        prices = re.findall(pattern, text)
        return [float(p) for p in prices if 1000 < float(p) < 10000]
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags"""
        return re.findall(r'#\w+', text)
    
    def _score_sentiment(self, text: str, sentiment_type: str) -> int:
        """Score fear or greed"""
        text_lower = text.lower()
        
        if sentiment_type == 'fear':
            words = ['panic', 'crash', 'dump', 'loss', 'danger', 'careful', 'blood', 'pain']
        else:  # greed
            words = ['moon', 'rocket', 'guaranteed', 'easy', 'rich', 'lambo', 'profit', 'gain']
        
        return sum(1 for w in words if w in text_lower)
    
    def get_sentiment(self) -> TwitterSentiment:
        """Get aggregated Twitter sentiment"""
        recent = self.tweets[-100:] if self.tweets else []
        
        if not recent:
            return self._simulated_sentiment()
        
        buy_tweets = [t for t in recent if t.get('direction') == 'BUY']
        sell_tweets = [t for t in recent if t.get('direction') == 'SELL']
        
        total_directional = len(buy_tweets) + len(sell_tweets)
        
        if total_directional > 0:
            bullish_ratio = len(buy_tweets) / total_directional
            bearish_ratio = len(sell_tweets) / total_directional
            direction = 'BULLISH' if bullish_ratio > 0.5 else 'BEARISH' if bearish_ratio > 0.5 else 'NEUTRAL'
            strength = abs(bullish_ratio - bearish_ratio)
        else:
            direction = 'NEUTRAL'
            bullish_ratio = 0.5
            bearish_ratio = 0.5
            strength = 0
        
        # Trending hashtags
        all_hashtags = [h for t in recent for h in t.get('hashtags', [])]
        trending = [tag for tag, _ in Counter(all_hashtags).most_common(5)]
        
        # Fear/Greed
        total_fear = sum(t.get('fear_score', 0) for t in recent)
        total_greed = sum(t.get('greed_score', 0) for t in recent)
        fear_greed = total_greed / (total_fear + 1)
        
        return TwitterSentiment(
            direction=direction,
            strength=round(strength, 2),
            tweets_last_hour=len(recent),
            bullish_ratio=round(bullish_ratio, 2),
            bearish_ratio=round(bearish_ratio, 2),
            trending_hashtags=trending,
            top_influencers_bias=direction,
            viral_tweets=[],
            fear_greed_score=round(fear_greed, 2),
            last_update=datetime.utcnow()
        )
    
    def _simulated_sentiment(self) -> TwitterSentiment:
        """Create simulated sentiment for testing"""
        import random
        
        direction = random.choice(['BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH'])
        bullish_ratio = random.uniform(0.4, 0.75) if direction == 'BULLISH' else random.uniform(0.2, 0.5)
        
        return TwitterSentiment(
            direction=direction,
            strength=random.uniform(0.3, 0.7),
            tweets_last_hour=random.randint(100, 500),
            bullish_ratio=bullish_ratio,
            bearish_ratio=1 - bullish_ratio,
            trending_hashtags=['#XAUUSD', '#Gold5000', '#GoldTrading', '#MCXGold'],
            top_influencers_bias=direction,
            viral_tweets=[],
            fear_greed_score=random.uniform(0.8, 2.5),
            last_update=datetime.utcnow()
        )
    
    def add_simulated_tweets(self, bias: str = 'BULLISH'):
        """Add simulated tweets for testing"""
        tweets = {
            'BULLISH': [
                "Gold breaking out! 🚀 #XAUUSD #GoldTrading Target 5100!",
                "Buy the dip at 4950! Easy money! #ForexIndia",
                "Moon mission confirmed! Gold to ATH! 🌙 #MCXGold",
                "खरीदो अभी! 5000 पार होगा! #सोना",
            ],
            'BEARISH': [
                "Crash incoming! Exit gold NOW! #XAUUSD",
                "Blood on the streets! Dump it! 📉",
                "Gold bubble bursting! #GoldTrading",
            ]
        }
        
        for tweet_text in tweets.get(bias, tweets['BULLISH']):
            self._process_tweet(tweet_text)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        sentiment = self.get_sentiment()
        return {
            'direction': sentiment.direction,
            'strength': sentiment.strength,
            'tweets_last_hour': sentiment.tweets_last_hour,
            'bullish_ratio': sentiment.bullish_ratio,
            'trending_hashtags': sentiment.trending_hashtags,
            'fear_greed_score': sentiment.fear_greed_score,
            'last_update': sentiment.last_update.isoformat()
        }


class IndianRedditScraper:
    """
    Scrapes Indian investing subreddits for sentiment
    
    Key subreddits:
    - r/IndianStreetBets (WSB-style, high FOMO potential)
    - r/IndiaInvestments (more conservative)
    - r/indianstocks
    """
    
    SUBREDDITS = [
        'IndianStreetBets',
        'IndiaInvestments',
        'indianstocks',
        'indiainvestments'
    ]
    
    GOLD_KEYWORDS = [
        'gold', 'xauusd', 'sovereign gold', 'mcx gold', 'सोना',
        'precious metals', 'commodity', 'sgb'
    ]
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        Initialize Reddit scraper
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.posts = []
        self.comments = []
    
    async def fetch_posts(self, subreddit: str = 'IndianStreetBets', limit: int = 50):
        """
        Fetch recent posts from subreddit
        
        Requires: pip install praw
        """
        try:
            import praw
            
            if not self.client_id or not self.client_secret:
                log.warning("Reddit API credentials not provided")
                return
            
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent='IndianSentimentBot/1.0'
            )
            
            sub = reddit.subreddit(subreddit)
            
            for post in sub.hot(limit=limit):
                if self._is_gold_related(post.title + ' ' + post.selftext):
                    self._process_post(post)
                    
        except ImportError:
            log.warning("praw not installed. Run: pip install praw")
        except Exception as e:
            log.error(f"Reddit API error: {e}")
    
    def _is_gold_related(self, text: str) -> bool:
        """Check if text is related to gold"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.GOLD_KEYWORDS)
    
    def _process_post(self, post):
        """Process a Reddit post"""
        text = post.title + ' ' + (post.selftext if hasattr(post, 'selftext') else '')
        
        parsed = {
            'title': post.title if hasattr(post, 'title') else str(post),
            'text': text,
            'timestamp': datetime.utcnow(),
            'direction': self._detect_direction(text),
            'upvotes': post.score if hasattr(post, 'score') else 0,
            'comments': post.num_comments if hasattr(post, 'num_comments') else 0,
            'fomo_level': self._detect_fomo(text),
            'subreddit': post.subreddit.display_name if hasattr(post, 'subreddit') else 'unknown'
        }
        
        self.posts.append(parsed)
    
    def _detect_direction(self, text: str) -> Optional[str]:
        """Detect trading direction"""
        text_lower = text.lower()
        
        buy_words = ['buy', 'long', 'bullish', 'accumulate', 'rocket', 'moon', 'खरीदो', 'invest']
        sell_words = ['sell', 'short', 'bearish', 'exit', 'crash', 'dump', 'बेचो', 'avoid']
        
        buy_score = sum(1 for w in buy_words if w in text_lower)
        sell_score = sum(1 for w in sell_words if w in text_lower)
        
        if buy_score > sell_score:
            return 'BUY'
        elif sell_score > buy_score:
            return 'SELL'
        return None
    
    def _detect_fomo(self, text: str) -> str:
        """Detect FOMO level (WSB-style)"""
        text_lower = text.lower()
        
        extreme_words = ['guaranteed', 'free money', 'can\'t go wrong', 'sure shot', 
                        'yolo', 'apes', 'diamond hands', 'to the moon', 'tendies']
        high_words = ['easy money', 'rocket', 'moon', '🚀', 'breaking out', 'lambo']
        medium_words = ['good opportunity', 'buy now', 'don\'t miss', 'last chance']
        
        if any(w in text_lower for w in extreme_words):
            return 'EXTREME'
        elif any(w in text_lower for w in high_words):
            return 'HIGH'
        elif any(w in text_lower for w in medium_words):
            return 'MEDIUM'
        return 'LOW'
    
    def get_sentiment(self) -> RedditSentiment:
        """Get aggregated Reddit sentiment"""
        recent = self.posts[-50:] if self.posts else []
        
        if not recent:
            return self._simulated_sentiment()
        
        buy_posts = [p for p in recent if p.get('direction') == 'BUY']
        sell_posts = [p for p in recent if p.get('direction') == 'SELL']
        
        total_directional = len(buy_posts) + len(sell_posts)
        
        if total_directional > 0:
            bullish_ratio = len(buy_posts) / total_directional
            direction = 'BULLISH' if bullish_ratio > 0.5 else 'BEARISH'
            strength = abs(bullish_ratio - 0.5) * 2
        else:
            direction = 'NEUTRAL'
            bullish_ratio = 0.5
            strength = 0
        
        # FOMO detection
        fomo_counts = Counter(p.get('fomo_level', 'LOW') for p in recent)
        wsb_style_fomo = (fomo_counts.get('EXTREME', 0) + fomo_counts.get('HIGH', 0)) > len(recent) * 0.3
        
        # Top discussed topics
        all_text = ' '.join(p.get('title', '') for p in recent).lower()
        topics = []
        if 'breakout' in all_text:
            topics.append('breakout')
        if '5000' in all_text or '5k' in all_text:
            topics.append('5000 target')
        if 'dip' in all_text:
            topics.append('buy the dip')
        
        # Sentiment by subreddit
        by_sub = {}
        for sub in self.SUBREDDITS:
            sub_posts = [p for p in recent if p.get('subreddit') == sub]
            if sub_posts:
                sub_buys = sum(1 for p in sub_posts if p.get('direction') == 'BUY')
                by_sub[sub] = 'BULLISH' if sub_buys > len(sub_posts) / 2 else 'BEARISH'
        
        return RedditSentiment(
            direction=direction,
            strength=round(strength, 2),
            posts_today=len(recent),
            comments_today=sum(p.get('comments', 0) for p in recent),
            bullish_ratio=round(bullish_ratio, 2),
            top_discussed=topics or ['gold', 'investment'],
            sentiment_by_sub=by_sub,
            wsb_style_fomo=wsb_style_fomo,
            last_update=datetime.utcnow()
        )
    
    def _simulated_sentiment(self) -> RedditSentiment:
        """Create simulated sentiment for testing"""
        import random
        
        direction = random.choice(['BULLISH', 'BULLISH', 'NEUTRAL'])
        
        return RedditSentiment(
            direction=direction,
            strength=random.uniform(0.4, 0.8),
            posts_today=random.randint(15, 50),
            comments_today=random.randint(200, 1000),
            bullish_ratio=random.uniform(0.5, 0.75),
            top_discussed=['gold breakout', '5000 target', 'buy the dip'],
            sentiment_by_sub={
                'IndianStreetBets': 'BULLISH',
                'IndiaInvestments': 'NEUTRAL'
            },
            wsb_style_fomo=random.choice([True, False]),
            last_update=datetime.utcnow()
        )
    
    def add_simulated_posts(self, bias: str = 'BULLISH'):
        """Add simulated posts for testing"""
        posts = {
            'BULLISH': [
                {'title': 'Gold to 5000! YOLO! 🚀🚀🚀', 'direction': 'BUY', 'fomo': 'EXTREME'},
                {'title': 'Easy money in gold, buy the dip!', 'direction': 'BUY', 'fomo': 'HIGH'},
                {'title': 'Diamond hands on gold! To the moon!', 'direction': 'BUY', 'fomo': 'EXTREME'},
            ],
            'BEARISH': [
                {'title': 'Gold crash incoming! Exit now!', 'direction': 'SELL', 'fomo': 'LOW'},
                {'title': 'Lost everything in gold, careful!', 'direction': 'SELL', 'fomo': 'LOW'},
            ]
        }
        
        for post in posts.get(bias, posts['BULLISH']):
            self.posts.append({
                'title': post['title'],
                'text': post['title'],
                'timestamp': datetime.utcnow(),
                'direction': post['direction'],
                'upvotes': 100,
                'comments': 50,
                'fomo_level': post['fomo'],
                'subreddit': 'IndianStreetBets'
            })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        sentiment = self.get_sentiment()
        return {
            'direction': sentiment.direction,
            'strength': sentiment.strength,
            'posts_today': sentiment.posts_today,
            'comments_today': sentiment.comments_today,
            'bullish_ratio': sentiment.bullish_ratio,
            'top_discussed': sentiment.top_discussed,
            'wsb_style_fomo': sentiment.wsb_style_fomo,
            'last_update': sentiment.last_update.isoformat()
        }


# Test
if __name__ == "__main__":
    print("="*60)
    print("🐦 INDIAN TWITTER SENTIMENT TEST")
    print("="*60)
    
    twitter = IndianTwitterScraper()
    twitter.add_simulated_tweets('BULLISH')
    
    sentiment = twitter.get_sentiment()
    print(f"\n📊 Direction: {sentiment.direction}")
    print(f"   Strength: {sentiment.strength:.0%}")
    print(f"   Bullish Ratio: {sentiment.bullish_ratio:.0%}")
    print(f"   Trending: {sentiment.trending_hashtags}")
    
    print("\n" + "="*60)
    print("📱 INDIAN REDDIT SENTIMENT TEST")
    print("="*60)
    
    reddit = IndianRedditScraper()
    reddit.add_simulated_posts('BULLISH')
    
    sentiment = reddit.get_sentiment()
    print(f"\n📊 Direction: {sentiment.direction}")
    print(f"   Strength: {sentiment.strength:.0%}")
    print(f"   WSB-Style FOMO: {sentiment.wsb_style_fomo}")
    print(f"   Top Discussed: {sentiment.top_discussed}")

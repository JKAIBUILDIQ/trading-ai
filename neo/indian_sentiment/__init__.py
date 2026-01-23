"""
Indian Forex Sentiment Intelligence Module
==========================================

Real-time sentiment analysis from Indian trading communities:
- Telegram groups (highest value!)
- YouTube live chats
- Twitter/X finance
- Reddit r/IndianStreetBets

Why India?
- 100M+ retail trading accounts
- Zerodha = World's largest broker by clients
- Same YouTube tutorials = Same strategies
- Predictable herd behavior

Usage:
    from indian_sentiment import IndianSentimentAggregator
    
    aggregator = IndianSentimentAggregator()
    sentiment = aggregator.get_live_sentiment()
"""

from .telegram_scraper import IndianTelegramScraper, TelegramSentiment
from .youtube_scraper import IndianYouTubeScraper
from .social_scraper import IndianTwitterScraper, IndianRedditScraper
from .aggregator import IndianSentimentAggregator, get_indian_sentiment

__all__ = [
    'IndianTelegramScraper',
    'IndianYouTubeScraper', 
    'IndianTwitterScraper',
    'IndianRedditScraper',
    'IndianSentimentAggregator',
    'get_indian_sentiment',
    'TelegramSentiment'
]

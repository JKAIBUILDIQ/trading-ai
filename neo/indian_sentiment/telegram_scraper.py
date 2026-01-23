"""
Indian Telegram Sentiment Scraper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monitors Indian forex/gold trading Telegram groups in real-time.

Why Telegram?
- Highest signal value for Indian traders
- Real-time discussion during market hours
- Stop loss levels mentioned explicitly
- Pure unfiltered retail sentiment

Groups to monitor:
- Gold Silver Trading India
- XAUUSD Signals India
- Zerodha Traders
- Bank Nifty Premium
- MCX Commodity Traders

Created: 2026-01-24
"""

import re
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import asyncio
import json

log = logging.getLogger(__name__)

# Keywords in multiple languages (English + Hindi)
BUY_KEYWORDS = [
    'buy', 'long', 'bullish', 'bull', 'entry', 'breakout', 'support',
    'accumulate', 'add', 'buying', 'bought', 'going up',
    # Hindi
    'खरीदो', 'लॉन्ग', 'बुलिश', 'एंट्री', 'ब्रेकआउट', 'सपोर्ट'
]

SELL_KEYWORDS = [
    'sell', 'short', 'bearish', 'bear', 'exit', 'breakdown', 'resistance',
    'dump', 'selling', 'sold', 'going down', 'crash',
    # Hindi
    'बेचो', 'शॉर्ट', 'बेयरिश', 'एग्जिट', 'रेजिस्टेंस'
]

FEAR_KEYWORDS = [
    'panic', 'crash', 'dump', 'scared', 'danger', 'careful', 'risk',
    'loss', 'losing', 'stop hit', 'blown', 'margin call', 'wipe out',
    'डर', 'खतरा', 'नुकसान'
]

GREED_KEYWORDS = [
    'moon', 'rocket', 'guaranteed', 'easy money', 'sure shot', 'free money',
    'double', 'triple', 'lakhs', 'crores', 'rich', 'lambo',
    'पक्का', 'गारंटी', 'आसान पैसा'
]


@dataclass
class TelegramMessage:
    """Parsed Telegram message"""
    text: str
    timestamp: datetime
    channel: str
    direction: Optional[str] = None  # 'BUY', 'SELL', None
    confidence: float = 0.0
    sl_mentioned: Optional[float] = None
    tp_mentioned: Optional[float] = None
    price_levels: List[float] = field(default_factory=list)
    fear_score: int = 0
    greed_score: int = 0


@dataclass
class TelegramSentiment:
    """Aggregated Telegram sentiment"""
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0-1
    message_count: int
    buy_count: int
    sell_count: int
    fear_greed_ratio: float  # >1 = greedy, <1 = fearful
    top_sl_levels: List[float]
    top_tp_levels: List[float]
    key_price_levels: List[float]
    fomo_detected: bool
    panic_detected: bool
    last_update: datetime


class IndianTelegramScraper:
    """
    Real-time scraping of Indian trading Telegram groups
    
    Usage:
        scraper = IndianTelegramScraper(api_id, api_hash)
        await scraper.start()
        sentiment = scraper.get_current_sentiment()
    """
    
    # Public Indian trading groups (search these on Telegram)
    DEFAULT_GROUPS = [
        'gold_silver_trading_india',
        'xauusd_signals_india', 
        'zerodha_traders_community',
        'bank_nifty_premium_official',
        'mcx_commodity_traders',
        'forex_india_traders',
        'intraday_gold_calls',
        'comex_gold_silver',
    ]
    
    def __init__(self, api_id: str = None, api_hash: str = None,
                 groups: List[str] = None, message_window: int = 60):
        """
        Initialize scraper
        
        Args:
            api_id: Telegram API ID (get from my.telegram.org)
            api_hash: Telegram API hash
            groups: List of group usernames to monitor
            message_window: Minutes of messages to keep in memory
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.groups = groups or self.DEFAULT_GROUPS
        self.message_window = message_window
        
        # Message buffer (last N minutes)
        self.messages: deque = deque(maxlen=1000)
        self.last_sentiment: Optional[TelegramSentiment] = None
        
        # Telethon client (requires: pip install telethon)
        self.client = None
        self.running = False
    
    async def start(self):
        """
        Start monitoring Telegram groups
        
        Requires telethon: pip install telethon
        """
        try:
            from telethon import TelegramClient, events
            
            if not self.api_id or not self.api_hash:
                log.warning("Telegram API credentials not provided. Using simulation mode.")
                return
            
            self.client = TelegramClient('indian_sentiment', self.api_id, self.api_hash)
            await self.client.start()
            
            # Register message handler
            @self.client.on(events.NewMessage(chats=self.groups))
            async def message_handler(event):
                try:
                    message = self._parse_message(event.message.text, 
                                                  event.chat.username or 'unknown')
                    self.messages.append(message)
                    log.debug(f"New message from {event.chat.username}: {message.direction}")
                except Exception as e:
                    log.error(f"Error parsing message: {e}")
            
            self.running = True
            log.info(f"Started monitoring {len(self.groups)} Telegram groups")
            
            # Keep running
            await self.client.run_until_disconnected()
            
        except ImportError:
            log.warning("telethon not installed. Run: pip install telethon")
        except Exception as e:
            log.error(f"Telegram scraper error: {e}")
    
    def _parse_message(self, text: str, channel: str) -> TelegramMessage:
        """
        Parse a Telegram message for trading intent
        """
        if not text:
            return TelegramMessage(text="", timestamp=datetime.utcnow(), channel=channel)
        
        text_lower = text.lower()
        
        # Direction detection
        buy_score = sum(1 for kw in BUY_KEYWORDS if kw in text_lower)
        sell_score = sum(1 for kw in SELL_KEYWORDS if kw in text_lower)
        
        if buy_score > sell_score:
            direction = 'BUY'
            confidence = min(1.0, buy_score / 3)
        elif sell_score > buy_score:
            direction = 'SELL'
            confidence = min(1.0, sell_score / 3)
        else:
            direction = None
            confidence = 0
        
        # Extract price levels (4-5 digit numbers for Gold)
        price_pattern = r'\b(\d{4,5}(?:\.\d{1,2})?)\b'
        prices = re.findall(price_pattern, text)
        price_levels = [float(p) for p in prices if 1000 < float(p) < 10000]
        
        # Extract SL (stop loss)
        sl_pattern = r'(?:sl|stop(?:\s*loss)?|स्टॉप)[:\s]*(\d{4,5}(?:\.\d{1,2})?)'
        sl_match = re.search(sl_pattern, text_lower)
        sl_mentioned = float(sl_match.group(1)) if sl_match else None
        
        # Extract TP (take profit)
        tp_pattern = r'(?:tp|target|t(?:ake)?p(?:rofit)?|टारगेट)[:\s]*(\d{4,5}(?:\.\d{1,2})?)'
        tp_match = re.search(tp_pattern, text_lower)
        tp_mentioned = float(tp_match.group(1)) if tp_match else None
        
        # Fear/Greed scores
        fear_score = sum(1 for kw in FEAR_KEYWORDS if kw in text_lower)
        greed_score = sum(1 for kw in GREED_KEYWORDS if kw in text_lower)
        
        return TelegramMessage(
            text=text,
            timestamp=datetime.utcnow(),
            channel=channel,
            direction=direction,
            confidence=confidence,
            sl_mentioned=sl_mentioned,
            tp_mentioned=tp_mentioned,
            price_levels=price_levels,
            fear_score=fear_score,
            greed_score=greed_score
        )
    
    def get_current_sentiment(self) -> TelegramSentiment:
        """
        Get aggregated sentiment from recent messages
        """
        cutoff = datetime.utcnow() - timedelta(minutes=self.message_window)
        recent = [m for m in self.messages if m.timestamp > cutoff]
        
        if not recent:
            return TelegramSentiment(
                direction='NEUTRAL',
                strength=0,
                message_count=0,
                buy_count=0,
                sell_count=0,
                fear_greed_ratio=1.0,
                top_sl_levels=[],
                top_tp_levels=[],
                key_price_levels=[],
                fomo_detected=False,
                panic_detected=False,
                last_update=datetime.utcnow()
            )
        
        # Count directions
        buy_messages = [m for m in recent if m.direction == 'BUY']
        sell_messages = [m for m in recent if m.direction == 'SELL']
        
        buy_count = len(buy_messages)
        sell_count = len(sell_messages)
        total_directional = buy_count + sell_count
        
        # Direction and strength
        if total_directional > 0:
            if buy_count > sell_count:
                direction = 'BULLISH'
                strength = buy_count / total_directional
            elif sell_count > buy_count:
                direction = 'BEARISH'
                strength = sell_count / total_directional
            else:
                direction = 'NEUTRAL'
                strength = 0.5
        else:
            direction = 'NEUTRAL'
            strength = 0
        
        # Fear/Greed
        total_fear = sum(m.fear_score for m in recent)
        total_greed = sum(m.greed_score for m in recent)
        fear_greed_ratio = (total_greed + 1) / (total_fear + 1)  # Avoid division by zero
        
        # Extract SL/TP clusters
        sl_levels = [m.sl_mentioned for m in recent if m.sl_mentioned]
        tp_levels = [m.tp_mentioned for m in recent if m.tp_mentioned]
        all_prices = [p for m in recent for p in m.price_levels]
        
        # Get most common levels
        def get_top_levels(levels: List[float], n: int = 5) -> List[float]:
            if not levels:
                return []
            from collections import Counter
            # Round to nearest 10 for clustering
            rounded = [round(l / 10) * 10 for l in levels]
            common = Counter(rounded).most_common(n)
            return [level for level, count in common]
        
        # FOMO/Panic detection
        fomo_detected = total_greed > 5 and fear_greed_ratio > 2.0
        panic_detected = total_fear > 5 and fear_greed_ratio < 0.5
        
        sentiment = TelegramSentiment(
            direction=direction,
            strength=round(strength, 2),
            message_count=len(recent),
            buy_count=buy_count,
            sell_count=sell_count,
            fear_greed_ratio=round(fear_greed_ratio, 2),
            top_sl_levels=get_top_levels(sl_levels),
            top_tp_levels=get_top_levels(tp_levels),
            key_price_levels=get_top_levels(all_prices),
            fomo_detected=fomo_detected,
            panic_detected=panic_detected,
            last_update=datetime.utcnow()
        )
        
        self.last_sentiment = sentiment
        return sentiment
    
    def add_simulated_message(self, text: str, channel: str = 'simulation'):
        """
        Add a simulated message for testing
        """
        message = self._parse_message(text, channel)
        self.messages.append(message)
    
    def simulate_market_session(self, bias: str = 'BULLISH'):
        """
        Simulate a typical Indian market session for testing
        """
        sample_messages = {
            'BULLISH': [
                "Gold looking strong! Buy at 4950 SL 4920 Target 5000 🚀",
                "Breakout confirmed above 4970, going to 5050 soon",
                "Entry at 4960, SL 4930. Moon incoming! 🌙",
                "Everyone buying gold today, easy money!",
                "खरीदो 4955 पर, स्टॉप 4925, टारगेट 5000",
                "Gold to 5100! Sure shot trade!",
                "Bought at 4965, target 5020, sl 4940",
                "Bull run continues! Add more longs!",
            ],
            'BEARISH': [
                "Sell gold at 4980, SL 5010, Target 4900",
                "Breakdown coming, short at resistance",
                "Panic selling! Dump incoming! 📉",
                "Exit all longs NOW! Crash alert!",
                "बेचो 4970 पर, स्टॉप 5000, टारगेट 4850",
                "Gold going down! Short it!",
                "Sold at 4975, target 4900",
                "Bear market starting, careful!",
            ],
            'MIXED': [
                "Range bound market, be careful",
                "Wait for breakout either side",
                "4950 support, 5000 resistance",
                "Sideways movement expected",
            ]
        }
        
        messages = sample_messages.get(bias, sample_messages['MIXED'])
        for msg in messages:
            self.add_simulated_message(msg)
    
    def to_dict(self) -> Dict:
        """Convert current sentiment to dictionary for API"""
        sentiment = self.get_current_sentiment()
        return {
            'direction': sentiment.direction,
            'strength': sentiment.strength,
            'message_count': sentiment.message_count,
            'buy_count': sentiment.buy_count,
            'sell_count': sentiment.sell_count,
            'fear_greed_ratio': sentiment.fear_greed_ratio,
            'top_sl_levels': sentiment.top_sl_levels,
            'top_tp_levels': sentiment.top_tp_levels,
            'key_price_levels': sentiment.key_price_levels,
            'fomo_detected': sentiment.fomo_detected,
            'panic_detected': sentiment.panic_detected,
            'last_update': sentiment.last_update.isoformat()
        }


# Test
if __name__ == "__main__":
    print("="*60)
    print("🇮🇳 INDIAN TELEGRAM SENTIMENT TEST")
    print("="*60)
    
    scraper = IndianTelegramScraper()
    
    # Simulate bullish session
    print("\n📈 Simulating BULLISH session...")
    scraper.simulate_market_session('BULLISH')
    
    sentiment = scraper.get_current_sentiment()
    
    print(f"\n📊 SENTIMENT ANALYSIS:")
    print(f"   Direction: {sentiment.direction}")
    print(f"   Strength: {sentiment.strength:.0%}")
    print(f"   Messages: {sentiment.message_count}")
    print(f"   Buy/Sell: {sentiment.buy_count}/{sentiment.sell_count}")
    print(f"   Fear/Greed Ratio: {sentiment.fear_greed_ratio:.2f}")
    print(f"   FOMO Detected: {sentiment.fomo_detected}")
    print(f"   Panic Detected: {sentiment.panic_detected}")
    
    if sentiment.top_sl_levels:
        print(f"\n🛑 Stop Loss Clusters:")
        for sl in sentiment.top_sl_levels:
            print(f"   ${sl}")
    
    if sentiment.key_price_levels:
        print(f"\n📍 Key Levels Discussed:")
        for level in sentiment.key_price_levels:
            print(f"   ${level}")

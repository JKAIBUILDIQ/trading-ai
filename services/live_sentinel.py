"""
LIVE SENTINEL AGENT
═══════════════════
Never sleeps. Always watches. Instant alerts.

This agent monitors the market in REAL-TIME using:
1. Continuous data feeds (polling or WebSocket)
2. TA-Lib pattern detection (61 patterns)
3. Claude Vision confirmation (on critical patterns)
4. Instant push alerts (Telegram, War Room, Ghost)
5. Optional auto-execution
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import pandas as pd
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveSentinel")

# Import pattern detector
import sys
sys.path.insert(0, '/home/jbot/trading_ai')
from api.talib_patterns import TALibPatternDetector

# Optional imports
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Anthropic not available - Claude confirmation disabled")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available")


class LiveSentinel:
    """
    Real-time trading sentinel powered by TA-Lib and Claude.
    
    Watches market 24/7, detects patterns instantly, 
    confirms with AI, alerts immediately, can auto-execute.
    """
    
    def __init__(
        self,
        symbols: List[str] = ["GC=F"],
        check_interval: int = 60,  # Seconds between checks
        telegram_token: str = None,
        telegram_chat_id: str = None,
        war_room_api_url: str = "http://localhost:8889",
        ghost_api_url: str = "http://localhost:8890",
        auto_execute: bool = False,
        claude_confirm_critical: bool = True,
    ):
        self.symbols = symbols
        self.check_interval = check_interval
        self.detector = TALibPatternDetector()
        
        # Claude client
        self.claude = anthropic.Anthropic() if CLAUDE_AVAILABLE else None
        self.claude_confirm_critical = claude_confirm_critical
        
        # Alert channels
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_bot = None
        if TELEGRAM_AVAILABLE and self.telegram_token:
            try:
                self.telegram_bot = Bot(token=self.telegram_token)
            except Exception as e:
                logger.warning(f"Telegram bot init failed: {e}")
        
        self.war_room_api_url = war_room_api_url
        self.ghost_api_url = ghost_api_url
        
        # State
        self.watching = True
        self.auto_execute = auto_execute
        self.candle_history: Dict[str, deque] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=15)  # Prevent alert spam
        self.last_patterns: Dict[str, List[Dict]] = {}
        
        # Stats
        self.patterns_detected = 0
        self.alerts_sent = 0
        self.auto_executions = 0
        self.started_at = None
        
        # Initialize candle history for each symbol
        for symbol in symbols:
            self.candle_history[symbol] = deque(maxlen=100)
            self.last_patterns[symbol] = []
    
    async def start(self):
        """Start the sentinel - runs forever."""
        
        self.started_at = datetime.now()
        
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║           👁️  LIVE SENTINEL AGENT ACTIVATED  👁️                   ║")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print(f"║  Watching: {', '.join(self.symbols):<52} ║")
        print(f"║  Patterns: {self.detector.talib_available and '61 TA-Lib' or 'Fallback':<52} ║")
        print(f"║  Check Interval: {self.check_interval}s{' ':<48}║")
        print(f"║  Auto-Execute: {'ENABLED' if self.auto_execute else 'DISABLED':<48} ║")
        print(f"║  Telegram: {'ENABLED' if self.telegram_bot else 'DISABLED':<52} ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        
        # Run parallel watchers for each symbol
        tasks = [self.watch_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)
    
    async def watch_symbol(self, symbol: str):
        """Watch a single symbol continuously."""
        
        logger.info(f"[SENTINEL] Starting watcher for {symbol}")
        
        while self.watching:
            try:
                # Fetch latest candles
                df = await self.fetch_candles(symbol)
                
                if df is not None and len(df) >= 3:
                    # Update history
                    self.update_candle_history(symbol, df)
                    
                    # Detect patterns on latest candle
                    patterns = self.detector.detect_patterns(df, priority_only=False)
                    
                    # Filter to only new patterns (not seen in last check)
                    new_patterns = self.filter_new_patterns(symbol, patterns)
                    
                    if new_patterns:
                        self.patterns_detected += len(new_patterns)
                        
                        # Check for high-priority patterns
                        critical_patterns = [p for p in new_patterns if p['severity'] in ['CRITICAL', 'HIGH']]
                        
                        if critical_patterns:
                            # Check cooldown
                            if self.can_send_alert(symbol):
                                logger.info(f"[SENTINEL] {len(critical_patterns)} critical/high patterns on {symbol}")
                                
                                # For CRITICAL patterns, get Claude confirmation
                                if any(p['severity'] == 'CRITICAL' for p in critical_patterns):
                                    if self.claude_confirm_critical and self.claude:
                                        confirmed = await self.claude_confirm(symbol, critical_patterns, df)
                                        if confirmed.get('confirmed'):
                                            await self.send_alert(symbol, critical_patterns, confirmed)
                                        else:
                                            logger.info(f"[SENTINEL] Claude rejected pattern on {symbol}")
                                    else:
                                        await self.send_alert(symbol, critical_patterns)
                                else:
                                    # HIGH patterns - alert directly
                                    await self.send_alert(symbol, critical_patterns)
                    
                    # Store patterns for next comparison
                    self.last_patterns[symbol] = patterns
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"[SENTINEL] Error watching {symbol}: {e}")
                await asyncio.sleep(30)  # Longer wait on error
    
    def filter_new_patterns(self, symbol: str, patterns: List[Dict]) -> List[Dict]:
        """Filter to only patterns not seen in last check."""
        
        last = self.last_patterns.get(symbol, [])
        last_names = {p['pattern'] for p in last}
        
        return [p for p in patterns if p['pattern'] not in last_names]
    
    async def fetch_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch latest candle data."""
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1h")
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            logger.error(f"[SENTINEL] Fetch error for {symbol}: {e}")
            return None
    
    def update_candle_history(self, symbol: str, df: pd.DataFrame):
        """Update candle history for a symbol."""
        
        latest = df.iloc[-1]
        candle = {
            'time': datetime.now(),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest.get('volume', 0)),
        }
        self.candle_history[symbol].append(candle)
    
    def can_send_alert(self, symbol: str) -> bool:
        """Check if we can send alert (cooldown check)."""
        
        last = self.last_alert_time.get(symbol)
        if last is None:
            return True
        return datetime.now() - last > self.alert_cooldown
    
    async def claude_confirm(
        self, 
        symbol: str, 
        patterns: List[Dict], 
        df: pd.DataFrame
    ) -> Dict:
        """
        Have Claude confirm the pattern.
        Returns confirmation and recommended action.
        """
        
        if not self.claude:
            return {'confirmed': True, 'action': 'HOLD', 'confidence': 5}
        
        pattern_names = [p['pattern'] for p in patterns]
        current_price = df['close'].iloc[-1]
        
        prompt = f"""
URGENT PATTERN ALERT - REAL-TIME ANALYSIS NEEDED

Symbol: {symbol}
Current Price: ${current_price:,.2f}
Patterns Detected by TA-Lib: {', '.join(pattern_names)}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Recent Price Action:
- Last 5 closes: {[round(float(df['close'].iloc[i]), 2) for i in range(-5, 0)]}
- Last 5 highs: {[round(float(df['high'].iloc[i]), 2) for i in range(-5, 0)]}
- Last 5 lows: {[round(float(df['low'].iloc[i]), 2) for i in range(-5, 0)]}

TASK:
1. CONFIRM or REJECT these pattern detections
2. If confirmed, recommend IMMEDIATE action:
   - CLOSE_ALL: Emergency exit all positions
   - CLOSE_PARTIAL: Close 50-75% of positions
   - TIGHTEN_STOPS: Protect profits with tighter stops
   - HOLD: Pattern valid but no immediate action needed
3. Confidence level (1-10)
4. Brief reasoning (1 sentence)

Be decisive. Trader's capital is at risk. This is REAL-TIME monitoring.

Respond in JSON format only:
{{"confirmed": true, "action": "CLOSE_PARTIAL", "confidence": 8, "reasoning": "Strong bearish reversal confirmed"}}
"""
        
        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            
            return {'confirmed': False, 'action': 'HOLD', 'confidence': 0}
            
        except Exception as e:
            logger.error(f"[SENTINEL] Claude confirmation error: {e}")
            # On error, treat CRITICAL patterns as confirmed for safety
            if any(p['severity'] == 'CRITICAL' for p in patterns):
                return {
                    'confirmed': True, 
                    'action': 'CLOSE_PARTIAL', 
                    'confidence': 5, 
                    'reasoning': 'Auto-confirmed due to CRITICAL severity (Claude error)'
                }
            return {'confirmed': False, 'action': 'HOLD', 'confidence': 0}
    
    async def send_alert(
        self, 
        symbol: str, 
        patterns: List[Dict], 
        confirmation: Dict = None
    ):
        """Send alert to all channels simultaneously."""
        
        self.last_alert_time[symbol] = datetime.now()
        self.alerts_sent += 1
        
        # Build alert message
        ghost_action = self.detector.get_ghost_action(patterns)
        
        alert = {
            'type': 'SENTINEL_ALERT',
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'patterns': [p['pattern'] for p in patterns],
            'directions': [p['direction'] for p in patterns],
            'severities': [p['severity'] for p in patterns],
            'current_price': patterns[0]['price'] if patterns else None,
            'ghost_action': ghost_action,
            'claude_confirmation': confirmation,
        }
        
        logger.info(f"[SENTINEL] Sending alert for {symbol}: {[p['pattern'] for p in patterns]}")
        
        # Send to all channels in parallel
        await asyncio.gather(
            self.send_telegram(alert),
            self.send_war_room(alert),
            self.save_to_db(alert),
            return_exceptions=True,
        )
        
        # Auto-execute if enabled
        if self.auto_execute and ghost_action['action'] not in ['NONE', 'OPPORTUNITY']:
            await self.execute_action(ghost_action)
    
    async def send_telegram(self, alert: Dict):
        """Send alert to Telegram."""
        
        if not self.telegram_bot or not self.telegram_chat_id:
            return
        
        try:
            severity_emoji = {
                'CRITICAL': '🚨🚨🚨',
                'HIGH': '⚠️⚠️',
                'MEDIUM': '📊',
                'LOW': '📝',
            }
            
            max_severity = max(
                alert['severities'], 
                key=lambda x: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x)
            )
            emoji = severity_emoji.get(max_severity, '📊')
            
            message = f"""
{emoji} SENTINEL ALERT {emoji}

Symbol: {alert['symbol']}
Price: ${alert['current_price']:,.2f}
Patterns: {', '.join(alert['patterns'])}
Direction: {', '.join(set(alert['directions']))}

Ghost Action: {alert['ghost_action']['action']}
{alert['ghost_action']['message']}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
            
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
            )
            logger.info(f"[SENTINEL] Telegram alert sent for {alert['symbol']}")
            
        except Exception as e:
            logger.error(f"[SENTINEL] Telegram error: {e}")
    
    async def send_war_room(self, alert: Dict):
        """Send alert to War Room API."""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.war_room_api_url}/war-room/alert",
                    json=alert,
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.info(f"[SENTINEL] War Room alert sent")
        except Exception as e:
            logger.debug(f"[SENTINEL] War Room API error: {e}")
    
    async def save_to_db(self, alert: Dict):
        """Save alert to MongoDB."""
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            client = AsyncIOMotorClient(mongo_uri)
            db = client.trading_ai
            
            await db.sentinel_alerts.insert_one({
                **alert,
                'created_at': datetime.now(),
            })
            
            logger.info(f"[SENTINEL] Alert saved to MongoDB")
        except Exception as e:
            logger.debug(f"[SENTINEL] MongoDB error: {e}")
    
    async def execute_action(self, action: Dict):
        """Auto-execute the recommended action."""
        
        self.auto_executions += 1
        
        logger.warning(f"[SENTINEL] AUTO-EXECUTING: {action['action']}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ghost_api_url}/signals/execute-action",
                    json={
                        'action': action['action'],
                        'close_percent': action.get('close_percent', 0),
                        'tighten_stops': action.get('tighten_stops', False),
                        'lot_multiplier': action.get('lot_multiplier', 1.0),
                        'source': 'LIVE_SENTINEL',
                        'timestamp': datetime.now().isoformat(),
                    },
                    timeout=10,
                )
                logger.info(f"[SENTINEL] Execution result: {response.status_code}")
        except Exception as e:
            logger.error(f"[SENTINEL] Execution error: {e}")
    
    def get_stats(self) -> Dict:
        """Get sentinel statistics."""
        
        uptime = None
        if self.started_at:
            uptime = str(datetime.now() - self.started_at).split('.')[0]
        
        return {
            'watching': self.watching,
            'symbols': self.symbols,
            'check_interval': self.check_interval,
            'auto_execute': self.auto_execute,
            'talib_available': self.detector.talib_available,
            'telegram_enabled': self.telegram_bot is not None,
            'patterns_detected': self.patterns_detected,
            'alerts_sent': self.alerts_sent,
            'auto_executions': self.auto_executions,
            'uptime': uptime,
            'started_at': self.started_at.isoformat() if self.started_at else None,
        }
    
    def stop(self):
        """Stop the sentinel."""
        
        logger.info("[SENTINEL] Shutting down...")
        self.watching = False


# Global instance for API access
_sentinel_instance: Optional[LiveSentinel] = None


def get_sentinel() -> Optional[LiveSentinel]:
    """Get the global sentinel instance."""
    return _sentinel_instance


def set_sentinel(sentinel: LiveSentinel):
    """Set the global sentinel instance."""
    global _sentinel_instance
    _sentinel_instance = sentinel


# === MAIN ENTRY POINT ===

async def run_sentinel(
    symbols: List[str] = None,
    check_interval: int = 60,
    auto_execute: bool = False,
):
    """Run the Live Sentinel Agent."""
    
    if symbols is None:
        symbols = ["GC=F"]  # Gold futures default
    
    sentinel = LiveSentinel(
        symbols=symbols,
        check_interval=check_interval,
        auto_execute=auto_execute,
    )
    
    set_sentinel(sentinel)
    
    try:
        await sentinel.start()
    except KeyboardInterrupt:
        sentinel.stop()


if __name__ == "__main__":
    asyncio.run(run_sentinel())

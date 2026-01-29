#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
CORRECTION FADE BOT - Mean Reversion on Bullish Trend
═══════════════════════════════════════════════════════════════════════════════

Strategy Logic:
1. PRIMARY BIAS: BULLISH (SuperTrend dictates overall direction)
2. OVERLAY: Fade extreme rises with SHORT positions
3. TAKE PROFIT: At benchmark levels (Fib, support zones)
4. RESPAWN: After TP, if price rises again, spawn new SHORT
5. ACCUMULATE: Buy HALF positions on dips at support levels

"Ride the bull, fade the extremes, accumulate the dips."

Flow:
┌─────────────────────────────────────────────────────────────────────┐
│                     CORRECTION FADE BOT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SuperTrend = BULLISH (background)                                  │
│       │                                                             │
│       ▼                                                             │
│  Price rises +X% above MA? ──YES──► OPEN SHORT (fade the rise)     │
│       │                              │                              │
│       │                              ▼                              │
│       │                         Price drops to TP?                  │
│       │                              │                              │
│       │                    ┌────────┴────────┐                     │
│       │                    ▼                 ▼                      │
│       │              CLOSE SHORT        BUY HALF LONG              │
│       │              (take profit)      (accumulate dip)           │
│       │                    │                                        │
│       │                    ▼                                        │
│       │              Price rises again?                             │
│       │                    │                                        │
│       │              ──YES──► RESPAWN SHORT                        │
│       │                                                             │
│  SuperTrend = BEARISH? ──► CLOSE ALL, WAIT                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Created: January 29, 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder

# Import indicators from local module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from indicators import Indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/home/jbot/trading_ai/logs/correction_fade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CorrectionFade')


@dataclass
class FadeLevel:
    """A level where we fade (short) the rise"""
    price: float
    extension_pct: float      # How far above MA
    short_size: int           # Contracts to short
    status: str = 'PENDING'   # PENDING, ACTIVE, CLOSED
    entry_price: float = 0
    entry_time: str = ''


@dataclass
class AccumulationLevel:
    """A level where we accumulate (buy) on dips"""
    price: float
    description: str
    buy_size: int             # Contracts to buy (HALF position)
    status: str = 'PENDING'   # PENDING, FILLED
    fill_price: float = 0
    fill_time: str = ''


@dataclass
class CorrectionFadeState:
    """State of the correction fade bot"""
    active: bool = True
    mode: str = 'CORRECTION_FADE'
    supertrend_bullish: bool = True
    
    # Current short position
    short_contracts: int = 0
    short_avg_entry: float = 0
    short_unrealized_pnl: float = 0
    
    # Current long accumulation
    long_contracts: int = 0
    long_avg_entry: float = 0
    
    # Levels
    fade_levels: List[Dict] = field(default_factory=list)
    accumulation_levels: List[Dict] = field(default_factory=list)
    
    # Stats
    total_short_profit: float = 0
    total_fades: int = 0
    successful_fades: int = 0
    
    last_update: str = ''


class CorrectionFadeBot:
    """
    Correction Fade Bot - Fades rises while respecting bullish trend
    """
    
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contract = None
        
        # State
        self.state = CorrectionFadeState()
        self.state_file = Path(__file__).parent / 'correction_fade_state.json'
        
        # Configuration
        self.config = {
            # Connection
            'host': '100.119.161.65',
            'port': 7497,
            'client_id': 300,  # Different from Ghost (100) and Casper (200)
            
            # SuperTrend settings (same as Ghost)
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
            
            # Fade settings - when to SHORT
            'fade_extension_pct': 2.0,      # Short when +2% above 20 EMA
            'fade_extension_levels': [2.0, 3.0, 4.0, 5.0],  # Multiple fade levels
            'fade_contracts_base': 3,        # Base contracts per fade
            'fade_contracts_increment': 1,   # Add per level
            
            # Take profit for shorts
            'short_tp_levels': [
                {'name': 'TP1', 'drop_pct': 1.0, 'close_pct': 30},
                {'name': 'TP2', 'drop_pct': 2.0, 'close_pct': 40},
                {'name': 'TP3', 'drop_pct': 3.0, 'close_pct': 100},
            ],
            
            # Accumulation settings - $20 spacing, tighter DCA
            # HALF = 2 contracts, FULL = 4 contracts, DOUBLE = 8 contracts
            'accumulation_levels': [
                # Tight $20 spacing from current level
                {'price': 5551, 'desc': '$20 drop - HALF', 'size': 2},
                {'price': 5531, 'desc': '$40 drop - HALF', 'size': 2},
                {'price': 5511, 'desc': '$60 drop - FULL', 'size': 4},
                {'price': 5491, 'desc': '$80 drop - FULL', 'size': 4},
                {'price': 5471, 'desc': '$100 drop - FULL', 'size': 4},
                {'price': 5451, 'desc': '$120 drop - FULL', 'size': 4},
                {'price': 5431, 'desc': '$140 drop - FULL', 'size': 4},
                {'price': 5411, 'desc': '$160 drop - FULL', 'size': 4},
                # Major levels
                {'price': 5231, 'desc': 'Fib 23.6% - DOUBLE', 'size': 8},
                {'price': 5000, 'desc': 'Psychological - DOUBLE', 'size': 8},
                {'price': 4991, 'desc': '$5000 zone - FULL', 'size': 4},
                {'price': 4971, 'desc': 'Below $5000 - DOUBLE', 'size': 8},
                {'price': 4650, 'desc': 'THE GAP - TRIPLE', 'size': 12},
            ],
            
            # Respawn settings
            'respawn_rise_pct': 1.5,  # Respawn short after +1.5% rise from TP
            'respawn_cooldown_minutes': 30,
            
            # Safety
            'max_short_contracts': 15,
            'max_long_contracts': 50,  # More levels = need more capacity
            'stop_loss_pct': 3.0,  # Stop loss on shorts at +3% above entry
        }
        
        # Tracking
        self.last_fade_time = None
        self.last_tp_price = None
        self.ema_20 = 0
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = CorrectionFadeState(**data)
                    logger.info(f"Loaded state: {self.state.short_contracts} short, {self.state.long_contracts} long")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save state to file"""
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
    
    def connect(self) -> bool:
        """Connect to IB"""
        try:
            self.ib.connect(
                self.config['host'],
                self.config['port'],
                clientId=self.config['client_id']
            )
            self.connected = True
            logger.info(f"✅ Connected to IB (clientId={self.config['client_id']})")
            
            # Initialize contract
            self._init_contract()
            
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _init_contract(self):
        """Initialize MGC contract"""
        # Get from existing position or use April 2026
        positions = self.ib.positions()
        for p in positions:
            if 'MGC' in str(p.contract.localSymbol):
                self.contract = p.contract
                self.contract.exchange = 'COMEX'
                logger.info(f"Using contract from position: {self.contract.localSymbol}")
                return
        
        # Default to April 2026
        self.contract = Future('MGC', '20260428', 'COMEX')
        self.ib.qualifyContracts(self.contract)
        logger.info(f"Using default contract: {self.contract.localSymbol}")
    
    def get_current_price(self) -> Optional[float]:
        """Get current price"""
        try:
            self.ib.reqMarketDataType(3)
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            
            price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            return price if price and price > 0 else None
        except Exception as e:
            logger.error(f"Price fetch error: {e}")
            return None
    
    def get_candles(self, duration: str = "5 D", bar_size: str = "1 hour") -> List[Dict]:
        """Get historical candles"""
        try:
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
            
            candles = []
            for bar in bars:
                candles.append({
                    'time': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            return candles
        except Exception as e:
            logger.error(f"Candle fetch error: {e}")
            return []
    
    def calculate_ema(self, candles: List[Dict], period: int = 20) -> float:
        """Calculate EMA"""
        if len(candles) < period:
            return 0
        
        closes = [c['close'] for c in candles]
        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        
        for close in closes[period:]:
            ema = (close - ema) * multiplier + ema
        
        return ema
    
    def check_supertrend(self, candles: List[Dict]) -> Tuple[bool, float]:
        """Check SuperTrend direction"""
        st = Indicators.supertrend(
            candles,
            period=self.config['supertrend_period'],
            multiplier=self.config['supertrend_multiplier']
        )
        
        is_bullish = st['trend'] == 1
        return is_bullish, st['value']
    
    def get_position(self) -> Tuple[int, float]:
        """Get current net position"""
        positions = self.ib.positions()
        for p in positions:
            if 'MGC' in str(p.contract.localSymbol):
                return int(p.position), p.avgCost / 10
        return 0, 0
    
    def check_fade_opportunity(self, current_price: float, ema_20: float) -> Optional[Dict]:
        """
        Check if we should fade (SHORT) the current rise
        
        Conditions:
        1. Price extended above EMA
        2. Not already at max short
        3. Cooldown passed since last fade
        """
        if ema_20 == 0:
            return None
        
        extension_pct = ((current_price - ema_20) / ema_20) * 100
        
        # Check if extended enough
        for i, level_pct in enumerate(self.config['fade_extension_levels']):
            if extension_pct >= level_pct:
                # Check if we already have a fade at this level
                level_active = any(
                    f['extension_pct'] == level_pct and f['status'] == 'ACTIVE'
                    for f in self.state.fade_levels
                )
                
                if level_active:
                    continue
                
                # Check max short
                if self.state.short_contracts >= self.config['max_short_contracts']:
                    logger.debug("Max short contracts reached")
                    return None
                
                # Check cooldown
                if self.last_fade_time:
                    cooldown = timedelta(minutes=self.config['respawn_cooldown_minutes'])
                    if datetime.now() - self.last_fade_time < cooldown:
                        logger.debug("Fade cooldown active")
                        return None
                
                # Calculate contracts
                contracts = self.config['fade_contracts_base'] + (self.config['fade_contracts_increment'] * i)
                contracts = min(contracts, self.config['max_short_contracts'] - self.state.short_contracts)
                
                if contracts > 0:
                    return {
                        'action': 'FADE',
                        'extension_pct': level_pct,
                        'actual_extension': extension_pct,
                        'contracts': contracts,
                        'price': current_price,
                        'ema_20': ema_20,
                        'reason': f"Price {extension_pct:.1f}% above EMA (level {level_pct}%)"
                    }
        
        return None
    
    def check_short_tp(self, current_price: float) -> Optional[Dict]:
        """Check if any short TP levels hit"""
        if self.state.short_contracts <= 0 or self.state.short_avg_entry == 0:
            return None
        
        drop_pct = ((self.state.short_avg_entry - current_price) / self.state.short_avg_entry) * 100
        
        for tp in self.config['short_tp_levels']:
            if drop_pct >= tp['drop_pct']:
                close_contracts = int(self.state.short_contracts * tp['close_pct'] / 100)
                close_contracts = max(1, min(close_contracts, self.state.short_contracts))
                
                return {
                    'action': 'SHORT_TP',
                    'level': tp['name'],
                    'drop_pct': drop_pct,
                    'close_contracts': close_contracts,
                    'price': current_price,
                    'profit_per_contract': (self.state.short_avg_entry - current_price) * 10
                }
        
        return None
    
    def check_short_stop(self, current_price: float) -> Optional[Dict]:
        """Check if short stop loss hit"""
        if self.state.short_contracts <= 0 or self.state.short_avg_entry == 0:
            return None
        
        rise_pct = ((current_price - self.state.short_avg_entry) / self.state.short_avg_entry) * 100
        
        if rise_pct >= self.config['stop_loss_pct']:
            return {
                'action': 'SHORT_STOP',
                'rise_pct': rise_pct,
                'close_contracts': self.state.short_contracts,
                'price': current_price,
                'loss_per_contract': (current_price - self.state.short_avg_entry) * 10
            }
        
        return None
    
    def check_accumulation(self, current_price: float) -> Optional[Dict]:
        """Check if we should accumulate (BUY) at a support level"""
        for level in self.config['accumulation_levels']:
            # Check if price at or below level
            if current_price <= level['price'] * 1.005:  # Within 0.5%
                # Check if already filled
                already_filled = any(
                    a['price'] == level['price'] and a['status'] == 'FILLED'
                    for a in self.state.accumulation_levels
                )
                
                if already_filled:
                    continue
                
                # Check max long
                if self.state.long_contracts >= self.config['max_long_contracts']:
                    return None
                
                buy_size = min(level['size'], self.config['max_long_contracts'] - self.state.long_contracts)
                
                if buy_size > 0:
                    return {
                        'action': 'ACCUMULATE',
                        'level_price': level['price'],
                        'description': level['desc'],
                        'contracts': buy_size,
                        'current_price': current_price,
                        'reason': f"Accumulation at {level['desc']} (${level['price']})"
                    }
        
        return None
    
    def check_respawn(self, current_price: float) -> bool:
        """Check if should respawn a short after taking profit"""
        if self.last_tp_price is None:
            return False
        
        rise_from_tp = ((current_price - self.last_tp_price) / self.last_tp_price) * 100
        
        if rise_from_tp >= self.config['respawn_rise_pct']:
            logger.info(f"🔄 RESPAWN triggered: +{rise_from_tp:.1f}% from last TP")
            self.last_tp_price = None  # Reset
            return True
        
        return False
    
    def execute_fade(self, fade: Dict) -> bool:
        """Execute a fade (SHORT) trade"""
        logger.info(f"📉 EXECUTING FADE: SELL {fade['contracts']} MGC")
        logger.info(f"   Reason: {fade['reason']}")
        
        try:
            order = MarketOrder('SELL', fade['contracts'])
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"✅ FADE FILLED @ ${fill_price:.2f}")
                
                # Update state
                total_value = (self.state.short_contracts * self.state.short_avg_entry) + (fade['contracts'] * fill_price)
                self.state.short_contracts += fade['contracts']
                self.state.short_avg_entry = total_value / self.state.short_contracts if self.state.short_contracts > 0 else 0
                
                # Track fade level
                self.state.fade_levels.append({
                    'extension_pct': fade['extension_pct'],
                    'entry_price': fill_price,
                    'contracts': fade['contracts'],
                    'status': 'ACTIVE',
                    'entry_time': datetime.now().isoformat()
                })
                
                self.state.total_fades += 1
                self.last_fade_time = datetime.now()
                self._save_state()
                
                return True
            else:
                logger.error(f"Fade order failed: {trade.orderStatus}")
                return False
                
        except Exception as e:
            logger.error(f"Fade execution error: {e}")
            return False
    
    def execute_short_close(self, action: Dict) -> bool:
        """Execute short position close (TP or stop)"""
        contracts = action['close_contracts']
        logger.info(f"📈 CLOSING SHORT: BUY {contracts} MGC ({action['action']})")
        
        try:
            order = MarketOrder('BUY', contracts)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                
                if 'profit_per_contract' in action:
                    profit = action['profit_per_contract'] * contracts
                    self.state.total_short_profit += profit
                    self.state.successful_fades += 1
                    logger.info(f"✅ SHORT TP: BUY {contracts} @ ${fill_price:.2f}, Profit: ${profit:+,.0f}")
                else:
                    loss = action['loss_per_contract'] * contracts
                    logger.info(f"🛑 SHORT STOP: BUY {contracts} @ ${fill_price:.2f}, Loss: ${loss:,.0f}")
                
                # Update state
                self.state.short_contracts -= contracts
                if self.state.short_contracts <= 0:
                    self.state.short_contracts = 0
                    self.state.short_avg_entry = 0
                
                # Mark fade levels as closed
                for f in self.state.fade_levels:
                    if f['status'] == 'ACTIVE':
                        f['status'] = 'CLOSED'
                
                self.last_tp_price = fill_price
                self._save_state()
                
                return True
            else:
                logger.error(f"Close order failed: {trade.orderStatus}")
                return False
                
        except Exception as e:
            logger.error(f"Close execution error: {e}")
            return False
    
    def execute_accumulation(self, acc: Dict) -> bool:
        """Execute accumulation (BUY) at support"""
        logger.info(f"📈 ACCUMULATING: BUY {acc['contracts']} MGC at {acc['description']}")
        
        try:
            order = MarketOrder('BUY', acc['contracts'])
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"✅ ACCUMULATED @ ${fill_price:.2f}")
                
                # Update state
                total_value = (self.state.long_contracts * self.state.long_avg_entry) + (acc['contracts'] * fill_price)
                self.state.long_contracts += acc['contracts']
                self.state.long_avg_entry = total_value / self.state.long_contracts if self.state.long_contracts > 0 else 0
                
                # Track accumulation
                self.state.accumulation_levels.append({
                    'price': acc['level_price'],
                    'description': acc['description'],
                    'contracts': acc['contracts'],
                    'fill_price': fill_price,
                    'status': 'FILLED',
                    'fill_time': datetime.now().isoformat()
                })
                
                self._save_state()
                return True
            else:
                logger.error(f"Accumulation order failed: {trade.orderStatus}")
                return False
                
        except Exception as e:
            logger.error(f"Accumulation error: {e}")
            return False
    
    def run_once(self):
        """Run one iteration of the correction fade bot"""
        if not self.connected:
            return
        
        # Get current price and data
        current_price = self.get_current_price()
        if not current_price:
            return
        
        candles = self.get_candles("10 D", "1 hour")
        if len(candles) < 50:
            return
        
        # Calculate indicators
        ema_20 = self.calculate_ema(candles, 20)
        self.ema_20 = ema_20
        
        is_bullish, st_value = self.check_supertrend(candles)
        self.state.supertrend_bullish = is_bullish
        
        # Update position from IB
        net_pos, avg_cost = self.get_position()
        
        extension_pct = ((current_price - ema_20) / ema_20) * 100 if ema_20 > 0 else 0
        
        logger.info(f"📊 Price: ${current_price:.2f} | EMA20: ${ema_20:.2f} | Ext: {extension_pct:+.1f}%")
        logger.info(f"   SuperTrend: {'BULLISH' if is_bullish else 'BEARISH'} | Net Pos: {net_pos}")
        
        # === SAFETY: If SuperTrend goes BEARISH, close all shorts ===
        if not is_bullish and self.state.short_contracts > 0:
            logger.warning("⚠️ SuperTrend BEARISH - closing all shorts")
            self.execute_short_close({
                'action': 'TREND_EXIT',
                'close_contracts': self.state.short_contracts,
                'profit_per_contract': (self.state.short_avg_entry - current_price) * 10
            })
            return
        
        # === CHECK SHORT STOP LOSS ===
        stop = self.check_short_stop(current_price)
        if stop:
            self.execute_short_close(stop)
            return
        
        # === CHECK SHORT TAKE PROFIT ===
        tp = self.check_short_tp(current_price)
        if tp:
            self.execute_short_close(tp)
            
            # Also check for accumulation at this level
            acc = self.check_accumulation(current_price)
            if acc:
                self.execute_accumulation(acc)
            return
        
        # === CHECK FOR NEW FADE OPPORTUNITY ===
        if is_bullish:  # Only fade in bullish trend
            # Check respawn first
            if self.check_respawn(current_price):
                pass  # Will trigger fade check below
            
            fade = self.check_fade_opportunity(current_price, ema_20)
            if fade:
                self.execute_fade(fade)
                return
        
        # === CHECK FOR ACCUMULATION (on dips) ===
        acc = self.check_accumulation(current_price)
        if acc:
            self.execute_accumulation(acc)
    
    def run(self, interval_seconds: int = 30):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("🔄 CORRECTION FADE BOT STARTING")
        logger.info("=" * 60)
        logger.info("Strategy: Fade rises, accumulate dips, respect bullish trend")
        logger.info(f"Fade levels: {self.config['fade_extension_levels']}%")
        logger.info(f"Accumulation levels: {[l['price'] for l in self.config['accumulation_levels']]}")
        
        if not self.connect():
            return
        
        while True:
            try:
                self.run_once()
                self.ib.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Run error: {e}")
                time.sleep(60)
        
        self.ib.disconnect()
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'mode': 'CORRECTION_FADE',
            'active': self.state.active,
            'supertrend': 'BULLISH' if self.state.supertrend_bullish else 'BEARISH',
            'short_position': {
                'contracts': self.state.short_contracts,
                'avg_entry': self.state.short_avg_entry,
                'unrealized_pnl': self.state.short_unrealized_pnl
            },
            'long_position': {
                'contracts': self.state.long_contracts,
                'avg_entry': self.state.long_avg_entry
            },
            'stats': {
                'total_fades': self.state.total_fades,
                'successful_fades': self.state.successful_fades,
                'total_profit': self.state.total_short_profit,
                'win_rate': f"{self.state.successful_fades/max(1,self.state.total_fades)*100:.0f}%"
            },
            'ema_20': self.ema_20,
            'last_update': self.state.last_update
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = CorrectionFadeBot()
    bot.run()

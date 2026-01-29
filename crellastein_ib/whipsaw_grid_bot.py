#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
WHIPSAW GRID BOT - Profit from Volatility in Both Directions
═══════════════════════════════════════════════════════════════════════════════

Strategy: Grid trading with $20 spacing
- BUY on drops (accumulate longs)
- SHORT on rises (fade the bounces)
- Take profit on BOTH sides
- Respawn after TP

"The chop is the opportunity."

                         SHORT ZONE (fade rises)
                              ▲
    $5,611 ──── SHORT 2 ─────┤
    $5,591 ──── SHORT 2 ─────┤
    $5,571 ════ CENTER ══════╋════ Current Price
    $5,551 ──── BUY 2 ───────┤
    $5,531 ──── BUY 2 ───────┤
    $5,511 ──── BUY 4 ───────┤
                              ▼
                         LONG ZONE (accumulate dips)

Each $20 move = opportunity to:
- Take profit on one side
- Add position on the other side

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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from indicators import Indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/home/jbot/trading_ai/logs/whipsaw_grid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WhipsawGrid')


@dataclass
class GridLevel:
    """A single grid level"""
    price: float
    side: str           # 'BUY' or 'SHORT'
    size: int           # Contracts
    status: str = 'PENDING'  # PENDING, FILLED, TP_HIT
    fill_price: float = 0
    fill_time: str = ''
    tp_price: float = 0
    tp_hit_time: str = ''


@dataclass 
class WhipsawState:
    """Bot state"""
    active: bool = True
    center_price: float = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # GRID MODE CONTROL
    # ═══════════════════════════════════════════════════════════════════
    grid_mode: str = 'BEARISH'  # BULLISH, BEARISH, CORRECTION
    buy_enabled: bool = False   # Allow new BUY orders
    short_enabled: bool = True  # Allow new SHORT orders
    
    # Bear flag specific
    bear_flag_mode: bool = True  # ENABLED - waiting for breakdown
    bear_flag_invalidation_price: float = 5611  # Price above which bear flag is debunked
    
    # Positions
    long_contracts: int = 0
    long_avg_entry: float = 0
    short_contracts: int = 0
    short_avg_entry: float = 0
    
    # Grid levels
    buy_levels: List[Dict] = field(default_factory=list)
    short_levels: List[Dict] = field(default_factory=list)
    
    # Stats
    long_tp_profit: float = 0
    short_tp_profit: float = 0
    total_trades: int = 0
    winning_trades: int = 0
    
    last_update: str = ''


class WhipsawGridBot:
    """
    Whipsaw Grid Bot - Two-sided grid trading
    
    Profits from volatility by:
    - Buying dips (every $20 down)
    - Shorting rises (every $20 up)
    - Taking profit when price reverses
    """
    
    def __init__(self, center_price: float = None):
        self.ib = IB()
        self.connected = False
        self.contract = None
        
        # State
        self.state = WhipsawState()
        self.state_file = Path(__file__).parent / 'whipsaw_state.json'
        
        # Grid Configuration
        self.config = {
            # Connection
            'host': '100.119.161.65',
            'port': 7497,
            'client_id': 400,  # Unique client ID
            
            # Grid settings
            'grid_spacing': 20.0,        # $20 between levels
            'center_price': center_price or 5571,  # Grid center
            
            # Position sizes
            'half_size': 2,              # HALF = 2 contracts
            'full_size': 4,              # FULL = 4 contracts
            'double_size': 8,            # DOUBLE = 8 contracts
            
            # Buy levels (below center) - $20 spacing
            'buy_grid': [
                {'offset': -20, 'size': 2, 'desc': 'BUY L1 HALF'},
                {'offset': -40, 'size': 2, 'desc': 'BUY L2 HALF'},
                {'offset': -60, 'size': 4, 'desc': 'BUY L3 FULL'},
                {'offset': -80, 'size': 4, 'desc': 'BUY L4 FULL'},
                {'offset': -100, 'size': 4, 'desc': 'BUY L5 FULL'},
                {'offset': -120, 'size': 4, 'desc': 'BUY L6 FULL'},
                {'offset': -140, 'size': 4, 'desc': 'BUY L7 FULL'},
                {'offset': -160, 'size': 4, 'desc': 'BUY L8 FULL'},
                {'offset': -340, 'size': 8, 'desc': 'BUY Fib DOUBLE'},
                {'offset': -571, 'size': 8, 'desc': 'BUY $5000 DOUBLE'},
                {'offset': -921, 'size': 12, 'desc': 'BUY GAP TRIPLE'},
            ],
            
            # Short levels (above center) - $20 spacing
            'short_grid': [
                {'offset': 20, 'size': 2, 'desc': 'SHORT L1 HALF'},
                {'offset': 40, 'size': 2, 'desc': 'SHORT L2 HALF'},
                {'offset': 60, 'size': 4, 'desc': 'SHORT L3 FULL'},
                {'offset': 80, 'size': 4, 'desc': 'SHORT L4 FULL'},
                {'offset': 100, 'size': 4, 'desc': 'SHORT L5 FULL'},
                {'offset': 120, 'size': 4, 'desc': 'SHORT L6 FULL'},
                {'offset': 140, 'size': 4, 'desc': 'SHORT L7 FULL'},
                {'offset': 160, 'size': 4, 'desc': 'SHORT L8 FULL'},
            ],
            
            # Take profit settings
            'tp_offset': 20.0,           # TP $20 from entry
            'partial_tp_pct': 50,        # Close 50% at first TP
            
            # Safety
            'max_long_contracts': 50,
            'max_short_contracts': 30,
            'stop_loss_offset': 100,     # Stop loss $100 from avg entry
            
            # SuperTrend safety
            'use_supertrend': True,
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
        }
        
        self._load_state()
        self._initialize_grid()
    
    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = WhipsawState(**data)
                    logger.info(f"Loaded state: L{self.state.long_contracts}/S{self.state.short_contracts}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save state"""
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
    
    def _initialize_grid(self):
        """Initialize grid levels"""
        center = self.config['center_price']
        self.state.center_price = center
        
        # Initialize buy levels
        if not self.state.buy_levels:
            for level in self.config['buy_grid']:
                self.state.buy_levels.append({
                    'price': center + level['offset'],
                    'offset': level['offset'],
                    'size': level['size'],
                    'desc': level['desc'],
                    'status': 'PENDING',
                    'fill_price': 0,
                    'tp_price': center + level['offset'] + self.config['tp_offset'],
                })
        
        # Initialize short levels
        if not self.state.short_levels:
            for level in self.config['short_grid']:
                self.state.short_levels.append({
                    'price': center + level['offset'],
                    'offset': level['offset'],
                    'size': level['size'],
                    'desc': level['desc'],
                    'status': 'PENDING',
                    'fill_price': 0,
                    'tp_price': center + level['offset'] - self.config['tp_offset'],
                })
        
        self._save_state()
    
    def connect(self) -> bool:
        """Connect to IB"""
        try:
            self.ib.connect(
                self.config['host'],
                self.config['port'],
                clientId=self.config['client_id']
            )
            self.connected = True
            logger.info(f"✅ Connected (clientId={self.config['client_id']})")
            self._init_contract()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _init_contract(self):
        """Initialize MGC contract"""
        positions = self.ib.positions()
        for p in positions:
            if 'MGC' in str(p.contract.localSymbol):
                self.contract = p.contract
                self.contract.exchange = 'COMEX'
                return
        
        self.contract = Future('MGC', '20260428', 'COMEX')
        self.ib.qualifyContracts(self.contract)
    
    def get_price(self) -> Optional[float]:
        """Get current price"""
        try:
            self.ib.reqMarketDataType(3)
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            return ticker.last if ticker.last and ticker.last > 0 else ticker.close
        except:
            return None
    
    def get_position(self) -> int:
        """Get net position"""
        positions = self.ib.positions()
        for p in positions:
            if 'MGC' in str(p.contract.localSymbol):
                return int(p.position)
        return 0
    
    def check_buy_triggers(self, price: float) -> Optional[Dict]:
        """Check if any buy level triggered"""
        # ═══════════════════════════════════════════════════════════════════
        # MODE CHECK - Are BUYs enabled?
        # ═══════════════════════════════════════════════════════════════════
        if not getattr(self.state, 'buy_enabled', True):
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # BEAR FLAG MODE - Block all new BUY orders
        # ═══════════════════════════════════════════════════════════════════
        if self.state.bear_flag_mode:
            # Check if bear flag should be invalidated (price breaks higher)
            if price >= self.state.bear_flag_invalidation_price:
                logger.info(f"🐻❌ BEAR FLAG INVALIDATED! Price ${price:.2f} >= ${self.state.bear_flag_invalidation_price}")
                self.state.bear_flag_mode = False
                self.state.buy_enabled = True
                self.state.grid_mode = 'CORRECTION'
                self._save_state()
            else:
                # Bear flag still active - NO BUYS
                return None
        
        for level in self.state.buy_levels:
            if level['status'] == 'PENDING':
                if price <= level['price'] * 1.002:  # Within 0.2%
                    # Check max long
                    if self.state.long_contracts >= self.config['max_long_contracts']:
                        continue
                    return {
                        'action': 'BUY',
                        'level': level,
                        'price': price,
                    }
        return None
    
    def check_short_triggers(self, price: float) -> Optional[Dict]:
        """Check if any short level triggered"""
        # ═══════════════════════════════════════════════════════════════════
        # MODE CHECK - Are SHORTs enabled?
        # ═══════════════════════════════════════════════════════════════════
        if not getattr(self.state, 'short_enabled', True):
            return None
        
        for level in self.state.short_levels:
            if level['status'] == 'PENDING':
                if price >= level['price'] * 0.998:  # Within 0.2%
                    # Check max short
                    if self.state.short_contracts >= self.config['max_short_contracts']:
                        continue
                    return {
                        'action': 'SHORT',
                        'level': level,
                        'price': price,
                    }
        return None
    
    def check_long_tp(self, price: float) -> Optional[Dict]:
        """Check if any long position hits TP"""
        for level in self.state.buy_levels:
            if level['status'] == 'FILLED':
                if price >= level['tp_price']:
                    return {
                        'action': 'LONG_TP',
                        'level': level,
                        'price': price,
                        'profit': (price - level['fill_price']) * 10 * level['size']
                    }
        return None
    
    def check_short_tp(self, price: float) -> Optional[Dict]:
        """Check if any short position hits TP"""
        for level in self.state.short_levels:
            if level['status'] == 'FILLED':
                if price <= level['tp_price']:
                    return {
                        'action': 'SHORT_TP',
                        'level': level,
                        'price': price,
                        'profit': (level['fill_price'] - price) * 10 * level['size']
                    }
        return None
    
    def execute_buy(self, trigger: Dict) -> bool:
        """Execute buy order"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"📈 BUY {size} @ ${trigger['price']:.2f} ({level['desc']})")
        
        try:
            order = MarketOrder('BUY', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                logger.info(f"✅ BOUGHT {size} @ ${fill:.2f}")
                
                # Update level
                level['status'] = 'FILLED'
                level['fill_price'] = fill
                level['fill_time'] = datetime.now().isoformat()
                level['tp_price'] = fill + self.config['tp_offset']
                
                # Update state
                total = (self.state.long_contracts * self.state.long_avg_entry) + (size * fill)
                self.state.long_contracts += size
                self.state.long_avg_entry = total / self.state.long_contracts if self.state.long_contracts else 0
                self.state.total_trades += 1
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Buy error: {e}")
        return False
    
    def execute_short(self, trigger: Dict) -> bool:
        """Execute short order"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"📉 SHORT {size} @ ${trigger['price']:.2f} ({level['desc']})")
        
        try:
            order = MarketOrder('SELL', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                logger.info(f"✅ SHORTED {size} @ ${fill:.2f}")
                
                # Update level
                level['status'] = 'FILLED'
                level['fill_price'] = fill
                level['fill_time'] = datetime.now().isoformat()
                level['tp_price'] = fill - self.config['tp_offset']
                
                # Update state
                total = (self.state.short_contracts * self.state.short_avg_entry) + (size * fill)
                self.state.short_contracts += size
                self.state.short_avg_entry = total / self.state.short_contracts if self.state.short_contracts else 0
                self.state.total_trades += 1
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Short error: {e}")
        return False
    
    def execute_long_tp(self, trigger: Dict) -> bool:
        """Execute long take profit"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"🎯 LONG TP: SELL {size} @ ${trigger['price']:.2f}")
        
        try:
            order = MarketOrder('SELL', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                profit = (fill - level['fill_price']) * 10 * size
                
                logger.info(f"✅ LONG TP HIT: ${profit:+,.0f}")
                
                # Update level - reset to PENDING for respawn
                level['status'] = 'PENDING'
                level['tp_hit_time'] = datetime.now().isoformat()
                
                # Update state
                self.state.long_contracts -= size
                self.state.long_tp_profit += profit
                self.state.winning_trades += 1
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Long TP error: {e}")
        return False
    
    def execute_short_tp(self, trigger: Dict) -> bool:
        """Execute short take profit"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"🎯 SHORT TP: BUY {size} @ ${trigger['price']:.2f}")
        
        try:
            order = MarketOrder('BUY', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                profit = (level['fill_price'] - fill) * 10 * size
                
                logger.info(f"✅ SHORT TP HIT: ${profit:+,.0f}")
                
                # Update level - reset to PENDING for respawn
                level['status'] = 'PENDING'
                level['tp_hit_time'] = datetime.now().isoformat()
                
                # Update state
                self.state.short_contracts -= size
                self.state.short_tp_profit += profit
                self.state.winning_trades += 1
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Short TP error: {e}")
        return False
    
    def run_once(self):
        """Run one iteration"""
        if not self.connected:
            return
        
        price = self.get_price()
        if not price:
            return
        
        net_pos = self.get_position()
        
        # Log status with mode indicator
        mode = getattr(self.state, 'grid_mode', 'UNKNOWN')
        mode_icon = {'BULLISH': '📈', 'BEARISH': '📉', 'CORRECTION': '🔄'}.get(mode, '')
        logger.info(f"💰 ${price:.2f} | {mode_icon} {mode} | Net:{net_pos} | L:{self.state.long_contracts} S:{self.state.short_contracts}")
        
        # Check LONG take profits first
        long_tp = self.check_long_tp(price)
        if long_tp:
            self.execute_long_tp(long_tp)
            return
        
        # Check SHORT take profits
        short_tp = self.check_short_tp(price)
        if short_tp:
            self.execute_short_tp(short_tp)
            return
        
        # Check BUY triggers (price dropping)
        buy = self.check_buy_triggers(price)
        if buy:
            self.execute_buy(buy)
        
        # Check SHORT triggers (price rising)
        short = self.check_short_triggers(price)
        if short:
            self.execute_short(short)
    
    def run(self, interval: int = 15):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("🔄 WHIPSAW GRID BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Center: ${self.config['center_price']}")
        logger.info(f"Grid spacing: ${self.config['grid_spacing']}")
        logger.info(f"Buy levels: {len(self.config['buy_grid'])}")
        logger.info(f"Short levels: {len(self.config['short_grid'])}")
        
        if not self.connect():
            return
        
        while True:
            try:
                self.run_once()
                self.ib.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)
        
        self.ib.disconnect()
    
    def get_grid_display(self) -> str:
        """Get visual grid display"""
        lines = []
        lines.append("=" * 50)
        lines.append("WHIPSAW GRID STATUS")
        lines.append("=" * 50)
        
        # Short levels (top)
        lines.append("\n📉 SHORT LEVELS (above center):")
        for level in reversed(self.state.short_levels):
            status = "🟢" if level['status'] == 'FILLED' else "⚪"
            lines.append(f"  {status} ${level['price']:.0f} | {level['size']} | {level['desc']} | {level['status']}")
        
        lines.append(f"\n═══ CENTER: ${self.state.center_price:.0f} ═══")
        
        # Buy levels (bottom)
        lines.append("\n📈 BUY LEVELS (below center):")
        for level in self.state.buy_levels:
            status = "🟢" if level['status'] == 'FILLED' else "⚪"
            lines.append(f"  {status} ${level['price']:.0f} | {level['size']} | {level['desc']} | {level['status']}")
        
        lines.append(f"\n💰 STATS:")
        lines.append(f"  Long P&L: ${self.state.long_tp_profit:+,.0f}")
        lines.append(f"  Short P&L: ${self.state.short_tp_profit:+,.0f}")
        lines.append(f"  Total: ${self.state.long_tp_profit + self.state.short_tp_profit:+,.0f}")
        lines.append(f"  Win Rate: {self.state.winning_trades}/{self.state.total_trades}")
        
        return "\n".join(lines)


def main():
    """Run the whipsaw grid bot"""
    import sys
    
    # Get center price from argument or use default
    center = float(sys.argv[1]) if len(sys.argv) > 1 else 5571
    
    bot = WhipsawGridBot(center_price=center)
    print(bot.get_grid_display())
    
    print("\nStarting bot...")
    bot.run()


if __name__ == "__main__":
    main()

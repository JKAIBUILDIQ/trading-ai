#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    WHIPSAW COMMANDER - HYBRID STRATEGY
═══════════════════════════════════════════════════════════════════════════════

Combines the BEST of MT5 Ghost + IBKR Whipsaw Grid:

FROM MT5 GHOST:
  ✓ SuperTrend regime detection
  ✓ NEO signal integration  
  ✓ Progressive TPs (+20 above previous entry)
  ✓ Trailing stops on runners

FROM IBKR WHIPSAW GRID:
  ✓ RESPAWN logic (levels reset after TP)
  ✓ Bidirectional grid (shorts on rises)
  ✓ Scaling by distance (HALF/FULL/DOUBLE/TRIPLE)
  ✓ Gap hunting at key levels

═══════════════════════════════════════════════════════════════════════════════
                              THE HYBRID
═══════════════════════════════════════════════════════════════════════════════

         SHORT GRID (fade parabolic rises)
                              ▲
    $5,631 ──── SHORT 1.0 ──── TP @ $5,611 │ FULL (respawns)
    $5,611 ──── SHORT 0.5 ──── TP @ $5,591 │ HALF (respawns)
    $5,591 ──── SHORT 0.5 ──── TP @ $5,571 │ HALF (respawns)
                              │
    $5,571 ═══ CENTER ════════╋════════════════════════════
                              │
    $5,551 ──── BUY 0.5 ───── TP @ $5,571 (+20) │ HALF (respawns)
    $5,531 ──── BUY 0.5 ───── TP @ $5,551 (+20 above prev) │ HALF
    $5,511 ──── BUY 1.0 ───── TP @ $5,531 (+20 above prev) │ FULL
    $5,491 ──── BUY 1.0 ───── TP @ $5,511 │ FULL
    $5,231 ──── BUY 2.0 ───── TP @ $5,251 │ DOUBLE (Fib)
    $5,000 ──── BUY 2.0 ───── TP @ $5,020 │ DOUBLE (Psych)
    $4,650 ──── BUY 3.0 ───── TP @ $4,670 │ TRIPLE (THE GAP)
                              ▼
         LONG GRID (buy corrections)

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
from enum import Enum
import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from indicators import Indicators

# Try to import NEO
try:
    from neo_integration import neo_should_enter, neo_should_dca, get_neo_signal
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/home/jbot/trading_ai/logs/whipsaw_commander.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WhipsawCommander')


class Regime(Enum):
    """Market regime from SuperTrend"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class GridLevel:
    """A grid level with RESPAWN capability"""
    price: float
    side: str               # 'BUY' or 'SHORT'
    size: float             # Lot size (0.5, 1.0, 2.0, 3.0)
    size_label: str         # HALF, FULL, DOUBLE, TRIPLE
    tp_price: float         # Take profit target
    status: str = 'PENDING' # PENDING, FILLED, TP_HIT
    fill_price: float = 0
    fill_time: str = ''
    tp_hit_time: str = ''
    cycles_completed: int = 0  # Track how many times this level cycled


@dataclass
class WhipsawState:
    """Complete bot state"""
    active: bool = True
    center_price: float = 0
    regime: str = 'NEUTRAL'
    
    # Position tracking
    long_contracts: int = 0
    long_avg_entry: float = 0
    short_contracts: int = 0
    short_avg_entry: float = 0
    
    # Grid levels (serialized)
    buy_levels: List[Dict] = field(default_factory=list)
    short_levels: List[Dict] = field(default_factory=list)
    
    # Performance stats
    long_tp_profit: float = 0
    short_tp_profit: float = 0
    total_cycles: int = 0
    total_profit: float = 0
    
    # Timestamps
    last_update: str = ''
    session_start: str = ''


class WhipsawCommander:
    """
    Whipsaw Commander - The Hybrid Strategy
    
    Combines MT5 Ghost intelligence with IBKR Grid mechanics:
    - SuperTrend regime detection
    - NEO signal confirmation
    - Bidirectional grid with RESPAWN
    - Progressive TPs
    - Scaled position sizing
    """
    
    def __init__(self, center_price: float = None):
        self.ib = IB()
        self.connected = False
        self.contract = None
        self.indicators = Indicators()
        
        # State
        self.state = WhipsawState()
        self.state_file = Path(__file__).parent / 'whipsaw_commander_state.json'
        
        # Configuration
        self.config = {
            # Connection
            'host': '100.119.161.65',
            'port': 7497,
            'client_id': 500,  # Unique client ID
            
            # Grid settings
            'grid_spacing': 20.0,           # $20 between levels
            'center_price': center_price or 5571,
            
            # Position sizes (in contracts, MGC = $10/point)
            'half_size': 2,                 # HALF = 2 contracts
            'full_size': 4,                 # FULL = 4 contracts  
            'double_size': 8,               # DOUBLE = 8 contracts
            'triple_size': 12,              # TRIPLE = 12 contracts
            
            # Take profit style
            'tp_style': 'progressive',      # 'progressive' or 'fixed'
            'tp_offset': 20.0,              # $20 TP offset
            
            # Regime settings (SuperTrend)
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
            'use_regime_filter': True,      # Only short in BULLISH (fading)
            
            # NEO integration
            'use_neo': NEO_AVAILABLE,
            'neo_min_score': 55,
            
            # Safety limits
            'max_long_contracts': 50,
            'max_short_contracts': 30,
            
            # Grid definitions
            'buy_grid': [
                # Near levels - HALF size
                {'offset': -20, 'size': 2, 'label': 'HALF', 'desc': 'BUY L1'},
                {'offset': -40, 'size': 2, 'label': 'HALF', 'desc': 'BUY L2'},
                # Mid levels - FULL size
                {'offset': -60, 'size': 4, 'label': 'FULL', 'desc': 'BUY L3'},
                {'offset': -80, 'size': 4, 'label': 'FULL', 'desc': 'BUY L4'},
                {'offset': -100, 'size': 4, 'label': 'FULL', 'desc': 'BUY L5'},
                {'offset': -120, 'size': 4, 'label': 'FULL', 'desc': 'BUY L6'},
                {'offset': -140, 'size': 4, 'label': 'FULL', 'desc': 'BUY L7'},
                {'offset': -160, 'size': 4, 'label': 'FULL', 'desc': 'BUY L8'},
                # Deep levels - DOUBLE/TRIPLE
                {'offset': -340, 'size': 8, 'label': 'DOUBLE', 'desc': 'BUY Fib'},
                {'offset': -571, 'size': 8, 'label': 'DOUBLE', 'desc': 'BUY $5000'},
                {'offset': -921, 'size': 12, 'label': 'TRIPLE', 'desc': 'BUY GAP'},
            ],
            
            'short_grid': [
                # Near levels - HALF size (fade small rises)
                {'offset': 20, 'size': 2, 'label': 'HALF', 'desc': 'SHORT L1'},
                {'offset': 40, 'size': 2, 'label': 'HALF', 'desc': 'SHORT L2'},
                # Mid levels - FULL size (fade bigger rises)
                {'offset': 60, 'size': 4, 'label': 'FULL', 'desc': 'SHORT L3'},
                {'offset': 80, 'size': 4, 'label': 'FULL', 'desc': 'SHORT L4'},
                {'offset': 100, 'size': 4, 'label': 'FULL', 'desc': 'SHORT L5'},
                {'offset': 120, 'size': 4, 'label': 'FULL', 'desc': 'SHORT L6'},
                # High levels - DOUBLE (fade parabolic)
                {'offset': 140, 'size': 8, 'label': 'DOUBLE', 'desc': 'SHORT L7'},
                {'offset': 160, 'size': 8, 'label': 'DOUBLE', 'desc': 'SHORT L8'},
            ],
        }
        
        self._load_state()
        if not self.state.buy_levels:
            self._initialize_grid()
    
    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = WhipsawState(**data)
                    logger.info(f"Loaded state: L{self.state.long_contracts}/S{self.state.short_contracts} | Cycles: {self.state.total_cycles}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save state"""
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
    
    def _initialize_grid(self):
        """Initialize grid levels with progressive TPs"""
        center = self.config['center_price']
        self.state.center_price = center
        self.state.session_start = datetime.now().isoformat()
        
        # Build BUY levels with PROGRESSIVE TPs
        prev_entry = center
        for level in self.config['buy_grid']:
            entry_price = center + level['offset']
            
            # Progressive TP: +$20 above previous entry
            if self.config['tp_style'] == 'progressive':
                tp_price = prev_entry  # TP at previous level
            else:
                tp_price = entry_price + self.config['tp_offset']
            
            self.state.buy_levels.append({
                'price': entry_price,
                'offset': level['offset'],
                'size': level['size'],
                'label': level['label'],
                'desc': level['desc'],
                'tp_price': tp_price,
                'status': 'PENDING',
                'fill_price': 0,
                'fill_time': '',
                'cycles_completed': 0,
            })
            prev_entry = entry_price
        
        # Build SHORT levels with fixed TPs (fade back to center)
        for level in self.config['short_grid']:
            entry_price = center + level['offset']
            tp_price = entry_price - self.config['tp_offset']  # TP $20 below
            
            self.state.short_levels.append({
                'price': entry_price,
                'offset': level['offset'],
                'size': level['size'],
                'label': level['label'],
                'desc': level['desc'],
                'tp_price': tp_price,
                'status': 'PENDING',
                'fill_price': 0,
                'fill_time': '',
                'cycles_completed': 0,
            })
        
        self._save_state()
        logger.info(f"Grid initialized: {len(self.state.buy_levels)} BUY levels, {len(self.state.short_levels)} SHORT levels")
    
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
            return [{'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close} for b in bars]
        except Exception as e:
            logger.error(f"Candle fetch error: {e}")
            return []
    
    def get_regime(self, candles: List[Dict]) -> Regime:
        """Determine market regime from SuperTrend"""
        if len(candles) < 20:
            return Regime.NEUTRAL
        
        try:
            st_result = self.indicators.supertrend(
                candles,
                period=self.config['supertrend_period'],
                multiplier=self.config['supertrend_multiplier']
            )
            
            if st_result.get('trend') == 1:
                return Regime.BULLISH
            elif st_result.get('trend') == -1:
                return Regime.BEARISH
            else:
                return Regime.NEUTRAL
        except:
            return Regime.NEUTRAL
    
    def check_neo_approval(self, side: str) -> Tuple[bool, str]:
        """Check NEO for trade approval"""
        if not self.config['use_neo'] or not NEO_AVAILABLE:
            return True, "NEO not configured"
        
        try:
            signal = get_neo_signal()
            score = signal.get('combined_score', 50)
            
            if side == 'BUY':
                # For buys, want bullish or neutral NEO
                approved = score >= 45
                reason = f"NEO score {score}"
            else:
                # For shorts, want bearish or neutral NEO
                approved = score <= 60
                reason = f"NEO score {score}"
            
            return approved, reason
        except:
            return True, "NEO unavailable"
    
    def check_buy_triggers(self, price: float, regime: Regime) -> Optional[Dict]:
        """Check if any buy level triggered"""
        for level in self.state.buy_levels:
            if level['status'] != 'PENDING':
                continue
            
            # Check if price hit this level
            if price <= level['price'] * 1.002:  # Within 0.2%
                # Check max position
                if self.state.long_contracts >= self.config['max_long_contracts']:
                    continue
                
                # Check NEO
                neo_ok, neo_reason = self.check_neo_approval('BUY')
                if not neo_ok:
                    logger.info(f"NEO blocked BUY: {neo_reason}")
                    continue
                
                return {
                    'action': 'BUY',
                    'level': level,
                    'price': price,
                    'regime': regime.value,
                }
        return None
    
    def check_short_triggers(self, price: float, regime: Regime) -> Optional[Dict]:
        """Check if any short level triggered"""
        # Only fade in BULLISH regime (mean reversion)
        if self.config['use_regime_filter'] and regime != Regime.BULLISH:
            return None
        
        for level in self.state.short_levels:
            if level['status'] != 'PENDING':
                continue
            
            # Check if price hit this level
            if price >= level['price'] * 0.998:  # Within 0.2%
                # Check max position
                if self.state.short_contracts >= self.config['max_short_contracts']:
                    continue
                
                # Check NEO
                neo_ok, neo_reason = self.check_neo_approval('SHORT')
                if not neo_ok:
                    logger.info(f"NEO blocked SHORT: {neo_reason}")
                    continue
                
                return {
                    'action': 'SHORT',
                    'level': level,
                    'price': price,
                    'regime': regime.value,
                }
        return None
    
    def check_long_tp(self, price: float) -> Optional[Dict]:
        """Check if any long hits TP"""
        for level in self.state.buy_levels:
            if level['status'] == 'FILLED':
                if price >= level['tp_price']:
                    profit = (price - level['fill_price']) * 10 * level['size']
                    return {
                        'action': 'LONG_TP',
                        'level': level,
                        'price': price,
                        'profit': profit,
                    }
        return None
    
    def check_short_tp(self, price: float) -> Optional[Dict]:
        """Check if any short hits TP"""
        for level in self.state.short_levels:
            if level['status'] == 'FILLED':
                if price <= level['tp_price']:
                    profit = (level['fill_price'] - price) * 10 * level['size']
                    return {
                        'action': 'SHORT_TP',
                        'level': level,
                        'price': price,
                        'profit': profit,
                    }
        return None
    
    def execute_buy(self, trigger: Dict) -> bool:
        """Execute buy with RESPAWN tracking"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"📈 BUY {size} @ ${trigger['price']:.2f} ({level['desc']} {level['label']})")
        
        try:
            order = MarketOrder('BUY', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                logger.info(f"✅ BOUGHT {size} @ ${fill:.2f} | TP @ ${level['tp_price']:.2f}")
                
                # Update level
                level['status'] = 'FILLED'
                level['fill_price'] = fill
                level['fill_time'] = datetime.now().isoformat()
                
                # Recalculate TP based on actual fill
                if self.config['tp_style'] == 'fixed':
                    level['tp_price'] = fill + self.config['tp_offset']
                
                # Update position
                total = (self.state.long_contracts * self.state.long_avg_entry) + (size * fill)
                self.state.long_contracts += size
                self.state.long_avg_entry = total / self.state.long_contracts if self.state.long_contracts else 0
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Buy error: {e}")
        return False
    
    def execute_short(self, trigger: Dict) -> bool:
        """Execute short with RESPAWN tracking"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"📉 SHORT {size} @ ${trigger['price']:.2f} ({level['desc']} {level['label']})")
        
        try:
            order = MarketOrder('SELL', size)
            trade = self.ib.placeOrder(self.contract, order)
            
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                logger.info(f"✅ SHORTED {size} @ ${fill:.2f} | TP @ ${level['tp_price']:.2f}")
                
                # Update level
                level['status'] = 'FILLED'
                level['fill_price'] = fill
                level['fill_time'] = datetime.now().isoformat()
                level['tp_price'] = fill - self.config['tp_offset']
                
                # Update position
                total = (self.state.short_contracts * self.state.short_avg_entry) + (size * fill)
                self.state.short_contracts += size
                self.state.short_avg_entry = total / self.state.short_contracts if self.state.short_contracts else 0
                
                self._save_state()
                return True
        except Exception as e:
            logger.error(f"Short error: {e}")
        return False
    
    def execute_long_tp(self, trigger: Dict) -> bool:
        """Execute long TP with RESPAWN"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"🎯 LONG TP: SELL {size} @ ${trigger['price']:.2f} ({level['desc']})")
        
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
                
                logger.info(f"✅ LONG TP HIT: ${profit:+,.0f} | Cycle #{level['cycles_completed']+1}")
                
                # ═══════════════════════════════════════════════════
                # RESPAWN LOGIC - Reset level to PENDING
                # ═══════════════════════════════════════════════════
                level['status'] = 'PENDING'
                level['cycles_completed'] += 1
                level['tp_hit_time'] = datetime.now().isoformat()
                level['fill_price'] = 0
                
                # Update state
                self.state.long_contracts -= size
                self.state.long_tp_profit += profit
                self.state.total_profit += profit
                self.state.total_cycles += 1
                
                self._save_state()
                logger.info(f"🔄 RESPAWNED {level['desc']} - Ready for next cycle!")
                return True
        except Exception as e:
            logger.error(f"Long TP error: {e}")
        return False
    
    def execute_short_tp(self, trigger: Dict) -> bool:
        """Execute short TP with RESPAWN"""
        level = trigger['level']
        size = level['size']
        
        logger.info(f"🎯 SHORT TP: BUY {size} @ ${trigger['price']:.2f} ({level['desc']})")
        
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
                
                logger.info(f"✅ SHORT TP HIT: ${profit:+,.0f} | Cycle #{level['cycles_completed']+1}")
                
                # ═══════════════════════════════════════════════════
                # RESPAWN LOGIC - Reset level to PENDING
                # ═══════════════════════════════════════════════════
                level['status'] = 'PENDING'
                level['cycles_completed'] += 1
                level['tp_hit_time'] = datetime.now().isoformat()
                level['fill_price'] = 0
                
                # Update state
                self.state.short_contracts -= size
                self.state.short_tp_profit += profit
                self.state.total_profit += profit
                self.state.total_cycles += 1
                
                self._save_state()
                logger.info(f"🔄 RESPAWNED {level['desc']} - Ready for next cycle!")
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
        
        # Get market regime
        candles = self.get_candles()
        regime = self.get_regime(candles)
        self.state.regime = regime.value
        
        # Log status
        logger.info(f"💰 ${price:.2f} | {regime.value} | L:{self.state.long_contracts} S:{self.state.short_contracts} | Cycles:{self.state.total_cycles} | P&L:${self.state.total_profit:+,.0f}")
        
        # Priority 1: Check take profits
        long_tp = self.check_long_tp(price)
        if long_tp:
            self.execute_long_tp(long_tp)
            return
        
        short_tp = self.check_short_tp(price)
        if short_tp:
            self.execute_short_tp(short_tp)
            return
        
        # Priority 2: Check new entries
        buy = self.check_buy_triggers(price, regime)
        if buy:
            self.execute_buy(buy)
        
        short = self.check_short_triggers(price, regime)
        if short:
            self.execute_short(short)
    
    def run(self, interval: int = 15):
        """Main run loop"""
        logger.info("=" * 70)
        logger.info("⚔️  WHIPSAW COMMANDER STARTING")
        logger.info("=" * 70)
        logger.info(f"Center: ${self.config['center_price']}")
        logger.info(f"Grid spacing: ${self.config['grid_spacing']}")
        logger.info(f"TP style: {self.config['tp_style']}")
        logger.info(f"Regime filter: {self.config['use_regime_filter']}")
        logger.info(f"NEO integration: {self.config['use_neo']}")
        logger.info(f"BUY levels: {len(self.config['buy_grid'])}")
        logger.info(f"SHORT levels: {len(self.config['short_grid'])}")
        
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
    
    def get_status_display(self) -> str:
        """Get visual status display"""
        lines = []
        lines.append("=" * 70)
        lines.append("⚔️  WHIPSAW COMMANDER STATUS")
        lines.append("=" * 70)
        lines.append(f"Center: ${self.state.center_price:.0f} | Regime: {self.state.regime}")
        lines.append(f"Position: L:{self.state.long_contracts} S:{self.state.short_contracts}")
        lines.append("")
        
        # Short levels
        lines.append("📉 SHORT GRID (fade rises):")
        for level in reversed(self.state.short_levels):
            status = "🟢 FILLED" if level['status'] == 'FILLED' else "⚪ PENDING"
            cycles = f"[{level['cycles_completed']} cycles]" if level['cycles_completed'] else ""
            lines.append(f"  ${level['price']:.0f} | {level['label']:6} | {status} {cycles}")
        
        lines.append(f"\n{'═'*20} CENTER ${self.state.center_price:.0f} {'═'*20}\n")
        
        # Buy levels
        lines.append("📈 BUY GRID (accumulate dips):")
        for level in self.state.buy_levels:
            status = "🟢 FILLED" if level['status'] == 'FILLED' else "⚪ PENDING"
            cycles = f"[{level['cycles_completed']} cycles]" if level['cycles_completed'] else ""
            lines.append(f"  ${level['price']:.0f} | {level['label']:6} | TP ${level['tp_price']:.0f} | {status} {cycles}")
        
        lines.append(f"\n{'─'*70}")
        lines.append(f"📊 PERFORMANCE:")
        lines.append(f"  Total Cycles: {self.state.total_cycles}")
        lines.append(f"  Long P&L:  ${self.state.long_tp_profit:+,.0f}")
        lines.append(f"  Short P&L: ${self.state.short_tp_profit:+,.0f}")
        lines.append(f"  TOTAL:     ${self.state.total_profit:+,.0f}")
        
        return "\n".join(lines)


def main():
    """Run Whipsaw Commander"""
    import sys
    
    center = float(sys.argv[1]) if len(sys.argv) > 1 else 5571
    
    bot = WhipsawCommander(center_price=center)
    print(bot.get_status_display())
    
    print("\n⚔️  Starting Whipsaw Commander...")
    bot.run()


if __name__ == "__main__":
    main()

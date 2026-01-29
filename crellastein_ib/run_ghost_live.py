#!/usr/bin/env python3
"""
Ghost Commander Live Runner v3.0 - FULL AUTONOMOUS TRADING
Runs 24/7 on H100, manages MGC positions automatically

NOW WITH ENTRY LOGIC:
- SuperTrend signal for initial entry
- DCA ladder on 30/60/90/120 pip drops
- Tiered TPs (TP1/TP2/TP3)
- Free Roll runner after TP1 + $500 profit
- LONG-ONLY MODE: Will NEVER go short

Usage:
    python run_ghost_live.py
    
Or as service:
    sudo systemctl start ghost-commander

Author: QUINN001
Created: January 29, 2026
v2.0: Race condition fix
v3.0: Full entry logic (SuperTrend + DCA)
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'neo'))

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder, LimitOrder
from indicators import Indicators

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IB_HOST = '100.119.161.65'      # Gringot's desktop via Tailscale
IB_PORT = 7497                   # Paper trading port (7496 for live)
CLIENT_ID = 100                  # Unique client ID for Ghost

# Alert webhook (Discord/Slack)
WEBHOOK_URL = os.environ.get('ALERT_WEBHOOK', '')

# Log file
LOG_DIR = Path('/home/jbot/trading_ai/logs')
LOG_FILE = LOG_DIR / 'ghost_commander.log'

# State file
STATE_DIR = Path(__file__).parent / 'ghost_data'
STATE_FILE = STATE_DIR / 'ghost_live_state.json'

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('GhostLive')


# ═══════════════════════════════════════════════════════════════════════════════
# GHOST COMMANDER LIVE v3.0 - FULL AUTONOMOUS TRADING
# ═══════════════════════════════════════════════════════════════════════════════

class GhostCommanderLive:
    """
    Ghost Commander for IB Gold Futures (MGC)
    
    v3.0 FEATURES:
    - SuperTrend-based initial entry (BULLISH only)
    - DCA ladder on pip drops (30/60/90/120)
    - Tiered take profits (TP1/TP2/TP3)
    - Free Roll runner
    - LONG-ONLY mode with safeguards
    """
    
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contract = None
        
        # Position tracking
        self.positions = []
        self.dca_total_contracts = 0
        self.dca_average_entry = 0
        self.dca_highest_entry = 0
        self.dca_ladder_count = 0
        
        # TP tracking
        self.tp1_hit = False
        self.tp2_hit = False
        self.daily_realized_pnl = 0
        
        # Runner
        self.runner_position = None
        
        # Settings
        self.settings = {
            # Entry settings
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
            'initial_contracts': 1,
            
            # DCA settings
            'dca_pip_spacing': 30,          # Pips between DCA levels
            'dca_max_levels': 5,            # Max DCA positions
            'dca_initial_lots': 0.5,        # Starting lots
            'dca_lot_increment': 0.25,      # Add per level
            'dca_min_entry_gap': 20,        # Min points between entries
            
            # TP settings
            'tp1_pips': 30,                 # +$3
            'tp2_pips': 60,                 # +$6
            'tp3_pips': 90,                 # +$9
            'tp1_close_percent': 30,
            'tp2_close_percent': 30,
            
            # Runner settings
            'runner_enabled': True,
            'runner_profit_threshold': 500,
            
            # Safety settings
            'max_total_contracts': 10,
            'entry_cooldown_seconds': 300,  # 5 min between entries
            'tp_cooldown_seconds': 10,
        }
        
        # Status tracking
        self.last_price = 0
        self.last_status_log = datetime.now()
        
        # Order safeguards
        self.pending_order = False
        self.order_in_flight = False
        self.last_tp_time = None
        self.last_entry_time = None
        
        # Candle cache
        self.candle_cache = []
        self.candle_cache_time = None
        
        # Load saved state
        self._load_state()
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    def connect(self) -> bool:
        """Connect to IB TWS"""
        try:
            if self.ib.isConnected():
                self.connected = True
                return True
            
            logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT}...")
            self.ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=30)
            self.ib.sleep(3)
            
            self.connected = self.ib.isConnected()
            
            if self.connected:
                accounts = self.ib.managedAccounts()
                logger.info(f"✅ Connected to account: {accounts}")
                
                self._init_contract()
                self.sync_positions()
                
                mode = "FULL AUTO" if self.dca_total_contracts == 0 else "MANAGING"
                send_alert(f"Ghost Commander v3.0 connected! {mode} mode, {self.dca_total_contracts} MGC contracts", "success")
            
            return self.connected
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IB"""
        self._save_state()
        if self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False
        logger.info("Disconnected from IB")
    
    def _init_contract(self):
        """Initialize MGC futures contract"""
        now = datetime.now()
        for month in [2, 4, 6, 8, 10, 12]:
            year = now.year if month > now.month else now.year + 1
            if (datetime(year, month, 25) - now).days >= 30:
                expiry = f"{year}{month:02d}"
                break
        else:
            expiry = f"{now.year + 1}04"
        
        self.contract = Future(
            symbol='MGC',
            lastTradeDateOrContractMonth=expiry,
            exchange='COMEX',
            currency='USD'
        )
        
        qualified = self.ib.qualifyContracts(self.contract)
        if qualified:
            self.contract = qualified[0]
            logger.info(f"Using contract: {self.contract.localSymbol}")
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def get_price(self) -> dict:
        """Get current MGC price"""
        if not self.connected or not self.contract:
            return None
        
        try:
            self.ib.reqMarketDataType(3)
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
            last = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            
            if last and last > 0:
                self.last_price = last
                return {'bid': bid or last, 'ask': ask or last, 'last': last}
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    def get_candles(self, duration: str = "5 D", bar_size: str = "1 hour") -> List[Dict]:
        """Get historical candles with caching"""
        # Check cache (5 min refresh)
        if self.candle_cache and self.candle_cache_time:
            age = (datetime.now() - self.candle_cache_time).total_seconds()
            if age < 300:  # 5 minutes
                return self.candle_cache
        
        if not self.connected or not self.contract:
            return self.candle_cache or []
        
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
            
            self.candle_cache = [
                {'open': b.open, 'high': b.high, 'low': b.low, 
                 'close': b.close, 'volume': b.volume}
                for b in bars
            ]
            self.candle_cache_time = datetime.now()
            
            return self.candle_cache
            
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return self.candle_cache or []
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def sync_positions(self):
        """Sync positions from IB"""
        if not self.connected:
            return
        
        try:
            self.positions = []
            self.ib.sleep(0.5)
            
            for pos in self.ib.positions():
                if pos.contract.symbol == 'MGC' and pos.position != 0:
                    self.positions.append({
                        'contracts': pos.position,
                        'avg_cost': pos.avgCost,
                        'entry_price': pos.avgCost / 10,
                    })
            
            self._calculate_ladder_stats()
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def _calculate_ladder_stats(self):
        """Calculate DCA ladder statistics"""
        self.dca_total_contracts = 0
        self.dca_average_entry = 0
        self.dca_highest_entry = 0
        self.dca_ladder_count = 0
        
        if not self.positions:
            return
        
        total_value = 0
        for pos in self.positions:
            contracts = pos['contracts']
            price = pos['entry_price']
            
            if contracts > 0:
                total_value += price * contracts
                self.dca_total_contracts += contracts
                self.dca_ladder_count += 1
                
                if price > self.dca_highest_entry:
                    self.dca_highest_entry = price
        
        if self.dca_total_contracts > 0:
            self.dca_average_entry = total_value / self.dca_total_contracts
    
    def get_net_long_position(self) -> int:
        """Get net LONG position count"""
        self.sync_positions()
        net = sum(p['contracts'] for p in self.positions)
        return max(0, int(net))
    
    # =========================================================================
    # ENTRY SAFEGUARDS
    # =========================================================================
    
    def can_execute_entry(self) -> bool:
        """Check if safe to execute new entry"""
        # Check pending order
        if self.pending_order or self.order_in_flight:
            return False
        
        # Entry cooldown (5 minutes)
        if self.last_entry_time:
            elapsed = (datetime.now() - self.last_entry_time).total_seconds()
            if elapsed < self.settings['entry_cooldown_seconds']:
                return False
        
        # TP cooldown (1 minute after TP)
        if self.last_tp_time:
            elapsed = (datetime.now() - self.last_tp_time).total_seconds()
            if elapsed < 60:
                return False
        
        # Max contracts limit
        if self.dca_total_contracts >= self.settings['max_total_contracts']:
            return False
        
        return True
    
    def can_execute_tp(self) -> bool:
        """Check if safe to execute TP"""
        if self.pending_order or self.order_in_flight:
            return False
        
        if self.last_tp_time:
            elapsed = (datetime.now() - self.last_tp_time).total_seconds()
            if elapsed < self.settings['tp_cooldown_seconds']:
                return False
        
        return True
    
    def validate_sell_quantity(self, requested: int) -> int:
        """CRITICAL: Never go short"""
        current_long = self.get_net_long_position()
        
        if current_long <= 0:
            logger.error("🚨 BLOCKED: No long positions to sell!")
            return 0
        
        if requested > current_long:
            logger.warning(f"⚠️ REDUCED: Selling {current_long} instead of {requested}")
            return current_long
        
        return requested
    
    # =========================================================================
    # ENTRY LOGIC - SuperTrend + DCA
    # =========================================================================
    
    def check_initial_entry(self, current_price: float) -> Optional[Dict]:
        """
        Check if should open initial position.
        REQUIRES: SuperTrend = BULLISH (trend = 1)
        """
        # Already have positions
        if self.dca_total_contracts > 0:
            return None
        
        # Get candles for SuperTrend
        candles = self.get_candles("5 D", "1 hour")
        if len(candles) < 20:
            logger.warning("Not enough candles for SuperTrend")
            return None
        
        # Calculate SuperTrend
        st = Indicators.supertrend(
            candles,
            period=self.settings['supertrend_period'],
            multiplier=self.settings['supertrend_multiplier']
        )
        
        # ONLY enter if BULLISH
        if st['trend'] != 1:
            logger.debug(f"SuperTrend BEARISH ({st['trend']}) - no entry")
            return None
        
        logger.info(f"✅ SuperTrend BULLISH - initiating entry @ ${current_price:.2f}")
        logger.info(f"   ST Value: ${st['value']:.2f}, Lower Band: ${st['lower_band']:.2f}")
        
        return {
            'type': 'INITIAL',
            'level': 1,
            'contracts': self.settings['initial_contracts'],
            'price': current_price,
            'comment': 'GHOST|ENTRY|ST_BULL'
        }
    
    def check_dca_ladder(self, current_price: float) -> Optional[Dict]:
        """
        Check if should add DCA position on dip.
        REQUIRES: Price dropped 30/60/90/120 pips from highest entry
        """
        if self.dca_total_contracts == 0:
            return None  # Use check_initial_entry first
        
        if self.dca_ladder_count >= self.settings['dca_max_levels']:
            return None  # Max levels reached
        
        # Calculate drop from highest entry
        drop_pips = (self.dca_highest_entry - current_price) * 10
        
        # Required drop for next level
        required_drop = self.settings['dca_pip_spacing'] * self.dca_ladder_count
        
        if drop_pips < required_drop:
            return None  # Not enough drop
        
        # Anti-stack check
        for pos in self.positions:
            distance = abs(current_price - pos['entry_price']) * 10
            if distance < self.settings['dca_min_entry_gap']:
                logger.debug("Anti-stack: Too close to existing position")
                return None
        
        # Calculate contracts for this level
        lots = self.settings['dca_initial_lots'] + (
            self.settings['dca_lot_increment'] * self.dca_ladder_count
        )
        contracts = max(1, int(lots))
        
        logger.info(f"📉 DCA TRIGGER: Drop {drop_pips:.1f} pips >= {required_drop} required")
        
        return {
            'type': 'DCA',
            'level': self.dca_ladder_count + 1,
            'contracts': contracts,
            'price': current_price,
            'drop_pips': drop_pips,
            'comment': f'GHOST|DCA{self.dca_ladder_count + 1}|{drop_pips:.0f}p'
        }
    
    def execute_entry(self, entry: Dict) -> bool:
        """Execute BUY entry with safeguards"""
        self.pending_order = True
        self.order_in_flight = True
        
        try:
            contracts = entry['contracts']
            
            order = MarketOrder('BUY', contracts)
            order.orderRef = entry['comment']
            
            logger.info(f"📈 EXECUTING ENTRY: {entry['comment']}")
            logger.info(f"   Contracts: {contracts}, Price: ~${entry['price']:.2f}")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            # Wait for fill
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    logger.error("⏱️ Entry order timeout!")
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                filled_qty = trade.orderStatus.filled
                
                logger.info(f"✅ FILLED: Bought {int(filled_qty)} @ ${fill_price:.2f}")
                send_alert(f"🟢 BOUGHT {int(filled_qty)} MGC @ ${fill_price:.2f} - {entry['comment']}", "success")
                
                self.last_entry_time = datetime.now()
                self._save_state()
                return True
            else:
                logger.error(f"❌ Entry failed: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"Entry execution failed: {e}")
            return False
            
        finally:
            # Re-sync positions
            self.ib.sleep(2)
            self.sync_positions()
            self.pending_order = False
            self.order_in_flight = False
    
    # =========================================================================
    # TAKE PROFIT LOGIC
    # =========================================================================
    
    def check_take_profits(self, current_price: float) -> Optional[Dict]:
        """Check tiered take profit levels"""
        if self.dca_total_contracts <= 0 or self.dca_average_entry == 0:
            return None
        
        profit_pips = (current_price - self.dca_average_entry) * 10
        
        # TP3: Full exit at +90 pips
        if profit_pips >= self.settings['tp3_pips']:
            return {
                'action': 'CLOSE_ALL',
                'level': 'TP3',
                'profit_pips': profit_pips,
                'contracts': self.dca_total_contracts
            }
        
        # TP2: Close 30% at +60 pips
        if not self.tp2_hit and profit_pips >= self.settings['tp2_pips']:
            close_pct = self.settings['tp2_close_percent'] / 100
            contracts = max(1, int(self.dca_total_contracts * close_pct))
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP2',
                'profit_pips': profit_pips,
                'contracts': contracts
            }
        
        # TP1: Close 30% at +30 pips
        if not self.tp1_hit and profit_pips >= self.settings['tp1_pips']:
            close_pct = self.settings['tp1_close_percent'] / 100
            contracts = max(1, int(self.dca_total_contracts * close_pct))
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP1',
                'profit_pips': profit_pips,
                'contracts': contracts
            }
        
        return None
    
    def execute_take_profit(self, tp_info: dict, current_price: float) -> bool:
        """Execute take profit order"""
        self.order_in_flight = True
        self.pending_order = True
        
        try:
            if tp_info['action'] == 'CLOSE_ALL':
                requested_qty = self.get_net_long_position()
            else:
                requested_qty = int(tp_info['contracts'])
            
            safe_qty = self.validate_sell_quantity(requested_qty)
            if safe_qty <= 0:
                logger.error(f"🚨 ABORT {tp_info['level']}: No valid quantity!")
                return False
            
            order = MarketOrder('SELL', safe_qty)
            order.orderRef = f"GHOST|{tp_info['level']}"
            
            logger.info(f"📤 Executing {tp_info['level']}: SELL {safe_qty} @ MKT")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    logger.error(f"⏱️ {tp_info['level']} timeout!")
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                filled_qty = trade.orderStatus.filled
                
                profit = (fill_price - self.dca_average_entry) * filled_qty * 10
                self.daily_realized_pnl += profit
                
                if tp_info['level'] == 'TP1':
                    self.tp1_hit = True
                elif tp_info['level'] == 'TP2':
                    self.tp2_hit = True
                elif tp_info['level'] == 'TP3':
                    self.tp1_hit = False
                    self.tp2_hit = False
                
                msg = f"🎯 {tp_info['level']} HIT! Closed {int(filled_qty)} @ ${fill_price:.2f} = +${profit:.2f}"
                logger.info(msg)
                send_alert(msg, "success")
                
                if tp_info['level'] == 'TP1' and self.check_runner_trigger():
                    self.deploy_runner(fill_price)
                
                self._save_state()
                return True
            else:
                logger.error(f"❌ {tp_info['level']} failed: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"TP failed: {e}")
            return False
            
        finally:
            self.last_tp_time = datetime.now()
            self.ib.sleep(3)
            self.sync_positions()
            self.order_in_flight = False
            self.pending_order = False
    
    # =========================================================================
    # FREE ROLL RUNNER
    # =========================================================================
    
    def check_runner_trigger(self) -> bool:
        """Check if should deploy Free Roll runner"""
        if not self.settings['runner_enabled']:
            return False
        if self.runner_position is not None:
            return False
        if self.daily_realized_pnl < self.settings['runner_profit_threshold']:
            return False
        return True
    
    def deploy_runner(self, current_price: float):
        """Deploy Free Roll runner - funded by profits, NO SL"""
        try:
            order = MarketOrder('BUY', 1)
            order.orderRef = 'GHOST|RUNNER|PROFIT_FUNDED'
            
            logger.info(f"🚀 Deploying FREE ROLL RUNNER @ ${current_price:.2f}")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                
                self.runner_position = {
                    'entry_price': fill_price,
                    'contracts': 1,
                    'funded_by': self.daily_realized_pnl,
                    'time': datetime.now().isoformat()
                }
                
                self._save_state()
                
                msg = f"🚀 RUNNER deployed! Entry: ${fill_price:.2f}, Funded: ${self.daily_realized_pnl:.2f}"
                logger.info(msg)
                send_alert(msg, "success")
                
        except Exception as e:
            logger.error(f"Runner failed: {e}")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run_once(self):
        """Run one iteration - FULL TRADING LOGIC"""
        if not self.connected:
            return
        
        try:
            self.sync_positions()
            
            price_data = self.get_price()
            if not price_data:
                return
            
            current_price = price_data['last']
            
            # === ENTRY LOGIC ===
            if self.can_execute_entry():
                if self.dca_total_contracts == 0:
                    # No positions - check for initial entry
                    entry = self.check_initial_entry(current_price)
                    if entry:
                        self.execute_entry(entry)
                
                elif self.dca_ladder_count < self.settings['dca_max_levels']:
                    # Have positions - check for DCA
                    dca = self.check_dca_ladder(current_price)
                    if dca:
                        self.execute_entry(dca)
            
            # === TP LOGIC ===
            if self.can_execute_tp() and self.dca_total_contracts > 0:
                tp_action = self.check_take_profits(current_price)
                if tp_action:
                    self.execute_take_profit(tp_action, current_price)
            
            # === STATUS LOG (every 60 seconds) ===
            if (datetime.now() - self.last_status_log).seconds >= 60:
                self._log_status(current_price)
                self.last_status_log = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
    
    def _log_status(self, current_price: float):
        """Log current status"""
        # Check SuperTrend for status
        candles = self.get_candles("5 D", "1 hour")
        st_trend = "?"
        if candles and len(candles) >= 20:
            st = Indicators.supertrend(candles, 10, 3.0)
            st_trend = "BULL" if st['trend'] == 1 else "BEAR"
        
        if self.dca_total_contracts > 0:
            unrealized = (current_price - self.dca_average_entry) * self.dca_total_contracts * 10
            tp1_price = self.dca_average_entry + 3
            tp2_price = self.dca_average_entry + 6
            tp3_price = self.dca_average_entry + 9
            
            logger.info(
                f"📊 MGC: ${current_price:.2f} | "
                f"L{self.dca_ladder_count} ({self.dca_total_contracts}x) @ ${self.dca_average_entry:.2f} | "
                f"P&L: ${unrealized:.2f} | "
                f"TPs: ${tp1_price:.2f}{'✓' if self.tp1_hit else ''} / "
                f"${tp2_price:.2f}{'✓' if self.tp2_hit else ''} / "
                f"${tp3_price:.2f} | "
                f"ST: {st_trend}"
            )
        else:
            logger.info(f"📊 MGC: ${current_price:.2f} | No positions | ST: {st_trend} | Watching for entry...")
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'daily_realized_pnl': self.daily_realized_pnl,
            'runner_position': self.runner_position,
            'last_tp_time': self.last_tp_time.isoformat() if self.last_tp_time else None,
            'last_entry_time': self.last_entry_time.isoformat() if self.last_entry_time else None,
            'last_update': datetime.now().isoformat()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from file"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.tp1_hit = state.get('tp1_hit', False)
                    self.tp2_hit = state.get('tp2_hit', False)
                    self.daily_realized_pnl = state.get('daily_realized_pnl', 0)
                    self.runner_position = state.get('runner_position')
                    
                    if state.get('last_tp_time'):
                        self.last_tp_time = datetime.fromisoformat(state['last_tp_time'])
                    if state.get('last_entry_time'):
                        self.last_entry_time = datetime.fromisoformat(state['last_entry_time'])
                    
                    logger.info(f"Loaded state: TP1={self.tp1_hit}, TP2={self.tp2_hit}, PnL=${self.daily_realized_pnl:.2f}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def get_status(self) -> dict:
        """Get current status for API"""
        return {
            'connected': self.connected,
            'contracts': self.dca_total_contracts,
            'dca_level': self.dca_ladder_count,
            'avg_entry': self.dca_average_entry,
            'current_price': self.last_price,
            'unrealized_pnl': (self.last_price - self.dca_average_entry) * self.dca_total_contracts * 10 if self.dca_total_contracts > 0 else 0,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'daily_realized': self.daily_realized_pnl,
            'runner_active': self.runner_position is not None,
            'mode': 'FULL_AUTO',
            'last_update': datetime.now().isoformat()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def send_alert(message: str, level: str = 'info'):
    """Send alert via webhook"""
    emoji = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}.get(level, '📢')
    logger.info(f"ALERT [{level.upper()}]: {message}")
    
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={
                'content': f"{emoji} **Ghost Commander v3.0** [{level.upper()}]\n{message}",
                'username': 'Ghost Commander'
            }, timeout=5)
        except:
            pass


def check_market_hours() -> bool:
    """Check if gold futures market is open"""
    now = datetime.utcnow()
    if now.weekday() == 5:  # Saturday
        return False
    if now.weekday() == 6 and now.hour < 22:  # Sunday before 6pm ET
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    logger.info("=" * 70)
    logger.info("🚀 GHOST COMMANDER v3.0 - FULL AUTONOMOUS TRADING")
    logger.info(f"   IB: {IB_HOST}:{IB_PORT}")
    logger.info(f"   Entry: SuperTrend BULLISH required")
    logger.info(f"   DCA: 30/60/90/120 pip drops, max 5 levels")
    logger.info(f"   TPs: +$3 / +$6 / +$9 (30%/30%/ALL)")
    logger.info(f"   Mode: LONG-ONLY")
    logger.info("=" * 70)
    
    send_alert("Ghost Commander v3.0 starting - FULL AUTO mode", "info")
    
    ghost = GhostCommanderLive()
    reconnect_delay = 30
    max_reconnect_delay = 300
    
    while True:
        try:
            if not check_market_hours():
                logger.info("Market closed. Sleeping 1 hour...")
                time.sleep(3600)
                continue
            
            if not ghost.connected:
                if ghost.connect():
                    reconnect_delay = 30
                else:
                    raise Exception("Connection failed")
            
            ghost.run_once()
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            send_alert("Ghost Commander shutting down", "warning")
            break
            
        except Exception as e:
            logger.error(f"Error: {e}")
            
            try:
                ghost.disconnect()
            except:
                pass
            ghost.connected = False
            
            logger.info(f"Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    ghost.disconnect()
    logger.info("Ghost Commander stopped.")


if __name__ == '__main__':
    main()

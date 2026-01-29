#!/usr/bin/env python3
"""
Casper Live Runner v1.0 - Drop-Buy Martingale Strategy
Runs 24/7 on H100, manages MGC positions automatically

CASPER STRATEGY:
- Track session high
- Buy on every $10 drop from session high
- Lot ladder: [0.5, 0.5, 1.0, 1.0, 2.0] = 5 levels
- TP at +$20 from average entry
- Trailing TP: Start at +$10, trail $8 behind
- LONG-ONLY MODE: Will NEVER go short

Usage:
    python run_casper_live.py
    
Or as service:
    sudo systemctl start casper

Author: QUINN001
Created: January 30, 2026
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

from ib_insync import IB, Future, MarketOrder

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IB_HOST = '100.119.161.65'      # Gringot's desktop via Tailscale
IB_PORT = 7497                   # Paper trading port (7496 for live)
CLIENT_ID = 200                  # Different from Ghost (100)

# Alert webhook
WEBHOOK_URL = os.environ.get('ALERT_WEBHOOK', '')

# Log file
LOG_DIR = Path('/home/jbot/trading_ai/logs')
LOG_FILE = LOG_DIR / 'casper.log'

# State file
STATE_DIR = Path(__file__).parent / 'casper_data'
STATE_FILE = STATE_DIR / 'casper_live_state.json'

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | CASPER | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CasperLive')


# ═══════════════════════════════════════════════════════════════════════════════
# CASPER LIVE - DROP-BUY MARTINGALE
# ═══════════════════════════════════════════════════════════════════════════════

class CasperLive:
    """
    Casper Drop-Buy Martingale Strategy
    
    STRATEGY:
    1. Track session high
    2. Buy on every $10 drop from session high
    3. Lot ladder: [0.5, 0.5, 1.0, 1.0, 2.0]
    4. TP at +$20 from average entry OR trailing stop
    5. LONG-ONLY mode
    """
    
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contract = None
        
        # Settings - HALF SIZE FOR TESTING (50% of MT5 equivalent)
        # Full MT5: [5, 5, 10, 10, 20] = 50 contracts (0.5, 0.5, 1.0, 1.0, 2.0 lots)
        self.settings = {
            # Drop-buy settings - HALF SIZE
            'trigger_drop': 10.0,           # $10 drop triggers buy
            'lot_ladder': [2, 2, 5, 5, 10], # HALF SIZE! (full=[5,5,10,10,20])
            'max_levels': 5,
            
            # TP settings (same as MT5)
            'tp_pips': 20.0,                # +$20 from avg entry
            'trail_start': 10.0,            # Start trailing at +$10
            'trail_distance': 8.0,          # Trail $8 behind
            
            # Safety - HALF SIZE LIMITS
            'max_contracts': 24,            # Sum of ladder (2+2+5+5+10)
            'entry_cooldown_seconds': 120,  # 2 min between entries
            
            # PARABOLIC MOVE SAFEGUARDS
            'max_drawdown_percent': 10.0,   # Close ALL if down 10% from peak
            'parabolic_threshold_pct': 5.0, # Warn if +5% in 5 days
            'max_position_percent': 30.0,   # Max 30% of account in gold
        }
        
        # Position tracking
        self.positions = []
        self.total_contracts = 0
        self.average_entry = 0
        
        # State tracking
        self.session_high = 0
        self.current_level = 0
        self.trailing_stop = 0
        
        # PARABOLIC SAFEGUARD TRACKING
        self.peak_equity = 0
        self.parabolic_mode = False
        self.safeguard_triggered = False
        
        # Status
        self.last_price = 0
        self.last_status_log = datetime.now()
        
        # Safeguards
        self.pending_order = False
        self.last_entry_time = None
        
        # Load state
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
                
                # Initialize session high
                if self.session_high == 0:
                    price = self.get_price()
                    if price:
                        self.session_high = price['last']
                
                send_alert(f"Casper connected! Session high: ${self.session_high:.2f}, Level: {self.current_level}", "success")
            
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
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def sync_positions(self):
        """Sync MGC positions from IB"""
        if not self.connected:
            return
        
        try:
            self.positions = []
            self.ib.sleep(0.5)
            
            for pos in self.ib.positions():
                if pos.contract.symbol == 'MGC' and pos.position > 0:
                    self.positions.append({
                        'quantity': pos.position,
                        'entry_price': pos.avgCost / 10,
                    })
            
            # Calculate totals
            self.total_contracts = sum(p['quantity'] for p in self.positions)
            
            if self.total_contracts > 0:
                total_value = sum(p['entry_price'] * p['quantity'] for p in self.positions)
                self.average_entry = total_value / self.total_contracts
            else:
                self.average_entry = 0
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    # =========================================================================
    # ENTRY SAFEGUARDS
    # =========================================================================
    
    def can_execute_entry(self) -> bool:
        """Check if safe to execute entry"""
        if self.pending_order:
            return False
        
        # Entry cooldown
        if self.last_entry_time:
            elapsed = (datetime.now() - self.last_entry_time).total_seconds()
            if elapsed < self.settings['entry_cooldown_seconds']:
                return False
        
        # Max contracts
        if self.total_contracts >= self.settings['max_contracts']:
            return False
        
        # Max levels
        if self.current_level >= self.settings['max_levels']:
            return False
        
        return True
    
    def validate_sell_quantity(self, requested: int) -> int:
        """Never go short"""
        self.sync_positions()
        current = int(self.total_contracts)
        
        if current <= 0:
            return 0
        
        return min(requested, current)
    
    # =========================================================================
    # DROP-BUY LOGIC
    # =========================================================================
    
    def check_drop_buy(self, current_price: float) -> Optional[Dict]:
        """
        Check if should execute drop-buy.
        Buy on every $10 drop from session high.
        """
        # Update session high
        if current_price > self.session_high:
            self.session_high = current_price
            logger.debug(f"New session high: ${self.session_high:.2f}")
            return None
        
        # Already at max level?
        if self.current_level >= self.settings['max_levels']:
            return None
        
        # Calculate drop from session high
        drop = self.session_high - current_price
        
        # Required drop for next level
        # Level 0->1: $10, Level 1->2: $20, etc.
        trigger_drop = self.settings['trigger_drop']
        
        # PARABOLIC MODE: Widen spacing by 50% ($10 → $15)
        if self.parabolic_mode:
            trigger_drop *= 1.5
        
        required_drop = trigger_drop * (self.current_level + 1)
        
        if drop >= required_drop:
            # Get contracts for this level
            contracts = self.settings['lot_ladder'][self.current_level]
            
            logger.info(f"📉 DROP-BUY TRIGGER: ${drop:.2f} drop from ${self.session_high:.2f}")
            
            return {
                'level': self.current_level + 1,
                'contracts': contracts,
                'drop': drop,
                'comment': f'CASPER|DROP{self.current_level + 1}|${drop:.0f}'
            }
        
        return None
    
    def execute_drop_buy(self, entry: Dict) -> bool:
        """Execute drop-buy entry"""
        self.pending_order = True
        
        try:
            contracts = entry['contracts']
            
            order = MarketOrder('BUY', contracts)
            order.orderRef = entry['comment']
            
            logger.info(f"📈 EXECUTING DROP-BUY: {entry['comment']}")
            logger.info(f"   Contracts: {contracts}")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            # Wait for fill
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    logger.error("⏱️ Order timeout!")
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                filled_qty = trade.orderStatus.filled
                
                logger.info(f"✅ FILLED: Bought {int(filled_qty)} @ ${fill_price:.2f}")
                send_alert(f"👻 CASPER bought {int(filled_qty)} MGC @ ${fill_price:.2f} - {entry['comment']}", "success")
                
                self.current_level = entry['level']
                self.last_entry_time = datetime.now()
                self._save_state()
                return True
            else:
                logger.error(f"❌ Order failed: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"Drop-buy failed: {e}")
            return False
            
        finally:
            self.ib.sleep(2)
            self.sync_positions()
            self.pending_order = False
    
    # =========================================================================
    # TAKE PROFIT LOGIC
    # =========================================================================
    
    def check_take_profit(self, current_price: float) -> Optional[Dict]:
        """
        Check TP conditions:
        1. Fixed TP: +$20 from average entry
        2. Trailing: Start at +$10, trail $8 behind
        """
        if self.total_contracts <= 0 or self.average_entry == 0:
            return None
        
        profit = current_price - self.average_entry
        
        # Fixed TP at +$20
        if profit >= self.settings['tp_pips']:
            return {
                'action': 'CLOSE_ALL',
                'reason': f'TP_HIT +${profit:.2f}',
                'profit': profit
            }
        
        # Trailing TP logic
        if profit >= self.settings['trail_start']:
            new_stop = current_price - self.settings['trail_distance']
            
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                logger.info(f"📈 Trailing stop moved to ${self.trailing_stop:.2f}")
                self._save_state()
        
        # Check if trailing stop hit
        if self.trailing_stop > 0 and current_price <= self.trailing_stop:
            return {
                'action': 'CLOSE_ALL',
                'reason': f'TRAILING_STOP @ ${self.trailing_stop:.2f}',
                'profit': profit
            }
        
        return None
    
    def execute_close_all(self, tp_info: Dict, current_price: float) -> bool:
        """Close all positions"""
        self.pending_order = True
        
        try:
            qty = self.validate_sell_quantity(int(self.total_contracts))
            if qty <= 0:
                logger.error("🚨 No positions to close!")
                return False
            
            order = MarketOrder('SELL', qty)
            order.orderRef = f'CASPER|TP|{tp_info["reason"][:20]}'
            
            logger.info(f"📤 CLOSING ALL: {qty} contracts - {tp_info['reason']}")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    logger.error("⏱️ Close timeout!")
                    break
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                profit = (fill_price - self.average_entry) * qty * 10
                
                logger.info(f"✅ CLOSED: Sold {qty} @ ${fill_price:.2f} = +${profit:.2f}")
                send_alert(f"👻 CASPER closed {qty} MGC @ ${fill_price:.2f} - {tp_info['reason']} = +${profit:.2f}", "success")
                
                # Reset state
                self.current_level = 0
                self.trailing_stop = 0
                # Reset session high to current price
                self.session_high = fill_price
                
                self._save_state()
                return True
            else:
                logger.error(f"❌ Close failed: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"Close failed: {e}")
            return False
            
        finally:
            self.ib.sleep(2)
            self.sync_positions()
            self.pending_order = False
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run_once(self):
        """Run one iteration - with SAFEGUARDS"""
        if not self.connected:
            return
        
        # If safeguard was triggered, don't trade
        if self.safeguard_triggered:
            return
        
        try:
            self.sync_positions()
            
            price_data = self.get_price()
            if not price_data:
                return
            
            current_price = price_data['last']
            
            # === SAFEGUARD CHECKS (FIRST!) ===
            if self.run_safeguard_checks():
                return  # Stop trading if safeguard triggered
            
            # === TP CHECK (first priority) ===
            tp = self.check_take_profit(current_price)
            if tp:
                self.execute_close_all(tp, current_price)
                return
            
            # === ENTRY CHECK ===
            if self.can_execute_entry():
                # Check position limit before any entry
                if not self.check_position_limit():
                    pass  # Position too large, skip entry
                else:
                    entry = self.check_drop_buy(current_price)
                    if entry:
                        # In parabolic mode, halve the contracts
                        if self.parabolic_mode:
                            entry['contracts'] = max(1, entry['contracts'] // 2)
                            entry['comment'] += '|PARA'
                        self.execute_drop_buy(entry)
            
            # === STATUS LOG ===
            if (datetime.now() - self.last_status_log).seconds >= 60:
                self._log_status(current_price)
                self.last_status_log = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
    
    def _log_status(self, current_price: float):
        """Log current status"""
        drop = self.session_high - current_price if self.session_high > 0 else 0
        next_trigger = self.settings['trigger_drop'] * (self.current_level + 1) if self.current_level < self.settings['max_levels'] else 0
        
        if self.total_contracts > 0:
            profit = current_price - self.average_entry
            unrealized = profit * self.total_contracts * 10
            
            trail_info = f"Trail:${self.trailing_stop:.2f}" if self.trailing_stop > 0 else ""
            
            logger.info(
                f"👻 MGC: ${current_price:.2f} | "
                f"High: ${self.session_high:.2f} | "
                f"Drop: ${drop:.2f} | "
                f"L{self.current_level} ({self.total_contracts}x) @ ${self.average_entry:.2f} | "
                f"P&L: ${unrealized:.2f} {trail_info}"
            )
        else:
            logger.info(
                f"👻 MGC: ${current_price:.2f} | "
                f"High: ${self.session_high:.2f} | "
                f"Drop: ${drop:.2f} | "
                f"Next trigger: ${next_trigger:.0f} drop | "
                f"Watching..."
            )
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save state"""
        state = {
            'session_high': self.session_high,
            'current_level': self.current_level,
            'trailing_stop': self.trailing_stop,
            'last_entry_time': self.last_entry_time.isoformat() if self.last_entry_time else None,
            'last_update': datetime.now().isoformat()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.session_high = state.get('session_high', 0)
                    self.current_level = state.get('current_level', 0)
                    self.trailing_stop = state.get('trailing_stop', 0)
                    
                    if state.get('last_entry_time'):
                        self.last_entry_time = datetime.fromisoformat(state['last_entry_time'])
                    
                    logger.info(f"Loaded state: High=${self.session_high:.2f}, Level={self.current_level}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def get_status(self) -> dict:
        """Get status for API"""
        return {
            'connected': self.connected,
            'contracts': self.total_contracts,
            'level': self.current_level,
            'avg_entry': self.average_entry,
            'session_high': self.session_high,
            'current_price': self.last_price,
            'drop': self.session_high - self.last_price if self.session_high > 0 else 0,
            'trailing_stop': self.trailing_stop,
            'unrealized_pnl': (self.last_price - self.average_entry) * self.total_contracts * 10 if self.total_contracts > 0 else 0,
            'parabolic_mode': self.parabolic_mode,
            'safeguard_triggered': self.safeguard_triggered,
            'last_update': datetime.now().isoformat()
        }
    
    # =========================================================================
    # PARABOLIC MOVE SAFEGUARDS
    # =========================================================================
    
    def get_account_equity(self) -> float:
        """Get current account equity from IB"""
        try:
            account_values = self.ib.accountSummary()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    return float(av.value)
            return 0
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            return 0
    
    def check_drawdown_stop(self) -> bool:
        """
        CRITICAL: Emergency stop if drawdown exceeds limit
        Close ALL if down 10% from peak equity
        """
        current_equity = self.get_account_equity()
        if current_equity <= 0:
            return False
        
        # Track peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown %
        if self.peak_equity > 0:
            drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
            
            if drawdown_pct >= self.settings['max_drawdown_percent']:
                logger.critical(f"🚨 DRAWDOWN STOP: {drawdown_pct:.1f}% >= {self.settings['max_drawdown_percent']}%")
                send_alert(f"🚨 CASPER DRAWDOWN STOP! {drawdown_pct:.1f}% loss - closing ALL", "error")
                self.emergency_close_all("DRAWDOWN_STOP")
                self.safeguard_triggered = True
                return True
        
        return False
    
    def check_position_limit(self) -> bool:
        """Check if position is too large relative to account - blocks new entries"""
        account_value = self.get_account_equity()
        if account_value <= 0:
            return True  # Can't check, allow
        
        # MGC = 10 oz per contract
        position_value = self.total_contracts * self.last_price * 10
        position_pct = (position_value / account_value) * 100
        
        if position_pct >= self.settings['max_position_percent']:
            logger.warning(f"⚠️ POSITION LIMIT: {position_pct:.1f}% of account (max {self.settings['max_position_percent']}%)")
            return False  # Block new entries
        
        return True  # OK to add
    
    def get_candles(self, duration: str = "5 D", bar_size: str = "1 hour") -> list:
        """Get historical candles for parabolic detection"""
        if not self.connected or not self.contract:
            return []
        
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
            return [{'open': b.open, 'high': b.high, 'low': b.low, 
                     'close': b.close, 'volume': b.volume} for b in bars]
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return []
    
    def is_parabolic_move(self) -> bool:
        """
        Detect parabolic move - high correction risk!
        Triggers if price rose >5% in 5 days
        """
        candles = self.get_candles("5 D", "1 hour")
        if len(candles) < 120:
            return False
        
        price_5_days_ago = candles[0]['close']
        current_price = candles[-1]['close']
        
        gain_pct = ((current_price - price_5_days_ago) / price_5_days_ago) * 100
        
        if gain_pct > self.settings['parabolic_threshold_pct']:
            logger.warning(f"⚠️ PARABOLIC WARNING: +{gain_pct:.1f}% in 5 days!")
            return True
        
        return False
    
    def adjust_for_parabolic(self):
        """Reduce risk during parabolic moves"""
        if not self.parabolic_mode:
            self.parabolic_mode = True
            logger.info("📉 ENTERING PARABOLIC MODE: Reduced size")
            send_alert("⚠️ CASPER: Parabolic mode - reducing risk", "warning")
    
    def emergency_close_all(self, reason: str):
        """Emergency close all positions"""
        if self.total_contracts <= 0:
            return
        
        try:
            qty = self.validate_sell_quantity(int(self.total_contracts))
            if qty <= 0:
                return
            
            order = MarketOrder('SELL', qty)
            order.orderRef = f'CASPER|EMERGENCY|{reason}'
            
            logger.critical(f"🚨 EMERGENCY CLOSE: {qty} contracts - {reason}")
            
            trade = self.ib.placeOrder(self.contract, order)
            
            timeout = 30
            start = datetime.now()
            while not trade.isDone():
                self.ib.sleep(0.5)
                if (datetime.now() - start).total_seconds() > timeout:
                    break
            
            if trade.orderStatus.status == 'Filled':
                logger.info(f"✅ EMERGENCY CLOSE FILLED @ ${trade.orderStatus.avgFillPrice:.2f}")
                send_alert(f"🚨 CASPER emergency close: {qty} @ ${trade.orderStatus.avgFillPrice:.2f}", "error")
            
            # Reset state
            self.current_level = 0
            self.trailing_stop = 0
            
            self.ib.sleep(2)
            self.sync_positions()
            
        except Exception as e:
            logger.error(f"Emergency close failed: {e}")
    
    def run_safeguard_checks(self) -> bool:
        """
        Run all safeguard checks - returns True if trading should STOP
        """
        # 1. Drawdown stop (most critical)
        if self.check_drawdown_stop():
            return True
        
        # 2. Parabolic move detection (adjust risk)
        if self.is_parabolic_move():
            self.adjust_for_parabolic()
        else:
            self.parabolic_mode = False
        
        return False


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
                'content': f"{emoji} **Casper** [{level.upper()}]\n{message}",
                'username': 'Casper'
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
    logger.info("👻 CASPER LIVE v1.0 - DROP-BUY MARTINGALE")
    logger.info(f"   IB: {IB_HOST}:{IB_PORT}")
    logger.info(f"   Drop Trigger: ${10} per level")
    logger.info(f"   Lot Ladder: [1, 1, 2, 2, 4] contracts")
    logger.info(f"   TP: +$20 from avg OR trailing (+$10 start, $8 trail)")
    logger.info(f"   Mode: LONG-ONLY")
    logger.info("=" * 70)
    
    send_alert("Casper starting - Drop-Buy Martingale", "info")
    
    casper = CasperLive()
    reconnect_delay = 30
    max_reconnect_delay = 300
    
    while True:
        try:
            if not check_market_hours():
                logger.info("Market closed. Sleeping 1 hour...")
                time.sleep(3600)
                continue
            
            if not casper.connected:
                if casper.connect():
                    reconnect_delay = 30
                else:
                    raise Exception("Connection failed")
            
            casper.run_once()
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            send_alert("Casper shutting down", "warning")
            break
            
        except Exception as e:
            logger.error(f"Error: {e}")
            
            try:
                casper.disconnect()
            except:
                pass
            casper.connected = False
            
            logger.info(f"Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    casper.disconnect()
    logger.info("Casper stopped.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Ghost Commander Live Runner
Runs 24/7 on H100, manages MGC positions automatically

Features:
- Auto-reconnect on disconnect
- Tiered TPs (TP1/TP2/TP3)
- Free Roll runner deployment
- Webhook alerts
- Survives reboots via systemd

Usage:
    python run_ghost_live.py
    
Or as service:
    sudo systemctl start ghost-commander

Author: QUINN001
Created: January 29, 2026
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'neo'))

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder, LimitOrder

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
# GHOST COMMANDER LIVE
# ═══════════════════════════════════════════════════════════════════════════════

class GhostCommanderLive:
    """
    Live trading version of Ghost Commander
    Monitors MGC positions and executes tiered TPs
    """
    
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contract = None
        
        # Position tracking
        self.positions = []
        self.dca_total_contracts = 0
        self.dca_average_entry = 0
        
        # TP tracking
        self.tp1_hit = False
        self.tp2_hit = False
        self.daily_realized_pnl = 0
        
        # Runner
        self.runner_position = None
        
        # Settings (from config)
        self.settings = {
            'tp1_pips': 30,              # +$3
            'tp2_pips': 60,              # +$6
            'tp3_pips': 90,              # +$9
            'tp1_close_percent': 30,
            'tp2_close_percent': 30,
            'runner_enabled': True,
            'runner_profit_threshold': 500,
        }
        
        # Status tracking
        self.last_price = 0
        self.last_status_log = datetime.now()
        
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
                
                # Initialize contract
                self._init_contract()
                
                # Sync positions
                self.sync_positions()
                
                send_alert(f"Ghost Commander connected! Managing {self.dca_total_contracts} MGC contracts", "success")
            
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
        # Get active contract month (at least 30 days out)
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
            self.ib.reqMarketDataType(3)  # Delayed data for paper
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
            last = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            
            if last and last > 0:
                self.last_price = last
                return {
                    'bid': bid or last,
                    'ask': ask or last,
                    'last': last
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def sync_positions(self):
        """Sync positions from IB"""
        if not self.connected:
            return
        
        try:
            self.positions = []
            
            for pos in self.ib.positions():
                if pos.contract.symbol == 'MGC' and pos.position != 0:
                    self.positions.append({
                        'contracts': pos.position,
                        'avg_cost': pos.avgCost,
                        'entry_price': pos.avgCost / 10,  # avgCost includes multiplier
                    })
            
            # Calculate totals
            self.dca_total_contracts = sum(p['contracts'] for p in self.positions)
            
            if self.dca_total_contracts > 0:
                total_value = sum(p['entry_price'] * p['contracts'] for p in self.positions)
                self.dca_average_entry = total_value / self.dca_total_contracts
            else:
                self.dca_average_entry = 0
            
            logger.debug(f"Synced: {self.dca_total_contracts} contracts @ ${self.dca_average_entry:.2f}")
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    # =========================================================================
    # TAKE PROFIT LOGIC
    # =========================================================================
    
    def check_take_profits(self, current_price: float) -> dict:
        """
        Check tiered take profit levels
        
        TP1: +30 pips ($3) = close 30%
        TP2: +60 pips ($6) = close 30%
        TP3: +90 pips ($9) = close ALL
        """
        if self.dca_total_contracts == 0 or self.dca_average_entry == 0:
            return None
        
        # Calculate profit in pips (1 pip = $0.10)
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
            contracts_to_close = max(1, int(self.dca_total_contracts * close_pct))
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP2',
                'profit_pips': profit_pips,
                'contracts': contracts_to_close
            }
        
        # TP1: Close 30% at +30 pips
        if not self.tp1_hit and profit_pips >= self.settings['tp1_pips']:
            close_pct = self.settings['tp1_close_percent'] / 100
            contracts_to_close = max(1, int(self.dca_total_contracts * close_pct))
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP1',
                'profit_pips': profit_pips,
                'contracts': contracts_to_close
            }
        
        return None
    
    def execute_take_profit(self, tp_info: dict, current_price: float) -> bool:
        """Execute take profit order"""
        try:
            contracts = int(tp_info['contracts'])
            
            order = MarketOrder('SELL', contracts)
            order.orderRef = f"GHOST|{tp_info['level']}"
            
            logger.info(f"📤 Executing {tp_info['level']}: SELL {contracts} @ MKT")
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(3)
            
            # Calculate realized profit
            profit = (current_price - self.dca_average_entry) * contracts * 10
            self.daily_realized_pnl += profit
            
            # Update TP flags
            if tp_info['level'] == 'TP1':
                self.tp1_hit = True
            elif tp_info['level'] == 'TP2':
                self.tp2_hit = True
            elif tp_info['level'] == 'TP3':
                # Full close - reset
                self.tp1_hit = False
                self.tp2_hit = False
            
            self._save_state()
            
            # Log and alert
            msg = f"🎯 {tp_info['level']} HIT! Closed {contracts} contracts @ ${current_price:.2f} = +${profit:.2f}"
            logger.info(msg)
            send_alert(msg, "success")
            
            # Check for runner deployment after TP1
            if tp_info['level'] == 'TP1' and self.check_runner_trigger():
                self.deploy_runner(current_price)
            
            return True
            
        except Exception as e:
            logger.error(f"TP execution failed: {e}")
            return False
    
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
            self.ib.sleep(3)
            
            self.runner_position = {
                'entry_price': current_price,
                'contracts': 1,
                'funded_by': self.daily_realized_pnl,
                'time': datetime.now().isoformat()
            }
            
            self._save_state()
            
            msg = f"🚀 FREE ROLL RUNNER deployed! Entry: ${current_price:.2f}, Funded by: ${self.daily_realized_pnl:.2f} profits"
            logger.info(msg)
            send_alert(msg, "success")
            
        except Exception as e:
            logger.error(f"Runner deployment failed: {e}")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run_once(self):
        """Run one iteration of the bot"""
        if not self.connected:
            return
        
        try:
            # Sync positions
            self.sync_positions()
            
            # Get current price
            price_data = self.get_price()
            if not price_data:
                return
            
            current_price = price_data['last']
            
            # Check take profits
            tp_action = self.check_take_profits(current_price)
            if tp_action:
                self.execute_take_profit(tp_action, current_price)
                # Re-sync after TP
                self.sync_positions()
            
            # Log status periodically (every 60 seconds)
            if (datetime.now() - self.last_status_log).seconds >= 60:
                self._log_status(current_price)
                self.last_status_log = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
    
    def _log_status(self, current_price: float):
        """Log current status"""
        if self.dca_total_contracts > 0:
            unrealized = (current_price - self.dca_average_entry) * self.dca_total_contracts * 10
            tp1_price = self.dca_average_entry + 3
            tp2_price = self.dca_average_entry + 6
            tp3_price = self.dca_average_entry + 9
            
            logger.info(
                f"📊 MGC: ${current_price:.2f} | "
                f"{self.dca_total_contracts} contracts @ ${self.dca_average_entry:.2f} | "
                f"P&L: ${unrealized:.2f} | "
                f"TPs: ${tp1_price:.2f}{'✓' if self.tp1_hit else ''} / "
                f"${tp2_price:.2f}{'✓' if self.tp2_hit else ''} / "
                f"${tp3_price:.2f}"
            )
        else:
            logger.info(f"📊 MGC: ${current_price:.2f} | No positions")
    
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
                    logger.info(f"Loaded state: TP1={self.tp1_hit}, TP2={self.tp2_hit}")
            except:
                pass
    
    def get_status(self) -> dict:
        """Get current status for API/dashboard"""
        return {
            'connected': self.connected,
            'contracts': self.dca_total_contracts,
            'avg_entry': self.dca_average_entry,
            'current_price': self.last_price,
            'unrealized_pnl': (self.last_price - self.dca_average_entry) * self.dca_total_contracts * 10 if self.dca_total_contracts > 0 else 0,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'daily_realized': self.daily_realized_pnl,
            'runner_active': self.runner_position is not None,
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
                'content': f"{emoji} **Ghost Commander** [{level.upper()}]\n{message}",
                'username': 'Ghost Commander'
            }, timeout=5)
        except Exception as e:
            logger.error(f"Webhook failed: {e}")


def check_market_hours() -> bool:
    """Check if gold futures market is open"""
    now = datetime.utcnow()
    weekday = now.weekday()
    
    # Closed Saturday and most of Sunday (until ~11pm UTC)
    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and now.hour < 22:  # Sunday before ~6pm ET
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    logger.info("=" * 70)
    logger.info("🚀 GHOST COMMANDER LIVE - Starting")
    logger.info(f"   IB: {IB_HOST}:{IB_PORT}")
    logger.info(f"   Client ID: {CLIENT_ID}")
    logger.info("=" * 70)
    
    send_alert("Ghost Commander starting on H100", "info")
    
    ghost = GhostCommanderLive()
    reconnect_delay = 30
    max_reconnect_delay = 300
    
    while True:
        try:
            # Check market hours
            if not check_market_hours():
                logger.info("Market closed. Sleeping 1 hour...")
                time.sleep(3600)
                continue
            
            # Connect if needed
            if not ghost.connected:
                if ghost.connect():
                    reconnect_delay = 30
                else:
                    raise Exception("Connection failed")
            
            # Run one iteration
            ghost.run_once()
            
            # Sleep between checks
            time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            send_alert("Ghost Commander shutting down", "warning")
            break
            
        except Exception as e:
            logger.error(f"Error: {e}")
            
            # Disconnect and prepare for reconnect
            try:
                ghost.disconnect()
            except:
                pass
            ghost.connected = False
            
            # Exponential backoff
            logger.info(f"Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    # Cleanup
    ghost.disconnect()
    logger.info("Ghost Commander stopped.")


if __name__ == '__main__':
    main()

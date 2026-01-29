"""
Ghost Commander for IB Gold Futures
Ported from Crellastein_v0201.mq5

Same logic as MT5:
- Regime detection (TRENDING vs QUIET)
- DCA ladder on drops (DROPBUYL1-L5)
- Free Roll Runner funded by profits
- Position management with partial TPs

Contract: MGC (Micro Gold Futures)
- 10 oz per contract
- ~$1,000 margin per contract
- Tick value: $1 per 0.10 move

Author: QUINN001
Created: 2026-01-28
"""

from gold_futures_connector import GoldFuturesConnector
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostCommander")

# State file
STATE_DIR = Path(__file__).parent / "ghost_data"
STATE_DIR.mkdir(exist_ok=True)


class GhostCommanderIB:
    """
    Ghost Commander Gold Strategy for Interactive Brokers
    
    Same logic as MT5 Crellastein:
    - DCA on drops (DROPBUYL1-L5)
    - Partial take profits
    - Free Roll Runner funded by profits
    - Regime-aware trading
    """
    
    def __init__(self, paper_trading: bool = True, client_id: int = 51):
        self.connector = GoldFuturesConnector(
            paper_trading=paper_trading, 
            client_id=client_id
        )
        self.name = "GHOST"
        
        # ═══════════════════════════════════════════════════════════════
        # GHOST COMMANDER SETTINGS (Same as MT5)
        # ═══════════════════════════════════════════════════════════════
        
        self.settings = {
            # DCA Settings (DROPBUY) - 1.1.2.2.4.4 Ladder
            'dca_enabled': True,
            'dca_levels': 6,
            'dca_drop_pips': [0, 30, 60, 100, 150, 200],   # Pips drop triggers ($3, $6, $10, $15, $20)
            'dca_contracts': [1, 1, 2, 2, 4, 4],           # 1.1.2.2.4.4 scaling (14 total)
            
            # Position Limits (1+1+2+2+4+4 = 14 max)
            'max_contracts': 14,
            'max_daily_trades': 10,
            
            # Direction (supertrend bullish = LONG ONLY)
            'direction': 'LONG',  # LONG only - no shorts
            
            # Take Profit (in pips, 1 pip = $0.10)
            'tp_pips': 50,                  # $5 take profit
            'partial_tp_enabled': True,
            'partial_tp_percent': 50,       # Close 50% at first TP
            
            # Free Roll Runner
            'runner_enabled': True,
            'runner_profit_threshold': 300,  # $300 min profit to open runner
            'runner_contracts': 1,
            'runner_tp_pips': 300,           # $30 runner target (300%)
            
            # Regime Detection
            'regime_atr_period': 14,
            'regime_atr_threshold': 20,      # Pips - above = TRENDING
            
            # Risk Management
            'stop_loss_pips': 200,           # $20 stop (200 pips)
            'trailing_stop_enabled': True,
            'trailing_stop_pips': 30,        # $3 trail
            
            # Trading Hours (UTC)
            'trade_start_hour': 0,
            'trade_end_hour': 23,
        }
        
        # ═══════════════════════════════════════════════════════════════
        # STATE
        # ═══════════════════════════════════════════════════════════════
        
        self.positions = []           # DCA positions
        self.runner_position = None   # Free roll runner
        self.current_regime = "QUIET"
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_price = 0
        self.entry_price = None       # First entry price (for DCA reference)
        
        # Load saved state
        self.state_file = STATE_DIR / "ghost_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load saved state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.positions = state.get('positions', [])
                    self.runner_position = state.get('runner_position')
                    self.daily_pnl = state.get('daily_pnl', 0)
                    self.entry_price = state.get('entry_price')
                    logger.info(f"Loaded state: {len(self.positions)} positions")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'positions': self.positions,
            'runner_position': self.runner_position,
            'daily_pnl': self.daily_pnl,
            'entry_price': self.entry_price,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def connect(self) -> bool:
        """Connect to TWS"""
        result = self.connector.connect()
        if result:
            logger.info(f"🔮 {self.name} COMMANDER connected to IB!")
        return result
    
    def disconnect(self):
        """Disconnect and save state"""
        self._save_state()
        self.connector.disconnect()
    
    # ═══════════════════════════════════════════════════════════════════
    # REGIME DETECTION (Same as MT5)
    # ═══════════════════════════════════════════════════════════════════
    
    def detect_regime(self) -> str:
        """
        Detect market regime: TRENDING or QUIET
        Based on ATR volatility
        """
        candles = self.connector.get_historical_candles(
            duration="2 D", 
            bar_size="1 hour"
        )
        
        if len(candles) < self.settings['regime_atr_period']:
            return self.current_regime
        
        # Calculate ATR in pips (1 pip = $0.10)
        atr_sum = 0
        period = self.settings['regime_atr_period']
        
        for i in range(-period, 0):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close'] if i > -period else candles[i]['open']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            atr_sum += tr
        
        atr = atr_sum / period
        atr_pips = atr * 10  # Convert to pips ($0.10 per pip)
        
        new_regime = "TRENDING" if atr_pips > self.settings['regime_atr_threshold'] else "QUIET"
        
        if new_regime != self.current_regime:
            self._on_regime_change(self.current_regime, new_regime)
            self.current_regime = new_regime
        
        return self.current_regime
    
    def _on_regime_change(self, old: str, new: str):
        """Handle regime change"""
        logger.info(f"📊 {self.name}: Regime changed {old} → {new}")
        
        if new == "TRENDING":
            # More aggressive in trends
            self.settings['tp_pips'] = 70
            self.settings['dca_drop_pips'] = [0, 40, 80, 120, 160]
        else:
            # Conservative in quiet
            self.settings['tp_pips'] = 50
            self.settings['dca_drop_pips'] = [0, 50, 100, 150, 200]
    
    # ═══════════════════════════════════════════════════════════════════
    # DCA LOGIC (DROPBUY - Same as MT5)
    # ═══════════════════════════════════════════════════════════════════
    
    def check_dca_trigger(self, current_price: float) -> Optional[Dict]:
        """
        Check if we should add DCA position
        
        Returns: {level, contracts, type} or None
        """
        if not self.settings['dca_enabled']:
            return None
        
        current_level = len(self.positions)
        
        # No positions - first entry
        if current_level == 0:
            return {
                'level': 1,
                'contracts': self.settings['dca_contracts'][0],
                'type': 'DROPBUYL1'
            }
        
        # Max DCA reached
        if current_level >= self.settings['dca_levels']:
            return None
        
        # Max contracts reached
        total_contracts = sum(p.get('contracts', 0) for p in self.positions)
        if total_contracts >= self.settings['max_contracts']:
            return None
        
        # Calculate drop from entry in pips
        if self.entry_price is None:
            return None
        
        drop_amount = self.entry_price - current_price
        drop_pips = drop_amount * 10  # $0.10 per pip
        
        # Check next DCA level
        next_level = current_level + 1
        trigger_pips = self.settings['dca_drop_pips'][current_level]
        
        if drop_pips >= trigger_pips:
            contracts = self.settings['dca_contracts'][current_level]
            
            # Check max contracts
            if total_contracts + contracts > self.settings['max_contracts']:
                contracts = self.settings['max_contracts'] - total_contracts
            
            if contracts > 0:
                return {
                    'level': next_level,
                    'contracts': contracts,
                    'type': f'DROPBUYL{next_level}'
                }
        
        return None
    
    def execute_dca(self, dca_info: Dict, price: float) -> bool:
        """Execute DCA buy"""
        if self.daily_trades >= self.settings['max_daily_trades']:
            logger.warning(f"⚠️ Max daily trades ({self.settings['max_daily_trades']}) reached")
            return False
        
        # Place order
        result = self.connector.buy_gold(
            quantity=dca_info['contracts'],
            limit_price=None,  # Market order
            order_ref=dca_info['type']
        )
        
        if result.get('success'):
            # Track position
            position = {
                'order_id': result['order_id'],
                'contracts': dca_info['contracts'],
                'price': price,
                'type': dca_info['type'],
                'level': dca_info['level'],
                'time': datetime.now().isoformat()
            }
            self.positions.append(position)
            
            # Set entry price on first position
            if dca_info['level'] == 1:
                self.entry_price = price
            
            self.daily_trades += 1
            self._save_state()
            
            logger.info(f"📈 {dca_info['type']}: Bought {dca_info['contracts']}x MGC @ ${price:.2f}")
            return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════════
    # TAKE PROFIT (Same as MT5)
    # ═══════════════════════════════════════════════════════════════════
    
    def check_take_profit(self, current_price: float) -> bool:
        """Check and execute take profits"""
        if not self.positions:
            return False
        
        # Calculate average entry
        total_contracts = sum(p['contracts'] for p in self.positions)
        avg_entry = sum(p['price'] * p['contracts'] for p in self.positions) / total_contracts
        
        # Calculate gain in pips
        gain_amount = current_price - avg_entry
        gain_pips = gain_amount * 10
        
        if gain_pips >= self.settings['tp_pips']:
            if self.settings['partial_tp_enabled'] and len(self.positions) > 1:
                # Partial TP - close 50%
                close_contracts = max(1, int(total_contracts * self.settings['partial_tp_percent'] / 100))
                
                result = self.connector.sell_gold(
                    quantity=close_contracts,
                    order_ref='PARTIAL_TP'
                )
                
                if result.get('success'):
                    profit = gain_amount * close_contracts * 10  # 10 oz per contract
                    self.daily_pnl += profit
                    
                    # Remove closed positions (FIFO)
                    remaining = close_contracts
                    while remaining > 0 and self.positions:
                        pos = self.positions[0]
                        if pos['contracts'] <= remaining:
                            remaining -= pos['contracts']
                            self.positions.pop(0)
                        else:
                            pos['contracts'] -= remaining
                            remaining = 0
                    
                    self._save_state()
                    logger.info(f"💰 PARTIAL TP: Closed {close_contracts}x @ ${current_price:.2f} = +${profit:.2f}")
                    
                    # Check for runner trigger
                    if self.check_runner_trigger():
                        self.open_runner(current_price)
                    
                    return True
            else:
                # Full TP - close all
                result = self.connector.sell_gold(
                    quantity=total_contracts,
                    order_ref='FULL_TP'
                )
                
                if result.get('success'):
                    profit = gain_amount * total_contracts * 10
                    self.daily_pnl += profit
                    
                    self.positions.clear()
                    self.entry_price = None
                    self._save_state()
                    
                    logger.info(f"💰 FULL TP: Closed all {total_contracts}x @ ${current_price:.2f} = +${profit:.2f}")
                    
                    # Check for runner
                    if self.check_runner_trigger():
                        self.open_runner(current_price)
                    
                    return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════════
    # FREE ROLL RUNNER (Same as MT5)
    # ═══════════════════════════════════════════════════════════════════
    
    def check_runner_trigger(self) -> bool:
        """Check if we should open Free Roll Runner"""
        if not self.settings['runner_enabled']:
            return False
        
        if self.runner_position is not None:
            return False  # Already have runner
        
        if self.daily_pnl < self.settings['runner_profit_threshold']:
            return False  # Not enough profit
        
        return True
    
    def open_runner(self, price: float):
        """Open Free Roll Runner - funded by profits, NO STOP LOSS"""
        result = self.connector.buy_gold(
            quantity=self.settings['runner_contracts'],
            order_ref='FREEROLLRUNNER'
        )
        
        if result.get('success'):
            self.runner_position = {
                'order_id': result['order_id'],
                'contracts': self.settings['runner_contracts'],
                'entry_price': price,
                'time': datetime.now().isoformat()
            }
            self._save_state()
            
            logger.info(f"🚀 FREE ROLL RUNNER OPENED!")
            logger.info(f"   {self.settings['runner_contracts']}x MGC @ ${price:.2f}")
            logger.info(f"   Funded by: ${self.daily_pnl:.2f} profits")
            logger.info(f"   Target: +${self.settings['runner_tp_pips'] / 10:.0f} (no stop loss!)")
    
    def manage_runner(self, current_price: float):
        """Manage Free Roll Runner - NO STOP LOSS, let profits run"""
        if not self.runner_position:
            return
        
        entry = self.runner_position['entry_price']
        gain_amount = current_price - entry
        gain_pips = gain_amount * 10
        
        # Take profit at target
        if gain_pips >= self.settings['runner_tp_pips']:
            result = self.connector.sell_gold(
                quantity=self.runner_position['contracts'],
                order_ref='RUNNER_TP'
            )
            
            if result.get('success'):
                profit = gain_amount * self.runner_position['contracts'] * 10
                logger.info(f"🎯 RUNNER HIT TARGET! +${profit:.2f}")
                self.runner_position = None
                self._save_state()
                return
        
        # Log status (no stop loss - funded by profits)
        gain_dollars = gain_amount * self.runner_position['contracts'] * 10
        status = "+" if gain_dollars >= 0 else ""
        logger.debug(f"🏃 Runner: {status}${gain_dollars:.2f} (target: +${self.settings['runner_tp_pips']/10:.0f})")
    
    # ═══════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════
    
    def run_once(self) -> Dict:
        """Run one iteration of the strategy"""
        if not self.connector.is_connected():
            return {'error': 'Not connected'}
        
        # Get current price
        price_data = self.connector.get_gold_price()
        current_price = price_data.get('last', 0)
        
        if current_price == 0:
            return {'status': 'waiting_for_price'}
        
        self.last_price = current_price
        
        # Detect regime
        regime = self.detect_regime()
        
        # Check DCA trigger
        dca = self.check_dca_trigger(current_price)
        if dca:
            self.execute_dca(dca, current_price)
        
        # Check take profit
        self.check_take_profit(current_price)
        
        # Manage runner
        self.manage_runner(current_price)
        
        # Status
        total_contracts = sum(p.get('contracts', 0) for p in self.positions)
        avg_entry = sum(p['price'] * p['contracts'] for p in self.positions) / total_contracts if total_contracts > 0 else 0
        unrealized_pnl = (current_price - avg_entry) * total_contracts * 10 if total_contracts > 0 else 0
        
        return {
            'price': current_price,
            'regime': regime,
            'positions': total_contracts,
            'avg_entry': avg_entry,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'runner_active': self.runner_position is not None,
            'dca_level': len(self.positions)
        }
    
    def run(self, check_interval: int = 30):
        """Main trading loop"""
        print("=" * 70)
        print(f"🔮 {self.name} COMMANDER IB - Gold Futures (MGC)")
        print("=" * 70)
        print(f"Contract: MGC (10 oz per contract)")
        print(f"DCA Levels: {self.settings['dca_levels']}")
        print(f"Max Contracts: {self.settings['max_contracts']}")
        print(f"TP: {self.settings['tp_pips']} pips (${self.settings['tp_pips']/10:.1f})")
        print(f"Runner: {'Enabled' if self.settings['runner_enabled'] else 'Disabled'}")
        print("=" * 70)
        
        while True:
            try:
                status = self.run_once()
                
                if 'error' in status:
                    logger.error(f"Error: {status['error']}")
                    time.sleep(10)
                    continue
                
                if 'price' in status:
                    runner_status = "🏃 RUNNER" if status['runner_active'] else ""
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"MGC: ${status['price']:.2f} | "
                          f"L{status['dca_level']}/{self.settings['dca_levels']} ({status['positions']} contracts) | "
                          f"P&L: ${status['unrealized_pnl']:.2f} | "
                          f"Daily: ${status['daily_pnl']:.2f} | "
                          f"{status['regime']} {runner_status}")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print(f"\n🔮 {self.name} COMMANDER shutting down...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)
    
    def reset_daily(self):
        """Reset daily counters (call at midnight)"""
        self.daily_pnl = 0
        self.daily_trades = 0
        logger.info(f"🔄 {self.name} daily counters reset")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ghost = GhostCommanderIB(paper_trading=True, client_id=51)
    
    if ghost.connect():
        try:
            ghost.run(check_interval=30)
        finally:
            ghost.disconnect()
    else:
        print("❌ Failed to connect")

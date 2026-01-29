"""
Ghost Commander IB - Ported from Crellastein_v020.mq5 + v0201_DCA_PATCH

Features:
- NEO signals + SuperTrend direction
- DCA ladder with pip-based spacing (30/60/90/120)
- Tiered take profits: TP1 +30p (30%), TP2 +60p (30%), TP3 +90p (all)
- Free Roll Runner funded by profits
- Regime detection (TRENDING vs QUIET)

Magic Number: 888020

Author: QUINN001
Ported: January 29, 2026
"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, LimitOrder, MarketOrder
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import json
import logging
from pathlib import Path

from .indicators import Indicators
from .config import GHOST_SETTINGS, IB_SETTINGS, MGC_SPECS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ghost")


class Regime(Enum):
    QUIET = "QUIET"
    TRENDING = "TRENDING"


class GhostCommanderIB:
    """
    Ghost Commander for IB Gold Futures (MGC)
    Ported from MT5 Crellastein_v020 + v0201_DCA_PATCH
    """
    
    def __init__(self, paper_trading: bool = True):
        self.ib = IB()
        self.paper_trading = paper_trading
        self.connected = False
        
        # Settings from config
        self.settings = GHOST_SETTINGS.copy()
        self.name = "GHOST"
        self.magic = self.settings['strategy_id']
        
        # State
        self.positions = []           # DCA ladder positions
        self.runner_position = None   # Free roll runner
        self.regime = Regime.QUIET
        
        # DCA tracking (from v0201 PATCH)
        self.dca_average_entry = 0
        self.dca_total_contracts = 0
        self.dca_highest_entry = 0
        self.dca_ladder_count = 0
        self.tp1_hit = False
        self.tp2_hit = False
        
        # Daily tracking
        self.daily_realized_pnl = 0
        self.daily_trades = 0
        self.dca_wins_today = 0
        
        # Contract
        self.contract = None
        self.contract_expiry = None
        self.last_price = 0
        
        # State persistence
        self.state_dir = Path(__file__).parent / "ghost_data"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "ghost_state.json"
        self._load_state()
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    def connect(self, host: str = None, port: int = None, 
                client_id: int = None) -> bool:
        """Connect to TWS"""
        try:
            if self.ib.isConnected():
                return True
            
            host = host or IB_SETTINGS['host']
            port = port or (IB_SETTINGS['paper_port'] if self.paper_trading 
                           else IB_SETTINGS['live_port'])
            client_id = client_id or IB_SETTINGS['client_id_ghost']
            
            mode = "PAPER" if self.paper_trading else "LIVE"
            logger.info(f"Connecting to IB {mode}...")
            
            self.ib.connect(host, port, clientId=client_id, timeout=30)
            self.ib.sleep(2)
            self.connected = True
            
            # Initialize contract
            self._init_contract()
            
            logger.info(f"🔮 {self.name} COMMANDER connected!")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect and save state"""
        self._save_state()
        if self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False
    
    def _init_contract(self):
        """Initialize MGC futures contract"""
        expiry = self._get_active_contract_month()
        self.contract = Future(
            symbol='MGC',
            lastTradeDateOrContractMonth=expiry,
            exchange='COMEX',
            currency='USD'
        )
        qualified = self.ib.qualifyContracts(self.contract)
        if qualified:
            self.contract = qualified[0]
            self.contract_expiry = expiry
            logger.info(f"Using contract: {self.contract.localSymbol}")
    
    def _get_active_contract_month(self) -> str:
        """Get actively traded contract month (avoids delivery)"""
        now = datetime.now()
        months = [2, 4, 6, 8, 10, 12]
        
        for month in months:
            year = now.year if month > now.month else now.year + 1
            expiry_approx = datetime(year, month, 25)
            if (expiry_approx - now).days >= 30:
                return f"{year}{month:02d}"
        
        return f"{now.year + 1}04"
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def get_price(self) -> float:
        """Get current MGC price"""
        if not self.connected:
            return self.last_price
        
        try:
            self.ib.reqMarketDataType(3)  # Delayed data
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            
            price = ticker.last or ticker.close or self.last_price
            if price and price > 0:
                self.last_price = price
            
            return self.last_price
        except:
            return self.last_price
    
    def get_candles(self, duration: str = "2 D", bar_size: str = "1 hour") -> List[Dict]:
        """Get historical candles"""
        if not self.connected:
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
            
            return [
                {'open': b.open, 'high': b.high, 'low': b.low, 
                 'close': b.close, 'volume': b.volume}
                for b in bars
            ]
        except:
            return []
    
    # =========================================================================
    # REGIME DETECTION (from MQ5)
    # =========================================================================
    
    def detect_regime(self) -> Regime:
        """
        Detect TRENDING vs QUIET regime
        Uses ADX + ATR like MT5 version
        """
        candles = self.get_candles("3 D", "1 hour")
        
        if len(candles) < 20:
            return self.regime
        
        # ADX for trend strength
        adx_value = Indicators.adx_single(candles, self.settings['adx_period'])
        
        # ATR for volatility
        atr = Indicators.atr_single(candles, self.settings['atr_period'])
        avg_price = candles[-1]['close']
        atr_percent = (atr / avg_price) * 100 if avg_price > 0 else 0
        
        # Determine regime
        if adx_value > self.settings['adx_trend_threshold'] or atr_percent > 0.5:
            new_regime = Regime.TRENDING
        else:
            new_regime = Regime.QUIET
        
        if new_regime != self.regime:
            logger.info(f"📊 Regime: {self.regime.value} → {new_regime.value}")
            self.regime = new_regime
        
        return self.regime
    
    # =========================================================================
    # DCA LADDER LOGIC (from v0201 PATCH)
    # =========================================================================
    
    def calculate_dca_ladder_stats(self):
        """
        Calculate DCA ladder statistics
        Port of MT5 CalculateDCALadderStats()
        """
        self.dca_average_entry = 0
        self.dca_total_contracts = 0
        self.dca_highest_entry = 0
        self.dca_ladder_count = 0
        
        if not self.positions:
            return
        
        total_value = 0
        for pos in self.positions:
            price = pos['entry_price']
            contracts = pos['contracts']
            
            total_value += price * contracts
            self.dca_total_contracts += contracts
            
            if price > self.dca_highest_entry:
                self.dca_highest_entry = price
            
            self.dca_ladder_count += 1
        
        if self.dca_total_contracts > 0:
            self.dca_average_entry = total_value / self.dca_total_contracts
    
    def is_too_close_to_existing(self, price: float) -> bool:
        """
        Anti-stack check: ensure minimum gap between positions
        Port of MT5 IsTooCloseToExistingPosition()
        """
        min_gap = self.settings['dca_min_entry_gap']
        
        for pos in self.positions:
            gap = abs(price - pos['entry_price']) * 10  # Convert to points
            if gap < min_gap:
                return True
        
        return False
    
    def check_dca_initial_entry(self, current_price: float) -> Optional[Dict]:
        """
        Check if should open initial DCA position
        Requires SuperTrend bullish
        """
        if self.positions:
            return None  # Already have positions
        
        # Check SuperTrend direction
        candles = self.get_candles("2 D", "1 hour")
        if len(candles) < 15:
            return None
        
        st = Indicators.supertrend(
            candles, 
            self.settings['supertrend_period'],
            self.settings['supertrend_multiplier']
        )
        
        if st['trend'] != 1:  # Not bullish
            return None
        
        contracts = 1  # Initial position = 1 contract
        
        return {
            'level': 1,
            'contracts': contracts,
            'comment': f'GHOST|ENTRY|ST_BULL'
        }
    
    def check_dca_ladder(self, current_price: float) -> Optional[Dict]:
        """
        Check if should add DCA position
        Port of MT5 CheckDCALadder()
        
        Uses pip-based spacing: 30, 60, 90, 120 pips from highest entry
        """
        self.calculate_dca_ladder_stats()
        
        if self.dca_ladder_count == 0:
            return self.check_dca_initial_entry(current_price)
        
        if self.dca_ladder_count >= self.settings['dca_max_levels']:
            return None  # Max levels reached
        
        # Calculate drop from highest entry in pips
        drop_pips = (self.dca_highest_entry - current_price) * 10
        
        # Required drop for next level
        required_drop = self.settings['dca_pip_spacing'] * self.dca_ladder_count
        
        if drop_pips < required_drop:
            return None  # Not deep enough
        
        # Anti-stack check
        if self.is_too_close_to_existing(current_price):
            return None
        
        # Calculate lot size (with increment)
        base_lots = self.settings['dca_initial_lots']
        increment = self.settings['dca_lot_increment']
        max_lots = self.settings['dca_max_lot_size']
        
        lots = base_lots + (increment * self.dca_ladder_count)
        lots = min(lots, max_lots)
        
        # Convert lots to contracts (1 lot = 10 MGC)
        contracts = max(1, int(lots * 10 / 10))  # Simplified: 1 lot = 1 contract
        
        return {
            'level': self.dca_ladder_count + 1,
            'contracts': contracts,
            'drop_pips': drop_pips,
            'comment': f'GHOST|DCA{self.dca_ladder_count}|{drop_pips:.0f}p'
        }
    
    def execute_dca_entry(self, dca_info: Dict, price: float) -> bool:
        """Execute DCA buy order"""
        try:
            order = MarketOrder('BUY', dca_info['contracts'])
            order.orderRef = dca_info['comment']
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            # Track position
            self.positions.append({
                'order_id': trade.order.orderId,
                'contracts': dca_info['contracts'],
                'entry_price': price,
                'level': dca_info['level'],
                'comment': dca_info['comment'],
                'time': datetime.now().isoformat()
            })
            
            self.daily_trades += 1
            self._save_state()
            
            logger.info(f"📈 DCA L{dca_info['level']}: +{dca_info['contracts']}x @ ${price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"DCA entry failed: {e}")
            return False
    
    # =========================================================================
    # TIERED TAKE PROFIT (from v0201 PATCH)
    # =========================================================================
    
    def check_dca_take_profits(self, current_price: float) -> Optional[Dict]:
        """
        Check tiered take profit levels
        Port of MT5 CheckDCATakeProfits()
        
        TP1: +30 pips = close 30%
        TP2: +60 pips = close 30%
        TP3: +90 pips = close ALL
        """
        self.calculate_dca_ladder_stats()
        
        if self.dca_ladder_count == 0 or self.dca_average_entry == 0:
            return None
        
        # Profit in pips from average entry
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
            close_percent = self.settings['tp2_close_percent'] / 100
            contracts_to_close = max(1, int(self.dca_total_contracts * close_percent))
            self.tp2_hit = True
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP2',
                'profit_pips': profit_pips,
                'contracts': contracts_to_close
            }
        
        # TP1: Close 30% at +30 pips
        if not self.tp1_hit and profit_pips >= self.settings['tp1_pips']:
            close_percent = self.settings['tp1_close_percent'] / 100
            contracts_to_close = max(1, int(self.dca_total_contracts * close_percent))
            self.tp1_hit = True
            
            # Check if should open runner
            if self.check_runner_trigger():
                return {
                    'action': 'PARTIAL_CLOSE_AND_RUNNER',
                    'level': 'TP1',
                    'profit_pips': profit_pips,
                    'contracts': contracts_to_close
                }
            
            return {
                'action': 'PARTIAL_CLOSE',
                'level': 'TP1',
                'profit_pips': profit_pips,
                'contracts': contracts_to_close
            }
        
        return None
    
    def execute_take_profit(self, tp_info: Dict, current_price: float) -> bool:
        """Execute take profit"""
        try:
            contracts = tp_info['contracts']
            
            order = MarketOrder('SELL', contracts)
            order.orderRef = f"GHOST|{tp_info['level']}"
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            # Calculate realized profit
            profit = (current_price - self.dca_average_entry) * contracts * 10
            self.daily_realized_pnl += profit
            
            # Update positions (FIFO)
            remaining = contracts
            while remaining > 0 and self.positions:
                pos = self.positions[0]
                if pos['contracts'] <= remaining:
                    remaining -= pos['contracts']
                    self.positions.pop(0)
                else:
                    pos['contracts'] -= remaining
                    remaining = 0
            
            # Check for full close
            if tp_info['action'] == 'CLOSE_ALL':
                self.positions.clear()
                self.tp1_hit = False
                self.tp2_hit = False
                self.dca_wins_today += 1
            
            self._save_state()
            
            logger.info(f"💰 {tp_info['level']}: -{contracts}x @ ${current_price:.2f} = +${profit:.2f}")
            
            # Open runner if triggered
            if tp_info['action'] == 'PARTIAL_CLOSE_AND_RUNNER':
                self.open_runner(current_price)
            
            return True
            
        except Exception as e:
            logger.error(f"Take profit failed: {e}")
            return False
    
    # =========================================================================
    # FREE ROLL RUNNER
    # =========================================================================
    
    def check_runner_trigger(self) -> bool:
        """Check if should open Free Roll Runner"""
        if not self.settings['runner_enabled']:
            return False
        
        if self.runner_position is not None:
            return False  # Already have runner
        
        if self.daily_realized_pnl < self.settings['runner_profit_threshold']:
            return False  # Not enough profit
        
        if self.dca_wins_today < 2:
            return False  # Need 2+ DCA wins first
        
        return True
    
    def open_runner(self, current_price: float):
        """Open Free Roll Runner - funded by profits, NO STOP LOSS"""
        try:
            budget = self.daily_realized_pnl * (self.settings['runner_budget_percent'] / 100)
            contracts = 1  # Runner = 1 contract
            
            order = MarketOrder('BUY', contracts)
            order.orderRef = 'GHOST|RUNNER|PROFIT_FUNDED'
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            self.runner_position = {
                'order_id': trade.order.orderId,
                'contracts': contracts,
                'entry_price': current_price,
                'funded_by': budget,
                'time': datetime.now().isoformat()
            }
            
            self._save_state()
            
            logger.info(f"🚀 FREE ROLL RUNNER OPENED!")
            logger.info(f"   Entry: ${current_price:.2f}")
            logger.info(f"   Funded by: ${budget:.2f} profits")
            logger.info(f"   Target: +{self.settings['runner_tp_percent']}%")
            logger.info(f"   Stop Loss: NONE (house money!)")
            
        except Exception as e:
            logger.error(f"Runner open failed: {e}")
    
    def manage_runner(self, current_price: float):
        """Manage Free Roll Runner - NO STOP LOSS"""
        if not self.runner_position:
            return
        
        entry = self.runner_position['entry_price']
        gain_pips = (current_price - entry) * 10
        gain_percent = (gain_pips / entry) * 1000 if entry > 0 else 0
        
        # Take profit at target
        if gain_percent >= self.settings['runner_tp_percent']:
            self._close_runner(current_price, 'TP_HIT')
            return
        
        # Check trend exit (death cross, below EMA50)
        candles = self.get_candles("2 D", "1 hour")
        if len(candles) >= 50:
            closes = [c['close'] for c in candles]
            ema20 = Indicators.ema(closes, 20)
            ema50 = Indicators.ema(closes, 50)
            
            if Indicators.is_death_cross(ema20, ema50):
                self._close_runner(current_price, 'DEATH_CROSS')
                return
    
    def _close_runner(self, price: float, reason: str):
        """Close runner position"""
        try:
            order = MarketOrder('SELL', self.runner_position['contracts'])
            order.orderRef = f'GHOST|RUNNER|{reason}'
            
            self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            profit = (price - self.runner_position['entry_price']) * self.runner_position['contracts'] * 10
            
            logger.info(f"🎯 RUNNER CLOSED: {reason} = ${profit:.2f}")
            self.runner_position = None
            self._save_state()
            
        except Exception as e:
            logger.error(f"Runner close failed: {e}")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run_once(self) -> Dict:
        """Run one iteration"""
        if not self.connected:
            return {'error': 'Not connected'}
        
        current_price = self.get_price()
        if current_price == 0:
            return {'status': 'waiting_for_price'}
        
        # Detect regime
        regime = self.detect_regime()
        
        # Check DCA entry
        dca = self.check_dca_ladder(current_price)
        if dca:
            self.execute_dca_entry(dca, current_price)
        
        # Check take profits
        tp = self.check_dca_take_profits(current_price)
        if tp:
            self.execute_take_profit(tp, current_price)
        
        # Manage runner
        self.manage_runner(current_price)
        
        # Return status
        self.calculate_dca_ladder_stats()
        unrealized_pnl = (current_price - self.dca_average_entry) * self.dca_total_contracts * 10 if self.dca_total_contracts > 0 else 0
        
        return {
            'price': current_price,
            'regime': regime.value,
            'dca_level': self.dca_ladder_count,
            'contracts': self.dca_total_contracts,
            'avg_entry': self.dca_average_entry,
            'unrealized_pnl': unrealized_pnl,
            'daily_realized': self.daily_realized_pnl,
            'runner_active': self.runner_position is not None,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit
        }
    
    def run(self, interval: int = 30):
        """Main trading loop"""
        print("=" * 70)
        print(f"🔮 {self.name} COMMANDER IB - MGC Gold Futures")
        print("=" * 70)
        print(f"DCA Spacing: {self.settings['dca_pip_spacing']} pips")
        print(f"Max Levels: {self.settings['dca_max_levels']}")
        print(f"TPs: +{self.settings['tp1_pips']}/{self.settings['tp2_pips']}/{self.settings['tp3_pips']} pips")
        print(f"Runner: {'Enabled' if self.settings['runner_enabled'] else 'Disabled'}")
        print("=" * 70)
        
        while True:
            try:
                status = self.run_once()
                
                if 'price' in status:
                    runner = "🏃" if status['runner_active'] else ""
                    tp_status = f"TP1:{'✓' if status['tp1_hit'] else '-'} TP2:{'✓' if status['tp2_hit'] else '-'}"
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"${status['price']:.2f} | "
                          f"L{status['dca_level']} ({status['contracts']}x) | "
                          f"Avg: ${status['avg_entry']:.2f} | "
                          f"P&L: ${status['unrealized_pnl']:.2f} | "
                          f"{status['regime']} | {tp_status} {runner}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print(f"\n🔮 {self.name} shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(10)
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'positions': self.positions,
            'runner_position': self.runner_position,
            'daily_realized_pnl': self.daily_realized_pnl,
            'dca_wins_today': self.dca_wins_today,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.positions = state.get('positions', [])
                    self.runner_position = state.get('runner_position')
                    self.daily_realized_pnl = state.get('daily_realized_pnl', 0)
                    self.dca_wins_today = state.get('dca_wins_today', 0)
                    self.tp1_hit = state.get('tp1_hit', False)
                    self.tp2_hit = state.get('tp2_hit', False)
                    logger.info(f"Loaded state: {len(self.positions)} positions")
            except:
                pass
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_realized_pnl = 0
        self.daily_trades = 0
        self.dca_wins_today = 0
        self._save_state()
        logger.info(f"🔄 Daily counters reset")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ghost = GhostCommanderIB(paper_trading=True)
    
    if ghost.connect():
        try:
            ghost.run(interval=30)
        finally:
            ghost.disconnect()
    else:
        print("❌ Failed to connect")

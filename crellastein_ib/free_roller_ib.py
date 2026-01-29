"""
Free Roller IB - Ported from Crellastein_v043.mq5

State Machine Strategy:
1. WAITING - Accumulate profits from other strategies
2. READY - Profit threshold met, waiting for entry signal
3. DEPLOYED - Position open, risking profits
4. FREE_ROLL - TP1 hit, now running risk-free!
5. COMPLETED - Position closed, cycle complete

Key Features:
- Uses accumulated profits for entries (house money)
- TP1 takes partial profit, moves to breakeven
- Remaining position runs with NO STOP LOSS
- Exit on death cross or below EMA50

Magic Number: 888043

Author: QUINN001
Ported: January 29, 2026
"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder, LimitOrder
from datetime import datetime
from typing import Dict, Optional
from enum import Enum
import time
import json
import logging
from pathlib import Path

from .indicators import Indicators
from .config import FREE_ROLLER_SETTINGS, IB_SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FreeRoller")


class RollState(Enum):
    """State machine states - matches MT5 ROLL_STATE enum"""
    WAITING = "WAITING"         # Waiting for profit to accumulate
    READY = "READY"             # Ready to deploy (profit threshold met)
    DEPLOYED = "DEPLOYED"       # Position open, risking profit
    FREE_ROLL = "FREE_ROLL"     # TP1 hit, now free rolling!
    COMPLETED = "COMPLETED"     # Position closed


class FreeRollerIB:
    """
    Free Roller - Profit-Funded Runner System
    Ported from MT5 Crellastein_v043.mq5
    """
    
    def __init__(self, paper_trading: bool = True):
        self.ib = IB()
        self.paper_trading = paper_trading
        self.connected = False
        
        # Settings
        self.settings = FREE_ROLLER_SETTINGS.copy()
        self.name = "FREE_ROLLER"
        
        # State Machine
        self.state = RollState.WAITING
        
        # Profit Tracking
        self.profit_available = 0      # From external sources (Ghost/Casper)
        self.profit_to_risk = 0        # Amount we're risking
        
        # Position Tracking
        self.position = None
        self.entry_price = 0
        self.initial_sl = 0
        self.tp1_price = 0
        self.breakeven_price = 0
        self.peak_profit = 0
        self.tp1_hit = False
        self.realized_from_tp1 = 0
        
        # Contract
        self.contract = None
        self.last_price = 0
        
        # State persistence
        self.state_dir = Path(__file__).parent / "roller_data"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "roller_state.json"
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
            client_id = client_id or IB_SETTINGS['client_id_roller']
            
            logger.info(f"Connecting to IB...")
            self.ib.connect(host, port, clientId=client_id, timeout=30)
            self.ib.sleep(2)
            self.connected = True
            
            self._init_contract()
            logger.info(f"🎰 {self.name} connected!")
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
        """Initialize MGC contract"""
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
        self.ib.qualifyContracts(self.contract)
    
    def get_price(self) -> float:
        """Get current price"""
        if not self.connected:
            return self.last_price
        try:
            self.ib.reqMarketDataType(3)
            ticker = self.ib.reqMktData(self.contract, '', False, False)
            self.ib.sleep(1)
            price = ticker.last or ticker.close or self.last_price
            if price and price > 0:
                self.last_price = price
            return self.last_price
        except:
            return self.last_price
    
    def get_candles(self, duration: str = "3 D", bar_size: str = "1 hour") -> list:
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
    # PROFIT TRACKING
    # =========================================================================
    
    def set_profit_available(self, profit: float):
        """Set available profit from external sources"""
        self.profit_available = profit
        self._check_ready()
    
    def add_profit(self, profit: float):
        """Add profit from external trade"""
        self.profit_available += profit
        self._check_ready()
    
    def _check_ready(self):
        """Check if ready to deploy"""
        if self.state == RollState.WAITING:
            if self.profit_available >= self.settings['min_profit_to_roll']:
                self.state = RollState.READY
                logger.info(f"🎰 READY! Profit available: ${self.profit_available:.2f}")
    
    # =========================================================================
    # STATE MACHINE HANDLERS
    # =========================================================================
    
    def handle_waiting(self, current_price: float):
        """
        STATE: WAITING
        - Accumulate profits
        - Check if threshold met
        """
        # Just waiting for profits to accumulate
        # Profits are added via set_profit_available() or add_profit()
        pass
    
    def handle_ready(self, current_price: float):
        """
        STATE: READY
        - Check for valid entry signal
        - Deploy position
        """
        candles = self.get_candles()
        if len(candles) < 50:
            return
        
        closes = [c['close'] for c in candles]
        
        # Check SuperTrend bullish
        st = Indicators.supertrend(candles, 10, 3)
        if st['trend'] != 1:
            return  # Wait for bullish
        
        # Check EMA alignment (price > EMA20 > EMA50)
        ema20 = Indicators.ema_single(closes, 20)
        ema50 = Indicators.ema_single(closes, 50)
        
        if not (current_price > ema20 > ema50):
            return  # Wait for bullish alignment
        
        # Deploy!
        self._deploy_position(current_price)
    
    def handle_deployed(self, current_price: float):
        """
        STATE: DEPLOYED
        - Monitor position
        - Check TP1
        - Check stop loss
        """
        if not self.position:
            self.state = RollState.WAITING
            return
        
        profit_pips = (current_price - self.entry_price) * 10
        
        # Track peak profit
        if profit_pips > self.peak_profit:
            self.peak_profit = profit_pips
        
        # Check TP1 (partial profit)
        tp1_pips = self.settings['initial_sl_pips'] * self.settings['risk_reward_ratio']
        
        if profit_pips >= tp1_pips:
            self._hit_tp1(current_price)
            return
        
        # Check stop loss (max drawdown of risked profit)
        if profit_pips < -self.settings['initial_sl_pips']:
            self._hit_stop(current_price)
            return
    
    def handle_free_roll(self, current_price: float):
        """
        STATE: FREE_ROLL
        - NO STOP LOSS (house money!)
        - Trail with trend
        - Exit on death cross or below EMA50
        """
        if not self.position:
            self.state = RollState.COMPLETED
            return
        
        profit_pips = (current_price - self.entry_price) * 10
        
        # Track peak
        if profit_pips > self.peak_profit:
            self.peak_profit = profit_pips
        
        candles = self.get_candles()
        if len(candles) < 50:
            return
        
        closes = [c['close'] for c in candles]
        ema20 = Indicators.ema(closes, 20)
        ema50 = Indicators.ema(closes, 50)
        
        # Exit on death cross
        if self.settings['exit_death_cross']:
            if Indicators.is_death_cross(ema20, ema50):
                self._exit_free_roll(current_price, 'DEATH_CROSS')
                return
        
        # Exit below EMA50
        if self.settings['exit_below_ema50']:
            if current_price < ema50[-1]:
                self._exit_free_roll(current_price, 'BELOW_EMA50')
                return
        
        # Exit on max drawdown from peak (optional)
        drawdown_pct = ((self.peak_profit - profit_pips) / self.peak_profit * 100) if self.peak_profit > 0 else 0
        if drawdown_pct > self.settings['exit_max_drawdown']:
            self._exit_free_roll(current_price, 'MAX_DRAWDOWN')
            return
    
    def handle_completed(self):
        """
        STATE: COMPLETED
        - Reset for next cycle
        """
        self.position = None
        self.entry_price = 0
        self.tp1_hit = False
        self.peak_profit = 0
        self.profit_to_risk = 0
        
        self.state = RollState.WAITING
        self._save_state()
        
        logger.info(f"🔄 Cycle complete. Waiting for next opportunity...")
    
    # =========================================================================
    # TRADING OPERATIONS
    # =========================================================================
    
    def _deploy_position(self, current_price: float):
        """Deploy position using accumulated profits"""
        try:
            # Calculate position size based on risk
            risk_amount = min(
                self.profit_available,
                self.profit_available * (self.settings['max_risk_percent'] / 100)
            )
            
            # 1 contract for simplicity
            contracts = 1
            
            order = MarketOrder('BUY', contracts)
            order.orderRef = 'FREEROLLER|DEPLOY'
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            self.position = {
                'order_id': trade.order.orderId,
                'contracts': contracts,
                'entry_price': current_price
            }
            
            self.entry_price = current_price
            self.profit_to_risk = risk_amount
            self.initial_sl = current_price - (self.settings['initial_sl_pips'] / 10)
            self.tp1_price = current_price + (self.settings['initial_sl_pips'] * self.settings['risk_reward_ratio'] / 10)
            self.breakeven_price = current_price
            
            self.state = RollState.DEPLOYED
            self._save_state()
            
            logger.info(f"🚀 DEPLOYED!")
            logger.info(f"   Entry: ${current_price:.2f}")
            logger.info(f"   Risking: ${risk_amount:.2f} profits")
            logger.info(f"   SL: ${self.initial_sl:.2f}")
            logger.info(f"   TP1: ${self.tp1_price:.2f}")
            
        except Exception as e:
            logger.error(f"Deploy failed: {e}")
    
    def _hit_tp1(self, current_price: float):
        """Handle TP1 hit - partial close, move to free roll"""
        try:
            # Close partial (50%)
            close_contracts = max(1, self.position['contracts'] // 2)
            
            if close_contracts > 0:
                order = MarketOrder('SELL', close_contracts)
                order.orderRef = 'FREEROLLER|TP1'
                
                self.ib.placeOrder(self.contract, order)
                self.ib.sleep(2)
                
                profit = (current_price - self.entry_price) * close_contracts * 10
                self.realized_from_tp1 = profit
                
                # Update position
                self.position['contracts'] -= close_contracts
            
            self.tp1_hit = True
            
            # If position remaining, move to free roll
            if self.position['contracts'] > 0:
                self.state = RollState.FREE_ROLL
                logger.info(f"🎯 TP1 HIT! +${self.realized_from_tp1:.2f}")
                logger.info(f"   Now FREE ROLLING {self.position['contracts']} contracts!")
                logger.info(f"   NO STOP LOSS - riding house money!")
            else:
                self.state = RollState.COMPLETED
            
            self._save_state()
            
        except Exception as e:
            logger.error(f"TP1 handling failed: {e}")
    
    def _hit_stop(self, current_price: float):
        """Handle stop loss hit"""
        try:
            order = MarketOrder('SELL', self.position['contracts'])
            order.orderRef = 'FREEROLLER|STOP'
            
            self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            loss = (self.entry_price - current_price) * self.position['contracts'] * 10
            self.profit_available -= loss
            
            logger.info(f"❌ STOP HIT: -${loss:.2f}")
            
            self.state = RollState.COMPLETED
            self._save_state()
            
        except Exception as e:
            logger.error(f"Stop handling failed: {e}")
    
    def _exit_free_roll(self, current_price: float, reason: str):
        """Exit free roll position"""
        try:
            order = MarketOrder('SELL', self.position['contracts'])
            order.orderRef = f'FREEROLLER|{reason}'
            
            self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            profit = (current_price - self.entry_price) * self.position['contracts'] * 10
            total_profit = self.realized_from_tp1 + profit
            
            logger.info(f"🏁 FREE ROLL EXIT: {reason}")
            logger.info(f"   TP1 profit: ${self.realized_from_tp1:.2f}")
            logger.info(f"   Runner profit: ${profit:.2f}")
            logger.info(f"   TOTAL: ${total_profit:.2f}")
            
            self.profit_available += total_profit
            self.state = RollState.COMPLETED
            self._save_state()
            
        except Exception as e:
            logger.error(f"Exit failed: {e}")
    
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
        
        # State machine
        if self.state == RollState.WAITING:
            self.handle_waiting(current_price)
        elif self.state == RollState.READY:
            self.handle_ready(current_price)
        elif self.state == RollState.DEPLOYED:
            self.handle_deployed(current_price)
        elif self.state == RollState.FREE_ROLL:
            self.handle_free_roll(current_price)
        elif self.state == RollState.COMPLETED:
            self.handle_completed()
        
        # Status
        unrealized = 0
        if self.position and self.entry_price > 0:
            unrealized = (current_price - self.entry_price) * self.position['contracts'] * 10
        
        return {
            'price': current_price,
            'state': self.state.value,
            'profit_available': self.profit_available,
            'position_contracts': self.position['contracts'] if self.position else 0,
            'entry_price': self.entry_price,
            'unrealized_pnl': unrealized,
            'tp1_hit': self.tp1_hit,
            'peak_profit_pips': self.peak_profit
        }
    
    def run(self, interval: int = 30):
        """Main loop"""
        print("=" * 70)
        print(f"🎰 {self.name} IB - Profit-Funded Runner")
        print("=" * 70)
        print(f"Min profit to deploy: ${self.settings['min_profit_to_roll']}")
        print(f"Risk/Reward: 1:{self.settings['risk_reward_ratio']}")
        print(f"TP1 partial: {self.settings['tp1_percent']}%")
        print("=" * 70)
        
        while True:
            try:
                status = self.run_once()
                
                if 'price' in status:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"${status['price']:.2f} | "
                          f"State: {status['state']} | "
                          f"Profit Pool: ${status['profit_available']:.2f} | "
                          f"Position: {status['position_contracts']}x | "
                          f"P&L: ${status['unrealized_pnl']:.2f}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print(f"\n🎰 {self.name} shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(10)
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save state"""
        state = {
            'state': self.state.value,
            'position': self.position,
            'entry_price': self.entry_price,
            'profit_available': self.profit_available,
            'profit_to_risk': self.profit_to_risk,
            'tp1_hit': self.tp1_hit,
            'realized_from_tp1': self.realized_from_tp1,
            'peak_profit': self.peak_profit,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.state = RollState(state.get('state', 'WAITING'))
                    self.position = state.get('position')
                    self.entry_price = state.get('entry_price', 0)
                    self.profit_available = state.get('profit_available', 0)
                    self.profit_to_risk = state.get('profit_to_risk', 0)
                    self.tp1_hit = state.get('tp1_hit', False)
                    self.realized_from_tp1 = state.get('realized_from_tp1', 0)
                    self.peak_profit = state.get('peak_profit', 0)
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    roller = FreeRollerIB(paper_trading=True)
    
    # Simulate having some profits
    roller.set_profit_available(600)  # $600 from other strategies
    
    if roller.connect():
        try:
            roller.run(interval=30)
        finally:
            roller.disconnect()
    else:
        print("❌ Failed to connect")

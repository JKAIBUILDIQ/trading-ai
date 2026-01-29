"""
Casper IB - Ported from Crellastein_Casper.mq5

Features:
- Meta Bot signals (10 weighted indicators)
- Drop-Buy Martingale DCA ($10 price drops)
- Fixed lot ladder: 0.5, 0.5, 1.0, 1.0, 2.0
- Trailing TP: +$10 start, $8 trail
- More conservative than Ghost ("prove it first")

Magic Number: 8880202

Author: QUINN001
Ported: January 29, 2026
"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, MarketOrder
from datetime import datetime
from typing import Dict, List, Optional
import time
import json
import logging
from pathlib import Path

from .indicators import Indicators
from .config import CASPER_SETTINGS, IB_SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Casper")


class CasperIB:
    """
    Casper IB - Conservative Drop-Buy Martingale
    Ported from MT5 Crellastein_Casper.mq5
    """
    
    def __init__(self, paper_trading: bool = True):
        self.ib = IB()
        self.paper_trading = paper_trading
        self.connected = False
        
        # Settings
        self.settings = CASPER_SETTINGS.copy()
        self.name = "CASPER"
        self.magic = self.settings['strategy_id']
        
        # Lot ladder: 0.5, 0.5, 1.0, 1.0, 2.0 = 5 lots max
        self.lot_ladder = self.settings['dropbuy_lot_ladder']
        
        # State
        self.positions = []
        self.session_high = 0
        self.current_level = 0
        self.trailing_sl = 0
        
        # Tracking
        self.dca_average_entry = 0
        self.dca_total_contracts = 0
        
        # Contract
        self.contract = None
        self.last_price = 0
        
        # State persistence
        self.state_dir = Path(__file__).parent / "casper_data"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "casper_state.json"
        self._load_state()
    
    # =========================================================================
    # CONNECTION (Same as Ghost)
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
            client_id = client_id or IB_SETTINGS['client_id_casper']
            
            logger.info(f"Connecting to IB...")
            self.ib.connect(host, port, clientId=client_id, timeout=30)
            self.ib.sleep(2)
            self.connected = True
            
            self._init_contract()
            logger.info(f"👻 {self.name} connected!")
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
        months = [2, 4, 6, 8, 10, 12]
        for month in months:
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
    
    # =========================================================================
    # DROP-BUY DCA (Martingale on $10 drops)
    # =========================================================================
    
    def check_drop_buy_dca(self, current_price: float) -> Optional[Dict]:
        """
        Drop-Buy Martingale: Buy on every $10 drop from session high
        Port of MT5 CheckDropBuyDCA()
        """
        # Track session high
        if current_price > self.session_high:
            self.session_high = current_price
            return None
        
        # Already at max level?
        if self.current_level >= len(self.lot_ladder):
            return None
        
        # Calculate drop from session high
        drop = self.session_high - current_price
        
        # Trigger price for next level
        # Level 1 at $10 drop, Level 2 at $20, etc.
        trigger_drop = self.settings['dropbuy_trigger_pips'] * (self.current_level + 1)
        
        if drop >= trigger_drop:
            # Get lot size from ladder
            lots = self.lot_ladder[self.current_level]
            level = self.current_level + 1
            
            return {
                'level': level,
                'lots': lots,
                'contracts': max(1, int(lots)),  # 1 lot = 1 contract for simplicity
                'drop': drop,
                'comment': f'CASPER|DROP{level}|${drop:.0f}'
            }
        
        return None
    
    def execute_drop_buy(self, dca_info: Dict, price: float) -> bool:
        """Execute Drop-Buy entry"""
        try:
            contracts = dca_info['contracts']
            
            order = MarketOrder('BUY', contracts)
            order.orderRef = dca_info['comment']
            
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            # Track position
            self.positions.append({
                'order_id': trade.order.orderId,
                'contracts': contracts,
                'entry_price': price,
                'level': dca_info['level'],
                'lots': dca_info['lots'],
                'time': datetime.now().isoformat()
            })
            
            self.current_level = dca_info['level']
            self._calculate_averages()
            self._save_state()
            
            logger.info(f"📈 DROP{dca_info['level']}: +{contracts}x @ ${price:.2f} (drop: ${dca_info['drop']:.0f})")
            return True
            
        except Exception as e:
            logger.error(f"Drop-buy failed: {e}")
            return False
    
    def _calculate_averages(self):
        """Calculate average entry and total contracts"""
        if not self.positions:
            self.dca_average_entry = 0
            self.dca_total_contracts = 0
            return
        
        total_value = sum(p['entry_price'] * p['contracts'] for p in self.positions)
        self.dca_total_contracts = sum(p['contracts'] for p in self.positions)
        self.dca_average_entry = total_value / self.dca_total_contracts if self.dca_total_contracts > 0 else 0
    
    # =========================================================================
    # TRAILING TAKE PROFIT
    # =========================================================================
    
    def manage_trailing_tp(self, current_price: float) -> Optional[Dict]:
        """
        Trailing TP: Start at +$10, trail $8 behind
        Port of MT5 ManageMetaTrailingTP()
        """
        if not self.positions or self.dca_average_entry == 0:
            return None
        
        # Calculate profit from average entry
        profit_per_contract = (current_price - self.dca_average_entry) * 10  # $10 per point
        total_profit = profit_per_contract * self.dca_total_contracts
        
        # Fixed TP at +$20 from average
        tp_price = self.dca_average_entry + self.settings['dropbuy_tp_pips']
        if current_price >= tp_price:
            return {
                'action': 'CLOSE_ALL',
                'reason': 'TP_HIT',
                'profit': total_profit
            }
        
        # Start trailing at +$10 profit
        if total_profit >= self.settings['trail_start']:
            new_sl = current_price - self.settings['trail_distance']
            
            if new_sl > self.trailing_sl:
                self.trailing_sl = new_sl
                logger.debug(f"   Trailing SL updated: ${new_sl:.2f}")
        
        # Check if trailing SL hit
        if self.trailing_sl > 0 and current_price <= self.trailing_sl:
            return {
                'action': 'CLOSE_ALL',
                'reason': 'TRAILING_SL',
                'profit': total_profit
            }
        
        return None
    
    def execute_close_all(self, close_info: Dict, current_price: float) -> bool:
        """Close all positions"""
        try:
            order = MarketOrder('SELL', self.dca_total_contracts)
            order.orderRef = f"CASPER|{close_info['reason']}"
            
            self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)
            
            logger.info(f"💰 {close_info['reason']}: ${close_info['profit']:.2f}")
            
            # Reset state
            self.positions.clear()
            self.current_level = 0
            self.trailing_sl = 0
            self.session_high = current_price  # Reset session high
            self._calculate_averages()
            self._save_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Close failed: {e}")
            return False
    
    # =========================================================================
    # META BOT SCORE (10 Weighted Indicators)
    # =========================================================================
    
    def calculate_meta_score(self, candles: List[Dict]) -> float:
        """
        Calculate Meta Bot composite score (0-100)
        Based on 10 weighted indicators
        """
        if len(candles) < 50:
            return 50  # Neutral
        
        closes = [c['close'] for c in candles]
        current_price = closes[-1]
        
        scores = []
        
        # 1. RSI (weight: 15%)
        rsi = Indicators.rsi_single(closes, 14)
        if rsi < 30:
            scores.append(('RSI', 80, 0.15))  # Oversold = bullish
        elif rsi > 70:
            scores.append(('RSI', 20, 0.15))  # Overbought = bearish
        else:
            scores.append(('RSI', 50, 0.15))
        
        # 2. EMA Trend (weight: 15%)
        ema20 = Indicators.ema_single(closes, 20)
        ema50 = Indicators.ema_single(closes, 50)
        if current_price > ema20 > ema50:
            scores.append(('EMA_TREND', 80, 0.15))
        elif current_price < ema20 < ema50:
            scores.append(('EMA_TREND', 20, 0.15))
        else:
            scores.append(('EMA_TREND', 50, 0.15))
        
        # 3. MACD (weight: 10%)
        macd = Indicators.macd(closes)
        if macd['histogram'] and macd['histogram'][-1] > 0:
            scores.append(('MACD', 70, 0.10))
        else:
            scores.append(('MACD', 30, 0.10))
        
        # 4. ADX Strength (weight: 10%)
        adx = Indicators.adx_single(candles, 14)
        if adx > 25:
            scores.append(('ADX', 70, 0.10))  # Strong trend
        else:
            scores.append(('ADX', 40, 0.10))  # Weak trend
        
        # 5. Price vs EMA20 (weight: 10%)
        if current_price > ema20:
            scores.append(('PRICE_EMA20', 70, 0.10))
        else:
            scores.append(('PRICE_EMA20', 30, 0.10))
        
        # 6. SuperTrend (weight: 15%)
        st = Indicators.supertrend(candles, 10, 3)
        if st['trend'] == 1:
            scores.append(('SUPERTREND', 80, 0.15))
        else:
            scores.append(('SUPERTREND', 20, 0.15))
        
        # 7. ATR Volatility (weight: 5%)
        atr = Indicators.atr_single(candles, 14)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        if atr_pct > 1:
            scores.append(('ATR', 60, 0.05))
        else:
            scores.append(('ATR', 40, 0.05))
        
        # 8. Price momentum (weight: 5%)
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            if momentum > 0.5:
                scores.append(('MOMENTUM', 70, 0.05))
            elif momentum < -0.5:
                scores.append(('MOMENTUM', 30, 0.05))
            else:
                scores.append(('MOMENTUM', 50, 0.05))
        else:
            scores.append(('MOMENTUM', 50, 0.05))
        
        # 9. Higher highs (weight: 5%)
        if len(candles) >= 3:
            if candles[-1]['high'] > candles[-2]['high'] > candles[-3]['high']:
                scores.append(('HH', 75, 0.05))
            elif candles[-1]['low'] < candles[-2]['low'] < candles[-3]['low']:
                scores.append(('HH', 25, 0.05))
            else:
                scores.append(('HH', 50, 0.05))
        else:
            scores.append(('HH', 50, 0.05))
        
        # 10. Volume trend (weight: 10%) - simplified
        scores.append(('VOLUME', 50, 0.10))  # Neutral without real volume
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in scores)
        
        return total_score
    
    def should_enter_meta(self, candles: List[Dict]) -> bool:
        """Check if Meta Bot score is above threshold"""
        score = self.calculate_meta_score(candles)
        return score >= self.settings['meta_min_score']
    
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
        
        # Check Drop-Buy DCA
        dca = self.check_drop_buy_dca(current_price)
        if dca:
            self.execute_drop_buy(dca, current_price)
        
        # Check trailing TP
        tp = self.manage_trailing_tp(current_price)
        if tp:
            self.execute_close_all(tp, current_price)
        
        # Calculate status
        self._calculate_averages()
        unrealized = (current_price - self.dca_average_entry) * self.dca_total_contracts * 10 if self.dca_total_contracts > 0 else 0
        
        return {
            'price': current_price,
            'session_high': self.session_high,
            'drop_level': self.current_level,
            'contracts': self.dca_total_contracts,
            'avg_entry': self.dca_average_entry,
            'unrealized_pnl': unrealized,
            'trailing_sl': self.trailing_sl
        }
    
    def run(self, interval: int = 30):
        """Main trading loop"""
        print("=" * 70)
        print(f"👻 {self.name} IB - Conservative Drop-Buy Martingale")
        print("=" * 70)
        print(f"Drop Trigger: ${self.settings['dropbuy_trigger_pips']} per level")
        print(f"Lot Ladder: {self.lot_ladder}")
        print(f"TP: +${self.settings['dropbuy_tp_pips']} from avg")
        print(f"Trail: +${self.settings['trail_start']} start, ${self.settings['trail_distance']} distance")
        print("=" * 70)
        
        while True:
            try:
                status = self.run_once()
                
                if 'price' in status:
                    trail = f"Trail:${status['trailing_sl']:.0f}" if status['trailing_sl'] > 0 else ""
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"${status['price']:.2f} | "
                          f"High:${status['session_high']:.2f} | "
                          f"L{status['drop_level']} ({status['contracts']}x) | "
                          f"Avg: ${status['avg_entry']:.2f} | "
                          f"P&L: ${status['unrealized_pnl']:.2f} {trail}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print(f"\n👻 {self.name} shutting down...")
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
            'positions': self.positions,
            'session_high': self.session_high,
            'current_level': self.current_level,
            'trailing_sl': self.trailing_sl,
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
                    self.positions = state.get('positions', [])
                    self.session_high = state.get('session_high', 0)
                    self.current_level = state.get('current_level', 0)
                    self.trailing_sl = state.get('trailing_sl', 0)
                    self._calculate_averages()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    casper = CasperIB(paper_trading=True)
    
    if casper.connect():
        try:
            casper.run(interval=30)
        finally:
            casper.disconnect()
    else:
        print("❌ Failed to connect")

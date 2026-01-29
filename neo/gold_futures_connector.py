"""
Gold Futures Connector for Interactive Brokers
Trade MGC (Micro Gold Futures) on COMEX

Contract Specs:
- Symbol: MGC
- Exchange: COMEX
- Size: 10 troy ounces per contract
- Tick: $0.10 ($1 per contract per tick)
- Margin: ~$1,000-1,500 per contract

Author: QUINN001
Created: 2026-01-28
"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Future, Contract, Order, LimitOrder, MarketOrder
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoldFutures")


class GoldFuturesConnector:
    """
    Trade Gold Futures (MGC) on Interactive Brokers
    Designed for Ghost Commander strategy
    """
    
    # Gold futures contract months: Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)
    GOLD_MONTHS = {
        2: 'G', 4: 'J', 6: 'M', 8: 'Q', 10: 'V', 12: 'Z'
    }
    
    def __init__(self, paper_trading: bool = True, host: str = "100.119.161.65", 
                 client_id: int = 50):
        """
        Initialize Gold Futures connector
        
        Args:
            paper_trading: True for paper (7497), False for live (7496)
            host: TWS host IP
            client_id: Unique client ID
        """
        self.ib = IB()
        self.paper_trading = paper_trading
        self.host = host
        self.client_id = client_id
        self.port = 7497 if paper_trading else 7496
        self.connected = False
        
        # Contract cache
        self.front_month_contract = None
        self.contract_expiry = None
        
        # Position tracking
        self.positions = {}
        self.orders = {}
        self.last_price = 0
        
    def connect(self, timeout: int = 30) -> bool:
        """Connect to TWS"""
        try:
            if self.ib.isConnected():
                logger.info("Already connected")
                return True
            
            mode = "PAPER" if self.paper_trading else "LIVE"
            logger.info(f"Connecting to IB {mode} on {self.host}:{self.port}...")
            
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id,
                timeout=timeout,
                readonly=False
            )
            
            # Wait for connection to be ready
            self.ib.sleep(3)
            self.connected = True
            
            logger.info(f"✅ Connected to IB {mode} for Gold Futures")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self.ib.isConnected()
    
    # =========================================================================
    # CONTRACT MANAGEMENT
    # =========================================================================
    
    def get_front_month(self, avoid_delivery: bool = True) -> str:
        """
        Get front month contract expiry (YYYYMM format)
        Gold futures: Feb, Apr, Jun, Aug, Oct, Dec
        
        Args:
            avoid_delivery: If True, skip contracts within 5 days of expiry
                           to avoid IBKR delivery window restrictions
        """
        now = datetime.now()
        
        # Gold contract months
        months = [2, 4, 6, 8, 10, 12]
        
        # Find next valid month
        for month in months:
            # Check if this month is valid
            if month > now.month:
                # Future month this year - check delivery window
                if avoid_delivery and month == (now.month + 1) and now.day > 20:
                    # Too close to next month's delivery, skip
                    continue
                return f"{now.year}{month:02d}"
            elif month == now.month:
                # Current month - need at least 10 days buffer
                if now.day < 10 and not avoid_delivery:
                    return f"{now.year}{month:02d}"
                # Otherwise skip current month
                continue
        
        # Roll to next year
        return f"{now.year + 1}02"
    
    def get_active_contract_month(self) -> str:
        """
        Get the actively traded contract month (avoids delivery issues)
        Uses the 2nd month out to be safe
        """
        now = datetime.now()
        months = [2, 4, 6, 8, 10, 12]
        
        # Find the contract that's at least 30 days out
        for month in months:
            year = now.year
            if month <= now.month:
                year += 1
            
            expiry_approx = datetime(year, month, 25)
            days_to_expiry = (expiry_approx - now).days
            
            if days_to_expiry >= 30:
                return f"{year}{month:02d}"
        
        # Default to next year April (safe)
        return f"{now.year + 1}04"
    
    def create_gold_contract(self, expiry: str = None) -> Future:
        """
        Create Micro Gold futures contract
        
        Args:
            expiry: "202602" format or None for active month
        """
        if expiry is None:
            # Use the actively traded month (avoids delivery window issues)
            expiry = self.get_active_contract_month()
        
        contract = Future(
            symbol='MGC',
            lastTradeDateOrContractMonth=expiry,
            exchange='COMEX',
            currency='USD'
        )
        
        # Qualify the contract
        if self.is_connected():
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self.front_month_contract = qualified[0]
                self.contract_expiry = expiry
                logger.info(f"🥇 Using MGC contract: {expiry} ({qualified[0].localSymbol})")
                return qualified[0]
        
        return contract
    
    def get_contract(self) -> Future:
        """Get or create the front month contract"""
        if self.front_month_contract is None:
            self.create_gold_contract()
        return self.front_month_contract
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def get_gold_price(self) -> Dict:
        """
        Get current gold futures price
        
        Returns:
            Dict with bid, ask, last, high, low, volume
        """
        if not self.is_connected():
            return {'error': 'Not connected'}
        
        try:
            contract = self.get_contract()
            
            # Request delayed data (for paper accounts without live subscription)
            self.ib.reqMarketDataType(3)  # 3 = delayed data
            
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Give more time for delayed data
            
            price_data = {
                'symbol': 'MGC',
                'expiry': self.contract_expiry,
                'bid': ticker.bid if ticker.bid and ticker.bid > 0 else ticker.bidPrice if hasattr(ticker, 'bidPrice') and ticker.bidPrice else None,
                'ask': ticker.ask if ticker.ask and ticker.ask > 0 else ticker.askPrice if hasattr(ticker, 'askPrice') and ticker.askPrice else None,
                'last': ticker.last if ticker.last and ticker.last > 0 else ticker.lastPrice if hasattr(ticker, 'lastPrice') and ticker.lastPrice else None,
                'close': ticker.close if ticker.close and ticker.close > 0 else None,
                'high': ticker.high if ticker.high and ticker.high > 0 else None,
                'low': ticker.low if ticker.low and ticker.low > 0 else None,
                'volume': ticker.volume if ticker.volume else 0,
                'delayed': True,  # Mark as delayed data
                'timestamp': datetime.now().isoformat()
            }
            
            # Use close or last price
            if price_data['last']:
                self.last_price = price_data['last']
            elif price_data['close']:
                self.last_price = price_data['close']
                price_data['last'] = price_data['close']
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting gold price: {e}")
            return {'error': str(e)}
    
    def get_gold_price_from_history(self) -> Dict:
        """
        Get gold price from historical data (more reliable for paper accounts)
        """
        if not self.is_connected():
            return {'error': 'Not connected'}
        
        try:
            contract = self.get_contract()
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
            
            if bars:
                last_bar = bars[-1]
                self.last_price = last_bar.close
                return {
                    'symbol': 'MGC',
                    'expiry': self.contract_expiry,
                    'last': last_bar.close,
                    'high': last_bar.high,
                    'low': last_bar.low,
                    'open': last_bar.open,
                    'bar_time': str(last_bar.date),
                    'source': 'historical',
                    'timestamp': datetime.now().isoformat()
                }
            
            return {'error': 'No historical data'}
            
        except Exception as e:
            logger.error(f"Error getting historical price: {e}")
            return {'error': str(e)}
    
    def get_historical_candles(self, duration: str = "1 D", bar_size: str = "1 hour") -> List[Dict]:
        """
        Get historical candles for regime detection
        
        Args:
            duration: "1 D", "1 W", etc.
            bar_size: "1 hour", "4 hours", "1 day"
        """
        if not self.is_connected():
            return []
        
        try:
            contract = self.get_contract()
            bars = self.ib.reqHistoricalData(
                contract,
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
            logger.error(f"Error getting historical data: {e}")
            return []
    
    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================
    
    def buy_gold(self, quantity: int, limit_price: float = None, 
                 order_ref: str = None) -> Dict:
        """
        Buy gold futures
        
        Args:
            quantity: Number of MGC contracts (1 MGC = 10 oz)
            limit_price: Limit price or None for market
            order_ref: Order reference (e.g., "DROPBUYL1")
        
        Returns:
            Dict with order_id, status, etc.
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected'}
        
        try:
            contract = self.get_contract()
            
            if limit_price:
                order = LimitOrder('BUY', quantity, limit_price)
            else:
                order = MarketOrder('BUY', quantity)
            
            if order_ref:
                order.orderRef = order_ref
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            result = {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'action': 'BUY',
                'symbol': 'MGC',
                'quantity': quantity,
                'limit_price': limit_price,
                'order_ref': order_ref,
                'timestamp': datetime.now().isoformat()
            }
            
            self.orders[str(trade.order.orderId)] = result
            logger.info(f"🥇 BUY {quantity}x MGC @ {'MKT' if not limit_price else f'${limit_price:.2f}'} [{order_ref or 'MANUAL'}]")
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'success': False, 'error': str(e)}
    
    def sell_gold(self, quantity: int, limit_price: float = None,
                  order_ref: str = None) -> Dict:
        """
        Sell gold futures
        
        Args:
            quantity: Number of MGC contracts
            limit_price: Limit price or None for market
            order_ref: Order reference (e.g., "TP1")
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected'}
        
        try:
            contract = self.get_contract()
            
            if limit_price:
                order = LimitOrder('SELL', quantity, limit_price)
            else:
                order = MarketOrder('SELL', quantity)
            
            if order_ref:
                order.orderRef = order_ref
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            result = {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'action': 'SELL',
                'symbol': 'MGC',
                'quantity': quantity,
                'limit_price': limit_price,
                'order_ref': order_ref,
                'timestamp': datetime.now().isoformat()
            }
            
            self.orders[str(trade.order.orderId)] = result
            logger.info(f"🥇 SELL {quantity}x MGC @ {'MKT' if not limit_price else f'${limit_price:.2f}'} [{order_ref or 'MANUAL'}]")
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def get_gold_positions(self) -> List[Dict]:
        """Get current gold futures positions"""
        if not self.is_connected():
            return []
        
        try:
            positions = self.ib.positions()
            gold_positions = []
            
            for pos in positions:
                if pos.contract.symbol == 'MGC':
                    gold_positions.append({
                        'symbol': 'MGC',
                        'quantity': pos.position,
                        'avg_cost': pos.avgCost,
                        'expiry': pos.contract.lastTradeDateOrContractMonth,
                        'market_value': pos.position * self.last_price * 10 if self.last_price else 0,
                        'unrealized_pnl': (self.last_price - pos.avgCost) * pos.position * 10 if self.last_price else 0
                    })
            
            return gold_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_open_orders(self) -> List[Dict]:
        """Get open orders"""
        if not self.is_connected():
            return []
        
        try:
            trades = self.ib.openTrades()
            return [
                {
                    'order_id': t.order.orderId,
                    'symbol': t.contract.symbol,
                    'action': t.order.action,
                    'quantity': t.order.totalQuantity,
                    'order_type': t.order.orderType,
                    'limit_price': t.order.lmtPrice if t.order.orderType == 'LMT' else None,
                    'status': t.orderStatus.status,
                    'filled': t.orderStatus.filled
                }
                for t in trades
                if t.contract.symbol == 'MGC'
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def cancel_order(self, order_id: int) -> Dict:
        """Cancel an open order"""
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected'}
        
        try:
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    return {'success': True, 'message': f'Cancelled order {order_id}'}
            
            return {'success': False, 'error': f'Order {order_id} not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🥇 GOLD FUTURES CONNECTOR TEST")
    print("=" * 60)
    
    gc = GoldFuturesConnector(paper_trading=True, client_id=50)
    
    if gc.connect():
        print(f"\n📊 Front month: {gc.get_front_month()}")
        
        print("\n📈 Getting gold price...")
        price = gc.get_gold_price()
        print(f"   Price: {price}")
        
        print("\n📋 Current positions:")
        positions = gc.get_gold_positions()
        print(f"   {positions}")
        
        gc.disconnect()
    else:
        print("❌ Connection failed")

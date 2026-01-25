"""
Interactive Brokers Connector for NEO Options Trading
Connects to TWS/IB Gateway for automated IREN options trading

Ports:
- Paper Trading: 7497 (TWS) or 4002 (IB Gateway)
- Live Trading: 7496 (TWS) or 4001 (IB Gateway)

Requirements:
1. TWS or IB Gateway running on local machine
2. API enabled in TWS settings
3. pip install ib_insync
"""

from ib_insync import IB, Stock, Option, Contract, Order, LimitOrder, MarketOrder
from ib_insync import util
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import logging
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "neo" / "ibkr_data"
DATA_DIR.mkdir(exist_ok=True)


class IBKRConnector:
    """
    Connect NEO to Interactive Brokers for options trading
    
    Usage:
        connector = IBKRConnector(paper_trading=True)
        connector.connect()
        connector.get_iren_options_chain()
        connector.buy_iren_call(expiry="20260220", strike=60, quantity=5)
    """
    
    # Paul's trading rules
    PAUL_RULES = {
        'allowed_actions': ['BUY_CALL'],  # LONG ONLY
        'blocked_actions': ['SELL_CALL', 'BUY_PUT', 'SELL_PUT'],
        'preferred_expirations': ['20260220', '20260227'],  # Feb 20, Feb 27
        'blocked_expirations': ['20260130', '20260205'],  # Jan 30, Feb 5 (earnings)
        'min_dte': 14,
        'max_contracts_per_trade': 50,
        'max_total_contracts': 200,
        'min_confidence': 70,
        'preferred_strikes': [55, 60, 65, 70],
    }
    
    def __init__(self, paper_trading: bool = True, host: str = "127.0.0.1", 
                 client_id: int = 1):
        """
        Initialize IBKR connector
        
        Args:
            paper_trading: True for paper trading (port 7497), False for live (7496)
            host: TWS/Gateway host (usually localhost)
            client_id: Unique client ID for this connection
        """
        self.ib = IB()
        self.paper_trading = paper_trading
        self.host = host
        self.client_id = client_id
        self.port = 7497 if paper_trading else 7496
        self.connected = False
        
        # Cache
        self.positions = {}
        self.account_summary = {}
        self.options_chains = {}
        self.orders = {}
        self.executions = []
        
        # State file
        self.state_file = DATA_DIR / "ibkr_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load saved state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.orders = state.get('orders', {})
                    self.executions = state.get('executions', [])
            except:
                pass
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'orders': self.orders,
            'executions': self.executions[-100:],  # Keep last 100
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONNECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def connect(self) -> bool:
        """
        Connect to TWS/IB Gateway
        
        Returns:
            True if connected successfully
        """
        try:
            if self.ib.isConnected():
                logger.info("Already connected to IBKR")
                self.connected = True
                return True
            
            mode = "PAPER" if self.paper_trading else "LIVE"
            logger.info(f"Connecting to IBKR {mode} on {self.host}:{self.port}...")
            
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            self.connected = True
            logger.info(f"✅ Connected to IBKR {mode}")
            
            # Subscribe to events
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_execution
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.ib.isConnected()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _on_order_status(self, trade):
        """Callback for order status updates"""
        order = trade.order
        status = trade.orderStatus
        
        order_info = {
            'order_id': order.orderId,
            'status': status.status,
            'filled': status.filled,
            'remaining': status.remaining,
            'avg_fill_price': status.avgFillPrice,
            'last_fill_price': status.lastFillPrice,
            'timestamp': datetime.now().isoformat()
        }
        
        self.orders[str(order.orderId)] = order_info
        self._save_state()
        
        logger.info(f"Order {order.orderId}: {status.status} | "
                   f"Filled: {status.filled}/{order.totalQuantity} @ {status.avgFillPrice}")
    
    def _on_execution(self, trade, fill):
        """Callback for execution details"""
        execution = {
            'order_id': trade.order.orderId,
            'symbol': trade.contract.symbol,
            'action': trade.order.action,
            'quantity': fill.execution.shares,
            'price': fill.execution.price,
            'time': fill.execution.time.isoformat() if fill.execution.time else None,
            'commission': fill.commissionReport.commission if fill.commissionReport else None
        }
        
        self.executions.append(execution)
        self._save_state()
        
        logger.info(f"✅ EXECUTED: {execution['action']} {execution['quantity']} "
                   f"{execution['symbol']} @ {execution['price']}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ACCOUNT & POSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary
        
        Returns:
            Dict with NetLiquidation, BuyingPower, AvailableFunds, etc.
        """
        if not self.is_connected():
            return {'error': 'Not connected to IBKR'}
        
        try:
            account_values = self.ib.accountSummary()
            
            summary = {}
            for av in account_values:
                if av.tag in ['NetLiquidation', 'BuyingPower', 'AvailableFunds', 
                             'TotalCashValue', 'GrossPositionValue', 'UnrealizedPnL',
                             'RealizedPnL']:
                    summary[av.tag] = float(av.value)
            
            self.account_summary = summary
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions
        
        Returns:
            List of position dicts
        """
        if not self.is_connected():
            return []
        
        try:
            positions = self.ib.positions()
            
            result = []
            for pos in positions:
                contract = pos.contract
                
                # Get current market price
                self.ib.qualifyContracts(contract)
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(1)
                
                current_price = ticker.marketPrice() if ticker.marketPrice() else pos.avgCost
                
                position_info = {
                    'symbol': contract.symbol,
                    'sec_type': contract.secType,
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': contract.right,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'current_price': current_price,
                    'market_value': pos.position * current_price * 100 if contract.secType == 'OPT' else pos.position * current_price,
                    'pnl': (current_price - pos.avgCost) * pos.position * (100 if contract.secType == 'OPT' else 1),
                    'pnl_pct': ((current_price / pos.avgCost) - 1) * 100 if pos.avgCost > 0 else 0
                }
                
                result.append(position_info)
                self.positions[f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"] = position_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_iren_positions(self) -> List[Dict]:
        """Get IREN options positions only"""
        positions = self.get_positions()
        return [p for p in positions if p['symbol'] == 'IREN']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # IREN OPTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_iren_option_contract(self, expiry: str, strike: float, 
                                     right: str = "C") -> Option:
        """
        Create IREN option contract
        
        Args:
            expiry: "20260220" format (YYYYMMDD)
            strike: Strike price (e.g., 60.0)
            right: "C" for Call, "P" for Put
            
        Returns:
            Option contract
        """
        contract = Option(
            symbol='IREN',
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='SMART',
            currency='USD'
        )
        
        if self.is_connected():
            self.ib.qualifyContracts(contract)
        
        return contract
    
    def get_iren_stock_price(self) -> float:
        """Get current IREN stock price"""
        if not self.is_connected():
            return 0.0
        
        try:
            stock = Stock('IREN', 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            ticker = self.ib.reqMktData(stock, '', False, False)
            self.ib.sleep(1)
            return ticker.marketPrice() or ticker.last or 0.0
        except Exception as e:
            logger.error(f"Error getting IREN price: {e}")
            return 0.0
    
    def get_iren_options_chain(self, min_dte: int = 14, max_dte: int = 90) -> Dict:
        """
        Get IREN options chain
        
        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            Dict with expirations, strikes, and chain data
        """
        if not self.is_connected():
            return {'error': 'Not connected to IBKR'}
        
        try:
            # Get underlying
            stock = Stock('IREN', 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            # Get chains
            chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
            
            if not chains:
                return {'error': 'No options chains found'}
            
            chain = chains[0]  # Usually SMART exchange
            
            # Filter expirations by DTE
            today = datetime.now().date()
            valid_expirations = []
            
            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, "%Y%m%d").date()
                dte = (exp_date - today).days
                
                if min_dte <= dte <= max_dte:
                    # Check if not in blocked list
                    if exp not in self.PAUL_RULES['blocked_expirations']:
                        valid_expirations.append({
                            'expiry': exp,
                            'dte': dte,
                            'is_paul_pick': exp in self.PAUL_RULES['preferred_expirations']
                        })
            
            # Sort by DTE
            valid_expirations.sort(key=lambda x: x['dte'])
            
            # Filter strikes around current price
            current_price = self.get_iren_stock_price()
            if current_price == 0:
                current_price = 55.0  # Default if can't get price
            
            valid_strikes = [s for s in chain.strikes 
                           if current_price * 0.7 <= s <= current_price * 1.5]
            
            result = {
                'symbol': 'IREN',
                'current_price': current_price,
                'expirations': valid_expirations,
                'strikes': sorted(valid_strikes),
                'exchange': chain.exchange,
                'multiplier': chain.tradingClass,
                'recommended': {
                    'expirations': self.PAUL_RULES['preferred_expirations'],
                    'strikes': self.PAUL_RULES['preferred_strikes'],
                    'reason': "Paul prefers Feb 20 & Feb 27, avoid Feb 5 earnings"
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.options_chains['IREN'] = result
            return result
            
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return {'error': str(e)}
    
    def get_option_quote(self, expiry: str, strike: float, right: str = "C") -> Dict:
        """
        Get quote for specific option
        
        Returns:
            Dict with bid, ask, last, volume, open interest, greeks
        """
        if not self.is_connected():
            return {'error': 'Not connected to IBKR'}
        
        try:
            contract = self.create_iren_option_contract(expiry, strike, right)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)
            
            return {
                'symbol': 'IREN',
                'expiry': expiry,
                'strike': strike,
                'right': right,
                'bid': ticker.bid or 0,
                'ask': ticker.ask or 0,
                'last': ticker.last or 0,
                'volume': ticker.volume or 0,
                'open_interest': ticker.openInterest or 0,
                'iv': ticker.modelGreeks.impliedVol if ticker.modelGreeks else None,
                'delta': ticker.modelGreeks.delta if ticker.modelGreeks else None,
                'gamma': ticker.modelGreeks.gamma if ticker.modelGreeks else None,
                'theta': ticker.modelGreeks.theta if ticker.modelGreeks else None,
                'vega': ticker.modelGreeks.vega if ticker.modelGreeks else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting option quote: {e}")
            return {'error': str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _validate_paul_rules(self, action: str, expiry: str, quantity: int) -> Dict:
        """
        Validate trade against Paul's rules
        
        Returns:
            Dict with 'valid': bool, 'reason': str
        """
        # Check action
        if action not in self.PAUL_RULES['allowed_actions']:
            return {
                'valid': False,
                'reason': f"Action '{action}' not allowed. Paul is LONG ONLY."
            }
        
        # Check expiration
        if expiry in self.PAUL_RULES['blocked_expirations']:
            return {
                'valid': False,
                'reason': f"Expiration {expiry} blocked (near earnings Feb 5)"
            }
        
        # Check DTE
        exp_date = datetime.strptime(expiry, "%Y%m%d").date()
        dte = (exp_date - datetime.now().date()).days
        if dte < self.PAUL_RULES['min_dte']:
            return {
                'valid': False,
                'reason': f"DTE {dte} is below minimum {self.PAUL_RULES['min_dte']}"
            }
        
        # Check quantity
        if quantity > self.PAUL_RULES['max_contracts_per_trade']:
            return {
                'valid': False,
                'reason': f"Quantity {quantity} exceeds max {self.PAUL_RULES['max_contracts_per_trade']} per trade"
            }
        
        # Check total position
        current_iren = sum(p.get('quantity', 0) for p in self.get_iren_positions())
        if current_iren + quantity > self.PAUL_RULES['max_total_contracts']:
            return {
                'valid': False,
                'reason': f"Total position would exceed {self.PAUL_RULES['max_total_contracts']} contracts"
            }
        
        return {'valid': True, 'reason': 'Passed all Paul rules'}
    
    def buy_iren_call(self, expiry: str, strike: float, quantity: int,
                      limit_price: float = None, validate_rules: bool = True) -> Dict:
        """
        Buy IREN call option
        
        Args:
            expiry: "20260220" format
            strike: Strike price
            quantity: Number of contracts
            limit_price: Limit price (None = market order)
            validate_rules: Check against Paul's rules
            
        Returns:
            Dict with order_id, status, message
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Validate Paul's rules
        if validate_rules:
            validation = self._validate_paul_rules('BUY_CALL', expiry, quantity)
            if not validation['valid']:
                return {'success': False, 'error': validation['reason']}
        
        try:
            contract = self.create_iren_option_contract(expiry, strike, "C")
            
            if limit_price:
                order = LimitOrder('BUY', quantity, limit_price)
            else:
                order = MarketOrder('BUY', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for order to be acknowledged
            
            order_info = {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'action': 'BUY_CALL',
                'symbol': 'IREN',
                'expiry': expiry,
                'strike': strike,
                'quantity': quantity,
                'limit_price': limit_price,
                'order_type': 'LMT' if limit_price else 'MKT',
                'message': f"BUY {quantity}x IREN {strike}C {expiry} @ {'MKT' if not limit_price else limit_price}",
                'timestamp': datetime.now().isoformat()
            }
            
            self.orders[str(trade.order.orderId)] = order_info
            self._save_state()
            
            logger.info(f"📈 {order_info['message']}")
            return order_info
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'success': False, 'error': str(e)}
    
    def sell_iren_call(self, expiry: str, strike: float, quantity: int,
                       limit_price: float = None) -> Dict:
        """
        Sell IREN call option (close position)
        NOTE: This is for CLOSING positions only, not opening shorts
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        try:
            contract = self.create_iren_option_contract(expiry, strike, "C")
            
            if limit_price:
                order = LimitOrder('SELL', quantity, limit_price)
            else:
                order = MarketOrder('SELL', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            order_info = {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'action': 'SELL_CALL',
                'symbol': 'IREN',
                'expiry': expiry,
                'strike': strike,
                'quantity': quantity,
                'limit_price': limit_price,
                'message': f"SELL {quantity}x IREN {strike}C {expiry}",
                'timestamp': datetime.now().isoformat()
            }
            
            self.orders[str(trade.order.orderId)] = order_info
            self._save_state()
            
            logger.info(f"📉 {order_info['message']}")
            return order_info
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return {'success': False, 'error': str(e)}
    
    def buy_iren_put(self, expiry: str, strike: float, quantity: int,
                     limit_price: float = None) -> Dict:
        """
        Buy IREN put option (for hedging only)
        """
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        try:
            contract = self.create_iren_option_contract(expiry, strike, "P")
            
            if limit_price:
                order = LimitOrder('BUY', quantity, limit_price)
            else:
                order = MarketOrder('BUY', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            order_info = {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'action': 'BUY_PUT',
                'symbol': 'IREN',
                'expiry': expiry,
                'strike': strike,
                'quantity': quantity,
                'limit_price': limit_price,
                'message': f"BUY {quantity}x IREN {strike}P {expiry} (HEDGE)",
                'timestamp': datetime.now().isoformat()
            }
            
            self.orders[str(trade.order.orderId)] = order_info
            self._save_state()
            
            logger.info(f"🛡️ {order_info['message']}")
            return order_info
            
        except Exception as e:
            logger.error(f"Error placing put order: {e}")
            return {'success': False, 'error': str(e)}
    
    def cancel_order(self, order_id: int) -> Dict:
        """Cancel an open order"""
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        try:
            # Find the trade
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    return {'success': True, 'message': f'Order {order_id} cancelled'}
            
            return {'success': False, 'error': f'Order {order_id} not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
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
                    'filled': t.orderStatus.filled,
                    'remaining': t.orderStatus.remaining
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FULL STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_full_status(self) -> Dict:
        """Get complete IBKR status"""
        return {
            'connected': self.is_connected(),
            'mode': 'PAPER' if self.paper_trading else 'LIVE',
            'port': self.port,
            'account': self.get_account_summary(),
            'iren_positions': self.get_iren_positions(),
            'open_orders': self.get_open_orders(),
            'recent_executions': self.executions[-10:],
            'paul_rules': self.PAUL_RULES,
            'timestamp': datetime.now().isoformat()
        }


# Standalone test
if __name__ == "__main__":
    print("=" * 70)
    print("🏦 IBKR CONNECTOR TEST")
    print("=" * 70)
    
    print("""
    To test, make sure:
    1. TWS or IB Gateway is running
    2. Logged into Paper Trading account
    3. API is enabled (port 7497)
    
    Then uncomment the test code below.
    """)
    
    # Uncomment to test:
    # connector = IBKRConnector(paper_trading=True)
    # 
    # if connector.connect():
    #     print("\n📊 Account Summary:")
    #     print(connector.get_account_summary())
    #     
    #     print("\n📋 IREN Options Chain:")
    #     chain = connector.get_iren_options_chain()
    #     print(f"Expirations: {[e['expiry'] for e in chain.get('expirations', [])]}")
    #     print(f"Strikes: {chain.get('strikes', [])}")
    #     
    #     print("\n💰 Option Quote (IREN Feb20 60C):")
    #     quote = connector.get_option_quote("20260220", 60, "C")
    #     print(quote)
    #     
    #     # Test order (uncomment carefully!)
    #     # result = connector.buy_iren_call("20260220", 60, 1, limit_price=2.50)
    #     # print(result)
    #     
    #     connector.disconnect()

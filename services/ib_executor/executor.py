"""
IB Trade Executor
=================
Executes trades on Interactive Brokers.

Paper Mode: Auto-execute all scout/sentinel signals
Live Mode: Require confirmation or use DCA
"""

import logging
from typing import Dict, List
from datetime import datetime

from ib_insync import MarketOrder, LimitOrder, StopOrder

from .connection import IBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IBExecutor")


class IBExecutor:
    """
    Executes trades on Interactive Brokers.
    
    Paper Mode: Auto-execute all scout/sentinel signals
    Live Mode: Require confirmation or use smaller sizes
    """
    
    def __init__(self, connection: IBConnection):
        self.conn = connection
        self.ib = connection.ib
        
        # Execution settings
        self.paper_auto_execute = True
        self.live_auto_execute = False  # Require confirmation in live
        
        # Position limits
        self.max_position_size = 500     # Max shares per position
        self.max_position_value = 10000  # Max $ per position
        self.max_positions = 15          # Max concurrent positions
        
        # Risk settings
        self.risk_per_trade = 0.01       # 1% of account per trade
        self.default_account_value = 100000  # Default if can't get real value
        
        # DCA settings (for live mode)
        self.dca_tranches = 4          # Number of DCA entries
        self.dca_spread = 0.02         # 2% between entries
        
        # Tracking
        self.pending_orders: List[Dict] = []
        self.executed_orders: List[Dict] = []
        self.failed_orders: List[Dict] = []
    
    def execute_scout_alert(self, alert: Dict) -> Dict:
        """
        Execute a trade based on scout alert.
        
        alert: {
            'symbol': 'IREN',
            'confidence': 82,
            'direction': 'LONG',
            'entry': 8.50,
            'stop': 8.00,
            'target1': 9.50,
            'target2': 10.50,
        }
        """
        
        symbol = alert.get('symbol', '')
        direction = alert.get('direction', '')
        confidence = alert.get('confidence', 0)
        entry_price = alert.get('entry', 0)
        stop_price = alert.get('stop', 0)
        
        logger.info(f"[EXECUTOR] Processing scout alert: {symbol} {direction} ({confidence}%)")
        
        # Validate
        if confidence < 75:
            return {'status': 'rejected', 'reason': f'Confidence {confidence}% below 75%'}
        
        if direction not in ['LONG', 'SHORT']:
            return {'status': 'rejected', 'reason': f'Invalid direction: {direction}'}
        
        if not symbol:
            return {'status': 'rejected', 'reason': 'No symbol provided'}
        
        # Skip crypto symbols (IB doesn't support direct crypto)
        crypto_symbols = ['BTC', 'ETH', 'BTCUSD', 'ETHUSD', 'BTC-USD', 'ETH-USD']
        if symbol.upper() in crypto_symbols:
            return {'status': 'skipped', 'reason': 'Crypto not supported on IB'}
        
        # Check mode and execute
        if self.conn.is_paper:
            # Paper mode: Auto-execute
            return self._execute_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                source='SCOUT',
                alert=alert,
            )
        else:
            # Live mode: Queue for confirmation or DCA
            if self.live_auto_execute:
                return self._execute_dca_trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    source='SCOUT',
                    alert=alert,
                )
            else:
                self.pending_orders.append({
                    'alert': alert,
                    'created_at': datetime.now().isoformat(),
                    'status': 'PENDING_CONFIRMATION',
                })
                return {'status': 'pending', 'message': 'Queued for confirmation (live mode)'}
    
    def execute_sentinel_alert(self, alert: Dict) -> Dict:
        """
        Execute based on Sentinel pattern alert.
        
        alert: {
            'symbol': 'GC=F',
            'patterns': ['Shooting Star'],
            'ghost_action': {
                'action': 'CLOSE_PARTIAL',
                'close_percent': 50,
            }
        }
        """
        
        ghost_action = alert.get('ghost_action', {})
        action = ghost_action.get('action', 'NONE')
        
        if action == 'NONE':
            return {'status': 'no_action', 'message': 'No action required'}
        
        symbol = alert.get('symbol', '')
        
        logger.info(f"[EXECUTOR] Processing sentinel alert: {symbol} - {action}")
        
        if action in ['CLOSE_PARTIAL', 'CLOSE_ALL', 'REDUCE_AND_PROTECT']:
            close_percent = ghost_action.get('close_percent', 50)
            return self._close_position(symbol, close_percent)
        
        elif action == 'OPPORTUNITY':
            # Bullish pattern - entry opportunity
            return self._execute_trade(
                symbol=symbol,
                direction='LONG',
                entry_price=0,  # Market order
                stop_price=0,
                source='SENTINEL',
                alert=alert,
            )
        
        elif action == 'CAUTION':
            # Just tighten stops, no new trades
            return {'status': 'caution', 'message': 'Tighten stops advised'}
        
        return {'status': 'unknown_action', 'action': action}
    
    def _execute_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        source: str,
        alert: Dict,
    ) -> Dict:
        """Execute a single trade."""
        
        try:
            # Get contract
            contract = self.conn.get_contract(symbol, sec_type="AUTO")
            
            if contract is None:
                return {'status': 'error', 'message': f'Could not create contract for {symbol}'}
            
            # Qualify contract
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return {'status': 'error', 'message': f'Could not qualify contract for {symbol}'}
            
            # Calculate position size
            quantity = self._calculate_position_size(symbol, entry_price, stop_price)
            
            if quantity <= 0:
                return {'status': 'error', 'message': 'Calculated quantity is 0'}
            
            # Create order
            action = 'BUY' if direction == 'LONG' else 'SELL'
            
            if entry_price > 0:
                order = LimitOrder(
                    action=action,
                    totalQuantity=quantity,
                    lmtPrice=round(entry_price, 2),
                )
            else:
                order = MarketOrder(
                    action=action,
                    totalQuantity=quantity,
                )
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait briefly for status update
            self.ib.sleep(1)
            
            result = {
                'status': 'submitted',
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'direction': direction,
                'action': action,
                'quantity': quantity,
                'entry_price': entry_price if entry_price > 0 else 'MARKET',
                'order_type': 'LIMIT' if entry_price > 0 else 'MARKET',
                'order_status': trade.orderStatus.status,
                'source': source,
                'mode': 'PAPER' if self.conn.is_paper else 'LIVE',
                'timestamp': datetime.now().isoformat(),
            }
            
            self.executed_orders.append(result)
            
            logger.info(f"[EXECUTOR] ✅ Order placed: {action} {quantity} {symbol} @ {entry_price if entry_price > 0 else 'MARKET'}")
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e),
                'symbol': symbol,
                'direction': direction,
                'source': source,
                'timestamp': datetime.now().isoformat(),
            }
            self.failed_orders.append(error_result)
            logger.error(f"[EXECUTOR] ❌ Order failed: {e}")
            return error_result
    
    def _execute_dca_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        source: str,
        alert: Dict,
    ) -> Dict:
        """
        Execute DCA (Dollar Cost Average) entry.
        Places multiple limit orders at different price levels.
        """
        
        try:
            contract = self.conn.get_contract(symbol, sec_type="AUTO")
            
            if contract is None:
                return {'status': 'error', 'message': f'Could not create contract for {symbol}'}
            
            self.ib.qualifyContracts(contract)
            
            total_quantity = self._calculate_position_size(symbol, entry_price, stop_price)
            tranche_quantity = max(1, total_quantity // self.dca_tranches)
            
            orders_placed = []
            
            for i in range(self.dca_tranches):
                # Calculate price for this tranche
                if direction == 'LONG':
                    price = entry_price * (1 - (i * self.dca_spread))
                else:
                    price = entry_price * (1 + (i * self.dca_spread))
                
                action = 'BUY' if direction == 'LONG' else 'SELL'
                
                order = LimitOrder(
                    action=action,
                    totalQuantity=tranche_quantity,
                    lmtPrice=round(price, 2),
                )
                
                trade = self.ib.placeOrder(contract, order)
                
                orders_placed.append({
                    'tranche': i + 1,
                    'quantity': tranche_quantity,
                    'price': round(price, 2),
                    'order_id': trade.order.orderId,
                })
            
            result = {
                'status': 'dca_submitted',
                'symbol': symbol,
                'direction': direction,
                'total_quantity': total_quantity,
                'tranches': self.dca_tranches,
                'orders': orders_placed,
                'source': source,
                'mode': 'LIVE_DCA',
                'timestamp': datetime.now().isoformat(),
            }
            
            self.executed_orders.append(result)
            
            logger.info(f"[EXECUTOR] ✅ DCA orders placed: {direction} {total_quantity} {symbol} in {self.dca_tranches} tranches")
            
            return result
            
        except Exception as e:
            logger.error(f"[EXECUTOR] ❌ DCA failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _close_position(self, symbol: str, percent: int) -> Dict:
        """Close a percentage of position."""
        
        try:
            position = self.conn.get_position(symbol)
            
            if position['quantity'] == 0:
                return {'status': 'no_position', 'symbol': symbol}
            
            quantity_to_close = int(abs(position['quantity']) * (percent / 100))
            
            if quantity_to_close == 0:
                return {'status': 'quantity_too_small', 'symbol': symbol}
            
            contract = self.conn.get_contract(symbol, sec_type="AUTO")
            if contract is None:
                return {'status': 'error', 'message': f'Could not create contract for {symbol}'}
            
            self.ib.qualifyContracts(contract)
            
            # Determine action based on current position
            action = 'SELL' if position['quantity'] > 0 else 'BUY'
            
            order = MarketOrder(
                action=action,
                totalQuantity=quantity_to_close,
            )
            
            trade = self.ib.placeOrder(contract, order)
            
            self.ib.sleep(1)
            
            result = {
                'status': 'close_submitted',
                'symbol': symbol,
                'action': action,
                'quantity_closed': quantity_to_close,
                'percent': percent,
                'order_id': trade.order.orderId,
                'order_status': trade.orderStatus.status,
                'timestamp': datetime.now().isoformat(),
            }
            
            self.executed_orders.append(result)
            
            logger.info(f"[EXECUTOR] ✅ Closing {percent}% of {symbol}: {quantity_to_close} units")
            
            return result
            
        except Exception as e:
            logger.error(f"[EXECUTOR] ❌ Close failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
    ) -> int:
        """
        Calculate position size based on risk.
        Uses 1% risk per trade rule.
        """
        
        # Get account value
        try:
            account_value = self.conn.get_account_value()
        except:
            account_value = self.default_account_value
        
        # Maximum risk per trade
        max_risk = account_value * self.risk_per_trade
        
        # If we have valid prices, calculate based on risk
        if entry_price > 0 and stop_price > 0:
            risk_per_share = abs(entry_price - stop_price)
            
            if risk_per_share > 0:
                quantity = int(max_risk / risk_per_share)
            else:
                # Default to position value limit
                quantity = int(self.max_position_value / entry_price) if entry_price > 0 else 100
        else:
            # No prices - use default sizing
            quantity = 100
        
        # Apply limits
        quantity = min(quantity, self.max_position_size)
        quantity = max(quantity, 1)
        
        # Also limit by max position value
        if entry_price > 0:
            max_by_value = int(self.max_position_value / entry_price)
            quantity = min(quantity, max_by_value)
        
        return quantity
    
    def get_pending_orders(self) -> List[Dict]:
        """Get orders pending confirmation."""
        return self.pending_orders
    
    def get_executed_orders(self, limit: int = 50) -> List[Dict]:
        """Get executed orders."""
        return self.executed_orders[-limit:]
    
    def get_failed_orders(self) -> List[Dict]:
        """Get failed orders."""
        return self.failed_orders
    
    def confirm_order(self, order_index: int) -> Dict:
        """Confirm a pending order (for live mode)."""
        
        if order_index >= len(self.pending_orders):
            return {'status': 'error', 'message': 'Invalid order index'}
        
        pending = self.pending_orders.pop(order_index)
        alert = pending['alert']
        
        return self._execute_trade(
            symbol=alert['symbol'],
            direction=alert['direction'],
            entry_price=alert.get('entry', 0),
            stop_price=alert.get('stop', 0),
            source='CONFIRMED',
            alert=alert,
        )
    
    def cancel_pending_order(self, order_index: int) -> Dict:
        """Cancel a pending order."""
        
        if order_index >= len(self.pending_orders):
            return {'status': 'error', 'message': 'Invalid order index'}
        
        cancelled = self.pending_orders.pop(order_index)
        return {'status': 'cancelled', 'order': cancelled}

"""
Paul's BTC Miners - IBKR Sync Service
Mirrors positions: IBKR (primary) → Dev Site (secondary)

Flow:
1. Read positions from Paul's dashboard
2. Sync to IBKR paper account
3. Update dashboard with IBKR fills

Author: QUINN001
Created: 2026-01-28
"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import IB, Stock, LimitOrder, MarketOrder
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaulSync")

# Configuration
PAUL_DASHBOARD_URL = "http://146.190.188.208:3456"
IBKR_HOST = "100.119.161.65"
IBKR_PORT = 7497  # Paper trading


class PaulIBKRSync:
    """
    Sync service between Paul's BTC Miners dashboard and IBKR
    
    Priority: IBKR first, then update dev site
    """
    
    def __init__(self, paper_trading: bool = True):
        self.ib = IB()
        self.paper_trading = paper_trading
        self.connected = False
        
        # Supported symbols
        self.symbols = ['IREN', 'CIFR', 'CLSK']
        
        # Position tracking
        self.paul_positions = {}
        self.ibkr_positions = {}
        
    def connect_ibkr(self, client_id: int = 70) -> bool:
        """Connect to IBKR TWS"""
        try:
            if self.ib.isConnected():
                return True
            
            port = 7497 if self.paper_trading else 7496
            logger.info(f"Connecting to IBKR {'PAPER' if self.paper_trading else 'LIVE'}...")
            
            self.ib.connect(IBKR_HOST, port, clientId=client_id, timeout=30)
            self.ib.sleep(2)
            self.connected = True
            
            logger.info("✅ Connected to IBKR")
            return True
            
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
    
    # =========================================================================
    # PAUL'S DASHBOARD API
    # =========================================================================
    
    def get_paul_positions(self) -> List[Dict]:
        """Get positions from Paul's dashboard"""
        try:
            response = requests.get(
                f"{PAUL_DASHBOARD_URL}/api/positions",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.paul_positions = data
                return data
            else:
                logger.error(f"Paul API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching Paul positions: {e}")
            return []
    
    def update_paul_position(self, symbol: str, fill_data: Dict) -> bool:
        """Update Paul's dashboard with IBKR fill data"""
        try:
            response = requests.post(
                f"{PAUL_DASHBOARD_URL}/api/positions/update",
                json={
                    'symbol': symbol,
                    'ibkr_fill': fill_data,
                    'synced_at': datetime.now().isoformat()
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error updating Paul dashboard: {e}")
            return False
    
    # =========================================================================
    # IBKR OPERATIONS
    # =========================================================================
    
    def get_ibkr_positions(self) -> Dict[str, Dict]:
        """Get current IBKR positions for BTC miners"""
        if not self.connected:
            return {}
        
        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            if symbol in self.symbols:
                positions[symbol] = {
                    'symbol': symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'sec_type': pos.contract.secType
                }
        
        self.ibkr_positions = positions
        return positions
    
    def create_stock_contract(self, symbol: str) -> Stock:
        """Create stock contract"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return contract
    
    def buy_stock(self, symbol: str, quantity: int, limit_price: float = None) -> Dict:
        """Buy stock on IBKR"""
        try:
            contract = self.create_stock_contract(symbol)
            
            if limit_price:
                order = LimitOrder('BUY', quantity, limit_price)
            else:
                order = MarketOrder('BUY', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(2)
            
            return {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'symbol': symbol,
                'quantity': quantity,
                'filled': trade.orderStatus.filled,
                'avg_fill_price': trade.orderStatus.avgFillPrice
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sell_stock(self, symbol: str, quantity: int, limit_price: float = None) -> Dict:
        """Sell stock on IBKR"""
        try:
            contract = self.create_stock_contract(symbol)
            
            if limit_price:
                order = LimitOrder('SELL', quantity, limit_price)
            else:
                order = MarketOrder('SELL', quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(2)
            
            return {
                'success': True,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'symbol': symbol,
                'quantity': quantity
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================
    
    def sync_position(self, symbol: str, target_quantity: int, 
                      target_price: float = None) -> Dict:
        """
        Sync a single position: IBKR first, then Paul dashboard
        
        Args:
            symbol: Stock symbol (IREN, CIFR, CLSK)
            target_quantity: Desired position size
            target_price: Optional limit price
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 SYNCING {symbol}: Target {target_quantity} shares")
        logger.info(f"{'='*50}")
        
        # Get current IBKR position
        ibkr_pos = self.ibkr_positions.get(symbol, {})
        current_qty = ibkr_pos.get('quantity', 0)
        
        logger.info(f"   IBKR current: {current_qty} shares")
        logger.info(f"   Target: {target_quantity} shares")
        
        diff = target_quantity - current_qty
        
        if diff == 0:
            logger.info(f"   ✅ Already synced!")
            return {'status': 'synced', 'quantity': current_qty}
        
        # Step 1: Execute on IBKR FIRST
        if diff > 0:
            # Need to BUY
            logger.info(f"   📈 IBKR: Buying {diff} shares...")
            result = self.buy_stock(symbol, int(diff), target_price)
        else:
            # Need to SELL
            logger.info(f"   📉 IBKR: Selling {abs(diff)} shares...")
            result = self.sell_stock(symbol, int(abs(diff)), target_price)
        
        if result.get('success'):
            logger.info(f"   ✅ IBKR order: {result.get('status')}")
            
            # Step 2: Update Paul's dashboard SECOND
            fill_data = {
                'order_id': result.get('order_id'),
                'action': 'BUY' if diff > 0 else 'SELL',
                'quantity': abs(diff),
                'fill_price': result.get('avg_fill_price'),
                'status': result.get('status'),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"   📤 Updating Paul dashboard...")
            if self.update_paul_position(symbol, fill_data):
                logger.info(f"   ✅ Dashboard updated!")
            else:
                logger.warning(f"   ⚠️ Dashboard update failed (non-critical)")
            
            return {
                'status': 'synced',
                'ibkr_result': result,
                'diff': diff
            }
        else:
            logger.error(f"   ❌ IBKR order failed: {result.get('error')}")
            return {'status': 'failed', 'error': result.get('error')}
    
    def sync_all_positions(self, paul_positions: List[Dict]) -> Dict:
        """
        Sync all positions from Paul's dashboard to IBKR
        
        Args:
            paul_positions: List of positions from Paul's dashboard
        """
        logger.info("\n" + "="*70)
        logger.info("🔄 PAUL → IBKR FULL SYNC")
        logger.info("="*70)
        
        # Refresh IBKR positions
        self.get_ibkr_positions()
        
        results = {}
        
        for pos in paul_positions:
            symbol = pos.get('symbol')
            if symbol not in self.symbols:
                continue
            
            target_qty = pos.get('quantity', pos.get('size', 0))
            target_price = pos.get('entry_price', pos.get('entry'))
            
            result = self.sync_position(symbol, target_qty, target_price)
            results[symbol] = result
        
        return results
    
    def get_sync_status(self) -> Dict:
        """Get current sync status between Paul and IBKR"""
        paul_pos = self.get_paul_positions()
        ibkr_pos = self.get_ibkr_positions()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'paul_positions': {},
            'ibkr_positions': {},
            'sync_needed': []
        }
        
        # Paul positions
        if isinstance(paul_pos, list):
            for p in paul_pos:
                symbol = p.get('symbol')
                if symbol in self.symbols:
                    status['paul_positions'][symbol] = p.get('quantity', p.get('size', 0))
        
        # IBKR positions
        for symbol, p in ibkr_pos.items():
            status['ibkr_positions'][symbol] = p.get('quantity', 0)
        
        # Check differences
        all_symbols = set(status['paul_positions'].keys()) | set(status['ibkr_positions'].keys())
        for symbol in all_symbols:
            paul_qty = status['paul_positions'].get(symbol, 0)
            ibkr_qty = status['ibkr_positions'].get(symbol, 0)
            if paul_qty != ibkr_qty:
                status['sync_needed'].append({
                    'symbol': symbol,
                    'paul': paul_qty,
                    'ibkr': ibkr_qty,
                    'diff': paul_qty - ibkr_qty
                })
        
        return status


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main sync interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paul's BTC Miners IBKR Sync")
    parser.add_argument('--status', action='store_true', help='Show sync status')
    parser.add_argument('--sync', action='store_true', help='Sync all positions')
    parser.add_argument('--symbol', type=str, help='Sync specific symbol')
    parser.add_argument('--quantity', type=int, help='Target quantity for symbol')
    
    args = parser.parse_args()
    
    sync = PaulIBKRSync(paper_trading=True)
    
    if not sync.connect_ibkr(client_id=70):
        print("❌ Failed to connect to IBKR")
        return
    
    try:
        if args.status:
            status = sync.get_sync_status()
            print("\n📊 SYNC STATUS")
            print("="*50)
            print(f"\nPaul's Dashboard:")
            for sym, qty in status['paul_positions'].items():
                print(f"   {sym}: {qty} shares")
            print(f"\nIBKR:")
            for sym, qty in status['ibkr_positions'].items():
                print(f"   {sym}: {qty} shares")
            if status['sync_needed']:
                print(f"\n⚠️ SYNC NEEDED:")
                for s in status['sync_needed']:
                    print(f"   {s['symbol']}: Paul={s['paul']}, IBKR={s['ibkr']}, Diff={s['diff']}")
            else:
                print(f"\n✅ All positions synced!")
        
        elif args.sync:
            paul_pos = sync.get_paul_positions()
            results = sync.sync_all_positions(paul_pos)
            print("\n📊 SYNC RESULTS")
            for sym, res in results.items():
                print(f"   {sym}: {res.get('status')}")
        
        elif args.symbol and args.quantity is not None:
            result = sync.sync_position(args.symbol, args.quantity)
            print(f"\n{args.symbol}: {result}")
        
        else:
            # Default: show status
            status = sync.get_sync_status()
            print(json.dumps(status, indent=2))
    
    finally:
        sync.disconnect()


if __name__ == '__main__':
    main()

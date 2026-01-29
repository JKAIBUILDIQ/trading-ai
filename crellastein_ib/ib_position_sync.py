#!/usr/bin/env python3
"""
IB Position Sync - Fetches live positions from Interactive Brokers
and writes to ib_live_positions.json for the tAIq dashboard.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from ib_insync import IB, Future
import nest_asyncio

nest_asyncio.apply()

# Output file for tAIq dashboard
OUTPUT_FILE = Path(__file__).parent / 'ib_live_positions.json'
PRICE_CACHE = Path(__file__).parent / 'price_cache.json'
STATE_FILE = Path(__file__).parent / 'whipsaw_state.json'

# IB Connection settings
IB_HOST = '100.119.161.65'  # Desktop TWS via Tailscale
IB_PORT = 7497
CLIENT_ID = 897  # Unique client ID for position sync

# MGC Contract
MGC_CONID = 706903676


def get_magic_and_comment(avg_cost: float, position: float, state: dict) -> tuple:
    """Determine magic number and comment based on position and state."""
    if position > 0:
        # Long position - check if it matches a buy level
        for i, level in enumerate(state.get('buy_levels', [])):
            if level.get('filled') and abs(level.get('fill_price', level['price']) - avg_cost) < 5:
                return '8880202', f"DROPBUY|L{i+1}|{abs(position)}"
        return '8880777', f"GRID|LONG|{avg_cost:.0f}"
    else:
        # Short position
        for i, level in enumerate(state.get('short_levels', [])):
            if level.get('filled') and abs(level.get('fill_price', level['price']) - avg_cost) < 5:
                return '8880333', f"SHORT|S{i+1}|{abs(position)}"
        return '8880999', f"CORRECTION|HEDGE|{avg_cost:.0f}"


def fetch_ib_positions():
    """Fetch positions from IB and return formatted data."""
    ib = IB()
    
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to TWS at {IB_HOST}:{IB_PORT}...")
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15)
        
        # Get MGC contract
        contract = Future(conId=MGC_CONID, symbol='MGC', exchange='COMEX')
        
        # Request market data
        ib.reqMarketDataType(3)  # Delayed data
        ticker = ib.reqMktData(contract)
        ib.sleep(2)
        
        current_price = ticker.last or ticker.close or 5580
        
        # Get all positions
        positions = ib.positions()
        
        # Get open orders for SL/TP
        open_orders = ib.openOrders()
        
        # Load state for magic/comment mapping
        state = {}
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
        
        # Format positions
        formatted_positions = []
        
        for pos in positions:
            if pos.contract.symbol == 'MGC':
                avg_cost = pos.avgCost / 10  # MGC avgCost is per contract * 10
                position_qty = int(pos.position)
                
                if position_qty == 0:
                    continue
                
                # Determine type
                pos_type = 'BUY' if position_qty > 0 else 'SELL'
                volume = abs(position_qty)
                
                # Calculate P&L
                if pos_type == 'BUY':
                    profit = (current_price - avg_cost) * volume * 10  # $10 per point
                else:
                    profit = (avg_cost - current_price) * volume * 10
                
                # Get SL/TP from open orders
                sl = None
                tp = None
                for order in open_orders:
                    if hasattr(order, 'contract') and order.contract.symbol == 'MGC':
                        # Check if it's a stop or limit order related to this position
                        if order.orderType == 'STP':
                            sl = order.auxPrice
                        elif order.orderType == 'LMT':
                            if pos_type == 'BUY' and order.action == 'SELL' and order.lmtPrice > avg_cost:
                                tp = order.lmtPrice
                            elif pos_type == 'SELL' and order.action == 'BUY' and order.lmtPrice < avg_cost:
                                tp = order.lmtPrice
                
                # Default SL/TP based on strategy and mode
                grid_mode = state.get('grid_mode', 'BULLISH')
                
                if sl is None:
                    if grid_mode == 'BULLISH' and pos_type == 'BUY':
                        sl = None  # No SL for bullish DCA - we average down
                    elif pos_type == 'BUY':
                        sl = avg_cost - 50  # Wider SL for correction mode
                    else:
                        sl = avg_cost + 30  # Short positions get SL
                
                if tp is None:
                    tp = avg_cost + 50 if pos_type == 'BUY' else avg_cost - 100
                
                # Get magic and comment
                magic, comment = get_magic_and_comment(avg_cost, position_qty, state)
                
                formatted_positions.append({
                    'ticket': f"IB{pos.contract.conId}",
                    'symbol': 'mgc',
                    'time': datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
                    'type': pos_type,
                    'volume': volume,
                    'entry_price': round(avg_cost, 2),
                    'current_price': round(current_price, 2),
                    'sl': round(sl, 2) if sl else None,
                    'tp': round(tp, 2) if tp else None,
                    'profit': round(profit, 2),
                    'magic': magic,
                    'comment': comment,
                })
        
        # Also get pending/working orders
        for order in open_orders:
            if hasattr(order, 'contract') and order.contract.symbol == 'MGC':
                if order.orderType in ['LMT', 'STP', 'STP LMT']:
                    # This is a working order, not a position
                    pass  # Could add to a separate orders list
        
        ib.disconnect()
        
        return {
            'positions': formatted_positions,
            'current_price': round(current_price, 2),
            'updated': datetime.now().isoformat(),
            'connection': 'OK',
        }
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
        if ib.isConnected():
            ib.disconnect()
        
        # Return cached data with error flag
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE) as f:
                data = json.load(f)
                data['connection'] = f'ERROR: {str(e)}'
                return data
        
        return {
            'positions': [],
            'current_price': 5580,
            'updated': datetime.now().isoformat(),
            'connection': f'ERROR: {str(e)}',
        }


def sync_once():
    """Single sync operation."""
    data = fetch_ib_positions()
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also update price cache
    with open(PRICE_CACHE, 'w') as f:
        json.dump({'price': data['current_price'], 'updated': data['updated']}, f)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Synced {len(data['positions'])} positions @ ${data['current_price']}")
    return data


def run_continuous(interval: int = 10):
    """Run continuous sync loop."""
    print(f"Starting IB Position Sync (interval: {interval}s)")
    print(f"Output: {OUTPUT_FILE}")
    
    while True:
        try:
            sync_once()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sync error: {e}")
        
        time.sleep(interval)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Single sync
        data = sync_once()
        print(json.dumps(data, indent=2))
    else:
        # Continuous sync
        run_continuous(interval=10)

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    DESKTOP BACKUP EXECUTOR
                    Run this on your Windows desktop when H100 can't connect
═══════════════════════════════════════════════════════════════════════════════

This script connects to TWS locally (127.0.0.1) and executes pending commands.

Usage:
    python execute_local.py              # Execute all pending commands
    python execute_local.py status       # Check position
    python execute_local.py sell 2       # Sell 2 MGC contracts
    python execute_local.py buy 2        # Buy 2 MGC contracts

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from ib_insync import IB, Future, MarketOrder
except ImportError:
    print("❌ ib_insync not installed. Run: pip install ib_insync")
    sys.exit(1)

PENDING_FILE = Path(__file__).parent / 'pending_commands.json'

# MGC Contract specs
MGC_CONTRACT = Future(
    conId=706903676,
    symbol='MGC',
    lastTradeDateOrContractMonth='20260428',
    exchange='COMEX',
    multiplier='10',
    currency='USD',
    localSymbol='MGCJ6',
    tradingClass='MGC',
)


def connect_local():
    """Connect to TWS locally"""
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=999, timeout=10)
        print("✅ Connected to TWS (localhost)")
        return ib
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None


def get_position(ib):
    """Get current MGC position"""
    positions = ib.positions()
    for pos in positions:
        if pos.contract.symbol == 'MGC':
            return {
                'position': pos.position,
                'avg_cost': pos.avgCost / 10,
            }
    return {'position': 0, 'avg_cost': 0}


def execute_order(ib, action: str, quantity: int):
    """Execute a market order"""
    order = MarketOrder(action.upper(), quantity)
    trade = ib.placeOrder(MGC_CONTRACT, order)
    
    print(f"📤 Placing order: {action.upper()} {quantity} MGC @ MARKET")
    ib.sleep(5)
    
    if trade.orderStatus.status == 'Filled':
        print(f"✅ FILLED @ ${trade.orderStatus.avgFillPrice}")
        return True
    else:
        print(f"⚠️ Order status: {trade.orderStatus.status}")
        return False


def execute_pending():
    """Execute all pending commands from H100"""
    if not PENDING_FILE.exists():
        print("No pending commands")
        return
    
    with open(PENDING_FILE, 'r') as f:
        data = json.load(f)
    
    pending = [c for c in data.get('commands', []) if c.get('status') == 'PENDING']
    
    if not pending:
        print("No pending commands to execute")
        return
    
    ib = connect_local()
    if not ib:
        return
    
    for cmd in pending:
        print(f"\n📋 Executing: {cmd['action']} {cmd['quantity']} MGC")
        print(f"   Note: {cmd.get('note', '')}")
        
        success = execute_order(ib, cmd['action'], cmd['quantity'])
        
        if success:
            cmd['status'] = 'EXECUTED'
            cmd['executed_at'] = datetime.now().isoformat()
    
    # Update file
    with open(PENDING_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Show final position
    pos = get_position(ib)
    print(f"\n📊 Final Position: {pos['position']} contracts @ ${pos['avg_cost']:.2f}")
    
    ib.disconnect()


def show_status():
    """Show current position"""
    ib = connect_local()
    if not ib:
        return
    
    pos = get_position(ib)
    print(f"\n📊 MGC Position: {pos['position']} contracts @ ${pos['avg_cost']:.2f}")
    ib.disconnect()


def quick_trade(action: str, quantity: int):
    """Execute a quick trade"""
    ib = connect_local()
    if not ib:
        return
    
    execute_order(ib, action, quantity)
    
    pos = get_position(ib)
    print(f"\n📊 New Position: {pos['position']} contracts @ ${pos['avg_cost']:.2f}")
    ib.disconnect()


def main():
    if len(sys.argv) < 2:
        execute_pending()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'status':
        show_status()
    elif cmd == 'sell' and len(sys.argv) >= 3:
        quick_trade('SELL', int(sys.argv[2]))
    elif cmd == 'buy' and len(sys.argv) >= 3:
        quick_trade('BUY', int(sys.argv[2]))
    elif cmd == 'pending':
        execute_pending()
    else:
        print("Usage:")
        print("  python execute_local.py              # Execute pending commands")
        print("  python execute_local.py status       # Check position")
        print("  python execute_local.py sell 2       # Sell 2 contracts")
        print("  python execute_local.py buy 2        # Buy 2 contracts")


if __name__ == "__main__":
    main()

"""
MGC Scale-Out Bot
Sells contracts at 10-point intervals to lock in profits.

Usage:
    python scale_out_mgc.py --start-price 5442 --contracts-per-level 4 --levels 5
    
Or just run it and it will auto-detect current price.
"""

import asyncio
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

try:
    from ib_insync import IB, Future, MarketOrder, LimitOrder
    HAS_IB = True
except ImportError:
    HAS_IB = False
    print("Warning: ib_insync not installed")

# Telegram notification
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = "/home/jbot/trading_ai/crellastein_ib/scale_out_state.json"


@dataclass
class ScaleOutLevel:
    level: int
    target_price: float
    contracts: int
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[str] = None


@dataclass 
class ScaleOutState:
    symbol: str
    start_price: float
    interval: float
    total_contracts_to_sell: int
    contracts_per_level: int
    levels: list  # List of ScaleOutLevel
    created_at: str
    active: bool = True
    

def send_telegram(message: str):
    """Send Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram disabled] {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")


def load_state() -> Optional[ScaleOutState]:
    """Load state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
            levels = [ScaleOutLevel(**l) for l in data.pop('levels')]
            return ScaleOutState(**data, levels=levels)
    return None


def save_state(state: ScaleOutState):
    """Save state to file."""
    data = asdict(state)
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def create_scale_out_plan(
    start_price: float,
    total_contracts: int,
    percent_to_sell: float = 0.5,
    interval: float = 10.0,
    num_levels: int = 5
) -> ScaleOutState:
    """
    Create a scale-out plan.
    
    Args:
        start_price: Current/starting price
        total_contracts: Total contracts held
        percent_to_sell: Percentage to scale out (0.5 = 50%)
        interval: Points between each sell level
        num_levels: Number of levels to spread sells across
    """
    contracts_to_sell = int(total_contracts * percent_to_sell)
    base_per_level = contracts_to_sell // num_levels
    remainder = contracts_to_sell % num_levels
    
    levels = []
    for i in range(num_levels):
        # Distribute remainder to early levels
        contracts = base_per_level + (1 if i < remainder else 0)
        if contracts > 0:
            levels.append(ScaleOutLevel(
                level=i,
                target_price=start_price + (i * interval),
                contracts=contracts
            ))
    
    state = ScaleOutState(
        symbol="MGC",
        start_price=start_price,
        interval=interval,
        total_contracts_to_sell=contracts_to_sell,
        contracts_per_level=base_per_level,
        levels=levels,
        created_at=datetime.now().isoformat(),
        active=True
    )
    
    return state


def print_plan(state: ScaleOutState):
    """Print the scale-out plan."""
    print("\n" + "="*60)
    print("📊 MGC SCALE-OUT PLAN")
    print("="*60)
    print(f"Start Price: ${state.start_price:.2f}")
    print(f"Interval: {state.interval} points")
    print(f"Total to Sell: {state.total_contracts_to_sell} contracts (50%)")
    print("-"*60)
    print(f"{'Level':<8} {'Price':<12} {'Contracts':<12} {'Status':<15}")
    print("-"*60)
    
    running_total = 0
    for level in state.levels:
        running_total += level.contracts
        status = "✅ FILLED" if level.filled else "⏳ Pending"
        if level.filled and level.fill_price:
            status = f"✅ @ ${level.fill_price:.2f}"
        print(f"{level.level:<8} ${level.target_price:<11.2f} {level.contracts:<12} {status:<15}")
    
    print("-"*60)
    print(f"{'TOTAL':<8} {'':<12} {running_total:<12}")
    print("="*60 + "\n")


async def place_limit_orders(state: ScaleOutState, ib: IB):
    """Place all limit orders at once."""
    contract = Future(
        symbol='MGC',
        exchange='COMEX',
        lastTradeDateOrContractMonth='202604'
    )
    await ib.qualifyContractsAsync(contract)
    
    orders_placed = []
    
    for level in state.levels:
        if not level.filled:
            order = LimitOrder(
                action='SELL',
                totalQuantity=level.contracts,
                lmtPrice=level.target_price,
                tif='GTC'  # Good til cancelled
            )
            trade = ib.placeOrder(contract, order)
            orders_placed.append({
                "level": level.level,
                "price": level.target_price,
                "contracts": level.contracts,
                "order_id": trade.order.orderId
            })
            print(f"📤 Placed SELL {level.contracts} @ ${level.target_price:.2f}")
    
    return orders_placed


async def run_monitor_mode(state: ScaleOutState):
    """Monitor price and execute sells as levels are hit."""
    if not HAS_IB:
        print("Error: ib_insync required for monitor mode")
        return
    
    ib = IB()
    
    try:
        await ib.connectAsync('100.119.161.65', 7497, clientId=20)
        print("✅ Connected to IB for scale-out monitoring")
        
        contract = Future(
            symbol='MGC',
            exchange='COMEX',
            lastTradeDateOrContractMonth='202604'
        )
        await ib.qualifyContractsAsync(contract)
        
        # Subscribe to market data
        ib.reqMktData(contract, '', False, False)
        
        print(f"📡 Monitoring {contract.localSymbol} for scale-out levels...")
        send_telegram(f"🎯 MGC Scale-Out Active\nLevels: {len(state.levels)}\nFirst target: ${state.levels[0].target_price:.2f}")
        
        while state.active:
            await asyncio.sleep(5)
            
            ticker = ib.ticker(contract)
            if ticker and ticker.last:
                current_price = ticker.last
                
                for level in state.levels:
                    if not level.filled and current_price >= level.target_price:
                        # Execute sell
                        order = MarketOrder('SELL', level.contracts)
                        trade = ib.placeOrder(contract, order)
                        
                        # Wait for fill
                        while not trade.isDone():
                            await asyncio.sleep(0.5)
                        
                        if trade.orderStatus.status == 'Filled':
                            level.filled = True
                            level.fill_price = trade.orderStatus.avgFillPrice
                            level.fill_time = datetime.now().isoformat()
                            
                            msg = f"✅ SCALE-OUT Level {level.level}\nSOLD {level.contracts} MGC @ ${level.fill_price:.2f}\nTarget was ${level.target_price:.2f}"
                            print(msg)
                            send_telegram(msg)
                            
                            save_state(state)
                
                # Check if all levels filled
                if all(l.filled for l in state.levels):
                    msg = f"🎉 SCALE-OUT COMPLETE\nAll {state.total_contracts_to_sell} contracts sold"
                    print(msg)
                    send_telegram(msg)
                    state.active = False
                    save_state(state)
    
    except Exception as e:
        print(f"Error: {e}")
        send_telegram(f"❌ Scale-out error: {e}")
    finally:
        ib.disconnect()


async def place_all_limits():
    """Connect to IB and place all limit orders."""
    state = load_state()
    if not state:
        print("No state file found. Create plan first.")
        return
    
    if not HAS_IB:
        print("Error: ib_insync required")
        return
    
    ib = IB()
    try:
        await ib.connectAsync('100.119.161.65', 7497, clientId=21)
        print("✅ Connected to IB")
        
        orders = await place_limit_orders(state, ib)
        print(f"\n📋 Placed {len(orders)} limit orders")
        
        send_telegram(f"📊 MGC Scale-Out Orders Placed\n{len(orders)} GTC limit sells")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ib.disconnect()


def main():
    parser = argparse.ArgumentParser(description='MGC Scale-Out Bot')
    parser.add_argument('command', choices=['plan', 'limits', 'monitor', 'status'],
                        help='Command to run')
    parser.add_argument('--start-price', type=float, default=5442,
                        help='Starting price for scale-out')
    parser.add_argument('--total-contracts', type=int, default=36,
                        help='Total contracts held')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='Percentage to scale out (0.5 = 50%%)')
    parser.add_argument('--interval', type=float, default=10,
                        help='Points between each level')
    parser.add_argument('--levels', type=int, default=5,
                        help='Number of scale-out levels')
    
    args = parser.parse_args()
    
    if args.command == 'plan':
        state = create_scale_out_plan(
            start_price=args.start_price,
            total_contracts=args.total_contracts,
            percent_to_sell=args.percent,
            interval=args.interval,
            num_levels=args.levels
        )
        print_plan(state)
        save_state(state)
        print(f"💾 Plan saved to {STATE_FILE}")
        
    elif args.command == 'limits':
        asyncio.run(place_all_limits())
        
    elif args.command == 'monitor':
        state = load_state()
        if state:
            print_plan(state)
            asyncio.run(run_monitor_mode(state))
        else:
            print("No state file. Run 'plan' first.")
            
    elif args.command == 'status':
        state = load_state()
        if state:
            print_plan(state)
        else:
            print("No active scale-out plan.")


if __name__ == "__main__":
    main()

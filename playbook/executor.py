"""
PLAYBOOK EXECUTOR
Executes today's approved playbook. No independent decisions.

Rules:
1. Only trades from approved playbook
2. Respects cooldowns
3. Respects max daily trades
4. Stops at market close
5. Alerts on stop levels, doesn't auto-close
"""

import json
import asyncio
import os
from datetime import datetime, date, time
from typing import Dict, Optional
import pytz
from dataclasses import dataclass

# IB connection (optional import if running standalone)
try:
    from ib_insync import IB, Future, Stock, Option, MarketOrder, LimitOrder
    HAS_IB = True
except ImportError:
    HAS_IB = False
    print("⚠️  ib_insync not available - running in simulation mode")


@dataclass
class ExecutionState:
    """Track today's execution state."""
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None
    filled_entries: Dict[str, list] = None
    filled_tps: Dict[str, list] = None
    alerts_sent: Dict[str, list] = None
    
    def __post_init__(self):
        self.filled_entries = self.filled_entries or {}
        self.filled_tps = self.filled_tps or {}
        self.alerts_sent = self.alerts_sent or {}


class PlaybookExecutor:
    """Executes trades based on daily playbook."""
    
    def __init__(self, playbook_path: str = None):
        self.et = pytz.timezone('US/Eastern')
        self.playbook = None
        self.state = ExecutionState()
        self.ib = None
        self.running = False
        
        # Load playbook
        if playbook_path:
            self.load_playbook(playbook_path)
        else:
            # Load today's playbook
            today = date.today().isoformat()
            default_path = f"/home/jbot/trading_ai/playbook/daily/playbook_{today}.json"
            if os.path.exists(default_path):
                self.load_playbook(default_path)
    
    def load_playbook(self, path: str):
        """Load playbook from file."""
        with open(path, 'r') as f:
            self.playbook = json.load(f)
        print(f"📋 Loaded playbook: {path}")
        print(f"   Date: {self.playbook['date']}")
        print(f"   Approved: {self.playbook['approved']}")
        print(f"   Symbols: {list(self.playbook['symbols'].keys())}")
    
    def approve_playbook(self):
        """Mark playbook as approved (would be called via API/Telegram)."""
        if self.playbook:
            self.playbook['approved'] = True
            self.playbook['approved_at'] = datetime.now().isoformat()
            # Save back
            today = date.today().isoformat()
            path = f"/home/jbot/trading_ai/playbook/daily/playbook_{today}.json"
            with open(path, 'w') as f:
                json.dump(self.playbook, f, indent=2)
            print("✅ Playbook APPROVED")
            return True
        return False
    
    def is_market_hours(self) -> bool:
        """Check if within trading hours."""
        now = datetime.now(self.et)
        
        # Check day of week (0=Mon, 6=Sun)
        if now.weekday() >= 5:
            return False
        
        hours = self.playbook.get('trading_hours', {})
        start = time(9, 30)  # Default 9:30am
        end = time(16, 0)    # Default 4:00pm
        
        if hours.get('start'):
            h, m = map(int, hours['start'].split(':'))
            start = time(h, m)
        if hours.get('end'):
            h, m = map(int, hours['end'].split(':'))
            end = time(h, m)
        
        current = now.time()
        return start <= current <= end
    
    def check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        if self.state.last_trade_time is None:
            return True
        
        cooldown = self.playbook.get('cooldown_minutes', 30)
        elapsed = (datetime.now() - self.state.last_trade_time).total_seconds() / 60
        return elapsed >= cooldown
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if not self.playbook:
            return False, "No playbook loaded"
        
        if not self.playbook.get('approved'):
            return False, "Playbook not approved"
        
        if not self.is_market_hours():
            return False, "Outside market hours"
        
        max_trades = self.playbook.get('max_daily_trades', 10)
        if self.state.trades_today >= max_trades:
            return False, f"Max daily trades ({max_trades}) reached"
        
        if not self.check_cooldown():
            cooldown = self.playbook.get('cooldown_minutes', 30)
            return False, f"Cooldown active ({cooldown} min)"
        
        return True, "OK"
    
    async def connect_ib(self) -> bool:
        """Connect to IB."""
        if not HAS_IB:
            print("⚠️  Running in simulation mode (no IB)")
            return False
        
        self.ib = IB()
        hosts = ['127.0.0.1', '100.119.161.65']
        
        for host in hosts:
            try:
                await self.ib.connectAsync(host, 7497, clientId=500)
                if self.ib.isConnected():
                    print(f"✅ Connected to IB at {host}")
                    return True
            except Exception as e:
                print(f"   Failed {host}: {e}")
        
        print("❌ Could not connect to IB")
        return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        if not self.ib or not self.ib.isConnected():
            return None
        
        try:
            if symbol == "MGC":
                contract = Future(conId=706903676, symbol='MGC', exchange='COMEX')
            else:
                contract = Stock(symbol, 'SMART', 'USD')
            
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)
            
            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)
            return price if price > 0 else None
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def check_entry_triggers(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if any entry levels are triggered."""
        if symbol not in self.playbook['symbols']:
            return None
        
        sym_playbook = self.playbook['symbols'][symbol]
        entries = sym_playbook.get('entry_levels', [])
        
        filled = self.state.filled_entries.get(symbol, [])
        
        for entry in entries:
            entry_price = entry['price']
            # Check if triggered (price at or below entry level for buys)
            if current_price <= entry_price:
                # Check if not already filled
                if entry_price not in filled:
                    return entry
        
        return None
    
    def check_tp_triggers(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if any TP levels are triggered."""
        if symbol not in self.playbook['symbols']:
            return None
        
        sym_playbook = self.playbook['symbols'][symbol]
        tps = sym_playbook.get('tp_levels', [])
        
        filled = self.state.filled_tps.get(symbol, [])
        
        for tp in tps:
            tp_price = tp['price']
            # Check if triggered (price at or above TP level for sells)
            if current_price >= tp_price:
                # Check if not already filled
                if tp_price not in filled:
                    return tp
        
        return None
    
    def check_stop_triggers(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if stop level is triggered."""
        if symbol not in self.playbook['symbols']:
            return None
        
        sym_playbook = self.playbook['symbols'][symbol]
        stop_price = sym_playbook.get('stop_price')
        stop_action = sym_playbook.get('stop_action', 'ALERT_ONLY')
        
        if stop_price and current_price <= stop_price:
            sent = self.state.alerts_sent.get(symbol, [])
            if 'stop' not in sent:
                return {
                    'price': stop_price,
                    'action': stop_action,
                    'current': current_price
                }
        
        return None
    
    async def execute_trade(self, symbol: str, action: str, quantity: int, note: str = "") -> bool:
        """Execute a trade."""
        can, reason = self.can_trade()
        if not can:
            print(f"⚠️  Cannot trade: {reason}")
            return False
        
        print(f"\n🔔 EXECUTING: {action} {quantity} {symbol}")
        print(f"   Note: {note}")
        
        if not self.ib or not self.ib.isConnected():
            print(f"   [SIMULATION] Would {action} {quantity} {symbol}")
            self.state.trades_today += 1
            self.state.last_trade_time = datetime.now()
            return True
        
        try:
            # Get contract
            if symbol == "MGC":
                contract = Future(conId=706903676, symbol='MGC', exchange='COMEX')
            else:
                contract = Stock(symbol, 'SMART', 'USD')
            
            self.ib.qualifyContracts(contract)
            
            # Create order
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for fill
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1
            
            if trade.orderStatus.status == 'Filled':
                print(f"   ✅ FILLED @ ${trade.orderStatus.avgFillPrice:.2f}")
                self.state.trades_today += 1
                self.state.last_trade_time = datetime.now()
                return True
            else:
                print(f"   ❌ Order status: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def send_alert(self, symbol: str, alert_type: str, message: str):
        """Send alert (Telegram, etc)."""
        print(f"\n🚨 ALERT [{symbol}] {alert_type}: {message}")
        
        # Track sent alerts
        if symbol not in self.state.alerts_sent:
            self.state.alerts_sent[symbol] = []
        self.state.alerts_sent[symbol].append(alert_type)
        
        # TODO: Send to Telegram
        # send_telegram_message(f"🚨 {symbol} {alert_type}: {message}")
    
    async def run_cycle(self):
        """Run one monitoring cycle."""
        if not self.playbook:
            return
        
        can, reason = self.can_trade()
        if not can and "approved" in reason.lower():
            print(f"⏸️  {reason}")
            return
        
        for symbol, sym_playbook in self.playbook['symbols'].items():
            # Get current price
            price = self.get_current_price(symbol)
            if not price:
                continue
            
            print(f"📊 {symbol}: ${price:.2f}")
            
            # Check stop triggers (alert only)
            stop = self.check_stop_triggers(symbol, price)
            if stop:
                self.send_alert(
                    symbol, 
                    'STOP', 
                    f"Price ${price:.2f} hit stop ${stop['price']:.2f}. Action: {stop['action']}"
                )
                if stop['action'] == 'CLOSE':
                    # Would execute close here if not ALERT_ONLY
                    pass
            
            # Only trade if allowed
            can, reason = self.can_trade()
            if not can:
                continue
            
            # Check entry triggers
            entry = self.check_entry_triggers(symbol, price)
            if entry:
                success = await self.execute_trade(
                    symbol,
                    'BUY',
                    entry['quantity'],
                    entry.get('note', '')
                )
                if success:
                    if symbol not in self.state.filled_entries:
                        self.state.filled_entries[symbol] = []
                    self.state.filled_entries[symbol].append(entry['price'])
            
            # Check TP triggers
            tp = self.check_tp_triggers(symbol, price)
            if tp:
                qty = tp['quantity'] if isinstance(tp['quantity'], int) else sym_playbook['current_position']
                success = await self.execute_trade(
                    symbol,
                    'SELL',
                    qty,
                    tp.get('note', '')
                )
                if success:
                    if symbol not in self.state.filled_tps:
                        self.state.filled_tps[symbol] = []
                    self.state.filled_tps[symbol].append(tp['price'])
    
    async def run(self, interval_seconds: int = 300):
        """Main run loop - checks every interval."""
        print("\n" + "="*60)
        print("PLAYBOOK EXECUTOR STARTED")
        print("="*60)
        
        if not self.playbook:
            print("❌ No playbook loaded")
            return
        
        if not self.playbook.get('approved'):
            print("⚠️  Playbook not approved - waiting for approval...")
        
        # Connect to IB
        await self.connect_ib()
        
        self.running = True
        
        while self.running:
            try:
                await self.run_cycle()
            except Exception as e:
                print(f"❌ Cycle error: {e}")
            
            # Wait for next cycle
            print(f"\n⏳ Next check in {interval_seconds}s...")
            await asyncio.sleep(interval_seconds)
            
            # Check if market closed
            if not self.is_market_hours():
                print("\n🔔 Market closed - stopping executor")
                break
        
        # Cleanup
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        
        print("\n" + "="*60)
        print("EXECUTOR STOPPED")
        print(f"Trades today: {self.state.trades_today}")
        print("="*60)
    
    def stop(self):
        """Stop the executor."""
        self.running = False


def print_playbook_summary(playbook: Dict):
    """Print human-readable playbook summary."""
    print("\n" + "="*60)
    print(f"📋 DAILY PLAYBOOK - {playbook['date']}")
    print("="*60)
    print(f"Status: {'✅ APPROVED' if playbook['approved'] else '⏳ PENDING APPROVAL'}")
    print(f"Context: {playbook['market_context']}")
    print(f"Max Trades: {playbook['max_daily_trades']} | Cooldown: {playbook['cooldown_minutes']}min")
    print("-"*60)
    
    for symbol, sp in playbook['symbols'].items():
        print(f"\n📌 {symbol}")
        print(f"   Bias: {sp['bias']} ({sp['confidence']} confidence)")
        print(f"   Action: {sp['primary_action']}")
        print(f"   Thesis: {sp['thesis'][:80]}...")
        
        if sp['entry_levels']:
            print(f"   Entry Levels:")
            for e in sp['entry_levels']:
                print(f"      ${e['price']:,.2f} → {e['action']} {e['quantity']} ({e['note']})")
        
        if sp['tp_levels']:
            print(f"   TP Levels:")
            for t in sp['tp_levels']:
                qty = t['quantity'] if isinstance(t['quantity'], int) else 'ALL'
                print(f"      ${t['price']:,.2f} → {t['action']} {qty} ({t['note']})")
        
        if sp['stop_price']:
            print(f"   Stop: ${sp['stop_price']:,.2f} ({sp['stop_action']})")
        
        print(f"   Invalidation: {sp['invalidation']}")


if __name__ == "__main__":
    import sys
    
    # Load today's playbook
    today = date.today().isoformat()
    path = f"/home/jbot/trading_ai/playbook/daily/playbook_{today}.json"
    
    if not os.path.exists(path):
        print(f"❌ No playbook found for {today}")
        print("Run: python generator.py first")
        sys.exit(1)
    
    with open(path, 'r') as f:
        playbook = json.load(f)
    
    # Print summary
    print_playbook_summary(playbook)
    
    # Check for command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "approve":
            executor = PlaybookExecutor(path)
            executor.approve_playbook()
        elif sys.argv[1] == "run":
            executor = PlaybookExecutor(path)
            if not playbook['approved']:
                print("\n⚠️  Playbook not approved! Run: python executor.py approve")
            else:
                asyncio.run(executor.run(interval_seconds=300))
    else:
        print("\n" + "-"*60)
        print("Commands:")
        print("  python executor.py approve  - Approve today's playbook")
        print("  python executor.py run      - Run the executor")

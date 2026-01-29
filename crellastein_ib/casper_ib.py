#!/usr/bin/env python3
"""
Casper IB - Virtual Position Tracking for DCA on Interactive Brokers

IB aggregates all buys into one position with average cost.
This module tracks "virtual positions" so we can apply Casper's
per-entry TP logic even on IB's aggregated system.

Author: Quinn (NEO Training)
"""

import json
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from ib_insync import IB, Future, MarketOrder
import nest_asyncio

nest_asyncio.apply()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VirtualEntry:
    """Represents one DCA ladder entry (virtual position)"""
    level: int                          # L1, L2, L3, etc.
    entry_price: float                  # Actual fill price
    entry_time: str                     # ISO format timestamp
    quantity: int                       # Contracts (usually 1)
    tp_price: float                     # Target price to close this entry
    status: str                         # 'open', 'tp_pending', 'closed'
    close_price: Optional[float] = None
    close_time: Optional[str] = None
    pnl: Optional[float] = None
    magic: str = "8880202"              # DROPBUY magic number
    comment: str = ""


class DCALadder:
    """
    Manages virtual position tracking for IB DCA.
    
    Each DCA buy creates a VirtualEntry with its own TP.
    When price hits a virtual TP, we sell 1 contract on IB.
    IB doesn't know which "virtual position" we're closing - we track it.
    """
    
    def __init__(self, symbol: str, tp_points: float = 50.0, state_file: str = None):
        self.symbol = symbol
        self.tp_points = tp_points  # $50 per entry
        self.entries: List[VirtualEntry] = []
        self.total_quantity = 0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        
        # Persistence
        self.state_file = Path(state_file) if state_file else Path(__file__).parent / 'casper_dca_state.json'
        self._load_state()
    
    def _load_state(self):
        """Load state from disk on startup"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.realized_pnl = data.get('realized_pnl', 0.0)
                for entry_data in data.get('entries', []):
                    entry = VirtualEntry(**entry_data)
                    self.entries.append(entry)
                self._recalc_average()
                print(f"📂 Loaded {len(self.entries)} virtual entries from {self.state_file}")
            except Exception as e:
                print(f"⚠️ Error loading state: {e}")
    
    def _save_state(self):
        """Save state to disk after changes"""
        data = {
            'symbol': self.symbol,
            'tp_points': self.tp_points,
            'realized_pnl': self.realized_pnl,
            'total_quantity': self.total_quantity,
            'avg_price': self.avg_price,
            'updated': datetime.now().isoformat(),
            'entries': [asdict(e) for e in self.entries]
        }
        self.state_file.write_text(json.dumps(data, indent=2))
    
    def _recalc_average(self):
        """Recalculate average price (mirrors IB's view)"""
        open_entries = [e for e in self.entries if e.status == 'open']
        if not open_entries:
            self.total_quantity = 0
            self.avg_price = 0.0
            return
        
        total_cost = sum(e.entry_price * e.quantity for e in open_entries)
        self.total_quantity = sum(e.quantity for e in open_entries)
        self.avg_price = total_cost / self.total_quantity if self.total_quantity > 0 else 0
    
    def add_entry(self, price: float, quantity: int = 1) -> VirtualEntry:
        """Record a new DCA buy"""
        level = len(self.entries) + 1
        entry = VirtualEntry(
            level=level,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            quantity=quantity,
            tp_price=price + self.tp_points,  # Each entry has its own TP
            status='open',
            comment=f"DROPBUY|L{level}|{quantity}"
        )
        self.entries.append(entry)
        self._recalc_average()
        self._save_state()
        print(f"📈 Added L{level} @ ${price:.2f} | TP: ${entry.tp_price:.2f} | Total: {self.total_quantity} contracts")
        return entry
    
    def check_tps(self, current_price: float) -> List[VirtualEntry]:
        """Check which entries have hit their TP"""
        ready_to_close = []
        for entry in self.entries:
            if entry.status == 'open' and current_price >= entry.tp_price:
                entry.status = 'tp_pending'
                ready_to_close.append(entry)
                print(f"🎯 L{entry.level} TP HIT! Price ${current_price:.2f} >= TP ${entry.tp_price:.2f}")
        
        if ready_to_close:
            self._save_state()
        return ready_to_close
    
    def close_entry(self, entry: VirtualEntry, close_price: float):
        """Mark an entry as closed after IB fill"""
        entry.status = 'closed'
        entry.close_price = close_price
        entry.close_time = datetime.now().isoformat()
        entry.pnl = (close_price - entry.entry_price) * entry.quantity * 10  # MGC = $10/point
        self.realized_pnl += entry.pnl
        self._recalc_average()
        self._save_state()
        print(f"✅ Closed L{entry.level} @ ${close_price:.2f} | P&L: ${entry.pnl:.2f} | Total Realized: ${self.realized_pnl:.2f}")
    
    def get_open_entries(self) -> List[VirtualEntry]:
        return [e for e in self.entries if e.status == 'open']
    
    def get_closed_entries(self) -> List[VirtualEntry]:
        return [e for e in self.entries if e.status == 'closed']
    
    def get_lowest_open(self) -> Optional[VirtualEntry]:
        """Get the lowest priced open entry"""
        open_entries = self.get_open_entries()
        return min(open_entries, key=lambda e: e.entry_price) if open_entries else None
    
    def get_highest_open(self) -> Optional[VirtualEntry]:
        """Get the highest priced open entry"""
        open_entries = self.get_open_entries()
        return max(open_entries, key=lambda e: e.entry_price) if open_entries else None
    
    def get_stats(self) -> dict:
        """Summary stats for display"""
        open_entries = self.get_open_entries()
        closed_entries = self.get_closed_entries()
        
        # Calculate unrealized P&L (would need current price)
        return {
            'open_count': len(open_entries),
            'closed_count': len(closed_entries),
            'total_quantity': self.total_quantity,
            'avg_price': self.avg_price,
            'realized_pnl': self.realized_pnl,
            'freeroll': self.realized_pnl > 0,
            'levels': [
                {
                    'level': e.level,
                    'entry': e.entry_price,
                    'tp': e.tp_price,
                    'status': e.status,
                    'pnl': e.pnl
                }
                for e in self.entries
            ]
        }
    
    def reset(self):
        """Reset the ladder (use with caution)"""
        self.entries = []
        self.total_quantity = 0
        self.avg_price = 0.0
        # Note: Don't reset realized_pnl - that's banked profit
        self._save_state()
        print("🔄 Ladder reset (realized P&L preserved)")


# ═══════════════════════════════════════════════════════════════════════════════
# CASPER IB BOT
# ═══════════════════════════════════════════════════════════════════════════════

class CasperIB:
    """
    Casper-style DCA trading for Interactive Brokers.
    
    Uses virtual position tracking to achieve MT5-like behavior
    where each entry has its own TP, even though IB aggregates positions.
    """
    
    def __init__(
        self,
        host: str = '100.119.161.65',
        port: int = 7497,
        client_id: int = 450,
        symbol: str = 'MGC',
        tp_points: float = 50.0,
        dca_spacing: float = 20.0,
        max_levels: int = 10,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.symbol = symbol
        self.dca_spacing = dca_spacing
        self.max_levels = max_levels
        
        self.ib = IB()
        self.contract = None
        self.ladder = DCALadder(symbol, tp_points=tp_points)
        
        self.running = False
        self.last_price = 0.0
    
    def connect(self) -> bool:
        """Connect to IB TWS"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
            
            # Get MGC contract
            self.contract = Future(conId=706903676, symbol='MGC', exchange='COMEX')
            self.ib.qualifyContracts(self.contract)
            
            print(f"✅ Connected to IB (clientId={self.client_id})")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB"""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("📴 Disconnected from IB")
    
    def get_price(self) -> float:
        """Get current MGC price"""
        self.ib.reqMarketDataType(3)  # Delayed data
        ticker = self.ib.reqMktData(self.contract)
        self.ib.sleep(1)
        price = ticker.last or ticker.close or 0
        self.ib.cancelMktData(self.contract)
        return price
    
    def get_ib_position(self) -> int:
        """Get actual IB position quantity"""
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == 'MGC':
                return int(pos.position)
        return 0
    
    def execute_buy(self, quantity: int = 1) -> Optional[float]:
        """Execute a buy order on IB"""
        try:
            order = MarketOrder('BUY', quantity)
            trade = self.ib.placeOrder(self.contract, order)
            
            # Wait for fill
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(0.5)
                timeout -= 0.5
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                print(f"🟢 BUY {quantity} @ ${fill_price:.2f}")
                return fill_price
            else:
                print(f"⚠️ Buy order not filled: {trade.orderStatus.status}")
                return None
        except Exception as e:
            print(f"❌ Buy error: {e}")
            return None
    
    def execute_sell(self, quantity: int = 1) -> Optional[float]:
        """Execute a sell order on IB"""
        try:
            order = MarketOrder('SELL', quantity)
            trade = self.ib.placeOrder(self.contract, order)
            
            # Wait for fill
            timeout = 30
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(0.5)
                timeout -= 0.5
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                print(f"🔴 SELL {quantity} @ ${fill_price:.2f}")
                return fill_price
            else:
                print(f"⚠️ Sell order not filled: {trade.orderStatus.status}")
                return None
        except Exception as e:
            print(f"❌ Sell error: {e}")
            return None
    
    def check_and_execute_tps(self, current_price: float):
        """Check for TP hits and execute sells"""
        ready = self.ladder.check_tps(current_price)
        
        for entry in ready:
            fill_price = self.execute_sell(entry.quantity)
            if fill_price:
                self.ladder.close_entry(entry, fill_price)
            else:
                # Reset to open if sell failed
                entry.status = 'open'
    
    def check_new_dca(self, current_price: float):
        """Check if we should add a new DCA level"""
        open_entries = self.ladder.get_open_entries()
        
        # Already at max depth
        if len(open_entries) >= self.max_levels:
            return
        
        # No positions - need external trigger for first entry
        if not open_entries:
            return
        
        # Check if price dropped enough from lowest entry
        lowest = self.ladder.get_lowest_open()
        if lowest and (lowest.entry_price - current_price) >= self.dca_spacing:
            fill_price = self.execute_buy(1)
            if fill_price:
                self.ladder.add_entry(fill_price, 1)
    
    def sync_with_ib(self):
        """
        Sync virtual ladder with actual IB position.
        Call this on startup to reconcile state.
        """
        ib_qty = self.get_ib_position()
        virtual_qty = self.ladder.total_quantity
        
        if ib_qty != virtual_qty:
            print(f"⚠️ Position mismatch: IB has {ib_qty}, virtual has {virtual_qty}")
            # TODO: Implement reconciliation logic
            # For now, just warn
        else:
            print(f"✅ Positions in sync: {ib_qty} contracts")
    
    def run(self, interval: int = 5):
        """Main run loop"""
        if not self.ib.isConnected():
            if not self.connect():
                return
        
        self.sync_with_ib()
        self.running = True
        
        print(f"\n{'═'*60}")
        print(f"  CASPER IB - Virtual DCA Tracking")
        print(f"  Symbol: {self.symbol} | TP: +${self.ladder.tp_points} | Spacing: ${self.dca_spacing}")
        print(f"  Max Levels: {self.max_levels} | Interval: {interval}s")
        print(f"{'═'*60}\n")
        
        try:
            while self.running:
                price = self.get_price()
                if price <= 0:
                    self.ib.sleep(interval)
                    continue
                
                self.last_price = price
                
                # Check TPs first (priority)
                self.check_and_execute_tps(price)
                
                # Check for new DCA entries
                self.check_new_dca(price)
                
                # Status display
                stats = self.ladder.get_stats()
                ib_pos = self.get_ib_position()
                
                print(f"💰 ${price:.2f} | IB: {ib_pos} | Virtual: {stats['open_count']} open, {stats['closed_count']} closed | Realized: ${stats['realized_pnl']:.2f}")
                
                self.ib.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Casper IB...")
        finally:
            self.running = False
            self.disconnect()
    
    def add_manual_entry(self, price: float, quantity: int = 1):
        """Manually add an entry (for existing positions)"""
        entry = self.ladder.add_entry(price, quantity)
        return entry
    
    def status(self):
        """Print current status"""
        stats = self.ladder.get_stats()
        print(f"\n{'═'*60}")
        print(f"  CASPER IB STATUS")
        print(f"{'═'*60}")
        print(f"  Open Entries:    {stats['open_count']}")
        print(f"  Closed Entries:  {stats['closed_count']}")
        print(f"  Total Contracts: {stats['total_quantity']}")
        print(f"  Avg Price:       ${stats['avg_price']:.2f}")
        print(f"  Realized P&L:    ${stats['realized_pnl']:.2f}")
        print(f"  Freeroll:        {'✅ YES' if stats['freeroll'] else '⏳ Building...'}")
        print(f"\n  Levels:")
        for lvl in stats['levels']:
            status_icon = '✅' if lvl['status'] == 'closed' else '🔵' if lvl['status'] == 'open' else '⏳'
            pnl_str = f"P&L: ${lvl['pnl']:.2f}" if lvl['pnl'] else f"TP: ${lvl['tp']:.2f}"
            print(f"    {status_icon} L{lvl['level']}: ${lvl['entry']:.2f} → {pnl_str} [{lvl['status']}]")
        print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    casper = CasperIB()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'status':
            casper.connect()
            casper.status()
            casper.disconnect()
            
        elif cmd == 'add' and len(sys.argv) >= 3:
            # Add manual entry: python casper_ib.py add 5560 1
            price = float(sys.argv[2])
            qty = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            casper.ladder.add_entry(price, qty)
            casper.status()
            
        elif cmd == 'reset':
            casper.ladder.reset()
            print("Ladder reset")
            
        elif cmd == 'run':
            casper.run()
            
        else:
            print("Usage:")
            print("  python casper_ib.py status     - Show current status")
            print("  python casper_ib.py add <price> [qty]  - Add manual entry")
            print("  python casper_ib.py reset      - Reset ladder")
            print("  python casper_ib.py run        - Run bot")
    else:
        # Default: show status
        casper.status()

#!/usr/bin/env python3
"""
NEO Position Tracker - Track Real Positions
NO RANDOM DATA - All positions from MT5 API.
"""

import requests
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

from config import (
    MT5_API_URL, ACCOUNT_SIZE, 
    MAX_POSITION_DOLLARS, MAX_DAILY_LOSS_DOLLARS,
    MAX_OPEN_POSITIONS
)


@dataclass
class Position:
    """Real position data from MT5."""
    ticket: int
    symbol: str
    direction: str  # BUY or SELL
    volume: float
    open_price: float
    current_price: float
    unrealized_pnl: float
    open_time: str
    source: str = "MT5_REAL"


@dataclass
class TradeResult:
    """Completed trade result."""
    ticket: int
    symbol: str
    direction: str
    volume: float
    open_price: float
    close_price: float
    realized_pnl: float
    open_time: str
    close_time: str
    duration_minutes: int
    source: str = "MT5_REAL"


@dataclass
class AccountState:
    """Current account state."""
    timestamp: str
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    open_positions: List[Position]
    total_unrealized_pnl: float
    today_realized_pnl: float
    position_count: int
    mt5_connected: bool
    
    # Safety metrics
    daily_loss_remaining: float
    position_capacity_remaining: int
    can_open_new: bool
    kill_switch_triggered: bool


class PositionTracker:
    """
    Tracks REAL positions from MT5.
    Calculates P&L and enforces safety limits.
    """
    
    def __init__(self):
        self.today_realized_pnl = 0.0
        self.last_reset_date = date.today()
        self.trade_history: List[TradeResult] = []
    
    def _reset_daily_if_needed(self):
        """Reset daily P&L counter at midnight."""
        if date.today() != self.last_reset_date:
            self.today_realized_pnl = 0.0
            self.last_reset_date = date.today()
    
    def _fetch_positions(self) -> List[Position]:
        """Fetch real open positions from MT5."""
        positions = []
        try:
            response = requests.get(
                f"{MT5_API_URL}/positions",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for pos in data.get("positions", []):
                    positions.append(Position(
                        ticket=pos.get("ticket", 0),
                        symbol=pos.get("symbol", "").replace("/", ""),
                        direction="BUY" if pos.get("type", 0) == 0 else "SELL",
                        volume=pos.get("volume", 0),
                        open_price=pos.get("open_price", 0),
                        current_price=pos.get("current_price", 0),
                        unrealized_pnl=pos.get("profit", 0),
                        open_time=pos.get("time", datetime.utcnow().isoformat()),
                        source="MT5_REAL"
                    ))
        except requests.exceptions.ConnectionError:
            pass  # MT5 not available
        except Exception as e:
            print(f"Position fetch error: {e}")
        
        return positions
    
    def _fetch_account_info(self) -> Dict[str, float]:
        """Fetch real account info from MT5."""
        try:
            response = requests.get(
                f"{MT5_API_URL}/account",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "balance": data.get("balance", ACCOUNT_SIZE),
                    "equity": data.get("equity", ACCOUNT_SIZE),
                    "margin_used": data.get("margin_used", 0),
                    "free_margin": data.get("free_margin", ACCOUNT_SIZE)
                }
        except:
            pass
        
        # Return defaults if MT5 not available
        return {
            "balance": ACCOUNT_SIZE,
            "equity": ACCOUNT_SIZE,
            "margin_used": 0,
            "free_margin": ACCOUNT_SIZE
        }
    
    def _fetch_today_trades(self) -> List[TradeResult]:
        """Fetch today's completed trades from MT5."""
        trades = []
        try:
            response = requests.get(
                f"{MT5_API_URL}/trades",
                params={"period": "today"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for trade in data.get("trades", []):
                    if trade.get("close_time"):  # Only completed trades
                        trades.append(TradeResult(
                            ticket=trade.get("ticket", 0),
                            symbol=trade.get("symbol", "").replace("/", ""),
                            direction="BUY" if trade.get("type", 0) == 0 else "SELL",
                            volume=trade.get("volume", 0),
                            open_price=trade.get("open_price", 0),
                            close_price=trade.get("close_price", 0),
                            realized_pnl=trade.get("profit", 0),
                            open_time=trade.get("time", ""),
                            close_time=trade.get("close_time", ""),
                            duration_minutes=trade.get("duration_minutes", 0),
                            source="MT5_REAL"
                        ))
        except:
            pass
        
        return trades
    
    def get_state(self) -> AccountState:
        """Get complete account state with safety checks."""
        self._reset_daily_if_needed()
        
        # Fetch real data
        positions = self._fetch_positions()
        account = self._fetch_account_info()
        today_trades = self._fetch_today_trades()
        
        # Calculate P&L
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        today_realized = sum(t.realized_pnl for t in today_trades)
        
        # Safety calculations
        total_daily_pnl = today_realized + total_unrealized
        daily_loss_remaining = MAX_DAILY_LOSS_DOLLARS + total_daily_pnl  # Positive means room
        position_capacity = MAX_OPEN_POSITIONS - len(positions)
        
        # Kill switch check
        kill_switch = total_daily_pnl <= -MAX_DAILY_LOSS_DOLLARS
        
        # Can open new position?
        can_open = (
            not kill_switch and
            position_capacity > 0 and
            daily_loss_remaining > 0
        )
        
        return AccountState(
            timestamp=datetime.utcnow().isoformat(),
            balance=account["balance"],
            equity=account["equity"],
            margin_used=account["margin_used"],
            free_margin=account["free_margin"],
            open_positions=positions,
            total_unrealized_pnl=total_unrealized,
            today_realized_pnl=today_realized,
            position_count=len(positions),
            mt5_connected=bool(positions) or account["balance"] != ACCOUNT_SIZE,
            daily_loss_remaining=max(0, daily_loss_remaining),
            position_capacity_remaining=max(0, position_capacity),
            can_open_new=can_open,
            kill_switch_triggered=kill_switch
        )
    
    def to_llm_context(self, state: AccountState) -> str:
        """Format account state for LLM consumption."""
        lines = [
            "=== ACCOUNT STATE (REAL) ===",
            f"Timestamp: {state.timestamp}",
            f"MT5 Connected: {state.mt5_connected}",
            "",
            f"Balance: ${state.balance:,.2f}",
            f"Equity: ${state.equity:,.2f}",
            f"Unrealized P&L: ${state.total_unrealized_pnl:+,.2f}",
            f"Today Realized: ${state.today_realized_pnl:+,.2f}",
            "",
            "=== SAFETY STATUS ===",
            f"Kill Switch: {'🚨 TRIGGERED' if state.kill_switch_triggered else '✅ OK'}",
            f"Daily Loss Remaining: ${state.daily_loss_remaining:,.2f}",
            f"Position Slots: {state.position_capacity_remaining}/{MAX_OPEN_POSITIONS}",
            f"Can Open New: {'✅ YES' if state.can_open_new else '❌ NO'}",
        ]
        
        if state.open_positions:
            lines.append("")
            lines.append("=== OPEN POSITIONS ===")
            for pos in state.open_positions:
                pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                lines.append(
                    f"  {pnl_emoji} #{pos.ticket} {pos.symbol} {pos.direction} "
                    f"{pos.volume} lots @ {pos.open_price:.5f} "
                    f"P&L: ${pos.unrealized_pnl:+,.2f}"
                )
        else:
            lines.append("")
            lines.append("=== OPEN POSITIONS ===")
            lines.append("  No open positions")
        
        return "\n".join(lines)


def test_position_tracker():
    """Test the position tracker."""
    print("=" * 60)
    print("NEO POSITION TRACKER TEST")
    print("=" * 60)
    
    tracker = PositionTracker()
    state = tracker.get_state()
    
    print(tracker.to_llm_context(state))
    print("")
    print("=" * 60)


if __name__ == "__main__":
    test_position_tracker()

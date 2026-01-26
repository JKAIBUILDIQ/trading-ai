#!/usr/bin/env python3
"""
Steady Climb Position Sizer for NEO Trading System
Implements Paul's 1,1,2,2,4,4,8,8 progression for forex/gold trading

Key Principle: Only risk 1 unit from bankroll, scale up with winnings
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class PositionState:
    """Current state in the Steady Climb progression"""
    position: int = 0          # 0-7 index in progression
    units: int = 1             # Current unit multiplier
    consecutive_wins: int = 0  # Wins since last reset
    cycle_profit: float = 0    # Profit accumulated this cycle
    cycles_completed: int = 0  # Full progressions completed
    total_profit: float = 0    # All-time profit
    trades_today: int = 0
    last_trade_time: str = ""
    last_reset_reason: str = ""

class SteadyClimbPositionSizer:
    """
    Steady Climb Position Sizing Strategy
    
    Progression: 1, 1, 2, 2, 4, 4, 8, 8
    - Win: Advance to next position
    - Loss: Reset to position 0 (1 unit)
    - Complete cycle (8 wins): Stay at 8 or reset
    
    Risk Management:
    - Only 1 unit is "real" risk from bankroll
    - Additional units come from accumulated winnings
    - Never chase losses - always reset to 1
    """
    
    PROGRESSION = [1, 1, 2, 2, 4, 4, 8, 8]
    MAX_POSITION = 7  # Index of last position
    
    def __init__(
        self,
        base_lot_size: float = 0.01,  # Base lot size (1 unit)
        max_lot_size: float = 0.10,   # Maximum allowed lot
        daily_loss_limit: float = 500,  # Stop trading if down this much
        daily_trade_limit: int = 10,    # Max trades per day
        state_file: str = None
    ):
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.daily_loss_limit = daily_loss_limit
        self.daily_trade_limit = daily_trade_limit
        
        self.state_file = state_file or os.path.join(
            os.path.dirname(__file__), 'steady_climb_state.json'
        )
        
        self.state = self._load_state()
    
    def _load_state(self) -> PositionState:
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return PositionState(**data)
            except:
                pass
        return PositionState()
    
    def _save_state(self):
        """Save state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
    
    def get_current_units(self) -> int:
        """Get current unit multiplier"""
        return self.PROGRESSION[self.state.position]
    
    def get_lot_size(self) -> float:
        """Calculate lot size for next trade"""
        units = self.get_current_units()
        lot_size = self.base_lot_size * units
        return min(lot_size, self.max_lot_size)
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        units = self.get_current_units()
        return {
            'position': self.state.position + 1,  # 1-indexed for display
            'units': units,
            'lot_size': self.get_lot_size(),
            'consecutive_wins': self.state.consecutive_wins,
            'cycle_profit': self.state.cycle_profit,
            'cycles_completed': self.state.cycles_completed,
            'total_profit': self.state.total_profit,
            'trades_today': self.state.trades_today,
            'progression_display': self._get_progression_display(),
            'next_win_units': self.PROGRESSION[min(self.state.position + 1, self.MAX_POSITION)],
            'risk_status': self._get_risk_status(),
        }
    
    def _get_progression_display(self) -> str:
        """Visual representation of current position"""
        display = []
        for i, units in enumerate(self.PROGRESSION):
            if i == self.state.position:
                display.append(f"[{units}]")
            elif i < self.state.position:
                display.append(f"✓{units}")
            else:
                display.append(f" {units} ")
        return " → ".join(display)
    
    def _get_risk_status(self) -> str:
        """Get risk status message"""
        if self.state.position == 0:
            return "STARTING - Risk: 1 unit (your money)"
        elif self.state.position <= 2:
            return f"BUILDING - Risk: {self.get_current_units()} units (mostly house money)"
        elif self.state.position <= 5:
            return f"CLIMBING - Risk: {self.get_current_units()} units (all house money)"
        else:
            return f"PEAK - Risk: {self.get_current_units()} units (maximum position)"
    
    def record_win(self, profit: float) -> Dict:
        """Record a winning trade"""
        old_position = self.state.position
        old_units = self.get_current_units()
        
        # Update profit tracking
        self.state.cycle_profit += profit
        self.state.total_profit += profit
        self.state.consecutive_wins += 1
        self.state.trades_today += 1
        self.state.last_trade_time = datetime.now().isoformat()
        
        # Advance position
        if self.state.position < self.MAX_POSITION:
            self.state.position += 1
        else:
            # Completed full cycle!
            self.state.cycles_completed += 1
            # Option: Reset or stay at max
            # Staying at max for continued high gains
        
        new_units = self.get_current_units()
        self._save_state()
        
        return {
            'action': 'WIN_RECORDED',
            'profit': profit,
            'old_position': old_position + 1,
            'new_position': self.state.position + 1,
            'old_units': old_units,
            'new_units': new_units,
            'cycle_profit': self.state.cycle_profit,
            'consecutive_wins': self.state.consecutive_wins,
            'message': f"Advanced to position {self.state.position + 1} ({new_units} units)"
        }
    
    def record_loss(self, loss: float) -> Dict:
        """Record a losing trade - RESET to position 0"""
        old_position = self.state.position
        old_units = self.get_current_units()
        old_cycle_profit = self.state.cycle_profit
        
        # Update tracking
        self.state.total_profit += loss  # loss is negative
        self.state.trades_today += 1
        self.state.last_trade_time = datetime.now().isoformat()
        
        # RESET - This is the key to Steady Climb
        self.state.position = 0
        self.state.consecutive_wins = 0
        self.state.cycle_profit = 0
        self.state.last_reset_reason = f"Loss at position {old_position + 1}"
        
        self._save_state()
        
        return {
            'action': 'LOSS_RESET',
            'loss': loss,
            'old_position': old_position + 1,
            'new_position': 1,
            'old_units': old_units,
            'new_units': 1,
            'cycle_profit_lost': old_cycle_profit,
            'consecutive_wins_lost': self.state.consecutive_wins,
            'message': f"Reset to position 1 (1 unit). Cycle profit was ${old_cycle_profit:.2f}"
        }
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if we can take another trade"""
        # Check daily trade limit
        if self.state.trades_today >= self.daily_trade_limit:
            return False, f"Daily trade limit reached ({self.daily_trade_limit})"
        
        # Check if we should reset for new day
        if self.state.last_trade_time:
            last_trade = datetime.fromisoformat(self.state.last_trade_time)
            if last_trade.date() < datetime.now().date():
                self._reset_daily_counters()
        
        return True, "Ready to trade"
    
    def _reset_daily_counters(self):
        """Reset daily counters"""
        self.state.trades_today = 0
        self._save_state()
    
    def reset_progression(self, reason: str = "Manual reset"):
        """Manually reset the progression"""
        self.state.position = 0
        self.state.consecutive_wins = 0
        self.state.cycle_profit = 0
        self.state.last_reset_reason = reason
        self._save_state()
    
    def get_signal_with_sizing(self, signal: Dict) -> Dict:
        """
        Take a trading signal and add Steady Climb position sizing
        
        Args:
            signal: Dict with keys like 'action', 'symbol', 'entry', 'sl', 'tp'
        
        Returns:
            Signal with lot_size and position info added
        """
        can_trade, reason = self.can_trade()
        
        if not can_trade:
            return {
                **signal,
                'can_execute': False,
                'reason': reason
            }
        
        position_info = self.get_position_info()
        
        return {
            **signal,
            'can_execute': True,
            'lot_size': position_info['lot_size'],
            'units': position_info['units'],
            'position': position_info['position'],
            'progression': position_info['progression_display'],
            'risk_status': position_info['risk_status'],
            'steady_climb': True
        }


# Singleton instance for NEO
_sizer_instance = None

def get_position_sizer() -> SteadyClimbPositionSizer:
    """Get or create the position sizer instance"""
    global _sizer_instance
    if _sizer_instance is None:
        _sizer_instance = SteadyClimbPositionSizer()
    return _sizer_instance


def main():
    """Test the position sizer"""
    sizer = SteadyClimbPositionSizer(base_lot_size=0.01)
    
    print("="*60)
    print("🎰 STEADY CLIMB POSITION SIZER TEST")
    print("="*60)
    
    # Simulate a series of trades
    trades = [
        ('WIN', 50),
        ('WIN', 50),
        ('WIN', 100),
        ('WIN', 100),
        ('WIN', 200),
        ('LOSS', -200),  # Reset!
        ('WIN', 50),
        ('WIN', 50),
    ]
    
    for outcome, pnl in trades:
        print(f"\n{'='*40}")
        info = sizer.get_position_info()
        print(f"Current: Position {info['position']}, {info['units']} units, Lot: {info['lot_size']}")
        print(f"Progression: {info['progression_display']}")
        
        if outcome == 'WIN':
            result = sizer.record_win(pnl)
        else:
            result = sizer.record_loss(pnl)
        
        print(f"Trade: {outcome} ${pnl}")
        print(f"Result: {result['message']}")
    
    print(f"\n{'='*60}")
    print("FINAL STATE:")
    info = sizer.get_position_info()
    print(f"Position: {info['position']}")
    print(f"Total Profit: ${sizer.state.total_profit:.2f}")
    print(f"Cycles Completed: {sizer.state.cycles_completed}")


if __name__ == "__main__":
    main()

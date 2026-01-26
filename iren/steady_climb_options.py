#!/usr/bin/env python3
"""
Steady Climb Options Strategy for IREN
Applies Paul's 1,1,2,2,4,4,8,8 progression to options trading

Key Differences from Forex:
- Options have expiration (prefer 21-35 DTE)
- Options have larger % swings (good for progression!)
- We're LONG ONLY (calls only per Paul's rules)
- Contract sizing instead of lot sizing
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import yfinance as yf

@dataclass 
class OptionsClimbState:
    """Current state in the Steady Climb progression for options"""
    position: int = 0              # 0-7 index in progression
    contracts: int = 1             # Current contract multiplier
    consecutive_wins: int = 0      # Wins since last reset
    cycle_profit: float = 0        # Profit accumulated this cycle
    cycles_completed: int = 0      # Full progressions completed
    total_profit: float = 0        # All-time profit
    total_trades: int = 0
    winning_trades: int = 0
    last_trade_time: str = ""
    last_reset_reason: str = ""
    current_position: Optional[Dict] = None  # Open position if any

class IrenSteadyClimbOptions:
    """
    Steady Climb Strategy for IREN Options
    
    Progression: 1, 1, 2, 2, 4, 4, 8, 8 contracts
    - Win (TP hit): Advance to next position
    - Loss (SL hit or expiry loss): Reset to position 0
    
    RULES (Paul's preferences):
    - LONG ONLY: Calls only
    - Min 14 DTE, prefer 21-35 DTE
    - Avoid earnings week (Feb 5, 2026)
    - Strike preference: $60, $70, $80
    """
    
    PROGRESSION = [1, 1, 2, 2, 4, 4, 8, 8]
    MAX_POSITION = 7
    
    # Paul's rules
    LONG_ONLY = True
    MIN_DTE = 14
    PREFERRED_DTE_MIN = 21
    PREFERRED_DTE_MAX = 35
    PREFERRED_STRIKES = [60, 70, 80]
    EARNINGS_DATE = datetime(2026, 2, 5)
    EARNINGS_BLACKOUT_DAYS = 7
    
    def __init__(
        self,
        base_contracts: int = 1,       # 1 contract = 1 unit
        max_contracts: int = 10,       # Cap at 10 contracts
        max_contract_price: float = 500,  # Max $500 per contract
        take_profit_pct: float = 50,   # TP at +50% gain
        stop_loss_pct: float = 30,     # SL at -30% loss
        state_file: str = None
    ):
        self.base_contracts = base_contracts
        self.max_contracts = max_contracts
        self.max_contract_price = max_contract_price
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        self.state_file = state_file or os.path.join(
            os.path.dirname(__file__), 'data', 'iren_climb_state.json'
        )
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        self.state = self._load_state()
    
    def _load_state(self) -> OptionsClimbState:
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return OptionsClimbState(**data)
            except:
                pass
        return OptionsClimbState()
    
    def _save_state(self):
        """Save state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
    
    def get_current_contracts(self) -> int:
        """Get current contract multiplier"""
        return self.PROGRESSION[self.state.position]
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        contracts = self.get_current_contracts()
        return {
            'position': self.state.position + 1,
            'max_position': 8,
            'contracts': contracts,
            'base_contracts': self.base_contracts,
            'total_contracts': contracts * self.base_contracts,
            'consecutive_wins': self.state.consecutive_wins,
            'cycle_profit': self.state.cycle_profit,
            'cycles_completed': self.state.cycles_completed,
            'total_profit': self.state.total_profit,
            'total_trades': self.state.total_trades,
            'win_rate': (self.state.winning_trades / self.state.total_trades * 100) if self.state.total_trades > 0 else 0,
            'progression_display': self._get_progression_display(),
            'risk_status': self._get_risk_status(),
            'has_open_position': self.state.current_position is not None,
            'open_position': self.state.current_position,
        }
    
    def _get_progression_display(self) -> str:
        """Visual representation of current position"""
        display = []
        for i, contracts in enumerate(self.PROGRESSION):
            if i == self.state.position:
                display.append(f"[{contracts}]")
            elif i < self.state.position:
                display.append(f"✓{contracts}")
            else:
                display.append(f" {contracts} ")
        return " → ".join(display)
    
    def _get_risk_status(self) -> str:
        """Get risk status message"""
        contracts = self.get_current_contracts()
        if self.state.position == 0:
            return f"STARTING - {contracts} contract (your capital)"
        elif self.state.position <= 2:
            return f"BUILDING - {contracts} contracts (mostly house money)"
        elif self.state.position <= 5:
            return f"CLIMBING - {contracts} contracts (all house money)"
        else:
            return f"PEAK - {contracts} contracts (maximum position)"
    
    def get_iren_price(self) -> float:
        """Get current IREN stock price"""
        try:
            ticker = yf.Ticker("IREN")
            data = ticker.history(period="1d")
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return 0
    
    def get_recommended_option(self) -> Optional[Dict]:
        """Get recommended IREN call option based on Paul's rules"""
        try:
            ticker = yf.Ticker("IREN")
            current_price = self.get_iren_price()
            
            if current_price == 0:
                return None
            
            # Get expiration dates
            expirations = ticker.options
            today = datetime.now()
            
            recommended = None
            
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - today).days
                
                # Skip if too close to expiry
                if dte < self.MIN_DTE:
                    continue
                
                # Skip if in earnings blackout
                days_to_earnings = (self.EARNINGS_DATE - exp_date).days
                if abs(days_to_earnings) < self.EARNINGS_BLACKOUT_DAYS:
                    continue
                
                # Prefer 21-35 DTE
                is_preferred_dte = self.PREFERRED_DTE_MIN <= dte <= self.PREFERRED_DTE_MAX
                
                # Get options chain
                chain = ticker.option_chain(exp_str)
                calls = chain.calls
                
                for strike in self.PREFERRED_STRIKES:
                    # Find this strike
                    strike_options = calls[calls['strike'] == strike]
                    if len(strike_options) == 0:
                        continue
                    
                    opt = strike_options.iloc[0]
                    last_price = float(opt['lastPrice'])
                    bid = float(opt['bid']) if opt['bid'] > 0 else last_price * 0.95
                    ask = float(opt['ask']) if opt['ask'] > 0 else last_price * 1.05
                    volume = int(opt['volume']) if not pd.isna(opt['volume']) else 0
                    open_interest = int(opt['openInterest']) if not pd.isna(opt['openInterest']) else 0
                    
                    # Skip if too expensive
                    if last_price * 100 > self.max_contract_price:
                        continue
                    
                    # Calculate greeks if available
                    delta = float(opt.get('delta', 0.5)) if 'delta' in opt else 0.5
                    
                    option_info = {
                        'symbol': 'IREN',
                        'type': 'CALL',
                        'strike': strike,
                        'expiration': exp_str,
                        'dte': dte,
                        'last_price': last_price,
                        'bid': bid,
                        'ask': ask,
                        'mid_price': (bid + ask) / 2,
                        'volume': volume,
                        'open_interest': open_interest,
                        'delta': delta,
                        'is_preferred_dte': is_preferred_dte,
                        'is_pauls_pick': is_preferred_dte and strike == 60,
                        'contract_cost': last_price * 100,
                        'current_stock_price': current_price,
                        'moneyness': 'ITM' if strike < current_price else ('ATM' if abs(strike - current_price) < 2 else 'OTM'),
                    }
                    
                    # Prefer Paul's pick
                    if recommended is None or option_info['is_pauls_pick']:
                        recommended = option_info
                        if option_info['is_pauls_pick']:
                            break
                
                if recommended and recommended['is_pauls_pick']:
                    break
            
            return recommended
            
        except Exception as e:
            print(f"Error getting options: {e}")
            return None
    
    def generate_signal(self) -> Dict:
        """Generate a trading signal with Steady Climb sizing"""
        
        # Check if we have an open position
        if self.state.current_position:
            return {
                'action': 'HOLD',
                'reason': 'Position already open',
                'current_position': self.state.current_position,
                'position_info': self.get_position_info()
            }
        
        # Get recommended option
        option = self.get_recommended_option()
        
        if option is None:
            return {
                'action': 'WAIT',
                'reason': 'No suitable options found',
                'position_info': self.get_position_info()
            }
        
        contracts = self.get_current_contracts() * self.base_contracts
        contracts = min(contracts, self.max_contracts)
        
        entry_price = option['mid_price']
        tp_price = entry_price * (1 + self.take_profit_pct / 100)
        sl_price = entry_price * (1 - self.stop_loss_pct / 100)
        
        total_cost = entry_price * 100 * contracts
        max_profit = (tp_price - entry_price) * 100 * contracts
        max_loss = (entry_price - sl_price) * 100 * contracts
        
        signal = {
            'action': 'BUY_CALL',
            'symbol': 'IREN',
            'option': option,
            'contracts': contracts,
            'entry_price': entry_price,
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'total_cost': total_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': max_profit / max_loss if max_loss > 0 else 0,
            'position_info': self.get_position_info(),
            'steady_climb': {
                'position': self.state.position + 1,
                'units': self.get_current_contracts(),
                'progression': self._get_progression_display(),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return signal
    
    def open_position(self, option: Dict, entry_price: float, contracts: int) -> Dict:
        """Record opening a new position"""
        self.state.current_position = {
            'option': option,
            'entry_price': entry_price,
            'contracts': contracts,
            'opened_at': datetime.now().isoformat(),
            'position_level': self.state.position,
            'tp_price': entry_price * (1 + self.take_profit_pct / 100),
            'sl_price': entry_price * (1 - self.stop_loss_pct / 100),
        }
        self._save_state()
        
        return {
            'action': 'POSITION_OPENED',
            'position': self.state.current_position,
            'message': f"Opened {contracts} IREN {option['strike']}C {option['expiration']} @ ${entry_price:.2f}"
        }
    
    def close_position(self, exit_price: float, reason: str = "Manual") -> Dict:
        """Close the current position and record result"""
        if not self.state.current_position:
            return {'action': 'NO_POSITION', 'message': 'No open position to close'}
        
        pos = self.state.current_position
        entry_price = pos['entry_price']
        contracts = pos['contracts']
        
        pnl = (exit_price - entry_price) * 100 * contracts
        pnl_pct = (exit_price / entry_price - 1) * 100
        
        is_win = pnl > 0
        
        # Update state
        self.state.total_trades += 1
        self.state.total_profit += pnl
        
        if is_win:
            self.state.winning_trades += 1
            result = self.record_win(pnl)
        else:
            result = self.record_loss(pnl)
        
        # Clear current position
        closed_position = self.state.current_position
        self.state.current_position = None
        self._save_state()
        
        return {
            'action': 'POSITION_CLOSED',
            'result': 'WIN' if is_win else 'LOSS',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'contracts': contracts,
            'reason': reason,
            'closed_position': closed_position,
            'climb_result': result,
            'new_position_info': self.get_position_info()
        }
    
    def record_win(self, profit: float) -> Dict:
        """Record a winning trade - advance position"""
        old_position = self.state.position
        old_contracts = self.get_current_contracts()
        
        self.state.cycle_profit += profit
        self.state.consecutive_wins += 1
        self.state.last_trade_time = datetime.now().isoformat()
        
        if self.state.position < self.MAX_POSITION:
            self.state.position += 1
        else:
            self.state.cycles_completed += 1
        
        new_contracts = self.get_current_contracts()
        self._save_state()
        
        return {
            'action': 'WIN_ADVANCE',
            'profit': profit,
            'old_position': old_position + 1,
            'new_position': self.state.position + 1,
            'old_contracts': old_contracts,
            'new_contracts': new_contracts,
            'message': f"🎉 WIN! Advanced to Position {self.state.position + 1} ({new_contracts} contracts)"
        }
    
    def record_loss(self, loss: float) -> Dict:
        """Record a losing trade - RESET to position 0"""
        old_position = self.state.position
        old_contracts = self.get_current_contracts()
        old_cycle_profit = self.state.cycle_profit
        
        self.state.last_trade_time = datetime.now().isoformat()
        
        # RESET
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
            'old_contracts': old_contracts,
            'new_contracts': 1,
            'cycle_profit_lost': old_cycle_profit,
            'message': f"❌ Loss. Reset to Position 1 (1 contract). Cycle profit was ${old_cycle_profit:.2f}"
        }
    
    def reset_progression(self, reason: str = "Manual reset"):
        """Manually reset the progression"""
        self.state.position = 0
        self.state.consecutive_wins = 0
        self.state.cycle_profit = 0
        self.state.last_reset_reason = reason
        self._save_state()
    
    def get_status(self) -> Dict:
        """Get full status for API/display"""
        position_info = self.get_position_info()
        
        return {
            'strategy': 'IREN Steady Climb Options',
            'status': 'ACTIVE',
            'position_info': position_info,
            'rules': {
                'long_only': self.LONG_ONLY,
                'min_dte': self.MIN_DTE,
                'preferred_dte': f"{self.PREFERRED_DTE_MIN}-{self.PREFERRED_DTE_MAX}",
                'preferred_strikes': self.PREFERRED_STRIKES,
                'take_profit_pct': self.take_profit_pct,
                'stop_loss_pct': self.stop_loss_pct,
            },
            'next_trade': {
                'contracts': self.get_current_contracts() * self.base_contracts,
                'max_cost': self.max_contract_price * self.get_current_contracts(),
            },
            'timestamp': datetime.now().isoformat()
        }


# Add pandas import at top if needed
import pandas as pd


def main():
    """Test the IREN Steady Climb options strategy"""
    climber = IrenSteadyClimbOptions()
    
    print("="*60)
    print("🎰 IREN STEADY CLIMB OPTIONS")
    print("="*60)
    
    status = climber.get_status()
    info = status['position_info']
    
    print(f"\nPosition: {info['position']}/8")
    print(f"Contracts: {info['contracts']}x")
    print(f"Progression: {info['progression_display']}")
    print(f"Total Profit: ${info['total_profit']:.2f}")
    print(f"Win Rate: {info['win_rate']:.1f}%")
    
    print(f"\n📊 GENERATING SIGNAL...")
    signal = climber.generate_signal()
    
    if signal['action'] == 'BUY_CALL':
        opt = signal['option']
        print(f"\n🎯 SIGNAL: {signal['action']}")
        print(f"   Option: IREN ${opt['strike']}C exp {opt['expiration']}")
        print(f"   DTE: {opt['dte']} days {'⭐ PAUL PICK' if opt['is_pauls_pick'] else ''}")
        print(f"   Contracts: {signal['contracts']}")
        print(f"   Entry: ${signal['entry_price']:.2f}")
        print(f"   TP: ${signal['take_profit']:.2f} (+{climber.take_profit_pct}%)")
        print(f"   SL: ${signal['stop_loss']:.2f} (-{climber.stop_loss_pct}%)")
        print(f"   Total Cost: ${signal['total_cost']:.2f}")
        print(f"   Max Profit: ${signal['max_profit']:.2f}")
        print(f"   Max Loss: ${signal['max_loss']:.2f}")
        print(f"   R:R: {signal['risk_reward']:.2f}")
    else:
        print(f"\n⏸ {signal['action']}: {signal.get('reason', 'N/A')}")
    
    return signal


if __name__ == "__main__":
    main()

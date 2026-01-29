#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                           NEO CALL TRACKER
            "NEO makes the call → We track it → We learn from it"
═══════════════════════════════════════════════════════════════════════════════

NEO's Job:
1. Receive intel (events, patterns, levels, market data)
2. Analyze and make a CALL (entry, SL, TP, R:R)
3. Log the call with timestamp
4. Track outcome (win/loss/pending)
5. Update Command Center with mode recommendation

This is what Crella does on MT5 → +$100k in a week!

═══════════════════════════════════════════════════════════════════════════════
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NEO')

# Files
CALLS_FILE = Path(__file__).parent / 'neo_calls.json'
INTEL_FILE = Path(__file__).parent / 'command_center_intel.json'


@dataclass
class NeoCall:
    """A specific trade call from NEO"""
    id: str
    timestamp: str
    
    # The Setup
    setup_name: str          # FADE_THE_SQUEEZE, GAP_FILL, BREAKOUT, etc.
    setup_description: str
    
    # The Call
    direction: str           # LONG, SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float = 0
    
    # Risk/Reward
    risk_pips: float = 0
    reward_pips: float = 0
    risk_reward: float = 0
    
    # Mode Recommendation
    current_mode: int = 2
    trigger_mode: int = 3    # Mode to switch to when triggered
    trigger_condition: str = ''  # "price fails at 5598"
    
    # Tracking
    status: str = 'PENDING'  # PENDING, TRIGGERED, HIT_TP, HIT_SL, CANCELLED
    triggered_at: str = ''
    outcome_price: float = 0
    outcome_pnl: float = 0
    notes: str = ''


class NeoCallTracker:
    """
    NEO's brain - receives intel, makes calls, tracks outcomes
    """
    
    def __init__(self):
        self.calls: List[Dict] = []
        self._load_calls()
    
    def _load_calls(self):
        """Load saved calls"""
        if CALLS_FILE.exists():
            try:
                with open(CALLS_FILE, 'r') as f:
                    self.calls = json.load(f)
            except:
                self.calls = []
    
    def _save_calls(self):
        """Save calls"""
        with open(CALLS_FILE, 'w') as f:
            json.dump(self.calls, f, indent=2)
    
    def make_call(
        self,
        setup_name: str,
        setup_description: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float = 0,
        current_mode: int = 2,
        trigger_mode: int = 3,
        trigger_condition: str = '',
    ) -> NeoCall:
        """
        NEO makes a trade call
        """
        # Calculate risk/reward
        if direction == 'SHORT':
            risk_pips = abs(stop_loss - entry_price)
            reward_pips = abs(entry_price - take_profit_1)
        else:
            risk_pips = abs(entry_price - stop_loss)
            reward_pips = abs(take_profit_1 - entry_price)
        
        risk_reward = reward_pips / risk_pips if risk_pips > 0 else 0
        
        call = NeoCall(
            id=f"NEO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            setup_name=setup_name,
            setup_description=setup_description,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            risk_reward=round(risk_reward, 2),
            current_mode=current_mode,
            trigger_mode=trigger_mode,
            trigger_condition=trigger_condition,
            status='PENDING',
        )
        
        self.calls.append(asdict(call))
        self._save_calls()
        
        # Update Command Center
        self._update_command_center(call)
        
        logger.info(f"🎯 NEO CALL: {setup_name} - {direction} @ ${entry_price}")
        logger.info(f"   SL: ${stop_loss} | TP1: ${take_profit_1} | R:R = 1:{risk_reward:.1f}")
        
        return call
    
    def _update_command_center(self, call: NeoCall):
        """Push NEO's call to Command Center"""
        try:
            if INTEL_FILE.exists():
                with open(INTEL_FILE, 'r') as f:
                    intel = json.load(f)
            else:
                intel = {}
            
            intel['neo_intel'] = {
                'active_call': call.setup_name,
                'direction': call.direction,
                'entry': call.entry_price,
                'stop_loss': call.stop_loss,
                'take_profit': call.take_profit_1,
                'risk_reward': f"1:{call.risk_reward}",
                'current_mode': call.current_mode,
                'trigger_mode': call.trigger_mode,
                'trigger_condition': call.trigger_condition,
                'mm_prediction': call.setup_description,
                'hunt_direction': 'UP' if call.direction == 'SHORT' else 'DOWN',
                'best_action': call.setup_name,
                'received_at': datetime.now().isoformat(),
            }
            
            with open(INTEL_FILE, 'w') as f:
                json.dump(intel, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update command center: {e}")
    
    def trigger_call(self, call_id: str, price: float):
        """Mark a call as triggered"""
        for call in self.calls:
            if call['id'] == call_id:
                call['status'] = 'TRIGGERED'
                call['triggered_at'] = datetime.now().isoformat()
                call['notes'] = f"Triggered at ${price}"
                self._save_calls()
                logger.info(f"⚡ TRIGGERED: {call['setup_name']} @ ${price}")
                return
    
    def close_call(self, call_id: str, outcome: str, price: float, pnl: float = 0):
        """Close a call with outcome"""
        for call in self.calls:
            if call['id'] == call_id:
                call['status'] = outcome  # HIT_TP, HIT_SL, MANUAL_CLOSE
                call['outcome_price'] = price
                call['outcome_pnl'] = pnl
                self._save_calls()
                emoji = '✅' if outcome == 'HIT_TP' else '❌'
                logger.info(f"{emoji} CLOSED: {call['setup_name']} - {outcome} @ ${price} (P&L: ${pnl})")
                return
    
    def get_active_calls(self) -> List[Dict]:
        """Get pending/triggered calls"""
        return [c for c in self.calls if c['status'] in ['PENDING', 'TRIGGERED']]
    
    def get_scorecard(self) -> Dict:
        """NEO's performance scorecard"""
        closed = [c for c in self.calls if c['status'] in ['HIT_TP', 'HIT_SL', 'MANUAL_CLOSE']]
        wins = [c for c in closed if c['status'] == 'HIT_TP']
        losses = [c for c in closed if c['status'] == 'HIT_SL']
        
        total_pnl = sum(c.get('outcome_pnl', 0) for c in closed)
        
        return {
            'total_calls': len(self.calls),
            'active': len(self.get_active_calls()),
            'closed': len(closed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': f"{len(wins)/len(closed)*100:.1f}%" if closed else "N/A",
            'total_pnl': total_pnl,
        }
    
    def get_status_display(self) -> str:
        """Display NEO's current calls"""
        active = self.get_active_calls()
        scorecard = self.get_scorecard()
        
        output = f"""
═══════════════════════════════════════════════════════════════════════════════
                              NEO CALLS
═══════════════════════════════════════════════════════════════════════════════

  📊 SCORECARD:
     Total Calls: {scorecard['total_calls']}
     Active: {scorecard['active']}
     Win Rate: {scorecard['win_rate']}
     Total P&L: ${scorecard['total_pnl']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🎯 ACTIVE CALLS:
"""
        if active:
            for call in active:
                status_emoji = '⏳' if call['status'] == 'PENDING' else '⚡'
                output += f"""
  {status_emoji} {call['setup_name']} ({call['status']})
     Direction: {call['direction']}
     Entry: ${call['entry_price']} | SL: ${call['stop_loss']} | TP1: ${call['take_profit_1']}
     R:R = 1:{call['risk_reward']}
     Trigger: {call['trigger_condition']}
     Mode: Stay {call['current_mode']} → Switch to {call['trigger_mode']} when triggered
"""
        else:
            output += "     (no active calls)\n"
        
        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Call History:
"""
        # Last 5 calls
        for call in self.calls[-5:]:
            status_emoji = {'PENDING': '⏳', 'TRIGGERED': '⚡', 'HIT_TP': '✅', 'HIT_SL': '❌'}.get(call['status'], '❓')
            date = call['timestamp'][:10]
            output += f"     {date} | {status_emoji} {call['setup_name']} | {call['direction']} @ ${call['entry_price']} | {call['status']}\n"
        
        output += """
═══════════════════════════════════════════════════════════════════════════════
"""
        return output


def main():
    """NEO Call Tracker CLI"""
    import sys
    
    neo = NeoCallTracker()
    
    if len(sys.argv) < 2:
        print(neo.get_status_display())
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'status':
        print(neo.get_status_display())
    
    elif cmd == 'call':
        # Pre-built calls
        if len(sys.argv) >= 3:
            call_type = sys.argv[2].upper()
            
            if call_type == 'FADE_SQUEEZE':
                neo.make_call(
                    setup_name='FADE_THE_SQUEEZE',
                    setup_description='MMs will squeeze shorts above $5598 then reverse down',
                    direction='SHORT',
                    entry_price=5598,
                    stop_loss=5613,
                    take_profit_1=5409,
                    take_profit_2=5200,
                    current_mode=2,
                    trigger_mode=3,
                    trigger_condition='Price fails at $5598 and reverses',
                )
                print("✅ NEO CALL LOGGED: FADE_THE_SQUEEZE")
            
            elif call_type == 'GAP_FILL':
                neo.make_call(
                    setup_name='GAP_FILL_ACCUMULATE',
                    setup_description='Accumulate longs as price fills gap to $4650',
                    direction='LONG',
                    entry_price=5200,
                    stop_loss=4500,
                    take_profit_1=5600,
                    take_profit_2=6000,
                    current_mode=3,
                    trigger_mode=1,
                    trigger_condition='Price reaches gap fill zone',
                )
                print("✅ NEO CALL LOGGED: GAP_FILL_ACCUMULATE")
            
            else:
                print(f"Unknown call type: {call_type}")
                print("Available: FADE_SQUEEZE, GAP_FILL")
    
    elif cmd == 'trigger':
        if len(sys.argv) >= 4:
            call_id = sys.argv[2]
            price = float(sys.argv[3])
            neo.trigger_call(call_id, price)
    
    elif cmd == 'close':
        if len(sys.argv) >= 5:
            call_id = sys.argv[2]
            outcome = sys.argv[3].upper()
            price = float(sys.argv[4])
            pnl = float(sys.argv[5]) if len(sys.argv) > 5 else 0
            neo.close_call(call_id, outcome, price, pnl)
    
    elif cmd == 'scorecard':
        sc = neo.get_scorecard()
        print(f"\n📊 NEO SCORECARD:")
        for k, v in sc.items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()

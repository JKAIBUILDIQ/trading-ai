#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    GRID CONTROL - Mode Switching
═══════════════════════════════════════════════════════════════════════════════

Quick commands to switch grid modes:

  python3 grid_control.py bullish     # Activate Bullish Grid (BUY only)
  python3 grid_control.py bearish     # Activate Bearish Setup (SHORT only)  
  python3 grid_control.py correction  # Activate Correction Grid (both ways)
  python3 grid_control.py status      # Show current mode

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from pathlib import Path
from datetime import datetime

STATE_FILE = Path(__file__).parent / 'whipsaw_state.json'


def load_state():
    """Load current state"""
    with open(STATE_FILE, 'r') as f:
        return json.load(f)


def save_state(state):
    """Save state"""
    state['last_update'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def activate_bullish():
    """
    BULLISH GRID MODE
    - BUY orders: ACTIVE
    - SHORT orders: BLOCKED
    - Use when: SuperTrend bullish, buying dips
    """
    state = load_state()
    state['bear_flag_mode'] = False
    state['grid_mode'] = 'BULLISH'
    state['buy_enabled'] = True
    state['short_enabled'] = False
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
                    📈 BULLISH GRID ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  BUY orders:   ✅ ACTIVE (accumulate dips)
  SHORT orders: ❌ BLOCKED

  Strategy: Buy the dips, ride the trend
  
  Grid Levels:
    $5,551 ── BUY 2 ✅
    $5,531 ── BUY 2 ✅
    $5,511 ── BUY 4 ✅
    ...all BUY levels active
    
    $5,591 ── SHORT 🚫 BLOCKED
    $5,611 ── SHORT 🚫 BLOCKED
    ...all SHORT levels blocked

═══════════════════════════════════════════════════════════════════════════════
""")


def activate_bearish():
    """
    BEARISH SETUP MODE (Bear Flag)
    - BUY orders: BLOCKED
    - SHORT orders: ACTIVE
    - Use when: Bear flag, correction expected
    """
    state = load_state()
    state['bear_flag_mode'] = True
    state['grid_mode'] = 'BEARISH'
    state['buy_enabled'] = False
    state['short_enabled'] = True
    state['bear_flag_invalidation_price'] = 5611  # Debunk level
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
                    📉 BEARISH SETUP ACTIVATED (Bear Flag Mode)
═══════════════════════════════════════════════════════════════════════════════

  BUY orders:   ❌ BLOCKED (no new longs)
  SHORT orders: ✅ ACTIVE (fade rises)

  Strategy: Lean into correction, SHORT only
  Auto-debunk: If price >= $5,611, switches to CORRECTION mode
  
  Grid Levels:
    $5,591 ── SHORT 2 ✅
    $5,611 ── SHORT 2 ✅ (also debunk trigger)
    $5,631 ── SHORT 4 ✅
    ...all SHORT levels active
    
    $5,551 ── BUY 🚫 BLOCKED
    $5,531 ── BUY 🚫 BLOCKED
    ...all BUY levels blocked

═══════════════════════════════════════════════════════════════════════════════
""")


def activate_correction():
    """
    CORRECTION/WHIPSAW GRID MODE
    - BUY orders: ACTIVE
    - SHORT orders: ACTIVE
    - Use when: Choppy market, profit from volatility
    """
    state = load_state()
    state['bear_flag_mode'] = False
    state['grid_mode'] = 'CORRECTION'
    state['buy_enabled'] = True
    state['short_enabled'] = True
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
                    🔄 CORRECTION GRID ACTIVATED (Whipsaw Mode)
═══════════════════════════════════════════════════════════════════════════════

  BUY orders:   ✅ ACTIVE (accumulate dips)
  SHORT orders: ✅ ACTIVE (fade rises)

  Strategy: Profit from BOTH directions - the chop is the opportunity!
  
  Grid Levels:
    $5,591 ── SHORT 2 ✅  TP @ $5,571
    $5,611 ── SHORT 2 ✅  TP @ $5,591
    ...
    ════ CENTER $5,571 ════
    ...
    $5,551 ── BUY 2 ✅    TP @ $5,571
    $5,531 ── BUY 2 ✅    TP @ $5,551
    
  Every $20 move = profit opportunity on BOTH sides

═══════════════════════════════════════════════════════════════════════════════
""")


def show_status():
    """Show current mode status"""
    state = load_state()
    
    mode = state.get('grid_mode', 'UNKNOWN')
    bear_flag = state.get('bear_flag_mode', False)
    buy_enabled = state.get('buy_enabled', True)
    short_enabled = state.get('short_enabled', True)
    
    # Determine actual mode from flags
    if bear_flag:
        mode = 'BEARISH'
    elif buy_enabled and short_enabled:
        mode = 'CORRECTION'
    elif buy_enabled and not short_enabled:
        mode = 'BULLISH'
    elif not buy_enabled and short_enabled:
        mode = 'BEARISH'
    
    mode_icon = {
        'BULLISH': '📈',
        'BEARISH': '📉',
        'CORRECTION': '🔄',
    }.get(mode, '❓')
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
                    GRID STATUS
═══════════════════════════════════════════════════════════════════════════════

  Current Mode: {mode_icon} {mode}
  
  BUY orders:   {'✅ ACTIVE' if buy_enabled else '❌ BLOCKED'}
  SHORT orders: {'✅ ACTIVE' if short_enabled else '❌ BLOCKED'}
  Bear Flag:    {'🐻 YES' if bear_flag else 'No'}
  
  Position:
    Long contracts:  {state.get('long_contracts', 0)}
    Short contracts: {state.get('short_contracts', 0)}
  
  P&L:
    Long TP profit:  ${state.get('long_tp_profit', 0):+,.0f}
    Short TP profit: ${state.get('short_tp_profit', 0):+,.0f}

═══════════════════════════════════════════════════════════════════════════════

  Commands:
    python3 grid_control.py bullish     # BUY only
    python3 grid_control.py bearish     # SHORT only (bear flag)
    python3 grid_control.py correction  # Both directions
    python3 grid_control.py status      # Show this

═══════════════════════════════════════════════════════════════════════════════
""")


def activate_supertrend():
    """
    SUPERTREND DEFAULT MODE
    - Follow SuperTrend direction (currently BULLISH)
    - No pattern override active
    - Use when: Pattern invalidated, return to baseline
    """
    state = load_state()
    state['bear_flag_mode'] = False
    state['grid_mode'] = 'BULLISH'  # SuperTrend is bullish
    state['buy_enabled'] = True
    state['short_enabled'] = False  # Only buy in bullish supertrend
    state['pattern_override'] = None
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
                    📈 SUPERTREND MODE (Baseline)
═══════════════════════════════════════════════════════════════════════════════

  SuperTrend: BULLISH
  Pattern Override: NONE (cleared)
  
  BUY orders:   ✅ ACTIVE (buy the dips)
  SHORT orders: ❌ BLOCKED (ride the trend)

  Strategy: Follow SuperTrend - buy dips, ride the bull
  
  "No pattern spotted. Following the trend."

═══════════════════════════════════════════════════════════════════════════════
""")


def main():
    if len(sys.argv) < 2:
        show_status()
        return
    
    command = sys.argv[1].lower()
    
    if command in ['bullish', 'bull', 'long', 'buy']:
        activate_bullish()
    elif command in ['bearish', 'bear', 'short', 'sell']:
        activate_bearish()
    elif command in ['correction', 'whipsaw', 'both', 'grid', 'neutral']:
        activate_correction()
    elif command in ['supertrend', 'default', 'baseline', 'trend']:
        activate_supertrend()
    elif command in ['status', 'show', 'info']:
        show_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: bullish, bearish, correction, supertrend, or status")


if __name__ == "__main__":
    main()

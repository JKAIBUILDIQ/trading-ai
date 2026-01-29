#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    GRID CONTROL - Trading Mode Commands
                    Ghost Commander IBKR - MGC Futures
═══════════════════════════════════════════════════════════════════════════════

Quick commands to switch trading modes:

  python3 grid_control.py 1           # Activate Bullish Grid (Mode 1)
  python3 grid_control.py 2           # Activate Correction Grid (Mode 2)
  python3 grid_control.py 3           # Activate Bearish Sighting (Mode 3)
  python3 grid_control.py status      # Show current mode

Voice/Text Commands:
  "Activate Bullish Grid"      → Mode 1
  "Activate Correction Grid"   → Mode 2  
  "Activate Bearish Sighting"  → Mode 3

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


def save_state_mode_only(updates: dict):
    """
    Save ONLY mode flags without resetting levels!
    This preserves buy_levels and short_levels arrays.
    """
    state = load_state()
    
    # Only update the mode-related fields
    for key, value in updates.items():
        state[key] = value
    
    state['last_update'] = datetime.now().isoformat()
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def mode_1_bullish():
    """
    MODE 1: BULLISH GRID (Default)
    
    All BUYs with DCA on drops. Normal bullish trading - buying dips.
    SHORT grid can scalp but bias is LONG.
    """
    save_state_mode_only({
        'trading_mode': 1,
        'grid_mode': 'BULLISH',
        'buy_enabled': True,      # ✅ DCA every $20 drop
        'short_enabled': True,    # ✅ Scalp shorts active
        'bear_flag_mode': False,
        'hedge_active': False,    # ❌ No hedge
        'pattern_override': None,
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📈 MODE 1: BULLISH GRID ACTIVATED (Default)
═══════════════════════════════════════════════════════════════════════════════

  When to use: SuperTrend bullish, no warning signs, normal conditions

  ✅ DCA BUY ladder:    ACTIVE (buy every $20 drop)
  ✅ Grid LONG levels:  ACTIVE (auto-buy at grid levels)
  ✅ Grid SHORT levels: ACTIVE (auto-scalp on rises)
  ❌ Hedge SELL:        OFF

  Grid:
    $5,611 ─── SHORT 2 ✅ scalp
    $5,591 ─── SHORT 2 ✅ scalp
         ══ CENTER ══
    $5,551 ─── BUY 2 ✅ 
    $5,531 ─── BUY 2 ✅
    ...all levels active

  "Normal bullish trading - buy dips, scalp rises"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_2_correction():
    """
    MODE 2: CORRECTION GRID
    
    FULL HEDGE POSITION on top of grid.
    Favors correction down to gap fills/necklines.
    Grid still trades both ways. Bias is EXPECTING DOWNSIDE but still accumulating.
    """
    save_state_mode_only({
        'trading_mode': 2,
        'grid_mode': 'CORRECTION',
        'buy_enabled': True,      # ✅ Accumulate on way down
        'short_enabled': True,    # ✅ Fade bounces
        'bear_flag_mode': False,
        'hedge_active': True,     # ✅ FULL HEDGE expecting drop
        'pattern_override': 'CORRECTION',
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📊 MODE 2: CORRECTION GRID ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  When to use: Overextended but trend still bullish, want to hedge profits

  ✅ DCA BUY ladder:    ACTIVE (keep buying dips)
  ✅ Grid LONG levels:  ACTIVE
  ✅ Grid SHORT levels: ACTIVE
  ✅ Hedge SELL:        ACTIVE (protection)

  Example scenarios:
    • Gold parabolic (+20% in 2 weeks)
    • RSI overbought (85+)
    • FOMC tomorrow
    • Want protection but still bullish long-term

  Grid: ALL levels active BOTH directions
  
  "Hedged but still bullish - protect profits, keep buying dips"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_3_bearish():
    """
    MODE 3: BEARISH SIGHTING
    
    Bear signal spotted (bear flag, breakdown).
    STOPS any new buys completely. Shorts only - ride the drop.
    """
    save_state_mode_only({
        'trading_mode': 3,
        'grid_mode': 'BEARISH',
        'buy_enabled': False,     # ❌ NO NEW BUYS
        'short_enabled': True,    # ✅ Profit from drops
        'bear_flag_mode': True,
        'hedge_active': True,     # ✅ Hedge active
        'bear_flag_invalidation_price': 5611,
        'pattern_override': 'BEAR_FLAG',
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            🐻 MODE 3: BEARISH SIGHTING ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  When to use: Bear flag, divergence, breakdown imminent

  ❌ DCA BUY ladder:    STOPPED (no new longs)
  ❌ Grid LONG levels:  STOPPED
  ✅ Grid SHORT levels: ACTIVE (profit from drops)
  ✅ Hedge SELL:        ACTIVE

  Example scenarios:
    • Bear flag pattern forming
    • RSI divergence (price higher, RSI lower)
    • Major support about to break
    • "What goes up must come down"

  Grid:
    $5,611 ─── SHORT 2 ✅
    $5,591 ─── SHORT 2 ✅
         ══ CENTER ══
    $5,551 ─── BUY 🚫 BLOCKED
    $5,531 ─── BUY 🚫 BLOCKED
    ...all BUY levels blocked

  Exit criteria:
    • Pattern breaks down → Keep mode 3, ride shorts
    • Pattern invalidated (price >= $5,611) → Switch to mode 1

  "Bearish sighting - shorts only, waiting for breakdown"

═══════════════════════════════════════════════════════════════════════════════
""")


def show_status():
    """Show current mode status"""
    state = load_state()
    
    mode = state.get('trading_mode', 1)
    grid_mode = state.get('grid_mode', 'BULLISH')
    buy_enabled = state.get('buy_enabled', True)
    short_enabled = state.get('short_enabled', True)
    hedge_active = state.get('hedge_active', False)
    pattern = state.get('pattern_override', None)
    
    mode_names = {
        1: ('📈 BULLISH GRID', 'Normal bullish trading'),
        2: ('📊 CORRECTION GRID', 'Hedged, still buying dips'),
        3: ('🐻 BEARISH SIGHTING', 'Shorts only, BUYs blocked'),
    }
    
    mode_name, mode_desc = mode_names.get(mode, ('❓ UNKNOWN', ''))
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
                    GRID STATUS - Ghost Commander IBKR
═══════════════════════════════════════════════════════════════════════════════

  Current Mode:    {mode_name} (Mode {mode})
  Description:     {mode_desc}
  Pattern Override: {pattern or 'None'}
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ DCA BUY ladder:    {'✅ ACTIVE' if buy_enabled else '❌ STOPPED':20} │
  │ Grid LONG levels:  {'✅ ACTIVE' if buy_enabled else '❌ STOPPED':20} │
  │ Grid SHORT levels: {'✅ ACTIVE' if short_enabled else '❌ STOPPED':20} │
  │ Hedge SELL:        {'✅ ACTIVE' if hedge_active else '❌ OFF':20} │
  └─────────────────────────────────────────────────────────────────────────┘
  
  Position:
    Long contracts:  {state.get('long_contracts', 0)}
    Short contracts: {state.get('short_contracts', 0)}
  
  P&L:
    Long TP profit:  ${state.get('long_tp_profit', 0):+,.0f}
    Short TP profit: ${state.get('short_tp_profit', 0):+,.0f}

═══════════════════════════════════════════════════════════════════════════════

  COMMANDS:
  
  │ Command                          │ Mode │ Description              │
  ├──────────────────────────────────┼──────┼──────────────────────────┤
  │ python3 grid_control.py 1        │  1   │ Activate Bullish Grid    │
  │ python3 grid_control.py 2        │  2   │ Activate Correction Grid │
  │ python3 grid_control.py 3        │  3   │ Activate Bearish Sighting│
  │ python3 grid_control.py status   │  -   │ Show this status         │
  └──────────────────────────────────┴──────┴──────────────────────────┘

  VOICE COMMANDS:
    "Activate Bullish Grid"      → Mode 1
    "Activate Correction Grid"   → Mode 2
    "Activate Bearish Sighting"  → Mode 3

═══════════════════════════════════════════════════════════════════════════════
""")


def main():
    if len(sys.argv) < 2:
        show_status()
        return
    
    command = sys.argv[1].lower()
    
    # Mode numbers
    if command == '1':
        mode_1_bullish()
    elif command == '2':
        mode_2_correction()
    elif command == '3':
        mode_3_bearish()
    
    # Voice command keywords
    elif command in ['bullish', 'bull', 'long', 'buy', 'normal', 'default']:
        mode_1_bullish()
    elif command in ['correction', 'hedge', 'protect', 'fomc']:
        mode_2_correction()
    elif command in ['bearish', 'bear', 'short', 'sell', 'sighting']:
        mode_3_bearish()
    
    # Status
    elif command in ['status', 'show', 'info', 's']:
        show_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: 1, 2, 3, or status")
        print('  1 = "Activate Bullish Grid"')
        print('  2 = "Activate Correction Grid"')
        print('  3 = "Activate Bearish Sighting"')


if __name__ == "__main__":
    main()

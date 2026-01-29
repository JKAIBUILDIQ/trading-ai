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
    MODE 1: BULLISH (Trend Following)
    Goal: MAXIMIZE GAINS
    
    Full confidence in uptrend. DCA on dips, TP on way back up.
    
    REQUIRED BASE POSITION: 1 LONG (freeroll)
    - Never miss runups
    - Ride downturns with DCA
    """
    save_state_mode_only({
        'trading_mode': 1,
        'grid_mode': 'BULLISH',
        'buy_enabled': True,      # ✅ DCA every $20 drop
        'short_enabled': False,   # ❌ No shorts - directional mode
        'bear_flag_mode': False,
        'hedge_active': False,    # ❌ No hedge - we're bullish
        'pattern_override': None,
        'required_position': 'LONG',
        'required_contracts': 1,
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📈 MODE 1: BULLISH GRID ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  When to use: SuperTrend bullish, no warning signs, normal conditions

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🎯 REQUIRED BASE POSITION: 1 LONG (FREEROLL)                          │
  │     Purpose: Never miss runups, ride downturns with DCA                 │
  └─────────────────────────────────────────────────────────────────────────┘

  ✅ Freeroll LONG:     REQUIRED (always have 1 long)
  ✅ DCA BUY ladder:    ACTIVE (add on every $20 drop)
  ❌ Grid SHORT levels: OFF (no shorts - we're bullish)
  ❌ Hedge SELL:        OFF

  Strategy:
    • Freeroll long captures any breakout
    • DCA adds contracts on dips → bigger position for recovery
    • TP as price returns → lock profits → rebuy lower if dips again

  "Freeroll long + DCA dips = never miss the move"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_2_correction():
    """
    MODE 2: CORRECTION (Choppy/Sideways)
    Goal: SAFEGUARD AGAINST LOSSES + Profit from Chop
    
    Market is choppy/uncertain. Use GRID to profit from oscillations both ways.
    
    REQUIRED BASE POSITION: 1 SELL (hedge)
    - Downside protection
    - Profit if correction happens
    """
    save_state_mode_only({
        'trading_mode': 2,
        'grid_mode': 'CORRECTION',
        'buy_enabled': True,      # ✅ Grid BUYs at support
        'short_enabled': True,    # ✅ Grid SHORTs at resistance
        'bear_flag_mode': False,
        'hedge_active': True,     # ✅ Hedge for insurance
        'pattern_override': 'CORRECTION',
        'required_position': 'SELL',
        'required_contracts': 1,
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📊 MODE 2: CORRECTION GRID ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  When to use: Overextended but trend still bullish, want to hedge profits

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🎯 REQUIRED BASE POSITION: 1 SELL (HEDGE)                             │
  │     Purpose: Downside protection while still participating in upside    │
  └─────────────────────────────────────────────────────────────────────────┘

  ✅ Hedge SELL:        REQUIRED (1 full contract for protection)
  ✅ Grid BUY levels:   ACTIVE (catch dips)
  ✅ Grid SHORT levels: ACTIVE (scalp bounces)
  ✅ DCA BUY ladder:    ACTIVE (accumulate on drops)

  Example scenarios:
    • Gold parabolic (+20% in 2 weeks)
    • RSI overbought (85+)
    • FOMC tomorrow ← WE ARE HERE
    • Bear flag forming but not confirmed

  Strategy:
    • Hedge SELL profits if correction happens
    • Grid catches oscillations both ways
    • Still accumulating longs for the eventual continuation

  "Hedged + grid both ways = profit from chop, protected from crash"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_3_bearish():
    """
    MODE 3: BEARISH (Supertrend Switch)
    Goal: KILL LONGS, SCALE IN SHORTS
    
    Trend is reversing. STOP all new buys. Ride the new trend DOWN.
    
    REQUIRED BASE POSITION: SELL + add more on bounces
    - Already short from hedge
    - ADD to shorts as price bounces
    - Ride the reversal down
    """
    save_state_mode_only({
        'trading_mode': 3,
        'grid_mode': 'BEARISH',
        'buy_enabled': False,     # ❌ STOP NEW BUYS
        'short_enabled': True,    # ✅ Scale in shorts on bounces
        'bear_flag_mode': True,
        'hedge_active': True,     # ✅ Hedge active
        'bear_flag_invalidation_price': 5611,
        'pattern_override': 'BEARISH',
        'required_position': 'SELL',
        'required_contracts': 2,  # Base + add
    })
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            🐻 MODE 3: BEARISH SIGHTING ACTIVATED
═══════════════════════════════════════════════════════════════════════════════

  When to use: Bear flag confirmed, squeeze failed, reversal confirmed

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🎯 REQUIRED BASE POSITION: SELL + ADD MORE                            │
  │     Purpose: Ride the reversal down, scale in on bounces               │
  └─────────────────────────────────────────────────────────────────────────┘

  ✅ Base SELL:         REQUIRED (already have from Mode 2 hedge)
  ✅ ADD shorts:        ACTIVE (scale in on every bounce)
  ❌ DCA BUY ladder:    STOPPED (no new longs!)
  ❌ Grid LONG levels:  STOPPED

  Example scenarios:
    • Squeeze failed at $5598 ← NEO's call
    • Bear flag breakdown confirmed
    • SuperTrend flipped bearish
    • Major support broken

  Strategy:
    • Already short from hedge → let it ride
    • ADD shorts on bounces → bigger position for the drop
    • Target: $5409 (NEO's TP1) → $5200 → $4650 (THE GAP)

  Grid (SHORT only):
    $5,598 ─── SHORT +1 ✅ (NEO's entry)
    $5,611 ─── SHORT +1 ✅ (scale in)
         ══ PRICE BELOW ══
    $5,551 ─── BUY 🚫 BLOCKED
    ...all BUY levels blocked

  "Shorts only, ride it down, add on bounces"

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
    required_pos = state.get('required_position', 'NONE')
    required_qty = state.get('required_contracts', 0)
    
    mode_names = {
        1: ('📈 BULLISH GRID', 'Freeroll long + DCA dips'),
        2: ('📊 CORRECTION GRID', 'Hedge sell + grid both ways'),
        3: ('🐻 BEARISH SIGHTING', 'Sell + add shorts on bounces'),
    }
    
    mode_name, mode_desc = mode_names.get(mode, ('❓ UNKNOWN', ''))
    
    # Required position display
    required_display = {
        1: '🎯 1 LONG (freeroll) - never miss runups',
        2: '🎯 1 SELL (hedge) - downside protection',
        3: '🎯 SELL + ADD - ride reversal down',
    }
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
                    GRID STATUS - Ghost Commander IBKR
═══════════════════════════════════════════════════════════════════════════════

  Current Mode:       {mode_name} (Mode {mode})
  Description:        {mode_desc}
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  REQUIRED BASE POSITION:                                                │
  │  {required_display.get(mode, 'NONE'):66} │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ DCA BUY ladder:    {'✅ ACTIVE' if buy_enabled else '❌ STOPPED':20} │
  │ Grid LONG levels:  {'✅ ACTIVE' if buy_enabled else '❌ STOPPED':20} │
  │ Grid SHORT levels: {'✅ ACTIVE' if short_enabled else '❌ STOPPED':20} │
  │ Hedge SELL:        {'✅ ACTIVE' if hedge_active else '❌ OFF':20} │
  └─────────────────────────────────────────────────────────────────────────┘
  
  Current Position:
    Long contracts:  {state.get('long_contracts', 0)}
    Short contracts: {state.get('short_contracts', 0)}
  
  Realized P&L:
    Long TP profit:  ${state.get('long_tp_profit', 0):+,.0f}
    Short TP profit: ${state.get('short_tp_profit', 0):+,.0f}

═══════════════════════════════════════════════════════════════════════════════

  MODE RULES (FIRST RULE OF EACH MODE):
  
  │ Mode │ Required Position │ Purpose                          │
  ├──────┼───────────────────┼──────────────────────────────────┤
  │  1   │ 1 LONG (freeroll) │ Never miss runups, ride dips     │
  │  2   │ 1 SELL (hedge)    │ Downside protection              │
  │  3   │ SELL + ADD        │ Ride reversal, scale in shorts   │
  └──────┴───────────────────┴──────────────────────────────────┘

  COMMANDS:
  
  │ python3 grid_control.py 1        │ Bullish - freeroll long          │
  │ python3 grid_control.py 2        │ Correction - hedge sell          │
  │ python3 grid_control.py 3        │ Bearish - sell + add shorts      │
  │ python3 grid_control.py status   │ Show this status                 │
  └──────────────────────────────────┴──────────────────────────────────┘

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

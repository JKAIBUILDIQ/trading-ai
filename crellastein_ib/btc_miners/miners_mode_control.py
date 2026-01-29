#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    BTC MINERS MODE CONTROL
                    CIFR | CLSK | IREN - Options DCA Strategy
═══════════════════════════════════════════════════════════════════════════════

Mode 1: BULLISH DCA
  - Always have exposure (freeroll position)
  - DCA by selling more options as premium drops
  - Let winners run

Mode 2: CORRECTION
  - Hedge with protective puts
  - Smaller position sizing
  - Wider DCA spacing

Mode 3: BEARISH
  - Close longs, no new sells
  - Wait for setup

Commands:
    python miners_mode_control.py 1        # Bullish DCA
    python miners_mode_control.py 2        # Correction
    python miners_mode_control.py 3        # Bearish
    python miners_mode_control.py status   # Show status

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from pathlib import Path
from datetime import datetime

STATE_FILE = Path(__file__).parent / 'miners_state.json'
SETUP_FILE = Path(__file__).parent / 'MODE1_DCA_SETUP.json'


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'mode': 1,
        'mode_name': 'BULLISH_DCA',
        'dca_enabled': True,
        'hedge_enabled': False,
        'symbols': ['CIFR', 'CLSK', 'IREN'],
    }


def save_state(state):
    state['last_update'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def mode_1_bullish():
    """Mode 1: Bullish DCA - Scale in on dips"""
    state = load_state()
    state.update({
        'mode': 1,
        'mode_name': 'BULLISH_DCA',
        'dca_enabled': True,
        'hedge_enabled': False,
        'new_sells_enabled': True,
        'close_longs': False,
    })
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📈 MODE 1: BULLISH DCA - BTC MINERS
═══════════════════════════════════════════════════════════════════════════════

  Symbols: CIFR | CLSK | IREN

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🎯 REQUIRED: Always have exposure (freeroll)                          │
  │  Strategy: Sell options, DCA as premium drops                          │
  └─────────────────────────────────────────────────────────────────────────┘

  ✅ DCA Selling:     ACTIVE (scale in on premium drops)
  ✅ New Positions:   ACTIVE (sell puts/calls on dips)
  ❌ Hedge:           OFF (full bullish)
  ✅ Let Winners Run: Yes

  DCA Rules:
    • Premium drops 20% → Add HALF position
    • Premium drops 40% → Add FULL position
    • Premium drops 60% → Add DOUBLE position

  "Stay long, DCA dips, let winners run"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_2_correction():
    """Mode 2: Correction - Hedge and reduce"""
    state = load_state()
    state.update({
        'mode': 2,
        'mode_name': 'CORRECTION',
        'dca_enabled': True,
        'hedge_enabled': True,
        'new_sells_enabled': True,
        'close_longs': False,
    })
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            📊 MODE 2: CORRECTION - BTC MINERS
═══════════════════════════════════════════════════════════════════════════════

  Symbols: CIFR | CLSK | IREN

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🎯 REQUIRED: Hedge with protective puts                               │
  │  Strategy: Smaller sizing, wider DCA spacing                           │
  └─────────────────────────────────────────────────────────────────────────┘

  ✅ DCA Selling:     ACTIVE (but smaller size)
  ✅ New Positions:   ACTIVE (cautious)
  ✅ Hedge:           ACTIVE (buy protective puts)
  ⚠️ Sizing:          HALF normal size

  "Protected but still participating"

═══════════════════════════════════════════════════════════════════════════════
""")


def mode_3_bearish():
    """Mode 3: Bearish - Close and wait"""
    state = load_state()
    state.update({
        'mode': 3,
        'mode_name': 'BEARISH',
        'dca_enabled': False,
        'hedge_enabled': True,
        'new_sells_enabled': False,
        'close_longs': True,
    })
    save_state(state)
    
    print("""
═══════════════════════════════════════════════════════════════════════════════
            🐻 MODE 3: BEARISH - BTC MINERS
═══════════════════════════════════════════════════════════════════════════════

  Symbols: CIFR | CLSK | IREN

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ⚠️ ACTION: Close longs, no new option sells                           │
  │  Strategy: Wait for setup, preserve capital                            │
  └─────────────────────────────────────────────────────────────────────────┘

  ❌ DCA Selling:     STOPPED
  ❌ New Positions:   STOPPED  
  ⚠️ Close Longs:     YES (or tight stops)
  ✅ Cash Ready:      For next opportunity

  "Preserve capital, wait for reversal"

═══════════════════════════════════════════════════════════════════════════════
""")


def show_status():
    """Show current status"""
    state = load_state()
    
    # Load setup if exists
    setup = {}
    if SETUP_FILE.exists():
        with open(SETUP_FILE, 'r') as f:
            setup = json.load(f)
    
    mode_emoji = {1: '📈', 2: '📊', 3: '🐻'}
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
                    BTC MINERS STATUS
═══════════════════════════════════════════════════════════════════════════════

  Current Mode: {mode_emoji.get(state['mode'], '❓')} {state['mode_name']} (Mode {state['mode']})

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ DCA Enabled:      {'✅ ACTIVE' if state.get('dca_enabled') else '❌ OFF':20} │
  │ Hedge Enabled:    {'✅ ACTIVE' if state.get('hedge_enabled') else '❌ OFF':20} │
  │ New Sells:        {'✅ ACTIVE' if state.get('new_sells_enabled') else '❌ OFF':20} │
  └─────────────────────────────────────────────────────────────────────────┘
""")
    
    if setup.get('symbols'):
        print("  📊 POSITIONS:")
        for sym, data in setup['symbols'].items():
            print(f"    {sym}: {data.get('current_position', 0)} @ ${data.get('current_price', 0):.2f} | P&L: ${data.get('unrealized_pnl', 0):+,.0f}")
    
    if setup.get('total_account'):
        acc = setup['total_account']
        print(f"""
  💰 ACCOUNT:
    Balance:    ${acc.get('balance', 0):,.0f}
    Total P&L:  ${acc.get('total_pnl', 0):+,.0f}
    Realized:   ${acc.get('realized', 0):+,.0f}
    Unrealized: ${acc.get('unrealized', 0):+,.0f}
""")
    
    print("""
═══════════════════════════════════════════════════════════════════════════════

  COMMANDS:
    python miners_mode_control.py 1        # Bullish DCA
    python miners_mode_control.py 2        # Correction
    python miners_mode_control.py 3        # Bearish

═══════════════════════════════════════════════════════════════════════════════
""")


def main():
    if len(sys.argv) < 2:
        show_status()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == '1' or cmd == 'bullish':
        mode_1_bullish()
    elif cmd == '2' or cmd == 'correction':
        mode_2_correction()
    elif cmd == '3' or cmd == 'bearish':
        mode_3_bearish()
    elif cmd == 'status':
        show_status()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: 1, 2, 3, or status")


if __name__ == "__main__":
    main()

"""
NEO DEFCON Controller
Manages DEFCON level and communicates with Ghost/Casper on MT5.

DEFCON LEVELS:
├── 5 (Green):  Normal - full trading
├── 4 (Blue):   Elevated - 80% sizing, wider gaps
├── 3 (Yellow): High Alert - 50% sizing, Casper hedge mode
├── 2 (Orange): Severe - Ghost paused, aggressive hedge
└── 1 (Red):    Maximum - survival mode, delta neutral

Communication via network share to Windows MT5:
- Writes to: \\\\100.119.161.65\\MT5_Share\\MT5_DEFCON.txt
- Also serves via HTTP API for redundancy
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from pathlib import Path

# Network share path to Windows MT5 machine
MT5_SHARE_PATH = "//100.119.161.65/MT5_Share"
DEFCON_FILE = "MT5_DEFCON.txt"
LOCAL_STATE_FILE = "/home/jbot/trading_ai/neo/defcon_state.json"

# Telegram alerts
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


@dataclass
class DefconState:
    level: int  # 1-5
    color: str  # RED, ORANGE, YELLOW, BLUE, GREEN
    triggers: List[str]
    set_at: str
    valid_until: str
    ghost_instructions: Dict
    casper_instructions: Dict
    notes: str = ""


# DEFCON configurations
DEFCON_CONFIG = {
    5: {
        "color": "GREEN",
        "name": "NORMAL",
        "ghost": {
            "lot_multiplier": 1.0,
            "max_positions": 5,
            "min_confidence": 40,
            "entry_gap_multiplier": 1.0,
            "pause_entries": False,
        },
        "casper": {
            "lot_multiplier": 1.0,
            "max_positions": 3,
            "min_confidence": 60,
            "hedge_mode": False,
            "hedge_percent": 0,
        }
    },
    4: {
        "color": "BLUE",
        "name": "ELEVATED",
        "ghost": {
            "lot_multiplier": 0.8,
            "max_positions": 4,
            "min_confidence": 70,
            "entry_gap_multiplier": 1.2,
            "pause_entries": False,
        },
        "casper": {
            "lot_multiplier": 0.8,
            "max_positions": 3,
            "min_confidence": 65,
            "hedge_mode": False,
            "hedge_percent": 0,
        }
    },
    3: {
        "color": "YELLOW",
        "name": "HIGH ALERT",
        "ghost": {
            "lot_multiplier": 0.5,
            "max_positions": 3,
            "min_confidence": 80,
            "entry_gap_multiplier": 2.0,
            "pause_entries": False,
        },
        "casper": {
            "lot_multiplier": 0.5,
            "max_positions": 2,
            "min_confidence": 75,
            "hedge_mode": True,
            "hedge_percent": 25,
        }
    },
    2: {
        "color": "ORANGE",
        "name": "SEVERE",
        "ghost": {
            "lot_multiplier": 0.0,
            "max_positions": 0,
            "min_confidence": 100,
            "entry_gap_multiplier": 999,
            "pause_entries": True,
            "tighten_stops": 50,
            "scale_out_percent": 30,
        },
        "casper": {
            "lot_multiplier": 0.5,
            "max_positions": 0,
            "min_confidence": 85,
            "hedge_mode": True,
            "hedge_percent": 50,
        }
    },
    1: {
        "color": "RED",
        "name": "MAXIMUM THREAT",
        "ghost": {
            "lot_multiplier": 0.0,
            "max_positions": 0,
            "min_confidence": 100,
            "entry_gap_multiplier": 999,
            "pause_entries": True,
            "survival_mode": True,
            "close_percent": 50,
            "set_breakeven": True,
        },
        "casper": {
            "lot_multiplier": 0.0,
            "max_positions": 0,
            "min_confidence": 100,
            "hedge_mode": True,
            "hedge_percent": 100,  # Full delta neutral
            "delta_neutral": True,
        }
    }
}

# DEFCON display
DEFCON_EMOJI = {
    5: "🟢",
    4: "🔵",
    3: "🟡",
    2: "🟠",
    1: "🔴"
}


def send_telegram(message: str):
    """Send Telegram alert."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram] {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")


def load_current_state() -> Optional[DefconState]:
    """Load current DEFCON state from file."""
    if os.path.exists(LOCAL_STATE_FILE):
        with open(LOCAL_STATE_FILE, 'r') as f:
            data = json.load(f)
            return DefconState(**data)
    return None


def save_state(state: DefconState):
    """Save DEFCON state to local file."""
    with open(LOCAL_STATE_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2)


def write_to_mt5(level: int) -> bool:
    """
    Write DEFCON level to MT5 share.
    Returns True if successful.
    """
    try:
        # Try network share first
        share_path = Path(MT5_SHARE_PATH) / DEFCON_FILE
        share_path.write_text(str(level))
        print(f"✅ Wrote DEFCON {level} to {share_path}")
        return True
    except Exception as e:
        print(f"⚠️ Network share failed: {e}")
        
        # Fallback: write to local file that can be synced
        local_mt5_file = Path("/home/jbot/trading_ai/neo/signals/MT5_DEFCON.txt")
        local_mt5_file.parent.mkdir(parents=True, exist_ok=True)
        local_mt5_file.write_text(str(level))
        print(f"📝 Wrote DEFCON {level} to local fallback: {local_mt5_file}")
        return True


def write_full_state_for_mt5(state: DefconState):
    """Write full state JSON for MT5 to read (more details than just the level)."""
    mt5_state = {
        "defcon": state.level,
        "color": state.color,
        "name": DEFCON_CONFIG[state.level]["name"],
        "ghost": state.ghost_instructions,
        "casper": state.casper_instructions,
        "set_at": state.set_at,
        "valid_until": state.valid_until,
        "triggers": state.triggers,
    }
    
    # Write to local (can be synced to MT5)
    state_path = Path("/home/jbot/trading_ai/neo/signals/MT5_DEFCON_STATE.json")
    with open(state_path, 'w') as f:
        json.dump(mt5_state, f, indent=2)
    
    print(f"📝 Full state written to {state_path}")


def set_defcon(
    level: int,
    triggers: List[str],
    duration_hours: int = 4,
    notes: str = ""
) -> DefconState:
    """
    Set DEFCON level with triggers and duration.
    
    Args:
        level: 1-5 (1=max threat, 5=normal)
        triggers: List of reasons for this DEFCON level
        duration_hours: How long this level should remain (default 4 hours)
        notes: Additional notes
    
    Returns:
        DefconState object
    """
    if level < 1 or level > 5:
        raise ValueError(f"DEFCON level must be 1-5, got {level}")
    
    config = DEFCON_CONFIG[level]
    current = load_current_state()
    
    # Create new state
    state = DefconState(
        level=level,
        color=config["color"],
        triggers=triggers,
        set_at=datetime.now().isoformat(),
        valid_until=(datetime.now() + timedelta(hours=duration_hours)).isoformat(),
        ghost_instructions=config["ghost"],
        casper_instructions=config["casper"],
        notes=notes
    )
    
    # Write to MT5
    write_to_mt5(level)
    write_full_state_for_mt5(state)
    save_state(state)
    
    # Send alert if level changed
    if current is None or current.level != level:
        emoji = DEFCON_EMOJI[level]
        direction = "⬆️ UPGRADE" if current and level < current.level else "⬇️ DOWNGRADE" if current else "SET"
        
        alert = f"""
{emoji} <b>DEFCON {level} - {config['name']}</b> {direction}

<b>Triggers:</b>
{chr(10).join('• ' + t for t in triggers)}

<b>Ghost:</b> {'⛔ PAUSED' if config['ghost']['pause_entries'] else f"{config['ghost']['lot_multiplier']*100:.0f}% sizing"}
<b>Casper:</b> {'🛡️ HEDGE ' + str(config['casper']['hedge_percent']) + '%' if config['casper']['hedge_mode'] else 'Normal'}

Valid for {duration_hours} hours
"""
        send_telegram(alert.strip())
        print(alert)
    
    return state


def get_current_defcon() -> Tuple[int, str]:
    """Get current DEFCON level and color."""
    state = load_current_state()
    if state:
        return state.level, state.color
    return 5, "GREEN"  # Default to normal


def format_defcon_status() -> str:
    """Format current DEFCON status for display."""
    state = load_current_state()
    if not state:
        return "🟢 DEFCON 5 (NORMAL) - No state set"
    
    emoji = DEFCON_EMOJI[state.level]
    config = DEFCON_CONFIG[state.level]
    
    lines = [
        f"{emoji} DEFCON {state.level} - {config['name']}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Set: {state.set_at[:16]}",
        f"Valid until: {state.valid_until[:16]}",
        f"",
        f"Triggers:",
    ]
    for t in state.triggers:
        lines.append(f"  • {t}")
    
    lines.extend([
        f"",
        f"Ghost Instructions:",
        f"  Lot Multiplier: {state.ghost_instructions.get('lot_multiplier', 1.0)*100:.0f}%",
        f"  Max Positions: {state.ghost_instructions.get('max_positions', 5)}",
        f"  Entries Paused: {'YES' if state.ghost_instructions.get('pause_entries') else 'NO'}",
        f"",
        f"Casper Instructions:",
        f"  Hedge Mode: {'ACTIVE' if state.casper_instructions.get('hedge_mode') else 'OFF'}",
        f"  Hedge Percent: {state.casper_instructions.get('hedge_percent', 0)}%",
    ])
    
    return "\n".join(lines)


# Pre-built DEFCON scenarios
def defcon_fomc_day():
    """Set DEFCON for FOMC day."""
    return set_defcon(
        level=3,
        triggers=[
            "FOMC rate decision today",
            "Powell press conference scheduled",
            "High volatility expected"
        ],
        duration_hours=6,
        notes="Upgrade to DEFCON 2 if hawkish, downgrade to 4 if dovish"
    )


def defcon_earnings_risk(company: str):
    """Set DEFCON for mega-cap earnings."""
    return set_defcon(
        level=4,
        triggers=[
            f"{company} earnings after market close",
            "Potential sector contagion risk",
            "Monitor after-hours reaction"
        ],
        duration_hours=18,
        notes=f"Upgrade to DEFCON 3 if {company} misses or dumps on beat"
    )


def defcon_correction_active():
    """Set DEFCON when correction is in progress."""
    return set_defcon(
        level=2,
        triggers=[
            "Active correction in progress",
            "Support levels breaking",
            "Risk-off sentiment spreading"
        ],
        duration_hours=8,
        notes="Upgrade to DEFCON 1 if -$100+ in session"
    )


def defcon_survival_mode():
    """Set DEFCON 1 - Maximum threat."""
    return set_defcon(
        level=1,
        triggers=[
            "Market crash in progress",
            "Multiple support levels broken",
            "Full risk-off across asset classes"
        ],
        duration_hours=12,
        notes="Survival mode - protect capital at all costs"
    )


def defcon_normal():
    """Return to normal operations."""
    return set_defcon(
        level=5,
        triggers=[
            "Market conditions normalized",
            "No pending high-impact events",
            "Clear trend established"
        ],
        duration_hours=24,
        notes="Full trading resumed"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python defcon_controller.py <command> [args]")
        print("Commands:")
        print("  status          - Show current DEFCON")
        print("  set <1-5>       - Set DEFCON level")
        print("  fomc            - Set DEFCON 3 for FOMC day")
        print("  earnings <CO>   - Set DEFCON 4 for earnings")
        print("  correction      - Set DEFCON 2 for active correction")
        print("  survival        - Set DEFCON 1 maximum threat")
        print("  normal          - Return to DEFCON 5")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "status":
        print(format_defcon_status())
    
    elif cmd == "set":
        if len(sys.argv) < 3:
            print("Usage: python defcon_controller.py set <1-5>")
            sys.exit(1)
        level = int(sys.argv[2])
        triggers = sys.argv[3:] if len(sys.argv) > 3 else ["Manual DEFCON set"]
        set_defcon(level, triggers)
    
    elif cmd == "fomc":
        defcon_fomc_day()
    
    elif cmd == "earnings":
        company = sys.argv[2] if len(sys.argv) > 2 else "MEGA-CAP"
        defcon_earnings_risk(company)
    
    elif cmd == "correction":
        defcon_correction_active()
    
    elif cmd == "survival":
        defcon_survival_mode()
    
    elif cmd == "normal":
        defcon_normal()
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

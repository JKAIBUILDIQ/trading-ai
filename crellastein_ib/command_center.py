#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    GHOST COMMANDER - COMMAND CENTER
                    "NEO's Intel + Your Sightings → Trading Decisions"
═══════════════════════════════════════════════════════════════════════════════

The MISSING LINK between:
- NEO's pre-market intel (MM predictions, liquidity, correlations)
- Your pattern sightings (bear flag, gap fill, FOMC)
- Ghost Commander's trading mode

FLOW:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  NEO'S INTEL     │    │  YOUR SIGHTINGS  │    │  MARKET EVENTS   │
│  - MM Prediction │ +  │  - Bear flag     │ +  │  - FOMC          │
│  - Liquidity     │    │  - Gap at $4,650 │    │  - NFP           │
│  - Correlations  │    │  - Divergence    │    │  - Earnings      │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    COMMAND CENTER       │
                    │    Combines all intel   │
                    │    → Recommends MODE    │
                    │    → Updates Ghost      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   GHOST COMMANDER       │
                    │   Executes the plan     │
                    └─────────────────────────┘

Created: January 29, 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CommandCenter')

# Files
STATE_FILE = Path(__file__).parent / 'whipsaw_state.json'
INTEL_FILE = Path(__file__).parent / 'command_center_intel.json'


@dataclass
class UserSighting:
    """A pattern or event you spotted"""
    type: str           # BEAR_FLAG, BULL_FLAG, GAP_FILL, DIVERGENCE, etc.
    description: str
    price_level: float
    confidence: str     # HIGH, MEDIUM, LOW
    timestamp: str
    active: bool = True


@dataclass
class MarketEvent:
    """Scheduled market event"""
    name: str           # FOMC, NFP, CPI, EARNINGS
    datetime: str
    impact: str         # HIGH, MEDIUM, LOW
    expected: str       # What's expected to happen
    bias: str           # BULLISH, BEARISH, UNKNOWN


@dataclass
class NeoIntel:
    """Intel from NEO's analysis"""
    mm_prediction: str  # What MMs will do
    liquidity_above: float  # Short stops level
    liquidity_below: float  # Long stops level
    hunt_direction: str     # UP, DOWN, BOTH
    correlations: Dict
    best_action: str
    timestamp: str


@dataclass
class CommandCenterState:
    """Combined intelligence state"""
    # Mode
    current_mode: int = 2
    mode_name: str = 'CORRECTION'
    
    # User Sightings
    sightings: List[Dict] = field(default_factory=list)
    
    # Market Events
    events: List[Dict] = field(default_factory=list)
    
    # NEO Intel
    neo_intel: Dict = field(default_factory=dict)
    
    # Key Levels
    key_levels: Dict = field(default_factory=dict)
    
    # Timestamps
    last_update: str = ''
    last_mode_change: str = ''


class CommandCenter:
    """
    The brain that combines all intelligence sources
    """
    
    def __init__(self):
        self.state = CommandCenterState()
        self._load_state()
    
    def _load_state(self):
        """Load saved state"""
        if INTEL_FILE.exists():
            try:
                with open(INTEL_FILE, 'r') as f:
                    data = json.load(f)
                    self.state = CommandCenterState(**data)
            except:
                pass
    
    def _save_state(self):
        """Save state"""
        self.state.last_update = datetime.now().isoformat()
        with open(INTEL_FILE, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════════
    # USER SIGHTINGS
    # ═══════════════════════════════════════════════════════════════════
    
    def add_sighting(self, sighting_type: str, description: str, 
                     price_level: float = 0, confidence: str = 'MEDIUM'):
        """Add a pattern/sighting you spotted"""
        sighting = UserSighting(
            type=sighting_type.upper(),
            description=description,
            price_level=price_level,
            confidence=confidence.upper(),
            timestamp=datetime.now().isoformat(),
            active=True,
        )
        self.state.sightings.append(asdict(sighting))
        self._save_state()
        logger.info(f"✅ Added sighting: {sighting_type} - {description}")
        return sighting
    
    def clear_sighting(self, sighting_type: str):
        """Clear/deactivate a sighting"""
        for s in self.state.sightings:
            if s['type'] == sighting_type.upper():
                s['active'] = False
        self._save_state()
    
    def get_active_sightings(self) -> List[Dict]:
        """Get all active sightings"""
        return [s for s in self.state.sightings if s.get('active', False)]
    
    # ═══════════════════════════════════════════════════════════════════
    # MARKET EVENTS
    # ═══════════════════════════════════════════════════════════════════
    
    def add_event(self, name: str, event_datetime: str, 
                  impact: str = 'HIGH', expected: str = '', bias: str = 'UNKNOWN'):
        """Add a market event"""
        event = MarketEvent(
            name=name.upper(),
            datetime=event_datetime,
            impact=impact.upper(),
            expected=expected,
            bias=bias.upper(),
        )
        self.state.events.append(asdict(event))
        self._save_state()
        logger.info(f"✅ Added event: {name} at {event_datetime}")
        return event
    
    def get_upcoming_events(self) -> List[Dict]:
        """Get events coming up"""
        now = datetime.now()
        upcoming = []
        for e in self.state.events:
            try:
                event_time = datetime.fromisoformat(e['datetime'])
                if event_time > now:
                    upcoming.append(e)
            except:
                pass
        return upcoming
    
    # ═══════════════════════════════════════════════════════════════════
    # KEY LEVELS
    # ═══════════════════════════════════════════════════════════════════
    
    def set_key_level(self, name: str, price: float, description: str = ''):
        """Set a key level"""
        self.state.key_levels[name] = {
            'price': price,
            'description': description,
            'set_at': datetime.now().isoformat(),
        }
        self._save_state()
        logger.info(f"✅ Set key level: {name} @ ${price}")
    
    # ═══════════════════════════════════════════════════════════════════
    # NEO INTEL
    # ═══════════════════════════════════════════════════════════════════
    
    def update_neo_intel(self, intel: Dict):
        """Update with NEO's latest intel"""
        self.state.neo_intel = {
            **intel,
            'received_at': datetime.now().isoformat(),
        }
        self._save_state()
        logger.info("✅ Updated NEO intel")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODE RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════
    
    def recommend_mode(self) -> Dict:
        """
        Combine all intelligence and recommend a trading mode
        """
        sightings = self.get_active_sightings()
        events = self.get_upcoming_events()
        neo = self.state.neo_intel
        levels = self.state.key_levels
        
        reasoning = []
        mode = 1  # Default BULLISH
        confidence = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK SIGHTINGS
        # ═══════════════════════════════════════════════════════════════════
        
        bearish_sightings = [s for s in sightings if s['type'] in 
                           ['BEAR_FLAG', 'DIVERGENCE', 'BREAKDOWN', 'LOWER_HIGH']]
        
        if bearish_sightings:
            mode = 3
            confidence += 20
            for s in bearish_sightings:
                reasoning.append(f"🐻 {s['type']}: {s['description']}")
        
        gap_sightings = [s for s in sightings if s['type'] == 'GAP_FILL']
        if gap_sightings:
            confidence += 10
            for s in gap_sightings:
                reasoning.append(f"🕳️ GAP: {s['description']} @ ${s['price_level']}")
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK EVENTS
        # ═══════════════════════════════════════════════════════════════════
        
        high_impact_events = [e for e in events if e['impact'] == 'HIGH']
        
        if high_impact_events:
            if mode == 1:  # Was bullish, switch to correction for safety
                mode = 2
            confidence += 15
            for e in high_impact_events:
                reasoning.append(f"📅 EVENT: {e['name']} - {e['expected']}")
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK NEO INTEL
        # ═══════════════════════════════════════════════════════════════════
        
        if neo:
            if neo.get('hunt_direction') == 'DOWN':
                if mode == 1:
                    mode = 2  # At least go to correction
                reasoning.append(f"🦊 NEO: MMs hunting DOWN - {neo.get('mm_prediction', '')}")
            elif neo.get('hunt_direction') == 'UP':
                reasoning.append(f"🦊 NEO: MMs hunting UP first - wait for squeeze")
            
            if neo.get('best_action'):
                reasoning.append(f"🎯 NEO Action: {neo.get('best_action')}")
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK KEY LEVELS
        # ═══════════════════════════════════════════════════════════════════
        
        if 'gap' in levels:
            gap = levels['gap']
            reasoning.append(f"📍 GAP LEVEL: ${gap['price']} - {gap['description']}")
        
        # Determine mode name
        mode_names = {1: 'BULLISH', 2: 'CORRECTION', 3: 'BEARISH'}
        
        return {
            'recommended_mode': mode,
            'mode_name': mode_names[mode],
            'confidence': min(95, confidence),
            'reasoning': reasoning,
            'sightings': len(sightings),
            'events': len(events),
            'has_neo_intel': bool(neo),
        }
    
    def apply_mode(self, mode: int):
        """Apply the recommended mode to Ghost Commander"""
        import subprocess
        
        result = subprocess.run(
            ['python3', 'grid_control.py', str(mode)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        
        mode_names = {1: 'BULLISH', 2: 'CORRECTION', 3: 'BEARISH'}
        self.state.current_mode = mode
        self.state.mode_name = mode_names[mode]
        self.state.last_mode_change = datetime.now().isoformat()
        self._save_state()
        
        return result.stdout
    
    # ═══════════════════════════════════════════════════════════════════
    # STATUS DISPLAY
    # ═══════════════════════════════════════════════════════════════════
    
    def get_status(self) -> str:
        """Get full command center status"""
        rec = self.recommend_mode()
        sightings = self.get_active_sightings()
        events = self.get_upcoming_events()
        
        mode_emoji = {1: '📈', 2: '📊', 3: '🐻'}
        
        output = f"""
═══════════════════════════════════════════════════════════════════════════════
                    COMMAND CENTER STATUS
═══════════════════════════════════════════════════════════════════════════════

  CURRENT MODE: {mode_emoji.get(self.state.current_mode, '❓')} {self.state.mode_name} (Mode {self.state.current_mode})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  📋 ACTIVE SIGHTINGS ({len(sightings)}):
"""
        if sightings:
            for s in sightings:
                output += f"    • {s['type']}: {s['description']}"
                if s.get('price_level'):
                    output += f" @ ${s['price_level']}"
                output += "\n"
        else:
            output += "    (none)\n"
        
        output += f"""
  📅 UPCOMING EVENTS ({len(events)}):
"""
        if events:
            for e in events:
                output += f"    • {e['name']}: {e['expected']} ({e['impact']} impact)\n"
        else:
            output += "    (none)\n"
        
        output += f"""
  📍 KEY LEVELS:
"""
        for name, level in self.state.key_levels.items():
            output += f"    • {name}: ${level['price']} - {level['description']}\n"
        
        if not self.state.key_levels:
            output += "    (none)\n"
        
        output += f"""
  🦊 NEO INTEL:
"""
        if self.state.neo_intel:
            neo = self.state.neo_intel
            output += f"    • MM Prediction: {neo.get('mm_prediction', 'N/A')}\n"
            output += f"    • Hunt Direction: {neo.get('hunt_direction', 'N/A')}\n"
            output += f"    • Best Action: {neo.get('best_action', 'N/A')}\n"
        else:
            output += "    (no intel received)\n"
        
        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🎯 RECOMMENDATION: {mode_emoji.get(rec['recommended_mode'], '❓')} MODE {rec['recommended_mode']} ({rec['mode_name']})
  
  Confidence: {rec['confidence']}%
  
  REASONING:
"""
        for r in rec['reasoning']:
            output += f"    {r}\n"
        
        if not rec['reasoning']:
            output += "    (no specific signals - default mode)\n"
        
        output += f"""
  COMMAND: python3 grid_control.py {rec['recommended_mode']}
  
═══════════════════════════════════════════════════════════════════════════════
"""
        return output


def main():
    """Command Center CLI"""
    import sys
    
    cc = CommandCenter()
    
    if len(sys.argv) < 2:
        print(cc.get_status())
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'status':
        print(cc.get_status())
    
    elif cmd == 'sighting':
        # sighting <type> <description> [price] [confidence]
        if len(sys.argv) >= 4:
            stype = sys.argv[2]
            desc = sys.argv[3]
            price = float(sys.argv[4]) if len(sys.argv) > 4 else 0
            conf = sys.argv[5] if len(sys.argv) > 5 else 'MEDIUM'
            cc.add_sighting(stype, desc, price, conf)
            print(f"✅ Added: {stype}")
        else:
            print("Usage: sighting <type> <description> [price] [confidence]")
            print("Types: BEAR_FLAG, BULL_FLAG, GAP_FILL, DIVERGENCE, BREAKDOWN")
    
    elif cmd == 'event':
        # event <name> <datetime> [impact] [expected]
        if len(sys.argv) >= 4:
            name = sys.argv[2]
            dt = sys.argv[3]
            impact = sys.argv[4] if len(sys.argv) > 4 else 'HIGH'
            expected = sys.argv[5] if len(sys.argv) > 5 else ''
            cc.add_event(name, dt, impact, expected)
            print(f"✅ Added event: {name}")
        else:
            print("Usage: event <name> <datetime> [impact] [expected]")
    
    elif cmd == 'level':
        # level <name> <price> [description]
        if len(sys.argv) >= 4:
            name = sys.argv[2]
            price = float(sys.argv[3])
            desc = sys.argv[4] if len(sys.argv) > 4 else ''
            cc.set_key_level(name, price, desc)
            print(f"✅ Set level: {name} @ ${price}")
        else:
            print("Usage: level <name> <price> [description]")
    
    elif cmd == 'neo':
        # neo <mm_prediction> <hunt_direction> <best_action>
        if len(sys.argv) >= 5:
            cc.update_neo_intel({
                'mm_prediction': sys.argv[2],
                'hunt_direction': sys.argv[3],
                'best_action': sys.argv[4],
            })
            print("✅ Updated NEO intel")
    
    elif cmd == 'apply':
        rec = cc.recommend_mode()
        print(f"Applying Mode {rec['recommended_mode']} ({rec['mode_name']})...")
        result = cc.apply_mode(rec['recommended_mode'])
        print(result)
    
    elif cmd == 'clear':
        if len(sys.argv) >= 3:
            cc.clear_sighting(sys.argv[2])
            print(f"✅ Cleared: {sys.argv[2]}")


if __name__ == "__main__":
    main()

"""
DAILY PLAYBOOK GENERATOR
Converts AI research (NEO/CLAUDIA/META) into actionable trading rules.

Each morning:
1. Reads latest research reports
2. Generates structured playbook
3. Sends for approval
4. Bot executes approved playbook
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Bias(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL_BULLISH = "NEUTRAL_BULLISH"
    NEUTRAL = "NEUTRAL"
    NEUTRAL_BEARISH = "NEUTRAL_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"
    CLOSE = "CLOSE"

@dataclass
class PriceLevel:
    price: float
    action: str
    quantity: int
    note: str = ""

@dataclass
class SymbolPlaybook:
    symbol: str
    bias: str
    confidence: str  # HIGH, MEDIUM, LOW
    thesis: str
    current_position: int
    current_avg: float
    
    # Actions
    primary_action: str  # HOLD, ADD, REDUCE, CLOSE
    
    # Entry levels (DCA opportunities)
    entry_levels: List[Dict]
    
    # Take profit levels
    tp_levels: List[Dict]
    
    # Stop/risk levels
    stop_price: Optional[float]
    stop_action: str  # CLOSE, ALERT_ONLY, REDUCE
    
    # Key catalysts
    catalysts: List[str]
    
    # Invalidation
    invalidation: str

@dataclass 
class DailyPlaybook:
    date: str
    generated_at: str
    expires_at: str
    market_context: str
    approved: bool
    approved_at: Optional[str]
    
    symbols: Dict[str, Dict]
    
    # Global rules
    max_daily_trades: int
    cooldown_minutes: int
    trading_hours: Dict[str, str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())


class PlaybookGenerator:
    """Generates daily playbook from research reports."""
    
    def __init__(self, research_dir: str = "/home/jbot/trading_ai"):
        self.research_dir = research_dir
        self.today = date.today().isoformat()
        
    def read_claudia_report(self, symbol: str) -> Optional[str]:
        """Read latest CLAUDIA report for symbol."""
        report_path = f"{self.research_dir}/claudia/research/{symbol}_INSTITUTIONAL_REPORT_{self.today.replace('-', '')}.md"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                return f.read()
        # Try without date
        for f in os.listdir(f"{self.research_dir}/claudia/research"):
            if f.startswith(f"{symbol}_INSTITUTIONAL"):
                with open(f"{self.research_dir}/claudia/research/{f}", 'r') as file:
                    return file.read()
        return None
    
    def read_neo_report(self, symbol: str) -> Optional[str]:
        """Read latest NEO report for symbol."""
        # Check neo research directory
        report_path = f"{self.research_dir}/neo/research/{symbol.lower()}_research.json"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                return f.read()
        return None
    
    def parse_levels_from_report(self, report: str) -> Dict:
        """Extract key price levels from report text."""
        levels = {
            "support": [],
            "resistance": [],
            "entry_zones": [],
            "tp_targets": []
        }
        # This would be enhanced with NLP/regex parsing
        # For now, return structure for manual population
        return levels
    
    def generate_symbol_playbook(
        self,
        symbol: str,
        bias: str,
        confidence: str,
        thesis: str,
        current_position: int,
        current_avg: float,
        entry_levels: List[Dict],
        tp_levels: List[Dict],
        stop_price: Optional[float] = None,
        stop_action: str = "ALERT_ONLY",
        catalysts: List[str] = None,
        invalidation: str = ""
    ) -> SymbolPlaybook:
        """Generate playbook for a single symbol."""
        
        # Determine primary action based on bias and position
        if current_position == 0:
            if "BULLISH" in bias:
                primary_action = "WAIT_FOR_ENTRY"
            else:
                primary_action = "WAIT"
        elif current_position > 0:
            if "BEARISH" in bias:
                primary_action = "REDUCE"
            elif "BULLISH" in bias and entry_levels:
                primary_action = "HOLD_ADD_ON_DIP"
            else:
                primary_action = "HOLD"
        else:  # Short position
            primary_action = "HOLD"
        
        return SymbolPlaybook(
            symbol=symbol,
            bias=bias,
            confidence=confidence,
            thesis=thesis,
            current_position=current_position,
            current_avg=current_avg,
            primary_action=primary_action,
            entry_levels=entry_levels or [],
            tp_levels=tp_levels or [],
            stop_price=stop_price,
            stop_action=stop_action,
            catalysts=catalysts or [],
            invalidation=invalidation
        )
    
    def generate_daily_playbook(
        self,
        symbols: Dict[str, SymbolPlaybook],
        market_context: str = "",
        max_daily_trades: int = 10,
        cooldown_minutes: int = 30
    ) -> DailyPlaybook:
        """Generate complete daily playbook."""
        
        now = datetime.now()
        
        return DailyPlaybook(
            date=self.today,
            generated_at=now.strftime("%Y-%m-%d %H:%M:%S EST"),
            expires_at=f"{self.today} 16:00:00 EST",
            market_context=market_context,
            approved=False,
            approved_at=None,
            symbols={k: asdict(v) for k, v in symbols.items()},
            max_daily_trades=max_daily_trades,
            cooldown_minutes=cooldown_minutes,
            trading_hours={
                "start": "09:30",
                "end": "16:00",
                "timezone": "US/Eastern"
            }
        )


def generate_todays_playbook() -> DailyPlaybook:
    """Generate today's playbook based on current research and positions."""
    
    generator = PlaybookGenerator()
    
    # ===== MGC PLAYBOOK =====
    mgc = generator.generate_symbol_playbook(
        symbol="MGC",
        bias="NEUTRAL_BULLISH",
        confidence="MEDIUM",
        thesis="Holding 56 contracts avg $5,465. Currently underwater but gold recovering. "
               "Wait for bounce to $5,400+ for partial TP. DCA only on significant dips.",
        current_position=56,
        current_avg=5465.40,
        entry_levels=[
            {"price": 5250, "action": "BUY", "quantity": 10, "note": "Strong support DCA"},
            {"price": 5200, "action": "BUY", "quantity": 10, "note": "Major support DCA"},
            {"price": 5150, "action": "BUY", "quantity": 15, "note": "Capitulation DCA"},
        ],
        tp_levels=[
            {"price": 5400, "action": "SELL", "quantity": 15, "note": "Partial TP - reduce risk"},
            {"price": 5465, "action": "SELL", "quantity": 20, "note": "Break-even TP"},
            {"price": 5500, "action": "SELL", "quantity": 15, "note": "Profit TP"},
        ],
        stop_price=5100,
        stop_action="ALERT_ONLY",  # Don't auto-close, just alert
        catalysts=["FOMC", "USD strength", "Bond yields"],
        invalidation="Close below $5,100 with volume invalidates bullish thesis"
    )
    
    # ===== IREN PLAYBOOK =====
    iren = generator.generate_symbol_playbook(
        symbol="IREN",
        bias="BULLISH",
        confidence="HIGH",
        thesis="Gap fill expected to $63. Stop hunt completed at $58.71 open. "
               "Decoupled from BTC (90-day correlation minimal). Microsoft $9.7B AI deal intact. "
               "Feb 5 earnings is key catalyst.",
        current_position=122,  # 60+60+2 calls
        current_avg=8.18,  # Approximate blended avg
        entry_levels=[
            {"price": 58.00, "action": "BUY_CALLS", "quantity": 20, "note": "Add on retest of today's low"},
            {"price": 56.00, "action": "BUY_CALLS", "quantity": 30, "note": "Key support - aggressive add"},
        ],
        tp_levels=[
            {"price": 62.00, "action": "SELL_PARTIAL", "quantity": 40, "note": "Partial at gap fill start"},
            {"price": 63.00, "action": "SELL_PARTIAL", "quantity": 40, "note": "Full gap fill"},
            {"price": 65.00, "action": "SELL_REMAINING", "quantity": "ALL", "note": "Extension target"},
        ],
        stop_price=54.00,
        stop_action="ALERT_ONLY",
        catalysts=["Feb 5 Earnings", "Microsoft AI updates", "GPU deployment news"],
        invalidation="Close below $54.95 (gap) with volume invalidates gap fill thesis"
    )
    
    # ===== CLSK PLAYBOOK =====
    clsk = generator.generate_symbol_playbook(
        symbol="CLSK",
        bias="NEUTRAL",
        confidence="LOW",
        thesis="Underwater on calls. Hold and wait for sector recovery. "
               "Not adding until IREN shows strength first.",
        current_position=80,  # 20+60 calls
        current_avg=1.45,
        entry_levels=[],  # No entries today
        tp_levels=[
            {"price": 14.00, "action": "SELL_PARTIAL", "quantity": 40, "note": "If rallies to $14"},
            {"price": 15.00, "action": "SELL_REMAINING", "quantity": "ALL", "note": "Full exit"},
        ],
        stop_price=None,
        stop_action="HOLD",
        catalysts=["BTC price", "IREN sympathy"],
        invalidation="Calls expire Feb 13 - time decay risk"
    )
    
    # ===== CIFR PLAYBOOK =====
    cifr = generator.generate_symbol_playbook(
        symbol="CIFR",
        bias="NEUTRAL",
        confidence="LOW", 
        thesis="Underwater on calls. Hold and wait for sector recovery. "
               "Not adding until IREN shows strength first.",
        current_position=80,  # 20+60 calls
        current_avg=2.10,
        entry_levels=[],  # No entries today
        tp_levels=[
            {"price": 19.00, "action": "SELL_PARTIAL", "quantity": 40, "note": "If rallies to $19"},
            {"price": 20.00, "action": "SELL_REMAINING", "quantity": "ALL", "note": "Full exit"},
        ],
        stop_price=None,
        stop_action="HOLD",
        catalysts=["BTC price", "IREN sympathy"],
        invalidation="Calls expire Feb 13 - time decay risk"
    )
    
    # Generate full playbook
    playbook = generator.generate_daily_playbook(
        symbols={
            "MGC": mgc,
            "IREN": iren,
            "CLSK": clsk,
            "CIFR": cifr
        },
        market_context="Post-FOMC volatility. Gold recovering from lows. "
                       "BTC miners gapping down but IREN decoupled. "
                       "Risk-off sentiment but opportunities in oversold names.",
        max_daily_trades=10,
        cooldown_minutes=30
    )
    
    return playbook


if __name__ == "__main__":
    # Generate and save today's playbook
    playbook = generate_todays_playbook()
    
    # Save to file
    save_path = f"/home/jbot/trading_ai/playbook/daily/playbook_{date.today().isoformat()}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    playbook.save(save_path)
    
    print(f"✅ Playbook generated: {save_path}")
    print("\n" + "="*60)
    print(playbook.to_json())

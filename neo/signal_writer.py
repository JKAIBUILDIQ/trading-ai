#!/usr/bin/env python3
"""
NEO Signal Writer - Output Signals for Ghost Commander
Writes trading signals to /tmp/neo_signal.json
Ghost Commander reads and executes.
"""

import json
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil

from config import SIGNAL_FILE, SIGNAL_HISTORY_DIR, MAX_POSITION_DOLLARS


@dataclass
class TradingSignal:
    """A trading signal for Ghost Commander."""
    timestamp: str
    signal_id: str
    
    # Trade parameters
    symbol: str
    direction: str  # BUY or SELL
    
    # Position sizing
    position_value: float  # Dollar amount
    stop_loss_pips: int
    take_profit_pips: int
    
    # NEO metadata
    confidence: float  # 0-100
    reasoning: str
    model_used: str
    
    # Safety
    max_hold_minutes: int = 180  # 3 hours default
    source: str = "NEO"
    
    # Execution preferences
    entry_type: str = "MARKET"  # MARKET or LIMIT
    limit_offset_pips: int = 0


class SignalWriter:
    """
    Writes trading signals for Ghost Commander to execute.
    Maintains signal history for learning.
    """
    
    def __init__(self):
        self.signal_file = SIGNAL_FILE
        self.history_dir = SIGNAL_HISTORY_DIR
    
    def write_signal(self, signal: TradingSignal) -> str:
        """Write signal to file and return signal ID."""
        # Create signal dict
        signal_dict = {
            "timestamp": signal.timestamp,
            "signal_id": signal.signal_id,
            "action": "OPEN",
            "status": {
                "state": "ACTIVE",  # ACTIVE → EXECUTED → CLOSED_TP/CLOSED_SL/EXPIRED
                "trade_ticket": None,  # Set when v020 executes
                "outcome_pips": None,  # Set when trade closes
                "created_at": signal.timestamp,
                "executed_at": None,
                "closed_at": None
            },
            "trade": {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "position_value_usd": signal.position_value,
                "stop_loss_pips": signal.stop_loss_pips,
                "take_profit_pips": signal.take_profit_pips,
                "entry_type": signal.entry_type,
                "limit_offset_pips": signal.limit_offset_pips,
                "max_hold_minutes": signal.max_hold_minutes
            },
            "metadata": {
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "model_used": signal.model_used,
                "source": signal.source
            },
            "safety": {
                "max_position_value": MAX_POSITION_DOLLARS,
                "auto_close_on_loss": True,
                "trail_after_profit_pips": 30
            }
        }
        
        # Write to main signal file
        with open(self.signal_file, 'w') as f:
            json.dump(signal_dict, f, indent=2)
        
        # Archive to history
        history_file = self.history_dir / f"signal_{signal.signal_id}.json"
        shutil.copy(self.signal_file, history_file)
        
        return signal.signal_id
    
    def write_close_signal(self, symbol: str, reason: str) -> str:
        """Write a signal to close existing position."""
        signal_id = f"close_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        signal_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "signal_id": signal_id,
            "action": "CLOSE",
            "trade": {
                "symbol": symbol
            },
            "metadata": {
                "reason": reason,
                "source": "NEO"
            }
        }
        
        with open(self.signal_file, 'w') as f:
            json.dump(signal_dict, f, indent=2)
        
        return signal_id
    
    def write_hold_signal(self, reasoning: str) -> str:
        """Write a hold/no-trade signal (for logging)."""
        signal_id = f"hold_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        signal_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "signal_id": signal_id,
            "action": "HOLD",
            "metadata": {
                "reasoning": reasoning,
                "source": "NEO"
            }
        }
        
        with open(self.signal_file, 'w') as f:
            json.dump(signal_dict, f, indent=2)
        
        return signal_id
    
    def get_current_signal(self) -> Optional[Dict]:
        """Read the current signal file."""
        try:
            if self.signal_file.exists():
                with open(self.signal_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading signal: {e}")
        return None
    
    def clear_signal(self):
        """Clear the current signal file."""
        if self.signal_file.exists():
            self.signal_file.unlink()


def test_signal_writer():
    """Test the signal writer."""
    print("=" * 60)
    print("NEO SIGNAL WRITER TEST")
    print("=" * 60)
    
    writer = SignalWriter()
    
    # Create test signal
    signal = TradingSignal(
        timestamp=datetime.utcnow().isoformat(),
        signal_id=f"NEO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        symbol="EURUSD",
        direction="BUY",
        position_value=2000.0,
        stop_loss_pips=30,
        take_profit_pips=60,
        confidence=75.0,
        reasoning="RSI(2) extreme oversold (<10) with price above 200 SMA. "
                  "Historical win rate 88% for this setup.",
        model_used="deepseek-r1:70b"
    )
    
    signal_id = writer.write_signal(signal)
    print(f"✅ Signal written: {signal_id}")
    print(f"📁 File: {writer.signal_file}")
    print("")
    
    # Read it back
    current = writer.get_current_signal()
    print("Signal content:")
    print(json.dumps(current, indent=2))
    print("")
    print("=" * 60)


if __name__ == "__main__":
    test_signal_writer()

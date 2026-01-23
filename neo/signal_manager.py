#!/usr/bin/env python3
"""
NEO SIGNAL STATE MANAGER
═══════════════════════════════════════════════════════════════════════

Prevents duplicate signal sending that causes Ghost Commander to stack positions.

Key Rules:
1. Only send signal when direction/symbol/price CHANGES
2. Track last signal sent per symbol
3. Wait for execution confirmation before new signals
4. Generate content-based signal IDs (not time-based)

"Send once. Execute once. Confirm once."
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalManager")

# State file for persistence across restarts
STATE_FILE = "/tmp/neo_signal_state.json"

# Configuration
SIGNAL_EXPIRY_MINUTES = 30  # How long before a signal can be resent
PRICE_CHANGE_THRESHOLD = 20  # Points change to trigger new signal for Gold
PRICE_CHANGE_THRESHOLD_FX = 0.0020  # 20 pips for forex


@dataclass
class SignalState:
    """Tracks the state of a sent signal."""
    signal_id: str
    symbol: str
    direction: str  # BUY, SELL, WAIT
    entry_price: float
    sent_at: float  # Unix timestamp
    executed: bool = False
    executed_at: Optional[float] = None
    execution_ticket: Optional[int] = None
    
    def age_minutes(self) -> float:
        """Get signal age in minutes."""
        return (time.time() - self.sent_at) / 60
    
    def is_expired(self) -> bool:
        """Check if signal is expired (can be resent)."""
        return self.age_minutes() > SIGNAL_EXPIRY_MINUTES
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SignalManager:
    """
    Manages signal state to prevent duplicates.
    
    Key responsibilities:
    - Track last signal per symbol
    - Determine if new signal should be sent
    - Generate content-based signal IDs
    - Handle execution confirmations
    """
    
    def __init__(self):
        self.signals: Dict[str, SignalState] = {}  # symbol -> SignalState
        self.pending_signals: Dict[str, SignalState] = {}  # signal_id -> SignalState
        self._load_state()
        
        logger.info("📊 Signal Manager initialized")
        logger.info(f"   Expiry: {SIGNAL_EXPIRY_MINUTES} minutes")
        logger.info(f"   Gold threshold: {PRICE_CHANGE_THRESHOLD} points")
        logger.info(f"   Forex threshold: {PRICE_CHANGE_THRESHOLD_FX * 10000} pips")
    
    # ═══════════════════════════════════════════════════════════════════
    # SIGNAL ID GENERATION
    # ═══════════════════════════════════════════════════════════════════
    
    def generate_signal_id(self, symbol: str, direction: str, entry_price: float) -> str:
        """
        Generate content-based signal ID.
        Same content = same ID = Ghost skips if already executed.
        
        Format: {SYMBOL}_{DIRECTION}_{PRICE_LEVEL}_{DATE}
        """
        # Round price to reduce tiny variations triggering new IDs
        if "JPY" in symbol:
            price_level = round(entry_price, 1)  # 150.5
        elif symbol == "XAUUSD":
            price_level = round(entry_price / 10) * 10  # 4930 → 4930
        else:
            price_level = round(entry_price, 3)  # 1.170
        
        date_str = datetime.utcnow().strftime("%Y%m%d")
        
        signal_id = f"{symbol}_{direction}_{price_level}_{date_str}"
        
        return signal_id
    
    # ═══════════════════════════════════════════════════════════════════
    # DUPLICATE DETECTION
    # ═══════════════════════════════════════════════════════════════════
    
    def should_send_signal(self, symbol: str, direction: str, entry_price: float) -> Tuple[bool, str]:
        """
        Determine if a new signal should be sent.
        
        Returns:
            (should_send, reason)
        """
        # Always send if no previous signal for this symbol
        if symbol not in self.signals:
            return True, "First signal for this symbol"
        
        last = self.signals[symbol]
        
        # Send if direction changed
        if last.direction != direction:
            return True, f"Direction changed: {last.direction} → {direction}"
        
        # Send if signal expired
        if last.is_expired():
            return True, f"Previous signal expired ({last.age_minutes():.0f} min old)"
        
        # Send if price changed significantly
        price_change = abs(entry_price - last.entry_price)
        threshold = PRICE_CHANGE_THRESHOLD if symbol == "XAUUSD" else PRICE_CHANGE_THRESHOLD_FX
        
        if price_change >= threshold:
            return True, f"Price changed: {last.entry_price} → {entry_price} ({price_change:.2f})"
        
        # Send if previous signal was executed (allow new entry)
        if last.executed:
            return True, "Previous signal already executed"
        
        # Don't send - same signal still pending
        return False, f"Same signal already pending (age: {last.age_minutes():.1f} min)"
    
    def is_duplicate(self, signal_id: str) -> bool:
        """Quick check if signal ID was already sent recently."""
        for state in self.signals.values():
            if state.signal_id == signal_id and not state.is_expired():
                return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════
    # SIGNAL TRACKING
    # ═══════════════════════════════════════════════════════════════════
    
    def record_signal_sent(self, symbol: str, direction: str, entry_price: float, 
                           signal_id: str = None) -> SignalState:
        """
        Record that a signal was sent.
        Call this AFTER successfully pushing to API.
        """
        if not signal_id:
            signal_id = self.generate_signal_id(symbol, direction, entry_price)
        
        state = SignalState(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            sent_at=time.time(),
            executed=False
        )
        
        self.signals[symbol] = state
        self.pending_signals[signal_id] = state
        self._save_state()
        
        logger.info(f"📤 Signal RECORDED: {signal_id}")
        logger.info(f"   {direction} {symbol} @ {entry_price}")
        
        return state
    
    def mark_executed(self, signal_id: str, ticket: int = None, price: float = None) -> bool:
        """
        Mark a signal as executed by Ghost Commander.
        Called when Ghost sends /neo/signals/executed callback.
        """
        if signal_id in self.pending_signals:
            state = self.pending_signals[signal_id]
            state.executed = True
            state.executed_at = time.time()
            state.execution_ticket = ticket
            
            # Also update in signals dict
            if state.symbol in self.signals:
                self.signals[state.symbol].executed = True
                self.signals[state.symbol].executed_at = state.executed_at
                self.signals[state.symbol].execution_ticket = ticket
            
            self._save_state()
            
            logger.info(f"✅ Signal EXECUTED: {signal_id}")
            logger.info(f"   Ticket: {ticket}")
            
            return True
        
        logger.warning(f"⚠️ Unknown signal ID: {signal_id}")
        return False
    
    def mark_expired(self, symbol: str):
        """Mark the current signal for a symbol as expired."""
        if symbol in self.signals:
            self.signals[symbol].sent_at = 0  # Force expiry
            self._save_state()
            logger.info(f"⏰ Signal EXPIRED: {symbol}")
    
    def clear_symbol(self, symbol: str):
        """Clear all signal state for a symbol."""
        if symbol in self.signals:
            del self.signals[symbol]
            self._save_state()
            logger.info(f"🗑️ Signal CLEARED: {symbol}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STATE QUERIES
    # ═══════════════════════════════════════════════════════════════════
    
    def get_current_state(self, symbol: str) -> Optional[SignalState]:
        """Get current signal state for a symbol."""
        return self.signals.get(symbol)
    
    def get_pending_count(self) -> int:
        """Get count of pending (unexecuted) signals."""
        return sum(1 for s in self.signals.values() if not s.executed and not s.is_expired())
    
    def get_status_report(self) -> Dict:
        """Get full status report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'signals': {},
            'pending_count': 0,
            'executed_count': 0,
            'expired_count': 0
        }
        
        for symbol, state in self.signals.items():
            status = 'EXECUTED' if state.executed else ('EXPIRED' if state.is_expired() else 'PENDING')
            
            if status == 'PENDING':
                report['pending_count'] += 1
            elif status == 'EXECUTED':
                report['executed_count'] += 1
            else:
                report['expired_count'] += 1
            
            report['signals'][symbol] = {
                'signal_id': state.signal_id,
                'direction': state.direction,
                'entry_price': state.entry_price,
                'age_minutes': round(state.age_minutes(), 1),
                'status': status,
                'execution_ticket': state.execution_ticket
            }
        
        return report
    
    # ═══════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════
    
    def _save_state(self):
        """Save state to file for persistence."""
        try:
            data = {
                'signals': {k: v.to_dict() for k, v in self.signals.items()},
                'pending': {k: v.to_dict() for k, v in self.pending_signals.items()},
                'saved_at': time.time()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from file."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                
                # Restore signals
                for symbol, state_dict in data.get('signals', {}).items():
                    self.signals[symbol] = SignalState(**state_dict)
                
                for signal_id, state_dict in data.get('pending', {}).items():
                    self.pending_signals[signal_id] = SignalState(**state_dict)
                
                # Clean up expired signals
                self._cleanup_expired()
                
                logger.info(f"📂 Loaded {len(self.signals)} signal states")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def _cleanup_expired(self):
        """Remove signals that expired more than 2 hours ago."""
        max_age = 2 * 60 * 60  # 2 hours in seconds
        
        to_remove = []
        for symbol, state in self.signals.items():
            if time.time() - state.sent_at > max_age:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del self.signals[symbol]
        
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old signals")


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════

_manager = None

def get_signal_manager() -> SignalManager:
    """Get or create the global SignalManager instance."""
    global _manager
    if _manager is None:
        _manager = SignalManager()
    return _manager


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def should_send_signal(symbol: str, direction: str, entry_price: float) -> Tuple[bool, str]:
    """Quick check if signal should be sent."""
    return get_signal_manager().should_send_signal(symbol, direction, entry_price)


def record_signal_sent(symbol: str, direction: str, entry_price: float, signal_id: str = None):
    """Record that a signal was sent."""
    return get_signal_manager().record_signal_sent(symbol, direction, entry_price, signal_id)


def mark_signal_executed(signal_id: str, ticket: int = None, price: float = None):
    """Mark a signal as executed."""
    return get_signal_manager().mark_executed(signal_id, ticket, price)


def generate_signal_id(symbol: str, direction: str, entry_price: float) -> str:
    """Generate content-based signal ID."""
    return get_signal_manager().generate_signal_id(symbol, direction, entry_price)


# ═══════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("📊 SIGNAL MANAGER - Test")
    print("=" * 60)
    
    mgr = SignalManager()
    
    # Test 1: First signal should be sent
    print("\n1️⃣ First SELL XAUUSD signal:")
    can_send, reason = mgr.should_send_signal("XAUUSD", "SELL", 4930.50)
    print(f"   Should send: {can_send}")
    print(f"   Reason: {reason}")
    
    if can_send:
        signal_id = mgr.generate_signal_id("XAUUSD", "SELL", 4930.50)
        mgr.record_signal_sent("XAUUSD", "SELL", 4930.50, signal_id)
        print(f"   Signal ID: {signal_id}")
    
    # Test 2: Same signal should NOT be sent
    print("\n2️⃣ Same SELL XAUUSD signal (immediate):")
    can_send, reason = mgr.should_send_signal("XAUUSD", "SELL", 4930.50)
    print(f"   Should send: {can_send}")
    print(f"   Reason: {reason}")
    
    # Test 3: Different direction SHOULD be sent
    print("\n3️⃣ BUY XAUUSD signal (direction change):")
    can_send, reason = mgr.should_send_signal("XAUUSD", "BUY", 4900.00)
    print(f"   Should send: {can_send}")
    print(f"   Reason: {reason}")
    
    # Test 4: Status report
    print("\n📊 Status Report:")
    report = mgr.get_status_report()
    print(json.dumps(report, indent=2))

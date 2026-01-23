#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: NEO Broadcasting Signals
═══════════════════════════════════════════════════════════════════════════════

Run this script to broadcast a test signal from NEO.

Usage:
    Terminal 1: python crella_listener.py  (listen)
    Terminal 2: python neo_broadcaster.py   (broadcast)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
sys.path.insert(0, '/home/jbot/trading_ai/neo')

from agent_comms import AgentBroadcaster
import time

def main():
    print("="*60)
    print("📡 NEO BROADCASTING...")
    print("="*60)
    
    broadcaster = AgentBroadcaster("NEO")
    
    # Send heartbeat
    print("\n💓 Sending heartbeat...")
    broadcaster.heartbeat({
        "status": "analyzing",
        "current_symbol": "XAUUSD",
        "uptime": 3600
    })
    
    time.sleep(1)
    
    # Broadcast signal
    print("\n📊 Broadcasting signal...")
    broadcaster.signal(
        symbol="XAUUSD",
        action="SELL",
        confidence=85,
        entry_price=2750.00,
        stop_loss=2780.00,
        take_profit=2700.00,
        reasoning="RSI divergence on H4, volume exhaustion, parabolic move detected",
        crowd_psychology={
            "crash_probability": 65,
            "risk_level": "medium",
            "btc_similarity": 45
        }
    )
    
    time.sleep(1)
    
    # Send chat to CRELLA
    print("\n💬 Chatting with CRELLA...")
    broadcaster.chat("CRELLA001", "Hey CRELLA, watch out for that H4 divergence on Gold!")
    
    time.sleep(1)
    
    # Send crash warning
    print("\n⚠️ Sending crash warning...")
    broadcaster.crash_warning("XAUUSD", 72, {
        "divergence": "bearish",
        "parabolic_score": 68,
        "patterns": ["rising_wedge"]
    })
    
    time.sleep(1)
    
    # Notify human
    print("\n📱 Notifying human...")
    broadcaster.notify_human("Test notification: NEO is online and monitoring Gold")
    
    print("\n" + "="*60)
    print("✅ All messages broadcasted!")
    print("="*60)


if __name__ == "__main__":
    main()

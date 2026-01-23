#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: CRELLA Listening to NEO
═══════════════════════════════════════════════════════════════════════════════

Run this script to see how CRELLA would receive NEO's signals in real-time.

Usage:
    Terminal 1: python crella_listener.py  (listen)
    Terminal 2: python neo_broadcaster.py   (broadcast)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
sys.path.insert(0, '/home/jbot/trading_ai/neo')

from agent_comms import AgentListener, AgentMemory
import time

def main():
    print("="*60)
    print("🎧 CRELLA001 LISTENING FOR AGENT MESSAGES...")
    print("="*60)
    
    listener = AgentListener("CRELLA001")
    
    @listener.on("agent:signals")
    def handle_neo_signal(msg):
        print(f"\n🧠 NEO SIGNAL RECEIVED!")
        print(f"   Symbol: {msg.get('symbol')}")
        print(f"   Action: {msg['data'].get('action')}")
        print(f"   Confidence: {msg['data'].get('confidence')}%")
        
        # Show crowd psychology if present
        crowd = msg['data'].get('crowd_psychology', {})
        if crowd.get('crash_probability'):
            print(f"   ⚠️ Crash Risk: {crowd['crash_probability']:.0f}%")
        
        # React to high-confidence signals
        if msg['data'].get('confidence', 0) >= 70:
            print(f"   ⚡ CRELLA: Adjusting Jackal strategy...")
    
    @listener.on("agent:alerts")
    def handle_alert(msg):
        print(f"\n⚠️ ALERT from {msg.get('agent')}!")
        print(f"   {msg['data'].get('message')}")
        
        severity = msg['data'].get('severity', 'info')
        if severity in ['high', 'critical']:
            print(f"   🚨 CRELLA: Taking protective action!")
    
    @listener.on("agent:inbox:CRELLA001")
    def handle_direct_message(msg):
        print(f"\n💬 DM from {msg.get('from')}: {msg.get('message')}")
    
    @listener.on("agent:heartbeat")
    def handle_heartbeat(msg):
        agent = msg.get('agent')
        if agent != "CRELLA001":  # Don't show own heartbeat
            print(f"\n💓 {agent} is alive")
    
    print("\nSubscribed to:")
    print("  - agent:signals")
    print("  - agent:alerts")
    print("  - agent:inbox:CRELLA001")
    print("  - agent:heartbeat")
    print("\nWaiting for messages... (Ctrl+C to stop)")
    print("-"*60)
    
    # Start listening (blocking)
    listener.listen_blocking()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 CRELLA listener stopped.")

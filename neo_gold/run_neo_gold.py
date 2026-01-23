#!/usr/bin/env python3
"""
NEO-GOLD Runner Script
Start the Gold Trading Specialist
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo_gold import NeoGold

def main():
    print("═══════════════════════════════════════════════════════════")
    print("🥇 NEO-GOLD: GOLD TRADING SPECIALIST")
    print("═══════════════════════════════════════════════════════════")
    print("Symbol: XAUUSD ONLY")
    print("Min Confidence: 75%")
    print("Max Signals/Day: 3")
    print("═══════════════════════════════════════════════════════════")
    
    neo = NeoGold()
    neo.run(interval_seconds=60)

if __name__ == "__main__":
    main()

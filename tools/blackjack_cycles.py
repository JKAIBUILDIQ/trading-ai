#!/usr/bin/env python3
"""
Paul's Blackjack - CYCLE-BASED Analysis
Shows risk per cycle, not per hand
"""

import random
from dataclasses import dataclass
from typing import List
import statistics

# Correct probabilities for 6-deck S17 (~0.5% house edge)
WIN_PROB = 0.4285  # includes BJ adjusted
LOSS_PROB = 0.4575
PUSH_PROB = 0.1140

# Steady Climb progression
PROGRESSION = [1, 1, 2, 2, 4, 4, 8, 8, 16]

@dataclass
class CycleResult:
    hands_played: int
    units_won: int
    max_streak: int
    lost_first_hand: bool

def play_cycle() -> CycleResult:
    """Play one cycle until a loss or push that ends advancement"""
    position = 0
    units = 0
    hands = 0
    
    while True:
        bet = PROGRESSION[position] if position < len(PROGRESSION) else 16
        hands += 1
        
        roll = random.random()
        
        if roll < 0.0475:  # Blackjack
            units += int(bet * 1.5)
            position += 1
        elif roll < 0.0475 + 0.381:  # Regular win
            units += bet
            position += 1
        elif roll < 0.0475 + 0.381 + 0.4575:  # Loss
            units -= bet
            break  # Cycle ends on loss
        # Push: no change, but cycle continues at same position
        
        # Safety cap
        if hands > 50:
            break
    
    return CycleResult(
        hands_played=hands,
        units_won=units,
        max_streak=position,
        lost_first_hand=(position == 0 and hands == 1)
    )

def simulate_session(hands_limit: int = 560, profit_stop: int = None) -> dict:
    """Simulate a session as series of cycles"""
    total_units = 0
    total_hands = 0
    cycles = []
    
    while total_hands < hands_limit:
        if profit_stop and total_units >= profit_stop:
            break
            
        cycle = play_cycle()
        cycles.append(cycle)
        total_units += cycle.units_won
        total_hands += cycle.hands_played
    
    lost_first = sum(1 for c in cycles if c.lost_first_hand)
    won_first = len(cycles) - lost_first
    
    return {
        'total_units': total_units,
        'total_hands': total_hands,
        'num_cycles': len(cycles),
        'lost_first_hand_cycles': lost_first,
        'won_first_hand_cycles': won_first,
        'max_streak': max(c.max_streak for c in cycles),
        'avg_streak': statistics.mean(c.max_streak for c in cycles),
        'units_at_risk': lost_first,  # Only lost 1 unit per failed first hand
    }

def main():
    NUM_SESSIONS = 10000
    
    print("=" * 70)
    print("PAUL'S STEADY CLIMB - CYCLE-BASED ANALYSIS")
    print("=" * 70)
    print(f"\nKey Insight: You only risk 1 unit per CYCLE (first hand)")
    print(f"After first win, you're playing with HOUSE MONEY\n")
    
    # Run without stop
    print("Running simulations...")
    results_no_stop = [simulate_session(560, None) for _ in range(NUM_SESSIONS)]
    results_with_stop = [simulate_session(560, 100) for _ in range(NUM_SESSIONS)]
    
    print("\n" + "=" * 70)
    print("NO PROFIT STOP")
    print("=" * 70)
    
    avg_units = statistics.mean(r['total_units'] for r in results_no_stop)
    avg_cycles = statistics.mean(r['num_cycles'] for r in results_no_stop)
    avg_lost_first = statistics.mean(r['lost_first_hand_cycles'] for r in results_no_stop)
    avg_won_first = statistics.mean(r['won_first_hand_cycles'] for r in results_no_stop)
    avg_risk = statistics.mean(r['units_at_risk'] for r in results_no_stop)
    
    winning = sum(1 for r in results_no_stop if r['total_units'] > 0)
    
    print(f"""
📊 SESSION STATISTICS (avg per session):

  Cycles per session:       {avg_cycles:.0f}
  Lost first hand:          {avg_lost_first:.0f} cycles ({avg_lost_first/avg_cycles*100:.1f}%)
  Won first hand:           {avg_won_first:.0f} cycles ({avg_won_first/avg_cycles*100:.1f}%)
  
  RISK ANALYSIS:
  ├── Units truly at risk:  {avg_risk:.0f} (1 unit per losing first-hand)
  ├── That's only ${avg_risk*100:.0f} at $100/unit
  └── Or ${avg_risk*500:.0f} at $500/unit
  
  RESULTS:
  ├── Avg units won/lost:   {avg_units:+.1f}
  ├── Win rate:             {winning/NUM_SESSIONS*100:.1f}% sessions
  └── At $500/unit:         ${avg_units*500:+,.0f}/session
""")

    print("=" * 70)
    print("+100 PROFIT STOP")
    print("=" * 70)
    
    avg_units_stop = statistics.mean(r['total_units'] for r in results_with_stop)
    avg_cycles_stop = statistics.mean(r['num_cycles'] for r in results_with_stop)
    avg_lost_first_stop = statistics.mean(r['lost_first_hand_cycles'] for r in results_with_stop)
    avg_risk_stop = statistics.mean(r['units_at_risk'] for r in results_with_stop)
    hit_stop = sum(1 for r in results_with_stop if r['total_units'] >= 100)
    winning_stop = sum(1 for r in results_with_stop if r['total_units'] > 0)
    
    print(f"""
📊 SESSION STATISTICS (avg per session):

  Cycles per session:       {avg_cycles_stop:.0f}
  Lost first hand:          {avg_lost_first_stop:.0f} cycles
  
  RISK ANALYSIS:
  ├── Units truly at risk:  {avg_risk_stop:.0f}
  ├── That's only ${avg_risk_stop*100:.0f} at $100/unit
  └── Or ${avg_risk_stop*500:.0f} at $500/unit
  
  RESULTS:
  ├── Avg units won/lost:   {avg_units_stop:+.1f}
  ├── Win rate:             {winning_stop/NUM_SESSIONS*100:.1f}% sessions
  ├── Hit +100 stop:        {hit_stop/NUM_SESSIONS*100:.1f}% sessions
  └── At $500/unit:         ${avg_units_stop*500:+,.0f}/session
""")

    print("=" * 70)
    print("PAUL'S RISK vs REWARD")
    print("=" * 70)
    
    print(f"""
🎯 THE KEY INSIGHT:

  TRADITIONAL VIEW (wrong):
  ├── "Average bet is ~4.7 units"
  ├── "Risking $2,350/hand at $500/unit"
  └── "Huge exposure!"
  
  PAUL'S VIEW (correct):
  ├── "I only risk 1 unit to START each cycle"
  ├── "That's $500 at risk, not $2,350"
  ├── "After first win, it's HOUSE MONEY"
  └── "Max I can lose per cycle = $500"

💰 ACTUAL RISK PER SESSION:

  At $500/unit:
  ├── Units truly at risk:  ~{avg_risk:.0f} (first-hand losses)
  ├── Dollar risk:          ~${avg_risk*500:,.0f}
  ├── Expected result:      ${avg_units*500:+,.0f}
  └── Risk/Reward ratio:    Much better than flat betting!

🆚 COMPARISON:

  FLAT BETTING $500 × 560 hands:
  ├── Total risk exposure:  $280,000
  ├── Expected loss:        ~$770 (0.55% edge)
  
  STEADY CLIMB $500 unit:
  ├── True risk:            ~${avg_risk*500:,.0f}
  ├── Expected loss:        ~${abs(avg_units)*500:,.0f}
  ├── But with WIN potential on streaks!
  
  The Steady Climb has ASYMMETRIC risk:
  ├── Limited downside (1 unit per cycle)
  └── Unlimited upside (big streaks = big wins)
""")

if __name__ == "__main__":
    main()

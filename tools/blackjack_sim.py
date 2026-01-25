#!/usr/bin/env python3
"""
Paul's Blackjack Strategy Simulator
Compares: Steady Climb with vs without +100 unit profit stop
"""

import random
import statistics
from dataclasses import dataclass
from typing import List, Tuple

# Game parameters (6-deck, S17, perfect basic strategy)
# Adjusted for correct ~0.5% house edge
WIN_PROB = 0.4262      # Player wins (non-BJ)
LOSS_PROB = 0.4802     # Dealer wins  
PUSH_PROB = 0.0936     # Tie
BJ_PROB = 0.0475       # Blackjack (subset, pays 3:2)
# Net: 42.62% + 4.75%*0.5 (BJ bonus) - 48.02% = ~-0.5% house edge

# Steady Climb betting progression
PROGRESSION = [1, 1, 2, 2, 4, 4, 8, 8, 16]
MAX_BET = 16  # Stay at 16 after reaching max

# Session parameters
HANDS_PER_HOUR = 70
HOURS_PER_SESSION = 8
HANDS_PER_SESSION = HANDS_PER_HOUR * HOURS_PER_SESSION  # 560 hands

# Stop parameters
PROFIT_STOP = 100      # Stop when up 100 units
LOSS_STOP = -500       # Stop when down 500 units

@dataclass
class SessionResult:
    final_units: int
    hands_played: int
    hit_profit_stop: bool
    hit_loss_stop: bool
    max_units: int
    min_units: int
    winning_streaks: List[int]

def play_hand() -> Tuple[str, float]:
    """Play a single hand, returns result and payout multiplier
    
    Correct probability distribution for 6-deck S17 (~0.5% house edge):
    - Blackjack: 4.75% (pays 1.5x) - adds 2.375% value
    - Regular win: 38.10% (pays 1x)
    - Loss: 45.75% (pays -1x)
    - Push: 11.40% (pays 0)
    Total: 100%
    
    EV check: 0.0475×1.5 + 0.381×1 - 0.4575×1 = 0.071 + 0.381 - 0.4575 = -0.0055 ≈ -0.5%
    """
    roll = random.random()
    
    # Cumulative probabilities (corrected for 0.5% house edge)
    if roll < 0.0475:  # 4.75% blackjack
        return 'blackjack', 1.5
    elif roll < 0.0475 + 0.3810:  # 38.10% regular win
        return 'win', 1.0
    elif roll < 0.0475 + 0.3810 + 0.4575:  # 45.75% loss
        return 'loss', -1.0
    else:  # 11.40% push
        return 'push', 0.0

def get_bet_amount(streak: int) -> int:
    """Get bet amount based on winning streak"""
    if streak < len(PROGRESSION):
        return PROGRESSION[streak]
    return MAX_BET

def simulate_session(use_profit_stop: bool = False) -> SessionResult:
    """Simulate a single session"""
    units = 0
    hands = 0
    streak = 0
    max_units = 0
    min_units = 0
    winning_streaks = []
    hit_profit_stop = False
    hit_loss_stop = False
    
    for _ in range(HANDS_PER_SESSION):
        # Check stops
        if use_profit_stop and units >= PROFIT_STOP:
            hit_profit_stop = True
            break
        if units <= LOSS_STOP:
            hit_loss_stop = True
            break
            
        # Get bet based on current streak
        bet = get_bet_amount(streak)
        
        # Play hand
        result, multiplier = play_hand()
        hands += 1
        
        if result == 'blackjack':
            units += int(bet * 1.5)
            streak += 1
        elif result == 'win':
            units += bet
            streak += 1
        elif result == 'loss':
            units -= bet
            if streak > 0:
                winning_streaks.append(streak)
            streak = 0  # Reset on loss
        # Push: no change to units or streak
        
        max_units = max(max_units, units)
        min_units = min(min_units, units)
    
    # Record final streak if any
    if streak > 0:
        winning_streaks.append(streak)
    
    return SessionResult(
        final_units=units,
        hands_played=hands,
        hit_profit_stop=hit_profit_stop,
        hit_loss_stop=hit_loss_stop,
        max_units=max_units,
        min_units=min_units,
        winning_streaks=winning_streaks
    )

def run_simulation(num_sessions: int, use_profit_stop: bool) -> dict:
    """Run multiple sessions and aggregate results"""
    results = []
    
    for _ in range(num_sessions):
        result = simulate_session(use_profit_stop)
        results.append(result)
    
    # Aggregate statistics
    final_units = [r.final_units for r in results]
    hands_played = [r.hands_played for r in results]
    
    winning_sessions = [r for r in results if r.final_units > 0]
    losing_sessions = [r for r in results if r.final_units < 0]
    breakeven_sessions = [r for r in results if r.final_units == 0]
    
    profit_stops = sum(1 for r in results if r.hit_profit_stop)
    loss_stops = sum(1 for r in results if r.hit_loss_stop)
    
    # Best and worst sessions
    best = max(results, key=lambda r: r.final_units)
    worst = min(results, key=lambda r: r.final_units)
    
    return {
        'num_sessions': num_sessions,
        'use_profit_stop': use_profit_stop,
        'total_units': sum(final_units),
        'avg_units_per_session': statistics.mean(final_units),
        'median_units': statistics.median(final_units),
        'std_dev': statistics.stdev(final_units) if len(final_units) > 1 else 0,
        'winning_sessions': len(winning_sessions),
        'losing_sessions': len(losing_sessions),
        'breakeven_sessions': len(breakeven_sessions),
        'win_rate': len(winning_sessions) / num_sessions * 100,
        'profit_stops_hit': profit_stops,
        'loss_stops_hit': loss_stops,
        'avg_hands_per_session': statistics.mean(hands_played),
        'best_session': best.final_units,
        'worst_session': worst.final_units,
        'best_max_units': best.max_units,
        'worst_min_units': worst.min_units,
    }

def main():
    NUM_SESSIONS = 10000
    
    print("=" * 70)
    print("PAUL'S BLACKJACK STRATEGY COMPARISON")
    print("Steady Climb: 1, 1, 2, 2, 4, 4, 8, 8, 16 (reset on loss)")
    print(f"Sessions: {NUM_SESSIONS:,}")
    print(f"Hands per session: {HANDS_PER_SESSION} (8 hours @ 70/hr)")
    print(f"Win (inc BJ): 42.85% / Loss: 45.75% / Push: 11.40%")
    print(f"House Edge: ~0.55% (6-deck, S17, perfect basic strategy)")
    print("=" * 70)
    
    # Run both strategies
    print("\nRunning simulations...")
    
    print("  Strategy A: No profit stop...")
    no_stop = run_simulation(NUM_SESSIONS, use_profit_stop=False)
    
    print("  Strategy B: +100 unit profit stop...")
    with_stop = run_simulation(NUM_SESSIONS, use_profit_stop=True)
    
    # Side-by-side comparison
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    
    headers = ["METRIC", "NO STOP", "+100 STOP", "DIFFERENCE"]
    print(f"\n{'METRIC':<35} {'NO STOP':>15} {'+100 STOP':>15} {'DIFF':>12}")
    print("-" * 80)
    
    metrics = [
        ("Total Units (all sessions)", no_stop['total_units'], with_stop['total_units']),
        ("Avg Units/Session", no_stop['avg_units_per_session'], with_stop['avg_units_per_session']),
        ("Median Units/Session", no_stop['median_units'], with_stop['median_units']),
        ("Std Deviation", no_stop['std_dev'], with_stop['std_dev']),
        ("Winning Sessions", no_stop['winning_sessions'], with_stop['winning_sessions']),
        ("Losing Sessions", no_stop['losing_sessions'], with_stop['losing_sessions']),
        ("Win Rate %", no_stop['win_rate'], with_stop['win_rate']),
        ("Avg Hands/Session", no_stop['avg_hands_per_session'], with_stop['avg_hands_per_session']),
        ("Best Session", no_stop['best_session'], with_stop['best_session']),
        ("Worst Session", no_stop['worst_session'], with_stop['worst_session']),
        ("Profit Stops Hit", no_stop['profit_stops_hit'], with_stop['profit_stops_hit']),
        ("Loss Stops Hit", no_stop['loss_stops_hit'], with_stop['loss_stops_hit']),
    ]
    
    for name, val_a, val_b in metrics:
        diff = val_b - val_a
        if isinstance(val_a, float):
            print(f"{name:<35} {val_a:>15.2f} {val_b:>15.2f} {diff:>+12.2f}")
        else:
            print(f"{name:<35} {val_a:>15,} {val_b:>15,} {diff:>+12,}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print(f"""
📊 KEY FINDINGS:

1. TOTAL EXPECTED LOSS:
   • No Stop:    {no_stop['total_units']:>+10,} units over {NUM_SESSIONS:,} sessions
   • +100 Stop:  {with_stop['total_units']:>+10,} units over {NUM_SESSIONS:,} sessions
   • Difference: {with_stop['total_units'] - no_stop['total_units']:>+10,} units

2. WIN RATE:
   • No Stop:    {no_stop['win_rate']:.1f}% winning sessions
   • +100 Stop:  {with_stop['win_rate']:.1f}% winning sessions
   • Difference: {with_stop['win_rate'] - no_stop['win_rate']:>+.1f}% more winning sessions

3. VARIANCE (Standard Deviation):
   • No Stop:    {no_stop['std_dev']:.1f} units
   • +100 Stop:  {with_stop['std_dev']:.1f} units
   • +100 Stop has {"LOWER" if with_stop['std_dev'] < no_stop['std_dev'] else "HIGHER"} variance

4. SESSION LENGTH:
   • No Stop:    {no_stop['avg_hands_per_session']:.0f} hands avg (full sessions)
   • +100 Stop:  {with_stop['avg_hands_per_session']:.0f} hands avg
   • Profit stop cuts sessions by {(1 - with_stop['avg_hands_per_session']/no_stop['avg_hands_per_session'])*100:.1f}%

5. PROFIT STOPS HIT:
   • {with_stop['profit_stops_hit']:,} sessions ({with_stop['profit_stops_hit']/NUM_SESSIONS*100:.1f}%) hit +100 unit stop

6. AT $100/UNIT:
   • No Stop Total:   ${no_stop['total_units']*100:>+12,}
   • +100 Stop Total: ${with_stop['total_units']*100:>+12,}
   
7. AT $500/UNIT:
   • No Stop Total:   ${no_stop['total_units']*500:>+12,}
   • +100 Stop Total: ${with_stop['total_units']*500:>+12,}
""")
    
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    units_saved = with_stop['total_units'] - no_stop['total_units']
    pct_hit_stop = with_stop['profit_stops_hit'] / NUM_SESSIONS * 100
    
    print(f"""
📊 REALITY CHECK:

The house edge (~0.5%) grinds down both strategies over time.
With 560 hands per session, variance can't overcome math.

🎰 +100 PROFIT STOP RESULTS:
   • Only {pct_hit_stop:.1f}% of sessions hit +100 target
   • Caps big wins (best: {with_stop['best_session']} vs {no_stop['best_session']} without stop)
   • Total units saved: {units_saved:+,} over {NUM_SESSIONS:,} sessions
   • Win rate: {with_stop['win_rate']:.1f}% vs {no_stop['win_rate']:.1f}% ({"better" if with_stop['win_rate'] > no_stop['win_rate'] else "worse"})

💡 WHY IT'S STILL SMART:

Even though the math is similar, the profit stop:
   1. ✅ Gets you off the table when ahead (stop giving it back)
   2. ✅ Creates a "mission accomplished" feeling  
   3. ✅ Reduces total table time ({(1 - with_stop['avg_hands_per_session']/no_stop['avg_hands_per_session'])*100:.1f}% less)
   4. ✅ Prevents the "I was up $10K now I'm down $5K" regret
   
⚠️ THE HARD TRUTH:
   • At $100/unit: Expect to lose ~${abs(with_stop['avg_units_per_session']*100):,.0f}/session
   • At $500/unit: Expect to lose ~${abs(with_stop['avg_units_per_session']*500):,.0f}/session
   • Over 100 sessions: ~${abs(with_stop['total_units']/100*100):,.0f} at $100/unit
   
🎯 PAUL'S OPTIMAL APPROACH:
   
   The +100 stop is PSYCHOLOGICALLY superior even if mathematically
   similar. When you hit +100, you WALK AWAY with a WIN. Without the
   stop, that same session might end at +50, +10, or even negative.
   
   Recreational value: HIGH
   Mathematical edge: NONE (house always wins long-term)
   Best use: Set the stop, enjoy the wins, accept the cost of entertainment
""")
    
    return no_stop, with_stop
    
    return no_stop, with_stop

if __name__ == "__main__":
    main()

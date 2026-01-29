# Trading Modes - Philosophy & Strategy

## Mode Overview

| Mode | Objective | SuperTrend | Mindset |
|------|-----------|------------|---------|
| **1 BULLISH** | MAXIMIZE GAINS | Bullish | Full risk, ride up |
| **2 CORRECTION** | SAFEGUARD LOSSES | Bullish (expecting pullback) | Protect profits |
| **3 BEARISH** | BREAKEVEN/PROFIT DOWN | Flipped bearish | Follow new trend |

---

## Mode 1: BULLISH 📈

### Objective: MAXIMIZE GAINS

**Philosophy:** Full risk on. Aggressive buying. Ride the trend up.

**When to use:**
- SuperTrend is bullish
- No warning signs
- Normal trending conditions

**Actions:**
- ✅ DCA buy every dip
- ✅ Grid shorts (scalp only)
- ❌ No hedge (full exposure)

**Mindset:** *"The trend is our friend - maximize upside."*

---

## Mode 2: CORRECTION 📊

### Objective: SAFEGUARD AGAINST LOSSES

**Philosophy:** Protect profits. Hedge exposure. Defensive posture.

**When to use:**
- SuperTrend still bullish BUT expecting pullback
- Parabolic move, overextended
- Pre-FOMC, pre-news protection
- RSI overbought

**Actions:**
- ✅ FULL HEDGE SHORT (protect profits)
- ✅ Grid BUYs active (accumulate cheaper on way down)
- ✅ Grid SHORTs active (capture correction moves)
- 🎯 Target: Gap fills, necklines, support levels

**Mindset:** *"Protect what we've made while positioning for next move."*

---

## Mode 3: BEARISH 🐻

### Objective: BREAKEVEN TO PROFIT RIDING DOWN

**Philosophy:** SuperTrend changed. Go with the new trend.

**When to use:**
- SuperTrend FLIPPED bearish
- Bear flag confirmed and broke down
- Major support broken
- Trend reversal confirmed

**Actions:**
- ❌ STOP all new buys (don't fight the trend)
- ✅ Ride shorts down
- ✅ Try to breakeven or profit from drop

**Mindset:** *"Trend changed - adapt and profit from the new direction."*

---

## Quick Reference

```
SITUATION                     → MODE
─────────────────────────────────────────
SuperTrend bullish, clear     → 1 BULLISH
Parabolic, want protection    → 2 CORRECTION
Pre-FOMC hedge                → 2 CORRECTION
Bear flag spotted             → 2 CORRECTION (then 3 if confirms)
SuperTrend flipped bearish    → 3 BEARISH
Major breakdown confirmed     → 3 BEARISH
Pattern invalidated (new high)→ 1 BULLISH
```

---

## Commands

```bash
python3 grid_control.py 1   # Activate Bullish Grid
python3 grid_control.py 2   # Activate Correction Grid
python3 grid_control.py 3   # Activate Bearish Sighting
```

**Voice Commands:**
- "Activate Bullish Grid" → Mode 1
- "Activate Correction Grid" → Mode 2
- "Activate Bearish Sighting" → Mode 3

---

*Ghost Commander IBKR - MGC Futures*
*Aligned with MT5 Ghost Commander v0201*

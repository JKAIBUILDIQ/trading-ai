# Quinn/Neo Trading Mode Commands
## Ghost Commander IBKR - MGC Futures

---

## 🎯 MODE PHILOSOPHY

| Mode | Goal | Mindset |
|------|------|---------|
| **BULLISH** | Maximize Gains | "Ride the trend, buy every dip" |
| **CORRECTION** | Safeguard Against Losses | "Protect profits, profit from chop" |
| **BEARISH** | Ride the Supertrend Change | "Go with the flow down, breakeven or profit" |

---

## 🎯 TRADING MODE COMMANDS

### Voice/Text Commands → Mode Input

| Command | Mode | Command |
|---------|------|---------|
| "Activate Bullish Grid" | Maximize gains | `python3 grid_control.py 1` |
| "Activate Correction Grid" | Safeguard losses | `python3 grid_control.py 2` |
| "Activate Bearish Sighting" | Ride trend change | `python3 grid_control.py 3` |

---

## 📈 MODE 1: BULLISH (Trend Following)
### Goal: MAXIMIZE GAINS

**Philosophy:** Full confidence in uptrend. DCA on dips, TP on the way back up.

**What happens:**
- ✅ DCA BUY ladder: ACTIVE (buy every $20 drop)
- ❌ Grid: OFF (no need - we know the direction!)
- ❌ Hedge SELL: OFF (no protection needed)

**Strategy:**
```
Price drops $20 → BUY 2 contracts
Price drops $40 → BUY 2 contracts  
Price drops $60 → BUY 4 contracts
Price bounces → TPs hit on way up!
```

**When to use:**
- Supertrend confirmed bullish
- Clear uptrend, no chop
- RSI healthy (40-70 range)
- "Ride the trend, buy the dips!"

---

## 📊 MODE 2: CORRECTION (Choppy/Sideways)
### Goal: SAFEGUARD AGAINST LOSSES + Profit from Chop

**Philosophy:** Market is choppy/uncertain. Use GRID to profit from oscillations both ways.

**What happens:**
- ❌ DCA BUY ladder: OFF (not trending, don't stack)
- ✅ Grid LONG levels: ACTIVE (buy at support levels)
- ✅ Grid SHORT levels: ACTIVE (sell at resistance levels)
- ✅ Hedge SELL: ACTIVE (insurance)

**Strategy:**
```
Price rises to $5,591 → SHORT 2, TP at $5,571
Price drops to $5,551 → BUY 2, TP at $5,571
RESPAWN after each TP → Profit from every oscillation!
```

**When to use:**
- Parabolic exhaustion (consolidation)
- Pre-news chop (FOMC, NFP)
- Range-bound / sideways market
- "Profit from the whipsaw!"

---

## 🐻 MODE 3: BEARISH (Supertrend Switch)
### Goal: KILL LONGS, SCALE IN SHORTS

**Philosophy:** Trend is reversing. CLOSE all longs. Ride the new trend DOWN.

**What happens:**
- ❌ DCA BUY ladder: OFF
- ❌ Grid LONG levels: OFF
- ✅ **STOP NEW BUYS** (exit before more damage!)
- ✅ **SCALE IN SHORTS** (DCA shorts on bounces)
- ✅ Hedge SELL: ACTIVE

**Strategy:**
```
On activation → Stop all new BUY orders
Price bounces $20 → SHORT 2 contracts
Price bounces $40 → SHORT 2 contracts
Price drops → TPs hit, ride it down!
```

**When to use:**
- Supertrend flips bearish
- Major support breakdown
- Bear flag confirms
- "The trend changed - flip with it!"

**Exit to Mode 1 when:**
- New ATH (reversal failed)
- Supertrend flips back bullish

---

## 🔄 WHIPSAW GRID (Mode 2 Only - Correction/Choppy)

Grid is ONLY active in Mode 2 for choppy/sideways markets:

```
SHORT LEVELS (fade rises):
$5,611 ─── SHORT 2 ─── TP @ $5,591
$5,591 ─── SHORT 2 ─── TP @ $5,571
         │
$5,571 ══╪══ CENTER ══════════════════
         │
$5,551 ─── BUY 2 ──── TP @ $5,571
$5,531 ─── BUY 2 ──── TP @ $5,551
$5,511 ─── BUY 4 ──── TP @ $5,531
$5,491 ─── BUY 4 ──── TP @ $5,511
...down to $4,650 (THE GAP)
```

**RESPAWN:** After TP hit, level resets for next cycle

**Grid OFF in Mode 1 & 3** - those modes are directional (trend following)

---

## 📋 QUICK REFERENCE

| Situation | Command | Mode | Goal |
|-----------|---------|------|------|
| Normal uptrend | `grid_control.py 1` | BULLISH | Maximize gains |
| Parabolic/overextended | `grid_control.py 2` | CORRECTION | Safeguard profits |
| Bear flag spotted | `grid_control.py 3` | BEARISH | Ride it down |
| Divergence detected | `grid_control.py 3` | BEARISH | Ride it down |
| Pre-FOMC/news | `grid_control.py 2` | CORRECTION | Safeguard profits |
| Support breakdown | `grid_control.py 3` | BEARISH | Ride it down |
| New ATH / pattern fails | `grid_control.py 1` | BULLISH | Back to maximize |

---

## 💰 P&L EXPECTATIONS BY MODE

| Mode | Best Case | Worst Case | Philosophy |
|------|-----------|------------|------------|
| **Bullish** | Big gains on rally | Losses on drop | "Go for max profit" |
| **Correction** | Protected both ways | Small hedge loss if rally | "Protect what I have" |
| **Bearish** | Profit riding down | Breakeven if reverses | "Don't fight the trend" |

---

## 🎓 PATTERN RECOGNITION TRIGGERS

### Switch to Mode 2 (Correction) when:
- RSI > 80 (overbought)
- Price > $150 above EMA20
- 5-day gain > 10%
- Major news event upcoming (FOMC, NFP)

### Switch to Mode 3 (Bearish) when:
- Bear flag pattern (consolidation after spike)
- RSI divergence (price up, RSI down)
- Failed breakout attempt
- Support breakdown imminent

### Switch to Mode 1 (Bullish) when:
- Bear flag invalidated (new high)
- RSI returns to 40-60 range
- Pullback complete, uptrend resumes
- Pattern resolves bullishly

---

*Ghost Commander IBKR - Aligned with MT5 v0201*
*Generated: January 29, 2026*

# MGC TRADING PLAN - FOMC WEEK
## January 29, 2026 | Pre-FOMC Setup

---

## ACCOUNT INFO
- **Account:** IBKR Paper (DUP177636)
- **Instrument:** MGC (Micro Gold Futures) - MGCJ6 April Contract
- **Contract Value:** $10 per point ($1 gold move = $10)

---

## CURRENT POSITION

| Component | Contracts | Entry |
|-----------|-----------|-------|
| Correction Hedge | -9 | ~$5,564 |
| Grid Short L1 | -2 | $5,583.15 |
| **NET POSITION** | **-11 SHORT** | ~$5,568 |

---

## STRATEGY: WHIPSAW GRID

**Concept:** Profit from BOTH directions with $20 grid spacing

> "The chop is the opportunity"

```
                     SHORT ZONE (fade rises)
                          ▲
$5,731 ──── SHORT 4 ─────┤ FULL
$5,711 ──── SHORT 4 ─────┤ FULL
$5,691 ──── SHORT 4 ─────┤ FULL
$5,671 ──── SHORT 4 ─────┤ FULL
$5,651 ──── SHORT 4 ─────┤ FULL
$5,631 ──── SHORT 4 ─────┤ FULL
$5,611 ──── SHORT 2 ─────┤ HALF ← Next short trigger
$5,591 ──── SHORT 2 ═════╡ HALF ✓ FILLED @ $5,583
                          │
$5,571 ════ CENTER ═══════╋═══════════════════════
                          │
$5,551 ──── BUY 2 ────────┤ HALF ← Next buy trigger
$5,531 ──── BUY 2 ────────┤ HALF
$5,511 ──── BUY 4 ────────┤ FULL
$5,491 ──── BUY 4 ────────┤ FULL
$5,471 ──── BUY 4 ────────┤ FULL
$5,451 ──── BUY 4 ────────┤ FULL
$5,431 ──── BUY 4 ────────┤ FULL
$5,411 ──── BUY 4 ────────┤ FULL
$5,231 ──── BUY 8 ────────┤ DOUBLE (Fib 23.6%)
$5,000 ──── BUY 8 ────────┤ DOUBLE (Psychological)
$4,650 ──── BUY 12 ───────┤ TRIPLE (THE GAP)
                          ▼
                     LONG ZONE (accumulate dips)
```

---

## TAKE PROFIT LOGIC

| Side | TP Rule | Example |
|------|---------|---------|
| SHORTS | TP when price drops $20 | SHORT @ $5,583 → TP @ $5,563 = **+$400** |
| LONGS | TP when price rises $20 | BUY @ $5,551 → TP @ $5,571 = **+$400** |

**RESPAWN:** After TP, level resets to PENDING for next cycle

---

## PROFIT MATH

### Per $20 Move (1 cycle)

| Position | Contracts | Profit |
|----------|-----------|--------|
| HALF | 2 | $400 |
| FULL | 4 | $800 |
| DOUBLE | 8 | $1,600 |
| TRIPLE | 12 | $2,400 |

### Daily Potential (5 oscillations)
- Cycling HALF: 5 × $400 = **$2,000/day**
- Cycling FULL: 5 × $800 = **$4,000/day**

---

## FOMC SCENARIOS

### Scenario A: HAWKISH (Gold Drops)
- Short hedge profits as price falls
- Grid BUY levels trigger, accumulating longs
- If drops to $5,551 → BUY 2, TP at $5,571
- Deep drop to $5,231 or lower = DOUBLE positions

### Scenario B: DOVISH (Gold Spikes)
- Short hedge takes heat (offset by grid)
- Grid SHORT levels trigger above
- If rises to $5,611 → SHORT 2, TP at $5,591
- Ride the whipsaw back down

### Scenario C: CHOP (Sideways) ⭐ IDEAL
- Both BUY and SHORT triggers cycling
- Maximum profit from oscillations

---

## RISK MANAGEMENT

| Parameter | Value |
|-----------|-------|
| Max Long Contracts | 50 |
| Max Short Contracts | 30 |
| Stop Loss | $100 from avg entry |
| Commission | $0.87/contract/side |

---

## ACTIVE BOTS

| Bot | Client ID | Status |
|-----|-----------|--------|
| Ghost Commander | 100 | Paused (SuperTrend DCA) |
| Casper | 200 | Paused (TP Management) |
| **Whipsaw Grid** | **400** | **✓ ACTIVE** |

---

## CURRENT TRIGGERS WATCHING

| Trigger | Price | Action |
|---------|-------|--------|
| 📈 Rise | $5,611 | SHORT L2 HALF (add 2 shorts) |
| 📉 Drop | $5,563 | SHORT TP (close 2 for +$400) |
| 📉 Drop | $5,551 | BUY L1 HALF (start accumulating) |

---

## SUMMARY

- **BIAS:** Neutral (profiting from volatility, not direction)
- **HEDGE:** 9 shorts for FOMC downside protection
- **GRID:** Active whipsaw trading on $20 intervals
- **GOAL:** Capture profits from post-FOMC volatility

> "We don't predict. We prepare for BOTH directions."

---

*Generated: January 29, 2026*

# Casper's DROPBUY DCA Strategy - The Breadwinner

**Source:** `Crellastein_Casper.mq5` - The strategy that generates consistent profits.

---

## Core Philosophy

```
"Buy on drops, not on rises. Let winners run. Don't fight the trend."
```

---

## DROPBUY Settings (Parabolic Mode)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Trigger** | $20 drops | Buy every $20 drop from session high |
| **Max Levels** | 3 | Maximum DCA entries per cycle |
| **TP** | +$50 from avg | Take profit $50 above average entry |

### Lot Sizing Ladder

| Level | Lots | When |
|-------|------|------|
| L1 | 0.25 | First $20 drop |
| L2 | 0.25 | Second $20 drop ($40 total) |
| L3 | 0.50 | Third $20 drop ($60 total) |
| L4 | 0.50 | Fourth $20 drop ($80 total) |
| L5 | 1.00 | Fifth $20 drop ($100 total) |

---

## The DROPBUY Cycle

```
Session High: $5600
                │
    ┌───────────┼───────────┐
    │           │           │
$5580 ─────────┼───────────┼─── DROPBUY L1 (0.25 lots)
    │           │           │
$5560 ─────────┼───────────┼─── DROPBUY L2 (0.25 lots)  
    │           │           │
$5540 ─────────┼───────────┼─── DROPBUY L3 (0.50 lots)
    │           │           │
    └───────────┼───────────┘
                │
         Average Entry: $5560
         TP Target: $5610 (+$50)
                │
                ▼
         Price Recovers to $5610
                │
         ┌──────┴──────┐
         │   TP HIT!   │
         │  All close  │
         │  +$50/lot   │
         └──────┬──────┘
                │
         Keep 1 as FREEROLL
         (paid by profits)
```

---

## Anti-Stacking Rules

**CRITICAL: $20 minimum gap between ANY position**

```python
# Check ALL Gold BUY positions - NOT JUST CASPER!
# This prevents stacking with Ghost, DropBuy, Breakout buys, etc.

if distance_from_any_position < $20:
    BLOCK_ENTRY  # Wait for bigger dip!
```

---

## Breakout Detection

When all positions close AND price breaks above session high:

```
All TPs Hit → Check for Breakout

If price > session_high + $5:
    → BREAKOUT BUY (don't wait for drop!)
    → New session high set
    → DCA cycle resets

Else:
    → Wait for next drop
    → DCA cycle resets at current price
```

---

## Trailing TP (Lock Profits)

| Parameter | Value |
|-----------|-------|
| Trail Start | +$10 profit |
| Trail Distance | $8 behind price |

```
Entry: $5500
Price rises to $5520 (+$20 profit)
    → Trail SL set to $5512 (locks $12 profit)

Price rises to $5540 (+$40 profit)  
    → Trail SL moves to $5532 (locks $32 profit)

Price drops to $5532
    → SL hit, exit with +$32 profit (not full +$40, but guaranteed!)
```

---

## IBKR Equivalent (MGC)

### Conversion: MT5 → IBKR

| MT5 (XAUUSD) | IBKR (MGC) | Ratio |
|--------------|------------|-------|
| 1.0 lot | 10 contracts | 1:10 |
| 0.5 lot | 5 contracts | 1:10 |
| 0.25 lot | 2-3 contracts | 1:10 |

### DROPBUY Levels for MGC

| Level | Price Drop | Contracts | Running Total |
|-------|------------|-----------|---------------|
| L1 | -$20 | 2 | 2 |
| L2 | -$40 | 3 | 5 |
| L3 | -$60 | 5 | 10 |
| **MAX** | **-$60** | **10** | **= 1 MT5 lot** |

---

## Code Translation

### MT5 (Casper)
```mq5
void CheckDropBuyDCA()
{
   // Track session high
   if(currentPrice > g_dropBuy_SessionHigh)
      g_dropBuy_SessionHigh = currentPrice;
   
   // Calculate drop
   double dropFromHigh = g_dropBuy_SessionHigh - currentPrice;
   int targetLevel = (int)(dropFromHigh / DropBuy_TriggerPips);
   
   // Buy if new level reached
   if(targetLevel > g_dropBuy_CurrentLevel)
   {
      // Anti-stacking check
      if(IsTooCloseToMetaPosition(symbol, ask, 20.0))
         return;  // BLOCK!
      
      // Execute DROPBUY
      g_trade.Buy(lots, symbol, ask, 0, 0, "DROPBUY|L" + level);
   }
}
```

### IBKR (Ghost Commander)
```python
def check_dropbuy_dca(self):
    # Track session high
    if current_price > self.session_high:
        self.session_high = current_price
    
    # Calculate drop
    drop_from_high = self.session_high - current_price
    target_level = int(drop_from_high / 20)  # $20 drops
    
    # Buy if new level reached
    if target_level > self.current_level:
        # Anti-stacking check
        if self.too_close_to_position(20.0):
            return  # BLOCK!
        
        # Execute DROPBUY
        self.buy(contracts, "DROPBUY_L" + str(level))
```

---

## Summary: Why Casper Wins

1. **Only buys on DROPS** - Never chases
2. **Scales in progressively** - Bigger size at better prices
3. **TP based on average** - Always profitable exit
4. **Anti-stacking** - No position clumping
5. **Breakout detection** - Catches momentum shifts
6. **Trailing TP** - Locks profits, lets winners run
7. **NO SHORTS** - Trusts the Gold thesis

---

## Commands

```bash
# Show current DROPBUY status
ghoststatus

# Set Mode 1 (DROPBUY active)
mode1

# Check positions
twscheck
```

---

*Casper = The Breadwinner. Copy his homework.*

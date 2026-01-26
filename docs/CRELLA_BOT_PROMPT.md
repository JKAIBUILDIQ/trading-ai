# 🤖 CRELLA BOT BUILD PROMPT

## Quest: Build Steady Climb MT5 Trading Bots

---

### TLDR
Build TWO MT5 Expert Advisors using Paul's Steady Climb betting progression (1,1,2,2,4,4,8,8) applied to forex trading. Reset to 1 unit on any loss. Scale up only with winnings.

---

## 🎯 WHAT TO BUILD

### Bot 1: `v0091_NEO_SteadyClimb.mq5`
- **Signal Source**: NEO API (http://146.190.188.208:8750/api/gaps/best)
- **When to trade**: When NEO returns high-confidence signals (65%+)
- **Position sizing**: Steady Climb progression

### Bot 2: `v0092_Technical_SteadyClimb.mq5`
- **Signal Source**: Pure technical indicators (no API)
- **Indicators**: Gap detection (74% win rate), RSI, EMA crossover, Supertrend
- **When to trade**: When 3+ indicators align
- **Position sizing**: Steady Climb progression

---

## 📊 THE STEADY CLIMB PROGRESSION

```
Position:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
Units:     1   1   2   2   4   4   8   8
```

### Rules
```cpp
// WIN: Advance to next position
if(tradeProfit > 0) {
    if(position < 7) position++;
    // Lock in gains, continue building
}

// LOSS: RESET to position 0 (1 unit) - NO EXCEPTIONS
if(tradeProfit < 0) {
    position = 0;  // Always reset
    // Never chase, never add to losers
}
```

---

## 🔑 KEY VARIABLES

```cpp
// Progression array
int PROGRESSION[] = {1, 1, 2, 2, 4, 4, 8, 8};

// State tracking
int    g_CurrentPosition = 0;     // 0-7 index
double g_CycleProfit = 0;         // Profit this cycle
int    g_ConsecutiveWins = 0;     // Wins since reset

// Lot calculation
double GetLot() {
    return BaseLotSize * PROGRESSION[g_CurrentPosition];
}
```

---

## 📡 NEO API FORMAT (Bot 1)

**Endpoint**: `GET http://146.190.188.208:8750/api/gaps/best`

**Response**:
```json
{
  "success": true,
  "has_trade": true,
  "trade": {
    "symbol": "XAUUSD",
    "action": "BUY",
    "confidence": 76,
    "entry": 5080.00,
    "stop_loss": 5065.00,
    "take_profit": 5120.00,
    "risk_reward": "2.67"
  }
}
```

**Trade when**: `confidence >= 65` AND `has_trade == true`

---

## 📊 TECHNICAL SIGNALS (Bot 2)

### Signal Score System (0-5 points)

| Signal | Condition | Points |
|--------|-----------|--------|
| Gap Fill | Active unfilled gap | +2 |
| RSI | Oversold (<30) or Overbought (>70) | +1 |
| EMA Cross | 9 EMA crosses 21 EMA | +1 |
| Supertrend | Price above/below ST | +1 |

**Trade when**: Score >= 3 AND direction agrees

### Gap Detection (Most Important - 74% Win Rate!)
```cpp
double prevClose = iClose(Symbol(), PERIOD_D1, 1);
double todayOpen = iOpen(Symbol(), PERIOD_D1, 0);
double gapSize = MathAbs(todayOpen - prevClose);

if(gapSize >= MinGapSize) {
    // Gap UP = SELL signal (fade it)
    // Gap DOWN = BUY signal (fade it)
    direction = (todayOpen > prevClose) ? -1 : 1;
}
```

---

## ⚙️ INPUT PARAMETERS

```cpp
// Steady Climb
input double SC_BaseLotSize = 0.01;    // 1 unit = 0.01 lot
input double SC_MaxLotSize = 0.10;     // Cap at 0.10
input bool   SC_ResetDaily = true;     // Reset progression each day

// Risk Management
input int    MaxTradesPerDay = 10;
input double DailyLossLimit = 500;     // Stop trading if down $500

// NEO API (Bot 1 only)
input string NEO_ApiUrl = "http://146.190.188.208:8750";
input int    NEO_MinConfidence = 65;

// Technical (Bot 2 only)
input int    MinSignalScore = 3;       // Need 3+ signals aligned
```

---

## 🖥️ DISPLAY PANEL

Show on chart:
```
🎰 STEADY CLIMB
Position: 5/8 | Units: 4 | Lot: 0.04
[✓1] → [✓1] → [✓2] → [✓2] → [4] → 4 → 8 → 8
Cycle P&L: +$320 | Wins: 4
```

---

## ⌨️ KEYBOARD SHORTCUTS

- **R** = Reset progression to position 1
- **S** = Print status to log
- **G** = Show gap status

---

## 🎯 EXPECTED RESULTS

From simulation (1000 sessions, 20 trades each):

| Win Rate | Strategy | Avg Session P&L |
|----------|----------|-----------------|
| 50% | Steady Climb | +$654 (+6.5%) |
| 55% | Steady Climb | +$1,090 (+10.9%) |
| **74%** | **Gap Fill + SC** | **+$3,049 (+30.5%)** |

---

## ✅ ACCEPTANCE CRITERIA

1. [ ] Both EAs compile without errors
2. [ ] Progression advances on win (1→1→2→2→4→4→8→8)
3. [ ] Progression resets to 1 on ANY loss
4. [ ] Lot size = BaseLot × Units
5. [ ] Daily trade limit enforced
6. [ ] NEO API signals parsed correctly (Bot 1)
7. [ ] Technical signals calculated correctly (Bot 2)
8. [ ] On-chart display working
9. [ ] Keyboard shortcuts functional
10. [ ] State persists across restarts

---

## 📁 DELIVERABLES

```
MQL5/Experts/
├── v0091_NEO_SteadyClimb.mq5      ← Bot 1
└── v0092_Technical_SteadyClimb.mq5 ← Bot 2

MQL5/Include/
└── SteadyClimbLib.mqh              ← Shared logic
```

---

**Priority**: HIGH
**Deadline**: ASAP
**Test on**: Demo account first

---

### THE ONE RULE THAT MATTERS

> **ON ANY LOSS: RESET TO 1 UNIT. NO EXCEPTIONS.**
> 
> This is not Martingale. We don't chase losses.
> We only risk 1 unit of OUR money.
> Everything else is house money.

---

Good luck Crella! 🚀

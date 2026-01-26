# 🤖 CRELLA MT5 BOT BUILD INSTRUCTIONS

## Mission: Build Dual-Model Steady Climb Trading Bot

Build TWO MT5 Expert Advisors that implement Paul's Steady Climb position sizing strategy:
1. **Model A: NEO Signals** - AI-driven signals from NEO API
2. **Model B: Technical Pure** - Pure technical analysis (no AI dependency)

---

## 📋 CORE STRATEGY: STEADY CLIMB POSITION SIZING

### The Progression
```
Position:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
Units:     1   1   2   2   4   4   8   8
```

### Rules
1. **START**: Begin at Position 1 (1 unit)
2. **WIN**: Advance to next position
3. **LOSS**: IMMEDIATELY reset to Position 1 (1 unit)
4. **MAX**: At Position 8, either stay or reset after completing cycle

### Key Principle
> "You only risk 1 unit from YOUR bankroll. Everything else is house money."

---

## 🔧 MODEL A: NEO SIGNALS EA

### File: `v0091_NEO_SteadyClimb.mq5`

### Signal Source
Fetch signals from NEO API:
```cpp
string NEO_API_URL = "http://146.190.188.208:8750/api/gaps/best";
// Also check:
// http://146.190.188.208:8700/api/neo/gold-forex
// http://146.190.188.208:8080/api/neo/brain
```

### Input Parameters
```cpp
input group "=== 🎰 STEADY CLIMB SETTINGS ==="
input double SC_BaseLotSize = 0.01;        // Base lot (1 unit)
input double SC_MaxLotSize = 0.10;         // Maximum lot size
input bool   SC_ResetOnNewDay = true;      // Reset progression daily
input bool   SC_StayAtMaxLevel = true;     // Stay at 8 units after full cycle

input group "=== 📡 NEO API SETTINGS ==="  
input string NEO_ApiUrl = "http://146.190.188.208:8750";
input int    NEO_MinConfidence = 65;       // Min confidence to trade
input int    NEO_CheckIntervalSec = 60;    // Check API every N seconds

input group "=== ⚡ TRADE SETTINGS ==="
input int    MaxTradesPerDay = 10;         // Daily trade limit
input double DailyLossLimit = 500;         // Stop if down this much
input int    MaxSlippagePoints = 30;       // Max slippage
input string TradeComment = "v0091_NEO";   // Trade comment
```

### Core Logic
```cpp
//+------------------------------------------------------------------+
//| Steady Climb State                                                |
//+------------------------------------------------------------------+
int    g_CurrentPosition = 0;     // 0-7 index
int    g_ConsecutiveWins = 0;
double g_CycleProfit = 0;
int    g_TradesToday = 0;
datetime g_LastTradeDay = 0;

int PROGRESSION[] = {1, 1, 2, 2, 4, 4, 8, 8};

//+------------------------------------------------------------------+
//| Get current lot size                                              |
//+------------------------------------------------------------------+
double GetSteadyClimbLot() {
   int units = PROGRESSION[g_CurrentPosition];
   double lot = SC_BaseLotSize * units;
   return MathMin(lot, SC_MaxLotSize);
}

//+------------------------------------------------------------------+
//| Record a win - advance position                                   |
//+------------------------------------------------------------------+
void RecordWin(double profit) {
   g_CycleProfit += profit;
   g_ConsecutiveWins++;
   
   if(g_CurrentPosition < 7) {
      g_CurrentPosition++;
      Print("🎰 WIN! Advanced to Position ", g_CurrentPosition + 1, 
            " (", PROGRESSION[g_CurrentPosition], " units)");
   } else {
      Print("🎰 WIN at MAX! Cycle profit: $", g_CycleProfit);
      if(!SC_StayAtMaxLevel) {
         ResetProgression("Cycle complete");
      }
   }
}

//+------------------------------------------------------------------+
//| Record a loss - RESET to position 0                               |
//+------------------------------------------------------------------+
void RecordLoss(double loss) {
   Print("❌ LOSS at Position ", g_CurrentPosition + 1, 
         " - Resetting. Cycle profit was: $", g_CycleProfit);
   ResetProgression("Loss");
}

//+------------------------------------------------------------------+
//| Reset progression                                                 |
//+------------------------------------------------------------------+
void ResetProgression(string reason) {
   g_CurrentPosition = 0;
   g_ConsecutiveWins = 0;
   g_CycleProfit = 0;
   Print("🔄 Progression RESET: ", reason);
}

//+------------------------------------------------------------------+
//| Fetch NEO signal                                                  |
//+------------------------------------------------------------------+
bool GetNEOSignal(string &symbol, int &direction, double &sl, double &tp, int &confidence) {
   // Use WebRequest to fetch from NEO API
   string url = NEO_ApiUrl + "/api/gaps/best";
   
   char post[], result[];
   string headers = "Content-Type: application/json\r\n";
   
   int res = WebRequest("GET", url, headers, 5000, post, result, headers);
   
   if(res != 200) return false;
   
   // Parse JSON response
   string response = CharArrayToString(result);
   
   // Extract fields (simplified - use proper JSON parser)
   if(StringFind(response, "\"has_trade\":true") < 0) return false;
   
   // Parse symbol, action, confidence, sl, tp from response
   // ... JSON parsing logic ...
   
   if(confidence < NEO_MinConfidence) return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Main trading logic                                                |
//+------------------------------------------------------------------+
void OnTick() {
   // Reset daily counters
   if(g_LastTradeDay != iTime(Symbol(), PERIOD_D1, 0)) {
      g_TradesToday = 0;
      g_LastTradeDay = iTime(Symbol(), PERIOD_D1, 0);
      if(SC_ResetOnNewDay) ResetProgression("New day");
   }
   
   // Check trade limits
   if(g_TradesToday >= MaxTradesPerDay) return;
   
   // Check for open positions
   if(PositionSelect(Symbol())) {
      CheckPositionClose();  // Monitor for TP/SL hit
      return;
   }
   
   // Get NEO signal
   string signalSymbol;
   int direction;
   double sl, tp;
   int confidence;
   
   if(!GetNEOSignal(signalSymbol, direction, sl, tp, confidence)) return;
   
   // Execute trade with Steady Climb lot size
   double lotSize = GetSteadyClimbLot();
   
   ExecuteTrade(direction, lotSize, sl, tp, TradeComment);
   g_TradesToday++;
}

//+------------------------------------------------------------------+
//| Check if position closed and record result                        |
//+------------------------------------------------------------------+
void CheckPositionClose() {
   // Check last closed trade
   if(HistorySelect(TimeCurrent() - 3600, TimeCurrent())) {
      int total = HistoryDealsTotal();
      for(int i = total - 1; i >= 0; i--) {
         ulong ticket = HistoryDealGetTicket(i);
         if(HistoryDealGetString(ticket, DEAL_COMMENT) == TradeComment) {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            if(profit > 0) {
               RecordWin(profit);
            } else {
               RecordLoss(profit);
            }
            break;
         }
      }
   }
}
```

### Display Panel
```cpp
void DisplaySteadyClimbPanel() {
   int x = 10, y = 50;
   string prefix = "SC_";
   
   // Title
   ObjectCreate(0, prefix+"TITLE", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0, prefix+"TITLE", OBJPROP_TEXT, "🎰 STEADY CLIMB");
   ObjectSetInteger(0, prefix+"TITLE", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix+"TITLE", OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, prefix+"TITLE", OBJPROP_COLOR, clrGold);
   
   // Position display
   string posText = "Position: " + IntegerToString(g_CurrentPosition + 1) + 
                    " | Units: " + IntegerToString(PROGRESSION[g_CurrentPosition]) +
                    " | Lot: " + DoubleToString(GetSteadyClimbLot(), 2);
   
   ObjectCreate(0, prefix+"POS", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0, prefix+"POS", OBJPROP_TEXT, posText);
   ObjectSetInteger(0, prefix+"POS", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix+"POS", OBJPROP_YDISTANCE, y + 20);
   ObjectSetInteger(0, prefix+"POS", OBJPROP_COLOR, clrWhite);
   
   // Progression visual
   string progText = "";
   for(int i = 0; i < 8; i++) {
      if(i == g_CurrentPosition)
         progText += "[" + IntegerToString(PROGRESSION[i]) + "]";
      else if(i < g_CurrentPosition)
         progText += "✓";
      else
         progText += " " + IntegerToString(PROGRESSION[i]) + " ";
      if(i < 7) progText += "→";
   }
   
   ObjectCreate(0, prefix+"PROG", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0, prefix+"PROG", OBJPROP_TEXT, progText);
   ObjectSetInteger(0, prefix+"PROG", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix+"PROG", OBJPROP_YDISTANCE, y + 40);
   ObjectSetInteger(0, prefix+"PROG", OBJPROP_COLOR, clrCyan);
}
```

---

## 🔧 MODEL B: TECHNICAL PURE EA

### File: `v0092_Technical_SteadyClimb.mq5`

### Signal Sources (No AI Dependency)
Uses pure technical indicators:

1. **Gap Fill Detection** (74% win rate from research)
2. **RSI Divergence** (oversold/overbought)
3. **Supertrend** (trend direction)
4. **EMA Crossover** (momentum)

### Input Parameters
```cpp
input group "=== 🎰 STEADY CLIMB SETTINGS ==="
input double SC_BaseLotSize = 0.01;        
input double SC_MaxLotSize = 0.10;         
input bool   SC_ResetOnNewDay = true;      

input group "=== 📊 GAP FILL SETTINGS ==="
input bool   EnableGapTrading = true;
input double MinGapPercent = 0.3;          // Minimum gap size (%)
input double MaxGapPercent = 2.0;          // Max gap (avoid breakaway)
input double GapFillTPPercent = 80;        // TP at 80% of gap
input double GapFillSLPercent = 50;        // SL at 50% extension

input group "=== 📈 TREND SETTINGS ==="
input bool   EnableTrendTrading = true;
input int    RSI_Period = 14;
input int    RSI_Oversold = 30;
input int    RSI_Overbought = 70;
input int    EMA_Fast = 9;
input int    EMA_Slow = 21;
input int    Supertrend_Period = 10;
input double Supertrend_Multiplier = 3.0;

input group "=== 🎯 SIGNAL FILTERS ==="
input int    MinSignalScore = 3;           // Minimum combined score (1-5)
input bool   RequireMultipleConfirm = true; // Need 2+ indicators aligned
```

### Technical Signal Logic
```cpp
//+------------------------------------------------------------------+
//| Calculate signal score (0-5)                                      |
//+------------------------------------------------------------------+
int CalculateSignalScore(int &direction) {
   int score = 0;
   int bullish = 0;
   int bearish = 0;
   
   // 1. GAP FILL CHECK (strongest signal - 74% win rate)
   GapInfo gap;
   if(DetectGap(gap) && gap.isActive && !gap.isFilled) {
      if(gap.direction == "UP") {
         bearish += 2;  // Fade the gap - SELL
      } else {
         bullish += 2;  // Fade the gap - BUY
      }
      score += 2;  // Gap signals worth more
   }
   
   // 2. RSI CHECK
   double rsi = iRSI(Symbol(), PERIOD_H1, RSI_Period, PRICE_CLOSE, 0);
   if(rsi < RSI_Oversold) {
      bullish++;
      score++;
   } else if(rsi > RSI_Overbought) {
      bearish++;
      score++;
   }
   
   // 3. EMA CROSSOVER
   double emaFast = iMA(Symbol(), PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE, 0);
   double emaSlow = iMA(Symbol(), PERIOD_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, 0);
   double emaFastPrev = iMA(Symbol(), PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaSlowPrev = iMA(Symbol(), PERIOD_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE, 1);
   
   // Bullish crossover
   if(emaFastPrev < emaSlowPrev && emaFast > emaSlow) {
      bullish++;
      score++;
   }
   // Bearish crossover
   if(emaFastPrev > emaSlowPrev && emaFast < emaSlow) {
      bearish++;
      score++;
   }
   
   // 4. SUPERTREND
   double st = GetSupertrend(Symbol(), PERIOD_H1, Supertrend_Period, Supertrend_Multiplier);
   double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   if(price > st) {
      bullish++;
      score++;
   } else {
      bearish++;
      score++;
   }
   
   // 5. MOMENTUM (price action)
   double close0 = iClose(Symbol(), PERIOD_H1, 0);
   double close3 = iClose(Symbol(), PERIOD_H1, 3);
   if(close0 > close3 * 1.002) {  // 0.2% up in 3 hours
      bullish++;
   } else if(close0 < close3 * 0.998) {
      bearish++;
   }
   
   // Determine direction
   if(bullish > bearish && bullish >= 2) {
      direction = 1;  // BUY
   } else if(bearish > bullish && bearish >= 2) {
      direction = -1;  // SELL
   } else {
      direction = 0;  // No clear signal
   }
   
   return score;
}

//+------------------------------------------------------------------+
//| Main trading logic                                                |
//+------------------------------------------------------------------+
void OnTick() {
   // Daily reset
   CheckDailyReset();
   
   // Check trade limits
   if(g_TradesToday >= MaxTradesPerDay) return;
   
   // Don't trade if position open
   if(PositionSelect(Symbol())) {
      CheckPositionClose();
      return;
   }
   
   // Calculate signal
   int direction;
   int score = CalculateSignalScore(direction);
   
   // Require minimum score
   if(score < MinSignalScore || direction == 0) return;
   
   // Get Steady Climb lot size
   double lot = GetSteadyClimbLot();
   
   // Calculate SL/TP based on ATR
   double atr = iATR(Symbol(), PERIOD_H1, 14, 0);
   double sl, tp;
   
   if(direction > 0) {  // BUY
      sl = SymbolInfoDouble(Symbol(), SYMBOL_ASK) - atr * 1.5;
      tp = SymbolInfoDouble(Symbol(), SYMBOL_ASK) + atr * 2.5;
   } else {  // SELL
      sl = SymbolInfoDouble(Symbol(), SYMBOL_BID) + atr * 1.5;
      tp = SymbolInfoDouble(Symbol(), SYMBOL_BID) - atr * 2.5;
   }
   
   // Execute trade
   string comment = "v0092_TECH|Score:" + IntegerToString(score);
   ExecuteTrade(direction, lot, sl, tp, comment);
   g_TradesToday++;
   
   Print("📊 Signal: ", direction > 0 ? "BUY" : "SELL", 
         " | Score: ", score, 
         " | Lot: ", lot,
         " | Position: ", g_CurrentPosition + 1);
}
```

---

## 📊 COMBINED FEATURES FOR BOTH MODELS

### Trade Management
```cpp
//+------------------------------------------------------------------+
//| Execute trade with proper error handling                          |
//+------------------------------------------------------------------+
bool ExecuteTrade(int direction, double lot, double sl, double tp, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = Symbol();
   request.volume = lot;
   request.type = direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = direction > 0 ? 
                   SymbolInfoDouble(Symbol(), SYMBOL_ASK) :
                   SymbolInfoDouble(Symbol(), SYMBOL_BID);
   request.sl = sl;
   request.tp = tp;
   request.deviation = MaxSlippagePoints;
   request.magic = 90910001;  // v0091 magic number
   request.comment = comment;
   
   if(!OrderSend(request, result)) {
      Print("❌ Order failed: ", GetLastError());
      return false;
   }
   
   Print("✅ Order placed: ", comment, " | Lot: ", lot);
   return true;
}

//+------------------------------------------------------------------+
//| Handle keyboard shortcuts                                         |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if(id == CHARTEVENT_KEYDOWN) {
      // 'R' = Reset progression
      if(lparam == 'R') {
         ResetProgression("Manual reset (R key)");
         Print("🔄 Manual progression reset");
      }
      
      // 'S' = Show status
      if(lparam == 'S') {
         PrintStatus();
      }
      
      // 'G' = Show gap status (both models)
      if(lparam == 'G') {
         PrintGapStatus();
      }
   }
}

void PrintStatus() {
   Print("========================================");
   Print("🎰 STEADY CLIMB STATUS");
   Print("Position: ", g_CurrentPosition + 1, " / 8");
   Print("Units: ", PROGRESSION[g_CurrentPosition]);
   Print("Lot Size: ", GetSteadyClimbLot());
   Print("Consecutive Wins: ", g_ConsecutiveWins);
   Print("Cycle Profit: $", g_CycleProfit);
   Print("Trades Today: ", g_TradesToday);
   Print("========================================");
}
```

---

## 🎯 RECOMMENDED SETTINGS

### Conservative (Lower Risk)
```
Base Lot: 0.01
Max Lot: 0.05
Daily Trade Limit: 5
Min Confidence (NEO): 70
Min Signal Score (Tech): 4
```

### Standard
```
Base Lot: 0.01
Max Lot: 0.08
Daily Trade Limit: 10
Min Confidence (NEO): 65
Min Signal Score (Tech): 3
```

### Aggressive
```
Base Lot: 0.02
Max Lot: 0.16
Daily Trade Limit: 15
Min Confidence (NEO): 60
Min Signal Score (Tech): 3
```

---

## 📁 FILE STRUCTURE

```
MQL5/
├── Experts/
│   ├── v0091_NEO_SteadyClimb.mq5       # Model A: NEO signals
│   └── v0092_Technical_SteadyClimb.mq5 # Model B: Pure technical
├── Include/
│   ├── SteadyClimbLib.mqh              # Shared progression logic
│   ├── GapDetection.mqh                # Gap detection (from earlier)
│   └── JSONParser.mqh                  # For API responses
└── Files/
    └── steady_climb_state.json         # Persistent state
```

---

## 🚀 DEPLOYMENT CHECKLIST

1. [ ] Compile both EAs without errors
2. [ ] Test on demo account for 1 week
3. [ ] Verify API connectivity (Model A)
4. [ ] Verify indicator calculations (Model B)
5. [ ] Test progression reset on loss
6. [ ] Test daily reset functionality
7. [ ] Verify lot size calculations at each position
8. [ ] Test keyboard shortcuts
9. [ ] Monitor memory usage
10. [ ] Document actual win rates

---

## 📞 API ENDPOINTS FOR MODEL A

| Endpoint | Purpose |
|----------|---------|
| `GET /api/gaps/best` | Best gap fill trade |
| `GET /api/gaps/tradeable` | All tradeable gaps |
| `GET /api/neo/gold-forex` | Gold correlation signals |
| `GET /api/neo/brain` | NEO brain decisions |

---

## 🎰 EXPECTED PERFORMANCE

Based on simulations:

| Win Rate | Avg Session P&L | Profitable Sessions |
|----------|-----------------|---------------------|
| 50% | +6.5% | 75% |
| 55% | +10.9% | 88% |
| 74% (Gap Fill) | +30.5% | 99% |

**Key**: The strategy amplifies edge. Higher win rate = exponentially better results.

---

**Build Status: READY FOR CRELLA**

Quest: `QUEST-MT5-STEADY-CLIMB-001`
Priority: HIGH
Assigned: Crella Swarm

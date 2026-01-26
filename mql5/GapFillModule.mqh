//+------------------------------------------------------------------+
//|                                              GapFillModule.mqh   |
//|                              Gap Fill Trading Module for v0071   |
//|                              Research-backed gap fill detection  |
//+------------------------------------------------------------------+
#property copyright "NEO Trading System"
#property link      "https://crella.ai"
#property version   "1.00"

//=== GAP FILL INPUTS ===
input group "=== 📊 GAP FILL TRADING ==="
input bool   GAP_EnableGapTrading = true;        // Enable gap fill trading
input double GAP_MinGapPoints_Gold = 10.0;       // Min gap size for Gold (points)
input double GAP_MinGapPips_Forex = 15.0;        // Min gap size for Forex (pips)
input double GAP_FillTP_Percent = 80.0;          // TP at % of gap fill (80 = 80%)
input double GAP_FillSL_Percent = 50.0;          // SL at % extension beyond gap
input int    GAP_MaxAgeHours = 24;               // Only trade gaps < N hours old
input bool   GAP_RequireConfirmation = true;     // Wait for 10% fill before entry
input int    GAP_MinConfidence = 60;             // Minimum confidence to trade

//=== GAP INFO STRUCTURE ===
struct GapInfo {
   bool     hasGap;
   string   direction;      // "UP" or "DOWN"
   double   gapSize;
   double   gapPercent;
   double   gapOpen;        // Where gap started (today's open)
   double   gapTarget;      // Previous close (fill target)
   datetime gapTime;
   bool     isActive;
   bool     isFilled;
   double   fillProgress;   // 0-100%
   double   fillProbability;
   int      confidence;
   string   gapType;        // "COMMON", "STANDARD", "LARGE", "BREAKAWAY"
   string   tradeAction;    // "BUY" or "SELL"
   double   entry;
   double   stopLoss;
   double   takeProfit;
   double   riskReward;
};

//=== GLOBAL GAP STATE ===
GapInfo g_currentGap;
datetime g_lastGapCheck = 0;

//=== RESEARCH-BASED FILL RATES ===
// From 365-day analysis:
// XAUUSD: 77.9% (DOWN: 89.8%, UP: 71.4%)
// USDJPY: 80.6% (DOWN: 82.7%, UP: 78.5%)
// EURUSD: 73.1% (DOWN: 81.3%, UP: 65.3%)
// GBPUSD: 71.9% (DOWN: 78.7%, UP: 65.7%)
// AUDUSD: 78.5% (DOWN: 85.7%, UP: 72.2%)

double GetFillRate(string symbol, string direction) {
   // DOWN gaps have higher fill rates - key finding!
   
   if(StringFind(symbol, "XAU") >= 0 || StringFind(symbol, "GOLD") >= 0) {
      return (direction == "DOWN") ? 0.898 : 0.714;
   }
   if(StringFind(symbol, "JPY") >= 0) {
      return (direction == "DOWN") ? 0.827 : 0.785;
   }
   if(StringFind(symbol, "EUR") >= 0) {
      return (direction == "DOWN") ? 0.813 : 0.653;
   }
   if(StringFind(symbol, "GBP") >= 0) {
      return (direction == "DOWN") ? 0.787 : 0.657;
   }
   if(StringFind(symbol, "AUD") >= 0) {
      return (direction == "DOWN") ? 0.857 : 0.722;
   }
   
   // Default
   return (direction == "DOWN") ? 0.80 : 0.70;
}

double GetMinGap(string symbol) {
   if(StringFind(symbol, "XAU") >= 0 || StringFind(symbol, "GOLD") >= 0) {
      return GAP_MinGapPoints_Gold;
   }
   // Forex pairs - convert pips to points
   return GAP_MinGapPips_Forex * _Point * 10;
}

//+------------------------------------------------------------------+
//| Detect Gap for current symbol                                     |
//+------------------------------------------------------------------+
bool DetectGap() {
   if(!GAP_EnableGapTrading) return false;
   
   // Only check once per hour
   if(TimeCurrent() - g_lastGapCheck < 3600 && g_currentGap.hasGap) {
      return g_currentGap.hasGap && g_currentGap.isActive;
   }
   g_lastGapCheck = TimeCurrent();
   
   // Get previous day's close and today's open
   double prevClose = iClose(_Symbol, PERIOD_D1, 1);
   double todayOpen = iOpen(_Symbol, PERIOD_D1, 0);
   double todayHigh = iHigh(_Symbol, PERIOD_D1, 0);
   double todayLow = iLow(_Symbol, PERIOD_D1, 0);
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double gapSize = todayOpen - prevClose;
   double minGap = GetMinGap(_Symbol);
   
   // Check if gap exists
   if(MathAbs(gapSize) < minGap) {
      g_currentGap.hasGap = false;
      return false;
   }
   
   // Populate gap info
   g_currentGap.hasGap = true;
   g_currentGap.direction = (gapSize > 0) ? "UP" : "DOWN";
   g_currentGap.gapSize = MathAbs(gapSize);
   g_currentGap.gapPercent = MathAbs(gapSize / prevClose) * 100;
   g_currentGap.gapOpen = todayOpen;
   g_currentGap.gapTarget = prevClose;
   g_currentGap.gapTime = iTime(_Symbol, PERIOD_D1, 0);
   
   // Check if already filled
   if(g_currentGap.direction == "UP") {
      g_currentGap.isFilled = (todayLow <= prevClose);
      g_currentGap.fillProgress = MathMax(0, MathMin(100, 
         (todayOpen - currentPrice) / g_currentGap.gapSize * 100));
   } else {
      g_currentGap.isFilled = (todayHigh >= prevClose);
      g_currentGap.fillProgress = MathMax(0, MathMin(100,
         (currentPrice - todayOpen) / g_currentGap.gapSize * 100));
   }
   
   g_currentGap.isActive = !g_currentGap.isFilled;
   
   // Check gap age
   if(TimeCurrent() - g_currentGap.gapTime > GAP_MaxAgeHours * 3600) {
      g_currentGap.isActive = false;
   }
   
   // Calculate fill probability
   g_currentGap.fillProbability = GetFillRate(_Symbol, g_currentGap.direction) * 100;
   
   // Classify gap type
   if(g_currentGap.gapPercent < 0.3) {
      g_currentGap.gapType = "COMMON";
   } else if(g_currentGap.gapPercent < 1.0) {
      g_currentGap.gapType = "STANDARD";
   } else if(g_currentGap.gapPercent < 2.0) {
      g_currentGap.gapType = "LARGE";
   } else {
      g_currentGap.gapType = "BREAKAWAY";
   }
   
   // Calculate confidence
   g_currentGap.confidence = (int)g_currentGap.fillProbability;
   
   // Adjust for direction (DOWN gaps are more reliable)
   if(g_currentGap.direction == "DOWN") {
      g_currentGap.confidence += 5;
   }
   
   // Adjust for gap size (optimal 0.3%-1.0%)
   if(g_currentGap.gapPercent >= 0.3 && g_currentGap.gapPercent <= 1.0) {
      g_currentGap.confidence += 5;
   } else if(g_currentGap.gapPercent > 1.5) {
      g_currentGap.confidence -= 10;
   }
   
   // Adjust for fill progress confirmation
   if(g_currentGap.fillProgress > 10) {
      g_currentGap.confidence += 8;
   }
   
   g_currentGap.confidence = MathMin(95, MathMax(30, g_currentGap.confidence));
   
   // Set trade action (fade the gap)
   g_currentGap.tradeAction = (g_currentGap.direction == "UP") ? "SELL" : "BUY";
   
   // Calculate entry, SL, TP
   g_currentGap.entry = currentPrice;
   
   double remainingGap = MathAbs(currentPrice - g_currentGap.gapTarget);
   double tpDist = remainingGap * (GAP_FillTP_Percent / 100.0);
   double slDist = g_currentGap.gapSize * (GAP_FillSL_Percent / 100.0);
   
   if(g_currentGap.tradeAction == "BUY") {
      g_currentGap.stopLoss = currentPrice - slDist;
      g_currentGap.takeProfit = currentPrice + tpDist;
   } else {
      g_currentGap.stopLoss = currentPrice + slDist;
      g_currentGap.takeProfit = currentPrice - tpDist;
   }
   
   g_currentGap.riskReward = tpDist / slDist;
   
   // Log detection
   if(g_currentGap.isActive) {
      Print("📊 GAP DETECTED!");
      Print("   Symbol: ", _Symbol);
      Print("   Direction: ", g_currentGap.direction);
      Print("   Size: ", g_currentGap.gapPercent, "%");
      Print("   Type: ", g_currentGap.gapType);
      Print("   Fill Prob: ", g_currentGap.fillProbability, "%");
      Print("   Confidence: ", g_currentGap.confidence, "%");
      Print("   Trade: ", g_currentGap.tradeAction);
      Print("   Target: ", g_currentGap.gapTarget);
   }
   
   return g_currentGap.hasGap && g_currentGap.isActive;
}

//+------------------------------------------------------------------+
//| Check if gap fill signal is valid for entry                       |
//+------------------------------------------------------------------+
bool CheckGapFillSignal() {
   if(!g_currentGap.hasGap || !g_currentGap.isActive) return false;
   if(g_currentGap.isFilled) return false;
   
   // Check confidence
   if(g_currentGap.confidence < GAP_MinConfidence) {
      Print("📊 Gap confidence too low: ", g_currentGap.confidence, "% < ", GAP_MinConfidence, "%");
      return false;
   }
   
   // Don't fade breakaway gaps
   if(g_currentGap.gapType == "BREAKAWAY") {
      Print("📊 Breakaway gap - not fading");
      return false;
   }
   
   // Check gap age
   if(TimeCurrent() - g_currentGap.gapTime > GAP_MaxAgeHours * 3600) {
      Print("📊 Gap expired (> ", GAP_MaxAgeHours, " hours)");
      g_currentGap.isActive = false;
      return false;
   }
   
   // Confirmation check - wait for 10% fill progress
   if(GAP_RequireConfirmation && g_currentGap.fillProgress < 10) {
      Print("📊 Waiting for fill confirmation (", g_currentGap.fillProgress, "% < 10%)");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Get gap fill trade direction (-1 = SELL, 1 = BUY, 0 = none)      |
//+------------------------------------------------------------------+
int GetGapTradeDirection() {
   if(!CheckGapFillSignal()) return 0;
   
   if(g_currentGap.tradeAction == "BUY") return 1;
   if(g_currentGap.tradeAction == "SELL") return -1;
   
   return 0;
}

//+------------------------------------------------------------------+
//| Get gap fill SL                                                   |
//+------------------------------------------------------------------+
double GetGapSL() {
   return g_currentGap.stopLoss;
}

//+------------------------------------------------------------------+
//| Get gap fill TP                                                   |
//+------------------------------------------------------------------+
double GetGapTP() {
   return g_currentGap.takeProfit;
}

//+------------------------------------------------------------------+
//| Get gap confidence (0-100)                                        |
//+------------------------------------------------------------------+
int GetGapConfidence() {
   return g_currentGap.confidence;
}

//+------------------------------------------------------------------+
//| Display gap status on chart                                       |
//+------------------------------------------------------------------+
void DisplayGapStatus(int x = 10, int y = 100) {
   string prefix = "GAP_";
   
   // Delete old objects
   ObjectDelete(0, prefix + "TITLE");
   ObjectDelete(0, prefix + "STATUS");
   ObjectDelete(0, prefix + "DETAILS");
   
   if(!g_currentGap.hasGap) {
      ObjectCreate(0, prefix + "STATUS", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "STATUS", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "STATUS", OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, prefix + "STATUS", OBJPROP_YDISTANCE, y);
      ObjectSetString(0, prefix + "STATUS", OBJPROP_TEXT, "📊 No Gap");
      ObjectSetInteger(0, prefix + "STATUS", OBJPROP_COLOR, clrGray);
      ObjectSetInteger(0, prefix + "STATUS", OBJPROP_FONTSIZE, 10);
      return;
   }
   
   // Title
   ObjectCreate(0, prefix + "TITLE", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix + "TITLE", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, prefix + "TITLE", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix + "TITLE", OBJPROP_YDISTANCE, y);
   ObjectSetString(0, prefix + "TITLE", OBJPROP_TEXT, 
      "📊 GAP: " + g_currentGap.direction + " " + 
      DoubleToString(g_currentGap.gapPercent, 2) + "%");
   ObjectSetInteger(0, prefix + "TITLE", OBJPROP_COLOR, 
      g_currentGap.direction == "UP" ? clrLimeGreen : clrOrangeRed);
   ObjectSetInteger(0, prefix + "TITLE", OBJPROP_FONTSIZE, 11);
   
   // Status
   string statusText = "";
   color statusColor = clrWhite;
   
   if(g_currentGap.isFilled) {
      statusText = "✅ FILLED";
      statusColor = clrLimeGreen;
   } else if(g_currentGap.isActive) {
      statusText = "🔄 ACTIVE | " + g_currentGap.tradeAction + " | " + 
                   IntegerToString(g_currentGap.confidence) + "% conf";
      statusColor = clrCyan;
   } else {
      statusText = "⏸ INACTIVE";
      statusColor = clrGray;
   }
   
   ObjectCreate(0, prefix + "STATUS", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix + "STATUS", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, prefix + "STATUS", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix + "STATUS", OBJPROP_YDISTANCE, y + 18);
   ObjectSetString(0, prefix + "STATUS", OBJPROP_TEXT, statusText);
   ObjectSetInteger(0, prefix + "STATUS", OBJPROP_COLOR, statusColor);
   ObjectSetInteger(0, prefix + "STATUS", OBJPROP_FONTSIZE, 10);
   
   // Details
   ObjectCreate(0, prefix + "DETAILS", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix + "DETAILS", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, prefix + "DETAILS", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, prefix + "DETAILS", OBJPROP_YDISTANCE, y + 36);
   ObjectSetString(0, prefix + "DETAILS", OBJPROP_TEXT, 
      "Fill: " + DoubleToString(g_currentGap.fillProgress, 0) + "% | " +
      "Prob: " + DoubleToString(g_currentGap.fillProbability, 0) + "% | " +
      "Type: " + g_currentGap.gapType);
   ObjectSetInteger(0, prefix + "DETAILS", OBJPROP_COLOR, clrSilver);
   ObjectSetInteger(0, prefix + "DETAILS", OBJPROP_FONTSIZE, 9);
}

//+------------------------------------------------------------------+
//| Draw gap zone on chart                                            |
//+------------------------------------------------------------------+
void DrawGapZone() {
   if(!g_currentGap.hasGap) return;
   
   string prefix = "GAP_ZONE_";
   
   // Delete old objects
   ObjectDelete(0, prefix + "RECT");
   ObjectDelete(0, prefix + "TARGET");
   
   datetime timeStart = g_currentGap.gapTime;
   datetime timeEnd = timeStart + 24 * 3600;
   
   // Draw gap rectangle
   ObjectCreate(0, prefix + "RECT", OBJ_RECTANGLE, 0, 
      timeStart, g_currentGap.gapOpen,
      timeEnd, g_currentGap.gapTarget);
   ObjectSetInteger(0, prefix + "RECT", OBJPROP_COLOR, 
      g_currentGap.direction == "UP" ? clrDarkGreen : clrDarkRed);
   ObjectSetInteger(0, prefix + "RECT", OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, prefix + "RECT", OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, prefix + "RECT", OBJPROP_BACK, true);
   ObjectSetInteger(0, prefix + "RECT", OBJPROP_FILL, true);
   
   // Draw target line
   ObjectCreate(0, prefix + "TARGET", OBJ_HLINE, 0, 0, g_currentGap.gapTarget);
   ObjectSetInteger(0, prefix + "TARGET", OBJPROP_COLOR, clrYellow);
   ObjectSetInteger(0, prefix + "TARGET", OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, prefix + "TARGET", OBJPROP_WIDTH, 1);
}

//+------------------------------------------------------------------+
//| Integration example for v0071/v0091                               |
//+------------------------------------------------------------------+
/*
// Add to OnTick():

void OnTick() {
   // ... existing code ...
   
   // Check for gap fill opportunity
   if(DetectGap()) {
      int gapDirection = GetGapTradeDirection();
      
      if(gapDirection != 0) {
         // Check if it aligns with Supertrend
         int supertrendDirection = GetSupertrendDirection(); // Your existing function
         
         // STRATEGY 1: Gap alone
         if(gapDirection != 0 && GetGapConfidence() >= 70) {
            Print("🎯 High-confidence gap fill signal!");
            // Place trade with GetGapSL() and GetGapTP()
         }
         
         // STRATEGY 2: Gap + Supertrend alignment
         if(gapDirection == supertrendDirection) {
            Print("🎯 Gap + Supertrend aligned!");
            // Higher confidence trade
         }
         
         // STRATEGY 3: Gap conflicts with Supertrend - WAIT
         if(gapDirection != supertrendDirection && gapDirection != 0) {
            Print("⚠️ Gap conflicts with Supertrend - waiting for gap fill");
            // Wait until gap fills before trading Supertrend direction
         }
      }
   }
   
   // Display status
   DisplayGapStatus();
   DrawGapZone();
}

// Keyboard handler for 'G' key:
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
   if(id == CHARTEVENT_KEYDOWN) {
      if(lparam == 'G') {  // 'G' key pressed
         DetectGap();
         PrintGapStatus();
      }
   }
}

void PrintGapStatus() {
   Print("========== GAP STATUS ==========");
   Print("Has Gap: ", g_currentGap.hasGap);
   if(g_currentGap.hasGap) {
      Print("Direction: ", g_currentGap.direction);
      Print("Size: ", g_currentGap.gapPercent, "%");
      Print("Type: ", g_currentGap.gapType);
      Print("Active: ", g_currentGap.isActive);
      Print("Filled: ", g_currentGap.isFilled);
      Print("Fill Progress: ", g_currentGap.fillProgress, "%");
      Print("Fill Probability: ", g_currentGap.fillProbability, "%");
      Print("Confidence: ", g_currentGap.confidence, "%");
      Print("Trade: ", g_currentGap.tradeAction);
      Print("Entry: ", g_currentGap.entry);
      Print("SL: ", g_currentGap.stopLoss);
      Print("TP: ", g_currentGap.takeProfit);
      Print("R:R: ", g_currentGap.riskReward);
   }
   Print("================================");
}
*/
//+------------------------------------------------------------------+

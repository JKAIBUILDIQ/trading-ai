//+------------------------------------------------------------------+
//|                                           GhostTimedTrader.mqh   |
//|                        Ghost Commander - Time-Based Stealth Mode |
//|                                 NEO Integration for 4H Predictions |
//+------------------------------------------------------------------+
#property copyright "Crella Trading AI"
#property link      "https://crella.ai"
#property version   "2.01"

//+------------------------------------------------------------------+
//| 🛡️ WHY TIME-BASED? (ANTI-MM STEALTH MODE!)                        |
//+------------------------------------------------------------------+
//| 
//| TRADITIONAL TRADING (VULNERABLE):
//| ├── BUY @ $5,090
//| ├── SL @ $5,070 ← MMs SEE THIS ON ORDER BOOK!
//| ├── TP @ $5,120 ← MMs SEE THIS TOO!
//| └── MMs hunt your stop, take your money
//|
//| TIME-BASED TRADING (INVISIBLE):
//| ├── BUY @ $5,090  
//| ├── NO SL visible ← MMs see NOTHING (emergency SL only)
//| ├── NO TP visible ← MMs see NOTHING
//| ├── EXIT after 4 hours at market price
//| └── MMs can't target what they can't see!
//|
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>

//--- Inputs
input group "=== TIME-BASED STEALTH MODE ==="
input bool   UseTimeBased = true;              // Enable time-based exits (stealth mode)
input int    HoldPeriodMinutes = 240;          // Hold period in minutes (240 = 4 hours)
input double EmergencySL_Pips = 200;           // Emergency SL (flash crash protection only)
input bool   HideTPFromBroker = true;          // No TP on broker side (stealth)

input group "=== NEO PREDICTION INTEGRATION ==="
input string NEO_API_URL = "http://localhost:8020"; // NEO Prediction API URL
input int    PollIntervalSeconds = 60;         // How often to check for new predictions
input double MinConfidence = 60.0;             // Minimum confidence to trade (%)
input bool   AutoTrade = false;                // Auto-execute NEO predictions (OFF by default!)

input group "=== POSITION SIZING ==="
input double RiskPerTrade = 1.0;               // Risk per trade (% of balance)
input double MaxLots = 5.0;                    // Maximum lots per trade
input double MinLots = 0.01;                   // Minimum lots

//--- Global variables
CTrade trade;
datetime g_entryTime = 0;                      // When position was opened
datetime g_exitTime = 0;                       // When to exit (entry + hold period)
string   g_currentPredictionId = "";           // Current NEO prediction ID
bool     g_hasOpenPosition = false;
int      g_timedPositionTicket = 0;

//+------------------------------------------------------------------+
//| STRUCTURE: NEO Prediction                                         |
//+------------------------------------------------------------------+
struct NEOPrediction
{
   string prediction_id;
   string predicted_direction;        // "UP", "DOWN", "FLAT"
   double predicted_change_pips;
   double predicted_price;
   double current_price;
   double confidence;
   string reasoning;
   datetime target_time;
   bool valid;
};

//+------------------------------------------------------------------+
//| Initialize timed trader                                           |
//+------------------------------------------------------------------+
void InitTimedTrader()
{
   Print("═══════════════════════════════════════════════════════════");
   Print("🛡️ GHOST TIMED TRADER INITIALIZED (STEALTH MODE)");
   Print("═══════════════════════════════════════════════════════════");
   Print("   Hold Period: ", HoldPeriodMinutes, " minutes (", HoldPeriodMinutes/60.0, " hours)");
   Print("   Emergency SL: ", EmergencySL_Pips, " pips (flash crash protection only)");
   Print("   Hide TP: ", HideTPFromBroker ? "YES (stealth)" : "NO");
   Print("   NEO API: ", NEO_API_URL);
   Print("   Auto Trade: ", AutoTrade ? "ENABLED ⚠️" : "DISABLED (manual confirmation)");
   Print("═══════════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| Execute a time-based trade (STEALTH MODE)                         |
//+------------------------------------------------------------------+
bool ExecuteTimedTrade(string direction, double lots, string comment = "")
{
   if(!UseTimeBased)
   {
      Print("❌ Time-based trading disabled");
      return false;
   }
   
   if(g_hasOpenPosition)
   {
      Print("⚠️ Already have a timed position open");
      return false;
   }
   
   // Normalize lots
   lots = MathMax(MinLots, MathMin(MaxLots, lots));
   
   double entry = 0;
   double sl = 0;
   double tp = 0;  // NO TP - stealth mode!
   
   // Calculate entry and emergency SL
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   // Emergency SL in points (very far away)
   double sl_points = EmergencySL_Pips * 10;  // Convert pips to points for Gold
   if(StringFind(_Symbol, "JPY") >= 0) sl_points = EmergencySL_Pips * 100;
   
   if(direction == "BUY")
   {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = NormalizeDouble(entry - sl_points * point, digits);
      tp = HideTPFromBroker ? 0 : 0;  // NO TP visible to broker!
   }
   else if(direction == "SELL")
   {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = NormalizeDouble(entry + sl_points * point, digits);
      tp = HideTPFromBroker ? 0 : 0;  // NO TP visible to broker!
   }
   else
   {
      Print("❌ Invalid direction: ", direction);
      return false;
   }
   
   // Build comment
   string tradeComment = "NEO|TIMED|" + IntegerToString(HoldPeriodMinutes) + "M";
   if(comment != "") tradeComment += "|" + comment;
   
   // Execute trade
   bool result = false;
   if(direction == "BUY")
      result = trade.Buy(lots, _Symbol, entry, sl, tp, tradeComment);
   else
      result = trade.Sell(lots, _Symbol, entry, sl, tp, tradeComment);
   
   if(result)
   {
      g_entryTime = TimeCurrent();
      g_exitTime = g_entryTime + HoldPeriodMinutes * 60;
      g_hasOpenPosition = true;
      g_timedPositionTicket = (int)trade.ResultOrder();
      
      Print("═══════════════════════════════════════════════════════════");
      Print("🛡️ TIMED TRADE OPENED (STEALTH MODE)");
      Print("═══════════════════════════════════════════════════════════");
      Print("   Direction: ", direction);
      Print("   Entry: ", entry);
      Print("   Lots: ", lots);
      Print("   Emergency SL: ", sl, " (", EmergencySL_Pips, " pips away - flash crash only)");
      Print("   TP: NONE (stealth - exit by time!)");
      Print("   Entry Time: ", TimeToString(g_entryTime, TIME_DATE|TIME_MINUTES));
      Print("   Exit Time: ", TimeToString(g_exitTime, TIME_DATE|TIME_MINUTES));
      Print("   Hold Period: ", HoldPeriodMinutes, " minutes");
      Print("═══════════════════════════════════════════════════════════");
      
      return true;
   }
   else
   {
      Print("❌ Trade failed: ", trade.ResultRetcodeDescription());
      return false;
   }
}

//+------------------------------------------------------------------+
//| Check if time to exit (called on every tick)                      |
//+------------------------------------------------------------------+
void CheckTimedExit()
{
   if(!UseTimeBased || !g_hasOpenPosition) return;
   
   datetime now = TimeCurrent();
   
   // Time remaining
   int remainingSeconds = (int)(g_exitTime - now);
   
   // Log progress every 15 minutes
   static datetime lastLog = 0;
   if(now - lastLog >= 900)  // 15 minutes
   {
      Print("⏱️ Timed position: ", remainingSeconds / 60, " minutes remaining until exit");
      lastLog = now;
   }
   
   // Check if time to exit
   if(now >= g_exitTime)
   {
      Print("═══════════════════════════════════════════════════════════");
      Print("⏰ TIME TO EXIT - Closing timed position");
      Print("═══════════════════════════════════════════════════════════");
      
      CloseTimedPosition();
   }
}

//+------------------------------------------------------------------+
//| Close the timed position and log results                          |
//+------------------------------------------------------------------+
void CloseTimedPosition()
{
   if(!g_hasOpenPosition) return;
   
   // Find and close the position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket))
      {
         string comment = PositionGetString(POSITION_COMMENT);
         if(StringFind(comment, "NEO|TIMED") >= 0)
         {
            double entry = PositionGetDouble(POSITION_PRICE_OPEN);
            double current = PositionGetDouble(POSITION_PRICE_CURRENT);
            double profit = PositionGetDouble(POSITION_PROFIT);
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            // Calculate actual change
            double change_pips = (type == POSITION_TYPE_BUY) ? 
                                 (current - entry) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) / 10 :
                                 (entry - current) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) / 10;
            
            // Close position
            if(trade.PositionClose(ticket))
            {
               Print("═══════════════════════════════════════════════════════════");
               Print("✅ TIMED POSITION CLOSED");
               Print("═══════════════════════════════════════════════════════════");
               Print("   Direction: ", type == POSITION_TYPE_BUY ? "BUY" : "SELL");
               Print("   Entry: ", entry);
               Print("   Exit: ", current);
               Print("   Change: ", change_pips, " pips");
               Print("   Profit: $", DoubleToString(profit, 2));
               Print("   Result: ", profit >= 0 ? "✅ WIN" : "❌ LOSS");
               Print("   Prediction ID: ", g_currentPredictionId);
               Print("═══════════════════════════════════════════════════════════");
               
               // Log result for NEO learning
               LogResultForLearning(g_currentPredictionId, profit >= 0, change_pips, profit);
            }
            
            break;
         }
      }
   }
   
   // Reset state
   g_hasOpenPosition = false;
   g_entryTime = 0;
   g_exitTime = 0;
   g_timedPositionTicket = 0;
   g_currentPredictionId = "";
}

//+------------------------------------------------------------------+
//| Log result to NEO for learning                                    |
//+------------------------------------------------------------------+
void LogResultForLearning(string predictionId, bool won, double change_pips, double profit)
{
   if(predictionId == "") return;
   
   // Create result JSON
   string json = "{";
   json += "\"prediction_id\":\"" + predictionId + "\",";
   json += "\"won\":" + (won ? "true" : "false") + ",";
   json += "\"change_pips\":" + DoubleToString(change_pips, 2) + ",";
   json += "\"profit\":" + DoubleToString(profit, 2) + ",";
   json += "\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES) + "\"";
   json += "}";
   
   // Would send to NEO API here
   // For now, save to file for manual processing
   string filename = "NEO_Results_" + TimeToString(TimeCurrent(), TIME_DATE) + ".json";
   int handle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_READ|FILE_ANSI);
   if(handle != INVALID_HANDLE)
   {
      FileSeek(handle, 0, SEEK_END);
      FileWriteString(handle, json + "\n");
      FileClose(handle);
      Print("📝 Result logged to: ", filename);
   }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                             |
//+------------------------------------------------------------------+
double CalculateLots(double riskPercent)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * riskPercent / 100.0;
   
   // For timed trades, base risk on expected move (not SL)
   // Assume worst case is 2x expected move
   double atr = iATR(_Symbol, PERIOD_H4, 14);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(tickValue == 0 || atr == 0) return MinLots;
   
   // Risk = 2x ATR (worst case scenario)
   double riskPips = atr * 2 / tickSize;
   double valuePerPip = tickValue / tickSize;
   
   double lots = riskAmount / (riskPips * valuePerPip);
   lots = NormalizeDouble(lots, 2);
   
   return MathMax(MinLots, MathMin(MaxLots, lots));
}

//+------------------------------------------------------------------+
//| Parse NEO prediction from JSON                                    |
//+------------------------------------------------------------------+
NEOPrediction ParseNEOPrediction(string json)
{
   NEOPrediction pred;
   pred.valid = false;
   
   // Simple JSON parsing (production should use proper JSON library)
   if(StringFind(json, "prediction_id") < 0) return pred;
   
   // Extract prediction_id
   int start = StringFind(json, "\"prediction_id\":");
   if(start >= 0)
   {
      start = StringFind(json, "\"", start + 16) + 1;
      int end = StringFind(json, "\"", start);
      pred.prediction_id = StringSubstr(json, start, end - start);
   }
   
   // Extract predicted_direction
   start = StringFind(json, "\"predicted_direction\":");
   if(start >= 0)
   {
      start = StringFind(json, "\"", start + 22) + 1;
      int end = StringFind(json, "\"", start);
      pred.predicted_direction = StringSubstr(json, start, end - start);
   }
   
   // Extract confidence
   start = StringFind(json, "\"confidence\":");
   if(start >= 0)
   {
      start += 13;
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      pred.confidence = StringToDouble(StringSubstr(json, start, end - start));
   }
   
   // Extract predicted_change_pips
   start = StringFind(json, "\"predicted_change_pips\":");
   if(start >= 0)
   {
      start += 24;
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      pred.predicted_change_pips = StringToDouble(StringSubstr(json, start, end - start));
   }
   
   // Extract current_price
   start = StringFind(json, "\"current_price\":");
   if(start >= 0)
   {
      start += 16;
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      pred.current_price = StringToDouble(StringSubstr(json, start, end - start));
   }
   
   pred.valid = (pred.prediction_id != "" && pred.predicted_direction != "");
   
   return pred;
}

//+------------------------------------------------------------------+
//| Get status string for display                                     |
//+------------------------------------------------------------------+
string GetTimedTraderStatus()
{
   string status = "";
   
   if(!UseTimeBased)
   {
      status = "⏹️ Time-based trading DISABLED";
   }
   else if(g_hasOpenPosition)
   {
      int remaining = (int)(g_exitTime - TimeCurrent()) / 60;
      status = StringFormat("🛡️ STEALTH MODE | Position open | Exit in %d min", remaining);
   }
   else
   {
      status = "🛡️ STEALTH MODE | Waiting for NEO prediction";
   }
   
   return status;
}

//+------------------------------------------------------------------+
//| Display status on chart                                           |
//+------------------------------------------------------------------+
void DisplayTimedStatus()
{
   string status = GetTimedTraderStatus();
   
   // Would display on chart using ObjectCreate, etc.
   Comment(
      "═══════════════════════════════════════════════════\n",
      "🛡️ GHOST TIMED TRADER (STEALTH MODE)\n",
      "═══════════════════════════════════════════════════\n",
      "Status: ", status, "\n",
      "Hold Period: ", HoldPeriodMinutes, " minutes\n",
      "Emergency SL: ", EmergencySL_Pips, " pips\n",
      "Auto Trade: ", AutoTrade ? "ON ⚠️" : "OFF\n",
      "═══════════════════════════════════════════════════"
   );
}
//+------------------------------------------------------------------+

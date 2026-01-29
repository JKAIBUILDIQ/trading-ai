//+------------------------------------------------------------------+
//|                                              NeoSignalReader.mqh |
//|                         NEO Signal Integration for Crellastein EA |
//|                                    Read conviction-scored signals |
//+------------------------------------------------------------------+
#property copyright "NEO Trading Intelligence"
#property version   "1.00"

#include <JAson.mqh>  // JSON parsing library

//--- Signal file path (network share from H100)
input string NEO_SignalPath = "\\\\100.119.161.65\\trading_ai\\neo\\signals\\ea_signal.json";
input int    NEO_CheckIntervalSeconds = 60;  // How often to check for new signals
input bool   NEO_AutoDefensive = true;       // Automatically enter defensive mode on HIGH/EXTREME

//--- NEO Signal Structure with DEFCON
struct NEOSignal {
    string   timestamp;
    string   symbol;
    string   direction;        // BULLISH, BEARISH, NEUTRAL
    int      conviction;       // 50-100
    int      defcon;           // 1-5 (1=max threat, 5=normal)
    string   defcon_color;     // RED, ORANGE, YELLOW, BLUE, GREEN
    string   action;           // PROCEED, MONITOR, REDUCE_SIZE, PAUSE_LONGS, PAUSE_SHORTS, DEFENSIVE_MODE
    double   tp;
    double   sl;
    double   hunt_zone;
    bool     pause_longs;
    bool     pause_shorts;
    double   lot_multiplier;   // 0.0, 0.5, 0.75, 1.0
    int      tighten_sl_pips;
    double   max_dd_override;
    int      close_partial;    // Percentage to close (0-100)
    bool     set_breakeven;
    bool     consider_hedge;
    string   valid_until;
};

//--- Global signal storage
NEOSignal g_CurrentNEOSignal;
datetime  g_LastSignalCheck = 0;
bool      g_SignalLoaded = false;

//+------------------------------------------------------------------+
//| Read NEO Signal from JSON file                                     |
//+------------------------------------------------------------------+
bool ReadNEOSignal(string filepath, NEOSignal &signal) {
    //--- Check if file exists
    if(!FileIsExist(filepath, FILE_COMMON)) {
        Print("NEO: Signal file not found: ", filepath);
        return false;
    }
    
    //--- Read file contents
    int handle = FileOpen(filepath, FILE_READ|FILE_TXT|FILE_COMMON);
    if(handle == INVALID_HANDLE) {
        Print("NEO: Cannot open signal file");
        return false;
    }
    
    string json_content = "";
    while(!FileIsEnding(handle)) {
        json_content += FileReadString(handle);
    }
    FileClose(handle);
    
    //--- Parse JSON
    CJAVal json;
    if(!json.Deserialize(json_content)) {
        Print("NEO: Failed to parse JSON");
        return false;
    }
    
    //--- Extract signal data
    signal.timestamp = json["timestamp"].ToStr();
    signal.symbol = json["symbol"].ToStr();
    signal.direction = json["direction"].ToStr();
    signal.conviction = (int)json["conviction"].ToInt();
    signal.defcon = (int)json["defcon"].ToInt();
    signal.defcon_color = json["defcon_color"].ToStr();
    signal.action = json["action"].ToStr();
    
    //--- Targets
    signal.tp = json["targets"]["tp"].ToDbl();
    signal.sl = json["targets"]["sl"].ToDbl();
    signal.hunt_zone = json["targets"]["hunt_zone"].ToDbl();
    
    //--- EA Instructions
    signal.pause_longs = json["ea_instructions"]["pause_longs"].ToBool();
    signal.pause_shorts = json["ea_instructions"]["pause_shorts"].ToBool();
    signal.lot_multiplier = json["ea_instructions"]["reduce_lot_multiplier"].ToDbl();
    signal.tighten_sl_pips = (int)json["ea_instructions"]["tighten_sl_pips"].ToInt();
    signal.max_dd_override = json["ea_instructions"]["max_drawdown_override"].ToDbl();
    signal.close_partial = (int)json["ea_instructions"]["close_partial"].ToInt();
    signal.set_breakeven = json["ea_instructions"]["set_breakeven"].ToBool();
    signal.consider_hedge = json["ea_instructions"]["consider_hedge"].ToBool();
    
    signal.valid_until = json["valid_until"].ToStr();
    
    Print("NEO: Signal loaded - DEFCON ", signal.defcon, " (", signal.defcon_color, ") - ", 
          signal.direction, " ", signal.conviction, "%");
    
    return true;
}

//+------------------------------------------------------------------+
//| Check if signal is still valid                                     |
//+------------------------------------------------------------------+
bool IsSignalValid(NEOSignal &signal) {
    if(signal.valid_until == "") return false;
    
    // Parse valid_until timestamp
    // Format: "2026-01-30T13:23:45.123456"
    datetime valid_dt = StringToTime(StringSubstr(signal.valid_until, 0, 10) + " " + 
                                      StringSubstr(signal.valid_until, 11, 8));
    
    return (TimeCurrent() < valid_dt);
}

//+------------------------------------------------------------------+
//| Check if symbol matches current chart                              |
//+------------------------------------------------------------------+
bool IsSignalForSymbol(NEOSignal &signal, string chart_symbol) {
    string neo_sym = signal.symbol;
    
    // Handle XAUUSD variants
    if(StringFind(chart_symbol, "XAUUSD") >= 0 || StringFind(chart_symbol, "GOLD") >= 0) {
        if(neo_sym == "XAUUSD" || neo_sym == "MGC" || neo_sym == "GOLD") {
            return true;
        }
    }
    
    return (neo_sym == chart_symbol);
}

//+------------------------------------------------------------------+
//| Main update function - call from EA OnTick or OnTimer              |
//+------------------------------------------------------------------+
void UpdateNEOSignal() {
    //--- Rate limit checks
    if(TimeCurrent() - g_LastSignalCheck < NEO_CheckIntervalSeconds) {
        return;
    }
    g_LastSignalCheck = TimeCurrent();
    
    //--- Read signal
    NEOSignal new_signal;
    if(ReadNEOSignal(NEO_SignalPath, new_signal)) {
        g_CurrentNEOSignal = new_signal;
        g_SignalLoaded = true;
        
        //--- Log if conviction changed significantly
        if(new_signal.conviction_level == "HIGH" || new_signal.conviction_level == "EXTREME") {
            Alert("NEO HIGH CONVICTION SIGNAL: ", new_signal.direction, 
                  " ", new_signal.conviction, "% - ", new_signal.action);
        }
    }
}

//+------------------------------------------------------------------+
//| Check if should pause long entries                                 |
//+------------------------------------------------------------------+
bool NEO_ShouldPauseLongs() {
    if(!g_SignalLoaded) return false;
    if(!IsSignalValid(g_CurrentNEOSignal)) return false;
    if(!NEO_AutoDefensive) return false;
    
    return g_CurrentNEOSignal.pause_longs;
}

//+------------------------------------------------------------------+
//| Check if should pause short entries                                |
//+------------------------------------------------------------------+
bool NEO_ShouldPauseShorts() {
    if(!g_SignalLoaded) return false;
    if(!IsSignalValid(g_CurrentNEOSignal)) return false;
    if(!NEO_AutoDefensive) return false;
    
    return g_CurrentNEOSignal.pause_shorts;
}

//+------------------------------------------------------------------+
//| Get lot size multiplier (1.0 = normal, 0.5 = half, etc)            |
//+------------------------------------------------------------------+
double NEO_GetLotMultiplier() {
    if(!g_SignalLoaded) return 1.0;
    if(!IsSignalValid(g_CurrentNEOSignal)) return 1.0;
    if(!NEO_AutoDefensive) return 1.0;
    
    double mult = g_CurrentNEOSignal.lot_multiplier;
    return (mult > 0 && mult <= 1.0) ? mult : 1.0;
}

//+------------------------------------------------------------------+
//| Get SL tightening in pips                                          |
//+------------------------------------------------------------------+
int NEO_GetSLTightenPips() {
    if(!g_SignalLoaded) return 0;
    if(!IsSignalValid(g_CurrentNEOSignal)) return 0;
    if(!NEO_AutoDefensive) return 0;
    
    return g_CurrentNEOSignal.tighten_sl_pips;
}

//+------------------------------------------------------------------+
//| Get max drawdown override (0 = use default)                        |
//+------------------------------------------------------------------+
double NEO_GetMaxDDOverride() {
    if(!g_SignalLoaded) return 0;
    if(!IsSignalValid(g_CurrentNEOSignal)) return 0;
    if(!NEO_AutoDefensive) return 0;
    
    return g_CurrentNEOSignal.max_dd_override;
}

//+------------------------------------------------------------------+
//| Get conviction level string                                        |
//+------------------------------------------------------------------+
string NEO_GetConvictionLevel() {
    if(!g_SignalLoaded) return "NONE";
    if(!IsSignalValid(g_CurrentNEOSignal)) return "EXPIRED";
    
    return g_CurrentNEOSignal.conviction_level;
}

//+------------------------------------------------------------------+
//| Get conviction percentage                                          |
//+------------------------------------------------------------------+
int NEO_GetConviction() {
    if(!g_SignalLoaded) return 50;
    if(!IsSignalValid(g_CurrentNEOSignal)) return 50;
    
    return g_CurrentNEOSignal.conviction;
}

//+------------------------------------------------------------------+
//| Get direction                                                       |
//+------------------------------------------------------------------+
string NEO_GetDirection() {
    if(!g_SignalLoaded) return "NEUTRAL";
    if(!IsSignalValid(g_CurrentNEOSignal)) return "NEUTRAL";
    
    return g_CurrentNEOSignal.direction;
}

//+------------------------------------------------------------------+
//| Get DEFCON level (1=max threat, 5=normal)                          |
//+------------------------------------------------------------------+
int NEO_GetDefcon() {
    if(!g_SignalLoaded) return 5;  // Default to normal
    if(!IsSignalValid(g_CurrentNEOSignal)) return 5;
    
    return g_CurrentNEOSignal.defcon;
}

//+------------------------------------------------------------------+
//| Check if should close partial positions                            |
//+------------------------------------------------------------------+
int NEO_GetClosePartialPercent() {
    if(!g_SignalLoaded) return 0;
    if(!IsSignalValid(g_CurrentNEOSignal)) return 0;
    if(!NEO_AutoDefensive) return 0;
    
    return g_CurrentNEOSignal.close_partial;
}

//+------------------------------------------------------------------+
//| Check if should set breakeven stops                                |
//+------------------------------------------------------------------+
bool NEO_ShouldSetBreakeven() {
    if(!g_SignalLoaded) return false;
    if(!IsSignalValid(g_CurrentNEOSignal)) return false;
    if(!NEO_AutoDefensive) return false;
    
    return g_CurrentNEOSignal.set_breakeven;
}

//+------------------------------------------------------------------+
//| Check if should consider hedge                                     |
//+------------------------------------------------------------------+
bool NEO_ShouldConsiderHedge() {
    if(!g_SignalLoaded) return false;
    if(!IsSignalValid(g_CurrentNEOSignal)) return false;
    if(!NEO_AutoDefensive) return false;
    
    return g_CurrentNEOSignal.consider_hedge;
}

//+------------------------------------------------------------------+
//| Print signal summary to log with DEFCON                            |
//+------------------------------------------------------------------+
void NEO_PrintSignalSummary() {
    if(!g_SignalLoaded) {
        Print("NEO: No signal loaded");
        return;
    }
    
    string defcon_emoji = "";
    switch(g_CurrentNEOSignal.defcon) {
        case 1: defcon_emoji = "🔴 DEFCON 1 - MAXIMUM THREAT"; break;
        case 2: defcon_emoji = "🟠 DEFCON 2 - SEVERE"; break;
        case 3: defcon_emoji = "🟡 DEFCON 3 - HIGH ALERT"; break;
        case 4: defcon_emoji = "🔵 DEFCON 4 - ELEVATED"; break;
        case 5: defcon_emoji = "🟢 DEFCON 5 - NORMAL"; break;
    }
    
    Print("╔══════════════════════════════════════════════════════════╗");
    Print("║             NEO SIGNAL SUMMARY                           ║");
    Print("╠══════════════════════════════════════════════════════════╣");
    Print("║ ", defcon_emoji);
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Symbol:     ", g_CurrentNEOSignal.symbol);
    Print("║ Direction:  ", g_CurrentNEOSignal.direction);
    Print("║ Conviction: ", g_CurrentNEOSignal.conviction, "%");
    Print("║ Action:     ", g_CurrentNEOSignal.action);
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Pause Longs:    ", g_CurrentNEOSignal.pause_longs ? "YES 🛑" : "NO");
    Print("║ Pause Shorts:   ", g_CurrentNEOSignal.pause_shorts ? "YES 🛑" : "NO");
    Print("║ Lot Multiplier: ", g_CurrentNEOSignal.lot_multiplier);
    Print("║ Tighten SL:     ", g_CurrentNEOSignal.tighten_sl_pips, " pips");
    Print("║ Close Partial:  ", g_CurrentNEOSignal.close_partial, "%");
    Print("║ Set Breakeven:  ", g_CurrentNEOSignal.set_breakeven ? "YES" : "NO");
    Print("║ Consider Hedge: ", g_CurrentNEOSignal.consider_hedge ? "YES ⚠️" : "NO");
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Valid Until: ", g_CurrentNEOSignal.valid_until);
    Print("╚══════════════════════════════════════════════════════════╝");
}

//+------------------------------------------------------------------+
//| Example usage in EA OnTick                                         |
//+------------------------------------------------------------------+
/*
#include "NeoSignalReader.mqh"

void OnTick() {
    // Update NEO signal (rate limited internally)
    UpdateNEOSignal();
    
    // Check if we should open long
    if(BuySignalDetected()) {
        // NEO says pause longs?
        if(NEO_ShouldPauseLongs()) {
            Print("NEO: Long entry blocked - HIGH conviction BEARISH signal");
            return;
        }
        
        // Get lot multiplier
        double lot = BaseLotSize * NEO_GetLotMultiplier();
        
        // Get tightened SL
        double sl = DefaultSL - NEO_GetSLTightenPips() * Point;
        
        // Execute with NEO adjustments
        OpenBuy(lot, sl, tp);
    }
}
*/
//+------------------------------------------------------------------+

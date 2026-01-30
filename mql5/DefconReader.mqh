//+------------------------------------------------------------------+
//|                                                 DefconReader.mqh |
//|                         NEO DEFCON System for Ghost/Casper EAs   |
//|                        Read DEFCON level and adjust bot behavior |
//+------------------------------------------------------------------+
#property copyright "NEO Trading Intelligence"
#property version   "1.00"

//--- DEFCON file paths
input string DEFCON_FilePath = "C:\\MT5_Share\\MT5_DEFCON.txt";
input string DEFCON_StatePath = "C:\\MT5_Share\\MT5_DEFCON_STATE.json";
input string GHOST_ExposurePath = "C:\\MT5_Share\\GHOST_EXPOSURE.txt";
input string CASPER_HedgePath = "C:\\MT5_Share\\CASPER_HEDGE.txt";
input int    DEFCON_CheckInterval = 30;  // Seconds between checks

//--- DEFCON Levels
#define DEFCON_NORMAL       5
#define DEFCON_ELEVATED     4
#define DEFCON_HIGH_ALERT   3
#define DEFCON_SEVERE       2
#define DEFCON_MAXIMUM      1

//--- DEFCON State Structure
struct DefconState {
    int      level;           // 1-5
    string   color;           // RED, ORANGE, YELLOW, BLUE, GREEN
    string   name;            // MAXIMUM THREAT, SEVERE, etc.
    double   lotMultiplier;   // 0.0 - 1.0
    int      maxPositions;    // Max allowed positions
    int      minConfidence;   // Minimum confidence to trade
    double   entryGapMult;    // Entry gap multiplier
    bool     pauseEntries;    // Stop all new entries
    bool     survivalMode;    // DEFCON 1 survival mode
    int      closePercent;    // Percent to close at market
    bool     setBreakeven;    // Set stops to breakeven
    bool     hedgeMode;       // Casper hedge active
    int      hedgePercent;    // Hedge coverage percent
    bool     deltaNeutral;    // Full delta neutral mode
    datetime lastUpdate;      // When file was last read
};

//--- Global state
DefconState g_Defcon;
datetime    g_LastDefconCheck = 0;
int         g_PreviousDefcon = 5;

//+------------------------------------------------------------------+
//| Read DEFCON level from file                                       |
//+------------------------------------------------------------------+
int ReadDefconLevel() {
    if(!FileIsExist(DEFCON_FilePath)) {
        Print("DEFCON: File not found, defaulting to DEFCON 5");
        return DEFCON_NORMAL;
    }
    
    int handle = FileOpen(DEFCON_FilePath, FILE_READ|FILE_TXT);
    if(handle == INVALID_HANDLE) {
        Print("DEFCON: Cannot open file, defaulting to DEFCON 5");
        return DEFCON_NORMAL;
    }
    
    string content = FileReadString(handle);
    FileClose(handle);
    
    int level = (int)StringToInteger(StringTrimRight(StringTrimLeft(content)));
    
    if(level < 1 || level > 5) {
        Print("DEFCON: Invalid level ", level, ", defaulting to DEFCON 5");
        return DEFCON_NORMAL;
    }
    
    return level;
}

//+------------------------------------------------------------------+
//| Apply DEFCON configuration based on level                         |
//+------------------------------------------------------------------+
void ApplyDefconConfig(int level) {
    g_Defcon.level = level;
    g_Defcon.lastUpdate = TimeCurrent();
    
    switch(level) {
        case 5:  // NORMAL
            g_Defcon.color = "GREEN";
            g_Defcon.name = "NORMAL";
            g_Defcon.lotMultiplier = 1.0;
            g_Defcon.maxPositions = 5;
            g_Defcon.minConfidence = 40;
            g_Defcon.entryGapMult = 1.0;
            g_Defcon.pauseEntries = false;
            g_Defcon.survivalMode = false;
            g_Defcon.closePercent = 0;
            g_Defcon.setBreakeven = false;
            g_Defcon.hedgeMode = false;
            g_Defcon.hedgePercent = 0;
            g_Defcon.deltaNeutral = false;
            break;
            
        case 4:  // ELEVATED
            g_Defcon.color = "BLUE";
            g_Defcon.name = "ELEVATED";
            g_Defcon.lotMultiplier = 0.8;
            g_Defcon.maxPositions = 4;
            g_Defcon.minConfidence = 70;
            g_Defcon.entryGapMult = 1.2;
            g_Defcon.pauseEntries = false;
            g_Defcon.survivalMode = false;
            g_Defcon.closePercent = 0;
            g_Defcon.setBreakeven = false;
            g_Defcon.hedgeMode = false;
            g_Defcon.hedgePercent = 0;
            g_Defcon.deltaNeutral = false;
            break;
            
        case 3:  // HIGH ALERT
            g_Defcon.color = "YELLOW";
            g_Defcon.name = "HIGH ALERT";
            g_Defcon.lotMultiplier = 0.5;
            g_Defcon.maxPositions = 3;
            g_Defcon.minConfidence = 80;
            g_Defcon.entryGapMult = 2.0;
            g_Defcon.pauseEntries = false;
            g_Defcon.survivalMode = false;
            g_Defcon.closePercent = 0;
            g_Defcon.setBreakeven = false;
            g_Defcon.hedgeMode = true;
            g_Defcon.hedgePercent = 25;
            g_Defcon.deltaNeutral = false;
            break;
            
        case 2:  // SEVERE
            g_Defcon.color = "ORANGE";
            g_Defcon.name = "SEVERE";
            g_Defcon.lotMultiplier = 0.0;
            g_Defcon.maxPositions = 0;
            g_Defcon.minConfidence = 100;
            g_Defcon.entryGapMult = 999;
            g_Defcon.pauseEntries = true;
            g_Defcon.survivalMode = false;
            g_Defcon.closePercent = 30;
            g_Defcon.setBreakeven = false;
            g_Defcon.hedgeMode = true;
            g_Defcon.hedgePercent = 50;
            g_Defcon.deltaNeutral = false;
            break;
            
        case 1:  // MAXIMUM THREAT
            g_Defcon.color = "RED";
            g_Defcon.name = "MAXIMUM THREAT";
            g_Defcon.lotMultiplier = 0.0;
            g_Defcon.maxPositions = 0;
            g_Defcon.minConfidence = 100;
            g_Defcon.entryGapMult = 999;
            g_Defcon.pauseEntries = true;
            g_Defcon.survivalMode = true;
            g_Defcon.closePercent = 50;
            g_Defcon.setBreakeven = true;
            g_Defcon.hedgeMode = true;
            g_Defcon.hedgePercent = 100;
            g_Defcon.deltaNeutral = true;
            break;
    }
}

//+------------------------------------------------------------------+
//| Update DEFCON - call from OnTick or OnTimer                       |
//+------------------------------------------------------------------+
bool UpdateDefcon() {
    //--- Rate limit checks
    if(TimeCurrent() - g_LastDefconCheck < DEFCON_CheckInterval) {
        return false;
    }
    g_LastDefconCheck = TimeCurrent();
    
    //--- Read current level
    int newLevel = ReadDefconLevel();
    
    //--- Check for change
    if(newLevel != g_PreviousDefcon) {
        string direction = (newLevel < g_PreviousDefcon) ? "⬆️ UPGRADED" : "⬇️ DOWNGRADED";
        
        Print("═══════════════════════════════════════════════════");
        Print("🚨 DEFCON CHANGE: ", g_PreviousDefcon, " → ", newLevel, " (", direction, ")");
        Print("═══════════════════════════════════════════════════");
        
        g_PreviousDefcon = newLevel;
        ApplyDefconConfig(newLevel);
        
        //--- Alert on significant changes
        if(newLevel <= 2) {
            Alert("🚨 DEFCON ", newLevel, " - ", g_Defcon.name, " - ", direction);
        }
        
        return true;  // State changed
    }
    
    return false;  // No change
}

//+------------------------------------------------------------------+
//| Ghost-specific functions                                          |
//+------------------------------------------------------------------+

// Should Ghost open new positions?
bool Ghost_CanOpenPosition() {
    if(g_Defcon.pauseEntries) {
        Print("DEFCON ", g_Defcon.level, ": Entries PAUSED");
        return false;
    }
    return true;
}

// Get Ghost lot multiplier
double Ghost_GetLotMultiplier() {
    return g_Defcon.lotMultiplier;
}

// Get Ghost max positions
int Ghost_GetMaxPositions() {
    return g_Defcon.maxPositions;
}

// Get Ghost entry gap multiplier (for spacing entries)
double Ghost_GetEntryGapMultiplier() {
    return g_Defcon.entryGapMult;
}

// Should Ghost close partial positions?
int Ghost_GetClosePercent() {
    return g_Defcon.closePercent;
}

// Should Ghost set breakeven stops?
bool Ghost_ShouldSetBreakeven() {
    return g_Defcon.setBreakeven;
}

// Is Ghost in survival mode?
bool Ghost_IsSurvivalMode() {
    return g_Defcon.survivalMode;
}

// Write Ghost exposure for Casper to read
void Ghost_WriteExposure(double exposure, double totalLots, int positionCount) {
    int handle = FileOpen(GHOST_ExposurePath, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE) {
        // Format: exposure,lots,count,defcon
        string data = StringFormat("%.2f,%.2f,%d,%d", exposure, totalLots, positionCount, g_Defcon.level);
        FileWriteString(handle, data);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Casper-specific functions                                         |
//+------------------------------------------------------------------+

// Should Casper hedge Ghost?
bool Casper_ShouldHedge() {
    return g_Defcon.hedgeMode;
}

// Get hedge percentage of Ghost exposure
int Casper_GetHedgePercent() {
    return g_Defcon.hedgePercent;
}

// Should Casper go full delta neutral?
bool Casper_IsDeltaNeutral() {
    return g_Defcon.deltaNeutral;
}

// Read Ghost exposure
bool Casper_ReadGhostExposure(double &exposure, double &lots, int &count) {
    if(!FileIsExist(GHOST_ExposurePath)) {
        return false;
    }
    
    int handle = FileOpen(GHOST_ExposurePath, FILE_READ|FILE_TXT);
    if(handle == INVALID_HANDLE) {
        return false;
    }
    
    string content = FileReadString(handle);
    FileClose(handle);
    
    // Parse: exposure,lots,count,defcon
    string parts[];
    int n = StringSplit(content, ',', parts);
    if(n >= 3) {
        exposure = StringToDouble(parts[0]);
        lots = StringToDouble(parts[1]);
        count = (int)StringToInteger(parts[2]);
        return true;
    }
    
    return false;
}

// Calculate required hedge size
double Casper_CalculateHedgeSize(double ghostLots) {
    if(!g_Defcon.hedgeMode) return 0;
    return ghostLots * (g_Defcon.hedgePercent / 100.0);
}

// Write Casper hedge status
void Casper_WriteHedgeStatus(double hedgeLots, bool hedgeActive) {
    int handle = FileOpen(CASPER_HedgePath, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE) {
        // Format: hedgeLots,hedgeActive,defcon
        string data = StringFormat("%.2f,%d,%d", hedgeLots, hedgeActive ? 1 : 0, g_Defcon.level);
        FileWriteString(handle, data);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Display DEFCON status on chart                                    |
//+------------------------------------------------------------------+
void DisplayDefconStatus(int x = 10, int y = 30) {
    string objName = "DEFCON_STATUS";
    
    color clr;
    switch(g_Defcon.level) {
        case 1: clr = clrRed; break;
        case 2: clr = clrOrange; break;
        case 3: clr = clrYellow; break;
        case 4: clr = clrDodgerBlue; break;
        default: clr = clrLime; break;
    }
    
    string text = StringFormat("DEFCON %d - %s", g_Defcon.level, g_Defcon.name);
    
    if(ObjectFind(0, objName) < 0) {
        ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
    }
    
    ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, 14);
    ObjectSetString(0, objName, OBJPROP_FONT, "Arial Bold");
    ObjectSetString(0, objName, OBJPROP_TEXT, text);
    
    // Additional status line
    string statusName = "DEFCON_DETAIL";
    string detail = "";
    
    if(g_Defcon.pauseEntries) {
        detail = "⛔ ENTRIES PAUSED";
    } else {
        detail = StringFormat("Lot: %.0f%% | Max: %d pos", g_Defcon.lotMultiplier * 100, g_Defcon.maxPositions);
    }
    
    if(g_Defcon.hedgeMode) {
        detail += StringFormat(" | 🛡️ HEDGE %d%%", g_Defcon.hedgePercent);
    }
    
    if(ObjectFind(0, statusName) < 0) {
        ObjectCreate(0, statusName, OBJ_LABEL, 0, 0, 0);
    }
    
    ObjectSetInteger(0, statusName, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, statusName, OBJPROP_YDISTANCE, y + 20);
    ObjectSetInteger(0, statusName, OBJPROP_COLOR, clrWhite);
    ObjectSetInteger(0, statusName, OBJPROP_FONTSIZE, 10);
    ObjectSetString(0, statusName, OBJPROP_FONT, "Arial");
    ObjectSetString(0, statusName, OBJPROP_TEXT, detail);
}

//+------------------------------------------------------------------+
//| Print full DEFCON summary to log                                  |
//+------------------------------------------------------------------+
void PrintDefconSummary() {
    Print("╔══════════════════════════════════════════════════════════╗");
    Print("║               DEFCON STATUS SUMMARY                      ║");
    Print("╠══════════════════════════════════════════════════════════╣");
    Print("║ Level:        DEFCON ", g_Defcon.level, " - ", g_Defcon.name);
    Print("║ Color:        ", g_Defcon.color);
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ GHOST SETTINGS:");
    Print("║   Lot Mult:     ", g_Defcon.lotMultiplier * 100, "%");
    Print("║   Max Pos:      ", g_Defcon.maxPositions);
    Print("║   Min Conf:     ", g_Defcon.minConfidence, "%");
    Print("║   Entry Gap:    ", g_Defcon.entryGapMult, "x");
    Print("║   Paused:       ", g_Defcon.pauseEntries ? "YES ⛔" : "NO");
    Print("║   Survival:     ", g_Defcon.survivalMode ? "YES 🚨" : "NO");
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ CASPER SETTINGS:");
    Print("║   Hedge Mode:   ", g_Defcon.hedgeMode ? "ACTIVE 🛡️" : "OFF");
    Print("║   Hedge %:      ", g_Defcon.hedgePercent, "%");
    Print("║   Delta Neut:   ", g_Defcon.deltaNeutral ? "YES" : "NO");
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Last Update:    ", TimeToString(g_Defcon.lastUpdate));
    Print("╚══════════════════════════════════════════════════════════╝");
}

//+------------------------------------------------------------------+
//| Initialize DEFCON system - call from OnInit                       |
//+------------------------------------------------------------------+
void InitDefcon() {
    int level = ReadDefconLevel();
    ApplyDefconConfig(level);
    g_PreviousDefcon = level;
    PrintDefconSummary();
}
//+------------------------------------------------------------------+

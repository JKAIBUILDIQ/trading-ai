//+------------------------------------------------------------------+
//|                                                   DefconGrid.mqh |
//|                    DEFCON-Aware Grid Trading Module for Ghost    |
//|                                                                  |
//| Grid trading ONLY active in DEFCON 4-5 (normal/elevated)         |
//| Automatically adjusts spacing, lots, and max levels by DEFCON    |
//+------------------------------------------------------------------+
#property copyright "NEO Trading Intelligence"
#property version   "1.00"

#include "DefconReader.mqh"

//=== GRID INPUT PARAMETERS ===
input group "=== 📊 GRID BASE CONFIGURATION ==="
input bool   Grid_Enabled = true;              // Enable grid trading
input double Grid_BasePipSpacing = 20.0;       // Base pip spacing (DEFCON 5)
input double Grid_BaseLotSize = 0.5;           // Base lot per level (DEFCON 5)
input int    Grid_MaxLevels = 5;               // Max grid positions (DEFCON 5)
input double Grid_TP_Pips = 25.0;              // TP pips above average entry
input double Grid_MaxDrawdown = 25000;         // Max drawdown before emergency exit

input group "=== 📊 GRID DEFCON MULTIPLIERS ==="
input double Grid_D4_SpacingMult = 1.25;       // DEFCON 4: Spacing multiplier
input double Grid_D4_LotMult = 0.8;            // DEFCON 4: Lot multiplier
input int    Grid_D4_MaxLevels = 4;            // DEFCON 4: Max levels
input double Grid_D3_SpacingMult = 2.0;        // DEFCON 3: Spacing multiplier
input double Grid_D3_LotMult = 0.5;            // DEFCON 3: Lot multiplier
input int    Grid_D3_MaxLevels = 3;            // DEFCON 3: Max levels

input group "=== 📊 GRID SAFETY ==="
input double Grid_ATR_Multiplier = 1.5;        // Spacing = ATR × this value
input bool   Grid_UseATRSpacing = true;        // Use ATR-based spacing
input int    Grid_ATR_Period = 14;             // ATR period for spacing calc
input double Grid_MinSpacing = 15.0;           // Minimum spacing regardless of ATR

//=== GRID STATE STRUCTURE ===
struct GridEntry {
    int      ticket;
    double   price;
    double   lots;
    datetime time;
    bool     active;
};

struct GridState {
    GridEntry entries[10];    // Max 10 grid levels
    int       entryCount;
    double    totalLots;
    double    averageEntry;
    double    currentDrawdown;
    double    takeProfitPrice;
    bool      gridActive;
    int       defconLevel;
};

//=== GLOBAL GRID STATE ===
GridState g_Grid;

//+------------------------------------------------------------------+
//| Initialize Grid System                                            |
//+------------------------------------------------------------------+
void Grid_Init() {
    ArrayInitialize(g_Grid.entries, GridEntry());
    g_Grid.entryCount = 0;
    g_Grid.totalLots = 0;
    g_Grid.averageEntry = 0;
    g_Grid.currentDrawdown = 0;
    g_Grid.takeProfitPrice = 0;
    g_Grid.gridActive = Grid_Enabled;
    g_Grid.defconLevel = 5;
    
    Print("📊 Grid System Initialized");
    Print("   Base Spacing: ", Grid_BasePipSpacing, " pips");
    Print("   Base Lots: ", Grid_BaseLotSize);
    Print("   Max Levels: ", Grid_MaxLevels);
}

//+------------------------------------------------------------------+
//| Check if Grid Trading is Allowed at Current DEFCON               |
//+------------------------------------------------------------------+
bool Grid_IsAllowed() {
    if(!Grid_Enabled) return false;
    
    int defcon = NEO_GetDefcon();
    g_Grid.defconLevel = defcon;
    
    // Grid ONLY allowed in DEFCON 4-5
    if(defcon <= 2) {
        if(g_Grid.gridActive) {
            Print("⛔ Grid DISABLED - DEFCON ", defcon, " (Severe/Maximum)");
            g_Grid.gridActive = false;
        }
        return false;
    }
    
    if(defcon == 3) {
        if(g_Grid.gridActive) {
            Print("⏸️ Grid PAUSED - DEFCON 3 (High Alert) - No new entries");
            g_Grid.gridActive = false;
        }
        return false;  // No new entries, but hold existing
    }
    
    // DEFCON 4-5: Grid allowed
    if(!g_Grid.gridActive) {
        Print("✅ Grid ACTIVE - DEFCON ", defcon);
        g_Grid.gridActive = true;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get Current Grid Pip Spacing Based on DEFCON and ATR             |
//+------------------------------------------------------------------+
double Grid_GetPipSpacing() {
    double spacing = Grid_BasePipSpacing;
    
    // Use ATR-based spacing if enabled
    if(Grid_UseATRSpacing) {
        double atr = iATR(_Symbol, PERIOD_H1, Grid_ATR_Period);
        double atrPips = atr / _Point / 10;  // Convert to pips
        spacing = MathMax(atrPips * Grid_ATR_Multiplier, Grid_MinSpacing);
    }
    
    // Apply DEFCON multiplier
    int defcon = NEO_GetDefcon();
    
    switch(defcon) {
        case 4:
            spacing *= Grid_D4_SpacingMult;
            break;
        case 3:
            spacing *= Grid_D3_SpacingMult;
            break;
        // DEFCON 5: base spacing (1.0x)
        // DEFCON 2-1: grid disabled
    }
    
    return spacing;
}

//+------------------------------------------------------------------+
//| Get Current Grid Lot Size Based on DEFCON                        |
//+------------------------------------------------------------------+
double Grid_GetLotSize() {
    double lots = Grid_BaseLotSize;
    int defcon = NEO_GetDefcon();
    
    switch(defcon) {
        case 4:
            lots *= Grid_D4_LotMult;
            break;
        case 3:
            lots *= Grid_D3_LotMult;
            break;
        case 2:
        case 1:
            lots = 0;  // No new entries
            break;
        // DEFCON 5: base lots (1.0x)
    }
    
    return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Get Current Max Grid Levels Based on DEFCON                      |
//+------------------------------------------------------------------+
int Grid_GetMaxLevels() {
    int defcon = NEO_GetDefcon();
    
    switch(defcon) {
        case 4:
            return Grid_D4_MaxLevels;
        case 3:
            return Grid_D3_MaxLevels;
        case 2:
        case 1:
            return 0;  // No grid
        default:
            return Grid_MaxLevels;  // DEFCON 5
    }
}

//+------------------------------------------------------------------+
//| Calculate Average Entry Price                                     |
//+------------------------------------------------------------------+
double Grid_CalculateAverageEntry() {
    if(g_Grid.entryCount == 0) return 0;
    
    double totalValue = 0;
    double totalLots = 0;
    
    for(int i = 0; i < g_Grid.entryCount; i++) {
        if(g_Grid.entries[i].active) {
            totalValue += g_Grid.entries[i].price * g_Grid.entries[i].lots;
            totalLots += g_Grid.entries[i].lots;
        }
    }
    
    g_Grid.totalLots = totalLots;
    g_Grid.averageEntry = (totalLots > 0) ? totalValue / totalLots : 0;
    
    return g_Grid.averageEntry;
}

//+------------------------------------------------------------------+
//| Calculate Take Profit Price                                       |
//+------------------------------------------------------------------+
double Grid_CalculateTP() {
    double avgEntry = Grid_CalculateAverageEntry();
    if(avgEntry == 0) return 0;
    
    double tpPips = Grid_TP_Pips;
    int defcon = NEO_GetDefcon();
    
    // Adjust TP based on DEFCON (tighter when higher alert)
    switch(defcon) {
        case 3:
            tpPips *= 0.6;  // Tighter TP - take profits faster
            break;
        case 4:
            tpPips *= 1.2;  // Slightly wider
            break;
        // DEFCON 5: standard TP
    }
    
    g_Grid.takeProfitPrice = avgEntry + (tpPips * _Point * 10);
    return g_Grid.takeProfitPrice;
}

//+------------------------------------------------------------------+
//| Check if Should Add Grid Entry                                    |
//+------------------------------------------------------------------+
bool Grid_ShouldAddEntry(double currentPrice, double lastEntryPrice) {
    if(!Grid_IsAllowed()) return false;
    if(g_Grid.entryCount >= Grid_GetMaxLevels()) return false;
    
    double spacing = Grid_GetPipSpacing() * _Point * 10;
    
    // For LONG grid: add entry if price dropped by spacing amount
    if(lastEntryPrice - currentPrice >= spacing) {
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Add Grid Entry (Call after opening position)                      |
//+------------------------------------------------------------------+
void Grid_AddEntry(int ticket, double price, double lots) {
    if(g_Grid.entryCount >= 10) return;
    
    g_Grid.entries[g_Grid.entryCount].ticket = ticket;
    g_Grid.entries[g_Grid.entryCount].price = price;
    g_Grid.entries[g_Grid.entryCount].lots = lots;
    g_Grid.entries[g_Grid.entryCount].time = TimeCurrent();
    g_Grid.entries[g_Grid.entryCount].active = true;
    g_Grid.entryCount++;
    
    Grid_CalculateAverageEntry();
    Grid_CalculateTP();
    
    Print("📊 Grid Entry Added: #", ticket, " @ ", price, " (", lots, " lots)");
    Print("   Total Entries: ", g_Grid.entryCount);
    Print("   Average Entry: ", g_Grid.averageEntry);
    Print("   Take Profit: ", g_Grid.takeProfitPrice);
}

//+------------------------------------------------------------------+
//| Remove Grid Entry (Call after closing position)                   |
//+------------------------------------------------------------------+
void Grid_RemoveEntry(int ticket) {
    for(int i = 0; i < g_Grid.entryCount; i++) {
        if(g_Grid.entries[i].ticket == ticket) {
            g_Grid.entries[i].active = false;
            Print("📊 Grid Entry Removed: #", ticket);
            break;
        }
    }
    
    // Recalculate
    Grid_CalculateAverageEntry();
    Grid_CalculateTP();
}

//+------------------------------------------------------------------+
//| Check if Grid Should Take Profit                                  |
//+------------------------------------------------------------------+
bool Grid_ShouldTakeProfit(double currentPrice) {
    if(g_Grid.entryCount == 0) return false;
    if(g_Grid.takeProfitPrice == 0) return false;
    
    return (currentPrice >= g_Grid.takeProfitPrice);
}

//+------------------------------------------------------------------+
//| Get Last Entry Price                                              |
//+------------------------------------------------------------------+
double Grid_GetLastEntryPrice() {
    for(int i = g_Grid.entryCount - 1; i >= 0; i--) {
        if(g_Grid.entries[i].active) {
            return g_Grid.entries[i].price;
        }
    }
    return 0;
}

//+------------------------------------------------------------------+
//| Calculate Current Grid P&L                                        |
//+------------------------------------------------------------------+
double Grid_CalculatePnL(double currentPrice) {
    if(g_Grid.totalLots == 0 || g_Grid.averageEntry == 0) return 0;
    
    double priceDiff = currentPrice - g_Grid.averageEntry;
    double pips = priceDiff / (_Point * 10);
    
    // Approximate pip value for Gold: $100 per pip per lot
    double pipValue = 100.0;
    
    return pips * g_Grid.totalLots * pipValue;
}

//+------------------------------------------------------------------+
//| Check for Emergency Grid Exit                                     |
//+------------------------------------------------------------------+
bool Grid_ShouldEmergencyExit(double currentPrice) {
    double pnl = Grid_CalculatePnL(currentPrice);
    g_Grid.currentDrawdown = (pnl < 0) ? MathAbs(pnl) : 0;
    
    // Emergency exit if drawdown exceeds threshold
    if(g_Grid.currentDrawdown >= Grid_MaxDrawdown) {
        Print("🚨 GRID EMERGENCY EXIT - Drawdown $", g_Grid.currentDrawdown);
        return true;
    }
    
    // Emergency exit on DEFCON 1
    if(NEO_GetDefcon() == 1 && g_Grid.entryCount > 0) {
        Print("🚨 GRID DEFCON 1 EXIT - Closing 50% of positions");
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Display Grid Status on Chart                                      |
//+------------------------------------------------------------------+
void Grid_DisplayStatus(int x = 10, int y = 80) {
    string prefix = "GRID_";
    
    // Status line
    string status = g_Grid.gridActive ? "ACTIVE" : "PAUSED";
    color statusColor = g_Grid.gridActive ? clrLime : clrOrange;
    
    string text = StringFormat("GRID: %s | Entries: %d/%d | Avg: %.2f",
                               status, g_Grid.entryCount, Grid_GetMaxLevels(),
                               g_Grid.averageEntry);
    
    if(ObjectFind(0, prefix + "STATUS") < 0) {
        ObjectCreate(0, prefix + "STATUS", OBJ_LABEL, 0, 0, 0);
    }
    ObjectSetInteger(0, prefix + "STATUS", OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, prefix + "STATUS", OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, prefix + "STATUS", OBJPROP_COLOR, statusColor);
    ObjectSetInteger(0, prefix + "STATUS", OBJPROP_FONTSIZE, 10);
    ObjectSetString(0, prefix + "STATUS", OBJPROP_TEXT, text);
    
    // P&L line
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double pnl = Grid_CalculatePnL(bid);
    color pnlColor = (pnl >= 0) ? clrLime : clrRed;
    
    string pnlText = StringFormat("P&L: $%.2f | TP: %.2f | Spacing: %.0f pips",
                                   pnl, g_Grid.takeProfitPrice, Grid_GetPipSpacing());
    
    if(ObjectFind(0, prefix + "PNL") < 0) {
        ObjectCreate(0, prefix + "PNL", OBJ_LABEL, 0, 0, 0);
    }
    ObjectSetInteger(0, prefix + "PNL", OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, prefix + "PNL", OBJPROP_YDISTANCE, y + 15);
    ObjectSetInteger(0, prefix + "PNL", OBJPROP_COLOR, pnlColor);
    ObjectSetInteger(0, prefix + "PNL", OBJPROP_FONTSIZE, 9);
    ObjectSetString(0, prefix + "PNL", OBJPROP_TEXT, pnlText);
}

//+------------------------------------------------------------------+
//| Print Grid Summary to Log                                         |
//+------------------------------------------------------------------+
void Grid_PrintSummary() {
    Print("╔══════════════════════════════════════════════════════════╗");
    Print("║                   GRID STATUS SUMMARY                    ║");
    Print("╠══════════════════════════════════════════════════════════╣");
    Print("║ Status:       ", g_Grid.gridActive ? "ACTIVE ✅" : "PAUSED ⏸️");
    Print("║ DEFCON:       ", g_Grid.defconLevel);
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Entries:      ", g_Grid.entryCount, " / ", Grid_GetMaxLevels());
    Print("║ Total Lots:   ", g_Grid.totalLots);
    Print("║ Avg Entry:    $", g_Grid.averageEntry);
    Print("║ TP Price:     $", g_Grid.takeProfitPrice);
    Print("║ ────────────────────────────────────────────────────────");
    Print("║ Pip Spacing:  ", Grid_GetPipSpacing(), " pips");
    Print("║ Lot Size:     ", Grid_GetLotSize(), " per level");
    Print("║ ────────────────────────────────────────────────────────");
    
    for(int i = 0; i < g_Grid.entryCount; i++) {
        if(g_Grid.entries[i].active) {
            Print("║ Entry ", i+1, ":     #", g_Grid.entries[i].ticket, 
                  " @ $", g_Grid.entries[i].price, 
                  " (", g_Grid.entries[i].lots, " lots)");
        }
    }
    
    Print("╚══════════════════════════════════════════════════════════╝");
}

//+------------------------------------------------------------------+
//| Reset Grid (after full TP or emergency exit)                      |
//+------------------------------------------------------------------+
void Grid_Reset() {
    ArrayInitialize(g_Grid.entries, GridEntry());
    g_Grid.entryCount = 0;
    g_Grid.totalLots = 0;
    g_Grid.averageEntry = 0;
    g_Grid.currentDrawdown = 0;
    g_Grid.takeProfitPrice = 0;
    
    Print("📊 Grid RESET - Ready for new cycle");
}
//+------------------------------------------------------------------+

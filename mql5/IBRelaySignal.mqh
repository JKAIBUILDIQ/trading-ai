//+------------------------------------------------------------------+
//|                                               IBRelaySignal.mqh  |
//|                                    MT5 → IB Signal Relay Module  |
//|                                                    Quinn / NEO   |
//+------------------------------------------------------------------+
#property copyright "Quinn/NEO Trading"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Configuration - Set your H100 Tailscale IP                       |
//+------------------------------------------------------------------+
input string IB_Relay_URL     = "http://100.74.123.157:5000/signal";  // H100 Tailscale IP
input bool   Enable_IB_Relay  = true;                                   // Enable IB relay
input string IB_Auth_Token    = "ghost-casper-relay-2026";             // Auth token

//+------------------------------------------------------------------+
//| Symbol mapping: MT5 → IB                                          |
//+------------------------------------------------------------------+
string GetIBSymbol(string mt5Symbol)
{
    // Gold
    if(mt5Symbol == "XAUUSD" || mt5Symbol == "GOLD" || mt5Symbol == "XAUUSDm")
        return "MGC";
    
    // Forex micros (if needed)
    if(mt5Symbol == "EURUSD") return "M6E";
    if(mt5Symbol == "GBPUSD") return "M6B";
    if(mt5Symbol == "USDJPY") return "M6J";
    if(mt5Symbol == "AUDUSD") return "M6A";
    
    // Default: return as-is
    return mt5Symbol;
}

//+------------------------------------------------------------------+
//| Lot size conversion: MT5 lots → IB contracts                      |
//+------------------------------------------------------------------+
int GetIBContracts(string mt5Symbol, double lots)
{
    // XAUUSD: 0.10 lot ≈ 1 MGC contract
    // 0.10 lot = 10 oz exposure (at 100 oz per lot)
    // 1 MGC = 10 oz exposure
    if(mt5Symbol == "XAUUSD" || mt5Symbol == "GOLD" || mt5Symbol == "XAUUSDm")
    {
        int contracts = (int)MathRound(lots * 10);  // 0.10 lot = 1 contract
        return MathMax(1, contracts);
    }
    
    // Default: 1 lot = 1 contract
    return MathMax(1, (int)MathRound(lots));
}

//+------------------------------------------------------------------+
//| Send signal to IB relay via HTTP POST                             |
//+------------------------------------------------------------------+
bool SendSignalToIB(string action, string symbol, double lots, double price, 
                    double tp = 0, double sl = 0, string comment = "")
{
    if(!Enable_IB_Relay)
    {
        Print("IB Relay disabled - signal not sent");
        return false;
    }
    
    // Convert symbol
    string ibSymbol = GetIBSymbol(symbol);
    
    // Build JSON payload
    string json = StringFormat(
        "{"
        "\"action\":\"%s\","
        "\"symbol\":\"%s\","
        "\"lots\":%.2f,"
        "\"price\":%.2f,"
        "\"tp\":%.2f,"
        "\"sl\":%.2f,"
        "\"comment\":\"%s\","
        "\"timestamp\":\"%s\","
        "\"source\":\"MT5_Ghost\","
        "\"token\":\"%s\""
        "}",
        action,
        ibSymbol,
        lots,
        price,
        tp,
        sl,
        comment,
        TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
        IB_Auth_Token
    );
    
    // Prepare request
    char post[];
    char result[];
    string headers = "Content-Type: application/json\r\n";
    
    StringToCharArray(json, post);
    ArrayResize(post, ArraySize(post) - 1);  // Remove null terminator
    
    int timeout = 5000;  // 5 second timeout
    string result_headers;
    
    // Send HTTP POST
    ResetLastError();
    int res = WebRequest("POST", IB_Relay_URL, headers, timeout, post, result, result_headers);
    
    if(res == -1)
    {
        int error = GetLastError();
        PrintFormat("⚠️ IB Relay WebRequest failed: Error %d", error);
        
        if(error == 4014)
        {
            Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Print("  ERROR 4014: WebRequest URL not allowed");
            Print("  Fix: Tools → Options → Expert Advisors → Allow WebRequest for listed URL");
            PrintFormat("  Add this URL: %s", IB_Relay_URL);
            Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }
        return false;
    }
    
    // Parse response
    string response = CharArrayToString(result);
    
    // Log success
    PrintFormat("📡 IB Relay: %s %s %.2f lots @ %.2f → %s", 
                action, ibSymbol, lots, price, 
                StringLen(response) > 100 ? StringSubstr(response, 0, 100) + "..." : response);
    
    return true;
}

//+------------------------------------------------------------------+
//| Convenience functions for common signals                          |
//+------------------------------------------------------------------+

// Send BUY signal
bool RelayBuy(string symbol, double lots, double price, double tp = 0, double sl = 0, string comment = "")
{
    return SendSignalToIB("BUY", symbol, lots, price, tp, sl, comment);
}

// Send SELL signal (open short or reduce long)
bool RelaySell(string symbol, double lots, double price, double tp = 0, double sl = 0, string comment = "")
{
    return SendSignalToIB("SELL", symbol, lots, price, tp, sl, comment);
}

// Send CLOSE signal (close specific quantity)
bool RelayClose(string symbol, double lots, double price, string comment = "")
{
    return SendSignalToIB("CLOSE", symbol, lots, price, 0, 0, comment);
}

// Close all long positions
bool RelayCloseAllLong(string symbol)
{
    return SendSignalToIB("CLOSE_ALL_LONG", symbol, 0, 0, 0, 0, "FLATTEN_LONGS");
}

// Close all short positions  
bool RelayCloseAllShort(string symbol)
{
    return SendSignalToIB("CLOSE_ALL_SHORT", symbol, 0, 0, 0, 0, "FLATTEN_SHORTS");
}

//+------------------------------------------------------------------+
//| Test the relay connection                                         |
//+------------------------------------------------------------------+
bool TestIBRelay()
{
    if(!Enable_IB_Relay)
    {
        Print("IB Relay is disabled");
        return false;
    }
    
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("  Testing IB Relay Connection...");
    PrintFormat("  URL: %s", IB_Relay_URL);
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Test health endpoint
    string health_url = StringReplace(IB_Relay_URL, "/signal", "/health");
    
    char post[];
    char result[];
    string result_headers;
    
    ResetLastError();
    int res = WebRequest("GET", health_url, "", 5000, post, result, result_headers);
    
    if(res == -1)
    {
        int error = GetLastError();
        PrintFormat("  ❌ Connection failed: Error %d", error);
        return false;
    }
    
    string response = CharArrayToString(result);
    PrintFormat("  ✅ Relay is reachable: %s", response);
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    return true;
}

//+------------------------------------------------------------------+
//| Example integration with existing EA                              |
//+------------------------------------------------------------------+
/*
// In your EA's OnTick() or trade function:

#include <IBRelaySignal.mqh>

void ExecuteDCABuy(double lots, double price, int level)
{
    // Execute on MT5 first
    if(OrderSend(...))  // Your existing MT5 order logic
    {
        // Then relay to IB
        string comment = StringFormat("DROPBUY|L%d|%.2f", level, lots);
        RelayBuy(_Symbol, lots, price, price + 50, 0, comment);
    }
}

void CloseTakeProfit(double lots, double price, int level)
{
    // Close on MT5 first
    if(OrderClose(...))  // Your existing close logic
    {
        // Then relay to IB
        string comment = StringFormat("TP_HIT|L%d", level);
        RelayClose(_Symbol, lots, price, comment);
    }
}

void OpenCorrectionHedge(double lots, double price)
{
    // Open hedge on MT5
    if(OrderSend(...))  // SELL order
    {
        // Then relay to IB
        string comment = StringFormat("CORRECTION|HEDGE|%.0f", price);
        RelaySell(_Symbol, lots, price, price - 100, price + 30, comment);
    }
}
*/
//+------------------------------------------------------------------+

#!/bin/bash
# NEO's War Room Startup Script
# Starts all intel services and NEO

echo "════════════════════════════════════════════════════════"
echo "🎖️  STARTING NEO'S WAR ROOM"
echo "════════════════════════════════════════════════════════"
echo ""

# Change to NEO directory
cd ~/trading_ai/neo

# Check if already running
check_running() {
    pgrep -f "$1" > /dev/null
}

# Kill existing processes if needed
if [ "$1" == "--restart" ]; then
    echo "🔄 Restarting all services..."
    pkill -f "position_monitor.py" 2>/dev/null
    pkill -f "intel_bots.py" 2>/dev/null
    pkill -f "neo_trader.py" 2>/dev/null
    sleep 2
fi

echo "📡 Starting Position Monitor..."
if ! check_running "position_monitor.py"; then
    python3 ~/trading_ai/neo/intel/position_monitor.py --run --interval 30 &
    echo "   ✅ Position Monitor started (PID: $!)"
else
    echo "   ⚠️  Position Monitor already running"
fi

echo ""
echo "🤖 Starting Intel Bots..."
if ! check_running "intel_bots.py"; then
    python3 ~/trading_ai/neo/intel/intel_bots.py --run --interval 300 &
    echo "   ✅ Intel Bots started (PID: $!)"
    echo "   📡 ALPHA Scanner (qwen3:32b)"
    echo "   📡 BRAVO News (llama3.1:8b)"
    echo "   📡 CHARLIE Analyst (deepseek-r1:32b)"
    echo "   📡 DELTA Threat (qwen3:32b)"
else
    echo "   ⚠️  Intel Bots already running"
fi

# Wait for initial data
echo ""
echo "⏳ Waiting for initial data collection (15s)..."
sleep 15

echo ""
echo "🧠 Starting NEO..."
if ! check_running "neo_trader.py"; then
    python3 ~/trading_ai/neo/neo_trader.py &
    echo "   ✅ NEO started (PID: $!)"
else
    echo "   ⚠️  NEO already running"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "🎖️  WAR ROOM OPERATIONAL"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📂 Data Files:"
echo "   Position data:  /tmp/neo_positions.json"
echo "   Battlefield:    /tmp/neo_battlefield.txt"
echo "   Intel report:   /tmp/neo_intel_report.json"
echo "   NEO signals:    /tmp/neo_signal.json"
echo ""
echo "📋 Commands:"
echo "   # View battlefield"
echo "   cat /tmp/neo_battlefield.txt"
echo ""
echo "   # View intel"
echo "   cat /tmp/neo_intel_report.json | jq ."
echo ""
echo "   # View NEO's latest signal"
echo "   cat /tmp/neo_signal.json | jq ."
echo ""
echo "   # View NEO logs"
echo "   tail -f ~/trading_ai/neo/logs/neo_\$(date +%Y%m%d).log"
echo ""
echo "   # Stop all"
echo "   pkill -f 'position_monitor.py|intel_bots.py|neo_trader.py'"
echo ""
echo "════════════════════════════════════════════════════════"

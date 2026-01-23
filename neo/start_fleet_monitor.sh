#!/bin/bash
# Start Fleet Monitor as daemon
# Updates portfolio_state.json every 30 seconds

echo "=============================================="
echo "🎖️ FLEET MONITOR - Starting Daemon"
echo "=============================================="

cd ~/trading_ai/neo

# Check if already running
if pgrep -f "fleet_monitor.py --daemon" > /dev/null; then
    echo "⚠️ Fleet Monitor already running"
    pgrep -fa "fleet_monitor.py"
    exit 0
fi

# Start daemon
echo "Starting Fleet Monitor daemon..."
nohup python3 fleet_monitor.py --daemon > /tmp/fleet_monitor.log 2>&1 &
MONITOR_PID=$!

echo "✅ Fleet Monitor started (PID: $MONITOR_PID)"
echo "   Log: /tmp/fleet_monitor.log"
echo "   State: ~/trading_ai/neo/portfolio_state.json"
echo ""
echo "To stop: pkill -f 'fleet_monitor.py --daemon'"
echo "To view: python3 fleet_monitor.py --show"

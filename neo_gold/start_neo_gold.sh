#!/bin/bash
# NEO-GOLD Startup Script

echo "═══════════════════════════════════════════════════════════"
echo "🥇 NEO-GOLD: GOLD TRADING SPECIALIST"
echo "═══════════════════════════════════════════════════════════"

cd /home/jbot/trading_ai/neo_gold

# Check if already running
if pm2 describe neo-gold > /dev/null 2>&1; then
    echo "NEO-GOLD is already running. Restarting..."
    pm2 restart neo-gold
else
    echo "Starting NEO-GOLD..."
    pm2 start run_neo_gold.py --name neo-gold --interpreter python3
fi

echo ""
echo "✅ NEO-GOLD started!"
echo ""
echo "Commands:"
echo "  pm2 logs neo-gold       # View logs"
echo "  pm2 stop neo-gold       # Stop"
echo "  pm2 restart neo-gold    # Restart"
echo ""
echo "Signal endpoint: http://localhost:8085/neo/gold/signal"
echo "Signal file: /tmp/neo_gold_signal.json"
echo ""

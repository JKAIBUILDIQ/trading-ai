#!/bin/bash
# Scout Swarm Runner
# Add to cron:
#   0 6 * * 1-5 /home/jbot/trading_ai/scout_swarm/run_scouts.sh morning
#   0 18 * * 1-5 /home/jbot/trading_ai/scout_swarm/run_scouts.sh evening

cd /home/jbot/trading_ai

# Activate environment if needed
# source venv/bin/activate

# Run scan
if [ "$1" == "morning" ]; then
    echo "🌅 Running MORNING scout scan..."
    python3 scout_swarm/swarm.py morning 2>&1 | tee scout_swarm/logs/morning_$(date +%Y%m%d).log
elif [ "$1" == "evening" ]; then
    echo "🌆 Running EVENING scout scan..."
    python3 scout_swarm/swarm.py evening 2>&1 | tee scout_swarm/logs/evening_$(date +%Y%m%d).log
else
    echo "Usage: run_scouts.sh [morning|evening]"
    exit 1
fi

# Send to Telegram (if configured)
if [ -f scout_swarm/reports/scout_report_$(date +%Y-%m-%d)_$1.txt ]; then
    # Uncomment when Telegram is configured:
    # python3 -c "
    # import requests
    # with open('scout_swarm/reports/scout_report_$(date +%Y-%m-%d)_$1.txt') as f:
    #     msg = f.read()
    # requests.post('https://api.telegram.org/bot\$BOT_TOKEN/sendMessage', 
    #     json={'chat_id': '\$CHAT_ID', 'text': msg[:4000]})
    # "
    echo "📱 Report ready for Telegram"
fi

echo "✅ Scout scan complete"

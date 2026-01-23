#!/bin/bash
# Setup cron job for MQL5 Intel Scraper

echo "Setting up MQL5 Intel cron job..."

# Create cron entry
CRON_CMD="*/15 * * * * /usr/bin/python3 /home/jbot/trading_ai/intel/run_mql5_intel.py >> /home/jbot/trading_ai/intel/logs/cron.log 2>&1"

# Check if already exists
if crontab -l 2>/dev/null | grep -q "run_mql5_intel.py"; then
    echo "Cron job already exists"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job added"
fi

# Show current crontab
echo ""
echo "Current crontab:"
crontab -l

echo ""
echo "MQL5 Intel will run every 15 minutes"
echo "Logs: ~/trading_ai/intel/logs/cron.log"

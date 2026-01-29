#!/bin/bash
# NEO Daily Options Research - Cron Script
# Runs at 8:30 AM ET on weekdays to research BTC miners and recommend options strategies
#
# Cron entry (add with: crontab -e):
# 30 8 * * 1-5 /home/jbot/trading_ai/scripts/run_daily_options_research.sh >> /home/jbot/trading_ai/logs/daily_options_research.log 2>&1

# Exit on error
set -e

# Change to project directory
cd /home/jbot/trading_ai

# Activate conda environment if needed
source /home/jbot/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set environment variables
export TELEGRAM_BOT_TOKEN="8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s"
export TELEGRAM_CHAT_ID="6776619257"
export META_BOT_API="http://127.0.0.1:8035"
export NEO_API="http://127.0.0.1:8036"

# Log start
echo ""
echo "=============================================="
echo "NEO Daily Options Research"
echo "Started: $(date)"
echo "=============================================="

# Run the research script
python3 /home/jbot/trading_ai/neo/daily_options_research.py

# Log completion
echo ""
echo "Completed: $(date)"
echo "=============================================="

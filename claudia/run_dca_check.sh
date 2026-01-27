#!/bin/bash
cd /home/jbot/trading_ai
python3 -m claudia.breakout_dca_system >> /home/jbot/trading_ai/claudia/logs/dca_system.log 2>&1

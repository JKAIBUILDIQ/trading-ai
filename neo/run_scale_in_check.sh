#!/bin/bash
cd /home/jbot/trading_ai
python3 -m neo.iren_scale_in_monitor >> /home/jbot/trading_ai/neo/logs/scale_in.log 2>&1

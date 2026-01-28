#!/bin/bash
cd /home/jbot/trading_ai/neo
python3 -c "
from btc_miners_trader import BTCMinersTrader
trader = BTCMinersTrader()
results = trader.run_daily_check()
import json
print(json.dumps(results, indent=2))
" >> /home/jbot/trading_ai/logs/btc_miners_trades.log 2>&1

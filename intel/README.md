# MQL5 Signal Intel Scraper

Scrapes top trading signals from MQL5.com and detects consensus for NEO.

## Overview

This system:
1. Scrapes top 20 signals from MQL5.com (sorted by growth)
2. Extracts trader stats: growth%, win rate, drawdown
3. Scrapes open positions from top performers
4. Detects **CONSENSUS** when 3+ top traders enter same symbol + direction
5. Writes intel to `/tmp/neo_intel.json` for NEO to read

## Files

```
~/trading_ai/intel/
├── config.py           # Configuration settings
├── mql5_scraper.py     # Main scraper
├── neo_integration.py  # NEO integration helpers
├── run_mql5_intel.py   # Runner script for cron
├── mql5_signals.json   # Full signal data (output)
├── consensus.json      # Consensus signals (output)
├── README.md           # This file
└── logs/               # Scraper logs
```

## Usage

### Test the scraper
```bash
cd ~/trading_ai/intel
python3 mql5_scraper.py --test
```

### Run once
```bash
python3 mql5_scraper.py --once
```

### Run as daemon (continuous)
```bash
python3 mql5_scraper.py --daemon
```

### Set up cron (every 15 minutes)
```bash
crontab -e
# Add:
*/15 * * * * /usr/bin/python3 /home/jbot/trading_ai/intel/run_mql5_intel.py >> /home/jbot/trading_ai/intel/logs/cron.log 2>&1
```

## NEO Integration

NEO reads `/tmp/neo_intel.json` every decision cycle.

```python
from neo_integration import MQL5Intel

intel = MQL5Intel()

# Get consensus signals
consensus = intel.get_consensus_signals()

# Check specific symbol
boost = intel.get_confidence_boost("XAUUSD", "BUY")

# Get formatted text for prompt
intel_text = intel.format_for_neo()
```

## Consensus Detection

A **CONSENSUS SIGNAL** is generated when:
- 3+ top traders (500%+ growth)
- Have open positions in the SAME symbol
- In the SAME direction (BUY or SELL)

This adds +15% confidence boost to NEO's decisions.

## Output Format

### /tmp/neo_intel.json
```json
{
  "timestamp": "2026-01-22T05:00:00Z",
  "source": "MQL5 Top Signals",
  "consensus_signals": [
    {
      "symbol": "XAUUSD",
      "direction": "BUY",
      "confidence": 85,
      "traders": ["Gold Pro", "Kenni Trades", "PM Capital"],
      "trader_count": 3,
      "avg_growth_pct": 2500.0
    }
  ],
  "top_traders_positions": [
    {"trader": "Gold Pro", "symbol": "XAUUSD", "direction": "BUY"}
  ],
  "summary": {
    "total_signals_tracked": 20,
    "consensus_count": 1,
    "confidence_boost": 15
  }
}
```

## Verification

**NO RANDOM DATA** - All data comes from actual MQL5 pages.

### Verify no random imports
```bash
grep -c "random" ~/trading_ai/intel/*.py
# Should return 0
```

### Check data source
```bash
cat /tmp/neo_intel.json | jq '.source'
# Should show "MQL5 Top Signals"
```

### View scraper logs
```bash
tail -f ~/trading_ai/intel/logs/mql5_*.log
```

## Limitations

1. **MQL5 may block scrapers** - If you get 403 errors, you may need:
   - VPN/proxy rotation
   - Browser automation (Selenium/Playwright)
   - MQL5 API (paid)

2. **Position data may be delayed** - MQL5 doesn't always show real-time positions

3. **Rate limiting** - Scraper waits 2 seconds between requests to be respectful

## Alternative Data Sources

If MQL5 blocking becomes an issue:
1. **Myfxbook API** - Has official API
2. **ZuluTrade** - Provides trader positions
3. **eToro Popular Investors** - Public portfolios
4. **Direct MT5 Signal Copier** - Most accurate but requires setup

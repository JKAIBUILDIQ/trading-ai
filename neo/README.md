# NEO - Neural Economic Oracle

Autonomous LLM-based forex trader that learns from outcomes.

**Like AlphaZero learning chess, but forex.**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NEO TRADER                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────┐    ┌─────────┐    ┌─────┐    ┌─────────┐          │
│  │ SEE │───>│  THINK  │───>│ ACT │───>│  LEARN  │          │
│  └─────┘    └─────────┘    └─────┘    └─────────┘          │
│     │           │             │             │               │
│     v           v             v             v               │
│  Market     deepseek      Signals      Memory              │
│   Feed       LLM          Writer       Store               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          v
              ┌─────────────────────┐
              │   Ghost Commander   │
              │   (Executes Trades) │
              └─────────────────────┘
```

## The Loop

1. **SEE** - Gather real market data (prices, positions, P&L)
2. **THINK** - LLM analyzes and forms a trading decision
3. **ACT** - Write signal for Ghost Commander to execute
4. **LEARN** - Review outcomes and extract lessons

## Files

```
neo/
├── neo_trader.py      # Main brain - SEE/THINK/ACT/LEARN loop
├── market_feed.py     # Real prices from MT5/CoinGecko/Frankfurter
├── position_tracker.py # Real positions from MT5
├── memory_store.py    # SQLite learning database
├── signal_writer.py   # Output signals for Ghost Commander
├── config.py          # All settings
├── signals/           # Signal history
├── logs/              # Daily logs
└── neo_memory.db      # Learning database
```

## NO RANDOM DATA

All data comes from real sources:
- **MT5 API** (localhost:8085) - Real trades and positions
- **CoinGecko** - Real crypto prices
- **Frankfurter** - Real forex rates

If an API is unavailable, NEO shows "DATA UNAVAILABLE" - never generates fake data.

## Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max Position | 5% ($4,400) | Single trade risk |
| Max Daily Loss | 3% ($2,640) | Kill switch trigger |
| Max Positions | 4 | Concentration limit |
| Min Confidence | 70% | Only high-confidence trades |

## LLM Configuration

| Role | Model | Use Case |
|------|-------|----------|
| Primary | deepseek-r1:70b | Best reasoning for trade decisions |
| Backup | qwen3:32b | Fallback if primary times out |
| Fast | mistral:7b | Quick decisions when needed |

## Usage

### Run NEO (Continuous)
```bash
cd ~/trading_ai/neo
python3 neo_trader.py
```

### Run One Cycle
```bash
python3 neo_trader.py --once
```

### Run Limited Cycles
```bash
python3 neo_trader.py --cycles 10
```

### Test Components
```bash
# Test market feed
python3 market_feed.py

# Test position tracker
python3 position_tracker.py

# Test memory store
python3 memory_store.py

# Test signal writer
python3 signal_writer.py
```

## Integration with Ghost Commander

NEO writes signals to `/tmp/neo_signal.json`. Ghost Commander should:

1. Watch this file for changes
2. Parse the signal
3. Execute on MT5
4. Report outcome back (for learning)

Signal format:
```json
{
    "timestamp": "2026-01-22T01:30:00Z",
    "signal_id": "NEO_123_013000",
    "action": "OPEN",
    "trade": {
        "symbol": "EURUSD",
        "direction": "BUY",
        "position_value_usd": 2000,
        "stop_loss_pips": 30,
        "take_profit_pips": 60,
        "max_hold_minutes": 180
    },
    "metadata": {
        "confidence": 78,
        "reasoning": "RSI(2) at 8.5, extreme oversold...",
        "model_used": "deepseek-r1:70b"
    }
}
```

## Learning System

NEO maintains a SQLite database with:

- **Decisions** - Every trade decision with reasoning
- **Outcomes** - Win/loss/P&L for each decision
- **Learnings** - Extracted patterns and rules

Over time, NEO builds a knowledge base of what works and what doesn't.

## Proven Parameters

NEO uses parameters from academic research:

| Parameter | Value | Source |
|-----------|-------|--------|
| RSI Period | 2 | Connors Research (88% win rate) |
| RSI Oversold | <10 | Connors Research |
| ATR Period | 20 | Turtle Trading |
| Stop Loss | 2 ATR | Turtle Trading |
| Trend Filter | 200 SMA | Standard |

## Monitoring

Logs are written to:
- Console (real-time)
- `logs/neo_YYYYMMDD.log` (daily file)

View today's log:
```bash
tail -f ~/trading_ai/neo/logs/neo_$(date +%Y%m%d).log
```

---

**⚠️ RISK WARNING**: This is experimental AI trading software. Use at your own risk. Start with paper trading.

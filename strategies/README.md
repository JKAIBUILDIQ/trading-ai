# Proven Trading Strategies Database

## ⚠️ IMPORTANT: NO RANDOM DATA

Every strategy in this database is documented from **real, verifiable sources**:
- Academic papers (SSRN, Journal of Finance, AER)
- Published books (ISBN provided)
- Historical data compilations

## Verification Sources

| Strategy | Primary Source | Verification URL |
|----------|----------------|------------------|
| Turtle Traders | Curtis Faith - Way of the Turtle (2007) | https://www.amazon.com/Way-Turtle-Secret-Methods-Ordinary/dp/007148664X |
| RSI(2) Mean Reversion | Larry Connors - Short Term Trading Strategies | https://www.tradingmarkets.com/ |
| Session Breakouts | Kathy Lien - Day Trading the Currency Market | https://www.amazon.com/Day-Trading-Swing-Currency-Market/dp/1119108411 |
| News Fade | Academic papers + Andersen et al. (2003) | https://www.aeaweb.org/articles?id=10.1257/000282803321455151 |
| Institutional Flow | Lakonishok & Smidt (1988) | Journal of Finance |

## Files

```
strategies/
├── turtle_traders.json      # Original Turtle rules + modern performance
├── mean_reversion.json      # RSI(2), Connors research
├── session_breakouts.json   # London, NY, Asian session strategies
├── news_fade.json           # News event fading with timing
├── institutional_flow.json  # Calendar effects, rebalancing patterns
├── loader.py                # Python utility to query strategies
└── README.md                # This file
```

## Strategy Schema

Each JSON file follows this structure:

```json
{
  "strategy_id": "unique_identifier",
  "name": "Human-readable name",
  "category": "trend_following|mean_reversion|breakout|event_trading|calendar_effects",
  
  "sources": {
    "primary": { "title": "", "author": "", "year": 0, "isbn": "" },
    "secondary": []
  },
  
  "systems": {
    "system_name": {
      "entry_rules": {},
      "exit_rules": {},
      "position_sizing": {}
    }
  },
  
  "backtest_results": {
    "period": "YYYY-YYYY",
    "win_rate": "XX%",
    "profit_factor": X.X,
    "max_drawdown": "XX%"
  },
  
  "still_works": {
    "verdict": true|false,
    "conditions": "..."
  },
  
  "key_parameters": {},
  
  "metadata": {
    "created": "YYYY-MM-DD",
    "data_verified": true,
    "random_data": false
  }
}
```

## Usage

### Python

```python
from loader import StrategyLoader

loader = StrategyLoader()

# Get all strategies
all_strategies = loader.list_strategies()

# Get specific strategy
turtle = loader.get_strategy("turtle_original")

# Get strategies by category
trend_strategies = loader.get_by_category("trend_following")

# Get all strategies that still work
working = loader.get_working_strategies()

# Search for specific parameters
rsi_strategies = loader.search("RSI")
```

### Command Line

```bash
# List all strategies
python loader.py --list

# Get specific strategy
python loader.py --get turtle_original

# Get by category
python loader.py --category mean_reversion

# Search
python loader.py --search "breakout"
```

## Important Disclaimers

1. **Past performance does not guarantee future results**
2. **All backtest results are historical** - actual trading will differ
3. **Strategy edge may have diminished** - many are now widely known
4. **Requires proper implementation** - parameters matter
5. **Risk management is critical** - position sizing not included in win rates

## How to Verify

Each strategy includes source citations. To verify:

1. Check the ISBN and look up the book
2. Visit the URLs provided
3. Search SSRN/Google Scholar for academic papers
4. Cross-reference with multiple sources

## No Random Data Policy

This database follows strict rules:
- ❌ NO `random.choice()`, `random.randint()`, `random.uniform()`
- ❌ NO made-up statistics
- ❌ NO fictional backtest results
- ✅ All data traceable to published sources
- ✅ Every number has a citation
- ✅ "Unknown" used when data unavailable

---

*Last Updated: 2026-01-22*
*Maintainer: Trading AI System*

# Trading AI Infrastructure

**REAL AI Trading System - NO Fake Data**

Built on H100 GPU for training reinforcement learning trading agents on **REAL historical data**.

## 🚨 CRITICAL RULES

1. **NO `random.choice()`, `random.randint()`, `random.uniform()`** for any displayed data
2. Every data point has a **TRACEABLE SOURCE** (API, database, or MT5)
3. Every service has `/health` endpoint with **REAL** connection status
4. All results are **VERIFIABLE** and **REPRODUCIBLE**

## 📁 Project Structure

```
trading_ai/
├── configs/
│   └── settings.py          # API keys, configuration
├── data_pipeline/
│   └── data_sources.py      # Polygon, Twelve Data, MT5, Dukascopy
├── feature_extraction/
│   └── indicators.py        # Technical indicators (RSI, MACD, BB, etc.)
├── backtesting/
│   └── engine.py            # GPU-accelerated backtesting
├── rl_training/
│   ├── trading_env.py       # Gymnasium environment
│   └── train.py             # RL training script
├── signal_generator/
│   └── generator.py         # ML signals → MT5
├── performance_tracker/
│   └── tracker.py           # Real P&L tracking
└── models/                   # Trained model storage
```

## 🔧 Setup

### 1. Install Dependencies

```bash
cd ~/trading_ai
pip install -r requirements.txt

# For GPU acceleration (H100):
pip install cupy-cuda12x
```

### 2. Configure Data Sources

Edit `configs/settings.py` or set environment variables:

```bash
# Market Data APIs (choose one)
export POLYGON_API_KEY="your_key"     # $29/mo stocks, $199/mo forex
export TWELVE_DATA_API_KEY="your_key" # $79/mo forex

# Or use FREE Dukascopy historical data
# Download from: https://www.dukascopy.com/swiss/english/marketwatch/historical/
```

### 3. Verify Configuration

```bash
python configs/settings.py
```

## 📊 Data Sources

| Source | Type | Cost | Data |
|--------|------|------|------|
| **MT5 API** | Real trades | Your bots | Crellastein trade history |
| **Polygon.io** | Real-time + Historical | $29-199/mo | Stocks, Forex, Crypto |
| **Twelve Data** | Real-time + Historical | $79/mo | Forex, Stocks |
| **Dukascopy** | Historical ticks | FREE | Forex (5+ years) |
| **CoinGecko** | Real-time | FREE | Crypto prices |

## 🚀 Usage

### Train a Model

```bash
# Train PPO agent on 1 year of EURUSD data
python rl_training/train.py --algo ppo --timesteps 1000000 --pair EURUSD --days 365

# Train with adversarial MM simulation
python rl_training/train.py --algo ppo --adversarial
```

### Run Backtests

```python
from backtesting.engine import BacktestEngine, ma_crossover_signal
from data_pipeline.data_sources import UnifiedDataManager

# Get REAL data
manager = UnifiedDataManager()
data = manager.get_historical_data("EURUSD", "minute", days=90)

# Run backtest
engine = BacktestEngine()
result = engine.run(data["data"]["bars"], ma_crossover_signal, data["source"])
print(f"Total P&L: ${result.total_pnl:,.2f}")
```

### Generate Live Signals

```python
from signal_generator.generator import SignalGenerator

# Load trained model
gen = SignalGenerator(model_path="models/EURUSD_ppo_final")

# Generate signal from real-time data
signal = gen.generate_signal()
print(f"Action: {signal['signal']['action']}")
```

### Track Performance

```bash
# Start performance API
python -c "from performance_tracker.tracker import PerformanceAPI; PerformanceAPI().run()"

# Access metrics
curl http://localhost:8091/performance/metrics
```

## 🔍 Verification

All services must pass verification:

```bash
# Check for random imports (should be 0)
grep -r "import random" ~/trading_ai/**/*.py | wc -l

# Health endpoints
curl localhost:8090/health        # H100 health monitor
curl localhost:8091/performance/metrics  # Performance tracker
```

## 📈 Training Pipeline

| Week | Task |
|------|------|
| 1-2 | Data collection (Polygon/Dukascopy) |
| 3-4 | Backtesting engine, strategy testing |
| 5-6 | RL training on H100 |
| 7-8 | Integration with MT5 |
| 9+ | Live paper trading, then real capital |

## ⚔️ Anti-MM Tactics

This system is designed to fight back against market makers:

1. **No visible stops** - Mental SL via shadow positions
2. **GTO randomization** - Random entry timing to avoid detection
3. **Multi-bot coordination** - Looks like 6 random traders
4. **Second-mover strategy** - Wait for MM moves, then counter

## 📜 License

Internal use only - Crellastein Trading Systems

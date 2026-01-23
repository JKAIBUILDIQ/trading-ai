# 🔬 RESEARCH SWARM REPORT: Open Source Forex/Trading Algorithms

**Generated:** 2026-01-22
**Query:** Find open source forex algos and GitHub repos for bot algo trading

---

## 🏆 TOP TIER - Production Ready

### 1. DRL Trading Bot (XAUUSD Specialist)
**GitHub:** `github.com/zero-was-here/tradingbot`
**Stars:** Active, MIT Licensed
**Focus:** Gold/XAUUSD Trading

| Feature | Details |
|---------|---------|
| **Algorithm** | PPO + Dreamer (Deep RL) |
| **Framework** | PyTorch |
| **Features** | 140+ multi-timeframe indicators |
| **MT5 Integration** | ✅ Yes - Live trading ready |
| **Target Returns** | 80-120% annual |
| **Training** | 2M+ steps, Colab notebooks |
| **Modes** | Aggressive / Swing / Standard |

**Key Files to Study:**
- `training/ppo_agent.py` - Core RL logic
- `features/technical.py` - 140+ indicator suite
- `live/mt5_connector.py` - MetaTrader bridge

---

### 2. Freqtrade
**GitHub:** `github.com/freqtrade/freqtrade`
**Stars:** 28K+
**Focus:** Crypto (adaptable to forex via CCXT)

| Feature | Details |
|---------|---------|
| **Language** | Python 3.10+ |
| **Backtesting** | Built-in, highly optimized |
| **ML Support** | FreqAI module (RL, classifiers) |
| **Exchanges** | 100+ via CCXT |
| **UI** | Web dashboard included |
| **Community** | Very active Discord |

**Why It's Great:**
- Most mature open source trading bot
- Plugin architecture for custom strategies
- Built-in hyperparameter optimization
- Dry-run mode for paper trading

---

### 3. QuantConnect LEAN
**GitHub:** `github.com/QuantConnect/Lean`
**Stars:** 9K+
**Focus:** Multi-asset (Stocks, Forex, Crypto, Futures)

| Feature | Details |
|---------|---------|
| **Language** | C# / Python |
| **Backtesting** | Institutional grade |
| **Data** | Free historical data included |
| **Live Trading** | Multiple broker integrations |
| **Research** | Jupyter notebook support |

**Pro Tip:** Use their free cloud backtesting at quantconnect.com, then deploy locally

---

### 4. Jesse
**GitHub:** `github.com/jesse-ai/jesse`
**Stars:** 5K+
**Focus:** Crypto, research-focused

| Feature | Details |
|---------|---------|
| **Framework** | Python, async |
| **Optimization** | Genetic algorithms built-in |
| **Indicators** | 200+ technical indicators |
| **Backtesting** | Candle-by-candle simulation |

---

## 🔧 BUILDING BLOCKS / LIBRARIES

### CCXT - Exchange Connectivity
**GitHub:** `github.com/ccxt/ccxt`
**Stars:** 32K+
**Purpose:** Unified API for 100+ exchanges

```python
import ccxt
exchange = ccxt.binance()
ticker = exchange.fetch_ticker('BTC/USDT')
```

### TA-Lib / Pandas-TA
**Purpose:** Technical Analysis Libraries

```python
import pandas_ta as ta
df['RSI'] = ta.rsi(df['close'], length=14)
df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
```

### Backtrader
**GitHub:** `github.com/mementum/backtrader`
**Purpose:** Python backtesting framework

---

## 🤖 AI/ML TRADING RESEARCH

### Trading-R1 (UCLA/Stanford)
**Paper:** "Financial Trading with LLM Reasoning via Reinforcement Learning"
**Approach:** LLM generates trading thesis → RL optimizes execution

**Key Innovation:** 
- Multi-stage curriculum learning
- Combines text reasoning with numerical analysis
- Better risk-adjusted returns than pure RL

### FinRL
**GitHub:** `github.com/AI4Finance-Foundation/FinRL`
**Stars:** 9K+
**Purpose:** Deep RL library for quantitative finance

| Algorithms | DQN, DDPG, PPO, A2C, SAC, TD3 |
|------------|-------------------------------|
| Environments | Stock, Forex, Crypto |
| Data | Yahoo Finance, Alpaca, etc. |

---

## 📋 MQL5 Expert Advisors (MT5 Native)

### EA Sources
1. **MQL5 Code Base:** `mql5.com/en/code`
2. **GitHub Search:** `mql5 expert advisor forex`

### Notable Open Source EAs:
- **Grid Traders** - Martingale with grid entries
- **Trend Followers** - MA crossover systems
- **Scalpers** - High-frequency pip catchers
- **News EAs** - Economic calendar integration

---

## 🎯 RECOMMENDED FOR YOUR USE CASE

Given your **Profit Lock + Jackal** strategy and MT5 setup:

### Option A: Adapt DRL Trading Bot
```
PROS:
✅ Already targets XAUUSD (your specialty)
✅ MT5 integration built-in
✅ Deep RL learns counter-MM tactics
✅ Multiple strategy modes

CONS:
❌ Needs GPU for training
❌ Learning curve for RL
```

### Option B: Build Custom with Freqtrade + FreqAI
```
PROS:
✅ Most mature codebase
✅ FreqAI handles ML complexity
✅ Excellent backtesting
✅ Active community support

CONS:
❌ Crypto-focused (needs adapter for MT5)
❌ Overkill for single-pair trading
```

### Option C: FinRL for Pure Research
```
PROS:
✅ Academic backing
✅ Multiple RL algorithms
✅ Good for learning

CONS:
❌ Research-focused, needs work for production
❌ Less community support
```

---

## 🚀 QUICK START: Clone & Test

### DRL Trading Bot (Gold Specialist)
```bash
cd ~/trading_ai/external
git clone https://github.com/zero-was-here/tradingbot.git drl-gold-bot
cd drl-gold-bot
pip install -r requirements.txt
# Review training config
cat config/training.yaml
```

### Freqtrade
```bash
cd ~/trading_ai/external
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
# Create strategy
freqtrade new-strategy --strategy MyStrategy
```

### FinRL
```bash
cd ~/trading_ai/external
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
pip install -e .
# Run tutorial notebook
jupyter notebook tutorials/
```

---

## 📊 COMPARISON MATRIX

| Repo | Forex? | MT5? | GPU? | Backtest | Live | Difficulty |
|------|--------|------|------|----------|------|------------|
| DRL Bot | ✅ XAUUSD | ✅ | ✅ Required | ✅ | ✅ | Hard |
| Freqtrade | ❌ Crypto | ❌ | Optional | ✅ Best | ✅ | Medium |
| LEAN | ✅ | Via adapter | Optional | ✅ | ✅ | Medium |
| Jesse | ❌ Crypto | ❌ | Optional | ✅ | ✅ | Easy |
| FinRL | ✅ | ❌ | ✅ Recommended | ✅ | ❌ Research | Hard |

---

## 💡 NEO INTEGRATION OPPORTUNITIES

1. **Extract indicator logic** from Freqtrade's 200+ indicators
2. **Port RL training** from DRL Bot to NEO's decision engine
3. **Use LEAN's backtester** for strategy validation
4. **Study Trading-R1** for LLM+RL hybrid approach (matches NEO architecture!)

---

*Research compiled by RONIN001 Research Swarm*
*Sources: GitHub, arXiv, MQL5 Community*

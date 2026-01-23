"""
═══════════════════════════════════════════════════════════════════
NEO BACKTEST: PROFIT LOCK + JACKAL COUNTER STRATEGY V2
═══════════════════════════════════════════════════════════════════

Strategy:
- Enter on technical setups (RSI oversold/overbought, Support/Resistance)
- Target: +$500 profit per position
- NORMALIZED lot sizes so $500 = ~20 pip/point move
- Jackal: 5x opposite when -15 pips underwater

NO RANDOM DATA - Uses real historical prices from Yahoo Finance
═══════════════════════════════════════════════════════════════════
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

# Setup logging
log_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Backtest] %(levelname)s: %(message)s'
)
logger = logging.getLogger("Backtest")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - NORMALIZED POSITION SIZING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentConfig:
    """Configuration for each tradeable instrument"""
    symbol: str
    signal_lots: float      # Base position size
    jackal_lots: float      # 5x counter position
    pip_value: float        # $ per pip per lot
    pip_size: float         # What constitutes 1 pip
    target_pips: float      # Pips needed for $500 profit
    jackal_trigger_pips: float = 15.0  # When to trigger Jackal
    spread_pips: float = 1.0  # Typical spread


INSTRUMENTS = {
    "XAUUSD": InstrumentConfig(
        symbol="XAUUSD",
        signal_lots=0.3,
        jackal_lots=1.5,
        pip_value=10.0,      # $10 per point per lot
        pip_size=0.10,       # Gold moves in 0.10 increments = 1 point
        target_pips=17,      # 17 points * $10/point * 0.3 lots = $51... need to recalc
        spread_pips=3.0      # Gold spread ~3 points
    ),
    "EURUSD": InstrumentConfig(
        symbol="EURUSD",
        signal_lots=2.5,
        jackal_lots=12.5,
        pip_value=10.0,      # $10 per pip per lot
        pip_size=0.0001,
        target_pips=20,      # 20 pips * $10 * 2.5 lots = $500
        spread_pips=1.0
    ),
    "GBPUSD": InstrumentConfig(
        symbol="GBPUSD",
        signal_lots=2.0,
        jackal_lots=10.0,
        pip_value=10.0,
        pip_size=0.0001,
        target_pips=25,      # 25 pips * $10 * 2.0 lots = $500
        spread_pips=1.5
    ),
    "USDJPY": InstrumentConfig(
        symbol="USDJPY",
        signal_lots=2.5,
        jackal_lots=12.5,
        pip_value=6.67,      # ~$6.67 per pip per lot (varies with USDJPY rate)
        pip_size=0.01,
        target_pips=30,      # 30 pips * $6.67 * 2.5 lots = $500
        spread_pips=1.0
    )
}

# Recalculate target pips for exact $500
for symbol, cfg in INSTRUMENTS.items():
    cfg.target_pips = round(500 / (cfg.pip_value * cfg.signal_lots), 1)
    logger.info(f"{symbol}: {cfg.signal_lots} lots, target {cfg.target_pips} pips for $500")


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING - REAL HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════════

def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real historical data from Yahoo Finance.
    Returns OHLCV DataFrame.
    """
    import yfinance as yf
    
    # Yahoo Finance ticker mapping
    yahoo_tickers = {
        "XAUUSD": "GC=F",      # Gold futures
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X", 
        "USDJPY": "USDJPY=X"
    }
    
    ticker = yahoo_tickers.get(symbol, symbol)
    logger.info(f"Fetching {symbol} ({ticker}) from {start_date} to {end_date}...")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.error(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Standardize column names
        col_mapping = {
            'Date': 'Date',
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        }
        
        # Keep only the columns we need
        df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        df['Symbol'] = symbol
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"  Got {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    """Find recent support and resistance levels."""
    recent = df.tail(lookback)
    support = recent['Low'].min()
    resistance = recent['High'].max()
    return support, resistance


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on technical analysis.
    
    Entry conditions:
    - RSI < 30 (oversold) → BUY
    - RSI > 70 (overbought) → SELL
    - Price near support → BUY
    - Price near resistance → SELL
    """
    df = df.copy()
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    # Initialize signal column
    df['Signal'] = 0
    df['Signal_Reason'] = ''
    
    for i in range(50, len(df)):
        # Get recent data for S/R
        support, resistance = find_support_resistance(df.iloc[:i], lookback=20)
        current_price = df.iloc[i]['Close']
        rsi = df.iloc[i]['RSI']
        
        # RSI signals
        if rsi < 30:
            df.iloc[i, df.columns.get_loc('Signal')] = 1  # BUY
            df.iloc[i, df.columns.get_loc('Signal_Reason')] = f'RSI Oversold ({rsi:.1f})'
        elif rsi > 70:
            df.iloc[i, df.columns.get_loc('Signal')] = -1  # SELL
            df.iloc[i, df.columns.get_loc('Signal_Reason')] = f'RSI Overbought ({rsi:.1f})'
        
        # Support/Resistance signals (within 0.5% of level)
        elif current_price < support * 1.005:
            df.iloc[i, df.columns.get_loc('Signal')] = 1  # BUY at support
            df.iloc[i, df.columns.get_loc('Signal_Reason')] = f'Support Test ({support:.2f})'
        elif current_price > resistance * 0.995:
            df.iloc[i, df.columns.get_loc('Signal')] = -1  # SELL at resistance
            df.iloc[i, df.columns.get_loc('Signal_Reason')] = f'Resistance Test ({resistance:.2f})'
    
    return df


# ═══════════════════════════════════════════════════════════════════
# POSITION & TRADE TRACKING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Represents an open position"""
    id: int
    symbol: str
    direction: int          # 1 = long, -1 = short
    entry_price: float
    entry_date: datetime
    lots: float
    is_jackal: bool = False
    parent_id: Optional[int] = None  # For Jackal positions
    
    # Tracking
    max_profit_pips: float = 0.0
    max_loss_pips: float = 0.0
    current_pnl: float = 0.0
    jackal_triggered: bool = False


@dataclass
class Trade:
    """Completed trade record"""
    id: int
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    lots: float
    pnl: float
    pnl_pips: float
    exit_reason: str
    is_jackal: bool
    time_to_target_hours: float


@dataclass 
class BacktestResult:
    """Backtest results for one instrument"""
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_time_to_target_hours: float
    jackal_triggers: int
    jackal_pnl: float
    wins_per_week: float
    trades: List[Trade] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════

class ProfitLockJackalBacktest:
    """
    Backtests the Profit Lock + Jackal Counter strategy.
    """
    
    def __init__(self, starting_balance: float = 100000, max_positions: int = 5):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.equity = starting_balance
        self.max_positions = max_positions
        
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.position_id = 0
        
        # Tracking
        self.equity_curve: List[float] = []
        self.peak_equity = starting_balance
        self.max_drawdown = 0.0
    
    def run(self, df: pd.DataFrame, config: InstrumentConfig) -> BacktestResult:
        """Run backtest on historical data."""
        symbol = config.symbol
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTESTING {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"  Signal lots: {config.signal_lots}")
        logger.info(f"  Jackal lots: {config.jackal_lots}")
        logger.info(f"  Target pips: {config.target_pips}")
        logger.info(f"  Jackal trigger: -{config.jackal_trigger_pips} pips")
        
        self.positions = []
        self.trades = []
        self.balance = self.starting_balance
        self.equity = self.starting_balance
        self.equity_curve = []
        self.peak_equity = self.starting_balance
        self.max_drawdown = 0.0
        
        # Generate signals
        df = generate_signals(df)
        
        # Process each bar
        for i, row in df.iterrows():
            current_price = row['Close']
            current_date = row['Date']
            signal = row['Signal']
            
            # Update existing positions
            self._update_positions(current_price, current_date, config)
            
            # Check for new entries (if we have room)
            if signal != 0 and len(self.positions) < self.max_positions:
                self._open_position(
                    symbol=symbol,
                    direction=signal,
                    price=current_price,
                    date=current_date,
                    lots=config.signal_lots,
                    config=config
                )
            
            # Track equity
            self._update_equity(current_price, config)
        
        # Close any remaining positions at end
        for pos in list(self.positions):
            self._close_position(pos, df.iloc[-1]['Close'], df.iloc[-1]['Date'], "End of backtest", config)
        
        # Generate results
        return self._generate_results(symbol, df)
    
    def _open_position(self, symbol: str, direction: int, price: float, 
                       date: datetime, lots: float, config: InstrumentConfig,
                       is_jackal: bool = False, parent_id: int = None):
        """Open a new position."""
        self.position_id += 1
        
        # Apply spread to entry
        spread_cost = config.spread_pips * config.pip_size
        if direction == 1:  # Long
            entry_price = price + spread_cost
        else:  # Short
            entry_price = price - spread_cost
        
        pos = Position(
            id=self.position_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_date=date,
            lots=lots,
            is_jackal=is_jackal,
            parent_id=parent_id
        )
        
        self.positions.append(pos)
        
        pos_type = "JACKAL" if is_jackal else "SIGNAL"
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.debug(f"  [{pos_type}] {dir_str} {lots} lots @ {entry_price:.5f}")
    
    def _update_positions(self, current_price: float, current_date: datetime, 
                          config: InstrumentConfig):
        """Update all positions, check for TP/SL/Jackal trigger."""
        for pos in list(self.positions):
            # Calculate current P&L in pips
            if pos.direction == 1:  # Long
                pnl_pips = (current_price - pos.entry_price) / config.pip_size
            else:  # Short
                pnl_pips = (pos.entry_price - current_price) / config.pip_size
            
            pnl_dollars = pnl_pips * config.pip_value * pos.lots
            pos.current_pnl = pnl_dollars
            
            # Track max profit/loss
            if pnl_pips > pos.max_profit_pips:
                pos.max_profit_pips = pnl_pips
            if pnl_pips < pos.max_loss_pips:
                pos.max_loss_pips = pnl_pips
            
            # Check for profit target
            target_pips = config.target_pips if not pos.is_jackal else config.target_pips * 0.5
            if pnl_pips >= target_pips:
                self._close_position(pos, current_price, current_date, 
                                   f"Target hit (+{pnl_pips:.1f} pips)", config)
                continue
            
            # Check for Jackal trigger (only for signal positions)
            if not pos.is_jackal and not pos.jackal_triggered:
                if pnl_pips <= -config.jackal_trigger_pips:
                    pos.jackal_triggered = True
                    # Open Jackal counter position
                    self._open_position(
                        symbol=pos.symbol,
                        direction=-pos.direction,  # Opposite direction
                        price=current_price,
                        date=current_date,
                        lots=config.jackal_lots,
                        config=config,
                        is_jackal=True,
                        parent_id=pos.id
                    )
                    logger.debug(f"  [JACKAL TRIGGERED] Position {pos.id} at -{config.jackal_trigger_pips} pips")
            
            # Stop loss for Jackal positions (tighter)
            if pos.is_jackal and pnl_pips <= -10:
                self._close_position(pos, current_price, current_date,
                                   f"Jackal SL (-{abs(pnl_pips):.1f} pips)", config)
            
            # Emergency stop loss for signal positions
            if not pos.is_jackal and pnl_pips <= -50:
                self._close_position(pos, current_price, current_date,
                                   f"Emergency SL (-{abs(pnl_pips):.1f} pips)", config)
    
    def _close_position(self, pos: Position, exit_price: float, exit_date: datetime,
                        reason: str, config: InstrumentConfig):
        """Close a position and record the trade."""
        if pos not in self.positions:
            return
        
        # Calculate final P&L
        if pos.direction == 1:
            pnl_pips = (exit_price - pos.entry_price) / config.pip_size
        else:
            pnl_pips = (pos.entry_price - exit_price) / config.pip_size
        
        pnl_dollars = pnl_pips * config.pip_value * pos.lots
        
        # Calculate time to close
        if isinstance(exit_date, str):
            exit_date = pd.to_datetime(exit_date)
        if isinstance(pos.entry_date, str):
            pos.entry_date = pd.to_datetime(pos.entry_date)
        
        try:
            time_hours = (exit_date - pos.entry_date).total_seconds() / 3600
        except:
            time_hours = 0
        
        # Record trade
        trade = Trade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            lots=pos.lots,
            pnl=pnl_dollars,
            pnl_pips=pnl_pips,
            exit_reason=reason,
            is_jackal=pos.is_jackal,
            time_to_target_hours=time_hours
        )
        
        self.trades.append(trade)
        self.balance += pnl_dollars
        self.positions.remove(pos)
        
        trade_type = "JACKAL" if pos.is_jackal else "SIGNAL"
        logger.debug(f"  [CLOSE {trade_type}] {reason} | P&L: ${pnl_dollars:+.2f} ({pnl_pips:+.1f} pips)")
    
    def _update_equity(self, current_price: float, config: InstrumentConfig):
        """Update equity curve and track drawdown."""
        floating_pnl = sum(pos.current_pnl for pos in self.positions)
        self.equity = self.balance + floating_pnl
        self.equity_curve.append(self.equity)
        
        # Track drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        drawdown = self.peak_equity - self.equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def _generate_results(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """Generate backtest results summary."""
        if not self.trades:
            return BacktestResult(
                symbol=symbol, total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_win=0, avg_loss=0, max_drawdown=0,
                max_drawdown_pct=0, profit_factor=0, avg_time_to_target_hours=0,
                jackal_triggers=0, jackal_pnl=0, wins_per_week=0
            )
        
        # Split trades
        signal_trades = [t for t in self.trades if not t.is_jackal]
        jackal_trades = [t for t in self.trades if t.is_jackal]
        
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        
        # Calculate weeks in backtest
        try:
            start_date = df.iloc[0]['Date']
            end_date = df.iloc[-1]['Date']
            weeks = (end_date - start_date).days / 7
        except:
            weeks = 26  # Default to 6 months
        
        return BacktestResult(
            symbol=symbol,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(self.trades) * 100 if self.trades else 0,
            total_pnl=self.balance - self.starting_balance,
            avg_win=total_wins / len(winning) if winning else 0,
            avg_loss=total_losses / len(losing) if losing else 0,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=(self.max_drawdown / self.starting_balance) * 100,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            avg_time_to_target_hours=np.mean([t.time_to_target_hours for t in winning]) if winning else 0,
            jackal_triggers=len(jackal_trades),
            jackal_pnl=sum(t.pnl for t in jackal_trades),
            wins_per_week=len(winning) / weeks if weeks > 0 else 0,
            trades=self.trades
        )


# ═══════════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_full_backtest():
    """Run backtest on all instruments and generate report."""
    
    START_DATE = "2025-07-01"
    END_DATE = "2026-01-22"
    STARTING_BALANCE = 100000
    
    print("\n" + "═" * 70)
    print("NEO BACKTEST: PROFIT LOCK + JACKAL COUNTER STRATEGY V2")
    print("═" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Starting Balance: ${STARTING_BALANCE:,}")
    print("═" * 70)
    
    results = {}
    
    for symbol, config in INSTRUMENTS.items():
        # Fetch data
        df = fetch_historical_data(symbol, START_DATE, END_DATE)
        
        if df.empty:
            logger.warning(f"Skipping {symbol} - no data")
            continue
        
        # Run backtest
        backtest = ProfitLockJackalBacktest(
            starting_balance=STARTING_BALANCE,
            max_positions=5
        )
        
        result = backtest.run(df, config)
        results[symbol] = result
    
    # Print comparative results
    print("\n" + "═" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("═" * 70)
    
    print(f"\n{'Symbol':<10} {'Trades':<8} {'Win%':<8} {'Total P&L':<15} {'Max DD':<12} {'PF':<8} {'Wins/Wk':<10}")
    print("-" * 70)
    
    for symbol, result in results.items():
        print(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:.1f}%{'':<4} "
              f"${result.total_pnl:>+12,.2f} ${result.max_drawdown:>10,.0f} "
              f"{result.profit_factor:>6.2f} {result.wins_per_week:>8.1f}")
    
    # Detailed analysis
    print("\n" + "═" * 70)
    print("DETAILED ANALYSIS BY INSTRUMENT")
    print("═" * 70)
    
    for symbol, result in results.items():
        config = INSTRUMENTS[symbol]
        print(f"\n🎯 {symbol}")
        print(f"   Position Size: {config.signal_lots} lots (Jackal: {config.jackal_lots} lots)")
        print(f"   Target: {config.target_pips} pips = $500")
        print()
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Winning: {result.winning_trades} | Losing: {result.losing_trades}")
        print(f"   Win Rate: {result.win_rate:.1f}%")
        print()
        print(f"   Total P&L: ${result.total_pnl:+,.2f}")
        print(f"   Avg Win: ${result.avg_win:,.2f}")
        print(f"   Avg Loss: ${result.avg_loss:,.2f}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print()
        print(f"   Max Drawdown: ${result.max_drawdown:,.0f} ({result.max_drawdown_pct:.1f}%)")
        print(f"   Avg Time to Target: {result.avg_time_to_target_hours:.1f} hours")
        print()
        print(f"   Jackal Triggers: {result.jackal_triggers}")
        print(f"   Jackal P&L: ${result.jackal_pnl:+,.2f}")
        print(f"   +$500 Wins per Week: {result.wins_per_week:.1f}")
    
    # NEO's Recommendation
    print("\n" + "═" * 70)
    print("🤖 NEO'S ANALYSIS & RECOMMENDATIONS")
    print("═" * 70)
    
    # Find best performer
    if results:
        best_pf = max(results.items(), key=lambda x: x[1].profit_factor if x[1].profit_factor != float('inf') else 0)
        best_wr = max(results.items(), key=lambda x: x[1].win_rate)
        safest = min(results.items(), key=lambda x: x[1].max_drawdown_pct)
        
        print(f"\n   📊 BEST PROFIT FACTOR: {best_pf[0]} (PF: {best_pf[1].profit_factor:.2f})")
        print(f"   📈 BEST WIN RATE: {best_wr[0]} ({best_wr[1].win_rate:.1f}%)")
        print(f"   🛡️ SAFEST (Lowest DD): {safest[0]} ({safest[1].max_drawdown_pct:.1f}% max DD)")
        
        # Risk analysis
        print("\n   ⚠️ RISK COMPARISON:")
        for symbol, result in results.items():
            config = INSTRUMENTS[symbol]
            margin_required = config.signal_lots * 100000 * 0.01  # Approximate margin
            risk_per_trade = config.jackal_trigger_pips * config.pip_value * config.signal_lots
            print(f"      {symbol}: Margin ~${margin_required:,.0f}, Risk/Trade ~${risk_per_trade:,.0f}")
        
        print("\n   💡 RECOMMENDATION:")
        print(f"      Based on risk-adjusted returns, NEO recommends:")
        print(f"      PRIMARY: {safest[0]} - Best risk profile")
        print(f"      SECONDARY: {best_pf[0]} - Best profit factor")
        print(f"      AVOID: High lot sizes on volatile pairs during news")
    
    # Save results
    output_file = os.path.join(log_dir, "backtest_results.json")
    output = {
        "period": f"{START_DATE} to {END_DATE}",
        "starting_balance": STARTING_BALANCE,
        "results": {
            symbol: {
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "total_pnl": r.total_pnl,
                "max_drawdown": r.max_drawdown,
                "max_drawdown_pct": r.max_drawdown_pct,
                "profit_factor": r.profit_factor if r.profit_factor != float('inf') else 999,
                "wins_per_week": r.wins_per_week,
                "jackal_triggers": r.jackal_triggers,
                "jackal_pnl": r.jackal_pnl,
                "avg_time_to_target_hours": r.avg_time_to_target_hours
            }
            for symbol, r in results.items()
        },
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n   📁 Results saved to: {output_file}")
    print("═" * 70)
    
    return results


if __name__ == "__main__":
    run_full_backtest()

"""
Mean Reversion RSI(2) Strategy
Based on Larry Connors' research

Win Rate: 55-65%
Sharpe: 1.2-1.6
Best For: Range-bound markets, counter-trend entries
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

def calculate_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """
    Calculate RSI with given period
    RSI(2) is more sensitive for mean reversion
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def connors_rsi_strategy(
    ohlcv: pd.DataFrame,
    rsi_period: int = 2,
    oversold: float = 10,
    overbought: float = 90,
    trend_filter_period: int = 200
) -> Dict:
    """
    Classic Connors RSI(2) Mean Reversion Strategy
    
    Rules:
    - LONG: RSI(2) < 10 AND price above 200 EMA (uptrend filter)
    - SHORT: RSI(2) > 90 AND price below 200 EMA (downtrend filter)
    - Exit: RSI(2) > 50 for longs, RSI(2) < 50 for shorts
    
    Returns:
        Signal dictionary with entry/exit conditions
    """
    close = ohlcv['close']
    high = ohlcv['high']
    low = ohlcv['low']
    
    # Calculate indicators
    rsi2 = calculate_rsi(close, rsi_period)
    ema200 = calculate_ema(close, trend_filter_period)
    
    # Current values
    current_close = close.iloc[-1]
    current_rsi = rsi2.iloc[-1]
    current_ema = ema200.iloc[-1]
    
    # Determine trend
    in_uptrend = current_close > current_ema
    in_downtrend = current_close < current_ema
    
    # Calculate recent swing for SL/TP
    recent_low = low.iloc[-20:].min()
    recent_high = high.iloc[-20:].max()
    
    signal = {
        'signal': 'NEUTRAL',
        'confidence': 0,
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'exit_condition': None,
        'reason': [],
        'indicators': {
            'rsi2': current_rsi,
            'ema200': current_ema,
            'trend': 'UPTREND' if in_uptrend else 'DOWNTREND'
        }
    }
    
    # LONG Signal: Oversold in uptrend
    if current_rsi < oversold and in_uptrend:
        signal['signal'] = 'BUY'
        signal['confidence'] = min(80, 50 + (oversold - current_rsi) * 3)
        signal['entry'] = current_close
        signal['stop_loss'] = recent_low - (recent_high - recent_low) * 0.1
        signal['take_profit'] = current_close + (current_close - signal['stop_loss']) * 2
        signal['exit_condition'] = 'RSI(2) > 50'
        signal['reason'] = [
            f"RSI(2) = {current_rsi:.1f} (OVERSOLD)",
            f"Price ${current_close:.2f} above 200 EMA ${current_ema:.2f}",
            "Uptrend confirmed - mean reversion buy"
        ]
    
    # SHORT Signal: Overbought in downtrend
    elif current_rsi > overbought and in_downtrend:
        signal['signal'] = 'SELL'
        signal['confidence'] = min(80, 50 + (current_rsi - overbought) * 3)
        signal['entry'] = current_close
        signal['stop_loss'] = recent_high + (recent_high - recent_low) * 0.1
        signal['take_profit'] = current_close - (signal['stop_loss'] - current_close) * 2
        signal['exit_condition'] = 'RSI(2) < 50'
        signal['reason'] = [
            f"RSI(2) = {current_rsi:.1f} (OVERBOUGHT)",
            f"Price ${current_close:.2f} below 200 EMA ${current_ema:.2f}",
            "Downtrend confirmed - mean reversion sell"
        ]
    
    # Check for EXIT signals
    elif current_rsi > 50 and in_uptrend:
        signal['signal'] = 'EXIT_LONG'
        signal['reason'] = [f"RSI(2) = {current_rsi:.1f} crossed above 50 - exit longs"]
    
    elif current_rsi < 50 and in_downtrend:
        signal['signal'] = 'EXIT_SHORT'
        signal['reason'] = [f"RSI(2) = {current_rsi:.1f} crossed below 50 - exit shorts"]
    
    return signal


def cumulative_rsi_strategy(
    ohlcv: pd.DataFrame,
    rsi_period: int = 2,
    cumulative_days: int = 2,
    threshold: float = 35
) -> Dict:
    """
    Cumulative RSI Strategy (advanced mean reversion)
    
    Entry: Sum of RSI(2) over N days < threshold
    This finds more extreme oversold conditions
    
    Returns:
        Signal dictionary
    """
    close = ohlcv['close']
    
    rsi2 = calculate_rsi(close, rsi_period)
    cumulative_rsi = rsi2.rolling(cumulative_days).sum()
    ema200 = calculate_ema(close, 200)
    
    current_close = close.iloc[-1]
    current_cum_rsi = cumulative_rsi.iloc[-1]
    current_ema = ema200.iloc[-1]
    
    signal = {
        'signal': 'NEUTRAL',
        'confidence': 0,
        'entry': None,
        'reason': [],
        'indicators': {
            'cumulative_rsi': current_cum_rsi,
            'threshold': threshold
        }
    }
    
    # Cumulative RSI < threshold AND above 200 EMA
    if current_cum_rsi < threshold and current_close > current_ema:
        signal['signal'] = 'BUY'
        signal['confidence'] = min(85, 60 + (threshold - current_cum_rsi) * 0.5)
        signal['entry'] = current_close
        signal['reason'] = [
            f"Cumulative RSI(2) over {cumulative_days} days = {current_cum_rsi:.1f}",
            f"Below threshold of {threshold}",
            "Extreme oversold - high probability bounce"
        ]
    
    return signal


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    data = yf.download("GC=F", period="3mo", interval="1d")
    data = data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    # Test Connors RSI
    signal = connors_rsi_strategy(data)
    print("=== Connors RSI(2) Strategy ===")
    print(f"Signal: {signal['signal']}")
    print(f"RSI(2): {signal['indicators']['rsi2']:.1f}")
    print(f"Trend: {signal['indicators']['trend']}")
    
    # Test Cumulative RSI
    cum_signal = cumulative_rsi_strategy(data)
    print("\n=== Cumulative RSI Strategy ===")
    print(f"Signal: {cum_signal['signal']}")
    print(f"Cumulative RSI: {cum_signal['indicators']['cumulative_rsi']:.1f}")

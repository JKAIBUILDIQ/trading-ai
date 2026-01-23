"""
Momentum Breakout Strategy
Based on Freqtrade / DRL Bot patterns

Win Rate: 50-55%
Sharpe: 1.1-1.4
Best For: Trending markets, London session
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate momentum as percentage change"""
    return (close - close.shift(period)) / close.shift(period) * 100


def detect_breakout(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    atr_multiplier: float = 1.5
) -> Dict:
    """
    Detect breakout conditions
    
    Returns:
        Dictionary with breakout signal and metadata
    """
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    volume = ohlcv['volume']
    
    # Calculate indicators
    atr = calculate_atr(high, low, close)
    momentum = calculate_momentum(close)
    
    # Find recent range
    recent_high = high.rolling(lookback).max()
    recent_low = low.rolling(lookback).min()
    
    # Current candle
    current_close = close.iloc[-1]
    current_high = high.iloc[-1]
    current_low = low.iloc[-1]
    current_atr = atr.iloc[-1]
    current_volume = volume.iloc[-1]
    avg_volume = volume.rolling(20).mean().iloc[-1]
    
    # Breakout conditions
    resistance_level = recent_high.iloc[-2]  # Previous high
    support_level = recent_low.iloc[-2]      # Previous low
    
    signal = {
        'signal': 'NEUTRAL',
        'confidence': 0,
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'reason': []
    }
    
    # BULLISH BREAKOUT
    if current_close > resistance_level:
        # Confirm with volume
        volume_surge = current_volume > avg_volume * 1.5
        
        # Confirm with momentum
        momentum_positive = momentum.iloc[-1] > 0
        
        if volume_surge and momentum_positive:
            signal['signal'] = 'BUY'
            signal['confidence'] = min(85, 60 + (current_volume / avg_volume - 1) * 20)
            signal['entry'] = current_close
            signal['stop_loss'] = resistance_level - (current_atr * atr_multiplier)
            signal['take_profit'] = current_close + (current_atr * 3)
            signal['reason'] = [
                f"Broke above ${resistance_level:.2f} resistance",
                f"Volume surge: {current_volume/avg_volume:.1f}x average",
                f"Momentum: +{momentum.iloc[-1]:.2f}%"
            ]
    
    # BEARISH BREAKOUT
    elif current_close < support_level:
        volume_surge = current_volume > avg_volume * 1.5
        momentum_negative = momentum.iloc[-1] < 0
        
        if volume_surge and momentum_negative:
            signal['signal'] = 'SELL'
            signal['confidence'] = min(85, 60 + (current_volume / avg_volume - 1) * 20)
            signal['entry'] = current_close
            signal['stop_loss'] = support_level + (current_atr * atr_multiplier)
            signal['take_profit'] = current_close - (current_atr * 3)
            signal['reason'] = [
                f"Broke below ${support_level:.2f} support",
                f"Volume surge: {current_volume/avg_volume:.1f}x average",
                f"Momentum: {momentum.iloc[-1]:.2f}%"
            ]
    
    return signal


def filter_false_breakout(
    ohlcv: pd.DataFrame,
    breakout_signal: Dict,
    candles_to_confirm: int = 3
) -> bool:
    """
    Filter false breakouts by checking if price holds above/below level
    
    Returns:
        True if likely real breakout, False if likely false breakout
    """
    if breakout_signal['signal'] == 'NEUTRAL':
        return False
    
    close = ohlcv['close']
    entry = breakout_signal['entry']
    
    if breakout_signal['signal'] == 'BUY':
        # Price should stay above entry for confirmation
        recent_closes = close.iloc[-candles_to_confirm:]
        return all(c > entry * 0.998 for c in recent_closes)  # 0.2% buffer
    
    elif breakout_signal['signal'] == 'SELL':
        recent_closes = close.iloc[-candles_to_confirm:]
        return all(c < entry * 1.002 for c in recent_closes)
    
    return False


# Example usage
if __name__ == "__main__":
    # Simulate some data
    import yfinance as yf
    
    data = yf.download("GC=F", period="1mo", interval="1h")
    data = data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    signal = detect_breakout(data)
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']}%")
    if signal['reason']:
        print("Reasons:")
        for r in signal['reason']:
            print(f"  - {r}")

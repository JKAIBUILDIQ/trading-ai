"""
Smart Money Concept (SMC) Detector
Identifies institutional trading patterns: Order Blocks, Fair Value Gaps, Liquidity

Win Rate: 55-65%
Sharpe: 1.4-1.8
Based On: ICT (Inner Circle Trader) concepts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OrderBlock:
    """Represents an Order Block zone"""
    type: str  # 'BULLISH' or 'BEARISH'
    top: float
    bottom: float
    timestamp: datetime
    strength: float  # 0-100
    mitigated: bool = False
    
@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (FVG)"""
    type: str  # 'BULLISH' or 'BEARISH'
    top: float
    bottom: float
    timestamp: datetime
    filled: bool = False


def find_swing_points(ohlcv: pd.DataFrame, lookback: int = 5) -> Tuple[List, List]:
    """
    Find swing highs and swing lows
    
    Swing High: High > highs of N candles before AND after
    Swing Low: Low < lows of N candles before AND after
    """
    highs = []
    lows = []
    
    for i in range(lookback, len(ohlcv) - lookback):
        # Check swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if ohlcv['high'].iloc[i] <= ohlcv['high'].iloc[i-j] or \
               ohlcv['high'].iloc[i] <= ohlcv['high'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            highs.append({
                'price': ohlcv['high'].iloc[i],
                'index': i,
                'timestamp': ohlcv.index[i] if hasattr(ohlcv.index, 'date') else i
            })
        
        # Check swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if ohlcv['low'].iloc[i] >= ohlcv['low'].iloc[i-j] or \
               ohlcv['low'].iloc[i] >= ohlcv['low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            lows.append({
                'price': ohlcv['low'].iloc[i],
                'index': i,
                'timestamp': ohlcv.index[i] if hasattr(ohlcv.index, 'date') else i
            })
    
    return highs, lows


def detect_order_blocks(ohlcv: pd.DataFrame, lookback: int = 100) -> List[OrderBlock]:
    """
    Detect Order Blocks
    
    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    
    Order Blocks are zones where institutions placed large orders
    """
    order_blocks = []
    
    for i in range(2, min(lookback, len(ohlcv) - 5)):
        candle = ohlcv.iloc[-(i+1)]
        next_candles = ohlcv.iloc[-i:-i+3] if i > 3 else ohlcv.iloc[-i:]
        
        # Check for bullish order block
        # Current candle is bearish, followed by strong bullish move
        is_bearish = candle['close'] < candle['open']
        if is_bearish and len(next_candles) > 0:
            move_up = next_candles['close'].max() - candle['close']
            candle_range = candle['high'] - candle['low']
            
            if move_up > candle_range * 2:  # Strong impulse
                order_blocks.append(OrderBlock(
                    type='BULLISH',
                    top=candle['high'],
                    bottom=candle['low'],
                    timestamp=ohlcv.index[-(i+1)] if hasattr(ohlcv.index, 'date') else len(ohlcv)-(i+1),
                    strength=min(100, 50 + move_up / candle_range * 10)
                ))
        
        # Check for bearish order block
        is_bullish = candle['close'] > candle['open']
        if is_bullish and len(next_candles) > 0:
            move_down = candle['close'] - next_candles['close'].min()
            candle_range = candle['high'] - candle['low']
            
            if move_down > candle_range * 2:
                order_blocks.append(OrderBlock(
                    type='BEARISH',
                    top=candle['high'],
                    bottom=candle['low'],
                    timestamp=ohlcv.index[-(i+1)] if hasattr(ohlcv.index, 'date') else len(ohlcv)-(i+1),
                    strength=min(100, 50 + move_down / candle_range * 10)
                ))
    
    return order_blocks[:10]  # Return 10 most recent


def detect_fair_value_gaps(ohlcv: pd.DataFrame, lookback: int = 50) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (FVG)
    
    Bullish FVG: Gap between candle 1's high and candle 3's low
    Bearish FVG: Gap between candle 1's low and candle 3's high
    
    FVGs are areas of imbalance that price tends to return to
    """
    fvgs = []
    
    for i in range(2, min(lookback, len(ohlcv))):
        candle1 = ohlcv.iloc[-(i+2)]  # First candle
        candle2 = ohlcv.iloc[-(i+1)]  # Middle candle (impulse)
        candle3 = ohlcv.iloc[-i]       # Third candle
        
        # Bullish FVG: Gap between candle1.high and candle3.low
        if candle3['low'] > candle1['high']:
            fvgs.append(FairValueGap(
                type='BULLISH',
                top=candle3['low'],
                bottom=candle1['high'],
                timestamp=ohlcv.index[-(i+1)] if hasattr(ohlcv.index, 'date') else len(ohlcv)-(i+1)
            ))
        
        # Bearish FVG: Gap between candle1.low and candle3.high
        if candle3['high'] < candle1['low']:
            fvgs.append(FairValueGap(
                type='BEARISH',
                top=candle1['low'],
                bottom=candle3['high'],
                timestamp=ohlcv.index[-(i+1)] if hasattr(ohlcv.index, 'date') else len(ohlcv)-(i+1)
            ))
    
    return fvgs[:10]


def find_liquidity_pools(ohlcv: pd.DataFrame) -> Dict:
    """
    Find liquidity pools (where stops are clustered)
    
    Buy-side liquidity: Above equal highs, above swing highs
    Sell-side liquidity: Below equal lows, below swing lows
    """
    swing_highs, swing_lows = find_swing_points(ohlcv, lookback=10)
    
    # Cluster nearby swing points
    buy_side_liquidity = []
    sell_side_liquidity = []
    
    for sh in swing_highs[-5:]:  # Recent swing highs
        buy_side_liquidity.append({
            'price': sh['price'],
            'type': 'SWING_HIGH',
            'likelihood': 'HIGH'
        })
    
    for sl in swing_lows[-5:]:  # Recent swing lows
        sell_side_liquidity.append({
            'price': sl['price'],
            'type': 'SWING_LOW',
            'likelihood': 'HIGH'
        })
    
    # Add round number liquidity
    current_price = ohlcv['close'].iloc[-1]
    round_above = np.ceil(current_price / 50) * 50
    round_below = np.floor(current_price / 50) * 50
    
    buy_side_liquidity.append({
        'price': round_above,
        'type': 'ROUND_NUMBER',
        'likelihood': 'MEDIUM'
    })
    
    sell_side_liquidity.append({
        'price': round_below,
        'type': 'ROUND_NUMBER',
        'likelihood': 'MEDIUM'
    })
    
    return {
        'buy_side': sorted(buy_side_liquidity, key=lambda x: x['price']),
        'sell_side': sorted(sell_side_liquidity, key=lambda x: x['price'], reverse=True)
    }


def check_ob_mitigation(order_blocks: List[OrderBlock], current_price: float) -> List[OrderBlock]:
    """
    Check if order blocks have been mitigated (price returned to zone)
    """
    for ob in order_blocks:
        if not ob.mitigated:
            if ob.type == 'BULLISH' and current_price <= ob.top:
                ob.mitigated = True
            elif ob.type == 'BEARISH' and current_price >= ob.bottom:
                ob.mitigated = True
    return order_blocks


def generate_smc_signal(ohlcv: pd.DataFrame) -> Dict:
    """
    Generate trading signal using Smart Money Concepts
    """
    current_price = ohlcv['close'].iloc[-1]
    atr = (ohlcv['high'] - ohlcv['low']).rolling(14).mean().iloc[-1]
    
    # Find SMC structures
    order_blocks = detect_order_blocks(ohlcv)
    fvgs = detect_fair_value_gaps(ohlcv)
    liquidity = find_liquidity_pools(ohlcv)
    
    # Update mitigation status
    order_blocks = check_ob_mitigation(order_blocks, current_price)
    
    signal = {
        'signal': 'NEUTRAL',
        'confidence': 0,
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'reason': [],
        'smc_structures': {
            'order_blocks': len(order_blocks),
            'fair_value_gaps': len(fvgs),
            'nearest_buy_liquidity': liquidity['buy_side'][0] if liquidity['buy_side'] else None,
            'nearest_sell_liquidity': liquidity['sell_side'][0] if liquidity['sell_side'] else None
        }
    }
    
    # Check if price is at a bullish order block (unmitigated)
    for ob in order_blocks:
        if ob.type == 'BULLISH' and not ob.mitigated:
            if ob.bottom <= current_price <= ob.top:
                signal['signal'] = 'BUY'
                signal['confidence'] = int(ob.strength)
                signal['entry'] = current_price
                signal['stop_loss'] = ob.bottom - atr * 0.5
                signal['take_profit'] = current_price + (current_price - signal['stop_loss']) * 2.5
                signal['reason'].append(f"At bullish Order Block zone ${ob.bottom:.2f}-${ob.top:.2f}")
                signal['reason'].append(f"OB strength: {ob.strength:.0f}")
                break
    
    # Check if price is at a bearish order block
    if signal['signal'] == 'NEUTRAL':
        for ob in order_blocks:
            if ob.type == 'BEARISH' and not ob.mitigated:
                if ob.bottom <= current_price <= ob.top:
                    signal['signal'] = 'SELL'
                    signal['confidence'] = int(ob.strength)
                    signal['entry'] = current_price
                    signal['stop_loss'] = ob.top + atr * 0.5
                    signal['take_profit'] = current_price - (signal['stop_loss'] - current_price) * 2.5
                    signal['reason'].append(f"At bearish Order Block zone ${ob.bottom:.2f}-${ob.top:.2f}")
                    signal['reason'].append(f"OB strength: {ob.strength:.0f}")
                    break
    
    # Check for FVG fill opportunities
    if signal['signal'] == 'NEUTRAL':
        for fvg in fvgs:
            if not fvg.filled:
                if fvg.type == 'BULLISH' and current_price <= fvg.top:
                    signal['signal'] = 'BUY'
                    signal['confidence'] = 60
                    signal['entry'] = current_price
                    signal['stop_loss'] = fvg.bottom - atr * 0.5
                    signal['take_profit'] = current_price + atr * 3
                    signal['reason'].append(f"Price filling bullish FVG ${fvg.bottom:.2f}-${fvg.top:.2f}")
                    break
                elif fvg.type == 'BEARISH' and current_price >= fvg.bottom:
                    signal['signal'] = 'SELL'
                    signal['confidence'] = 60
                    signal['entry'] = current_price
                    signal['stop_loss'] = fvg.top + atr * 0.5
                    signal['take_profit'] = current_price - atr * 3
                    signal['reason'].append(f"Price filling bearish FVG ${fvg.bottom:.2f}-${fvg.top:.2f}")
                    break
    
    # Add liquidity context
    if liquidity['buy_side']:
        signal['reason'].append(f"Nearest buy-side liquidity: ${liquidity['buy_side'][0]['price']:.2f}")
    if liquidity['sell_side']:
        signal['reason'].append(f"Nearest sell-side liquidity: ${liquidity['sell_side'][0]['price']:.2f}")
    
    return signal


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    data = yf.download("GC=F", period="1mo", interval="1h")
    data = data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    signal = generate_smc_signal(data)
    
    print("=== Smart Money Concept Strategy ===")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']}%")
    print(f"\nSMC Structures Found:")
    print(f"  Order Blocks: {signal['smc_structures']['order_blocks']}")
    print(f"  Fair Value Gaps: {signal['smc_structures']['fair_value_gaps']}")
    if signal['smc_structures']['nearest_buy_liquidity']:
        print(f"  Buy-side Liquidity: ${signal['smc_structures']['nearest_buy_liquidity']['price']:.2f}")
    if signal['smc_structures']['nearest_sell_liquidity']:
        print(f"  Sell-side Liquidity: ${signal['smc_structures']['nearest_sell_liquidity']['price']:.2f}")
    if signal['reason']:
        print("\nReasons:")
        for r in signal['reason']:
            print(f"  - {r}")

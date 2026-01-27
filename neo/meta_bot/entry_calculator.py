"""
Entry Calculator for Crellastein Meta Bot
Calculates optimal entry zones and take profit targets using indicator confluence
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EntryCalculator")


class EntryCalculator:
    """
    Calculates optimal entry zones and TP targets based on technical confluence
    """
    
    ASSET_CONFIG = {
        "XAUUSD": {
            "ticker": "GC=F",
            "pip_size": 0.01,  # 1 pip = $0.01
            "entry_zone_pips": 15,  # ±15 pips around optimal
            "min_rr_ratio": 1.5,  # Minimum risk:reward
            "atr_multiplier": 0.5,  # Entry zone = 0.5 * ATR
        },
        "IREN": {
            "ticker": "IREN",
            "pip_size": 0.01,
            "entry_zone_pips": 50,  # ±$0.50 around optimal
            "min_rr_ratio": 2.0,
            "atr_multiplier": 0.5,
        }
    }
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.config = self.ASSET_CONFIG.get(symbol, self.ASSET_CONFIG["XAUUSD"])
        self._df = None
        self._last_fetch = None
    
    def _get_data(self, period: str = "30d", interval: str = "1h") -> pd.DataFrame:
        """Fetch market data with caching"""
        now = datetime.now()
        if self._df is not None and self._last_fetch and (now - self._last_fetch).seconds < 300:
            return self._df
        
        try:
            ticker = yf.Ticker(self.config["ticker"])
            self._df = ticker.history(period=period, interval=interval)
            self._last_fetch = now
            return self._df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _calc_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return float(vwap.iloc[-1])
        except:
            return 0
    
    def _calc_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate EMA"""
        try:
            return float(df['Close'].ewm(span=period).mean().iloc[-1])
        except:
            return 0
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            
            return float(tr.rolling(period).mean().iloc[-1])
        except:
            return 0
    
    def _find_swing_low(self, df: pd.DataFrame, periods: int = 20) -> float:
        """Find recent swing low"""
        try:
            return float(df['Low'].tail(periods).min())
        except:
            return 0
    
    def _find_swing_high(self, df: pd.DataFrame, periods: int = 20) -> float:
        """Find recent swing high"""
        try:
            return float(df['High'].tail(periods).max())
        except:
            return 0
    
    def _calc_fib_retracement(self, df: pd.DataFrame, level: float = 0.618) -> float:
        """Calculate Fibonacci retracement level"""
        try:
            high = df['High'].tail(50).max()
            low = df['Low'].tail(50).min()
            # For uptrend, fib retracement is from high
            return float(high - (high - low) * level)
        except:
            return 0
    
    def _find_order_blocks(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find demand (bullish) and supply (bearish) order blocks"""
        try:
            demand_zone = None
            supply_zone = None
            
            # Look for last bullish order block (down candle before big move up)
            for i in range(-20, -1):
                if i >= -len(df):
                    # Check for bullish OB: bearish candle followed by strong bullish move
                    if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Bearish candle
                        # Check if next 3 candles moved up significantly
                        future_high = df['High'].iloc[i+1:min(i+4, -1)].max() if i+4 < 0 else df['High'].iloc[-1]
                        if future_high > df['High'].iloc[i] * 1.005:  # 0.5% move
                            demand_zone = float(df['Low'].iloc[i])
                            break
            
            # Look for last bearish order block (up candle before big move down)
            for i in range(-20, -1):
                if i >= -len(df):
                    if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Bullish candle
                        future_low = df['Low'].iloc[i+1:min(i+4, -1)].min() if i+4 < 0 else df['Low'].iloc[-1]
                        if future_low < df['Low'].iloc[i] * 0.995:
                            supply_zone = float(df['High'].iloc[i])
                            break
            
            return demand_zone, supply_zone
        except:
            return None, None
    
    def _find_fvg(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find Fair Value Gaps (imbalances)"""
        try:
            bullish_fvg = None
            bearish_fvg = None
            
            # Look for bullish FVG (gap up that hasn't been filled)
            for i in range(-15, -2):
                if i >= -len(df):
                    candle1_high = df['High'].iloc[i]
                    candle3_low = df['Low'].iloc[i+2]
                    
                    if candle3_low > candle1_high:  # Gap exists
                        current_price = df['Close'].iloc[-1]
                        if current_price > candle3_low:  # Above the gap
                            bullish_fvg = float(candle3_low)  # Gap acts as support
                            break
            
            # Look for bearish FVG
            for i in range(-15, -2):
                if i >= -len(df):
                    candle1_low = df['Low'].iloc[i]
                    candle3_high = df['High'].iloc[i+2]
                    
                    if candle3_high < candle1_low:  # Gap exists
                        current_price = df['Close'].iloc[-1]
                        if current_price < candle3_high:
                            bearish_fvg = float(candle3_high)  # Gap acts as resistance
                            break
            
            return bullish_fvg, bearish_fvg
        except:
            return None, None
    
    def get_optimal_entry(self) -> Dict:
        """
        Calculate optimal entry zone based on indicator confluence
        """
        df = self._get_data()
        if df.empty:
            return self._default_entry()
        
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate all support levels
        vwap = self._calc_vwap(df)
        ema20 = self._calc_ema(df, 20)
        ema50 = self._calc_ema(df, 50)
        ema200 = self._calc_ema(df, 200)
        atr = self._calc_atr(df)
        swing_low = self._find_swing_low(df, 20)
        fib_618 = self._calc_fib_retracement(df, 0.618)
        fib_50 = self._calc_fib_retracement(df, 0.50)
        demand_zone, supply_zone = self._find_order_blocks(df)
        bullish_fvg, bearish_fvg = self._find_fvg(df)
        
        # Collect valid support levels (below current price = potential entry)
        support_levels = []
        support_names = []
        
        if vwap and vwap < current_price:
            support_levels.append(vwap)
            support_names.append(f"VWAP({vwap:.2f})")
        
        if ema20 and ema20 < current_price:
            support_levels.append(ema20)
            support_names.append(f"EMA20({ema20:.2f})")
        
        if ema50 and ema50 < current_price:
            support_levels.append(ema50)
            support_names.append(f"EMA50({ema50:.2f})")
        
        if swing_low and swing_low < current_price:
            support_levels.append(swing_low)
            support_names.append(f"SwingLow({swing_low:.2f})")
        
        if fib_618 and fib_618 < current_price:
            support_levels.append(fib_618)
            support_names.append(f"Fib618({fib_618:.2f})")
        
        if demand_zone and demand_zone < current_price:
            support_levels.append(demand_zone)
            support_names.append(f"DemandZone({demand_zone:.2f})")
        
        if bullish_fvg and bullish_fvg < current_price:
            support_levels.append(bullish_fvg)
            support_names.append(f"FVG({bullish_fvg:.2f})")
        
        # Calculate optimal entry
        if support_levels:
            # Find nearest cluster of support levels
            support_levels.sort(reverse=True)  # Sort descending (nearest first)
            
            # Weight by proximity to current price
            weights = [1.0 / (i + 1) for i in range(len(support_levels))]
            optimal_entry = np.average(support_levels[:5], weights=weights[:len(support_levels[:5])])
        else:
            # No support found below - use ATR-based entry
            optimal_entry = current_price - (atr * self.config["atr_multiplier"])
        
        # Entry zone boundaries
        zone_size = min(atr * 0.3, self.config["entry_zone_pips"] * self.config["pip_size"])
        entry_zone_high = optimal_entry + zone_size
        entry_zone_low = optimal_entry - zone_size
        
        # Danger zone (trend break)
        if ema200 > 0:
            danger_zone = min(ema200, swing_low if swing_low else ema200) - (atr * 0.5)
        else:
            danger_zone = optimal_entry - (atr * 2)
        
        # Build entry basis explanation
        if support_names:
            entry_basis = f"Confluence: {' + '.join(support_names[:3])}"
        else:
            entry_basis = f"ATR-based entry ({atr:.2f} ATR)"
        
        return {
            "optimal_entry": float(round(optimal_entry, 2)),
            "entry_zone_high": float(round(entry_zone_high, 2)),
            "entry_zone_low": float(round(entry_zone_low, 2)),
            "danger_zone": float(round(danger_zone, 2)),
            "entry_basis": entry_basis,
            "current_price": float(round(current_price, 2)),
            "distance_to_entry_pips": float(round((current_price - optimal_entry) / self.config["pip_size"], 1)),
            "in_entry_zone": bool(entry_zone_low <= current_price <= entry_zone_high),
            "support_levels": [float(round(x, 2)) for x in support_levels[:5]],
        }
    
    def get_target_tp(self, entry_price: float = None) -> Dict:
        """
        Calculate take profit targets based on resistance levels
        """
        df = self._get_data()
        if df.empty:
            return self._default_tp(entry_price or 5000)
        
        current_price = float(df['Close'].iloc[-1])
        entry = entry_price or current_price
        
        # Calculate resistance levels
        vwap = self._calc_vwap(df)
        ema20 = self._calc_ema(df, 20)
        ema50 = self._calc_ema(df, 50)
        atr = self._calc_atr(df)
        swing_high = self._find_swing_high(df, 20)
        fib_382 = self._calc_fib_retracement(df, 0.382)
        _, supply_zone = self._find_order_blocks(df)
        _, bearish_fvg = self._find_fvg(df)
        
        # Collect valid resistance levels (above entry)
        resistance_levels = []
        resistance_names = []
        
        if vwap and vwap > entry:
            resistance_levels.append(vwap)
            resistance_names.append(f"VWAP({vwap:.2f})")
        
        if ema20 and ema20 > entry:
            resistance_levels.append(ema20)
            resistance_names.append(f"EMA20({ema20:.2f})")
        
        if swing_high and swing_high > entry:
            resistance_levels.append(swing_high)
            resistance_names.append(f"SwingHigh({swing_high:.2f})")
        
        if fib_382 and fib_382 > entry:
            resistance_levels.append(fib_382)
            resistance_names.append(f"Fib382({fib_382:.2f})")
        
        if supply_zone and supply_zone > entry:
            resistance_levels.append(supply_zone)
            resistance_names.append(f"SupplyZone({supply_zone:.2f})")
        
        if bearish_fvg and bearish_fvg > entry:
            resistance_levels.append(bearish_fvg)
            resistance_names.append(f"FVG({bearish_fvg:.2f})")
        
        # Calculate TPs
        if resistance_levels:
            resistance_levels.sort()  # Ascending
            target_tp = resistance_levels[0]  # First resistance
            target_tp_2 = resistance_levels[1] if len(resistance_levels) > 1 else target_tp + (atr * 1.5)
            target_tp_3 = resistance_levels[2] if len(resistance_levels) > 2 else target_tp_2 + atr
            tp_basis = f"Resistance: {resistance_names[0]}"
        else:
            # No resistance found - use ATR-based TPs
            target_tp = entry + (atr * 1.0)
            target_tp_2 = entry + (atr * 2.0)
            target_tp_3 = entry + (atr * 3.0)
            tp_basis = f"ATR-based TP ({atr:.2f} ATR)"
        
        # Calculate risk:reward
        entry_data = self.get_optimal_entry()
        risk = entry - entry_data.get("danger_zone", entry - atr)
        reward = target_tp - entry
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            "target_tp": float(round(target_tp, 2)),
            "target_tp_2": float(round(target_tp_2, 2)),
            "target_tp_3": float(round(target_tp_3, 2)),
            "tp_basis": tp_basis,
            "resistance_levels": [float(round(x, 2)) for x in resistance_levels[:5]],
            "risk_reward_ratio": float(round(rr_ratio, 2)),
            "potential_profit_pips": float(round((target_tp - entry) / self.config["pip_size"], 1)),
        }
    
    def get_complete_signal(self) -> Dict:
        """Get complete entry and TP data"""
        entry_data = self.get_optimal_entry()
        tp_data = self.get_target_tp(entry_data["optimal_entry"])
        
        return {
            **entry_data,
            **tp_data,
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _default_entry(self) -> Dict:
        """Default entry when data unavailable"""
        return {
            "optimal_entry": 0,
            "entry_zone_high": 0,
            "entry_zone_low": 0,
            "danger_zone": 0,
            "entry_basis": "Data unavailable",
            "current_price": 0,
            "distance_to_entry_pips": 0,
            "in_entry_zone": False,
            "support_levels": [],
        }
    
    def _default_tp(self, entry: float) -> Dict:
        """Default TP when data unavailable"""
        return {
            "target_tp": entry * 1.01,
            "target_tp_2": entry * 1.02,
            "target_tp_3": entry * 1.03,
            "tp_basis": "Default 1% TP",
            "resistance_levels": [],
            "risk_reward_ratio": 1.0,
            "potential_profit_pips": 100,
        }


# Convenience functions
def get_xauusd_entry() -> Dict:
    calc = EntryCalculator("XAUUSD")
    return calc.get_complete_signal()


def get_iren_entry() -> Dict:
    calc = EntryCalculator("IREN")
    return calc.get_complete_signal()


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 ENTRY CALCULATOR TEST")
    print("=" * 70)
    
    for symbol in ["XAUUSD", "IREN"]:
        print(f"\n📊 {symbol}:")
        calc = EntryCalculator(symbol)
        data = calc.get_complete_signal()
        
        print(f"   Current Price: ${data['current_price']}")
        print(f"   Optimal Entry: ${data['optimal_entry']}")
        print(f"   Entry Zone: ${data['entry_zone_low']} - ${data['entry_zone_high']}")
        print(f"   In Entry Zone: {data['in_entry_zone']}")
        print(f"   Distance: {data['distance_to_entry_pips']} pips")
        print(f"   Danger Zone: ${data['danger_zone']}")
        print(f"   Entry Basis: {data['entry_basis']}")
        print(f"   Target TP1: ${data['target_tp']}")
        print(f"   Target TP2: ${data['target_tp_2']}")
        print(f"   R:R Ratio: {data['risk_reward_ratio']}")

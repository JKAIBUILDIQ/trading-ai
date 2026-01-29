"""
Technical Indicators - Pure Python (No TA-Lib Required)

Matches MT5 indicator calculations for:
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- ADX (Average Directional Index)
- MACD (Moving Average Convergence Divergence)
- SuperTrend

Author: QUINN001
"""

from typing import List, Dict, Tuple, Optional
import math


class Indicators:
    """Technical indicators without external dependencies - matches MT5 calculations"""
    
    # =========================================================================
    # MOVING AVERAGES
    # =========================================================================
    
    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(data) < period:
            return []
        
        result = []
        for i in range(period - 1, len(data)):
            avg = sum(data[i - period + 1:i + 1]) / period
            result.append(avg)
        return result
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """
        Exponential Moving Average
        Matches MT5 iMA(..., MODE_EMA)
        """
        if len(data) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        
        # First EMA is SMA of first 'period' values
        ema_values = [sum(data[:period]) / period]
        
        # Calculate EMA for rest
        for i in range(period, len(data)):
            ema_val = (data[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema_val)
        
        return ema_values
    
    @staticmethod
    def ema_single(data: List[float], period: int) -> float:
        """Get just the current EMA value"""
        ema_vals = Indicators.ema(data, period)
        return ema_vals[-1] if ema_vals else 0
    
    # =========================================================================
    # RSI
    # =========================================================================
    
    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """
        Relative Strength Index
        Matches MT5 iRSI()
        """
        if len(data) < period + 1:
            return []
        
        # Calculate price changes
        changes = [data[i] - data[i-1] for i in range(1, len(data))]
        
        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [max(0, -c) for c in changes]
        
        # First average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # First RSI
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Subsequent RSI values using smoothed averages
        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    @staticmethod
    def rsi_single(data: List[float], period: int = 14) -> float:
        """Get just the current RSI value"""
        rsi_vals = Indicators.rsi(data, period)
        return rsi_vals[-1] if rsi_vals else 50
    
    # =========================================================================
    # ATR (Average True Range)
    # =========================================================================
    
    @staticmethod
    def true_range(candles: List[Dict]) -> List[float]:
        """Calculate True Range for each candle"""
        if len(candles) < 2:
            return []
        
        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        return tr_values
    
    @staticmethod
    def atr(candles: List[Dict], period: int = 14) -> List[float]:
        """
        Average True Range
        Matches MT5 iATR()
        """
        tr_values = Indicators.true_range(candles)
        
        if len(tr_values) < period:
            return []
        
        # First ATR is SMA of TR
        atr_values = [sum(tr_values[:period]) / period]
        
        # Subsequent ATR using smoothed average
        for i in range(period, len(tr_values)):
            atr_val = (atr_values[-1] * (period - 1) + tr_values[i]) / period
            atr_values.append(atr_val)
        
        return atr_values
    
    @staticmethod
    def atr_single(candles: List[Dict], period: int = 14) -> float:
        """Get just the current ATR value"""
        atr_vals = Indicators.atr(candles, period)
        return atr_vals[-1] if atr_vals else 0
    
    # =========================================================================
    # ADX (Average Directional Index)
    # =========================================================================
    
    @staticmethod
    def adx(candles: List[Dict], period: int = 14) -> Dict[str, List[float]]:
        """
        Average Directional Index with +DI and -DI
        Matches MT5 iADX()
        
        Returns: {'adx': [...], 'plus_di': [...], 'minus_di': [...]}
        """
        if len(candles) < period + 1:
            return {'adx': [], 'plus_di': [], 'minus_di': []}
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        tr_values = []
        
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_high = candles[i-1]['high']
            prev_low = candles[i-1]['low']
            prev_close = candles[i-1]['close']
            
            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
            
            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        # Smooth with Wilder's method
        def wilder_smooth(data: List[float], period: int) -> List[float]:
            if len(data) < period:
                return []
            result = [sum(data[:period])]
            for i in range(period, len(data)):
                val = result[-1] - (result[-1] / period) + data[i]
                result.append(val)
            return result
        
        smooth_tr = wilder_smooth(tr_values, period)
        smooth_plus_dm = wilder_smooth(plus_dm, period)
        smooth_minus_dm = wilder_smooth(minus_dm, period)
        
        # Calculate +DI and -DI
        plus_di = []
        minus_di = []
        dx = []
        
        for i in range(len(smooth_tr)):
            if smooth_tr[i] == 0:
                plus_di.append(0)
                minus_di.append(0)
            else:
                plus_di.append(100 * smooth_plus_dm[i] / smooth_tr[i])
                minus_di.append(100 * smooth_minus_dm[i] / smooth_tr[i])
            
            # DX
            di_sum = plus_di[-1] + minus_di[-1]
            if di_sum == 0:
                dx.append(0)
            else:
                dx.append(100 * abs(plus_di[-1] - minus_di[-1]) / di_sum)
        
        # ADX is smoothed DX
        adx_values = wilder_smooth(dx, period) if len(dx) >= period else []
        
        return {
            'adx': adx_values,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def adx_single(candles: List[Dict], period: int = 14) -> float:
        """Get just the current ADX value"""
        result = Indicators.adx(candles, period)
        return result['adx'][-1] if result['adx'] else 0
    
    # =========================================================================
    # MACD
    # =========================================================================
    
    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Dict[str, List[float]]:
        """
        MACD Indicator
        Matches MT5 iMACD()
        
        Returns: {'macd': [...], 'signal': [...], 'histogram': [...]}
        """
        if len(data) < slow + signal:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        ema_fast = Indicators.ema(data, fast)
        ema_slow = Indicators.ema(data, slow)
        
        # Align lengths (slow EMA starts later)
        offset = len(ema_fast) - len(ema_slow)
        ema_fast = ema_fast[offset:]
        
        # MACD line = fast EMA - slow EMA
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        
        # Signal line = EMA of MACD
        signal_line = Indicators.ema(macd_line, signal)
        
        # Align MACD with signal
        offset = len(macd_line) - len(signal_line)
        macd_aligned = macd_line[offset:]
        
        # Histogram
        histogram = [m - s for m, s in zip(macd_aligned, signal_line)]
        
        return {
            'macd': macd_aligned,
            'signal': signal_line,
            'histogram': histogram
        }
    
    # =========================================================================
    # SUPERTREND
    # =========================================================================
    
    @staticmethod
    def supertrend(candles: List[Dict], period: int = 10, 
                   multiplier: float = 3.0) -> Dict[str, any]:
        """
        SuperTrend Indicator
        
        Returns: {
            'trend': 1 (bullish) or -1 (bearish),
            'value': supertrend value,
            'upper_band': upper band,
            'lower_band': lower band
        }
        """
        if len(candles) < period + 1:
            return {'trend': 0, 'value': 0, 'upper_band': 0, 'lower_band': 0}
        
        atr_values = Indicators.atr(candles, period)
        if not atr_values:
            return {'trend': 0, 'value': 0, 'upper_band': 0, 'lower_band': 0}
        
        # Align candles with ATR (ATR starts at index period)
        start_idx = len(candles) - len(atr_values)
        
        upper_band = []
        lower_band = []
        supertrend = []
        trend = []
        
        for i, atr in enumerate(atr_values):
            candle_idx = start_idx + i
            hl2 = (candles[candle_idx]['high'] + candles[candle_idx]['low']) / 2
            
            basic_upper = hl2 + (multiplier * atr)
            basic_lower = hl2 - (multiplier * atr)
            
            if i == 0:
                upper_band.append(basic_upper)
                lower_band.append(basic_lower)
                supertrend.append(basic_lower)
                trend.append(1)
            else:
                prev_close = candles[candle_idx - 1]['close']
                
                # Final upper band
                if basic_upper < upper_band[-1] or prev_close > upper_band[-1]:
                    upper_band.append(basic_upper)
                else:
                    upper_band.append(upper_band[-1])
                
                # Final lower band
                if basic_lower > lower_band[-1] or prev_close < lower_band[-1]:
                    lower_band.append(basic_lower)
                else:
                    lower_band.append(lower_band[-1])
                
                # Trend direction
                close = candles[candle_idx]['close']
                
                if trend[-1] == 1:
                    if close < lower_band[-1]:
                        trend.append(-1)
                        supertrend.append(upper_band[-1])
                    else:
                        trend.append(1)
                        supertrend.append(lower_band[-1])
                else:
                    if close > upper_band[-1]:
                        trend.append(1)
                        supertrend.append(lower_band[-1])
                    else:
                        trend.append(-1)
                        supertrend.append(upper_band[-1])
        
        return {
            'trend': trend[-1] if trend else 0,
            'value': supertrend[-1] if supertrend else 0,
            'upper_band': upper_band[-1] if upper_band else 0,
            'lower_band': lower_band[-1] if lower_band else 0,
            'trend_history': trend,
            'values': supertrend
        }
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def is_death_cross(ema_fast: List[float], ema_slow: List[float]) -> bool:
        """Check for death cross (fast EMA crosses below slow EMA)"""
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return False
        return ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]
    
    @staticmethod
    def is_golden_cross(ema_fast: List[float], ema_slow: List[float]) -> bool:
        """Check for golden cross (fast EMA crosses above slow EMA)"""
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return False
        return ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
    
    @staticmethod
    def get_trend_strength(adx_value: float) -> str:
        """Classify trend strength based on ADX"""
        if adx_value < 20:
            return "WEAK"
        elif adx_value < 25:
            return "MODERATE"
        elif adx_value < 50:
            return "STRONG"
        else:
            return "VERY_STRONG"


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test data
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 
              107, 108, 107, 109, 110, 109, 111, 112, 111, 113]
    
    candles = [
        {'open': c-0.5, 'high': c+1, 'low': c-1, 'close': c}
        for c in closes
    ]
    
    print("EMA(5):", Indicators.ema(closes, 5)[-3:])
    print("RSI(14):", Indicators.rsi_single(closes, 14))
    print("ATR(14):", Indicators.atr_single(candles, 14))
    print("ADX(14):", Indicators.adx_single(candles, 14))
    print("SuperTrend:", Indicators.supertrend(candles, 10, 3))

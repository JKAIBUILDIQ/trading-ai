"""
NEO Asian Retail Intelligence - Counter the Real Herd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHT: MMs don't worry about US traders. The REAL herd is 
India + China - that's who we infiltrate and counter.

India: 100M+ retail accounts, Zerodha = World's largest broker
China: 200M+ retail accounts, massive copy-trade culture

= 300M+ traders using SIMILAR STRATEGIES
= PREDICTABLE HERD BEHAVIOR = OUR EDGE!

Created: 2026-01-24
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
import pandas as pd
import numpy as np
import math

log = logging.getLogger(__name__)


# ============================================================================
# INDICATOR SETTINGS DATABASE (Reverse-Engineered from Asian Retail)
# ============================================================================

INDIAN_RETAIL_SETTINGS = {
    # SUPERTREND - #1 MOST POPULAR in India (90%+ use this!)
    'supertrend': {
        'atr_period': 10,
        'multiplier': 3.0,
        'popularity': 0.90,
        'source': 'Zerodha Streak default'
    },
    
    # RSI - Standard oversold/overbought
    'rsi': {
        'period': 14,
        'oversold': 30,
        'overbought': 70,
        'popularity': 0.85,
        'source': 'Universal retail'
    },
    
    # EMA Crossovers - Very popular in India
    'ema': {
        'fast': 9,
        'slow': 21,
        'trend': 200,
        'scalp_fast': 5,
        'scalp_slow': 13,
        'popularity': 0.75,
        'source': 'Streak templates'
    },
    
    # MACD - Standard settings
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'popularity': 0.60,
        'source': 'Universal'
    },
    
    # Bollinger Bands
    'bollinger': {
        'period': 20,
        'std': 2.0,
        'popularity': 0.55,
        'source': 'Universal'
    },
    
    # VWAP - Popular for intraday
    'vwap': {
        'use_as_support': True,
        'popularity': 0.70,
        'source': 'Zerodha Kite'
    },
    
    # Pivot Points
    'pivots': {
        'type': 'standard',
        'popularity': 0.50,
        'source': 'Universal'
    }
}

CHINESE_RETAIL_SETTINGS = {
    # Moving Averages - China uses different periods!
    'ma': {
        'periods': [5, 10, 20, 60, 120, 250],  # 250 = annual
        'popularity': 0.95,
        'source': '同花顺 default'
    },
    
    # MACD - Same as global
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'popularity': 0.90,
        'source': 'Universal'
    },
    
    # KDJ - Very popular in China (like Stochastic but different)
    'kdj': {
        'k_period': 9,
        'd_period': 3,
        'j_multiplier': 3,  # J = 3D - 2K
        'oversold': 20,
        'overbought': 80,
        'popularity': 0.80,
        'source': '东方财富 default'
    },
    
    # BOLL (Bollinger)
    'boll': {
        'period': 20,
        'std': 2.0,
        'popularity': 0.70,
        'source': 'Universal'
    },
    
    # RSI - China uses 6-period often!
    'rsi': {
        'period': 6,  # Different from West!
        'oversold': 20,
        'overbought': 80,
        'popularity': 0.50,
        'source': 'Chinese retail'
    },
    
    # Round Number Obsession (Cultural)
    'round_numbers': {
        'levels': [50, 100, 500, 1000],  # Multiples
        'popularity': 0.95,
        'source': 'Cultural - lucky numbers'
    }
}

# Festival/Cultural Trading Patterns
INDIAN_CALENDAR = {
    'diwali': {
        'months': [10, 11],  # Oct-Nov
        'behavior': 'BUY_GOLD',
        'reason': 'Auspicious to buy gold for Lakshmi Puja',
        'strength': 0.8
    },
    'akshaya_tritiya': {
        'months': [4, 5],  # Apr-May
        'behavior': 'BUY_GOLD',
        'reason': 'Most auspicious day for gold purchases',
        'strength': 0.9
    },
    'dhanteras': {
        'months': [10, 11],
        'behavior': 'BUY_GOLD',
        'reason': 'Day before Diwali - wealth worship',
        'strength': 0.85
    },
    'month_end': {
        'days': [28, 29, 30, 31, 1],
        'behavior': 'REBALANCE',
        'reason': 'Salary credits, SIP investments',
        'strength': 0.6
    }
}

CHINESE_CALENDAR = {
    'spring_festival': {
        'months': [1, 2],  # Chinese New Year
        'behavior': 'BUY_GOLD',
        'reason': 'Gift-giving, wealth tradition',
        'strength': 0.9
    },
    'golden_week': {
        'months': [10],  # National Day
        'behavior': 'LOW_VOLUME',
        'reason': 'Market holiday',
        'strength': 0.7
    },
    'government_meetings': {
        'months': [3],  # Two Sessions
        'behavior': 'CAUTIOUS',
        'reason': 'Policy uncertainty',
        'strength': 0.6
    }
}


@dataclass
class RetailSignal:
    """A predicted retail signal"""
    indicator: str
    direction: str  # 'BUY', 'SELL'
    strength: float  # 0-1 based on popularity
    trigger_price: Optional[float]
    time_estimate: str  # 'IMMINENT', 'SOON', 'POSSIBLE'
    region: str  # 'INDIA', 'CHINA', 'BOTH'


@dataclass 
class RetailState:
    """Current state of Asian retail herd"""
    herd_direction: str
    herd_strength: float
    stop_clusters: List[float]
    next_entry_triggers: List[Dict]
    exhaustion_probability: float
    recommended_action: str
    counter_confidence: float
    reasoning: str


class IndianRetailSimulator:
    """
    Simulates what 90% of Indian retail algos do
    
    Based on:
    - Zerodha Streak popular strategies
    - Kite Connect open-source bots
    - YouTube India algo tutorials
    - r/IndianStreetBets patterns
    """
    
    def __init__(self):
        self.settings = INDIAN_RETAIL_SETTINGS
        self.calendar = INDIAN_CALENDAR
    
    def calculate_supertrend(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Calculate Supertrend - THE indicator Indian retail uses
        
        90%+ of Zerodha Streak strategies use Supertrend!
        """
        period = self.settings['supertrend']['atr_period']
        multiplier = self.settings['supertrend']['multiplier']
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Basic upper/lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Supertrend calculation
        supertrend = pd.Series(index=ohlcv.index, dtype=float)
        direction = pd.Series(index=ohlcv.index, dtype=int)
        
        for i in range(period, len(ohlcv)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1  # Bullish
                supertrend.iloc[i] = lower_band.iloc[i]
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1  # Bearish
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
        
        return supertrend, direction
    
    def get_entry_signals(self, ohlcv: pd.DataFrame) -> List[RetailSignal]:
        """
        Predict what Indian retail will do
        
        Returns signals ordered by probability
        """
        signals = []
        close = ohlcv['close']
        current_price = close.iloc[-1]
        
        # 1. SUPERTREND (90% of bots)
        try:
            st, st_dir = self.calculate_supertrend(ohlcv)
            st_current = st.iloc[-1]
            st_dir_current = st_dir.iloc[-1]
            st_dir_prev = st_dir.iloc[-2]
            
            # Supertrend flip = MASS ENTRY
            if st_dir_current != st_dir_prev:
                direction = 'BUY' if st_dir_current == 1 else 'SELL'
                signals.append(RetailSignal(
                    indicator='SUPERTREND_FLIP',
                    direction=direction,
                    strength=0.90,
                    trigger_price=current_price,
                    time_estimate='IMMINENT',
                    region='INDIA'
                ))
            
            # About to flip (within 0.5%)
            elif st_dir_current == -1 and current_price > st_current * 0.995:
                signals.append(RetailSignal(
                    indicator='SUPERTREND_FLIP_IMMINENT',
                    direction='BUY',
                    strength=0.85,
                    trigger_price=st_current,
                    time_estimate='SOON',
                    region='INDIA'
                ))
        except Exception as e:
            log.warning(f"Supertrend calculation error: {e}")
        
        # 2. RSI (85% of bots)
        rsi = self._calculate_rsi(close, 14)
        if rsi < 30:
            signals.append(RetailSignal(
                indicator='RSI_OVERSOLD',
                direction='BUY',
                strength=0.85,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='INDIA'
            ))
        elif rsi > 70:
            signals.append(RetailSignal(
                indicator='RSI_OVERBOUGHT',
                direction='SELL',
                strength=0.85,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='INDIA'
            ))
        elif rsi < 35:  # Approaching oversold
            signals.append(RetailSignal(
                indicator='RSI_APPROACHING_OVERSOLD',
                direction='BUY',
                strength=0.70,
                trigger_price=current_price * 0.995,
                time_estimate='SOON',
                region='INDIA'
            ))
        
        # 3. EMA Cross (75% of bots)
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        
        if ema_9.iloc[-1] > ema_21.iloc[-1] and ema_9.iloc[-2] <= ema_21.iloc[-2]:
            signals.append(RetailSignal(
                indicator='EMA_GOLDEN_CROSS',
                direction='BUY',
                strength=0.75,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='INDIA'
            ))
        elif ema_9.iloc[-1] < ema_21.iloc[-1] and ema_9.iloc[-2] >= ema_21.iloc[-2]:
            signals.append(RetailSignal(
                indicator='EMA_DEATH_CROSS',
                direction='SELL',
                strength=0.75,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='INDIA'
            ))
        
        # Sort by strength
        signals.sort(key=lambda x: x.strength, reverse=True)
        return signals
    
    def get_stop_clusters(self, ohlcv: pd.DataFrame, direction: str = 'long') -> List[float]:
        """
        Where Indian retail stops are clustered
        
        Indian traders typically use:
        1. Supertrend line as stop
        2. ATR-based stops (1.5x, 2x)
        3. Round numbers
        4. Recent swing low/high
        """
        clusters = []
        close = ohlcv['close']
        current_price = close.iloc[-1]
        
        # 1. Supertrend line (MOST COMMON!)
        try:
            st, _ = self.calculate_supertrend(ohlcv)
            if not np.isnan(st.iloc[-1]):
                clusters.append(round(st.iloc[-1], 2))
        except:
            pass
        
        # 2. ATR-based stops
        atr = self._calculate_atr(ohlcv, 14)
        if direction == 'long':
            clusters.append(round(current_price - 1.5 * atr, 2))
            clusters.append(round(current_price - 2.0 * atr, 2))
        else:
            clusters.append(round(current_price + 1.5 * atr, 2))
            clusters.append(round(current_price + 2.0 * atr, 2))
        
        # 3. Round numbers
        if direction == 'long':
            round_below = math.floor(current_price / 50) * 50
            clusters.append(round_below)
            clusters.append(round_below - 50)
        else:
            round_above = math.ceil(current_price / 50) * 50
            clusters.append(round_above)
            clusters.append(round_above + 50)
        
        # 4. Recent swing
        if direction == 'long':
            swing_low = ohlcv['low'].tail(20).min()
            clusters.append(round(swing_low - 10, 2))
        else:
            swing_high = ohlcv['high'].tail(20).max()
            clusters.append(round(swing_high + 10, 2))
        
        # Remove duplicates and sort
        clusters = sorted(list(set(clusters)))
        if direction == 'long':
            clusters = sorted(clusters, reverse=True)  # Highest first for longs
        
        return clusters[:10]
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate ATR"""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def get_festival_bias(self) -> Dict:
        """Get current festival/cultural bias for Indian traders"""
        now = datetime.now()
        month = now.month
        day = now.day
        
        for festival, info in self.calendar.items():
            if 'months' in info and month in info['months']:
                return {
                    'event': festival,
                    'behavior': info['behavior'],
                    'reason': info['reason'],
                    'strength': info['strength']
                }
            if 'days' in info and day in info['days']:
                return {
                    'event': festival,
                    'behavior': info['behavior'],
                    'reason': info['reason'],
                    'strength': info['strength']
                }
        
        return {'event': None, 'behavior': 'NEUTRAL', 'strength': 0}


class ChineseRetailSimulator:
    """
    Simulates Chinese retail trading behavior
    
    Based on:
    - 同花顺 (Tonghuashun) popular indicators
    - 东方财富 (East Money) community patterns
    - Weibo finance influencers
    - Shanghai Gold Exchange patterns
    """
    
    def __init__(self):
        self.settings = CHINESE_RETAIL_SETTINGS
        self.calendar = CHINESE_CALENDAR
    
    def calculate_kdj(self, ohlcv: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate KDJ indicator - VERY popular in China
        
        KDJ is like Stochastic but with J line = 3D - 2K
        """
        period = self.settings['kdj']['k_period']
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * d - 2 * k  # Chinese-specific J calculation
        
        return k.iloc[-1], d.iloc[-1], j.iloc[-1]
    
    def get_entry_signals(self, ohlcv: pd.DataFrame) -> List[RetailSignal]:
        """
        Predict what Chinese retail will do
        """
        signals = []
        close = ohlcv['close']
        current_price = close.iloc[-1]
        
        # 1. MA System (95% of Chinese traders)
        ma_5 = close.rolling(5).mean().iloc[-1]
        ma_10 = close.rolling(10).mean().iloc[-1]
        ma_20 = close.rolling(20).mean().iloc[-1]
        
        # Golden cross 5/10
        ma_5_prev = close.rolling(5).mean().iloc[-2]
        ma_10_prev = close.rolling(10).mean().iloc[-2]
        
        if ma_5 > ma_10 and ma_5_prev <= ma_10_prev:
            signals.append(RetailSignal(
                indicator='MA_5_10_CROSS',
                direction='BUY',
                strength=0.95,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        
        # Price above all MAs (strong buy)
        if current_price > ma_5 > ma_10 > ma_20:
            signals.append(RetailSignal(
                indicator='MA_BULLISH_ALIGNMENT',
                direction='BUY',
                strength=0.80,
                trigger_price=current_price,
                time_estimate='ACTIVE',
                region='CHINA'
            ))
        
        # 2. KDJ Indicator (80% of Chinese traders)
        k, d, j = self.calculate_kdj(ohlcv)
        
        if k < 20 and d < 20:
            signals.append(RetailSignal(
                indicator='KDJ_OVERSOLD',
                direction='BUY',
                strength=0.80,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        elif k > 80 and d > 80:
            signals.append(RetailSignal(
                indicator='KDJ_OVERBOUGHT',
                direction='SELL',
                strength=0.80,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        
        # J line extreme (unique to China)
        if j < 0:  # Very oversold
            signals.append(RetailSignal(
                indicator='KDJ_J_EXTREME_LOW',
                direction='BUY',
                strength=0.85,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        elif j > 100:  # Very overbought
            signals.append(RetailSignal(
                indicator='KDJ_J_EXTREME_HIGH',
                direction='SELL',
                strength=0.85,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        
        # 3. Round number obsession (95% - cultural)
        for mult in [100, 500, 1000]:
            nearest_round = round(current_price / mult) * mult
            if abs(current_price - nearest_round) / current_price < 0.01:  # Within 1%
                signals.append(RetailSignal(
                    indicator=f'ROUND_NUMBER_{nearest_round}',
                    direction='WATCH',
                    strength=0.95,
                    trigger_price=nearest_round,
                    time_estimate='AT_LEVEL',
                    region='CHINA'
                ))
        
        # 4. MACD (90% of Chinese traders)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signals.append(RetailSignal(
                indicator='MACD_CROSS_UP',
                direction='BUY',
                strength=0.90,
                trigger_price=current_price,
                time_estimate='IMMINENT',
                region='CHINA'
            ))
        
        signals.sort(key=lambda x: x.strength, reverse=True)
        return signals
    
    def get_session_bias(self) -> Dict:
        """
        Get Shanghai trading session bias
        
        Chinese traders are VERY session-focused:
        - Shanghai open: 9:00 AM CST
        - Lunch break: 11:30-13:00
        - Shanghai close: 15:00
        """
        now = datetime.utcnow()
        shanghai_hour = (now.hour + 8) % 24  # UTC+8
        
        if 9 <= shanghai_hour < 10:
            return {
                'session': 'SHANGHAI_OPEN',
                'behavior': 'HIGH_VOLUME',
                'bias': 'FOLLOW_OVERNIGHT_MOVE',
                'strength': 0.8
            }
        elif 14 <= shanghai_hour < 15:
            return {
                'session': 'SHANGHAI_CLOSE',
                'behavior': 'POSITION_SQUARING',
                'bias': 'REVERSAL_RISK',
                'strength': 0.7
            }
        elif 11 <= shanghai_hour < 13:
            return {
                'session': 'LUNCH_BREAK',
                'behavior': 'LOW_VOLUME',
                'bias': 'NEUTRAL',
                'strength': 0.5
            }
        
        return {
            'session': 'NORMAL',
            'behavior': 'STANDARD',
            'bias': 'NEUTRAL',
            'strength': 0.5
        }


class AsianRetailIntelligence:
    """
    Combined Asian Retail Intelligence for NEO
    
    Aggregates Indian and Chinese retail behavior to:
    1. Predict herd movements
    2. Identify stop clusters
    3. Front-run retail entries
    4. Fade exhaustion
    """
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.india = IndianRetailSimulator()
        self.china = ChineseRetailSimulator()
    
    def analyze(self, ohlcv: pd.DataFrame) -> RetailState:
        """
        Complete analysis of Asian retail positioning
        """
        ohlcv.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in ohlcv.columns]
        
        current_price = ohlcv['close'].iloc[-1]
        
        # Get signals from both regions
        india_signals = self.india.get_entry_signals(ohlcv)
        china_signals = self.china.get_entry_signals(ohlcv)
        
        # Combine signals
        all_signals = india_signals + china_signals
        
        # Calculate herd direction
        buy_strength = sum(s.strength for s in all_signals if s.direction == 'BUY')
        sell_strength = sum(s.strength for s in all_signals if s.direction == 'SELL')
        
        total_strength = buy_strength + sell_strength
        if total_strength > 0:
            herd_strength = abs(buy_strength - sell_strength) / total_strength
        else:
            herd_strength = 0
        
        herd_direction = 'LONG' if buy_strength > sell_strength else 'SHORT' if sell_strength > buy_strength else 'NEUTRAL'
        
        # Get stop clusters
        stop_clusters = self.india.get_stop_clusters(
            ohlcv, 
            'long' if herd_direction == 'LONG' else 'short'
        )
        
        # Detect exhaustion
        exhaustion_prob = self._detect_exhaustion(ohlcv, all_signals, herd_direction)
        
        # Generate recommendation
        recommended_action, counter_confidence, reasoning = self._generate_recommendation(
            herd_direction, herd_strength, exhaustion_prob, all_signals, stop_clusters, current_price
        )
        
        # Build next entry triggers
        next_triggers = self._get_next_triggers(all_signals)
        
        return RetailState(
            herd_direction=herd_direction,
            herd_strength=round(herd_strength, 2),
            stop_clusters=stop_clusters,
            next_entry_triggers=next_triggers,
            exhaustion_probability=round(exhaustion_prob, 2),
            recommended_action=recommended_action,
            counter_confidence=round(counter_confidence, 2),
            reasoning=reasoning
        )
    
    def _detect_exhaustion(self, ohlcv: pd.DataFrame, signals: List[RetailSignal],
                           herd_direction: str) -> float:
        """
        Detect if the Asian retail herd is exhausted
        """
        exhaustion = 0.0
        
        # Many signals but declining momentum
        direction_signals = [s for s in signals if 
                          (s.direction == 'BUY' and herd_direction == 'LONG') or
                          (s.direction == 'SELL' and herd_direction == 'SHORT')]
        
        if len(direction_signals) >= 4:
            exhaustion += 0.2  # Too many agree = crowded
        
        # Volume check
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([1]*len(ohlcv))
        vol_ma_5 = volume.tail(5).mean()
        vol_ma_20 = volume.tail(20).mean()
        
        if vol_ma_5 < vol_ma_20 * 0.8:  # Volume declining
            exhaustion += 0.3
        
        # Price extended from MA
        close = ohlcv['close']
        ma_20 = close.rolling(20).mean().iloc[-1]
        extension = abs(close.iloc[-1] - ma_20) / ma_20
        
        if extension > 0.05:  # 5%+ extended
            exhaustion += 0.2
        if extension > 0.10:  # 10%+ extended
            exhaustion += 0.2
        
        # RSI extreme
        rsi = self.india._calculate_rsi(close, 14)
        if rsi > 75 or rsi < 25:
            exhaustion += 0.1
        
        return min(1.0, exhaustion)
    
    def _generate_recommendation(self, direction: str, strength: float,
                                  exhaustion: float, signals: List[RetailSignal],
                                  stop_clusters: List[float], price: float) -> Tuple[str, float, str]:
        """
        Generate trading recommendation based on retail analysis
        """
        
        # High strength + High exhaustion = FADE opportunity
        if strength > 0.7 and exhaustion > 0.6:
            fade_direction = 'SELL' if direction == 'LONG' else 'BUY'
            return (
                f'FADE_{fade_direction}',
                0.75,
                f"Herd is {direction} ({strength:.0%}) but EXHAUSTED ({exhaustion:.0%}). "
                f"300M+ Asian traders already positioned - fade the crowd!"
            )
        
        # Approaching stop cluster = WAIT for stop hunt
        if stop_clusters:
            nearest_stop = stop_clusters[0]
            distance = abs(price - nearest_stop) / price
            if distance < 0.01:  # Within 1%
                return (
                    'WAIT_FOR_STOP_HUNT',
                    0.70,
                    f"Price near stop cluster at ${nearest_stop:.0f}. "
                    "Wait for MMs to sweep stops, then reverse."
                )
        
        # Strong signals imminent = FRONT RUN
        imminent = [s for s in signals if s.time_estimate == 'IMMINENT']
        if imminent:
            top_signal = max(imminent, key=lambda x: x.strength)
            return (
                f'FRONT_RUN_{top_signal.direction}',
                top_signal.strength * 0.9,
                f"{top_signal.indicator} about to trigger for {top_signal.region} retail. "
                f"Enter before the {strength*100:.0f}% of 300M traders!"
            )
        
        # Moderate strength = RIDE with herd
        if strength > 0.5 and exhaustion < 0.4:
            return (
                f'RIDE_{direction}',
                0.60,
                f"Herd is {direction} ({strength:.0%}) with room to run. "
                "Join the trend but trail stops tight."
            )
        
        # Default = WAIT
        return (
            'WAIT',
            0.30,
            f"No clear edge. Herd strength {strength:.0%}, exhaustion {exhaustion:.0%}. "
            "Wait for better setup."
        )
    
    def _get_next_triggers(self, signals: List[RetailSignal]) -> List[Dict]:
        """Get upcoming trigger points"""
        triggers = []
        
        for sig in signals[:5]:  # Top 5
            if sig.trigger_price:
                triggers.append({
                    'indicator': sig.indicator,
                    'direction': sig.direction,
                    'trigger_price': sig.trigger_price,
                    'timing': sig.time_estimate,
                    'region': sig.region,
                    'strength': sig.strength
                })
        
        return triggers
    
    def get_supertrend_level(self, ohlcv: pd.DataFrame) -> float:
        """Get current Supertrend level (where Indian stops are)"""
        st, _ = self.india.calculate_supertrend(ohlcv)
        return st.iloc[-1] if not np.isnan(st.iloc[-1]) else 0
    
    def get_session_context(self) -> Dict:
        """Get current Asian session context"""
        india_festival = self.india.get_festival_bias()
        china_session = self.china.get_session_bias()
        
        return {
            'india': india_festival,
            'china': china_session,
            'combined_bias': self._combine_biases(india_festival, china_session)
        }
    
    def _combine_biases(self, india: Dict, china: Dict) -> str:
        """Combine Indian and Chinese biases"""
        if india.get('behavior') == 'BUY_GOLD' and india.get('strength', 0) > 0.7:
            return 'BULLISH_INDIA_FESTIVAL'
        if china.get('behavior') == 'HIGH_VOLUME':
            return 'WATCH_SHANGHAI_OPEN'
        return 'NEUTRAL'


def get_asian_retail_analysis(ohlcv: pd.DataFrame, symbol: str = 'XAUUSD') -> Dict:
    """Quick function for NEO integration"""
    intel = AsianRetailIntelligence(symbol)
    state = intel.analyze(ohlcv)
    session = intel.get_session_context()
    
    return {
        'available': True,
        'herd_direction': state.herd_direction,
        'herd_strength': state.herd_strength,
        'stop_clusters': state.stop_clusters,
        'next_triggers': state.next_entry_triggers,
        'exhaustion_probability': state.exhaustion_probability,
        'recommended_action': state.recommended_action,
        'counter_confidence': state.counter_confidence,
        'reasoning': state.reasoning,
        'session_context': session,
        'supertrend_level': intel.get_supertrend_level(ohlcv)
    }


# Test
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*70)
    print("🇮🇳🇨🇳 ASIAN RETAIL INTELLIGENCE TEST")
    print("="*70)
    
    # Get Gold data
    gold = yf.download('GC=F', period='3mo', interval='1d', progress=False)
    
    if hasattr(gold.columns, 'levels'):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]
    
    intel = AsianRetailIntelligence('XAUUSD')
    state = intel.analyze(gold)
    
    print(f"\n📊 HERD STATE:")
    print(f"   Direction: {state.herd_direction}")
    print(f"   Strength: {state.herd_strength:.0%}")
    print(f"   Exhaustion: {state.exhaustion_probability:.0%}")
    
    print(f"\n🎯 RECOMMENDATION: {state.recommended_action}")
    print(f"   Confidence: {state.counter_confidence:.0%}")
    print(f"   {state.reasoning}")
    
    print(f"\n🛑 STOP CLUSTERS (where 300M traders have stops):")
    for stop in state.stop_clusters[:5]:
        print(f"   ${stop:.2f}")
    
    print(f"\n⏰ NEXT TRIGGERS:")
    for trigger in state.next_entry_triggers[:3]:
        print(f"   {trigger['indicator']}: {trigger['direction']} @ ${trigger.get('trigger_price', 'N/A')}")
        print(f"      Timing: {trigger['timing']} | Region: {trigger['region']} | Strength: {trigger['strength']:.0%}")
    
    # Session context
    session = intel.get_session_context()
    print(f"\n🌏 SESSION CONTEXT:")
    print(f"   India: {session['india']}")
    print(f"   China: {session['china']}")
    print(f"   Combined: {session['combined_bias']}")
    
    # Supertrend level
    st_level = intel.get_supertrend_level(gold)
    print(f"\n📈 SUPERTREND LEVEL (Indian stop zone): ${st_level:.2f}")

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
XAUUSD 12-MONTH PATTERN ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Backtest Gold behavior across 6 categories:
1. Day of Week Patterns
2. Session Patterns (Asian, London, NY)
3. Technical Indicator Patterns
4. Price Level Patterns
5. Volatility Regime Patterns
6. News & Event Patterns

Output: Pattern database + Weekly prediction model for NEO

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XAUUSD-Backtest")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pattern:
    """A trading pattern with statistics"""
    id: str
    category: str
    description: str
    conditions: List[str]
    action: str
    win_rate: float
    avg_profit_pts: float
    avg_loss_pts: float
    sample_size: int
    confidence: float
    best_entry_hour: int = None
    best_exit_hours: int = None
    optimal_tp: float = None
    optimal_sl: float = None


@dataclass  
class DailyPrediction:
    """Daily trading prediction based on patterns"""
    date: str
    day_of_week: str
    bias: str  # BULLISH, BEARISH, NEUTRAL
    strategy: str  # TREND_FOLLOW, MEAN_REVERT, BREAKOUT
    expected_range: float
    confidence: float
    key_levels: List[float]
    session_predictions: Dict[str, Dict]
    pattern_matches: List[Dict]
    avoid_windows: List[str]
    notes: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

class XAUUSDDataCollector:
    """Collect and prepare XAUUSD historical data"""
    
    def __init__(self, months: int = 12):
        self.months = months
        self.h1_data: pd.DataFrame = None
        self.d1_data: pd.DataFrame = None
        
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch 12 months of XAUUSD data from Yahoo Finance"""
        logger.info(f"📊 Fetching {self.months} months of XAUUSD data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months * 30)
        
        # Gold futures (GC=F) as proxy for XAUUSD
        ticker = yf.Ticker("GC=F")
        
        # Fetch H1 data (1 hour)
        logger.info("   Fetching H1 data...")
        self.h1_data = ticker.history(
            start=start_date,
            end=end_date,
            interval="1h"
        )
        
        # Fetch D1 data (daily)
        logger.info("   Fetching D1 data...")
        self.d1_data = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d"
        )
        
        # Clean and prepare data
        self.h1_data = self._prepare_data(self.h1_data, "H1")
        self.d1_data = self._prepare_data(self.d1_data, "D1")
        
        logger.info(f"   ✅ H1: {len(self.h1_data)} candles")
        logger.info(f"   ✅ D1: {len(self.d1_data)} candles")
        
        return self.h1_data, self.d1_data
    
    def _prepare_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Clean and enrich data with metadata"""
        if df.empty:
            return df
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Add time-based features
        df['timestamp'] = df.index
        df['day_of_week'] = df.index.dayofweek  # 0=Monday
        df['day_name'] = df.index.day_name()
        df['hour_utc'] = df.index.hour
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        
        # Add session classification
        df['session'] = df['hour_utc'].apply(self._classify_session)
        
        # Add price-based features
        df['range'] = df['high'] - df['low']
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open'] * 100
        df['is_bullish'] = df['close'] > df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Previous candle reference
        df['prev_close'] = df['close'].shift(1)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['gap'] = df['open'] - df['prev_close']
        
        return df
    
    def _classify_session(self, hour: int) -> str:
        """Classify trading session based on hour (UTC)"""
        if 0 <= hour < 8:
            return "ASIAN"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 16:
            return "OVERLAP"  # London-NY overlap
        elif 16 <= hour < 21:
            return "NY"
        else:
            return "ASIAN"  # Late night


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalIndicators:
    """Calculate all technical indicators needed for pattern analysis"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = TechnicalIndicators.add_rsi(df, 14)
        df = TechnicalIndicators.add_rsi(df, 2)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df, 14)
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df[f'rsi_{period}_overbought'] = df[f'rsi_{period}'] > 70
        df[f'rsi_{period}_oversold'] = df[f'rsi_{period}'] < 30
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD signals
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB touches
        df['bb_upper_touch'] = df['high'] >= df['bb_upper']
        df['bb_lower_touch'] = df['low'] <= df['bb_lower']
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        # ATR percentile (volatility regime)
        df['atr_percentile'] = df['atr'].rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # Volatility regime
        df['vol_regime'] = pd.cut(
            df['atr_percentile'],
            bins=[0, 25, 75, 100],
            labels=['LOW', 'NORMAL', 'HIGH']
        )
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        for period in [20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # MA crossovers
        df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
        
        # Price vs MAs
        df['above_sma_50'] = df['close'] > df['sma_50']
        df['above_sma_200'] = df['close'] > df['sma_200']
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators"""
        if 'volume' not in df.columns or df['volume'].isna().all():
            df['volume'] = 1000000  # Default if no volume data
        
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_spike'] = df['volume_ratio'] > 2.0
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: DAY OF WEEK PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class DayOfWeekAnalyzer:
    """Analyze patterns by day of week"""
    
    def __init__(self, h1_data: pd.DataFrame, d1_data: pd.DataFrame):
        self.h1 = h1_data
        self.d1 = d1_data
        self.results = {}
    
    def analyze(self) -> Dict:
        """Run complete day of week analysis"""
        logger.info("📅 Analyzing Day of Week patterns...")
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        for day in days:
            day_data = self.d1[self.d1['day_name'] == day]
            
            if len(day_data) < 10:
                continue
            
            self.results[day] = {
                'sample_size': len(day_data),
                'avg_range': day_data['range'].mean(),
                'avg_body': day_data['body'].mean(),
                'bullish_pct': (day_data['is_bullish'].sum() / len(day_data)) * 100,
                'avg_high_from_open': (day_data['high'] - day_data['open']).mean(),
                'avg_low_from_open': (day_data['open'] - day_data['low']).mean(),
                'gap_fill_rate': self._calculate_gap_fill_rate(day_data),
                'trend_continuation_rate': self._calculate_trend_continuation(day_data),
                'best_entry_hour': self._find_best_entry_hour(day),
                'patterns': []
            }
            
            # Find specific patterns
            self._find_day_patterns(day, day_data)
        
        return self.results
    
    def _calculate_gap_fill_rate(self, day_data: pd.DataFrame) -> float:
        """Calculate how often gaps get filled"""
        gaps = day_data[day_data['gap'].abs() > 5]  # Significant gaps (>$5)
        if len(gaps) == 0:
            return 0
        
        filled = 0
        for idx, row in gaps.iterrows():
            if row['gap'] > 0:  # Gap up
                if row['low'] <= row['prev_close']:
                    filled += 1
            else:  # Gap down
                if row['high'] >= row['prev_close']:
                    filled += 1
        
        return (filled / len(gaps)) * 100 if len(gaps) > 0 else 0
    
    def _calculate_trend_continuation(self, day_data: pd.DataFrame) -> float:
        """Calculate trend continuation from previous day"""
        if len(day_data) < 2:
            return 50
        
        # Compare direction with previous day
        continued = 0
        for i in range(1, len(day_data)):
            prev_bullish = day_data.iloc[i-1]['is_bullish']
            curr_bullish = day_data.iloc[i]['is_bullish']
            if prev_bullish == curr_bullish:
                continued += 1
        
        return (continued / (len(day_data) - 1)) * 100
    
    def _find_best_entry_hour(self, day_name: str) -> int:
        """Find the hour with best risk-reward for entries"""
        day_h1 = self.h1[self.h1['day_name'] == day_name]
        
        hourly_stats = []
        for hour in range(24):
            hour_data = day_h1[day_h1['hour_utc'] == hour]
            if len(hour_data) < 5:
                continue
            
            # Calculate forward returns (next 4 hours)
            hour_data = hour_data.copy()
            hour_data['fwd_return'] = hour_data['close'].shift(-4) - hour_data['close']
            
            avg_return = hour_data['fwd_return'].mean()
            win_rate = (hour_data['fwd_return'] > 0).mean() * 100
            
            hourly_stats.append({
                'hour': hour,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'score': win_rate * abs(avg_return) if not np.isnan(avg_return) else 0
            })
        
        if not hourly_stats:
            return 8  # Default to London open
        
        best = max(hourly_stats, key=lambda x: x['score'])
        return best['hour']
    
    def _find_day_patterns(self, day: str, day_data: pd.DataFrame):
        """Find specific patterns for this day"""
        patterns = []
        
        # Monday gap fill pattern
        if day == 'Monday':
            gap_data = day_data[day_data['gap'].abs() > 10]
            if len(gap_data) >= 10:
                gap_fill = self._calculate_gap_fill_rate(gap_data)
                if gap_fill > 55:
                    patterns.append({
                        'id': 'monday_gap_fill',
                        'description': 'Monday gaps tend to fill',
                        'win_rate': gap_fill,
                        'sample_size': len(gap_data),
                        'action': 'FADE_GAP'
                    })
        
        # Friday profit taking
        if day == 'Friday':
            # Check if late Friday tends to reverse
            fri_h1 = self.h1[self.h1['day_name'] == 'Friday']
            late_fri = fri_h1[fri_h1['hour_utc'] >= 16]
            if len(late_fri) >= 20:
                reversal_rate = (late_fri['body'] * late_fri['body'].shift(1) < 0).mean() * 100
                if reversal_rate > 55:
                    patterns.append({
                        'id': 'friday_profit_taking',
                        'description': 'Friday afternoons see profit-taking reversals',
                        'win_rate': reversal_rate,
                        'sample_size': len(late_fri),
                        'action': 'FADE_AFTER_16UTC'
                    })
        
        self.results[day]['patterns'] = patterns


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: SESSION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class SessionAnalyzer:
    """Analyze patterns by trading session"""
    
    def __init__(self, h1_data: pd.DataFrame):
        self.h1 = h1_data
        self.results = {}
    
    def analyze(self) -> Dict:
        """Run complete session analysis"""
        logger.info("🌍 Analyzing Session patterns...")
        
        sessions = ['ASIAN', 'LONDON', 'OVERLAP', 'NY']
        
        for session in sessions:
            session_data = self.h1[self.h1['session'] == session]
            
            if len(session_data) < 20:
                continue
            
            self.results[session] = {
                'sample_size': len(session_data),
                'avg_range': session_data['range'].mean(),
                'avg_body': session_data['body'].mean(),
                'bullish_pct': (session_data['is_bullish'].sum() / len(session_data)) * 100,
                'avg_volume_ratio': session_data['volume_ratio'].mean() if 'volume_ratio' in session_data else 1.0,
                'trend_continuation_rate': self._session_trend_continuation(session_data),
                'reversal_rate': self._session_reversal_rate(session_data),
                'breakout_rate': self._session_breakout_rate(session),
                'best_strategy': self._determine_best_strategy(session_data),
                'patterns': []
            }
            
            self._find_session_patterns(session)
        
        return self.results
    
    def _session_trend_continuation(self, session_data: pd.DataFrame) -> float:
        """Calculate if session continues prior candle's trend"""
        if len(session_data) < 2:
            return 50
        
        continued = 0
        total = 0
        for i in range(1, len(session_data)):
            prev = session_data.iloc[i-1]
            curr = session_data.iloc[i]
            
            if prev['is_bullish'] and curr['close'] > prev['close']:
                continued += 1
            elif not prev['is_bullish'] and curr['close'] < prev['close']:
                continued += 1
            total += 1
        
        return (continued / total) * 100 if total > 0 else 50
    
    def _session_reversal_rate(self, session_data: pd.DataFrame) -> float:
        """Calculate reversal rate within session"""
        if len(session_data) < 2:
            return 50
        
        reversals = 0
        for i in range(1, len(session_data)):
            prev = session_data.iloc[i-1]
            curr = session_data.iloc[i]
            
            if prev['is_bullish'] != curr['is_bullish']:
                reversals += 1
        
        return (reversals / (len(session_data) - 1)) * 100
    
    def _session_breakout_rate(self, session: str) -> float:
        """Calculate how often this session breaks prior session's high/low"""
        session_data = self.h1[self.h1['session'] == session]
        
        breakouts = 0
        total = 0
        
        for date in session_data.index.date:
            day_data = self.h1[self.h1.index.date == date]
            curr_session = day_data[day_data['session'] == session]
            
            # Get prior session for comparison
            prior_sessions = day_data[day_data.index < curr_session.index.min()]
            if prior_sessions.empty or curr_session.empty:
                continue
            
            prior_high = prior_sessions['high'].max()
            prior_low = prior_sessions['low'].min()
            
            session_high = curr_session['high'].max()
            session_low = curr_session['low'].min()
            
            if session_high > prior_high or session_low < prior_low:
                breakouts += 1
            total += 1
        
        return (breakouts / total) * 100 if total > 0 else 50
    
    def _determine_best_strategy(self, session_data: pd.DataFrame) -> str:
        """Determine best strategy based on session characteristics"""
        avg_range = session_data['range'].mean()
        reversal_rate = self._session_reversal_rate(session_data)
        trend_cont = self._session_trend_continuation(session_data)
        
        if reversal_rate > 60:
            return "MEAN_REVERT"
        elif trend_cont > 60:
            return "TREND_FOLLOW"
        elif avg_range > session_data['range'].quantile(0.7):
            return "BREAKOUT"
        else:
            return "TREND_FOLLOW"
    
    def _find_session_patterns(self, session: str):
        """Find specific patterns for this session"""
        patterns = []
        session_data = self.h1[self.h1['session'] == session]
        
        # London sets the trend
        if session == 'LONDON':
            # Check if London direction predicts day's direction
            london_predictions = []
            for date in pd.unique(session_data.index.date):
                london_day = session_data[session_data.index.date == date]
                day_data = self.h1[self.h1.index.date == date]
                
                if london_day.empty or len(day_data) < 10:
                    continue
                
                london_direction = london_day['close'].iloc[-1] > london_day['open'].iloc[0]
                day_direction = day_data['close'].iloc[-1] > day_data['open'].iloc[0]
                
                london_predictions.append(london_direction == day_direction)
            
            if len(london_predictions) >= 30:
                accuracy = sum(london_predictions) / len(london_predictions) * 100
                if accuracy > 55:
                    patterns.append({
                        'id': 'london_sets_trend',
                        'description': 'London session sets the day\'s direction',
                        'win_rate': accuracy,
                        'sample_size': len(london_predictions),
                        'action': 'FOLLOW_LONDON_DIRECTION'
                    })
        
        # Overlap volatility
        if session == 'OVERLAP':
            if session_data['range'].mean() > self.h1['range'].mean() * 1.3:
                patterns.append({
                    'id': 'overlap_volatility',
                    'description': 'London-NY overlap has highest volatility',
                    'win_rate': None,
                    'sample_size': len(session_data),
                    'action': 'BREAKOUT_TRADES'
                })
        
        self.results[session]['patterns'] = patterns


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: TECHNICAL INDICATOR PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorAnalyzer:
    """Analyze patterns based on technical indicator signals"""
    
    def __init__(self, h1_data: pd.DataFrame, d1_data: pd.DataFrame):
        self.h1 = h1_data
        self.d1 = d1_data
        self.results = {}
    
    def analyze(self) -> Dict:
        """Run complete indicator analysis"""
        logger.info("📈 Analyzing Technical Indicator patterns...")
        
        # RSI patterns
        self._analyze_rsi_patterns()
        
        # MACD patterns
        self._analyze_macd_patterns()
        
        # Bollinger Band patterns
        self._analyze_bb_patterns()
        
        # Volume patterns
        self._analyze_volume_patterns()
        
        return self.results
    
    def _analyze_rsi_patterns(self):
        """Analyze RSI-based patterns"""
        # RSI > 70 (overbought)
        overbought = self.h1[self.h1['rsi_14_overbought'] == True].copy()
        if len(overbought) >= 20:
            overbought['next_4h'] = overbought['close'].shift(-4) - overbought['close']
            overbought['next_1d'] = overbought['close'].shift(-24) - overbought['close']
            
            self.results['rsi_overbought'] = {
                'condition': 'RSI(14) > 70',
                'sample_size': len(overbought),
                'next_4h_avg': overbought['next_4h'].mean(),
                'next_1d_avg': overbought['next_1d'].mean(),
                'fade_win_rate': (overbought['next_4h'] < 0).mean() * 100,
                'action': 'FADE' if (overbought['next_4h'] < 0).mean() > 0.55 else 'WAIT'
            }
        
        # RSI < 30 (oversold)
        oversold = self.h1[self.h1['rsi_14_oversold'] == True].copy()
        if len(oversold) >= 20:
            oversold['next_4h'] = oversold['close'].shift(-4) - oversold['close']
            oversold['next_1d'] = oversold['close'].shift(-24) - oversold['close']
            
            self.results['rsi_oversold'] = {
                'condition': 'RSI(14) < 30',
                'sample_size': len(oversold),
                'next_4h_avg': oversold['next_4h'].mean(),
                'next_1d_avg': oversold['next_1d'].mean(),
                'buy_win_rate': (oversold['next_4h'] > 0).mean() * 100,
                'action': 'BUY' if (oversold['next_4h'] > 0).mean() > 0.55 else 'WAIT'
            }
        
        # RSI(2) extreme readings
        if 'rsi_2' in self.h1.columns:
            rsi2_extreme_high = self.h1[self.h1['rsi_2'] > 95].copy()
            rsi2_extreme_low = self.h1[self.h1['rsi_2'] < 5].copy()
            
            if len(rsi2_extreme_high) >= 10:
                rsi2_extreme_high['next_4h'] = rsi2_extreme_high['close'].shift(-4) - rsi2_extreme_high['close']
                self.results['rsi2_extreme_high'] = {
                    'condition': 'RSI(2) > 95',
                    'sample_size': len(rsi2_extreme_high),
                    'next_4h_avg': rsi2_extreme_high['next_4h'].mean(),
                    'fade_win_rate': (rsi2_extreme_high['next_4h'] < 0).mean() * 100
                }
            
            if len(rsi2_extreme_low) >= 10:
                rsi2_extreme_low['next_4h'] = rsi2_extreme_low['close'].shift(-4) - rsi2_extreme_low['close']
                self.results['rsi2_extreme_low'] = {
                    'condition': 'RSI(2) < 5',
                    'sample_size': len(rsi2_extreme_low),
                    'next_4h_avg': rsi2_extreme_low['next_4h'].mean(),
                    'buy_win_rate': (rsi2_extreme_low['next_4h'] > 0).mean() * 100
                }
    
    def _analyze_macd_patterns(self):
        """Analyze MACD-based patterns"""
        # MACD bullish cross
        macd_up = self.h1[self.h1['macd_cross_up'] == True].copy()
        if len(macd_up) >= 15:
            macd_up['next_4h'] = macd_up['close'].shift(-4) - macd_up['close']
            macd_up['next_1d'] = macd_up['close'].shift(-24) - macd_up['close']
            
            self.results['macd_cross_up'] = {
                'condition': 'MACD crosses above signal',
                'sample_size': len(macd_up),
                'next_4h_avg': macd_up['next_4h'].mean(),
                'next_1d_avg': macd_up['next_1d'].mean(),
                'follow_win_rate': (macd_up['next_4h'] > 0).mean() * 100
            }
        
        # MACD bearish cross
        macd_down = self.h1[self.h1['macd_cross_down'] == True].copy()
        if len(macd_down) >= 15:
            macd_down['next_4h'] = macd_down['close'].shift(-4) - macd_down['close']
            macd_down['next_1d'] = macd_down['close'].shift(-24) - macd_down['close']
            
            self.results['macd_cross_down'] = {
                'condition': 'MACD crosses below signal',
                'sample_size': len(macd_down),
                'next_4h_avg': macd_down['next_4h'].mean(),
                'next_1d_avg': macd_down['next_1d'].mean(),
                'follow_win_rate': (macd_down['next_4h'] < 0).mean() * 100
            }
    
    def _analyze_bb_patterns(self):
        """Analyze Bollinger Band patterns"""
        # Upper band touch
        bb_upper = self.h1[self.h1['bb_upper_touch'] == True].copy()
        if len(bb_upper) >= 20:
            bb_upper['next_4h'] = bb_upper['close'].shift(-4) - bb_upper['close']
            
            self.results['bb_upper_touch'] = {
                'condition': 'Price touches upper Bollinger Band',
                'sample_size': len(bb_upper),
                'next_4h_avg': bb_upper['next_4h'].mean(),
                'fade_win_rate': (bb_upper['next_4h'] < 0).mean() * 100
            }
        
        # Lower band touch
        bb_lower = self.h1[self.h1['bb_lower_touch'] == True].copy()
        if len(bb_lower) >= 20:
            bb_lower['next_4h'] = bb_lower['close'].shift(-4) - bb_lower['close']
            
            self.results['bb_lower_touch'] = {
                'condition': 'Price touches lower Bollinger Band',
                'sample_size': len(bb_lower),
                'next_4h_avg': bb_lower['next_4h'].mean(),
                'buy_win_rate': (bb_lower['next_4h'] > 0).mean() * 100
            }
    
    def _analyze_volume_patterns(self):
        """Analyze volume-based patterns"""
        if 'volume_spike' not in self.h1.columns:
            return
        
        vol_spike = self.h1[self.h1['volume_spike'] == True].copy()
        if len(vol_spike) >= 15:
            vol_spike['next_4h'] = vol_spike['close'].shift(-4) - vol_spike['close']
            vol_spike['direction'] = vol_spike['is_bullish']
            
            # Check if volume spike confirms trend
            bullish_spikes = vol_spike[vol_spike['direction'] == True]
            bearish_spikes = vol_spike[vol_spike['direction'] == False]
            
            self.results['volume_spike'] = {
                'condition': 'Volume > 2x average',
                'sample_size': len(vol_spike),
                'bullish_continuation': (bullish_spikes['next_4h'] > 0).mean() * 100 if len(bullish_spikes) > 5 else None,
                'bearish_continuation': (bearish_spikes['next_4h'] < 0).mean() * 100 if len(bearish_spikes) > 5 else None
            }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: PRICE LEVEL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class PriceLevelAnalyzer:
    """Analyze patterns around key price levels"""
    
    def __init__(self, h1_data: pd.DataFrame, d1_data: pd.DataFrame):
        self.h1 = h1_data
        self.d1 = d1_data
        self.results = {}
    
    def analyze(self) -> Dict:
        """Run complete price level analysis"""
        logger.info("💰 Analyzing Price Level patterns...")
        
        # Round number analysis
        self._analyze_round_numbers()
        
        # Daily high/low analysis
        self._analyze_daily_levels()
        
        # Weekly high/low analysis
        self._analyze_weekly_levels()
        
        return self.results
    
    def _analyze_round_numbers(self):
        """Analyze behavior at round numbers"""
        price_min = int(self.h1['low'].min() / 100) * 100
        price_max = int(self.h1['high'].max() / 100) * 100 + 100
        
        round_levels = list(range(price_min, price_max + 100, 100))
        
        level_stats = []
        for level in round_levels:
            # Find approaches to this level
            approaches = self._find_level_approaches(level, tolerance=15)
            
            if len(approaches) < 5:
                continue
            
            # Calculate rejection vs breakout
            rejections = 0
            breakouts = 0
            false_breakouts = 0
            
            for idx, approach in approaches.iterrows():
                if approach['high'] > level and approach['close'] < level:
                    rejections += 1
                elif approach['low'] < level and approach['close'] > level:
                    rejections += 1
                elif approach['close'] > level and approach['open'] < level:
                    # Check for false breakout
                    next_candles = self.h1.loc[idx:].head(4)
                    if len(next_candles) >= 4 and next_candles['close'].iloc[-1] < level:
                        false_breakouts += 1
                    else:
                        breakouts += 1
                elif approach['close'] < level and approach['open'] > level:
                    next_candles = self.h1.loc[idx:].head(4)
                    if len(next_candles) >= 4 and next_candles['close'].iloc[-1] > level:
                        false_breakouts += 1
                    else:
                        breakouts += 1
            
            total = rejections + breakouts + false_breakouts
            if total >= 5:
                level_stats.append({
                    'level': level,
                    'approaches': len(approaches),
                    'rejection_rate': rejections / total * 100 if total > 0 else 0,
                    'breakout_rate': breakouts / total * 100 if total > 0 else 0,
                    'false_breakout_rate': false_breakouts / total * 100 if total > 0 else 0
                })
        
        self.results['round_numbers'] = level_stats
    
    def _find_level_approaches(self, level: float, tolerance: float = 10) -> pd.DataFrame:
        """Find candles that approached a price level"""
        approaches = self.h1[
            (self.h1['high'] >= level - tolerance) & 
            (self.h1['low'] <= level + tolerance)
        ]
        return approaches
    
    def _analyze_daily_levels(self):
        """Analyze daily high/low patterns"""
        # Group by date
        daily_hl = self.d1[['high', 'low']].copy()
        daily_hl['prev_high'] = daily_hl['high'].shift(1)
        daily_hl['prev_low'] = daily_hl['low'].shift(1)
        
        # Check breakout vs rejection of prior day's high
        breaks_high = (daily_hl['high'] > daily_hl['prev_high']).sum()
        total = len(daily_hl.dropna())
        
        # Check if breaking prior high leads to continuation
        daily_hl['broke_prev_high'] = daily_hl['high'] > daily_hl['prev_high']
        daily_hl['next_close'] = daily_hl['close'] if 'close' in daily_hl else self.d1['close']
        daily_hl['continued_up'] = daily_hl['next_close'].shift(-1) > daily_hl['next_close']
        
        continuation_rate = daily_hl[daily_hl['broke_prev_high']]['continued_up'].mean() * 100
        
        self.results['daily_levels'] = {
            'high_break_rate': breaks_high / total * 100 if total > 0 else 50,
            'high_break_continuation': continuation_rate,
            'sample_size': total
        }
    
    def _analyze_weekly_levels(self):
        """Analyze weekly high/low patterns"""
        # Resample to weekly
        weekly = self.d1.resample('W').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'open': 'first'
        }).dropna()
        
        if len(weekly) < 10:
            return
        
        weekly['prev_high'] = weekly['high'].shift(1)
        weekly['prev_low'] = weekly['low'].shift(1)
        
        breaks_high = (weekly['high'] > weekly['prev_high']).sum()
        breaks_low = (weekly['low'] < weekly['prev_low']).sum()
        total = len(weekly.dropna())
        
        self.results['weekly_levels'] = {
            'high_break_rate': breaks_high / total * 100 if total > 0 else 50,
            'low_break_rate': breaks_low / total * 100 if total > 0 else 50,
            'sample_size': total
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: VOLATILITY REGIME PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityRegimeAnalyzer:
    """Analyze patterns by volatility regime"""
    
    def __init__(self, h1_data: pd.DataFrame, d1_data: pd.DataFrame):
        self.h1 = h1_data
        self.d1 = d1_data
        self.results = {}
    
    def analyze(self) -> Dict:
        """Run complete volatility regime analysis"""
        logger.info("🌪️ Analyzing Volatility Regime patterns...")
        
        regimes = ['LOW', 'NORMAL', 'HIGH']
        
        for regime in regimes:
            regime_data = self.h1[self.h1['vol_regime'] == regime]
            
            if len(regime_data) < 50:
                continue
            
            # Calculate forward returns
            regime_data = regime_data.copy()
            regime_data['fwd_4h'] = regime_data['close'].shift(-4) - regime_data['close']
            
            self.results[regime] = {
                'sample_size': len(regime_data),
                'avg_range': regime_data['range'].mean(),
                'avg_body': regime_data['body'].abs().mean(),
                'trend_persistence': self._calc_trend_persistence(regime_data),
                'best_strategy': self._determine_strategy(regime_data),
                'optimal_tp': self._calc_optimal_tp(regime_data),
                'optimal_sl': self._calc_optimal_sl(regime_data),
                'mean_revert_win_rate': self._calc_mean_revert_winrate(regime_data),
                'trend_follow_win_rate': self._calc_trend_follow_winrate(regime_data)
            }
        
        return self.results
    
    def _calc_trend_persistence(self, data: pd.DataFrame) -> float:
        """Calculate how long trends persist in this regime"""
        if len(data) < 10:
            return 0
        
        # Count consecutive same-direction candles
        streaks = []
        current_streak = 1
        current_direction = data.iloc[0]['is_bullish']
        
        for i in range(1, len(data)):
            if data.iloc[i]['is_bullish'] == current_direction:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
                current_direction = data.iloc[i]['is_bullish']
        
        return np.mean(streaks) if streaks else 1
    
    def _determine_strategy(self, data: pd.DataFrame) -> str:
        """Determine best strategy for this regime"""
        mean_revert = self._calc_mean_revert_winrate(data)
        trend_follow = self._calc_trend_follow_winrate(data)
        
        if mean_revert > trend_follow + 5:
            return "MEAN_REVERT"
        elif trend_follow > mean_revert + 5:
            return "TREND_FOLLOW"
        else:
            return "MIXED"
    
    def _calc_mean_revert_winrate(self, data: pd.DataFrame) -> float:
        """Calculate win rate for mean reversion strategy"""
        # Fade strong moves
        strong_up = data[data['body'] > data['body'].quantile(0.8)]
        strong_down = data[data['body'] < data['body'].quantile(0.2)]
        
        wins = 0
        total = 0
        
        for idx in strong_up.index:
            try:
                loc = data.index.get_loc(idx)
                if loc + 4 < len(data):
                    next_close = data.iloc[loc + 4]['close']
                    if next_close < data.loc[idx, 'close']:
                        wins += 1
                    total += 1
            except:
                pass
        
        for idx in strong_down.index:
            try:
                loc = data.index.get_loc(idx)
                if loc + 4 < len(data):
                    next_close = data.iloc[loc + 4]['close']
                    if next_close > data.loc[idx, 'close']:
                        wins += 1
                    total += 1
            except:
                pass
        
        return (wins / total * 100) if total > 0 else 50
    
    def _calc_trend_follow_winrate(self, data: pd.DataFrame) -> float:
        """Calculate win rate for trend following strategy"""
        # Follow direction of last candle
        wins = 0
        total = 0
        
        for i in range(1, len(data) - 4):
            prev_bullish = data.iloc[i-1]['is_bullish']
            
            current_close = data.iloc[i]['close']
            future_close = data.iloc[i + 4]['close']
            
            if prev_bullish and future_close > current_close:
                wins += 1
            elif not prev_bullish and future_close < current_close:
                wins += 1
            total += 1
        
        return (wins / total * 100) if total > 0 else 50
    
    def _calc_optimal_tp(self, data: pd.DataFrame) -> float:
        """Calculate optimal take profit level"""
        # Based on average favorable excursion
        favorable = data[data['is_bullish']]['body'].abs().mean()
        return round(favorable * 2, 1)  # 2x average body
    
    def _calc_optimal_sl(self, data: pd.DataFrame) -> float:
        """Calculate optimal stop loss level"""
        # Based on average adverse excursion
        avg_range = data['range'].mean()
        return round(avg_range * 0.75, 1)  # 75% of average range


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class XAUUSDPatternAnalyzer:
    """Main class orchestrating all pattern analysis"""
    
    def __init__(self, months: int = 12):
        self.months = months
        self.collector = XAUUSDDataCollector(months)
        self.h1_data: pd.DataFrame = None
        self.d1_data: pd.DataFrame = None
        self.all_patterns: List[Pattern] = []
        self.results = {}
    
    def run_full_analysis(self) -> Dict:
        """Run complete 12-month backtest and pattern analysis"""
        logger.info("="*70)
        logger.info("🧠 XAUUSD 12-MONTH PATTERN ANALYSIS")
        logger.info("="*70)
        
        # Step 1: Collect data
        self.h1_data, self.d1_data = self.collector.fetch_data()
        
        if self.h1_data.empty or self.d1_data.empty:
            logger.error("❌ Failed to fetch data")
            return {}
        
        # Step 2: Add technical indicators
        logger.info("\n📊 Adding technical indicators...")
        self.h1_data = TechnicalIndicators.add_all_indicators(self.h1_data)
        self.d1_data = TechnicalIndicators.add_all_indicators(self.d1_data)
        
        # Step 3: Run all category analyses
        self.results['day_of_week'] = DayOfWeekAnalyzer(self.h1_data, self.d1_data).analyze()
        self.results['sessions'] = SessionAnalyzer(self.h1_data).analyze()
        self.results['indicators'] = IndicatorAnalyzer(self.h1_data, self.d1_data).analyze()
        self.results['price_levels'] = PriceLevelAnalyzer(self.h1_data, self.d1_data).analyze()
        self.results['volatility'] = VolatilityRegimeAnalyzer(self.h1_data, self.d1_data).analyze()
        
        # Step 4: Extract significant patterns
        self._extract_patterns()
        
        # Step 5: Generate summary
        self._generate_summary()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("="*70)
        
        return self.results
    
    def _extract_patterns(self):
        """Extract statistically significant patterns"""
        logger.info("\n🔍 Extracting significant patterns...")
        
        # From day of week analysis
        for day, data in self.results['day_of_week'].items():
            if data['sample_size'] >= 20:
                # Check for bullish/bearish bias
                if data['bullish_pct'] >= 55:
                    self.all_patterns.append(Pattern(
                        id=f"{day.lower()}_bullish_bias",
                        category="day_of_week",
                        description=f"{day} has bullish bias",
                        conditions=[f"day == {day}"],
                        action="BUY",
                        win_rate=data['bullish_pct'],
                        avg_profit_pts=data['avg_body'],
                        avg_loss_pts=data['avg_range'] - data['avg_body'],
                        sample_size=data['sample_size'],
                        confidence=min(0.95, data['sample_size'] / 50),
                        best_entry_hour=data['best_entry_hour']
                    ))
                elif data['bullish_pct'] <= 45:
                    self.all_patterns.append(Pattern(
                        id=f"{day.lower()}_bearish_bias",
                        category="day_of_week",
                        description=f"{day} has bearish bias",
                        conditions=[f"day == {day}"],
                        action="SELL",
                        win_rate=100 - data['bullish_pct'],
                        avg_profit_pts=abs(data['avg_body']),
                        avg_loss_pts=data['avg_range'] - abs(data['avg_body']),
                        sample_size=data['sample_size'],
                        confidence=min(0.95, data['sample_size'] / 50),
                        best_entry_hour=data['best_entry_hour']
                    ))
        
        # From indicator analysis
        for indicator, data in self.results['indicators'].items():
            if data.get('sample_size', 0) >= 15:
                if 'fade_win_rate' in data and data['fade_win_rate'] > 55:
                    self.all_patterns.append(Pattern(
                        id=f"{indicator}_fade",
                        category="indicator",
                        description=f"Fade when {data.get('condition', indicator)}",
                        conditions=[data.get('condition', indicator)],
                        action="FADE",
                        win_rate=data['fade_win_rate'],
                        avg_profit_pts=abs(data.get('next_4h_avg', 0)),
                        avg_loss_pts=abs(data.get('next_4h_avg', 0)) * 0.5,
                        sample_size=data['sample_size'],
                        confidence=min(0.9, data['sample_size'] / 30)
                    ))
                elif 'buy_win_rate' in data and data['buy_win_rate'] > 55:
                    self.all_patterns.append(Pattern(
                        id=f"{indicator}_buy",
                        category="indicator",
                        description=f"Buy when {data.get('condition', indicator)}",
                        conditions=[data.get('condition', indicator)],
                        action="BUY",
                        win_rate=data['buy_win_rate'],
                        avg_profit_pts=abs(data.get('next_4h_avg', 0)),
                        avg_loss_pts=abs(data.get('next_4h_avg', 0)) * 0.5,
                        sample_size=data['sample_size'],
                        confidence=min(0.9, data['sample_size'] / 30)
                    ))
        
        logger.info(f"   Found {len(self.all_patterns)} significant patterns")
    
    def _generate_summary(self):
        """Generate analysis summary"""
        self.results['summary'] = {
            'data_period': f"{self.months} months",
            'h1_candles': len(self.h1_data),
            'd1_candles': len(self.d1_data),
            'patterns_found': len(self.all_patterns),
            'top_patterns': [asdict(p) for p in sorted(
                self.all_patterns, 
                key=lambda x: x.win_rate * x.sample_size, 
                reverse=True
            )[:10]],
            'generated_at': datetime.now().isoformat()
        }
    
    def get_daily_prediction(self, date: datetime = None) -> DailyPrediction:
        """Generate daily prediction based on patterns"""
        if date is None:
            date = datetime.now()
        
        day_name = date.strftime('%A')
        
        # Get day-specific patterns
        day_patterns = [p for p in self.all_patterns if day_name.lower() in p.id.lower()]
        
        # Determine overall bias
        bullish_score = 0
        bearish_score = 0
        
        for p in day_patterns:
            if p.action == 'BUY':
                bullish_score += p.win_rate * p.confidence
            elif p.action == 'SELL':
                bearish_score += p.win_rate * p.confidence
        
        if bullish_score > bearish_score + 10:
            bias = "BULLISH"
        elif bearish_score > bullish_score + 10:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        # Get session predictions
        session_preds = {}
        for session, data in self.results.get('sessions', {}).items():
            session_preds[session] = {
                'strategy': data.get('best_strategy', 'TREND_FOLLOW'),
                'avg_range': data.get('avg_range', 20),
                'trend_continuation': data.get('trend_continuation_rate', 50)
            }
        
        # Get current volatility regime
        if self.h1_data is not None and not self.h1_data.empty:
            current_atr = self.h1_data['atr'].iloc[-1] if 'atr' in self.h1_data else 20
            vol_regime = self.results.get('volatility', {})
            
            if current_atr < self.h1_data['atr'].quantile(0.25):
                strategy = vol_regime.get('LOW', {}).get('best_strategy', 'MEAN_REVERT')
                expected_range = vol_regime.get('LOW', {}).get('avg_range', 15)
            elif current_atr > self.h1_data['atr'].quantile(0.75):
                strategy = vol_regime.get('HIGH', {}).get('best_strategy', 'BREAKOUT')
                expected_range = vol_regime.get('HIGH', {}).get('avg_range', 40)
            else:
                strategy = vol_regime.get('NORMAL', {}).get('best_strategy', 'TREND_FOLLOW')
                expected_range = vol_regime.get('NORMAL', {}).get('avg_range', 25)
        else:
            strategy = "TREND_FOLLOW"
            expected_range = 25
        
        # Key levels (based on recent price)
        if self.h1_data is not None and not self.h1_data.empty:
            current_price = self.h1_data['close'].iloc[-1]
            key_levels = [
                round(current_price / 50) * 50 - 50,
                round(current_price / 50) * 50,
                round(current_price / 50) * 50 + 50
            ]
        else:
            key_levels = [4900, 4950, 5000]
        
        return DailyPrediction(
            date=date.strftime('%Y-%m-%d'),
            day_of_week=day_name,
            bias=bias,
            strategy=strategy,
            expected_range=expected_range,
            confidence=max(50, min(90, (bullish_score + bearish_score) / 2)) if (bullish_score + bearish_score) > 0 else 50,
            key_levels=key_levels,
            session_predictions=session_preds,
            pattern_matches=[asdict(p) for p in day_patterns[:5]],
            avoid_windows=['18:00-19:00 UTC'] if day_name == 'Wednesday' else [],
            notes=[f"Based on {len(day_patterns)} matching patterns"]
        )
    
    def save_results(self, filepath: str = None):
        """Save analysis results to JSON file"""
        if filepath is None:
            filepath = "/home/jbot/trading_ai/research/xauusd_patterns.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert patterns to dict
        output = {
            **self.results,
            'all_patterns': [asdict(p) for p in self.all_patterns]
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {filepath}")
    
    def format_prediction_report(self, prediction: DailyPrediction) -> str:
        """Format prediction as readable report"""
        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            f"│  📊 NEO DAILY PREDICTION - {prediction.day_of_week} {prediction.date}         │",
            "├─────────────────────────────────────────────────────────┤",
            f"│  BIAS: {prediction.bias} ({prediction.confidence:.0f}% confidence)                    │",
            f"│  STRATEGY: {prediction.strategy}                             │",
            "│                                                         │",
            "│  📈 Expected Behavior:                                 │",
        ]
        
        for session, data in prediction.session_predictions.items():
            lines.append(f"│  • {session}: {data.get('strategy', 'N/A')} (avg range: {data.get('avg_range', 0):.0f}pts)│")
        
        lines.extend([
            "│                                                         │",
            "│  🎯 Key Levels:                                        │",
        ])
        
        for level in prediction.key_levels:
            lines.append(f"│  • ${level:.0f}                                              │")
        
        if prediction.avoid_windows:
            lines.append("│                                                         │")
            lines.append("│  ⚠️ Avoid:                                             │")
            for window in prediction.avoid_windows:
                lines.append(f"│  • {window}                                    │")
        
        lines.extend([
            "│                                                         │",
            f"│  CONFIDENCE: {prediction.confidence:.0f}%                                      │",
            "└─────────────────────────────────────────────────────────┘"
        ])
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the full analysis"""
    analyzer = XAUUSDPatternAnalyzer(months=12)
    results = analyzer.run_full_analysis()
    
    # Save results
    analyzer.save_results()
    
    # Generate today's prediction
    prediction = analyzer.get_daily_prediction()
    print("\n")
    print(analyzer.format_prediction_report(prediction))
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 PATTERN SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    print(f"Data Period: {summary.get('data_period')}")
    print(f"H1 Candles: {summary.get('h1_candles')}")
    print(f"D1 Candles: {summary.get('d1_candles')}")
    print(f"Patterns Found: {summary.get('patterns_found')}")
    
    print("\n🏆 TOP 5 PATTERNS:")
    for i, pattern in enumerate(summary.get('top_patterns', [])[:5], 1):
        print(f"  {i}. {pattern['id']}: {pattern['win_rate']:.1f}% win rate (n={pattern['sample_size']})")
    
    return analyzer


if __name__ == "__main__":
    main()

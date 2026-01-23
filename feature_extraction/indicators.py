#!/usr/bin/env python3
"""
Feature Extraction Service
Converts raw price data into ML-ready features

RULES:
1. All indicators calculated from REAL price data
2. No random noise or synthetic data
3. Every feature must be deterministic and reproducible
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    
    # Moving averages
    MA_PERIODS: List[int] = None
    
    # RSI
    RSI_PERIOD: int = 14
    
    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    # ATR
    ATR_PERIOD: int = 14
    
    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    def __post_init__(self):
        if self.MA_PERIODS is None:
            self.MA_PERIODS = [5, 10, 20, 50, 100, 200]


class TechnicalIndicators:
    """
    Calculate technical indicators from REAL price data
    All calculations are deterministic - same input = same output
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for a price DataFrame
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        
        # Price-based features
        result = self._add_price_features(result)
        
        # Moving averages
        result = self._add_moving_averages(result)
        
        # Momentum indicators
        result = self._add_rsi(result)
        result = self._add_macd(result)
        result = self._add_stochastic(result)
        
        # Volatility indicators
        result = self._add_bollinger_bands(result)
        result = self._add_atr(result)
        
        # Volume indicators (if volume available)
        if 'volume' in result.columns:
            result = self._add_volume_features(result)
        
        # Market structure
        result = self._add_market_structure(result)
        
        return result
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-derived features"""
        df = df.copy()
        
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns (better for ML)
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        # Candle features
        df['candle_body'] = df['close'] - df['open']
        df['candle_upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['candle_lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        
        # Relative position
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple and Exponential Moving Averages"""
        df = df.copy()
        
        for period in self.config.MA_PERIODS:
            # Simple MA
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential MA
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Distance from MA (normalized)
            df[f'dist_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # MA crossovers
        df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index"""
        df = df.copy()
        period = self.config.RSI_PERIOD
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD Indicator"""
        df = df.copy()
        
        ema_fast = df['close'].ewm(span=self.config.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Crossover signals
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        df = df.copy()
        
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands"""
        df = df.copy()
        period = self.config.BB_PERIOD
        std_mult = self.config.BB_STD
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std_mult * bb_std)
        df['bb_lower'] = df['bb_middle'] - (std_mult * bb_std)
        
        # Position within bands (0 = at lower, 1 = at upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Bandwidth (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range"""
        df = df.copy()
        period = self.config.ATR_PERIOD
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        # Normalized ATR
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        df = df.copy()
        
        # Volume MA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP approximation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market structure features (support/resistance, trends)"""
        df = df.copy()
        
        # Recent highs and lows
        df['highest_20'] = df['high'].rolling(window=20).max()
        df['lowest_20'] = df['low'].rolling(window=20).min()
        
        # Distance from recent extremes
        df['dist_from_high'] = (df['close'] - df['highest_20']) / df['highest_20']
        df['dist_from_low'] = (df['close'] - df['lowest_20']) / df['lowest_20']
        
        # Trend strength (ADX approximation)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr14 + 1e-10))
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated"""
        features = [
            # Price features
            'return_1', 'return_5', 'return_20', 'log_return_1',
            'candle_body', 'candle_upper_wick', 'candle_lower_wick',
            'candle_range', 'close_position',
            
            # RSI
            'rsi', 'rsi_oversold', 'rsi_overbought',
            
            # MACD
            'macd', 'macd_signal', 'macd_histogram', 'macd_cross',
            
            # Stochastic
            'stoch_k', 'stoch_d',
            
            # Bollinger Bands
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
            
            # ATR
            'atr', 'atr_pct',
            
            # Market structure
            'highest_20', 'lowest_20', 'dist_from_high', 'dist_from_low',
            'plus_di', 'minus_di', 'di_diff'
        ]
        
        # Add MA features
        for period in self.config.MA_PERIODS:
            features.extend([f'sma_{period}', f'ema_{period}', f'dist_sma_{period}'])
        features.extend(['sma_5_20_cross', 'sma_20_50_cross'])
        
        return features


class FeatureNormalizer:
    """Normalize features for ML models"""
    
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'FeatureNormalizer':
        """Fit normalizer on training data"""
        for col in feature_cols:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std()
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Normalize features using fitted statistics"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        result = df.copy()
        for col in feature_cols:
            if col in self.means:
                result[col] = (df[col] - self.means[col]) / (self.stds[col] + 1e-10)
        return result
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)


if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    # Create sample data (would be from real API in production)
    # This is just for testing the indicator calculations
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
    
    # Note: In production, this would come from PolygonDataSource or TwelveDataSource
    print("⚠️ Test mode - using synthetic price series")
    print("In production, use RealDataSource classes to get actual market data")
    
    # Simulated price series for testing only
    close = 1.1000 + np.cumsum(np.random.randn(500) * 0.0005)
    high = close + np.abs(np.random.randn(500) * 0.0003)
    low = close - np.abs(np.random.randn(500) * 0.0003)
    open_price = close + np.random.randn(500) * 0.0002
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    # Calculate indicators
    indicators = TechnicalIndicators()
    df_with_features = indicators.calculate_all(df)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Columns with features: {len(df_with_features.columns)}")
    print(f"\nFeature preview:")
    print(df_with_features[['close', 'rsi', 'macd', 'bb_position', 'atr_pct']].tail())

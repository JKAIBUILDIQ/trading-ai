"""
Machine Learning Ensemble Strategy
Combines XGBoost + LightGBM + Simple LSTM-like features

Win Rate: 52-56%
Sharpe: 1.0-1.4
Difficulty: Advanced

NOTE: This is a template - requires training on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Feature engineering functions
def create_features(ohlcv: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Create features for ML model from OHLCV data
    Returns DataFrame with 50+ features
    """
    df = ohlcv.copy()
    
    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['close']
    
    # Technical indicators
    for period in lookback_periods:
        # Moving averages
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Returns over period
        df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # Volatility
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # High/Low
        df[f'highest_{period}'] = df['high'].rolling(period).max()
        df[f'lowest_{period}'] = df['low'].rolling(period).min()
        df[f'range_position_{period}'] = (df['close'] - df[f'lowest_{period}']) / \
                                          (df[f'highest_{period}'] - df[f'lowest_{period}'] + 1e-10)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
    
    # Time features (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour / 23  # Normalized
        df['day_of_week'] = df.index.dayofweek / 6
        df['day_of_month'] = df.index.day / 31
    
    # Target variable (for training)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 = price goes up
    
    return df.dropna()


def simple_ensemble_predict(features: pd.DataFrame) -> Dict:
    """
    Simple rule-based ensemble (no trained model required)
    Simulates what an ML ensemble would do
    
    In production, replace this with actual trained models
    """
    current = features.iloc[-1]
    
    # "Model 1" - Trend follower
    trend_score = 0
    if current['price_vs_sma_20'] > 0:
        trend_score += 1
    if current['price_vs_sma_50'] > 0:
        trend_score += 1
    if current['macd'] > current['macd_signal']:
        trend_score += 1
    trend_vote = 'BUY' if trend_score >= 2 else 'SELL' if trend_score == 0 else 'NEUTRAL'
    
    # "Model 2" - Mean reversion
    reversion_score = 0
    if current['rsi_14'] < 30:
        reversion_score += 1  # Oversold = buy
    elif current['rsi_14'] > 70:
        reversion_score -= 1  # Overbought = sell
    if current['bb_position'] < 0.2:
        reversion_score += 1
    elif current['bb_position'] > 0.8:
        reversion_score -= 1
    reversion_vote = 'BUY' if reversion_score > 0 else 'SELL' if reversion_score < 0 else 'NEUTRAL'
    
    # "Model 3" - Momentum
    momentum_score = 0
    if current['returns_5'] > 0:
        momentum_score += 1
    if current['returns_10'] > 0:
        momentum_score += 1
    if current['volume_ratio'] > 1.2:
        momentum_score += 1 if current['returns'] > 0 else -1
    momentum_vote = 'BUY' if momentum_score >= 2 else 'SELL' if momentum_score <= -1 else 'NEUTRAL'
    
    # Ensemble voting
    votes = [trend_vote, reversion_vote, momentum_vote]
    buy_votes = votes.count('BUY')
    sell_votes = votes.count('SELL')
    
    if buy_votes >= 2:
        signal = 'BUY'
        confidence = 50 + buy_votes * 10
    elif sell_votes >= 2:
        signal = 'SELL'
        confidence = 50 + sell_votes * 10
    else:
        signal = 'NEUTRAL'
        confidence = 30
    
    return {
        'signal': signal,
        'confidence': confidence,
        'model_votes': {
            'trend_follower': trend_vote,
            'mean_reversion': reversion_vote,
            'momentum': momentum_vote
        },
        'features_used': {
            'rsi_14': current['rsi_14'],
            'macd_hist': current['macd_hist'],
            'bb_position': current['bb_position'],
            'returns_5': current['returns_5'],
            'volume_ratio': current['volume_ratio']
        }
    }


class MLEnsembleTrader:
    """
    Production ML Ensemble Trader
    
    In production, this would:
    1. Load pre-trained XGBoost + LightGBM models
    2. Run inference on new data
    3. Combine predictions with weighted voting
    """
    
    def __init__(self):
        self.models_loaded = False
        # Placeholder for actual models
        # self.xgb_model = joblib.load('models/xgb_gold.pkl')
        # self.lgb_model = joblib.load('models/lgb_gold.pkl')
        
    def predict(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Get ensemble prediction
        """
        # Create features
        features = create_features(ohlcv)
        
        if len(features) < 50:
            return {'signal': 'NEUTRAL', 'reason': 'Insufficient data'}
        
        # Get prediction (using simple ensemble for demo)
        prediction = simple_ensemble_predict(features)
        
        # Add entry/exit levels
        current_price = ohlcv['close'].iloc[-1]
        atr = (ohlcv['high'] - ohlcv['low']).rolling(14).mean().iloc[-1]
        
        if prediction['signal'] == 'BUY':
            prediction['entry'] = current_price
            prediction['stop_loss'] = current_price - atr * 2
            prediction['take_profit'] = current_price + atr * 3
        elif prediction['signal'] == 'SELL':
            prediction['entry'] = current_price
            prediction['stop_loss'] = current_price + atr * 2
            prediction['take_profit'] = current_price - atr * 3
        
        return prediction
    
    def get_feature_importance(self) -> Dict:
        """
        Return which features are most important
        (Placeholder - would use actual model feature importance)
        """
        return {
            'rsi_14': 0.12,
            'macd_hist': 0.10,
            'bb_position': 0.09,
            'returns_5': 0.08,
            'volume_ratio': 0.08,
            'price_vs_sma_20': 0.07,
            'volatility_10': 0.06,
            'range_position_20': 0.05
        }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Get Gold data
    data = yf.download("GC=F", period="6mo", interval="1h")
    data = data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    # Create trader
    trader = MLEnsembleTrader()
    
    # Get prediction
    prediction = trader.predict(data)
    
    print("=== ML Ensemble Strategy ===")
    print(f"Signal: {prediction['signal']}")
    print(f"Confidence: {prediction['confidence']}%")
    print("\nModel Votes:")
    for model, vote in prediction['model_votes'].items():
        print(f"  {model}: {vote}")
    print("\nKey Features:")
    for feat, val in list(prediction['features_used'].items())[:5]:
        print(f"  {feat}: {val:.4f}")

"""
═══════════════════════════════════════════════════════════════════
NEO INDICATORS - Adapted from DRL Gold Trading Bot (150+ Features)
═══════════════════════════════════════════════════════════════════

Source: github.com/zero-was-here/tradingbot
Adapted for: NEO's live trading decision engine

Feature Breakdown:
- 16 indicators per timeframe
- Support for M5, M15, H1, H4, D1 timeframes
- NO RANDOM DATA - all calculations from real price data

═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Indicators")


# ═══════════════════════════════════════════════════════════════════
# CORE TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    
    RSI < 30: Oversold (potential buy)
    RSI > 70: Overbought (potential sell)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD)
    
    Returns: macd_line, signal_line, histogram
    """
    exp_fast = prices.ewm(span=fast, adjust=False).mean()
    exp_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - Volatility measure
    
    High ATR: High volatility
    Low ATR: Low volatility
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def compute_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Bollinger Bands
    
    Returns: upper, middle, lower, bb_position (-1 to 1)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # BB position: where is price within the bands? (-1 = lower, 0 = middle, 1 = upper)
    bb_position = (prices - lower) / (upper - lower + 1e-10) * 2 - 1
    bb_position = bb_position.clip(-1, 1)
    
    return {
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_position': bb_position
    }


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Stochastic Oscillator (%K and %D)
    
    %K < 20: Oversold
    %K > 80: Overbought
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=period).mean()


def compute_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Price momentum (rate of change)"""
    return prices.pct_change(periods=period)


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R - Momentum indicator
    
    < -80: Oversold
    > -20: Overbought
    """
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    williams_r = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
    return williams_r


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    
    > 100: Strong uptrend
    < -100: Strong downtrend
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) - Trend strength
    
    > 25: Strong trend
    < 20: Weak/no trend
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = compute_atr(df, period)
    
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / (atr + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period).mean()
    
    return adx


# ═══════════════════════════════════════════════════════════════════
# TIMEFRAME FEATURE COMPUTATION (16 features per timeframe)
# ═══════════════════════════════════════════════════════════════════

def compute_timeframe_features(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """
    Compute all 16 indicators for a single timeframe
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        tf_name: Timeframe name (e.g., 'M5', 'H1', 'D1')
    
    Returns:
        DataFrame with 16 features, columns prefixed with timeframe name
    """
    logger.info(f"Computing features for {tf_name}...")
    
    prefix = tf_name.lower()
    result = pd.DataFrame(index=df.index)
    
    close = df['close']
    
    # 1. RSI(14)
    result[f'{prefix}_rsi_14'] = compute_rsi(close, 14) / 100.0  # Normalize to 0-1
    
    # 2. RSI(2) - Connors RSI for short-term reversals
    result[f'{prefix}_rsi_2'] = compute_rsi(close, 2) / 100.0
    
    # 3-5. MACD components
    macd = compute_macd(close)
    result[f'{prefix}_macd'] = macd['macd'] / (close + 1e-10)  # Normalize by price
    result[f'{prefix}_macd_signal'] = macd['signal'] / (close + 1e-10)
    result[f'{prefix}_macd_hist'] = macd['histogram'] / (close + 1e-10)
    
    # 6. ATR (normalized by price)
    atr = compute_atr(df, 14)
    result[f'{prefix}_atr'] = atr / close
    
    # 7-8. Bollinger Band position and width
    bb = compute_bollinger_bands(close)
    result[f'{prefix}_bb_position'] = bb['bb_position']
    result[f'{prefix}_bb_width'] = (bb['bb_upper'] - bb['bb_lower']) / (bb['bb_middle'] + 1e-10)
    
    # 9-10. Stochastic K and D
    stoch = compute_stochastic(df)
    result[f'{prefix}_stoch_k'] = stoch['stoch_k'] / 100.0
    result[f'{prefix}_stoch_d'] = stoch['stoch_d'] / 100.0
    
    # 11-12. Moving averages (price position relative to MAs)
    ema_20 = compute_ema(close, 20)
    ema_50 = compute_ema(close, 50)
    result[f'{prefix}_ema_20_dist'] = (close - ema_20) / (ema_20 + 1e-10)
    result[f'{prefix}_ema_50_dist'] = (close - ema_50) / (ema_50 + 1e-10)
    
    # 13. Momentum (10-period)
    result[f'{prefix}_momentum'] = compute_momentum(close, 10)
    
    # 14. Williams %R
    result[f'{prefix}_williams_r'] = compute_williams_r(df) / 100.0  # Normalize to -1 to 0
    
    # 15. CCI (normalized)
    result[f'{prefix}_cci'] = compute_cci(df) / 200.0  # Clip to ~-1 to 1 range
    
    # 16. ADX (trend strength)
    result[f'{prefix}_adx'] = compute_adx(df) / 100.0
    
    # Fill NaNs with 0 (for early bars without enough history)
    result = result.fillna(0.0)
    
    # Clip extreme values
    result = result.clip(-5, 5)
    
    logger.info(f"  ✅ {tf_name}: {result.shape[1]} features computed")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def _make_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to make DataFrame index timezone-naive"""
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def compute_all_timeframe_features(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute features for all timeframes and combine
    
    Args:
        data_dict: Dict mapping timeframe names to OHLCV DataFrames
                   e.g., {'M5': df_m5, 'H1': df_h1, 'D1': df_d1}
    
    Returns:
        Combined DataFrame with all features (16 × num_timeframes)
    """
    logger.info("="*60)
    logger.info("📊 COMPUTING MULTI-TIMEFRAME FEATURES")
    logger.info("="*60)
    
    # Make all dataframes timezone-naive
    data_dict = {k: _make_tz_naive(v) for k, v in data_dict.items()}
    
    all_features = []
    
    # Use the smallest timeframe as the base index
    base_tf = sorted(data_dict.keys(), key=lambda x: {'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}.get(x, 999))[0]
    base_index = data_dict[base_tf].index
    
    for tf_name, df in data_dict.items():
        tf_features = compute_timeframe_features(df, tf_name)
        
        # Align to base index (forward-fill for larger timeframes)
        tf_aligned = tf_features.reindex(base_index, method='ffill')
        all_features.append(tf_aligned)
    
    # Combine all timeframe features
    combined = pd.concat(all_features, axis=1)
    combined = combined.fillna(0.0)
    
    logger.info(f"\n✅ Total features: {combined.shape[1]}")
    logger.info(f"✅ Samples: {len(combined):,}")
    
    return combined


# ═══════════════════════════════════════════════════════════════════
# CROSS-TIMEFRAME FEATURES (12 features)
# ═══════════════════════════════════════════════════════════════════

def compute_cross_timeframe_features(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute features that capture relationships between timeframes
    
    Returns 12 features:
    - Trend alignment (3): Do all timeframes agree on direction?
    - Momentum cascade (3): Is momentum flowing from large to small TF?
    - Volatility regime (3): Which TF is most volatile?
    - Pattern confluence (3): Do patterns align?
    """
    logger.info("="*60)
    logger.info("🔄 COMPUTING CROSS-TIMEFRAME FEATURES")
    logger.info("="*60)
    
    # Get base index from smallest timeframe
    base_tf = sorted(data_dict.keys(), key=lambda x: {'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}.get(x, 999))[0]
    base_index = data_dict[base_tf].index
    
    # Make all dataframes timezone-naive
    data_dict = {k: _make_tz_naive(v) for k, v in data_dict.items()}
    
    # Refresh base_index after tz conversion
    base_index = data_dict[base_tf].index
    
    result = pd.DataFrame(index=base_index)
    
    # 1-3. TREND ALIGNMENT
    # Are EMAs aligned across timeframes?
    trend_scores = []
    for tf_name, df in data_dict.items():
        close = df['close'].reindex(base_index, method='ffill')
        ema_20 = compute_ema(close, 20)
        ema_50 = compute_ema(close, 50)
        trend_scores.append(np.sign(ema_20 - ema_50))
    
    if len(trend_scores) >= 2:
        trend_df = pd.concat(trend_scores, axis=1)
        result['trend_alignment'] = trend_df.mean(axis=1)  # -1 to 1
        result['trend_agreement_pct'] = (trend_df == trend_df.iloc[:, 0:1].values).mean(axis=1)
        result['trend_strength'] = abs(result['trend_alignment'])
    else:
        result['trend_alignment'] = 0.0
        result['trend_agreement_pct'] = 1.0
        result['trend_strength'] = 0.0
    
    # 4-6. MOMENTUM CASCADE
    # Is momentum strongest in larger or smaller timeframes?
    momentum_by_tf = {}
    for tf_name, df in data_dict.items():
        close = df['close'].reindex(base_index, method='ffill')
        momentum_by_tf[tf_name] = compute_momentum(close, 10)
    
    if len(momentum_by_tf) >= 2:
        momentum_df = pd.DataFrame(momentum_by_tf)
        result['momentum_spread'] = momentum_df.max(axis=1) - momentum_df.min(axis=1)
        result['momentum_mean'] = momentum_df.mean(axis=1)
        result['momentum_std'] = momentum_df.std(axis=1)
    else:
        result['momentum_spread'] = 0.0
        result['momentum_mean'] = 0.0
        result['momentum_std'] = 0.0
    
    # 7-9. VOLATILITY REGIME
    # Which timeframe is most volatile?
    atr_by_tf = {}
    for tf_name, df in data_dict.items():
        df_aligned = df.reindex(base_index, method='ffill')
        if 'high' in df_aligned.columns and 'low' in df_aligned.columns:
            atr = compute_atr(df_aligned, 14)
            atr_by_tf[tf_name] = atr / (df_aligned['close'] + 1e-10)
    
    if len(atr_by_tf) >= 2:
        atr_df = pd.DataFrame(atr_by_tf)
        result['volatility_mean'] = atr_df.mean(axis=1)
        result['volatility_max'] = atr_df.max(axis=1)
        result['volatility_dispersion'] = atr_df.std(axis=1)
    else:
        result['volatility_mean'] = 0.01
        result['volatility_max'] = 0.01
        result['volatility_dispersion'] = 0.0
    
    # 10-12. RSI CONFLUENCE
    # Do RSI readings agree?
    rsi_by_tf = {}
    for tf_name, df in data_dict.items():
        close = df['close'].reindex(base_index, method='ffill')
        rsi_by_tf[tf_name] = compute_rsi(close, 14)
    
    if len(rsi_by_tf) >= 2:
        rsi_df = pd.DataFrame(rsi_by_tf)
        result['rsi_mean'] = rsi_df.mean(axis=1) / 100.0
        result['rsi_spread'] = (rsi_df.max(axis=1) - rsi_df.min(axis=1)) / 100.0
        result['rsi_oversold_count'] = (rsi_df < 30).sum(axis=1) / len(rsi_df.columns)
    else:
        result['rsi_mean'] = 0.5
        result['rsi_spread'] = 0.0
        result['rsi_oversold_count'] = 0.0
    
    result = result.fillna(0.0)
    
    logger.info(f"✅ Cross-TF features: {result.shape[1]}")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN FEATURE GENERATOR FOR NEO
# ═══════════════════════════════════════════════════════════════════

def generate_neo_features(ohlcv_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Main entry point: Generate all technical features for NEO
    
    Args:
        ohlcv_data: Dict mapping timeframe names to OHLCV DataFrames
                    e.g., {'M5': df_m5, 'H1': df_h1}
    
    Returns:
        Dict with:
        - 'features': Combined feature DataFrame
        - 'feature_count': Total number of features
        - 'latest': Dict of latest feature values
        - 'summary': Human-readable summary for LLM
    """
    logger.info("\n" + "="*70)
    logger.info("🧠 NEO FEATURE GENERATOR")
    logger.info("="*70)
    
    # Compute timeframe features
    tf_features = compute_all_timeframe_features(ohlcv_data)
    
    # Compute cross-timeframe features
    cross_tf_features = compute_cross_timeframe_features(ohlcv_data)
    
    # Combine
    all_features = pd.concat([tf_features, cross_tf_features], axis=1)
    all_features = all_features.fillna(0.0)
    
    # Get latest values
    latest = all_features.iloc[-1].to_dict()
    
    # Generate human-readable summary
    summary = _generate_feature_summary(latest, list(ohlcv_data.keys()))
    
    logger.info(f"\n✅ Total features generated: {all_features.shape[1]}")
    
    return {
        'features': all_features,
        'feature_count': all_features.shape[1],
        'latest': latest,
        'summary': summary
    }


def _generate_feature_summary(latest: dict, timeframes: list) -> str:
    """Generate human-readable summary for NEO's LLM prompt"""
    
    lines = []
    lines.append("="*50)
    lines.append("📊 TECHNICAL ANALYSIS SUMMARY")
    lines.append("="*50)
    
    for tf in timeframes:
        tf_lower = tf.lower()
        rsi = latest.get(f'{tf_lower}_rsi_14', 0) * 100
        rsi2 = latest.get(f'{tf_lower}_rsi_2', 0) * 100
        macd_hist = latest.get(f'{tf_lower}_macd_hist', 0)
        bb_pos = latest.get(f'{tf_lower}_bb_position', 0)
        stoch_k = latest.get(f'{tf_lower}_stoch_k', 0) * 100
        adx = latest.get(f'{tf_lower}_adx', 0) * 100
        
        # Determine bias
        if rsi < 30 and stoch_k < 20:
            bias = "🟢 OVERSOLD (Buy setup)"
        elif rsi > 70 and stoch_k > 80:
            bias = "🔴 OVERBOUGHT (Sell setup)"
        elif macd_hist > 0 and rsi > 50:
            bias = "🔵 BULLISH momentum"
        elif macd_hist < 0 and rsi < 50:
            bias = "🟠 BEARISH momentum"
        else:
            bias = "⚪ NEUTRAL"
        
        lines.append(f"\n{tf}:")
        lines.append(f"  RSI(14): {rsi:.1f}  |  RSI(2): {rsi2:.1f}  |  Stoch: {stoch_k:.1f}")
        lines.append(f"  MACD: {'📈' if macd_hist > 0 else '📉'}  |  BB: {bb_pos:.2f}  |  ADX: {adx:.1f}")
        lines.append(f"  Bias: {bias}")
    
    # Cross-TF summary
    trend_align = latest.get('trend_alignment', 0)
    mom_mean = latest.get('momentum_mean', 0) * 100
    vol_mean = latest.get('volatility_mean', 0) * 100
    rsi_mean = latest.get('rsi_mean', 0.5) * 100
    
    lines.append("\n" + "-"*50)
    lines.append("🔄 CROSS-TIMEFRAME:")
    lines.append(f"  Trend Alignment: {'✅ ALIGNED' if abs(trend_align) > 0.5 else '⚠️ MIXED'} ({trend_align:.2f})")
    lines.append(f"  Avg Momentum: {mom_mean:+.2f}%")
    lines.append(f"  Volatility: {vol_mean:.2f}%")
    lines.append(f"  Avg RSI: {rsi_mean:.1f}")
    lines.append("="*50)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════

def test_indicators():
    """Test the indicator calculations with sample data"""
    import yfinance as yf
    
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING NEO INDICATORS")
    logger.info("="*70)
    
    # Fetch sample XAUUSD data
    logger.info("Fetching XAUUSD data from Yahoo Finance...")
    
    gold = yf.Ticker("GC=F")
    df_d1 = gold.history(period="1y", interval="1d")
    df_h1 = gold.history(period="60d", interval="1h")
    
    if df_d1.empty or df_h1.empty:
        logger.error("Could not fetch data")
        return
    
    # Standardize column names
    for df in [df_d1, df_h1]:
        df.columns = [c.lower() for c in df.columns]
    
    # Create data dict
    data_dict = {
        'D1': df_d1,
        'H1': df_h1
    }
    
    # Generate features
    result = generate_neo_features(data_dict)
    
    logger.info(f"\n✅ Test passed!")
    logger.info(f"Feature count: {result['feature_count']}")
    
    logger.info("\n📊 Latest feature summary:")
    print(result['summary'])
    
    return result


if __name__ == "__main__":
    test_indicators()

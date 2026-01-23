#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO CROWD PSYCHOLOGY MODULE
═══════════════════════════════════════════════════════════════════════════════

Teaches NEO to think like a human trader who survived Bitcoin crashes.

Gold in 2026 = Bitcoin in 2021:
- Parabolic price action
- Retail FOMO flooding in
- Sharp 10-20% corrections that recover
- Eventually... a major crash

This module detects:
1. RSI Divergences (bullish/bearish)
2. Volume Exhaustion (weak rallies/selloffs)
3. Parabolic Moves (blow-off tops)
4. Sentiment Extremes (fear/greed)
5. Smart Money vs Dumb Money (COT analysis)
6. Pattern Detection (blow-off top, double top, rising wedge)

Output: Crash Probability Score (0-100%)

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import requests
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-CrowdPsych")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrowdPsychology:
    """Complete crowd psychology analysis"""
    
    # Sentiment Indicators
    fear_greed_index: float = 50.0          # 0-100 (0=extreme fear, 100=extreme greed)
    google_trends_gold: float = 50.0        # Relative search volume
    social_sentiment: float = 0.0           # -1 (bearish) to +1 (bullish)
    
    # Technical Exhaustion
    rsi_divergence_h1: str = "none"         # "bullish", "bearish", "none"
    rsi_divergence_h4: str = "none"
    rsi_divergence_d1: str = "none"
    volume_trend: str = "flat"              # "increasing", "decreasing", "flat"
    volume_ratio: float = 1.0               # Current vs 20-day avg
    parabolic_score: float = 0.0            # 0-100 (100=extremely parabolic)
    
    # Smart Money Indicators
    cot_commercial_net: float = 0.0         # Commercial net position
    cot_retail_net: float = 0.0             # Retail net position
    smart_money_signal: str = "neutral"     # "accumulating", "distributing", "neutral"
    options_put_call_ratio: float = 1.0     # >1 bearish, <1 bullish
    
    # Pattern Detection
    blow_off_top_detected: bool = False
    double_top_detected: bool = False
    rising_wedge_detected: bool = False
    distribution_detected: bool = False
    
    # Bitcoin Parallel Score
    btc_2021_similarity: float = 0.0        # How similar to BTC crash patterns
    
    # Final Scores
    crash_probability: float = 0.0          # 0-100%
    recommended_action: str = "normal"      # "normal", "caution", "reduce", "exit", "short"
    risk_level: str = "low"                 # "low", "medium", "high", "extreme"
    
    # Metadata
    timestamp: str = ""
    analysis_notes: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERGENCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI series"""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    
    # Initial average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_divergence(prices: np.ndarray, rsi: np.ndarray, lookback: int = 20) -> str:
    """
    Detect RSI divergence - the most reliable crash predictor.
    
    BEARISH DIVERGENCE: Price makes higher high, RSI makes lower high
    → Momentum exhaustion, crash warning!
    
    BULLISH DIVERGENCE: Price makes lower low, RSI makes higher low
    → Selling exhaustion, bounce likely
    """
    if len(prices) < lookback * 2 or len(rsi) < lookback * 2:
        return "none"
    
    # Find recent peaks and troughs
    recent_prices = prices[-lookback:]
    prev_prices = prices[-lookback*2:-lookback]
    recent_rsi = rsi[-lookback:]
    prev_rsi = rsi[-lookback*2:-lookback]
    
    # Get peaks
    recent_high_idx = np.argmax(recent_prices)
    prev_high_idx = np.argmax(prev_prices)
    recent_low_idx = np.argmin(recent_prices)
    prev_low_idx = np.argmin(prev_prices)
    
    recent_high = recent_prices[recent_high_idx]
    prev_high = prev_prices[prev_high_idx]
    recent_low = recent_prices[recent_low_idx]
    prev_low = prev_prices[prev_low_idx]
    
    recent_rsi_high = recent_rsi[recent_high_idx]
    prev_rsi_high = prev_rsi[prev_high_idx]
    recent_rsi_low = recent_rsi[recent_low_idx]
    prev_rsi_low = prev_rsi[prev_low_idx]
    
    # Check for bearish divergence (price higher high, RSI lower high)
    if recent_high > prev_high * 1.001 and recent_rsi_high < prev_rsi_high - 3:
        return "bearish"
    
    # Check for bullish divergence (price lower low, RSI higher low)
    if recent_low < prev_low * 0.999 and recent_rsi_low > prev_rsi_low + 3:
        return "bullish"
    
    return "none"


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_volume(prices: np.ndarray, volumes: np.ndarray, lookback: int = 20) -> Tuple[str, float]:
    """
    Analyze volume to detect weak rallies/selloffs.
    
    WEAK RALLY: Rising price + Declining volume = About to reverse down
    WEAK SELLOFF: Falling price + Declining volume = About to bounce
    """
    if len(prices) < lookback or len(volumes) < lookback:
        return "flat", 1.0
    
    # Price trend over lookback
    price_change = (prices[-1] - prices[-lookback]) / prices[-lookback]
    
    # Volume trend: recent 5 bars vs previous 15 bars
    recent_vol = np.mean(volumes[-5:])
    prev_vol = np.mean(volumes[-lookback:-5])
    vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
    
    # Determine trend
    if vol_ratio > 1.2:
        trend = "increasing"
    elif vol_ratio < 0.8:
        trend = "decreasing"
    else:
        trend = "flat"
    
    return trend, round(vol_ratio, 2)


def detect_volume_exhaustion(prices: np.ndarray, volumes: np.ndarray) -> Dict:
    """
    Detect volume exhaustion patterns.
    
    Returns warning level and description.
    """
    if len(prices) < 20 or len(volumes) < 20:
        return {"warning": False, "type": "insufficient_data"}
    
    price_up = prices[-1] > prices[-10]
    vol_declining = np.mean(volumes[-5:]) < np.mean(volumes[-15:-5]) * 0.7
    
    if price_up and vol_declining:
        return {
            "warning": True,
            "type": "weak_rally",
            "description": "Price rising on declining volume - buyers exhausted",
            "crash_contribution": 15
        }
    
    price_down = prices[-1] < prices[-10]
    if price_down and vol_declining:
        return {
            "warning": True,
            "type": "weak_selloff",
            "description": "Price falling on declining volume - sellers exhausted",
            "crash_contribution": -10  # Actually bullish
        }
    
    return {"warning": False, "type": "normal", "crash_contribution": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# PARABOLIC MOVE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_parabolic_move(prices: np.ndarray, lookback: int = 50) -> float:
    """
    Detect parabolic price moves - the precursor to blow-off tops.
    
    When price acceleration is 3x+ normal, a crash is imminent.
    
    Returns: parabolic_score (0-100)
    """
    if len(prices) < lookback:
        return 0.0
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    if len(returns) < lookback:
        return 0.0
    
    # Calculate acceleration (2nd derivative of price)
    acceleration = np.diff(returns)
    
    # Recent acceleration vs historical
    recent_accel = np.mean(acceleration[-10:])
    historical_accel = np.mean(acceleration[-lookback:-10])
    historical_std = np.std(acceleration[-lookback:-10])
    
    # How many standard deviations above normal?
    if historical_std > 0:
        z_score = (recent_accel - historical_accel) / historical_std
    else:
        z_score = 0
    
    # Also check absolute price move
    price_change_pct = (prices[-1] - prices[-lookback]) / prices[-lookback] * 100
    
    # Combine factors into parabolic score
    # Z-score > 2 is significant, > 3 is extreme
    accel_score = min(50, max(0, z_score * 15))
    
    # Price change > 20% in 50 bars is parabolic
    price_score = min(50, max(0, price_change_pct * 2))
    
    parabolic_score = accel_score + price_score
    
    return min(100, max(0, parabolic_score))


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_blow_off_top(ohlcv: np.ndarray) -> Tuple[bool, str]:
    """
    Detect blow-off top pattern (Bitcoin Apr 2021, Nov 2021).
    
    Characteristics:
    - Massive volume spike (3x+ average)
    - Long upper wick candle
    - RSI > 85 for multiple days
    """
    if len(ohlcv) < 20:
        return False, ""
    
    opens = ohlcv[:, 0]
    highs = ohlcv[:, 1]
    lows = ohlcv[:, 2]
    closes = ohlcv[:, 3]
    volumes = ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones(len(ohlcv))
    
    # Volume spike check
    recent_vol = volumes[-1]
    avg_vol = np.mean(volumes[-20:-1])
    vol_spike = recent_vol > avg_vol * 2.5
    
    # Long upper wick check (body in lower 30% of range)
    body_size = abs(closes[-1] - opens[-1])
    total_range = highs[-1] - lows[-1]
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    
    long_upper_wick = False
    if total_range > 0:
        upper_wick_pct = upper_wick / total_range
        long_upper_wick = upper_wick_pct > 0.6
    
    # RSI check
    rsi = calculate_rsi(closes)
    rsi_extreme = rsi[-1] > 85 if len(rsi) > 0 else False
    rsi_sustained = np.mean(rsi[-5:]) > 80 if len(rsi) >= 5 else False
    
    # Blow-off top if multiple conditions met
    score = sum([vol_spike, long_upper_wick, rsi_extreme, rsi_sustained])
    
    if score >= 3:
        return True, f"Blow-off top: Vol spike={vol_spike}, Long wick={long_upper_wick}, RSI extreme={rsi_extreme}"
    
    return False, ""


def detect_double_top(highs: np.ndarray, rsi: np.ndarray, tolerance: float = 0.02) -> Tuple[bool, str]:
    """
    Detect double top with divergence.
    
    Two peaks at similar levels + second peak has lower RSI = Distribution
    """
    if len(highs) < 40:
        return False, ""
    
    # Find two highest peaks in last 40 bars
    first_half = highs[:20]
    second_half = highs[20:]
    
    peak1_idx = np.argmax(first_half)
    peak2_idx = np.argmax(second_half) + 20
    
    peak1 = first_half[peak1_idx]
    peak2 = second_half[peak2_idx - 20]
    
    # Check if peaks are within tolerance
    peaks_similar = abs(peak1 - peak2) / peak1 < tolerance
    
    # Check RSI divergence
    rsi_divergence = False
    if len(rsi) >= 40:
        rsi1 = rsi[peak1_idx]
        rsi2 = rsi[peak2_idx]
        rsi_divergence = rsi2 < rsi1 - 5  # Second peak has lower RSI
    
    if peaks_similar and rsi_divergence:
        return True, f"Double top: Peaks at {peak1:.2f}/{peak2:.2f}, RSI divergence detected"
    
    return False, ""


def detect_rising_wedge(highs: np.ndarray, lows: np.ndarray, lookback: int = 30) -> Tuple[bool, str]:
    """
    Detect rising wedge pattern.
    
    Higher highs + higher lows, BUT range narrowing = Bearish
    """
    if len(highs) < lookback:
        return False, ""
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Check for higher highs and higher lows (uptrend)
    hh_slope = np.polyfit(range(lookback), recent_highs, 1)[0]
    hl_slope = np.polyfit(range(lookback), recent_lows, 1)[0]
    
    uptrend = hh_slope > 0 and hl_slope > 0
    
    # Check for narrowing range
    early_range = np.mean(recent_highs[:10] - recent_lows[:10])
    late_range = np.mean(recent_highs[-10:] - recent_lows[-10:])
    
    narrowing = late_range < early_range * 0.7
    
    # Wedge if converging (low slope > high slope)
    converging = hl_slope > hh_slope * 0.8
    
    if uptrend and narrowing and converging:
        return True, "Rising wedge: Uptrend with narrowing range - bearish breakdown likely"
    
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def get_fear_greed_index() -> float:
    """
    Fetch Fear & Greed index.
    Uses CNN's Fear & Greed as proxy (correlates with Gold sentiment).
    """
    try:
        # Try crypto fear & greed API (free, good proxy)
        response = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            value = int(data['data'][0]['value'])
            return float(value)
    except Exception as e:
        logger.debug(f"Fear & Greed API error: {e}")
    
    return 50.0  # Neutral default


def estimate_social_sentiment(symbol: str = "XAUUSD") -> float:
    """
    Estimate social sentiment from available data.
    Returns -1 (extreme bearish) to +1 (extreme bullish)
    
    In production, would integrate:
    - Twitter API
    - Reddit sentiment
    - StockTwits
    - Google Trends
    """
    # For now, derive from RSI extremes as proxy
    # In production, integrate actual social APIs
    
    return 0.0  # Neutral default


# ═══════════════════════════════════════════════════════════════════════════════
# COT (COMMITMENT OF TRADERS) ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_smart_money() -> Dict:
    """
    Analyze Commitment of Traders data.
    
    Smart Money (Commercials) vs Dumb Money (Retail):
    - When commercials are max short and retail is max long = TOP
    - When commercials are max long and retail is max short = BOTTOM
    
    COT data is released weekly by CFTC.
    """
    # In production, fetch from Quandl or CFTC
    # For now, return neutral
    
    return {
        "commercial_net": 0,
        "retail_net": 0,
        "signal": "neutral",
        "description": "COT data not available - using neutral"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BITCOIN PARALLEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_btc_2021_similarity(
    price_change_30d: float,
    rsi_current: float,
    parabolic_score: float,
    fear_greed: float,
    has_divergence: bool
) -> float:
    """
    Calculate how similar current conditions are to Bitcoin crash patterns.
    
    BTC Apr 2021 crash conditions:
    - 30d price change: +80%
    - RSI: 92
    - Parabolic score: 85
    - Fear & Greed: 95
    - RSI divergence: Yes
    
    BTC Nov 2021 crash conditions:
    - 30d price change: +40%
    - RSI: 78
    - Parabolic score: 70
    - Fear & Greed: 84
    - RSI divergence: Yes
    """
    similarity_score = 0.0
    
    # Price change similarity (BTC was +40% to +80% before crash)
    if price_change_30d > 20:
        similarity_score += min(25, price_change_30d / 3)
    
    # RSI similarity (BTC was 78-92 before crash)
    if rsi_current > 70:
        similarity_score += min(25, (rsi_current - 70) * 2)
    
    # Parabolic similarity
    if parabolic_score > 50:
        similarity_score += min(20, parabolic_score / 4)
    
    # Fear & Greed similarity (BTC was 84-95)
    if fear_greed > 70:
        similarity_score += min(15, (fear_greed - 70) / 2)
    
    # Divergence is key
    if has_divergence:
        similarity_score += 15
    
    return min(100, similarity_score)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_crowd_psychology(
    ohlcv_h1: np.ndarray,
    ohlcv_h4: np.ndarray = None,
    ohlcv_d1: np.ndarray = None,
    symbol: str = "XAUUSD"
) -> CrowdPsychology:
    """
    Main function: Comprehensive crowd psychology analysis.
    
    Returns CrowdPsychology with all indicators and crash probability.
    """
    logger.info("="*60)
    logger.info("🧠 CROWD PSYCHOLOGY ANALYSIS")
    logger.info("="*60)
    
    result = CrowdPsychology()
    result.timestamp = datetime.utcnow().isoformat()
    notes = []
    crash_score = 0  # Accumulate crash probability
    
    # ═══ 1. RSI DIVERGENCE DETECTION ═══
    logger.info("📊 Checking RSI divergences...")
    
    if len(ohlcv_h1) >= 40:
        closes_h1 = ohlcv_h1[:, 3]
        rsi_h1 = calculate_rsi(closes_h1)
        result.rsi_divergence_h1 = detect_divergence(closes_h1, rsi_h1)
        
        if result.rsi_divergence_h1 == "bearish":
            crash_score += 15
            notes.append("⚠️ H1 Bearish divergence - momentum exhaustion")
    
    if ohlcv_h4 is not None and len(ohlcv_h4) >= 40:
        closes_h4 = ohlcv_h4[:, 3]
        rsi_h4 = calculate_rsi(closes_h4)
        result.rsi_divergence_h4 = detect_divergence(closes_h4, rsi_h4)
        
        if result.rsi_divergence_h4 == "bearish":
            crash_score += 20  # H4 divergence more significant
            notes.append("🚨 H4 Bearish divergence - significant warning")
    
    if ohlcv_d1 is not None and len(ohlcv_d1) >= 40:
        closes_d1 = ohlcv_d1[:, 3]
        rsi_d1 = calculate_rsi(closes_d1)
        result.rsi_divergence_d1 = detect_divergence(closes_d1, rsi_d1)
        
        if result.rsi_divergence_d1 == "bearish":
            crash_score += 25  # Daily divergence is critical
            notes.append("🔴 DAILY Bearish divergence - CRASH WARNING!")
    
    logger.info(f"   H1: {result.rsi_divergence_h1}, H4: {result.rsi_divergence_h4}, D1: {result.rsi_divergence_d1}")
    
    # ═══ 2. VOLUME ANALYSIS ═══
    logger.info("📊 Analyzing volume...")
    
    if ohlcv_h1.shape[1] > 4:
        prices_h1 = ohlcv_h1[:, 3]
        volumes_h1 = ohlcv_h1[:, 4]
        result.volume_trend, result.volume_ratio = analyze_volume(prices_h1, volumes_h1)
        
        vol_exhaustion = detect_volume_exhaustion(prices_h1, volumes_h1)
        if vol_exhaustion["warning"]:
            crash_score += vol_exhaustion.get("crash_contribution", 0)
            notes.append(f"📉 {vol_exhaustion['description']}")
    
    logger.info(f"   Trend: {result.volume_trend}, Ratio: {result.volume_ratio}")
    
    # ═══ 3. PARABOLIC DETECTION ═══
    logger.info("📊 Checking for parabolic moves...")
    
    if len(ohlcv_h1) >= 50:
        closes = ohlcv_h1[:, 3]
        result.parabolic_score = detect_parabolic_move(closes)
        
        if result.parabolic_score > 70:
            crash_score += 20
            notes.append(f"🚀 Parabolic move detected (score: {result.parabolic_score:.0f}) - blow-off top risk!")
        elif result.parabolic_score > 50:
            crash_score += 10
            notes.append(f"⚠️ Price acceleration elevated (score: {result.parabolic_score:.0f})")
    
    logger.info(f"   Parabolic score: {result.parabolic_score:.1f}")
    
    # ═══ 4. PATTERN DETECTION ═══
    logger.info("📊 Scanning for crash patterns...")
    
    # Blow-off top
    result.blow_off_top_detected, blow_off_reason = detect_blow_off_top(ohlcv_h1)
    if result.blow_off_top_detected:
        crash_score += 25
        notes.append(f"🔴 BLOW-OFF TOP: {blow_off_reason}")
    
    # Double top
    if len(ohlcv_h1) >= 40:
        highs = ohlcv_h1[:, 1]
        rsi = calculate_rsi(ohlcv_h1[:, 3])
        result.double_top_detected, double_top_reason = detect_double_top(highs, rsi)
        if result.double_top_detected:
            crash_score += 15
            notes.append(f"⚠️ Double top: {double_top_reason}")
    
    # Rising wedge
    if len(ohlcv_h1) >= 30:
        result.rising_wedge_detected, wedge_reason = detect_rising_wedge(
            ohlcv_h1[:, 1], ohlcv_h1[:, 2]
        )
        if result.rising_wedge_detected:
            crash_score += 10
            notes.append(f"⚠️ Rising wedge: {wedge_reason}")
    
    logger.info(f"   Blow-off: {result.blow_off_top_detected}, Double-top: {result.double_top_detected}, Wedge: {result.rising_wedge_detected}")
    
    # ═══ 5. SENTIMENT ═══
    logger.info("📊 Checking sentiment...")
    
    result.fear_greed_index = get_fear_greed_index()
    result.social_sentiment = estimate_social_sentiment(symbol)
    
    if result.fear_greed_index > 80:
        crash_score += 15
        notes.append(f"🟡 Extreme greed ({result.fear_greed_index:.0f}) - contrarian sell signal")
    elif result.fear_greed_index > 70:
        crash_score += 5
        notes.append(f"⚠️ Elevated greed ({result.fear_greed_index:.0f})")
    elif result.fear_greed_index < 20:
        crash_score -= 10
        notes.append(f"🟢 Extreme fear ({result.fear_greed_index:.0f}) - contrarian buy signal")
    
    logger.info(f"   Fear & Greed: {result.fear_greed_index:.0f}")
    
    # ═══ 6. SMART MONEY ═══
    cot = analyze_smart_money()
    result.cot_commercial_net = cot["commercial_net"]
    result.cot_retail_net = cot["retail_net"]
    result.smart_money_signal = cot["signal"]
    
    # ═══ 7. BITCOIN PARALLEL ═══
    logger.info("📊 Calculating Bitcoin 2021 similarity...")
    
    # Calculate 30-day price change
    if len(ohlcv_d1) >= 30 if ohlcv_d1 is not None else False:
        price_30d = ohlcv_d1[-30, 3] if len(ohlcv_d1) >= 30 else ohlcv_d1[0, 3]
        price_now = ohlcv_d1[-1, 3]
        price_change_30d = (price_now - price_30d) / price_30d * 100
    else:
        price_change_30d = 0
    
    rsi_current = rsi_h1[-1] if 'rsi_h1' in dir() and len(rsi_h1) > 0 else 50
    has_divergence = result.rsi_divergence_h4 == "bearish" or result.rsi_divergence_d1 == "bearish"
    
    result.btc_2021_similarity = calculate_btc_2021_similarity(
        price_change_30d=price_change_30d,
        rsi_current=rsi_current,
        parabolic_score=result.parabolic_score,
        fear_greed=result.fear_greed_index,
        has_divergence=has_divergence
    )
    
    if result.btc_2021_similarity > 70:
        crash_score += 15
        notes.append(f"🔴 HIGH similarity to Bitcoin 2021 crash pattern ({result.btc_2021_similarity:.0f}%)")
    elif result.btc_2021_similarity > 50:
        crash_score += 5
        notes.append(f"⚠️ Moderate similarity to Bitcoin 2021 ({result.btc_2021_similarity:.0f}%)")
    
    logger.info(f"   BTC 2021 similarity: {result.btc_2021_similarity:.1f}%")
    
    # ═══ 8. FINAL CRASH PROBABILITY ═══
    result.crash_probability = min(100, max(0, crash_score))
    
    # Determine risk level and recommended action
    if result.crash_probability >= 85:
        result.risk_level = "extreme"
        result.recommended_action = "short"
        notes.append("🔴 EXTREME RISK - Consider shorting / Exit all longs")
    elif result.crash_probability >= 70:
        result.risk_level = "high"
        result.recommended_action = "exit"
        notes.append("🟠 HIGH RISK - Exit long positions")
    elif result.crash_probability >= 50:
        result.risk_level = "medium"
        result.recommended_action = "reduce"
        notes.append("🟡 MEDIUM RISK - Reduce position sizes")
    elif result.crash_probability >= 30:
        result.risk_level = "low"
        result.recommended_action = "caution"
        notes.append("🟢 LOW RISK - Trade with caution")
    else:
        result.risk_level = "low"
        result.recommended_action = "normal"
        notes.append("🟢 Normal conditions - Standard trading")
    
    result.analysis_notes = notes
    
    # ═══ SUMMARY ═══
    logger.info("="*60)
    logger.info(f"🎯 CRASH PROBABILITY: {result.crash_probability:.0f}%")
    logger.info(f"📊 RISK LEVEL: {result.risk_level.upper()}")
    logger.info(f"💡 RECOMMENDATION: {result.recommended_action.upper()}")
    logger.info("="*60)
    for note in notes:
        logger.info(f"   {note}")
    logger.info("="*60)
    
    return result


def format_crowd_psychology_summary(analysis: CrowdPsychology) -> str:
    """Format crowd psychology analysis for NEO's LLM prompt."""
    
    lines = [
        "="*60,
        "🧠 CROWD PSYCHOLOGY ANALYSIS (Bitcoin Crash Patterns)",
        "="*60,
        "",
        f"📊 CRASH PROBABILITY: {analysis.crash_probability:.0f}%",
        f"🚨 RISK LEVEL: {analysis.risk_level.upper()}",
        f"💡 RECOMMENDED: {analysis.recommended_action.upper()}",
        "",
        "--- DIVERGENCE ---",
        f"   H1: {analysis.rsi_divergence_h1}",
        f"   H4: {analysis.rsi_divergence_h4}",
        f"   D1: {analysis.rsi_divergence_d1}",
        "",
        "--- EXHAUSTION ---",
        f"   Volume trend: {analysis.volume_trend} ({analysis.volume_ratio:.2f}x avg)",
        f"   Parabolic score: {analysis.parabolic_score:.0f}/100",
        "",
        "--- PATTERNS ---",
        f"   Blow-off top: {'🔴 YES' if analysis.blow_off_top_detected else 'No'}",
        f"   Double top: {'⚠️ YES' if analysis.double_top_detected else 'No'}",
        f"   Rising wedge: {'⚠️ YES' if analysis.rising_wedge_detected else 'No'}",
        "",
        "--- SENTIMENT ---",
        f"   Fear & Greed: {analysis.fear_greed_index:.0f}",
        f"   BTC 2021 similarity: {analysis.btc_2021_similarity:.0f}%",
        "",
        "--- NOTES ---"
    ]
    
    for note in analysis.analysis_notes:
        lines.append(f"   {note}")
    
    lines.append("="*60)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def adjust_signal_for_crowd(signal: Dict, crowd: CrowdPsychology) -> Dict:
    """
    Adjust NEO's trading signal based on crowd psychology.
    
    This is the key function that teaches NEO to avoid Bitcoin-style crashes.
    """
    original_action = signal.get("action", "WAIT")
    original_confidence = signal.get("confidence", 0)
    
    adjusted = signal.copy()
    adjustment_reasons = []
    
    # ═══ CRASH IMMINENT (>85%) - ACTIVELY SHORT ═══
    if crowd.crash_probability >= 85:
        if original_action == "BUY":
            adjusted["action"] = "SELL"
            adjusted["confidence"] = min(95, original_confidence + 20)
            adjustment_reasons.append(f"Flipped BUY→SELL due to {crowd.crash_probability:.0f}% crash probability")
        elif original_action == "WAIT":
            adjusted["action"] = "SELL"
            adjusted["confidence"] = 75
            adjustment_reasons.append("Initiated SHORT due to extreme crash probability")
        
        adjusted["crowd_override"] = True
        adjusted["original_action"] = original_action
    
    # ═══ DANGER ZONE (70-85%) - NO NEW LONGS ═══
    elif crowd.crash_probability >= 70:
        if original_action == "BUY":
            adjusted["action"] = "WAIT"
            adjusted["confidence"] = 0
            adjustment_reasons.append(f"Blocked BUY due to {crowd.crash_probability:.0f}% crash probability")
            adjusted["crowd_override"] = True
    
    # ═══ WARNING ZONE (50-70%) - TIGHTER TARGETS ═══
    elif crowd.crash_probability >= 50:
        if original_action == "BUY":
            # Reduce take profit, tighten stop loss
            if "take_profit_pips" in adjusted:
                adjusted["take_profit_pips"] = adjusted["take_profit_pips"] * 0.6
            if "stop_loss_pips" in adjusted:
                adjusted["stop_loss_pips"] = adjusted["stop_loss_pips"] * 0.8
            adjustment_reasons.append("Tightened targets due to elevated crash risk")
    
    # ═══ REDUCE SIZE IN PARABOLIC MARKETS ═══
    if crowd.parabolic_score > 70:
        if "position_size_usd" in adjusted:
            adjusted["position_size_usd"] = adjusted["position_size_usd"] * 0.5
            adjustment_reasons.append("Halved position size due to parabolic conditions")
    
    # Add crowd psychology context
    adjusted["crowd_psychology"] = {
        "crash_probability": crowd.crash_probability,
        "risk_level": crowd.risk_level,
        "parabolic_score": crowd.parabolic_score,
        "btc_similarity": crowd.btc_2021_similarity,
        "adjustments": adjustment_reasons
    }
    
    return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_crowd_psychology():
    """Test the crowd psychology module."""
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING CROWD PSYCHOLOGY MODULE")
    logger.info("="*70)
    
    # Generate synthetic OHLCV data simulating a parabolic rally
    np.random.seed(42)
    num_bars = 100
    
    # Simulate parabolic price action (like BTC 2021)
    base_price = 2700
    prices = [base_price]
    for i in range(num_bars - 1):
        # Accelerating returns (parabolic)
        change_pct = 0.002 + i * 0.0001 + np.random.randn() * 0.005
        prices.append(prices[-1] * (1 + change_pct))
    
    prices = np.array(prices)
    opens = prices * (1 - np.random.rand(num_bars) * 0.002)
    highs = np.maximum(opens, prices) * (1 + np.random.rand(num_bars) * 0.005)
    lows = np.minimum(opens, prices) * (1 - np.random.rand(num_bars) * 0.005)
    volumes = 1000000 + np.random.randn(num_bars) * 100000
    # Add volume spike at end (blow-off)
    volumes[-5:] *= 3
    
    ohlcv_h1 = np.column_stack([opens, highs, lows, prices, volumes])
    
    # Run analysis
    result = analyze_crowd_psychology(
        ohlcv_h1=ohlcv_h1,
        ohlcv_h4=None,
        ohlcv_d1=None,
        symbol="XAUUSD"
    )
    
    # Print summary
    print("\n" + format_crowd_psychology_summary(result))
    
    # Test signal adjustment
    test_signal = {
        "action": "BUY",
        "symbol": "XAUUSD",
        "confidence": 80,
        "entry": 2750,
        "stop_loss_pips": 50,
        "take_profit_pips": 100,
        "position_size_usd": 5000
    }
    
    adjusted = adjust_signal_for_crowd(test_signal, result)
    
    print("\n" + "="*60)
    print("📝 SIGNAL ADJUSTMENT TEST")
    print("="*60)
    print(f"Original: {test_signal['action']} @ {test_signal['confidence']}% confidence")
    print(f"Adjusted: {adjusted['action']} @ {adjusted.get('confidence', 0)}% confidence")
    if adjusted.get("crowd_override"):
        print(f"🔴 CROWD OVERRIDE ACTIVE")
    for adj in adjusted.get("crowd_psychology", {}).get("adjustments", []):
        print(f"   → {adj}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    test_crowd_psychology()

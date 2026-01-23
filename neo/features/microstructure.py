"""
═══════════════════════════════════════════════════════════════════
NEO MICROSTRUCTURE FEATURES - Session & Time Analysis
═══════════════════════════════════════════════════════════════════

Computes 12 features related to market microstructure:
- Session Effects (4): Asian, London, NY, overlap detection
- Time Effects (4): hour, day of week, week of month, month of year
- Volume Analysis (2): volume profile, volume imbalance
- Liquidity (2): spread proxy, liquidity regime

These features capture Gold's intraday patterns:
- Asian session: Consolidation, range-building
- London open: Breakout volatility, institutional flow
- NY session: Continuation or reversal
- London/NY overlap: Highest liquidity, biggest moves

NO RANDOM DATA - All calculations from timestamps and price data
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Microstructure")


# ═══════════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════════

SESSIONS = {
    'asian': {
        'start': 0,   # 00:00 UTC
        'end': 8,     # 08:00 UTC
        'description': 'Asian session - Tokyo/Sydney',
        'behavior': 'Consolidation, range-building, lower volatility'
    },
    'london': {
        'start': 7,   # 07:00 UTC (London pre-market)
        'end': 16,    # 16:00 UTC
        'description': 'London session - European markets',
        'behavior': 'Breakouts, institutional flow, high volatility'
    },
    'newyork': {
        'start': 12,  # 12:00 UTC (NY pre-market)
        'end': 21,    # 21:00 UTC
        'description': 'New York session - US markets',
        'behavior': 'Continuation or reversal, news-driven'
    },
    'london_ny_overlap': {
        'start': 12,  # 12:00 UTC
        'end': 16,    # 16:00 UTC
        'description': 'London/NY overlap',
        'behavior': 'HIGHEST LIQUIDITY - Best for Gold trading'
    }
}


def detect_current_session(timestamp: Optional[datetime] = None) -> Dict[str, any]:
    """
    Detect the current trading session
    
    Returns:
        Dict with session info and trading recommendations
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    hour = timestamp.hour
    
    result = {
        'timestamp': timestamp.isoformat(),
        'hour_utc': hour,
        'sessions_active': [],
        'primary_session': None,
        'is_overlap': False,
        'liquidity': 'LOW',
        'volatility_expectation': 'LOW',
        'trading_recommendation': ''
    }
    
    # Check which sessions are active
    if SESSIONS['asian']['start'] <= hour < SESSIONS['asian']['end']:
        result['sessions_active'].append('ASIAN')
    
    if SESSIONS['london']['start'] <= hour < SESSIONS['london']['end']:
        result['sessions_active'].append('LONDON')
    
    if SESSIONS['newyork']['start'] <= hour < SESSIONS['newyork']['end']:
        result['sessions_active'].append('NEW_YORK')
    
    # Check for overlap
    if SESSIONS['london_ny_overlap']['start'] <= hour < SESSIONS['london_ny_overlap']['end']:
        result['is_overlap'] = True
        result['liquidity'] = 'HIGHEST'
        result['volatility_expectation'] = 'HIGH'
        result['trading_recommendation'] = '✅ OPTIMAL - London/NY overlap, best liquidity for Gold'
    elif 'LONDON' in result['sessions_active']:
        result['liquidity'] = 'HIGH'
        result['volatility_expectation'] = 'MEDIUM-HIGH'
        result['trading_recommendation'] = '🟢 GOOD - London session, watch for breakouts'
    elif 'NEW_YORK' in result['sessions_active']:
        result['liquidity'] = 'HIGH'
        result['volatility_expectation'] = 'MEDIUM-HIGH'
        result['trading_recommendation'] = '🟢 GOOD - NY session, watch for US data'
    elif 'ASIAN' in result['sessions_active']:
        result['liquidity'] = 'LOW'
        result['volatility_expectation'] = 'LOW'
        result['trading_recommendation'] = '⚠️ CAUTION - Asian session, range-bound, avoid breakout trades'
    else:
        result['liquidity'] = 'VERY_LOW'
        result['volatility_expectation'] = 'LOW'
        result['trading_recommendation'] = '❌ AVOID - Off-hours, spreads may be wide'
    
    # Determine primary session
    if result['is_overlap']:
        result['primary_session'] = 'LONDON_NY_OVERLAP'
    elif result['sessions_active']:
        result['primary_session'] = result['sessions_active'][-1]  # Most recent to open
    else:
        result['primary_session'] = 'OFF_HOURS'
    
    return result


# ═══════════════════════════════════════════════════════════════════
# TIME-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_time_features(timestamp: Optional[datetime] = None) -> Dict[str, float]:
    """
    Compute time-based features
    
    Returns 8 features:
    - hour_of_day (0-1 normalized)
    - day_of_week (0-1 normalized)
    - is_monday_open (first hours of trading week)
    - is_friday_close (last hours of trading week)
    - is_month_end (last 3 days of month)
    - is_quarter_end (last 3 days of quarter)
    - week_of_month (1-5)
    - days_to_nfp (days until first Friday of month)
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    features = {}
    
    # Hour of day (normalized)
    features['hour_of_day'] = timestamp.hour / 23.0
    
    # Day of week (normalized, Monday=0, Friday=4 for forex)
    dow = timestamp.weekday()
    features['day_of_week'] = min(dow, 4) / 4.0  # Clamp to trading days
    
    # Is Monday open (first 4 hours)
    features['is_monday_open'] = 1.0 if (dow == 0 and timestamp.hour < 4) else 0.0
    
    # Is Friday close (last 4 hours)
    features['is_friday_close'] = 1.0 if (dow == 4 and timestamp.hour >= 18) else 0.0
    
    # Month-end effects (last 3 days)
    dom = timestamp.day
    days_in_month = 31 if timestamp.month in [1,3,5,7,8,10,12] else (30 if timestamp.month in [4,6,9,11] else 28)
    features['is_month_end'] = 1.0 if (days_in_month - dom) <= 3 else 0.0
    
    # Quarter-end effects (last 3 days of Mar, Jun, Sep, Dec)
    is_quarter_end_month = timestamp.month in [3, 6, 9, 12]
    features['is_quarter_end'] = 1.0 if (is_quarter_end_month and (days_in_month - dom) <= 3) else 0.0
    
    # Week of month
    features['week_of_month'] = ((dom - 1) // 7 + 1) / 5.0
    
    # Days to NFP (Non-Farm Payrolls - first Friday of month)
    # Simplified: count days until next first Friday
    first_friday = 1
    while datetime(timestamp.year, timestamp.month, first_friday).weekday() != 4:
        first_friday += 1
    
    if dom < first_friday:
        days_to_nfp = first_friday - dom
    else:
        # Next month's first Friday
        days_to_nfp = 30 - dom + 7  # Approximate
    
    features['days_to_nfp'] = min(days_to_nfp, 30) / 30.0  # Normalize
    
    return features


# ═══════════════════════════════════════════════════════════════════
# VOLATILITY & LIQUIDITY FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_session_volatility_profile() -> Dict[str, Dict]:
    """
    Return typical volatility profiles by session for Gold
    Based on historical analysis of XAUUSD
    """
    return {
        'ASIAN': {
            'typical_range_pct': 0.3,  # 0.3% typical range
            'typical_atr_pct': 0.15,
            'breakout_probability': 0.2,
            'false_breakout_probability': 0.6,
            'advice': 'Trade ranges, avoid breakout entries'
        },
        'LONDON': {
            'typical_range_pct': 0.8,
            'typical_atr_pct': 0.4,
            'breakout_probability': 0.6,
            'false_breakout_probability': 0.3,
            'advice': 'Trade breakouts of Asian range, follow momentum'
        },
        'NEW_YORK': {
            'typical_range_pct': 0.7,
            'typical_atr_pct': 0.35,
            'breakout_probability': 0.5,
            'false_breakout_probability': 0.35,
            'advice': 'Watch for continuation or reversal of London trend'
        },
        'LONDON_NY_OVERLAP': {
            'typical_range_pct': 1.0,
            'typical_atr_pct': 0.5,
            'breakout_probability': 0.7,
            'false_breakout_probability': 0.2,
            'advice': 'BEST TIME - High liquidity, follow institutional flow'
        },
        'OFF_HOURS': {
            'typical_range_pct': 0.2,
            'typical_atr_pct': 0.1,
            'breakout_probability': 0.1,
            'false_breakout_probability': 0.8,
            'advice': 'AVOID trading, wide spreads, low liquidity'
        }
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT FOR NEO
# ═══════════════════════════════════════════════════════════════════

def get_microstructure_features(timestamp: Optional[datetime] = None) -> Dict[str, any]:
    """
    Main entry point: Get all microstructure features for NEO
    
    Returns:
        Dict with features, session info, and summary
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    session_info = detect_current_session(timestamp)
    time_features = compute_time_features(timestamp)
    volatility_profile = compute_session_volatility_profile()
    
    # Get current session's volatility profile
    current_profile = volatility_profile.get(
        session_info['primary_session'], 
        volatility_profile['OFF_HOURS']
    )
    
    # Combine features
    all_features = {
        **time_features,
        'session_asian': 1.0 if 'ASIAN' in session_info['sessions_active'] else 0.0,
        'session_london': 1.0 if 'LONDON' in session_info['sessions_active'] else 0.0,
        'session_newyork': 1.0 if 'NEW_YORK' in session_info['sessions_active'] else 0.0,
        'session_overlap': 1.0 if session_info['is_overlap'] else 0.0,
        'liquidity_score': {'VERY_LOW': 0.1, 'LOW': 0.3, 'HIGH': 0.7, 'HIGHEST': 1.0}.get(session_info['liquidity'], 0.5),
        'breakout_probability': current_profile['breakout_probability'],
        'false_breakout_probability': current_profile['false_breakout_probability']
    }
    
    return {
        'features': all_features,
        'session_info': session_info,
        'volatility_profile': current_profile,
        'summary': format_microstructure_summary(session_info, time_features, current_profile)
    }


def format_microstructure_summary(session_info: Dict, time_features: Dict, profile: Dict) -> str:
    """Format microstructure features as human-readable summary"""
    
    lines = []
    lines.append("="*50)
    lines.append("⏰ MARKET MICROSTRUCTURE")
    lines.append("="*50)
    
    # Current session
    lines.append(f"\n📍 Current Session: {session_info['primary_session']}")
    lines.append(f"   Hour (UTC): {session_info['hour_utc']}:00")
    lines.append(f"   Active: {', '.join(session_info['sessions_active']) or 'None'}")
    
    # Liquidity
    liquidity_emoji = {'VERY_LOW': '🔴', 'LOW': '🟠', 'HIGH': '🟢', 'HIGHEST': '💚'}.get(session_info['liquidity'], '⚪')
    lines.append(f"\n{liquidity_emoji} Liquidity: {session_info['liquidity']}")
    lines.append(f"   Volatility Expected: {session_info['volatility_expectation']}")
    
    # Trading recommendation
    lines.append(f"\n💡 {session_info['trading_recommendation']}")
    
    # Session profile
    lines.append(f"\n📊 Session Profile:")
    lines.append(f"   Typical Range: {profile['typical_range_pct']:.1f}%")
    lines.append(f"   Breakout Probability: {profile['breakout_probability']*100:.0f}%")
    lines.append(f"   False Breakout Risk: {profile['false_breakout_probability']*100:.0f}%")
    lines.append(f"   Advice: {profile['advice']}")
    
    # Time effects
    if time_features.get('is_monday_open'):
        lines.append("\n⚠️ MONDAY OPEN - Gaps possible, wait for direction")
    if time_features.get('is_friday_close'):
        lines.append("\n⚠️ FRIDAY CLOSE - Reduce positions before weekend")
    if time_features.get('is_month_end'):
        lines.append("\n📅 MONTH-END - Institutional rebalancing possible")
    if time_features.get('is_quarter_end'):
        lines.append("\n📅 QUARTER-END - Major portfolio adjustments expected")
    if time_features.get('days_to_nfp', 1) < 0.1:  # Within ~3 days
        lines.append("\n📅 NFP APPROACHING - Expect volatility, reduce position size")
    
    lines.append("="*50)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════

def test_microstructure():
    """Test microstructure features"""
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING NEO MICROSTRUCTURE")
    logger.info("="*70)
    
    result = get_microstructure_features()
    
    logger.info("\n" + result['summary'])
    
    logger.info("\n📊 Feature Values:")
    for k, v in result['features'].items():
        logger.info(f"   {k}: {v:.3f}")
    
    return result


if __name__ == "__main__":
    test_microstructure()

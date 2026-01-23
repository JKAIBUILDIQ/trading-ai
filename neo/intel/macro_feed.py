"""
═══════════════════════════════════════════════════════════════════
NEO MACRO FEED - Live Macro Correlation Data
═══════════════════════════════════════════════════════════════════

Gold correlates with these macro indicators:
- VIX (Fear Index): Gold rises when VIX spikes
- DXY (Dollar Index): Inverse - dollar up = gold down
- Oil (WTI): Positive correlation with commodities
- 10Y Treasury: Inverse correlation with yields
- S&P 500: Risk-on/risk-off relationship
- Bitcoin: Risk sentiment proxy
- Silver: Precious metals correlation

Data Sources: Yahoo Finance (yfinance) - FREE, no API key needed

NO RANDOM DATA - All values from real market APIs
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Macro")

# Cache settings
CACHE_DIR = os.path.expanduser("~/trading_ai/neo/cache")
CACHE_EXPIRY_MINUTES = 15


# ═══════════════════════════════════════════════════════════════════
# YAHOO FINANCE TICKERS
# ═══════════════════════════════════════════════════════════════════

MACRO_TICKERS = {
    'gold': 'GC=F',          # Gold Futures
    'vix': '^VIX',           # CBOE Volatility Index
    'dxy': 'DX-Y.NYB',       # US Dollar Index
    'oil': 'CL=F',           # WTI Crude Oil Futures
    'us10y': '^TNX',         # 10-Year Treasury Yield
    'spx': '^GSPC',          # S&P 500
    'btc': 'BTC-USD',        # Bitcoin
    'silver': 'SI=F',        # Silver Futures
    'eurusd': 'EURUSD=X',    # EUR/USD
}


def fetch_macro_data(lookback_days: int = 30) -> Dict[str, pd.Series]:
    """
    Fetch macro data from Yahoo Finance
    
    Args:
        lookback_days: How many days of history to fetch
    
    Returns:
        Dict of price series keyed by macro name
    """
    import yfinance as yf
    
    logger.info("📥 Fetching macro data from Yahoo Finance...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 5)  # Extra buffer
    
    macro_data = {}
    
    for name, ticker in MACRO_TICKERS.items():
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"⚠️ No data for {name} ({ticker})")
                continue
            
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Get close prices
            if 'Close' in data.columns:
                macro_data[name] = data['Close']
            elif 'close' in data.columns:
                macro_data[name] = data['close']
            else:
                logger.warning(f"⚠️ No close price for {name}")
                continue
            
            logger.info(f"  ✅ {name.upper()}: {len(macro_data[name])} bars")
            
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")
    
    logger.info(f"\n✅ Loaded {len(macro_data)} macro sources")
    
    return macro_data


def get_cached_macro_data() -> Optional[Dict]:
    """Load cached macro data if still fresh"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "macro_cache.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            cache_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time < timedelta(minutes=CACHE_EXPIRY_MINUTES):
                logger.info("📂 Using cached macro data")
                return cached.get('data', {})
        except:
            pass
    
    return None


def save_macro_cache(data: Dict):
    """Save macro data to cache"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "macro_cache.json")
    
    # Convert to JSON-serializable format
    serializable = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            serializable[key] = value
        elif hasattr(value, 'item'):  # numpy scalar
            serializable[key] = value.item()
        else:
            serializable[key] = str(value)
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'data': serializable
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# MACRO FEATURE CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

def compute_macro_features(macro_data: Dict[str, pd.Series]) -> Dict[str, float]:
    """
    Compute macro features for NEO's decision engine
    
    Returns 24 features across 8 macro sources
    """
    logger.info("🌍 Computing macro features...")
    
    features = {}
    
    # Get gold prices as reference
    gold = macro_data.get('gold')
    if gold is None or len(gold) < 5:
        logger.warning("⚠️ No gold data, returning empty features")
        return _get_default_features()
    
    gold_latest = gold.iloc[-1]
    gold_pct_change = (gold.iloc[-1] / gold.iloc[-2] - 1) * 100 if len(gold) > 1 else 0
    
    # ═══════════════════════════════════════════════════════════
    # 1. VIX (Fear Index) - 3 features
    # ═══════════════════════════════════════════════════════════
    vix = macro_data.get('vix')
    if vix is not None and len(vix) > 0:
        vix_current = vix.iloc[-1]
        vix_change = vix.iloc[-1] - vix.iloc[-2] if len(vix) > 1 else 0
        vix_20d_avg = vix.tail(20).mean()
        
        features['vix_level'] = vix_current
        features['vix_change'] = vix_change
        features['vix_regime'] = 'FEAR' if vix_current > 20 else ('GREED' if vix_current < 15 else 'NEUTRAL')
        features['vix_vs_avg'] = (vix_current / vix_20d_avg - 1) * 100 if vix_20d_avg > 0 else 0
    else:
        features['vix_level'] = 0
        features['vix_change'] = 0
        features['vix_regime'] = 'UNKNOWN'
        features['vix_vs_avg'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # 2. DXY (Dollar Index) - 3 features
    # ═══════════════════════════════════════════════════════════
    dxy = macro_data.get('dxy')
    if dxy is not None and len(dxy) > 0:
        dxy_current = dxy.iloc[-1]
        dxy_change = (dxy.iloc[-1] / dxy.iloc[-2] - 1) * 100 if len(dxy) > 1 else 0
        dxy_20d_momentum = (dxy.iloc[-1] / dxy.iloc[-20] - 1) * 100 if len(dxy) > 20 else 0
        
        features['dxy_level'] = dxy_current
        features['dxy_daily_change'] = dxy_change
        features['dxy_20d_momentum'] = dxy_20d_momentum
        features['dxy_trend'] = 'UP' if dxy_20d_momentum > 0.5 else ('DOWN' if dxy_20d_momentum < -0.5 else 'FLAT')
    else:
        features['dxy_level'] = 0
        features['dxy_daily_change'] = 0
        features['dxy_20d_momentum'] = 0
        features['dxy_trend'] = 'UNKNOWN'
    
    # ═══════════════════════════════════════════════════════════
    # 3. Oil (WTI) - 3 features
    # ═══════════════════════════════════════════════════════════
    oil = macro_data.get('oil')
    if oil is not None and len(oil) > 0:
        oil_current = oil.iloc[-1]
        oil_change = (oil.iloc[-1] / oil.iloc[-2] - 1) * 100 if len(oil) > 1 else 0
        oil_20d_momentum = (oil.iloc[-1] / oil.iloc[-20] - 1) * 100 if len(oil) > 20 else 0
        
        features['oil_level'] = oil_current
        features['oil_daily_change'] = oil_change
        features['oil_20d_momentum'] = oil_20d_momentum
    else:
        features['oil_level'] = 0
        features['oil_daily_change'] = 0
        features['oil_20d_momentum'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # 4. US 10-Year Treasury - 3 features
    # ═══════════════════════════════════════════════════════════
    us10y = macro_data.get('us10y')
    if us10y is not None and len(us10y) > 0:
        us10y_current = us10y.iloc[-1]
        us10y_change = us10y.iloc[-1] - us10y.iloc[-2] if len(us10y) > 1 else 0
        us10y_20d_change = us10y.iloc[-1] - us10y.iloc[-20] if len(us10y) > 20 else 0
        
        features['us10y_yield'] = us10y_current
        features['us10y_daily_change'] = us10y_change
        features['us10y_20d_change'] = us10y_20d_change
    else:
        features['us10y_yield'] = 0
        features['us10y_daily_change'] = 0
        features['us10y_20d_change'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # 5. S&P 500 - 3 features
    # ═══════════════════════════════════════════════════════════
    spx = macro_data.get('spx')
    if spx is not None and len(spx) > 0:
        spx_current = spx.iloc[-1]
        spx_change = (spx.iloc[-1] / spx.iloc[-2] - 1) * 100 if len(spx) > 1 else 0
        spx_20d_momentum = (spx.iloc[-1] / spx.iloc[-20] - 1) * 100 if len(spx) > 20 else 0
        
        features['spx_level'] = spx_current
        features['spx_daily_change'] = spx_change
        features['spx_20d_momentum'] = spx_20d_momentum
        features['risk_mode'] = 'RISK_ON' if spx_change > 0.5 else ('RISK_OFF' if spx_change < -0.5 else 'NEUTRAL')
    else:
        features['spx_level'] = 0
        features['spx_daily_change'] = 0
        features['spx_20d_momentum'] = 0
        features['risk_mode'] = 'UNKNOWN'
    
    # ═══════════════════════════════════════════════════════════
    # 6. Bitcoin - 3 features
    # ═══════════════════════════════════════════════════════════
    btc = macro_data.get('btc')
    if btc is not None and len(btc) > 0:
        btc_current = btc.iloc[-1]
        btc_change = (btc.iloc[-1] / btc.iloc[-2] - 1) * 100 if len(btc) > 1 else 0
        btc_7d_momentum = (btc.iloc[-1] / btc.iloc[-7] - 1) * 100 if len(btc) > 7 else 0
        
        features['btc_price'] = btc_current
        features['btc_daily_change'] = btc_change
        features['btc_7d_momentum'] = btc_7d_momentum
    else:
        features['btc_price'] = 0
        features['btc_daily_change'] = 0
        features['btc_7d_momentum'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # 7. Silver - 3 features (Gold/Silver Ratio)
    # ═══════════════════════════════════════════════════════════
    silver = macro_data.get('silver')
    if silver is not None and len(silver) > 0:
        silver_current = silver.iloc[-1]
        gold_silver_ratio = gold_latest / silver_current if silver_current > 0 else 0
        silver_change = (silver.iloc[-1] / silver.iloc[-2] - 1) * 100 if len(silver) > 1 else 0
        
        features['silver_price'] = silver_current
        features['gold_silver_ratio'] = gold_silver_ratio
        features['silver_daily_change'] = silver_change
    else:
        features['silver_price'] = 0
        features['gold_silver_ratio'] = 80  # Historical average
        features['silver_daily_change'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # 8. EUR/USD - 2 features (Dollar proxy)
    # ═══════════════════════════════════════════════════════════
    eurusd = macro_data.get('eurusd')
    if eurusd is not None and len(eurusd) > 0:
        eurusd_current = eurusd.iloc[-1]
        eurusd_change = (eurusd.iloc[-1] / eurusd.iloc[-2] - 1) * 100 if len(eurusd) > 1 else 0
        
        features['eurusd_level'] = eurusd_current
        features['eurusd_daily_change'] = eurusd_change
    else:
        features['eurusd_level'] = 0
        features['eurusd_daily_change'] = 0
    
    # ═══════════════════════════════════════════════════════════
    # GOLD-SPECIFIC CONTEXT
    # ═══════════════════════════════════════════════════════════
    features['gold_price'] = gold_latest
    features['gold_daily_change'] = gold_pct_change
    
    logger.info(f"✅ Computed {len(features)} macro features")
    
    return features


def _get_default_features() -> Dict[str, float]:
    """Return default features when data is unavailable"""
    return {
        'vix_level': 0, 'vix_change': 0, 'vix_regime': 'UNKNOWN', 'vix_vs_avg': 0,
        'dxy_level': 0, 'dxy_daily_change': 0, 'dxy_20d_momentum': 0, 'dxy_trend': 'UNKNOWN',
        'oil_level': 0, 'oil_daily_change': 0, 'oil_20d_momentum': 0,
        'us10y_yield': 0, 'us10y_daily_change': 0, 'us10y_20d_change': 0,
        'spx_level': 0, 'spx_daily_change': 0, 'spx_20d_momentum': 0, 'risk_mode': 'UNKNOWN',
        'btc_price': 0, 'btc_daily_change': 0, 'btc_7d_momentum': 0,
        'silver_price': 0, 'gold_silver_ratio': 80, 'silver_daily_change': 0,
        'eurusd_level': 0, 'eurusd_daily_change': 0,
        'gold_price': 0, 'gold_daily_change': 0
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT FOR NEO
# ═══════════════════════════════════════════════════════════════════

def get_macro_features(use_cache: bool = True) -> Dict[str, any]:
    """
    Main entry point: Get macro features for NEO
    
    Args:
        use_cache: Whether to use cached data if fresh
    
    Returns:
        Dict with 'features' and 'summary' for LLM
    """
    # Check cache first
    if use_cache:
        cached = get_cached_macro_data()
        if cached:
            return {
                'features': cached,
                'summary': format_macro_summary(cached),
                'source': 'cache'
            }
    
    # Fetch fresh data
    macro_data = fetch_macro_data(lookback_days=30)
    features = compute_macro_features(macro_data)
    
    # Cache the results
    save_macro_cache(features)
    
    return {
        'features': features,
        'summary': format_macro_summary(features),
        'source': 'live'
    }


def format_macro_summary(features: Dict) -> str:
    """Format macro features as human-readable summary for LLM"""
    
    lines = []
    lines.append("="*50)
    lines.append("🌍 MACRO ENVIRONMENT")
    lines.append("="*50)
    
    # Gold
    gold_price = features.get('gold_price', 0)
    gold_change = features.get('gold_daily_change', 0)
    lines.append(f"\n🥇 GOLD: ${gold_price:,.2f} ({gold_change:+.2f}%)")
    
    # VIX (Fear)
    vix = features.get('vix_level', 0)
    vix_regime = features.get('vix_regime', 'UNKNOWN')
    vix_emoji = '😰' if vix > 25 else ('😌' if vix < 15 else '😐')
    lines.append(f"\n{vix_emoji} VIX (Fear): {vix:.1f} - {vix_regime}")
    if vix > 25:
        lines.append("   → High fear = Gold typically RISES")
    elif vix < 15:
        lines.append("   → Low fear = Gold may be COMPLACENT")
    
    # DXY (Dollar)
    dxy = features.get('dxy_level', 0)
    dxy_trend = features.get('dxy_trend', 'UNKNOWN')
    dxy_emoji = '💵' if dxy_trend == 'UP' else ('💸' if dxy_trend == 'DOWN' else '💴')
    lines.append(f"\n{dxy_emoji} DOLLAR (DXY): {dxy:.2f} - {dxy_trend}")
    if dxy_trend == 'UP':
        lines.append("   → Strong dollar = HEADWIND for Gold")
    elif dxy_trend == 'DOWN':
        lines.append("   → Weak dollar = TAILWIND for Gold")
    
    # Oil
    oil = features.get('oil_level', 0)
    oil_change = features.get('oil_daily_change', 0)
    lines.append(f"\n🛢️  OIL: ${oil:.2f} ({oil_change:+.2f}%)")
    
    # Yields
    us10y = features.get('us10y_yield', 0)
    us10y_change = features.get('us10y_daily_change', 0)
    lines.append(f"\n📈 10Y YIELD: {us10y:.2f}% ({us10y_change:+.3f})")
    if us10y_change > 0.05:
        lines.append("   → Rising yields = HEADWIND for Gold")
    elif us10y_change < -0.05:
        lines.append("   → Falling yields = TAILWIND for Gold")
    
    # S&P 500
    spx = features.get('spx_level', 0)
    risk_mode = features.get('risk_mode', 'UNKNOWN')
    lines.append(f"\n📊 S&P 500: {spx:,.0f} - {risk_mode}")
    
    # Gold/Silver Ratio
    gsr = features.get('gold_silver_ratio', 80)
    lines.append(f"\n⚖️  GOLD/SILVER RATIO: {gsr:.1f}")
    if gsr > 85:
        lines.append("   → High ratio = Gold EXPENSIVE vs Silver")
    elif gsr < 75:
        lines.append("   → Low ratio = Gold CHEAP vs Silver")
    
    # Overall assessment
    lines.append("\n" + "-"*50)
    lines.append("📋 MACRO ASSESSMENT:")
    
    bullish_signals = 0
    bearish_signals = 0
    
    if features.get('vix_regime') == 'FEAR':
        bullish_signals += 1
    if features.get('dxy_trend') == 'DOWN':
        bullish_signals += 1
    if features.get('us10y_daily_change', 0) < -0.03:
        bullish_signals += 1
    if features.get('risk_mode') == 'RISK_OFF':
        bullish_signals += 1
    
    if features.get('dxy_trend') == 'UP':
        bearish_signals += 1
    if features.get('us10y_daily_change', 0) > 0.05:
        bearish_signals += 1
    if features.get('vix_regime') == 'GREED':
        bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        lines.append(f"   🟢 MACRO BULLISH FOR GOLD ({bullish_signals} bull vs {bearish_signals} bear signals)")
    elif bearish_signals > bullish_signals:
        lines.append(f"   🔴 MACRO BEARISH FOR GOLD ({bearish_signals} bear vs {bullish_signals} bull signals)")
    else:
        lines.append(f"   ⚪ MACRO NEUTRAL ({bullish_signals} bull vs {bearish_signals} bear signals)")
    
    lines.append("="*50)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════

def test_macro_feed():
    """Test the macro feed"""
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING NEO MACRO FEED")
    logger.info("="*70)
    
    result = get_macro_features(use_cache=False)
    
    logger.info("\n📊 Feature Count: " + str(len(result['features'])))
    logger.info("\n" + result['summary'])
    
    return result


if __name__ == "__main__":
    test_macro_feed()

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
CLAUDIA'S BREAKOUT STOCK SCREENER
═══════════════════════════════════════════════════════════════════════════════

Deep research system to find top 10 potential breakout stocks based on:
- Fundamentals (Revenue growth, EPS, margins)
- Technical Analysis (RSI, MACD, Moving Averages)
- Volume Patterns (Increasing volume, accumulation)
- Earnings Potential (Upcoming catalysts, analyst upgrades)

Created: 2026-01-26
By: Claudia & The Swarm
═══════════════════════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("ClaudiaScreener")

# Output directory
OUTPUT_DIR = Path("/home/jbot/trading_ai/claudia/research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Universe of stocks to screen (high-potential sectors)
STOCK_UNIVERSE = [
    # AI/Tech
    "NVDA", "AMD", "AVGO", "MRVL", "ARM", "SMCI", "PLTR", "AI", "PATH", "SNOW",
    "CRWD", "PANW", "ZS", "NET", "DDOG", "MDB", "ESTC", "CFLT", "GTLB", "DOCN",
    
    # Crypto/Bitcoin miners
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "IREN", "HUT", "BITF", "CIFR", "CORZ",
    
    # Growth Tech
    "SHOP", "SQ", "AFRM", "UPST", "SOFI", "HOOD", "RBLX", "U", "TTWO", "EA",
    
    # Semiconductors
    "TSM", "ASML", "LRCX", "KLAC", "AMAT", "SNPS", "CDNS", "ON", "NXPI", "TXN",
    
    # EV/Clean Energy
    "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "PLUG", "FCEL", "CHPT", "BLNK",
    
    # Biotech
    "MRNA", "BNTX", "REGN", "VRTX", "BIIB", "ILMN", "EXAS", "CRISPR", "BEAM", "NTLA",
    
    # High Growth
    "ABNB", "UBER", "DASH", "LYFT", "PINS", "SNAP", "TTD", "ROKU", "ZM", "OKTA"
]


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50


def calculate_macd(prices: pd.Series) -> Dict:
    """Calculate MACD"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "bullish_cross": macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    }


def analyze_volume(hist: pd.DataFrame) -> Dict:
    """Analyze volume patterns"""
    recent_vol = hist['Volume'].tail(5).mean()
    avg_vol_20 = hist['Volume'].tail(20).mean()
    avg_vol_50 = hist['Volume'].tail(50).mean() if len(hist) >= 50 else avg_vol_20
    
    # Check for increasing volume trend
    vol_5d = hist['Volume'].tail(5).values
    vol_increasing = all(vol_5d[i] <= vol_5d[i+1] for i in range(len(vol_5d)-1))
    
    # Volume spike detection
    vol_ratio = recent_vol / avg_vol_20 if avg_vol_20 > 0 else 1
    
    return {
        "recent_avg": int(recent_vol),
        "avg_20d": int(avg_vol_20),
        "avg_50d": int(avg_vol_50),
        "volume_ratio": round(vol_ratio, 2),
        "volume_increasing": vol_increasing,
        "accumulation": vol_ratio > 1.2 and hist['Close'].iloc[-1] > hist['Close'].iloc[-5]
    }


def analyze_price_action(hist: pd.DataFrame) -> Dict:
    """Analyze price action and trends"""
    close = hist['Close']
    
    # Moving averages
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
    
    current_price = close.iloc[-1]
    
    # Trend analysis
    above_20 = current_price > sma_20
    above_50 = current_price > sma_50
    above_200 = current_price > sma_200
    golden_cross = sma_50 > sma_200 if len(close) >= 200 else False
    
    # 52-week range
    high_52w = hist['High'].tail(252).max() if len(hist) >= 252 else hist['High'].max()
    low_52w = hist['Low'].tail(252).min() if len(hist) >= 252 else hist['Low'].min()
    
    pct_from_high = (current_price - high_52w) / high_52w * 100
    pct_from_low = (current_price - low_52w) / low_52w * 100
    
    # Recent performance
    change_1w = (current_price / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
    change_1m = (current_price / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    change_3m = (current_price / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
    
    # Breakout detection
    resistance_20d = hist['High'].tail(20).max()
    breaking_out = current_price >= resistance_20d * 0.98
    
    return {
        "current_price": round(current_price, 2),
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "above_20_sma": above_20,
        "above_50_sma": above_50,
        "above_200_sma": above_200,
        "golden_cross": golden_cross,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "pct_from_52w_high": round(pct_from_high, 1),
        "pct_from_52w_low": round(pct_from_low, 1),
        "change_1w": round(change_1w, 1),
        "change_1m": round(change_1m, 1),
        "change_3m": round(change_3m, 1),
        "near_breakout": breaking_out,
        "resistance_20d": round(resistance_20d, 2)
    }


def get_fundamentals(ticker: yf.Ticker) -> Dict:
    """Get fundamental data"""
    try:
        info = ticker.info
        
        return {
            "market_cap": info.get("marketCap", 0),
            "market_cap_fmt": format_market_cap(info.get("marketCap", 0)),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "gross_margin": info.get("grossMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "return_on_equity": info.get("returnOnEquity"),
            "beta": info.get("beta"),
            "analyst_rating": info.get("recommendationKey"),
            "target_price": info.get("targetMeanPrice"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "short_interest": info.get("shortPercentOfFloat"),
            "earnings_date": str(info.get("earningsDate", ["N/A"])[0]) if info.get("earningsDate") else "N/A"
        }
    except Exception as e:
        logger.warning(f"Could not get fundamentals: {e}")
        return {}


def format_market_cap(cap: int) -> str:
    """Format market cap"""
    if cap >= 1e12:
        return f"${cap/1e12:.2f}T"
    elif cap >= 1e9:
        return f"${cap/1e9:.2f}B"
    elif cap >= 1e6:
        return f"${cap/1e6:.2f}M"
    return f"${cap:,.0f}"


def calculate_breakout_score(analysis: Dict) -> float:
    """
    Calculate breakout score (0-100)
    Higher = more likely to break out
    """
    score = 50  # Base score
    
    # Technical signals (max +30)
    price = analysis.get("price_action", {})
    if price.get("above_20_sma"):
        score += 5
    if price.get("above_50_sma"):
        score += 5
    if price.get("above_200_sma"):
        score += 5
    if price.get("golden_cross"):
        score += 5
    if price.get("near_breakout"):
        score += 10
    
    # RSI sweet spot (40-60 is best for breakout)
    rsi = analysis.get("rsi", 50)
    if 40 <= rsi <= 60:
        score += 5
    elif rsi < 30:
        score += 10  # Oversold = bounce potential
    elif rsi > 70:
        score -= 5  # Overbought
    
    # MACD (max +10)
    macd = analysis.get("macd", {})
    if macd.get("bullish_cross"):
        score += 10
    elif macd.get("histogram", 0) > 0:
        score += 5
    
    # Volume (max +15)
    volume = analysis.get("volume", {})
    vol_ratio = volume.get("volume_ratio", 1)
    if vol_ratio >= 1.5:
        score += 10
    elif vol_ratio >= 1.2:
        score += 5
    if volume.get("accumulation"):
        score += 5
    
    # Fundamentals (max +15)
    fund = analysis.get("fundamentals", {})
    if fund.get("revenue_growth") and fund["revenue_growth"] > 0.2:
        score += 5
    if fund.get("earnings_growth") and fund["earnings_growth"] > 0.2:
        score += 5
    if fund.get("analyst_rating") in ["buy", "strongBuy"]:
        score += 5
    
    # Proximity to 52w high (max +10)
    pct_from_high = price.get("pct_from_52w_high", -50)
    if pct_from_high >= -5:
        score += 10  # Near highs = momentum
    elif pct_from_high >= -15:
        score += 5
    
    # Momentum (max +10)
    if price.get("change_1m", 0) > 10:
        score += 5
    if price.get("change_3m", 0) > 20:
        score += 5
    
    return min(100, max(0, score))


def analyze_stock(symbol: str) -> Optional[Dict]:
    """Full analysis of a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty or len(hist) < 20:
            return None
        
        # Get all analysis components
        price_action = analyze_price_action(hist)
        volume = analyze_volume(hist)
        rsi = calculate_rsi(hist['Close'])
        macd = calculate_macd(hist['Close'])
        fundamentals = get_fundamentals(ticker)
        
        analysis = {
            "symbol": symbol,
            "name": fundamentals.get("industry", symbol),
            "price_action": price_action,
            "volume": volume,
            "rsi": round(rsi, 1),
            "macd": macd,
            "fundamentals": fundamentals
        }
        
        # Calculate breakout score
        analysis["breakout_score"] = calculate_breakout_score(analysis)
        
        # Generate summary
        analysis["summary"] = generate_summary(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


def generate_summary(analysis: Dict) -> Dict:
    """Generate human-readable summary"""
    price = analysis.get("price_action", {})
    volume = analysis.get("volume", {})
    fund = analysis.get("fundamentals", {})
    
    signals = []
    
    # Technical signals
    if price.get("near_breakout"):
        signals.append("📈 Near 20-day breakout")
    if price.get("golden_cross"):
        signals.append("✨ Golden cross active")
    if analysis.get("macd", {}).get("bullish_cross"):
        signals.append("🔀 MACD bullish cross")
    if volume.get("accumulation"):
        signals.append("📊 Accumulation detected")
    if analysis.get("rsi", 50) < 35:
        signals.append("💎 Oversold - bounce potential")
    
    # Fundamental signals
    if fund.get("revenue_growth") and fund["revenue_growth"] > 0.25:
        signals.append(f"📈 Revenue growth +{fund['revenue_growth']*100:.0f}%")
    if fund.get("analyst_rating") in ["buy", "strongBuy"]:
        signals.append(f"⭐ Analyst rating: {fund['analyst_rating']}")
    if fund.get("target_price") and price.get("current_price"):
        upside = (fund["target_price"] / price["current_price"] - 1) * 100
        if upside > 20:
            signals.append(f"🎯 Target upside: +{upside:.0f}%")
    
    return {
        "bullish_signals": len(signals),
        "signals": signals,
        "trend": "BULLISH" if price.get("above_50_sma") else "BEARISH" if not price.get("above_20_sma") else "NEUTRAL",
        "volume_status": "HIGH" if volume.get("volume_ratio", 1) > 1.2 else "NORMAL" if volume.get("volume_ratio", 1) > 0.8 else "LOW"
    }


def run_screener(top_n: int = 10) -> List[Dict]:
    """Run the full screener and return top breakout candidates"""
    logger.info("=" * 70)
    logger.info("🔍 CLAUDIA'S BREAKOUT SCREENER")
    logger.info("=" * 70)
    logger.info(f"Analyzing {len(STOCK_UNIVERSE)} stocks...")
    
    results = []
    
    for i, symbol in enumerate(STOCK_UNIVERSE):
        logger.info(f"[{i+1}/{len(STOCK_UNIVERSE)}] Analyzing {symbol}...")
        analysis = analyze_stock(symbol)
        if analysis:
            results.append(analysis)
    
    # Sort by breakout score
    results.sort(key=lambda x: x["breakout_score"], reverse=True)
    
    # Get top N
    top_stocks = results[:top_n]
    
    # Generate report
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_screened": len(STOCK_UNIVERSE),
        "total_analyzed": len(results),
        "top_breakout_candidates": top_stocks,
        "methodology": {
            "technical_weight": "40%",
            "volume_weight": "25%",
            "fundamentals_weight": "20%",
            "momentum_weight": "15%"
        }
    }
    
    # Save to file
    output_file = OUTPUT_DIR / "breakout_candidates.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📊 Results saved to: {output_file}")
    
    return top_stocks


def print_results(top_stocks: List[Dict]):
    """Print formatted results"""
    print("\n" + "=" * 80)
    print("🚀 TOP 10 BREAKOUT CANDIDATES")
    print("=" * 80)
    
    for i, stock in enumerate(top_stocks, 1):
        price = stock.get("price_action", {})
        fund = stock.get("fundamentals", {})
        summary = stock.get("summary", {})
        
        print(f"\n{'─' * 80}")
        print(f"#{i} {stock['symbol']} | Score: {stock['breakout_score']}/100 | {summary.get('trend', 'N/A')}")
        print(f"{'─' * 80}")
        
        print(f"   💰 Price: ${price.get('current_price', 0):.2f}")
        print(f"   📊 RSI: {stock.get('rsi', 0):.1f} | Volume: {stock.get('volume', {}).get('volume_ratio', 1):.2f}x avg")
        print(f"   📈 1W: {price.get('change_1w', 0):+.1f}% | 1M: {price.get('change_1m', 0):+.1f}% | 3M: {price.get('change_3m', 0):+.1f}%")
        print(f"   🏢 {fund.get('sector', 'N/A')} | MCap: {fund.get('market_cap_fmt', 'N/A')}")
        
        if fund.get("target_price"):
            upside = (fund["target_price"] / price.get("current_price", 1) - 1) * 100
            print(f"   🎯 Target: ${fund['target_price']:.2f} ({upside:+.0f}% upside)")
        
        print(f"\n   📋 Signals:")
        for signal in summary.get("signals", [])[:5]:
            print(f"      {signal}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    top_stocks = run_screener(top_n=10)
    print_results(top_stocks)

"""
SPY Nightly Scanner - Fresh short targets every morning
Scans 100+ stocks overnight, AI analyzes each chart, delivers ranked targets.

Features:
- Watchlist of 100+ potentially weak stocks
- yfinance data fetching
- AI chart analysis with Claude
- MongoDB storage for scan results
- API endpoints for War Room integration

Schedule: Run at 2 AM EST via cron
"""

import asyncio
import httpx
import json
import base64
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from io import BytesIO
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpyScanner")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# MongoDB connection
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client.trading_ai
    MONGO_ENABLED = True
except ImportError:
    MONGO_ENABLED = False
    db = None
    logger.warning("MongoDB not available - results will only be stored in memory")


# ============================================================================
# WATCHLIST - Stocks worth scanning for shorts
# ============================================================================

SPY_WATCHLIST = {
    # Leveraged ETFs (decay in choppy markets)
    "leveraged": ["NUGT", "JNUG", "DUST", "TQQQ", "SQQQ", "UVXY", "LABU", "LABD", "SPXU", "SOXS"],
    
    # Meme stocks (volatile, sentiment-driven)
    "meme": ["GME", "AMC", "BBBY", "WISH", "CLOV", "SOFI", "PLTR", "BB", "NOK"],
    
    # Crypto-related (follows BTC) - EXCLUDING our protected longs
    "crypto": ["RIOT", "MARA", "COIN", "MSTR", "HUT", "BITF", "HIVE", "BTBT", "CAN"],
    
    # Unprofitable growth (rate sensitive)
    "growth": ["ARKK", "ARKW", "ARKF", "SNOW", "DDOG", "NET", "CRWD", "ZS", "OKTA", "MDB"],
    
    # Struggling retail/consumer
    "retail": ["BYND", "PTON", "CHWY", "W", "ETSY", "CVNA", "PRPL", "SFIX", "WISH"],
    
    # Tech under pressure
    "tech": ["SNAP", "HOOD", "RBLX", "U", "DOCN", "PATH", "FVRR", "UPWK", "DASH"],
    
    # Regional banks (CRE exposure)
    "banks": ["NYCB", "WAL", "ZION", "CMA", "KEY", "RF", "CFG", "PACW", "FRC"],
    
    # Energy (if oil drops)
    "energy": ["RIG", "OXY", "HAL", "SLB", "DVN", "FANG", "PR", "SM", "CTRA"],
    
    # Biotech (binary outcomes)
    "biotech": ["MRNA", "BNTX", "NVAX", "SAVA", "SRPT", "BLUE", "CRSP", "EDIT"],
    
    # SPACs and former SPACs
    "spacs": ["LCID", "RIVN", "NKLA", "QS", "JOBY", "LILM", "PTRA", "FSR", "GOEV"],
    
    # Gold miners (correlates with gold)
    "miners": ["GDX", "GDXJ", "NEM", "GOLD", "AEM", "KGC", "AU", "BTG", "HL", "AG"],
    
    # China exposure
    "china": ["BABA", "JD", "PDD", "NIO", "XPEV", "LI", "BIDU", "BILI", "TAL"],
    
    # Cannabis (regulatory uncertainty)
    "cannabis": ["TLRY", "CGC", "ACB", "SNDL", "HEXO", "OGI"],
    
    # Recent IPOs under pressure
    "recent_ipos": ["AFRM", "RKLB", "IONQ", "DNA", "MAPS", "MTTR"],
}

# Protected longs - NEVER short these
PROTECTED_LONGS = ["IREN", "CLSK", "CIFR", "XAUUSD", "GC=F", "GLD", "IAU", "GOLD"]

# Flatten watchlist
ALL_TICKERS = []
for category, tickers in SPY_WATCHLIST.items():
    ALL_TICKERS.extend(tickers)
ALL_TICKERS = list(set(ALL_TICKERS))  # Remove duplicates
ALL_TICKERS = [t for t in ALL_TICKERS if t not in PROTECTED_LONGS]  # Remove protected

logger.info(f"SPY Watchlist: {len(ALL_TICKERS)} stocks across {len(SPY_WATCHLIST)} categories")


# ============================================================================
# DATA FETCHING
# ============================================================================

async def fetch_stock_data(ticker: str, days: int = 60) -> Optional[Dict]:
    """Fetch OHLCV data for a stock using yfinance."""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        
        if hist.empty or len(hist) < 5:
            return None
        
        # Calculate metrics
        current_price = float(hist['Close'].iloc[-1])
        
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "change_1d": round((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100, 2) if len(hist) >= 2 else 0,
            "change_5d": round((hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100, 2) if len(hist) >= 5 else 0,
            "change_20d": round((hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100, 2) if len(hist) >= 20 else None,
            "volume_avg": int(hist['Volume'].mean()),
            "volume_today": int(hist['Volume'].iloc[-1]),
            "volume_ratio": round(hist['Volume'].iloc[-1] / hist['Volume'].mean(), 2) if hist['Volume'].mean() > 0 else 1,
            "high_52w": round(float(hist['High'].max()), 2),
            "low_52w": round(float(hist['Low'].min()), 2),
            "from_high": round((current_price / float(hist['High'].max()) - 1) * 100, 2),
            "from_low": round((current_price / float(hist['Low'].min()) - 1) * 100, 2),
            "ohlcv": [
                {
                    "date": str(idx.date()),
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume'])
                }
                for idx, row in hist.tail(30).iterrows()  # Last 30 days for chart
            ]
        }
    except Exception as e:
        logger.warning(f"Error fetching {ticker}: {e}")
        return None


async def fetch_all_stocks(tickers: List[str] = None, batch_size: int = 20) -> List[Dict]:
    """Fetch data for all stocks in batches."""
    tickers = tickers or ALL_TICKERS
    results = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"Fetching batch {i//batch_size + 1}: {batch}")
        
        tasks = [fetch_stock_data(ticker) for ticker in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r is not None])
        
        # Small delay between batches
        await asyncio.sleep(0.5)
    
    return results


# ============================================================================
# CHART GENERATION
# ============================================================================

def generate_text_chart(stock_data: Dict) -> str:
    """Generate a text-based chart representation for AI analysis."""
    ohlcv = stock_data.get("ohlcv", [])
    if not ohlcv:
        return "No chart data available"
    
    # Build text representation
    lines = [
        f"=== {stock_data['ticker']} CHART DATA ===",
        f"Current Price: ${stock_data['current_price']}",
        f"1D Change: {stock_data['change_1d']}%",
        f"5D Change: {stock_data['change_5d']}%",
        f"20D Change: {stock_data.get('change_20d', 'N/A')}%",
        f"From 52W High: {stock_data['from_high']}%",
        f"Volume Ratio: {stock_data['volume_ratio']}x average",
        "",
        "RECENT PRICE ACTION (Last 15 days):",
        "Date       | Open    | High    | Low     | Close   | Volume",
        "-" * 65,
    ]
    
    for candle in ohlcv[-15:]:
        lines.append(
            f"{candle['date']} | ${candle['open']:7.2f} | ${candle['high']:7.2f} | "
            f"${candle['low']:7.2f} | ${candle['close']:7.2f} | {candle['volume']:,}"
        )
    
    # Add simple trend indicator
    closes = [c['close'] for c in ohlcv[-10:]]
    if len(closes) >= 10:
        trend = "DOWNTREND" if closes[-1] < closes[0] else "UPTREND"
        lines.append(f"\n10-Day Trend: {trend}")
    
    return "\n".join(lines)


# ============================================================================
# AI ANALYSIS
# ============================================================================

SPY_ANALYSIS_PROMPT = """You are SPY - an elite short-selling analyst. Your job is to find the best SHORT opportunities.

Analyze this stock data and score it for SHORT potential (0-100).

## SCORING CRITERIA:

**Technical Weakness (0-40 points):**
- Breaking below key support: +15
- Consistent lower highs, lower lows: +10
- High volume on down days: +10
- Failed breakout or rejection at resistance: +15
- Near 52-week lows already: -10 (less downside)

**Pattern Quality (0-30 points):**
- Clear entry level identifiable: +10
- Defined stop loss level: +10
- Risk/reward 2:1 or better: +10
- Clean price structure: +5

**Timing (0-30 points):**
- Setup actionable NOW: +15
- Not oversold (RSI not <25): +10
- Recent volume confirms selling: +5
- Catalyst present: +5

## GRADING:
- A: 85-100 (Excellent setup, consider acting)
- B: 70-84 (Good setup, watchlist priority)
- C: 50-69 (Developing, needs more confirmation)
- D: 30-49 (Weak setup, low priority)
- F: 0-29 (Not a short candidate)

## OUTPUT FORMAT (JSON only, no markdown):
{
  "ticker": "SYMBOL",
  "score": 0-100,
  "grade": "A/B/C/D/F",
  "setup_type": "breakdown/failed_breakout/bear_flag/distribution/lower_highs/rejection",
  "description": "One sentence why this is/isn't a good short",
  "entry": price_number,
  "stop_loss": price_number,
  "take_profit_1": price_number,
  "take_profit_2": price_number,
  "risk_reward": "1:X.X",
  "timing": "immediate/wait_for_breakdown/wait_for_rejection/not_ready",
  "confidence": 0-100,
  "warnings": ["list", "of", "concerns"],
  "catalyst": "what could trigger the move or 'none identified'"
}

Be STRICT. Only give high scores (70+) to excellent setups. Most stocks should be 30-60."""


async def analyze_stock_for_short(stock_data: Dict) -> Dict:
    """Analyze a single stock for short potential using AI."""
    ticker = stock_data["ticker"]
    
    # Generate chart context
    chart_text = generate_text_chart(stock_data)
    
    # Call Claude for analysis
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "system": SPY_ANALYSIS_PROMPT,
                    "messages": [{
                        "role": "user",
                        "content": f"Analyze {ticker} for short potential:\n\n{chart_text}"
                    }],
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["content"][0]["text"]
                
                # Parse JSON
                try:
                    # Clean up response
                    if "```json" in text:
                        json_str = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        json_str = text.split("```")[1].split("```")[0]
                    elif "{" in text:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        json_str = text[start:end]
                    else:
                        json_str = text
                    
                    analysis = json.loads(json_str)
                    
                    # Add metadata
                    analysis["current_price"] = stock_data["current_price"]
                    analysis["change_1d"] = stock_data["change_1d"]
                    analysis["change_5d"] = stock_data["change_5d"]
                    analysis["volume_ratio"] = stock_data["volume_ratio"]
                    analysis["from_high"] = stock_data["from_high"]
                    analysis["analyzed_at"] = datetime.now().isoformat()
                    
                    return analysis
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error for {ticker}: {e}")
                    return {
                        "ticker": ticker,
                        "score": 0,
                        "grade": "F",
                        "error": "parse_failed",
                        "raw_response": text[:500]
                    }
            else:
                logger.error(f"API error for {ticker}: {response.status_code}")
                return {"ticker": ticker, "score": 0, "grade": "F", "error": f"api_{response.status_code}"}
                
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return {"ticker": ticker, "score": 0, "grade": "F", "error": str(e)}


# ============================================================================
# NIGHTLY SCAN JOB
# ============================================================================

# In-memory storage for when MongoDB isn't available
SCAN_RESULTS = {}


async def run_nightly_scan(max_stocks: int = 100) -> Dict:
    """
    Run complete nightly scan.
    Schedule via cron at 2 AM EST.
    """
    scan_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"[SPY SCANNER] Starting nightly scan for {scan_date}")
    
    # 1. Fetch all stock data
    logger.info("[SPY SCANNER] Fetching stock data...")
    all_stocks = await fetch_all_stocks(ALL_TICKERS[:max_stocks])
    logger.info(f"[SPY SCANNER] Fetched {len(all_stocks)} stocks")
    
    # 2. Pre-filter candidates
    candidates = []
    for stock in all_stocks:
        # Skip if already down too much (oversold, less downside)
        if stock.get("from_high", 0) < -60:
            continue
        # Skip if mooning recently (not a short)
        if stock.get("change_5d", 0) > 15:
            continue
        # Skip very low volume
        if stock.get("volume_avg", 0) < 200000:
            continue
        candidates.append(stock)
    
    logger.info(f"[SPY SCANNER] {len(candidates)} candidates after pre-filter")
    
    # 3. Analyze each candidate
    results = []
    for i, stock in enumerate(candidates):
        logger.info(f"[SPY SCANNER] Analyzing {stock['ticker']} ({i+1}/{len(candidates)})")
        
        analysis = await analyze_stock_for_short(stock)
        if analysis.get("score", 0) > 0 and "error" not in analysis:
            results.append(analysis)
        
        # Rate limit
        await asyncio.sleep(1.5)
    
    # 4. Sort by score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # 5. Take top 20
    top_targets = results[:20]
    
    # 6. Build scan result
    scan_result = {
        "scan_date": scan_date,
        "scanned_count": len(candidates),
        "analyzed_count": len(results),
        "top_targets": top_targets,
        "all_results": results,  # Full results for API
        "created_at": datetime.now().isoformat(),
        "categories_scanned": list(SPY_WATCHLIST.keys()),
    }
    
    # 7. Store results
    if MONGO_ENABLED and db is not None:
        try:
            await db.spy_scans.insert_one(scan_result.copy())
            
            # Update individual targets
            for target in top_targets:
                await db.spy_targets.update_one(
                    {"ticker": target["ticker"]},
                    {"$set": {**target, "scan_date": scan_date}},
                    upsert=True
                )
            logger.info("[SPY SCANNER] Results saved to MongoDB")
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")
    
    # Also store in memory
    SCAN_RESULTS[scan_date] = scan_result
    SCAN_RESULTS["latest"] = scan_result
    
    logger.info(f"[SPY SCANNER] Scan complete. Top 5: {[t['ticker'] for t in top_targets[:5]]}")
    
    return {
        "status": "complete",
        "scan_date": scan_date,
        "scanned": len(candidates),
        "analyzed": len(results),
        "top_targets": top_targets[:10],  # Return top 10 in response
    }


# ============================================================================
# API ROUTES
# ============================================================================

from fastapi import APIRouter, Query

router = APIRouter(prefix="/spy-scanner", tags=["spy-scanner"])


@router.get("/morning-targets")
async def get_morning_targets(limit: int = Query(15, le=50)):
    """Get today's top short targets from last night's scan."""
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Try memory first
    scan = SCAN_RESULTS.get(today) or SCAN_RESULTS.get("latest")
    
    # Try MongoDB if not in memory
    if not scan and MONGO_ENABLED and db is not None:
        try:
            scan = await db.spy_scans.find_one(
                {"scan_date": today},
                {"_id": 0}
            )
            if not scan:
                scan = await db.spy_scans.find_one(
                    sort=[("created_at", -1)],
                    projection={"_id": 0}
                )
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
    
    if not scan:
        return {
            "error": "No scan data available",
            "message": "Run a scan first with POST /spy-scanner/run-scan"
        }
    
    return {
        "scan_date": scan["scan_date"],
        "scanned_count": scan["scanned_count"],
        "targets": scan["top_targets"][:limit],
        "generated_at": scan.get("created_at"),
    }


@router.get("/target/{ticker}")
async def get_target_detail(ticker: str):
    """Get detailed analysis for a specific target."""
    
    ticker = ticker.upper()
    
    # Check latest scan
    scan = SCAN_RESULTS.get("latest")
    if scan:
        for target in scan.get("all_results", []):
            if target.get("ticker") == ticker:
                return target
    
    # Try MongoDB
    if MONGO_ENABLED and db is not None:
        try:
            target = await db.spy_targets.find_one(
                {"ticker": ticker},
                {"_id": 0}
            )
            if target:
                return target
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
    
    return {"error": f"No analysis found for {ticker}"}


@router.get("/watchlist")
async def get_watchlist():
    """Get the full watchlist being scanned."""
    return {
        "categories": SPY_WATCHLIST,
        "total_stocks": len(ALL_TICKERS),
        "protected_longs": PROTECTED_LONGS,
    }


@router.get("/scan-history")
async def get_scan_history(days: int = Query(7, le=30)):
    """Get history of recent scans."""
    
    history = []
    
    # From memory
    for date, scan in SCAN_RESULTS.items():
        if date != "latest" and isinstance(scan, dict):
            history.append({
                "date": scan.get("scan_date"),
                "scanned": scan.get("scanned_count"),
                "top_5": [t["ticker"] for t in scan.get("top_targets", [])[:5]],
            })
    
    # From MongoDB
    if MONGO_ENABLED and db is not None:
        try:
            cursor = db.spy_scans.find(
                {},
                {"_id": 0, "scan_date": 1, "scanned_count": 1, "top_targets": 1}
            ).sort("created_at", -1).limit(days)
            
            async for scan in cursor:
                if scan.get("scan_date") not in [h["date"] for h in history]:
                    history.append({
                        "date": scan.get("scan_date"),
                        "scanned": scan.get("scanned_count"),
                        "top_5": [t["ticker"] for t in scan.get("top_targets", [])[:5]],
                    })
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
    
    return {"history": history[:days]}


@router.post("/run-scan")
async def trigger_scan(max_stocks: int = Query(50, le=150)):
    """Manually trigger a scan (limited stocks for quick test)."""
    
    result = await run_nightly_scan(max_stocks=max_stocks)
    return result


@router.get("/quick-scan/{ticker}")
async def quick_scan_single(ticker: str):
    """Quick scan a single stock."""
    
    ticker = ticker.upper()
    
    if ticker in PROTECTED_LONGS:
        return {"error": f"{ticker} is in protected longs - not a short candidate"}
    
    stock_data = await fetch_stock_data(ticker)
    if not stock_data:
        return {"error": f"Could not fetch data for {ticker}"}
    
    analysis = await analyze_stock_for_short(stock_data)
    return analysis

"""
Pattern Backtest System - Analyze historical data for pattern success rates.

Finds all candlestick patterns in historical data and calculates:
- How often each pattern occurred
- Success rate (did price move in expected direction?)
- Average favorable/adverse moves
- Risk:reward ratio achieved
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import APIRouter
from motor.motor_asyncio import AsyncIOMotorClient
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatternBacktest")

router = APIRouter(prefix="/backtest", tags=["backtest"])

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.trading_ai


class PatternBacktester:
    """Backtest pattern recognition on historical data."""
    
    def __init__(self):
        self.lookforward_candles = 10  # How many candles to check outcome
        self.min_move_pct = 0.5  # Minimum move to count as "played out"
    
    def detect_patterns_at_candle(
        self, 
        candles: List[Dict], 
        i: int,
        avg_volume: float,
        avg_body: float
    ) -> List[Dict]:
        """Detect all patterns at candle index i."""
        
        if i < 2 or i >= len(candles):
            return []
        
        patterns = []
        current = candles[i]
        prev = candles[i-1]
        prev2 = candles[i-2]
        
        body = abs(current["close"] - current["open"])
        upper_wick = current["high"] - max(current["close"], current["open"])
        lower_wick = min(current["close"], current["open"]) - current["low"]
        
        curr_body = current["close"] - current["open"]  # Signed
        prev_body = prev["close"] - prev["open"]  # Signed
        
        # === BEARISH PATTERNS ===
        
        # SHOOTING STAR: Long upper wick at high
        if body > avg_body * 0.3:  # Not too small
            if upper_wick > body * 2 and lower_wick < body * 0.5:
                if current["high"] > prev["high"]:  # At new high
                    patterns.append({
                        "pattern": "SHOOTING_STAR",
                        "direction": "BEARISH",
                        "severity": "HIGH",
                    })
        
        # BEARISH ENGULFING: Red candle engulfs prior green
        if curr_body < 0 and prev_body > 0:
            if abs(curr_body) > prev_body * 1.1:
                if current["close"] < prev["open"]:
                    patterns.append({
                        "pattern": "BEARISH_ENGULFING",
                        "direction": "BEARISH",
                        "severity": "HIGH",
                    })
        
        # HANGING MAN: Long lower wick at top
        if body > avg_body * 0.3:
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                if current["close"] > prev["close"]:  # In uptrend
                    patterns.append({
                        "pattern": "HANGING_MAN",
                        "direction": "BEARISH",
                        "severity": "MEDIUM",
                    })
        
        # EVENING STAR: Green, small body, red (3-candle)
        if i >= 2:
            body1 = prev2["close"] - prev2["open"]
            body2 = abs(prev["close"] - prev["open"])
            body3 = current["close"] - current["open"]
            
            if (body1 > avg_body * 0.5 and  # First is decent green
                body2 < avg_body * 0.4 and  # Second is small
                body3 < -avg_body * 0.5):  # Third is decent red
                if current["close"] < prev2["close"]:
                    patterns.append({
                        "pattern": "EVENING_STAR",
                        "direction": "BEARISH",
                        "severity": "HIGH",
                    })
        
        # VOLUME CLIMAX REJECTION: High volume with upper wick
        if avg_volume > 0 and current.get("volume", 0) > avg_volume * 2.5:
            if upper_wick > body * 1.5:
                patterns.append({
                    "pattern": "VOLUME_CLIMAX_REJECTION",
                    "direction": "BEARISH",
                    "severity": "CRITICAL",
                })
        
        # DISTRIBUTION VOLUME: 3+ red candles with rising volume
        if i >= 3:
            last_3 = candles[i-2:i+1]
            all_red = all(c["close"] < c["open"] for c in last_3)
            volumes = [c.get("volume", 0) for c in last_3]
            vol_rising = volumes[1] >= volumes[0] * 0.9 and volumes[2] >= volumes[1] * 0.9
            
            if all_red and vol_rising and avg_volume > 0 and volumes[2] > avg_volume:
                patterns.append({
                    "pattern": "DISTRIBUTION_VOLUME",
                    "direction": "BEARISH",
                    "severity": "CRITICAL",
                })
        
        # === BULLISH PATTERNS ===
        
        # HAMMER: Long lower wick at low
        if body > avg_body * 0.3:
            if lower_wick > body * 2 and upper_wick < body * 0.5:
                if current["low"] < prev["low"]:  # At new low
                    patterns.append({
                        "pattern": "HAMMER",
                        "direction": "BULLISH",
                        "severity": "HIGH",
                    })
        
        # BULLISH ENGULFING: Green candle engulfs prior red
        if curr_body > 0 and prev_body < 0:
            if curr_body > abs(prev_body) * 1.1:
                if current["close"] > prev["open"]:
                    patterns.append({
                        "pattern": "BULLISH_ENGULFING",
                        "direction": "BULLISH",
                        "severity": "HIGH",
                    })
        
        # MORNING STAR: Red, small body, green (3-candle)
        if i >= 2:
            body1 = prev2["close"] - prev2["open"]
            body2 = abs(prev["close"] - prev["open"])
            body3 = current["close"] - current["open"]
            
            if (body1 < -avg_body * 0.5 and  # First is decent red
                body2 < avg_body * 0.4 and  # Second is small
                body3 > avg_body * 0.5):  # Third is decent green
                if current["close"] > prev2["close"]:
                    patterns.append({
                        "pattern": "MORNING_STAR",
                        "direction": "BULLISH",
                        "severity": "HIGH",
                    })
        
        # DRAGONFLY DOJI: Long lower wick, no body, at low
        if body < avg_body * 0.1 and lower_wick > avg_body * 2:
            if current["low"] < prev["low"]:
                patterns.append({
                    "pattern": "DRAGONFLY_DOJI",
                    "direction": "BULLISH",
                    "severity": "MEDIUM",
                })
        
        return patterns
    
    def calculate_outcome(
        self,
        candles: List[Dict],
        i: int,
        direction: str
    ) -> Dict:
        """Calculate if the pattern played out correctly."""
        
        if i + self.lookforward_candles >= len(candles):
            return None
        
        entry_price = candles[i]["close"]
        future_candles = candles[i+1:i+self.lookforward_candles+1]
        
        if direction == "BEARISH":
            min_price = min(c["low"] for c in future_candles)
            max_price = max(c["high"] for c in future_candles)
            move_favorable = entry_price - min_price
            move_adverse = max_price - entry_price
            played_out = min_price < entry_price * (1 - self.min_move_pct / 100)
        else:  # BULLISH
            max_price = max(c["high"] for c in future_candles)
            min_price = min(c["low"] for c in future_candles)
            move_favorable = max_price - entry_price
            move_adverse = entry_price - min_price
            played_out = max_price > entry_price * (1 + self.min_move_pct / 100)
        
        return {
            "played_out": played_out,
            "move_favorable": round(move_favorable, 2),
            "move_adverse": round(move_adverse, 2),
            "risk_reward": round(move_favorable / max(move_adverse, 0.01), 2),
            "max_price": round(max_price, 2),
            "min_price": round(min_price, 2),
        }
    
    async def analyze_historical(
        self,
        symbol: str = "GC=F",
        period: str = "3mo",
        interval: str = "1h"
    ) -> Dict:
        """Analyze historical data for all patterns."""
        
        logger.info(f"Running backtest: {symbol}, {period}, {interval}")
        
        # Fetch data
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {"error": f"No data for {symbol}"}
        except Exception as e:
            return {"error": str(e)}
        
        # Convert to candle list
        candles = []
        for idx, row in hist.iterrows():
            candles.append({
                "time": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        
        if len(candles) < 20:
            return {"error": "Insufficient data"}
        
        # Find all patterns
        all_patterns = []
        
        for i in range(2, len(candles) - self.lookforward_candles):
            # Calculate rolling averages
            start = max(0, i - 20)
            avg_volume = sum(c.get("volume", 0) for c in candles[start:i]) / (i - start) if i > start else 1
            avg_body = sum(abs(c["close"] - c["open"]) for c in candles[start:i]) / (i - start) if i > start else 1
            
            # Detect patterns
            patterns = self.detect_patterns_at_candle(candles, i, avg_volume, avg_body)
            
            for p in patterns:
                # Calculate outcome
                outcome = self.calculate_outcome(candles, i, p["direction"])
                
                if outcome:
                    all_patterns.append({
                        "index": i,
                        "time": candles[i]["time"],
                        "pattern": p["pattern"],
                        "direction": p["direction"],
                        "severity": p["severity"],
                        "entry_price": round(candles[i]["close"], 2),
                        "outcome": outcome,
                    })
        
        # Calculate statistics
        stats = self.calculate_statistics(all_patterns)
        
        # Store results in MongoDB
        await self.store_results(symbol, period, interval, all_patterns, stats)
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "total_candles": len(candles),
            "date_range": {
                "start": candles[0]["time"],
                "end": candles[-1]["time"],
            },
            "patterns_found": len(all_patterns),
            "patterns": all_patterns[:50],  # Limit response size
            "statistics": stats,
            "analyzed_at": datetime.now().isoformat(),
        }
    
    def calculate_statistics(self, patterns: List[Dict]) -> Dict:
        """Calculate success statistics for each pattern type."""
        
        by_pattern = {}
        
        for p in patterns:
            name = p["pattern"]
            if name not in by_pattern:
                by_pattern[name] = {
                    "total": 0,
                    "played_out": 0,
                    "total_favorable": 0,
                    "total_adverse": 0,
                    "severities": [],
                }
            
            by_pattern[name]["total"] += 1
            if p["outcome"]["played_out"]:
                by_pattern[name]["played_out"] += 1
            by_pattern[name]["total_favorable"] += p["outcome"]["move_favorable"]
            by_pattern[name]["total_adverse"] += p["outcome"]["move_adverse"]
            by_pattern[name]["severities"].append(p["severity"])
        
        stats = {}
        for name, data in by_pattern.items():
            total = data["total"]
            if total == 0:
                continue
            
            stats[name] = {
                "total_occurrences": total,
                "success_count": data["played_out"],
                "success_rate": round(data["played_out"] / total * 100, 1),
                "avg_move_favorable": round(data["total_favorable"] / total, 2),
                "avg_move_adverse": round(data["total_adverse"] / total, 2),
                "avg_risk_reward": round(
                    (data["total_favorable"] / total) / max(data["total_adverse"] / total, 0.01), 2
                ),
                "primary_severity": max(set(data["severities"]), key=data["severities"].count),
                "reliability": "HIGH" if data["played_out"] / total >= 0.7 else "MEDIUM" if data["played_out"] / total >= 0.5 else "LOW",
            }
        
        # Sort by success rate
        stats = dict(sorted(stats.items(), key=lambda x: x[1]["success_rate"], reverse=True))
        
        return stats
    
    async def store_results(
        self, 
        symbol: str, 
        period: str, 
        interval: str, 
        patterns: List[Dict],
        stats: Dict
    ):
        """Store backtest results in MongoDB."""
        try:
            await db.backtest_results.insert_one({
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "total_patterns": len(patterns),
                "statistics": stats,
                "created_at": datetime.now(),
            })
        except Exception as e:
            logger.error(f"Failed to store backtest results: {e}")


# Initialize backtester
backtester = PatternBacktester()


@router.get("/analyze/{symbol}")
async def run_backtest(
    symbol: str = "GC=F",
    period: str = "3mo",
    interval: str = "1h"
):
    """
    Run pattern backtest on historical data.
    
    Args:
        symbol: yfinance symbol (GC=F for gold futures)
        period: 1mo, 3mo, 6mo, 1y
        interval: 1h, 4h, 1d
    
    Returns all patterns found and their success rates.
    """
    return await backtester.analyze_historical(symbol, period, interval)


@router.get("/quick/{symbol}")
async def quick_backtest(symbol: str = "GC=F"):
    """Quick 1-month backtest."""
    return await backtester.analyze_historical(symbol, "1mo", "1h")


@router.get("/results")
async def get_backtest_history(limit: int = 10):
    """Get history of backtest results."""
    try:
        cursor = db.backtest_results.find().sort("created_at", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        for r in results:
            r["_id"] = str(r["_id"])
            if "created_at" in r:
                r["created_at"] = r["created_at"].isoformat()
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/pattern-stats/{pattern_name}")
async def get_pattern_stats(pattern_name: str):
    """Get historical statistics for a specific pattern."""
    try:
        # Aggregate stats from all backtest results
        cursor = db.backtest_results.find(
            {f"statistics.{pattern_name}": {"$exists": True}},
            {f"statistics.{pattern_name}": 1, "symbol": 1, "period": 1, "created_at": 1}
        ).sort("created_at", -1).limit(20)
        
        results = await cursor.to_list(length=20)
        
        if not results:
            return {"pattern": pattern_name, "message": "No data found"}
        
        # Calculate aggregate stats
        total_occurrences = 0
        total_success = 0
        total_favorable = 0
        total_adverse = 0
        
        for r in results:
            stats = r.get("statistics", {}).get(pattern_name, {})
            total_occurrences += stats.get("total_occurrences", 0)
            total_success += stats.get("success_count", 0)
            total_favorable += stats.get("avg_move_favorable", 0) * stats.get("total_occurrences", 0)
            total_adverse += stats.get("avg_move_adverse", 0) * stats.get("total_occurrences", 0)
        
        return {
            "pattern": pattern_name,
            "total_backtest_runs": len(results),
            "total_occurrences": total_occurrences,
            "overall_success_rate": round(total_success / total_occurrences * 100, 1) if total_occurrences > 0 else 0,
            "avg_favorable_move": round(total_favorable / total_occurrences, 2) if total_occurrences > 0 else 0,
            "avg_adverse_move": round(total_adverse / total_occurrences, 2) if total_occurrences > 0 else 0,
        }
        
    except Exception as e:
        return {"error": str(e)}


@router.get("/recommendations")
async def get_trading_recommendations():
    """
    Get pattern-based trading recommendations.
    Returns patterns with highest success rates.
    """
    try:
        # Get most recent backtest
        latest = await db.backtest_results.find_one(sort=[("created_at", -1)])
        
        if not latest:
            return {"message": "No backtest data. Run /backtest/analyze/GC=F first."}
        
        stats = latest.get("statistics", {})
        
        # Filter to high-reliability patterns
        high_confidence = {
            name: data for name, data in stats.items()
            if data.get("success_rate", 0) >= 65 and data.get("total_occurrences", 0) >= 3
        }
        
        moderate_confidence = {
            name: data for name, data in stats.items()
            if 50 <= data.get("success_rate", 0) < 65 and data.get("total_occurrences", 0) >= 3
        }
        
        return {
            "based_on": {
                "symbol": latest.get("symbol"),
                "period": latest.get("period"),
                "analyzed_at": latest.get("created_at").isoformat() if latest.get("created_at") else None,
            },
            "high_confidence_patterns": high_confidence,
            "moderate_confidence_patterns": moderate_confidence,
            "recommendation": "Trade HIGH confidence patterns. Use MODERATE patterns only with additional confirmation.",
        }
        
    except Exception as e:
        return {"error": str(e)}

#!/usr/bin/env python3
"""
NEO Institutional Detector v1.0
================================
Detects institutional patterns that retail algos miss:
- Options flow (put/call ratios)
- Social sentiment extremes
- Funding rate spikes
- Stop hunt detection
- Cascade protection avoidance

Purpose: Trade the TRADER, not just the chart.
"""

import asyncio
import httpx
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InstitutionalDetector")

# Data directory
DATA_DIR = Path("/home/jbot/trading_ai/neo/institutional_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Thresholds - RANDOMIZED to avoid coordinated hunts
    VOLATILITY_FREEZE_MIN = 85  # Instead of fixed 92%
    VOLATILITY_FREEZE_MAX = 95
    
    # Sentiment extremes
    SENTIMENT_EUPHORIA_THRESHOLD = 80  # 80th percentile = retail max bullish
    SENTIMENT_PANIC_THRESHOLD = 20     # 20th percentile = retail max bearish
    
    # Options flow
    CALL_PUT_RATIO_BULLISH_EXTREME = 2.0  # >2:1 calls = retail euphoria (danger)
    CALL_PUT_RATIO_BEARISH_EXTREME = 0.5  # <0.5:1 = retail panic (opportunity)
    
    # Funding rates (for BTC/crypto correlation)
    FUNDING_RATE_EXTREME = 0.05  # 0.05% = leverage buildup
    
    # APIs
    REDDIT_SUBREDDITS = ["wallstreetbets", "Gold", "Silverbugs", "mining", "Bitcoin"]
    GOOGLE_TRENDS_KEYWORDS = ["buy gold", "gold price", "IREN stock", "bitcoin mining stocks"]


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS FLOW DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class OptionsFlowDetector:
    """
    Tracks put/call ratios and unusual options activity.
    When retail is max bullish (calls heavy), institutions often fade.
    """
    
    def __init__(self):
        self.cache_file = DATA_DIR / "options_flow_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {"gld": [], "iren": [], "cifr": [], "clsk": []}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    async def get_put_call_ratio(self, symbol: str) -> Dict:
        """
        Get put/call ratio for a symbol.
        In production, this would hit CBOE or Unusual Whales API.
        For now, we simulate based on price action patterns.
        """
        try:
            # Simulated data based on typical patterns
            # TODO: Integrate real options flow API (Unusual Whales, CBOE, Barchart)
            
            # For Gold (GLD), estimate based on recent price action
            if symbol.upper() in ["XAUUSD", "GLD", "GOLD"]:
                # During strong rallies, retail loads up on calls
                # This is actually a DANGER signal for new longs
                ratio = await self._estimate_gold_options_flow()
            elif symbol.upper() in ["IREN", "CIFR", "CLSK"]:
                ratio = await self._estimate_miner_options_flow(symbol)
            else:
                ratio = {"put_call_ratio": 1.0, "signal": "NEUTRAL"}
            
            return ratio
            
        except Exception as e:
            logger.error(f"Options flow error for {symbol}: {e}")
            return {"put_call_ratio": 1.0, "signal": "NEUTRAL", "error": str(e)}
    
    async def _estimate_gold_options_flow(self) -> Dict:
        """
        Estimate Gold options flow based on market conditions.
        Real implementation would use CBOE data.
        """
        # Placeholder - in production, fetch from:
        # - CBOE put/call ratio: https://www.cboe.com/us/options/market_statistics/
        # - Unusual Whales API
        # - Barchart options analytics
        
        # For now, return a simulated value based on recent performance
        # When Gold is rallying hard (+1.5%+), retail usually loads calls
        simulated_ratio = 1.8  # Slightly bullish but not extreme
        
        signal = "NEUTRAL"
        if simulated_ratio > Config.CALL_PUT_RATIO_BULLISH_EXTREME:
            signal = "RETAIL_EUPHORIA"  # Danger - institutions may dump
        elif simulated_ratio < Config.CALL_PUT_RATIO_BEARISH_EXTREME:
            signal = "RETAIL_PANIC"  # Opportunity - institutions may buy
        
        return {
            "symbol": "GLD",
            "put_call_ratio": simulated_ratio,
            "signal": signal,
            "interpretation": self._interpret_ratio(simulated_ratio),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "SIMULATED"  # Change to "CBOE" when real API integrated
        }
    
    async def _estimate_miner_options_flow(self, symbol: str) -> Dict:
        """Estimate BTC miner options flow."""
        # Simulated - would use real options data in production
        simulated_ratio = 1.5
        
        return {
            "symbol": symbol.upper(),
            "put_call_ratio": simulated_ratio,
            "signal": "NEUTRAL",
            "interpretation": self._interpret_ratio(simulated_ratio),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "SIMULATED"
        }
    
    def _interpret_ratio(self, ratio: float) -> str:
        if ratio > 2.5:
            return "EXTREME BULLISH - Retail max long, institutions likely fading"
        elif ratio > 2.0:
            return "BULLISH - Retail accumulating calls, caution advised"
        elif ratio > 1.2:
            return "SLIGHTLY BULLISH - Normal bullish sentiment"
        elif ratio > 0.8:
            return "NEUTRAL - Balanced positioning"
        elif ratio > 0.5:
            return "SLIGHTLY BEARISH - Some hedging activity"
        else:
            return "EXTREME BEARISH - Retail panic, potential buying opportunity"


# ═══════════════════════════════════════════════════════════════════════════════
# SOCIAL SENTIMENT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SocialSentimentDetector:
    """
    Tracks social media sentiment to detect FOMO peaks and panic bottoms.
    When everyone is bullish = institutions are selling.
    When everyone is bearish = institutions are buying.
    """
    
    def __init__(self):
        self.cache_file = DATA_DIR / "sentiment_cache.json"
        self.history_file = DATA_DIR / "sentiment_history.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    async def get_sentiment_score(self, asset: str) -> Dict:
        """
        Get social sentiment score for an asset.
        Score: 0-100 (0=max bearish, 100=max bullish)
        """
        try:
            # Aggregate from multiple sources
            reddit_score = await self._get_reddit_sentiment(asset)
            trends_score = await self._get_google_trends(asset)
            
            # Weighted average
            combined_score = (reddit_score * 0.6) + (trends_score * 0.4)
            
            # Determine signal
            signal = "NEUTRAL"
            if combined_score > Config.SENTIMENT_EUPHORIA_THRESHOLD:
                signal = "EUPHORIA"  # Retail max bullish - danger for new longs
            elif combined_score < Config.SENTIMENT_PANIC_THRESHOLD:
                signal = "PANIC"  # Retail max bearish - opportunity to buy
            
            return {
                "asset": asset,
                "score": round(combined_score, 1),
                "percentile": self._calculate_percentile(combined_score),
                "signal": signal,
                "components": {
                    "reddit": reddit_score,
                    "google_trends": trends_score
                },
                "interpretation": self._interpret_sentiment(combined_score),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment error for {asset}: {e}")
            return {"asset": asset, "score": 50, "signal": "NEUTRAL", "error": str(e)}
    
    async def _get_reddit_sentiment(self, asset: str) -> float:
        """
        Get Reddit sentiment for an asset.
        In production, use Reddit API or services like Quiver Quant.
        """
        # Simulated - would use Reddit API in production
        # High score = lots of bullish posts
        # For now, return moderate bullish (reflecting current Gold rally sentiment)
        
        if asset.upper() in ["GOLD", "XAUUSD", "GLD"]:
            return 72.0  # Bullish but not extreme (given the rally)
        elif asset.upper() in ["IREN", "CIFR", "CLSK"]:
            return 68.0  # Moderately bullish
        else:
            return 50.0
    
    async def _get_google_trends(self, asset: str) -> float:
        """
        Get Google Trends score for asset-related keywords.
        In production, use pytrends or Google Trends API.
        """
        # Simulated - would use pytrends in production
        if asset.upper() in ["GOLD", "XAUUSD"]:
            return 65.0  # "Buy gold" searches elevated
        elif asset.upper() in ["IREN"]:
            return 75.0  # IREN getting more attention
        else:
            return 50.0
    
    def _calculate_percentile(self, score: float) -> int:
        """Calculate percentile vs historical readings."""
        # In production, compare against stored history
        # For now, simple mapping
        return int(score)
    
    def _interpret_sentiment(self, score: float) -> str:
        if score > 90:
            return "🚨 EXTREME EUPHORIA - Everyone bullish, institutions likely selling"
        elif score > 80:
            return "⚠️ HIGH BULLISH - FOMO building, be cautious adding"
        elif score > 60:
            return "📈 BULLISH - Healthy sentiment, trend intact"
        elif score > 40:
            return "😐 NEUTRAL - Mixed signals"
        elif score > 20:
            return "📉 BEARISH - Fear building, watch for capitulation"
        else:
            return "🟢 EXTREME PANIC - Maximum fear, potential buying opportunity"


# ═══════════════════════════════════════════════════════════════════════════════
# STOP HUNT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class StopHuntDetector:
    """
    Detects when price is approaching areas where retail stops cluster.
    Institutions often trigger these stops before reversing.
    """
    
    def __init__(self):
        # Common stop placement patterns
        self.stop_patterns = {
            "round_numbers": True,  # $5,200, $5,250, $5,300
            "recent_lows": True,    # Just below recent swing lows
            "ema_levels": True,     # Near EMA 20/50
            "fibonacci": True       # 38.2%, 50%, 61.8% retracements
        }
    
    def detect_stop_hunt_risk(self, current_price: float, recent_lows: List[float], 
                              recent_highs: List[float]) -> Dict:
        """
        Analyze if price is approaching a likely stop cluster.
        """
        risks = []
        
        # Check round number proximity
        round_levels = self._get_round_numbers_near(current_price)
        for level in round_levels:
            if abs(current_price - level) < current_price * 0.003:  # Within 0.3%
                risks.append({
                    "type": "ROUND_NUMBER",
                    "level": level,
                    "distance_pct": abs(current_price - level) / current_price * 100
                })
        
        # Check recent lows (stops likely below)
        for low in recent_lows[-5:]:  # Last 5 swing lows
            stop_zone = low * 0.998  # Just below the low
            if current_price < low and current_price > stop_zone:
                risks.append({
                    "type": "BELOW_RECENT_LOW",
                    "level": low,
                    "stop_zone": stop_zone
                })
        
        # Risk assessment
        risk_level = "LOW"
        if len(risks) >= 3:
            risk_level = "HIGH"
        elif len(risks) >= 1:
            risk_level = "MEDIUM"
        
        return {
            "current_price": current_price,
            "risk_level": risk_level,
            "risks_detected": risks,
            "recommendation": self._get_recommendation(risk_level),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_round_numbers_near(self, price: float) -> List[float]:
        """Get round numbers near current price."""
        base = int(price / 50) * 50
        return [base - 50, base, base + 50, base + 100]
    
    def _get_recommendation(self, risk_level: str) -> str:
        if risk_level == "HIGH":
            return "⚠️ HIGH STOP HUNT RISK - Consider widening stops or moving to breakeven"
        elif risk_level == "MEDIUM":
            return "🟡 MODERATE RISK - Monitor price action closely"
        else:
            return "🟢 LOW RISK - Normal trading conditions"


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE PROTECTION (RANDOMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

class CascadeProtection:
    """
    Randomized cascade protection to avoid being hunted with other algos.
    Instead of fixed 92% volatility freeze (like everyone else), we use random thresholds.
    """
    
    def __init__(self):
        self.current_threshold = self._generate_new_threshold()
        self.last_threshold_change = datetime.utcnow()
    
    def _generate_new_threshold(self) -> float:
        """Generate random volatility freeze threshold."""
        return random.uniform(Config.VOLATILITY_FREEZE_MIN, Config.VOLATILITY_FREEZE_MAX)
    
    def should_freeze(self, current_volatility_percentile: float) -> Tuple[bool, str]:
        """
        Determine if we should freeze trading.
        Returns (should_freeze, reason)
        """
        # Rotate threshold every 4 hours to stay unpredictable
        if datetime.utcnow() - self.last_threshold_change > timedelta(hours=4):
            self.current_threshold = self._generate_new_threshold()
            self.last_threshold_change = datetime.utcnow()
            logger.info(f"🔄 Rotated cascade threshold to {self.current_threshold:.1f}%")
        
        if current_volatility_percentile > self.current_threshold:
            return True, f"Volatility {current_volatility_percentile:.1f}% > threshold {self.current_threshold:.1f}%"
        
        return False, f"Volatility {current_volatility_percentile:.1f}% OK (threshold: {self.current_threshold:.1f}%)"
    
    def get_current_threshold(self) -> Dict:
        """Get current protection status."""
        return {
            "current_threshold": round(self.current_threshold, 1),
            "last_rotated": self.last_threshold_change.isoformat(),
            "next_rotation": (self.last_threshold_change + timedelta(hours=4)).isoformat(),
            "note": "Threshold randomized to avoid coordinated algo hunts"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER INSTITUTIONAL DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class InstitutionalDetector:
    """
    Master class that combines all detection methods.
    This is what NEO uses to "see" institutional patterns.
    """
    
    def __init__(self):
        self.options_detector = OptionsFlowDetector()
        self.sentiment_detector = SocialSentimentDetector()
        self.stop_hunt_detector = StopHuntDetector()
        self.cascade_protection = CascadeProtection()
    
    async def get_full_analysis(self, symbol: str, current_price: float,
                                recent_lows: List[float] = None,
                                recent_highs: List[float] = None) -> Dict:
        """
        Get complete institutional analysis for a symbol.
        This is the main method NEO should call.
        """
        recent_lows = recent_lows or []
        recent_highs = recent_highs or []
        
        # Gather all signals
        options_flow = await self.options_detector.get_put_call_ratio(symbol)
        sentiment = await self.sentiment_detector.get_sentiment_score(symbol)
        stop_hunt = self.stop_hunt_detector.detect_stop_hunt_risk(
            current_price, recent_lows, recent_highs
        )
        cascade_status = self.cascade_protection.get_current_threshold()
        
        # Determine overall institutional signal
        institutional_signal = self._calculate_institutional_signal(
            options_flow, sentiment, stop_hunt
        )
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "timestamp": datetime.utcnow().isoformat(),
            
            "institutional_signal": institutional_signal,
            
            "components": {
                "options_flow": options_flow,
                "sentiment": sentiment,
                "stop_hunt_risk": stop_hunt,
                "cascade_protection": cascade_status
            },
            
            "trading_recommendation": self._get_trading_recommendation(
                institutional_signal, options_flow, sentiment
            )
        }
    
    def _calculate_institutional_signal(self, options_flow: Dict, 
                                        sentiment: Dict, stop_hunt: Dict) -> Dict:
        """
        Calculate the overall institutional signal.
        """
        # Danger signals (institutions likely selling)
        danger_score = 0
        if options_flow.get("signal") == "RETAIL_EUPHORIA":
            danger_score += 30
        if sentiment.get("signal") == "EUPHORIA":
            danger_score += 30
        if stop_hunt.get("risk_level") == "HIGH":
            danger_score += 20
        
        # Opportunity signals (institutions likely buying)
        opportunity_score = 0
        if options_flow.get("signal") == "RETAIL_PANIC":
            opportunity_score += 30
        if sentiment.get("signal") == "PANIC":
            opportunity_score += 30
        
        # Determine signal
        if danger_score >= 50:
            signal = "INSTITUTIONS_LIKELY_SELLING"
            action = "REDUCE_EXPOSURE"
        elif opportunity_score >= 50:
            signal = "INSTITUTIONS_LIKELY_BUYING"
            action = "ACCUMULATE_AGGRESSIVELY"
        else:
            signal = "NEUTRAL"
            action = "FOLLOW_TREND"
        
        return {
            "signal": signal,
            "action": action,
            "danger_score": danger_score,
            "opportunity_score": opportunity_score,
            "confidence": max(danger_score, opportunity_score, 50)
        }
    
    def _get_trading_recommendation(self, inst_signal: Dict, 
                                     options: Dict, sentiment: Dict) -> str:
        """Generate human-readable trading recommendation."""
        action = inst_signal.get("action", "FOLLOW_TREND")
        
        if action == "REDUCE_EXPOSURE":
            return (
                "⚠️ CAUTION: Retail euphoria detected. Institutions may be distributing.\n"
                f"Options: {options.get('interpretation', 'N/A')}\n"
                f"Sentiment: {sentiment.get('interpretation', 'N/A')}\n"
                "Recommendation: Tighten stops, take partial profits, avoid new longs at highs."
            )
        elif action == "ACCUMULATE_AGGRESSIVELY":
            return (
                "🟢 OPPORTUNITY: Retail panic detected. Institutions may be accumulating.\n"
                f"Options: {options.get('interpretation', 'N/A')}\n"
                f"Sentiment: {sentiment.get('interpretation', 'N/A')}\n"
                "Recommendation: DCA aggressively, add to positions on dips."
            )
        else:
            return (
                "📊 NEUTRAL: No extreme institutional signals detected.\n"
                f"Options: {options.get('interpretation', 'N/A')}\n"
                f"Sentiment: {sentiment.get('interpretation', 'N/A')}\n"
                "Recommendation: Follow trend, use standard DCA strategy."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Test the institutional detector."""
    detector = InstitutionalDetector()
    
    # Test with Gold
    print("\n" + "="*60)
    print("🏛️ INSTITUTIONAL DETECTOR - GOLD ANALYSIS")
    print("="*60)
    
    analysis = await detector.get_full_analysis(
        symbol="XAUUSD",
        current_price=5270.00,
        recent_lows=[5200, 5180, 5150, 5100, 5050],
        recent_highs=[5280, 5300, 5320]
    )
    
    print(json.dumps(analysis, indent=2))
    
    # Test with IREN
    print("\n" + "="*60)
    print("🏛️ INSTITUTIONAL DETECTOR - IREN ANALYSIS")
    print("="*60)
    
    iren_analysis = await detector.get_full_analysis(
        symbol="IREN",
        current_price=59.94,
        recent_lows=[52, 48, 45, 42, 40],
        recent_highs=[60, 62, 65]
    )
    
    print(json.dumps(iren_analysis, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
PRE-MARKET MM ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

Daily pre-market intelligence report generated at 6:00 AM UTC (before London).

Answers the critical question: "Where will MMs hunt stops today?"

Components:
1. Overnight Activity Summary (Asian session range)
2. Liquidity Pool Mapping (where are the stops?)
3. MM Playbook Prediction (what will they do?)
4. Key Levels for the Day
5. Correlation Check (USDJPY, DXY, etc.)
6. Today's Trading Plan

Schedule: 6:00 AM UTC, Monday-Friday
Delivery: Telegram notification + AgentDB

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler("/home/jbot/trading_ai/logs/premarket_report.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PREMARKET")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OvernightRange:
    """Asian session (overnight) activity"""
    high: float
    low: float
    open_price: float
    close_price: float
    range_pts: float
    volatility: str  # LOW, NORMAL, HIGH
    bias: str        # BULLISH, BEARISH, CONSOLIDATION
    volume_ratio: float  # vs average


@dataclass
class LiquidityPool:
    """Stop loss cluster area"""
    level: float
    pool_type: str       # LONG_STOPS, SHORT_STOPS
    distance: float      # Points from current price
    strength: int        # 1-100
    reasons: List[str]
    hunt_probability: int  # 0-100


@dataclass 
class MMPrediction:
    """Market Maker behavior prediction"""
    scenario: str        # A, B, C
    probability: int     # 0-100
    description: str
    expected_hunt: str   # LONGS, SHORTS, NONE
    hunt_level: float
    reversal_target: float
    action: str
    timing: str


@dataclass
class TradingPlan:
    """Daily trading plan"""
    primary_setup: Dict
    alternative_setup: Dict
    best_times: List[str]
    avoid_times: List[str]
    risk_level: str


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-MARKET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PreMarketAnalyzer:
    """
    Generates comprehensive pre-market analysis report.
    """
    
    def __init__(self):
        self.symbol = "XAUUSD"
        
    def fetch_ohlcv_data(self, symbol: str = "XAUUSD", timeframe: str = "H1", 
                         period: str = "7d") -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            import yfinance as yf
            
            ticker_map = {
                "XAUUSD": "GC=F",
                "USDJPY": "USDJPY=X",
                "EURUSD": "EURUSD=X",
                "DXY": "DX-Y.NYB"
            }
            
            ticker = ticker_map.get(symbol, symbol)
            interval_map = {"H1": "1h", "H4": "1h", "D1": "1d"}
            
            data = yf.Ticker(ticker).history(
                period=period,
                interval=interval_map.get(timeframe, "1h")
            )
            
            if not data.empty:
                data.columns = [c.lower() for c in data.columns]
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str = "XAUUSD") -> float:
        """Get current price"""
        df = self.fetch_ohlcv_data(symbol, "H1", "1d")
        if not df.empty:
            return float(df['close'].iloc[-1])
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. OVERNIGHT ACTIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_overnight_range(self, df: pd.DataFrame) -> OvernightRange:
        """
        Analyze Asian session (overnight) activity.
        Asian session: 00:00 - 08:00 UTC
        """
        if df.empty:
            return OvernightRange(0, 0, 0, 0, 0, "UNKNOWN", "UNKNOWN", 0)
        
        # Get last 8 hours of data (Asian session)
        asian_hours = df.tail(8)
        
        if asian_hours.empty:
            return OvernightRange(0, 0, 0, 0, 0, "UNKNOWN", "UNKNOWN", 0)
        
        high = float(asian_hours['high'].max())
        low = float(asian_hours['low'].min())
        open_price = float(asian_hours['open'].iloc[0])
        close_price = float(asian_hours['close'].iloc[-1])
        range_pts = high - low
        
        # Calculate average range for comparison
        if len(df) >= 24:
            avg_range = (df['high'] - df['low']).tail(24).mean()
            if range_pts < avg_range * 0.7:
                volatility = "LOW"
            elif range_pts > avg_range * 1.3:
                volatility = "HIGH"
            else:
                volatility = "NORMAL"
        else:
            volatility = "NORMAL"
        
        # Determine bias
        if close_price > open_price + (range_pts * 0.3):
            bias = "BULLISH"
        elif close_price < open_price - (range_pts * 0.3):
            bias = "BEARISH"
        else:
            bias = "CONSOLIDATION"
        
        # Volume ratio
        if 'volume' in asian_hours.columns:
            asian_vol = asian_hours['volume'].mean()
            avg_vol = df['volume'].tail(48).mean() if len(df) >= 48 else asian_vol
            volume_ratio = asian_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        return OvernightRange(
            high=high,
            low=low,
            open_price=open_price,
            close_price=close_price,
            range_pts=range_pts,
            volatility=volatility,
            bias=bias,
            volume_ratio=volume_ratio
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. LIQUIDITY POOL MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_liquidity_pools(self, df: pd.DataFrame, current_price: float) -> List[LiquidityPool]:
        """
        Find areas where retail stops likely cluster.
        """
        pools = []
        
        if df.empty:
            return pools
        
        # Get recent price action
        recent_20 = df.tail(20)
        recent_5 = df.tail(5)
        
        swing_low = float(recent_20['low'].min())
        swing_high = float(recent_20['high'].max())
        yesterday_low = float(recent_5['low'].min())
        yesterday_high = float(recent_5['high'].max())
        
        # ═══ LONG STOPS (Below current price) ═══
        
        # Round number pools
        base_100 = int(current_price / 100) * 100
        for level in [base_100 - 100, base_100 - 50, base_100]:
            if level < current_price:
                distance = current_price - level
                hunt_prob = 70 if distance < 30 else 50 if distance < 50 else 30
                
                pools.append(LiquidityPool(
                    level=level,
                    pool_type="LONG_STOPS",
                    distance=distance,
                    strength=80 if level == base_100 else 60,
                    reasons=["Round psychological number", "Common stop placement"],
                    hunt_probability=hunt_prob
                ))
        
        # Swing low pool
        if swing_low < current_price:
            distance = current_price - swing_low
            pools.append(LiquidityPool(
                level=swing_low - 5,  # Just below
                pool_type="LONG_STOPS",
                distance=distance + 5,
                strength=90,
                reasons=["Recent swing low", "Obvious stop level", "Breakout trader stops"],
                hunt_probability=85 if distance < 30 else 60
            ))
        
        # Yesterday's low
        if yesterday_low < current_price and abs(yesterday_low - swing_low) > 10:
            distance = current_price - yesterday_low
            pools.append(LiquidityPool(
                level=yesterday_low - 3,
                pool_type="LONG_STOPS",
                distance=distance + 3,
                strength=75,
                reasons=["Yesterday's low", "Day trader stops"],
                hunt_probability=70 if distance < 40 else 45
            ))
        
        # ═══ SHORT STOPS (Above current price) ═══
        
        # Round number pools above
        for level in [base_100 + 50, base_100 + 100]:
            if level > current_price:
                distance = level - current_price
                hunt_prob = 60 if distance < 30 else 40 if distance < 50 else 25
                
                pools.append(LiquidityPool(
                    level=level,
                    pool_type="SHORT_STOPS",
                    distance=distance,
                    strength=70 if level == base_100 + 100 else 55,
                    reasons=["Round psychological number", "Short squeeze target"],
                    hunt_probability=hunt_prob
                ))
        
        # Swing high pool
        if swing_high > current_price:
            distance = swing_high - current_price
            pools.append(LiquidityPool(
                level=swing_high + 5,
                pool_type="SHORT_STOPS",
                distance=distance + 5,
                strength=85,
                reasons=["Recent swing high", "Obvious resistance", "Short stop level"],
                hunt_probability=75 if distance < 30 else 50
            ))
        
        # Yesterday's high
        if yesterday_high > current_price and abs(yesterday_high - swing_high) > 10:
            distance = yesterday_high - current_price
            pools.append(LiquidityPool(
                level=yesterday_high + 3,
                pool_type="SHORT_STOPS",
                distance=distance + 3,
                strength=70,
                reasons=["Yesterday's high", "Momentum trader stops"],
                hunt_probability=65 if distance < 40 else 40
            ))
        
        # Sort by hunt probability
        pools.sort(key=lambda x: x.hunt_probability, reverse=True)
        
        return pools
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. MM PLAYBOOK PREDICTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def predict_mm_behavior(
        self, 
        current_price: float,
        liquidity_pools: List[LiquidityPool],
        overnight: OvernightRange
    ) -> List[MMPrediction]:
        """
        Predict Market Maker behavior for the day.
        
        Based on:
        - Liquidity pool locations
        - Overnight bias
        - Day of week patterns
        """
        predictions = []
        
        # Find most huntable pools
        long_pools = [p for p in liquidity_pools if p.pool_type == "LONG_STOPS"]
        short_pools = [p for p in liquidity_pools if p.pool_type == "SHORT_STOPS"]
        
        # Best long hunt target
        best_long_hunt = long_pools[0] if long_pools else None
        best_short_hunt = short_pools[0] if short_pools else None
        
        day_of_week = datetime.utcnow().strftime("%A")
        
        # ═══ SCENARIO A: Hunt Longs First ═══
        if best_long_hunt and best_long_hunt.hunt_probability >= 60:
            hunt_level = best_long_hunt.level
            
            # Adjust probability based on overnight bias
            prob = best_long_hunt.hunt_probability
            if overnight.bias == "BEARISH":
                prob = min(90, prob + 10)
            elif overnight.bias == "BULLISH":
                prob = max(40, prob - 15)
            
            # Monday tends to be bullish after initial dip
            if day_of_week == "Monday":
                prob = min(85, prob + 10)
            
            reversal_target = current_price + (current_price - hunt_level) * 1.5
            
            predictions.append(MMPrediction(
                scenario="A",
                probability=prob,
                description=f"Push down to ${hunt_level:.0f} to hunt long stops, then reverse up",
                expected_hunt="LONGS",
                hunt_level=hunt_level,
                reversal_target=reversal_target,
                action=f"WAIT for hunt at ${hunt_level:.0f}, then BUY on reversal",
                timing="London Open (08:00 UTC)"
            ))
        
        # ═══ SCENARIO B: Hunt Shorts First ═══
        if best_short_hunt and best_short_hunt.hunt_probability >= 50:
            hunt_level = best_short_hunt.level
            
            prob = best_short_hunt.hunt_probability - 10  # Less common than long hunts
            if overnight.bias == "BULLISH":
                prob = min(80, prob + 15)
            
            reversal_target = current_price - (hunt_level - current_price) * 1.2
            
            predictions.append(MMPrediction(
                scenario="B",
                probability=prob,
                description=f"Break above ${hunt_level:.0f} to squeeze shorts, then reverse down",
                expected_hunt="SHORTS",
                hunt_level=hunt_level,
                reversal_target=reversal_target,
                action=f"WAIT for squeeze above ${hunt_level:.0f}, SELL on failure",
                timing="Late London/NY overlap"
            ))
        
        # ═══ SCENARIO C: Range Day ═══
        remaining_prob = 100 - sum(p.probability for p in predictions)
        if remaining_prob > 5:
            predictions.append(MMPrediction(
                scenario="C",
                probability=max(10, remaining_prob),
                description=f"Choppy range between ${overnight.low:.0f} - ${overnight.high:.0f}",
                expected_hunt="NONE",
                hunt_level=0,
                reversal_target=0,
                action="SKIP TRADING - wait for clearer setup",
                timing="N/A"
            ))
        
        # Sort by probability
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        return predictions
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. CORRELATION CHECK
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_correlation_status(self) -> Dict:
        """Get correlation status for Gold"""
        correlations = {}
        
        try:
            # USDJPY
            usdjpy_df = self.fetch_ohlcv_data("USDJPY", "H1", "7d")
            if not usdjpy_df.empty:
                usdjpy_price = float(usdjpy_df['close'].iloc[-1])
                usdjpy_prev = float(usdjpy_df['close'].iloc[-24]) if len(usdjpy_df) >= 24 else usdjpy_price
                usdjpy_change = ((usdjpy_price - usdjpy_prev) / usdjpy_prev) * 100
                
                # USDJPY falling = bullish for Gold
                if usdjpy_price > 158:
                    gold_signal = "BULLISH (BOJ intervention zone)"
                elif usdjpy_change < -0.3:
                    gold_signal = "BULLISH (JPY strengthening)"
                elif usdjpy_change > 0.3:
                    gold_signal = "BEARISH (JPY weakening)"
                else:
                    gold_signal = "NEUTRAL"
                
                correlations["USDJPY"] = {
                    "price": usdjpy_price,
                    "change_pct": usdjpy_change,
                    "gold_signal": gold_signal
                }
        except Exception as e:
            logger.warning(f"USDJPY correlation error: {e}")
        
        try:
            # DXY (Dollar Index)
            dxy_df = self.fetch_ohlcv_data("DXY", "H1", "7d")
            if not dxy_df.empty:
                dxy_price = float(dxy_df['close'].iloc[-1])
                dxy_prev = float(dxy_df['close'].iloc[-24]) if len(dxy_df) >= 24 else dxy_price
                dxy_change = ((dxy_price - dxy_prev) / dxy_prev) * 100
                
                # DXY falling = bullish for Gold
                if dxy_change < -0.2:
                    gold_signal = "BULLISH (Dollar weak)"
                elif dxy_change > 0.2:
                    gold_signal = "BEARISH (Dollar strong)"
                else:
                    gold_signal = "NEUTRAL"
                
                correlations["DXY"] = {
                    "price": dxy_price,
                    "change_pct": dxy_change,
                    "gold_signal": gold_signal
                }
        except Exception as e:
            logger.warning(f"DXY correlation error: {e}")
        
        # Overall assessment
        bullish_count = sum(1 for c in correlations.values() if "BULLISH" in c.get("gold_signal", ""))
        bearish_count = sum(1 for c in correlations.values() if "BEARISH" in c.get("gold_signal", ""))
        
        if bullish_count > bearish_count:
            overall = "ALIGNED BULLISH ✅"
        elif bearish_count > bullish_count:
            overall = "ALIGNED BEARISH ⚠️"
        else:
            overall = "MIXED SIGNALS ⚪"
        
        correlations["overall"] = overall
        
        return correlations
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. GENERATE TRADING PLAN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_trading_plan(
        self,
        current_price: float,
        liquidity_pools: List[LiquidityPool],
        predictions: List[MMPrediction],
        overnight: OvernightRange
    ) -> TradingPlan:
        """Generate actionable trading plan for the day"""
        
        primary_pred = predictions[0] if predictions else None
        alt_pred = predictions[1] if len(predictions) > 1 else None
        
        # Primary setup
        if primary_pred and primary_pred.expected_hunt == "LONGS":
            primary_setup = {
                "type": "BUY_THE_DIP",
                "wait_for": f"Push to ${primary_pred.hunt_level:.0f}",
                "entry": f"After reversal candle at ${primary_pred.hunt_level:.0f}-${primary_pred.hunt_level+10:.0f}",
                "stop_loss": f"${primary_pred.hunt_level - 20:.0f}",
                "take_profit_1": f"${overnight.high:.0f} (Asian high)",
                "take_profit_2": f"${current_price + 30:.0f}",
                "risk_reward": "1:2 minimum"
            }
        elif primary_pred and primary_pred.expected_hunt == "SHORTS":
            primary_setup = {
                "type": "FADE_THE_SQUEEZE",
                "wait_for": f"Break above ${primary_pred.hunt_level:.0f}",
                "entry": f"SELL on failure below ${primary_pred.hunt_level:.0f}",
                "stop_loss": f"${primary_pred.hunt_level + 15:.0f}",
                "take_profit_1": f"${overnight.low:.0f}",
                "take_profit_2": f"${current_price - 25:.0f}",
                "risk_reward": "1:1.5 minimum"
            }
        else:
            primary_setup = {
                "type": "WAIT",
                "description": "No clear hunt setup - wait for better opportunity"
            }
        
        # Alternative setup
        if alt_pred:
            alternative_setup = {
                "trigger": alt_pred.description,
                "action": alt_pred.action
            }
        else:
            alternative_setup = {"description": "No alternative setup"}
        
        # Best times
        day_of_week = datetime.utcnow().strftime("%A")
        best_times = [
            "08:00-09:00 UTC (London open - watch for initial move)",
            "13:00-14:00 UTC (NY open - watch for reversal)",
        ]
        
        avoid_times = ["12:00-13:00 UTC (London lunch, choppy)"]
        if day_of_week == "Friday":
            avoid_times.append("16:00+ UTC (Weekend profit taking)")
        
        # Risk level
        if overnight.volatility == "HIGH":
            risk_level = "HIGH - Reduce position size"
        elif overnight.volatility == "LOW" and overnight.bias == "CONSOLIDATION":
            risk_level = "LOW - Good for entries"
        else:
            risk_level = "NORMAL"
        
        return TradingPlan(
            primary_setup=primary_setup,
            alternative_setup=alternative_setup,
            best_times=best_times,
            avoid_times=avoid_times,
            risk_level=risk_level
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN REPORT GENERATOR
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_report(self, symbol: str = "XAUUSD") -> Dict:
        """Generate comprehensive pre-market report"""
        
        logger.info("\n" + "="*70)
        logger.info("🌅 GENERATING PRE-MARKET REPORT")
        logger.info("="*70)
        
        self.symbol = symbol
        
        # 1. Get data
        logger.info("\n📊 Fetching market data...")
        h1_data = self.fetch_ohlcv_data(symbol, "H1", "7d")
        d1_data = self.fetch_ohlcv_data(symbol, "D1", "30d")
        current_price = self.get_current_price(symbol)
        
        logger.info(f"   Current Price: ${current_price:.2f}")
        
        # 2. Overnight analysis
        logger.info("\n🌙 Analyzing overnight activity...")
        overnight = self.analyze_overnight_range(h1_data)
        logger.info(f"   Range: ${overnight.low:.2f} - ${overnight.high:.2f} ({overnight.range_pts:.0f} pts)")
        logger.info(f"   Volatility: {overnight.volatility}")
        logger.info(f"   Bias: {overnight.bias}")
        
        # 3. Liquidity pools (use H1 for recent swings, D1 for major levels)
        logger.info("\n💧 Mapping liquidity pools...")
        # Combine H1 (recent) and D1 (major) for better level detection
        liquidity_pools = self.find_liquidity_pools(h1_data if not h1_data.empty else d1_data, current_price)
        for pool in liquidity_pools[:3]:
            logger.info(f"   ${pool.level:.0f} ({pool.pool_type}) - {pool.hunt_probability}% hunt probability")
        
        # 4. MM predictions
        logger.info("\n🦊 Predicting MM behavior...")
        predictions = self.predict_mm_behavior(current_price, liquidity_pools, overnight)
        for pred in predictions[:2]:
            logger.info(f"   Scenario {pred.scenario}: {pred.probability}% - {pred.description[:50]}...")
        
        # 5. Correlations
        logger.info("\n🔗 Checking correlations...")
        correlations = self.get_correlation_status()
        logger.info(f"   Overall: {correlations.get('overall', 'UNKNOWN')}")
        
        # 6. Trading plan
        logger.info("\n📋 Generating trading plan...")
        trading_plan = self.generate_trading_plan(
            current_price, liquidity_pools, predictions, overnight
        )
        
        logger.info(f"   Primary: {trading_plan.primary_setup.get('type', 'N/A')}")
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "overnight": {
                "high": overnight.high,
                "low": overnight.low,
                "range_pts": overnight.range_pts,
                "volatility": overnight.volatility,
                "bias": overnight.bias,
                "volume_ratio": overnight.volume_ratio
            },
            "liquidity_pools": [
                {
                    "level": p.level,
                    "type": p.pool_type,
                    "distance": p.distance,
                    "strength": p.strength,
                    "hunt_probability": p.hunt_probability,
                    "reasons": p.reasons
                }
                for p in liquidity_pools[:6]
            ],
            "predictions": [
                {
                    "scenario": p.scenario,
                    "probability": p.probability,
                    "description": p.description,
                    "expected_hunt": p.expected_hunt,
                    "hunt_level": p.hunt_level,
                    "reversal_target": p.reversal_target,
                    "action": p.action,
                    "timing": p.timing
                }
                for p in predictions
            ],
            "correlations": correlations,
            "trading_plan": {
                "primary": trading_plan.primary_setup,
                "alternative": trading_plan.alternative_setup,
                "best_times": trading_plan.best_times,
                "avoid_times": trading_plan.avoid_times,
                "risk_level": trading_plan.risk_level
            }
        }
        
        logger.info("\n✅ Pre-market report generated!")
        
        return report
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TELEGRAM FORMATTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def format_telegram_report(self, report: Dict) -> str:
        """Format report for Telegram delivery"""
        
        symbol = report.get("symbol", "XAUUSD")
        price = report.get("current_price", 0)
        overnight = report.get("overnight", {})
        pools = report.get("liquidity_pools", [])
        predictions = report.get("predictions", [])
        correlations = report.get("correlations", {})
        plan = report.get("trading_plan", {})
        
        # Day of week
        day = datetime.utcnow().strftime("%A")
        date = datetime.utcnow().strftime("%B %d, %Y")
        
        lines = [
            f"🌅 <b>PRE-MARKET REPORT - {symbol}</b>",
            f"📅 {day}, {date}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Current:</b> ${price:.2f}",
            "",
            f"📊 <b>OVERNIGHT (Asian Session):</b>",
            f"├── High: ${overnight.get('high', 0):.2f}",
            f"├── Low: ${overnight.get('low', 0):.2f}",
            f"├── Range: {overnight.get('range_pts', 0):.0f} pts ({overnight.get('volatility', 'N/A')})",
            f"└── Bias: {overnight.get('bias', 'N/A')}",
            "",
            f"💧 <b>LIQUIDITY POOLS (Hunt Targets):</b>",
        ]
        
        # Long stops
        long_pools = [p for p in pools if p.get('type') == 'LONG_STOPS'][:2]
        if long_pools:
            lines.append("<i>LONG STOPS (MMs hunt FIRST):</i>")
            for p in long_pools:
                emoji = "🎯" if p.get('hunt_probability', 0) >= 70 else "📍"
                lines.append(f"{emoji} ${p.get('level', 0):.0f} - {p.get('distance', 0):.0f}pts away ({p.get('hunt_probability', 0)}%)")
        
        # Short stops
        short_pools = [p for p in pools if p.get('type') == 'SHORT_STOPS'][:2]
        if short_pools:
            lines.append("<i>SHORT STOPS:</i>")
            for p in short_pools:
                lines.append(f"📍 ${p.get('level', 0):.0f} - {p.get('distance', 0):.0f}pts away ({p.get('hunt_probability', 0)}%)")
        
        lines.append("")
        lines.append(f"🦊 <b>MM PREDICTION:</b>")
        
        if predictions:
            for pred in predictions[:2]:
                emoji = "🎯" if pred.get('probability', 0) >= 60 else "📌"
                lines.append(f"{emoji} <b>Scenario {pred.get('scenario', '?')}:</b> ({pred.get('probability', 0)}%)")
                lines.append(f"   {pred.get('description', 'N/A')}")
                if pred.get('action'):
                    lines.append(f"   ➜ <i>{pred.get('action', '')}</i>")
        
        lines.append("")
        lines.append(f"🔗 <b>CORRELATIONS:</b>")
        
        for key, val in correlations.items():
            if key == "overall":
                lines.append(f"<b>Overall:</b> {val}")
            elif isinstance(val, dict):
                change = val.get('change_pct', 0)
                change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
                lines.append(f"├── {key}: {val.get('price', 0):.2f} ({change_str}%) → {val.get('gold_signal', 'N/A')}")
        
        lines.append("")
        lines.append(f"📋 <b>TODAY'S PLAN:</b>")
        
        primary = plan.get('primary', {})
        if primary.get('type') and primary.get('type') != 'WAIT':
            lines.append(f"<b>Primary:</b> {primary.get('type', 'N/A')}")
            lines.append(f"├── Wait for: {primary.get('wait_for', 'N/A')}")
            lines.append(f"├── Entry: {primary.get('entry', 'N/A')}")
            lines.append(f"├── SL: {primary.get('stop_loss', 'N/A')}")
            lines.append(f"├── TP1: {primary.get('take_profit_1', 'N/A')}")
            lines.append(f"└── R:R: {primary.get('risk_reward', 'N/A')}")
        else:
            lines.append("<b>⚠️ No clear setup - wait for opportunity</b>")
        
        lines.append("")
        lines.append(f"⏰ <b>BEST TIMES:</b>")
        for t in plan.get('best_times', []):
            lines.append(f"└── {t}")
        
        if plan.get('avoid_times'):
            lines.append(f"🚫 <b>AVOID:</b> {', '.join(plan.get('avoid_times', []))}")
        
        lines.append("")
        lines.append(f"⚠️ <b>RISK:</b> {plan.get('risk_level', 'NORMAL')}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM DELIVERY
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram_message(message: str, chat_id: str = None) -> bool:
    """Send message via Telegram"""
    try:
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s')
        if not chat_id:
            chat_id = os.environ.get('ADMIN_CHAT_ID', '6776619257')
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_premarket_report(symbol: str = "XAUUSD") -> Dict:
    """Generate pre-market report (convenience function)"""
    analyzer = PreMarketAnalyzer()
    return analyzer.generate_report(symbol)


def push_to_ghost_commander(report: Dict) -> Optional[Dict]:
    """
    Push the daily trading plan to Ghost Commander.
    Format must match what Ghost expects!
    
    Ghost will use this as a LIMIT ORDER setup - waiting for price
    to reach the hunt zone before executing.
    """
    plan = report.get("trading_plan", {}).get("primary", {})
    predictions = report.get("predictions", [])
    
    # Only push if we have a clear BUY or SELL setup
    plan_type = plan.get("type", "WAIT")
    if plan_type == "WAIT" or plan_type is None:
        logger.info("📊 No clear bias - skipping Ghost signal")
        return None
    
    # Determine direction
    if "BUY" in plan_type.upper() or "DIP" in plan_type.upper():
        direction = "BUY"
    elif "SELL" in plan_type.upper() or "FADE" in plan_type.upper():
        direction = "SELL"
    else:
        logger.info(f"📊 Unknown plan type '{plan_type}' - skipping Ghost signal")
        return None
    
    # Extract prices from plan
    # Plan has format like: "Push to $4900" or "Entry: After reversal at $4900-$4910"
    wait_for = plan.get("wait_for", "")
    entry_str = plan.get("entry", "")
    sl_str = plan.get("stop_loss", "")
    tp_str = plan.get("take_profit_1", "") or plan.get("take_profit", "")
    
    # Parse prices (remove $ and extract numbers)
    import re
    
    def extract_price(s):
        if not s:
            return None
        matches = re.findall(r'\$?([\d,]+\.?\d*)', str(s))
        if matches:
            return float(matches[0].replace(',', ''))
        return None
    
    # Get entry from wait_for or entry field
    entry_price = extract_price(wait_for) or extract_price(entry_str)
    stop_loss = extract_price(sl_str)
    take_profit = extract_price(tp_str)
    
    if not entry_price:
        logger.warning("⚠️ Could not extract entry price from plan")
        return None
    
    # Get confidence from prediction
    confidence = 70  # Default
    if predictions:
        confidence = predictions[0].get("probability", 70)
    
    # Build signal in Ghost-compatible format
    signal = {
        "signal_id": f"PREMARKET_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "OPEN",
        "trade": {
            "symbol": report.get("symbol", "XAUUSD"),
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss or (entry_price - 20 if direction == "BUY" else entry_price + 20),
            "take_profit": take_profit or (entry_price + 40 if direction == "BUY" else entry_price - 40),
            "position_size_usd": 3500,
            "confidence": confidence
        },
        "metadata": {
            "confidence": confidence,
            "source": "PREMARKET_MM",
            "signal_type": "NEO_PREMARKET",
            "reasoning": [
                predictions[0].get("description", "Pre-market analysis") if predictions else "Pre-market setup",
                f"Hunt zone: ${entry_price:.0f}",
                f"Expected reversal after stop hunt"
            ],
            "features_used": 150
        },
        "safety": {
            "max_position_pct": 5,
            "daily_loss_limit": 3,
            "kill_switch": False
        }
    }
    
    # Save to signal file (Ghost reads this)
    signal_file = "/tmp/neo_signal.json"
    try:
        with open(signal_file, "w") as f:
            json.dump(signal, f, indent=2)
        logger.info(f"📁 Ghost signal saved to {signal_file}")
    except Exception as e:
        logger.error(f"Failed to save Ghost signal: {e}")
    
    # Push to API endpoint
    try:
        response = requests.post(
            "http://localhost:8085/neo/signal",
            json=signal,
            timeout=10
        )
        if response.status_code == 200:
            logger.info(f"✅ Ghost signal pushed: {direction} {signal['trade']['symbol']} @ ${entry_price:.0f}")
        else:
            logger.warning(f"⚠️ Ghost API returned {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Ghost signal push failed: {e}")
    
    return signal


def send_premarket_report(symbol: str = "XAUUSD") -> Dict:
    """
    Generate and send pre-market report.
    
    Dual delivery:
    1. Telegram (human visibility)
    2. Ghost Commander (auto-execution)
    3. File backup (reference)
    """
    analyzer = PreMarketAnalyzer()
    
    # Generate report
    logger.info("📊 Generating pre-market analysis...")
    report = analyzer.generate_report(symbol)
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "telegram_sent": False,
        "ghost_signal": None,
        "file_saved": False
    }
    
    # 1. Send to Telegram (human visibility)
    logger.info("📱 Sending to Telegram...")
    message = analyzer.format_telegram_report(report)
    telegram_success = send_telegram_message(message)
    result["telegram_sent"] = telegram_success
    
    if telegram_success:
        logger.info("✅ Telegram report sent!")
    else:
        logger.warning("⚠️ Telegram send failed")
    
    # 2. Send to Ghost Commander (auto-execution)
    logger.info("👻 Pushing to Ghost Commander...")
    ghost_signal = push_to_ghost_commander(report)
    result["ghost_signal"] = ghost_signal
    
    if ghost_signal:
        # Add Ghost status to Telegram message
        ghost_msg = f"\n\n👻 <b>GHOST STATUS:</b> Signal pushed ✅\n"
        ghost_msg += f"├── Direction: {ghost_signal['trade']['direction']}\n"
        ghost_msg += f"├── Entry: ${ghost_signal['trade']['entry_price']:.0f}\n"
        ghost_msg += f"├── SL: ${ghost_signal['trade']['stop_loss']:.0f}\n"
        ghost_msg += f"└── TP: ${ghost_signal['trade']['take_profit']:.0f}"
        send_telegram_message(ghost_msg)
        logger.info("✅ Ghost signal pushed!")
    else:
        logger.info("📊 No Ghost signal generated (no clear setup)")
    
    # 3. Save to file for reference
    report_file = f"/home/jbot/trading_ai/research/premarket_{datetime.utcnow().strftime('%Y%m%d')}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        result["file_saved"] = True
        logger.info(f"📁 Report saved to {report_file}")
    except Exception as e:
        logger.warning(f"Could not save report file: {e}")
    
    # 4. Log summary
    logger.info("\n" + "="*60)
    logger.info("🌅 PRE-MARKET REPORT DELIVERY SUMMARY")
    logger.info("="*60)
    logger.info(f"   Telegram: {'✅ Sent' if result['telegram_sent'] else '❌ Failed'}")
    logger.info(f"   Ghost: {'✅ Armed' if result['ghost_signal'] else '📊 No setup'}")
    logger.info(f"   File: {'✅ Saved' if result['file_saved'] else '❌ Failed'}")
    logger.info("="*60)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Check if we should send to Telegram
    send_telegram = "--send" in sys.argv or "-s" in sys.argv
    symbol = "XAUUSD"
    
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            symbol = arg.upper()
    
    print(f"\n🌅 PRE-MARKET MM ANALYSIS - {symbol}")
    print("=" * 60)
    
    analyzer = PreMarketAnalyzer()
    report = analyzer.generate_report(symbol)
    
    # Print formatted report
    telegram_msg = analyzer.format_telegram_report(report)
    # Convert HTML to plain text for console
    import re
    plain_msg = re.sub(r'<[^>]+>', '', telegram_msg)
    print(plain_msg)
    
    if send_telegram:
        print("\n📤 Sending to Telegram...")
        success = send_telegram_message(telegram_msg)
        print("✅ Sent!" if success else "❌ Failed to send")

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IREN PRE-MARKET ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════

Daily pre-market intelligence report for IREN (Bitcoin miner).

Components:
1. Overnight Activity Summary (After-hours + Pre-market)
2. Liquidity Pool Mapping (where are retail stops?)
3. MM/Algo Prediction (what will they do?)
4. BTC Correlation Check (key driver for miners)
5. Options Flow Analysis (big money positioning)
6. Today's Trading Plan

Schedule: 9:00 AM EST (30 min before market open)
Delivery: Telegram notification

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler("/home/jbot/trading_ai/logs/iren_premarket.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IREN_PREMARKET")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OvernightRange:
    """After-hours + pre-market activity"""
    high: float
    low: float
    prev_close: float
    current: float
    change_pct: float
    gap_pct: float
    volatility: str  # LOW, NORMAL, HIGH
    bias: str        # BULLISH, BEARISH, CONSOLIDATION


@dataclass
class LiquidityPool:
    """Stop loss cluster area"""
    level: float
    pool_type: str       # LONG_STOPS, SHORT_STOPS
    distance_pct: float  # % from current price
    strength: int        # 1-100
    reasons: List[str]
    hunt_probability: int


@dataclass
class MMPrediction:
    """Market Maker behavior prediction"""
    scenario: str
    probability: int
    description: str
    expected_move: str
    target_level: float
    action: str
    timing: str


@dataclass
class TradingPlan:
    """Daily trading plan"""
    share_plan: Dict
    option_plan: Dict
    best_times: List[str]
    avoid_times: List[str]
    risk_level: str


# ═══════════════════════════════════════════════════════════════════════════════
# IREN PRE-MARKET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class IrenPreMarketAnalyzer:
    """
    Generates comprehensive pre-market analysis for IREN.
    """
    
    def __init__(self):
        self.symbol = "IREN"
        
    def fetch_iren_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch IREN OHLCV data"""
        try:
            ticker = yf.Ticker("IREN")
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                data.columns = [c.lower() for c in data.columns]
            return data
        except Exception as e:
            logger.error(f"Failed to fetch IREN data: {e}")
            return pd.DataFrame()
    
    def fetch_btc_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch BTC data for correlation"""
        try:
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                data.columns = [c.lower() for c in data.columns]
            return data
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self) -> Tuple[float, float]:
        """Get current IREN and BTC prices"""
        iren = 0.0
        btc = 0.0
        try:
            iren_data = self.fetch_iren_data("1d", "1m")
            if not iren_data.empty:
                iren = float(iren_data['close'].iloc[-1])
            
            btc_data = self.fetch_btc_data("1d", "1m")
            if not btc_data.empty:
                btc = float(btc_data['close'].iloc[-1])
        except:
            pass
        return iren, btc
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. OVERNIGHT ACTIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_overnight_range(self, df: pd.DataFrame) -> OvernightRange:
        """Analyze after-hours and pre-market activity"""
        if df.empty:
            return OvernightRange(0, 0, 0, 0, 0, 0, "UNKNOWN", "UNKNOWN")
        
        # Get yesterday's close and today's pre-market
        recent = df.tail(24)  # Last 24 hours
        
        if len(recent) < 5:
            return OvernightRange(0, 0, 0, 0, 0, 0, "UNKNOWN", "UNKNOWN")
        
        # Find previous close (end of regular hours)
        prev_close = float(recent['close'].iloc[-8] if len(recent) >= 8 else recent['close'].iloc[0])
        current = float(recent['close'].iloc[-1])
        high = float(recent['high'].max())
        low = float(recent['low'].min())
        
        change_pct = ((current - prev_close) / prev_close) * 100
        gap_pct = change_pct  # For stocks, gap = overnight change
        
        # Volatility assessment
        range_pct = ((high - low) / prev_close) * 100
        if range_pct < 2:
            volatility = "LOW"
        elif range_pct > 5:
            volatility = "HIGH"
        else:
            volatility = "NORMAL"
        
        # Bias
        if change_pct > 1.5:
            bias = "BULLISH"
        elif change_pct < -1.5:
            bias = "BEARISH"
        else:
            bias = "CONSOLIDATION"
        
        return OvernightRange(
            high=high,
            low=low,
            prev_close=prev_close,
            current=current,
            change_pct=change_pct,
            gap_pct=gap_pct,
            volatility=volatility,
            bias=bias
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. LIQUIDITY POOL MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_liquidity_pools(self, df: pd.DataFrame, current_price: float) -> List[LiquidityPool]:
        """Find areas where retail stops likely cluster"""
        pools = []
        
        if df.empty or current_price <= 0:
            return pools
        
        # Get recent price action
        recent_20d = df.tail(100)  # ~20 days of hourly data
        recent_5d = df.tail(30)
        
        swing_low = float(recent_20d['low'].min())
        swing_high = float(recent_20d['high'].max())
        yesterday_low = float(recent_5d['low'].min())
        yesterday_high = float(recent_5d['high'].max())
        
        # ═══ LONG STOPS (Below current price) ═══
        
        # Round dollar levels
        base = int(current_price)
        for level in [base - 5, base - 3, base - 2, base - 1, base]:
            if 0 < level < current_price:
                distance_pct = ((current_price - level) / current_price) * 100
                hunt_prob = 75 if distance_pct < 5 else 50 if distance_pct < 10 else 30
                
                pools.append(LiquidityPool(
                    level=float(level),
                    pool_type="LONG_STOPS",
                    distance_pct=distance_pct,
                    strength=80 if level == base else 60,
                    reasons=[f"${level} psychological level", "Common stop placement"],
                    hunt_probability=hunt_prob
                ))
        
        # Swing low (key level)
        if swing_low < current_price:
            distance_pct = ((current_price - swing_low) / current_price) * 100
            pools.append(LiquidityPool(
                level=swing_low - 0.10,
                pool_type="LONG_STOPS",
                distance_pct=distance_pct + 0.5,
                strength=90,
                reasons=["20-day swing low", "Obvious stop level", "Breakout stops"],
                hunt_probability=85 if distance_pct < 10 else 60
            ))
        
        # Yesterday's low
        if yesterday_low < current_price and abs(yesterday_low - swing_low) > 0.50:
            distance_pct = ((current_price - yesterday_low) / current_price) * 100
            pools.append(LiquidityPool(
                level=yesterday_low - 0.05,
                pool_type="LONG_STOPS",
                distance_pct=distance_pct,
                strength=75,
                reasons=["Yesterday's low", "Day trader stops"],
                hunt_probability=70 if distance_pct < 8 else 45
            ))
        
        # ═══ SHORT STOPS (Above current price) ═══
        
        # Round dollar levels above
        for level in [base + 1, base + 2, base + 3, base + 5]:
            if level > current_price:
                distance_pct = ((level - current_price) / current_price) * 100
                hunt_prob = 60 if distance_pct < 5 else 40 if distance_pct < 10 else 25
                
                pools.append(LiquidityPool(
                    level=float(level),
                    pool_type="SHORT_STOPS",
                    distance_pct=distance_pct,
                    strength=70,
                    reasons=[f"${level} round number", "Short squeeze target"],
                    hunt_probability=hunt_prob
                ))
        
        # Swing high
        if swing_high > current_price:
            distance_pct = ((swing_high - current_price) / current_price) * 100
            pools.append(LiquidityPool(
                level=swing_high + 0.10,
                pool_type="SHORT_STOPS",
                distance_pct=distance_pct + 0.5,
                strength=85,
                reasons=["20-day swing high", "Short stops above"],
                hunt_probability=75 if distance_pct < 10 else 50
            ))
        
        # Yesterday's high
        if yesterday_high > current_price and abs(yesterday_high - swing_high) > 0.50:
            distance_pct = ((yesterday_high - current_price) / current_price) * 100
            pools.append(LiquidityPool(
                level=yesterday_high + 0.05,
                pool_type="SHORT_STOPS",
                distance_pct=distance_pct,
                strength=70,
                reasons=["Yesterday's high", "Momentum stops"],
                hunt_probability=65 if distance_pct < 8 else 40
            ))
        
        pools.sort(key=lambda x: x.hunt_probability, reverse=True)
        return pools
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. MM/ALGO PREDICTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def predict_mm_behavior(
        self, 
        current_price: float,
        liquidity_pools: List[LiquidityPool],
        overnight: OvernightRange,
        btc_change: float
    ) -> List[MMPrediction]:
        """Predict Market Maker/Algo behavior for the day"""
        predictions = []
        
        long_pools = [p for p in liquidity_pools if p.pool_type == "LONG_STOPS"]
        short_pools = [p for p in liquidity_pools if p.pool_type == "SHORT_STOPS"]
        
        best_long_hunt = long_pools[0] if long_pools else None
        best_short_hunt = short_pools[0] if short_pools else None
        
        day_of_week = datetime.utcnow().strftime("%A")
        
        # ═══ SCENARIO A: Gap Fill / Hunt Longs ═══
        if overnight.gap_pct > 2 and best_long_hunt:
            # Gaps often fill - look for reversal
            prob = 65 + int(overnight.gap_pct * 5)
            prob = min(85, prob)
            
            predictions.append(MMPrediction(
                scenario="A",
                probability=prob,
                description=f"Gap fill: Hunt stops at ${best_long_hunt.level:.2f}, then bounce",
                expected_move="DOWN_THEN_UP",
                target_level=best_long_hunt.level,
                action=f"WAIT for dip to ${best_long_hunt.level:.2f}, BUY the reversal",
                timing="First 30 min after open"
            ))
        elif overnight.gap_pct < -2 and best_short_hunt:
            prob = 60 + int(abs(overnight.gap_pct) * 4)
            prob = min(80, prob)
            
            predictions.append(MMPrediction(
                scenario="A",
                probability=prob,
                description=f"Gap fill: Squeeze shorts at ${best_short_hunt.level:.2f}, then fade",
                expected_move="UP_THEN_DOWN",
                target_level=best_short_hunt.level,
                action=f"WAIT for spike to ${best_short_hunt.level:.2f}, SELL the failure",
                timing="First 30 min after open"
            ))
        
        # ═══ SCENARIO B: BTC Correlation Play ═══
        if abs(btc_change) > 2:
            btc_bias = "BULLISH" if btc_change > 0 else "BEARISH"
            if btc_change > 3:
                prob = 70
                target = current_price * 1.05  # 5% up target
                predictions.append(MMPrediction(
                    scenario="B",
                    probability=prob,
                    description=f"BTC rally +{btc_change:.1f}% → IREN follows higher",
                    expected_move="UP",
                    target_level=target,
                    action=f"BUY on any dip, target ${target:.2f}",
                    timing="All day bullish bias"
                ))
            elif btc_change < -3:
                prob = 65
                target = current_price * 0.95
                predictions.append(MMPrediction(
                    scenario="B",
                    probability=prob,
                    description=f"BTC dump {btc_change:.1f}% → IREN weakness expected",
                    expected_move="DOWN",
                    target_level=target,
                    action=f"AVOID longs, watch for bounce at ${target:.2f}",
                    timing="Morning weakness likely"
                ))
        
        # ═══ SCENARIO C: Range/Consolidation ═══
        remaining_prob = 100 - sum(p.probability for p in predictions)
        if remaining_prob > 15 or not predictions:
            range_high = overnight.high
            range_low = overnight.low
            predictions.append(MMPrediction(
                scenario="C",
                probability=max(25, remaining_prob),
                description=f"Range day: ${range_low:.2f} - ${range_high:.2f}",
                expected_move="SIDEWAYS",
                target_level=0,
                action="Scalp the range or WAIT for breakout",
                timing="Power hour (3-4pm) may provide direction"
            ))
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. BTC CORRELATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_btc_correlation(self) -> Dict:
        """Get BTC correlation status"""
        try:
            btc_df = self.fetch_btc_data("7d", "1h")
            if btc_df.empty:
                return {"price": 0, "change_24h": 0, "signal": "UNKNOWN"}
            
            btc_price = float(btc_df['close'].iloc[-1])
            btc_prev = float(btc_df['close'].iloc[-24]) if len(btc_df) >= 24 else btc_price
            btc_change = ((btc_price - btc_prev) / btc_prev) * 100
            
            # IREN typically follows BTC
            if btc_change > 3:
                signal = "BULLISH 🚀 (Strong BTC rally)"
            elif btc_change > 1:
                signal = "BULLISH (BTC green)"
            elif btc_change < -3:
                signal = "BEARISH ⚠️ (BTC dump)"
            elif btc_change < -1:
                signal = "BEARISH (BTC red)"
            else:
                signal = "NEUTRAL"
            
            return {
                "price": btc_price,
                "change_24h": btc_change,
                "signal": signal
            }
        except Exception as e:
            logger.error(f"BTC correlation error: {e}")
            return {"price": 0, "change_24h": 0, "signal": "ERROR"}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. OPTIONS FLOW (simplified)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_options_flow(self, current_price: float) -> Dict:
        """Get simplified options flow analysis"""
        try:
            ticker = yf.Ticker("IREN")
            exp_dates = ticker.options
            
            if not exp_dates:
                return {"signal": "NO_DATA", "max_pain": 0, "call_put_ratio": 1.0}
            
            # Get nearest expiration
            nearest_exp = exp_dates[0]
            chain = ticker.option_chain(nearest_exp)
            
            calls = chain.calls
            puts = chain.puts
            
            # Calculate call/put volume ratio
            call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            cp_ratio = call_vol / put_vol if put_vol > 0 else 2.0
            
            # Estimate max pain (simplified)
            # Max pain is typically around where most options expire worthless
            if not calls.empty:
                max_pain = float(calls[calls['volume'] == calls['volume'].max()]['strike'].values[0]) if calls['volume'].max() > 0 else current_price
            else:
                max_pain = current_price
            
            if cp_ratio > 1.5:
                signal = "BULLISH (Call heavy)"
            elif cp_ratio < 0.7:
                signal = "BEARISH (Put heavy)"
            else:
                signal = "NEUTRAL"
            
            return {
                "signal": signal,
                "max_pain": max_pain,
                "call_put_ratio": cp_ratio,
                "nearest_expiry": nearest_exp
            }
        except Exception as e:
            logger.warning(f"Options flow error: {e}")
            return {"signal": "N/A", "max_pain": current_price, "call_put_ratio": 1.0}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. TRADING PLAN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_trading_plan(
        self,
        current_price: float,
        liquidity_pools: List[LiquidityPool],
        predictions: List[MMPrediction],
        overnight: OvernightRange,
        btc_info: Dict
    ) -> TradingPlan:
        """Generate actionable trading plan"""
        
        primary_pred = predictions[0] if predictions else None
        
        # Share accumulation plan
        if primary_pred and "UP" in primary_pred.expected_move:
            if overnight.change_pct < -3:
                # Gap down = buy opportunity
                share_plan = {
                    "action": "BUY",
                    "entry": f"${current_price:.2f} (gap down = discount)",
                    "add_more": f"${current_price * 0.95:.2f} (5% lower)",
                    "stop_loss": f"${current_price * 0.90:.2f} (10% max)",
                    "target": f"${current_price * 1.15:.2f} (15% gain)"
                }
            else:
                share_plan = {
                    "action": "WAIT_FOR_DIP",
                    "entry": f"${overnight.low:.2f} (yesterday's low)",
                    "add_more": f"${current_price * 0.97:.2f}",
                    "stop_loss": f"${current_price * 0.92:.2f}",
                    "target": f"${current_price * 1.10:.2f}"
                }
        elif primary_pred and "DOWN" in primary_pred.expected_move:
            share_plan = {
                "action": "WAIT",
                "description": "Bearish setup - wait for lower entries",
                "buy_zone": f"${current_price * 0.90:.2f} - ${current_price * 0.95:.2f}"
            }
        else:
            share_plan = {
                "action": "HOLD",
                "description": "No clear direction - hold existing positions"
            }
        
        # Options plan
        if btc_info.get("change_24h", 0) > 3 or overnight.change_pct > 3:
            option_plan = {
                "strategy": "BUY_CALLS",
                "strike": f"${int(current_price) + 5}",
                "expiry": "2-3 weeks out",
                "size": "2-5 contracts",
                "reasoning": "Strong momentum, ride the wave"
            }
        elif btc_info.get("change_24h", 0) < -3 or overnight.change_pct < -3:
            option_plan = {
                "strategy": "WAIT_FOR_BOTTOM",
                "reasoning": "Let the dump play out, then buy calls on reversal"
            }
        else:
            option_plan = {
                "strategy": "SELL_COVERED_CALLS",
                "strike": f"${int(current_price) + 3}",
                "expiry": "1 week out",
                "reasoning": "Range-bound, collect premium"
            }
        
        # Timing
        best_times = [
            "9:30-10:00 EST (Opening volatility)",
            "10:30-11:00 EST (Post-opening reversal)",
            "15:00-16:00 EST (Power hour momentum)"
        ]
        
        avoid_times = ["12:00-14:00 EST (Lunch lull, low volume)"]
        
        day_of_week = datetime.utcnow().strftime("%A")
        if day_of_week == "Friday":
            avoid_times.append("15:30-16:00 EST (Options expiry volatility)")
        
        # Risk
        if overnight.volatility == "HIGH" or abs(btc_info.get("change_24h", 0)) > 5:
            risk_level = "HIGH ⚠️ - Reduce position size 50%"
        elif overnight.volatility == "LOW":
            risk_level = "LOW - Good for larger positions"
        else:
            risk_level = "NORMAL"
        
        return TradingPlan(
            share_plan=share_plan,
            option_plan=option_plan,
            best_times=best_times,
            avoid_times=avoid_times,
            risk_level=risk_level
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN REPORT GENERATOR
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_report(self) -> Dict:
        """Generate comprehensive pre-market report for IREN"""
        
        logger.info("=" * 60)
        logger.info("🌅 GENERATING IREN PRE-MARKET REPORT")
        logger.info("=" * 60)
        
        # Fetch data
        iren_data = self.fetch_iren_data("10d", "1h")
        iren_price, btc_price = self.get_current_price()
        
        if iren_price == 0 and not iren_data.empty:
            iren_price = float(iren_data['close'].iloc[-1])
        
        logger.info(f"IREN: ${iren_price:.2f} | BTC: ${btc_price:,.0f}")
        
        # Analysis
        overnight = self.analyze_overnight_range(iren_data)
        btc_info = self.get_btc_correlation()
        liquidity_pools = self.find_liquidity_pools(iren_data, iren_price)
        predictions = self.predict_mm_behavior(iren_price, liquidity_pools, overnight, btc_info.get("change_24h", 0))
        options_flow = self.get_options_flow(iren_price)
        trading_plan = self.generate_trading_plan(iren_price, liquidity_pools, predictions, overnight, btc_info)
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": "IREN",
            "current_price": iren_price,
            "overnight": {
                "high": overnight.high,
                "low": overnight.low,
                "prev_close": overnight.prev_close,
                "change_pct": overnight.change_pct,
                "gap_pct": overnight.gap_pct,
                "volatility": overnight.volatility,
                "bias": overnight.bias
            },
            "btc_correlation": btc_info,
            "liquidity_pools": [
                {
                    "level": p.level,
                    "type": p.pool_type,
                    "distance_pct": p.distance_pct,
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
                    "expected_move": p.expected_move,
                    "target_level": p.target_level,
                    "action": p.action,
                    "timing": p.timing
                }
                for p in predictions
            ],
            "options_flow": options_flow,
            "trading_plan": {
                "shares": trading_plan.share_plan,
                "options": trading_plan.option_plan,
                "best_times": trading_plan.best_times,
                "avoid_times": trading_plan.avoid_times,
                "risk_level": trading_plan.risk_level
            }
        }
        
        logger.info("✅ IREN pre-market report generated!")
        return report
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TELEGRAM FORMATTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def format_telegram_report(self, report: Dict) -> str:
        """Format report for Telegram delivery"""
        
        price = report.get("current_price", 0)
        overnight = report.get("overnight", {})
        btc = report.get("btc_correlation", {})
        pools = report.get("liquidity_pools", [])
        predictions = report.get("predictions", [])
        options = report.get("options_flow", {})
        plan = report.get("trading_plan", {})
        
        day = datetime.utcnow().strftime("%A")
        date = datetime.utcnow().strftime("%B %d, %Y")
        
        # Bias emoji
        if overnight.get("bias") == "BULLISH":
            bias_emoji = "🟢"
        elif overnight.get("bias") == "BEARISH":
            bias_emoji = "🔴"
        else:
            bias_emoji = "🟡"
        
        lines = [
            f"🌅 <b>PRE-MARKET REPORT - IREN</b>",
            f"📅 {day}, {date}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Current:</b> ${price:.2f}",
            "",
            f"📊 <b>OVERNIGHT:</b>",
            f"├── High: ${overnight.get('high', 0):.2f}",
            f"├── Low: ${overnight.get('low', 0):.2f}",
            f"├── Change: {overnight.get('change_pct', 0):+.1f}%",
            f"├── Volatility: {overnight.get('volatility', 'N/A')}",
            f"└── Bias: {bias_emoji} {overnight.get('bias', 'N/A')}",
            "",
            f"₿ <b>BTC CORRELATION:</b>",
            f"├── BTC: ${btc.get('price', 0):,.0f}",
            f"├── 24h: {btc.get('change_24h', 0):+.1f}%",
            f"└── Signal: {btc.get('signal', 'N/A')}",
            "",
            f"💧 <b>LIQUIDITY POOLS:</b>",
        ]
        
        # Long stops
        long_pools = [p for p in pools if p.get('type') == 'LONG_STOPS'][:2]
        if long_pools:
            lines.append("<i>🔻 LONG STOPS (Dip targets):</i>")
            for p in long_pools:
                emoji = "🎯" if p.get('hunt_probability', 0) >= 70 else "📍"
                lines.append(f"{emoji} ${p.get('level', 0):.2f} ({p.get('distance_pct', 0):.1f}% away)")
        
        # Short stops
        short_pools = [p for p in pools if p.get('type') == 'SHORT_STOPS'][:2]
        if short_pools:
            lines.append("<i>🔺 SHORT STOPS (Squeeze targets):</i>")
            for p in short_pools:
                lines.append(f"📍 ${p.get('level', 0):.2f} ({p.get('distance_pct', 0):.1f}% away)")
        
        lines.append("")
        lines.append(f"🦊 <b>MM PREDICTION:</b>")
        
        for pred in predictions[:2]:
            emoji = "🎯" if pred.get('probability', 0) >= 60 else "📌"
            lines.append(f"{emoji} <b>Scenario {pred.get('scenario', '?')}:</b> ({pred.get('probability', 0)}%)")
            lines.append(f"   {pred.get('description', 'N/A')}")
            if pred.get('action'):
                lines.append(f"   ➜ <i>{pred.get('action', '')}</i>")
        
        lines.append("")
        lines.append(f"📈 <b>OPTIONS FLOW:</b>")
        lines.append(f"├── Signal: {options.get('signal', 'N/A')}")
        lines.append(f"├── C/P Ratio: {options.get('call_put_ratio', 1):.2f}")
        lines.append(f"└── Max Pain: ${options.get('max_pain', 0):.2f}")
        
        lines.append("")
        lines.append(f"📋 <b>TODAY'S PLAN:</b>")
        
        shares = plan.get('shares', {})
        if shares.get('action'):
            lines.append(f"<b>Shares:</b> {shares.get('action', 'N/A')}")
            if shares.get('entry'):
                lines.append(f"├── Entry: {shares.get('entry', 'N/A')}")
            if shares.get('target'):
                lines.append(f"├── Target: {shares.get('target', 'N/A')}")
            if shares.get('stop_loss'):
                lines.append(f"└── SL: {shares.get('stop_loss', 'N/A')}")
            if shares.get('description'):
                lines.append(f"   {shares.get('description', '')}")
        
        opts = plan.get('options', {})
        if opts.get('strategy'):
            lines.append(f"<b>Options:</b> {opts.get('strategy', 'N/A')}")
            if opts.get('strike'):
                lines.append(f"   Strike: {opts.get('strike', 'N/A')}, Expiry: {opts.get('expiry', 'N/A')}")
            if opts.get('reasoning'):
                lines.append(f"   → {opts.get('reasoning', '')}")
        
        lines.append("")
        lines.append(f"⏰ <b>BEST TIMES:</b>")
        for t in plan.get('best_times', [])[:2]:
            lines.append(f"└── {t}")
        
        if plan.get('avoid_times'):
            lines.append(f"🚫 <b>AVOID:</b> {plan.get('avoid_times', [''])[0]}")
        
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


def send_iren_premarket_report() -> Dict:
    """Generate and send IREN pre-market report"""
    analyzer = IrenPreMarketAnalyzer()
    
    logger.info("📊 Generating IREN pre-market analysis...")
    report = analyzer.generate_report()
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "telegram_sent": False,
        "file_saved": False
    }
    
    # Send to Telegram
    logger.info("📱 Sending to Telegram...")
    message = analyzer.format_telegram_report(report)
    telegram_success = send_telegram_message(message)
    result["telegram_sent"] = telegram_success
    
    if telegram_success:
        logger.info("✅ IREN Telegram report sent!")
    else:
        logger.warning("⚠️ Telegram send failed")
    
    # Save to file
    report_file = f"/home/jbot/trading_ai/research/iren_premarket_{datetime.utcnow().strftime('%Y%m%d')}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        result["file_saved"] = True
    except:
        pass
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    send_telegram = "--send" in sys.argv or "-s" in sys.argv
    
    print("\n🌅 IREN PRE-MARKET ANALYSIS")
    print("=" * 60)
    
    analyzer = IrenPreMarketAnalyzer()
    report = analyzer.generate_report()
    
    # Print formatted report
    telegram_msg = analyzer.format_telegram_report(report)
    import re
    plain_msg = re.sub(r'<[^>]+>', '', telegram_msg)
    print(plain_msg)
    
    if send_telegram:
        print("\n📤 Sending to Telegram...")
        success = send_telegram_message(telegram_msg)
        print("✅ Sent!" if success else "❌ Failed to send")

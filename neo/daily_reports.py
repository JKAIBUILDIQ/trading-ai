#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO DAILY REPORT SYSTEM - 3 REPORTS PER DAY
═══════════════════════════════════════════════════════════════════════════════

1. PRE-MARKET:  Predictions & trading plan
2. MID-MARKET:  How predictions are playing out
3. POST-MARKET: Score predictions, learn, adjust weights

This creates a learning loop where NEO can evaluate and improve.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import requests
import yfinance as yf
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("NEO_DAILY_REPORTS")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s')
ADMIN_CHAT_ID = os.environ.get('ADMIN_CHAT_ID', '6776619257')
DATA_DIR = Path("/home/jbot/trading_ai/neo/daily_data")
DATA_DIR.mkdir(exist_ok=True)

# Consensus Engine endpoint
CONSENSUS_API = "http://localhost:8037"


def get_consensus_signal(symbol: str) -> Dict:
    """Get unified consensus signal from NEO + Meta Bot"""
    try:
        resp = requests.get(f"{CONSENSUS_API}/api/consensus/{symbol.lower()}", timeout=10)
        if resp.ok:
            return resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch consensus for {symbol}: {e}")
    return {"action": "HOLD", "confidence": 0, "consensus_level": "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DailyPrediction:
    """Stores pre-market predictions for later evaluation"""
    date: str
    symbol: str
    prediction_time: str
    
    # Pre-market state
    pre_market_price: float
    overnight_high: float
    overnight_low: float
    overnight_bias: str
    
    # Predictions
    primary_scenario: str
    primary_probability: int
    hunt_target: float
    expected_direction: str
    entry_level: float
    stop_loss: float
    take_profit: float
    
    # Actual outcomes (filled by post-market)
    actual_high: float = 0.0
    actual_low: float = 0.0
    actual_close: float = 0.0
    hunt_hit: bool = False
    direction_correct: bool = False
    tp_hit: bool = False
    sl_hit: bool = False
    pnl_pips: float = 0.0
    score: int = 0  # 0-100
    
    # Learning notes
    lesson_learned: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str, chat_id: str = None) -> bool:
    """Send message via Telegram"""
    try:
        if not chat_id:
            chat_id = ADMIN_CHAT_ID
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
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


def fetch_ohlcv(symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data"""
    try:
        ticker_map = {
            "XAUUSD": "GC=F",
            "IREN": "IREN",
            "BTC": "BTC-USD"
        }
        ticker = ticker_map.get(symbol, symbol)
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if not data.empty:
            data.columns = [c.lower() for c in data.columns]
        return data
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def get_today_file(symbol: str) -> Path:
    """Get path to today's prediction file"""
    date_str = datetime.utcnow().strftime("%Y%m%d")
    return DATA_DIR / f"{symbol}_{date_str}.json"


def save_prediction(pred: DailyPrediction):
    """Save prediction to file"""
    file_path = get_today_file(pred.symbol)
    with open(file_path, 'w') as f:
        json.dump(asdict(pred), f, indent=2)
    logger.info(f"Saved prediction to {file_path}")


def load_prediction(symbol: str) -> Optional[DailyPrediction]:
    """Load today's prediction (handles extra fields gracefully)"""
    file_path = get_today_file(symbol)
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
            # Filter to only known DailyPrediction fields
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(DailyPrediction)}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
            return DailyPrediction(**filtered_data)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# XAUUSD REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

class XAUUSDReports:
    """XAUUSD Daily Report Generator"""
    
    def __init__(self):
        self.symbol = "XAUUSD"
    
    def get_current_price(self) -> float:
        """Get current Gold price"""
        df = fetch_ohlcv("XAUUSD", "1d", "1m")
        if not df.empty:
            return float(df['close'].iloc[-1])
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRE-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_premarket(self) -> str:
        """Generate pre-market report and save predictions"""
        
        # Import existing pre-market analyzer
        sys.path.insert(0, '/home/jbot/trading_ai/neo')
        from premarket_report import PreMarketAnalyzer
        
        analyzer = PreMarketAnalyzer()
        report = analyzer.generate_report("XAUUSD")
        
        # Extract key predictions to save
        predictions = report.get("predictions", [])
        plan = report.get("trading_plan", {}).get("primary", {})
        overnight = report.get("overnight", {})
        pools = report.get("liquidity_pools", [])
        
        primary_pred = predictions[0] if predictions else {}
        
        # Find hunt target
        hunt_target = primary_pred.get("hunt_level", 0)
        if not hunt_target and pools:
            for pool in pools:
                if pool.get("hunt_probability", 0) >= 60:
                    hunt_target = pool.get("level", 0)
                    break
        
        # Parse entry/SL/TP from plan
        import re
        def extract_price(s):
            if not s:
                return 0
            matches = re.findall(r'\$?([\d,]+\.?\d*)', str(s))
            return float(matches[0].replace(',', '')) if matches else 0
        
        entry = extract_price(plan.get("wait_for", "")) or extract_price(plan.get("entry", ""))
        sl = extract_price(plan.get("stop_loss", ""))
        tp = extract_price(plan.get("take_profit_1", ""))
        
        # Save prediction for later evaluation
        pred = DailyPrediction(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            symbol="XAUUSD",
            prediction_time=datetime.utcnow().isoformat(),
            pre_market_price=report.get("current_price", 0),
            overnight_high=overnight.get("high", 0),
            overnight_low=overnight.get("low", 0),
            overnight_bias=overnight.get("bias", "UNKNOWN"),
            primary_scenario=primary_pred.get("scenario", "N/A"),
            primary_probability=primary_pred.get("probability", 0),
            hunt_target=hunt_target,
            expected_direction=primary_pred.get("expected_hunt", "NONE"),
            entry_level=entry or hunt_target,
            stop_loss=sl or (entry - 20 if entry else 0),
            take_profit=tp or (entry + 40 if entry else 0)
        )
        save_prediction(pred)
        
        # Return formatted Telegram message
        return analyzer.format_telegram_report(report)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MID-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_midmarket(self) -> str:
        """Generate mid-market report - how predictions are playing out"""
        
        pred = load_prediction("XAUUSD")
        if not pred:
            return "⚠️ No pre-market prediction found for today"
        
        # Get current market data
        df = fetch_ohlcv("XAUUSD", "1d", "5m")
        if df.empty:
            return "⚠️ Could not fetch market data"
        
        current_price = float(df['close'].iloc[-1])
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        
        # Calculate moves
        move_from_open = current_price - pred.pre_market_price
        move_pct = (move_from_open / pred.pre_market_price) * 100
        
        # Check prediction status
        hunt_status = "❌ NOT HIT"
        if pred.hunt_target > 0:
            if pred.expected_direction == "LONGS" and session_low <= pred.hunt_target:
                hunt_status = f"✅ HIT at ${pred.hunt_target:.0f}!"
            elif pred.expected_direction == "SHORTS" and session_high >= pred.hunt_target:
                hunt_status = f"✅ HIT at ${pred.hunt_target:.0f}!"
            elif abs(current_price - pred.hunt_target) < 20:
                hunt_status = f"⏳ APPROACHING (${abs(current_price - pred.hunt_target):.0f} away)"
        
        # Entry status
        entry_status = "⏳ WAITING"
        if pred.entry_level > 0:
            if pred.expected_direction == "LONGS" and session_low <= pred.entry_level:
                entry_status = f"✅ TRIGGERED at ${pred.entry_level:.0f}"
            elif pred.expected_direction == "SHORTS" and session_high >= pred.entry_level:
                entry_status = f"✅ TRIGGERED at ${pred.entry_level:.0f}"
        
        # Direction check
        if pred.expected_direction == "LONGS":
            direction_status = "✅ CORRECT" if move_from_open > 0 else "⚠️ WRONG (so far)"
        elif pred.expected_direction == "SHORTS":
            direction_status = "✅ CORRECT" if move_from_open < 0 else "⚠️ WRONG (so far)"
        else:
            direction_status = "N/A"
        
        # Volatility assessment
        session_range = session_high - session_low
        avg_range = 50  # XAUUSD typical
        if session_range > avg_range * 1.5:
            volatility = "🔥 HIGH"
        elif session_range < avg_range * 0.7:
            volatility = "😴 LOW"
        else:
            volatility = "📊 NORMAL"
        
        # Get UNIFIED CONSENSUS signal (NEO + Meta Bot combined)
        consensus = get_consensus_signal("XAUUSD")
        consensus_action = consensus.get("action", "HOLD")
        consensus_conf = consensus.get("confidence", 0)
        consensus_level = consensus.get("consensus_level", "UNKNOWN")
        neo_signal = consensus.get("neo", {})
        meta_signal = consensus.get("meta", {})
        supertrend = consensus.get("supertrend", {})
        conflict_warning = consensus.get("conflict_warning")
        
        # Consensus emoji
        level_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "WEAK": "🟠", "NEUTRAL": "⚪", "CONFLICT": "🔴"}
        action_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}
        
        day = datetime.utcnow().strftime("%A")
        time_now = datetime.utcnow().strftime("%H:%M UTC")
        
        lines = [
            f"📊 <b>MID-MARKET REPORT - XAUUSD</b>",
            f"📅 {day} | ⏰ {time_now}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📍 <b>CURRENT STATUS:</b>",
            f"├── Price: ${current_price:.2f}",
            f"├── Open: ${pred.pre_market_price:.2f}",
            f"├── Move: {move_from_open:+.0f} pts ({move_pct:+.1f}%)",
            f"├── High: ${session_high:.2f}",
            f"├── Low: ${session_low:.2f}",
            f"└── Volatility: {volatility}",
            f"",
            f"🎯 <b>PREDICTION CHECK:</b>",
            f"├── Scenario: {pred.primary_scenario} ({pred.primary_probability}%)",
            f"├── Expected: Hunt {pred.expected_direction}",
            f"├── Hunt Target: ${pred.hunt_target:.0f} → {hunt_status}",
            f"├── Entry Level: ${pred.entry_level:.0f} → {entry_status}",
            f"└── Direction: {direction_status}",
            f"",
            f"🤝 <b>UNIFIED CONSENSUS:</b>",
            f"{level_emoji.get(consensus_level, '⚪')} {action_emoji.get(consensus_action, '❓')} <b>{consensus_action}</b> ({consensus_conf}%) - {consensus_level}",
            f"├── NEO: {neo_signal.get('action', 'N/A')} ({neo_signal.get('confidence', 0)}%)",
            f"├── Meta Bot: {meta_signal.get('action', 'N/A')} ({meta_signal.get('confidence', 0)}%)",
            f"└── SuperTrend: {supertrend.get('direction', 'N/A')} {'✅' if supertrend.get('agrees') else '⚠️'}",
            f"",
        ]
        
        # Add conflict warning if present
        if conflict_warning:
            lines.append(f"<b>{conflict_warning}</b>")
            lines.append(f"")
        
        # Add recommendation
        if "TRIGGERED" in entry_status or "HIT" in hunt_status:
            lines.append(f"✅ <b>SETUP PLAYED OUT!</b>")
            lines.append(f"   Check if reversal happened as predicted")
        elif "APPROACHING" in hunt_status:
            lines.append(f"⏳ <b>SETUP DEVELOPING</b>")
            lines.append(f"   Hunt target approaching - watch for reversal")
        elif consensus_level == "CONFLICT":
            lines.append(f"🔴 <b>CONFLICTING SIGNALS - HOLD</b>")
            lines.append(f"   Wait for NEO and Meta to align before trading")
        else:
            lines.append(f"📋 <b>WAITING FOR SETUP</b>")
            lines.append(f"   Pre-market levels not yet tested")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POST-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_postmarket(self) -> str:
        """Generate post-market report - score predictions and learn"""
        
        pred = load_prediction("XAUUSD")
        if not pred:
            return "⚠️ No pre-market prediction found for today"
        
        # Get full day's data
        df = fetch_ohlcv("XAUUSD", "1d", "5m")
        if df.empty:
            return "⚠️ Could not fetch market data"
        
        current_price = float(df['close'].iloc[-1])
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        
        # Update prediction with actuals
        pred.actual_high = session_high
        pred.actual_low = session_low
        pred.actual_close = current_price
        
        # ═══ SCORING ═══
        score = 0
        lessons = []
        
        # 1. Hunt target hit? (30 points)
        if pred.hunt_target > 0:
            if pred.expected_direction == "LONGS" and session_low <= pred.hunt_target:
                pred.hunt_hit = True
                score += 30
                lessons.append("✅ Hunt target HIT")
            elif pred.expected_direction == "SHORTS" and session_high >= pred.hunt_target:
                pred.hunt_hit = True
                score += 30
                lessons.append("✅ Hunt target HIT")
            else:
                lessons.append(f"❌ Hunt target MISSED (needed ${pred.hunt_target:.0f})")
        
        # 2. Direction correct? (25 points)
        move = current_price - pred.pre_market_price
        if pred.expected_direction == "LONGS" and move > 20:
            pred.direction_correct = True
            score += 25
            lessons.append("✅ Direction CORRECT (bullish)")
        elif pred.expected_direction == "SHORTS" and move < -20:
            pred.direction_correct = True
            score += 25
            lessons.append("✅ Direction CORRECT (bearish)")
        elif abs(move) < 20:
            score += 10  # Partial credit for range
            lessons.append("⚪ RANGE day - direction unclear")
        else:
            lessons.append(f"❌ Direction WRONG (moved {move:+.0f})")
        
        # 3. TP hit? (25 points)
        if pred.take_profit > 0:
            if pred.expected_direction == "LONGS" and session_high >= pred.take_profit:
                pred.tp_hit = True
                score += 25
                lessons.append(f"✅ TP HIT at ${pred.take_profit:.0f}")
            elif pred.expected_direction == "SHORTS" and session_low <= pred.take_profit:
                pred.tp_hit = True
                score += 25
                lessons.append(f"✅ TP HIT at ${pred.take_profit:.0f}")
            else:
                lessons.append(f"❌ TP NOT hit (target ${pred.take_profit:.0f})")
        
        # 4. SL NOT hit? (20 points)
        if pred.stop_loss > 0:
            if pred.expected_direction == "LONGS" and session_low > pred.stop_loss:
                score += 20
                lessons.append("✅ SL NOT hit (good risk management)")
            elif pred.expected_direction == "SHORTS" and session_high < pred.stop_loss:
                score += 20
                lessons.append("✅ SL NOT hit (good risk management)")
            else:
                pred.sl_hit = True
                lessons.append(f"❌ SL HIT at ${pred.stop_loss:.0f}")
        
        # Calculate P&L (if trade was taken)
        if pred.entry_level > 0:
            if pred.expected_direction == "LONGS":
                if pred.tp_hit:
                    pred.pnl_pips = pred.take_profit - pred.entry_level
                elif pred.sl_hit:
                    pred.pnl_pips = pred.stop_loss - pred.entry_level
                else:
                    pred.pnl_pips = current_price - pred.entry_level
            else:  # SHORTS
                if pred.tp_hit:
                    pred.pnl_pips = pred.entry_level - pred.take_profit
                elif pred.sl_hit:
                    pred.pnl_pips = pred.entry_level - pred.stop_loss
                else:
                    pred.pnl_pips = pred.entry_level - current_price
        
        pred.score = score
        
        # Generate lesson learned
        if score >= 80:
            pred.lesson_learned = "EXCELLENT prediction - maintain current approach"
        elif score >= 60:
            pred.lesson_learned = "GOOD prediction - minor adjustments needed"
        elif score >= 40:
            pred.lesson_learned = "FAIR prediction - review hunt level calculations"
        else:
            pred.lesson_learned = "POOR prediction - significant model adjustment needed"
        
        # Save updated prediction
        save_prediction(pred)
        
        # ═══ FORMAT REPORT ═══
        grade = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "D" if score >= 50 else "F"
        grade_emoji = "🏆" if grade in ["A+", "A"] else "✅" if grade == "B" else "😐" if grade == "C" else "⚠️"
        
        pnl_emoji = "💰" if pred.pnl_pips > 0 else "🔴"
        
        day = datetime.utcnow().strftime("%A")
        
        lines = [
            f"🌙 <b>POST-MARKET REPORT - XAUUSD</b>",
            f"📅 {day}, {pred.date}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 <b>DAY SUMMARY:</b>",
            f"├── Open: ${pred.pre_market_price:.2f}",
            f"├── High: ${session_high:.2f}",
            f"├── Low: ${session_low:.2f}",
            f"├── Close: ${current_price:.2f}",
            f"└── Move: {current_price - pred.pre_market_price:+.0f} pts",
            f"",
            f"🎯 <b>PREDICTION SCORECARD:</b>",
            f"",
        ]
        
        for lesson in lessons:
            lines.append(f"   {lesson}")
        
        lines.extend([
            f"",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{grade_emoji} <b>GRADE: {grade}</b> ({score}/100)",
            f"",
            f"{pnl_emoji} <b>P&L:</b> {pred.pnl_pips:+.0f} pips",
            f"",
            f"📚 <b>LESSON:</b>",
            f"   {pred.lesson_learned}",
            f"",
        ])
        
        # Add specific recommendations
        if not pred.hunt_hit and pred.hunt_target > 0:
            lines.append(f"🔧 <b>ADJUSTMENT:</b>")
            if session_low < pred.hunt_target:
                lines.append(f"   Hunt target was too high - lower by {pred.hunt_target - session_low:.0f} pts")
            else:
                lines.append(f"   Hunt target was too low - raise by {session_low - pred.hunt_target:.0f} pts")
        
        if not pred.direction_correct:
            lines.append(f"🔧 <b>ADJUSTMENT:</b>")
            lines.append(f"   Bias was wrong - review correlation signals")
        
        # Calculate running accuracy
        lines.append(f"")
        lines.append(f"📈 <b>LEARNING STATS:</b>")
        lines.append(f"   (Run weekly accuracy review for trend)")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# IREN REPORTS  
# ═══════════════════════════════════════════════════════════════════════════════

class IRENReports:
    """IREN Daily Report Generator"""
    
    def __init__(self):
        self.symbol = "IREN"
    
    def get_current_price(self) -> float:
        """Get current IREN price"""
        df = fetch_ohlcv("IREN", "1d", "1m")
        if not df.empty:
            return float(df['close'].iloc[-1])
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRE-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_premarket(self) -> str:
        """Generate IREN pre-market report"""
        from iren_premarket_report import IrenPreMarketAnalyzer
        
        analyzer = IrenPreMarketAnalyzer()
        report = analyzer.generate_report()
        
        # Save prediction
        predictions = report.get("predictions", [])
        plan = report.get("trading_plan", {})
        overnight = report.get("overnight", {})
        pools = report.get("liquidity_pools", [])
        
        primary_pred = predictions[0] if predictions else {}
        
        # Find key levels
        hunt_target = 0
        for pool in pools:
            if pool.get("hunt_probability", 0) >= 60:
                hunt_target = pool.get("level", 0)
                break
        
        shares_plan = plan.get("shares", {})
        import re
        def extract_price(s):
            if not s:
                return 0
            matches = re.findall(r'\$?([\d.]+)', str(s))
            return float(matches[0]) if matches else 0
        
        entry = extract_price(shares_plan.get("entry", ""))
        sl = extract_price(shares_plan.get("stop_loss", ""))
        tp = extract_price(shares_plan.get("target", ""))
        
        pred = DailyPrediction(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            symbol="IREN",
            prediction_time=datetime.utcnow().isoformat(),
            pre_market_price=report.get("current_price", 0),
            overnight_high=overnight.get("high", 0),
            overnight_low=overnight.get("low", 0),
            overnight_bias=overnight.get("bias", "UNKNOWN"),
            primary_scenario=primary_pred.get("scenario", "N/A"),
            primary_probability=primary_pred.get("probability", 0),
            hunt_target=hunt_target,
            expected_direction="LONGS" if "UP" in primary_pred.get("expected_move", "") else "SHORTS" if "DOWN" in primary_pred.get("expected_move", "") else "NONE",
            entry_level=entry,
            stop_loss=sl,
            take_profit=tp
        )
        save_prediction(pred)
        
        return analyzer.format_telegram_report(report)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MID-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_midmarket(self) -> str:
        """Generate IREN mid-market report"""
        
        pred = load_prediction("IREN")
        if not pred:
            return "⚠️ No pre-market prediction found for IREN today"
        
        df = fetch_ohlcv("IREN", "1d", "5m")
        if df.empty:
            return "⚠️ Could not fetch IREN data"
        
        current_price = float(df['close'].iloc[-1])
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        
        move = current_price - pred.pre_market_price
        move_pct = (move / pred.pre_market_price) * 100
        
        # Get BTC status
        btc_df = fetch_ohlcv("BTC", "1d", "1h")
        btc_price = float(btc_df['close'].iloc[-1]) if not btc_df.empty else 0
        btc_change = ((btc_price - float(btc_df['close'].iloc[-24])) / float(btc_df['close'].iloc[-24])) * 100 if len(btc_df) >= 24 else 0
        
        # Direction check
        if pred.expected_direction == "LONGS":
            direction_status = "✅ CORRECT" if move > 0 else "⚠️ WRONG"
        elif pred.expected_direction == "SHORTS":
            direction_status = "✅ CORRECT" if move < 0 else "⚠️ WRONG"
        else:
            direction_status = "N/A"
        
        # Get UNIFIED CONSENSUS signal (NEO + Meta Bot combined)
        consensus = get_consensus_signal("IREN")
        consensus_action = consensus.get("action", "HOLD")
        consensus_conf = consensus.get("confidence", 0)
        consensus_level = consensus.get("consensus_level", "UNKNOWN")
        neo_signal = consensus.get("neo", {})
        meta_signal = consensus.get("meta", {})
        supertrend = consensus.get("supertrend", {})
        conflict_warning = consensus.get("conflict_warning")
        
        # Consensus emoji
        level_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "WEAK": "🟠", "NEUTRAL": "⚪", "CONFLICT": "🔴"}
        action_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}
        
        time_now = datetime.utcnow().strftime("%H:%M UTC")
        
        lines = [
            f"📊 <b>MID-MARKET REPORT - IREN</b>",
            f"⏰ {time_now}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📍 <b>CURRENT STATUS:</b>",
            f"├── Price: ${current_price:.2f}",
            f"├── Open: ${pred.pre_market_price:.2f}",
            f"├── Move: {move:+.2f} ({move_pct:+.1f}%)",
            f"├── High: ${session_high:.2f}",
            f"├── Low: ${session_low:.2f}",
            f"",
            f"₿ <b>BTC:</b> ${btc_price:,.0f} ({btc_change:+.1f}%)",
            f"",
            f"🎯 <b>PREDICTION CHECK:</b>",
            f"├── Scenario: {pred.primary_scenario} ({pred.primary_probability}%)",
            f"├── Expected: {pred.expected_direction}",
            f"└── Direction: {direction_status}",
            f"",
            f"🤝 <b>UNIFIED CONSENSUS:</b>",
            f"{level_emoji.get(consensus_level, '⚪')} {action_emoji.get(consensus_action, '❓')} <b>{consensus_action}</b> ({consensus_conf}%) - {consensus_level}",
            f"├── NEO: {neo_signal.get('action', 'N/A')} ({neo_signal.get('confidence', 0)}%)",
            f"├── Meta Bot: {meta_signal.get('action', 'N/A')} ({meta_signal.get('confidence', 0)}%)",
            f"└── SuperTrend: {supertrend.get('direction', 'N/A')} {'✅' if supertrend.get('agrees') else '⚠️'}",
        ]
        
        # Add conflict warning if present
        if conflict_warning:
            lines.append(f"")
            lines.append(f"<b>{conflict_warning}</b>")
        
        # Trading recommendation
        if move_pct < -5:
            lines.append(f"")
            lines.append(f"🛒 <b>DIP OPPORTUNITY?</b>")
            lines.append(f"   -{abs(move_pct):.1f}% drop - consider accumulation")
        elif move_pct > 5:
            lines.append(f"")
            lines.append(f"🎯 <b>RALLY IN PROGRESS</b>")
            lines.append(f"   +{move_pct:.1f}% - consider taking profits")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POST-MARKET REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_postmarket(self) -> str:
        """Generate IREN post-market report"""
        
        pred = load_prediction("IREN")
        if not pred:
            return "⚠️ No pre-market prediction found for IREN today"
        
        df = fetch_ohlcv("IREN", "1d", "5m")
        if df.empty:
            return "⚠️ Could not fetch IREN data"
        
        current_price = float(df['close'].iloc[-1])
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        
        # Update prediction
        pred.actual_high = session_high
        pred.actual_low = session_low
        pred.actual_close = current_price
        
        # Scoring
        score = 0
        lessons = []
        
        move = current_price - pred.pre_market_price
        move_pct = (move / pred.pre_market_price) * 100
        
        # Direction (40 points)
        if pred.expected_direction == "LONGS" and move > 0:
            pred.direction_correct = True
            score += 40
            lessons.append(f"✅ Direction CORRECT (+{move_pct:.1f}%)")
        elif pred.expected_direction == "SHORTS" and move < 0:
            pred.direction_correct = True
            score += 40
            lessons.append(f"✅ Direction CORRECT ({move_pct:.1f}%)")
        elif abs(move_pct) < 2:
            score += 20
            lessons.append(f"⚪ FLAT day ({move_pct:+.1f}%)")
        else:
            lessons.append(f"❌ Direction WRONG ({move_pct:+.1f}%)")
        
        # Entry hit (30 points)
        if pred.entry_level > 0:
            if session_low <= pred.entry_level <= session_high:
                score += 30
                lessons.append(f"✅ Entry level HIT (${pred.entry_level:.2f})")
            else:
                lessons.append(f"❌ Entry level MISSED (${pred.entry_level:.2f})")
        
        # TP/SL (30 points)
        if pred.take_profit > 0 and session_high >= pred.take_profit:
            pred.tp_hit = True
            score += 30
            lessons.append(f"✅ TP HIT (${pred.take_profit:.2f})")
        elif pred.stop_loss > 0 and session_low <= pred.stop_loss:
            pred.sl_hit = True
            lessons.append(f"❌ SL HIT (${pred.stop_loss:.2f})")
        else:
            if pred.take_profit > 0:
                lessons.append(f"⏳ TP not reached (${pred.take_profit:.2f})")
        
        pred.score = score
        pred.pnl_pips = move  # For stocks, use $ instead of pips
        
        if score >= 70:
            pred.lesson_learned = "Good IREN prediction - BTC correlation worked"
        elif score >= 50:
            pred.lesson_learned = "Fair prediction - review entry timing"
        else:
            pred.lesson_learned = "Poor prediction - check BTC divergence"
        
        save_prediction(pred)
        
        grade = "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "D" if score >= 50 else "F"
        
        lines = [
            f"🌙 <b>POST-MARKET REPORT - IREN</b>",
            f"📅 {pred.date}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 <b>DAY SUMMARY:</b>",
            f"├── Open: ${pred.pre_market_price:.2f}",
            f"├── High: ${session_high:.2f}",
            f"├── Low: ${session_low:.2f}",
            f"├── Close: ${current_price:.2f}",
            f"└── Move: {move_pct:+.1f}%",
            f"",
            f"🎯 <b>SCORECARD:</b>",
        ]
        
        for lesson in lessons:
            lines.append(f"   {lesson}")
        
        lines.extend([
            f"",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 <b>GRADE: {grade}</b> ({score}/100)",
            f"💰 <b>P&L:</b> ${pred.pnl_pips:+.2f}/share",
            f"",
            f"📚 <b>LESSON:</b> {pred.lesson_learned}",
        ])
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def send_xauusd_premarket():
    """Send XAUUSD pre-market report"""
    reporter = XAUUSDReports()
    msg = reporter.generate_premarket()
    return send_telegram(msg)

def send_xauusd_midmarket():
    """Send XAUUSD mid-market report"""
    reporter = XAUUSDReports()
    msg = reporter.generate_midmarket()
    return send_telegram(msg)

def send_xauusd_postmarket():
    """Send XAUUSD post-market report"""
    reporter = XAUUSDReports()
    msg = reporter.generate_postmarket()
    return send_telegram(msg)

def send_iren_premarket():
    """Send IREN pre-market report"""
    reporter = IRENReports()
    msg = reporter.generate_premarket()
    return send_telegram(msg)

def send_iren_midmarket():
    """Send IREN mid-market report"""
    reporter = IRENReports()
    msg = reporter.generate_midmarket()
    return send_telegram(msg)

def send_iren_postmarket():
    """Send IREN post-market report"""
    reporter = IRENReports()
    msg = reporter.generate_postmarket()
    return send_telegram(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python daily_reports.py <symbol> <report_type>")
        print("  symbol: XAUUSD or IREN")
        print("  report_type: pre, mid, or post")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    report_type = sys.argv[2].lower()
    send = "--send" in sys.argv or "-s" in sys.argv
    
    if symbol == "XAUUSD":
        reporter = XAUUSDReports()
    elif symbol == "IREN":
        reporter = IRENReports()
    else:
        print(f"Unknown symbol: {symbol}")
        sys.exit(1)
    
    if report_type == "pre":
        msg = reporter.generate_premarket()
    elif report_type == "mid":
        msg = reporter.generate_midmarket()
    elif report_type == "post":
        msg = reporter.generate_postmarket()
    else:
        print(f"Unknown report type: {report_type}")
        sys.exit(1)
    
    # Print plain text
    import re
    print(re.sub(r'<[^>]+>', '', msg))
    
    if send:
        print("\n📤 Sending to Telegram...")
        success = send_telegram(msg)
        print("✅ Sent!" if success else "❌ Failed")

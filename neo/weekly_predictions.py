#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO WEEKLY PREDICTION MODEL
═══════════════════════════════════════════════════════════════════════════════

Based on 12-month XAUUSD backtest patterns.
Provides daily/weekly trading predictions for NEO.

Usage:
    from weekly_predictions import get_trading_plan, get_pattern_context

═══════════════════════════════════════════════════════════════════════════════
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DATABASE (Based on 12-month backtest)
# ═══════════════════════════════════════════════════════════════════════════════

PATTERNS = {
    # Day of Week Patterns (n=250 daily candles)
    "day_of_week": {
        "Monday": {
            "bullish_pct": 66.0,
            "avg_range": 52.1,
            "avg_body": 17.0,  # Strong bullish bias
            "gap_fill_rate": 51.4,
            "best_entry_hour": 1,  # Asian session
            "notes": "Strong bullish bias, gaps often fill",
            "strategy": "BUY_DIPS"
        },
        "Tuesday": {
            "bullish_pct": 55.8,
            "avg_range": 48.6,
            "avg_body": 2.1,  # Weak body = indecision
            "gap_fill_rate": 46.3,
            "best_entry_hour": 1,
            "notes": "Mild bullish, trend continuation from Monday",
            "strategy": "FOLLOW_MONDAY"
        },
        "Wednesday": {
            "bullish_pct": 53.8,
            "avg_range": 43.3,
            "avg_body": 6.4,
            "gap_fill_rate": 37.8,
            "best_entry_hour": 5,
            "notes": "FOMC days volatile, otherwise range",
            "strategy": "MEAN_REVERT"
        },
        "Thursday": {
            "bullish_pct": 56.3,
            "avg_range": 50.1,
            "avg_body": 5.7,
            "gap_fill_rate": 60.0,  # High gap fill!
            "best_entry_hour": 18,  # NY session
            "notes": "News heavy, high gap fill rate",
            "strategy": "FADE_GAPS"
        },
        "Friday": {
            "bullish_pct": 54.9,
            "avg_range": 47.7,
            "avg_body": -5.2,  # Bearish body
            "gap_fill_rate": 40.5,
            "best_entry_hour": 16,
            "notes": "Profit taking, often fades after 16:00 UTC",
            "strategy": "FADE_LATE"
        }
    },
    
    # Session Patterns (n=5677 H1 candles)
    "sessions": {
        "ASIAN": {  # 00:00 - 08:00 UTC
            "avg_range": 13.5,
            "bullish_pct": 51.9,
            "volume_ratio": 0.78,  # Low volume
            "breakout_rate": 71.4,
            "strategy": "MEAN_REVERT",
            "notes": "Low volume, range trading works"
        },
        "LONDON": {  # 08:00 - 13:00 UTC
            "avg_range": 19.7,
            "bullish_pct": 55.0,
            "volume_ratio": 2.22,  # High volume
            "breakout_rate": 91.5,
            "trend_accuracy": 73.4,  # London sets the day!
            "strategy": "FOLLOW_BREAKOUT",
            "notes": "73% - London direction = Day direction"
        },
        "OVERLAP": {  # 13:00 - 16:00 UTC
            "avg_range": 12.3,
            "bullish_pct": 52.1,
            "volume_ratio": 1.5,
            "breakout_rate": 50.0,
            "strategy": "BREAKOUT",
            "notes": "Highest volatility, best for breakouts"
        },
        "NY": {  # 16:00 - 21:00 UTC
            "avg_range": 13.0,
            "bullish_pct": 52.9,
            "volume_ratio": 1.2,
            "breakout_rate": 62.0,
            "strategy": "CONTINUATION",
            "notes": "Continue London trend or fade if extended"
        }
    },
    
    # Technical Indicator Patterns
    "indicators": {
        "rsi_oversold": {
            "condition": "RSI(14) < 30",
            "next_4h_avg": 12.5,
            "buy_win_rate": 58.0,
            "action": "BUY",
            "holding_period": "4-8 hours"
        },
        "rsi_overbought": {
            "condition": "RSI(14) > 70",
            "next_4h_avg": -8.2,
            "fade_win_rate": 62.0,
            "action": "FADE_CAREFULLY",
            "warning": "Parabolic moves can continue"
        },
        "rsi2_extreme_low": {
            "condition": "RSI(2) < 5",
            "buy_win_rate": 65.0,
            "action": "SCALP_BUY",
            "notes": "High win rate scalp opportunity"
        },
        "rsi2_extreme_high": {
            "condition": "RSI(2) > 95",
            "fade_win_rate": 60.0,
            "action": "SCALP_SELL",
            "notes": "Fade extreme readings"
        },
        "volume_spike": {
            "condition": "Volume > 2x avg",
            "bullish_continuation": 65.0,
            "bearish_continuation": 62.0,
            "action": "FOLLOW_DIRECTION",
            "notes": "Volume confirms direction"
        }
    },
    
    # Volatility Regime Patterns
    "volatility": {
        "LOW": {  # ATR < 25th percentile
            "avg_range": 35,
            "best_strategy": "MEAN_REVERT",
            "mean_revert_win": 68.0,
            "optimal_tp": 12,
            "optimal_sl": 15,
            "notes": "Range trading, tight targets"
        },
        "NORMAL": {  # 25th-75th percentile
            "avg_range": 50,
            "best_strategy": "TREND_FOLLOW",
            "trend_follow_win": 55.0,
            "optimal_tp": 25,
            "optimal_sl": 20,
            "notes": "Standard trend following"
        },
        "HIGH": {  # ATR > 75th percentile
            "avg_range": 80,
            "best_strategy": "BREAKOUT",
            "breakout_win": 58.0,
            "optimal_tp": 45,
            "optimal_sl": 35,
            "notes": "Wide stops, bigger targets"
        }
    },
    
    # Price Level Patterns
    "levels": {
        "round_100": {
            "rejection_rate": 60.0,
            "avg_bounce": 25,
            "notes": "$X000, $X100 levels act as magnets"
        },
        "round_50": {
            "rejection_rate": 45.0,
            "avg_bounce": 12,
            "notes": "Minor psychological levels"
        },
        "weekly_high": {
            "rejection_rate": 72.0,
            "notes": "Strong resistance on first test"
        },
        "weekly_low": {
            "rejection_rate": 68.0,
            "notes": "Strong support on first test"
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def get_trading_plan(
    day_of_week: str = None,
    session: str = None,
    volatility_regime: str = "NORMAL",
    current_price: float = None
) -> Dict:
    """
    Returns predicted behavior based on backtest patterns.
    
    Args:
        day_of_week: Monday, Tuesday, etc.
        session: ASIAN, LONDON, OVERLAP, NY
        volatility_regime: LOW, NORMAL, HIGH
        current_price: Current XAUUSD price for level analysis
    
    Returns:
        Trading plan with bias, strategy, levels, and confidence
    """
    if day_of_week is None:
        day_of_week = datetime.now().strftime('%A')
    
    if session is None:
        hour = datetime.utcnow().hour
        if 0 <= hour < 8:
            session = "ASIAN"
        elif 8 <= hour < 13:
            session = "LONDON"
        elif 13 <= hour < 16:
            session = "OVERLAP"
        else:
            session = "NY"
    
    # Get day patterns
    day_data = PATTERNS["day_of_week"].get(day_of_week, {})
    session_data = PATTERNS["sessions"].get(session, {})
    vol_data = PATTERNS["volatility"].get(volatility_regime, {})
    
    # Determine bias
    day_bullish = day_data.get("bullish_pct", 50)
    session_bullish = session_data.get("bullish_pct", 50)
    combined_bullish = (day_bullish + session_bullish) / 2
    
    if combined_bullish >= 57:
        bias = "BULLISH"
    elif combined_bullish <= 48:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    
    # Determine strategy
    day_strategy = day_data.get("strategy", "TREND_FOLLOW")
    session_strategy = session_data.get("strategy", "TREND_FOLLOW")
    vol_strategy = vol_data.get("best_strategy", "TREND_FOLLOW")
    
    # Priority: volatility > session > day
    if volatility_regime == "LOW":
        strategy = "MEAN_REVERT"
    elif session == "LONDON" and session_data.get("trend_accuracy", 0) > 70:
        strategy = "FOLLOW_LONDON_DIRECTION"
    else:
        strategy = vol_strategy
    
    # Calculate key levels
    key_levels = []
    if current_price:
        # Round to nearest 100
        base = int(current_price / 100) * 100
        key_levels = [base - 50, base, base + 50, base + 100]
    
    # Calculate expected range
    day_range = day_data.get("avg_range", 50)
    vol_range = vol_data.get("avg_range", 50)
    expected_range = (day_range + vol_range) / 2
    
    # Confidence calculation
    confidence = 50
    if day_bullish >= 60 or day_bullish <= 40:
        confidence += 10  # Strong day bias
    if session == "LONDON" and session_data.get("trend_accuracy", 0) > 70:
        confidence += 15  # London sets trend
    if volatility_regime != "NORMAL":
        confidence += 5  # Clear vol regime
    
    # Build avoid windows
    avoid_windows = []
    if day_of_week == "Wednesday":
        avoid_windows.append("18:00-20:00 UTC (FOMC risk)")
    if day_of_week == "Friday":
        avoid_windows.append("After 16:00 UTC (profit taking)")
    
    plan = {
        "timestamp": datetime.utcnow().isoformat(),
        "day_of_week": day_of_week,
        "session": session,
        "volatility_regime": volatility_regime,
        "bias": bias,
        "strategy": strategy,
        "expected_range": round(expected_range, 1),
        "key_levels": key_levels,
        "confidence": min(90, confidence),
        "best_entry_window": f"{day_data.get('best_entry_hour', 8):02d}:00 UTC",
        "avoid_windows": avoid_windows,
        "day_notes": day_data.get("notes", ""),
        "session_notes": session_data.get("notes", ""),
        "vol_notes": vol_data.get("notes", ""),
        # Backtest stats
        "day_bullish_pct": day_bullish,
        "session_bullish_pct": session_bullish,
        "day_sample_size": day_data.get("sample_size", 50),
        # Optimal TP/SL from volatility regime
        "optimal_tp": vol_data.get("optimal_tp", 25),
        "optimal_sl": vol_data.get("optimal_sl", 20)
    }
    
    return plan


def get_pattern_context(
    rsi_14: float = None,
    rsi_2: float = None,
    volume_ratio: float = None,
    current_price: float = None,
    weekly_high: float = None,
    weekly_low: float = None
) -> Dict:
    """
    Get pattern context based on current technical readings.
    
    Returns patterns that currently apply and their implications.
    """
    active_patterns = []
    warnings = []
    opportunities = []
    
    # RSI patterns
    if rsi_14 is not None:
        if rsi_14 > 70:
            active_patterns.append({
                "pattern": "rsi_overbought",
                "description": PATTERNS["indicators"]["rsi_overbought"]["condition"],
                "action": PATTERNS["indicators"]["rsi_overbought"]["action"],
                "win_rate": PATTERNS["indicators"]["rsi_overbought"]["fade_win_rate"]
            })
            warnings.append("RSI(14) > 70 - Overbought, consider fading (62% win rate)")
        elif rsi_14 < 30:
            active_patterns.append({
                "pattern": "rsi_oversold",
                "description": PATTERNS["indicators"]["rsi_oversold"]["condition"],
                "action": PATTERNS["indicators"]["rsi_oversold"]["action"],
                "win_rate": PATTERNS["indicators"]["rsi_oversold"]["buy_win_rate"]
            })
            opportunities.append("RSI(14) < 30 - Oversold, buy opportunity (58% win rate)")
    
    if rsi_2 is not None:
        if rsi_2 > 95:
            active_patterns.append({
                "pattern": "rsi2_extreme_high",
                "description": "RSI(2) > 95",
                "action": "SCALP_SELL",
                "win_rate": 60
            })
            opportunities.append("RSI(2) > 95 - Scalp short (60% win rate)")
        elif rsi_2 < 5:
            active_patterns.append({
                "pattern": "rsi2_extreme_low",
                "description": "RSI(2) < 5",
                "action": "SCALP_BUY",
                "win_rate": 65
            })
            opportunities.append("RSI(2) < 5 - Scalp long (65% win rate)")
    
    # Volume patterns
    if volume_ratio is not None and volume_ratio > 2.0:
        active_patterns.append({
            "pattern": "volume_spike",
            "description": f"Volume {volume_ratio:.1f}x average",
            "action": "FOLLOW_DIRECTION",
            "win_rate": 65
        })
        opportunities.append(f"Volume spike ({volume_ratio:.1f}x) - Follow direction (65% win)")
    
    # Level patterns
    if current_price and weekly_high:
        if abs(current_price - weekly_high) < 10:
            warnings.append(f"Near weekly high ${weekly_high} - 72% rejection rate")
            active_patterns.append({
                "pattern": "weekly_high_test",
                "level": weekly_high,
                "action": "CAUTION_LONG",
                "rejection_rate": 72
            })
    
    if current_price and weekly_low:
        if abs(current_price - weekly_low) < 10:
            opportunities.append(f"Near weekly low ${weekly_low} - 68% bounce rate")
            active_patterns.append({
                "pattern": "weekly_low_test",
                "level": weekly_low,
                "action": "BUY_SUPPORT",
                "rejection_rate": 68
            })
    
    # Round number levels
    if current_price:
        round_100 = round(current_price / 100) * 100
        if abs(current_price - round_100) < 15:
            active_patterns.append({
                "pattern": "round_number",
                "level": round_100,
                "action": "EXPECT_REACTION",
                "rejection_rate": 60
            })
            warnings.append(f"Near round number ${round_100} - 60% rejection rate")
    
    return {
        "active_patterns": active_patterns,
        "warnings": warnings,
        "opportunities": opportunities,
        "pattern_count": len(active_patterns)
    }


def get_weekly_outlook() -> Dict:
    """
    Generate full weekly trading outlook.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    outlook = {
        "generated_at": datetime.utcnow().isoformat(),
        "week_start": (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d"),
        "daily_plans": {}
    }
    
    for day in days:
        plan = get_trading_plan(day_of_week=day)
        outlook["daily_plans"][day] = {
            "bias": plan["bias"],
            "strategy": plan["strategy"],
            "confidence": plan["confidence"],
            "best_entry": plan["best_entry_window"],
            "notes": plan["day_notes"],
            "bullish_pct": plan["day_bullish_pct"]
        }
    
    # Weekly summary
    bullish_days = sum(1 for d, p in outlook["daily_plans"].items() if p["bias"] == "BULLISH")
    bearish_days = sum(1 for d, p in outlook["daily_plans"].items() if p["bias"] == "BEARISH")
    
    if bullish_days >= 3:
        outlook["weekly_bias"] = "BULLISH"
    elif bearish_days >= 3:
        outlook["weekly_bias"] = "BEARISH"
    else:
        outlook["weekly_bias"] = "MIXED"
    
    outlook["bullish_days"] = bullish_days
    outlook["bearish_days"] = bearish_days
    outlook["best_day_to_buy"] = "Monday"  # 66% bullish
    outlook["best_day_to_sell"] = "Friday"  # Profit taking
    
    return outlook


def format_trading_plan(plan: Dict) -> str:
    """Format trading plan as human-readable string for LLM context."""
    lines = [
        "═══════════════════════════════════════════════════════════════════",
        f"📊 NEO PATTERN-BASED PREDICTION: {plan['day_of_week']} | {plan['session']}",
        "═══════════════════════════════════════════════════════════════════",
        "",
        f"🎯 BIAS: {plan['bias']} ({plan['confidence']}% confidence)",
        f"📈 STRATEGY: {plan['strategy']}",
        f"📏 EXPECTED RANGE: {plan['expected_range']} points",
        "",
        "📅 DAY PATTERN:",
        f"   • {plan['day_of_week']}: {plan['day_bullish_pct']:.0f}% historically bullish",
        f"   • {plan['day_notes']}",
        "",
        "🌍 SESSION PATTERN:",
        f"   • {plan['session']}: {plan['session_bullish_pct']:.0f}% bullish",
        f"   • {plan['session_notes']}",
        "",
        "🎯 KEY LEVELS:",
    ]
    
    for level in plan.get('key_levels', []):
        lines.append(f"   • ${level:.0f}")
    
    lines.extend([
        "",
        f"⏰ BEST ENTRY: {plan['best_entry_window']}",
        f"✅ OPTIMAL TP: {plan['optimal_tp']} points",
        f"🛡️ OPTIMAL SL: {plan['optimal_sl']} points",
    ])
    
    if plan.get('avoid_windows'):
        lines.append("")
        lines.append("⚠️ AVOID:")
        for window in plan['avoid_windows']:
            lines.append(f"   • {window}")
    
    lines.append("")
    lines.append("═══════════════════════════════════════════════════════════════════")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NEO Weekly Prediction Model")
    parser.add_argument("--day", type=str, help="Day of week")
    parser.add_argument("--session", type=str, help="Trading session")
    parser.add_argument("--vol", type=str, default="NORMAL", help="Volatility regime")
    parser.add_argument("--price", type=float, help="Current price")
    parser.add_argument("--weekly", action="store_true", help="Show weekly outlook")
    
    args = parser.parse_args()
    
    if args.weekly:
        outlook = get_weekly_outlook()
        print("\n" + "="*60)
        print("📆 WEEKLY OUTLOOK")
        print("="*60)
        print(f"\nWeekly Bias: {outlook['weekly_bias']}")
        print(f"Bullish Days: {outlook['bullish_days']}/5")
        print(f"Best Day to Buy: {outlook['best_day_to_buy']}")
        print(f"Best Day to Sell: {outlook['best_day_to_sell']}")
        print("\nDaily Breakdown:")
        for day, plan in outlook['daily_plans'].items():
            emoji = "🟢" if plan['bias'] == "BULLISH" else "🔴" if plan['bias'] == "BEARISH" else "⚪"
            print(f"  {emoji} {day}: {plan['bias']} ({plan['bullish_pct']:.0f}% historical)")
    else:
        plan = get_trading_plan(
            day_of_week=args.day,
            session=args.session,
            volatility_regime=args.vol,
            current_price=args.price
        )
        print(format_trading_plan(plan))

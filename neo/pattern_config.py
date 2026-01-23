#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO PATTERN BOT CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Trading rules based on 12-month XAUUSD backtest:
- 5,677 H1 candles
- 250 D1 candles  
- Statistical patterns extracted with >55% win rate

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import time

# ═══════════════════════════════════════════════════════════════════════════════
# DAY OF WEEK RULES (From 12-month backtest)
# ═══════════════════════════════════════════════════════════════════════════════

DAY_RULES = {
    "Monday": {
        "bias": "BULLISH",
        "win_rate": 0.66,        # 66% historically bullish
        "avg_range": 52.1,       # Average daily range in points
        "avg_body": 17.0,        # Strong bullish body
        "gap_fill_rate": 0.51,   # 51% of gaps fill
        "strategy": "BUY_DIPS",
        "best_entry_hour": 1,    # Asian session
        "trade": True,
        "notes": "Strong bullish bias, buy pullbacks"
    },
    "Tuesday": {
        "bias": "FOLLOW_MONDAY",
        "win_rate": 0.56,        # 56% bullish
        "avg_range": 48.6,
        "avg_body": 2.1,         # Weak body = indecision
        "gap_fill_rate": 0.46,
        "strategy": "CONTINUATION",
        "best_entry_hour": 1,
        "trade": True,
        "notes": "Continue Monday's direction"
    },
    "Wednesday": {
        "bias": "NEUTRAL",
        "win_rate": 0.54,        # Near 50/50
        "avg_range": 43.3,       # Lowest range
        "avg_body": 6.4,
        "gap_fill_rate": 0.38,
        "strategy": "MEAN_REVERT",
        "best_entry_hour": 5,
        "trade": True,
        "avoid_hours": [18, 19, 20],  # FOMC risk
        "notes": "FOMC days volatile, otherwise range"
    },
    "Thursday": {
        "bias": "FADE_GAPS",
        "win_rate": 0.56,
        "avg_range": 50.1,
        "avg_body": 5.7,
        "gap_fill_rate": 0.60,   # High gap fill!
        "strategy": "GAP_FILL",
        "best_entry_hour": 18,   # NY session
        "trade": True,
        "notes": "High gap fill rate, fade overnight gaps"
    },
    "Friday": {
        "bias": "NEUTRAL",
        "win_rate": 0.55,
        "avg_range": 47.7,
        "avg_body": -5.2,        # Bearish body (profit taking)
        "gap_fill_rate": 0.41,
        "strategy": "FADE_LATE",
        "best_entry_hour": 16,
        "trade": True,
        "trade_cutoff_hour": 16, # No new trades after 16:00 UTC
        "notes": "Profit taking after 16:00, fade extensions"
    },
    "Saturday": {
        "bias": "NO_TRADE",
        "trade": False
    },
    "Sunday": {
        "bias": "NO_TRADE",
        "trade": False
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION RULES (From session analysis)
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_RULES = {
    "ASIAN": {
        "start_hour": 0,
        "end_hour": 8,
        "avg_range": 13.5,
        "bullish_pct": 51.9,
        "volume_ratio": 0.78,    # Low volume
        "breakout_rate": 0.71,
        "strategy": "MEAN_REVERT",
        "best_for": "scalping",
        "notes": "Low volume, mean reversion works best"
    },
    "LONDON": {
        "start_hour": 8,
        "end_hour": 13,
        "avg_range": 19.7,
        "bullish_pct": 55.0,
        "volume_ratio": 2.22,    # High volume!
        "breakout_rate": 0.91,   # Very high breakout rate
        "sets_day_direction": True,  # 73% accurate!
        "trend_accuracy": 0.73,
        "strategy": "FOLLOW_BREAKOUT",
        "best_for": "trend_trading",
        "notes": "73% - London direction = Day direction. Follow breakouts!"
    },
    "OVERLAP": {
        "start_hour": 13,
        "end_hour": 16,
        "avg_range": 12.3,
        "bullish_pct": 52.1,
        "volume_ratio": 1.5,
        "breakout_rate": 0.50,
        "strategy": "CAUTION",
        "high_volatility": True,
        "best_for": "experienced_only",
        "notes": "Highest volatility, unpredictable"
    },
    "NY": {
        "start_hour": 16,
        "end_hour": 21,
        "avg_range": 13.0,
        "bullish_pct": 52.9,
        "volume_ratio": 1.2,
        "breakout_rate": 0.62,
        "strategy": "FOLLOW_OR_FADE_LONDON",
        "best_for": "continuation_or_reversal",
        "notes": "Continue London or fade if extended"
    },
    "LATE_NY": {
        "start_hour": 21,
        "end_hour": 24,
        "avg_range": 8.0,
        "bullish_pct": 50.0,
        "volume_ratio": 0.5,
        "strategy": "NO_TRADE",
        "best_for": "sleep",
        "notes": "Low liquidity, avoid"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING RULES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSizing:
    base_lot: float = 0.01          # Minimum lot
    max_lot: float = 0.10           # Maximum per trade
    risk_per_trade: float = 0.01    # 1% of balance per trade
    max_daily_risk: float = 0.03    # 3% max daily drawdown
    max_positions: int = 3          # Max concurrent positions
    one_per_symbol: bool = True     # Only 1 position per symbol
    
POSITION_SIZING = PositionSizing()

# ═══════════════════════════════════════════════════════════════════════════════
# TAKE PROFIT / STOP LOSS RULES (From volatility analysis)
# ═══════════════════════════════════════════════════════════════════════════════

TP_SL_RULES = {
    "XAUUSD": {
        # Volatility-based defaults
        "LOW_VOL": {
            "tp_points": 12,
            "sl_points": 15,
            "strategy": "MEAN_REVERT"
        },
        "NORMAL_VOL": {
            "tp_points": 25,
            "sl_points": 20,
            "strategy": "TREND_FOLLOW"
        },
        "HIGH_VOL": {
            "tp_points": 45,
            "sl_points": 35,
            "strategy": "BREAKOUT"
        },
        # Day-specific adjustments
        "monday_tp_bonus": 5,      # Monday has bigger moves
        "friday_tp_reduction": 5,  # Friday tighter targets
        "max_sl": 40,              # Never risk more than 40 points
        "min_rr_ratio": 1.0        # Minimum risk:reward
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR RULES (From indicator analysis)
# ═══════════════════════════════════════════════════════════════════════════════

INDICATOR_RULES = {
    "rsi_14": {
        "overbought": 70,
        "oversold": 30,
        "overbought_action": "CAUTION_LONG",  # Don't buy when overbought
        "oversold_action": "BUY_OPPORTUNITY",  # 58% win rate on buy
        "fade_overbought_win_rate": 0.62
    },
    "rsi_2": {
        "extreme_high": 95,
        "extreme_low": 5,
        "extreme_high_action": "SCALP_SELL",  # 60% win rate
        "extreme_low_action": "SCALP_BUY",    # 65% win rate
    },
    "volume": {
        "spike_threshold": 2.0,  # 2x average
        "spike_action": "FOLLOW_DIRECTION",  # 65% continuation
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# CROWD PSYCHOLOGY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

CROWD_PSYCHOLOGY_RULES = {
    "crash_probability": {
        "green_light": 30,       # < 30% = trade normally
        "yellow_light": 50,      # 30-50% = reduce size
        "red_light": 70,         # > 70% = no new longs
        "extreme": 85            # > 85% = consider shorts
    },
    "fear_greed": {
        "extreme_fear": 20,      # Buy opportunity
        "fear": 40,
        "neutral": 50,
        "greed": 60,
        "extreme_greed": 80      # Caution
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# BOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BotConfig:
    # Identity
    bot_name: str = "NEO-PATTERN-BOT"
    magic_number: int = 777002          # Different from Ghost (777001)
    
    # Trading limits
    max_daily_trades: int = 5           # Max trades per day
    max_daily_loss_pct: float = 0.03    # 3% max daily loss
    max_positions: int = 3              # Max concurrent positions
    
    # Timing
    check_interval_seconds: int = 60    # Check every minute
    min_time_between_trades: int = 300  # 5 min between trades
    
    # Filters
    min_confidence: float = 55          # Minimum pattern confidence
    min_win_rate: float = 0.55          # Minimum historical win rate
    
    # MT5 Connection
    mt5_api_url: str = "http://146.190.188.208:8085"
    
    # Telegram
    telegram_enabled: bool = True
    telegram_alert_trades: bool = True
    
    # Logging
    log_file: str = "/home/jbot/trading_ai/logs/pattern_bot.log"
    
BOT_CONFIG = BotConfig()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_session(hour: int) -> str:
    """Get current trading session based on UTC hour"""
    for session, rules in SESSION_RULES.items():
        start = rules.get("start_hour", 0)
        end = rules.get("end_hour", 24)
        if start <= hour < end:
            return session
    return "LATE_NY"


def get_session_strategy(session: str) -> str:
    """Get recommended strategy for session"""
    return SESSION_RULES.get(session, {}).get("strategy", "NEUTRAL")


def get_day_bias(day_name: str) -> str:
    """Get trading bias for day of week"""
    return DAY_RULES.get(day_name, {}).get("bias", "NEUTRAL")


def should_trade_day(day_name: str) -> bool:
    """Check if trading is allowed on this day"""
    return DAY_RULES.get(day_name, {}).get("trade", False)


def get_tp_sl(volatility_regime: str = "NORMAL_VOL", day_name: str = None) -> Dict:
    """Get optimal TP/SL based on volatility and day"""
    base = TP_SL_RULES["XAUUSD"].get(volatility_regime, TP_SL_RULES["XAUUSD"]["NORMAL_VOL"])
    tp = base["tp_points"]
    sl = base["sl_points"]
    
    # Day adjustments
    if day_name == "Monday":
        tp += TP_SL_RULES["XAUUSD"]["monday_tp_bonus"]
    elif day_name == "Friday":
        tp -= TP_SL_RULES["XAUUSD"]["friday_tp_reduction"]
    
    return {"tp": tp, "sl": sl, "strategy": base["strategy"]}


def calculate_lot_size(
    confidence: float,
    win_rate: float,
    account_balance: float,
    sl_points: float,
    pip_value: float = 1.0  # Approximate for Gold
) -> float:
    """
    Calculate position size based on pattern confidence and risk rules.
    
    Args:
        confidence: Pattern confidence (0-100)
        win_rate: Historical win rate (0-1)
        account_balance: Current account balance
        sl_points: Stop loss in points
        pip_value: Value per pip/point
    
    Returns:
        Lot size clamped between min and max
    """
    base_risk = account_balance * POSITION_SIZING.risk_per_trade  # 1% risk
    
    # Confidence multiplier (0.5x to 1.5x)
    if confidence >= 80:
        multiplier = 1.5
    elif confidence >= 65:
        multiplier = 1.0
    elif confidence >= 55:
        multiplier = 0.75
    else:
        multiplier = 0.5
    
    # Win rate bonus
    if win_rate >= 0.65:
        multiplier *= 1.2  # Monday's 66% gets a boost
    elif win_rate < 0.55:
        multiplier *= 0.8  # Reduce for lower win rates
    
    # Calculate risk amount
    risk_amount = base_risk * multiplier
    
    # Convert to lot size
    # For Gold: 1 lot = 100 oz, $1 move = $100
    # sl_points in $ terms
    lot = risk_amount / (sl_points * 100 * pip_value) if sl_points > 0 else POSITION_SIZING.base_lot
    
    # Clamp between min and max
    return round(min(max(lot, POSITION_SIZING.base_lot), POSITION_SIZING.max_lot), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT ALL CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DAY_RULES',
    'SESSION_RULES', 
    'POSITION_SIZING',
    'TP_SL_RULES',
    'INDICATOR_RULES',
    'CROWD_PSYCHOLOGY_RULES',
    'BOT_CONFIG',
    'get_current_session',
    'get_session_strategy',
    'get_day_bias',
    'should_trade_day',
    'get_tp_sl',
    'calculate_lot_size'
]

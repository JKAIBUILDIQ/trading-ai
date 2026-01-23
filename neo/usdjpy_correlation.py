#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
USDJPY-GOLD CORRELATION MODULE
═══════════════════════════════════════════════════════════════════════════════

The Insight: USDJPY and XAUUSD have stronger correlation than EURUSD.

Why?
- Both JPY and Gold are SAFE HAVEN assets
- Risk-off: JPY strengthens (USDJPY ↓) AND Gold rises (XAUUSD ↑) → INVERSE
- Risk-on: JPY weakens (USDJPY ↑) AND Gold falls (XAUUSD ↓) → INVERSE
- BOJ policy affects global liquidity (carry trade)
- Japan is a major Gold market participant

KEY LEVELS:
- USDJPY 160: Multi-decade resistance, BOJ intervention zone
- USDJPY 158: BOJ starts verbal intervention
- USDJPY 155: Key support
- USDJPY 150: Major psychological level

TRADING IMPLICATIONS:
- USDJPY at 160 resistance + reversal → BULLISH for Gold
- USDJPY breaks above 160 → USD strength → BEARISH for Gold
- USDJPY breaks below 155 → Risk-off → BULLISH for Gold

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("USDJPY-CORRELATION")

# ═══════════════════════════════════════════════════════════════════════════════
# KEY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

USDJPY_KEY_LEVELS = {
    "major_resistance": 160.00,     # Multi-decade high
    "boj_verbal": 158.00,           # BOJ starts verbal intervention
    "current_range_high": 157.00,   # Recent range high
    "current_range_low": 155.00,    # Recent range low
    "support_1": 152.00,            # Key support
    "support_2": 150.00,            # Major psychological
    "support_3": 145.00,            # Strong historical support
    "support_4": 140.00,            # 2024 low area
}

# Correlation rules for Gold based on USDJPY
CORRELATION_RULES = {
    "usdjpy_above_158": {
        "gold_bias": "BULLISH",
        "reason": "USDJPY in BOJ intervention zone - USD weakness likely",
        "confidence_boost": 10
    },
    "usdjpy_rejects_160": {
        "gold_bias": "STRONG_BULLISH",
        "reason": "USDJPY rejected 160 resistance - major USD reversal signal",
        "confidence_boost": 15
    },
    "usdjpy_breaks_160": {
        "gold_bias": "BEARISH",
        "reason": "USDJPY breakout above 160 - USD strength, Gold correction risk",
        "confidence_boost": -15
    },
    "usdjpy_below_155": {
        "gold_bias": "BULLISH",
        "reason": "USDJPY breakdown - risk-off environment, Gold accelerating",
        "confidence_boost": 10
    },
    "usdjpy_below_150": {
        "gold_bias": "STRONG_BULLISH",
        "reason": "USDJPY below 150 - major USD weakness, Gold could go parabolic",
        "confidence_boost": 20
    },
    "usdjpy_trending_down": {
        "gold_bias": "BULLISH",
        "reason": "USDJPY in downtrend - risk-off correlation supports Gold",
        "confidence_boost": 5
    },
    "usdjpy_trending_up": {
        "gold_bias": "CAUTIOUS",
        "reason": "USDJPY in uptrend - carry trade risk-on, caution on Gold longs",
        "confidence_boost": -5
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class USDJPYContext:
    """Current USDJPY market context"""
    price: float
    trend: str                      # UP, DOWN, SIDEWAYS
    trend_strength: float           # 0-100
    rsi_14: float
    at_resistance: bool
    at_support: bool
    key_level_distance: float       # Distance to nearest key level
    nearest_level: float
    nearest_level_type: str         # RESISTANCE or SUPPORT
    boj_intervention_risk: bool
    weekly_change_pct: float
    monthly_change_pct: float


@dataclass
class CorrelationSignal:
    """Signal generated from USDJPY-Gold correlation"""
    signal_type: str               # BULLISH_GOLD, BEARISH_GOLD, WARNING, NEUTRAL
    direction: str                 # BUY, SELL, HOLD
    confidence_adjustment: int     # Add/subtract from base confidence
    reason: str
    usdjpy_price: float
    usdjpy_condition: str
    urgency: str                   # HIGH, MEDIUM, LOW


# ═══════════════════════════════════════════════════════════════════════════════
# USDJPY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class USDJPYAnalyzer:
    """
    Analyzes USDJPY for Gold correlation signals.
    """
    
    def __init__(self):
        self.key_levels = USDJPY_KEY_LEVELS
        self.rules = CORRELATION_RULES
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI for price series."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def calculate_trend(self, prices: np.ndarray, lookback: int = 20) -> Tuple[str, float]:
        """
        Calculate trend direction and strength.
        
        Returns:
            (direction, strength) where direction is UP/DOWN/SIDEWAYS
            and strength is 0-100
        """
        if len(prices) < lookback:
            return "SIDEWAYS", 0
        
        recent = prices[-lookback:]
        
        # Calculate slope using linear regression
        x = np.arange(lookback)
        slope = np.polyfit(x, recent, 1)[0]
        
        # Normalize slope relative to price
        slope_pct = (slope * lookback / recent[0]) * 100
        
        # Calculate R-squared for trend strength
        y_pred = np.poly1d(np.polyfit(x, recent, 1))(x)
        ss_res = np.sum((recent - y_pred) ** 2)
        ss_tot = np.sum((recent - np.mean(recent)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        strength = min(100, abs(r_squared * 100))
        
        if slope_pct > 0.5:
            return "UP", strength
        elif slope_pct < -0.5:
            return "DOWN", strength
        else:
            return "SIDEWAYS", strength
    
    def check_divergence(self, prices: np.ndarray, rsi_values: np.ndarray) -> Optional[str]:
        """
        Check for RSI divergence.
        
        Returns:
            "BULLISH" if price lower low but RSI higher low
            "BEARISH" if price higher high but RSI lower high
            None if no divergence
        """
        if len(prices) < 20 or len(rsi_values) < 20:
            return None
        
        # Find recent peaks/troughs
        recent_prices = prices[-20:]
        recent_rsi = rsi_values[-20:]
        
        # Simple peak/trough detection
        price_high_1 = max(recent_prices[:10])
        price_high_2 = max(recent_prices[10:])
        rsi_high_1 = max(recent_rsi[:10])
        rsi_high_2 = max(recent_rsi[10:])
        
        price_low_1 = min(recent_prices[:10])
        price_low_2 = min(recent_prices[10:])
        rsi_low_1 = min(recent_rsi[:10])
        rsi_low_2 = min(recent_rsi[10:])
        
        # Bearish divergence: higher price high, lower RSI high
        if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1 - 5:
            return "BEARISH"
        
        # Bullish divergence: lower price low, higher RSI low
        if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1 + 5:
            return "BULLISH"
        
        return None
    
    def get_usdjpy_context(self, df: pd.DataFrame) -> USDJPYContext:
        """
        Get comprehensive USDJPY context.
        
        Args:
            df: OHLCV DataFrame with USDJPY data
        
        Returns:
            USDJPYContext with all relevant data
        """
        if df.empty:
            return USDJPYContext(
                price=155.0, trend="SIDEWAYS", trend_strength=0,
                rsi_14=50, at_resistance=False, at_support=False,
                key_level_distance=0, nearest_level=155,
                nearest_level_type="SUPPORT", boj_intervention_risk=False,
                weekly_change_pct=0, monthly_change_pct=0
            )
        
        current_price = float(df['close'].iloc[-1])
        prices = df['close'].values
        
        # Calculate indicators
        trend, trend_strength = self.calculate_trend(prices)
        rsi_14 = self.calculate_rsi(prices, 14)
        
        # Check proximity to key levels
        at_resistance = current_price > 158
        at_support = current_price < 152
        
        # Find nearest key level
        nearest_level = min(
            self.key_levels.values(),
            key=lambda x: abs(x - current_price)
        )
        key_level_distance = abs(current_price - nearest_level)
        nearest_level_type = "RESISTANCE" if nearest_level > current_price else "SUPPORT"
        
        # BOJ intervention risk (above 158)
        boj_intervention_risk = current_price > 158
        
        # Calculate changes
        weekly_change_pct = 0
        monthly_change_pct = 0
        if len(prices) >= 5:
            weekly_change_pct = ((current_price - prices[-5]) / prices[-5]) * 100
        if len(prices) >= 20:
            monthly_change_pct = ((current_price - prices[-20]) / prices[-20]) * 100
        
        return USDJPYContext(
            price=current_price,
            trend=trend,
            trend_strength=trend_strength,
            rsi_14=rsi_14,
            at_resistance=at_resistance,
            at_support=at_support,
            key_level_distance=key_level_distance,
            nearest_level=nearest_level,
            nearest_level_type=nearest_level_type,
            boj_intervention_risk=boj_intervention_risk,
            weekly_change_pct=weekly_change_pct,
            monthly_change_pct=monthly_change_pct
        )
    
    def generate_gold_signals(self, context: USDJPYContext) -> List[CorrelationSignal]:
        """
        Generate Gold trading signals based on USDJPY correlation.
        
        Args:
            context: USDJPYContext from get_usdjpy_context()
        
        Returns:
            List of CorrelationSignal objects
        """
        signals = []
        price = context.price
        
        # Rule 1: USDJPY above 158 (BOJ intervention zone)
        if price > 158:
            signals.append(CorrelationSignal(
                signal_type="BULLISH_GOLD",
                direction="BUY",
                confidence_adjustment=10,
                reason=f"USDJPY at {price:.2f} in BOJ intervention zone (158-162) - USD weakness likely",
                usdjpy_price=price,
                usdjpy_condition="BOJ_INTERVENTION_ZONE",
                urgency="HIGH"
            ))
            
            # Check for rejection at 160
            if price > 159.5 and context.rsi_14 > 70:
                signals.append(CorrelationSignal(
                    signal_type="STRONG_BULLISH_GOLD",
                    direction="BUY",
                    confidence_adjustment=15,
                    reason=f"USDJPY overbought (RSI {context.rsi_14:.0f}) near 160 - reversal signal for Gold",
                    usdjpy_price=price,
                    usdjpy_condition="OVERBOUGHT_AT_RESISTANCE",
                    urgency="HIGH"
                ))
        
        # Rule 2: USDJPY breaks above 160 (USD strength)
        if price > 160 and context.trend == "UP":
            signals.append(CorrelationSignal(
                signal_type="BEARISH_GOLD",
                direction="SELL",
                confidence_adjustment=-15,
                reason=f"USDJPY broke above 160 - USD strength, Gold correction risk",
                usdjpy_price=price,
                usdjpy_condition="BREAKOUT_ABOVE_160",
                urgency="HIGH"
            ))
        
        # Rule 3: USDJPY below 155 (risk-off)
        if price < 155:
            signals.append(CorrelationSignal(
                signal_type="BULLISH_GOLD",
                direction="BUY",
                confidence_adjustment=10,
                reason=f"USDJPY below 155 - risk-off environment, Gold accelerating",
                usdjpy_price=price,
                usdjpy_condition="BELOW_KEY_SUPPORT",
                urgency="MEDIUM"
            ))
        
        # Rule 4: USDJPY below 150 (major USD weakness)
        if price < 150:
            signals.append(CorrelationSignal(
                signal_type="STRONG_BULLISH_GOLD",
                direction="BUY",
                confidence_adjustment=20,
                reason=f"USDJPY below 150 - major USD weakness, Gold could go parabolic",
                usdjpy_price=price,
                usdjpy_condition="MAJOR_USD_WEAKNESS",
                urgency="HIGH"
            ))
        
        # Rule 5: Trend-based signals
        if context.trend == "DOWN" and context.trend_strength > 50:
            signals.append(CorrelationSignal(
                signal_type="BULLISH_GOLD",
                direction="BUY",
                confidence_adjustment=5,
                reason=f"USDJPY in strong downtrend ({context.trend_strength:.0f}%) - risk-off supports Gold",
                usdjpy_price=price,
                usdjpy_condition="DOWNTREND",
                urgency="LOW"
            ))
        elif context.trend == "UP" and context.trend_strength > 50:
            signals.append(CorrelationSignal(
                signal_type="CAUTIOUS_GOLD",
                direction="HOLD",
                confidence_adjustment=-5,
                reason=f"USDJPY in uptrend ({context.trend_strength:.0f}%) - carry trade risk-on, caution on Gold",
                usdjpy_price=price,
                usdjpy_condition="UPTREND",
                urgency="LOW"
            ))
        
        # Rule 6: Oversold USDJPY (potential bounce = Gold correction)
        if context.rsi_14 < 30:
            signals.append(CorrelationSignal(
                signal_type="WARNING",
                direction="HOLD",
                confidence_adjustment=-5,
                reason=f"USDJPY oversold (RSI {context.rsi_14:.0f}) - potential bounce could pressure Gold",
                usdjpy_price=price,
                usdjpy_condition="OVERSOLD",
                urgency="MEDIUM"
            ))
        
        return signals
    
    def format_correlation_summary(
        self, 
        context: USDJPYContext, 
        signals: List[CorrelationSignal]
    ) -> str:
        """Format correlation analysis as human-readable string."""
        lines = [
            "",
            "🔗 USDJPY-GOLD CORRELATION:",
            f"   USDJPY: {context.price:.2f}",
            f"   Trend: {context.trend} ({context.trend_strength:.0f}%)",
            f"   RSI(14): {context.rsi_14:.1f}",
        ]
        
        if context.boj_intervention_risk:
            lines.append("   ⚠️ In BOJ intervention zone (>158)")
        
        if context.at_resistance:
            lines.append("   📍 Near major resistance (160)")
        elif context.at_support:
            lines.append("   📍 Near key support")
        
        lines.append(f"   Nearest Level: {context.nearest_level} ({context.nearest_level_type})")
        
        if signals:
            lines.append("")
            lines.append("   Correlation Signals:")
            for sig in signals:
                emoji = "🟢" if "BULLISH" in sig.signal_type else "🔴" if "BEARISH" in sig.signal_type else "⚠️"
                lines.append(f"   {emoji} {sig.reason}")
        
        # Net confidence adjustment
        total_adjustment = sum(s.confidence_adjustment for s in signals)
        if total_adjustment != 0:
            adj_str = f"+{total_adjustment}" if total_adjustment > 0 else str(total_adjustment)
            lines.append(f"   📊 Net Gold Confidence: {adj_str}%")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer = None

def get_analyzer() -> USDJPYAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = USDJPYAnalyzer()
    return _analyzer


def get_usdjpy_gold_correlation(usdjpy_df: pd.DataFrame) -> Dict:
    """
    Main function to get USDJPY-Gold correlation analysis.
    
    Usage:
        from usdjpy_correlation import get_usdjpy_gold_correlation
        
        # Fetch USDJPY data (H4 or D1)
        usdjpy_df = yf.Ticker("USDJPY=X").history(period="3mo", interval="1d")
        
        correlation = get_usdjpy_gold_correlation(usdjpy_df)
    
    Returns:
        Dict with:
        - context: USDJPYContext
        - signals: List[CorrelationSignal]
        - summary: Human-readable string
        - net_confidence_adjustment: Total confidence boost/reduction for Gold
        - gold_bias: BULLISH, BEARISH, or NEUTRAL
    """
    analyzer = get_analyzer()
    
    # Normalize columns
    if 'Close' in usdjpy_df.columns:
        usdjpy_df.columns = [c.lower() for c in usdjpy_df.columns]
    
    context = analyzer.get_usdjpy_context(usdjpy_df)
    signals = analyzer.generate_gold_signals(context)
    summary = analyzer.format_correlation_summary(context, signals)
    
    # Calculate net effect
    net_adjustment = sum(s.confidence_adjustment for s in signals)
    
    # Determine overall bias
    bullish_signals = sum(1 for s in signals if "BULLISH" in s.signal_type)
    bearish_signals = sum(1 for s in signals if "BEARISH" in s.signal_type)
    
    if bullish_signals > bearish_signals:
        gold_bias = "BULLISH"
    elif bearish_signals > bullish_signals:
        gold_bias = "BEARISH"
    else:
        gold_bias = "NEUTRAL"
    
    return {
        "available": True,
        "context": {
            "price": context.price,
            "trend": context.trend,
            "trend_strength": context.trend_strength,
            "rsi_14": context.rsi_14,
            "at_resistance": context.at_resistance,
            "at_support": context.at_support,
            "boj_intervention_risk": context.boj_intervention_risk,
            "weekly_change_pct": context.weekly_change_pct,
            "monthly_change_pct": context.monthly_change_pct
        },
        "signals": [
            {
                "type": s.signal_type,
                "direction": s.direction,
                "adjustment": s.confidence_adjustment,
                "reason": s.reason,
                "urgency": s.urgency
            }
            for s in signals
        ],
        "summary": summary,
        "net_confidence_adjustment": net_adjustment,
        "gold_bias": gold_bias
    }


def fetch_usdjpy_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch USDJPY data from Yahoo Finance.
    
    Args:
        period: Data period (1mo, 3mo, 6mo, 1y)
        interval: Candle interval (1h, 4h, 1d)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("USDJPY=X")
        df = ticker.history(period=period, interval=interval)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Failed to fetch USDJPY data: {e}")
        return pd.DataFrame()


def get_quick_usdjpy_signal() -> Dict:
    """
    Quick function to get current USDJPY-Gold correlation signal.
    
    Usage:
        signal = get_quick_usdjpy_signal()
        print(f"Gold bias from USDJPY: {signal['gold_bias']}")
    """
    df = fetch_usdjpy_data(period="1mo", interval="1h")
    if df.empty:
        return {"available": False, "error": "Could not fetch data"}
    
    return get_usdjpy_gold_correlation(df)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🔗 USDJPY-GOLD CORRELATION ANALYZER")
    print("=" * 60)
    
    print("\n📊 Fetching USDJPY data...")
    df = fetch_usdjpy_data(period="3mo", interval="1d")
    
    if not df.empty:
        print(f"Data points: {len(df)}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Run analysis
        result = get_usdjpy_gold_correlation(df)
        
        print(result["summary"])
        print()
        print("=" * 60)
        print(f"📈 GOLD BIAS: {result['gold_bias']}")
        print(f"📊 Net Confidence Adjustment: {result['net_confidence_adjustment']:+d}%")
        print("=" * 60)
        
        # Show key levels
        print("\n📍 KEY USDJPY LEVELS:")
        for name, level in USDJPY_KEY_LEVELS.items():
            marker = "◀️ CURRENT" if abs(level - result['context']['price']) < 2 else ""
            print(f"   {level:.2f} - {name.replace('_', ' ').title()} {marker}")
    else:
        print("❌ Could not fetch USDJPY data")

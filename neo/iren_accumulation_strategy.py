"""
IREN Accumulation Strategy
==========================
User's Core Belief: IREN to $150

Strategy:
- SHARES: Buy dips, accumulate, HOLD to $150
- OPTIONS: Scalp bounces, sell covered calls for income
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IrenAccumulation")


class IrenAccumulationStrategy:
    """
    Strategy aligned with long-term $150 target
    """
    
    TARGET_PRICE = 150.00  # Long-term target
    
    def __init__(self):
        self.ticker = yf.Ticker("IREN")
        self._cache = {}
        self._cache_time = None
        
    def get_current_analysis(self) -> Dict[str, Any]:
        """Get current IREN analysis with buy zones"""
        
        # Cache for 1 minute
        if self._cache_time and (datetime.now() - self._cache_time).seconds < 60:
            return self._cache
            
        hist = self.ticker.history(period="3mo")
        current = float(hist['Close'].iloc[-1])
        high_3m = float(hist['High'].max())
        low_3m = float(hist['Low'].min())
        
        # Calculate key levels
        analysis = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "current_price": current,
            "target_price": self.TARGET_PRICE,
            "upside_pct": round((self.TARGET_PRICE - current) / current * 100, 1),
            
            # Price context
            "high_3m": high_3m,
            "low_3m": low_3m,
            "from_high_pct": round((current - high_3m) / high_3m * 100, 1),
            "from_low_pct": round((current - low_3m) / low_3m * 100, 1),
            
            # Buy zones for SHARES
            "share_zones": {
                "strong_buy": round(low_3m * 1.05, 2),  # Near 3mo low
                "good_buy": round(current * 0.95, 2),   # 5% below current
                "hold": current,
                "wait": round(current * 1.10, 2),       # 10% above = wait
            },
            
            # Scalp zones for OPTIONS
            "option_zones": {
                "buy_calls_at": round(current * 0.93, 2),   # 7% dip
                "sell_calls_at": round(current * 1.08, 2),  # 8% rally
            },
            
            # Current signal
            "share_signal": self._get_share_signal(current, low_3m),
            "option_signal": self._get_option_signal(current, low_3m, high_3m),
            
            # Recommended actions
            "actions": self._get_recommended_actions(current, low_3m, high_3m)
        }
        
        self._cache = analysis
        self._cache_time = datetime.now()
        
        return analysis
    
    def _get_share_signal(self, current: float, low_3m: float) -> Dict:
        """Determine share accumulation signal"""
        
        # Calculate how far from 3mo low
        from_low_pct = (current - low_3m) / low_3m * 100
        
        if from_low_pct < 10:
            return {
                "signal": "STRONG_BUY",
                "reason": f"Near 3-month low (${low_3m:.2f})",
                "action": "Accumulate heavily - rare opportunity",
                "confidence": 90
            }
        elif from_low_pct < 30:
            return {
                "signal": "BUY",
                "reason": f"Good value zone ({from_low_pct:.0f}% above 3mo low)",
                "action": "Add to position on any dip",
                "confidence": 75
            }
        elif from_low_pct < 60:
            return {
                "signal": "HOLD",
                "reason": f"Mid-range ({from_low_pct:.0f}% above 3mo low)",
                "action": "Hold existing, wait for better entry",
                "confidence": 60
            }
        else:
            return {
                "signal": "WAIT",
                "reason": f"Extended ({from_low_pct:.0f}% above 3mo low)",
                "action": "Don't chase - wait for pullback",
                "confidence": 50
            }
    
    def _get_option_signal(self, current: float, low_3m: float, high_3m: float) -> Dict:
        """Determine options scalping signal"""
        
        from_low_pct = (current - low_3m) / low_3m * 100
        from_high_pct = (high_3m - current) / high_3m * 100
        
        if from_low_pct < 15:
            return {
                "signal": "BUY_CALLS",
                "reason": "Near support - bounce expected",
                "strike": round(current * 1.10, 0),  # 10% OTM
                "expiry": "2-3 weeks",
                "action": "Buy calls for bounce scalp",
                "confidence": 75
            }
        elif from_high_pct < 15:
            return {
                "signal": "SELL_COVERED_CALLS",
                "reason": "Near resistance - premium opportunity",
                "strike": round(current * 1.15, 0),  # 15% OTM
                "expiry": "3-4 weeks",
                "action": "Sell covered calls for income",
                "confidence": 70
            }
        else:
            return {
                "signal": "WAIT",
                "reason": "Mid-range - no clear scalp setup",
                "action": "No options trade recommended now",
                "confidence": 50
            }
    
    def _get_recommended_actions(self, current: float, low_3m: float, high_3m: float) -> list:
        """Get prioritized action list"""
        
        actions = []
        from_low_pct = (current - low_3m) / low_3m * 100
        
        # Always show target reminder
        actions.append({
            "priority": 1,
            "type": "REMINDER",
            "message": f"🎯 Target: $150 ({((150 - current) / current * 100):.0f}% upside)",
            "action": "HOLD all core shares until target"
        })
        
        # Share accumulation
        if from_low_pct < 30:
            actions.append({
                "priority": 2,
                "type": "BUY_SHARES",
                "message": f"📈 Good accumulation zone (${current:.2f})",
                "action": f"Consider adding 50-100 shares"
            })
        
        # Dip alert levels
        actions.append({
            "priority": 3,
            "type": "ALERT_LEVELS",
            "message": "🔔 Set alerts for these levels:",
            "levels": {
                f"${current * 0.95:.2f}": "Add 100 shares",
                f"${current * 0.90:.2f}": "Add 100 shares + buy calls",
                f"${low_3m * 1.05:.2f}": "LOAD UP - near 3mo low"
            }
        })
        
        # Covered call opportunity
        if from_low_pct > 50:  # Stock has run up
            actions.append({
                "priority": 4,
                "type": "COVERED_CALL",
                "message": "💰 Consider selling covered call",
                "action": f"Sell ${round(current * 1.15, 0):.0f} call, 3-4 weeks out"
            })
        
        return actions
    
    def get_dip_buy_signal(self, threshold_pct: float = 5.0) -> Optional[Dict]:
        """Check if there's a dip worth buying"""
        
        analysis = self.get_current_analysis()
        current = analysis['current_price']
        
        # Get recent high (5 days)
        hist = self.ticker.history(period="5d")
        recent_high = float(hist['High'].max())
        
        dip_pct = (recent_high - current) / recent_high * 100
        
        if dip_pct >= threshold_pct:
            return {
                "signal": "DIP_BUY",
                "current_price": current,
                "recent_high": recent_high,
                "dip_pct": round(dip_pct, 2),
                "recommendation": f"Buy shares - {dip_pct:.1f}% dip from recent high",
                "suggested_size": 100 if dip_pct < 10 else 200
            }
        
        return None


# FastAPI endpoints for this strategy
def get_strategy_routes():
    """Return FastAPI router for this strategy"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/api/iren/accumulation", tags=["IREN Accumulation"])
    strategy = IrenAccumulationStrategy()
    
    @router.get("/analysis")
    def get_analysis():
        """Get current IREN analysis with buy zones"""
        return strategy.get_current_analysis()
    
    @router.get("/dip-alert")
    def check_dip(threshold: float = 5.0):
        """Check if there's a dip worth buying"""
        signal = strategy.get_dip_buy_signal(threshold)
        if signal:
            return signal
        return {"signal": "NO_DIP", "message": f"No dip > {threshold}% detected"}
    
    @router.get("/summary")
    def get_summary():
        """Get quick summary for dashboard"""
        analysis = strategy.get_current_analysis()
        return {
            "price": analysis["current_price"],
            "target": analysis["target_price"],
            "upside": f"{analysis['upside_pct']}%",
            "share_signal": analysis["share_signal"]["signal"],
            "option_signal": analysis["option_signal"]["signal"],
            "next_buy_at": analysis["share_zones"]["good_buy"]
        }
    
    return router


# Test
if __name__ == "__main__":
    strategy = IrenAccumulationStrategy()
    analysis = strategy.get_current_analysis()
    
    print("=" * 60)
    print("IREN ACCUMULATION STRATEGY")
    print("=" * 60)
    print(f"\nCurrent: ${analysis['current_price']:.2f}")
    print(f"Target:  ${analysis['target_price']:.2f} ({analysis['upside_pct']}% upside)")
    print(f"\nShare Signal: {analysis['share_signal']['signal']}")
    print(f"  → {analysis['share_signal']['action']}")
    print(f"\nOption Signal: {analysis['option_signal']['signal']}")
    print(f"  → {analysis['option_signal']['action']}")
    print("\nRecommended Actions:")
    for action in analysis['actions']:
        print(f"  {action['priority']}. {action['message']}")

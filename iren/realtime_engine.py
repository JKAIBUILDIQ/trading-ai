"""
IREN Real-Time Trading Engine
Complete metrics, paper trading, and earnings strategy
"""

import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import time

# Constants
IREN_TICKER = "IREN"
BTC_TICKER = "BTC-USD"
EARNINGS_DATE = datetime(2026, 2, 5)  # Q4 2025 earnings
PAUL_STRIKES = [60, 70, 80]
PAUL_PREFERRED_DTE = (21, 35)  # Sweet spot

# File paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.json"
METRICS_CACHE_FILE = DATA_DIR / "metrics_cache.json"


class IRENRealTimeEngine:
    """Real-time IREN trading metrics and paper trading engine"""
    
    def __init__(self):
        self.iren = yf.Ticker(IREN_TICKER)
        self.btc = yf.Ticker(BTC_TICKER)
        self._load_paper_trades()
        
    def _load_paper_trades(self):
        """Load existing paper trades"""
        if PAPER_TRADES_FILE.exists():
            with open(PAPER_TRADES_FILE, 'r') as f:
                self.paper_trades = json.load(f)
        else:
            self.paper_trades = {
                "balance": 100000,  # $100K starting balance
                "positions": [],
                "history": [],
                "created": datetime.now().isoformat()
            }
            self._save_paper_trades()
    
    def _save_paper_trades(self):
        """Save paper trades to file"""
        with open(PAPER_TRADES_FILE, 'w') as f:
            json.dump(self.paper_trades, f, indent=2, default=str)
    
    def get_realtime_metrics(self) -> Dict:
        """Get comprehensive real-time IREN metrics"""
        try:
            # Get current data
            iren_info = self.iren.info
            iren_hist = self.iren.history(period="1mo")
            btc_hist = self.btc.history(period="1mo")
            
            # Current price
            current_price = iren_info.get('currentPrice') or iren_info.get('regularMarketPrice', 0)
            prev_close = iren_info.get('previousClose', current_price)
            
            # Calculate change
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close else 0
            
            # Volume analysis
            current_volume = iren_info.get('volume', 0)
            avg_volume = iren_info.get('averageVolume', 1)
            volume_ratio = current_volume / avg_volume if avg_volume else 1
            
            # Technical indicators
            if len(iren_hist) >= 14:
                closes = iren_hist['Close'].values
                
                # RSI
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                
                # Moving averages
                ma_5 = np.mean(closes[-5:])
                ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
                
                # Momentum
                momentum = ((closes[-1] / closes[-5]) - 1) * 100 if len(closes) >= 5 else 0
            else:
                rsi = 50
                ma_5 = current_price
                ma_20 = current_price
                momentum = 0
            
            # BTC correlation (30-day)
            if len(iren_hist) >= 20 and len(btc_hist) >= 20:
                iren_returns = iren_hist['Close'].pct_change().dropna()[-20:]
                btc_returns = btc_hist['Close'].pct_change().dropna()[-20:]
                
                if len(iren_returns) > 5 and len(btc_returns) > 5:
                    min_len = min(len(iren_returns), len(btc_returns))
                    correlation = np.corrcoef(
                        iren_returns.values[-min_len:],
                        btc_returns.values[-min_len:]
                    )[0, 1]
                else:
                    correlation = 0
            else:
                correlation = 0
            
            # BTC current
            btc_info = self.btc.info
            btc_price = btc_info.get('currentPrice') or btc_info.get('regularMarketPrice', 0)
            
            # Earnings countdown
            days_to_earnings = (EARNINGS_DATE - datetime.now()).days
            
            # Trend determination
            if current_price > ma_5 > ma_20:
                trend = "BULLISH"
                trend_strength = min(100, (current_price / ma_20 - 1) * 500)
            elif current_price < ma_5 < ma_20:
                trend = "BEARISH"
                trend_strength = min(100, (ma_20 / current_price - 1) * 500)
            else:
                trend = "NEUTRAL"
                trend_strength = 50
            
            # Signal determination (fundamentals-based)
            signal_score = 0
            signal_reasons = []
            
            # RSI signal
            if rsi < 40:
                signal_score += 30
                signal_reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < 50:
                signal_score += 15
                signal_reasons.append(f"RSI neutral-low ({rsi:.1f})")
            elif rsi > 70:
                signal_score -= 20
                signal_reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # Trend signal
            if trend == "BULLISH":
                signal_score += 25
                signal_reasons.append("Bullish trend")
            elif trend == "BEARISH":
                signal_score -= 15
                signal_reasons.append("Bearish trend - wait")
            
            # Volume signal
            if volume_ratio > 1.5:
                signal_score += 15
                signal_reasons.append(f"High volume ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.2:
                signal_score += 10
                signal_reasons.append(f"Above-avg volume ({volume_ratio:.1f}x)")
            
            # Momentum signal
            if momentum > 5:
                signal_score += 15
                signal_reasons.append(f"Strong momentum (+{momentum:.1f}%)")
            elif momentum > 0:
                signal_score += 10
                signal_reasons.append(f"Positive momentum (+{momentum:.1f}%)")
            elif momentum < -5:
                signal_score -= 10
                signal_reasons.append(f"Negative momentum ({momentum:.1f}%)")
            
            # Earnings proximity (reduce aggression near earnings)
            if days_to_earnings <= 7:
                signal_score -= 20
                signal_reasons.append("⚠️ Earnings within 7 days - reduce size")
            elif days_to_earnings <= 14:
                signal_score -= 10
                signal_reasons.append("⚠️ Earnings approaching - be cautious")
            
            # Determine action
            if signal_score >= 50:
                action = "BUY"
                confidence = min(95, signal_score + 30)
            elif signal_score >= 25:
                action = "ACCUMULATE"
                confidence = min(80, signal_score + 40)
            elif signal_score >= 0:
                action = "HOLD"
                confidence = 50
            else:
                action = "WAIT"
                confidence = max(20, 50 + signal_score)
            
            # Build metrics object
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "symbol": IREN_TICKER,
                
                # Price data
                "price": {
                    "current": round(current_price, 2),
                    "previous_close": round(prev_close, 2),
                    "change": round(price_change, 2),
                    "change_pct": round(price_change_pct, 2),
                    "day_high": iren_info.get('dayHigh', current_price),
                    "day_low": iren_info.get('dayLow', current_price),
                    "52w_high": iren_info.get('fiftyTwoWeekHigh', 0),
                    "52w_low": iren_info.get('fiftyTwoWeekLow', 0)
                },
                
                # Volume
                "volume": {
                    "current": current_volume,
                    "average": avg_volume,
                    "ratio": round(volume_ratio, 2)
                },
                
                # Technicals
                "technicals": {
                    "rsi": round(rsi, 1),
                    "ma_5": round(ma_5, 2),
                    "ma_20": round(ma_20, 2),
                    "momentum_5d": round(momentum, 2),
                    "trend": trend,
                    "trend_strength": round(trend_strength, 1)
                },
                
                # BTC correlation
                "btc_correlation": {
                    "btc_price": round(btc_price, 2),
                    "correlation_30d": round(correlation, 3),
                    "coupling_status": "COUPLED" if abs(correlation) > 0.6 else "DECOUPLING"
                },
                
                # Earnings
                "earnings": {
                    "date": EARNINGS_DATE.strftime("%Y-%m-%d"),
                    "days_away": days_to_earnings,
                    "phase": self._get_earnings_phase(days_to_earnings)
                },
                
                # Signal
                "signal": {
                    "action": action,
                    "confidence": confidence,
                    "score": signal_score,
                    "reasons": signal_reasons
                },
                
                # Paul's strikes
                "options_focus": {
                    "primary_strike": 60,
                    "strikes": PAUL_STRIKES,
                    "preferred_dte": PAUL_PREFERRED_DTE
                }
            }
            
            # Cache metrics
            with open(METRICS_CACHE_FILE, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            # Return cached if available
            if METRICS_CACHE_FILE.exists():
                with open(METRICS_CACHE_FILE, 'r') as f:
                    return json.load(f)
            return {"error": str(e)}
    
    def _get_earnings_phase(self, days_away: int) -> str:
        """Determine current earnings phase"""
        if days_away <= 0:
            return "POST_EARNINGS"
        elif days_away <= 3:
            return "EARNINGS_WEEK"
        elif days_away <= 7:
            return "PRE_EARNINGS_FINAL"
        elif days_away <= 14:
            return "PRE_EARNINGS_SETUP"
        else:
            return "ACCUMULATION"
    
    def get_options_chain(self) -> Dict:
        """Get IREN options chain for Paul's strikes"""
        try:
            # Get available expiration dates
            expirations = self.iren.options
            
            if not expirations:
                return {"error": "No options data available"}
            
            # Filter to next 6 expiries (current week + 3 weeks + 2 months)
            today = datetime.now().date()
            valid_expiries = []
            
            for exp in expirations[:10]:  # Check first 10
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days_to_exp = (exp_date - today).days
                
                # Skip if too close to earnings (within 3 days after)
                days_from_earnings = (exp_date - EARNINGS_DATE.date()).days
                if -3 <= days_from_earnings <= 0:
                    continue
                
                if days_to_exp >= 3:  # At least 3 days out
                    valid_expiries.append({
                        "date": exp,
                        "dte": days_to_exp,
                        "is_paul_pick": PAUL_PREFERRED_DTE[0] <= days_to_exp <= PAUL_PREFERRED_DTE[1],
                        "earnings_warning": days_to_exp <= (EARNINGS_DATE.date() - today).days + 3
                    })
                
                if len(valid_expiries) >= 6:
                    break
            
            # Get options data for each expiry and strike
            options_data = []
            
            for exp_info in valid_expiries:
                try:
                    chain = self.iren.option_chain(exp_info["date"])
                    calls = chain.calls
                    
                    for strike in PAUL_STRIKES:
                        strike_data = calls[calls['strike'] == strike]
                        
                        if not strike_data.empty:
                            row = strike_data.iloc[0]
                            
                            # Determine strike action
                            if strike == 60:
                                strike_action = "PRIMARY"
                                strike_note = "Core accumulation strike"
                            elif strike == 70:
                                strike_action = "MOMENTUM"
                                strike_note = "Add on breakout above $65"
                            else:
                                strike_action = "EARNINGS_LOTTERY"
                                strike_note = "Small position for earnings pop"
                            
                            options_data.append({
                                "expiry": exp_info["date"],
                                "dte": exp_info["dte"],
                                "is_paul_pick": exp_info["is_paul_pick"],
                                "strike": strike,
                                "type": "CALL",
                                "bid": row.get('bid', 0),
                                "ask": row.get('ask', 0),
                                "last": row.get('lastPrice', 0),
                                "volume": int(row.get('volume', 0) or 0),
                                "open_interest": int(row.get('openInterest', 0) or 0),
                                "iv": round(row.get('impliedVolatility', 0) * 100, 1),
                                "delta": round(row.get('delta', 0.5), 3) if 'delta' in row else None,
                                "strike_action": strike_action,
                                "strike_note": strike_note
                            })
                except Exception as e:
                    print(f"Error getting chain for {exp_info['date']}: {e}")
                    continue
            
            return {
                "timestamp": datetime.now().isoformat(),
                "expirations": valid_expiries,
                "options": options_data,
                "earnings_date": EARNINGS_DATE.strftime("%Y-%m-%d"),
                "paul_preferred_dte": PAUL_PREFERRED_DTE
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_paper_trade(self, action: str, strike: float, expiry: str, 
                           contracts: int, price: float) -> Dict:
        """Execute a paper trade"""
        trade_value = contracts * price * 100  # Options = 100 shares per contract
        
        if action.upper() == "BUY":
            if trade_value > self.paper_trades["balance"]:
                return {"error": "Insufficient balance", "required": trade_value}
            
            self.paper_trades["balance"] -= trade_value
            
            position = {
                "id": len(self.paper_trades["positions"]) + 1,
                "type": "CALL",
                "strike": strike,
                "expiry": expiry,
                "contracts": contracts,
                "entry_price": price,
                "entry_date": datetime.now().isoformat(),
                "cost_basis": trade_value,
                "status": "OPEN"
            }
            self.paper_trades["positions"].append(position)
            
            trade_record = {
                **position,
                "action": "BUY",
                "executed_at": datetime.now().isoformat()
            }
            self.paper_trades["history"].append(trade_record)
            
            self._save_paper_trades()
            
            return {
                "success": True,
                "trade": position,
                "new_balance": self.paper_trades["balance"]
            }
        
        elif action.upper() == "SELL":
            # Find matching position
            for pos in self.paper_trades["positions"]:
                if (pos["status"] == "OPEN" and 
                    pos["strike"] == strike and 
                    pos["expiry"] == expiry):
                    
                    sell_value = contracts * price * 100
                    profit = sell_value - pos["cost_basis"]
                    
                    pos["status"] = "CLOSED"
                    pos["exit_price"] = price
                    pos["exit_date"] = datetime.now().isoformat()
                    pos["profit"] = profit
                    
                    self.paper_trades["balance"] += sell_value
                    
                    trade_record = {
                        **pos,
                        "action": "SELL",
                        "executed_at": datetime.now().isoformat()
                    }
                    self.paper_trades["history"].append(trade_record)
                    
                    self._save_paper_trades()
                    
                    return {
                        "success": True,
                        "trade": pos,
                        "profit": profit,
                        "new_balance": self.paper_trades["balance"]
                    }
            
            return {"error": "No matching position found"}
        
        return {"error": "Invalid action"}
    
    def get_portfolio_status(self) -> Dict:
        """Get current paper trading portfolio status"""
        open_positions = [p for p in self.paper_trades["positions"] if p["status"] == "OPEN"]
        closed_positions = [p for p in self.paper_trades["positions"] if p["status"] == "CLOSED"]
        
        total_invested = sum(p["cost_basis"] for p in open_positions)
        realized_pnl = sum(p.get("profit", 0) for p in closed_positions)
        
        return {
            "balance": self.paper_trades["balance"],
            "total_invested": total_invested,
            "available": self.paper_trades["balance"],
            "open_positions": open_positions,
            "position_count": len(open_positions),
            "realized_pnl": realized_pnl,
            "total_trades": len(self.paper_trades["history"]),
            "win_rate": self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions"""
        closed = [p for p in self.paper_trades["positions"] if p["status"] == "CLOSED"]
        if not closed:
            return 0
        wins = sum(1 for p in closed if p.get("profit", 0) > 0)
        return round(wins / len(closed) * 100, 1)
    
    def get_earnings_strategy(self) -> Dict:
        """Get complete earnings play strategy"""
        days_to_earnings = (EARNINGS_DATE - datetime.now()).days
        
        # Phase-specific strategy
        if days_to_earnings > 14:
            phase = "ACCUMULATION"
            strategy = {
                "primary_action": "ACCUMULATE $60 CALLS",
                "size": "25% of planned position",
                "target_dte": "Feb 20 or Feb 27",
                "notes": [
                    "Build core position in $60 calls",
                    "Add on any dips below $55",
                    "Keep 50% cash for earnings week"
                ]
            }
        elif days_to_earnings > 7:
            phase = "PRE_EARNINGS_SETUP"
            strategy = {
                "primary_action": "COMPLETE POSITION",
                "size": "50% of planned position",
                "target_dte": "Feb 20 (15 DTE post-earnings)",
                "notes": [
                    "Complete $60 call position",
                    "Consider small $70 call lottery",
                    "Set stop-loss at entry -30%",
                    "Watch IV expansion"
                ]
            }
        elif days_to_earnings > 3:
            phase = "EARNINGS_WEEK"
            strategy = {
                "primary_action": "HOLD / TRIM",
                "size": "No new positions",
                "target_dte": "N/A",
                "notes": [
                    "Hold existing positions",
                    "Consider trimming 25-50% before earnings",
                    "IV will be elevated - premium sellers active",
                    "Do NOT add new positions"
                ]
            }
        elif days_to_earnings > 0:
            phase = "EARNINGS_EVE"
            strategy = {
                "primary_action": "DECISION TIME",
                "size": "Trim to comfort",
                "target_dte": "N/A",
                "notes": [
                    "DECISION: Hold through or sell before?",
                    "If holding: Keep only what you can afford to lose",
                    "If selling: Exit 50-75% of position",
                    "Either way: Have a plan for both outcomes"
                ]
            }
        else:
            phase = "POST_EARNINGS"
            strategy = {
                "primary_action": "REACT TO RESULTS",
                "size": "Based on outcome",
                "target_dte": "March expiry",
                "notes": [
                    "BEAT: Add on any pullback, target $80+",
                    "MISS: Cut losses quickly, reassess thesis",
                    "IN-LINE: Hold, watch guidance for direction"
                ]
            }
        
        return {
            "earnings_date": EARNINGS_DATE.strftime("%Y-%m-%d"),
            "days_away": days_to_earnings,
            "phase": phase,
            "strategy": strategy,
            "key_levels": {
                "support": [52, 55, 58],
                "resistance": [62, 65, 70],
                "breakout_target": 80,
                "stop_loss": 50
            },
            "scenarios": {
                "bull_case": {
                    "trigger": "Revenue beat + AI datacenter growth > 50%",
                    "target": "$75-80",
                    "probability": "40%"
                },
                "base_case": {
                    "trigger": "In-line results, positive guidance",
                    "target": "$62-68",
                    "probability": "45%"
                },
                "bear_case": {
                    "trigger": "Miss or weak guidance",
                    "target": "$48-52",
                    "probability": "15%"
                }
            }
        }


def get_full_dashboard() -> Dict:
    """Get complete trading dashboard data"""
    engine = IRENRealTimeEngine()
    
    return {
        "metrics": engine.get_realtime_metrics(),
        "options": engine.get_options_chain(),
        "portfolio": engine.get_portfolio_status(),
        "earnings_strategy": engine.get_earnings_strategy(),
        "generated_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test the engine
    engine = IRENRealTimeEngine()
    
    print("=" * 70)
    print("IREN REAL-TIME TRADING ENGINE")
    print("=" * 70)
    
    metrics = engine.get_realtime_metrics()
    
    print(f"""
📊 IREN REAL-TIME METRICS
{'=' * 50}

PRICE:
  Current: ${metrics['price']['current']}
  Change:  {metrics['price']['change']:+.2f} ({metrics['price']['change_pct']:+.2f}%)
  
TECHNICALS:
  RSI:      {metrics['technicals']['rsi']}
  Trend:    {metrics['technicals']['trend']} ({metrics['technicals']['trend_strength']:.0f}%)
  Momentum: {metrics['technicals']['momentum_5d']:+.2f}%
  
BTC CORRELATION:
  BTC Price:   ${metrics['btc_correlation']['btc_price']:,.0f}
  Correlation: {metrics['btc_correlation']['correlation_30d']:.3f}
  Status:      {metrics['btc_correlation']['coupling_status']}
  
EARNINGS:
  Date:       {metrics['earnings']['date']}
  Days Away:  {metrics['earnings']['days_away']}
  Phase:      {metrics['earnings']['phase']}
  
SIGNAL:
  Action:     {metrics['signal']['action']}
  Confidence: {metrics['signal']['confidence']}%
  
  Reasons:
""")
    for reason in metrics['signal']['reasons']:
        print(f"    • {reason}")
    
    print(f"""
{'=' * 50}
PORTFOLIO STATUS
{'=' * 50}
""")
    portfolio = engine.get_portfolio_status()
    print(f"  Balance: ${portfolio['balance']:,.2f}")
    print(f"  Open Positions: {portfolio['position_count']}")
    print(f"  Realized P&L: ${portfolio['realized_pnl']:,.2f}")
    
    print(f"""
{'=' * 50}
EARNINGS STRATEGY
{'=' * 50}
""")
    strategy = engine.get_earnings_strategy()
    print(f"  Phase: {strategy['phase']}")
    print(f"  Action: {strategy['strategy']['primary_action']}")
    print(f"  Size: {strategy['strategy']['size']}")

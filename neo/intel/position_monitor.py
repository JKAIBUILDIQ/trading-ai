#!/usr/bin/env python3
"""
POSITION MONITOR
Fetches live positions from MT5 API and explains WHY they exist.
Part of NEO's War Room.

NO RANDOM DATA - All from MT5 API
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MT5_API_URL


class PositionMonitor:
    """
    Monitors live positions and adds context:
    - Which bot opened it
    - WHY it was opened
    - Health assessment
    - Action suggestions
    """
    
    def __init__(self):
        self.mt5_api = MT5_API_URL
        
        # Bot magic number registry
        self.bots = {
            888007: {
                "name": "v007 The Ultimate",
                "strategy": "Trend + Price Action",
                "expected_hold": "hours to days",
                "win_rate": "50-55%"
            },
            888008: {
                "name": "v008 The Contrarian",
                "strategy": "RSI(2) Mean Reversion",
                "expected_hold": "3-5 days max",
                "win_rate": "88%"
            },
            888010: {
                "name": "v010 Second Mover",
                "strategy": "Liquidity Sweeps",
                "expected_hold": "hours",
                "win_rate": "65%"
            },
            888015: {
                "name": "v015 Big Brother",
                "strategy": "GTO Multi-Strategy",
                "expected_hold": "variable",
                "win_rate": "55%"
            },
            888016: {
                "name": "v016 Observer",
                "strategy": "Fleet Heat Scalping",
                "expected_hold": "minutes to hours",
                "win_rate": "60%"
            },
            888020: {
                "name": "v020 Ghost Commander",
                "strategy": "Raid Coordination",
                "expected_hold": "variable",
                "win_rate": "varies"
            },
            888021: {
                "name": "v021 Shadow",
                "strategy": "Stop Protection",
                "expected_hold": "hours",
                "win_rate": "55%"
            },
            888022: {
                "name": "v022 Viper",
                "strategy": "Sweep Hunting",
                "expected_hold": "hours",
                "win_rate": "60%"
            },
            888023: {
                "name": "v023 Phantom",
                "strategy": "Fade Master",
                "expected_hold": "hours",
                "win_rate": "58%"
            },
            888024: {
                "name": "v024 Chaos",
                "strategy": "Volatility Harvest",
                "expected_hold": "minutes to hours",
                "win_rate": "52%"
            },
            0: {
                "name": "NEO Direct",
                "strategy": "AI Autonomous",
                "expected_hold": "variable",
                "win_rate": "learning"
            }
        }
    
    def get_live_positions(self) -> Dict[str, Any]:
        """
        Fetch all current positions from MT5 API.
        Returns real data or error status - NEVER fake data.
        """
        try:
            response = requests.get(f"{self.mt5_api}/positions", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "positions": data.get("positions", []),
                    "source": "MT5_REAL",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"MT5 API returned {response.status_code}",
                    "positions": [],
                    "source": "ERROR"
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "MT5 API not available",
                "positions": [],
                "source": "OFFLINE"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "positions": [],
                "source": "ERROR"
            }
    
    def identify_bot(self, magic: int) -> Dict[str, str]:
        """Map magic number to bot info."""
        return self.bots.get(magic, {
            "name": f"Unknown (magic: {magic})",
            "strategy": "Unknown",
            "expected_hold": "unknown",
            "win_rate": "unknown"
        })
    
    def infer_reason(self, pos: Dict, bot_info: Dict) -> str:
        """
        Infer WHY this position was opened based on bot strategy.
        This is the critical "show your work" feature.
        """
        strategy = bot_info.get("strategy", "")
        direction = "BUY" if pos.get("type", 0) == 0 else "SELL"
        symbol = pos.get("symbol", "")
        
        reasons = {
            "RSI(2) Mean Reversion": (
                f"RSI(2) was extreme - likely "
                f"{'oversold <10 (Connors 88% win rate)' if direction == 'BUY' else 'overbought >90'}"
            ),
            "Liquidity Sweeps": (
                "Detected stop hunt + reversal pattern (WWCD playbook). "
                "Entered after sweep completed."
            ),
            "Trend + Price Action": (
                f"Trend alignment detected with price action pattern "
                f"(pin bar/inside bar/fakey). {'Bullish' if direction == 'BUY' else 'Bearish'} setup."
            ),
            "GTO Multi-Strategy": (
                "Multiple signals aligned + GTO randomizer approved entry. "
                "Unpredictable timing by design."
            ),
            "Raid Coordination": (
                "Ghost Commander coordinated raid signal from NEO or fleet confluence."
            ),
            "Stop Protection": (
                "Mental stop trade - hidden SL/TP for MM protection. "
                "Managed by internal logic."
            ),
            "Sweep Hunting": (
                "Post-sweep entry after liquidity grab at key level."
            ),
            "Fade Master": (
                "Extreme reversal setup - fading recent spike (news fade or session fade)."
            ),
            "Volatility Harvest": (
                "High volatility opportunity - quick scalp during expansion."
            ),
            "AI Autonomous": (
                "NEO-generated signal based on full knowledge base analysis."
            )
        }
        
        return reasons.get(strategy, f"Strategy-based entry ({strategy})")
    
    def calculate_pips(self, pos: Dict) -> float:
        """Calculate P&L in pips."""
        entry = pos.get("open_price", 0)
        current = pos.get("current_price", 0)
        symbol = pos.get("symbol", "")
        direction = pos.get("type", 0)  # 0 = BUY, 1 = SELL
        
        if entry == 0:
            return 0.0
        
        diff = current - entry
        if direction == 1:  # SELL
            diff = -diff
        
        # JPY pairs have different pip calculation
        if "JPY" in symbol:
            return round(diff * 100, 1)
        else:
            return round(diff * 10000, 1)
    
    def calculate_hold_time(self, pos: Dict) -> float:
        """Calculate how long position has been open in hours."""
        open_time = pos.get("time", pos.get("open_time", ""))
        if not open_time:
            return 0.0
        
        try:
            if isinstance(open_time, (int, float)):
                # Unix timestamp
                open_dt = datetime.fromtimestamp(open_time)
            else:
                # ISO string
                open_dt = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
            
            delta = datetime.now() - open_dt
            return round(delta.total_seconds() / 3600, 1)
        except:
            return 0.0
    
    def suggest_action(self, pos: Dict, bot_info: Dict, pnl: float, hold_hours: float) -> str:
        """
        Suggest what to do with this position based on strategy and current state.
        """
        strategy = bot_info.get("strategy", "")
        
        # RSI(2) positions should be short-term
        if "RSI(2)" in strategy:
            if hold_hours > 120:  # 5 days
                return "⚠️ CONSIDER CLOSING - RSI(2) max hold is 5 days per Connors rules"
            if hold_hours > 72 and pnl < 0:
                return "⚠️ REVIEW - RSI(2) losing trades rarely recover after 3 days"
            if pnl > 200:
                return "💰 CONSIDER PARTIAL - Take some profit, let rest run"
        
        # Stop hunt positions should resolve quickly
        if "Sweep" in strategy or "Liquidity" in strategy:
            if hold_hours > 8 and pnl < 0:
                return "⚠️ REVIEW - Sweep trades should resolve within hours"
            if hold_hours > 24:
                return "⚠️ THESIS CHECK - Original sweep may have failed"
        
        # Volatility scalps
        if "Volatility" in strategy:
            if hold_hours > 4:
                return "⚠️ REVIEW - Volatility trades should be quick"
        
        # General P&L based suggestions
        if pnl > 500:
            return "🟢 STRONG WINNER - Consider trailing stop"
        elif pnl > 200:
            return "🟢 WINNING - Monitor for exit signals"
        elif pnl < -500:
            return "🔴 HEAVY LOSS - Check if thesis still valid"
        elif pnl < -300:
            return "🟡 LOSING - Review stop placement"
        elif pnl < 0:
            return "🟡 UNDERWATER - Hold if thesis intact"
        else:
            return "✅ HOLD - Within normal parameters"
    
    def get_position_context(self, positions: List[Dict]) -> List[Dict]:
        """
        Add full context to each position.
        This is the core value - understanding WHY each position exists.
        """
        context = []
        
        for pos in positions:
            magic = pos.get("magic", 0)
            bot_info = self.identify_bot(magic)
            pnl = pos.get("profit", 0)
            hold_hours = self.calculate_hold_time(pos)
            pips = self.calculate_pips(pos)
            
            position_context = {
                # Basic info
                "ticket": pos.get("ticket"),
                "symbol": pos.get("symbol"),
                "direction": "BUY" if pos.get("type", 0) == 0 else "SELL",
                "volume": pos.get("volume"),
                "entry_price": pos.get("open_price"),
                "current_price": pos.get("current_price"),
                
                # P&L
                "pnl_dollars": round(pnl, 2),
                "pnl_pips": pips,
                
                # WHO opened it
                "bot_magic": magic,
                "bot_name": bot_info["name"],
                "bot_strategy": bot_info["strategy"],
                "expected_win_rate": bot_info["win_rate"],
                
                # WHY it was opened
                "likely_reason": self.infer_reason(pos, bot_info),
                
                # HEALTH assessment
                "health": "🟢 winning" if pnl > 0 else "🔴 losing" if pnl < 0 else "⚪ breakeven",
                "hold_time_hours": hold_hours,
                
                # RECOMMENDATION
                "action_suggestion": self.suggest_action(pos, bot_info, pnl, hold_hours),
                
                # Timestamps
                "open_time": pos.get("time", pos.get("open_time")),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            context.append(position_context)
        
        return context
    
    def format_for_neo(self, positions_context: List[Dict]) -> str:
        """Format position data for NEO's context window."""
        if not positions_context:
            return """=== CURRENT BATTLEFIELD ===

No open positions.
Fleet is standing by.
"""
        
        lines = [
            "=== CURRENT BATTLEFIELD ===",
            ""
        ]
        
        total_pnl = sum(p["pnl_dollars"] for p in positions_context)
        winning = sum(1 for p in positions_context if p["pnl_dollars"] > 0)
        losing = sum(1 for p in positions_context if p["pnl_dollars"] < 0)
        
        lines.append(f"📊 Total Open P&L: ${total_pnl:+,.2f}")
        lines.append(f"📈 Active Positions: {len(positions_context)} ({winning}W / {losing}L)")
        lines.append("")
        
        for p in positions_context:
            lines.append("─" * 50)
            lines.append(f"📍 {p['symbol']} {p['direction']} ({p['volume']} lots)")
            lines.append(f"   Bot: {p['bot_name']}")
            lines.append(f"   Strategy: {p['bot_strategy']}")
            lines.append(f"   P&L: ${p['pnl_dollars']:+.2f} ({p['pnl_pips']:+.1f} pips)")
            lines.append(f"   Hold Time: {p['hold_time_hours']:.1f} hours")
            lines.append(f"   WHY: {p['likely_reason']}")
            lines.append(f"   💡 {p['action_suggestion']}")
        
        lines.append("")
        lines.append("─" * 50)
        
        return "\n".join(lines)
    
    def run_continuous(self, interval_seconds: int = 30):
        """Run continuous monitoring."""
        print(f"[{datetime.now()}] Position Monitor started (interval: {interval_seconds}s)")
        
        while True:
            try:
                result = self.get_live_positions()
                
                if result["success"]:
                    context = self.get_position_context(result["positions"])
                    battlefield = self.format_for_neo(context)
                    
                    # Write for NEO to read
                    with open("/tmp/neo_battlefield.txt", "w") as f:
                        f.write(battlefield)
                    
                    # Also save as JSON for programmatic access
                    with open("/tmp/neo_positions.json", "w") as f:
                        json.dump(context, f, indent=2)
                    
                    print(f"[{datetime.now()}] Updated battlefield: {len(context)} positions, "
                          f"${sum(p['pnl_dollars'] for p in context):+,.2f} P&L")
                else:
                    print(f"[{datetime.now()}] MT5 status: {result['source']} - {result.get('error', 'Unknown')}")
                
            except Exception as e:
                print(f"[{datetime.now()}] Error: {e}")
            
            time.sleep(interval_seconds)


def test_position_monitor():
    """Test the position monitor."""
    print("=" * 60)
    print("POSITION MONITOR TEST")
    print("=" * 60)
    
    monitor = PositionMonitor()
    result = monitor.get_live_positions()
    
    print(f"MT5 Status: {result['source']}")
    print(f"Positions: {len(result['positions'])}")
    
    if result["positions"]:
        context = monitor.get_position_context(result["positions"])
        print("\n" + monitor.format_for_neo(context))
    else:
        print("\nNo open positions (or MT5 not connected)")
        print("This is normal if MT5 API is not running locally")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--run", action="store_true", help="Run continuous")
    parser.add_argument("--interval", type=int, default=30, help="Update interval")
    args = parser.parse_args()
    
    if args.test:
        test_position_monitor()
    elif args.run:
        monitor = PositionMonitor()
        monitor.run_continuous(args.interval)
    else:
        test_position_monitor()

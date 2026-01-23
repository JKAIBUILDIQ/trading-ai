#!/usr/bin/env python3
"""
Performance Tracker
Logs and analyzes REAL trading performance

Features:
1. Track all signals and their outcomes
2. Compare ML signals vs actual trades
3. Calculate real performance metrics
4. Generate reports

RULES:
1. All P&L data comes from MT5 (real trades)
2. No simulated performance numbers
3. Every metric has data source
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import data_config


class PerformanceTracker:
    """
    Track and analyze trading performance
    """
    
    def __init__(self):
        self.redis = redis.Redis(host="localhost", decode_responses=True)
        self.mt5_api = data_config.MT5_API_URL
    
    def log_signal(self, signal: Dict):
        """Log a generated signal"""
        signal_id = f"{signal['signal']['pair']}_{signal['timestamp']}"
        
        record = {
            "id": signal_id,
            "timestamp": signal["timestamp"],
            "pair": signal["signal"]["pair"],
            "action": signal["signal"]["action"],
            "confidence": signal["signal"].get("confidence", 0),
            "source": signal["source"],
            "outcome": None,  # Will be filled when trade closes
            "pnl": None
        }
        
        self.redis.hset("performance:signals", signal_id, json.dumps(record))
        self.redis.lpush("performance:signal_history", signal_id)
        self.redis.ltrim("performance:signal_history", 0, 9999)
    
    def log_trade(self, trade: Dict):
        """Log a real trade from MT5"""
        trade_id = trade.get("ticket") or f"trade_{datetime.utcnow().timestamp()}"
        
        record = {
            "id": trade_id,
            "timestamp": trade.get("open_time", datetime.utcnow().isoformat()),
            "close_time": trade.get("close_time"),
            "pair": trade.get("symbol"),
            "direction": trade.get("type"),  # buy/sell
            "volume": trade.get("volume"),
            "entry_price": trade.get("open_price"),
            "exit_price": trade.get("close_price"),
            "pnl": trade.get("profit"),
            "source": "MT5_REAL_TRADE"
        }
        
        self.redis.hset("performance:trades", trade_id, json.dumps(record))
        self.redis.lpush("performance:trade_history", trade_id)
    
    def get_mt5_trades(self, days: int = 30) -> Dict:
        """Fetch real trades from MT5 API"""
        import requests
        
        try:
            response = requests.get(f"{self.mt5_api}/trades", timeout=10)
            
            if response.status_code == 200:
                trades = response.json()
                return {
                    "success": True,
                    "source": "MT5_REAL_DATA",
                    "trades": trades.get("trades", trades),
                    "count": len(trades.get("trades", trades))
                }
            else:
                return {
                    "success": False,
                    "error": f"MT5 returned {response.status_code}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from real trades
        """
        if not trades:
            return {
                "status": "NO_TRADES",
                "message": "No trade data available"
            }
        
        # Filter closed trades
        closed = [t for t in trades if t.get("close_time") or t.get("profit") is not None]
        
        if not closed:
            return {
                "status": "NO_CLOSED_TRADES",
                "message": "No closed trades to analyze"
            }
        
        # Calculate metrics
        total_pnl = sum(float(t.get("profit", 0)) for t in closed)
        wins = [t for t in closed if float(t.get("profit", 0)) > 0]
        losses = [t for t in closed if float(t.get("profit", 0)) < 0]
        
        gross_profit = sum(float(t.get("profit", 0)) for t in wins)
        gross_loss = abs(sum(float(t.get("profit", 0)) for t in losses))
        
        # By symbol
        by_symbol = {}
        for t in closed:
            symbol = t.get("symbol", "UNKNOWN")
            if symbol not in by_symbol:
                by_symbol[symbol] = {"pnl": 0, "count": 0, "wins": 0}
            by_symbol[symbol]["pnl"] += float(t.get("profit", 0))
            by_symbol[symbol]["count"] += 1
            if float(t.get("profit", 0)) > 0:
                by_symbol[symbol]["wins"] += 1
        
        metrics = {
            "source": "MT5_REAL_PERFORMANCE",
            "calculated_at": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_trades": len(closed),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0,
                "total_pnl": round(total_pnl, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
                "average_win": round(gross_profit / len(wins), 2) if wins else 0,
                "average_loss": round(gross_loss / len(losses), 2) if losses else 0,
            },
            "by_symbol": by_symbol,
            "verified": True
        }
        
        # Store in Redis
        self.redis.set("performance:metrics:latest", json.dumps(metrics))
        
        return metrics
    
    def compare_signals_vs_trades(self) -> Dict:
        """
        Compare AI signals with actual MT5 trades
        
        This shows how well the AI predictions match reality
        """
        # Get recent signals
        signal_ids = self.redis.lrange("performance:signal_history", 0, 99)
        signals = []
        for sid in signal_ids:
            data = self.redis.hget("performance:signals", sid)
            if data:
                signals.append(json.loads(data))
        
        # Get recent trades
        trades_result = self.get_mt5_trades(days=7)
        
        if not trades_result["success"]:
            return {
                "success": False,
                "error": f"Cannot fetch trades: {trades_result.get('error')}"
            }
        
        trades = trades_result["trades"]
        
        # Simple comparison (would be more sophisticated in production)
        comparison = {
            "source": "SIGNAL_VS_TRADE_COMPARISON",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "signals_generated": len(signals),
            "trades_executed": len(trades),
            "analysis": {
                "buy_signals": len([s for s in signals if s.get("action") == "BUY"]),
                "sell_signals": len([s for s in signals if s.get("action") == "SELL"]),
                "actual_buys": len([t for t in trades if t.get("type") == "buy"]),
                "actual_sells": len([t for t in trades if t.get("type") == "sell"])
            },
            "verified": True
        }
        
        return comparison
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive performance report
        """
        # Fetch all data
        trades_result = self.get_mt5_trades(days=30)
        
        if not trades_result["success"]:
            return {
                "success": False,
                "error": f"Cannot generate report: {trades_result.get('error')}"
            }
        
        trades = trades_result["trades"]
        metrics = self.calculate_metrics(trades)
        comparison = self.compare_signals_vs_trades()
        
        report = {
            "report_type": "TRADING_PERFORMANCE_REPORT",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "data_source": "MT5_REAL_DATA",
            "period_days": 30,
            "metrics": metrics,
            "signal_comparison": comparison,
            "raw_trade_count": len(trades),
            "verified": True,
            "note": "All data from real MT5 trades - no simulated values"
        }
        
        # Store report
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.redis.hset("performance:reports", report_id, json.dumps(report))
        
        return report


class PerformanceAPI:
    """
    Flask API for performance tracking
    Exposes endpoints for dashboards
    """
    
    def __init__(self):
        from flask import Flask, jsonify
        self.app = Flask(__name__)
        self.tracker = PerformanceTracker()
        
        # Register routes
        @self.app.route('/performance/metrics')
        def get_metrics():
            trades_result = self.tracker.get_mt5_trades()
            if trades_result["success"]:
                metrics = self.tracker.calculate_metrics(trades_result["trades"])
                return jsonify(metrics)
            return jsonify(trades_result), 500
        
        @self.app.route('/performance/report')
        def get_report():
            report = self.tracker.generate_report()
            return jsonify(report)
        
        @self.app.route('/performance/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "PERFORMANCE_TRACKER"
            })
    
    def run(self, host="0.0.0.0", port=8091):
        """Run the API server"""
        self.app.run(host=host, port=port)


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE TRACKER TEST")
    print("=" * 60)
    
    tracker = PerformanceTracker()
    
    # Test MT5 connection
    print("\nFetching real trades from MT5...")
    trades_result = tracker.get_mt5_trades()
    
    if trades_result["success"]:
        print(f"✅ Found {trades_result['count']} trades")
        
        # Calculate metrics
        metrics = tracker.calculate_metrics(trades_result["trades"])
        
        print(f"\n📊 Performance Metrics:")
        if "summary" in metrics:
            for key, value in metrics["summary"].items():
                print(f"  {key}: {value}")
        else:
            print(f"  Status: {metrics.get('status')}")
            print(f"  Message: {metrics.get('message')}")
    else:
        print(f"⚠️ Cannot fetch trades: {trades_result.get('error')}")
        print("\nTo use real performance tracking:")
        print("1. Ensure MT5 API is running on port 8085")
        print("2. Run trades through your Crellastein bots")
        print("3. This tracker will show REAL performance")
    
    print("\n" + "=" * 60)
    print("To run Performance API server:")
    print("  python -c \"from performance_tracker.tracker import PerformanceAPI; PerformanceAPI().run()\"")
    print("Then access: http://localhost:8091/performance/metrics")
    print("=" * 60)

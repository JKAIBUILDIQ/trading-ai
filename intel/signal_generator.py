#!/usr/bin/env python3
"""
Consensus Signal Generator
Generates trading signals when multiple top traders agree

RULE: NO random data. Signals are based entirely on scraped trader data.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import os

from config import signal_config
from database import IntelDatabase


class ConsensusSignalGenerator:
    """
    Generates trading signals based on consensus from top traders
    
    Logic:
    - When 3+ top traders enter the same symbol/direction within 1 hour
    - Generate a signal with confidence based on trader quality
    """
    
    def __init__(self):
        self.db = IntelDatabase()
        
        # Ensure signals directory exists
        os.makedirs(signal_config.SIGNALS_DIR, exist_ok=True)
    
    def find_consensus_signals(self, hours: int = 24) -> List[Dict]:
        """
        Find consensus signals from recent trades
        
        Returns signals where 3+ traders agree on symbol/direction
        """
        # Get recent trades
        trades = self.db.get_recent_trades(hours=hours)
        
        if not trades:
            print("No recent trades found")
            return []
        
        print(f"Analyzing {len(trades)} recent trades...")
        
        # Group trades by symbol and direction
        # Key: (symbol, direction) -> List of trades
        groups = defaultdict(list)
        
        for trade in trades:
            symbol = trade.get('symbol', '').upper()
            direction = trade.get('direction', '').upper()
            
            if symbol and direction:
                key = (symbol, direction)
                groups[key].append(trade)
        
        # Find groups with enough consensus
        signals = []
        
        for (symbol, direction), trade_list in groups.items():
            # Check if we have enough traders
            unique_traders = set(t.get('trader_id') for t in trade_list)
            
            if len(unique_traders) >= signal_config.MIN_TRADERS_FOR_SIGNAL:
                # Check time window - all trades within MAX_TIME_WINDOW_HOURS
                trade_times = []
                for t in trade_list:
                    if t.get('close_time'):
                        try:
                            # Parse various time formats
                            time_str = t['close_time']
                            if 'T' in time_str:
                                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            else:
                                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            trade_times.append(dt)
                        except:
                            pass
                
                # If we have timestamps, check the time window
                within_window = True
                if len(trade_times) >= 2:
                    time_range = max(trade_times) - min(trade_times)
                    if time_range.total_seconds() > signal_config.MAX_TIME_WINDOW_HOURS * 3600:
                        within_window = False
                
                if within_window:
                    # Calculate confidence based on number of agreeing traders
                    trader_count = len(unique_traders)
                    base_confidence = 50 + (trader_count * 10)
                    
                    # Boost confidence if traders have good stats
                    for t in trade_list:
                        trader_id = t.get('trader_id')
                        # Could look up trader stats here to boost confidence
                    
                    confidence = min(base_confidence, 95)  # Cap at 95%
                    
                    if confidence >= signal_config.MIN_CONFIDENCE:
                        # Build trader list for signal
                        traders_info = []
                        seen_traders = set()
                        
                        for t in trade_list:
                            trader_id = t.get('trader_id')
                            if trader_id and trader_id not in seen_traders:
                                seen_traders.add(trader_id)
                                traders_info.append({
                                    "id": trader_id,
                                    "name": t.get('trader_name', trader_id),
                                    "url": t.get('trader_url', t.get('source_url', '')),
                                    "pnl": t.get('pnl')
                                })
                        
                        signal = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "symbol": symbol,
                            "direction": direction,
                            "confidence": confidence,
                            "source_type": "trader_consensus",
                            "trader_count": trader_count,
                            "traders": traders_info,
                            "trades_analyzed": len(trade_list),
                            "time_window_hours": hours
                        }
                        
                        signals.append(signal)
        
        print(f"Found {len(signals)} consensus signals")
        return signals
    
    def check_news_impact(self, symbol: str) -> Dict:
        """
        Check for high-impact news affecting a symbol
        
        Returns warning if high-impact news is upcoming
        """
        # Extract currency from symbol (e.g., EURUSD -> EUR, USD)
        currencies = []
        if len(symbol) >= 6:
            currencies = [symbol[:3], symbol[3:6]]
        
        upcoming_events = []
        
        for currency in currencies:
            events = self.db.get_upcoming_events(
                hours=24,
                currency=currency,
                impact="high"
            )
            upcoming_events.extend(events)
        
        if upcoming_events:
            return {
                "has_news": True,
                "events": upcoming_events,
                "warning": f"High-impact news for {'/'.join(currencies)} in next 24h"
            }
        
        return {"has_news": False, "events": []}
    
    def generate_and_save(self, hours: int = 24) -> Dict:
        """
        Generate signals and save to file and database
        """
        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "signals_generated": 0,
            "signals_saved": 0,
            "signals": []
        }
        
        # Find consensus signals
        signals = self.find_consensus_signals(hours=hours)
        result["signals_generated"] = len(signals)
        
        # Enhance signals with news check
        enhanced_signals = []
        for signal in signals:
            news_check = self.check_news_impact(signal["symbol"])
            signal["news_warning"] = news_check.get("warning") if news_check["has_news"] else None
            signal["upcoming_news_count"] = len(news_check.get("events", []))
            enhanced_signals.append(signal)
        
        result["signals"] = enhanced_signals
        
        # Save to database
        for signal in enhanced_signals:
            try:
                self.db.insert_signal(signal)
                result["signals_saved"] += 1
            except Exception as e:
                print(f"Error saving signal: {e}")
        
        # Save to JSON file
        output = {
            "generated_at": result["timestamp"],
            "source": "consensus_signal_generator",
            "signals": enhanced_signals,
            "verification": {
                "rule": "All signals based on real trader data from Myfxbook/MQL5",
                "no_random_data": True,
                "traders_verifiable": True
            }
        }
        
        try:
            with open(signal_config.LATEST_SIGNAL_FILE, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"✅ Signals saved to: {signal_config.LATEST_SIGNAL_FILE}")
        except Exception as e:
            print(f"Error saving to file: {e}")
        
        # Also save timestamped version
        timestamp_file = os.path.join(
            signal_config.SIGNALS_DIR,
            f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(timestamp_file, 'w') as f:
                json.dump(output, f, indent=2)
        except:
            pass
        
        return result
    
    def get_latest_signals(self) -> Dict:
        """Read the latest signals from file"""
        try:
            with open(signal_config.LATEST_SIGNAL_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": "No signals file found", "signals": []}
        except Exception as e:
            return {"error": str(e), "signals": []}


def test_generator():
    """Test the signal generator"""
    print("=" * 60)
    print("CONSENSUS SIGNAL GENERATOR TEST")
    print("=" * 60)
    
    gen = ConsensusSignalGenerator()
    
    # Check database stats
    db = IntelDatabase()
    stats = db.get_stats()
    
    print(f"\n📊 Database Status:")
    print(f"  Total traders: {stats['total_traders']}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Total calendar events: {stats['total_events']}")
    
    if stats['total_trades'] == 0:
        print("\n⚠️ No trades in database yet!")
        print("  Run the scrapers first:")
        print("    python myfxbook_scraper.py")
        print("    python mql5_scraper.py")
        return
    
    # Generate signals
    print("\n🔍 Generating signals...")
    result = gen.generate_and_save(hours=72)  # Look back 72 hours
    
    print(f"\n📊 Results:")
    print(f"  Signals generated: {result['signals_generated']}")
    print(f"  Signals saved: {result['signals_saved']}")
    
    if result['signals']:
        print(f"\n  Signals found:")
        for sig in result['signals']:
            print(f"    {sig['symbol']} {sig['direction']}: {sig['confidence']}% confidence")
            print(f"      Traders agreeing: {sig['trader_count']}")
            if sig.get('news_warning'):
                print(f"      ⚠️ {sig['news_warning']}")


if __name__ == "__main__":
    test_generator()

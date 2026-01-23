#!/usr/bin/env python3
"""
INTEL BOTS
Specialized Ollama models feeding intelligence to NEO.
Each bot has ONE job and does it well.

Like having a team of analysts working for you.

ALPHA - Market Scanner (qwen3:32b)
BRAVO - News Monitor (llama3.1:8b)  
CHARLIE - Position Analyst (deepseek-r1:32b)
DELTA - Threat Detector (qwen3:32b)
"""

import subprocess
import json
import time
from datetime import datetime
import threading
from typing import Dict, List, Any, Optional
import sys
sys.path.append('..')


class IntelBot:
    """Base class for all intel bots."""
    
    def __init__(self, name: str, model: str, specialty: str):
        self.name = name
        self.model = model
        self.specialty = specialty
        self.last_intel = None
        self.last_update = None
        self.timeout = 90
    
    def ask(self, prompt: str) -> Optional[str]:
        """Query the Ollama model."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ {self.name} timed out ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"  ❌ {self.name} error: {e}")
            return None
    
    def parse_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from response."""
        if not response:
            return None
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        return None
    
    def gather_intel(self, data: Any) -> Dict:
        """Override in subclass."""
        raise NotImplementedError


class MarketScanner(IntelBot):
    """
    ALPHA - Scans all pairs for setups.
    Uses qwen3:32b for fast analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="ALPHA Scanner",
            model="qwen3:32b",
            specialty="Setup detection across all pairs"
        )
    
    def gather_intel(self, prices: Dict) -> Dict:
        prompt = f"""You are ALPHA, a market scanner. Your ONLY job is to identify trading setups.

CURRENT PRICES:
{json.dumps(prices, indent=2)}

Scan for these setups:
1. RSI(2) extremes (< 10 oversold, > 90 overbought) - 88% win rate per Connors
2. Price at session high/low (potential stop hunt)
3. Round number proximity (within 20 pips of .x000 or .x500)
4. Potential stop hunt levels (obvious support/resistance)

OUTPUT JSON only (no explanation):
{{
  "setups_found": [
    {{
      "symbol": "EURUSD",
      "setup_type": "RSI2_OVERSOLD",
      "confidence": 85,
      "notes": "brief note"
    }}
  ],
  "best_opportunity": "symbol or NONE",
  "avoid": ["symbols to avoid and why"]
}}"""
        
        response = self.ask(prompt)
        parsed = self.parse_json(response)
        
        self.last_intel = parsed or {"raw": response, "error": "Could not parse"}
        self.last_update = datetime.now()
        
        return {
            "status": "success" if parsed else "partial",
            "intel": self.last_intel
        }


class NewsMonitor(IntelBot):
    """
    BRAVO - Monitors for news events.
    Uses llama3.1:8b for speed.
    """
    
    def __init__(self):
        super().__init__(
            name="BRAVO News",
            model="llama3.1:8b",
            specialty="News and event monitoring"
        )
        self.timeout = 45  # Faster model, shorter timeout
    
    def gather_intel(self, data: Dict) -> Dict:
        # Get current time info
        now = datetime.now()
        calendar_events = data.get("calendar", [])
        
        prompt = f"""You are BRAVO, a news monitor. Your ONLY job is to assess news risk.

CURRENT TIME: {now.isoformat()}
DAY: {now.strftime('%A')}

KNOWN UPCOMING EVENTS:
{json.dumps(calendar_events, indent=2) if calendar_events else "No calendar data available"}

Based on typical forex calendar:
- NFP is first Friday of month
- FOMC is every 6 weeks
- CPI is mid-month

Assess:
1. Any high-impact news likely in next 2 hours?
2. Which currencies affected?
3. Should we avoid trading or prepare for volatility?

OUTPUT JSON only:
{{
  "news_risk": "LOW",
  "next_high_impact": "event name or NONE",
  "time_until": "minutes or N/A",
  "currencies_affected": ["USD", "EUR"],
  "recommendation": "TRADE_NORMAL"
}}

Note: recommendation must be one of: TRADE_NORMAL, REDUCE_SIZE, AVOID, PREPARE_FADE"""
        
        response = self.ask(prompt)
        parsed = self.parse_json(response)
        
        self.last_intel = parsed or {"raw": response, "error": "Could not parse", "news_risk": "UNKNOWN"}
        self.last_update = datetime.now()
        
        return {
            "status": "success" if parsed else "partial",
            "intel": self.last_intel
        }


class PositionAnalyst(IntelBot):
    """
    CHARLIE - Analyzes current positions.
    Uses deepseek-r1:32b for deep reasoning.
    """
    
    def __init__(self):
        super().__init__(
            name="CHARLIE Analyst",
            model="deepseek-r1:32b",
            specialty="Position analysis and recommendations"
        )
        self.timeout = 120  # More reasoning time
    
    def gather_intel(self, data: Dict) -> Dict:
        positions = data.get("positions", [])
        market_state = data.get("market", {})
        
        if not positions:
            return {
                "status": "success",
                "intel": {
                    "portfolio_health": "NEUTRAL",
                    "total_risk": "0%",
                    "recommendations": [],
                    "correlation_warning": None,
                    "overall_advice": "No open positions. Standing by for opportunities."
                }
            }
        
        prompt = f"""You are CHARLIE, a position analyst. Your ONLY job is to analyze open positions.

CURRENT POSITIONS:
{json.dumps(positions, indent=2)}

MARKET STATE:
{json.dumps(market_state, indent=2) if market_state else "No market state data"}

For each position, assess:
1. Is the original thesis still valid?
2. Has the market regime changed?
3. Should we add, hold, reduce, or close?

OUTPUT JSON only:
{{
  "portfolio_health": "GOOD",
  "total_risk": "3.5%",
  "recommendations": [
    {{
      "ticket": 12345,
      "symbol": "EURUSD",
      "current_action": "HOLD",
      "reasoning": "brief"
    }}
  ],
  "correlation_warning": "any correlated positions?",
  "overall_advice": "one sentence"
}}

Note: current_action must be one of: HOLD, ADD, REDUCE, CLOSE
Note: portfolio_health must be one of: GOOD, CONCERNING, CRITICAL"""
        
        response = self.ask(prompt)
        parsed = self.parse_json(response)
        
        self.last_intel = parsed or {"raw": response, "error": "Could not parse"}
        self.last_update = datetime.now()
        
        return {
            "status": "success" if parsed else "partial",
            "intel": self.last_intel
        }


class ThreatDetector(IntelBot):
    """
    DELTA - Watches for MM activity / threats.
    Uses qwen3:32b for pattern recognition.
    Implements WWCD (What Would Citadel Do).
    """
    
    def __init__(self):
        super().__init__(
            name="DELTA Threat",
            model="qwen3:32b",
            specialty="MM activity and threat detection (WWCD)"
        )
    
    def gather_intel(self, data: Dict) -> Dict:
        prices = data.get("prices", {})
        positions = data.get("positions", [])
        
        prompt = f"""You are DELTA, a threat detector using WWCD (What Would Citadel Do).

Your job is to think like a market maker and identify:
1. Where would MMs hunt stops?
2. Are they currently running a hunt?
3. What traps are being set?

CURRENT PRICES:
{json.dumps(prices, indent=2) if prices else "No price data"}

OUR POSITIONS:
{json.dumps(positions, indent=2) if positions else "No positions"}

Watch for:
1. Stop hunt patterns (spike through level, quick reversal)
2. Unusual volatility spikes
3. Liquidity grabs at round numbers (.x000, .x500)
4. Session manipulation (London/NY open)

OUTPUT JSON only:
{{
  "threat_level": "GREEN",
  "active_hunts": [
    {{
      "symbol": "EURUSD",
      "type": "STOP_HUNT_BEARISH",
      "level": 1.0850,
      "status": "IN_PROGRESS"
    }}
  ],
  "opportunities": "any post-hunt opportunities?",
  "warning": "what to watch out for"
}}

Note: threat_level must be: GREEN, YELLOW, or RED
Note: hunt status must be: IN_PROGRESS, COMPLETED, or FAILED"""
        
        response = self.ask(prompt)
        parsed = self.parse_json(response)
        
        self.last_intel = parsed or {"raw": response, "error": "Could not parse", "threat_level": "UNKNOWN"}
        self.last_update = datetime.now()
        
        return {
            "status": "success" if parsed else "partial",
            "intel": self.last_intel
        }


class IntelCoordinator:
    """
    Coordinates all intel bots and feeds to NEO.
    Runs bots in parallel for speed.
    """
    
    def __init__(self):
        self.alpha = MarketScanner()
        self.bravo = NewsMonitor()
        self.charlie = PositionAnalyst()
        self.delta = ThreatDetector()
        
        self.bots = {
            "ALPHA Scanner": self.alpha,
            "BRAVO News": self.bravo,
            "CHARLIE Analyst": self.charlie,
            "DELTA Threat": self.delta
        }
    
    def gather_all_intel(self, market_data: Dict) -> Dict:
        """
        Run all intel bots and compile report for NEO.
        Runs in parallel for speed.
        """
        intel_report = {
            "timestamp": datetime.now().isoformat(),
            "intel": {}
        }
        
        results = {}
        threads = []
        
        def run_bot(bot_name: str, bot: IntelBot, data: Dict):
            try:
                print(f"  📡 {bot_name} gathering intel...")
                result = bot.gather_intel(data)
                results[bot_name] = result
                print(f"  ✅ {bot_name} complete")
            except Exception as e:
                results[bot_name] = {
                    "status": "error",
                    "intel": {"error": str(e)}
                }
                print(f"  ❌ {bot_name} failed: {e}")
        
        # Prepare data for each bot
        bot_data = {
            "ALPHA Scanner": market_data.get("prices", {}),
            "BRAVO News": {"calendar": market_data.get("calendar", [])},
            "CHARLIE Analyst": {
                "positions": market_data.get("positions", []),
                "market": market_data.get("prices", {})
            },
            "DELTA Threat": {
                "prices": market_data.get("prices", {}),
                "positions": market_data.get("positions", [])
            }
        }
        
        # Start all bots in parallel
        for bot_name, bot in self.bots.items():
            t = threading.Thread(
                target=run_bot,
                args=(bot_name, bot, bot_data[bot_name])
            )
            threads.append(t)
            t.start()
        
        # Wait for all to complete (max 120 seconds total)
        for t in threads:
            t.join(timeout=120)
        
        intel_report["intel"] = results
        
        # Save for NEO
        with open("/tmp/neo_intel_report.json", "w") as f:
            json.dump(intel_report, f, indent=2)
        
        return intel_report
    
    def format_briefing(self, intel_report: Dict) -> str:
        """Create human-readable briefing for NEO."""
        lines = [
            "=== INTEL BRIEFING ===",
            f"Timestamp: {intel_report['timestamp']}",
            ""
        ]
        
        for bot_name, data in intel_report.get("intel", {}).items():
            lines.append(f"📡 {bot_name}")
            lines.append(f"   Status: {data.get('status', 'unknown')}")
            
            intel = data.get("intel", {})
            if isinstance(intel, dict):
                # Format key fields
                if "threat_level" in intel:
                    emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(intel["threat_level"], "⚪")
                    lines.append(f"   Threat Level: {emoji} {intel['threat_level']}")
                if "news_risk" in intel:
                    lines.append(f"   News Risk: {intel['news_risk']}")
                if "best_opportunity" in intel:
                    lines.append(f"   Best Setup: {intel['best_opportunity']}")
                if "portfolio_health" in intel:
                    lines.append(f"   Portfolio: {intel['portfolio_health']}")
                if "overall_advice" in intel:
                    lines.append(f"   Advice: {intel['overall_advice']}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def run_continuous(self, interval_seconds: int = 300):
        """Run intel gathering continuously."""
        print(f"[{datetime.now()}] Intel Coordinator started (interval: {interval_seconds}s)")
        
        # Import here to avoid circular imports
        sys.path.append('..')
        from market_feed import MarketFeed
        from position_tracker import PositionTracker
        
        market_feed = MarketFeed()
        position_tracker = PositionTracker()
        
        while True:
            print(f"\n[{datetime.now()}] Gathering intel from all bots...")
            
            # Get real data
            market_snapshot = market_feed.get_snapshot()
            account_state = position_tracker.get_state()
            
            # Prepare market data
            prices = {}
            for symbol, data in market_snapshot.forex.items():
                prices[symbol] = data.price
            for symbol, data in market_snapshot.crypto.items():
                prices[symbol] = data.price
            
            market_data = {
                "prices": prices,
                "calendar": [],  # Would need forex factory scraper
                "positions": [
                    {
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": 0 if p.direction == "BUY" else 1,
                        "volume": p.volume,
                        "profit": p.unrealized_pnl
                    }
                    for p in account_state.open_positions
                ]
            }
            
            report = self.gather_all_intel(market_data)
            briefing = self.format_briefing(report)
            
            print(briefing)
            
            time.sleep(interval_seconds)


def test_intel_bots():
    """Test the intel bots with sample data."""
    print("=" * 60)
    print("INTEL BOTS TEST")
    print("=" * 60)
    
    coordinator = IntelCoordinator()
    
    # Sample market data
    sample_data = {
        "prices": {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 148.50
        },
        "calendar": [],
        "positions": []
    }
    
    print("\nGathering intel from all bots (this may take a minute)...\n")
    report = coordinator.gather_all_intel(sample_data)
    
    print("\n" + coordinator.format_briefing(report))
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--run", action="store_true", help="Run continuous")
    parser.add_argument("--interval", type=int, default=300, help="Update interval")
    args = parser.parse_args()
    
    if args.test:
        test_intel_bots()
    elif args.run:
        coordinator = IntelCoordinator()
        coordinator.run_continuous(args.interval)
    else:
        print("Usage: python3 intel_bots.py --test | --run")

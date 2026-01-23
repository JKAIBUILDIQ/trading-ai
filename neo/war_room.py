#!/usr/bin/env python3
"""
NEO'S WAR ROOM
Combines all intel into one view for NEO's decision making.

The War Room aggregates:
- Live positions with WHY they exist
- Intel from all bots (ALPHA, BRAVO, CHARLIE, DELTA)
- Battlefield summary
- Threat assessment

This becomes part of NEO's THINK prompt.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class WarRoom:
    """
    NEO's War Room - Complete battlefield awareness.
    """
    
    def __init__(self):
        self.battlefield_file = "/tmp/neo_battlefield.txt"
        self.positions_file = "/tmp/neo_positions.json"
        self.intel_file = "/tmp/neo_intel_report.json"
    
    def get_positions_context(self) -> Dict[str, Any]:
        """Get current positions with context."""
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file) as f:
                    return {
                        "available": True,
                        "data": json.load(f)
                    }
            except:
                pass
        return {"available": False, "data": []}
    
    def get_intel_report(self) -> Dict[str, Any]:
        """Get latest intel from all bots."""
        if os.path.exists(self.intel_file):
            try:
                with open(self.intel_file) as f:
                    return {
                        "available": True,
                        "data": json.load(f)
                    }
            except:
                pass
        return {"available": False, "data": {}}
    
    def get_battlefield_summary(self) -> str:
        """Get human-readable battlefield summary."""
        if os.path.exists(self.battlefield_file):
            try:
                with open(self.battlefield_file) as f:
                    return f.read()
            except:
                pass
        return "Battlefield data not available. Position monitor may not be running."
    
    def get_full_context(self) -> Dict[str, Any]:
        """Compile everything NEO needs to see."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": self.get_positions_context(),
            "intel": self.get_intel_report(),
            "battlefield": self.get_battlefield_summary()
        }
    
    def extract_intel_summary(self, intel_data: Dict) -> Dict[str, str]:
        """Extract key intel from each bot."""
        summaries = {
            "alpha": "No data",
            "bravo": "No data",
            "charlie": "No data",
            "delta": "No data"
        }
        
        intel = intel_data.get("data", {}).get("intel", {})
        
        # ALPHA Scanner
        alpha = intel.get("ALPHA Scanner", {}).get("intel", {})
        if alpha:
            best = alpha.get("best_opportunity", "NONE")
            setups = len(alpha.get("setups_found", []))
            summaries["alpha"] = f"Best: {best}, {setups} setups found"
        
        # BRAVO News
        bravo = intel.get("BRAVO News", {}).get("intel", {})
        if bravo:
            risk = bravo.get("news_risk", "UNKNOWN")
            rec = bravo.get("recommendation", "")
            summaries["bravo"] = f"Risk: {risk}, Action: {rec}"
        
        # CHARLIE Analyst
        charlie = intel.get("CHARLIE Analyst", {}).get("intel", {})
        if charlie:
            health = charlie.get("portfolio_health", "UNKNOWN")
            advice = charlie.get("overall_advice", "")[:100]
            summaries["charlie"] = f"Portfolio: {health}. {advice}"
        
        # DELTA Threat
        delta = intel.get("DELTA Threat", {}).get("intel", {})
        if delta:
            threat = delta.get("threat_level", "UNKNOWN")
            hunts = len(delta.get("active_hunts", []))
            warning = delta.get("warning", "")[:100]
            summaries["delta"] = f"Threat: {threat}, {hunts} active hunts. {warning}"
        
        return summaries
    
    def format_for_neo_prompt(self) -> str:
        """
        Format the war room into NEO's thinking prompt.
        This is the key output - everything NEO needs to see.
        """
        context = self.get_full_context()
        positions = context["positions"]
        intel = context["intel"]
        battlefield = context["battlefield"]
        
        # Extract intel summaries
        intel_summaries = self.extract_intel_summary(intel)
        
        # Format positions
        if positions["available"] and positions["data"]:
            positions_text = json.dumps(positions["data"], indent=2)
            total_pnl = sum(p.get("pnl_dollars", 0) for p in positions["data"])
            position_summary = f"{len(positions['data'])} positions, ${total_pnl:+,.2f} P&L"
        else:
            positions_text = "No positions data available"
            position_summary = "No open positions"
        
        prompt_addition = f"""
═══════════════════════════════════════════════════════════════
WAR ROOM BRIEFING
═══════════════════════════════════════════════════════════════

📊 PORTFOLIO STATUS: {position_summary}

📡 INTEL FROM YOUR BOTS:
────────────────────────
🔎 ALPHA (Scanner): {intel_summaries['alpha']}
📰 BRAVO (News): {intel_summaries['bravo']}
📈 CHARLIE (Analyst): {intel_summaries['charlie']}
⚠️ DELTA (Threats): {intel_summaries['delta']}

🎯 CURRENT POSITIONS & WHY THEY EXIST:
────────────────────────────────────────
{positions_text}

📋 BATTLEFIELD SUMMARY:
────────────────────────────────────────
{battlefield}

═══════════════════════════════════════════════════════════════
Based on this complete picture, make your decision.
Remember: You have FULL battlefield awareness.
═══════════════════════════════════════════════════════════════
"""
        return prompt_addition
    
    def get_threat_level(self) -> str:
        """Get current threat level from DELTA."""
        intel = self.get_intel_report()
        delta = intel.get("data", {}).get("intel", {}).get("DELTA Threat", {}).get("intel", {})
        return delta.get("threat_level", "UNKNOWN")
    
    def get_news_risk(self) -> str:
        """Get current news risk from BRAVO."""
        intel = self.get_intel_report()
        bravo = intel.get("data", {}).get("intel", {}).get("BRAVO News", {}).get("intel", {})
        return bravo.get("news_risk", "UNKNOWN")
    
    def should_trade(self) -> Dict[str, Any]:
        """Quick check if conditions allow trading."""
        threat = self.get_threat_level()
        news = self.get_news_risk()
        
        # Don't trade in high-risk conditions
        if threat == "RED":
            return {
                "can_trade": False,
                "reason": "DELTA reports RED threat level - active MM manipulation"
            }
        
        if news == "HIGH":
            return {
                "can_trade": False,
                "reason": "BRAVO reports HIGH news risk - avoid until clear"
            }
        
        return {
            "can_trade": True,
            "reason": f"Conditions OK (Threat: {threat}, News: {news})"
        }


def test_war_room():
    """Test the war room."""
    print("=" * 60)
    print("NEO'S WAR ROOM TEST")
    print("=" * 60)
    
    war_room = WarRoom()
    
    print("\n📊 Full Context:")
    context = war_room.get_full_context()
    print(f"  Positions available: {context['positions']['available']}")
    print(f"  Intel available: {context['intel']['available']}")
    
    print("\n🎯 Should Trade Check:")
    trade_check = war_room.should_trade()
    print(f"  Can trade: {trade_check['can_trade']}")
    print(f"  Reason: {trade_check['reason']}")
    
    print("\n📋 War Room Briefing for NEO:")
    print(war_room.format_for_neo_prompt())
    
    print("=" * 60)


if __name__ == "__main__":
    test_war_room()

#!/usr/bin/env python3
"""
MQL5 Intel Reader for NEO
Reads MQL5 top trader intel and provides it to NEO's decision loop.

NO RANDOM DATA - All data from /tmp/neo_intel.json (scraped from MQL5.com)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


class MQL5Reader:
    """
    Reads MQL5 intel for NEO's decision making.
    """
    
    INTEL_FILE = Path("/tmp/neo_intel.json")
    MAX_AGE_MINUTES = 30  # Consider data stale after this
    
    def __init__(self):
        self._cache = None
        self._cache_time = None
    
    def get_intel(self) -> Optional[Dict]:
        """Get current MQL5 intel, returns None if unavailable or stale."""
        if not self.INTEL_FILE.exists():
            return None
        
        try:
            with open(self.INTEL_FILE) as f:
                data = json.load(f)
            
            # Check freshness
            timestamp = data.get("timestamp")
            if timestamp:
                data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age = datetime.utcnow() - data_time.replace(tzinfo=None)
                if age > timedelta(minutes=self.MAX_AGE_MINUTES):
                    return None
            
            return data
            
        except Exception as e:
            print(f"[MQL5Reader] Error loading intel: {e}")
            return None
    
    def get_consensus_signals(self) -> List[Dict]:
        """Get consensus signals if available."""
        intel = self.get_intel()
        if not intel:
            return []
        return intel.get("consensus_signals", [])
    
    def get_top_signals(self) -> List[Dict]:
        """Get top performing signals."""
        intel = self.get_intel()
        if not intel:
            return []
        return intel.get("top_signals", [])
    
    def get_confidence_boost(self, symbol: str, direction: str) -> int:
        """
        Get confidence boost if there's consensus for this trade.
        Returns 0 if no consensus, or boost amount (typically +15) if consensus.
        """
        for signal in self.get_consensus_signals():
            if signal.get("symbol") == symbol and signal.get("direction") == direction:
                # Return the boost from the consensus confidence - base confidence
                return max(0, signal.get("confidence", 60) - 60)
        return 0
    
    def has_consensus(self, symbol: str, direction: str) -> bool:
        """Check if there's consensus for this symbol/direction."""
        return self.get_confidence_boost(symbol, direction) > 0
    
    def format_for_neo(self) -> str:
        """Format MQL5 intel as text for NEO's prompt."""
        intel = self.get_intel()
        
        if not intel:
            return "MQL5 Intel: No recent data available"
        
        lines = [
            "=" * 50,
            "📊 MQL5 TOP TRADER INTEL",
            f"Updated: {intel.get('timestamp', 'Unknown')[:19]}",
            "=" * 50
        ]
        
        # Consensus signals
        consensus = intel.get("consensus_signals", [])
        if consensus:
            lines.append("\n🎯 CONSENSUS SIGNALS (multiple top traders agree):")
            for sig in consensus:
                lines.append(f"   {sig['symbol']} {sig['direction']}")
                lines.append(f"   → {sig.get('trader_count', 0)} traders, {sig.get('confidence', 0)}% confidence")
                lines.append(f"   → Avg growth: {sig.get('avg_growth_pct', 0)}%")
        else:
            lines.append("\n📊 No consensus signals (top traders not aligned)")
        
        # Top signals
        top = intel.get("top_signals", [])
        if top:
            lines.append("\n📈 TOP PERFORMING SIGNALS:")
            for s in top[:5]:
                lines.append(f"   • {s.get('name', 'Unknown')}: {s.get('growth_pct', 0)}% growth")
        
        # Summary
        summary = intel.get("summary", {})
        lines.append(f"\n📊 {summary.get('total_signals_tracked', 0)} signals tracked")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def test_reader():
    """Test the MQL5 reader."""
    reader = MQL5Reader()
    
    print("=" * 60)
    print("MQL5 INTEL READER TEST")
    print("=" * 60)
    
    intel = reader.get_intel()
    if intel:
        print("✅ Intel file found")
        print(f"   Timestamp: {intel.get('timestamp')}")
        print(f"   Consensus: {len(intel.get('consensus_signals', []))}")
        print(f"   Top signals: {len(intel.get('top_signals', []))}")
    else:
        print("⚠️ No recent intel available")
    
    print("\nFormatted for NEO:")
    print(reader.format_for_neo())
    
    # Test consensus check
    print("\nConsensus checks:")
    for symbol in ["XAUUSD", "EURUSD", "GBPUSD"]:
        for direction in ["BUY", "SELL"]:
            boost = reader.get_confidence_boost(symbol, direction)
            if boost > 0:
                print(f"   {symbol} {direction}: +{boost}% confidence boost!")


if __name__ == "__main__":
    test_reader()

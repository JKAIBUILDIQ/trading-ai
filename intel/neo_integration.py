#!/usr/bin/env python3
"""
NEO Integration for MQL5 Intel

This module allows NEO to read and use MQL5 consensus signals
in its decision-making process.

NO RANDOM DATA - All intel from real MQL5 scrapes.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Intel files
NEO_INTEL_FILE = Path("/tmp/neo_intel.json")
MQL5_SIGNALS_FILE = Path(os.path.dirname(__file__)) / "mql5_signals.json"
CONSENSUS_FILE = Path(os.path.dirname(__file__)) / "consensus.json"


class MQL5Intel:
    """
    Provides MQL5 intel to NEO.
    """
    
    def __init__(self):
        self.last_refresh = None
        self._cache = {}
        self.max_age_minutes = 30  # Consider data stale after this
    
    def get_consensus_signals(self) -> List[Dict]:
        """
        Get current consensus signals.
        Returns empty list if no consensus or data is stale.
        """
        data = self._load_intel()
        
        if not data:
            return []
        
        consensus = data.get("consensus_signals", [])
        
        # Check data freshness
        timestamp = data.get("timestamp")
        if timestamp:
            try:
                data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if datetime.utcnow() - data_time.replace(tzinfo=None) > timedelta(minutes=self.max_age_minutes):
                    print(f"[MQL5Intel] Data is stale ({timestamp})")
                    return []
            except:
                pass
        
        return consensus
    
    def get_top_positions(self) -> List[Dict]:
        """
        Get current positions from top traders.
        """
        data = self._load_intel()
        return data.get("top_traders_positions", []) if data else []
    
    def get_confidence_boost(self, symbol: str, direction: str) -> int:
        """
        Get confidence boost if there's consensus for this symbol/direction.
        Returns 0 if no consensus, or the boost amount if consensus exists.
        """
        consensus = self.get_consensus_signals()
        
        for signal in consensus:
            if signal.get("symbol") == symbol and signal.get("direction") == direction:
                return signal.get("confidence", 0) - 60  # Return boost above base
        
        return 0
    
    def check_consensus(self, symbol: str, direction: str) -> Optional[Dict]:
        """
        Check if there's consensus for a specific symbol/direction.
        Returns consensus details or None.
        """
        consensus = self.get_consensus_signals()
        
        for signal in consensus:
            if signal.get("symbol") == symbol and signal.get("direction") == direction:
                return signal
        
        return None
    
    def format_for_neo(self) -> str:
        """
        Format intel as a string for NEO's prompt.
        """
        data = self._load_intel()
        
        if not data:
            return "MQL5 Intel: No data available"
        
        lines = [
            "=" * 50,
            "MQL5 TOP TRADER INTEL",
            f"Updated: {data.get('timestamp', 'Unknown')}",
            "=" * 50
        ]
        
        # Consensus signals
        consensus = data.get("consensus_signals", [])
        if consensus:
            lines.append("\n🎯 CONSENSUS SIGNALS (Multiple top traders agree):")
            for sig in consensus:
                lines.append(f"  {sig['symbol']} {sig['direction']}")
                lines.append(f"    Traders: {', '.join(sig.get('traders', []))}")
                lines.append(f"    Confidence: {sig['confidence']}%")
                lines.append(f"    Avg Growth: {sig.get('avg_growth_pct', 0)}%")
                lines.append("")
        else:
            lines.append("\n📊 No consensus signals (top traders not aligned)")
        
        # Top positions
        positions = data.get("top_traders_positions", [])
        if positions:
            lines.append("\n📈 TOP TRADER POSITIONS:")
            by_symbol = {}
            for pos in positions:
                sym = pos.get('symbol', 'Unknown')
                if sym not in by_symbol:
                    by_symbol[sym] = {'BUY': [], 'SELL': []}
                direction = pos.get('direction', 'BUY')
                by_symbol[sym][direction].append(pos.get('trader', 'Unknown'))
            
            for symbol, dirs in by_symbol.items():
                if dirs['BUY'] or dirs['SELL']:
                    lines.append(f"  {symbol}:")
                    if dirs['BUY']:
                        lines.append(f"    BUY: {', '.join(dirs['BUY'][:3])}")
                    if dirs['SELL']:
                        lines.append(f"    SELL: {', '.join(dirs['SELL'][:3])}")
        
        # Summary
        summary = data.get("summary", {})
        lines.append(f"\n📊 Summary: {summary.get('total_signals_tracked', 0)} signals tracked")
        if summary.get('confidence_boost', 0) > 0:
            lines.append(f"⚡ CONFIDENCE BOOST ACTIVE: +{summary['confidence_boost']}%")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def _load_intel(self) -> Optional[Dict]:
        """Load intel from file."""
        # Try NEO intel file first
        if NEO_INTEL_FILE.exists():
            try:
                with open(NEO_INTEL_FILE) as f:
                    return json.load(f)
            except:
                pass
        
        # Fallback to consensus file
        if CONSENSUS_FILE.exists():
            try:
                with open(CONSENSUS_FILE) as f:
                    return json.load(f)
            except:
                pass
        
        return None


def test_integration():
    """Test the NEO integration."""
    intel = MQL5Intel()
    
    print("=" * 60)
    print("MQL5 INTEL - NEO INTEGRATION TEST")
    print("=" * 60)
    
    # Get consensus
    consensus = intel.get_consensus_signals()
    print(f"\nConsensus signals: {len(consensus)}")
    for sig in consensus:
        print(f"  - {sig['symbol']} {sig['direction']} ({sig['confidence']}%)")
    
    # Get positions
    positions = intel.get_top_positions()
    print(f"\nTop trader positions: {len(positions)}")
    for pos in positions[:5]:
        print(f"  - {pos['trader']}: {pos['symbol']} {pos['direction']}")
    
    # Format for NEO
    print("\n" + "=" * 60)
    print("FORMATTED FOR NEO:")
    print("=" * 60)
    print(intel.format_for_neo())


if __name__ == "__main__":
    test_integration()

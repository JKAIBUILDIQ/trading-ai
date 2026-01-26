#!/usr/bin/env python3
"""
Gap Detector - Real-time gap detection and fill probability calculator
Part of NEO Trading System

Research-Backed Fill Rates (365-day analysis):
- XAUUSD: 77.9% (DOWN gaps: 89.8%, UP gaps: 71.4%)
- USDJPY: 80.6% (DOWN gaps: 82.7%, UP gaps: 78.5%)
- AUDUSD: 78.5% (DOWN gaps: 85.7%, UP gaps: 72.2%)
- EURUSD: 73.1% (DOWN gaps: 81.3%, UP gaps: 65.3%)
- GBPUSD: 71.9% (DOWN gaps: 78.7%, UP gaps: 65.7%)

Key Finding: DOWN gaps fill more reliably than UP gaps!
Strategy: Favor buying down gaps over selling up gaps
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
import os

class GapDetector:
    """
    Real-time gap detection and trading signal generator
    """
    
    # Yahoo Finance symbols
    YAHOO_SYMBOLS = {
        'XAUUSD': 'GC=F',
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X',
        'NZDUSD': 'NZDUSD=X',
        'USDCHF': 'USDCHF=X',
        'USDCAD': 'USDCAD=X',
    }
    
    # Minimum gap sizes to be tradeable
    MIN_GAP_SIZE = {
        'XAUUSD': 10.0,    # $10 for Gold
        'EURUSD': 0.0015,  # 15 pips
        'GBPUSD': 0.0015,  # 15 pips
        'USDJPY': 0.15,    # 15 pips
        'AUDUSD': 0.0010,  # 10 pips
        'NZDUSD': 0.0010,  # 10 pips
        'USDCHF': 0.0010,  # 10 pips
        'USDCAD': 0.0010,  # 10 pips
    }
    
    # Research-backed fill rates
    FILL_RATES = {
        'XAUUSD': {'total': 0.779, 'up': 0.714, 'down': 0.898, 'avg_days': 1.4},
        'EURUSD': {'total': 0.731, 'up': 0.653, 'down': 0.813, 'avg_days': 1.8},
        'GBPUSD': {'total': 0.719, 'up': 0.657, 'down': 0.787, 'avg_days': 1.6},
        'USDJPY': {'total': 0.806, 'up': 0.785, 'down': 0.827, 'avg_days': 1.7},
        'AUDUSD': {'total': 0.785, 'up': 0.722, 'down': 0.857, 'avg_days': 2.1},
    }
    
    def __init__(self):
        self.active_gaps: Dict[str, dict] = {}
        self.gap_history: List[dict] = []
        self.state_file = os.path.join(os.path.dirname(__file__), 'gap_state.json')
        self._load_state()
    
    def _load_state(self):
        """Load persisted gap state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.active_gaps = data.get('active_gaps', {})
                    self.gap_history = data.get('history', [])[-50:]  # Keep last 50
            except:
                pass
    
    def _save_state(self):
        """Persist gap state"""
        with open(self.state_file, 'w') as f:
            json.dump({
                'active_gaps': self.active_gaps,
                'history': self.gap_history[-50:],
                'updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def fetch_price_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Fetch recent price data"""
        yahoo_symbol = self.YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")
        end = datetime.now()
        start = end - timedelta(days=days)
        
        try:
            df = yf.download(yahoo_symbol, start=start, end=end, interval='1d', progress=False)
            if len(df) > 0:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return None
    
    def detect_gap(self, symbol: str) -> Dict:
        """
        Detect if there's a tradeable gap for the symbol
        
        Returns:
            dict with gap info or {'has_gap': False}
        """
        df = self.fetch_price_data(symbol, days=5)
        if df is None or len(df) < 2:
            return {'has_gap': False, 'error': 'Insufficient data'}
        
        # Get yesterday's close and today's open
        prev_close = float(df.iloc[-2]['Close'])
        today_open = float(df.iloc[-1]['Open'])
        current_price = float(df.iloc[-1]['Close'])
        today_high = float(df.iloc[-1]['High'])
        today_low = float(df.iloc[-1]['Low'])
        
        gap_size = today_open - prev_close
        min_gap = self.MIN_GAP_SIZE.get(symbol, 0.001)
        
        if abs(gap_size) < min_gap:
            return {
                'has_gap': False,
                'symbol': symbol,
                'gap_size': abs(gap_size),
                'min_required': min_gap,
                'reason': 'Gap too small'
            }
        
        gap_direction = "UP" if gap_size > 0 else "DOWN"
        gap_percent = abs(gap_size / prev_close) * 100
        
        # Check if gap already filled today
        gap_filled = False
        fill_progress = 0.0
        
        if gap_direction == "UP":
            gap_filled = today_low <= prev_close
            fill_progress = max(0, min(1, (today_open - current_price) / abs(gap_size)))
        else:
            gap_filled = today_high >= prev_close
            fill_progress = max(0, min(1, (current_price - today_open) / abs(gap_size)))
        
        # Calculate fill probability
        fill_prob = self._calculate_fill_probability(symbol, gap_direction, gap_percent)
        
        # Generate trading recommendation
        fade_action = "SELL" if gap_direction == "UP" else "BUY"
        
        # Calculate entry, SL, TP
        entry = current_price
        target = prev_close  # Full gap fill
        
        if fade_action == "SELL":
            sl = today_open + (abs(gap_size) * 0.5)  # 50% extension beyond gap
            tp = entry - ((entry - target) * 0.8)    # 80% of gap fill
            risk = sl - entry
            reward = entry - tp
        else:
            sl = today_open - (abs(gap_size) * 0.5)
            tp = entry + ((target - entry) * 0.8)
            risk = entry - sl
            reward = tp - entry
        
        risk_reward = reward / risk if risk > 0 else 0
        
        # Confidence score
        confidence = self._calculate_confidence(
            fill_prob, gap_percent, fill_progress, gap_direction
        )
        
        gap_info = {
            'has_gap': True,
            'symbol': symbol,
            'direction': gap_direction,
            'gap_size': abs(gap_size),
            'gap_percent': gap_percent,
            'prev_close': prev_close,
            'today_open': today_open,
            'current_price': current_price,
            'fill_target': prev_close,
            'is_filled': gap_filled,
            'fill_progress': fill_progress * 100,
            'fill_probability': fill_prob * 100,
            'avg_fill_days': self.FILL_RATES.get(symbol, {}).get('avg_days', 2.0),
            
            # Trading Signal
            'trade_action': fade_action if not gap_filled else 'WAIT',
            'entry': entry,
            'stop_loss': sl,
            'take_profit': tp,
            'risk_reward': risk_reward,
            'confidence': confidence,
            
            # Classification
            'gap_type': self._classify_gap(gap_percent),
            'tradeable': not gap_filled and confidence >= 60,
            'detected_at': datetime.now().isoformat(),
        }
        
        # Store active gap
        if not gap_filled:
            self.active_gaps[symbol] = gap_info
            self._save_state()
        
        return gap_info
    
    def _calculate_fill_probability(self, symbol: str, direction: str, gap_percent: float) -> float:
        """Calculate probability of gap fill based on research data"""
        rates = self.FILL_RATES.get(symbol, {'total': 0.75, 'up': 0.70, 'down': 0.80})
        
        # Base probability from direction
        if direction == "UP":
            base_prob = rates['up']
        else:
            base_prob = rates['down']
        
        # Adjust for gap size
        if gap_percent < 0.3:
            # Small gaps fill more often
            base_prob += 0.05
        elif gap_percent > 1.5:
            # Large gaps may be breakaway gaps - lower fill rate
            base_prob -= 0.15
        elif gap_percent > 1.0:
            base_prob -= 0.08
        
        return min(0.95, max(0.40, base_prob))
    
    def _calculate_confidence(self, fill_prob: float, gap_percent: float, 
                             fill_progress: float, direction: str) -> int:
        """Calculate confidence score 0-100"""
        confidence = fill_prob * 100
        
        # Adjust for direction (DOWN gaps are more reliable)
        if direction == "DOWN":
            confidence += 5
        
        # Adjust for gap size (optimal range 0.3% - 1.0%)
        if 0.3 <= gap_percent <= 1.0:
            confidence += 5
        elif gap_percent > 1.5:
            confidence -= 10
        
        # Adjust for fill progress (if already started filling, more confident)
        if fill_progress > 0.1:
            confidence += 8  # Confirmation of fill beginning
        
        return int(min(95, max(30, confidence)))
    
    def _classify_gap(self, gap_percent: float) -> str:
        """Classify the gap type"""
        if gap_percent < 0.3:
            return "COMMON"      # Likely to fill, but may be noise
        elif gap_percent < 1.0:
            return "STANDARD"    # Good trading gap
        elif gap_percent < 2.0:
            return "LARGE"       # May be breakaway
        else:
            return "BREAKAWAY"   # Likely trend continuation - don't fade
    
    def scan_all_gaps(self) -> Dict[str, dict]:
        """Scan all tracked symbols for gaps"""
        results = {}
        for symbol in self.YAHOO_SYMBOLS.keys():
            gap = self.detect_gap(symbol)
            results[symbol] = gap
        return results
    
    def get_tradeable_gaps(self) -> List[dict]:
        """Get all currently tradeable gaps sorted by confidence"""
        gaps = []
        for symbol in self.YAHOO_SYMBOLS.keys():
            gap = self.detect_gap(symbol)
            if gap.get('tradeable'):
                gaps.append(gap)
        
        return sorted(gaps, key=lambda x: x['confidence'], reverse=True)
    
    def get_gap_status(self, symbol: str = None) -> dict:
        """Get status of gaps (for API)"""
        if symbol:
            return self.detect_gap(symbol)
        
        all_gaps = self.scan_all_gaps()
        tradeable = [g for g in all_gaps.values() if g.get('tradeable')]
        active = [g for g in all_gaps.values() if g.get('has_gap') and not g.get('is_filled')]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_scanned': len(all_gaps),
            'gaps_found': len([g for g in all_gaps.values() if g.get('has_gap')]),
            'tradeable_count': len(tradeable),
            'active_gaps': active,
            'tradeable_gaps': tradeable,
            'best_trade': tradeable[0] if tradeable else None,
            'all_gaps': all_gaps,
        }


def main():
    """Test the gap detector"""
    detector = GapDetector()
    
    print("\n" + "="*60)
    print("🔍 GAP DETECTOR - SCANNING ALL PAIRS")
    print("="*60)
    
    status = detector.get_gap_status()
    
    print(f"\n📊 SCAN RESULTS:")
    print(f"   Pairs Scanned: {status['total_scanned']}")
    print(f"   Gaps Found: {status['gaps_found']}")
    print(f"   Tradeable: {status['tradeable_count']}")
    
    if status['tradeable_gaps']:
        print(f"\n🎯 TRADEABLE GAPS:")
        for gap in status['tradeable_gaps']:
            print(f"\n   {gap['symbol']} - {gap['direction']} GAP")
            print(f"   Gap Size: {gap['gap_percent']:.2f}%")
            print(f"   Fill Probability: {gap['fill_probability']:.1f}%")
            print(f"   Trade: {gap['trade_action']} @ {gap['entry']:.5f}")
            print(f"   TP: {gap['take_profit']:.5f} | SL: {gap['stop_loss']:.5f}")
            print(f"   R:R: {gap['risk_reward']:.2f}")
            print(f"   Confidence: {gap['confidence']}%")
    else:
        print("\n   No tradeable gaps at this time")
    
    # Show all gaps status
    print(f"\n📋 ALL GAP STATUS:")
    for symbol, gap in status['all_gaps'].items():
        if gap.get('has_gap'):
            status_icon = "✅" if gap.get('is_filled') else "🔄"
            print(f"   {status_icon} {symbol}: {gap['direction']} {gap['gap_percent']:.2f}% "
                  f"(Fill: {gap['fill_progress']:.0f}%)")
        else:
            print(f"   ⚪ {symbol}: No gap")
    
    return status


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gap Fill Trading Research
Analyzes historical gap patterns for Gold and Forex pairs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class GapResearcher:
    """Research historical gap fill patterns"""
    
    SYMBOLS = {
        'XAUUSD': 'GC=F',      # Gold Futures
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X',
    }
    
    # Minimum gap sizes (in price units)
    MIN_GAPS = {
        'XAUUSD': 10.0,   # $10 for Gold
        'EURUSD': 0.0015, # 15 pips
        'GBPUSD': 0.0015, # 15 pips
        'USDJPY': 0.15,   # 15 pips
        'AUDUSD': 0.0015, # 15 pips
    }
    
    def __init__(self, lookback_days=365):
        self.lookback = lookback_days
        self.results = {}
        
    def fetch_data(self, symbol):
        """Fetch historical daily data"""
        yahoo_symbol = self.SYMBOLS.get(symbol, symbol)
        end = datetime.now()
        start = end - timedelta(days=self.lookback)
        
        try:
            df = yf.download(yahoo_symbol, start=start, end=end, interval='1d', progress=False)
            if len(df) > 0:
                df = df.reset_index()
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return None
    
    def detect_gaps(self, df, symbol):
        """Detect all gaps in the data"""
        gaps = []
        min_gap = self.MIN_GAPS.get(symbol, 0.001)
        
        for i in range(1, len(df)):
            prev_close = float(df.iloc[i-1]['Close'])
            curr_open = float(df.iloc[i]['Open'])
            curr_high = float(df.iloc[i]['High'])
            curr_low = float(df.iloc[i]['Low'])
            curr_close = float(df.iloc[i]['Close'])
            
            gap_size = curr_open - prev_close
            
            if abs(gap_size) >= min_gap:
                gap_direction = "UP" if gap_size > 0 else "DOWN"
                
                # Check same-day fill
                same_day_fill = False
                if gap_direction == "UP":
                    same_day_fill = curr_low <= prev_close
                else:
                    same_day_fill = curr_high >= prev_close
                
                # Check multi-day fill (look ahead up to 10 days)
                fill_days = None
                for j in range(i, min(i+10, len(df))):
                    future_low = float(df.iloc[j]['Low'])
                    future_high = float(df.iloc[j]['High'])
                    
                    if gap_direction == "UP" and future_low <= prev_close:
                        fill_days = j - i
                        break
                    elif gap_direction == "DOWN" and future_high >= prev_close:
                        fill_days = j - i
                        break
                
                gap_date = df.iloc[i]['Date']
                if hasattr(gap_date, 'strftime'):
                    gap_date = gap_date.strftime('%Y-%m-%d')
                else:
                    gap_date = str(gap_date)[:10]
                
                gaps.append({
                    'date': gap_date,
                    'prev_close': prev_close,
                    'open': curr_open,
                    'gap_size': abs(gap_size),
                    'gap_percent': abs(gap_size / prev_close) * 100,
                    'direction': gap_direction,
                    'same_day_fill': same_day_fill,
                    'fill_days': fill_days,
                    'filled_within_week': fill_days is not None and fill_days <= 5,
                })
        
        return gaps
    
    def analyze_symbol(self, symbol):
        """Complete analysis for one symbol"""
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING {symbol}")
        print(f"{'='*60}")
        
        df = self.fetch_data(symbol)
        if df is None or len(df) < 50:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        gaps = self.detect_gaps(df, symbol)
        
        if len(gaps) == 0:
            print(f"No significant gaps found for {symbol}")
            return None
        
        # Calculate statistics
        total_gaps = len(gaps)
        same_day_fills = sum(1 for g in gaps if g['same_day_fill'])
        week_fills = sum(1 for g in gaps if g['filled_within_week'])
        any_fills = sum(1 for g in gaps if g['fill_days'] is not None)
        
        up_gaps = [g for g in gaps if g['direction'] == 'UP']
        down_gaps = [g for g in gaps if g['direction'] == 'DOWN']
        
        up_fill_rate = sum(1 for g in up_gaps if g['fill_days'] is not None) / len(up_gaps) * 100 if up_gaps else 0
        down_fill_rate = sum(1 for g in down_gaps if g['fill_days'] is not None) / len(down_gaps) * 100 if down_gaps else 0
        
        avg_gap_size = np.mean([g['gap_size'] for g in gaps])
        avg_gap_pct = np.mean([g['gap_percent'] for g in gaps])
        avg_fill_time = np.mean([g['fill_days'] for g in gaps if g['fill_days'] is not None]) if any_fills > 0 else None
        
        results = {
            'symbol': symbol,
            'period_days': self.lookback,
            'total_gaps': total_gaps,
            'gaps_per_month': total_gaps / (self.lookback / 30),
            'same_day_fill_rate': same_day_fills / total_gaps * 100,
            'week_fill_rate': week_fills / total_gaps * 100,
            'total_fill_rate': any_fills / total_gaps * 100,
            'up_gap_fill_rate': up_fill_rate,
            'down_gap_fill_rate': down_fill_rate,
            'avg_gap_size': float(avg_gap_size),
            'avg_gap_percent': float(avg_gap_pct),
            'avg_fill_time_days': float(avg_fill_time) if avg_fill_time else None,
            'up_gaps': len(up_gaps),
            'down_gaps': len(down_gaps),
            'recent_gaps': gaps[-5:] if len(gaps) >= 5 else gaps,  # Last 5 gaps
        }
        
        # Print results
        print(f"\n📈 RESULTS FOR {symbol}:")
        print(f"   Total Gaps Found: {total_gaps}")
        print(f"   Gaps/Month: {results['gaps_per_month']:.1f}")
        print(f"   ")
        print(f"   📊 FILL RATES:")
        print(f"   Same Day Fill:    {results['same_day_fill_rate']:.1f}%")
        print(f"   Within Week Fill: {results['week_fill_rate']:.1f}%")
        print(f"   Total Fill Rate:  {results['total_fill_rate']:.1f}%")
        print(f"   ")
        print(f"   📊 BY DIRECTION:")
        print(f"   UP Gaps:   {len(up_gaps)} (Fill rate: {up_fill_rate:.1f}%)")
        print(f"   DOWN Gaps: {len(down_gaps)} (Fill rate: {down_fill_rate:.1f}%)")
        print(f"   ")
        print(f"   📊 GAP SIZE:")
        print(f"   Average Gap Size: {avg_gap_size:.4f}")
        print(f"   Average Gap %:    {avg_gap_pct:.2f}%")
        if avg_fill_time:
            print(f"   Avg Fill Time:    {avg_fill_time:.1f} days")
        
        self.results[symbol] = results
        return results
    
    def run_full_analysis(self):
        """Run analysis on all symbols"""
        print("\n" + "="*60)
        print("🔬 GAP FILL TRADING RESEARCH")
        print(f"   Lookback: {self.lookback} days")
        print("="*60)
        
        for symbol in self.SYMBOLS.keys():
            self.analyze_symbol(symbol)
        
        # Summary
        print("\n" + "="*60)
        print("📋 SUMMARY - GAP FILL RATES")
        print("="*60)
        print(f"{'Symbol':<10} {'Gaps':<8} {'Same Day':<12} {'Within Week':<12} {'Total Fill':<12}")
        print("-"*60)
        
        for symbol, res in self.results.items():
            print(f"{symbol:<10} {res['total_gaps']:<8} {res['same_day_fill_rate']:.1f}%{'':<6} {res['week_fill_rate']:.1f}%{'':<6} {res['total_fill_rate']:.1f}%")
        
        return self.results


if __name__ == "__main__":
    researcher = GapResearcher(lookback_days=365)
    results = researcher.run_full_analysis()
    
    # Save results
    with open('/home/jbot/trading_ai/neo/gap_trading/research_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to research_results.json")

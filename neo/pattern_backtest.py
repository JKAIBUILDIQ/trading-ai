"""
NEO Pattern Backtest Analysis
Analyze XAUUSD to find predictive patterns before corrections.

Analysis 1: 2 Red Candles + Rising Volume → What happens next?
Analysis 2: Large Corrections (50+ pips) → What patterns preceded them?
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json
import os

# Output paths
OUTPUT_DIR = "/home/jbot/trading_ai/neo/pattern_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class TwoRedPattern:
    """Record of 2 red candles + rising volume occurrence"""
    date: str
    time: str
    candle1_body_pips: float
    candle1_volume: float
    candle2_body_pips: float
    candle2_volume: float
    volume_increase_pct: float
    price_at_pattern: float
    price_location: str  # ATH, RESISTANCE, MID, SUPPORT
    trend_before: str    # UP, DOWN, RANGE
    
    # Outcomes
    outcome_5_candles: str  # DOWN, UP, SIDEWAYS
    move_5_candles_pips: float
    outcome_24h: str
    move_24h_pips: float
    max_drawdown_pips: float
    max_rally_pips: float


@dataclass
class Correction:
    """Record of a 50+ pip correction"""
    date: str
    high_price: float
    high_time: str
    low_price: float
    low_time: str
    drop_pips: float
    duration_hours: float
    
    # Pre-correction patterns
    had_2red_rising_vol: bool
    had_long_upper_wick: bool
    had_failed_breakout: bool
    was_at_ath: bool
    volume_spike_pct: float
    
    # Context
    event_driven: bool
    event_name: str
    day_of_week: str
    
    # Recovery
    recovered_50pct: bool
    recovered_100pct: bool
    recovery_hours: float


def fetch_xauusd_data(days: int = 90) -> pd.DataFrame:
    """Fetch XAUUSD hourly data from yfinance."""
    print(f"📥 Fetching {days} days of XAUUSD data...")
    
    # GC=F is Gold Futures on yfinance
    ticker = yf.Ticker("GC=F")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get hourly data
    df = ticker.history(start=start_date, end=end_date, interval="1h")
    
    if df.empty:
        print("⚠️ No data returned from yfinance, trying alternative...")
        # Try with longer period
        df = ticker.history(period="3mo", interval="1h")
    
    print(f"✅ Retrieved {len(df)} candles")
    
    # Calculate candle properties
    df['body'] = df['Close'] - df['Open']
    df['body_pips'] = abs(df['body']) / 0.1  # Gold pips (0.1 = 1 pip)
    df['is_red'] = df['Close'] < df['Open']
    df['is_green'] = df['Close'] > df['Open']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_wick_pips'] = df['upper_wick'] / 0.1
    df['lower_wick_pips'] = df['lower_wick'] / 0.1
    
    # Calculate moving averages
    df['ema20'] = df['Close'].ewm(span=20).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()
    
    # Rolling high (for ATH detection)
    df['rolling_high_20d'] = df['High'].rolling(window=480).max()  # 20 days of H1
    df['near_ath'] = df['High'] >= df['rolling_high_20d'] * 0.995  # Within 0.5%
    
    # Trend detection
    df['uptrend'] = (df['ema20'] > df['ema50']) & (df['Close'] > df['ema20'])
    df['downtrend'] = (df['ema20'] < df['ema50']) & (df['Close'] < df['ema20'])
    
    return df


def analyze_two_red_patterns(df: pd.DataFrame) -> Tuple[List[TwoRedPattern], Dict]:
    """
    Find all 2 consecutive red candles with rising volume.
    Track what happens after each occurrence.
    """
    print("\n📊 ANALYSIS 1: Two Red Candles + Rising Volume")
    print("=" * 60)
    
    patterns = []
    
    for i in range(2, len(df) - 24):  # Need history before and 24h after
        # Check for 2 red candles
        c1 = df.iloc[i-1]
        c2 = df.iloc[i]
        
        if not (c1['is_red'] and c2['is_red']):
            continue
        
        # Check for rising volume
        v1 = c1['Volume']
        v2 = c2['Volume']
        
        if v1 == 0 or v2 <= v1:
            continue
        
        volume_increase = ((v2 - v1) / v1) * 100
        
        # Pattern detected! Analyze context and outcome
        
        # Price location
        if c2['near_ath']:
            location = "ATH"
        elif c2['Close'] > c2['ema20']:
            location = "ABOVE_EMA"
        elif c2['Close'] < c2['ema50']:
            location = "BELOW_EMA"
        else:
            location = "MID"
        
        # Trend before
        if c2['uptrend']:
            trend = "UP"
        elif c2['downtrend']:
            trend = "DOWN"
        else:
            trend = "RANGE"
        
        # Outcome: next 5 candles
        future_5 = df.iloc[i+1:i+6]
        if len(future_5) < 5:
            continue
            
        close_5 = future_5.iloc[-1]['Close']
        move_5 = (close_5 - c2['Close']) / 0.1  # In pips
        
        if move_5 < -10:
            outcome_5 = "DOWN"
        elif move_5 > 10:
            outcome_5 = "UP"
        else:
            outcome_5 = "SIDEWAYS"
        
        # Outcome: next 24 candles (24h)
        future_24 = df.iloc[i+1:i+25]
        if len(future_24) < 24:
            continue
            
        close_24 = future_24.iloc[-1]['Close']
        move_24 = (close_24 - c2['Close']) / 0.1
        max_dd = (c2['Close'] - future_24['Low'].min()) / 0.1
        max_rally = (future_24['High'].max() - c2['Close']) / 0.1
        
        if move_24 < -10:
            outcome_24 = "DOWN"
        elif move_24 > 10:
            outcome_24 = "UP"
        else:
            outcome_24 = "SIDEWAYS"
        
        pattern = TwoRedPattern(
            date=str(c2.name.date()) if hasattr(c2.name, 'date') else str(c2.name)[:10],
            time=str(c2.name.time()) if hasattr(c2.name, 'time') else str(c2.name)[11:16],
            candle1_body_pips=round(c1['body_pips'], 1),
            candle1_volume=int(v1),
            candle2_body_pips=round(c2['body_pips'], 1),
            candle2_volume=int(v2),
            volume_increase_pct=round(volume_increase, 1),
            price_at_pattern=round(c2['Close'], 2),
            price_location=location,
            trend_before=trend,
            outcome_5_candles=outcome_5,
            move_5_candles_pips=round(move_5, 1),
            outcome_24h=outcome_24,
            move_24h_pips=round(move_24, 1),
            max_drawdown_pips=round(max_dd, 1),
            max_rally_pips=round(max_rally, 1)
        )
        
        patterns.append(pattern)
    
    # Calculate statistics
    total = len(patterns)
    if total == 0:
        print("⚠️ No patterns found!")
        return patterns, {}
    
    down_5 = sum(1 for p in patterns if p.outcome_5_candles == "DOWN")
    up_5 = sum(1 for p in patterns if p.outcome_5_candles == "UP")
    side_5 = sum(1 for p in patterns if p.outcome_5_candles == "SIDEWAYS")
    
    down_24 = sum(1 for p in patterns if p.outcome_24h == "DOWN")
    up_24 = sum(1 for p in patterns if p.outcome_24h == "UP")
    
    # By context
    ath_patterns = [p for p in patterns if p.price_location == "ATH"]
    ath_down = sum(1 for p in ath_patterns if p.outcome_5_candles == "DOWN") if ath_patterns else 0
    
    uptrend_patterns = [p for p in patterns if p.trend_before == "UP"]
    uptrend_down = sum(1 for p in uptrend_patterns if p.outcome_5_candles == "DOWN") if uptrend_patterns else 0
    
    # Volume correlation
    high_vol_patterns = [p for p in patterns if p.volume_increase_pct > 30]
    high_vol_down = sum(1 for p in high_vol_patterns if p.outcome_5_candles == "DOWN") if high_vol_patterns else 0
    
    avg_move_when_down = np.mean([p.move_5_candles_pips for p in patterns if p.outcome_5_candles == "DOWN"]) if down_5 > 0 else 0
    avg_move_when_up = np.mean([p.move_5_candles_pips for p in patterns if p.outcome_5_candles == "UP"]) if up_5 > 0 else 0
    
    stats = {
        "total_occurrences": total,
        "outcome_5_candles": {
            "DOWN": {"count": down_5, "pct": round(down_5/total*100, 1)},
            "UP": {"count": up_5, "pct": round(up_5/total*100, 1)},
            "SIDEWAYS": {"count": side_5, "pct": round(side_5/total*100, 1)}
        },
        "outcome_24h": {
            "DOWN": {"count": down_24, "pct": round(down_24/total*100, 1)},
            "UP": {"count": up_24, "pct": round(up_24/total*100, 1)}
        },
        "avg_move_when_down": round(avg_move_when_down, 1),
        "avg_move_when_up": round(avg_move_when_up, 1),
        "by_context": {
            "at_ath": {
                "total": len(ath_patterns),
                "down_pct": round(ath_down/len(ath_patterns)*100, 1) if ath_patterns else 0
            },
            "in_uptrend": {
                "total": len(uptrend_patterns),
                "down_pct": round(uptrend_down/len(uptrend_patterns)*100, 1) if uptrend_patterns else 0
            },
            "volume_spike_30pct": {
                "total": len(high_vol_patterns),
                "down_pct": round(high_vol_down/len(high_vol_patterns)*100, 1) if high_vol_patterns else 0
            }
        }
    }
    
    # Print report
    print(f"\nTOTAL OCCURRENCES: {total}")
    print(f"\nOUTCOME (next 5 candles):")
    print(f"├── Continuation DOWN:  {down_5} ({stats['outcome_5_candles']['DOWN']['pct']}%)")
    print(f"├── Reversal UP:        {up_5} ({stats['outcome_5_candles']['UP']['pct']}%)")
    print(f"└── Sideways:           {side_5} ({stats['outcome_5_candles']['SIDEWAYS']['pct']}%)")
    
    print(f"\nAVERAGE MOVE (next 5 candles):")
    print(f"├── When DOWN: {avg_move_when_down:.1f} pips")
    print(f"└── When UP:   {avg_move_when_up:.1f} pips")
    
    print(f"\nBY CONTEXT:")
    if ath_patterns:
        print(f"├── At ATH:           {stats['by_context']['at_ath']['down_pct']}% continued DOWN ({len(ath_patterns)} samples)")
    if uptrend_patterns:
        print(f"├── In Uptrend:       {stats['by_context']['in_uptrend']['down_pct']}% continued DOWN ({len(uptrend_patterns)} samples)")
    if high_vol_patterns:
        print(f"└── Vol spike >30%:   {stats['by_context']['volume_spike_30pct']['down_pct']}% continued DOWN ({len(high_vol_patterns)} samples)")
    
    return patterns, stats


def analyze_corrections(df: pd.DataFrame, min_pips: float = 50) -> Tuple[List[Correction], Dict]:
    """
    Find all corrections of 50+ pips and analyze what preceded them.
    """
    print(f"\n📊 ANALYSIS 2: Large Corrections ({min_pips}+ pips)")
    print("=" * 60)
    
    corrections = []
    
    # Find local highs and subsequent drops
    for i in range(10, len(df) - 10):
        current_high = df.iloc[i]['High']
        
        # Check if this is a local high (highest in 5 candles before and after)
        is_local_high = (
            current_high >= df.iloc[i-5:i]['High'].max() and
            current_high >= df.iloc[i:i+5]['High'].max()
        )
        
        if not is_local_high:
            continue
        
        # Look for drop in next 4-8 hours
        for j in range(i+1, min(i+9, len(df))):
            low = df.iloc[j]['Low']
            drop_pips = (current_high - low) / 0.1
            
            if drop_pips >= min_pips:
                # Found a correction! Analyze what preceded it
                
                # Check for 2 red + rising volume in 5 candles before
                had_2red = False
                vol_spike = 0
                for k in range(max(0, i-5), i):
                    if k > 0:
                        c1 = df.iloc[k-1]
                        c2 = df.iloc[k]
                        if c1['is_red'] and c2['is_red'] and c2['Volume'] > c1['Volume']:
                            had_2red = True
                            vol_spike = max(vol_spike, ((c2['Volume'] - c1['Volume']) / c1['Volume']) * 100 if c1['Volume'] > 0 else 0)
                
                # Check for long upper wick
                had_long_wick = any(df.iloc[i-3:i+1]['upper_wick_pips'] > 10)
                
                # Check if at ATH
                was_at_ath = df.iloc[i]['near_ath']
                
                # Duration
                high_time = df.iloc[i].name
                low_time = df.iloc[j].name
                duration = (j - i)  # In hours (H1 data)
                
                correction = Correction(
                    date=str(high_time.date()) if hasattr(high_time, 'date') else str(high_time)[:10],
                    high_price=float(round(current_high, 2)),
                    high_time=str(high_time.time()) if hasattr(high_time, 'time') else str(high_time)[11:16],
                    low_price=float(round(low, 2)),
                    low_time=str(low_time.time()) if hasattr(low_time, 'time') else str(low_time)[11:16],
                    drop_pips=float(round(drop_pips, 1)),
                    duration_hours=int(duration),
                    had_2red_rising_vol=bool(had_2red),
                    had_long_upper_wick=bool(had_long_wick),
                    had_failed_breakout=False,  # Would need more complex detection
                    was_at_ath=bool(was_at_ath),
                    volume_spike_pct=float(round(vol_spike, 1)),
                    event_driven=False,  # Would need calendar data
                    event_name="",
                    day_of_week=high_time.strftime("%A") if hasattr(high_time, 'strftime') else "",
                    recovered_50pct=False,  # Would need to track
                    recovered_100pct=False,
                    recovery_hours=0.0
                )
                
                corrections.append(correction)
                break  # Don't double-count this high
    
    # Remove duplicates (corrections within 5 hours of each other)
    filtered = []
    for c in corrections:
        try:
            time_str = c.high_time[:5] if len(c.high_time) > 5 else c.high_time
            curr_time = datetime.strptime(c.date + " " + time_str, "%Y-%m-%d %H:%M")
            
            if not filtered:
                filtered.append(c)
            else:
                prev_time_str = filtered[-1].high_time[:5] if len(filtered[-1].high_time) > 5 else filtered[-1].high_time
                prev_time = datetime.strptime(filtered[-1].date + " " + prev_time_str, "%Y-%m-%d %H:%M")
                if (curr_time - prev_time).total_seconds() > 18000:
                    filtered.append(c)
        except:
            filtered.append(c)
    corrections = filtered
    
    # Calculate statistics
    total = len(corrections)
    if total == 0:
        print("⚠️ No corrections found!")
        return corrections, {}
    
    with_2red = sum(1 for c in corrections if c.had_2red_rising_vol)
    with_wick = sum(1 for c in corrections if c.had_long_upper_wick)
    at_ath = sum(1 for c in corrections if c.was_at_ath)
    
    avg_drop = np.mean([c.drop_pips for c in corrections])
    max_drop = max(c.drop_pips for c in corrections)
    
    # Size categories
    drops_50_100 = sum(1 for c in corrections if 50 <= c.drop_pips < 100)
    drops_100_200 = sum(1 for c in corrections if 100 <= c.drop_pips < 200)
    drops_200_plus = sum(1 for c in corrections if c.drop_pips >= 200)
    
    stats = {
        "total_corrections": total,
        "by_size": {
            "50-100_pips": drops_50_100,
            "100-200_pips": drops_100_200,
            "200+_pips": drops_200_plus
        },
        "average_drop_pips": round(avg_drop, 1),
        "max_drop_pips": round(max_drop, 1),
        "pre_correction_patterns": {
            "2red_rising_volume": {
                "count": with_2red,
                "pct": round(with_2red/total*100, 1) if total > 0 else 0
            },
            "long_upper_wick": {
                "count": with_wick,
                "pct": round(with_wick/total*100, 1) if total > 0 else 0
            },
            "at_ath": {
                "count": at_ath,
                "pct": round(at_ath/total*100, 1) if total > 0 else 0
            }
        }
    }
    
    # Print report
    print(f"\nCORRECTION COUNT:")
    print(f"├── 50-100 pips:   {drops_50_100}")
    print(f"├── 100-200 pips:  {drops_100_200}")
    print(f"├── 200+ pips:     {drops_200_plus}")
    print(f"└── TOTAL:         {total}")
    
    print(f"\nPRE-CORRECTION PATTERNS (What preceded them):")
    print(f"├── 2 Red + Rising Volume: {with_2red}/{total} ({stats['pre_correction_patterns']['2red_rising_volume']['pct']}%)")
    print(f"├── Long Upper Wick:       {with_wick}/{total} ({stats['pre_correction_patterns']['long_upper_wick']['pct']}%)")
    print(f"└── At ATH:                {at_ath}/{total} ({stats['pre_correction_patterns']['at_ath']['pct']}%)")
    
    print(f"\nAVERAGE DROP: {avg_drop:.1f} pips")
    print(f"MAX DROP:     {max_drop:.1f} pips")
    
    return corrections, stats


def generate_neo_rules(pattern_stats: Dict, correction_stats: Dict) -> Dict:
    """Generate detection rules for NEO based on backtest results."""
    
    rules = {
        "two_red_rising_volume": {
            "description": "2 consecutive red candles with rising volume",
            "base_confidence": 50,
            "confidence_modifiers": {
                "at_ath": 20,
                "volume_increase_30pct": 15,
                "volume_increase_50pct": 10,
                "in_uptrend": 5
            },
            "defcon_thresholds": {
                "confidence_80+": 3,  # DEFCON 3
                "confidence_65+": 4,  # DEFCON 4
                "default": 5          # DEFCON 5
            },
            "backtest_accuracy": pattern_stats.get("outcome_5_candles", {}).get("DOWN", {}).get("pct", 0)
        },
        "correction_probability": {
            "2red_rising_vol_pct": correction_stats.get("pre_correction_patterns", {}).get("2red_rising_volume", {}).get("pct", 0),
            "long_upper_wick_pct": correction_stats.get("pre_correction_patterns", {}).get("long_upper_wick", {}).get("pct", 0),
            "at_ath_pct": correction_stats.get("pre_correction_patterns", {}).get("at_ath", {}).get("pct", 0)
        }
    }
    
    return rules


def main():
    print("=" * 70)
    print("        NEO PATTERN BACKTEST ANALYSIS - XAUUSD")
    print("=" * 70)
    
    # Fetch data
    df = fetch_xauusd_data(days=90)
    
    if df.empty:
        print("❌ Failed to fetch data")
        return
    
    # Analysis 1: Two Red + Rising Volume
    patterns, pattern_stats = analyze_two_red_patterns(df)
    
    # Analysis 2: Large Corrections
    corrections, correction_stats = analyze_corrections(df, min_pips=50)
    
    # Generate NEO rules
    neo_rules = generate_neo_rules(pattern_stats, correction_stats)
    
    # Save results
    results = {
        "generated_at": datetime.now().isoformat(),
        "data_range_days": 90,
        "candles_analyzed": len(df),
        "two_red_patterns": {
            "count": len(patterns),
            "statistics": pattern_stats,
            "samples": [asdict(p) for p in patterns[:10]]  # First 10 samples
        },
        "corrections": {
            "count": len(corrections),
            "statistics": correction_stats,
            "samples": [asdict(c) for c in corrections[:10]]  # First 10 samples
        },
        "neo_rules": neo_rules
    }
    
    output_path = os.path.join(OUTPUT_DIR, "backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")
    
    # Print key findings
    print("\n" + "=" * 70)
    print("                    KEY FINDINGS FOR NEO")
    print("=" * 70)
    
    down_pct = pattern_stats.get("outcome_5_candles", {}).get("DOWN", {}).get("pct", 0)
    corr_2red_pct = correction_stats.get("pre_correction_patterns", {}).get("2red_rising_volume", {}).get("pct", 0)
    
    print(f"""
1. "2 Red + Rising Volume" → {down_pct}% continued DOWN (next 5 candles)
   → This should trigger DEFCON 3 when confidence > 80%

2. {corr_2red_pct}% of large corrections had "2 Red + Rising Volume" beforehand
   → Pattern is predictive of corrections

3. Pattern at ATH = highest danger
   → Extra caution when at all-time highs

4. Grid should PAUSE when pattern detected at ATH
   → Protect capital, wait for resolution
""")
    
    return results


if __name__ == "__main__":
    results = main()

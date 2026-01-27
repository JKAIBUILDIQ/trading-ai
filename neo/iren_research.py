#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IREN TRADING RESEARCH & ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Deep analysis of IREN trading patterns including:
- Time-based volume analysis
- Post-rally and post-drop behavior
- Take profit timing
- Volume divergences
- Support/resistance levels
- Optimal TP/SL targets

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("/home/jbot/trading_ai/neo/research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_iren_data(period: str = "1y") -> pd.DataFrame:
    """Fetch IREN historical data"""
    print("📥 Fetching IREN historical data...")
    ticker = yf.Ticker("IREN")
    df = ticker.history(period=period, interval="1d")
    
    if df.empty:
        print("❌ No data received!")
        return pd.DataFrame()
    
    df.columns = [c.lower() for c in df.columns]
    df['daily_return'] = df['close'].pct_change() * 100
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    print(f"   ✅ Loaded {len(df)} days of data")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df


def analyze_time_based_volume(df: pd.DataFrame) -> Dict:
    """Analyze volume patterns by day of week"""
    print("\n📊 Analyzing time-based volume patterns...")
    
    df['day_of_week'] = df.index.dayofweek
    df['day_name'] = df.index.day_name()
    
    volume_by_day = df.groupby('day_name').agg({
        'volume': ['mean', 'std', 'sum'],
        'daily_return': ['mean', 'std', 'count']
    }).round(2)
    
    # Order by day
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    results = {
        "volume_by_day": {},
        "best_volume_day": "",
        "best_return_day": "",
        "insights": []
    }
    
    best_vol = 0
    best_ret = -999
    
    for day in day_order:
        if day in df['day_name'].values:
            day_data = df[df['day_name'] == day]
            avg_vol = day_data['volume'].mean()
            avg_ret = day_data['daily_return'].mean()
            win_rate = (day_data['daily_return'] > 0).sum() / len(day_data) * 100
            
            results["volume_by_day"][day] = {
                "avg_volume": int(avg_vol),
                "avg_return": round(avg_ret, 2),
                "win_rate": round(win_rate, 1),
                "sample_size": len(day_data)
            }
            
            if avg_vol > best_vol:
                best_vol = avg_vol
                results["best_volume_day"] = day
            if avg_ret > best_ret:
                best_ret = avg_ret
                results["best_return_day"] = day
    
    # Insights
    for day, data in results["volume_by_day"].items():
        if data["win_rate"] > 55:
            results["insights"].append(f"✅ {day}: {data['win_rate']}% win rate (bullish bias)")
        elif data["win_rate"] < 45:
            results["insights"].append(f"⚠️ {day}: {data['win_rate']}% win rate (bearish bias)")
    
    return results


def analyze_post_rally_behavior(df: pd.DataFrame, threshold: float = 5.0) -> Dict:
    """Analyze IREN behavior after 5%+ gains"""
    print(f"\n📈 Analyzing behavior after {threshold}%+ rallies...")
    
    rally_days = df[df['daily_return'] >= threshold].index.tolist()
    
    results = {
        "threshold": threshold,
        "total_rallies": len(rally_days),
        "rally_dates": [],
        "next_day_stats": {},
        "next_3day_stats": {},
        "next_5day_stats": {},
        "take_profit_timing": [],
        "insights": []
    }
    
    next_day_returns = []
    next_3day_returns = []
    next_5day_returns = []
    
    for rally_date in rally_days:
        try:
            idx = df.index.get_loc(rally_date)
            rally_return = df.iloc[idx]['daily_return']
            
            # Next day
            if idx + 1 < len(df):
                next_day_returns.append(df.iloc[idx + 1]['daily_return'])
            
            # Next 3 days cumulative
            if idx + 3 < len(df):
                cumret = ((df.iloc[idx+1:idx+4]['close'].iloc[-1] / df.iloc[idx]['close']) - 1) * 100
                next_3day_returns.append(cumret)
            
            # Next 5 days cumulative
            if idx + 5 < len(df):
                cumret = ((df.iloc[idx+1:idx+6]['close'].iloc[-1] / df.iloc[idx]['close']) - 1) * 100
                next_5day_returns.append(cumret)
                
                # Check for take profit (first negative day)
                for i in range(1, 6):
                    if df.iloc[idx + i]['daily_return'] < -1:
                        results["take_profit_timing"].append(i)
                        break
            
            results["rally_dates"].append({
                "date": rally_date.strftime('%Y-%m-%d'),
                "rally_pct": round(rally_return, 2)
            })
            
        except Exception as e:
            continue
    
    if next_day_returns:
        results["next_day_stats"] = {
            "avg_return": round(np.mean(next_day_returns), 2),
            "win_rate": round((np.array(next_day_returns) > 0).mean() * 100, 1),
            "median_return": round(np.median(next_day_returns), 2),
            "worst": round(min(next_day_returns), 2),
            "best": round(max(next_day_returns), 2)
        }
    
    if next_3day_returns:
        results["next_3day_stats"] = {
            "avg_return": round(np.mean(next_3day_returns), 2),
            "win_rate": round((np.array(next_3day_returns) > 0).mean() * 100, 1),
            "median_return": round(np.median(next_3day_returns), 2)
        }
    
    if next_5day_returns:
        results["next_5day_stats"] = {
            "avg_return": round(np.mean(next_5day_returns), 2),
            "win_rate": round((np.array(next_5day_returns) > 0).mean() * 100, 1),
            "median_return": round(np.median(next_5day_returns), 2)
        }
    
    # Take profit timing analysis
    if results["take_profit_timing"]:
        avg_tp_days = np.mean(results["take_profit_timing"])
        results["avg_take_profit_days"] = round(avg_tp_days, 1)
        results["insights"].append(
            f"📊 After {threshold}%+ rally, average take-profit happens on day {avg_tp_days:.1f}"
        )
    
    # Key insight
    if results["next_day_stats"]:
        nd_win = results["next_day_stats"]["win_rate"]
        if nd_win < 50:
            results["insights"].append(
                f"⚠️ CAUTION: After {threshold}%+ rally, next day is DOWN {100-nd_win:.0f}% of the time!"
            )
        else:
            results["insights"].append(
                f"✅ After {threshold}%+ rally, momentum continues {nd_win:.0f}% of the time"
            )
    
    return results


def analyze_post_drop_behavior(df: pd.DataFrame, threshold: float = -3.0) -> Dict:
    """Analyze IREN behavior after 3%+ drops"""
    print(f"\n📉 Analyzing behavior after {abs(threshold)}%+ drops...")
    
    drop_days = df[df['daily_return'] <= threshold].index.tolist()
    
    results = {
        "threshold": threshold,
        "total_drops": len(drop_days),
        "drop_dates": [],
        "next_day_stats": {},
        "next_3day_stats": {},
        "bounce_rate": 0,
        "insights": []
    }
    
    next_day_returns = []
    next_3day_returns = []
    bounces = 0
    
    for drop_date in drop_days:
        try:
            idx = df.index.get_loc(drop_date)
            drop_return = df.iloc[idx]['daily_return']
            
            if idx + 1 < len(df):
                nd_ret = df.iloc[idx + 1]['daily_return']
                next_day_returns.append(nd_ret)
                if nd_ret > 0:
                    bounces += 1
            
            if idx + 3 < len(df):
                cumret = ((df.iloc[idx+1:idx+4]['close'].iloc[-1] / df.iloc[idx]['close']) - 1) * 100
                next_3day_returns.append(cumret)
            
            results["drop_dates"].append({
                "date": drop_date.strftime('%Y-%m-%d'),
                "drop_pct": round(drop_return, 2)
            })
            
        except Exception:
            continue
    
    if next_day_returns:
        results["next_day_stats"] = {
            "avg_return": round(np.mean(next_day_returns), 2),
            "bounce_rate": round(bounces / len(next_day_returns) * 100, 1),
            "median_return": round(np.median(next_day_returns), 2),
            "worst": round(min(next_day_returns), 2),
            "best": round(max(next_day_returns), 2)
        }
        results["bounce_rate"] = results["next_day_stats"]["bounce_rate"]
    
    if next_3day_returns:
        results["next_3day_stats"] = {
            "avg_return": round(np.mean(next_3day_returns), 2),
            "recovery_rate": round((np.array(next_3day_returns) > 0).mean() * 100, 1)
        }
    
    # Insights
    if results["bounce_rate"] > 50:
        results["insights"].append(
            f"✅ BUY THE DIP: After {abs(threshold)}%+ drop, bounces {results['bounce_rate']:.0f}% next day!"
        )
    else:
        results["insights"].append(
            f"⚠️ FALLING KNIFE: After {abs(threshold)}%+ drop, continues falling {100-results['bounce_rate']:.0f}% of time"
        )
    
    return results


def analyze_volume_divergences(df: pd.DataFrame) -> Dict:
    """Detect volume/price divergences"""
    print("\n🔍 Analyzing volume divergences...")
    
    df = df.copy()
    
    # Detect divergences
    df['price_up'] = df['daily_return'] > 0.5
    df['vol_down'] = df['volume_ratio'] < 0.8  # Below average volume
    df['bearish_div'] = df['price_up'] & df['vol_down']  # Price up on low volume = bearish
    
    df['price_down'] = df['daily_return'] < -0.5
    df['vol_up'] = df['volume_ratio'] > 1.2  # Above average volume
    df['bullish_div'] = df['price_down'] & df['vol_up']  # Price down on high volume = potential capitulation
    
    results = {
        "bearish_divergences": [],
        "bullish_divergences": [],
        "bearish_div_accuracy": 0,
        "bullish_div_accuracy": 0,
        "insights": []
    }
    
    # Analyze bearish divergences (price up, volume down)
    bearish_divs = df[df['bearish_div']].index.tolist()
    correct_bearish = 0
    
    for div_date in bearish_divs[-20:]:  # Last 20
        try:
            idx = df.index.get_loc(div_date)
            if idx + 2 < len(df):
                next_2day = df.iloc[idx+1:idx+3]['daily_return'].sum()
                if next_2day < 0:
                    correct_bearish += 1
                results["bearish_divergences"].append({
                    "date": div_date.strftime('%Y-%m-%d'),
                    "return_on_day": round(df.iloc[idx]['daily_return'], 2),
                    "volume_ratio": round(df.iloc[idx]['volume_ratio'], 2),
                    "next_2day_return": round(next_2day, 2)
                })
        except:
            continue
    
    if len(results["bearish_divergences"]) > 0:
        results["bearish_div_accuracy"] = round(correct_bearish / len(results["bearish_divergences"]) * 100, 1)
    
    # Analyze bullish divergences (price down, volume spike - capitulation)
    bullish_divs = df[df['bullish_div']].index.tolist()
    correct_bullish = 0
    
    for div_date in bullish_divs[-20:]:
        try:
            idx = df.index.get_loc(div_date)
            if idx + 2 < len(df):
                next_2day = df.iloc[idx+1:idx+3]['daily_return'].sum()
                if next_2day > 0:
                    correct_bullish += 1
                results["bullish_divergences"].append({
                    "date": div_date.strftime('%Y-%m-%d'),
                    "return_on_day": round(df.iloc[idx]['daily_return'], 2),
                    "volume_ratio": round(df.iloc[idx]['volume_ratio'], 2),
                    "next_2day_return": round(next_2day, 2)
                })
        except:
            continue
    
    if len(results["bullish_divergences"]) > 0:
        results["bullish_div_accuracy"] = round(correct_bullish / len(results["bullish_divergences"]) * 100, 1)
    
    # Insights
    if results["bearish_div_accuracy"] > 55:
        results["insights"].append(
            f"📉 BEARISH DIVERGENCE WORKS: Price up + low volume → drop {results['bearish_div_accuracy']:.0f}% of time"
        )
    
    if results["bullish_div_accuracy"] > 55:
        results["insights"].append(
            f"📈 CAPITULATION SIGNAL: Price down + high volume → bounce {results['bullish_div_accuracy']:.0f}% of time"
        )
    
    return results


def calculate_support_resistance(df: pd.DataFrame) -> Dict:
    """Calculate key support and resistance levels"""
    print("\n📏 Calculating support and resistance levels...")
    
    recent = df.tail(60)  # Last 60 days
    
    # Key levels
    current_price = recent['close'].iloc[-1]
    
    # Support levels (lows that held multiple times)
    lows = recent['low'].values
    
    # Find clusters of lows
    low_sorted = sorted(lows)
    supports = []
    
    # Fibonacci levels from recent range
    high_60d = recent['high'].max()
    low_60d = recent['low'].min()
    fib_range = high_60d - low_60d
    
    fib_levels = {
        "0.0 (Low)": low_60d,
        "0.236": low_60d + fib_range * 0.236,
        "0.382": low_60d + fib_range * 0.382,
        "0.5": low_60d + fib_range * 0.5,
        "0.618": low_60d + fib_range * 0.618,
        "0.786": low_60d + fib_range * 0.786,
        "1.0 (High)": high_60d
    }
    
    # Simple support/resistance from key price points
    results = {
        "current_price": round(current_price, 2),
        "60_day_high": round(high_60d, 2),
        "60_day_low": round(low_60d, 2),
        "fibonacci_levels": {k: round(v, 2) for k, v in fib_levels.items()},
        "key_supports": [],
        "key_resistances": [],
        "insights": []
    }
    
    # Find support levels (price bounced from these)
    for fib_name, fib_price in fib_levels.items():
        if fib_price < current_price:
            results["key_supports"].append({
                "level": fib_name,
                "price": round(fib_price, 2),
                "distance_pct": round((current_price - fib_price) / current_price * 100, 1)
            })
        elif fib_price > current_price:
            results["key_resistances"].append({
                "level": fib_name,
                "price": round(fib_price, 2),
                "distance_pct": round((fib_price - current_price) / current_price * 100, 1)
            })
    
    # Add moving averages as dynamic support/resistance
    ma20 = recent['close'].rolling(20).mean().iloc[-1]
    ma50 = recent['close'].rolling(50).mean().iloc[-1] if len(recent) >= 50 else None
    
    if ma20:
        if ma20 < current_price:
            results["key_supports"].append({
                "level": "20-day MA",
                "price": round(ma20, 2),
                "distance_pct": round((current_price - ma20) / current_price * 100, 1)
            })
        else:
            results["key_resistances"].append({
                "level": "20-day MA",
                "price": round(ma20, 2),
                "distance_pct": round((ma20 - current_price) / current_price * 100, 1)
            })
    
    # Sort by distance
    results["key_supports"] = sorted(results["key_supports"], key=lambda x: x["distance_pct"])
    results["key_resistances"] = sorted(results["key_resistances"], key=lambda x: x["distance_pct"])
    
    # Insights
    if results["key_supports"]:
        nearest = results["key_supports"][0]
        results["insights"].append(
            f"💪 Nearest support: ${nearest['price']:.2f} ({nearest['distance_pct']:.1f}% below)"
        )
    
    if results["key_resistances"]:
        nearest = results["key_resistances"][0]
        results["insights"].append(
            f"🚧 Nearest resistance: ${nearest['price']:.2f} ({nearest['distance_pct']:.1f}% above)"
        )
    
    return results


def calculate_optimal_tp_sl(df: pd.DataFrame) -> Dict:
    """Calculate optimal take-profit and stop-loss levels"""
    print("\n🎯 Calculating optimal TP/SL targets...")
    
    # Calculate ATR (Average True Range)
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    current_price = df['close'].iloc[-1]
    current_atr = df['atr'].iloc[-1]
    atr_pct = current_atr / current_price * 100
    
    # Historical win rate at different TP/SL ratios
    results = {
        "current_price": round(current_price, 2),
        "atr_14": round(current_atr, 2),
        "atr_pct": round(atr_pct, 2),
        "recommendations": {},
        "scenarios": [],
        "insights": []
    }
    
    # Test different R:R ratios
    for tp_pct in [3, 5, 7, 10]:
        for sl_pct in [2, 3, 5]:
            wins = 0
            losses = 0
            
            for i in range(len(df) - 20):
                entry = df.iloc[i]['close']
                tp_target = entry * (1 + tp_pct / 100)
                sl_target = entry * (1 - sl_pct / 100)
                
                # Simulate next 10 days
                for j in range(1, min(11, len(df) - i)):
                    high = df.iloc[i + j]['high']
                    low = df.iloc[i + j]['low']
                    
                    if high >= tp_target:
                        wins += 1
                        break
                    elif low <= sl_target:
                        losses += 1
                        break
            
            total = wins + losses
            if total > 0:
                win_rate = wins / total * 100
                rr = tp_pct / sl_pct
                expected_value = (win_rate/100 * tp_pct) - ((100-win_rate)/100 * sl_pct)
                
                results["scenarios"].append({
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "r_r": round(rr, 1),
                    "win_rate": round(win_rate, 1),
                    "expected_value": round(expected_value, 2),
                    "trades": total
                })
    
    # Sort by expected value
    results["scenarios"] = sorted(results["scenarios"], key=lambda x: x["expected_value"], reverse=True)
    
    # Best recommendation
    if results["scenarios"]:
        best = results["scenarios"][0]
        results["recommendations"] = {
            "take_profit_pct": best["tp_pct"],
            "stop_loss_pct": best["sl_pct"],
            "take_profit_price": round(current_price * (1 + best["tp_pct"]/100), 2),
            "stop_loss_price": round(current_price * (1 - best["sl_pct"]/100), 2),
            "expected_win_rate": best["win_rate"],
            "risk_reward": best["r_r"],
            "expected_value_per_trade": best["expected_value"]
        }
        
        results["insights"].append(
            f"🎯 OPTIMAL: TP +{best['tp_pct']}% / SL -{best['sl_pct']}% "
            f"(Win rate: {best['win_rate']:.0f}%, EV: +{best['expected_value']:.2f}%)"
        )
    
    return results


def generate_trading_strategy(all_results: Dict) -> Dict:
    """Generate a complete trading strategy for Paul"""
    print("\n📋 Generating trading strategy...")
    
    strategy = {
        "title": "IREN TRADING STRATEGY - Research-Based",
        "generated": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "summary": {},
        "entry_rules": [],
        "exit_rules": [],
        "risk_management": {},
        "key_levels": {},
        "warnings": [],
        "paul_summary": ""
    }
    
    # Extract key findings
    tp_sl = all_results.get("optimal_tp_sl", {})
    post_rally = all_results.get("post_rally", {})
    post_drop = all_results.get("post_drop", {})
    divergences = all_results.get("divergences", {})
    support_resistance = all_results.get("support_resistance", {})
    
    current_price = support_resistance.get("current_price", 53.57)
    
    # Summary
    strategy["summary"] = {
        "current_price": current_price,
        "recommended_tp_pct": tp_sl.get("recommendations", {}).get("take_profit_pct", 7),
        "recommended_sl_pct": tp_sl.get("recommendations", {}).get("stop_loss_pct", 3),
        "take_profit_target": tp_sl.get("recommendations", {}).get("take_profit_price", round(current_price * 1.07, 2)),
        "stop_loss_target": tp_sl.get("recommendations", {}).get("stop_loss_price", round(current_price * 0.97, 2))
    }
    
    # Entry rules
    strategy["entry_rules"] = [
        "🟢 BUY after 3%+ drop if volume is 1.2x+ average (capitulation signal)",
        f"🟢 BUY near support levels: {[s['price'] for s in support_resistance.get('key_supports', [])[:3]]}",
        "🟢 BUY on bearish divergence reversal (volume up, price stabilizing)",
        "⚠️ AVOID buying day after 5%+ rally (tends to pull back)"
    ]
    
    # Exit rules
    strategy["exit_rules"] = [
        f"✅ TAKE PROFIT at +{tp_sl.get('recommendations', {}).get('take_profit_pct', 7)}% from entry",
        f"❌ STOP LOSS at -{tp_sl.get('recommendations', {}).get('stop_loss_pct', 3)}% from entry",
        f"⏰ Take profit typically triggers within {post_rally.get('avg_take_profit_days', 3):.0f} days after rally",
        "📉 EXIT if price rises on falling volume (bearish divergence)"
    ]
    
    # Risk management
    strategy["risk_management"] = {
        "position_size": "Max 10% of portfolio per trade",
        "max_daily_loss": "Stop trading if down 5% in a day",
        "scaling": "Scale in: 50% initial, 50% on confirmation",
        "holding_period": "Optimal: 3-5 days for swings"
    }
    
    # Key levels
    strategy["key_levels"] = {
        "strong_support": support_resistance.get("key_supports", [{}])[0].get("price") if support_resistance.get("key_supports") else round(current_price * 0.9, 2),
        "first_resistance": support_resistance.get("key_resistances", [{}])[0].get("price") if support_resistance.get("key_resistances") else round(current_price * 1.1, 2),
        "60_day_high": support_resistance.get("60_day_high", round(current_price * 1.15, 2)),
        "60_day_low": support_resistance.get("60_day_low", round(current_price * 0.85, 2))
    }
    
    # Warnings
    if post_rally.get("next_day_stats", {}).get("win_rate", 50) < 50:
        strategy["warnings"].append(
            f"⚠️ After 5%+ rally, next day is DOWN {100-post_rally['next_day_stats']['win_rate']:.0f}% of the time"
        )
    
    if divergences.get("bearish_div_accuracy", 0) > 55:
        strategy["warnings"].append(
            "⚠️ Price up + low volume = likely drop (bearish divergence)"
        )
    
    # Paul summary (text for messaging)
    strategy["paul_summary"] = f"""
📊 IREN TRADING STRATEGY
Generated: {strategy['generated']}

💰 CURRENT: ${current_price:.2f}
🎯 TAKE PROFIT: ${strategy['summary']['take_profit_target']:.2f} (+{strategy['summary']['recommended_tp_pct']}%)
🛑 STOP LOSS: ${strategy['summary']['stop_loss_target']:.2f} (-{strategy['summary']['recommended_sl_pct']}%)

📈 WHEN TO BUY:
• After 3%+ drop with high volume (bounce rate: {post_drop.get('bounce_rate', 50):.0f}%)
• Near support: ${strategy['key_levels']['strong_support']:.2f}
• On volume spike capitulation

📉 WHEN TO SELL:
• At TP target (+{strategy['summary']['recommended_tp_pct']}%)
• After 5%+ rally (next day drops {100-post_rally.get('next_day_stats', {}).get('win_rate', 50):.0f}% of time)
• Price up on LOW volume = exit soon

⏰ TIMING:
• Take profits usually hit in {post_rally.get('avg_take_profit_days', 3):.0f} days
• Best volume day: {all_results.get('time_based_volume', {}).get('best_volume_day', 'N/A')}

📏 KEY LEVELS:
• Support: ${strategy['key_levels']['strong_support']:.2f}
• Resistance: ${strategy['key_levels']['first_resistance']:.2f}
• 60-day range: ${strategy['key_levels']['60_day_low']:.2f} - ${strategy['key_levels']['60_day_high']:.2f}
"""
    
    return strategy


def run_full_research():
    """Run complete IREN research"""
    print("=" * 70)
    print("🔬 IREN TRADING RESEARCH & ANALYSIS")
    print("=" * 70)
    
    # Fetch data
    df = fetch_iren_data("1y")
    
    if df.empty:
        return {"error": "No data available"}
    
    # Run all analyses
    all_results = {}
    
    all_results["time_based_volume"] = analyze_time_based_volume(df)
    all_results["post_rally"] = analyze_post_rally_behavior(df, threshold=5.0)
    all_results["post_drop"] = analyze_post_drop_behavior(df, threshold=-3.0)
    all_results["divergences"] = analyze_volume_divergences(df)
    all_results["support_resistance"] = calculate_support_resistance(df)
    all_results["optimal_tp_sl"] = calculate_optimal_tp_sl(df)
    
    # Generate strategy
    all_results["trading_strategy"] = generate_trading_strategy(all_results)
    
    # Save results
    output_file = OUTPUT_DIR / "iren_research.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 RESEARCH COMPLETE!")
    print("=" * 70)
    
    print("\n📈 KEY FINDINGS:")
    
    for section, data in all_results.items():
        if isinstance(data, dict) and "insights" in data:
            for insight in data.get("insights", []):
                print(f"   {insight}")
    
    print("\n" + all_results["trading_strategy"]["paul_summary"])
    
    print(f"\n📁 Full results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_research()

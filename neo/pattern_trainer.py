#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO PATTERN TRAINER
═══════════════════════════════════════════════════════════════════════════════

Fetches 90 days of historical data for XAUUSD and IREN,
detects patterns, and trains NEO on their outcomes.

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

from pattern_detector import PatternDetector, PatternResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternTrainer")


class PatternTrainer:
    """
    Trains NEO on 90 days of historical patterns for XAUUSD and IREN.
    """
    
    def __init__(self):
        self.data_dir = Path("/home/jbot/trading_ai/neo/pattern_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Symbol mappings for yfinance
        self.symbol_map = {
            "XAUUSD": "GC=F",   # Gold futures
            "IREN": "IREN"      # IREN stock
        }
        
        # Timeframes to analyze
        self.timeframes = {
            "H4": "60m",  # 4-hour (yfinance uses 60m, we'll resample)
            "H1": "60m",  # 1-hour
            "D1": "1d"    # Daily
        }
        
    def fetch_historical_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch historical OHLC data from Yahoo Finance"""
        yf_symbol = self.symbol_map.get(symbol, symbol)
        
        logger.info(f"📥 Fetching {days} days of {symbol} ({yf_symbol}) data...")
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # For intraday data, yfinance limits to 60 days max
            # We'll get daily data for full 90 days, hourly for recent 60
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get daily data
            daily_df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            # Get hourly data (max 60 days)
            hourly_start = end_date - timedelta(days=59)
            hourly_df = ticker.history(start=hourly_start, end=end_date, interval="1h")
            
            if daily_df.empty:
                logger.error(f"No daily data for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"   ✅ Daily: {len(daily_df)} candles")
            logger.info(f"   ✅ Hourly: {len(hourly_df)} candles")
            
            # Save raw data
            daily_df.to_csv(self.data_dir / f"{symbol.lower()}_daily.csv")
            if not hourly_df.empty:
                hourly_df.to_csv(self.data_dir / f"{symbol.lower()}_hourly.csv")
            
            # Create H4 by resampling hourly
            if not hourly_df.empty:
                h4_df = hourly_df.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                h4_df.to_csv(self.data_dir / f"{symbol.lower()}_h4.csv")
                logger.info(f"   ✅ H4: {len(h4_df)} candles")
                return h4_df
            
            return daily_df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def backtest_patterns(self, symbol: str, df: pd.DataFrame, 
                          lookahead_bars: int = 4) -> Dict:
        """
        Backtest patterns on historical data.
        
        For each detected pattern, check if the predicted direction
        was correct N bars later.
        """
        detector = PatternDetector(symbol)
        
        results = {
            "symbol": symbol,
            "total_patterns": 0,
            "correct": 0,
            "incorrect": 0,
            "pattern_breakdown": {}
        }
        
        if len(df) < 50 + lookahead_bars:
            logger.warning(f"Insufficient data for backtesting: {len(df)} bars")
            return results
        
        logger.info(f"🔬 Backtesting patterns on {symbol}...")
        logger.info(f"   Using {len(df)} bars, lookahead: {lookahead_bars} bars")
        
        # Standardize columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Slide through historical data
        for i in range(50, len(df) - lookahead_bars):
            # Get data up to this point
            window_df = df.iloc[:i+1]
            
            # Detect patterns
            patterns = detector.analyze(window_df)
            
            if not patterns:
                continue
            
            # Get future price to evaluate
            current_close = df.iloc[i]['close']
            future_close = df.iloc[i + lookahead_bars]['close']
            actual_direction = "UP" if future_close > current_close else "DOWN"
            
            for pattern in patterns:
                pattern_name = pattern.pattern_type.value
                predicted = "UP" if pattern.direction == "BUY" else "DOWN"
                correct = predicted == actual_direction
                
                # Track overall
                results["total_patterns"] += 1
                if correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                
                # Track by pattern type
                if pattern_name not in results["pattern_breakdown"]:
                    results["pattern_breakdown"][pattern_name] = {
                        "total": 0, "correct": 0, "examples": []
                    }
                
                results["pattern_breakdown"][pattern_name]["total"] += 1
                if correct:
                    results["pattern_breakdown"][pattern_name]["correct"] += 1
                
                # Record outcome for learning
                detector.record_pattern_outcome(pattern_name, correct)
        
        # Calculate accuracies
        if results["total_patterns"] > 0:
            results["overall_accuracy"] = results["correct"] / results["total_patterns"] * 100
        else:
            results["overall_accuracy"] = 0
        
        for pattern_name, data in results["pattern_breakdown"].items():
            if data["total"] > 0:
                data["accuracy"] = data["correct"] / data["total"] * 100
            else:
                data["accuracy"] = 0
        
        return results
    
    def train_symbol(self, symbol: str, days: int = 90) -> Dict:
        """Full training pipeline for one symbol"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎓 TRAINING NEO ON {symbol} ({days} DAYS)")
        logger.info(f"{'='*60}\n")
        
        # Fetch data
        df = self.fetch_historical_data(symbol, days)
        
        if df.empty:
            return {"error": f"No data for {symbol}"}
        
        # Run backtest
        results = self.backtest_patterns(symbol, df)
        
        # Save results
        results_file = self.data_dir / f"{symbol.lower()}_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n📊 {symbol} TRAINING RESULTS:")
        logger.info(f"   Total patterns: {results['total_patterns']}")
        logger.info(f"   Correct: {results['correct']}")
        logger.info(f"   Accuracy: {results.get('overall_accuracy', 0):.1f}%")
        
        if results.get("pattern_breakdown"):
            logger.info(f"\n   Pattern breakdown:")
            sorted_patterns = sorted(
                results["pattern_breakdown"].items(),
                key=lambda x: x[1].get("accuracy", 0),
                reverse=True
            )
            for pattern, data in sorted_patterns:
                if data["total"] >= 3:
                    acc = data.get("accuracy", 0)
                    emoji = "✅" if acc >= 60 else "⚠️" if acc >= 50 else "❌"
                    logger.info(f"   {emoji} {pattern}: {acc:.0f}% ({data['total']} samples)")
        
        return results
    
    def train_all(self, days: int = 90) -> Dict:
        """Train on both XAUUSD and IREN"""
        results = {}
        
        for symbol in ["XAUUSD", "IREN"]:
            try:
                results[symbol] = self.train_symbol(symbol, days)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to train {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        # Save combined results
        summary_file = self.data_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "trained_at": datetime.now().isoformat(),
                "days": days,
                "results": results
            }, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎓 TRAINING COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {summary_file}")
        
        return results
    
    def get_best_patterns(self, symbol: str, min_samples: int = 5) -> List[Dict]:
        """Get highest performing patterns for a symbol"""
        detector = PatternDetector(symbol)
        stats = detector.get_stats()
        
        best = []
        for pattern, data in stats.items():
            if data["total"] >= min_samples and data["accuracy"] >= 55:
                best.append({
                    "pattern": pattern,
                    "accuracy": data["accuracy"],
                    "samples": data["total"]
                })
        
        return sorted(best, key=lambda x: x["accuracy"], reverse=True)


def main():
    """Run full training"""
    trainer = PatternTrainer()
    
    print("🧠 NEO PATTERN RECOGNITION TRAINING")
    print("=" * 60)
    print("Training on 90 days of historical data for XAUUSD and IREN")
    print("This will take a few minutes...")
    print()
    
    results = trainer.train_all(days=90)
    
    print("\n" + "=" * 60)
    print("📊 BEST PERFORMING PATTERNS")
    print("=" * 60)
    
    for symbol in ["XAUUSD", "IREN"]:
        print(f"\n{symbol}:")
        best = trainer.get_best_patterns(symbol)
        if best:
            for p in best[:5]:
                print(f"  ✅ {p['pattern']}: {p['accuracy']:.0f}% ({p['samples']} samples)")
        else:
            print("  No reliable patterns yet (need more data)")
    
    return results


if __name__ == "__main__":
    main()

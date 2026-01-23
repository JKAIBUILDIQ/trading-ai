"""
XAUUSD Backtest and Pattern Analysis Module

Usage:
    from backtest.xauusd_patterns import XAUUSDPatternAnalyzer
    
    analyzer = XAUUSDPatternAnalyzer(months=12)
    results = analyzer.run_full_analysis()
    prediction = analyzer.get_daily_prediction()
"""

from .xauusd_patterns import XAUUSDPatternAnalyzer, XAUUSDDataCollector

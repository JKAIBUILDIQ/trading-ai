#!/usr/bin/env python3
"""
Pattern Monitor Service - 24/7 chart pattern detection.

Monitors gold (and other symbols) for candlestick patterns and sends alerts
when danger/opportunity patterns are detected.

Run with: pm2 start services/pattern_monitor.py --name pattern-monitor --interpreter python3
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from api.pattern_detector import PatternDetector
from api.alert_system import AlertSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PatternMonitor")


class PriceMonitor:
    """Fetch and cache price data."""
    
    def __init__(self):
        self.candle_history: Dict[str, List[Dict]] = {}
        self.last_fetch: Dict[str, datetime] = {}
    
    async def fetch_candles(self, symbol: str, interval: str = "1h", period: str = "5d") -> List[Dict]:
        """Fetch recent candles for analysis."""
        try:
            # Use appropriate yfinance symbol
            yf_symbol = symbol
            if symbol == "XAUUSD":
                yf_symbol = "GC=F"  # Gold futures
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                return []
            
            candles = []
            for idx, row in hist.iterrows():
                candles.append({
                    "time": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row.get("Volume", 0)),
                })
            
            self.candle_history[symbol] = candles
            self.last_fetch[symbol] = datetime.now()
            
            logger.debug(f"Fetched {len(candles)} candles for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return self.candle_history.get(symbol, [])  # Return cached if available


class PatternMonitorService:
    """Main monitoring service."""
    
    def __init__(self):
        self.price_monitor = PriceMonitor()
        self.detector = PatternDetector()
        self.alerts = AlertSystem()
        self.running = False
        
        # Symbols to monitor
        self.symbols = [
            {"symbol": "XAUUSD", "name": "Gold Spot", "yf": "GC=F"},
            # Add more as needed:
            # {"symbol": "IREN", "name": "IREN", "yf": "IREN"},
            # {"symbol": "BTCUSD", "name": "Bitcoin", "yf": "BTC-USD"},
        ]
        
        # Check intervals (minutes)
        self.check_interval = 5  # Check every 5 minutes
        self.candle_interval = "1h"  # Analyze 1-hour candles
    
    async def start(self):
        """Start continuous monitoring."""
        self.running = True
        logger.info("=" * 50)
        logger.info("PATTERN MONITOR SERVICE STARTED")
        logger.info(f"Monitoring: {[s['symbol'] for s in self.symbols]}")
        logger.info(f"Check interval: {self.check_interval} minutes")
        logger.info(f"Candle interval: {self.candle_interval}")
        logger.info("=" * 50)
        
        while self.running:
            try:
                await self.check_all_symbols()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
            
            # Wait for next check
            await asyncio.sleep(self.check_interval * 60)
    
    async def check_all_symbols(self):
        """Check all monitored symbols for patterns."""
        logger.info(f"Checking {len(self.symbols)} symbols...")
        
        for symbol_config in self.symbols:
            symbol = symbol_config["symbol"]
            yf_symbol = symbol_config.get("yf", symbol)
            
            try:
                # Fetch candles
                candles = await self.price_monitor.fetch_candles(yf_symbol, self.candle_interval)
                
                if not candles or len(candles) < 10:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Detect patterns
                patterns = self.detector.detect_all_patterns(candles, symbol)
                
                if patterns:
                    logger.info(f"Detected {len(patterns)} pattern(s) on {symbol}")
                    
                    for pattern in patterns:
                        logger.info(f"  - {pattern['pattern']} ({pattern['severity']}): {pattern['message']}")
                        
                        # Send alert
                        sent = await self.alerts.send_alert(pattern, symbol)
                        if sent:
                            logger.info(f"  ✓ Alert sent for {pattern['pattern']}")
                        else:
                            logger.debug(f"  - Alert skipped (cooldown)")
                else:
                    logger.debug(f"No patterns detected on {symbol}")
                    
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
        
        logger.info("Check complete")
    
    async def check_single(self, symbol: str) -> List[Dict]:
        """Check a single symbol on demand."""
        try:
            candles = await self.price_monitor.fetch_candles(symbol, self.candle_interval)
            
            if not candles:
                return []
            
            patterns = self.detector.detect_all_patterns(candles, symbol)
            
            for pattern in patterns:
                await self.alerts.send_alert(pattern, symbol)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Single check error for {symbol}: {e}")
            return []
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping pattern monitor...")
        self.running = False


# Run as standalone service
if __name__ == "__main__":
    service = PatternMonitorService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        service.stop()
        logger.info("Service stopped by user")

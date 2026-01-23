#!/usr/bin/env python3
"""
Real Data Sources for Trading AI

CRITICAL RULES:
1. NO random data generation - EVER
2. Every data point must have timestamp + source
3. If API fails, return error - NOT fake data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import data_config


class RealDataSource:
    """Base class for all real data sources"""
    
    def __init__(self):
        self.source_name = "UNKNOWN"
    
    def _create_response(self, data: Any, success: bool, error: str = None) -> Dict:
        """Standard response format with source tracking"""
        return {
            "source": self.source_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": success,
            "error": error,
            "data": data,
            "verified": success
        }


class PolygonDataSource(RealDataSource):
    """
    Polygon.io - Real-time and historical market data
    https://polygon.io/docs/forex
    """
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.source_name = "POLYGON_IO_REAL"
        self.api_key = api_key or data_config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"
    
    def get_forex_snapshot(self, pair: str) -> Dict:
        """Get real-time forex quote"""
        if not self.api_key:
            return self._create_response(None, False, "POLYGON_API_KEY not configured")
        
        try:
            # Convert pair format: EURUSD -> C:EURUSD
            ticker = f"C:{pair}"
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {"apiKey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._create_response(data, True)
            else:
                return self._create_response(None, False, f"API returned {response.status_code}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))
    
    def get_historical_bars(self, pair: str, timeframe: str = "minute", 
                           from_date: str = None, to_date: str = None,
                           limit: int = 5000) -> Dict:
        """
        Get historical OHLCV bars
        
        Args:
            pair: Currency pair (e.g., "EURUSD")
            timeframe: "minute", "hour", "day"
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max bars to return
        """
        if not self.api_key:
            return self._create_response(None, False, "POLYGON_API_KEY not configured")
        
        try:
            ticker = f"C:{pair}"
            
            # Default to last 30 days
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{from_date}/{to_date}"
            params = {
                "apiKey": self.api_key,
                "limit": limit,
                "sort": "asc"
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Convert to DataFrame-friendly format
                bars = []
                for bar in results:
                    bars.append({
                        "timestamp": datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar.get("v", 0),
                        "vwap": bar.get("vw"),
                        "transactions": bar.get("n", 0)
                    })
                
                return self._create_response({
                    "pair": pair,
                    "timeframe": timeframe,
                    "bar_count": len(bars),
                    "bars": bars
                }, True)
            else:
                return self._create_response(None, False, f"API returned {response.status_code}: {response.text}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))


class TwelveDataSource(RealDataSource):
    """
    Twelve Data - Alternative real-time data source
    https://twelvedata.com/docs
    """
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.source_name = "TWELVE_DATA_REAL"
        self.api_key = api_key or data_config.TWELVE_DATA_API_KEY
        self.base_url = "https://api.twelvedata.com"
    
    def get_forex_price(self, pair: str) -> Dict:
        """Get real-time forex price"""
        if not self.api_key:
            return self._create_response(None, False, "TWELVE_DATA_API_KEY not configured")
        
        try:
            # Convert format: EURUSD -> EUR/USD
            symbol = f"{pair[:3]}/{pair[3:]}"
            url = f"{self.base_url}/price"
            params = {"symbol": symbol, "apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "price" in data:
                    return self._create_response({
                        "pair": pair,
                        "price": float(data["price"]),
                        "fetched_at": datetime.utcnow().isoformat()
                    }, True)
                else:
                    return self._create_response(None, False, f"Invalid response: {data}")
            else:
                return self._create_response(None, False, f"API returned {response.status_code}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))
    
    def get_time_series(self, pair: str, interval: str = "1min", 
                        outputsize: int = 5000) -> Dict:
        """Get historical time series data"""
        if not self.api_key:
            return self._create_response(None, False, "TWELVE_DATA_API_KEY not configured")
        
        try:
            symbol = f"{pair[:3]}/{pair[3:]}"
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "values" in data:
                    bars = []
                    for bar in data["values"]:
                        bars.append({
                            "timestamp": bar["datetime"],
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"])
                        })
                    
                    return self._create_response({
                        "pair": pair,
                        "interval": interval,
                        "bar_count": len(bars),
                        "bars": bars
                    }, True)
                else:
                    return self._create_response(None, False, f"Invalid response: {data}")
            else:
                return self._create_response(None, False, f"API returned {response.status_code}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))


class MT5DataSource(RealDataSource):
    """
    MT5 API - Your REAL trading data from Crellastein bots
    This is the most important data source!
    """
    
    def __init__(self, api_url: str = None):
        super().__init__()
        self.source_name = "MT5_REAL_TRADES"
        self.api_url = api_url or data_config.MT5_API_URL
    
    def get_trades(self) -> Dict:
        """Get all real trades from MT5"""
        try:
            response = requests.get(f"{self.api_url}/trades", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._create_response(data, True)
            else:
                return self._create_response(None, False, f"MT5 API returned {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            return self._create_response(None, False, "MT5 API offline")
        except Exception as e:
            return self._create_response(None, False, str(e))
    
    def get_positions(self) -> Dict:
        """Get open positions from MT5"""
        try:
            response = requests.get(f"{self.api_url}/positions", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._create_response(data, True)
            else:
                return self._create_response(None, False, f"MT5 API returned {response.status_code}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))
    
    def get_account_info(self) -> Dict:
        """Get account balance and equity"""
        try:
            response = requests.get(f"{self.api_url}/account", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._create_response(data, True)
            else:
                return self._create_response(None, False, f"MT5 API returned {response.status_code}")
                
        except Exception as e:
            return self._create_response(None, False, str(e))


class DukascopyDataSource(RealDataSource):
    """
    Dukascopy - FREE historical tick data
    Download from: https://www.dukascopy.com/swiss/english/marketwatch/historical/
    
    This provides MILLIONS of real tick data points for FREE
    """
    
    def __init__(self, data_path: str = None):
        super().__init__()
        self.source_name = "DUKASCOPY_HISTORICAL"
        self.data_path = data_path or data_config.DUKASCOPY_DATA_PATH
    
    def load_tick_data(self, pair: str, year: int, month: int) -> Dict:
        """
        Load tick data from downloaded Dukascopy files
        
        Files should be in format: {data_path}/{pair}/{year}/{month}.csv
        """
        try:
            file_path = os.path.join(self.data_path, pair, str(year), f"{month:02d}.csv")
            
            if not os.path.exists(file_path):
                return self._create_response(None, False, f"File not found: {file_path}")
            
            # Dukascopy CSV format: timestamp,ask,bid,ask_volume,bid_volume
            df = pd.read_csv(file_path, names=["timestamp", "ask", "bid", "ask_vol", "bid_vol"])
            
            # Convert to list of dicts
            ticks = df.to_dict(orient="records")
            
            return self._create_response({
                "pair": pair,
                "year": year,
                "month": month,
                "tick_count": len(ticks),
                "ticks": ticks[:1000]  # Return first 1000 for preview
            }, True)
            
        except Exception as e:
            return self._create_response(None, False, str(e))
    
    def get_available_data(self) -> Dict:
        """List available downloaded tick data"""
        try:
            available = {}
            
            if not os.path.exists(self.data_path):
                return self._create_response({"available": {}}, True)
            
            for pair in os.listdir(self.data_path):
                pair_path = os.path.join(self.data_path, pair)
                if os.path.isdir(pair_path):
                    available[pair] = []
                    for year in os.listdir(pair_path):
                        year_path = os.path.join(pair_path, year)
                        if os.path.isdir(year_path):
                            for month_file in os.listdir(year_path):
                                if month_file.endswith(".csv"):
                                    available[pair].append(f"{year}/{month_file}")
            
            return self._create_response({"available": available}, True)
            
        except Exception as e:
            return self._create_response(None, False, str(e))


class UnifiedDataManager:
    """
    Unified interface for all data sources
    Automatically selects best available source
    """
    
    def __init__(self):
        self.polygon = PolygonDataSource()
        self.twelve = TwelveDataSource()
        self.mt5 = MT5DataSource()
        self.dukascopy = DukascopyDataSource()
    
    def get_realtime_price(self, pair: str) -> Dict:
        """Get real-time price from best available source"""
        
        # Try Polygon first (fastest)
        if data_config.POLYGON_API_KEY:
            result = self.polygon.get_forex_snapshot(pair)
            if result["success"]:
                return result
        
        # Fallback to Twelve Data
        if data_config.TWELVE_DATA_API_KEY:
            result = self.twelve.get_forex_price(pair)
            if result["success"]:
                return result
        
        # No API available
        return {
            "source": "NO_API_CONFIGURED",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "error": "No market data API configured. Set POLYGON_API_KEY or TWELVE_DATA_API_KEY",
            "data": None,
            "verified": False
        }
    
    def get_historical_data(self, pair: str, timeframe: str = "minute",
                           days: int = 30) -> Dict:
        """Get historical data from best available source"""
        
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try Polygon first
        if data_config.POLYGON_API_KEY:
            result = self.polygon.get_historical_bars(pair, timeframe, from_date, to_date)
            if result["success"]:
                return result
        
        # Fallback to Twelve Data
        if data_config.TWELVE_DATA_API_KEY:
            result = self.twelve.get_time_series(pair, "1min" if timeframe == "minute" else "1h")
            if result["success"]:
                return result
        
        return {
            "source": "NO_API_CONFIGURED",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "error": "No market data API configured",
            "data": None,
            "verified": False
        }
    
    def get_mt5_trades(self) -> Dict:
        """Get real trades from MT5 (your Crellastein bots)"""
        return self.mt5.get_trades()
    
    def get_all_sources_status(self) -> Dict:
        """Check status of all data sources"""
        status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sources": {}
        }
        
        # Check Polygon
        status["sources"]["polygon"] = {
            "configured": bool(data_config.POLYGON_API_KEY),
            "type": "realtime + historical"
        }
        
        # Check Twelve Data
        status["sources"]["twelve_data"] = {
            "configured": bool(data_config.TWELVE_DATA_API_KEY),
            "type": "realtime + historical"
        }
        
        # Check MT5
        mt5_result = self.mt5.get_trades()
        status["sources"]["mt5"] = {
            "configured": True,
            "online": mt5_result["success"],
            "type": "real trades"
        }
        
        # Check Dukascopy
        duka_result = self.dukascopy.get_available_data()
        status["sources"]["dukascopy"] = {
            "configured": os.path.exists(data_config.DUKASCOPY_DATA_PATH),
            "type": "historical ticks",
            "available_pairs": list(duka_result.get("data", {}).get("available", {}).keys()) if duka_result["success"] else []
        }
        
        return status


if __name__ == "__main__":
    print("=" * 60)
    print("TRADING AI DATA SOURCES - REAL DATA ONLY")
    print("=" * 60)
    
    manager = UnifiedDataManager()
    status = manager.get_all_sources_status()
    
    print(json.dumps(status, indent=2))

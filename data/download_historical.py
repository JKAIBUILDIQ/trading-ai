#!/usr/bin/env python3
"""
Historical Data Downloader
Downloads free historical forex data from Dukascopy and other sources.

Sources:
- Dukascopy (free tick data)
- MT5 API (local broker data)

NO RANDOM DATA - All data from real market sources.
"""

import os
import gzip
import struct
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataDownloader")

# Paths
DATA_DIR = Path(__file__).parent
HISTORICAL_DIR = DATA_DIR / "historical"
TRAINING_DIR = DATA_DIR / "training"
HISTORICAL_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)

# Dukascopy configuration
DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# Symbol mappings (Dukascopy uses different naming)
SYMBOL_MAP = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
}

# Point values for converting prices
POINT_VALUES = {
    "EURUSD": 100000,
    "GBPUSD": 100000,
    "USDJPY": 1000,
    "USDCHF": 100000,
    "AUDUSD": 100000,
    "USDCAD": 100000,
    "NZDUSD": 100000,
    "XAUUSD": 1000,
    "XAGUSD": 1000,
}


class DukascopyDownloader:
    """
    Downloads tick data from Dukascopy.
    
    Dukascopy provides free historical tick data for major forex pairs.
    Data is in bi5 format (compressed binary).
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.dukascopy_symbol = SYMBOL_MAP.get(symbol, symbol)
        self.point_value = POINT_VALUES.get(symbol, 100000)
        
    def download_hour(self, dt: datetime) -> Optional[pd.DataFrame]:
        """
        Download one hour of tick data.
        
        Returns DataFrame with columns: [time, bid, ask, bid_volume, ask_volume]
        """
        url = DUKASCOPY_URL.format(
            symbol=self.dukascopy_symbol,
            year=dt.year,
            month=dt.month - 1,  # Dukascopy uses 0-indexed months
            day=dt.day,
            hour=dt.hour
        )
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                # No data for this hour (weekend, holiday)
                return None
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
            
            # Decompress bi5 data
            try:
                decompressed = gzip.decompress(response.content)
            except:
                # Not gzipped or empty
                return None
            
            if len(decompressed) == 0:
                return None
            
            # Parse binary data
            # Each tick is 20 bytes: time(4) + bid(4) + ask(4) + bid_vol(4) + ask_vol(4)
            num_ticks = len(decompressed) // 20
            
            if num_ticks == 0:
                return None
            
            ticks = []
            for i in range(num_ticks):
                offset = i * 20
                tick_data = struct.unpack('>IIIff', decompressed[offset:offset+20])
                
                time_ms = tick_data[0]
                bid = tick_data[1] / self.point_value
                ask = tick_data[2] / self.point_value
                bid_vol = tick_data[3]
                ask_vol = tick_data[4]
                
                tick_time = dt.replace(minute=0, second=0, microsecond=0) + timedelta(milliseconds=time_ms)
                
                ticks.append({
                    'time': tick_time,
                    'bid': bid,
                    'ask': ask,
                    'bid_volume': bid_vol,
                    'ask_volume': ask_vol
                })
            
            return pd.DataFrame(ticks)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def download_day(self, date: datetime) -> pd.DataFrame:
        """
        Download one day of tick data.
        """
        all_ticks = []
        
        for hour in range(24):
            dt = date.replace(hour=hour, minute=0, second=0, microsecond=0)
            df = self.download_hour(dt)
            
            if df is not None and len(df) > 0:
                all_ticks.append(df)
        
        if all_ticks:
            return pd.concat(all_ticks, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def download_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Download tick data for a date range.
        """
        logger.info(f"Downloading {self.symbol} from {start_date.date()} to {end_date.date()}")
        
        all_data = []
        current = start_date
        
        while current <= end_date:
            df = self.download_day(current)
            
            if len(df) > 0:
                all_data.append(df)
                logger.info(f"  {current.date()}: {len(df)} ticks")
            else:
                logger.debug(f"  {current.date()}: No data (weekend/holiday)")
            
            current += timedelta(days=1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total: {len(result)} ticks")
            return result
        else:
            return pd.DataFrame()


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str = "1H") -> pd.DataFrame:
    """
    Convert tick data to OHLCV candles.
    
    Args:
        ticks: DataFrame with columns [time, bid, ask, ...]
        timeframe: Candle timeframe (1T, 5T, 15T, 30T, 1H, 4H, 1D)
    
    Returns:
        DataFrame with columns [time, open, high, low, close, volume]
    """
    if len(ticks) == 0:
        return pd.DataFrame()
    
    # Use mid price
    ticks = ticks.copy()
    ticks['price'] = (ticks['bid'] + ticks['ask']) / 2
    ticks['volume'] = ticks.get('bid_volume', 0) + ticks.get('ask_volume', 0)
    
    # Set time as index
    ticks['time'] = pd.to_datetime(ticks['time'])
    ticks.set_index('time', inplace=True)
    
    # Resample to candles
    ohlcv = ticks['price'].resample(timeframe).ohlc()
    ohlcv['volume'] = ticks['volume'].resample(timeframe).sum()
    
    # Clean up column names
    ohlcv = ohlcv.reset_index()
    ohlcv.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # Drop rows with NaN
    ohlcv = ohlcv.dropna()
    
    return ohlcv


def download_training_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1H"
) -> dict:
    """
    Download and prepare training data for multiple symbols.
    
    Returns:
        Dictionary mapping symbol to OHLCV DataFrame
    """
    results = {}
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {symbol}")
        logger.info(f"{'='*50}")
        
        downloader = DukascopyDownloader(symbol)
        ticks = downloader.download_range(start_date, end_date)
        
        if len(ticks) > 0:
            ohlcv = ticks_to_ohlcv(ticks, timeframe)
            
            # Save to file
            filename = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = TRAINING_DIR / filename
            ohlcv.to_csv(filepath, index=False)
            
            logger.info(f"Saved {len(ohlcv)} candles to {filepath}")
            results[symbol] = ohlcv
        else:
            logger.warning(f"No data downloaded for {symbol}")
            results[symbol] = pd.DataFrame()
    
    return results


def download_from_mt5(symbol: str, timeframe: str = "H1", num_candles: int = 1000) -> Optional[pd.DataFrame]:
    """
    Download data from local MT5 API.
    
    This connects to your running MT5 terminal via the API on port 8085.
    """
    try:
        url = f"http://localhost:8085/history/{symbol}"
        params = {
            "timeframe": timeframe,
            "count": num_candles
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} candles from MT5 API")
            return df
        else:
            logger.warning(f"MT5 API returned {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"MT5 API error: {e}")
        return None


def prepare_cnn_training_data(ohlcv: pd.DataFrame, window: int = 100) -> List[np.ndarray]:
    """
    Prepare OHLCV data for CNN training.
    Creates sliding windows of candle data.
    
    Returns list of numpy arrays, each of shape (window, 5)
    """
    if len(ohlcv) < window:
        return []
    
    samples = []
    
    for i in range(len(ohlcv) - window):
        sample = ohlcv.iloc[i:i+window][['open', 'high', 'low', 'close', 'volume']].values
        samples.append(sample)
    
    return samples


def prepare_lstm_training_data(ohlcv: pd.DataFrame, window: int = 100, horizon: int = 12) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Prepare OHLCV data for LSTM training.
    Creates (features, labels) pairs where label is future price direction.
    
    Returns:
        features: List of numpy arrays, each of shape (window, 5)
        labels: List of dicts with direction, magnitude
    """
    if len(ohlcv) < window + horizon:
        return [], []
    
    features = []
    labels = []
    
    for i in range(len(ohlcv) - window - horizon):
        # Features: last `window` candles
        sample = ohlcv.iloc[i:i+window][['open', 'high', 'low', 'close', 'volume']].values
        features.append(sample)
        
        # Label: price direction over next `horizon` candles
        current_close = ohlcv.iloc[i+window-1]['close']
        future_close = ohlcv.iloc[i+window+horizon-1]['close']
        
        change = (future_close - current_close) / current_close
        change_pips = change * 10000  # Convert to pips
        
        if change > 0.0010:  # > 10 pips up
            direction = 'UP'
        elif change < -0.0010:  # > 10 pips down
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        labels.append({
            'direction': direction,
            'magnitude_pips': abs(change_pips)
        })
    
    return features, labels


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical Data Downloader")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol to download")
    parser.add_argument("--days", type=int, default=7, help="Number of days to download")
    parser.add_argument("--timeframe", type=str, default="1H", help="Candle timeframe")
    parser.add_argument("--mt5", action="store_true", help="Use MT5 API instead of Dukascopy")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HISTORICAL DATA DOWNLOADER")
    print("=" * 60)
    
    if args.mt5:
        # Download from MT5 API
        print(f"Downloading {args.symbol} from MT5 API...")
        df = download_from_mt5(args.symbol, args.timeframe, num_candles=1000)
        
        if df is not None:
            print(f"Downloaded {len(df)} candles")
            print(df.head())
        else:
            print("Failed to download from MT5")
    else:
        # Download from Dukascopy
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        print(f"Downloading {args.symbol} from Dukascopy...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        downloader = DukascopyDownloader(args.symbol)
        ticks = downloader.download_range(start_date, end_date)
        
        if len(ticks) > 0:
            print(f"\nDownloaded {len(ticks)} ticks")
            
            # Convert to candles
            ohlcv = ticks_to_ohlcv(ticks, args.timeframe)
            print(f"Converted to {len(ohlcv)} {args.timeframe} candles")
            
            # Save
            filename = f"{args.symbol}_{args.timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = TRAINING_DIR / filename
            ohlcv.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
            
            print("\nSample data:")
            print(ohlcv.head(10))
        else:
            print("No data downloaded (check if market was open)")
    
    print("\n" + "=" * 60)

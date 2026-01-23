#!/usr/bin/env python3
"""
Trading AI Configuration
All API keys and settings in one place

RULE: NO fake data. Every source must be traceable.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataSourceConfig:
    """Configuration for data sources - ALL REAL APIs"""
    
    # Polygon.io (Real-time + Historical)
    # Sign up: https://polygon.io/ ($29/mo for stocks, $199/mo for forex)
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    
    # Twelve Data (Alternative - $79/mo for forex)
    # Sign up: https://twelvedata.com/
    TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
    
    # Alpha Vantage (Free tier available - 5 calls/min)
    # Sign up: https://www.alphavantage.co/
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # Dukascopy (FREE historical tick data)
    # Download: https://www.dukascopy.com/swiss/english/marketwatch/historical/
    DUKASCOPY_DATA_PATH: str = os.getenv("DUKASCOPY_DATA_PATH", "/home/jbot/trading_ai/data/dukascopy")
    
    # MT5 Connection (Your REAL trading data)
    MT5_API_URL: str = os.getenv("MT5_API_URL", "http://localhost:8085")


@dataclass
class DatabaseConfig:
    """PostgreSQL configuration for tick data storage"""
    
    HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    DATABASE: str = os.getenv("POSTGRES_DB", "trading_ai")
    USER: str = os.getenv("POSTGRES_USER", "jbot")
    PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"


@dataclass
class ModelConfig:
    """ML/RL Model configuration"""
    
    # Device for training
    DEVICE: str = "cuda"  # H100 GPU
    
    # RL Training parameters
    TOTAL_TIMESTEPS: int = 10_000_000
    LEARNING_RATE: float = 3e-4
    BATCH_SIZE: int = 2048
    N_EPOCHS: int = 10
    
    # Model save path
    MODELS_PATH: str = "/home/jbot/trading_ai/models"
    
    # Feature dimensions
    STATE_DIM: int = 50  # Price features + indicators + position info
    ACTION_DIM: int = 4   # HOLD, BUY, SELL, CLOSE


@dataclass
class TradingConfig:
    """Trading parameters"""
    
    # Symbols to trade
    FOREX_PAIRS: list = None
    CRYPTO_PAIRS: list = None
    
    # Risk management
    MAX_POSITION_SIZE: float = 0.02  # 2% of capital per trade
    MAX_DRAWDOWN: float = 0.10       # 10% max drawdown before stop
    STOP_LOSS_PIPS: int = 30
    TAKE_PROFIT_PIPS: int = 60
    
    def __post_init__(self):
        if self.FOREX_PAIRS is None:
            self.FOREX_PAIRS = [
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
                "AUDUSD", "NZDUSD", "USDCAD", "EURJPY"
            ]
        if self.CRYPTO_PAIRS is None:
            self.CRYPTO_PAIRS = [
                "BTCUSD", "ETHUSD", "SOLUSD"
            ]


# Singleton instances
data_config = DataSourceConfig()
db_config = DatabaseConfig()
model_config = ModelConfig()
trading_config = TradingConfig()


def verify_config():
    """Verify all configurations are set correctly"""
    issues = []
    
    # Check for API keys
    if not data_config.POLYGON_API_KEY and not data_config.TWELVE_DATA_API_KEY:
        issues.append("WARNING: No market data API key configured. Set POLYGON_API_KEY or TWELVE_DATA_API_KEY")
    
    # Check MT5 connection
    import requests
    try:
        resp = requests.get(f"{data_config.MT5_API_URL}/health", timeout=5)
        if resp.status_code != 200:
            issues.append(f"WARNING: MT5 API not healthy at {data_config.MT5_API_URL}")
    except:
        issues.append(f"WARNING: Cannot connect to MT5 API at {data_config.MT5_API_URL}")
    
    # Check GPU
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("WARNING: CUDA not available. Training will be slow on CPU.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {gpu_name}")
    except ImportError:
        issues.append("WARNING: PyTorch not installed")
    
    return issues


if __name__ == "__main__":
    print("=" * 60)
    print("TRADING AI CONFIGURATION CHECK")
    print("=" * 60)
    issues = verify_config()
    
    if issues:
        print("\n⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All configurations valid!")

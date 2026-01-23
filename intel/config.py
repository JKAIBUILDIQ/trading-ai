#!/usr/bin/env python3
"""
MQL5 Intel Scraper Configuration
NO RANDOM DATA - All settings are real values
"""

import os
from pathlib import Path

# Directories
INTEL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = INTEL_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Output files
MQL5_SIGNALS_FILE = INTEL_DIR / "mql5_signals.json"
CONSENSUS_FILE = INTEL_DIR / "consensus.json"
NEO_INTEL_FILE = Path("/tmp/neo_intel.json")

# MQL5 URLs
MQL5_BASE_URL = "https://www.mql5.com"
MQL5_SIGNALS_URL = f"{MQL5_BASE_URL}/en/signals"
MQL5_SIGNAL_URL = f"{MQL5_BASE_URL}/en/signals/{{signal_id}}"

# Scraping settings
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2  # Seconds between requests (be nice to MQL5)
MAX_RETRIES = 3

# Signal filters
MIN_GROWTH_PCT = 200  # Minimum growth percentage (lowered from 500 to capture more signals)
MIN_HISTORY_WEEKS = 24  # Minimum 6 months history
MIN_ALGO_TRADING_PCT = 80  # Minimum algo trading percentage
MAX_DRAWDOWN_PCT = 50  # Maximum drawdown
TOP_SIGNALS_COUNT = 20  # Number of top signals to track

# Consensus detection
CONSENSUS_THRESHOLD = 3  # Minimum traders for consensus
CONSENSUS_WINDOW_HOURS = 2  # Trades within this window count as consensus
CONFIDENCE_BOOST = 15  # NEO confidence boost when consensus exists

# Update schedule
UPDATE_INTERVAL_MINUTES = 15

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [MQL5] %(levelname)s: %(message)s"

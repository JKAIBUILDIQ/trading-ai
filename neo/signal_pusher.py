#!/usr/bin/env python3
"""
NEO Signal Pusher
Pushes NEO's signals to MT5 API for Ghost Commander to read.

THE BRIDGE:
NEO (H100) → signal_pusher → MT5 API → Ghost Commander → Crellastein Fleet

NO RANDOM DATA - Only pushes actual NEO decisions
"""

import json
import requests
import time
import os
from datetime import datetime
from typing import Dict, Optional
import hashlib

# Configuration
MT5_API_URL = os.environ.get("MT5_API_URL", "http://localhost:8085")
SIGNAL_FILE = "/tmp/neo_signal.json"
PUSH_INTERVAL = 5  # Check every 5 seconds
RETRY_INTERVAL = 30  # Wait 30s on connection error


class SignalPusher:
    """
    Pushes NEO signals to MT5 API.
    Only pushes when signal changes (to avoid spam).
    """
    
    def __init__(self, mt5_api_url: str = MT5_API_URL):
        self.api_url = mt5_api_url
        self.last_signal_hash = None
        self.push_count = 0
        self.error_count = 0
        self.last_push_time = None
    
    def _hash_signal(self, signal: Dict) -> str:
        """Create hash of signal for change detection."""
        # Hash key fields that indicate a new signal
        key_data = {
            "signal_id": signal.get("signal_id"),
            "action": signal.get("action"),
            "symbol": signal.get("trade", {}).get("symbol"),
            "direction": signal.get("trade", {}).get("direction"),
            "timestamp": signal.get("timestamp")
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def read_signal(self) -> Optional[Dict]:
        """Read current signal from NEO's output file."""
        try:
            if os.path.exists(SIGNAL_FILE):
                with open(SIGNAL_FILE) as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print(f"[{datetime.now()}] ⚠️ Invalid JSON in signal file")
        except Exception as e:
            print(f"[{datetime.now()}] ⚠️ Error reading signal: {e}")
        return None
    
    def push_signal(self, signal: Dict) -> bool:
        """Push signal to MT5 API."""
        try:
            # Add push metadata
            push_data = signal.copy()
            push_data["pushed_at"] = datetime.utcnow().isoformat()
            push_data["push_source"] = "NEO_H100"
            
            response = requests.post(
                f"{self.api_url}/neo/signal",
                json=push_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.push_count += 1
                self.last_push_time = datetime.now()
                return True
            else:
                print(f"[{datetime.now()}] ⚠️ Push failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.error_count += 1
            print(f"[{datetime.now()}] ❌ MT5 API not reachable at {self.api_url}")
            return False
        except Exception as e:
            self.error_count += 1
            print(f"[{datetime.now()}] ❌ Push error: {e}")
            return False
    
    def check_api_health(self) -> bool:
        """Check if MT5 API is reachable."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run(self):
        """Main loop - watch for signals and push changes."""
        print("=" * 60)
        print("NEO SIGNAL PUSHER")
        print("=" * 60)
        print(f"Signal file: {SIGNAL_FILE}")
        print(f"MT5 API: {self.api_url}")
        print(f"Check interval: {PUSH_INTERVAL}s")
        print("=" * 60)
        
        # Check API health
        if self.check_api_health():
            print(f"✅ MT5 API is reachable")
        else:
            print(f"⚠️ MT5 API not responding - will retry")
        
        print("")
        print("Watching for NEO signals...")
        print("")
        
        while True:
            try:
                signal = self.read_signal()
                
                if signal:
                    signal_hash = self._hash_signal(signal)
                    
                    # Only push if signal changed
                    if signal_hash != self.last_signal_hash:
                        action = signal.get("action", "UNKNOWN")
                        trade = signal.get("trade", {})
                        symbol = trade.get("symbol", "")
                        direction = trade.get("direction", "")
                        confidence = signal.get("metadata", {}).get("confidence", 0)
                        
                        print(f"[{datetime.now()}] 📡 New signal detected:")
                        print(f"    Action: {action}")
                        if symbol:
                            print(f"    Trade: {symbol} {direction}")
                            print(f"    Confidence: {confidence}%")
                        
                        if self.push_signal(signal):
                            print(f"    ✅ Pushed to MT5 API")
                            self.last_signal_hash = signal_hash
                        else:
                            print(f"    ❌ Push failed - will retry")
                
                time.sleep(PUSH_INTERVAL)
                
            except KeyboardInterrupt:
                print("\nStopping signal pusher...")
                break
            except Exception as e:
                print(f"[{datetime.now()}] Error: {e}")
                time.sleep(RETRY_INTERVAL)
        
        print(f"\nStats: {self.push_count} pushed, {self.error_count} errors")


def test_pusher():
    """Test the signal pusher."""
    print("Testing Signal Pusher...")
    
    pusher = SignalPusher()
    
    # Read current signal
    signal = pusher.read_signal()
    if signal:
        print(f"✅ Current signal: {signal.get('action', 'UNKNOWN')}")
        print(f"   Signal ID: {signal.get('signal_id', 'N/A')}")
    else:
        print("⚠️ No signal file found")
    
    # Check API
    if pusher.check_api_health():
        print(f"✅ MT5 API is reachable")
    else:
        print(f"⚠️ MT5 API not responding at {pusher.api_url}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test connectivity")
    parser.add_argument("--api", type=str, default=MT5_API_URL, help="MT5 API URL")
    args = parser.parse_args()
    
    if args.test:
        test_pusher()
    else:
        pusher = SignalPusher(args.api)
        pusher.run()

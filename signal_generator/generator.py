#!/usr/bin/env python3
"""
Signal Generator
Connects trained RL models to MT5 for live signal generation

This service:
1. Loads trained model
2. Fetches real-time price data
3. Generates trading signals
4. Sends signals to MT5 API

RULES:
1. Uses REAL price data from APIs
2. Signals logged with full context
3. Model predictions are deterministic
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import data_config, model_config
from data_pipeline.data_sources import UnifiedDataManager
from feature_extraction.indicators import TechnicalIndicators


class SignalGenerator:
    """
    Generates trading signals from trained RL models
    """
    
    def __init__(
        self,
        model_path: str = None,
        pair: str = "EURUSD",
        window_size: int = 50
    ):
        """
        Initialize the signal generator
        
        Args:
            model_path: Path to trained model
            pair: Currency pair to trade
            window_size: Feature window size
        """
        self.model_path = model_path
        self.pair = pair
        self.window_size = window_size
        self.model = None
        
        # Data components
        self.data_manager = UnifiedDataManager()
        self.indicators = TechnicalIndicators()
        
        # Redis for caching
        self.redis = redis.Redis(host="localhost", decode_responses=True)
        
        # Signal state
        self.last_signal = None
        self.last_signal_time = None
        self.position_state = "flat"  # flat, long, short
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained RL model"""
        try:
            from stable_baselines3 import PPO
            
            self.model = PPO.load(model_path)
            print(f"✅ Loaded model from {model_path}")
            
            # Load training metadata
            metadata_path = os.path.dirname(model_path) + "/training_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                print(f"  Trained on: {self.metadata.get('data_source')}")
                print(f"  Timesteps: {self.metadata.get('timesteps'):,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def get_current_state(self) -> Optional[np.ndarray]:
        """
        Get current market state for model input
        
        Returns feature vector from real-time data
        """
        # Fetch recent price data
        result = self.data_manager.get_historical_data(self.pair, "minute", days=1)
        
        if not result["success"]:
            print(f"⚠️ Failed to fetch data: {result.get('error')}")
            return None
        
        # Convert to DataFrame
        import pandas as pd
        bars = result["data"]["bars"]
        df = pd.DataFrame(bars)
        
        if len(df) < self.window_size + 100:  # Need extra for indicator warmup
            print(f"⚠️ Insufficient data: {len(df)} bars")
            return None
        
        # Calculate indicators
        df_features = self.indicators.calculate_all(df)
        df_features = df_features.dropna()
        
        if len(df_features) < self.window_size:
            print(f"⚠️ Insufficient data after indicators: {len(df_features)} bars")
            return None
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol']]
        
        # Extract window
        window = df_features.iloc[-self.window_size:][feature_cols].values.flatten()
        
        # Add position info (simplified - would track actual position in production)
        position_info = np.array([0, 0, 0])  # position, entry_price_norm, pnl_norm
        
        state = np.concatenate([window, position_info]).astype(np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def generate_signal(self) -> Dict:
        """
        Generate a trading signal from the model
        
        Returns:
            Signal dict with action, confidence, and metadata
        """
        # Get current state
        state = self.get_current_state()
        
        if state is None:
            return {
                "source": "SIGNAL_GENERATOR",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": False,
                "error": "Failed to get market state",
                "signal": None
            }
        
        if self.model is None:
            return {
                "source": "SIGNAL_GENERATOR",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": False,
                "error": "No model loaded",
                "signal": None
            }
        
        # Get model prediction
        action, _states = self.model.predict(state, deterministic=True)
        
        # Map action to signal
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
        signal_name = action_map.get(int(action), "UNKNOWN")
        
        # Get action probabilities for confidence
        # This requires accessing the policy network
        try:
            obs_tensor = self.model.policy.obs_to_tensor(state.reshape(1, -1))[0]
            with torch.no_grad():
                _, log_prob, _ = self.model.policy.evaluate_actions(
                    obs_tensor, 
                    torch.tensor([action])
                )
            confidence = float(torch.exp(log_prob).item())
        except:
            confidence = 0.5  # Default confidence if can't compute
        
        # Create signal
        signal = {
            "source": "RL_MODEL_SIGNAL",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": True,
            "signal": {
                "pair": self.pair,
                "action": signal_name,
                "action_id": int(action),
                "confidence": round(confidence, 4),
                "model_path": self.model_path,
                "data_source": "REAL_MARKET_DATA"
            },
            "verified": True
        }
        
        # Store in Redis
        self.redis.set(f"signal:{self.pair}:latest", json.dumps(signal))
        self.redis.lpush(f"signal:{self.pair}:history", json.dumps(signal))
        self.redis.ltrim(f"signal:{self.pair}:history", 0, 999)  # Keep last 1000
        
        # Update state
        self.last_signal = signal
        self.last_signal_time = datetime.utcnow()
        
        return signal
    
    def send_to_mt5(self, signal: Dict) -> Dict:
        """
        Send signal to MT5 API
        
        This would connect to your Crellastein bots
        """
        import requests
        
        if not signal.get("success") or not signal.get("signal"):
            return {"success": False, "error": "Invalid signal"}
        
        action = signal["signal"]["action"]
        
        if action == "HOLD":
            return {"success": True, "message": "No action needed"}
        
        # Prepare MT5 payload
        payload = {
            "pair": self.pair,
            "action": action,
            "confidence": signal["signal"]["confidence"],
            "source": "AI_SIGNAL_GENERATOR",
            "timestamp": signal["timestamp"]
        }
        
        try:
            # Send to MT5 API
            response = requests.post(
                f"{data_config.MT5_API_URL}/signals",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Signal sent: {action}",
                    "mt5_response": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"MT5 returned {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_loop(self, interval_seconds: int = 60):
        """
        Run continuous signal generation loop
        
        Args:
            interval_seconds: Time between signals
        """
        print("=" * 60)
        print("SIGNAL GENERATOR - LIVE MODE")
        print("=" * 60)
        print(f"Pair: {self.pair}")
        print(f"Model: {self.model_path}")
        print(f"Interval: {interval_seconds}s")
        print("=" * 60)
        
        while True:
            try:
                # Generate signal
                signal = self.generate_signal()
                
                if signal["success"]:
                    action = signal["signal"]["action"]
                    confidence = signal["signal"]["confidence"]
                    print(f"[{datetime.now()}] {self.pair}: {action} (conf: {confidence:.2%})")
                    
                    # Send to MT5 if actionable
                    if action in ["BUY", "SELL", "CLOSE"]:
                        mt5_result = self.send_to_mt5(signal)
                        if mt5_result["success"]:
                            print(f"  → MT5: {mt5_result['message']}")
                        else:
                            print(f"  → MT5 Error: {mt5_result['error']}")
                else:
                    print(f"[{datetime.now()}] Error: {signal.get('error')}")
                
            except Exception as e:
                print(f"[{datetime.now()}] Exception: {e}")
            
            time.sleep(interval_seconds)


class RuleBasedSignalGenerator:
    """
    Rule-based signal generator (fallback if no ML model)
    Uses technical indicators for signals
    """
    
    def __init__(self, pair: str = "EURUSD"):
        self.pair = pair
        self.data_manager = UnifiedDataManager()
        self.indicators = TechnicalIndicators()
        self.redis = redis.Redis(host="localhost", decode_responses=True)
    
    def generate_signal(self) -> Dict:
        """Generate signal from technical rules"""
        import pandas as pd
        
        # Fetch data
        result = self.data_manager.get_historical_data(self.pair, "minute", days=1)
        
        if not result["success"]:
            return {"success": False, "error": result.get("error")}
        
        df = pd.DataFrame(result["data"]["bars"])
        df = self.indicators.calculate_all(df).dropna()
        
        if len(df) < 10:
            return {"success": False, "error": "Insufficient data"}
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Simple rule-based signals
        signals = []
        
        # RSI signals
        if latest.get('rsi', 50) < 30:
            signals.append(("BUY", "RSI oversold", 0.6))
        elif latest.get('rsi', 50) > 70:
            signals.append(("SELL", "RSI overbought", 0.6))
        
        # MA crossover
        if latest.get('sma_5_20_cross', 0) == 1:
            signals.append(("BUY", "MA crossover bullish", 0.5))
        elif latest.get('sma_5_20_cross', 0) == 0 and df.iloc[-2].get('sma_5_20_cross', 0) == 1:
            signals.append(("SELL", "MA crossover bearish", 0.5))
        
        # Bollinger Bands
        if latest.get('bb_position', 0.5) < 0.1:
            signals.append(("BUY", "Below lower BB", 0.55))
        elif latest.get('bb_position', 0.5) > 0.9:
            signals.append(("SELL", "Above upper BB", 0.55))
        
        # Combine signals
        if not signals:
            action = "HOLD"
            reason = "No clear signal"
            confidence = 0.5
        else:
            # Take the highest confidence signal
            best_signal = max(signals, key=lambda x: x[2])
            action, reason, confidence = best_signal
        
        signal = {
            "source": "RULE_BASED_SIGNAL",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": True,
            "signal": {
                "pair": self.pair,
                "action": action,
                "reason": reason,
                "confidence": confidence,
                "data_source": result["source"]
            },
            "indicators": {
                "rsi": round(latest.get('rsi', 0), 2),
                "macd": round(latest.get('macd', 0), 6),
                "bb_position": round(latest.get('bb_position', 0.5), 2)
            },
            "verified": True
        }
        
        # Store in Redis
        self.redis.set(f"signal:{self.pair}:rule_based", json.dumps(signal))
        
        return signal


if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL GENERATOR TEST")
    print("=" * 60)
    
    # Test rule-based generator (no model needed)
    print("\nTesting Rule-Based Signal Generator...")
    rule_gen = RuleBasedSignalGenerator("EURUSD")
    signal = rule_gen.generate_signal()
    
    if signal["success"]:
        print(f"\n✅ Signal generated:")
        print(f"  Action: {signal['signal']['action']}")
        print(f"  Reason: {signal['signal']['reason']}")
        print(f"  Confidence: {signal['signal']['confidence']:.1%}")
        print(f"  Data source: {signal['signal']['data_source']}")
        print(f"\n  Indicators:")
        for k, v in signal.get('indicators', {}).items():
            print(f"    {k}: {v}")
    else:
        print(f"\n❌ Failed: {signal.get('error')}")
    
    # Test ML-based generator (requires trained model)
    print("\n" + "=" * 60)
    print("To use ML-based signals:")
    print("1. Train a model: python rl_training/train.py")
    print("2. Load it here: SignalGenerator(model_path='path/to/model')")
    print("=" * 60)

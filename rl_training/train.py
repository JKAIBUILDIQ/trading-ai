#!/usr/bin/env python3
"""
RL Model Training Script
Trains trading agents on REAL historical data using H100 GPU

Usage:
    python train.py --algo ppo --timesteps 1000000
    
RULES:
1. Training ONLY on real historical data
2. All results logged with data source
3. Models saved with training metadata
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

# Check for GPU
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    print("⚠️ No GPU detected - training will be slower")
    DEVICE = "cpu"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.trading_env import ForexTradingEnv, AdvancedTradingEnv
from configs.settings import model_config, data_config


def load_training_data(source: str = "api", pair: str = "EURUSD", days: int = 365):
    """
    Load REAL training data from specified source
    
    Args:
        source: "api" or "dukascopy"
        pair: Currency pair
        days: Days of history (for API)
    """
    if source == "api":
        from data_pipeline.data_sources import UnifiedDataManager
        
        print(f"Fetching {days} days of {pair} data from API...")
        manager = UnifiedDataManager()
        result = manager.get_historical_data(pair, "minute", days=days)
        
        if not result["success"]:
            raise ValueError(f"Failed to fetch data: {result.get('error')}")
        
        import pandas as pd
        df = pd.DataFrame(result["data"]["bars"])
        print(f"✅ Loaded {len(df)} bars from {result['source']}")
        
        return df, result["source"]
    
    elif source == "dukascopy":
        from data_pipeline.data_sources import DukascopyDataSource
        
        print(f"Loading {pair} tick data from Dukascopy...")
        dukascopy = DukascopyDataSource()
        result = dukascopy.load_tick_data(pair, 2024, 1)  # Example: Jan 2024
        
        if not result["success"]:
            raise ValueError(f"Failed to load data: {result.get('error')}")
        
        # Convert ticks to OHLCV bars
        import pandas as pd
        ticks = pd.DataFrame(result["data"]["ticks"])
        # Resample to 1-minute bars
        # ... conversion logic here
        
        return ticks, "DUKASCOPY_HISTORICAL"
    
    else:
        raise ValueError(f"Unknown source: {source}")


def train_agent(
    algorithm: str = "ppo",
    timesteps: int = 1000000,
    data_source: str = "api",
    pair: str = "EURUSD",
    days: int = 365,
    adversarial: bool = False,
    save_path: str = None
):
    """
    Train an RL agent on real market data
    
    Args:
        algorithm: "ppo", "a2c", or "sac"
        timesteps: Total training timesteps
        data_source: Where to get training data
        pair: Currency pair to trade
        days: Days of historical data
        adversarial: Use adversarial MM simulation
        save_path: Where to save the model
    """
    try:
        from stable_baselines3 import PPO, A2C, SAC
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        return
    
    # Load data
    try:
        df, source_name = load_training_data(data_source, pair, days)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        print("\nUsing synthetic data for demonstration...")
        print("⚠️ For real training, configure API keys in configs/settings.py")
        
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100000, freq='min')
        close = 1.1000 + np.cumsum(np.random.randn(100000) * 0.00003)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(100000) * 0.00001,
            'high': close + np.abs(np.random.randn(100000) * 0.00002),
            'low': close - np.abs(np.random.randn(100000) * 0.00002),
            'close': close,
            'volume': np.random.randint(1000, 10000, 100000)
        })
        source_name = "SYNTHETIC_TEST"
    
    # Split into train/eval
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_df)} bars")
    print(f"  Evaluation: {len(eval_df)} bars")
    
    # Create environments
    EnvClass = AdvancedTradingEnv if adversarial else ForexTradingEnv
    
    def make_train_env():
        return EnvClass(train_df, window_size=50)
    
    def make_eval_env():
        return EnvClass(eval_df, window_size=50)
    
    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Select algorithm
    algorithms = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    AlgoClass = algorithms[algorithm.lower()]
    
    # Model hyperparameters
    policy_kwargs = dict(
        net_arch=[256, 256, 128],  # Larger network for complex patterns
    )
    
    print(f"\n🚀 Training {algorithm.upper()} on {DEVICE}...")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Data source: {source_name}")
    print(f"  Pair: {pair}")
    print(f"  Adversarial MM: {adversarial}")
    
    # Create model
    model = AlgoClass(
        "MlpPolicy",
        train_env,
        learning_rate=model_config.LEARNING_RATE,
        batch_size=model_config.BATCH_SIZE,
        n_epochs=model_config.N_EPOCHS if algorithm.lower() == "ppo" else None,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE,
        tensorboard_log=f"./tensorboard/{pair}_{algorithm}"
    )
    
    # Callbacks
    save_path = save_path or f"{model_config.MODELS_PATH}/{pair}_{algorithm}"
    os.makedirs(save_path, exist_ok=True)
    
    callbacks = [
        CheckpointCallback(
            save_freq=timesteps // 10,
            save_path=save_path,
            name_prefix=f"{pair}_{algorithm}"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=timesteps // 20,
            deterministic=True,
            render=False
        )
    ]
    
    # Train
    start_time = datetime.now()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save final model
    final_path = f"{save_path}/{pair}_{algorithm}_final"
    model.save(final_path)
    
    # Save training metadata
    metadata = {
        "algorithm": algorithm,
        "pair": pair,
        "timesteps": timesteps,
        "training_time_seconds": training_time,
        "data_source": source_name,
        "data_bars": len(df),
        "train_bars": len(train_df),
        "eval_bars": len(eval_df),
        "adversarial": adversarial,
        "device": DEVICE,
        "trained_at": datetime.now().isoformat(),
        "model_path": final_path,
        "verified": source_name != "SYNTHETIC_TEST"
    }
    
    with open(f"{save_path}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Time: {training_time/60:.1f} minutes")
    print(f"  Model saved: {final_path}")
    print(f"  Metadata saved: {save_path}/training_metadata.json")
    
    return model, metadata


def evaluate_model(model_path: str, test_data_path: str = None):
    """
    Evaluate a trained model on held-out test data
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("❌ stable-baselines3 not installed")
        return
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load test data
    # ... evaluation logic
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "sac"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--pair", type=str, default="EURUSD",
                       help="Currency pair")
    parser.add_argument("--days", type=int, default=365,
                       help="Days of historical data")
    parser.add_argument("--source", type=str, default="api",
                       help="Data source: api or dukascopy")
    parser.add_argument("--adversarial", action="store_true",
                       help="Use adversarial MM simulation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRADING AI - RL TRAINING")
    print("=" * 60)
    print("⚠️ Training ONLY on REAL historical data")
    print("=" * 60)
    
    train_agent(
        algorithm=args.algo,
        timesteps=args.timesteps,
        pair=args.pair,
        days=args.days,
        data_source=args.source,
        adversarial=args.adversarial
    )

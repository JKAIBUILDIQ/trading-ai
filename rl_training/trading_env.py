#!/usr/bin/env python3
"""
Reinforcement Learning Trading Environment
Uses REAL historical data for training

Compatible with:
- Stable-Baselines3 (PPO, A2C, SAC)
- Custom RL algorithms

RULES:
1. Train ONLY on real historical data
2. No synthetic data generation during training
3. Results must be reproducible
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction.indicators import TechnicalIndicators, FeatureNormalizer
from configs.settings import model_config


class ForexTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for forex trading
    
    Observation Space: Price features + indicators + position info
    Action Space: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    
    CRITICAL: This environment uses REAL historical data
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        position_size_pct: float = 0.02,
        spread_pips: float = 1.5,
        commission_per_lot: float = 7.0,
        max_position: int = 1,
        window_size: int = 50,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the trading environment
        
        Args:
            data: DataFrame with OHLCV data (MUST be real historical data)
            initial_balance: Starting capital
            position_size_pct: Fraction of balance per trade
            spread_pips: Bid-ask spread in pips
            commission_per_lot: Commission per round trip
            max_position: Maximum simultaneous positions
            window_size: Number of bars to include in observation
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Validate data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Store parameters
        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.max_position = max_position
        self.window_size = window_size
        
        # Process data with indicators
        self.indicators = TechnicalIndicators()
        self.raw_data = data.copy()
        self.data = self.indicators.calculate_all(data)
        
        # Feature columns (excluding raw OHLCV and timestamp)
        self.feature_cols = [col for col in self.data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol']]
        
        # Normalize features
        self.normalizer = FeatureNormalizer()
        self.data = self.normalizer.fit_transform(self.data, self.feature_cols)
        
        # Drop NaN rows (from indicator calculations)
        self.data = self.data.dropna().reset_index(drop=True)
        
        # Define spaces
        # Observation: [window_size * num_features] + [position_info (3)]
        num_features = len(self.feature_cols)
        obs_dim = (window_size * num_features) + 3  # +3 for position, entry_price_norm, pnl_norm
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Actions: HOLD=0, BUY=1, SELL=2, CLOSE=3
        self.action_space = spaces.Discrete(4)
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_pnl = 0.0
        self.trades = []
        
        # For tracking
        self.equity_curve = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Get current price
        current_price = self.data['close'].iloc[self.current_step]
        
        # Calculate reward (before action)
        reward = 0.0
        
        # Execute action
        if action == 1 and self.position == 0:  # BUY
            self._open_position("long", current_price)
            
        elif action == 2 and self.position == 0:  # SELL
            self._open_position("short", current_price)
            
        elif action == 3 and self.position != 0:  # CLOSE
            reward = self._close_position(current_price)
        
        # Calculate unrealized P&L for reward shaping
        if self.position != 0:
            unrealized = self._calculate_unrealized_pnl(current_price)
            reward += unrealized * 0.01  # Small reward for profitable positions
        
        # Update equity curve
        equity = self.balance + self._calculate_unrealized_pnl(current_price)
        self.equity_curve.append(equity)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # End of data
        if self.current_step >= len(self.data) - 1:
            if self.position != 0:
                reward += self._close_position(current_price)
            truncated = True
        
        # Bankruptcy
        if equity <= 0:
            terminated = True
            reward = -10.0  # Large negative reward
        
        # Max drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - equity) / peak
        if drawdown > 0.3:  # 30% drawdown
            terminated = True
            reward = -5.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector"""
        # Get feature window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx][self.feature_cols].values
        features = window_data.flatten()
        
        # Add position info
        current_price = self.data['close'].iloc[self.current_step]
        entry_price_norm = (self.entry_price / current_price - 1) if self.entry_price > 0 else 0
        pnl_norm = self.total_pnl / self.initial_balance
        
        position_info = np.array([
            self.position,
            entry_price_norm,
            pnl_norm
        ])
        
        observation = np.concatenate([features, position_info]).astype(np.float32)
        
        # Handle NaN/Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "num_trades": len(self.trades),
            "equity": self.equity_curve[-1] if self.equity_curve else self.initial_balance
        }
    
    def _open_position(self, direction: str, price: float):
        """Open a new position"""
        # Apply spread
        if direction == "long":
            self.entry_price = price + (self.spread_pips * 0.0001 / 2)
            self.position = 1
        else:
            self.entry_price = price - (self.spread_pips * 0.0001 / 2)
            self.position = -1
        
        # Calculate position size
        self.position_size = (self.balance * self.position_size_pct) / (100000 * price)
    
    def _close_position(self, price: float) -> float:
        """Close current position and return P&L"""
        if self.position == 0:
            return 0.0
        
        # Apply spread
        if self.position == 1:  # Long
            exit_price = price - (self.spread_pips * 0.0001 / 2)
            pnl_pips = (exit_price - self.entry_price) / 0.0001
        else:  # Short
            exit_price = price + (self.spread_pips * 0.0001 / 2)
            pnl_pips = (self.entry_price - exit_price) / 0.0001
        
        # Calculate P&L
        pnl = pnl_pips * 10 * self.position_size  # $10 per pip per lot
        pnl -= self.commission_per_lot * self.position_size
        
        # Update state
        self.balance += pnl
        self.total_pnl += pnl
        
        # Record trade
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "direction": "long" if self.position == 1 else "short",
            "pnl": pnl
        })
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        
        return pnl / 100  # Normalize reward
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.position == 0:
            return 0.0
        
        if self.position == 1:  # Long
            pnl_pips = (current_price - self.entry_price) / 0.0001
        else:  # Short
            pnl_pips = (self.entry_price - current_price) / 0.0001
        
        return pnl_pips * 10 * self.position_size


class AdversarialMMSimulator:
    """
    Simulates Market Maker behavior to train against
    This creates a more challenging environment
    
    MM Goals:
    1. Collect spread (we pay it)
    2. Hunt stop losses (we hide them)
    3. Fade retail (we randomize)
    4. Front-run orders (we delay randomly)
    """
    
    def __init__(self, aggression: float = 0.5):
        """
        Args:
            aggression: How aggressive the MM is (0-1)
        """
        self.aggression = aggression
        self.visible_stops: List[float] = []
    
    def simulate_stop_hunt(self, price: float, direction: str) -> float:
        """
        Simulate MM pushing price to trigger stops
        
        Args:
            price: Current price
            direction: Direction of most retail positions
            
        Returns:
            Adjusted price after MM manipulation
        """
        # If retail is long, MM might push down to trigger stops
        # Then reverse and push up
        if direction == "long" and np.random.random() < self.aggression * 0.3:
            # Spike down then recover
            return price * (1 - 0.001 * self.aggression)
        elif direction == "short" and np.random.random() < self.aggression * 0.3:
            # Spike up then recover
            return price * (1 + 0.001 * self.aggression)
        
        return price
    
    def widen_spread(self, base_spread: float, volatility: float) -> float:
        """
        MM widens spread during high volatility or low liquidity
        
        Args:
            base_spread: Normal spread in pips
            volatility: Current market volatility
            
        Returns:
            Adjusted spread
        """
        # Spread widens with volatility
        volatility_factor = 1 + (volatility * self.aggression * 5)
        return base_spread * volatility_factor


class AdvancedTradingEnv(ForexTradingEnv):
    """
    Advanced trading environment with MM simulation
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        mm_aggression: float = 0.3,
        **kwargs
    ):
        super().__init__(data, **kwargs)
        self.mm = AdversarialMMSimulator(aggression=mm_aggression)
    
    def step(self, action: int):
        """Step with MM simulation"""
        # Get price before potential MM manipulation
        current_price = self.data['close'].iloc[self.current_step]
        
        # Simulate MM behavior if we have a position
        if self.position != 0:
            direction = "long" if self.position == 1 else "short"
            current_price = self.mm.simulate_stop_hunt(current_price, direction)
        
        # Dynamic spread based on volatility
        if 'atr_pct' in self.data.columns:
            volatility = abs(self.data['atr_pct'].iloc[self.current_step])
            self.spread_pips = self.mm.widen_spread(1.5, volatility)
        
        # Continue with normal step
        return super().step(action)


def create_training_data_from_api():
    """
    Create training environment from real API data
    
    This function fetches REAL data and creates an environment
    """
    from data_pipeline.data_sources import UnifiedDataManager
    
    print("Fetching real historical data...")
    manager = UnifiedDataManager()
    
    # Get historical data
    result = manager.get_historical_data("EURUSD", "minute", days=365)
    
    if not result["success"]:
        raise ValueError(f"Failed to fetch data: {result.get('error')}")
    
    # Convert to DataFrame
    bars = result["data"]["bars"]
    df = pd.DataFrame(bars)
    
    print(f"Loaded {len(df)} bars from {result['source']}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("RL TRADING ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test with synthetic data (for demonstration)
    # In production, use create_training_data_from_api()
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=10000, freq='h')
    close = 1.1000 + np.cumsum(np.random.randn(10000) * 0.0003)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(10000) * 0.0001,
        'high': close + np.abs(np.random.randn(10000) * 0.0002),
        'low': close - np.abs(np.random.randn(10000) * 0.0002),
        'close': close,
        'volume': np.random.randint(1000, 10000, 10000)
    })
    
    print("Creating environment...")
    env = ForexTradingEnv(test_data, window_size=30)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Test environment
    print("\nRunning random agent for 100 steps...")
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final balance: ${info['balance']:,.2f}")
    print(f"Total P&L: ${info['total_pnl']:,.2f}")
    print(f"Number of trades: {info['num_trades']}")
    
    print("\n⚠️ This was a TEST with synthetic data")
    print("For real training, use create_training_data_from_api()")

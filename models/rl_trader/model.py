#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent
Learns optimal entry/exit from trade outcomes using PPO.

State: Market features + current position + P&L
Actions: BUY, SELL, HOLD, CLOSE
Reward: Realized P&L (with drawdown penalty)

Algorithm: PPO (Proximal Policy Optimization)
Framework: Stable-Baselines3 compatible + custom PyTorch

NO RANDOM DATA in inference - All predictions from trained RL policy.
Note: Training uses controlled randomness for exploration, but inference is deterministic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


# State configuration
STATE_FEATURES = [
    # Market features (from LSTM/CNN)
    'price_change_1', 'price_change_5', 'price_change_20',
    'rsi_2', 'rsi_14',
    'bb_position',
    'atr_normalized',
    'volume_ratio',
    'trend_strength',  # ADX or similar
    'pattern_signal',  # From CNN (-1 bearish, 0 neutral, 1 bullish)
    'lstm_direction',  # From LSTM predictor
    'lstm_confidence',
    # Position features
    'has_position',  # 0 = no position, 1 = long, -1 = short
    'position_pnl',  # Current P&L normalized
    'position_duration',  # How long held (normalized)
    'distance_from_entry',  # Price vs entry (normalized)
    # Risk features
    'daily_pnl',  # Today's P&L
    'drawdown',  # Current drawdown
    'exposure_pct',  # Position size as % of capital
]

STATE_DIM = len(STATE_FEATURES)
NUM_ACTIONS = len(Action)


@dataclass
class TradingState:
    """Trading environment state."""
    market_features: np.ndarray  # Shape: (12,)
    position_features: np.ndarray  # Shape: (4,)
    risk_features: np.ndarray  # Shape: (3,)
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.concatenate([
            self.market_features,
            self.position_features,
            self.risk_features
        ])


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Policy network that outputs action probabilities
    Critic: Value network that estimates state value
    
    Architecture:
    - Shared feature extractor
    - Separate actor and critic heads
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch, state_dim)
        
        Returns:
            action_logits: (batch, action_dim)
            state_value: (batch, 1)
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from policy.
        
        Args:
            state: State tensor of shape (1, state_dim)
            deterministic: If True, take greedy action
        
        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, state_value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action]).item()
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, state_value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_logits, values = self.forward(states)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class RLTradingAgent:
    """
    Reinforcement Learning Trading Agent using PPO.
    
    Handles:
    - State preprocessing
    - Action selection
    - Policy inference
    
    For training, use train_rl.py which implements the full PPO algorithm.
    """
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = ActorCritic().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
        # State normalization (to be loaded with model)
        self.state_mean = None
        self.state_std = None
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.state_mean = checkpoint.get('state_mean')
            self.state_std = checkpoint.get('state_std')
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded RL model from {path}")
    
    def preprocess_state(self, state: TradingState) -> torch.Tensor:
        """Preprocess state for model input."""
        state_array = state.to_array()
        
        # Normalize if scalers available
        if self.state_mean is not None and self.state_std is not None:
            state_array = (state_array - self.state_mean) / (self.state_std + 1e-8)
        
        # Clip extreme values
        state_array = np.clip(state_array, -10, 10)
        
        return torch.from_numpy(state_array).float().unsqueeze(0).to(self.device)
    
    def act(self, state: TradingState, deterministic: bool = True) -> Dict:
        """
        Select action based on current state.
        
        Args:
            state: Current trading state
            deterministic: If True, use greedy action selection
        
        Returns:
            {
                "action": "BUY",
                "action_idx": 1,
                "confidence": 0.85,
                "value_estimate": 0.12,
                "action_probs": {"HOLD": 0.1, "BUY": 0.85, "SELL": 0.03, "CLOSE": 0.02}
            }
        """
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            action_logits, value = self.model.forward(state_tensor)
            probs = F.softmax(action_logits, dim=-1)[0]
            
            if deterministic:
                action_idx = torch.argmax(probs).item()
            else:
                action_idx = Categorical(probs).sample().item()
            
            action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE']
            
            return {
                "action": action_names[action_idx],
                "action_idx": action_idx,
                "confidence": probs[action_idx].item(),
                "value_estimate": value.item(),
                "action_probs": {
                    name: probs[i].item()
                    for i, name in enumerate(action_names)
                },
                "model": "rl_ppo_v1",
                "source": "reinforcement_learning"
            }
    
    def get_action_mask(self, has_position: bool) -> np.ndarray:
        """
        Get valid action mask based on current position.
        
        If no position: HOLD, BUY, SELL valid (CLOSE invalid)
        If has position: HOLD, CLOSE valid (BUY/SELL depends on direction)
        """
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        
        if not has_position:
            mask[Action.CLOSE] = 0  # Can't close if no position
        
        return mask


class TradingEnvironment:
    """
    Trading environment for RL training.
    Simulates trading with realistic constraints.
    """
    
    def __init__(
        self,
        initial_capital: float = 88000,
        max_position_pct: float = 0.05,
        max_daily_loss_pct: float = 0.03,
        spread_pips: float = 1.5,
        commission_per_lot: float = 7.0
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        
        self.reset()
    
    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self.capital = self.initial_capital
        self.position = None  # {'direction': 'LONG', 'entry_price': 1.1000, 'size': 0.1, 'entry_time': 0}
        self.daily_pnl = 0
        self.max_capital = self.initial_capital
        self.timestep = 0
        
        return self._get_state()
    
    def step(self, action: int, market_price: float, market_features: np.ndarray) -> Tuple[TradingState, float, bool, Dict]:
        """
        Take action in environment.
        
        Returns:
            next_state, reward, done, info
        """
        self.timestep += 1
        reward = 0
        info = {}
        
        # Execute action
        if action == Action.BUY and self.position is None:
            # Open long position
            position_value = self.capital * self.max_position_pct
            self.position = {
                'direction': 'LONG',
                'entry_price': market_price + self.spread_pips * 0.0001,  # Add spread
                'size': position_value,
                'entry_time': self.timestep
            }
            reward -= self.commission_per_lot * 0.1  # Commission
            info['action'] = 'opened_long'
            
        elif action == Action.SELL and self.position is None:
            # Open short position
            position_value = self.capital * self.max_position_pct
            self.position = {
                'direction': 'SHORT',
                'entry_price': market_price - self.spread_pips * 0.0001,
                'size': position_value,
                'entry_time': self.timestep
            }
            reward -= self.commission_per_lot * 0.1
            info['action'] = 'opened_short'
            
        elif action == Action.CLOSE and self.position is not None:
            # Close position
            pnl = self._calculate_pnl(market_price)
            self.capital += pnl
            self.daily_pnl += pnl
            reward = pnl  # Reward is realized P&L
            info['action'] = 'closed'
            info['pnl'] = pnl
            self.position = None
            
        else:
            # HOLD
            info['action'] = 'hold'
            
            # Small penalty for holding too long in a losing position
            if self.position is not None:
                unrealized_pnl = self._calculate_pnl(market_price)
                if unrealized_pnl < 0:
                    reward -= abs(unrealized_pnl) * 0.001  # Drawdown penalty
        
        # Update max capital
        self.max_capital = max(self.max_capital, self.capital)
        
        # Check termination conditions
        done = False
        
        # Daily loss limit
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            done = True
            reward -= 100  # Big penalty for hitting daily loss
            info['termination'] = 'daily_loss_limit'
        
        # Capital depleted
        if self.capital < self.initial_capital * 0.5:
            done = True
            reward -= 200  # Severe penalty for losing half
            info['termination'] = 'capital_depleted'
        
        return self._get_state(market_features), reward, done, info
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L for current position."""
        if self.position is None:
            return 0
        
        entry = self.position['entry_price']
        size = self.position['size']
        
        if self.position['direction'] == 'LONG':
            pips = (current_price - entry) / 0.0001
        else:
            pips = (entry - current_price) / 0.0001
        
        # Simplified: $10 per pip per standard lot
        return pips * (size / 100000) * 10
    
    def _get_state(self, market_features: np.ndarray = None) -> TradingState:
        """Get current state."""
        if market_features is None:
            market_features = np.zeros(12)
        
        # Position features
        if self.position is not None:
            has_position = 1 if self.position['direction'] == 'LONG' else -1
            position_pnl = 0  # Would need current price
            position_duration = (self.timestep - self.position['entry_time']) / 100  # Normalized
            distance_from_entry = 0
        else:
            has_position = 0
            position_pnl = 0
            position_duration = 0
            distance_from_entry = 0
        
        position_features = np.array([
            has_position,
            position_pnl,
            position_duration,
            distance_from_entry
        ])
        
        # Risk features
        drawdown = (self.max_capital - self.capital) / self.max_capital
        exposure = self.position['size'] / self.capital if self.position else 0
        
        risk_features = np.array([
            self.daily_pnl / self.initial_capital,  # Normalized daily P&L
            drawdown,
            exposure
        ])
        
        return TradingState(
            market_features=market_features,
            position_features=position_features,
            risk_features=risk_features
        )


def get_rl_agent(model_path: str = None) -> RLTradingAgent:
    """Factory function to get RL trading agent."""
    return RLTradingAgent(model_path=model_path)


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("RL TRADING AGENT - Model Test")
    print("=" * 60)
    
    # Create model
    model = ActorCritic()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    state = torch.randn(1, STATE_DIM)
    action_logits, value = model(state)
    
    print(f"State dim: {STATE_DIM}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test action selection
    action, log_prob, value_est = model.get_action(state, deterministic=True)
    print(f"Selected action: {Action(action).name}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value estimate: {value_est:.4f}")
    
    # Test agent
    print("\n--- Testing RLTradingAgent ---")
    agent = RLTradingAgent()
    
    test_state = TradingState(
        market_features=np.zeros(12),
        position_features=np.zeros(4),
        risk_features=np.zeros(3)
    )
    
    result = agent.act(test_state)
    print(f"Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Action probs: {result['action_probs']}")
    
    # Check GPU
    if torch.cuda.is_available():
        model = model.cuda()
        state = state.cuda()
        action_logits, value = model(state)
        print(f"\n✅ GPU inference working on {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 60)

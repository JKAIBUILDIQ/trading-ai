#!/usr/bin/env python3
"""Continuous RL Training for NEO"""
import sys
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from datetime import datetime
from pathlib import Path

from models.rl_trader.model import ActorCritic, TradingEnvironment, STATE_DIM, NUM_ACTIONS

print("=" * 60)
print("RL TRADING AGENT - CONTINUOUS TRAINING (PPO)")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Hyperparameters
EPISODES = 5000
MAX_STEPS = 500
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 64
UPDATE_EPOCHS = 4

# Create environment and model
env = TradingEnvironment(initial_capital=88000)
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training for {EPISODES} episodes...")

best_reward = float('-inf')
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    state_tensor = torch.from_numpy(state.to_array()).float().unsqueeze(0).to(device)
    
    episode_reward = 0
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    
    for step in range(MAX_STEPS):
        # Generate random market features for simulation
        market_features = np.random.randn(12) * 0.1
        
        with torch.no_grad():
            action_logits, value = model(state_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # Generate random price for simulation
        price = 1.1 + np.random.randn() * 0.01
        
        next_state, reward, done, info = env.step(action.item(), price, market_features)
        
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)
        
        episode_reward += reward
        state = next_state
        state_tensor = torch.from_numpy(state.to_array()).float().unsqueeze(0).to(device)
        
        if done:
            break
    
    episode_rewards.append(episode_reward)
    
    # PPO Update
    if len(states) > BATCH_SIZE:
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R * (1 - dones[i])
            returns.insert(0, R)
        
        returns = torch.tensor(returns).float().to(device)
        values_tensor = torch.cat(values).squeeze()
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        states_tensor = torch.cat(states)
        actions_tensor = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        
        for _ in range(UPDATE_EPOCHS):
            action_logits, new_values = model(states_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_tensor.squeeze())
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((new_values.squeeze() - returns) ** 2).mean()
            
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    
    # Log progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Capital: ${env.capital:,.2f}")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'avg_reward': avg_reward
            }, 'policy.pt')
            print(f"  → New best model saved!")

print(f"\n✅ RL training complete! Best avg reward: {best_reward:.2f}")

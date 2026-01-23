#!/bin/bash
# NEO Training Launcher - Runs all training on H100
# Usage: ./train_all.sh

LOG_DIR=~/trading_ai/logs
mkdir -p $LOG_DIR

echo "══════════════════════════════════════════════════════════════════"
echo "🚀 NEO TRAINING LAUNCHER"
echo "══════════════════════════════════════════════════════════════════"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Log directory: $LOG_DIR"
echo ""

# ===== PHASE 1: Generate Synthetic Training Data =====
echo "📊 PHASE 1: Generating synthetic training data..."
cd ~/trading_ai/models/cnn_patterns

# Generate data for CNN (faster than downloading)
python3 -c "
import sys
sys.path.insert(0, '../..')
from models.cnn_patterns.train_cnn import generate_synthetic_data, ChartPatternDataset
import numpy as np
import pickle
from pathlib import Path

print('Generating 10,000 synthetic chart patterns...')
ohlcv_data, labels = generate_synthetic_data(10000)

# Save for reuse
data_path = Path('../../data/training/synthetic_patterns.pkl')
with open(data_path, 'wb') as f:
    pickle.dump({'ohlcv': ohlcv_data, 'labels': labels}, f)

print(f'✅ Saved {len(ohlcv_data)} patterns to {data_path}')
print(f'   Pattern distribution: {np.bincount(labels)}')
"

if [ $? -eq 0 ]; then
    echo "✅ Synthetic data generated"
else
    echo "❌ Data generation failed"
    exit 1
fi

# ===== PHASE 2: Train CNN Pattern Detector =====
echo ""
echo "🧠 PHASE 2: Training CNN Pattern Detector (50 epochs)..."
echo "   Log: $LOG_DIR/cnn_training.log"

cd ~/trading_ai/models/cnn_patterns
nohup python3 train_cnn.py --synthetic --samples 10000 --epochs 50 --batch-size 64 > $LOG_DIR/cnn_training.log 2>&1 &
CNN_PID=$!
echo "   Started CNN training (PID: $CNN_PID)"

# Wait for CNN to finish (or run in parallel if you have enough GPU memory)
wait $CNN_PID
CNN_STATUS=$?

if [ $CNN_STATUS -eq 0 ]; then
    echo "✅ CNN training complete!"
    ls -la model.pt 2>/dev/null
else
    echo "⚠️ CNN training finished with status $CNN_STATUS"
fi

# ===== PHASE 3: Train LSTM Price Predictor =====
echo ""
echo "📈 PHASE 3: Training LSTM Price Predictor (100 epochs)..."
echo "   Log: $LOG_DIR/lstm_training.log"

cd ~/trading_ai/models/lstm_price

# Create LSTM training script
python3 << 'LSTM_TRAIN'
import sys
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json

from models.lstm_price.model import LSTMPricePredictor, NUM_FEATURES

print("=" * 60)
print("LSTM PRICE PREDICTOR TRAINING")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Generate synthetic training data for LSTM
print("\nGenerating synthetic LSTM training data...")
np.random.seed(42)

def generate_lstm_data(num_samples=8000, seq_len=100):
    """Generate synthetic price sequences with known patterns."""
    X = []
    y_direction = []
    y_magnitude = []
    
    for i in range(num_samples):
        # Generate base price series
        base = 1.1 + np.random.randn() * 0.05
        
        # Random trend
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.2, 0.5])
        trend_strength = np.random.uniform(0.0001, 0.0005)
        
        prices = np.zeros(seq_len + 12)  # Extra for label
        prices[0] = base
        
        for t in range(1, len(prices)):
            noise = np.random.randn() * 0.001
            prices[t] = prices[t-1] + trend * trend_strength + noise
        
        # Build feature matrix (simplified)
        features = np.zeros((seq_len, NUM_FEATURES))
        for t in range(seq_len):
            o = prices[t]
            h = prices[t] + abs(np.random.randn() * 0.0005)
            l = prices[t] - abs(np.random.randn() * 0.0005)
            c = prices[t] + np.random.randn() * 0.0002
            v = 1000
            
            features[t, 0] = o / c if c != 0 else 1
            features[t, 1] = h / c if c != 0 else 1
            features[t, 2] = l / c if c != 0 else 1
            features[t, 3] = 1.0
            features[t, 4] = 1.0
            # RSI placeholder
            features[t, 5] = 0.5 + np.random.randn() * 0.2
            features[t, 6] = 0.5 + np.random.randn() * 0.1
            # BB position
            features[t, 7] = 0.5 + trend * 0.3
            # Other features
            features[t, 8:] = np.random.randn(NUM_FEATURES - 8) * 0.1
        
        X.append(features)
        
        # Label: future direction
        future_change = (prices[seq_len + 11] - prices[seq_len - 1]) / prices[seq_len - 1]
        if future_change > 0.001:
            y_direction.append(2)  # UP
        elif future_change < -0.001:
            y_direction.append(0)  # DOWN
        else:
            y_direction.append(1)  # NEUTRAL
        
        y_magnitude.append(abs(future_change) * 10000)  # pips
    
    return np.array(X), np.array(y_direction), np.array(y_magnitude)

X, y_dir, y_mag = generate_lstm_data(8000)
print(f"Generated {len(X)} samples")
print(f"Direction distribution: DOWN={sum(y_dir==0)}, NEUTRAL={sum(y_dir==1)}, UP={sum(y_dir==2)}")

# Split
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
y_mag_train, y_mag_val = y_mag[:split], y_mag[split:]

# Create datasets
class LSTMDataset(Dataset):
    def __init__(self, X, y_dir, y_mag):
        self.X = torch.from_numpy(X).float()
        self.y_dir = torch.from_numpy(y_dir).long()
        self.y_mag = torch.from_numpy(y_mag).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_dir[idx], self.y_mag[idx]

train_dataset = LSTMDataset(X_train, y_dir_train, y_mag_train)
val_dataset = LSTMDataset(X_val, y_dir_val, y_mag_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Create model
model = LSTMPricePredictor().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
criterion_dir = nn.CrossEntropyLoss()
criterion_mag = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

best_val_acc = 0
epochs = 100

print(f"\nTraining for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_dir_batch, y_mag_batch in train_loader:
        X_batch = X_batch.to(device)
        y_dir_batch = y_dir_batch.to(device)
        y_mag_batch = y_mag_batch.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            dir_logits, mag_pred, conf_pred = model(X_batch)
            loss_dir = criterion_dir(dir_logits, y_dir_batch)
            loss_mag = criterion_mag(mag_pred.squeeze(), y_mag_batch)
            loss = loss_dir + 0.1 * loss_mag
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = dir_logits.max(1)
        train_total += y_dir_batch.size(0)
        train_correct += predicted.eq(y_dir_batch).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_dir_batch, y_mag_batch in val_loader:
            X_batch = X_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)
            
            dir_logits, _, _ = model(X_batch)
            _, predicted = dir_logits.max(1)
            val_total += y_dir_batch.size(0)
            val_correct += predicted.eq(y_dir_batch).sum().item()
    
    val_acc = 100. * val_correct / val_total
    scheduler.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc
        }, 'model.pt')

print(f"\n✅ LSTM training complete! Best val_acc: {best_val_acc:.1f}%")
print(f"Model saved to: model.pt")
LSTM_TRAIN

LSTM_STATUS=$?
if [ $LSTM_STATUS -eq 0 ]; then
    echo "✅ LSTM training complete!"
else
    echo "⚠️ LSTM training finished with status $LSTM_STATUS"
fi

# ===== PHASE 4: Start RL Training (Continuous) =====
echo ""
echo "🤖 PHASE 4: Starting RL Training (continuous)..."
echo "   Log: $LOG_DIR/rl_training.log"

cd ~/trading_ai/models/rl_trader

# Create RL training script
cat > train_rl_continuous.py << 'RLTRAIN'
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
RLTRAIN

# Start RL training in background
nohup python3 train_rl_continuous.py > $LOG_DIR/rl_training.log 2>&1 &
RL_PID=$!
echo "   Started RL training (PID: $RL_PID)"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "🎉 NEO TRAINING LAUNCHED!"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor training:"
echo "  tail -f $LOG_DIR/cnn_training.log"
echo "  tail -f $LOG_DIR/lstm_training.log"
echo "  tail -f $LOG_DIR/rl_training.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check trained models:"
echo "  ls -la ~/trading_ai/models/*/model.pt"
echo "  ls -la ~/trading_ai/models/rl_trader/policy.pt"
echo ""

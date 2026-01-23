#!/usr/bin/env python3
"""
CNN Pattern Detector Training Script
Trains the CNN on labeled chart pattern data.

Usage:
    python train_cnn.py --data /path/to/patterns --epochs 50

Requirements:
- Labeled pattern data (images or OHLCV + labels)
- GPU recommended for faster training

NO RANDOM SEEDS IN PRODUCTION - Training uses controlled randomness for
data augmentation and weight initialization, but inference is deterministic.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.cnn_patterns.model import ChartPatternCNN, PATTERNS, NUM_PATTERNS
from models.cnn_patterns.chart_renderer import CandlestickRenderer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNN-Trainer")

# Paths
MODEL_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training"


class ChartPatternDataset(Dataset):
    """
    Dataset for chart pattern training.
    
    Expects data in format:
    - OHLCV sequences with pattern labels
    - Or pre-rendered images with labels
    """
    
    def __init__(
        self,
        ohlcv_data: List[np.ndarray],
        labels: List[int],
        augment: bool = True
    ):
        self.ohlcv_data = ohlcv_data
        self.labels = labels
        self.augment = augment
        self.renderer = CandlestickRenderer(width=224, height=224)
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.ohlcv_data)
    
    def __getitem__(self, idx):
        ohlcv = self.ohlcv_data[idx]
        label = self.labels[idx]
        
        # Augmentation (training only)
        if self.augment:
            # Random crop (shift window by 0-10 candles)
            shift = np.random.randint(0, min(10, len(ohlcv) - 90))
            ohlcv = ohlcv[shift:shift+100] if len(ohlcv) > 100 else ohlcv
            
            # Random scaling (simulate different price ranges)
            scale = np.random.uniform(0.95, 1.05)
            ohlcv = ohlcv.copy()
            ohlcv[:, :4] *= scale  # Scale OHLC, not volume
        
        # Render to image
        img = self.renderer.render(ohlcv)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - self.mean) / self.std
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
        return tensor, label


def generate_synthetic_data(num_samples: int = 1000) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic training data with known patterns.
    
    In production, you would use real labeled data from:
    1. Historical charts manually labeled
    2. Auto-labeled based on price action after pattern
    3. Purchased pattern datasets
    
    This synthetic data is for demonstration only.
    """
    logger.info("Generating synthetic training data...")
    
    ohlcv_samples = []
    labels = []
    
    # Use controlled seed for reproducible synthetic data
    rng = np.random.RandomState(42)
    
    for i in range(num_samples):
        # Randomly select a pattern
        pattern_idx = rng.randint(0, NUM_PATTERNS)
        
        # Generate OHLCV that exhibits the pattern
        ohlcv = _generate_pattern_ohlcv(pattern_idx, rng)
        
        ohlcv_samples.append(ohlcv)
        labels.append(pattern_idx)
    
    logger.info(f"Generated {num_samples} synthetic samples")
    return ohlcv_samples, labels


def _generate_pattern_ohlcv(pattern_idx: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate OHLCV data exhibiting a specific pattern.
    This is simplified - real patterns are more complex.
    """
    num_candles = 100
    base_price = 1.0 + rng.uniform(0.1, 0.5)
    
    ohlcv = np.zeros((num_candles, 5))
    
    # Pattern-specific price movements
    pattern_name = PATTERNS[pattern_idx]['name']
    
    if pattern_name == 'head_shoulders':
        # Create head and shoulders shape
        trend = _head_shoulders_trend(num_candles, rng)
    elif pattern_name == 'inv_head_shoulders':
        # Inverted
        trend = -_head_shoulders_trend(num_candles, rng)
    elif pattern_name == 'double_top':
        trend = _double_top_trend(num_candles, rng)
    elif pattern_name == 'double_bottom':
        trend = -_double_top_trend(num_candles, rng)
    elif pattern_name in ['triangle_ascending', 'triangle_descending', 'triangle_symmetric']:
        trend = _triangle_trend(num_candles, rng, pattern_name)
    elif pattern_name in ['flag_bullish', 'flag_bearish']:
        trend = _flag_trend(num_candles, rng, pattern_name)
    else:
        # Random walk for no_pattern and others
        trend = np.cumsum(rng.randn(num_candles) * 0.002)
    
    # Add trend to base price
    prices = base_price + trend
    
    # Generate OHLCV from price series
    for i in range(num_candles):
        noise = rng.randn(4) * 0.001
        o = prices[i] + noise[0]
        c = prices[i] + noise[1]
        h = max(o, c) + abs(noise[2])
        l = min(o, c) - abs(noise[3])
        v = 1000 + rng.randn() * 100
        
        ohlcv[i] = [o, h, l, c, max(0, v)]
    
    return ohlcv


def _head_shoulders_trend(n: int, rng) -> np.ndarray:
    """Generate head and shoulders pattern."""
    trend = np.zeros(n)
    
    # Left shoulder (rise then fall)
    trend[:20] = np.linspace(0, 0.02, 20) + rng.randn(20) * 0.001
    trend[20:35] = np.linspace(0.02, 0.01, 15) + rng.randn(15) * 0.001
    
    # Head (higher rise then fall)
    trend[35:55] = np.linspace(0.01, 0.035, 20) + rng.randn(20) * 0.001
    trend[55:70] = np.linspace(0.035, 0.01, 15) + rng.randn(15) * 0.001
    
    # Right shoulder (rise to shoulder level then fall)
    trend[70:85] = np.linspace(0.01, 0.02, 15) + rng.randn(15) * 0.001
    trend[85:] = np.linspace(0.02, -0.01, n-85) + rng.randn(n-85) * 0.001
    
    return trend


def _double_top_trend(n: int, rng) -> np.ndarray:
    """Generate double top pattern."""
    trend = np.zeros(n)
    
    # First peak
    trend[:25] = np.linspace(0, 0.03, 25) + rng.randn(25) * 0.001
    trend[25:50] = np.linspace(0.03, 0.01, 25) + rng.randn(25) * 0.001
    
    # Second peak (similar height)
    trend[50:75] = np.linspace(0.01, 0.028, 25) + rng.randn(25) * 0.001
    trend[75:] = np.linspace(0.028, -0.01, n-75) + rng.randn(n-75) * 0.001
    
    return trend


def _triangle_trend(n: int, rng, pattern: str) -> np.ndarray:
    """Generate triangle pattern."""
    trend = np.zeros(n)
    
    # Converging highs and lows
    highs = np.linspace(0.02, 0.01, n)
    lows = np.linspace(-0.01, 0, n)
    
    if pattern == 'triangle_ascending':
        lows = np.linspace(-0.01, 0.005, n)  # Rising lows
    elif pattern == 'triangle_descending':
        highs = np.linspace(0.02, 0.005, n)  # Falling highs
    
    # Oscillate between high and low
    for i in range(n):
        if i % 10 < 5:
            trend[i] = highs[i] + rng.randn() * 0.002
        else:
            trend[i] = lows[i] + rng.randn() * 0.002
    
    return trend


def _flag_trend(n: int, rng, pattern: str) -> np.ndarray:
    """Generate flag pattern."""
    trend = np.zeros(n)
    
    if pattern == 'flag_bullish':
        # Strong move up
        trend[:30] = np.linspace(0, 0.04, 30) + rng.randn(30) * 0.001
        # Gentle consolidation down
        trend[30:70] = 0.04 - np.linspace(0, 0.01, 40) + rng.randn(40) * 0.001
        # Continuation up
        trend[70:] = np.linspace(0.03, 0.05, n-70) + rng.randn(n-70) * 0.001
    else:
        # Bearish flag is inverted
        trend[:30] = np.linspace(0, -0.04, 30) + rng.randn(30) * 0.001
        trend[30:70] = -0.04 + np.linspace(0, 0.01, 40) + rng.randn(40) * 0.001
        trend[70:] = np.linspace(-0.03, -0.05, n-70) + rng.randn(n-70) * 0.001
    
    return trend


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = None
) -> Dict:
    """
    Train the CNN model.
    
    Uses:
    - AdamW optimizer
    - CosineAnnealing LR scheduler
    - Mixed precision training (for H100)
    - Early stopping
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision
    scaler = GradScaler()
    
    best_val_acc = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                   f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, save_path)
                logger.info(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
    
    logger.info(f"\nTraining complete. Best val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train CNN Pattern Detector")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CNN PATTERN DETECTOR TRAINING")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load or generate data
    if args.synthetic or args.data is None:
        print("\nGenerating synthetic training data...")
        ohlcv_data, labels = generate_synthetic_data(args.samples)
    else:
        print(f"\nLoading data from {args.data}...")
        # TODO: Implement loading real labeled data
        raise NotImplementedError("Real data loading not yet implemented")
    
    # Split into train/val
    split = int(len(ohlcv_data) * 0.8)
    train_data = ohlcv_data[:split]
    train_labels = labels[:split]
    val_data = ohlcv_data[split:]
    val_labels = labels[split:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = ChartPatternDataset(train_data, train_labels, augment=True)
    val_dataset = ChartPatternDataset(val_data, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = ChartPatternCNN()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    save_path = MODEL_DIR / "model.pt"
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Model will be saved to: {save_path}")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=str(save_path)
    )
    
    # Save training history
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"\nTraining history saved to {history_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

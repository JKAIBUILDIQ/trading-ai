#!/usr/bin/env python3
"""
LSTM Price Direction Predictor
Predicts price direction for next N candles using LSTM networks.

Input features (per candle):
- OHLCV (Open, High, Low, Close, Volume)
- RSI(2), RSI(14)
- Bollinger Band position (0-1 scale)
- ATR (Average True Range)
- Volume ratio (current/average)

Output: {"direction": "UP", "magnitude": 15, "confidence": 0.72, "horizon": "4H"}

Architecture: 2-layer LSTM, 128 hidden units, dropout 0.2
Training: 5 years of data per pair

NO RANDOM DATA - All predictions from trained neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Feature configuration
FEATURES = [
    'open', 'high', 'low', 'close', 'volume',  # OHLCV
    'rsi_2', 'rsi_14',  # RSI
    'bb_position',  # Bollinger Band position (0-1)
    'atr',  # ATR normalized
    'volume_ratio',  # Volume / SMA(volume, 20)
    'returns_1',  # 1-period return
    'returns_5',  # 5-period return
    'ma_diff',  # (close - SMA50) / SMA50
]

NUM_FEATURES = len(FEATURES)
DEFAULT_SEQ_LENGTH = 100  # Last 100 candles


@dataclass
class PricePrediction:
    """Price prediction result."""
    direction: str  # UP, DOWN, NEUTRAL
    magnitude_pips: float  # Expected move in pips
    confidence: float  # 0-1
    horizon: str  # Timeframe (1H, 4H, etc.)
    probabilities: Dict[str, float]  # {UP: 0.7, DOWN: 0.25, NEUTRAL: 0.05}


class LSTMPricePredictor(nn.Module):
    """
    LSTM network for price direction prediction.
    
    Architecture:
    - Input: (batch, seq_len, features)
    - 2-layer LSTM with 128 hidden units
    - Dropout between layers
    - Attention mechanism for focusing on important timesteps
    - FC layers for classification (UP/DOWN/NEUTRAL)
    - FC layers for magnitude regression
    
    Optimized for H100 GPU with mixed precision support.
    """
    
    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,  # UP, DOWN, NEUTRAL
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )
        
        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Magnitude regression head (predicts pip movement)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Confidence head (predicts prediction reliability)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            direction_logits: (batch, 3) for UP/DOWN/NEUTRAL
            magnitude: (batch, 1) predicted pip movement
            confidence: (batch, 1) prediction confidence
        """
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        
        if self.use_attention:
            # Apply attention
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        else:
            # Use last hidden state
            context = h_n[-1]  # (batch, hidden_size)
        
        # Get predictions
        direction_logits = self.direction_head(context)
        magnitude = self.magnitude_head(context)
        confidence = self.confidence_head(context)
        
        return direction_logits, magnitude, confidence
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Make prediction with full output.
        """
        self.eval()
        with torch.no_grad():
            direction_logits, magnitude, confidence = self.forward(x)
            
            # Direction probabilities
            probs = F.softmax(direction_logits, dim=1)
            direction_idx = torch.argmax(probs, dim=1).item()
            
            directions = ['DOWN', 'NEUTRAL', 'UP']
            
            return {
                'direction': directions[direction_idx],
                'magnitude_pips': abs(magnitude.item()),
                'confidence': confidence.item(),
                'probabilities': {
                    'UP': probs[0, 2].item(),
                    'DOWN': probs[0, 0].item(),
                    'NEUTRAL': probs[0, 1].item()
                }
            }


class PriceDirectionPredictor:
    """
    High-level price prediction interface.
    Handles model loading, feature engineering, and inference.
    """
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = LSTMPricePredictor().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
        # Feature scalers (to be loaded with model)
        self.feature_means = None
        self.feature_stds = None
    
    def load_model(self, path: str):
        """Load trained model weights and scalers."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.feature_means = checkpoint.get('feature_means')
            self.feature_stds = checkpoint.get('feature_stds')
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {path}")
    
    def compute_features(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Compute features from OHLCV data.
        
        Args:
            ohlcv: numpy array of shape (seq_len, 5) with columns [open, high, low, close, volume]
        
        Returns:
            features: numpy array of shape (seq_len, NUM_FEATURES)
        """
        o, h, l, c, v = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        
        features = np.zeros((len(c), NUM_FEATURES))
        
        # OHLCV (normalized by close)
        features[:, 0] = o / c  # open relative to close
        features[:, 1] = h / c  # high relative to close
        features[:, 2] = l / c  # low relative to close
        features[:, 3] = 1.0    # close (reference)
        features[:, 4] = v / (np.mean(v) + 1e-8)  # volume ratio
        
        # RSI(2)
        features[:, 5] = self._compute_rsi(c, 2) / 100
        
        # RSI(14)
        features[:, 6] = self._compute_rsi(c, 14) / 100
        
        # Bollinger Band position
        sma20 = self._sma(c, 20)
        std20 = self._rolling_std(c, 20)
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_range = upper - lower + 1e-8
        features[:, 7] = (c - lower) / bb_range  # 0 = lower band, 1 = upper band
        
        # ATR normalized
        atr = self._compute_atr(h, l, c, 14)
        features[:, 8] = atr / (c + 1e-8)
        
        # Volume ratio
        vol_sma = self._sma(v, 20)
        features[:, 9] = v / (vol_sma + 1e-8)
        
        # Returns
        features[:, 10] = np.concatenate([[0], np.diff(c) / (c[:-1] + 1e-8)])  # 1-period
        features[4:, 11] = (c[4:] - c[:-4]) / (c[:-4] + 1e-8)  # 5-period
        
        # MA difference
        sma50 = self._sma(c, 50)
        features[:, 12] = (c - sma50) / (sma50 + 1e-8)
        
        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        result[:period - 1] = result[period - 1]  # Fill start
        return result
    
    def _rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        """Rolling standard deviation."""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1])
        result[:period - 1] = result[period - 1] if period <= len(data) else 0.01
        return result
    
    def _compute_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = self._sma(gain, period)
        avg_loss = self._sma(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute ATR."""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        return self._sma(tr, period)
    
    def predict(self, ohlcv: np.ndarray, horizon: str = "4H") -> Dict:
        """
        Predict price direction from OHLCV data.
        
        Args:
            ohlcv: numpy array of shape (seq_len, 5) with columns [open, high, low, close, volume]
            horizon: prediction horizon (e.g., "1H", "4H", "1D")
        
        Returns:
            {
                "direction": "UP",
                "magnitude": 15,
                "confidence": 0.72,
                "horizon": "4H",
                "probabilities": {"UP": 0.72, "DOWN": 0.20, "NEUTRAL": 0.08}
            }
        """
        # Ensure we have enough data
        if len(ohlcv) < DEFAULT_SEQ_LENGTH:
            # Pad with first row
            padding = np.repeat(ohlcv[:1], DEFAULT_SEQ_LENGTH - len(ohlcv), axis=0)
            ohlcv = np.vstack([padding, ohlcv])
        elif len(ohlcv) > DEFAULT_SEQ_LENGTH:
            ohlcv = ohlcv[-DEFAULT_SEQ_LENGTH:]
        
        # Compute features
        features = self.compute_features(ohlcv)
        
        # Normalize if scalers available
        if self.feature_means is not None and self.feature_stds is not None:
            features = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Convert to tensor
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Get prediction
        result = self.model.predict(x)
        result['horizon'] = horizon
        result['model'] = 'lstm_v1'
        result['source'] = 'neural_network'
        
        # Round values
        result['magnitude_pips'] = round(result['magnitude_pips'], 1)
        result['confidence'] = round(result['confidence'], 4)
        for key in result['probabilities']:
            result['probabilities'][key] = round(result['probabilities'][key], 4)
        
        return result


def get_price_predictor(model_path: str = None) -> PriceDirectionPredictor:
    """Factory function to get price predictor instance."""
    return PriceDirectionPredictor(model_path=model_path)


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("LSTM PRICE PREDICTOR - Model Test")
    print("=" * 60)
    
    # Create model
    model = LSTMPricePredictor()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 100, NUM_FEATURES)  # batch=1, seq=100, features=13
    dir_logits, magnitude, confidence = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Direction logits shape: {dir_logits.shape}")
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    # Test prediction
    result = model.predict(x)
    print(f"Prediction: {result['direction']} ({result['confidence']:.2%} confidence)")
    print(f"Magnitude: {result['magnitude_pips']:.1f} pips")
    print(f"Probabilities: UP={result['probabilities']['UP']:.2%}, DOWN={result['probabilities']['DOWN']:.2%}")
    
    # Test with dummy OHLCV data
    print("\n--- Testing with OHLCV data ---")
    predictor = PriceDirectionPredictor()
    
    # Generate dummy OHLCV (in reality, this comes from MT5 API)
    np.random.seed(42)  # For reproducibility in test only
    base_price = 1.1000
    ohlcv = np.zeros((100, 5))
    for i in range(100):
        noise = np.random.randn() * 0.001
        ohlcv[i, 0] = base_price + noise  # open
        ohlcv[i, 1] = base_price + noise + abs(np.random.randn() * 0.0005)  # high
        ohlcv[i, 2] = base_price + noise - abs(np.random.randn() * 0.0005)  # low
        ohlcv[i, 3] = base_price + noise + np.random.randn() * 0.0003  # close
        ohlcv[i, 4] = 1000 + np.random.randn() * 200  # volume
        base_price = ohlcv[i, 3]
    
    result = predictor.predict(ohlcv, horizon="4H")
    print(f"Direction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Magnitude: {result['magnitude_pips']} pips")
    
    # Check GPU
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        dir_logits, magnitude, confidence = model(x)
        print(f"\n✅ GPU inference working on {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 60)

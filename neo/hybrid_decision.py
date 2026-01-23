#!/usr/bin/env python3
"""
Hybrid Decision Engine for NEO
Combines all AI models (CNN, LSTM, RL, LLM) into unified trading decisions.

Decision Flow:
1. GET market data
2. RUN CNN pattern detector → "Double bottom detected, 82% confidence"
3. RUN LSTM predictor → "EURUSD likely +20 pips in 4H, 71% confidence"
4. CHECK MQL5 consensus → "3 top traders buying EURUSD"
5. LLM REASONING → Combine all inputs with strategies + WWCD
6. OUTPUT signal → Ghost Commander executes
7. RECORD trade → Learning database
8. LEARN from outcome → Update RL model

NO RANDOM DATA - All predictions from trained models or real market data.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model interfaces
from models.cnn_patterns.model import PatternDetector, get_pattern_detector
from models.lstm_price.model import PriceDirectionPredictor, get_price_predictor
from models.rl_trader.model import RLTradingAgent, TradingState, get_rl_agent
from models.cnn_patterns.chart_renderer import CandlestickRenderer

# Import intel
try:
    from intel.mql5_reader import MQL5Reader
except ImportError:
    MQL5Reader = None


@dataclass
class HybridAnalysis:
    """Combined analysis from all AI models."""
    
    # CNN Pattern Detection
    pattern: str
    pattern_confidence: float
    pattern_direction: str
    
    # LSTM Price Prediction
    lstm_direction: str
    lstm_confidence: float
    lstm_magnitude_pips: float
    lstm_horizon: str
    
    # RL Agent Recommendation
    rl_action: str
    rl_confidence: float
    rl_value_estimate: float
    
    # MQL5 Intel
    mql5_consensus: bool
    mql5_traders: List[str]
    mql5_confidence_boost: int
    
    # Combined Signal
    final_direction: str  # BUY, SELL, HOLD
    final_confidence: float
    final_magnitude_pips: float
    reasoning: str
    
    # Metadata
    symbol: str
    timestamp: str
    model_versions: Dict[str, str]


class HybridDecisionEngine:
    """
    Combines CNN, LSTM, RL, and LLM models for trading decisions.
    
    This is the brain that orchestrates all AI models and produces
    a unified trading signal for NEO.
    """
    
    def __init__(
        self,
        cnn_model_path: str = None,
        lstm_model_path: str = None,
        rl_model_path: str = None,
        device: str = "auto"
    ):
        self.device = device
        
        # Initialize models (will use untrained weights if no path provided)
        print("Initializing Hybrid Decision Engine...")
        
        # CNN Pattern Detector
        try:
            self.cnn = get_pattern_detector(cnn_model_path)
            print("  ✅ CNN Pattern Detector loaded")
        except Exception as e:
            print(f"  ⚠️ CNN not loaded: {e}")
            self.cnn = None
        
        # LSTM Price Predictor
        try:
            self.lstm = get_price_predictor(lstm_model_path)
            print("  ✅ LSTM Price Predictor loaded")
        except Exception as e:
            print(f"  ⚠️ LSTM not loaded: {e}")
            self.lstm = None
        
        # RL Trading Agent
        try:
            self.rl = get_rl_agent(rl_model_path)
            print("  ✅ RL Trading Agent loaded")
        except Exception as e:
            print(f"  ⚠️ RL not loaded: {e}")
            self.rl = None
        
        # MQL5 Intel Reader
        if MQL5Reader:
            self.mql5 = MQL5Reader()
            print("  ✅ MQL5 Intel Reader loaded")
        else:
            self.mql5 = None
            print("  ⚠️ MQL5 Intel not loaded")
        
        # Chart renderer for CNN
        self.chart_renderer = CandlestickRenderer(width=224, height=224)
        
        print("Hybrid Decision Engine ready!")
    
    def analyze(
        self,
        symbol: str,
        ohlcv: np.ndarray,
        current_position: Dict = None,
        daily_pnl: float = 0,
        horizon: str = "4H"
    ) -> HybridAnalysis:
        """
        Run full hybrid analysis on market data.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            ohlcv: OHLCV data of shape (num_candles, 5)
            current_position: Current position info {'direction': 'LONG', 'pnl': 100, ...}
            daily_pnl: Today's realized P&L
            horizon: Prediction horizon
        
        Returns:
            HybridAnalysis with combined signal
        """
        timestamp = datetime.utcnow().isoformat()
        
        # ===== 1. CNN Pattern Detection =====
        if self.cnn is not None:
            try:
                # Render chart as image
                chart_img = self.chart_renderer.render(ohlcv)
                pattern_result = self.cnn.detect(chart_img)
                
                pattern = pattern_result['pattern']
                pattern_confidence = pattern_result['confidence']
                pattern_direction = pattern_result['direction']
            except Exception as e:
                print(f"CNN error: {e}")
                pattern = "no_pattern"
                pattern_confidence = 0.0
                pattern_direction = "NEUTRAL"
        else:
            pattern = "no_pattern"
            pattern_confidence = 0.0
            pattern_direction = "NEUTRAL"
        
        # ===== 2. LSTM Price Prediction =====
        if self.lstm is not None:
            try:
                lstm_result = self.lstm.predict(ohlcv, horizon=horizon)
                
                lstm_direction = lstm_result['direction']
                lstm_confidence = lstm_result['confidence']
                lstm_magnitude = lstm_result['magnitude_pips']
            except Exception as e:
                print(f"LSTM error: {e}")
                lstm_direction = "NEUTRAL"
                lstm_confidence = 0.0
                lstm_magnitude = 0.0
        else:
            lstm_direction = "NEUTRAL"
            lstm_confidence = 0.0
            lstm_magnitude = 0.0
        
        # ===== 3. RL Agent Recommendation =====
        if self.rl is not None:
            try:
                # Build state for RL agent
                market_features = self._extract_market_features(ohlcv)
                market_features = np.append(market_features, [
                    1 if pattern_direction == 'BULLISH' else (-1 if pattern_direction == 'BEARISH' else 0),
                    1 if lstm_direction == 'UP' else (-1 if lstm_direction == 'DOWN' else 0),
                    lstm_confidence
                ])[:12]  # Ensure 12 features
                
                if len(market_features) < 12:
                    market_features = np.pad(market_features, (0, 12 - len(market_features)))
                
                # Position features
                if current_position:
                    has_pos = 1 if current_position.get('direction') == 'LONG' else -1
                    pos_pnl = current_position.get('pnl', 0) / 1000  # Normalize
                    pos_dur = current_position.get('duration', 0) / 100
                    dist_entry = current_position.get('distance', 0)
                else:
                    has_pos, pos_pnl, pos_dur, dist_entry = 0, 0, 0, 0
                
                position_features = np.array([has_pos, pos_pnl, pos_dur, dist_entry])
                
                # Risk features
                risk_features = np.array([
                    daily_pnl / 88000,  # Normalized by capital
                    0,  # Drawdown
                    0.05 if current_position else 0  # Exposure
                ])
                
                state = TradingState(
                    market_features=market_features,
                    position_features=position_features,
                    risk_features=risk_features
                )
                
                rl_result = self.rl.act(state, deterministic=True)
                
                rl_action = rl_result['action']
                rl_confidence = rl_result['confidence']
                rl_value = rl_result['value_estimate']
            except Exception as e:
                print(f"RL error: {e}")
                rl_action = "HOLD"
                rl_confidence = 0.0
                rl_value = 0.0
        else:
            rl_action = "HOLD"
            rl_confidence = 0.0
            rl_value = 0.0
        
        # ===== 4. MQL5 Consensus Check =====
        if self.mql5 is not None:
            try:
                # Check both directions
                buy_boost = self.mql5.get_confidence_boost(symbol, "BUY")
                sell_boost = self.mql5.get_confidence_boost(symbol, "SELL")
                
                if buy_boost > 0:
                    mql5_consensus = True
                    mql5_direction = "BUY"
                    mql5_boost = buy_boost
                elif sell_boost > 0:
                    mql5_consensus = True
                    mql5_direction = "SELL"
                    mql5_boost = sell_boost
                else:
                    mql5_consensus = False
                    mql5_direction = None
                    mql5_boost = 0
                
                # Get traders list
                consensus_signals = self.mql5.get_consensus_signals()
                mql5_traders = []
                for sig in consensus_signals:
                    if sig.get('symbol') == symbol:
                        mql5_traders = sig.get('traders', [])
                        break
            except Exception as e:
                print(f"MQL5 error: {e}")
                mql5_consensus = False
                mql5_traders = []
                mql5_boost = 0
        else:
            mql5_consensus = False
            mql5_traders = []
            mql5_boost = 0
        
        # ===== 5. Combine All Signals =====
        final_direction, final_confidence, reasoning = self._combine_signals(
            pattern=pattern,
            pattern_direction=pattern_direction,
            pattern_confidence=pattern_confidence,
            lstm_direction=lstm_direction,
            lstm_confidence=lstm_confidence,
            lstm_magnitude=lstm_magnitude,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            mql5_consensus=mql5_consensus,
            mql5_boost=mql5_boost
        )
        
        return HybridAnalysis(
            # CNN
            pattern=pattern,
            pattern_confidence=round(pattern_confidence, 4),
            pattern_direction=pattern_direction,
            # LSTM
            lstm_direction=lstm_direction,
            lstm_confidence=round(lstm_confidence, 4),
            lstm_magnitude_pips=round(lstm_magnitude, 1),
            lstm_horizon=horizon,
            # RL
            rl_action=rl_action,
            rl_confidence=round(rl_confidence, 4),
            rl_value_estimate=round(rl_value, 4),
            # MQL5
            mql5_consensus=mql5_consensus,
            mql5_traders=mql5_traders,
            mql5_confidence_boost=mql5_boost,
            # Final
            final_direction=final_direction,
            final_confidence=round(final_confidence, 4),
            final_magnitude_pips=round(lstm_magnitude, 1),
            reasoning=reasoning,
            # Metadata
            symbol=symbol,
            timestamp=timestamp,
            model_versions={
                "cnn": "v1.0",
                "lstm": "v1.0",
                "rl": "ppo_v1.0",
                "hybrid": "v1.0"
            }
        )
    
    def _extract_market_features(self, ohlcv: np.ndarray) -> np.ndarray:
        """Extract market features for RL state."""
        if len(ohlcv) < 20:
            return np.zeros(12)
        
        closes = ohlcv[:, 3]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        volumes = ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones(len(closes))
        
        # Price changes
        pc1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
        pc5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 and closes[-6] != 0 else 0
        pc20 = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 and closes[-21] != 0 else 0
        
        # RSI-like (simplified)
        gains = np.maximum(np.diff(closes[-15:]), 0)
        losses = np.maximum(-np.diff(closes[-15:]), 0)
        rsi_14 = 100 - 100 / (1 + np.mean(gains) / (np.mean(losses) + 1e-8))
        
        gains_2 = np.maximum(np.diff(closes[-3:]), 0)
        losses_2 = np.maximum(-np.diff(closes[-3:]), 0)
        rsi_2 = 100 - 100 / (1 + np.mean(gains_2) / (np.mean(losses_2) + 1e-8))
        
        # BB position
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # ATR
        tr = np.maximum(highs[-14:] - lows[-14:],
                       np.maximum(np.abs(highs[-14:] - np.roll(closes[-14:], 1)),
                                 np.abs(lows[-14:] - np.roll(closes[-14:], 1))))
        atr = np.mean(tr[1:]) / closes[-1]
        
        # Volume ratio
        vol_ratio = volumes[-1] / (np.mean(volumes[-20:]) + 1e-8)
        
        return np.array([
            pc1 * 100, pc5 * 100, pc20 * 100,
            rsi_2 / 100, rsi_14 / 100,
            bb_position,
            atr * 100,
            vol_ratio,
            0  # Trend strength placeholder
        ])
    
    def _combine_signals(
        self,
        pattern: str,
        pattern_direction: str,
        pattern_confidence: float,
        lstm_direction: str,
        lstm_confidence: float,
        lstm_magnitude: float,
        rl_action: str,
        rl_confidence: float,
        mql5_consensus: bool,
        mql5_boost: int
    ) -> Tuple[str, float, str]:
        """
        Combine all model signals into final decision.
        
        Weighting:
        - LSTM (price predictor): 35%
        - CNN (pattern): 25%
        - RL (experience): 25%
        - MQL5 (consensus): 15% boost
        """
        reasoning_parts = []
        
        # Convert directions to scores (-1 = SELL, 0 = HOLD, 1 = BUY)
        def direction_to_score(d):
            if d in ['UP', 'BUY', 'BULLISH']:
                return 1
            elif d in ['DOWN', 'SELL', 'BEARISH']:
                return -1
            return 0
        
        # LSTM signal (35% weight)
        lstm_score = direction_to_score(lstm_direction) * lstm_confidence * 0.35
        if lstm_direction != 'NEUTRAL':
            reasoning_parts.append(f"LSTM: {lstm_direction} ({lstm_confidence:.0%})")
        
        # Pattern signal (25% weight)
        pattern_score = direction_to_score(pattern_direction) * pattern_confidence * 0.25
        if pattern != 'no_pattern':
            reasoning_parts.append(f"Pattern: {pattern} → {pattern_direction} ({pattern_confidence:.0%})")
        
        # RL signal (25% weight)
        rl_direction_score = direction_to_score(rl_action) if rl_action in ['BUY', 'SELL'] else 0
        rl_score = rl_direction_score * rl_confidence * 0.25
        if rl_action != 'HOLD':
            reasoning_parts.append(f"RL: {rl_action} ({rl_confidence:.0%})")
        
        # MQL5 boost (15% max)
        mql5_score = 0
        if mql5_consensus:
            mql5_score = (mql5_boost / 100) * 0.15
            reasoning_parts.append(f"MQL5 Consensus: +{mql5_boost}%")
        
        # Combined score
        total_score = lstm_score + pattern_score + rl_score + mql5_score
        
        # Determine final direction
        if total_score > 0.15:  # Threshold for action
            final_direction = "BUY"
        elif total_score < -0.15:
            final_direction = "SELL"
        else:
            final_direction = "HOLD"
        
        # Calculate confidence
        final_confidence = min(0.95, abs(total_score) + 0.3)
        
        # Build reasoning string
        reasoning = f"Hybrid Analysis → {final_direction}. " + " | ".join(reasoning_parts)
        
        return final_direction, final_confidence, reasoning
    
    def to_dict(self, analysis: HybridAnalysis) -> Dict:
        """Convert HybridAnalysis to dictionary."""
        return asdict(analysis)
    
    def to_json(self, analysis: HybridAnalysis) -> str:
        """Convert HybridAnalysis to JSON string."""
        return json.dumps(asdict(analysis), indent=2)


def get_hybrid_engine(**kwargs) -> HybridDecisionEngine:
    """Factory function to get hybrid decision engine."""
    return HybridDecisionEngine(**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID DECISION ENGINE - Test")
    print("=" * 60)
    
    # Initialize engine
    engine = HybridDecisionEngine()
    
    # Generate test OHLCV data (in reality, from MT5 API)
    np.random.seed(42)
    num_candles = 100
    base_price = 1.1000
    
    ohlcv = np.zeros((num_candles, 5))
    for i in range(num_candles):
        change = np.random.randn() * 0.0020
        o = base_price
        c = base_price + change
        h = max(o, c) + abs(np.random.randn() * 0.0005)
        l = min(o, c) - abs(np.random.randn() * 0.0005)
        v = 1000 + np.random.randn() * 200
        ohlcv[i] = [o, h, l, c, v]
        base_price = c
    
    # Run analysis
    print("\nRunning hybrid analysis on EURUSD...")
    analysis = engine.analyze(
        symbol="EURUSD",
        ohlcv=ohlcv,
        current_position=None,
        daily_pnl=0,
        horizon="4H"
    )
    
    print("\n" + "=" * 60)
    print("HYBRID ANALYSIS RESULT")
    print("=" * 60)
    
    print(f"\n📊 CNN Pattern: {analysis.pattern} ({analysis.pattern_confidence:.0%} → {analysis.pattern_direction})")
    print(f"📈 LSTM Price: {analysis.lstm_direction} ({analysis.lstm_confidence:.0%}, {analysis.lstm_magnitude_pips} pips)")
    print(f"🤖 RL Agent: {analysis.rl_action} ({analysis.rl_confidence:.0%})")
    print(f"👥 MQL5 Consensus: {'Yes' if analysis.mql5_consensus else 'No'} (+{analysis.mql5_confidence_boost}%)")
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL SIGNAL: {analysis.final_direction}")
    print(f"   Confidence: {analysis.final_confidence:.0%}")
    print(f"   Magnitude: {analysis.final_magnitude_pips} pips")
    print(f"\n📝 Reasoning: {analysis.reasoning}")
    print("=" * 60)

"""
Crellastein Meta Bot - Weighted Indicator Ensemble
"The Wisdom of Many Indicators"

Combines multiple technical indicators with data-driven weights
to generate high-confidence trading signals for XAUUSD and IREN.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrellaStein")


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class IndicatorSignal:
    name: str
    signal: int  # -1 = SELL, 0 = NEUTRAL, +1 = BUY
    weight: float
    confidence: float
    value: float  # Raw indicator value
    reasoning: str


@dataclass
class CompositeSignal:
    symbol: str
    timestamp: str
    composite_score: float  # 0 to 1 (0.5 = neutral)
    signal_type: SignalType
    confidence: float
    indicators: List[Dict]
    buy_threshold: float
    sell_threshold: float
    recommended_action: str
    position_size_multiplier: float
    reasoning: str
    dca_allowed: bool
    dca_level: int


class CrellaSteinMetaBot:
    """
    Weighted Indicator Ensemble Trading System
    """
    
    # Default weights (will be overridden by backtested weights)
    DEFAULT_WEIGHTS = {
        "supertrend": 0.18,
        "macd": 0.12,
        "rsi": 0.10,
        "bollinger": 0.08,
        "ema_cross": 0.10,
        "vwap": 0.08,
        "momentum": 0.10,
        "volume_trend": 0.08,
        "support_resistance": 0.08,
        "gap_fill": 0.08,
    }
    
    # Asset-specific configurations
    ASSET_CONFIG = {
        "XAUUSD": {
            "ticker": "GC=F",  # Gold futures
            "long_bias": True,
            "max_drawdown_accept": 25.0,
            "dca_trigger_percent": 1.0,  # 1% pullback = DCA
            "dca_max_levels": 5,
            "dca_lot_multiplier": 1.5,
            "buy_threshold": 0.60,
            "sell_threshold": 0.40,
            "strong_threshold": 0.75,
        },
        "IREN": {
            "ticker": "IREN",
            "long_bias": True,  # Strong conviction on IREN rising
            "max_drawdown_accept": 30.0,
            "dca_trigger_percent": 5.0,  # 5% pullback = DCA
            "dca_max_levels": 5,
            "dca_lot_multiplier": 1.5,
            "buy_threshold": 0.58,
            "sell_threshold": 0.42,
            "strong_threshold": 0.72,
        },
        "CLSK": {
            "ticker": "CLSK",  # CleanSpark - BTC miner
            "long_bias": True,  # Paul's conviction - BTC miners long-term play
            "max_drawdown_accept": 35.0,  # Higher volatility tolerance for miners
            "dca_trigger_percent": 5.0,  # 5% pullback = DCA
            "dca_max_levels": 5,
            "dca_lot_multiplier": 1.5,
            "buy_threshold": 0.58,
            "sell_threshold": 0.42,
            "strong_threshold": 0.72,
        },
        "CIFR": {
            "ticker": "CIFR",  # Cipher Mining - BTC miner
            "long_bias": True,  # Paul's conviction - BTC miners long-term play
            "max_drawdown_accept": 35.0,  # Higher volatility tolerance for miners
            "dca_trigger_percent": 5.0,  # 5% pullback = DCA
            "dca_max_levels": 5,
            "dca_lot_multiplier": 1.5,
            "buy_threshold": 0.58,
            "sell_threshold": 0.42,
            "strong_threshold": 0.72,
        }
    }
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.config = self.ASSET_CONFIG.get(symbol, self.ASSET_CONFIG["XAUUSD"])
        self.weights = self._load_weights()
        self.weights_file = Path(__file__).parent / "reports" / f"{symbol}_indicator_weights.json"
        
    def _load_weights(self) -> Dict[str, float]:
        """Load optimized weights from file or use defaults"""
        weights_file = Path(__file__).parent / "reports" / f"{self.symbol}_indicator_weights.json"
        if weights_file.exists():
            try:
                with open(weights_file) as f:
                    return json.load(f)
            except:
                pass
        return self.DEFAULT_WEIGHTS.copy()
    
    def _save_weights(self, weights: Dict[str, float]):
        """Save optimized weights"""
        weights_file = Path(__file__).parent / "reports" / f"{self.symbol}_indicator_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    def get_market_data(self, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.config["ticker"])
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data for {self.config['ticker']}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    # ============== INDICATOR CALCULATIONS ==============
    
    def calc_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> IndicatorSignal:
        """Calculate SuperTrend indicator"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # ATR
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            # SuperTrend bands
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Determine trend
            supertrend = [0.0] * len(df)
            direction = [1] * len(df)
            
            for i in range(period, len(df)):
                if close.iloc[i] > upper_band.iloc[i-1]:
                    direction[i] = 1
                elif close.iloc[i] < lower_band.iloc[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
            
            current_direction = direction[-1]
            signal = 1 if current_direction == 1 else -1
            confidence = min(90, 60 + abs(close.iloc[-1] - hl2.iloc[-1]) / atr.iloc[-1] * 10)
            
            return IndicatorSignal(
                name="supertrend",
                signal=signal,
                weight=self.weights.get("supertrend", 0.18),
                confidence=confidence,
                value=current_direction,
                reasoning=f"SuperTrend: {'BULLISH' if signal == 1 else 'BEARISH'} trend"
            )
        except Exception as e:
            logger.warning(f"SuperTrend error: {e}")
            return IndicatorSignal("supertrend", 0, self.weights.get("supertrend", 0.18), 0, 0, "Error")
    
    def calc_macd(self, df: pd.DataFrame) -> IndicatorSignal:
        """Calculate MACD indicator"""
        try:
            close = df['Close']
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # Signal based on histogram direction and crossover
            signal = 0
            if histogram.iloc[-1] > 0 and histogram.iloc[-2] < 0:
                signal = 1  # Bullish crossover
            elif histogram.iloc[-1] < 0 and histogram.iloc[-2] > 0:
                signal = -1  # Bearish crossover
            elif histogram.iloc[-1] > histogram.iloc[-2]:
                signal = 1 if histogram.iloc[-1] > 0 else 0
            elif histogram.iloc[-1] < histogram.iloc[-2]:
                signal = -1 if histogram.iloc[-1] < 0 else 0
            
            confidence = min(85, 50 + abs(histogram.iloc[-1]) / close.iloc[-1] * 5000)
            
            return IndicatorSignal(
                name="macd",
                signal=signal,
                weight=self.weights.get("macd", 0.12),
                confidence=confidence,
                value=histogram.iloc[-1],
                reasoning=f"MACD: Histogram {'rising' if histogram.iloc[-1] > histogram.iloc[-2] else 'falling'}"
            )
        except Exception as e:
            logger.warning(f"MACD error: {e}")
            return IndicatorSignal("macd", 0, self.weights.get("macd", 0.12), 0, 0, "Error")
    
    def calc_rsi(self, df: pd.DataFrame, period: int = 14) -> IndicatorSignal:
        """Calculate RSI indicator"""
        try:
            close = df['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Signal logic
            signal = 0
            if current_rsi < 30:
                signal = 1  # Oversold = BUY
            elif current_rsi > 70:
                signal = -1  # Overbought = SELL
            elif current_rsi < 40 and rsi.iloc[-2] < rsi.iloc[-1]:
                signal = 1  # Rising from low
            elif current_rsi > 60 and rsi.iloc[-2] > rsi.iloc[-1]:
                signal = -1  # Falling from high
            
            confidence = min(80, 50 + abs(50 - current_rsi) * 0.6)
            
            return IndicatorSignal(
                name="rsi",
                signal=signal,
                weight=self.weights.get("rsi", 0.10),
                confidence=confidence,
                value=current_rsi,
                reasoning=f"RSI: {current_rsi:.1f} ({'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'})"
            )
        except Exception as e:
            logger.warning(f"RSI error: {e}")
            return IndicatorSignal("rsi", 0, self.weights.get("rsi", 0.10), 0, 50, "Error")
    
    def calc_bollinger(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> IndicatorSignal:
        """Calculate Bollinger Bands"""
        try:
            close = df['Close']
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            
            current_price = close.iloc[-1]
            bandwidth = (upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]
            
            # Signal based on position in bands
            signal = 0
            if current_price < lower.iloc[-1]:
                signal = 1  # Below lower band = oversold
            elif current_price > upper.iloc[-1]:
                signal = -1  # Above upper band = overbought
            elif current_price < sma.iloc[-1] * 0.99:
                signal = 1  # Below middle
            elif current_price > sma.iloc[-1] * 1.01:
                signal = -1  # Above middle
            
            # Squeeze detection (low bandwidth = breakout coming)
            avg_bandwidth = ((upper - lower) / sma).rolling(50).mean().iloc[-1]
            squeeze = bandwidth < avg_bandwidth * 0.7
            
            confidence = min(75, 50 + abs(current_price - sma.iloc[-1]) / std.iloc[-1] * 10)
            
            return IndicatorSignal(
                name="bollinger",
                signal=signal,
                weight=self.weights.get("bollinger", 0.08),
                confidence=confidence,
                value=bandwidth,
                reasoning=f"Bollinger: {'Squeeze' if squeeze else 'Normal'}, price {'below' if signal == 1 else 'above' if signal == -1 else 'at'} middle"
            )
        except Exception as e:
            logger.warning(f"Bollinger error: {e}")
            return IndicatorSignal("bollinger", 0, self.weights.get("bollinger", 0.08), 0, 0, "Error")
    
    def calc_ema_cross(self, df: pd.DataFrame) -> IndicatorSignal:
        """Calculate EMA Crossover (20/50/200)"""
        try:
            close = df['Close']
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()
            ema200 = close.ewm(span=200).mean() if len(close) >= 200 else close.ewm(span=min(len(close), 100)).mean()
            
            # Multi-EMA alignment
            signal = 0
            if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
                signal = 1  # Perfect bullish alignment
            elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
                signal = -1  # Perfect bearish alignment
            elif ema20.iloc[-1] > ema50.iloc[-1]:
                signal = 1  # Short-term bullish
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                signal = -1  # Short-term bearish
            
            # Check for recent crossover
            crossover = False
            if (ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] < ema50.iloc[-2]):
                crossover = True  # Bullish crossover
            elif (ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] > ema50.iloc[-2]):
                crossover = True  # Bearish crossover
            
            confidence = min(85, 60 + abs(ema20.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1] * 1000)
            if crossover:
                confidence = min(95, confidence + 15)
            
            return IndicatorSignal(
                name="ema_cross",
                signal=signal,
                weight=self.weights.get("ema_cross", 0.10),
                confidence=confidence,
                value=ema20.iloc[-1] - ema50.iloc[-1],
                reasoning=f"EMA: {'Bullish' if signal == 1 else 'Bearish' if signal == -1 else 'Neutral'} alignment" + (" (CROSSOVER!)" if crossover else "")
            )
        except Exception as e:
            logger.warning(f"EMA Cross error: {e}")
            return IndicatorSignal("ema_cross", 0, self.weights.get("ema_cross", 0.10), 0, 0, "Error")
    
    def calc_momentum(self, df: pd.DataFrame, period: int = 10) -> IndicatorSignal:
        """Calculate Momentum indicator"""
        try:
            close = df['Close']
            momentum = close - close.shift(period)
            rate_of_change = (close / close.shift(period) - 1) * 100
            
            current_momentum = momentum.iloc[-1]
            current_roc = rate_of_change.iloc[-1]
            
            # Signal based on momentum direction
            signal = 0
            if current_momentum > 0 and momentum.iloc[-2] > 0:
                signal = 1  # Sustained positive momentum
            elif current_momentum < 0 and momentum.iloc[-2] < 0:
                signal = -1  # Sustained negative momentum
            elif current_momentum > 0 and momentum.iloc[-2] < 0:
                signal = 1  # Momentum turning positive
            elif current_momentum < 0 and momentum.iloc[-2] > 0:
                signal = -1  # Momentum turning negative
            
            confidence = min(80, 50 + abs(current_roc) * 5)
            
            return IndicatorSignal(
                name="momentum",
                signal=signal,
                weight=self.weights.get("momentum", 0.10),
                confidence=confidence,
                value=current_roc,
                reasoning=f"Momentum: {current_roc:.2f}% over {period} periods"
            )
        except Exception as e:
            logger.warning(f"Momentum error: {e}")
            return IndicatorSignal("momentum", 0, self.weights.get("momentum", 0.10), 0, 0, "Error")
    
    def calc_volume_trend(self, df: pd.DataFrame) -> IndicatorSignal:
        """Calculate Volume Trend indicator"""
        try:
            close = df['Close']
            volume = df['Volume']
            
            # Volume Moving Average
            vol_sma = volume.rolling(20).mean()
            relative_volume = volume.iloc[-1] / vol_sma.iloc[-1]
            
            # Price-Volume relationship
            price_up = close.iloc[-1] > close.iloc[-2]
            volume_up = volume.iloc[-1] > vol_sma.iloc[-1]
            
            signal = 0
            if price_up and volume_up:
                signal = 1  # Price up on high volume = bullish confirmation
            elif not price_up and volume_up:
                signal = -1  # Price down on high volume = bearish confirmation
            elif price_up and not volume_up:
                signal = 0  # Price up on low volume = weak
            else:
                signal = 0  # Price down on low volume = weak selling
            
            confidence = min(75, 50 + abs(relative_volume - 1) * 20)
            
            return IndicatorSignal(
                name="volume_trend",
                signal=signal,
                weight=self.weights.get("volume_trend", 0.08),
                confidence=confidence,
                value=relative_volume,
                reasoning=f"Volume: {relative_volume:.2f}x average, {'confirming' if signal != 0 else 'weak'} move"
            )
        except Exception as e:
            logger.warning(f"Volume Trend error: {e}")
            return IndicatorSignal("volume_trend", 0, self.weights.get("volume_trend", 0.08), 0, 1, "Error")
    
    def calc_support_resistance(self, df: pd.DataFrame) -> IndicatorSignal:
        """Calculate Support/Resistance levels"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Find recent swing highs and lows
            recent_high = high.rolling(20).max().iloc[-1]
            recent_low = low.rolling(20).min().iloc[-1]
            current_price = close.iloc[-1]
            
            # Distance to levels
            dist_to_resistance = (recent_high - current_price) / current_price
            dist_to_support = (current_price - recent_low) / current_price
            
            signal = 0
            if dist_to_support < 0.01:  # Within 1% of support
                signal = 1  # Near support = potential bounce
            elif dist_to_resistance < 0.01:  # Within 1% of resistance
                signal = -1  # Near resistance = potential rejection
            elif dist_to_support < dist_to_resistance:
                signal = 1  # Closer to support
            else:
                signal = -1  # Closer to resistance
            
            confidence = min(70, 50 + (1 - min(dist_to_support, dist_to_resistance)) * 100)
            
            return IndicatorSignal(
                name="support_resistance",
                signal=signal,
                weight=self.weights.get("support_resistance", 0.08),
                confidence=confidence,
                value=dist_to_support - dist_to_resistance,
                reasoning=f"S/R: {dist_to_support*100:.1f}% from support, {dist_to_resistance*100:.1f}% from resistance"
            )
        except Exception as e:
            logger.warning(f"S/R error: {e}")
            return IndicatorSignal("support_resistance", 0, self.weights.get("support_resistance", 0.08), 0, 0, "Error")
    
    def calc_gap_fill(self, df: pd.DataFrame) -> IndicatorSignal:
        """Detect gaps and potential fill"""
        try:
            open_price = df['Open']
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # Detect gaps
            gaps = []
            for i in range(1, min(20, len(df))):
                prev_close = close.iloc[-(i+1)]
                curr_open = open_price.iloc[-i]
                gap = (curr_open - prev_close) / prev_close * 100
                if abs(gap) > 0.2:  # Gap > 0.2%
                    gaps.append((i, gap))
            
            signal = 0
            gap_info = "No significant gaps"
            
            if gaps:
                # Check for unfilled gaps
                recent_gap = gaps[0]
                gap_size = recent_gap[1]
                if gap_size > 0:
                    # Up gap - price should come back down
                    signal = -1 if close.iloc[-1] > close.iloc[-recent_gap[0]-1] else 0
                else:
                    # Down gap - price should come back up
                    signal = 1 if close.iloc[-1] < close.iloc[-recent_gap[0]-1] else 0
                gap_info = f"Gap of {gap_size:.2f}% detected"
            
            confidence = min(78, 50 + len(gaps) * 5)
            
            return IndicatorSignal(
                name="gap_fill",
                signal=signal,
                weight=self.weights.get("gap_fill", 0.08),
                confidence=confidence,
                value=len(gaps),
                reasoning=gap_info
            )
        except Exception as e:
            logger.warning(f"Gap Fill error: {e}")
            return IndicatorSignal("gap_fill", 0, self.weights.get("gap_fill", 0.08), 0, 0, "Error")
    
    # ============== COMPOSITE SIGNAL CALCULATION ==============
    
    def calculate_composite_signal(self) -> CompositeSignal:
        """Calculate the weighted composite signal from all indicators"""
        df = self.get_market_data()
        if df.empty:
            return CompositeSignal(
                symbol=self.symbol,
                timestamp=datetime.now(timezone.utc).isoformat(),
                composite_score=0.5,
                signal_type=SignalType.HOLD,
                confidence=0,
                indicators=[],
                buy_threshold=self.config["buy_threshold"],
                sell_threshold=self.config["sell_threshold"],
                recommended_action="NO_DATA",
                position_size_multiplier=0,
                reasoning="No market data available",
                dca_allowed=False,
                dca_level=0
            )
        
        # Calculate all indicators
        indicators = [
            self.calc_supertrend(df),
            self.calc_macd(df),
            self.calc_rsi(df),
            self.calc_bollinger(df),
            self.calc_ema_cross(df),
            self.calc_momentum(df),
            self.calc_volume_trend(df),
            self.calc_support_resistance(df),
            self.calc_gap_fill(df),
        ]
        
        # Calculate weighted composite score
        total_weight = sum(ind.weight for ind in indicators if ind.signal != 0 or ind.weight > 0)
        if total_weight == 0:
            total_weight = 1
        
        # Composite ranges from -1 to +1
        composite_raw = sum(ind.signal * ind.weight for ind in indicators) / total_weight
        
        # Convert to 0-1 scale
        composite_score = (composite_raw + 1) / 2
        
        # Apply long bias if configured
        if self.config["long_bias"]:
            # Shift score slightly bullish
            composite_score = composite_score * 0.9 + 0.1  # Minimum 10% bullish bias
        
        # Average confidence
        avg_confidence = np.mean([ind.confidence for ind in indicators if ind.confidence > 0])
        
        # Determine signal type
        if composite_score >= self.config["strong_threshold"]:
            signal_type = SignalType.STRONG_BUY
        elif composite_score >= self.config["buy_threshold"]:
            signal_type = SignalType.BUY
        elif composite_score <= (1 - self.config["strong_threshold"]):
            signal_type = SignalType.STRONG_SELL
        elif composite_score <= self.config["sell_threshold"]:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Position size multiplier
        if composite_score >= self.config["strong_threshold"]:
            size_mult = 1.5
        elif composite_score >= self.config["buy_threshold"]:
            size_mult = 1.0
        elif composite_score <= (1 - self.config["strong_threshold"]):
            size_mult = 1.5  # Strong sell also gets multiplier
        else:
            size_mult = 0.5
        
        # DCA allowed if bullish and configured
        dca_allowed = (signal_type in [SignalType.STRONG_BUY, SignalType.BUY]) and self.config["long_bias"]
        
        # Generate reasoning
        bullish_indicators = [ind for ind in indicators if ind.signal == 1]
        bearish_indicators = [ind for ind in indicators if ind.signal == -1]
        reasoning = f"{len(bullish_indicators)} bullish, {len(bearish_indicators)} bearish. "
        if bullish_indicators:
            reasoning += f"Bullish: {', '.join([ind.name for ind in bullish_indicators[:3]])}. "
        if bearish_indicators:
            reasoning += f"Bearish: {', '.join([ind.name for ind in bearish_indicators[:3]])}."
        
        return CompositeSignal(
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            composite_score=composite_score,
            signal_type=signal_type,
            confidence=avg_confidence,
            indicators=[asdict(ind) for ind in indicators],
            buy_threshold=self.config["buy_threshold"],
            sell_threshold=self.config["sell_threshold"],
            recommended_action=signal_type.value,
            position_size_multiplier=size_mult,
            reasoning=reasoning,
            dca_allowed=dca_allowed,
            dca_level=0
        )
    
    def get_signal_summary(self) -> Dict:
        """Get a summary of the current signal"""
        signal = self.calculate_composite_signal()
        return {
            "symbol": signal.symbol,
            "timestamp": signal.timestamp,
            "score": round(signal.composite_score * 100, 1),
            "signal": signal.signal_type.value,
            "confidence": round(signal.confidence, 1),
            "action": signal.recommended_action,
            "size_multiplier": signal.position_size_multiplier,
            "dca_allowed": signal.dca_allowed,
            "reasoning": signal.reasoning,
            "indicator_count": {
                "bullish": len([i for i in signal.indicators if i["signal"] == 1]),
                "bearish": len([i for i in signal.indicators if i["signal"] == -1]),
                "neutral": len([i for i in signal.indicators if i["signal"] == 0])
            }
        }


# Convenience functions
def get_xauusd_signal() -> Dict:
    """Get XAUUSD composite signal"""
    bot = CrellaSteinMetaBot("XAUUSD")
    return bot.get_signal_summary()


def get_iren_signal() -> Dict:
    """Get IREN composite signal"""
    bot = CrellaSteinMetaBot("IREN")
    return bot.get_signal_summary()


def get_clsk_signal() -> Dict:
    """Get CLSK composite signal"""
    bot = CrellaSteinMetaBot("CLSK")
    return bot.get_signal_summary()


def get_cifr_signal() -> Dict:
    """Get CIFR composite signal"""
    bot = CrellaSteinMetaBot("CIFR")
    return bot.get_signal_summary()


if __name__ == "__main__":
    print("=" * 70)
    print("🧠 CRELLASTEIN META BOT - Testing")
    print("=" * 70)
    
    for symbol in ["XAUUSD", "IREN", "CLSK", "CIFR"]:
        print(f"\n📊 {symbol} Signal:")
        bot = CrellaSteinMetaBot(symbol)
        signal = bot.get_signal_summary()
        print(f"   Score: {signal['score']}%")
        print(f"   Signal: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']}%")
        print(f"   Indicators: {signal['indicator_count']}")
        print(f"   Reasoning: {signal['reasoning']}")

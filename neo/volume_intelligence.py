"""
NEO Volume Intelligence Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Volume is the TELL. The Dec 2025 volume explosion preceded Gold's $800 rally.
This module detects:
- Accumulation patterns (institutional buying)
- Distribution patterns (smart money selling)
- Volume divergences (reversal warnings)
- Volume spikes (big moves incoming)

Created: 2026-01-23
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VolumeAnalysis:
    """Complete volume analysis result"""
    
    # Core metrics
    current_volume: float
    volume_sma_10: float
    volume_sma_20: float
    volume_sma_50: float
    volume_ratio: float  # current / sma_20
    
    # Trend
    volume_trend: str  # 'RISING', 'FALLING', 'STABLE'
    volume_trend_strength: float  # 0-100
    
    # Patterns
    is_spike: bool  # >2x average
    spike_magnitude: float  # How many x above average
    accumulation_score: int  # 0-100
    distribution_score: int  # 0-100
    
    # Divergences
    divergence: str  # 'BULLISH', 'BEARISH', 'NONE'
    divergence_strength: float
    
    # Signals
    signal: str  # 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
    confidence_adjustment: int  # -30 to +30
    size_multiplier: float  # 0.5 to 1.5
    
    # Warnings
    warnings: List[str]
    
    # Summary
    summary: str


class VolumeIntelligence:
    """
    Volume Analysis Engine for NEO
    
    Key Insight: The Dec 10, 2025 volume explosion directly correlates
    with Gold's rally from $4,200 to $5,000.
    
    Usage:
        analyzer = VolumeIntelligence()
        analysis = analyzer.analyze(ohlcv_data)
        
        if analysis.accumulation_score > 70:
            # Institutional buying detected
            pass
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional config overrides"""
        self.config = {
            # Volume thresholds
            'spike_threshold': 2.0,      # 2x average = spike
            'high_volume_threshold': 1.5,  # 1.5x average = high
            'low_volume_threshold': 0.5,   # 0.5x average = low
            
            # Trend settings
            'trend_lookback': 10,
            'divergence_lookback': 14,
            
            # Accumulation/Distribution
            'accumulation_threshold': 70,
            'distribution_threshold': 30,
            
            **(config or {})
        }
    
    def analyze(self, ohlcv: pd.DataFrame) -> VolumeAnalysis:
        """
        Perform complete volume analysis
        
        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            VolumeAnalysis with all metrics and signals
        """
        warnings = []
        
        # Ensure lowercase columns
        ohlcv.columns = ohlcv.columns.str.lower()
        
        # Core volume metrics
        current_volume = ohlcv['volume'].iloc[-1]
        vol_sma_10 = ohlcv['volume'].rolling(10).mean().iloc[-1]
        vol_sma_20 = ohlcv['volume'].rolling(20).mean().iloc[-1]
        vol_sma_50 = ohlcv['volume'].rolling(50).mean().iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / vol_sma_20 if vol_sma_20 > 0 else 1.0
        
        # Spike detection
        is_spike = volume_ratio >= self.config['spike_threshold']
        spike_magnitude = volume_ratio if is_spike else 0.0
        
        if is_spike:
            warnings.append(f"🔥 VOLUME SPIKE: {volume_ratio:.1f}x average - BIG MOVE INCOMING")
        
        # Volume trend
        volume_trend, trend_strength = self._calculate_volume_trend(ohlcv)
        
        # Accumulation/Distribution scores
        accumulation_score = self._calculate_accumulation_score(ohlcv)
        distribution_score = 100 - accumulation_score  # Inverse
        
        if accumulation_score >= self.config['accumulation_threshold']:
            warnings.append(f"📈 ACCUMULATION DETECTED: {accumulation_score}% - Institutional buying")
        elif distribution_score >= self.config['accumulation_threshold']:
            warnings.append(f"📉 DISTRIBUTION DETECTED: {distribution_score}% - Smart money selling")
        
        # Divergence detection
        divergence, div_strength = self._detect_divergence(ohlcv)
        
        if divergence == 'BEARISH':
            warnings.append(f"⚠️ BEARISH DIVERGENCE: Price up but volume down - Reversal warning")
        elif divergence == 'BULLISH':
            warnings.append(f"📊 BULLISH DIVERGENCE: Price down but volume up - Accumulation on dip")
        
        # Generate signal
        signal, confidence_adj, size_mult = self._generate_signal(
            volume_ratio, volume_trend, accumulation_score, divergence, ohlcv
        )
        
        # Build summary
        summary = self._build_summary(
            volume_ratio, volume_trend, accumulation_score, divergence, signal
        )
        
        return VolumeAnalysis(
            current_volume=current_volume,
            volume_sma_10=vol_sma_10,
            volume_sma_20=vol_sma_20,
            volume_sma_50=vol_sma_50,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            volume_trend_strength=trend_strength,
            is_spike=is_spike,
            spike_magnitude=spike_magnitude,
            accumulation_score=accumulation_score,
            distribution_score=distribution_score,
            divergence=divergence,
            divergence_strength=div_strength,
            signal=signal,
            confidence_adjustment=confidence_adj,
            size_multiplier=size_mult,
            warnings=warnings,
            summary=summary
        )
    
    def _calculate_volume_trend(self, ohlcv: pd.DataFrame) -> Tuple[str, float]:
        """
        Calculate volume trend direction and strength
        
        Returns:
            (trend: 'RISING'|'FALLING'|'STABLE', strength: 0-100)
        """
        lookback = self.config['trend_lookback']
        volumes = ohlcv['volume'].tail(lookback + 10)
        
        if len(volumes) < lookback:
            return 'STABLE', 50.0
        
        # Linear regression on volume
        x = np.arange(len(volumes))
        slope, _ = np.polyfit(x, volumes, 1)
        
        # Normalize slope
        avg_vol = volumes.mean()
        normalized_slope = (slope / avg_vol) * 100 if avg_vol > 0 else 0
        
        # Also check SMA relationship
        vol_sma_10 = volumes.rolling(10).mean().iloc[-1]
        vol_sma_20 = volumes.rolling(20).mean().iloc[-1] if len(volumes) >= 20 else vol_sma_10
        
        sma_trend = (vol_sma_10 - vol_sma_20) / vol_sma_20 * 100 if vol_sma_20 > 0 else 0
        
        # Combined score
        combined = normalized_slope + sma_trend
        
        if combined > 10:
            return 'RISING', min(100, 50 + combined)
        elif combined < -10:
            return 'FALLING', max(0, 50 + combined)
        else:
            return 'STABLE', 50.0
    
    def _calculate_accumulation_score(self, ohlcv: pd.DataFrame) -> int:
        """
        Calculate accumulation score (0-100)
        
        0 = Heavy distribution (selling)
        50 = Neutral
        100 = Heavy accumulation (buying)
        """
        score = 50  # Start neutral
        
        recent = ohlcv.tail(10)
        
        # 1. Volume trend component (25 points max)
        vol_sma_5 = ohlcv['volume'].tail(5).mean()
        vol_sma_20 = ohlcv['volume'].tail(20).mean()
        
        if vol_sma_5 > vol_sma_20 * 1.2:
            score += 15  # Rising volume
        elif vol_sma_5 > vol_sma_20:
            score += 8
        elif vol_sma_5 < vol_sma_20 * 0.8:
            score -= 15  # Falling volume
        elif vol_sma_5 < vol_sma_20:
            score -= 8
        
        # 2. Price action component (25 points max)
        green_candles = sum(recent['close'] > recent['open'])
        red_candles = len(recent) - green_candles
        
        if green_candles >= 7:
            score += 15
        elif green_candles >= 5:
            score += 8
        elif red_candles >= 7:
            score -= 15
        elif red_candles >= 5:
            score -= 8
        
        # 3. Higher lows pattern (15 points)
        lows = recent['low'].values
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        if higher_lows >= 6:
            score += 15
        elif higher_lows >= 4:
            score += 8
        
        # 4. Lower highs pattern (negative for distribution)
        highs = recent['high'].values
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        
        if lower_highs >= 6:
            score -= 15
        elif lower_highs >= 4:
            score -= 8
        
        # 5. Close near high (accumulation) vs close near low (distribution)
        for _, candle in recent.iterrows():
            candle_range = candle['high'] - candle['low']
            if candle_range > 0:
                close_position = (candle['close'] - candle['low']) / candle_range
                if close_position > 0.7:
                    score += 1  # Closes near high = buying
                elif close_position < 0.3:
                    score -= 1  # Closes near low = selling
        
        return max(0, min(100, score))
    
    def _detect_divergence(self, ohlcv: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect price-volume divergence
        
        Returns:
            (divergence_type: 'BULLISH'|'BEARISH'|'NONE', strength: 0-100)
        """
        lookback = self.config['divergence_lookback']
        
        if len(ohlcv) < lookback:
            return 'NONE', 0.0
        
        recent = ohlcv.tail(lookback)
        
        # Calculate price trend
        prices = recent['close'].values
        price_slope, _ = np.polyfit(np.arange(len(prices)), prices, 1)
        
        # Calculate volume trend
        volumes = recent['volume'].values
        vol_slope, _ = np.polyfit(np.arange(len(volumes)), volumes, 1)
        
        # Normalize
        price_change_pct = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
        vol_change_pct = (volumes[-1] - volumes[0]) / volumes[0] * 100 if volumes[0] > 0 else 0
        
        # Detect divergence
        # Bearish: Price up, Volume down
        if price_change_pct > 2 and vol_change_pct < -10:
            strength = min(100, abs(vol_change_pct))
            return 'BEARISH', strength
        
        # Bullish: Price down, Volume up
        if price_change_pct < -2 and vol_change_pct > 10:
            strength = min(100, abs(vol_change_pct))
            return 'BULLISH', strength
        
        return 'NONE', 0.0
    
    def _generate_signal(
        self,
        volume_ratio: float,
        volume_trend: str,
        accumulation_score: int,
        divergence: str,
        ohlcv: pd.DataFrame
    ) -> Tuple[str, int, float]:
        """
        Generate trading signal based on volume analysis
        
        Returns:
            (signal, confidence_adjustment, size_multiplier)
        """
        signal = 'NEUTRAL'
        confidence_adj = 0
        size_mult = 1.0
        
        # Get price direction
        recent_close = ohlcv['close'].tail(5)
        price_rising = recent_close.iloc[-1] > recent_close.iloc[0]
        
        # High volume scenarios
        if volume_ratio >= 1.5:
            if price_rising:
                if volume_trend == 'RISING':
                    signal = 'STRONG_BUY'
                    confidence_adj = 20
                    size_mult = 1.25
                else:
                    signal = 'BUY'
                    confidence_adj = 10
                    size_mult = 1.0
            else:
                if volume_trend == 'RISING':
                    signal = 'STRONG_SELL'
                    confidence_adj = -20  # Reduces confidence for buys
                    size_mult = 0.5  # Don't fight high volume selling
                else:
                    signal = 'SELL'
                    confidence_adj = -10
                    size_mult = 0.75
        
        # Low volume - weak moves
        elif volume_ratio < 0.5:
            if price_rising:
                signal = 'NEUTRAL'  # Don't trust low volume rallies
                confidence_adj = -15
                size_mult = 0.5
            else:
                signal = 'NEUTRAL'  # Weak selling, potential buy
                confidence_adj = 5
                size_mult = 0.75
        
        # Accumulation/Distribution override
        if accumulation_score >= 75:
            if signal in ['NEUTRAL', 'BUY']:
                signal = 'STRONG_BUY'
                confidence_adj = max(confidence_adj, 15)
                size_mult = max(size_mult, 1.2)
        elif accumulation_score <= 25:
            if signal in ['NEUTRAL', 'SELL']:
                signal = 'STRONG_SELL'
                confidence_adj = min(confidence_adj, -15)
                size_mult = min(size_mult, 0.6)
        
        # Divergence override
        if divergence == 'BEARISH':
            confidence_adj -= 20
            size_mult *= 0.5
            if signal in ['BUY', 'STRONG_BUY']:
                signal = 'NEUTRAL'  # Cancel buy on bearish divergence
        elif divergence == 'BULLISH':
            confidence_adj += 15
            if signal == 'SELL':
                signal = 'NEUTRAL'  # Cancel sell on bullish divergence
        
        return signal, confidence_adj, round(size_mult, 2)
    
    def _build_summary(
        self,
        volume_ratio: float,
        volume_trend: str,
        accumulation_score: int,
        divergence: str,
        signal: str
    ) -> str:
        """Build human-readable summary"""
        lines = []
        lines.append("📊 VOLUME INTELLIGENCE")
        lines.append(f"   Ratio: {volume_ratio:.2f}x avg ({self._ratio_description(volume_ratio)})")
        lines.append(f"   Trend: {volume_trend}")
        lines.append(f"   Accumulation: {accumulation_score}%")
        
        if divergence != 'NONE':
            lines.append(f"   ⚠️ Divergence: {divergence}")
        
        lines.append(f"   Signal: {signal}")
        
        return "\n".join(lines)
    
    def _ratio_description(self, ratio: float) -> str:
        """Get description for volume ratio"""
        if ratio >= 2.0:
            return "🔥 SPIKE"
        elif ratio >= 1.5:
            return "HIGH"
        elif ratio >= 0.8:
            return "NORMAL"
        elif ratio >= 0.5:
            return "LOW"
        else:
            return "VERY LOW"
    
    def get_whale_detection(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Detect potential whale/institutional activity
        
        Patterns:
        - Sustained high volume (not spike-and-die)
        - Consistent accumulation
        - Price steadily rising without violent pullbacks
        """
        # Check last 10 days
        recent = ohlcv.tail(10)
        
        # Volume consistency (are high volume days sustained?)
        vol_sma_20 = ohlcv['volume'].tail(20).mean()
        high_vol_days = sum(recent['volume'] > vol_sma_20 * 1.3)
        
        # Price consistency (steady rise, no major drops)
        price_changes = recent['close'].pct_change().dropna()
        max_drop = price_changes.min() if len(price_changes) > 0 else 0
        avg_change = price_changes.mean() if len(price_changes) > 0 else 0
        
        # Accumulation pattern
        accumulation = self._calculate_accumulation_score(ohlcv)
        
        # Score whale probability
        whale_score = 0
        reasons = []
        
        if high_vol_days >= 6:
            whale_score += 30
            reasons.append(f"Sustained high volume ({high_vol_days}/10 days)")
        
        if avg_change > 0.003 and max_drop > -0.03:
            whale_score += 30
            reasons.append("Steady price rise, controlled pullbacks")
        
        if accumulation >= 70:
            whale_score += 25
            reasons.append(f"Strong accumulation ({accumulation}%)")
        
        if high_vol_days >= 4 and accumulation >= 60:
            whale_score += 15
            reasons.append("Volume + accumulation confluence")
        
        is_whale = whale_score >= 60
        
        return {
            'detected': is_whale,
            'probability': min(100, whale_score),
            'reasons': reasons,
            'high_vol_days': high_vol_days,
            'accumulation': accumulation,
            'recommendation': "Follow the whale - this is systematic buying" if is_whale else "No clear institutional pattern"
        }


def get_volume_analysis(ohlcv: pd.DataFrame) -> Dict:
    """
    Quick function to get volume analysis for NEO integration
    """
    analyzer = VolumeIntelligence()
    analysis = analyzer.analyze(ohlcv)
    
    return {
        'volume_ratio': round(analysis.volume_ratio, 2),
        'volume_trend': analysis.volume_trend,
        'is_spike': analysis.is_spike,
        'accumulation_score': analysis.accumulation_score,
        'divergence': analysis.divergence,
        'signal': analysis.signal,
        'confidence_adjustment': analysis.confidence_adjustment,
        'size_multiplier': analysis.size_multiplier,
        'warnings': analysis.warnings,
        'summary': analysis.summary
    }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("VOLUME INTELLIGENCE TEST")
    print("="*60)
    
    # Get Gold data
    gold = yf.download('GC=F', period='3mo', interval='1d', progress=False)
    # Handle MultiIndex columns from yfinance
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = gold.columns.str.lower()
    
    analyzer = VolumeIntelligence()
    
    # Full analysis
    analysis = analyzer.analyze(gold)
    
    print(f"\n{analysis.summary}")
    print(f"\nVolume Ratio: {analysis.volume_ratio:.2f}x")
    print(f"Is Spike: {analysis.is_spike}")
    print(f"Accumulation Score: {analysis.accumulation_score}%")
    print(f"Divergence: {analysis.divergence} (strength: {analysis.divergence_strength:.0f})")
    print(f"Signal: {analysis.signal}")
    print(f"Confidence Adjustment: {analysis.confidence_adjustment:+d}%")
    print(f"Size Multiplier: {analysis.size_multiplier}x")
    
    if analysis.warnings:
        print("\nWarnings:")
        for w in analysis.warnings:
            print(f"  {w}")
    
    # Whale detection
    print("\n" + "="*60)
    print("WHALE DETECTION")
    print("="*60)
    whale = analyzer.get_whale_detection(gold)
    print(f"Whale Detected: {whale['detected']}")
    print(f"Probability: {whale['probability']}%")
    print(f"High Volume Days: {whale['high_vol_days']}/10")
    if whale['reasons']:
        print("Reasons:")
        for r in whale['reasons']:
            print(f"  • {r}")
    print(f"Recommendation: {whale['recommendation']}")

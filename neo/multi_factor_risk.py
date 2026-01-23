"""
Multi-Factor Risk Scoring Module
Ported from Hunter Bot patterns (https://github.com/mikegianfelice/Hunter)

Features:
- 6 weighted risk factors
- Dynamic position sizing based on risk score
- Market regime detection
- Correlation-aware risk

Risk Score: 0-100
- 0-25: LOW RISK → Full position size
- 25-50: MODERATE → 75% position
- 50-75: HIGH → 50% position
- 75-100: EXTREME → 25% position or skip
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    total_score: float  # 0-100
    risk_level: str  # LOW, MODERATE, HIGH, EXTREME
    position_multiplier: float  # 0.25-1.0
    factors: Dict[str, float]  # Individual factor scores
    warnings: List[str]
    recommendation: str


class MultiFactorRisk:
    """
    Hunter-inspired Multi-Factor Risk Scoring System
    
    Factors:
    1. Volatility (25%) - ATR-based, regime detection
    2. Trend Strength (20%) - ADX, trend alignment
    3. Volume Profile (20%) - Volume anomalies, liquidity
    4. Correlation (15%) - Cross-asset risk
    5. Sentiment (10%) - Crowd positioning
    6. MM Activity (10%) - Stop hunt probability
    
    Usage:
        risk = MultiFactorRisk()
        assessment = risk.assess(
            volatility_data={'atr': 25, 'atr_percentile': 75},
            trend_data={'adx': 35, 'trend_aligned': True},
            volume_data={'volume_ratio': 1.5, 'volume_trend': 'rising'},
            correlation_data={'gold_dxy': -0.8, 'gold_spy': 0.3},
            sentiment_data={'retail_long_pct': 75, 'fear_greed': 80},
            mm_data={'stop_hunt_prob': 0.6, 'liquidity_thin': False}
        )
    """
    
    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'volatility': 0.25,
        'trend_strength': 0.20,
        'volume_profile': 0.20,
        'correlation': 0.15,
        'sentiment': 0.10,
        'mm_activity': 0.10
    }
    
    # Risk thresholds
    RISK_LEVELS = {
        'LOW': (0, 25),
        'MODERATE': (25, 50),
        'HIGH': (50, 75),
        'EXTREME': (75, 100)
    }
    
    # Position sizing multipliers
    POSITION_MULTIPLIERS = {
        'LOW': 1.0,
        'MODERATE': 0.75,
        'HIGH': 0.50,
        'EXTREME': 0.25
    }
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Initialize with optional custom weights
        
        Args:
            weights: Override default factor weights
        """
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def assess(
        self,
        volatility_data: Dict,
        trend_data: Dict,
        volume_data: Dict,
        correlation_data: Dict,
        sentiment_data: Dict,
        mm_data: Dict
    ) -> RiskAssessment:
        """
        Perform full risk assessment
        
        Returns:
            RiskAssessment with score, level, and recommendations
        """
        warnings = []
        
        # Calculate individual factor scores (0-100, higher = more risk)
        factors = {}
        
        factors['volatility'] = self._score_volatility(volatility_data, warnings)
        factors['trend_strength'] = self._score_trend(trend_data, warnings)
        factors['volume_profile'] = self._score_volume(volume_data, warnings)
        factors['correlation'] = self._score_correlation(correlation_data, warnings)
        factors['sentiment'] = self._score_sentiment(sentiment_data, warnings)
        factors['mm_activity'] = self._score_mm_activity(mm_data, warnings)
        
        # Calculate weighted total
        total_score = sum(
            factors[factor] * self.weights[factor]
            for factor in factors
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(total_score)
        position_multiplier = self.POSITION_MULTIPLIERS[risk_level]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            total_score, risk_level, factors, warnings
        )
        
        return RiskAssessment(
            total_score=round(total_score, 1),
            risk_level=risk_level,
            position_multiplier=position_multiplier,
            factors={k: round(v, 1) for k, v in factors.items()},
            warnings=warnings,
            recommendation=recommendation
        )
    
    def _score_volatility(self, data: Dict, warnings: List) -> float:
        """
        Score volatility risk (0-100)
        
        Inputs:
            - atr: Current ATR
            - atr_percentile: Where current ATR ranks (0-100)
            - vix: VIX level (if available)
            - regime: 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
        """
        score = 0
        
        # ATR percentile (50% weight of volatility score)
        atr_pct = data.get('atr_percentile', 50)
        score += atr_pct * 0.5
        
        # VIX level (30% weight)
        vix = data.get('vix', 20)
        if vix > 30:
            score += 30
            warnings.append(f"⚠️ VIX elevated: {vix}")
        elif vix > 25:
            score += 20
        elif vix > 20:
            score += 10
        
        # Regime adjustment (20% weight)
        regime = data.get('regime', 'NORMAL')
        regime_scores = {'LOW': 0, 'NORMAL': 10, 'HIGH': 15, 'EXTREME': 20}
        score += regime_scores.get(regime, 10)
        
        if regime == 'EXTREME':
            warnings.append("🔴 EXTREME volatility regime - reduce size")
        
        return min(100, score)
    
    def _score_trend(self, data: Dict, warnings: List) -> float:
        """
        Score trend risk (0-100)
        Higher score = riskier (no clear trend or fighting trend)
        
        Inputs:
            - adx: ADX value (0-100)
            - trend_aligned: Whether our signal aligns with trend
            - trend_strength: 'WEAK', 'MODERATE', 'STRONG'
        """
        score = 50  # Neutral baseline
        
        adx = data.get('adx', 25)
        trend_aligned = data.get('trend_aligned', True)
        
        # Strong trend = lower risk IF aligned
        if adx > 40:
            if trend_aligned:
                score -= 25  # Very favorable
            else:
                score += 30  # Fighting strong trend = danger
                warnings.append("⚠️ Trading against strong trend (ADX > 40)")
        elif adx > 25:
            if trend_aligned:
                score -= 10
            else:
                score += 15
        else:
            # Weak trend (ranging) - moderate risk
            score += 5
            if data.get('signal_type') == 'TREND_FOLLOW':
                warnings.append("📊 Weak trend - consider range strategy")
        
        return max(0, min(100, score))
    
    def _score_volume(self, data: Dict, warnings: List) -> float:
        """
        Score volume risk (0-100)
        
        Inputs:
            - volume_ratio: Current vol / average vol
            - volume_trend: 'rising', 'falling', 'stable'
            - liquidity: 'HIGH', 'NORMAL', 'LOW'
        """
        score = 25  # Neutral baseline
        
        vol_ratio = data.get('volume_ratio', 1.0)
        vol_trend = data.get('volume_trend', 'stable')
        liquidity = data.get('liquidity', 'NORMAL')
        
        # Very high volume can signal exhaustion
        if vol_ratio > 3.0:
            score += 40
            warnings.append("📈 Volume spike (3x+) - possible exhaustion")
        elif vol_ratio > 2.0:
            score += 20
        elif vol_ratio < 0.5:
            score += 30  # Low volume = thin market
            warnings.append("📉 Low volume - watch for gaps")
        
        # Volume trend
        if vol_trend == 'falling' and data.get('price_rising', False):
            score += 20  # Bearish divergence
            warnings.append("⚠️ Rising price on falling volume")
        
        # Liquidity
        if liquidity == 'LOW':
            score += 25
            warnings.append("💧 Low liquidity - wider spreads expected")
        elif liquidity == 'HIGH':
            score -= 10
        
        return max(0, min(100, score))
    
    def _score_correlation(self, data: Dict, warnings: List) -> float:
        """
        Score correlation risk (0-100)
        
        Inputs:
            - gold_dxy: Gold-DXY correlation (-1 to 1)
            - gold_spy: Gold-SPY correlation
            - correlation_breakdown: Whether normal correlations are breaking
        """
        score = 25  # Neutral baseline
        
        gold_dxy = data.get('gold_dxy', -0.5)  # Normally negative
        gold_spy = data.get('gold_spy', 0.0)
        breakdown = data.get('correlation_breakdown', False)
        
        # Gold-DXY should be negative; if positive, something's off
        if gold_dxy > 0.3:
            score += 30
            warnings.append("⚠️ Gold-DXY correlation breakdown (usually negative)")
        elif gold_dxy > 0:
            score += 15
        
        # Strong correlation with risk assets = risk-on regime
        if abs(gold_spy) > 0.7:
            score += 20
            warnings.append(f"🔗 High Gold-SPY correlation ({gold_spy:.2f})")
        
        # General breakdown
        if breakdown:
            score += 25
            warnings.append("🔴 Correlation breakdown - markets unstable")
        
        return max(0, min(100, score))
    
    def _score_sentiment(self, data: Dict, warnings: List) -> float:
        """
        Score sentiment risk (0-100)
        Extreme sentiment = contrarian risk
        
        Inputs:
            - retail_long_pct: % of retail traders long
            - fear_greed: Fear & Greed index (0-100)
            - social_hype: Social media hype score
        """
        score = 25  # Neutral baseline
        
        retail_long = data.get('retail_long_pct', 50)
        fear_greed = data.get('fear_greed', 50)
        social_hype = data.get('social_hype', 50)
        
        # Extreme retail positioning
        if retail_long > 80:
            score += 35
            warnings.append("🐑 80%+ retail LONG - contrarian risk")
        elif retail_long > 70:
            score += 20
        elif retail_long < 20:
            score += 25  # Extreme bearishness = potential bounce
            warnings.append("🐻 80%+ retail SHORT - potential squeeze")
        
        # Fear & Greed extremes
        if fear_greed > 80:
            score += 25
            warnings.append("🔴 Extreme Greed - correction risk")
        elif fear_greed < 20:
            score += 15
            warnings.append("😨 Extreme Fear - capitulation possible")
        
        # Social hype (AHI-like)
        if social_hype > 75:
            score += 15
        
        return max(0, min(100, score))
    
    def _score_mm_activity(self, data: Dict, warnings: List) -> float:
        """
        Score Market Maker activity risk (0-100)
        
        Inputs:
            - stop_hunt_prob: Probability of stop hunt (0-1)
            - liquidity_thin: Whether we're in thin liquidity zone
            - recent_hunt: Whether a stop hunt just occurred
        """
        score = 20  # Neutral baseline
        
        hunt_prob = data.get('stop_hunt_prob', 0.3)
        liquidity_thin = data.get('liquidity_thin', False)
        recent_hunt = data.get('recent_hunt', False)
        
        # Stop hunt probability
        if hunt_prob > 0.7:
            score += 40
            warnings.append("🎯 High stop hunt probability (>70%)")
        elif hunt_prob > 0.5:
            score += 25
        
        # Thin liquidity zones
        if liquidity_thin:
            score += 25
            warnings.append("💧 Price in thin liquidity zone")
        
        # Recent hunt (better entry)
        if recent_hunt:
            score -= 20  # Actually reduces risk (trap already sprung)
        
        return max(0, min(100, score))
    
    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        for level, (low, high) in self.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return 'EXTREME'
    
    def _generate_recommendation(
        self,
        score: float,
        level: str,
        factors: Dict,
        warnings: List
    ) -> str:
        """Generate human-readable recommendation"""
        if level == 'LOW':
            return f"✅ LOW RISK ({score:.0f}/100) - Full position size recommended"
        elif level == 'MODERATE':
            # Find the highest risk factor
            top_risk = max(factors, key=factors.get)
            return f"⚠️ MODERATE RISK ({score:.0f}/100) - Reduce to 75% size. Watch: {top_risk}"
        elif level == 'HIGH':
            top_risks = sorted(factors, key=factors.get, reverse=True)[:2]
            return f"🔴 HIGH RISK ({score:.0f}/100) - Reduce to 50% size. Concerns: {', '.join(top_risks)}"
        else:
            return f"🚨 EXTREME RISK ({score:.0f}/100) - Consider skipping trade or 25% size only"
    
    def calculate_position_size(
        self,
        base_lots: float,
        risk_assessment: RiskAssessment,
        max_reduction: float = 0.75  # Maximum reduction from base
    ) -> Tuple[float, str]:
        """
        Calculate position size based on risk assessment
        
        Args:
            base_lots: Normal position size
            risk_assessment: Output from assess()
            max_reduction: Maximum % to reduce (default 75%)
            
        Returns:
            (adjusted_lots, reason)
        """
        adjusted = base_lots * risk_assessment.position_multiplier
        adjusted = max(adjusted, base_lots * (1 - max_reduction))
        
        reason = f"Base {base_lots} × {risk_assessment.position_multiplier:.2f} = {adjusted:.2f} ({risk_assessment.risk_level} risk)"
        
        return round(adjusted, 2), reason


# Quick risk check function
def quick_risk_check(
    atr_percentile: float = 50,
    adx: float = 25,
    volume_ratio: float = 1.0,
    retail_long_pct: float = 50,
    stop_hunt_prob: float = 0.3
) -> Dict:
    """
    Quick risk assessment with minimal inputs
    
    Returns:
        Dictionary with risk_score, risk_level, position_multiplier
    """
    risk = MultiFactorRisk()
    
    assessment = risk.assess(
        volatility_data={'atr_percentile': atr_percentile, 'regime': 'NORMAL'},
        trend_data={'adx': adx, 'trend_aligned': True},
        volume_data={'volume_ratio': volume_ratio, 'liquidity': 'NORMAL'},
        correlation_data={'gold_dxy': -0.5, 'gold_spy': 0.1},
        sentiment_data={'retail_long_pct': retail_long_pct, 'fear_greed': 50},
        mm_data={'stop_hunt_prob': stop_hunt_prob, 'liquidity_thin': False}
    )
    
    return {
        'risk_score': assessment.total_score,
        'risk_level': assessment.risk_level,
        'position_multiplier': assessment.position_multiplier,
        'recommendation': assessment.recommendation
    }


# Example usage and test
if __name__ == "__main__":
    risk = MultiFactorRisk()
    
    print("=== Multi-Factor Risk Assessment Tests ===\n")
    
    # Test 1: Low risk scenario
    assessment = risk.assess(
        volatility_data={'atr_percentile': 30, 'vix': 15, 'regime': 'NORMAL'},
        trend_data={'adx': 35, 'trend_aligned': True},
        volume_data={'volume_ratio': 1.2, 'liquidity': 'HIGH'},
        correlation_data={'gold_dxy': -0.6, 'gold_spy': 0.1},
        sentiment_data={'retail_long_pct': 55, 'fear_greed': 55},
        mm_data={'stop_hunt_prob': 0.2, 'liquidity_thin': False}
    )
    print("Test 1 - Low Risk Scenario:")
    print(f"  Total Score: {assessment.total_score}/100")
    print(f"  Level: {assessment.risk_level}")
    print(f"  Position Multiplier: {assessment.position_multiplier}")
    print(f"  Factors: {assessment.factors}")
    print(f"  Recommendation: {assessment.recommendation}\n")
    
    # Test 2: High risk scenario
    assessment = risk.assess(
        volatility_data={'atr_percentile': 85, 'vix': 32, 'regime': 'HIGH'},
        trend_data={'adx': 45, 'trend_aligned': False},  # Fighting trend!
        volume_data={'volume_ratio': 3.5, 'liquidity': 'LOW'},
        correlation_data={'gold_dxy': 0.4, 'correlation_breakdown': True},
        sentiment_data={'retail_long_pct': 85, 'fear_greed': 88},
        mm_data={'stop_hunt_prob': 0.75, 'liquidity_thin': True}
    )
    print("Test 2 - High Risk Scenario:")
    print(f"  Total Score: {assessment.total_score}/100")
    print(f"  Level: {assessment.risk_level}")
    print(f"  Position Multiplier: {assessment.position_multiplier}")
    print(f"  Warnings:")
    for w in assessment.warnings:
        print(f"    {w}")
    print(f"  Recommendation: {assessment.recommendation}\n")
    
    # Test 3: Position sizing
    base_lots = 0.5
    adjusted, reason = risk.calculate_position_size(base_lots, assessment)
    print(f"Test 3 - Position Sizing:")
    print(f"  Base: {base_lots} lots")
    print(f"  Adjusted: {adjusted} lots")
    print(f"  Reason: {reason}\n")
    
    # Test 4: Quick check
    quick = quick_risk_check(
        atr_percentile=60,
        adx=30,
        volume_ratio=1.5,
        retail_long_pct=70,
        stop_hunt_prob=0.4
    )
    print("Test 4 - Quick Risk Check:")
    print(f"  Score: {quick['risk_score']}/100")
    print(f"  Level: {quick['risk_level']}")
    print(f"  Multiplier: {quick['position_multiplier']}")

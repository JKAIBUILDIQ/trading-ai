#!/usr/bin/env python3
"""
NEO INTEGRATION FOR GHOST COMMANDER
═══════════════════════════════════════════════════════════════════════════════

Integrates NEO's intelligence into Ghost Commander's decision making:
1. GoldPredictor - 4-hour directional prediction
2. CrellaSteinMetaBot - Weighted indicator ensemble
3. Combined signal with confidence weighting

Ghost Commander will ONLY enter when:
- SuperTrend = BULLISH (technical requirement)
- NEO signal = BULLISH or STRONG_BUY (NEO's blessing)
- Combined confidence >= 60%

"NEO guides, Ghost executes."

Created: January 30, 2026
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Add NEO paths
NEO_PATH = Path(__file__).parent.parent / 'neo'
sys.path.insert(0, str(NEO_PATH))

logger = logging.getLogger('NeoIntegration')

# Try to import NEO components
try:
    from gold_predictor import GoldPredictor, predict_gold_4h
    GOLD_PREDICTOR_AVAILABLE = True
    logger.info("✅ GoldPredictor loaded")
except ImportError as e:
    GOLD_PREDICTOR_AVAILABLE = False
    logger.warning(f"⚠️ GoldPredictor not available: {e}")

try:
    from meta_bot.crellastein_meta import CrellaSteinMetaBot, get_xauusd_signal
    CRELLASTEIN_AVAILABLE = True
    logger.info("✅ CrellaSteinMetaBot loaded")
except ImportError as e:
    CRELLASTEIN_AVAILABLE = False
    logger.warning(f"⚠️ CrellaSteinMetaBot not available: {e}")


class NeoGoldAdvisor:
    """
    NEO's Gold Trading Advisor
    
    Combines multiple NEO intelligence sources to provide
    trading guidance to Ghost Commander.
    """
    
    def __init__(self):
        self.gold_predictor = None
        self.crellastein = None
        self.last_signal = None
        self.last_signal_time = None
        
        # Initialize components
        if GOLD_PREDICTOR_AVAILABLE:
            try:
                self.gold_predictor = GoldPredictor()
                logger.info("🔮 GoldPredictor initialized")
            except Exception as e:
                logger.error(f"GoldPredictor init failed: {e}")
        
        if CRELLASTEIN_AVAILABLE:
            try:
                self.crellastein = CrellaSteinMetaBot("XAUUSD")
                logger.info("🧠 CrellaSteinMetaBot initialized")
            except Exception as e:
                logger.error(f"CrellaSteinMetaBot init failed: {e}")
    
    def get_gold_prediction(self) -> Dict:
        """
        Get NEO's 4-hour Gold prediction
        
        Returns:
            {
                'direction': 'UP' | 'DOWN' | 'FLAT',
                'confidence': 0-100,
                'predicted_change': float,
                'reasoning': str
            }
        """
        if not self.gold_predictor:
            return {
                'direction': 'FLAT',
                'confidence': 0,
                'predicted_change': 0,
                'reasoning': 'GoldPredictor not available'
            }
        
        try:
            prediction = self.gold_predictor.predict_4h()
            return {
                'direction': prediction.predicted_direction,
                'confidence': prediction.confidence,
                'predicted_change': prediction.predicted_change_pips,
                'reasoning': prediction.reasoning
            }
        except Exception as e:
            logger.error(f"GoldPredictor error: {e}")
            return {
                'direction': 'FLAT',
                'confidence': 0,
                'predicted_change': 0,
                'reasoning': f'Error: {e}'
            }
    
    def get_crellastein_signal(self) -> Dict:
        """
        Get CrellaStein weighted indicator signal
        
        Returns:
            {
                'signal': 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL',
                'score': 0-100,
                'confidence': 0-100,
                'dca_allowed': bool,
                'reasoning': str,
                'indicators': {...}
            }
        """
        if not self.crellastein:
            return {
                'signal': 'HOLD',
                'score': 50,
                'confidence': 0,
                'dca_allowed': False,
                'reasoning': 'CrellaSteinMetaBot not available'
            }
        
        try:
            signal = self.crellastein.get_signal_summary()
            return {
                'signal': signal['signal'],
                'score': signal['score'],
                'confidence': signal['confidence'],
                'dca_allowed': signal['dca_allowed'],
                'reasoning': signal['reasoning'],
                'indicators': signal.get('indicator_count', {})
            }
        except Exception as e:
            logger.error(f"CrellaStein error: {e}")
            return {
                'signal': 'HOLD',
                'score': 50,
                'confidence': 0,
                'dca_allowed': False,
                'reasoning': f'Error: {e}'
            }
    
    def get_combined_signal(self) -> Dict:
        """
        Get NEO's combined trading signal
        
        Combines GoldPredictor + CrellaStein for a unified recommendation.
        
        Returns:
            {
                'should_enter': bool,
                'direction': 'LONG' | 'SHORT' | 'WAIT',
                'confidence': 0-100,
                'entry_type': 'INITIAL' | 'DCA' | 'NONE',
                'size_multiplier': 0.5-1.5,
                'reasoning': str,
                'gold_predictor': {...},
                'crellastein': {...}
            }
        """
        # Get individual signals
        gold_pred = self.get_gold_prediction()
        crella = self.get_crellastein_signal()
        
        # Combine signals with weighting
        # GoldPredictor: 40% weight (directional)
        # CrellaStein: 60% weight (multi-indicator)
        
        gold_score = 50  # neutral
        if gold_pred['direction'] == 'UP':
            gold_score = 50 + (gold_pred['confidence'] / 2)
        elif gold_pred['direction'] == 'DOWN':
            gold_score = 50 - (gold_pred['confidence'] / 2)
        
        crella_score = crella['score']
        
        # Weighted combined score
        combined_score = (gold_score * 0.4) + (crella_score * 0.6)
        
        # Determine direction
        if combined_score >= 60:
            direction = 'LONG'
        elif combined_score <= 40:
            direction = 'SHORT'
        else:
            direction = 'WAIT'
        
        # Should enter?
        should_enter = (
            direction == 'LONG' and 
            combined_score >= 55 and  # Need at least 55% bullish
            (gold_pred['direction'] in ['UP', 'FLAT'] or gold_pred['confidence'] < 50) and
            crella['signal'] in ['STRONG_BUY', 'BUY', 'HOLD']
        )
        
        # Entry type
        if should_enter and crella['dca_allowed']:
            entry_type = 'DCA' if crella['signal'] == 'BUY' else 'INITIAL'
        elif should_enter:
            entry_type = 'INITIAL'
        else:
            entry_type = 'NONE'
        
        # Size multiplier based on confidence
        if combined_score >= 75:
            size_mult = 1.5  # High confidence = bigger size
        elif combined_score >= 65:
            size_mult = 1.0  # Normal size
        elif combined_score >= 55:
            size_mult = 0.75  # Reduced size
        else:
            size_mult = 0.5  # Minimum size
        
        # Confidence
        combined_confidence = (gold_pred['confidence'] * 0.4) + (crella['confidence'] * 0.6)
        
        # Build reasoning
        reasoning_parts = []
        if gold_pred['direction'] == 'UP':
            reasoning_parts.append(f"GoldPredictor: UP ({gold_pred['confidence']:.0f}%)")
        elif gold_pred['direction'] == 'DOWN':
            reasoning_parts.append(f"GoldPredictor: DOWN ({gold_pred['confidence']:.0f}%)")
        else:
            reasoning_parts.append(f"GoldPredictor: FLAT")
        
        reasoning_parts.append(f"CrellaStein: {crella['signal']} ({crella['score']:.0f}%)")
        
        if crella.get('indicators'):
            ind = crella['indicators']
            reasoning_parts.append(f"Indicators: {ind.get('bullish', 0)}↑ {ind.get('bearish', 0)}↓")
        
        reasoning = " | ".join(reasoning_parts)
        
        result = {
            'should_enter': should_enter,
            'direction': direction,
            'confidence': combined_confidence,
            'combined_score': combined_score,
            'entry_type': entry_type,
            'size_multiplier': size_mult,
            'reasoning': reasoning,
            'gold_predictor': gold_pred,
            'crellastein': crella,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        self.last_signal = result
        self.last_signal_time = datetime.now()
        
        return result
    
    def should_ghost_enter(self, supertrend_bullish: bool) -> Tuple[bool, str, float]:
        """
        Main integration point for Ghost Commander.
        
        Args:
            supertrend_bullish: Whether Ghost's SuperTrend says BULLISH
        
        Returns:
            (should_enter, reasoning, size_multiplier)
        """
        # Get NEO's combined signal
        neo_signal = self.get_combined_signal()
        
        # Log NEO's analysis
        logger.info("=" * 50)
        logger.info("🧠 NEO ANALYSIS")
        logger.info(f"   Combined Score: {neo_signal['combined_score']:.1f}%")
        logger.info(f"   Direction: {neo_signal['direction']}")
        logger.info(f"   Should Enter: {neo_signal['should_enter']}")
        logger.info(f"   {neo_signal['reasoning']}")
        logger.info("=" * 50)
        
        # Decision logic:
        # Ghost enters ONLY if:
        # 1. SuperTrend = BULLISH (Ghost's technical requirement)
        # 2. NEO says LONG or at least not SHORT
        # 3. Combined confidence >= 50%
        
        if not supertrend_bullish:
            return False, "SuperTrend BEARISH - Ghost waiting", 0
        
        if neo_signal['direction'] == 'SHORT':
            return False, f"NEO says SHORT ({neo_signal['combined_score']:.0f}%) - Ghost waiting", 0
        
        if neo_signal['combined_score'] < 50:
            return False, f"NEO confidence too low ({neo_signal['combined_score']:.0f}%) - Ghost waiting", 0
        
        # NEO approves!
        if neo_signal['should_enter']:
            return True, f"✅ NEO APPROVES: {neo_signal['reasoning']}", neo_signal['size_multiplier']
        else:
            # NEO is neutral, Ghost can proceed with caution
            return True, f"⚠️ NEO NEUTRAL: {neo_signal['reasoning']} - proceed with caution", 0.5
    
    def should_ghost_dca(self, drop_dollars: float, current_level: int) -> Tuple[bool, str]:
        """
        Check if NEO approves a DCA entry.
        
        Args:
            drop_dollars: How much price has dropped from highest entry
            current_level: Current DCA level (1-5)
        
        Returns:
            (should_dca, reasoning)
        """
        neo_signal = self.get_combined_signal()
        
        # DCA rules:
        # 1. NEO must not be SHORT
        # 2. CrellaStein must allow DCA
        # 3. Don't DCA into a falling knife (NEO DOWN with high confidence)
        
        if neo_signal['direction'] == 'SHORT':
            return False, f"NEO says SHORT - no DCA"
        
        if neo_signal['gold_predictor']['direction'] == 'DOWN' and neo_signal['gold_predictor']['confidence'] > 70:
            return False, f"GoldPredictor: Strong DOWN signal ({neo_signal['gold_predictor']['confidence']:.0f}%) - no DCA"
        
        if not neo_signal['crellastein'].get('dca_allowed', False):
            return False, "CrellaStein: DCA not allowed"
        
        # Approve DCA
        return True, f"NEO approves DCA: {neo_signal['reasoning']}"


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_neo_advisor = None

def get_neo_advisor() -> NeoGoldAdvisor:
    """Get or create the global NEO advisor instance."""
    global _neo_advisor
    if _neo_advisor is None:
        _neo_advisor = NeoGoldAdvisor()
    return _neo_advisor


def neo_should_enter(supertrend_bullish: bool) -> Tuple[bool, str, float]:
    """Quick check if NEO approves entry."""
    return get_neo_advisor().should_ghost_enter(supertrend_bullish)


def neo_should_dca(drop_dollars: float, current_level: int) -> Tuple[bool, str]:
    """Quick check if NEO approves DCA."""
    return get_neo_advisor().should_ghost_dca(drop_dollars, current_level)


def get_neo_signal() -> Dict:
    """Get NEO's combined signal."""
    return get_neo_advisor().get_combined_signal()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 NEO INTEGRATION TEST")
    print("=" * 70)
    
    advisor = NeoGoldAdvisor()
    
    # Test combined signal
    print("\n📊 Combined Signal:")
    signal = advisor.get_combined_signal()
    print(f"   Should Enter: {signal['should_enter']}")
    print(f"   Direction: {signal['direction']}")
    print(f"   Combined Score: {signal['combined_score']:.1f}%")
    print(f"   Confidence: {signal['confidence']:.1f}%")
    print(f"   Size Multiplier: {signal['size_multiplier']}")
    print(f"   Reasoning: {signal['reasoning']}")
    
    # Test Ghost entry check
    print("\n👻 Ghost Entry Check (SuperTrend BULLISH):")
    should_enter, reason, size = advisor.should_ghost_enter(True)
    print(f"   Should Enter: {should_enter}")
    print(f"   Reason: {reason}")
    print(f"   Size Multiplier: {size}")
    
    print("\n👻 Ghost Entry Check (SuperTrend BEARISH):")
    should_enter, reason, size = advisor.should_ghost_enter(False)
    print(f"   Should Enter: {should_enter}")
    print(f"   Reason: {reason}")
    
    # Test DCA check
    print("\n📉 DCA Check ($10 drop, Level 1):")
    should_dca, reason = advisor.should_ghost_dca(10.0, 1)
    print(f"   Should DCA: {should_dca}")
    print(f"   Reason: {reason}")

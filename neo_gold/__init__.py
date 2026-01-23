"""
NEO-GOLD: Gold Trading Specialist
═══════════════════════════════════════════════════════════════════════

A specialized AI trader focused ONLY on Gold (XAUUSD).

Components:
- features.py: Gold-specific feature extraction
- patterns.py: Chart pattern detection
- mm_predictor.py: Market Maker behavior prediction
- rules.py: Hardcoded trading rules
- neo_gold.py: Main brain

Usage:
    from neo_gold import NeoGold
    
    neo = NeoGold()
    neo.run()
"""

from .neo_gold import NeoGold, main
from .features import GoldFeatureExtractor
from .patterns import GoldPatternDetector, PatternType, DetectedPattern
from .mm_predictor import MarketMakerPredictor, MMTactic, MMPrediction
from .rules import GoldTradingRules, RuleCheck

__all__ = [
    'NeoGold',
    'main',
    'GoldFeatureExtractor',
    'GoldPatternDetector',
    'PatternType',
    'DetectedPattern',
    'MarketMakerPredictor',
    'MMTactic',
    'MMPrediction',
    'GoldTradingRules',
    'RuleCheck'
]

__version__ = "1.0.0"

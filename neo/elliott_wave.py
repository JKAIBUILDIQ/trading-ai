"""
NEO Elliott Wave Analysis - Wave Counting & Projections for Gold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Elliott Wave Theory: Markets move in 5-wave impulses + 3-wave corrections

Gold Supercycle ($1,615 → $5,000+):

Wave (I):   $1,615 → $2,075 (impulse)
Wave (II):  $2,075 → $1,810 (correction, 61.8% retrace)
Wave (III): $1,810 → $5,500+? (CURRENT - extended wave!)
  └── Wave 1: $1,810 → $2,450
  └── Wave 2: $2,450 → $2,280 (38.2% retrace)
  └── Wave 3: $2,280 → $4,200 (extended!)
  └── Wave 4: $4,200 → $3,950 (shallow)
  └── Wave 5: $3,950 → $5,500+? ← CURRENT!
Wave (IV): Future correction
Wave (V):  Final push to ATH

"The market is a device for transferring money from the impatient to the patient."

Created: 2026-01-23
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# ============================================================================
# GOLD ELLIOTT WAVE STRUCTURE (Historical)
# ============================================================================

GOLD_WAVE_STRUCTURE = {
    'supercycle': {
        'wave_I': {
            'start': 1615, 'end': 2075, 'start_date': '2022-09', 'end_date': '2023-05',
            'type': 'IMPULSE', 'size': 460
        },
        'wave_II': {
            'start': 2075, 'end': 1810, 'start_date': '2023-05', 'end_date': '2023-10',
            'type': 'CORRECTION', 'retrace': 0.618, 'size': -265
        },
        'wave_III': {
            'start': 1810, 'end': None,  # Still in progress
            'start_date': '2023-10', 'end_date': None,
            'type': 'IMPULSE', 'extended': True,
            'sub_waves': {
                'wave_1': {'start': 1810, 'end': 2450, 'size': 640},
                'wave_2': {'start': 2450, 'end': 2280, 'retrace': 0.382, 'size': -170},
                'wave_3': {'start': 2280, 'end': 4200, 'extended': True, 'size': 1920},
                'wave_4': {'start': 4200, 'end': 3950, 'retrace': 0.13, 'size': -250},
                'wave_5': {'start': 3950, 'end': None, 'current': True},  # IN PROGRESS
            }
        },
        'wave_IV': {'start': None, 'end': None, 'type': 'CORRECTION', 'future': True},
        'wave_V': {'start': None, 'end': None, 'type': 'IMPULSE', 'future': True},
    }
}

# Fibonacci ratios for wave projections
FIBONACCI_RATIOS = {
    'extensions': [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.0, 4.236],
    'retracements': [0.236, 0.382, 0.5, 0.618, 0.786, 0.886],
    'common_wave_3': [1.618, 2.618, 3.0],
    'common_wave_5': [0.618, 1.0, 1.618],
}


@dataclass
class WaveCount:
    """Elliott Wave count result"""
    current_wave: str        # e.g., "Wave 5 of III"
    wave_degree: str         # SUPERCYCLE, CYCLE, PRIMARY, etc.
    wave_direction: str      # UP or DOWN
    completion_pct: float    # Estimated % of wave complete
    targets: List[Dict]      # Price targets with methods
    invalidation: float      # Level that invalidates count
    key_levels: Dict        # Support/resistance
    warnings: List[str]     # Wave completion warnings
    summary: str


class ElliottWaveAnalyzer:
    """
    Elliott Wave Analysis for Gold
    
    Counts waves, projects targets, identifies invalidation levels
    """
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.wave_structure = GOLD_WAVE_STRUCTURE
    
    def analyze(self, ohlcv: pd.DataFrame) -> WaveCount:
        """
        Perform Elliott Wave analysis
        
        Args:
            ohlcv: DataFrame with OHLCV data
            
        Returns:
            WaveCount with wave identification and projections
        """
        ohlcv.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in ohlcv.columns]
        
        current_price = ohlcv['close'].iloc[-1]
        
        # Identify current wave
        current_wave, degree, direction = self._identify_current_wave(ohlcv)
        
        # Calculate completion percentage
        completion_pct = self._estimate_wave_completion(current_price)
        
        # Calculate targets
        targets = self._calculate_targets(current_price)
        
        # Find invalidation level
        invalidation = self._find_invalidation_level(current_price)
        
        # Key levels
        key_levels = self._get_key_levels(current_price)
        
        # Warnings
        warnings = self._check_wave_warnings(ohlcv, current_price, completion_pct)
        
        # Summary
        summary = self._build_summary(current_wave, degree, current_price, 
                                      targets, completion_pct, warnings)
        
        return WaveCount(
            current_wave=current_wave,
            wave_degree=degree,
            wave_direction=direction,
            completion_pct=completion_pct,
            targets=targets,
            invalidation=invalidation,
            key_levels=key_levels,
            warnings=warnings,
            summary=summary
        )
    
    def _identify_current_wave(self, ohlcv: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Identify current wave based on price structure
        
        Gold is currently in Wave 5 of Wave III of the Supercycle
        """
        current_price = ohlcv['close'].iloc[-1]
        
        # Based on our wave structure
        wave_III = self.wave_structure['supercycle']['wave_III']
        sub_waves = wave_III['sub_waves']
        
        # Wave 5 started at $3,950
        wave_5_start = sub_waves['wave_5']['start']
        
        if current_price > wave_5_start:
            # We're in Wave 5 of III
            return ("Wave 5 of III", "SUPERCYCLE", "UP")
        
        # Fallback
        return ("Wave III", "SUPERCYCLE", "UP")
    
    def _estimate_wave_completion(self, current_price: float) -> float:
        """
        Estimate how complete the current wave is
        
        Wave 5 targets: $5,136 - $5,500
        Wave 5 start: $3,950
        """
        wave_5_start = 3950
        wave_5_target_min = 5136  # 0.618 extension
        wave_5_target_max = 5500  # Full extension
        
        if current_price <= wave_5_start:
            return 0.0
        
        # Progress toward minimum target
        progress = (current_price - wave_5_start) / (wave_5_target_min - wave_5_start)
        
        return min(100, max(0, progress * 100))
    
    def _calculate_targets(self, current_price: float) -> List[Dict]:
        """
        Calculate Elliott Wave price targets using Fibonacci
        
        Wave 5 targets based on:
        1. Wave 5 = Wave 1 (common)
        2. Wave 5 = 0.618 x Wave 3 (common)
        3. Wave 5 = 2.618 x Wave 1 (extended)
        """
        targets = []
        
        # Wave measurements
        wave_1_size = 640   # $1,810 → $2,450
        wave_3_size = 1920  # $2,280 → $4,200
        wave_4_low = 3950   # Wave 5 starting point
        
        # Target 1: Wave 5 = Wave 1
        target_1 = wave_4_low + wave_1_size
        targets.append({
            'price': target_1,
            'method': 'Wave 5 = Wave 1',
            'probability': 70,
            'reached': current_price >= target_1
        })
        
        # Target 2: Wave 5 = 0.618 x Wave 3
        target_2 = wave_4_low + (wave_3_size * 0.618)
        targets.append({
            'price': round(target_2, 0),
            'method': 'Wave 5 = 0.618 × Wave 3',
            'probability': 75,
            'reached': current_price >= target_2
        })
        
        # Target 3: Wave 5 = 1.0 x Wave 3
        target_3 = wave_4_low + wave_3_size
        targets.append({
            'price': round(target_3, 0),
            'method': 'Wave 5 = Wave 3',
            'probability': 50,
            'reached': current_price >= target_3
        })
        
        # Target 4: Psychological $5,000
        targets.append({
            'price': 5000,
            'method': 'Psychological Level',
            'probability': 95,
            'reached': current_price >= 5000
        })
        
        # Target 5: Wave 5 = 2.618 x Wave 1 (extended)
        target_5 = wave_4_low + (wave_1_size * 2.618)
        targets.append({
            'price': round(target_5, 0),
            'method': 'Wave 5 = 2.618 × Wave 1 (Extended)',
            'probability': 40,
            'reached': current_price >= target_5
        })
        
        # Sort by price
        targets = sorted(targets, key=lambda x: x['price'])
        
        return targets
    
    def _find_invalidation_level(self, current_price: float) -> float:
        """
        Find the level that would invalidate the current wave count
        
        Rule: Wave 4 cannot overlap Wave 1 territory
        Wave 1 top: $2,450
        Wave 4 low: $3,950
        
        If price drops below Wave 4 low significantly, count may be invalid
        """
        wave_4_low = 3950
        wave_1_top = 2450
        
        # Primary invalidation: Below Wave 4 low with momentum
        return wave_4_low 
    
    def _get_key_levels(self, current_price: float) -> Dict:
        """Get key Elliott Wave levels"""
        return {
            'support': [
                {'level': 4850, 'type': 'SHORT_TERM', 'desc': 'Recent pullback support'},
                {'level': 4700, 'type': 'STRUCTURE', 'desc': 'Major structure support'},
                {'level': 4500, 'type': 'FIBONACCI', 'desc': '38.2% retrace of Wave 5'},
                {'level': 4200, 'type': 'WAVE_3_TOP', 'desc': 'Top of Wave 3'},
                {'level': 3950, 'type': 'WAVE_4_LOW', 'desc': 'Wave 5 start - CRITICAL'},
            ],
            'resistance': [
                {'level': 5000, 'type': 'PSYCHOLOGICAL', 'desc': 'Major psychological'},
                {'level': 5136, 'type': 'ELLIOTT', 'desc': '0.618 × Wave 3'},
                {'level': 5154, 'type': 'ELLIOTT', 'desc': '2.618 × Wave 1'},
                {'level': 5250, 'type': 'FIBONACCI', 'desc': 'Fib cluster zone'},
                {'level': 5500, 'type': 'WAVE_III_END', 'desc': 'Potential Wave III completion'},
                {'level': 5870, 'type': 'EXTENDED', 'desc': 'Wave 5 = Wave 3'},
            ]
        }
    
    def _check_wave_warnings(self, ohlcv: pd.DataFrame, price: float, 
                             completion: float) -> List[str]:
        """Check for wave completion warnings"""
        warnings = []
        
        close = ohlcv['close']
        
        # Wave 5 completion warning
        if completion > 80:
            warnings.append(f"⚠️ Wave 5 estimated {completion:.0f}% complete - watch for reversal")
        
        # RSI divergence check (common at Wave 5 tops)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) >= 20:
            rsi_now = rsi.iloc[-1]
            if rsi_now > 70:
                warnings.append(f"⚠️ RSI at {rsi_now:.0f} - overbought, Wave 5 exhaustion possible")
            
            # Check for divergence
            price_hh = close.iloc[-1] > close.iloc[-20:].max() * 0.99
            rsi_lh = rsi.iloc[-1] < rsi.iloc[-20:].max()
            if price_hh and rsi_lh:
                warnings.append("⚠️ RSI DIVERGENCE - Classic Wave 5 completion signal!")
        
        # Near major target
        if price >= 5000:
            warnings.append("📍 Above $5,000 - Major psychological resistance")
        
        # Extended Wave 5
        wave_5_size = price - 3950
        wave_3_size = 1920
        if wave_5_size > wave_3_size * 0.618:
            warnings.append(f"📈 Wave 5 extended ({wave_5_size/wave_3_size*100:.0f}% of Wave 3)")
        
        return warnings
    
    def _build_summary(self, wave: str, degree: str, price: float,
                       targets: List[Dict], completion: float, 
                       warnings: List[str]) -> str:
        """Build human-readable summary"""
        lines = []
        
        lines.append("═══ ELLIOTT WAVE ANALYSIS ═══")
        lines.append(f"   🌊 Current: {wave}")
        lines.append(f"   📊 Degree: {degree}")
        lines.append(f"   💰 Price: ${price:.2f}")
        lines.append(f"   📈 Completion: ~{completion:.0f}%")
        
        # Next target
        unreached = [t for t in targets if not t['reached']]
        if unreached:
            next_target = unreached[0]
            lines.append(f"\n   🎯 Next Target: ${next_target['price']:.0f}")
            lines.append(f"      Method: {next_target['method']}")
            lines.append(f"      Probability: {next_target['probability']}%")
        
        if warnings:
            lines.append("\n   ⚠️ Warnings:")
            for warn in warnings[:3]:
                lines.append(f"      {warn}")
        
        return "\n".join(lines)
    
    def get_wave_projections(self) -> Dict:
        """Get all wave projections for reference"""
        return {
            'current_wave': 'Wave 5 of III',
            'wave_III_targets': {
                'minimum': 5136,   # 0.618 × Wave 3
                'normal': 5250,    # Fib cluster
                'maximum': 5500,   # Wave III completion
                'extended': 5870,  # Wave 5 = Wave 3
            },
            'wave_IV_projection': {
                'shallow': 5000,   # 23.6% retrace
                'normal': 4700,    # 38.2% retrace
                'deep': 4500,      # 50% retrace
            },
            'wave_V_projection': {
                'minimum': 5500,   # After Wave IV
                'target': 6000,    # Major target
                'extended': 6500,  # If wave extends
            }
        }


def get_elliott_analysis(ohlcv: pd.DataFrame, symbol: str = 'XAUUSD') -> Dict:
    """Quick function for NEO integration"""
    analyzer = ElliottWaveAnalyzer(symbol)
    result = analyzer.analyze(ohlcv)
    projections = analyzer.get_wave_projections()
    
    return {
        'available': True,
        'current_wave': result.current_wave,
        'wave_degree': result.wave_degree,
        'wave_direction': result.wave_direction,
        'completion_pct': result.completion_pct,
        'targets': result.targets,
        'invalidation': result.invalidation,
        'key_levels': result.key_levels,
        'projections': projections,
        'warnings': result.warnings,
        'summary': result.summary
    }


# Test
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("ELLIOTT WAVE ANALYSIS TEST - GOLD")
    print("="*60)
    
    # Get Gold data
    gold = yf.download('GC=F', period='1y', interval='1d', progress=False)
    
    if hasattr(gold.columns, 'levels'):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]
    
    analyzer = ElliottWaveAnalyzer('XAUUSD')
    result = analyzer.analyze(gold)
    
    print(f"\n{result.summary}")
    
    print(f"\n🚫 Invalidation: ${result.invalidation:.2f}")
    
    print("\n🎯 All Targets:")
    for t in result.targets:
        status = "✅" if t['reached'] else "⭕"
        print(f"   {status} ${t['price']:.0f} - {t['method']} ({t['probability']}%)")
    
    print("\n📊 Key Support Levels:")
    for level in result.key_levels['support']:
        print(f"   ${level['level']:.0f} - {level['desc']}")
    
    print("\n📊 Key Resistance Levels:")
    for level in result.key_levels['resistance']:
        print(f"   ${level['level']:.0f} - {level['desc']}")

"""
NEO Wyckoff Analysis - Phase Detection for Gold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Wyckoff Method identifies 4 market phases:
1. ACCUMULATION - Smart money buying quietly
2. MARKUP - Price rising, trend established
3. DISTRIBUTION - Smart money selling to retail
4. MARKDOWN - Price falling, trend down

Gold Journey $1,615 → $5,000+:
- Accumulation: 2022-2023 ($1,615-$2,100)
- Markup: 2024-2026 ($2,100 → $5,000+) ← CURRENT!
- Distribution: Future (watch for signs)
- Markdown: After distribution

"Follow the smart money, not the crowd."

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
# GOLD-SPECIFIC WYCKOFF LEVELS (Historical)
# ============================================================================

GOLD_WYCKOFF_HISTORY = {
    'accumulation_2022_2023': {
        'phase': 'ACCUMULATION',
        'start_date': '2022-09-01',
        'end_date': '2023-10-01',
        'ps': 1800,      # Preliminary Support
        'sc': 1615,      # Selling Climax (THE LOW)
        'ar': 1800,      # Automatic Rally
        'st': 1620,      # Secondary Test
        'spring': 1615,  # Spring (false breakdown)
        'sos': 1850,     # Sign of Strength (break above)
        'lps': 1810,     # Last Point of Support
        'creek': 2100,   # Resistance to break
    },
    'markup_2024_2026': {
        'phase': 'MARKUP',
        'start_date': '2023-10-01',
        'creek_jump': 2100,  # Major resistance broken
        'backup_levels': [2050, 2100, 2200, 2400, 2700, 3000, 3500, 4000, 4500],
        'current_phase': 'LATE_MARKUP',  # Approaching potential distribution
    }
}

# Current key levels for Gold (2026)
GOLD_WYCKOFF_LEVELS = {
    # Support levels (potential LPS zones)
    'lps_1': 4950,      # Recent support
    'lps_2': 4850,      # Stronger support
    'lps_3': 4700,      # Major structure
    'lps_4': 4500,      # Psychological + structure
    'lps_5': 4200,      # Previous wave top
    
    # Resistance levels (potential distribution zones)
    'res_1': 5000,      # Psychological
    'res_2': 5100,      # Extension target
    'res_3': 5250,      # Fibonacci cluster
    'res_4': 5500,      # Major target
    'res_5': 6000,      # Extended target
}


@dataclass
class WyckoffPhase:
    """Wyckoff phase detection result"""
    phase: str           # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
    sub_phase: str       # Early, Mid, Late
    confidence: float    # 0-100
    volume_confirms: bool
    key_level: float     # Current key level being tested
    next_level: float    # Next target/support
    events: List[str]    # Detected Wyckoff events
    warnings: List[str]  # Distribution/exhaustion warnings
    summary: str


class WyckoffAnalyzer:
    """
    Wyckoff Phase Detection for Gold
    
    Identifies:
    - Current market phase (accumulation/markup/distribution/markdown)
    - Key Wyckoff events (SC, AR, ST, Spring, SOS, etc.)
    - Volume confirmation
    - Phase transition warnings
    """
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.levels = GOLD_WYCKOFF_LEVELS
    
    def analyze(self, ohlcv: pd.DataFrame) -> WyckoffPhase:
        """
        Perform complete Wyckoff analysis
        
        Args:
            ohlcv: DataFrame with OHLCV data (daily preferred)
            
        Returns:
            WyckoffPhase with phase detection and analysis
        """
        # Ensure lowercase columns
        ohlcv.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in ohlcv.columns]
        
        current_price = ohlcv['close'].iloc[-1]
        
        # Detect current phase
        phase, sub_phase = self._detect_phase(ohlcv)
        
        # Detect Wyckoff events
        events = self._detect_events(ohlcv)
        
        # Check volume confirmation
        volume_confirms = self._check_volume_confirmation(ohlcv, phase)
        
        # Find key levels
        key_level = self._find_key_level(current_price)
        next_level = self._find_next_level(current_price, phase)
        
        # Check for distribution warnings
        warnings = self._check_distribution_warnings(ohlcv)
        
        # Calculate confidence
        confidence = self._calculate_phase_confidence(ohlcv, phase, volume_confirms, events)
        
        # Build summary
        summary = self._build_summary(phase, sub_phase, current_price, confidence, events, warnings)
        
        return WyckoffPhase(
            phase=phase,
            sub_phase=sub_phase,
            confidence=confidence,
            volume_confirms=volume_confirms,
            key_level=key_level,
            next_level=next_level,
            events=events,
            warnings=warnings,
            summary=summary
        )
    
    def _detect_phase(self, ohlcv: pd.DataFrame) -> Tuple[str, str]:
        """
        Detect current Wyckoff phase
        
        Uses:
        - Price position relative to moving averages
        - Trend direction and strength
        - Volume patterns
        """
        close = ohlcv['close']
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([1]*len(ohlcv))
        
        current_price = close.iloc[-1]
        
        # Moving averages
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma_50
        
        # Price trend (last 50 bars)
        price_change_50 = (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50] * 100 if len(close) >= 50 else 0
        
        # Volume trend
        vol_ma_20 = volume.rolling(20).mean().iloc[-1]
        vol_ma_50 = volume.rolling(50).mean().iloc[-1]
        vol_trend = 'RISING' if vol_ma_20 > vol_ma_50 else 'FALLING'
        
        # Higher highs / Higher lows (markup sign)
        recent_highs = ohlcv['high'].tail(20)
        recent_lows = ohlcv['low'].tail(20)
        
        # Check trend structure
        hh = recent_highs.iloc[-1] > recent_highs.iloc[0]  # Higher high
        hl = recent_lows.iloc[-1] > recent_lows.iloc[0]   # Higher low
        lh = recent_highs.iloc[-1] < recent_highs.iloc[0]  # Lower high
        ll = recent_lows.iloc[-1] < recent_lows.iloc[0]   # Lower low
        
        # Phase detection logic
        if current_price > ma_50 > ma_200:
            if hh and hl:
                if price_change_50 > 5:
                    return ('MARKUP', 'MID')
                elif price_change_50 > 10:
                    return ('MARKUP', 'LATE')
                else:
                    return ('MARKUP', 'EARLY')
            elif lh or not hh:
                return ('DISTRIBUTION', 'EARLY')
        
        elif current_price < ma_50 < ma_200:
            if ll and lh:
                if price_change_50 < -5:
                    return ('MARKDOWN', 'MID')
                elif price_change_50 < -10:
                    return ('MARKDOWN', 'LATE')
                else:
                    return ('MARKDOWN', 'EARLY')
            elif hl or not ll:
                return ('ACCUMULATION', 'EARLY')
        
        elif current_price > ma_200 and current_price > ma_50:
            # Strong uptrend
            if price_change_50 > 15:
                return ('MARKUP', 'LATE')  # Extended move
            return ('MARKUP', 'MID')
        
        elif current_price < ma_200 and current_price < ma_50:
            # Strong downtrend
            return ('MARKDOWN', 'MID')
        
        else:
            # Transitional / Range
            if vol_trend == 'FALLING':
                return ('ACCUMULATION', 'MID')
            else:
                return ('DISTRIBUTION', 'MID')
        
        return ('UNKNOWN', 'UNKNOWN')
    
    def _detect_events(self, ohlcv: pd.DataFrame) -> List[str]:
        """Detect specific Wyckoff events in recent price action"""
        events = []
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([1]*len(ohlcv))
        
        recent = ohlcv.tail(20)
        
        # Volume analysis
        vol_avg = volume.rolling(20).mean()
        vol_high = volume > vol_avg * 2
        
        # Sign of Strength (SOS): Break above resistance with volume
        recent_high = high.tail(50).max()
        if close.iloc[-1] > recent_high * 0.99 and volume.iloc[-1] > vol_avg.iloc[-1] * 1.5:
            events.append("SOS: Sign of Strength (breakout with volume)")
        
        # Last Point of Support (LPS): Higher low on lower volume
        if len(ohlcv) >= 10:
            prev_low = low.iloc[-10:-5].min()
            recent_low = low.tail(5).min()
            if recent_low > prev_low * 1.01:  # Higher low
                vol_at_low = volume.tail(5).mean()
                if vol_at_low < vol_avg.iloc[-1] * 0.8:  # Lower volume
                    events.append("LPS: Last Point of Support (higher low, lower volume)")
        
        # Backup to Edge of Creek (BU): Pullback to breakout level
        if len(ohlcv) >= 20:
            breakout_level = ohlcv['high'].iloc[-20:-10].max()
            if abs(close.iloc[-1] - breakout_level) / breakout_level < 0.02:
                events.append("BU: Back-Up to Creek (testing breakout)")
        
        # Jump Across Creek (JAC): Strong move above resistance
        if close.iloc[-1] > close.iloc[-5] * 1.03:  # 3%+ in 5 days
            if volume.iloc[-1] > vol_avg.iloc[-1] * 1.5:
                events.append("JAC: Jump Across Creek (strong breakout)")
        
        # Preliminary Supply (PSY): First sign of selling
        if len(recent) >= 10:
            if high.iloc[-1] < high.iloc[-5:-1].max():  # Lower high
                if volume.iloc[-1] > vol_avg.iloc[-1] * 1.3:  # Higher volume
                    events.append("PSY: Preliminary Supply (distribution warning)")
        
        # Upthrust After Distribution (UTAD): False breakout
        recent_max = high.tail(10).max()
        prev_max = high.iloc[-20:-10].max() if len(ohlcv) >= 20 else recent_max
        if high.iloc[-3:].max() > prev_max and close.iloc[-1] < prev_max:
            events.append("⚠️ UTAD: Upthrust After Distribution (false breakout)")
        
        return events
    
    def _check_volume_confirmation(self, ohlcv: pd.DataFrame, phase: str) -> bool:
        """
        Check if volume confirms the current phase
        
        Markup: Volume rises on rallies, declines on pullbacks
        Distribution: Volume rises at tops, climactic spikes
        """
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([1]*len(ohlcv))
        close = ohlcv['close']
        
        # Calculate up/down day volumes
        up_days = close.diff() > 0
        down_days = close.diff() < 0
        
        recent_up_vol = volume[up_days].tail(10).mean()
        recent_down_vol = volume[down_days].tail(10).mean()
        
        if phase == 'MARKUP':
            # Volume should be higher on up days
            return recent_up_vol > recent_down_vol * 1.2
        
        elif phase == 'DISTRIBUTION':
            # Volume should be higher overall, especially at highs
            return recent_up_vol < recent_down_vol * 1.2
        
        elif phase == 'ACCUMULATION':
            # Volume should be declining overall
            vol_ma_10 = volume.tail(10).mean()
            vol_ma_30 = volume.tail(30).mean()
            return vol_ma_10 < vol_ma_30
        
        elif phase == 'MARKDOWN':
            # Volume should be high on down days
            return recent_down_vol > recent_up_vol * 1.2
        
        return True
    
    def _find_key_level(self, price: float) -> float:
        """Find the nearest key Wyckoff level"""
        all_levels = list(self.levels.values())
        nearest = min(all_levels, key=lambda x: abs(x - price))
        return nearest
    
    def _find_next_level(self, price: float, phase: str) -> float:
        """Find the next target/support based on phase"""
        all_levels = sorted(self.levels.values())
        
        if phase == 'MARKUP':
            # Next resistance above
            above = [l for l in all_levels if l > price]
            return above[0] if above else all_levels[-1]
        
        elif phase in ['MARKDOWN', 'DISTRIBUTION']:
            # Next support below
            below = [l for l in all_levels if l < price]
            return below[-1] if below else all_levels[0]
        
        else:
            # Nearest level
            return self._find_key_level(price)
    
    def _check_distribution_warnings(self, ohlcv: pd.DataFrame) -> List[str]:
        """Check for signs of distribution (phase transition warning)"""
        warnings = []
        
        close = ohlcv['close']
        high = ohlcv['high']
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([1]*len(ohlcv))
        
        # 1. Volume exhaustion (volume declining at new highs)
        if len(ohlcv) >= 20:
            is_near_high = close.iloc[-1] > high.tail(50).max() * 0.98
            vol_declining = volume.tail(5).mean() < volume.tail(20).mean() * 0.8
            if is_near_high and vol_declining:
                warnings.append("⚠️ Volume exhaustion at highs")
        
        # 2. RSI divergence
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) >= 20:
            price_hh = close.iloc[-1] > close.iloc[-10]  # Higher price
            rsi_lh = rsi.iloc[-1] < rsi.iloc[-10]       # Lower RSI
            if price_hh and rsi_lh:
                warnings.append("⚠️ Bearish RSI divergence (price up, RSI down)")
        
        # 3. Long upper wicks (selling pressure)
        recent = ohlcv.tail(10)
        wick_ratio = (recent['high'] - recent['close']) / (recent['high'] - recent['low'] + 0.001)
        if wick_ratio.mean() > 0.5:
            warnings.append("⚠️ Long upper wicks (selling pressure)")
        
        # 4. Price extended from MA
        ma_50 = close.rolling(50).mean().iloc[-1]
        extension = (close.iloc[-1] - ma_50) / ma_50 * 100
        if extension > 10:
            warnings.append(f"⚠️ Price extended {extension:.1f}% above 50 MA")
        
        # 5. Near major psychological level
        current = close.iloc[-1]
        psych_levels = [5000, 5500, 6000]
        for level in psych_levels:
            if abs(current - level) / level < 0.02:  # Within 2%
                warnings.append(f"⚠️ Near psychological ${level} - watch for resistance")
        
        return warnings
    
    def _calculate_phase_confidence(self, ohlcv: pd.DataFrame, phase: str, 
                                    volume_confirms: bool, events: List[str]) -> float:
        """Calculate confidence in phase detection"""
        confidence = 50.0  # Base confidence
        
        # Volume confirmation adds confidence
        if volume_confirms:
            confidence += 20
        
        # Events add confidence
        positive_events = ['SOS', 'LPS', 'JAC', 'BU']
        negative_events = ['PSY', 'UTAD']
        
        for event in events:
            if any(pe in event for pe in positive_events):
                confidence += 10
            if any(ne in event for ne in negative_events):
                confidence -= 10
        
        # Trend alignment
        close = ohlcv['close']
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma_50
        
        if phase == 'MARKUP':
            if close.iloc[-1] > ma_50 > ma_200:
                confidence += 15
        elif phase == 'MARKDOWN':
            if close.iloc[-1] < ma_50 < ma_200:
                confidence += 15
        
        return max(0, min(100, confidence))
    
    def _build_summary(self, phase: str, sub_phase: str, price: float,
                       confidence: float, events: List[str], warnings: List[str]) -> str:
        """Build human-readable summary"""
        lines = []
        
        phase_emoji = {
            'ACCUMULATION': '📦',
            'MARKUP': '📈',
            'DISTRIBUTION': '📤',
            'MARKDOWN': '📉'
        }
        
        lines.append("═══ WYCKOFF ANALYSIS ═══")
        lines.append(f"   {phase_emoji.get(phase, '❓')} Phase: {phase} ({sub_phase})")
        lines.append(f"   📊 Confidence: {confidence:.0f}%")
        lines.append(f"   💰 Current: ${price:.2f}")
        
        if events:
            lines.append("\n   📌 Events Detected:")
            for event in events[:3]:
                lines.append(f"      • {event}")
        
        if warnings:
            lines.append("\n   ⚠️ Warnings:")
            for warn in warnings[:3]:
                lines.append(f"      {warn}")
        
        return "\n".join(lines)
    
    def get_support_resistance(self) -> Dict:
        """Get all key Wyckoff levels"""
        return {
            'support': [
                {'level': 4950, 'type': 'LPS_1', 'strength': 'MEDIUM'},
                {'level': 4850, 'type': 'LPS_2', 'strength': 'STRONG'},
                {'level': 4700, 'type': 'STRUCTURE', 'strength': 'VERY_STRONG'},
                {'level': 4500, 'type': 'PSYCHOLOGICAL', 'strength': 'MAJOR'},
                {'level': 4200, 'type': 'WAVE_TOP', 'strength': 'CRITICAL'},
            ],
            'resistance': [
                {'level': 5000, 'type': 'PSYCHOLOGICAL', 'strength': 'MAJOR'},
                {'level': 5100, 'type': 'TARGET_1', 'strength': 'MEDIUM'},
                {'level': 5250, 'type': 'FIB_CLUSTER', 'strength': 'STRONG'},
                {'level': 5500, 'type': 'TARGET_2', 'strength': 'STRONG'},
                {'level': 6000, 'type': 'EXTENDED', 'strength': 'MAJOR'},
            ]
        }


def get_wyckoff_analysis(ohlcv: pd.DataFrame, symbol: str = 'XAUUSD') -> Dict:
    """Quick function for NEO integration"""
    analyzer = WyckoffAnalyzer(symbol)
    result = analyzer.analyze(ohlcv)
    levels = analyzer.get_support_resistance()
    
    return {
        'available': True,
        'phase': result.phase,
        'sub_phase': result.sub_phase,
        'confidence': result.confidence,
        'volume_confirms': result.volume_confirms,
        'key_level': result.key_level,
        'next_level': result.next_level,
        'events': result.events,
        'warnings': result.warnings,
        'support_levels': levels['support'],
        'resistance_levels': levels['resistance'],
        'summary': result.summary
    }


# Test
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("WYCKOFF ANALYSIS TEST - GOLD")
    print("="*60)
    
    # Get Gold data
    gold = yf.download('GC=F', period='1y', interval='1d', progress=False)
    
    if hasattr(gold.columns, 'levels'):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]
    
    analyzer = WyckoffAnalyzer('XAUUSD')
    result = analyzer.analyze(gold)
    
    print(f"\n{result.summary}")
    
    print(f"\n📍 Key Level: ${result.key_level:.2f}")
    print(f"📍 Next Target: ${result.next_level:.2f}")
    
    if result.events:
        print("\n📌 Wyckoff Events:")
        for event in result.events:
            print(f"   • {event}")
    
    if result.warnings:
        print("\n⚠️ Distribution Warnings:")
        for warn in result.warnings:
            print(f"   {warn}")
    
    levels = analyzer.get_support_resistance()
    print("\n📊 Key Levels:")
    print("   Support:", [l['level'] for l in levels['support']])
    print("   Resistance:", [l['level'] for l in levels['resistance']])

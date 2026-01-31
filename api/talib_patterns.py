"""
TA-Lib Pattern Detector - Professional candlestick pattern detection.

Uses TA-Lib's 61 candlestick pattern recognition functions for
institutional-grade pattern detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger("TALibPatterns")

# Try to import talib, provide fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info(f"TA-Lib loaded successfully")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available - using fallback detection")


class TALibPatternDetector:
    """
    Professional candlestick pattern detection using TA-Lib.
    61 patterns, industry standard, battle-tested.
    """
    
    # All TA-Lib candlestick pattern functions
    ALL_PATTERNS = [
        # CRITICAL PATTERNS (Immediate Action Required)
        ('CDL3BLACKCROWS', 'Three Black Crows', 'BEARISH', 'CRITICAL'),
        ('CDL3WHITESOLDIERS', 'Three White Soldiers', 'BULLISH', 'CRITICAL'),
        ('CDLIDENTICAL3CROWS', 'Identical Three Crows', 'BEARISH', 'CRITICAL'),
        
        # HIGH PRIORITY - Reversal Signals
        ('CDLENGULFING', 'Engulfing', 'BOTH', 'HIGH'),
        ('CDLPIERCING', 'Piercing Line', 'BULLISH', 'HIGH'),
        ('CDLDARKCLOUDCOVER', 'Dark Cloud Cover', 'BEARISH', 'HIGH'),
        ('CDLMORNINGSTAR', 'Morning Star', 'BULLISH', 'HIGH'),
        ('CDLEVENINGSTAR', 'Evening Star', 'BEARISH', 'HIGH'),
        ('CDLMORNINGDOJISTAR', 'Morning Doji Star', 'BULLISH', 'HIGH'),
        ('CDLEVENINGDOJISTAR', 'Evening Doji Star', 'BEARISH', 'HIGH'),
        ('CDLHAMMER', 'Hammer', 'BULLISH', 'HIGH'),
        ('CDLHANGINGMAN', 'Hanging Man', 'BEARISH', 'HIGH'),
        ('CDLSHOOTINGSTAR', 'Shooting Star', 'BEARISH', 'HIGH'),
        ('CDLABANDONEDBABY', 'Abandoned Baby', 'BOTH', 'HIGH'),
        ('CDLBREAKAWAY', 'Breakaway', 'BOTH', 'HIGH'),
        ('CDLCONCEALBABYSWALL', 'Concealing Baby Swallow', 'BULLISH', 'HIGH'),
        ('CDLKICKING', 'Kicking', 'BOTH', 'HIGH'),
        ('CDLKICKINGBYLENGTH', 'Kicking by Length', 'BOTH', 'HIGH'),
        ('CDLMATHOLD', 'Mat Hold', 'BULLISH', 'HIGH'),
        ('CDLTAKURI', 'Takuri', 'BULLISH', 'HIGH'),
        
        # MEDIUM PRIORITY - Confirmation Patterns
        ('CDL2CROWS', 'Two Crows', 'BEARISH', 'MEDIUM'),
        ('CDLHARAMI', 'Harami', 'BOTH', 'MEDIUM'),
        ('CDLHARAMICROSS', 'Harami Cross', 'BOTH', 'MEDIUM'),
        ('CDL3INSIDE', 'Three Inside Up/Down', 'BOTH', 'MEDIUM'),
        ('CDL3OUTSIDE', 'Three Outside Up/Down', 'BOTH', 'MEDIUM'),
        ('CDL3STARSINSOUTH', 'Three Stars In South', 'BULLISH', 'MEDIUM'),
        ('CDLDOJISTAR', 'Doji Star', 'BOTH', 'MEDIUM'),
        ('CDLDRAGONFLYDOJI', 'Dragonfly Doji', 'BULLISH', 'MEDIUM'),
        ('CDLGRAVESTONEDOJI', 'Gravestone Doji', 'BEARISH', 'MEDIUM'),
        ('CDLINVERTEDHAMMER', 'Inverted Hammer', 'BULLISH', 'MEDIUM'),
        ('CDLMARUBOZU', 'Marubozu', 'BOTH', 'MEDIUM'),
        ('CDLGAPSIDESIDEWHITE', 'Gap Side-by-Side White', 'BULLISH', 'MEDIUM'),
        ('CDLTASUKIGAP', 'Tasuki Gap', 'BOTH', 'MEDIUM'),
        ('CDLBELTHOLD', 'Belt Hold', 'BOTH', 'MEDIUM'),
        ('CDLCLOSINGMARUBOZU', 'Closing Marubozu', 'BOTH', 'MEDIUM'),
        ('CDLCOUNTERATTACK', 'Counterattack', 'BOTH', 'MEDIUM'),
        ('CDLHIKKAKE', 'Hikkake', 'BOTH', 'MEDIUM'),
        ('CDLHIKKAKEMOD', 'Modified Hikkake', 'BOTH', 'MEDIUM'),
        ('CDLHOMINGPIGEON', 'Homing Pigeon', 'BULLISH', 'MEDIUM'),
        ('CDLLADDERBOTTOM', 'Ladder Bottom', 'BULLISH', 'MEDIUM'),
        ('CDLMATCHINGLOW', 'Matching Low', 'BULLISH', 'MEDIUM'),
        ('CDLRISEFALL3METHODS', 'Rising/Falling Three Methods', 'BOTH', 'MEDIUM'),
        ('CDLSTALLEDPATTERN', 'Stalled Pattern', 'BEARISH', 'MEDIUM'),
        ('CDLSTICKSANDWICH', 'Stick Sandwich', 'BULLISH', 'MEDIUM'),
        ('CDLTRISTAR', 'Tristar', 'BOTH', 'MEDIUM'),
        ('CDLUNIQUE3RIVER', 'Unique 3 River', 'BULLISH', 'MEDIUM'),
        ('CDLUPSIDEGAP2CROWS', 'Upside Gap Two Crows', 'BEARISH', 'MEDIUM'),
        ('CDLXSIDEGAP3METHODS', 'Upside/Downside Gap Three Methods', 'BOTH', 'MEDIUM'),
        ('CDLADVANCEBLOCK', 'Advance Block', 'BEARISH', 'MEDIUM'),
        ('CDLHIGHWAVE', 'High Wave', 'NEUTRAL', 'MEDIUM'),
        
        # LOW PRIORITY - Context Patterns
        ('CDLDOJI', 'Doji', 'NEUTRAL', 'LOW'),
        ('CDLSPINNINGTOP', 'Spinning Top', 'NEUTRAL', 'LOW'),
        ('CDLLONGLEGGEDDOJI', 'Long Legged Doji', 'NEUTRAL', 'LOW'),
        ('CDLSEPARATINGLINES', 'Separating Lines', 'BOTH', 'LOW'),
        ('CDLINNECK', 'In-Neck', 'BEARISH', 'LOW'),
        ('CDLLONGLINE', 'Long Line', 'BOTH', 'LOW'),
        ('CDLONNECK', 'On-Neck', 'BEARISH', 'LOW'),
        ('CDLRICKSHAWMAN', 'Rickshaw Man', 'NEUTRAL', 'LOW'),
        ('CDLTHRUSTING', 'Thrusting', 'BEARISH', 'LOW'),
        ('CDLSHORTLINE', 'Short Line', 'BOTH', 'LOW'),
    ]
    
    # Priority lists for fast filtering
    PRIORITY_BEARISH = [
        'CDL3BLACKCROWS', 'CDLIDENTICAL3CROWS', 'CDLEVENINGSTAR',
        'CDLSHOOTINGSTAR', 'CDLHANGINGMAN', 'CDLDARKCLOUDCOVER',
        'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLGRAVESTONEDOJI',
    ]
    
    PRIORITY_BULLISH = [
        'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLHAMMER',
        'CDLPIERCING', 'CDLENGULFING', 'CDLMORNINGDOJISTAR', 
        'CDLTAKURI', 'CDLDRAGONFLYDOJI', 'CDLINVERTEDHAMMER',
    ]
    
    def __init__(self):
        self.pattern_stats = {}
        self.last_patterns = []
        self.talib_available = TALIB_AVAILABLE
    
    def detect_patterns(self, df: pd.DataFrame, priority_only: bool = False) -> List[Dict]:
        """Detect candlestick patterns on latest candle."""
        
        if not self.talib_available:
            return self._fallback_detection(df)
        
        # Ensure proper column names
        if 'Open' in df.columns:
            df.columns = df.columns.str.lower()
        
        open_prices = df['open'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64)
        low_prices = df['low'].values.astype(np.float64)
        close_prices = df['close'].values.astype(np.float64)
        
        detected = []
        
        patterns_to_check = self.ALL_PATTERNS
        if priority_only:
            priority_funcs = self.PRIORITY_BEARISH + self.PRIORITY_BULLISH
            patterns_to_check = [p for p in self.ALL_PATTERNS if p[0] in priority_funcs]
        
        for func_name, pattern_name, direction, severity in patterns_to_check:
            try:
                func = getattr(talib, func_name)
                result = func(open_prices, high_prices, low_prices, close_prices)
                
                latest_signal = result[-1]
                
                if latest_signal != 0:
                    actual_direction = direction
                    if direction == 'BOTH':
                        actual_direction = 'BULLISH' if latest_signal > 0 else 'BEARISH'
                    
                    detected.append({
                        'pattern': pattern_name,
                        'function': func_name,
                        'direction': actual_direction,
                        'severity': severity,
                        'signal': int(latest_signal),
                        'price': float(close_prices[-1]),
                        'timestamp': datetime.now().isoformat(),
                    })
            except Exception as e:
                pass  # Silent fail for unavailable patterns
        
        self.last_patterns = detected
        return detected
    
    def _fallback_detection(self, df: pd.DataFrame) -> List[Dict]:
        """Fallback pattern detection when TA-Lib is not available."""
        
        if 'Open' in df.columns:
            df.columns = df.columns.str.lower()
        
        detected = []
        
        if len(df) < 3:
            return detected
        
        # Get recent candles
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Simple pattern detection
        body = abs(curr['close'] - curr['open'])
        upper_wick = curr['high'] - max(curr['close'], curr['open'])
        lower_wick = min(curr['close'], curr['open']) - curr['low']
        
        # Calculate average body size
        bodies = abs(df['close'] - df['open'])
        avg_body = bodies.tail(20).mean() if len(bodies) >= 20 else bodies.mean()
        
        # Shooting Star
        if body > 0 and upper_wick > body * 2 and lower_wick < body * 0.5:
            if curr['high'] > prev['high']:
                detected.append({
                    'pattern': 'Shooting Star',
                    'function': 'CDLSHOOTINGSTAR',
                    'direction': 'BEARISH',
                    'severity': 'HIGH',
                    'signal': -100,
                    'price': float(curr['close']),
                    'timestamp': datetime.now().isoformat(),
                })
        
        # Hammer
        if body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5:
            if curr['low'] < prev['low']:
                detected.append({
                    'pattern': 'Hammer',
                    'function': 'CDLHAMMER',
                    'direction': 'BULLISH',
                    'severity': 'HIGH',
                    'signal': 100,
                    'price': float(curr['close']),
                    'timestamp': datetime.now().isoformat(),
                })
        
        # Engulfing
        curr_body = curr['close'] - curr['open']
        prev_body = prev['close'] - prev['open']
        
        if curr_body > 0 and prev_body < 0 and curr_body > abs(prev_body):
            detected.append({
                'pattern': 'Engulfing',
                'function': 'CDLENGULFING',
                'direction': 'BULLISH',
                'severity': 'HIGH',
                'signal': 100,
                'price': float(curr['close']),
                'timestamp': datetime.now().isoformat(),
            })
        elif curr_body < 0 and prev_body > 0 and abs(curr_body) > prev_body:
            detected.append({
                'pattern': 'Engulfing',
                'function': 'CDLENGULFING',
                'direction': 'BEARISH',
                'severity': 'HIGH',
                'signal': -100,
                'price': float(curr['close']),
                'timestamp': datetime.now().isoformat(),
            })
        
        # Doji
        if pd.notna(avg_body) and body < avg_body * 0.1:
            detected.append({
                'pattern': 'Doji',
                'function': 'CDLDOJI',
                'direction': 'NEUTRAL',
                'severity': 'LOW',
                'signal': 0,
                'price': float(curr['close']),
                'timestamp': datetime.now().isoformat(),
            })
        
        self.last_patterns = detected
        return detected
    
    def detect_all_timeframe(
        self, 
        df: pd.DataFrame, 
        lookback: int = 10
    ) -> List[Dict]:
        """Detect patterns across the last N candles."""
        
        all_patterns = []
        
        for i in range(lookback):
            if i >= len(df) - 3:
                break
            
            subset = df.iloc[:len(df)-i]
            patterns = self.detect_patterns(subset, priority_only=True)
            
            for p in patterns:
                p['candles_ago'] = i
                all_patterns.append(p)
        
        return all_patterns
    
    def get_defcon_impact(self, patterns: List[Dict]) -> int:
        """Calculate DEFCON level adjustment from patterns."""
        
        if not patterns:
            return 0
        
        impact = 0
        for p in patterns:
            severity = p.get('severity', 'LOW')
            direction = p.get('direction', 'NEUTRAL')
            
            if direction == 'BEARISH':
                if severity == 'CRITICAL': impact -= 2
                elif severity == 'HIGH': impact -= 1
                elif severity == 'MEDIUM': impact -= 0.5
            elif direction == 'BULLISH':
                if severity == 'CRITICAL': impact += 2
                elif severity == 'HIGH': impact += 1
                elif severity == 'MEDIUM': impact += 0.5
        
        return int(round(impact))
    
    def get_ghost_action(self, patterns: List[Dict]) -> Dict:
        """Get recommended Ghost action based on detected patterns."""
        
        if not patterns:
            return {
                'action': 'NONE', 
                'message': 'No patterns detected',
                'close_percent': 0,
                'tighten_stops': False,
                'new_entries': True,
                'lot_multiplier': 1.0,
                'urgency': 'NONE',
            }
        
        bearish = [p for p in patterns if p['direction'] == 'BEARISH']
        bullish = [p for p in patterns if p['direction'] == 'BULLISH']
        
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        if bearish:
            worst = max(bearish, key=lambda x: severity_order.get(x['severity'], 0))
            
            if worst['severity'] == 'CRITICAL':
                return {
                    'action': 'CLOSE_PARTIAL',
                    'close_percent': 75,
                    'tighten_stops': True,
                    'new_entries': False,
                    'lot_multiplier': 0.25,
                    'message': f"CRITICAL: {worst['pattern']} - Close 75%, halt entries",
                    'trigger_pattern': worst['pattern'],
                    'urgency': 'IMMEDIATE',
                }
            elif worst['severity'] == 'HIGH':
                return {
                    'action': 'REDUCE_AND_PROTECT',
                    'close_percent': 50,
                    'tighten_stops': True,
                    'new_entries': False,
                    'lot_multiplier': 0.50,
                    'message': f"HIGH: {worst['pattern']} - Close 50%, tighten stops",
                    'trigger_pattern': worst['pattern'],
                    'urgency': 'HIGH',
                }
            else:
                return {
                    'action': 'CAUTION',
                    'close_percent': 0,
                    'tighten_stops': True,
                    'new_entries': True,
                    'lot_multiplier': 0.75,
                    'message': f"CAUTION: {worst['pattern']} - Tighten stops",
                    'trigger_pattern': worst['pattern'],
                    'urgency': 'MEDIUM',
                }
        
        if bullish:
            best = max(bullish, key=lambda x: severity_order.get(x['severity'], 0))
            return {
                'action': 'OPPORTUNITY',
                'close_percent': 0,
                'tighten_stops': False,
                'new_entries': True,
                'lot_multiplier': 1.0,
                'message': f"BULLISH: {best['pattern']} - Entry opportunity",
                'trigger_pattern': best['pattern'],
                'urgency': 'LOW',
            }
        
        return {
            'action': 'NONE', 
            'message': 'No actionable patterns',
            'close_percent': 0,
            'tighten_stops': False,
            'new_entries': True,
            'lot_multiplier': 1.0,
            'urgency': 'NONE',
        }
    
    def get_pattern_summary(self, patterns: List[Dict]) -> str:
        """Generate human-readable pattern summary."""
        
        if not patterns:
            return "No patterns detected"
        
        critical = [p for p in patterns if p['severity'] == 'CRITICAL']
        high = [p for p in patterns if p['severity'] == 'HIGH']
        medium = [p for p in patterns if p['severity'] == 'MEDIUM']
        
        summary = []
        
        if critical:
            summary.append(f"🚨 CRITICAL: {', '.join(p['pattern'] for p in critical)}")
        if high:
            summary.append(f"⚠️ HIGH: {', '.join(p['pattern'] for p in high)}")
        if medium:
            summary.append(f"📊 MEDIUM: {', '.join(p['pattern'] for p in medium)}")
        
        return "\n".join(summary) if summary else "Only low-priority patterns detected"

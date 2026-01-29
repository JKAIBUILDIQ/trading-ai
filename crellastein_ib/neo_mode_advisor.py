#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    NEO MODE ADVISOR
                    "The Tiebreaker"
═══════════════════════════════════════════════════════════════════════════════

NEO's job: Analyze technicals → Decide mode → Track calls → Learn

The Learning Loop:
1. NEO analyzes charts (SuperTrend, RSI, EMA, patterns)
2. NEO makes mode call
3. Track the call (date, price, reasoning)
4. Evaluate outcome (24h later)
5. Learn from result

"When we don't know, NEO makes the call."

Created: January 29, 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'neo'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NeoModeAdvisor')

# Import NEO components
try:
    from meta_bot.crellastein_meta import CrellaSteinMetaBot
    from gold_predictor import GoldPredictor
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False
    logger.warning("NEO components not available")


@dataclass
class TechnicalAnalysis:
    """Technical analysis snapshot"""
    timestamp: str
    price: float
    
    # SuperTrend
    supertrend_direction: str  # 'BULLISH' or 'BEARISH'
    supertrend_value: float
    
    # RSI
    rsi: float
    rsi_signal: str  # 'OVERBOUGHT', 'OVERSOLD', 'NEUTRAL'
    
    # EMA
    ema_20: float
    price_vs_ema: float  # How far above/below EMA
    overextended: bool
    
    # Pattern
    pattern_detected: str  # 'BEAR_FLAG', 'BULL_FLAG', 'CONSOLIDATION', 'NONE'
    
    # Composite
    composite_score: float  # 0-100 from CrellaStein


@dataclass
class ModeCall:
    """A mode recommendation from NEO"""
    id: str
    timestamp: str
    price_at_call: float
    
    # The call
    recommended_mode: int  # 1, 2, or 3
    mode_name: str  # BULLISH, CORRECTION, BEARISH
    confidence: float
    
    # Reasoning
    reasoning: List[str]
    technical_snapshot: Dict
    
    # Outcome tracking
    price_24h_later: Optional[float] = None
    actual_move: Optional[float] = None
    was_correct: Optional[bool] = None
    lesson_learned: Optional[str] = None


class NeoModeAdvisor:
    """
    NEO's brain for mode decisions
    
    Analyzes technicals and recommends:
    - Mode 1 (BULLISH): Maximize gains
    - Mode 2 (CORRECTION): Safeguard losses
    - Mode 3 (BEARISH): Ride trend change
    """
    
    def __init__(self):
        self.call_log_file = Path(__file__).parent / 'neo_mode_calls.json'
        self.calls: List[ModeCall] = []
        self._load_calls()
        
        # Technical thresholds
        self.thresholds = {
            # CORRECTION triggers (Mode 2)
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'overextension_pts': 100,  # Points above EMA20
            'gain_5d_pct': 8,  # 5-day gain %
            
            # BEARISH triggers (Mode 3)
            'divergence_threshold': 0.15,
            'support_break_pts': 50,
        }
        
        # Try to initialize NEO analyzers
        if NEO_AVAILABLE:
            try:
                self.crella = CrellaSteinMetaBot()
                logger.info("✅ CrellaSteinMetaBot initialized")
            except:
                self.crella = None
        else:
            self.crella = None
    
    def _load_calls(self):
        """Load historical calls"""
        if self.call_log_file.exists():
            try:
                with open(self.call_log_file, 'r') as f:
                    data = json.load(f)
                    self.calls = [ModeCall(**c) for c in data.get('calls', [])]
            except:
                self.calls = []
    
    def _save_calls(self):
        """Save calls to file"""
        with open(self.call_log_file, 'w') as f:
            json.dump({
                'calls': [asdict(c) for c in self.calls],
                'last_updated': datetime.now().isoformat(),
            }, f, indent=2)
    
    def analyze_technicals(self, price: float = None) -> TechnicalAnalysis:
        """Analyze current technical state"""
        # Get real data if possible
        if self.crella:
            try:
                signal = self.crella.calculate_composite_signal('GC=F')  # Gold futures
                composite_score = signal.composite_score * 100
                
                # Extract individual indicators
                indicators = {i['name']: i for i in signal.indicators}
                
                rsi_val = indicators.get('rsi', {}).get('value', 50)
                supertrend_signal = indicators.get('supertrend', {}).get('signal', 0)
                
            except Exception as e:
                logger.warning(f"Could not get real data: {e}")
                composite_score = 50
                rsi_val = 50
                supertrend_signal = 1
        else:
            # Defaults
            composite_score = 50
            rsi_val = 50
            supertrend_signal = 1
        
        # Determine signals
        if rsi_val > self.thresholds['rsi_overbought']:
            rsi_signal = 'OVERBOUGHT'
        elif rsi_val < self.thresholds['rsi_oversold']:
            rsi_signal = 'OVERSOLD'
        else:
            rsi_signal = 'NEUTRAL'
        
        supertrend_dir = 'BULLISH' if supertrend_signal > 0 else 'BEARISH'
        
        return TechnicalAnalysis(
            timestamp=datetime.now().isoformat(),
            price=price or 0,
            supertrend_direction=supertrend_dir,
            supertrend_value=0,
            rsi=rsi_val,
            rsi_signal=rsi_signal,
            ema_20=0,
            price_vs_ema=0,
            overextended=rsi_val > 80 or rsi_val < 20,
            pattern_detected='NONE',
            composite_score=composite_score,
        )
    
    def recommend_mode(self, price: float = None, pattern: str = None) -> ModeCall:
        """
        Analyze and recommend a trading mode
        
        Returns ModeCall with recommendation and reasoning
        """
        analysis = self.analyze_technicals(price)
        
        reasoning = []
        mode = 1  # Default to BULLISH
        confidence = 0.5
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK FOR BEARISH (Mode 3) - Trend reversal
        # ═══════════════════════════════════════════════════════════════════
        bearish_signals = 0
        
        if analysis.supertrend_direction == 'BEARISH':
            bearish_signals += 2
            reasoning.append("SuperTrend BEARISH - trend flipped")
        
        if pattern and pattern.upper() in ['BEAR_FLAG', 'BREAKDOWN', 'DIVERGENCE']:
            bearish_signals += 2
            reasoning.append(f"Pattern detected: {pattern}")
        
        if analysis.rsi_signal == 'OVERBOUGHT' and analysis.composite_score < 40:
            bearish_signals += 1
            reasoning.append("RSI overbought with weak composite - reversal risk")
        
        if bearish_signals >= 2:
            mode = 3
            confidence = min(0.9, 0.5 + bearish_signals * 0.15)
            reasoning.insert(0, "🐻 BEARISH CALL: Multiple reversal signals")
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK FOR CORRECTION (Mode 2) - Choppy/Safeguard
        # ═══════════════════════════════════════════════════════════════════
        elif analysis.supertrend_direction == 'BULLISH':
            correction_signals = 0
            
            if analysis.rsi_signal == 'OVERBOUGHT':
                correction_signals += 1
                reasoning.append(f"RSI {analysis.rsi:.0f} - overbought")
            
            if analysis.overextended:
                correction_signals += 1
                reasoning.append("Price overextended from EMA")
            
            if pattern and pattern.upper() in ['CONSOLIDATION', 'RANGE', 'CHOP']:
                correction_signals += 1
                reasoning.append(f"Pattern: {pattern} - choppy market")
            
            if analysis.composite_score > 70 or analysis.composite_score < 30:
                correction_signals += 1
                reasoning.append("Extreme composite score - caution warranted")
            
            if correction_signals >= 2:
                mode = 2
                confidence = min(0.85, 0.5 + correction_signals * 0.12)
                reasoning.insert(0, "📊 CORRECTION CALL: Safeguard mode")
            else:
                # Default to BULLISH
                mode = 1
                confidence = 0.6 + (analysis.composite_score / 200)
                reasoning.insert(0, "📈 BULLISH CALL: Trend intact, maximize gains")
        
        # Create the call
        mode_names = {1: 'BULLISH', 2: 'CORRECTION', 3: 'BEARISH'}
        
        call = ModeCall(
            id=f"NEO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            price_at_call=price or 0,
            recommended_mode=mode,
            mode_name=mode_names[mode],
            confidence=confidence,
            reasoning=reasoning,
            technical_snapshot=asdict(analysis),
        )
        
        # Log the call
        self.calls.append(call)
        self._save_calls()
        
        return call
    
    def update_call_outcome(self, call_id: str, current_price: float) -> Optional[ModeCall]:
        """Update a call with its outcome"""
        for call in self.calls:
            if call.id == call_id:
                call.price_24h_later = current_price
                call.actual_move = current_price - call.price_at_call
                
                # Determine if call was correct
                if call.recommended_mode == 1:  # BULLISH
                    call.was_correct = call.actual_move > 0
                elif call.recommended_mode == 2:  # CORRECTION
                    call.was_correct = abs(call.actual_move) < 30  # Sideways
                elif call.recommended_mode == 3:  # BEARISH
                    call.was_correct = call.actual_move < 0
                
                # Generate lesson
                if call.was_correct:
                    call.lesson_learned = f"Correct! {call.mode_name} call with {call.reasoning[0]}"
                else:
                    call.lesson_learned = f"Wrong. Recommended {call.mode_name} but market moved {'up' if call.actual_move > 0 else 'down'} ${abs(call.actual_move):.0f}"
                
                self._save_calls()
                return call
        return None
    
    def get_scorecard(self) -> Dict:
        """Get accuracy scorecard"""
        evaluated = [c for c in self.calls if c.was_correct is not None]
        
        if not evaluated:
            return {'message': 'No evaluated calls yet'}
        
        total = len(evaluated)
        correct = len([c for c in evaluated if c.was_correct])
        
        by_mode = {}
        for mode in [1, 2, 3]:
            mode_calls = [c for c in evaluated if c.recommended_mode == mode]
            if mode_calls:
                mode_correct = len([c for c in mode_calls if c.was_correct])
                by_mode[mode] = {
                    'calls': len(mode_calls),
                    'correct': mode_correct,
                    'accuracy': f"{mode_correct/len(mode_calls)*100:.0f}%"
                }
        
        return {
            'total_calls': total,
            'correct': correct,
            'overall_accuracy': f"{correct/total*100:.0f}%",
            'by_mode': by_mode,
        }
    
    def get_recommendation(self, price: float = None, pattern: str = None) -> str:
        """Get a formatted recommendation"""
        call = self.recommend_mode(price, pattern)
        
        mode_emoji = {1: '📈', 2: '📊', 3: '🐻'}
        
        output = f"""
═══════════════════════════════════════════════════════════════════════════════
                    NEO MODE RECOMMENDATION
═══════════════════════════════════════════════════════════════════════════════

  {mode_emoji[call.recommended_mode]} RECOMMENDED MODE: {call.recommended_mode} ({call.mode_name})
  
  Confidence: {call.confidence*100:.0f}%
  Price: ${call.price_at_call:.2f}
  Time: {call.timestamp}
  
  REASONING:
"""
        for r in call.reasoning:
            output += f"    • {r}\n"
        
        output += f"""
  COMMAND: python3 grid_control.py {call.recommended_mode}
  
═══════════════════════════════════════════════════════════════════════════════
  Call ID: {call.id} (for tracking)
═══════════════════════════════════════════════════════════════════════════════
"""
        return output


def main():
    """Run NEO Mode Advisor"""
    import sys
    
    advisor = NeoModeAdvisor()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'recommend':
            price = float(sys.argv[2]) if len(sys.argv) > 2 else 5570
            pattern = sys.argv[3] if len(sys.argv) > 3 else None
            print(advisor.get_recommendation(price, pattern))
            
        elif cmd == 'scorecard':
            scorecard = advisor.get_scorecard()
            print(json.dumps(scorecard, indent=2))
            
        elif cmd == 'update':
            if len(sys.argv) >= 4:
                call_id = sys.argv[2]
                price = float(sys.argv[3])
                result = advisor.update_call_outcome(call_id, price)
                if result:
                    print(f"Updated call {call_id}")
                    print(f"Was correct: {result.was_correct}")
                    print(f"Lesson: {result.lesson_learned}")
    else:
        # Default: show recommendation
        print(advisor.get_recommendation(5570))


if __name__ == "__main__":
    main()

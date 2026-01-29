#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    NEO CHART ANALYZER
                    Multi-Timeframe Analysis → Mode Recommendation
═══════════════════════════════════════════════════════════════════════════════

Implements the Quinn Chart Reading Training Guide for automated analysis.

Commands:
    python neo_chart_analyzer.py analyze          # Full chart analysis
    python neo_chart_analyzer.py mode             # Quick mode recommendation
    python neo_chart_analyzer.py levels           # Key S/R levels
    python neo_chart_analyzer.py scorecard        # Training scorecard

═══════════════════════════════════════════════════════════════════════════════
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sys

# Files
SCORECARD_FILE = Path(__file__).parent / 'neo_scorecard.json'
ANALYSIS_FILE = Path(__file__).parent / 'latest_analysis.json'


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe"""
    timeframe: str
    trend: str  # BULLISH, BEARISH, NEUTRAL
    structure: str  # HH/HL, LH/LL, RANGING
    pattern: str  # Pattern name or "None"
    key_level: float
    summary: str


@dataclass
class ChartAnalysis:
    """Complete multi-timeframe analysis"""
    timestamp: str
    symbol: str
    
    # Timeframe analyses
    weekly: Dict
    daily: Dict
    h4: Dict
    
    # Overall
    dominant_trend: str
    last_3_swings: str
    key_pattern: str
    
    # Levels
    ath: float
    resistance: List[float]
    support: List[float]
    invalidation: float
    
    # Indicators
    supertrend: str
    ema_status: str
    rsi: float
    divergence: str
    
    # Recommendation
    recommended_mode: int
    mode_name: str
    confidence: str
    reasoning: str
    alert_conditions: List[str]


class NeoChartAnalyzer:
    """
    NEO's chart reading brain
    """
    
    def __init__(self):
        self.scorecard = self._load_scorecard()
    
    def _load_scorecard(self) -> Dict:
        """Load training scorecard"""
        if SCORECARD_FILE.exists():
            with open(SCORECARD_FILE, 'r') as f:
                return json.load(f)
        return {
            'total_calls': 0,
            'correct_calls': 0,
            'calls': [],
        }
    
    def _save_scorecard(self):
        """Save scorecard"""
        with open(SCORECARD_FILE, 'w') as f:
            json.dump(self.scorecard, f, indent=2)
    
    def analyze_manual(
        self,
        # Weekly
        weekly_trend: str,
        weekly_summary: str,
        # Daily
        daily_trend: str,
        daily_pattern: str,
        daily_summary: str,
        # 4H
        h4_trend: str,
        h4_signal: str,
        h4_summary: str,
        # Levels
        ath: float,
        resistance: List[float],
        support: List[float],
        # Indicators
        supertrend: str,
        ema_status: str,
        rsi: float,
        divergence: str = 'None',
        # User sighting
        user_pattern: str = None,
    ) -> ChartAnalysis:
        """
        Manual input analysis - user provides the data, NEO makes the call
        """
        
        # Determine dominant trend (Weekly > Daily > 4H)
        if weekly_trend == 'BULLISH':
            dominant = 'UPTREND'
        elif weekly_trend == 'BEARISH':
            dominant = 'DOWNTREND'
        else:
            dominant = 'RANGE'
        
        # Apply Mode Decision Matrix
        mode, mode_name, confidence, reasoning = self._decide_mode(
            weekly_trend=weekly_trend,
            daily_trend=daily_trend,
            daily_pattern=daily_pattern,
            h4_signal=h4_signal,
            supertrend=supertrend,
            rsi=rsi,
            divergence=divergence,
            user_pattern=user_pattern,
        )
        
        # Determine invalidation level
        if mode == 1:  # Bullish
            invalidation = support[0] if support else 0
        elif mode == 3:  # Bearish
            invalidation = resistance[0] if resistance else 0
        else:  # Correction
            invalidation = ath
        
        # Alert conditions
        alerts = self._generate_alerts(mode, resistance, support, daily_pattern)
        
        analysis = ChartAnalysis(
            timestamp=datetime.now().isoformat(),
            symbol='XAUUSD/MGC',
            weekly={'trend': weekly_trend, 'summary': weekly_summary},
            daily={'trend': daily_trend, 'pattern': daily_pattern, 'summary': daily_summary},
            h4={'trend': h4_trend, 'signal': h4_signal, 'summary': h4_summary},
            dominant_trend=dominant,
            last_3_swings='See chart',
            key_pattern=user_pattern or daily_pattern or 'None',
            ath=ath,
            resistance=resistance,
            support=support,
            invalidation=invalidation,
            supertrend=supertrend,
            ema_status=ema_status,
            rsi=rsi,
            divergence=divergence,
            recommended_mode=mode,
            mode_name=mode_name,
            confidence=confidence,
            reasoning=reasoning,
            alert_conditions=alerts,
        )
        
        # Save analysis
        with open(ANALYSIS_FILE, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        
        return analysis
    
    def _decide_mode(
        self,
        weekly_trend: str,
        daily_trend: str,
        daily_pattern: str,
        h4_signal: str,
        supertrend: str,
        rsi: float,
        divergence: str,
        user_pattern: str,
    ) -> tuple:
        """
        Apply Mode Decision Matrix from training guide
        Returns: (mode, mode_name, confidence, reasoning)
        """
        
        # User pattern override (human-in-the-loop)
        if user_pattern:
            if 'BEAR' in user_pattern.upper():
                return (3, 'BEARISH', 'High', 
                       f"User identified {user_pattern}. Respecting human pattern recognition.")
            elif 'BULL' in user_pattern.upper():
                return (1, 'BULLISH', 'High',
                       f"User identified {user_pattern}. Respecting human pattern recognition.")
        
        # Check for divergence (reversal signal)
        if divergence == 'Bearish' and rsi > 70:
            return (2, 'CORRECTION', 'High',
                   "Bearish divergence with RSI overbought. Reversal likely.")
        
        if divergence == 'Bullish' and rsi < 30:
            return (1, 'BULLISH', 'Medium',
                   "Bullish divergence with RSI oversold. Reversal setup forming.")
        
        # Weekly Bullish scenarios
        if weekly_trend == 'BULLISH':
            if daily_pattern in ['Bull Flag', 'HH/HL intact']:
                if h4_signal in ['Pullback to EMA', 'Consolidating']:
                    return (1, 'BULLISH', 'High',
                           f"Weekly bullish, daily {daily_pattern}, 4H {h4_signal}. Trend continuation expected.")
            
            if daily_pattern in ['Bear Flag', 'Double Top', 'Head and Shoulders']:
                return (2, 'CORRECTION', 'High',
                       f"Weekly bullish BUT daily showing {daily_pattern}. Hedge recommended.")
            
            if supertrend == 'SELL':
                return (2, 'CORRECTION', 'Medium',
                       "Weekly bullish but SuperTrend flipped SELL. Correction mode for safety.")
        
        # Weekly Bearish scenarios
        if weekly_trend == 'BEARISH':
            if daily_pattern in ['Bear Flag', 'LH/LL intact']:
                return (3, 'BEARISH', 'High',
                       f"Weekly bearish, daily {daily_pattern}. Downtrend continuation.")
            
            if h4_signal == 'Rally to resistance':
                return (3, 'BEARISH', 'High',
                       "Weekly bearish, 4H rallying to resistance. Short opportunity.")
        
        # Ranging / Unclear
        if weekly_trend == 'NEUTRAL' or daily_trend == 'NEUTRAL':
            return (2, 'CORRECTION', 'Medium',
                   "No clear trend direction. Grid mode for range trading.")
        
        # Default: When in doubt, CORRECTION
        return (2, 'CORRECTION', 'Low',
               "Mixed signals. Defaulting to CORRECTION mode for safety.")
    
    def _generate_alerts(
        self,
        mode: int,
        resistance: List[float],
        support: List[float],
        pattern: str,
    ) -> List[str]:
        """Generate alert conditions"""
        alerts = []
        
        if mode == 1:  # Bullish
            if support:
                alerts.append(f"Switch to Mode 2 if price breaks ${support[0]}")
            alerts.append("Watch for bearish divergence on new highs")
        
        elif mode == 2:  # Correction
            if resistance:
                alerts.append(f"Switch to Mode 1 if price breaks ${resistance[0]} with volume")
            if support:
                alerts.append(f"Switch to Mode 3 if price breaks ${support[0]}")
        
        elif mode == 3:  # Bearish
            if resistance:
                alerts.append(f"Switch to Mode 1 if price breaks ${resistance[0]}")
            alerts.append("Watch for bullish divergence at support")
        
        return alerts
    
    def record_outcome(self, mode_called: int, actual_outcome: str, notes: str = ''):
        """Record outcome for training scorecard"""
        correct = (
            (mode_called == 1 and actual_outcome == 'UP') or
            (mode_called == 3 and actual_outcome == 'DOWN') or
            (mode_called == 2 and actual_outcome in ['CHOPPY', 'RANGE'])
        )
        
        self.scorecard['total_calls'] += 1
        if correct:
            self.scorecard['correct_calls'] += 1
        
        self.scorecard['calls'].append({
            'date': datetime.now().isoformat(),
            'mode_called': mode_called,
            'actual_outcome': actual_outcome,
            'correct': correct,
            'notes': notes,
        })
        
        self._save_scorecard()
        
        accuracy = self.scorecard['correct_calls'] / self.scorecard['total_calls'] * 100
        return {
            'correct': correct,
            'total_calls': self.scorecard['total_calls'],
            'accuracy': f"{accuracy:.1f}%",
            'graduated': accuracy >= 70 and self.scorecard['total_calls'] >= 20,
        }
    
    def get_scorecard(self) -> str:
        """Get training scorecard"""
        total = self.scorecard['total_calls']
        correct = self.scorecard['correct_calls']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        graduated = "✅ GRADUATED" if (accuracy >= 70 and total >= 20) else "🎓 IN TRAINING"
        
        output = f"""
═══════════════════════════════════════════════════════════════════════════════
                         NEO TRAINING SCORECARD
═══════════════════════════════════════════════════════════════════════════════

  Status: {graduated}
  
  Total Calls:   {total}
  Correct:       {correct}
  Accuracy:      {accuracy:.1f}%
  
  Graduation Criteria:
    • 70%+ accuracy: {"✅" if accuracy >= 70 else "❌"} ({accuracy:.1f}%)
    • 20+ sessions:  {"✅" if total >= 20 else "❌"} ({total}/20)

───────────────────────────────────────────────────────────────────────────────
  Recent Calls:
"""
        for call in self.scorecard['calls'][-5:]:
            emoji = "✅" if call['correct'] else "❌"
            output += f"    {call['date'][:10]} | Mode {call['mode_called']} | {call['actual_outcome']} | {emoji}\n"
        
        output += """
═══════════════════════════════════════════════════════════════════════════════
"""
        return output
    
    def format_analysis(self, analysis: ChartAnalysis) -> str:
        """Format analysis as report"""
        mode_emoji = {1: '📈', 2: '📊', 3: '🐻'}
        
        return f"""
═══════════════════════════════════════════════════════════════════════════════
                    {analysis.symbol} CHART ANALYSIS
                    {analysis.timestamp[:10]}
═══════════════════════════════════════════════════════════════════════════════

📊 MULTI-TIMEFRAME SUMMARY
───────────────────────────────────────────────────────────────────────────────
  Weekly:  {analysis.weekly['trend']:8} - {analysis.weekly['summary']}
  Daily:   {analysis.daily['trend']:8} - {analysis.daily['summary']}
  4H:      {analysis.h4['trend']:8} - {analysis.h4['summary']}

📈 STRUCTURE ANALYSIS
───────────────────────────────────────────────────────────────────────────────
  Dominant Trend: {analysis.dominant_trend}
  Key Pattern:    {analysis.key_pattern}

🎯 KEY LEVELS
───────────────────────────────────────────────────────────────────────────────
  ATH:              ${analysis.ath:,.2f}
  Resistance:       {', '.join(f'${r:,.2f}' for r in analysis.resistance)}
  Support:          {', '.join(f'${s:,.2f}' for s in analysis.support)}
  Invalidation:     ${analysis.invalidation:,.2f}

📊 INDICATOR STATUS
───────────────────────────────────────────────────────────────────────────────
  SuperTrend: {analysis.supertrend}
  EMA Status: {analysis.ema_status}
  RSI (14):   {analysis.rsi} - {"Overbought" if analysis.rsi > 70 else "Oversold" if analysis.rsi < 30 else "Neutral"}
  Divergence: {analysis.divergence}

🚦 MODE RECOMMENDATION
───────────────────────────────────────────────────────────────────────────────
  {mode_emoji.get(analysis.recommended_mode, '❓')} RECOMMENDED MODE: {analysis.recommended_mode} ({analysis.mode_name})
  
  Confidence: {analysis.confidence}
  
  REASONING:
    {analysis.reasoning}

  ALERT CONDITIONS:
"""
        for alert in analysis.alert_conditions:
            output += f"    • {alert}\n"
        
        output += """
═══════════════════════════════════════════════════════════════════════════════
"""
        return output


def main():
    """CLI interface"""
    analyzer = NeoChartAnalyzer()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python neo_chart_analyzer.py analyze   # Manual analysis input")
        print("  python neo_chart_analyzer.py scorecard # Training scorecard")
        print("  python neo_chart_analyzer.py record <mode> <outcome> # Record outcome")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'scorecard':
        print(analyzer.get_scorecard())
    
    elif cmd == 'record':
        if len(sys.argv) >= 4:
            mode = int(sys.argv[2])
            outcome = sys.argv[3].upper()
            notes = sys.argv[4] if len(sys.argv) > 4 else ''
            result = analyzer.record_outcome(mode, outcome, notes)
            print(f"Recorded: {'✅ Correct' if result['correct'] else '❌ Incorrect'}")
            print(f"Accuracy: {result['accuracy']} ({result['total_calls']} calls)")
    
    elif cmd == 'analyze':
        # Interactive analysis
        print("\n📊 NEO CHART ANALYSIS - Manual Input")
        print("─" * 50)
        
        # For now, use example values - in production, would prompt for input
        analysis = analyzer.analyze_manual(
            weekly_trend='BULLISH',
            weekly_summary='Strong uptrend, new ATH territory',
            daily_trend='BULLISH',
            daily_pattern='Bear Flag forming',
            daily_summary='Consolidating after spike, flag pattern',
            h4_trend='NEUTRAL',
            h4_signal='Consolidating',
            h4_summary='Choppy price action in range',
            ath=5620.0,
            resistance=[5610.0, 5630.0, 5650.0],
            support=[5550.0, 5500.0, 5400.0],
            supertrend='BUY',
            ema_status='20>50 Bullish',
            rsi=65,
            divergence='None',
            user_pattern='Bear Flag',
        )
        
        print(analyzer.format_analysis(analysis))


if __name__ == "__main__":
    main()

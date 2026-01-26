#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO SIGNAL LEARNER - Learn From Mistakes + Breakout Intelligence
═══════════════════════════════════════════════════════════════════════════════

KEY INSIGHT: NEO should get SMARTER over time, not repeat the same mistakes!

This module:
1. TRACKS all signals and their outcomes (win/loss/BE)
2. LEARNS patterns (what works, what doesn't)
3. ADJUSTS confidence based on historical accuracy
4. RECOGNIZES breakouts (real vs fake)
5. KNOWS when to ride vs fade

"Those who cannot remember the past are condemned to repeat it."

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalLearner")

# Database path
DB_PATH = Path(__file__).parent / "neo_learning.db"


@dataclass
class SignalOutcome:
    """Record of a signal and its outcome"""
    signal_id: str
    symbol: str
    direction: str  # BUY, SELL
    entry_price: float
    entry_time: str
    
    # Signal context
    rsi_at_signal: float = 0.0
    trend_at_signal: str = ""  # STRONG_UP, UP, RANGING, DOWN, STRONG_DOWN
    was_breakout: bool = False
    breakout_type: str = ""  # RESISTANCE_BREAK, SUPPORT_BREAK, RANGE_BREAK
    herd_direction: str = ""  # What was herd doing?
    
    # Outcome (filled later)
    exit_price: float = 0.0
    exit_time: str = ""
    outcome: str = ""  # WIN, LOSS, BREAKEVEN, OPEN
    pnl_pips: float = 0.0
    max_favorable: float = 0.0  # Max favorable excursion (pips)
    max_adverse: float = 0.0    # Max adverse excursion (pips)
    
    # Learning tags
    mistake_type: str = ""  # FOUGHT_TREND, EARLY_ENTRY, LATE_EXIT, BREAKOUT_FADE
    lesson: str = ""


@dataclass
class BreakoutPattern:
    """Track breakout patterns for learning"""
    timestamp: str
    symbol: str
    breakout_type: str  # RESISTANCE, SUPPORT, RANGE
    breakout_level: float
    pre_breakout_rsi: float
    pre_breakout_volume_ratio: float  # vs 20-day avg
    
    # Outcome
    was_real: bool = False  # True = continued, False = fake/reversal
    continuation_pips: float = 0.0  # How far did it go?
    reversal_pips: float = 0.0  # How far did it reverse?
    time_to_resolution: int = 0  # Minutes until we knew if real/fake


class SignalLearner:
    """
    Learn from signal outcomes to improve future predictions.
    
    Core Capabilities:
    1. Track every signal and outcome
    2. Calculate accuracy by pattern type
    3. Learn breakout vs fake-out patterns
    4. Adjust confidence dynamically
    5. Flag high-risk patterns based on history
    """
    
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()
        
        # Cache for quick lookups
        self.pattern_accuracy = {}
        self.breakout_stats = {}
        
        # Load historical stats
        self._load_stats()
        
        logger.info("=" * 60)
        logger.info("📚 SIGNAL LEARNER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Total signals tracked: {self._get_total_signals()}")
        logger.info(f"   Overall win rate: {self._get_overall_win_rate():.1f}%")
        logger.info("=" * 60)
    
    def _init_db(self):
        """Initialize SQLite database for learning"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Signals table
        c.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                entry_time TEXT,
                rsi_at_signal REAL,
                trend_at_signal TEXT,
                was_breakout INTEGER,
                breakout_type TEXT,
                herd_direction TEXT,
                exit_price REAL,
                exit_time TEXT,
                outcome TEXT,
                pnl_pips REAL,
                max_favorable REAL,
                max_adverse REAL,
                mistake_type TEXT,
                lesson TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Breakout patterns table
        c.execute('''
            CREATE TABLE IF NOT EXISTS breakouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                breakout_type TEXT,
                breakout_level REAL,
                pre_breakout_rsi REAL,
                pre_breakout_volume_ratio REAL,
                was_real INTEGER,
                continuation_pips REAL,
                reversal_pips REAL,
                time_to_resolution INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pattern stats table (pre-computed)
        c.execute('''
            CREATE TABLE IF NOT EXISTS pattern_stats (
                pattern_key TEXT PRIMARY KEY,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                avg_win_pips REAL,
                avg_loss_pips REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_total_signals(self) -> int:
        """Get total number of tracked signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM signals")
            result = c.fetchone()[0]
            conn.close()
            return result
        except:
            return 0
    
    def _get_overall_win_rate(self) -> float:
        """Get overall win rate"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM signals WHERE outcome = 'WIN'")
            wins = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM signals WHERE outcome IN ('WIN', 'LOSS')")
            total = c.fetchone()[0]
            conn.close()
            return (wins / total * 100) if total > 0 else 50.0
        except:
            return 50.0
    
    def _load_stats(self):
        """Load pre-computed pattern stats"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT * FROM pattern_stats")
            rows = c.fetchall()
            conn.close()
            
            for row in rows:
                pattern_key = row[0]
                self.pattern_accuracy[pattern_key] = {
                    'total': row[1],
                    'wins': row[2],
                    'losses': row[3],
                    'win_rate': row[4],
                    'avg_win': row[5],
                    'avg_loss': row[6]
                }
        except:
            pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL TRACKING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_signal(self, signal: SignalOutcome) -> str:
        """Record a new signal for tracking"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO signals 
            (signal_id, symbol, direction, entry_price, entry_time, 
             rsi_at_signal, trend_at_signal, was_breakout, breakout_type, 
             herd_direction, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id, signal.symbol, signal.direction,
            signal.entry_price, signal.entry_time, signal.rsi_at_signal,
            signal.trend_at_signal, 1 if signal.was_breakout else 0,
            signal.breakout_type, signal.herd_direction, 'OPEN'
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Signal recorded: {signal.signal_id}")
        return signal.signal_id
    
    def update_outcome(
        self, 
        signal_id: str, 
        exit_price: float, 
        outcome: str,
        pnl_pips: float = 0,
        max_favorable: float = 0,
        max_adverse: float = 0,
        mistake_type: str = "",
        lesson: str = ""
    ):
        """Update a signal with its outcome"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE signals SET
                exit_price = ?,
                exit_time = ?,
                outcome = ?,
                pnl_pips = ?,
                max_favorable = ?,
                max_adverse = ?,
                mistake_type = ?,
                lesson = ?
            WHERE signal_id = ?
        ''', (
            exit_price, datetime.utcnow().isoformat(), outcome,
            pnl_pips, max_favorable, max_adverse, mistake_type, lesson,
            signal_id
        ))
        
        conn.commit()
        conn.close()
        
        # Update pattern stats
        self._update_pattern_stats(signal_id)
        
        logger.info(f"📊 Outcome recorded: {signal_id} → {outcome} ({pnl_pips:+.1f} pips)")
        
        if mistake_type:
            logger.warning(f"   ⚠️ Mistake: {mistake_type}")
            logger.info(f"   📚 Lesson: {lesson}")
    
    def _update_pattern_stats(self, signal_id: str):
        """Update pattern statistics after outcome"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get signal details
        c.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return
        
        # Build pattern keys
        symbol = row[1]
        direction = row[2]
        rsi = row[5]
        trend = row[6]
        was_breakout = row[7]
        outcome = row[12]
        pnl = row[13] or 0
        
        # Pattern key: "XAUUSD_SELL_STRONG_UP_RSI_80+"
        rsi_bucket = "RSI_90+" if rsi > 90 else "RSI_80+" if rsi > 80 else "RSI_70+" if rsi > 70 else "RSI_30-" if rsi < 30 else "RSI_20-" if rsi < 20 else "RSI_10-" if rsi < 10 else "RSI_MID"
        pattern_key = f"{symbol}_{direction}_{trend}_{rsi_bucket}"
        
        # Update stats
        if pattern_key not in self.pattern_accuracy:
            self.pattern_accuracy[pattern_key] = {
                'total': 0, 'wins': 0, 'losses': 0,
                'win_rate': 50, 'avg_win': 0, 'avg_loss': 0,
                'win_pips': [], 'loss_pips': []
            }
        
        stats = self.pattern_accuracy[pattern_key]
        stats['total'] += 1
        
        if outcome == 'WIN':
            stats['wins'] += 1
            stats['win_pips'] = stats.get('win_pips', []) + [pnl]
        elif outcome == 'LOSS':
            stats['losses'] += 1
            stats['loss_pips'] = stats.get('loss_pips', []) + [abs(pnl)]
        
        if stats['total'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total'] * 100
        if stats['win_pips']:
            stats['avg_win'] = np.mean(stats['win_pips'])
        if stats['loss_pips']:
            stats['avg_loss'] = np.mean(stats['loss_pips'])
        
        # Save to DB
        c.execute('''
            INSERT OR REPLACE INTO pattern_stats
            (pattern_key, total_signals, wins, losses, win_rate, avg_win_pips, avg_loss_pips, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern_key, stats['total'], stats['wins'], stats['losses'],
            stats['win_rate'], stats.get('avg_win', 0), stats.get('avg_loss', 0),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BREAKOUT LEARNING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_breakout(self, breakout: BreakoutPattern):
        """Record a breakout for learning"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO breakouts
            (timestamp, symbol, breakout_type, breakout_level, 
             pre_breakout_rsi, pre_breakout_volume_ratio,
             was_real, continuation_pips, reversal_pips, time_to_resolution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            breakout.timestamp, breakout.symbol, breakout.breakout_type,
            breakout.breakout_level, breakout.pre_breakout_rsi,
            breakout.pre_breakout_volume_ratio, 1 if breakout.was_real else 0,
            breakout.continuation_pips, breakout.reversal_pips,
            breakout.time_to_resolution
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Breakout recorded: {breakout.breakout_type} @ ${breakout.breakout_level:.2f}")
    
    def get_breakout_stats(self, symbol: str = None) -> Dict:
        """Get breakout statistics for learning"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if symbol:
            c.execute('''
                SELECT breakout_type, 
                       COUNT(*) as total,
                       SUM(was_real) as real_breakouts,
                       AVG(pre_breakout_rsi) as avg_rsi,
                       AVG(pre_breakout_volume_ratio) as avg_volume,
                       AVG(continuation_pips) as avg_continuation,
                       AVG(reversal_pips) as avg_reversal
                FROM breakouts
                WHERE symbol = ?
                GROUP BY breakout_type
            ''', (symbol,))
        else:
            c.execute('''
                SELECT breakout_type, 
                       COUNT(*) as total,
                       SUM(was_real) as real_breakouts,
                       AVG(pre_breakout_rsi) as avg_rsi,
                       AVG(pre_breakout_volume_ratio) as avg_volume,
                       AVG(continuation_pips) as avg_continuation,
                       AVG(reversal_pips) as avg_reversal
                FROM breakouts
                GROUP BY breakout_type
            ''')
        
        rows = c.fetchall()
        conn.close()
        
        stats = {}
        for row in rows:
            breakout_type = row[0]
            total = row[1]
            real = row[2] or 0
            
            stats[breakout_type] = {
                'total': total,
                'real_breakouts': real,
                'fake_breakouts': total - real,
                'real_rate': (real / total * 100) if total > 0 else 50,
                'avg_rsi_at_breakout': row[3] or 50,
                'avg_volume_ratio': row[4] or 1.0,
                'avg_continuation': row[5] or 0,
                'avg_reversal': row[6] or 0
            }
        
        return stats
    
    def predict_breakout_validity(
        self, 
        symbol: str,
        breakout_type: str,
        current_rsi: float,
        volume_ratio: float
    ) -> Tuple[float, str]:
        """
        Predict if a breakout is real or fake based on historical patterns.
        
        Returns: (probability_real_0_100, reasoning)
        """
        stats = self.get_breakout_stats(symbol)
        
        if breakout_type not in stats or stats[breakout_type]['total'] < 5:
            return 50.0, "Insufficient breakout history - treating as 50/50"
        
        type_stats = stats[breakout_type]
        base_rate = type_stats['real_rate']
        
        # Adjust based on RSI
        rsi_adjustment = 0
        avg_rsi = type_stats['avg_rsi_at_breakout']
        
        # Higher RSI at resistance break = more likely fake (exhaustion)
        if breakout_type == "RESISTANCE" and current_rsi > 80:
            rsi_adjustment = -15
        elif breakout_type == "RESISTANCE" and current_rsi < 70:
            rsi_adjustment = +10
        
        # Lower RSI at support break = more likely fake (capitulation)
        if breakout_type == "SUPPORT" and current_rsi < 20:
            rsi_adjustment = -15
        elif breakout_type == "SUPPORT" and current_rsi > 30:
            rsi_adjustment = +10
        
        # Adjust based on volume
        vol_adjustment = 0
        avg_vol = type_stats['avg_volume_ratio']
        
        # Higher volume = more likely real
        if volume_ratio > 2.0:
            vol_adjustment = +20
        elif volume_ratio > 1.5:
            vol_adjustment = +10
        elif volume_ratio < 0.8:
            vol_adjustment = -15  # Low volume breakout = likely fake
        
        probability = base_rate + rsi_adjustment + vol_adjustment
        probability = max(10, min(90, probability))  # Clamp to 10-90%
        
        reasoning = (
            f"Historical: {base_rate:.0f}% real rate, "
            f"RSI adj: {rsi_adjustment:+d}%, "
            f"Volume adj: {vol_adjustment:+d}%"
        )
        
        return probability, reasoning
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE ADJUSTMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_pattern_confidence(
        self,
        symbol: str,
        direction: str,
        trend: str,
        rsi: float
    ) -> Tuple[float, str]:
        """
        Get historical confidence for a signal pattern.
        
        If we've lost 70% of similar trades, REDUCE confidence!
        If we've won 80% of similar trades, BOOST confidence!
        
        Returns: (confidence_multiplier, reasoning)
        """
        # Build pattern key
        rsi_bucket = "RSI_90+" if rsi > 90 else "RSI_80+" if rsi > 80 else "RSI_70+" if rsi > 70 else "RSI_30-" if rsi < 30 else "RSI_20-" if rsi < 20 else "RSI_10-" if rsi < 10 else "RSI_MID"
        pattern_key = f"{symbol}_{direction}_{trend}_{rsi_bucket}"
        
        if pattern_key in self.pattern_accuracy:
            stats = self.pattern_accuracy[pattern_key]
            
            if stats['total'] >= 5:  # Need at least 5 samples
                win_rate = stats['win_rate']
                
                # If win rate < 40%, reduce confidence significantly
                if win_rate < 40:
                    multiplier = 0.5
                    reasoning = f"⚠️ DANGER PATTERN: {pattern_key} has only {win_rate:.0f}% win rate ({stats['total']} samples)!"
                elif win_rate < 50:
                    multiplier = 0.7
                    reasoning = f"⚠️ Weak pattern: {pattern_key} has {win_rate:.0f}% win rate"
                elif win_rate > 70:
                    multiplier = 1.2
                    reasoning = f"✅ Strong pattern: {pattern_key} has {win_rate:.0f}% win rate!"
                elif win_rate > 60:
                    multiplier = 1.1
                    reasoning = f"✅ Good pattern: {pattern_key} has {win_rate:.0f}% win rate"
                else:
                    multiplier = 1.0
                    reasoning = f"Pattern {pattern_key} has {win_rate:.0f}% win rate (neutral)"
                
                return multiplier, reasoning
        
        # No historical data
        return 1.0, f"No historical data for pattern {pattern_key}"
    
    def analyze_mistakes(self, last_n: int = 20) -> Dict:
        """
        Analyze recent mistakes to identify patterns.
        
        Returns common mistake types and lessons.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT mistake_type, COUNT(*) as count, 
                   AVG(pnl_pips) as avg_loss,
                   GROUP_CONCAT(lesson, ' | ') as lessons
            FROM signals 
            WHERE outcome = 'LOSS' AND mistake_type != ''
            GROUP BY mistake_type
            ORDER BY count DESC
            LIMIT 10
        ''')
        
        rows = c.fetchall()
        conn.close()
        
        mistakes = {}
        for row in rows:
            mistakes[row[0]] = {
                'count': row[1],
                'avg_loss': row[2],
                'lessons': row[3].split(' | ')[:3] if row[3] else []
            }
        
        return mistakes
    
    def get_learning_summary(self) -> str:
        """Get a summary of what NEO has learned"""
        lines = []
        lines.append("=" * 60)
        lines.append("📚 NEO LEARNING SUMMARY")
        lines.append("=" * 60)
        
        # Overall stats
        total = self._get_total_signals()
        win_rate = self._get_overall_win_rate()
        lines.append(f"\n📊 OVERALL: {total} signals tracked, {win_rate:.1f}% win rate")
        
        # Pattern performance
        lines.append("\n📈 PATTERN PERFORMANCE (top 5 by volume):")
        sorted_patterns = sorted(
            self.pattern_accuracy.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:5]
        
        for pattern, stats in sorted_patterns:
            emoji = "🟢" if stats['win_rate'] > 60 else "🔴" if stats['win_rate'] < 40 else "🟡"
            lines.append(f"   {emoji} {pattern}: {stats['win_rate']:.0f}% WR ({stats['total']} trades)")
        
        # Breakout stats
        breakout_stats = self.get_breakout_stats()
        if breakout_stats:
            lines.append("\n📊 BREAKOUT LEARNING:")
            for b_type, stats in breakout_stats.items():
                lines.append(f"   {b_type}: {stats['real_rate']:.0f}% real ({stats['total']} samples)")
                lines.append(f"      Avg continuation: {stats['avg_continuation']:.1f} pips")
                lines.append(f"      Avg reversal: {stats['avg_reversal']:.1f} pips")
        
        # Common mistakes
        mistakes = self.analyze_mistakes()
        if mistakes:
            lines.append("\n⚠️ COMMON MISTAKES:")
            for mistake, data in list(mistakes.items())[:3]:
                lines.append(f"   {mistake}: {data['count']}x (avg loss: {data['avg_loss']:.1f} pips)")
                if data['lessons']:
                    lines.append(f"      📚 Lesson: {data['lessons'][0]}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BREAKOUT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_breakout(
        self, 
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[Dict]:
        """
        Detect if price is breaking out of a range/level.
        
        Returns breakout info if detected, None otherwise.
        """
        if len(df) < lookback:
            return None
        
        # Ensure lowercase columns
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        
        recent = df.tail(lookback)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = current['close']
        current_high = current['high']
        current_low = current['low']
        
        # Find range
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        
        # Calculate volume ratio
        current_vol = current.get('volume', 0)
        avg_vol = recent['volume'].mean() if 'volume' in recent.columns else 1
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Calculate RSI
        closes = df['close']
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ═══ CHECK FOR RESISTANCE BREAKOUT ═══
        # Price closing above range high
        if current_price > range_high and prev['close'] <= range_high:
            # Get breakout probability
            prob, reason = self.predict_breakout_validity(
                self.symbol if hasattr(self, 'symbol') else 'XAUUSD',
                'RESISTANCE',
                rsi,
                vol_ratio
            )
            
            return {
                'type': 'RESISTANCE_BREAK',
                'level': range_high,
                'current_price': current_price,
                'breakout_amount': current_price - range_high,
                'rsi': rsi,
                'volume_ratio': vol_ratio,
                'probability_real': prob,
                'reasoning': reason,
                'action': 'BUY' if prob > 60 else 'WAIT' if prob > 40 else 'FADE_SELL'
            }
        
        # ═══ CHECK FOR SUPPORT BREAKOUT (BREAKDOWN) ═══
        if current_price < range_low and prev['close'] >= range_low:
            prob, reason = self.predict_breakout_validity(
                self.symbol if hasattr(self, 'symbol') else 'XAUUSD',
                'SUPPORT',
                rsi,
                vol_ratio
            )
            
            return {
                'type': 'SUPPORT_BREAK',
                'level': range_low,
                'current_price': current_price,
                'breakout_amount': range_low - current_price,
                'rsi': rsi,
                'volume_ratio': vol_ratio,
                'probability_real': prob,
                'reasoning': reason,
                'action': 'SELL' if prob > 60 else 'WAIT' if prob > 40 else 'FADE_BUY'
            }
        
        # ═══ CHECK FOR RANGE EXPANSION (Breakout imminent) ═══
        # Price near edge of range with momentum
        distance_to_high = range_high - current_price
        distance_to_low = current_price - range_low
        
        if distance_to_high < range_size * 0.1:  # Within 10% of high
            return {
                'type': 'NEAR_RESISTANCE',
                'level': range_high,
                'current_price': current_price,
                'distance': distance_to_high,
                'rsi': rsi,
                'volume_ratio': vol_ratio,
                'warning': 'Breakout imminent! Watch for confirmation.',
                'action': 'WAIT_FOR_BREAK'
            }
        
        if distance_to_low < range_size * 0.1:  # Within 10% of low
            return {
                'type': 'NEAR_SUPPORT',
                'level': range_low,
                'current_price': current_price,
                'distance': distance_to_low,
                'rsi': rsi,
                'volume_ratio': vol_ratio,
                'warning': 'Breakdown imminent! Watch for confirmation.',
                'action': 'WAIT_FOR_BREAK'
            }
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_learner = None

def get_learner() -> SignalLearner:
    """Get singleton learner instance"""
    global _learner
    if _learner is None:
        _learner = SignalLearner()
    return _learner


def record_signal_outcome(
    signal_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    rsi: float,
    trend: str,
    exit_price: float = None,
    outcome: str = None,
    pnl_pips: float = 0,
    mistake_type: str = "",
    lesson: str = ""
):
    """Quick function to record a signal and optionally its outcome"""
    learner = get_learner()
    
    signal = SignalOutcome(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        entry_time=datetime.utcnow().isoformat(),
        rsi_at_signal=rsi,
        trend_at_signal=trend
    )
    
    learner.record_signal(signal)
    
    if outcome:
        learner.update_outcome(
            signal_id, exit_price or entry_price, outcome,
            pnl_pips, mistake_type=mistake_type, lesson=lesson
        )


def get_pattern_confidence(symbol: str, direction: str, trend: str, rsi: float) -> Tuple[float, str]:
    """Quick function to get confidence adjustment"""
    return get_learner().get_pattern_confidence(symbol, direction, trend, rsi)


def detect_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """Quick function to detect breakouts"""
    return get_learner().detect_breakout(df)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_learner():
    """Test the signal learner"""
    print("\n" + "=" * 70)
    print("🧪 TESTING SIGNAL LEARNER")
    print("=" * 70)
    
    learner = SignalLearner()
    
    # Record some test signals
    test_signals = [
        # Pattern: XAUUSD SELL in STRONG_UP with RSI 80+ (bad pattern!)
        {"id": "test_001", "symbol": "XAUUSD", "direction": "SELL", "price": 5000, "rsi": 85, "trend": "STRONG_UP", "outcome": "LOSS", "pnl": -50, "mistake": "FOUGHT_TREND", "lesson": "Don't sell in strong uptrend!"},
        {"id": "test_002", "symbol": "XAUUSD", "direction": "SELL", "price": 5020, "rsi": 82, "trend": "STRONG_UP", "outcome": "LOSS", "pnl": -45, "mistake": "FOUGHT_TREND", "lesson": "RSI overbought doesn't mean sell in uptrend!"},
        {"id": "test_003", "symbol": "XAUUSD", "direction": "SELL", "price": 5050, "rsi": 88, "trend": "STRONG_UP", "outcome": "LOSS", "pnl": -60, "mistake": "FOUGHT_TREND", "lesson": "Wait for trend change!"},
        
        # Pattern: XAUUSD BUY in UP with RSI 30- (good pattern!)
        {"id": "test_004", "symbol": "XAUUSD", "direction": "BUY", "price": 4950, "rsi": 25, "trend": "UP", "outcome": "WIN", "pnl": 80, "mistake": "", "lesson": ""},
        {"id": "test_005", "symbol": "XAUUSD", "direction": "BUY", "price": 4980, "rsi": 28, "trend": "UP", "outcome": "WIN", "pnl": 70, "mistake": "", "lesson": ""},
        {"id": "test_006", "symbol": "XAUUSD", "direction": "BUY", "price": 4960, "rsi": 22, "trend": "UP", "outcome": "WIN", "pnl": 90, "mistake": "", "lesson": ""},
    ]
    
    for s in test_signals:
        signal = SignalOutcome(
            signal_id=s["id"],
            symbol=s["symbol"],
            direction=s["direction"],
            entry_price=s["price"],
            entry_time=datetime.utcnow().isoformat(),
            rsi_at_signal=s["rsi"],
            trend_at_signal=s["trend"]
        )
        learner.record_signal(signal)
        learner.update_outcome(
            s["id"], s["price"] + (s["pnl"] if s["direction"] == "BUY" else -s["pnl"]),
            s["outcome"], s["pnl"],
            mistake_type=s["mistake"], lesson=s["lesson"]
        )
    
    # Test confidence adjustment
    print("\n📊 CONFIDENCE ADJUSTMENTS:")
    
    # Bad pattern (should reduce confidence)
    conf, reason = learner.get_pattern_confidence("XAUUSD", "SELL", "STRONG_UP", 85)
    print(f"\n   SELL XAUUSD in STRONG_UP with RSI 85:")
    print(f"   → Confidence multiplier: {conf:.1f}x")
    print(f"   → {reason}")
    
    # Good pattern (should boost confidence)
    conf, reason = learner.get_pattern_confidence("XAUUSD", "BUY", "UP", 25)
    print(f"\n   BUY XAUUSD in UP with RSI 25:")
    print(f"   → Confidence multiplier: {conf:.1f}x")
    print(f"   → {reason}")
    
    # Print learning summary
    print(learner.get_learning_summary())
    
    # Test breakout detection
    print("\n📊 BREAKOUT DETECTION TEST:")
    import yfinance as yf
    
    gold = yf.download('GC=F', period='1mo', interval='1h', progress=False)
    if not gold.empty:
        if hasattr(gold.columns, 'levels'):
            gold.columns = [col[0].lower() for col in gold.columns]
        else:
            gold.columns = [c.lower() for c in gold.columns]
        
        breakout = learner.detect_breakout(gold)
        if breakout:
            print(f"   ⚡ BREAKOUT DETECTED!")
            print(f"   Type: {breakout['type']}")
            print(f"   Level: ${breakout['level']:.2f}")
            print(f"   Action: {breakout['action']}")
            if 'probability_real' in breakout:
                print(f"   Probability real: {breakout['probability_real']:.0f}%")
        else:
            print("   No breakout currently detected")


if __name__ == "__main__":
    test_learner()

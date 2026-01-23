#!/usr/bin/env python3
"""
NEO Memory Store - Learning Database
Stores all decisions, outcomes, and learnings in SQLite.
NO RANDOM DATA - All entries from real analysis and outcomes.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from config import DATABASE_PATH


@dataclass
class Decision:
    """A trading decision made by NEO."""
    id: Optional[int]
    timestamp: str
    decision_type: str  # SIGNAL, HOLD, CLOSE, ADJUST
    symbol: Optional[str]
    direction: Optional[str]  # BUY, SELL, None
    confidence: float  # 0-100
    reasoning: str  # LLM's explanation
    market_context: str  # Snapshot at decision time
    model_used: str  # Which LLM made decision


@dataclass 
class Outcome:
    """The result of a decision."""
    decision_id: int
    timestamp: str
    result: str  # WIN, LOSS, BREAKEVEN, PENDING
    pnl: float
    duration_minutes: int
    market_move: float  # How much price moved
    lesson_learned: str  # LLM's reflection


@dataclass
class Learning:
    """A lesson NEO has extracted from outcomes."""
    id: Optional[int]
    timestamp: str
    category: str  # PATTERN, MISTAKE, SUCCESS, RULE
    content: str
    confidence: float
    supporting_decisions: List[int]  # Decision IDs that support this
    times_validated: int = 0


class MemoryStore:
    """
    SQLite-based memory for NEO's learning.
    Tracks decisions, outcomes, and extracted learnings.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATABASE_PATH)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    symbol TEXT,
                    direction TEXT,
                    confidence REAL,
                    reasoning TEXT NOT NULL,
                    market_context TEXT,
                    model_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Outcomes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    result TEXT NOT NULL,
                    pnl REAL,
                    duration_minutes INTEGER,
                    market_move REAL,
                    lesson_learned TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES decisions(id)
                )
            """)
            
            # Learnings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    supporting_decisions TEXT,
                    times_validated INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    decisions_made INTEGER,
                    signals_generated INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    best_decision_id INTEGER,
                    worst_decision_id INTEGER
                )
            """)
            
            conn.commit()
    
    def save_decision(self, decision: Decision) -> int:
        """Save a decision and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decisions 
                (timestamp, decision_type, symbol, direction, confidence, 
                 reasoning, market_context, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.timestamp, decision.decision_type, decision.symbol,
                decision.direction, decision.confidence, decision.reasoning,
                decision.market_context, decision.model_used
            ))
            conn.commit()
            return cursor.lastrowid
    
    def save_outcome(self, outcome: Outcome):
        """Save an outcome for a decision."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO outcomes
                (decision_id, timestamp, result, pnl, duration_minutes,
                 market_move, lesson_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.decision_id, outcome.timestamp, outcome.result,
                outcome.pnl, outcome.duration_minutes, outcome.market_move,
                outcome.lesson_learned
            ))
            conn.commit()
    
    def save_learning(self, learning: Learning) -> int:
        """Save a learning and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learnings
                (timestamp, category, content, confidence, supporting_decisions, times_validated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                learning.timestamp, learning.category, learning.content,
                learning.confidence, json.dumps(learning.supporting_decisions),
                learning.times_validated
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_decisions(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent decisions for context."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            cursor.execute("""
                SELECT d.*, o.result, o.pnl, o.lesson_learned
                FROM decisions d
                LEFT JOIN outcomes o ON d.id = o.decision_id
                WHERE d.timestamp > ?
                ORDER BY d.timestamp DESC
                LIMIT ?
            """, (cutoff, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_learnings_by_category(self, category: str = None, min_confidence: float = 0) -> List[Dict]:
        """Get learnings, optionally filtered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if category:
                cursor.execute("""
                    SELECT * FROM learnings 
                    WHERE category = ? AND confidence >= ?
                    ORDER BY confidence DESC, times_validated DESC
                """, (category, min_confidence))
            else:
                cursor.execute("""
                    SELECT * FROM learnings 
                    WHERE confidence >= ?
                    ORDER BY confidence DESC, times_validated DESC
                """, (min_confidence,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get performance statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Total decisions
            cursor.execute("SELECT COUNT(*) FROM decisions WHERE timestamp > ?", (cutoff,))
            total_decisions = cursor.fetchone()[0]
            
            # Decisions with outcomes
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM outcomes
                WHERE timestamp > ?
            """, (cutoff,))
            
            row = cursor.fetchone()
            
            return {
                "period_days": days,
                "total_decisions": total_decisions,
                "outcomes_recorded": row[0] or 0,
                "wins": row[1] or 0,
                "losses": row[2] or 0,
                "win_rate": (row[1] / row[0] * 100) if row[0] else 0,
                "total_pnl": row[3] or 0,
                "avg_pnl": row[4] or 0
            }
    
    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get performance for a specific symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as trades,
                    SUM(CASE WHEN o.result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(o.pnl) as total_pnl
                FROM decisions d
                JOIN outcomes o ON d.id = o.decision_id
                WHERE d.symbol = ? AND d.timestamp > ?
            """, (symbol, cutoff))
            
            row = cursor.fetchone()
            return {
                "symbol": symbol,
                "trades": row[0] or 0,
                "wins": row[1] or 0,
                "win_rate": (row[1] / row[0] * 100) if row[0] else 0,
                "total_pnl": row[2] or 0
            }
    
    def to_llm_context(self, include_learnings: bool = True) -> str:
        """Format memory for LLM consumption."""
        lines = ["=== NEO MEMORY ===", ""]
        
        # Recent performance
        stats = self.get_performance_stats(7)
        lines.extend([
            "LAST 7 DAYS PERFORMANCE:",
            f"  Decisions: {stats['total_decisions']}",
            f"  Wins: {stats['wins']} | Losses: {stats['losses']}",
            f"  Win Rate: {stats['win_rate']:.1f}%",
            f"  Total P&L: ${stats['total_pnl']:+,.2f}",
            ""
        ])
        
        # Recent decisions
        recent = self.get_recent_decisions(hours=24, limit=5)
        if recent:
            lines.append("RECENT DECISIONS (24h):")
            for d in recent:
                result = d.get('result', 'PENDING')
                pnl = d.get('pnl', 0) or 0
                lines.append(
                    f"  [{d['decision_type']}] {d.get('symbol', 'N/A')} "
                    f"{d.get('direction', '')} -> {result} (${pnl:+.2f})"
                )
            lines.append("")
        
        # Key learnings
        if include_learnings:
            learnings = self.get_learnings_by_category(min_confidence=70)[:5]
            if learnings:
                lines.append("KEY LEARNINGS:")
                for l in learnings:
                    lines.append(f"  [{l['category']}] {l['content'][:100]}...")
        
        return "\n".join(lines)


def test_memory_store():
    """Test the memory store."""
    print("=" * 60)
    print("NEO MEMORY STORE TEST")
    print("=" * 60)
    
    store = MemoryStore()
    
    # Save a test decision
    decision = Decision(
        id=None,
        timestamp=datetime.utcnow().isoformat(),
        decision_type="SIGNAL",
        symbol="EURUSD",
        direction="BUY",
        confidence=75.0,
        reasoning="RSI(2) showing extreme oversold conditions below 10. "
                  "Price above 200 SMA indicating uptrend. "
                  "Expecting mean reversion bounce.",
        market_context="EURUSD: 1.08500, RSI(2): 8.5, Trend: UP",
        model_used="deepseek-r1:70b"
    )
    
    decision_id = store.save_decision(decision)
    print(f"✅ Saved decision #{decision_id}")
    
    # Print context
    print("")
    print(store.to_llm_context())
    print("")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_store()

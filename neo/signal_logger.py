#!/usr/bin/env python3
"""
NEO Signal Logger - Persistent SQLite storage for signal history
Tracks: generation, sending, execution, and outcomes
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("NEO-SignalLogger")

DB_PATH = "/tmp/neo_signals.db"


class SignalLogger:
    """Logs and tracks NEO trading signals with full lifecycle management"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
        log.info(f"✅ SignalLogger initialized: {db_path}")
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
    
    def _create_tables(self):
        """Create the signals table if it doesn't exist"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size_usd REAL,
                reasoning TEXT,
                features_json TEXT,
                source TEXT DEFAULT 'NEO_AUTO',
                status TEXT DEFAULT 'GENERATED',
                ghost_received_at DATETIME,
                ghost_executed_at DATETIME,
                execution_price REAL,
                outcome TEXT,
                pnl REAL,
                notes TEXT
            )
        ''')
        
        # Create index for faster queries
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
            ON signals(timestamp DESC)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_symbol 
            ON signals(symbol)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_status 
            ON signals(status)
        ''')
        
        self.conn.commit()
        log.info("📊 Signal tables ready")
    
    def log_signal(self, signal: Dict[str, Any], source: str = "NEO_AUTO") -> int:
        """
        Log a new signal to the database
        
        Args:
            signal: Signal dict with keys like symbol, action, confidence, etc.
            source: Where the signal came from (NEO_AUTO, NEO_ASK_BUTTON, etc.)
        
        Returns:
            The inserted row ID
        """
        # Extract signal data
        trade = signal.get('trade', {})
        metadata = signal.get('metadata', {})
        
        signal_id = signal.get('signal_id') or f"NEO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        symbol = trade.get('symbol') or signal.get('symbol', 'UNKNOWN')
        action = trade.get('direction') or signal.get('action', 'UNKNOWN')
        confidence = trade.get('confidence') or metadata.get('confidence') or signal.get('confidence', 0)
        entry_price = trade.get('entry_price') or signal.get('entry_price')
        stop_loss = trade.get('stop_loss') or signal.get('stop_loss')
        take_profit = trade.get('take_profit') or signal.get('take_profit')
        position_size = trade.get('position_size_usd') or trade.get('position_value_usd') or signal.get('position_size_usd', 0)
        
        # Handle reasoning - could be string or list
        reasoning = metadata.get('reasoning') or signal.get('reasoning', '')
        if isinstance(reasoning, list):
            reasoning = ' | '.join(reasoning)
        
        # Features as JSON
        features = signal.get('features', {})
        features_json = json.dumps(features) if features else '{}'
        
        try:
            cursor = self.conn.execute('''
                INSERT INTO signals (
                    signal_id, symbol, action, confidence, entry_price,
                    stop_loss, take_profit, position_size_usd, reasoning,
                    features_json, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'GENERATED')
            ''', (
                signal_id, symbol, action, confidence, entry_price,
                stop_loss, take_profit, position_size, reasoning,
                features_json, source
            ))
            self.conn.commit()
            row_id = cursor.lastrowid
            log.info(f"📝 Signal logged: {signal_id} ({symbol} {action} {confidence}%)")
            return row_id
        except sqlite3.IntegrityError:
            # Signal already exists (duplicate signal_id)
            log.warning(f"⚠️ Signal already exists: {signal_id}")
            return -1
        except Exception as e:
            log.error(f"❌ Error logging signal: {e}")
            return -1
    
    def update_status(self, signal_id: str, status: str, **kwargs) -> bool:
        """
        Update signal status and optional fields
        
        Args:
            signal_id: The unique signal identifier
            status: New status (GENERATED, SENT, EXECUTED, EXPIRED, CANCELLED)
            **kwargs: Additional fields to update (ghost_received_at, execution_price, etc.)
        """
        valid_statuses = ['GENERATED', 'SENT', 'EXECUTED', 'CLOSED', 'EXPIRED', 'CANCELLED']
        if status not in valid_statuses:
            log.warning(f"Invalid status: {status}")
            return False
        
        # Build update query
        updates = ['status = ?']
        values = [status]
        
        if 'ghost_received_at' in kwargs:
            updates.append('ghost_received_at = ?')
            values.append(kwargs['ghost_received_at'] or datetime.utcnow().isoformat())
        
        if 'ghost_executed_at' in kwargs:
            updates.append('ghost_executed_at = ?')
            values.append(kwargs['ghost_executed_at'] or datetime.utcnow().isoformat())
        
        if 'execution_price' in kwargs:
            updates.append('execution_price = ?')
            values.append(kwargs['execution_price'])
        
        if 'outcome' in kwargs:
            updates.append('outcome = ?')
            values.append(kwargs['outcome'])
        
        if 'pnl' in kwargs:
            updates.append('pnl = ?')
            values.append(kwargs['pnl'])
        
        if 'notes' in kwargs:
            updates.append('notes = ?')
            values.append(kwargs['notes'])
        
        values.append(signal_id)
        
        query = f"UPDATE signals SET {', '.join(updates)} WHERE signal_id = ?"
        
        try:
            cursor = self.conn.execute(query, values)
            self.conn.commit()
            if cursor.rowcount > 0:
                log.info(f"📊 Signal {signal_id} → {status}")
                return True
            else:
                log.warning(f"⚠️ Signal not found: {signal_id}")
                return False
        except Exception as e:
            log.error(f"❌ Error updating signal: {e}")
            return False
    
    def mark_sent(self, signal_id: str) -> bool:
        """Mark signal as sent to Ghost Commander"""
        return self.update_status(signal_id, 'SENT', ghost_received_at=datetime.utcnow().isoformat())
    
    def mark_executed(self, signal_id: str, execution_price: float) -> bool:
        """Mark signal as executed"""
        return self.update_status(
            signal_id, 'EXECUTED',
            ghost_executed_at=datetime.utcnow().isoformat(),
            execution_price=execution_price
        )
    
    def mark_outcome(self, signal_id: str, outcome: str, pnl: float) -> bool:
        """Record signal outcome (WIN/LOSS) - keeps status as EXECUTED"""
        return self.update_status(signal_id, 'EXECUTED', outcome=outcome, pnl=pnl)
    
    def record_outcome(self, signal_id: str, outcome: str, pnl: float, 
                       exit_price: float = None, pips: float = None) -> bool:
        """
        Record trade outcome and close the signal.
        Called by Ghost Commander v0201 when trade closes.
        
        Args:
            signal_id: The unique signal identifier
            outcome: WIN or LOSS
            pnl: Profit/Loss in dollars
            exit_price: Trade exit price (optional)
            pips: Profit/Loss in pips (optional)
        """
        # Build update - change status to CLOSED (not EXECUTED)
        updates = ['status = ?', 'outcome = ?', 'pnl = ?']
        values = ['CLOSED', outcome, pnl]
        
        if exit_price:
            updates.append('notes = COALESCE(notes, "") || ?')
            values.append(f" | Exit: {exit_price}")
        
        values.append(signal_id)
        
        query = f"UPDATE signals SET {', '.join(updates)} WHERE signal_id = ?"
        
        try:
            cursor = self.conn.execute(query, values)
            self.conn.commit()
            if cursor.rowcount > 0:
                emoji = "✅" if outcome == "WIN" else "❌"
                sign = "+" if pnl >= 0 else ""
                log.info(f"{emoji} Signal {signal_id} CLOSED: {outcome} | {sign}${pnl:.2f}")
                return True
            else:
                log.warning(f"⚠️ Signal not found for outcome: {signal_id}")
                return False
        except Exception as e:
            log.error(f"❌ Error recording outcome: {e}")
            return False
    
    def get_signals(self, limit: int = 50, offset: int = 0, symbol: str = None, 
                    status: str = None) -> List[Dict[str, Any]]:
        """
        Get paginated signal history
        
        Args:
            limit: Max signals to return
            offset: Pagination offset
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)
        
        Returns:
            List of signal dicts
        """
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self.conn.execute(query, params)
        signals = []
        
        for row in cursor.fetchall():
            signals.append({
                'id': row['id'],
                'signal_id': row['signal_id'],
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'action': row['action'],
                'confidence': row['confidence'],
                'entry_price': row['entry_price'],
                'stop_loss': row['stop_loss'],
                'take_profit': row['take_profit'],
                'position_size': row['position_size_usd'],
                'reasoning': row['reasoning'],
                'features': json.loads(row['features_json'] or '{}'),
                'source': row['source'],
                'status': row['status'],
                'ghost_received': row['ghost_received_at'],
                'ghost_executed': row['ghost_executed_at'],
                'execution_price': row['execution_price'],
                'outcome': row['outcome'],
                'pnl': row['pnl'],
                'notes': row['notes']
            })
        
        return signals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal statistics"""
        stats = {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'by_symbol': {},
            'by_status': {}
        }
        
        # Total count
        stats['total'] = self.conn.execute(
            "SELECT COUNT(*) FROM signals"
        ).fetchone()[0]
        
        # Wins/Losses
        stats['wins'] = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'WIN'"
        ).fetchone()[0]
        
        stats['losses'] = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'LOSS'"
        ).fetchone()[0]
        
        stats['pending'] = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome IS NULL OR outcome = 'PENDING'"
        ).fetchone()[0]
        
        # Total P&L
        result = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM signals WHERE pnl IS NOT NULL"
        ).fetchone()
        stats['total_pnl'] = result[0] if result else 0.0
        
        # Win rate
        completed = stats['wins'] + stats['losses']
        if completed > 0:
            stats['win_rate'] = (stats['wins'] / completed) * 100
        
        # By symbol
        for row in self.conn.execute(
            "SELECT symbol, COUNT(*) as count FROM signals GROUP BY symbol"
        ):
            stats['by_symbol'][row['symbol']] = row['count']
        
        # By status
        for row in self.conn.execute(
            "SELECT status, COUNT(*) as count FROM signals GROUP BY status"
        ):
            stats['by_status'][row['status']] = row['count']
        
        return stats
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent signals"""
        return self.get_signals(limit=count)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Global instance for easy import
_logger_instance = None

def get_signal_logger() -> SignalLogger:
    """Get or create the global signal logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SignalLogger()
    return _logger_instance


def log_signal(signal: Dict[str, Any], source: str = "NEO_AUTO") -> int:
    """Convenience function to log a signal"""
    return get_signal_logger().log_signal(signal, source)


def mark_sent(signal_id: str) -> bool:
    """Convenience function to mark signal as sent"""
    return get_signal_logger().mark_sent(signal_id)


def mark_executed(signal_id: str, execution_price: float) -> bool:
    """Convenience function to mark signal as executed"""
    return get_signal_logger().mark_executed(signal_id, execution_price)


if __name__ == "__main__":
    # Test the signal logger
    print("🧪 Testing Signal Logger...")
    
    sl = SignalLogger()
    
    # Log a test signal
    test_signal = {
        'signal_id': f'TEST_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
        'trade': {
            'symbol': 'XAUUSD',
            'direction': 'SELL',
            'confidence': 85,
            'entry_price': 2750.50,
            'stop_loss': 2765.50,
            'take_profit': 2720.50,
            'position_size_usd': 5000
        },
        'metadata': {
            'reasoning': ['RSI overbought', 'Trend exhaustion'],
            'confidence': 85
        },
        'features': {
            'rsi2_h1': 100,
            'vix': 15.5
        }
    }
    
    row_id = sl.log_signal(test_signal, source='TEST')
    print(f"✅ Logged signal (row {row_id})")
    
    # Get signals
    signals = sl.get_signals(limit=5)
    print(f"\n📊 Recent signals ({len(signals)}):")
    for s in signals:
        print(f"   {s['signal_id']}: {s['symbol']} {s['action']} ({s['confidence']}%) - {s['status']}")
    
    # Get stats
    stats = sl.get_stats()
    print(f"\n📈 Stats:")
    print(f"   Total: {stats['total']}")
    print(f"   Wins: {stats['wins']}")
    print(f"   Losses: {stats['losses']}")
    print(f"   P&L: ${stats['total_pnl']:.2f}")
    
    print("\n✅ Signal Logger test complete!")

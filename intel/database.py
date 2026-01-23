#!/usr/bin/env python3
"""
Intel Database - SQLite storage for trading intel

Stores data from:
- Myfxbook verified traders
- MQL5 signal providers
- Forex Factory calendar
- Generated signals

RULE: Every record stores source URL for verification
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from config import db_config


class IntelDatabase:
    """
    SQLite database for trading intelligence
    
    Tables:
    - top_traders: Verified traders from Myfxbook/MQL5
    - trades: Recent trades from top traders
    - calendar: Economic events from Forex Factory
    - signals: Generated consensus signals
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or db_config.DB_PATH
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Top traders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS top_traders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,           -- 'myfxbook' or 'mql5'
                    source_url TEXT NOT NULL,       -- VERIFICATION URL
                    gain_pct REAL,
                    drawdown_pct REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    verified BOOLEAN DEFAULT 0,
                    subscribers INTEGER DEFAULT 0,
                    weeks_trading INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    raw_data TEXT                   -- JSON of all scraped fields
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,        -- 'BUY' or 'SELL'
                    open_time TEXT,
                    close_time TEXT,
                    lots REAL,
                    pnl REAL,
                    pips REAL,
                    source_url TEXT NOT NULL,       -- VERIFICATION URL
                    scraped_at TEXT NOT NULL,
                    FOREIGN KEY (trader_id) REFERENCES top_traders(trader_id)
                )
            """)
            
            # Calendar events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calendar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    datetime TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    impact TEXT NOT NULL,           -- 'high', 'medium', 'low'
                    previous TEXT,
                    forecast TEXT,
                    actual TEXT,
                    source_url TEXT NOT NULL,       -- VERIFICATION URL
                    scraped_at TEXT NOT NULL
                )
            """)
            
            # Generated signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence INTEGER,
                    source_type TEXT NOT NULL,      -- 'myfxbook_consensus', 'mql5_consensus', etc.
                    traders_json TEXT NOT NULL,     -- JSON array of trader info
                    status TEXT DEFAULT 'pending', -- 'pending', 'sent', 'executed'
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_trader ON trades(trader_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(close_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_calendar_datetime ON calendar(datetime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
    
    # ==================== TOP TRADERS ====================
    
    def upsert_trader(self, trader_data: Dict) -> bool:
        """Insert or update a trader record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO top_traders 
                (trader_id, name, source, source_url, gain_pct, drawdown_pct, 
                 win_rate, total_trades, verified, subscribers, weeks_trading,
                 last_updated, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trader_id) DO UPDATE SET
                    name = excluded.name,
                    gain_pct = excluded.gain_pct,
                    drawdown_pct = excluded.drawdown_pct,
                    win_rate = excluded.win_rate,
                    total_trades = excluded.total_trades,
                    verified = excluded.verified,
                    subscribers = excluded.subscribers,
                    weeks_trading = excluded.weeks_trading,
                    last_updated = excluded.last_updated,
                    raw_data = excluded.raw_data
            """, (
                trader_data["trader_id"],
                trader_data["name"],
                trader_data["source"],
                trader_data["source_url"],
                trader_data.get("gain_pct"),
                trader_data.get("drawdown_pct"),
                trader_data.get("win_rate"),
                trader_data.get("total_trades"),
                trader_data.get("verified", False),
                trader_data.get("subscribers", 0),
                trader_data.get("weeks_trading", 0),
                datetime.utcnow().isoformat() + "Z",
                json.dumps(trader_data)
            ))
            
            return True
    
    def get_top_traders(self, source: str = None, min_gain: float = None, 
                        max_drawdown: float = None, limit: int = 100) -> List[Dict]:
        """Get top traders matching criteria"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM top_traders WHERE 1=1"
            params = []
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            if min_gain is not None:
                query += " AND gain_pct >= ?"
                params.append(min_gain)
            
            if max_drawdown is not None:
                query += " AND drawdown_pct <= ?"
                params.append(max_drawdown)
            
            query += " ORDER BY gain_pct DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== TRADES ====================
    
    def insert_trade(self, trade_data: Dict) -> bool:
        """Insert a trade record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades 
                (trader_id, symbol, direction, open_time, close_time, 
                 lots, pnl, pips, source_url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data["trader_id"],
                trade_data["symbol"],
                trade_data["direction"],
                trade_data.get("open_time"),
                trade_data.get("close_time"),
                trade_data.get("lots"),
                trade_data.get("pnl"),
                trade_data.get("pips"),
                trade_data["source_url"],
                datetime.utcnow().isoformat() + "Z"
            ))
            
            return True
    
    def get_recent_trades(self, hours: int = 24, symbol: str = None) -> List[Dict]:
        """Get recent trades within time window"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff = datetime.utcnow().isoformat() + "Z"
            
            query = """
                SELECT t.*, tr.name as trader_name, tr.source_url as trader_url
                FROM trades t
                JOIN top_traders tr ON t.trader_id = tr.trader_id
                WHERE t.scraped_at >= datetime('now', ?)
            """
            params = [f"-{hours} hours"]
            
            if symbol:
                query += " AND t.symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY t.close_time DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_by_trader(self, trader_id: str, limit: int = 20) -> List[Dict]:
        """Get recent trades for a specific trader"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM trades 
                WHERE trader_id = ?
                ORDER BY close_time DESC
                LIMIT ?
            """, (trader_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== CALENDAR ====================
    
    def upsert_calendar_event(self, event_data: Dict) -> bool:
        """Insert or update a calendar event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            event_id = event_data.get("event_id") or f"{event_data['datetime']}_{event_data['event_name']}"
            
            cursor.execute("""
                INSERT INTO calendar 
                (event_id, datetime, currency, event_name, impact, 
                 previous, forecast, actual, source_url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    actual = excluded.actual,
                    scraped_at = excluded.scraped_at
            """, (
                event_id,
                event_data["datetime"],
                event_data["currency"],
                event_data["event_name"],
                event_data["impact"],
                event_data.get("previous"),
                event_data.get("forecast"),
                event_data.get("actual"),
                event_data["source_url"],
                datetime.utcnow().isoformat() + "Z"
            ))
            
            return True
    
    def get_upcoming_events(self, hours: int = 24, currency: str = None,
                           impact: str = None) -> List[Dict]:
        """Get upcoming calendar events"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM calendar 
                WHERE datetime >= datetime('now')
                AND datetime <= datetime('now', ?)
            """
            params = [f"+{hours} hours"]
            
            if currency:
                query += " AND currency = ?"
                params.append(currency)
            
            if impact:
                query += " AND impact = ?"
                params.append(impact)
            
            query += " ORDER BY datetime ASC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== SIGNALS ====================
    
    def insert_signal(self, signal_data: Dict) -> int:
        """Insert a generated signal"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals 
                (timestamp, symbol, direction, confidence, source_type, 
                 traders_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data["timestamp"],
                signal_data["symbol"],
                signal_data["direction"],
                signal_data.get("confidence", 0),
                signal_data["source_type"],
                json.dumps(signal_data.get("traders", [])),
                "pending",
                datetime.utcnow().isoformat() + "Z"
            ))
            
            return cursor.lastrowid
    
    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """Get recently generated signals"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM signals 
                WHERE created_at >= datetime('now', ?)
                ORDER BY created_at DESC
            """, (f"-{hours} hours",))
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data["traders"] = json.loads(data["traders_json"])
                results.append(data)
            
            return results
    
    # ==================== STATS ====================
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            cursor.execute("SELECT COUNT(*) FROM top_traders")
            stats["total_traders"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM top_traders WHERE source = 'myfxbook'")
            stats["myfxbook_traders"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM top_traders WHERE source = 'mql5'")
            stats["mql5_traders"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            stats["total_trades"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM calendar")
            stats["total_events"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM signals")
            stats["total_signals"] = cursor.fetchone()[0]
            
            return stats


if __name__ == "__main__":
    print("=" * 60)
    print("INTEL DATABASE TEST")
    print("=" * 60)
    
    db = IntelDatabase()
    
    # Test insert
    test_trader = {
        "trader_id": "myfxbook_test123",
        "name": "TestTrader",
        "source": "myfxbook",
        "source_url": "https://www.myfxbook.com/members/TestTrader",
        "gain_pct": 45.5,
        "drawdown_pct": 12.3,
        "win_rate": 68.2,
        "total_trades": 150,
        "verified": True
    }
    
    db.upsert_trader(test_trader)
    print("✅ Test trader inserted")
    
    # Get stats
    stats = db.get_stats()
    print(f"\n📊 Database Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Database initialized at: {db.db_path}")

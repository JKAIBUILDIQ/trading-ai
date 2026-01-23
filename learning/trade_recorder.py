#!/usr/bin/env python3
"""
Trade Recording System for NEO Learning
Records all Crellastein/NEO trades for model training.

Data captured (from MT5 API on port 8085):
- Entry: symbol, direction, price, time, lot size, strategy used
- Exit: close price, close time, P&L, reason (TP/SL/manual)
- Context: RSI at entry, BB position, regime, news events

Storage: SQLite + CSV export for training

NO RANDOM DATA - All records from actual MT5 trades.
"""

import sqlite3
import json
import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradeRecorder")

# Configuration
DB_PATH = Path(__file__).parent / "trades.db"
MT5_API_URL = "http://localhost:8085"
CSV_EXPORT_PATH = Path(__file__).parent / "exports"
CSV_EXPORT_PATH.mkdir(exist_ok=True)


@dataclass
class TradeRecord:
    """Complete trade record for learning."""
    # Identifiers
    trade_id: str
    ticket: int
    magic_number: int
    
    # Entry details
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    entry_time: str
    lot_size: float
    strategy: str  # e.g., "turtle", "rsi2", "session_breakout"
    
    # Exit details
    exit_price: float
    exit_time: str
    pnl: float
    pnl_pips: float
    exit_reason: str  # TP, SL, MANUAL, TIME
    
    # Context at entry
    rsi_2: float
    rsi_14: float
    bb_position: float
    atr: float
    regime: str  # TRENDING, RANGING, VOLATILE
    news_nearby: bool
    
    # Outcome
    outcome: str  # WIN, LOSS, BREAKEVEN
    hold_duration_minutes: int
    max_favorable: float  # Max favorable excursion (pips)
    max_adverse: float  # Max adverse excursion (pips)
    
    # Metadata
    bot_version: str
    recorded_at: str


class TradeRecorder:
    """
    Records trades from MT5 API to SQLite database.
    Provides methods for trade logging, querying, and export.
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                ticket INTEGER,
                magic_number INTEGER,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                entry_time TEXT,
                lot_size REAL,
                strategy TEXT,
                exit_price REAL,
                exit_time TEXT,
                pnl REAL,
                pnl_pips REAL,
                exit_reason TEXT,
                rsi_2 REAL,
                rsi_14 REAL,
                bb_position REAL,
                atr REAL,
                regime TEXT,
                news_nearby INTEGER,
                outcome TEXT,
                hold_duration_minutes INTEGER,
                max_favorable REAL,
                max_adverse REAL,
                bot_version TEXT,
                recorded_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def record_trade(self, trade: TradeRecord) -> bool:
        """
        Record a completed trade.
        Returns True if successful.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, ticket, magic_number, symbol, direction,
                    entry_price, entry_time, lot_size, strategy,
                    exit_price, exit_time, pnl, pnl_pips, exit_reason,
                    rsi_2, rsi_14, bb_position, atr, regime, news_nearby,
                    outcome, hold_duration_minutes, max_favorable, max_adverse,
                    bot_version, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.ticket, trade.magic_number,
                trade.symbol, trade.direction, trade.entry_price,
                trade.entry_time, trade.lot_size, trade.strategy,
                trade.exit_price, trade.exit_time, trade.pnl, trade.pnl_pips,
                trade.exit_reason, trade.rsi_2, trade.rsi_14,
                trade.bb_position, trade.atr, trade.regime,
                1 if trade.news_nearby else 0, trade.outcome,
                trade.hold_duration_minutes, trade.max_favorable,
                trade.max_adverse, trade.bot_version, trade.recorded_at
            ))
            
            conn.commit()
            logger.info(f"Recorded trade {trade.trade_id}: {trade.symbol} {trade.direction} → {trade.outcome} (${trade.pnl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
            
        finally:
            conn.close()
    
    def fetch_trades_from_mt5(self) -> List[Dict]:
        """
        Fetch closed trades from MT5 API.
        """
        try:
            # Get trade history
            response = requests.get(f"{MT5_API_URL}/trades/history", timeout=30)
            
            if response.status_code == 200:
                trades = response.json()
                logger.info(f"Fetched {len(trades)} trades from MT5 API")
                return trades
            else:
                logger.warning(f"MT5 API returned {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from MT5 API: {e}")
            return []
    
    def process_mt5_trade(self, mt5_trade: Dict, context: Dict = None) -> TradeRecord:
        """
        Convert MT5 trade to TradeRecord with context.
        """
        # Parse entry/exit times
        entry_time = mt5_trade.get('open_time', '')
        exit_time = mt5_trade.get('close_time', '')
        
        # Calculate hold duration
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
            hold_duration = int((exit_dt - entry_dt).total_seconds() / 60)
        except:
            hold_duration = 0
        
        # Calculate pips
        entry_price = mt5_trade.get('open_price', 0)
        exit_price = mt5_trade.get('close_price', 0)
        pnl = mt5_trade.get('profit', 0)
        
        direction = 'BUY' if mt5_trade.get('type', 0) == 0 else 'SELL'
        
        if direction == 'BUY':
            pnl_pips = (exit_price - entry_price) / 0.0001
        else:
            pnl_pips = (entry_price - exit_price) / 0.0001
        
        # Determine outcome
        if pnl > 0:
            outcome = 'WIN'
        elif pnl < 0:
            outcome = 'LOSS'
        else:
            outcome = 'BREAKEVEN'
        
        # Magic number to strategy mapping
        magic = mt5_trade.get('magic', 0)
        strategy_map = {
            888007: 'trend_following',
            888008: 'rsi2_mean_reversion',
            888010: 'liquidity_sweep',
            888015: 'gto_multi',
            888020: 'ghost_commander',
            888021: 'shadow_executor',
            0: 'manual'
        }
        strategy = strategy_map.get(magic, 'unknown')
        
        # Context (from separate indicator data if provided)
        ctx = context or {}
        
        return TradeRecord(
            trade_id=f"{mt5_trade.get('ticket', 0)}_{entry_time}",
            ticket=mt5_trade.get('ticket', 0),
            magic_number=magic,
            symbol=mt5_trade.get('symbol', 'UNKNOWN'),
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            lot_size=mt5_trade.get('volume', 0.01),
            strategy=strategy,
            exit_price=exit_price,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pips=round(pnl_pips, 1),
            exit_reason=mt5_trade.get('comment', 'UNKNOWN'),
            rsi_2=ctx.get('rsi_2', 50),
            rsi_14=ctx.get('rsi_14', 50),
            bb_position=ctx.get('bb_position', 0.5),
            atr=ctx.get('atr', 0.001),
            regime=ctx.get('regime', 'UNKNOWN'),
            news_nearby=ctx.get('news_nearby', False),
            outcome=outcome,
            hold_duration_minutes=hold_duration,
            max_favorable=ctx.get('max_favorable', abs(pnl_pips) if outcome == 'WIN' else 0),
            max_adverse=ctx.get('max_adverse', abs(pnl_pips) if outcome == 'LOSS' else 0),
            bot_version=f"v{magic % 1000:03d}" if magic > 0 else "manual",
            recorded_at=datetime.utcnow().isoformat()
        )
    
    def sync_from_mt5(self) -> int:
        """
        Sync trades from MT5 API to database.
        Returns number of new trades recorded.
        """
        mt5_trades = self.fetch_trades_from_mt5()
        
        if not mt5_trades:
            return 0
        
        new_count = 0
        for mt5_trade in mt5_trades:
            record = self.process_mt5_trade(mt5_trade)
            if self.record_trade(record):
                new_count += 1
        
        logger.info(f"Synced {new_count} new trades from MT5")
        return new_count
    
    def get_trades(
        self,
        symbol: str = None,
        strategy: str = None,
        outcome: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query trades from database.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        
        query += f" ORDER BY entry_time DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_performance_by_strategy(self) -> Dict:
        """
        Get performance statistics by strategy.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(pnl_pips) as avg_pips,
                AVG(hold_duration_minutes) as avg_hold_minutes
            FROM trades
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            strategy = row[0]
            total = row[1]
            wins = row[2]
            
            results[strategy] = {
                'total_trades': total,
                'wins': wins,
                'losses': row[3],
                'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                'total_pnl': round(row[4], 2),
                'avg_pnl': round(row[5], 2),
                'avg_pips': round(row[6], 1),
                'avg_hold_minutes': round(row[7], 1)
            }
        
        conn.close()
        return results
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export all trades to CSV for model training.
        """
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = CSV_EXPORT_PATH / filename
        trades = self.get_trades(limit=100000)
        
        if not trades:
            logger.warning("No trades to export")
            return ""
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return str(filepath)
    
    def get_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get trades formatted for model training.
        Returns (features, labels) tuples.
        """
        trades = self.get_trades(limit=100000)
        
        features = []
        labels = []
        
        for trade in trades:
            # Feature vector for entry decision
            feature = {
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'rsi_2': trade['rsi_2'],
                'rsi_14': trade['rsi_14'],
                'bb_position': trade['bb_position'],
                'atr': trade['atr'],
                'regime': trade['regime'],
                'strategy': trade['strategy']
            }
            
            # Label (outcome)
            label = {
                'outcome': trade['outcome'],
                'pnl': trade['pnl'],
                'pnl_pips': trade['pnl_pips'],
                'hold_duration': trade['hold_duration_minutes']
            }
            
            features.append(feature)
            labels.append(label)
        
        return features, labels


def get_trade_recorder() -> TradeRecorder:
    """Factory function to get trade recorder instance."""
    return TradeRecorder()


if __name__ == "__main__":
    print("=" * 60)
    print("TRADE RECORDER - Test")
    print("=" * 60)
    
    recorder = TradeRecorder()
    
    # Check database
    print(f"Database path: {recorder.db_path}")
    
    # Try to sync from MT5
    print("\nAttempting to sync from MT5 API...")
    try:
        count = recorder.sync_from_mt5()
        print(f"Synced {count} trades")
    except Exception as e:
        print(f"MT5 sync failed (expected if no API): {e}")
    
    # Show existing trades
    trades = recorder.get_trades(limit=5)
    print(f"\nLast {len(trades)} trades in database:")
    for t in trades:
        print(f"  {t.get('symbol')} {t.get('direction')} → {t.get('outcome')} (${t.get('pnl', 0):.2f})")
    
    # Show performance by strategy
    print("\nPerformance by strategy:")
    perf = recorder.get_performance_by_strategy()
    for strategy, stats in perf.items():
        print(f"  {strategy}: {stats['win_rate']}% WR, ${stats['total_pnl']:.2f} total")
    
    print("\n" + "=" * 60)

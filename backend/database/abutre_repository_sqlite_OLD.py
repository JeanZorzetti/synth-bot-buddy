"""
ABUTRE REPOSITORY

Camada de persistência para eventos do Deriv Bot XML
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "abutre.db"


class AbutreRepository:
    """Repository para eventos do Abutre"""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        """Ensure all tables exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table: abutre_candles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abutre_candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL DEFAULT '1HZ100V',
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                color INTEGER NOT NULL,
                source TEXT DEFAULT 'deriv_bot_xml',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_triggers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abutre_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                streak_count INTEGER NOT NULL,
                direction TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abutre_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                contract_id TEXT,

                -- Entry
                entry_time DATETIME NOT NULL,
                direction TEXT NOT NULL,
                initial_stake REAL NOT NULL,

                -- Progression
                max_level_reached INTEGER NOT NULL,
                total_staked REAL NOT NULL,

                -- Exit
                exit_time DATETIME,
                result TEXT,
                profit REAL,
                balance_after REAL,

                -- Metadata
                source TEXT DEFAULT 'deriv_bot_xml',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_balance_history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abutre_balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                balance REAL NOT NULL,
                peak_balance REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                roi_pct REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Abutre tables ensured")

    # ==================== CANDLES ====================

    def insert_candle(
        self,
        timestamp: datetime,
        symbol: str,
        open: float,
        high: float,
        low: float,
        close: float,
        color: int
    ) -> int:
        """Insert candle event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO abutre_candles (timestamp, symbol, open, high, low, close, color)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, open, high, low, close, color))

        candle_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Inserted candle #{candle_id}: {symbol} @ {close:.2f} (color={color})")
        return candle_id

    def get_recent_candles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent candles"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM abutre_candles
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        candles = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return candles

    # ==================== TRIGGERS ====================

    def insert_trigger(
        self,
        timestamp: datetime,
        streak_count: int,
        direction: str
    ) -> int:
        """Insert trigger event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO abutre_triggers (timestamp, streak_count, direction)
            VALUES (?, ?, ?)
        """, (timestamp, streak_count, direction))

        trigger_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"🚨 Trigger #{trigger_id}: {streak_count} {direction} candles")
        return trigger_id

    # ==================== TRADES ====================

    def insert_trade_opened(
        self,
        trade_id: str,
        timestamp: datetime,
        direction: str,
        stake: float,
        level: int,
        contract_id: Optional[str] = None
    ) -> int:
        """Insert trade opened event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO abutre_trades
            (trade_id, contract_id, entry_time, direction, initial_stake, max_level_reached, total_staked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, contract_id, timestamp, direction, stake, level, stake))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"📈 Trade opened: {trade_id} | {direction} | ${stake:.2f} @ Level {level}")
        return row_id

    def update_trade_closed(
        self,
        trade_id: str,
        exit_time: datetime,
        result: str,
        profit: float,
        balance: float,
        max_level: int
    ) -> bool:
        """Update trade with closed info"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE abutre_trades
            SET exit_time = ?,
                result = ?,
                profit = ?,
                balance_after = ?,
                max_level_reached = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE trade_id = ?
        """, (exit_time, result, profit, balance, max_level, trade_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            emoji = "🎯" if result == "WIN" else "❌"
            logger.info(f"{emoji} Trade closed: {trade_id} | {result} | {profit:+.2f} | Balance: ${balance:.2f}")
        else:
            logger.warning(f"⚠️ Trade {trade_id} not found for update")

        return success

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM abutre_trades
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))

        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return trades

    def get_trade_stats(self) -> Dict[str, Any]:
        """Get aggregated trade statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result IN ('LOSS', 'STOP_LOSS') THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(profit), 0) as total_profit,
                COALESCE(AVG(CASE WHEN result = 'WIN' THEN profit END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN result IN ('LOSS', 'STOP_LOSS') THEN profit END), 0) as avg_loss,
                MAX(max_level_reached) as max_level_used
            FROM abutre_trades
            WHERE result IS NOT NULL
        """)

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate_pct': 0.0,
                'total_profit': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_level_used': 0
            }

        stats = dict(row)
        stats['win_rate_pct'] = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0.0

        return stats

    # ==================== BALANCE ====================

    def insert_balance_snapshot(
        self,
        timestamp: datetime,
        balance: float,
        peak_balance: float,
        drawdown_pct: float,
        total_trades: int,
        wins: int,
        losses: int,
        roi_pct: float
    ) -> int:
        """Insert balance snapshot"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO abutre_balance_history
            (timestamp, balance, peak_balance, drawdown_pct, total_trades, wins, losses, roi_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, balance, peak_balance, drawdown_pct, total_trades, wins, losses, roi_pct))

        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Balance snapshot #{snapshot_id}: ${balance:.2f} (ROI: {roi_pct:.2f}%)")
        return snapshot_id

    def get_balance_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get balance history for equity curve"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM abutre_balance_history
            ORDER BY timestamp ASC
            LIMIT ?
        """, (limit,))

        history = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return history

    def get_latest_balance(self) -> Optional[float]:
        """Get latest balance"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT balance FROM abutre_balance_history
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        return row['balance'] if row else None


# Singleton instance
_repository: Optional[AbutreRepository] = None


def get_abutre_repository() -> AbutreRepository:
    """Get singleton repository instance"""
    global _repository
    if _repository is None:
        _repository = AbutreRepository()
    return _repository

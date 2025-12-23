"""
ABUTRE REPOSITORY - PostgreSQL Version

Camada de persistência para eventos do Deriv Bot XML usando PostgreSQL
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://botderiv:PAzo18**@dados_botderiv:5432/botderiv")


class AbutreRepositoryPostgres:
    """Repository para eventos do Abutre usando PostgreSQL"""

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self._ensure_tables()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def _ensure_tables(self):
        """Ensure all tables exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Table: abutre_candles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abutre_candles (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    symbol TEXT NOT NULL DEFAULT '1HZ100V',
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    color INTEGER NOT NULL,
                    source TEXT DEFAULT 'deriv_bot_xml',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table: abutre_triggers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abutre_triggers (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    streak_count INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    source TEXT DEFAULT 'deriv_bot_xml',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table: abutre_trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abutre_trades (
                    id SERIAL PRIMARY KEY,
                    trade_id TEXT UNIQUE NOT NULL,
                    contract_id TEXT,
                    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    direction TEXT NOT NULL,
                    initial_stake REAL NOT NULL,
                    max_level_reached INTEGER DEFAULT 1,
                    total_staked REAL NOT NULL,
                    exit_time TIMESTAMP WITH TIME ZONE,
                    result TEXT,
                    profit REAL,
                    balance_after REAL,
                    source TEXT DEFAULT 'deriv_bot_xml',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table: abutre_balance_history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abutre_balance_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    balance REAL NOT NULL,
                    peak_balance REAL,
                    drawdown_pct REAL,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    roi_pct REAL DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abutre_candles_timestamp ON abutre_candles(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abutre_triggers_timestamp ON abutre_triggers(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abutre_trades_entry_time ON abutre_trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abutre_trades_trade_id ON abutre_trades(trade_id)")

            conn.commit()
            logger.info("PostgreSQL tables created successfully")

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    # Insert methods (same logic as SQLite version but with PostgreSQL syntax)

    def insert_candle(
        self,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        color: int,
        symbol: str = '1HZ100V',
        source: str = 'deriv_bot_xml'
    ) -> int:
        """Insert candle event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO abutre_candles (timestamp, symbol, open, high, low, close, color, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                color,
                source
            ))

            row = cursor.fetchone()
            conn.commit()
            return row['id']

        except Exception as e:
            logger.error(f"Error inserting candle: {e}")
            conn.rollback()
            return -1
        finally:
            cursor.close()
            conn.close()

    def insert_trigger(
        self,
        timestamp: datetime,
        streak_count: int,
        direction: str,
        source: str = 'deriv_bot_xml'
    ) -> int:
        """Insert trigger event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO abutre_triggers (timestamp, streak_count, direction, source)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                timestamp,
                streak_count,
                direction,
                source
            ))

            row = cursor.fetchone()
            conn.commit()
            return row['id']

        except Exception as e:
            logger.error(f"Error inserting trigger: {e}")
            conn.rollback()
            return -1
        finally:
            cursor.close()
            conn.close()

    def insert_trade_opened(
        self,
        trade_id: str,
        timestamp: datetime,
        direction: str,
        stake: float,
        level: int = 1,
        contract_id: Optional[str] = None,
        source: str = 'deriv_bot_xml'
    ) -> int:
        """Insert trade opened event"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO abutre_trades
                (trade_id, contract_id, entry_time, direction, initial_stake, total_staked, max_level_reached, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_id) DO NOTHING
                RETURNING id
            """, (
                trade_id,
                contract_id,
                timestamp,
                direction,
                stake,
                stake,
                level,
                source
            ))

            row = cursor.fetchone()
            conn.commit()
            return row['id'] if row else -1

        except Exception as e:
            logger.error(f"Error inserting trade: {e}")
            conn.rollback()
            return -1
        finally:
            cursor.close()
            conn.close()

    def update_trade_closed(
        self,
        trade_id: str,
        exit_time: datetime,
        result: str,
        profit: float,
        balance: float,
        max_level: int = 1
    ) -> bool:
        """Update trade with closed information"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE abutre_trades
                SET exit_time = %s,
                    result = %s,
                    profit = %s,
                    balance_after = %s,
                    max_level_reached = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = %s
            """, (
                exit_time,
                result,
                profit,
                balance,
                max_level,
                trade_id
            ))

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()

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
        """Insert balance history snapshot"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO abutre_balance_history
                (timestamp, balance, peak_balance, drawdown_pct, total_trades, wins, losses, roi_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                timestamp,
                balance,
                peak_balance,
                drawdown_pct,
                total_trades,
                wins,
                losses,
                roi_pct
            ))

            row = cursor.fetchone()
            conn.commit()
            return row['id'] if row else -1

        except Exception as e:
            logger.error(f"Error inserting balance snapshot: {e}")
            conn.rollback()
            return -1
        finally:
            cursor.close()
            conn.close()

    def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM abutre_trades
                ORDER BY entry_time DESC
                LIMIT %s
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            cursor.close()
            conn.close()

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Alias for get_trades() - for compatibility with API endpoints"""
        return self.get_trades(limit=limit)

    def get_trades_by_period(self, date_from: datetime, date_to: datetime, limit: int = 1000) -> List[Dict]:
        """Get trades within a specific date range"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM abutre_trades
                WHERE entry_time >= %s AND entry_time <= %s
                ORDER BY entry_time DESC
                LIMIT %s
            """, (date_from, date_to, limit))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            cursor.close()
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(profit), 0) as total_profit,
                    COALESCE(AVG(CASE WHEN result = 'WIN' THEN profit END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN result = 'LOSS' THEN profit END), 0) as avg_loss,
                    MAX(max_level_reached) as max_level_used
                FROM abutre_trades
                WHERE result IS NOT NULL
            """)

            row = cursor.fetchone()
            stats = dict(row)

            # Calculate win rate
            total = stats['total_trades']
            wins = stats['wins'] or 0
            stats['win_rate_pct'] = (wins / total * 100) if total > 0 else 0.0

            # Get current balance
            cursor.execute("""
                SELECT balance_after FROM abutre_trades
                WHERE balance_after IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 1
            """)

            balance_row = cursor.fetchone()
            stats['current_balance'] = balance_row['balance_after'] if balance_row else 10000.0

            # Calculate ROI
            initial_balance = 10000.0
            stats['roi_pct'] = ((stats['current_balance'] - initial_balance) / initial_balance * 100)

            return stats

        finally:
            cursor.close()
            conn.close()

    def get_trade_stats(self) -> Dict[str, Any]:
        """Alias for get_stats() - for compatibility with API endpoints"""
        return self.get_stats()

    def get_latest_balance(self) -> Optional[float]:
        """Get latest balance from balance history"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT balance FROM abutre_balance_history
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            return row['balance'] if row else None

        finally:
            cursor.close()
            conn.close()

    def get_balance_history(self, limit: int = 1000) -> List[Dict]:
        """Get balance history snapshots"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM abutre_balance_history
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            cursor.close()
            conn.close()


# Singleton instance
_repository: Optional[AbutreRepositoryPostgres] = None


def get_abutre_repository() -> AbutreRepositoryPostgres:
    """Get singleton repository instance"""
    global _repository
    if _repository is None:
        _repository = AbutreRepositoryPostgres()
    return _repository

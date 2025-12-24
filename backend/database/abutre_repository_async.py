"""
ABUTRE REPOSITORY - Async Version with Connection Pooling

Performance improvements:
- asyncpg (async driver) instead of psycopg2 (sync)
- Connection pooling (reuse connections)
- No event loop blocking
"""
import asyncpg
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class AbutreRepositoryAsync:
    """Async PostgreSQL repository for Abutre trades with connection pooling"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool (call once on startup)"""
        if not self._pool:
            logger.info("Creating asyncpg connection pool...")
            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("✅ Connection pool created successfully")

    async def close(self):
        """Close connection pool (call on shutdown)"""
        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed")

    async def save_trade(self, trade: Dict[str, Any]) -> bool:
        """Save trade to database (INSERT IGNORE on conflict)"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO abutre_trades (
                        trade_id, contract_id, entry_time, exit_time,
                        direction, initial_stake, result, profit,
                        balance_after, max_level_reached
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (trade_id) DO NOTHING
                """,
                    trade.get("trade_id"),
                    trade.get("contract_id"),
                    trade.get("entry_time"),
                    trade.get("exit_time"),
                    trade.get("direction"),
                    trade.get("initial_stake"),
                    trade.get("result"),
                    trade.get("profit"),
                    trade.get("balance_after"),
                    trade.get("max_level_reached")
                )
                return True
            except Exception as e:
                logger.error(f"Error saving trade: {e}")
                return False

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing trade"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                set_clauses = []
                values = []
                param_num = 1

                for key, value in updates.items():
                    set_clauses.append(f"{key} = ${param_num}")
                    values.append(value)
                    param_num += 1

                values.append(trade_id)
                query = f"UPDATE abutre_trades SET {', '.join(set_clauses)} WHERE trade_id = ${param_num}"

                await conn.execute(query, *values)
                return True
            except Exception as e:
                logger.error(f"Error updating trade: {e}")
                return False

    async def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades (ordered by entry_time DESC)"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM abutre_trades
                ORDER BY entry_time DESC
                LIMIT $1
            """, limit)

            return [dict(row) for row in rows]

    async def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Alias for get_trades - for compatibility"""
        return await self.get_trades(limit=limit)

    async def get_trades_by_period(
        self,
        date_from: datetime,
        date_to: datetime,
        limit: int = 10000
    ) -> List[Dict]:
        """Get trades within a specific date range"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM abutre_trades
                WHERE entry_time >= $1 AND entry_time <= $2
                ORDER BY entry_time DESC
                LIMIT $3
            """, date_from, date_to, limit)

            return [dict(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            stats = {}

            # Total trades
            total_trades = await conn.fetchval("SELECT COUNT(*) FROM abutre_trades")
            stats['total_trades'] = total_trades or 0

            # Wins and losses
            wins = await conn.fetchval("SELECT COUNT(*) FROM abutre_trades WHERE result = 'WIN'")
            losses = await conn.fetchval("SELECT COUNT(*) FROM abutre_trades WHERE result = 'LOSS'")

            stats['wins'] = wins or 0
            stats['losses'] = losses or 0
            stats['win_rate'] = (wins / total_trades * 100) if total_trades > 0 else 0

            # Total profit
            total_profit = await conn.fetchval("SELECT SUM(profit) FROM abutre_trades WHERE profit IS NOT NULL")
            stats['total_profit'] = float(total_profit) if total_profit else 0.0

            # Current balance (last trade)
            current_balance = await conn.fetchval("""
                SELECT balance_after FROM abutre_trades
                WHERE balance_after IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 1
            """)
            stats['current_balance'] = float(current_balance) if current_balance else 10000.0

            # ROI
            initial_balance = 10000.0
            stats['roi_pct'] = ((stats['current_balance'] - initial_balance) / initial_balance * 100)

            return stats

    async def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of the most recent trade in DB"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT MAX(entry_time) FROM abutre_trades
            """)
            return result

    async def insert_candle(
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
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval("""
                    INSERT INTO abutre_candles (timestamp, symbol, open, high, low, close, color, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, timestamp, symbol, open, high, low, close, color, source)
                return result
            except Exception as e:
                logger.error(f"Error inserting candle: {e}")
                return -1

    async def insert_trigger(
        self,
        timestamp: datetime,
        streak_count: int,
        direction: str,
        source: str = 'deriv_bot_xml'
    ) -> int:
        """Insert trigger event"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval("""
                    INSERT INTO abutre_triggers (timestamp, streak_count, direction, source)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """, timestamp, streak_count, direction, source)
                return result
            except Exception as e:
                logger.error(f"Error inserting trigger: {e}")
                return -1

    async def insert_trade_opened(
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
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval("""
                    INSERT INTO abutre_trades
                    (trade_id, contract_id, entry_time, direction, initial_stake, total_staked, max_level_reached, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (trade_id) DO NOTHING
                    RETURNING id
                """, trade_id, contract_id, timestamp, direction, stake, stake, level, source)
                return result if result else -1
            except Exception as e:
                logger.error(f"Error inserting trade: {e}")
                return -1

    async def update_trade_closed(
        self,
        trade_id: str,
        exit_time: datetime,
        result: str,
        profit: float,
        balance: float,
        max_level: int = 1
    ) -> bool:
        """Update trade with closed information"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                await conn.execute("""
                    UPDATE abutre_trades
                    SET exit_time = $1,
                        result = $2,
                        profit = $3,
                        balance_after = $4,
                        max_level_reached = $5,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = $6
                """, exit_time, result, profit, balance, max_level, trade_id)
                return True
            except Exception as e:
                logger.error(f"Error updating trade: {e}")
                return False

    async def get_trade_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
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

            stats = dict(row)

            # Calculate win rate
            total = stats['total_trades']
            wins = stats['wins'] or 0
            stats['win_rate_pct'] = (wins / total * 100) if total > 0 else 0.0

            # Get current balance
            balance_row = await conn.fetchrow("""
                SELECT balance_after FROM abutre_trades
                WHERE balance_after IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 1
            """)

            stats['current_balance'] = balance_row['balance_after'] if balance_row else 10000.0

            # Calculate ROI
            initial_balance = 10000.0
            stats['roi_pct'] = ((stats['current_balance'] - initial_balance) / initial_balance * 100)

            return stats

    async def get_latest_balance(self) -> Optional[float]:
        """Get latest balance from balance history"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT balance FROM abutre_balance_history
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            return result

    async def insert_balance_snapshot(
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
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval("""
                    INSERT INTO abutre_balance_history
                    (timestamp, balance, peak_balance, drawdown_pct, total_trades, wins, losses, roi_pct)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, timestamp, balance, peak_balance, drawdown_pct, total_trades, wins, losses, roi_pct)
                return result if result else -1
            except Exception as e:
                logger.error(f"Error inserting balance snapshot: {e}")
                return -1

    async def get_balance_history(self, limit: int = 1000) -> List[Dict]:
        """Get balance history snapshots"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM abutre_balance_history
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)

            return [dict(row) for row in rows]


# Singleton instance
_repository_instance: Optional[AbutreRepositoryAsync] = None


async def get_async_repository() -> AbutreRepositoryAsync:
    """Get singleton async repository instance"""
    global _repository_instance

    if _repository_instance is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not configured!")

        _repository_instance = AbutreRepositoryAsync(db_url)
        await _repository_instance.connect()

    return _repository_instance

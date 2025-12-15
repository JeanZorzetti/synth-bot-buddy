"""
Trade History Manager - FASE 7
Gerencia o histórico de trades usando SQLite
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class TradesHistoryManager:
    """Manager for trades history database"""

    def __init__(self, db_path: str = "trades_history.db"):
        """Initialize the trades history manager"""
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database and table exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL CHECK(trade_type IN ('BUY', 'SELL', 'CALL', 'PUT')),
                entry_price REAL NOT NULL,
                exit_price REAL,
                stake REAL NOT NULL,
                profit_loss REAL,
                result TEXT CHECK(result IN ('win', 'loss', 'pending')),
                confidence REAL CHECK(confidence >= 0 AND confidence <= 100),
                strategy TEXT CHECK(strategy IN ('ml', 'technical', 'hybrid', 'order_flow')),
                indicators_used TEXT,
                ml_prediction REAL,
                order_flow_signal TEXT,
                stop_loss REAL,
                take_profit REAL,
                exit_reason TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Add a new trade to the history

        Args:
            trade_data: Dictionary with trade information

        Returns:
            ID of the inserted trade
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert indicators_used dict to JSON string if needed
        if 'indicators_used' in trade_data and isinstance(trade_data['indicators_used'], dict):
            trade_data['indicators_used'] = json.dumps(trade_data['indicators_used'])

        cursor.execute("""
            INSERT INTO trades_history (
                symbol, trade_type, entry_price, exit_price, stake, profit_loss,
                result, confidence, strategy, indicators_used, ml_prediction,
                order_flow_signal, stop_loss, take_profit, exit_reason, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get('symbol'),
            trade_data.get('trade_type'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('stake'),
            trade_data.get('profit_loss'),
            trade_data.get('result', 'pending'),
            trade_data.get('confidence'),
            trade_data.get('strategy'),
            trade_data.get('indicators_used'),
            trade_data.get('ml_prediction'),
            trade_data.get('order_flow_signal'),
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('exit_reason'),
            trade_data.get('notes')
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return trade_id

    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade

        Args:
            trade_id: ID of the trade to update
            update_data: Dictionary with fields to update

        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build SET clause dynamically
        set_clauses = []
        values = []

        for key, value in update_data.items():
            if key == 'indicators_used' and isinstance(value, dict):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)

        # Add updated_at
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())

        # Add trade_id for WHERE clause
        values.append(trade_id)

        query = f"""
            UPDATE trades_history
            SET {', '.join(set_clauses)}
            WHERE id = ?
        """

        cursor.execute(query, values)
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific trade by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades_history WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            trade = dict(row)
            # Parse JSON fields
            if trade.get('indicators_used'):
                try:
                    trade['indicators_used'] = json.loads(trade['indicators_used'])
                except:
                    pass
            return trade
        return None

    def get_trades(
        self,
        page: int = 1,
        limit: int = 25,
        symbol: Optional[str] = None,
        trade_type: Optional[str] = None,
        result: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy: Optional[str] = None,
        sort_by: str = 'timestamp',
        sort_order: str = 'DESC'
    ) -> Dict[str, Any]:
        """
        Get trades with pagination and filters

        Returns:
            Dictionary with trades list and pagination info
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build WHERE clause
        where_clauses = []
        params = []

        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol)

        if trade_type:
            where_clauses.append("trade_type = ?")
            params.append(trade_type)

        if result:
            where_clauses.append("result = ?")
            params.append(result)

        if strategy:
            where_clauses.append("strategy = ?")
            params.append(strategy)

        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM trades_history {where_sql}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()['total']

        # Get paginated results
        offset = (page - 1) * limit

        # Validate sort_by to prevent SQL injection
        allowed_sort_fields = ['timestamp', 'symbol', 'profit_loss', 'confidence', 'created_at']
        if sort_by not in allowed_sort_fields:
            sort_by = 'timestamp'

        if sort_order.upper() not in ['ASC', 'DESC']:
            sort_order = 'DESC'

        query = f"""
            SELECT * FROM trades_history
            {where_sql}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """

        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        trades = []
        for row in rows:
            trade = dict(row)
            # Parse JSON fields
            if trade.get('indicators_used'):
                try:
                    trade['indicators_used'] = json.loads(trade['indicators_used'])
                except:
                    pass
            trades.append(trade)

        conn.close()

        total_pages = (total + limit - 1) // limit  # Ceiling division

        return {
            'trades': trades,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }

    def get_trade_stats(self) -> Dict[str, Any]:
        """Get overall trading statistics"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss) as avg_pnl,
                MAX(profit_loss) as max_profit,
                MIN(profit_loss) as max_loss,
                AVG(confidence) as avg_confidence
            FROM trades_history
        """)

        overall = dict(cursor.fetchone())

        # Calculate win rate
        total_completed = (overall['wins'] or 0) + (overall['losses'] or 0)
        overall['win_rate'] = (overall['wins'] / total_completed * 100) if total_completed > 0 else 0

        # Stats by symbol
        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as trades,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as pnl
            FROM trades_history
            GROUP BY symbol
            ORDER BY trades DESC
        """)

        by_symbol = [dict(row) for row in cursor.fetchall()]

        # Stats by strategy
        cursor.execute("""
            SELECT
                strategy,
                COUNT(*) as trades,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as pnl,
                AVG(confidence) as avg_confidence
            FROM trades_history
            WHERE strategy IS NOT NULL
            GROUP BY strategy
            ORDER BY trades DESC
        """)

        by_strategy = [dict(row) for row in cursor.fetchall()]

        # Recent performance (last 7 days)
        cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as trades,
                SUM(profit_loss) as pnl
            FROM trades_history
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)

        recent_performance = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            'overall': overall,
            'by_symbol': by_symbol,
            'by_strategy': by_strategy,
            'recent_performance': recent_performance
        }

    def delete_trade(self, trade_id: int) -> bool:
        """Delete a trade by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM trades_history WHERE id = ?", (trade_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def clear_all_trades(self) -> bool:
        """Clear all trades (use with caution!)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM trades_history")

        conn.commit()
        conn.close()

        return True


# Singleton instance
_trades_manager = None

def get_trades_manager() -> TradesHistoryManager:
    """Get or create the trades history manager singleton"""
    global _trades_manager
    if _trades_manager is None:
        _trades_manager = TradesHistoryManager()
    return _trades_manager

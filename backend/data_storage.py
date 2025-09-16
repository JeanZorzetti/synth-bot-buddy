"""
🤖 AI Trading Bot - Data Storage System
Time-series database for tick data storage and retrieval

Author: Claude Code
Created: 2025-01-16
"""

import asyncio
import sqlite3
import json
import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import asdict
from contextlib import asynccontextmanager
import aiosqlite
import threading
from queue import Queue
import pickle
import gzip

from .tick_data_collector import TickData, TickSequence

logger = logging.getLogger(__name__)

class TimeSeriesDB:
    """High-performance time-series database for tick data"""

    def __init__(self, db_path: str = "data/ticks.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        self.connection_pool = {}
        self.batch_queue = Queue(maxsize=10000)
        self.batch_size = 1000
        self.batch_timeout = 5.0  # seconds

        # Background batch processor
        self.batch_processor_running = False
        self.batch_thread = None

        # Compression settings
        self.compress_older_than_days = 7
        self.compression_level = 6

        logger.info(f"TimeSeriesDB initialized with path: {db_path}")

    def ensure_db_directory(self):
        """Ensure database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    async def initialize(self) -> bool:
        """Initialize database schema"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Main ticks table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS ticks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        epoch INTEGER,
                        ask REAL,
                        bid REAL,
                        spread REAL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Indexes for performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
                    ON ticks(symbol, timestamp)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON ticks(timestamp)
                """)

                # Compressed historical data table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS ticks_compressed (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        data BLOB NOT NULL,
                        tick_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Training datasets table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS training_datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe_hours INTEGER,
                        features_count INTEGER,
                        accuracy REAL,
                        data BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Performance metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS storage_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        timestamp REAL,
                        metadata TEXT
                    )
                """)

                await db.commit()

            # Start batch processor
            self.start_batch_processor()

            logger.info("Database initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False

    def start_batch_processor(self):
        """Start background batch processor"""
        if not self.batch_processor_running:
            self.batch_processor_running = True
            self.batch_thread = threading.Thread(target=self._batch_processor_worker, daemon=True)
            self.batch_thread.start()
            logger.info("Batch processor started")

    def stop_batch_processor(self):
        """Stop background batch processor"""
        self.batch_processor_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=10)
        logger.info("Batch processor stopped")

    def _batch_processor_worker(self):
        """Background worker for batch processing"""
        batch = []
        last_flush = time.time()

        while self.batch_processor_running:
            try:
                # Try to get item from queue
                try:
                    item = self.batch_queue.get(timeout=1.0)
                    batch.append(item)
                except:
                    pass  # Timeout, continue

                # Flush batch if size or time threshold reached
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (current_time - last_flush) >= self.batch_timeout)
                )

                if should_flush and batch:
                    asyncio.run(self._flush_batch(batch))
                    batch.clear()
                    last_flush = current_time

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    async def _flush_batch(self, batch: List[TickData]):
        """Flush batch of ticks to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                tick_data = []
                for tick in batch:
                    tick_data.append((
                        tick.symbol,
                        tick.price,
                        tick.timestamp,
                        tick.epoch,
                        tick.ask,
                        tick.bid,
                        tick.spread,
                        tick.volume
                    ))

                await db.executemany("""
                    INSERT INTO ticks (symbol, price, timestamp, epoch, ask, bid, spread, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, tick_data)

                await db.commit()

            logger.debug(f"Flushed batch of {len(batch)} ticks to database")

        except Exception as e:
            logger.error(f"Error flushing batch: {e}")

    def store_tick_async(self, tick: TickData) -> bool:
        """Store tick asynchronously via batch queue"""
        try:
            self.batch_queue.put_nowait(tick)
            return True
        except:
            logger.warning("Batch queue full, dropping tick")
            return False

    async def store_tick(self, tick: TickData) -> bool:
        """Store single tick immediately"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO ticks (symbol, price, timestamp, epoch, ask, bid, spread, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tick.symbol,
                    tick.price,
                    tick.timestamp,
                    tick.epoch,
                    tick.ask,
                    tick.bid,
                    tick.spread,
                    tick.volume
                ))
                await db.commit()
            return True

        except Exception as e:
            logger.error(f"Error storing tick: {e}")
            return False

    async def get_ticks(self,
                       symbol: str,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       limit: Optional[int] = None) -> List[TickData]:
        """Retrieve ticks with optional filtering"""
        try:
            query = "SELECT symbol, price, timestamp, epoch, ask, bid, spread, volume FROM ticks WHERE symbol = ?"
            params = [symbol]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

            # Convert to TickData objects
            ticks = []
            for row in rows:
                tick = TickData(
                    symbol=row[0],
                    price=row[1],
                    timestamp=row[2],
                    epoch=row[3],
                    ask=row[4],
                    bid=row[5],
                    spread=row[6],
                    volume=row[7]
                )
                ticks.append(tick)

            return ticks

        except Exception as e:
            logger.error(f"Error retrieving ticks: {e}")
            return []

    async def get_tick_statistics(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical analysis of tick data"""
        try:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

            async with aiosqlite.connect(self.db_path) as db:
                # Basic statistics
                async with db.execute("""
                    SELECT
                        COUNT(*) as count,
                        MIN(price) as min_price,
                        MAX(price) as max_price,
                        AVG(price) as avg_price,
                        MIN(timestamp) as first_tick,
                        MAX(timestamp) as last_tick
                    FROM ticks
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                """, (symbol, start_time, end_time)) as cursor:
                    stats_row = await cursor.fetchone()

                if not stats_row or stats_row[0] == 0:
                    return {'error': 'No data available'}

                # Get prices for volatility calculation
                async with db.execute("""
                    SELECT price FROM ticks
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (symbol, start_time, end_time)) as cursor:
                    price_rows = await cursor.fetchall()

                prices = [row[0] for row in price_rows]

                # Calculate additional metrics
                volatility = float(np.std(prices)) if len(prices) > 1 else 0.0
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

                return {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'tick_count': stats_row[0],
                    'min_price': stats_row[1],
                    'max_price': stats_row[2],
                    'avg_price': stats_row[3],
                    'volatility': volatility,
                    'price_range': stats_row[2] - stats_row[1],
                    'first_tick': stats_row[4],
                    'last_tick': stats_row[5],
                    'duration_seconds': stats_row[5] - stats_row[4],
                    'avg_price_change': float(np.mean(price_changes)) if price_changes else 0.0,
                    'ticks_per_minute': stats_row[0] / (hours * 60) if hours > 0 else 0
                }

        except Exception as e:
            logger.error(f"Error getting tick statistics: {e}")
            return {'error': str(e)}

    async def compress_old_data(self, days_old: int = 7) -> Dict[str, Any]:
        """Compress old tick data to save space"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 3600)
            compression_stats = {'compressed_symbols': 0, 'ticks_compressed': 0, 'space_saved': 0}

            async with aiosqlite.connect(self.db_path) as db:
                # Get distinct symbols with old data
                async with db.execute("""
                    SELECT DISTINCT symbol FROM ticks WHERE timestamp < ?
                """, (cutoff_time,)) as cursor:
                    symbols = [row[0] for row in await cursor.fetchall()]

                for symbol in symbols:
                    # Get old ticks for this symbol
                    async with db.execute("""
                        SELECT symbol, price, timestamp, epoch, ask, bid, spread, volume
                        FROM ticks
                        WHERE symbol = ? AND timestamp < ?
                        ORDER BY timestamp
                    """, (symbol, cutoff_time)) as cursor:
                        old_ticks = await cursor.fetchall()

                    if not old_ticks:
                        continue

                    # Group by date
                    ticks_by_date = {}
                    for tick in old_ticks:
                        date_key = datetime.fromtimestamp(tick[2]).strftime('%Y-%m-%d')
                        if date_key not in ticks_by_date:
                            ticks_by_date[date_key] = []
                        ticks_by_date[date_key].append(tick)

                    # Compress each date
                    for date_key, date_ticks in ticks_by_date.items():
                        # Create compressed data
                        compressed_data = gzip.compress(
                            pickle.dumps(date_ticks),
                            compresslevel=self.compression_level
                        )

                        # Store compressed data
                        await db.execute("""
                            INSERT INTO ticks_compressed (symbol, date, data, tick_count)
                            VALUES (?, ?, ?, ?)
                        """, (symbol, date_key, compressed_data, len(date_ticks)))

                        compression_stats['ticks_compressed'] += len(date_ticks)

                    compression_stats['compressed_symbols'] += 1

                # Remove old uncompressed data
                result = await db.execute("DELETE FROM ticks WHERE timestamp < ?", (cutoff_time,))
                await db.commit()

                # Calculate space saved (approximate)
                compression_stats['space_saved'] = compression_stats['ticks_compressed'] * 50  # Approx bytes per tick

            logger.info(f"Compressed {compression_stats['ticks_compressed']} ticks for {compression_stats['compressed_symbols']} symbols")
            return compression_stats

        except Exception as e:
            logger.error(f"Error compressing old data: {e}")
            return {'error': str(e)}

    async def save_training_dataset(self, symbol: str, dataset: Dict[str, Any]) -> bool:
        """Save prepared training dataset"""
        try:
            # Serialize dataset
            dataset_blob = gzip.compress(pickle.dumps(dataset), compresslevel=6)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO training_datasets (symbol, timeframe_hours, features_count, accuracy, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol,
                    dataset.get('timeframe_hours', 0),
                    dataset.get('total_samples', 0),
                    dataset.get('accuracy', 0.0),
                    dataset_blob
                ))
                await db.commit()

            logger.info(f"Saved training dataset for {symbol} with {dataset.get('total_samples', 0)} samples")
            return True

        except Exception as e:
            logger.error(f"Error saving training dataset: {e}")
            return False

    async def load_training_dataset(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load latest training dataset for symbol"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT data FROM training_datasets
                    WHERE symbol = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (symbol,)) as cursor:
                    row = await cursor.fetchone()

            if not row:
                return None

            # Deserialize dataset
            dataset = pickle.loads(gzip.decompress(row[0]))
            return dataset

        except Exception as e:
            logger.error(f"Error loading training dataset: {e}")
            return None

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Basic table stats
                async with db.execute("SELECT COUNT(*) FROM ticks") as cursor:
                    total_ticks = (await cursor.fetchone())[0]

                async with db.execute("SELECT COUNT(DISTINCT symbol) FROM ticks") as cursor:
                    total_symbols = (await cursor.fetchone())[0]

                async with db.execute("SELECT COUNT(*) FROM ticks_compressed") as cursor:
                    compressed_records = (await cursor.fetchone())[0]

                async with db.execute("SELECT COUNT(*) FROM training_datasets") as cursor:
                    training_datasets = (await cursor.fetchone())[0]

                # Time range
                async with db.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ticks") as cursor:
                    time_range = await cursor.fetchone()

                # Database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                return {
                    'total_ticks': total_ticks,
                    'total_symbols': total_symbols,
                    'compressed_records': compressed_records,
                    'training_datasets': training_datasets,
                    'earliest_tick': time_range[0],
                    'latest_tick': time_range[1],
                    'timespan_hours': (time_range[1] - time_range[0]) / 3600 if time_range[0] else 0,
                    'database_size_mb': db_size / (1024 * 1024),
                    'batch_queue_size': self.batch_queue.qsize(),
                    'batch_processor_running': self.batch_processor_running
                }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}

    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up very old data to save space"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)

            async with aiosqlite.connect(self.db_path) as db:
                # Count records to be deleted
                async with db.execute("SELECT COUNT(*) FROM ticks WHERE timestamp < ?", (cutoff_time,)) as cursor:
                    ticks_to_delete = (await cursor.fetchone())[0]

                async with db.execute("SELECT COUNT(*) FROM ticks_compressed WHERE date < ?",
                                    (datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d'),)) as cursor:
                    compressed_to_delete = (await cursor.fetchone())[0]

                # Delete old data
                await db.execute("DELETE FROM ticks WHERE timestamp < ?", (cutoff_time,))
                await db.execute("DELETE FROM ticks_compressed WHERE date < ?",
                               (datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d'),))
                await db.execute("DELETE FROM training_datasets WHERE created_at < datetime('now', '-{} days')".format(days_to_keep))

                await db.commit()

                # Vacuum to reclaim space
                await db.execute("VACUUM")

            logger.info(f"Cleaned up {ticks_to_delete} ticks and {compressed_to_delete} compressed records")

            return {
                'ticks_deleted': ticks_to_delete,
                'compressed_deleted': compressed_to_delete,
                'cutoff_date': datetime.fromtimestamp(cutoff_time).isoformat()
            }

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'error': str(e)}

    async def export_data(self, symbol: str, format: str = 'csv', hours: int = 24) -> Optional[str]:
        """Export tick data to file"""
        try:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

            ticks = await self.get_ticks(symbol, start_time, end_time)

            if not ticks:
                return None

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format.lower() == 'csv':
                filename = f"data/export_{symbol}_{timestamp}.csv"

                # Convert to DataFrame
                data = []
                for tick in ticks:
                    data.append(asdict(tick))

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

                return filename

            elif format.lower() == 'json':
                filename = f"data/export_{symbol}_{timestamp}.json"

                data = [asdict(tick) for tick in ticks]
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)

                return filename

            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_batch_processor()
        except:
            pass
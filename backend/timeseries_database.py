"""
🗄️ TIME-SERIES DATABASE INTEGRATION
InfluxDB integration for high-frequency tick data storage and retrieval
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import pandas as pd
from influxdb_client import InfluxDBClient, Point, QueryOptions
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import numpy as np
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import gzip
import os

from real_tick_processor import ProcessedTickData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetentionPolicy(Enum):
    """Políticas de retenção de dados"""
    REALTIME = "1h"      # 1 hour - real-time data
    HOURLY = "7d"        # 7 days - hourly aggregates
    DAILY = "90d"        # 90 days - daily aggregates
    MONTHLY = "2y"       # 2 years - monthly aggregates
    YEARLY = "10y"       # 10 years - yearly aggregates


@dataclass
class DatabaseConfig:
    """Configuração do banco de dados time-series"""
    url: str = "http://localhost:8086"
    token: str = "your-influxdb-token"
    org: str = "trading-org"
    bucket_realtime: str = "trading-realtime"
    bucket_historical: str = "trading-historical"
    bucket_analytics: str = "trading-analytics"
    timeout: int = 30000  # 30 seconds
    batch_size: int = 5000
    flush_interval: int = 1000  # 1 second
    enable_compression: bool = True
    enable_ssl: bool = False


@dataclass
class QueryParams:
    """Parâmetros para consultas"""
    start_time: datetime
    end_time: datetime
    symbols: List[str] = field(default_factory=list)
    aggregation: Optional[str] = None  # mean, max, min, first, last
    window: Optional[str] = None       # 1m, 5m, 1h, 1d
    limit: Optional[int] = None


class TimeSeriesDatabase:
    """Integração com InfluxDB para dados de time-series"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self.write_api = None
        self.query_api = None
        self.delete_api = None

        # Performance tracking
        self.write_count = 0
        self.write_errors = 0
        self.query_count = 0
        self.query_errors = 0

        # Thread pool for heavy operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Batch writing
        self.batch_buffer = []
        self.batch_lock = threading.Lock()
        self.last_flush = time.time()

    async def connect(self) -> bool:
        """Conectar ao InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout,
                enable_gzip=self.config.enable_compression,
                ssl=self.config.enable_ssl
            )

            # Test connection
            health = self.client.health()
            if health.status == "pass":
                logger.info("Connected to InfluxDB successfully")

                # Initialize APIs
                self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
                self.query_api = self.client.query_api()
                self.delete_api = self.client.delete_api()

                # Setup buckets
                await self._ensure_buckets()

                return True
            else:
                logger.error(f"InfluxDB health check failed: {health.status}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    async def _ensure_buckets(self):
        """Garantir que buckets necessários existem"""
        try:
            buckets_api = self.client.buckets_api()
            existing_buckets = buckets_api.find_buckets()

            bucket_names = [bucket.name for bucket in existing_buckets.buckets]

            # Create missing buckets
            for bucket_name in [self.config.bucket_realtime,
                               self.config.bucket_historical,
                               self.config.bucket_analytics]:
                if bucket_name not in bucket_names:
                    bucket = buckets_api.create_bucket(
                        bucket_name=bucket_name,
                        org=self.config.org,
                        retention_rules=[{
                            "type": "expire",
                            "everySeconds": 86400 if "realtime" in bucket_name else 2592000  # 1 day or 30 days
                        }]
                    )
                    logger.info(f"Created bucket: {bucket_name}")

        except Exception as e:
            logger.error(f"Error ensuring buckets: {e}")

    async def write_tick(self, tick: ProcessedTickData, bucket: str = None) -> bool:
        """Escrever tick individual para o banco"""
        try:
            bucket = bucket or self.config.bucket_realtime

            # Create point
            point = Point("tick_data") \
                .tag("symbol", tick.symbol) \
                .field("price", float(tick.price)) \
                .field("price_change", float(tick.price_change)) \
                .field("price_change_pct", float(tick.price_change_pct)) \
                .field("price_velocity", float(tick.price_velocity)) \
                .field("price_acceleration", float(tick.price_acceleration)) \
                .field("volatility_1m", float(tick.volatility_1m)) \
                .field("volatility_5m", float(tick.volatility_5m)) \
                .field("volatility_15m", float(tick.volatility_15m)) \
                .field("sma_5", float(tick.sma_5)) \
                .field("sma_20", float(tick.sma_20)) \
                .field("ema_12", float(tick.ema_12)) \
                .field("ema_26", float(tick.ema_26)) \
                .field("rsi", float(tick.rsi)) \
                .field("macd", float(tick.macd)) \
                .field("macd_signal", float(tick.macd_signal)) \
                .field("bollinger_upper", float(tick.bollinger_upper)) \
                .field("bollinger_lower", float(tick.bollinger_lower)) \
                .field("bollinger_position", float(tick.bollinger_position)) \
                .field("tick_direction", int(tick.tick_direction)) \
                .field("momentum_score", float(tick.momentum_score)) \
                .field("price_position_sma", float(tick.price_position_sma)) \
                .field("volatility_rank", float(tick.volatility_rank)) \
                .field("trend_strength", float(tick.trend_strength)) \
                .time(tick.timestamp)

            # Add spread if available
            if tick.spread is not None:
                point = point.field("spread", float(tick.spread))

            # Write point
            self.write_api.write(bucket=bucket, org=self.config.org, record=point)
            self.write_count += 1

            return True

        except Exception as e:
            logger.error(f"Error writing tick to database: {e}")
            self.write_errors += 1
            return False

    async def write_batch(self, ticks: List[ProcessedTickData], bucket: str = None) -> bool:
        """Escrever batch de ticks"""
        try:
            bucket = bucket or self.config.bucket_realtime

            points = []
            for tick in ticks:
                point = Point("tick_data") \
                    .tag("symbol", tick.symbol) \
                    .field("price", float(tick.price)) \
                    .field("price_change", float(tick.price_change)) \
                    .field("price_velocity", float(tick.price_velocity)) \
                    .field("volatility_1m", float(tick.volatility_1m)) \
                    .field("rsi", float(tick.rsi)) \
                    .field("macd", float(tick.macd)) \
                    .field("momentum_score", float(tick.momentum_score)) \
                    .field("trend_strength", float(tick.trend_strength)) \
                    .time(tick.timestamp)

                points.append(point)

            # Write batch
            self.write_api.write(bucket=bucket, org=self.config.org, record=points)
            self.write_count += len(points)

            logger.info(f"Wrote batch of {len(points)} ticks to {bucket}")
            return True

        except Exception as e:
            logger.error(f"Error writing batch to database: {e}")
            self.write_errors += len(ticks)
            return False

    async def query_ticks(self, params: QueryParams, bucket: str = None) -> pd.DataFrame:
        """Consultar ticks do banco"""
        try:
            bucket = bucket or self.config.bucket_realtime
            self.query_count += 1

            # Build query
            query_parts = [
                f'from(bucket: "{bucket}")',
                f'|> range(start: {params.start_time.isoformat()}, stop: {params.end_time.isoformat()})',
                '|> filter(fn: (r) => r._measurement == "tick_data")'
            ]

            # Add symbol filter
            if params.symbols:
                symbol_filter = ' or '.join([f'r.symbol == "{symbol}"' for symbol in params.symbols])
                query_parts.append(f'|> filter(fn: (r) => {symbol_filter})')

            # Add aggregation
            if params.aggregation and params.window:
                query_parts.append(f'|> aggregateWindow(every: {params.window}, fn: {params.aggregation})')

            # Add limit
            if params.limit:
                query_parts.append(f'|> limit(n: {params.limit})')

            query = '\n  '.join(query_parts)

            # Execute query
            result = self.query_api.query_data_frame(query)

            if isinstance(result, list) and len(result) > 0:
                df = result[0]
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                df = pd.DataFrame()

            logger.info(f"Query returned {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error querying database: {e}")
            self.query_errors += 1
            return pd.DataFrame()

    async def get_latest_ticks(self, symbol: str, count: int = 100, bucket: str = None) -> pd.DataFrame:
        """Obter últimos ticks de um símbolo"""
        try:
            bucket = bucket or self.config.bucket_realtime

            query = f'''
            from(bucket: "{bucket}")
              |> range(start: -1h)
              |> filter(fn: (r) => r._measurement == "tick_data")
              |> filter(fn: (r) => r.symbol == "{symbol}")
              |> sort(columns: ["_time"], desc: true)
              |> limit(n: {count})
            '''

            result = self.query_api.query_data_frame(query)

            if isinstance(result, list) and len(result) > 0:
                df = result[0]
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                df = pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error getting latest ticks: {e}")
            return pd.DataFrame()

    async def aggregate_data(self, symbol: str, start_time: datetime, end_time: datetime,
                           window: str = "1m", bucket: str = None) -> pd.DataFrame:
        """Agregar dados em janelas de tempo"""
        try:
            bucket = bucket or self.config.bucket_realtime

            query = f'''
            from(bucket: "{bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "tick_data")
              |> filter(fn: (r) => r.symbol == "{symbol}")
              |> filter(fn: (r) => r._field == "price" or r._field == "volatility_1m" or r._field == "volume")
              |> aggregateWindow(every: {window}, fn: mean)
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            result = self.query_api.query_data_frame(query)

            if isinstance(result, list) and len(result) > 0:
                df = result[0]
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                df = pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return pd.DataFrame()

    async def get_price_history(self, symbol: str, hours: int = 24, bucket: str = None) -> pd.DataFrame:
        """Obter histórico de preços"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            params = QueryParams(
                start_time=start_time,
                end_time=end_time,
                symbols=[symbol],
                limit=10000
            )

            df = await self.query_ticks(params, bucket)

            # Filter for price data
            if not df.empty and '_field' in df.columns:
                price_df = df[df['_field'] == 'price'].copy()
                price_df['timestamp'] = pd.to_datetime(price_df['_time'])
                price_df = price_df[['timestamp', '_value', 'symbol']].rename(columns={'_value': 'price'})
                price_df = price_df.sort_values('timestamp')
                return price_df.reset_index(drop=True)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return pd.DataFrame()

    async def calculate_ohlcv(self, symbol: str, start_time: datetime, end_time: datetime,
                            interval: str = "1m", bucket: str = None) -> pd.DataFrame:
        """Calcular dados OHLCV"""
        try:
            bucket = bucket or self.config.bucket_realtime

            query = f'''
            data = from(bucket: "{bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "tick_data")
              |> filter(fn: (r) => r.symbol == "{symbol}")
              |> filter(fn: (r) => r._field == "price")

            ohlc = data
              |> aggregateWindow(every: {interval}, fn: (column, tables=<-) =>
                tables
                |> reduce(fn: (r, accumulator) => ({{
                  open: if accumulator.open == 0.0 then r._value else accumulator.open,
                  high: if r._value > accumulator.high then r._value else accumulator.high,
                  low: if r._value < accumulator.low then r._value else accumulator.low,
                  close: r._value,
                  count: accumulator.count + 1.0
                }}), identity: {{ open: 0.0, high: 0.0, low: 999999.0, close: 0.0, count: 0.0 }})
              )

            ohlc
              |> drop(columns: ["_start", "_stop", "_measurement", "symbol"])
            '''

            result = self.query_api.query_data_frame(query)

            if isinstance(result, list) and len(result) > 0:
                df = result[0]
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                df = pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error calculating OHLCV: {e}")
            return pd.DataFrame()

    async def delete_old_data(self, bucket: str, days: int = 7) -> bool:
        """Deletar dados antigos"""
        try:
            start_time = "1970-01-01T00:00:00Z"
            end_time = (datetime.now() - timedelta(days=days)).isoformat() + "Z"

            self.delete_api.delete(
                start=start_time,
                stop=end_time,
                predicate='_measurement="tick_data"',
                bucket=bucket,
                org=self.config.org
            )

            logger.info(f"Deleted data older than {days} days from {bucket}")
            return True

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return False

    async def backup_data(self, symbol: str, start_time: datetime, end_time: datetime,
                         output_file: str) -> bool:
        """Fazer backup de dados para arquivo"""
        try:
            params = QueryParams(
                start_time=start_time,
                end_time=end_time,
                symbols=[symbol]
            )

            df = await self.query_ticks(params)

            if not df.empty:
                # Compress and save
                with gzip.open(output_file, 'wt') as f:
                    df.to_csv(f, index=False)

                logger.info(f"Backed up {len(df)} records to {output_file}")
                return True
            else:
                logger.warning(f"No data found for backup: {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error backing up data: {e}")
            return False

    async def restore_data(self, input_file: str, bucket: str = None) -> bool:
        """Restaurar dados de arquivo"""
        try:
            bucket = bucket or self.config.bucket_realtime

            # Read compressed file
            with gzip.open(input_file, 'rt') as f:
                df = pd.read_csv(f)

            if df.empty:
                logger.warning("No data in backup file")
                return False

            # Convert back to points
            points = []
            for _, row in df.iterrows():
                point = Point("tick_data") \
                    .tag("symbol", row['symbol']) \
                    .field("price", float(row['_value'])) \
                    .time(pd.to_datetime(row['_time']))

                points.append(point)

            # Write in batches
            batch_size = self.config.batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.write_api.write(bucket=bucket, org=self.config.org, record=batch)

            logger.info(f"Restored {len(points)} records from {input_file}")
            return True

        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do banco de dados"""
        try:
            # Get bucket info
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets()

            bucket_stats = []
            for bucket in buckets.buckets:
                if bucket.name in [self.config.bucket_realtime,
                                  self.config.bucket_historical,
                                  self.config.bucket_analytics]:
                    bucket_stats.append({
                        'name': bucket.name,
                        'id': bucket.id,
                        'retention_rules': bucket.retention_rules
                    })

            return {
                'write_count': self.write_count,
                'write_errors': self.write_errors,
                'write_success_rate': self.write_count / max(1, self.write_count + self.write_errors),
                'query_count': self.query_count,
                'query_errors': self.query_errors,
                'query_success_rate': self.query_count / max(1, self.query_count + self.query_errors),
                'buckets': bucket_stats,
                'connection_status': 'connected' if self.client else 'disconnected'
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}

    async def health_check(self) -> bool:
        """Verificar saúde do banco de dados"""
        try:
            if not self.client:
                return False

            health = self.client.health()
            return health.status == "pass"

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def disconnect(self):
        """Desconectar do banco de dados"""
        try:
            if self.write_api:
                self.write_api.close()

            if self.client:
                self.client.close()

            self.thread_pool.shutdown(wait=True)

            logger.info("Disconnected from InfluxDB")

        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")


# 🧪 Função de teste
async def test_timeseries_database():
    """Testar integração com InfluxDB"""
    config = DatabaseConfig(
        url="http://localhost:8086",
        token="your-test-token",
        org="test-org",
        bucket_realtime="test-realtime"
    )

    db = TimeSeriesDatabase(config)

    # Connect
    connected = await db.connect()
    if not connected:
        print("❌ Failed to connect to InfluxDB")
        return

    print("✅ Connected to InfluxDB")

    # Create test data
    test_tick = ProcessedTickData(
        symbol="R_100",
        timestamp=datetime.now(),
        price=245.67,
        price_change=0.01,
        price_change_pct=0.004,
        price_velocity=0.001,
        volatility_1m=0.02,
        rsi=65.5,
        macd=0.05,
        momentum_score=0.3,
        trend_strength=0.2
    )

    # Write test tick
    success = await db.write_tick(test_tick)
    print(f"📝 Write test: {'✅' if success else '❌'}")

    # Wait a moment for write to complete
    await asyncio.sleep(2)

    # Query test data
    params = QueryParams(
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        symbols=["R_100"],
        limit=10
    )

    df = await db.query_ticks(params)
    print(f"📊 Query test: {len(df)} records returned")

    # Get latest ticks
    latest_df = await db.get_latest_ticks("R_100", count=5)
    print(f"📈 Latest ticks: {len(latest_df)} records")

    # Get stats
    stats = db.get_database_stats()
    print(f"📊 Database stats: {stats}")

    # Disconnect
    await db.disconnect()
    print("🔌 Disconnected from InfluxDB")


if __name__ == "__main__":
    print("🗄️ TESTING TIME-SERIES DATABASE")
    print("=" * 40)
    print("⚠️  Make sure InfluxDB is running locally")
    print("   docker run -p 8086:8086 influxdb:2.0")
    print("=" * 40)

    # Uncomment to run test (requires InfluxDB running)
    # asyncio.run(test_timeseries_database())
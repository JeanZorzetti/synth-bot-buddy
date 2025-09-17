"""
InfluxDB Time Series Database - Phase 13 Real-Time Data Pipeline
Sistema de armazenamento de séries temporais para dados de mercado em alta frequência
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.query_api import QueryApi
import logging

from real_logging_system import logging_system, LogComponent, LogLevel

@dataclass
class TimeSeriesPoint:
    measurement: str
    timestamp: datetime
    fields: Dict[str, Union[float, int, str]]
    tags: Dict[str, str]

class InfluxDBTimeSeriesManager:
    """High-performance time series data manager using InfluxDB"""

    def __init__(self):
        # InfluxDB Configuration
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', 'your-influxdb-token')
        self.org = os.getenv('INFLUXDB_ORG', 'trading-org')
        self.bucket = os.getenv('INFLUXDB_BUCKET', 'market-data')

        # Client configuration
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api: Optional[QueryApi] = None

        # Write configuration
        self.batch_size = int(os.getenv('INFLUXDB_BATCH_SIZE', '1000'))
        self.flush_interval = int(os.getenv('INFLUXDB_FLUSH_INTERVAL', '1000'))  # ms

        # Data retention
        self.tick_retention_days = int(os.getenv('TICK_RETENTION_DAYS', '30'))
        self.minute_retention_days = int(os.getenv('MINUTE_RETENTION_DAYS', '365'))
        self.hour_retention_days = int(os.getenv('HOUR_RETENTION_DAYS', '1095'))  # 3 years

        # Logging
        self.logger = logging_system.loggers.get('database', logging.getLogger(__name__))

    async def initialize(self) -> bool:
        """Initialize InfluxDB connection"""
        try:
            # Create InfluxDB client
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                enable_gzip=True,
                timeout=30000
            )

            # Initialize write API with batching
            self.write_api = self.client.write_api(
                write_options=ASYNCHRONOUS,
                batch_size=self.batch_size,
                flush_interval=self.flush_interval,
                jitter_interval=2000,
                retry_interval=5000,
                max_retries=3,
                max_retry_delay=30000,
                exponential_base=2
            )

            # Initialize query API
            self.query_api = self.client.query_api()

            # Test connection
            await self._test_connection()

            # Setup retention policies
            await self._setup_retention_policies()

            logging_system.log(
                LogComponent.DATABASE,
                LogLevel.INFO,
                "InfluxDB Time Series Manager initialized successfully",
                {'url': self.url, 'bucket': self.bucket}
            )

            return True

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'initialize_influxdb'}
            )
            return False

    async def _test_connection(self):
        """Test InfluxDB connection"""
        try:
            # Test with a simple query
            health = self.client.health()
            if health.status != "pass":
                raise Exception(f"InfluxDB health check failed: {health.status}")

            logging_system.log(
                LogComponent.DATABASE,
                LogLevel.INFO,
                "InfluxDB connection test successful"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'test_influxdb_connection'}
            )
            raise

    async def _setup_retention_policies(self):
        """Setup retention policies for different data granularities"""
        try:
            buckets_api = self.client.buckets_api()

            # Define retention policies
            retention_policies = [
                {'name': 'tick-data', 'retention_period': f'{self.tick_retention_days}d'},
                {'name': 'minute-data', 'retention_period': f'{self.minute_retention_days}d'},
                {'name': 'hour-data', 'retention_period': f'{self.hour_retention_days}d'}
            ]

            for policy in retention_policies:
                try:
                    bucket_name = f"{self.bucket}-{policy['name']}"
                    retention_seconds = self._parse_retention_period(policy['retention_period'])

                    # Check if bucket exists
                    existing_buckets = buckets_api.find_buckets()
                    bucket_exists = any(b.name == bucket_name for b in existing_buckets.buckets or [])

                    if not bucket_exists:
                        # Create bucket with retention policy
                        buckets_api.create_bucket(
                            bucket_name=bucket_name,
                            org=self.org,
                            retention_rules=[{
                                'type': 'expire',
                                'everySeconds': retention_seconds
                            }]
                        )

                        logging_system.log(
                            LogComponent.DATABASE,
                            LogLevel.INFO,
                            f"Created InfluxDB bucket: {bucket_name}",
                            {'retention_period': policy['retention_period']}
                        )

                except Exception as e:
                    logging_system.log_error(
                        LogComponent.DATABASE,
                        e,
                        {'action': 'setup_retention_policy', 'policy': policy}
                    )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'setup_retention_policies'}
            )

    def _parse_retention_period(self, period: str) -> int:
        """Parse retention period string to seconds"""
        if period.endswith('d'):
            return int(period[:-1]) * 24 * 3600
        elif period.endswith('h'):
            return int(period[:-1]) * 3600
        elif period.endswith('m'):
            return int(period[:-1]) * 60
        else:
            return int(period)

    async def write_tick_data(self, symbol: str, tick_data: Dict[str, Any]):
        """Write real-time tick data to InfluxDB"""
        try:
            point = Point("tick") \
                .tag("symbol", symbol) \
                .tag("data_type", "tick") \
                .field("bid", float(tick_data.get('bid', 0))) \
                .field("ask", float(tick_data.get('ask', 0))) \
                .field("price", float(tick_data.get('price', 0))) \
                .field("spread", float(tick_data.get('spread', 0))) \
                .field("pip_size", float(tick_data.get('pip_size', 0.0001))) \
                .time(tick_data.get('timestamp', datetime.utcnow()), WritePrecision.NS)

            # Add quote_id if available
            if 'quote_id' in tick_data:
                point = point.tag("quote_id", str(tick_data['quote_id']))

            # Write to tick data bucket
            self.write_api.write(
                bucket=f"{self.bucket}-tick-data",
                org=self.org,
                record=point
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'write_tick_data', 'symbol': symbol}
            )

    async def write_candle_data(self, symbol: str, candle_data: Dict[str, Any], granularity: str = '1m'):
        """Write candle/OHLCV data to InfluxDB"""
        try:
            point = Point("candle") \
                .tag("symbol", symbol) \
                .tag("granularity", granularity) \
                .tag("data_type", "candle") \
                .field("open", float(candle_data.get('open_price', 0))) \
                .field("high", float(candle_data.get('high_price', 0))) \
                .field("low", float(candle_data.get('low_price', 0))) \
                .field("close", float(candle_data.get('close_price', 0))) \
                .field("volume", int(candle_data.get('volume', 0))) \
                .time(candle_data.get('timestamp', datetime.utcnow()), WritePrecision.NS)

            # Determine bucket based on granularity
            if granularity in ['1m', '5m', '15m', '30m']:
                bucket_name = f"{self.bucket}-minute-data"
            else:
                bucket_name = f"{self.bucket}-hour-data"

            self.write_api.write(
                bucket=bucket_name,
                org=self.org,
                record=point
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'write_candle_data', 'symbol': symbol}
            )

    async def write_feature_data(self, symbol: str, features: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Write processed features to InfluxDB"""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()

            point = Point("features") \
                .tag("symbol", symbol) \
                .tag("data_type", "features") \
                .time(timestamp, WritePrecision.NS)

            # Add all feature fields
            for feature_name, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    point = point.field(feature_name, float(value))

            self.write_api.write(
                bucket=f"{self.bucket}-minute-data",
                org=self.org,
                record=point
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'write_feature_data', 'symbol': symbol}
            )

    async def write_ai_prediction(self, model_id: str, symbol: str, prediction: Dict[str, Any]):
        """Write AI model predictions to InfluxDB"""
        try:
            point = Point("prediction") \
                .tag("model_id", model_id) \
                .tag("symbol", symbol) \
                .tag("data_type", "prediction") \
                .field("signal", prediction.get('signal', 0)) \
                .field("confidence", float(prediction.get('confidence', 0))) \
                .field("probability_up", float(prediction.get('probability_up', 0.5))) \
                .field("probability_down", float(prediction.get('probability_down', 0.5))) \
                .time(prediction.get('timestamp', datetime.utcnow()), WritePrecision.NS)

            # Add additional prediction fields
            for field_name, value in prediction.items():
                if field_name not in ['signal', 'confidence', 'probability_up', 'probability_down', 'timestamp']:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        point = point.field(field_name, float(value))

            self.write_api.write(
                bucket=f"{self.bucket}-minute-data",
                org=self.org,
                record=point
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'write_ai_prediction', 'model_id': model_id, 'symbol': symbol}
            )

    async def query_tick_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Query tick data from InfluxDB"""
        try:
            if end_time is None:
                end_time = datetime.utcnow()

            query = f'''
                from(bucket: "{self.bucket}-tick-data")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "tick")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            if limit:
                query += f' |> limit(n: {limit})'

            result = self.query_api.query(query=query, org=self.org)

            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {
                        'timestamp': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'bid': record.values.get('bid'),
                        'ask': record.values.get('ask'),
                        'price': record.values.get('price'),
                        'spread': record.values.get('spread'),
                        'pip_size': record.values.get('pip_size')
                    }
                    data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'query_tick_data', 'symbol': symbol}
            )
            return pd.DataFrame()

    async def query_candle_data(
        self,
        symbol: str,
        granularity: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Query candle data from InfluxDB"""
        try:
            if end_time is None:
                end_time = datetime.utcnow()

            # Determine bucket
            bucket_name = f"{self.bucket}-minute-data" if granularity in ['1m', '5m', '15m', '30m'] else f"{self.bucket}-hour-data"

            query = f'''
                from(bucket: "{bucket_name}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "candle")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.granularity == "{granularity}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            if limit:
                query += f' |> limit(n: {limit})'

            result = self.query_api.query(query=query, org=self.org)

            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {
                        'timestamp': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    }
                    data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'query_candle_data', 'symbol': symbol}
            )
            return pd.DataFrame()

    async def query_features(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Query processed features from InfluxDB"""
        try:
            if end_time is None:
                end_time = datetime.utcnow()

            query = f'''
                from(bucket: "{self.bucket}-minute-data")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "features")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            '''

            if feature_names:
                feature_filter = ' or '.join([f'r._field == "{name}"' for name in feature_names])
                query += f' |> filter(fn: (r) => {feature_filter})'

            query += ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

            result = self.query_api.query(query=query, org=self.org)

            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {'timestamp': record.get_time(), 'symbol': record.values.get('symbol')}
                    # Add all feature values
                    for key, value in record.values.items():
                        if key not in ['timestamp', 'symbol', '_measurement', 'data_type']:
                            row[key] = value
                    data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'query_features', 'symbol': symbol}
            )
            return pd.DataFrame()

    async def get_latest_data(self, symbol: str, data_type: str = 'tick', limit: int = 100) -> Dict[str, Any]:
        """Get latest data points for symbol"""
        try:
            if data_type == 'tick':
                bucket_name = f"{self.bucket}-tick-data"
                measurement = "tick"
            elif data_type == 'candle':
                bucket_name = f"{self.bucket}-minute-data"
                measurement = "candle"
            else:
                bucket_name = f"{self.bucket}-minute-data"
                measurement = data_type

            query = f'''
                from(bucket: "{bucket_name}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> last()
                |> limit(n: {limit})
            '''

            result = self.query_api.query(query=query, org=self.org)

            # Convert to dictionary
            data = {}
            for table in result:
                for record in table.records:
                    field_name = record.get_field()
                    data[field_name] = record.get_value()
                    data['timestamp'] = record.get_time()
                    data['symbol'] = record.values.get('symbol')

            return data

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'get_latest_data', 'symbol': symbol, 'data_type': data_type}
            )
            return {}

    async def aggregate_data(
        self,
        symbol: str,
        measurement: str,
        aggregation_window: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Aggregate time series data"""
        try:
            if end_time is None:
                end_time = datetime.utcnow()

            bucket_name = f"{self.bucket}-minute-data"

            query = f'''
                from(bucket: "{bucket_name}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)
                |> yield(name: "mean")
            '''

            result = self.query_api.query(query=query, org=self.org)

            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {
                        'timestamp': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'field': record.get_field(),
                        'value': record.get_value()
                    }
                    data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'aggregate_data', 'symbol': symbol}
            )
            return pd.DataFrame()

    async def delete_old_data(self, days_to_keep: int = 30):
        """Delete old data based on retention policy"""
        try:
            delete_api = self.client.delete_api()
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)

            # Delete from tick data bucket
            delete_api.delete(
                start=datetime(1970, 1, 1),
                stop=cutoff_time,
                predicate='_measurement="tick"',
                bucket=f"{self.bucket}-tick-data",
                org=self.org
            )

            logging_system.log(
                LogComponent.DATABASE,
                LogLevel.INFO,
                f"Deleted tick data older than {days_to_keep} days"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'delete_old_data', 'days_to_keep': days_to_keep}
            )

    async def get_data_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}

            # Get bucket information
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets()

            for bucket in buckets.buckets or []:
                if bucket.name.startswith(self.bucket):
                    # Get data count for each bucket
                    query = f'''
                        from(bucket: "{bucket.name}")
                        |> range(start: -24h)
                        |> count()
                    '''

                    try:
                        result = self.query_api.query(query=query, org=self.org)
                        count = 0
                        for table in result:
                            for record in table.records:
                                count += record.get_value() or 0

                        stats[bucket.name] = {
                            'data_points_24h': count,
                            'retention_rules': bucket.retention_rules
                        }
                    except Exception:
                        stats[bucket.name] = {'data_points_24h': 0}

            return stats

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'get_data_stats'}
            )
            return {}

    async def close(self):
        """Close InfluxDB connection"""
        try:
            if self.write_api:
                self.write_api.close()
            if self.client:
                self.client.close()

            logging_system.log(
                LogComponent.DATABASE,
                LogLevel.INFO,
                "InfluxDB connection closed"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.DATABASE,
                e,
                {'action': 'close_influxdb'}
            )

# Global InfluxDB manager instance
influxdb_manager = InfluxDBTimeSeriesManager()

async def get_influxdb_manager() -> InfluxDBTimeSeriesManager:
    """Get initialized InfluxDB manager"""
    if not influxdb_manager.client:
        await influxdb_manager.initialize()
    return influxdb_manager
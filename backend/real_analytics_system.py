"""
📊 REAL ANALYTICS & REPORTING SYSTEM
Sistema completo de analytics com data warehouse real e business intelligence
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncpg
import aiofiles
from pathlib import Path
import boto3
from google.cloud import bigquery
import snowflake.connector
import redis
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
import schedule
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataWarehouseProvider(Enum):
    """🏢 Provedores de Data Warehouse"""
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    POSTGRESQL = "postgresql"


class ReportType(Enum):
    """📈 Tipos de relatórios"""
    TRADING_PERFORMANCE = "trading_performance"
    USER_ANALYTICS = "user_analytics"
    FINANCIAL_METRICS = "financial_metrics"
    RISK_ANALYSIS = "risk_analysis"
    SYSTEM_HEALTH = "system_health"
    MARKET_ANALYSIS = "market_analysis"
    COMPLIANCE_REPORT = "compliance_report"


class AlertSeverity(Enum):
    """🚨 Severidade dos alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataWarehouseConfig:
    """⚙️ Configuração do Data Warehouse"""
    provider: DataWarehouseProvider
    connection_string: str
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    warehouse_name: Optional[str] = None
    schema_name: Optional[str] = None
    region: str = "us-central1"
    max_connections: int = 10
    timeout_seconds: int = 300


@dataclass
class AnalyticsMetric:
    """📊 Métrica de analytics"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""
    value: Union[float, int, str] = 0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    dimensions: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Report:
    """📄 Relatório"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ReportType = ReportType.TRADING_PERFORMANCE
    description: str = ""
    sql_query: str = ""
    parameters: Dict = field(default_factory=dict)
    schedule_cron: str = ""
    output_format: str = "html"  # html, pdf, csv, json
    recipients: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_run_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class Alert:
    """🚨 Alerta"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric_name: str = ""
    condition: str = ""  # >, <, >=, <=, ==, !=
    threshold_value: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    is_active: bool = True
    recipients: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered_at: Optional[datetime] = None
    trigger_count: int = 0


class BigQueryManager:
    """📊 Gerenciador BigQuery"""

    def __init__(self, project_id: str, dataset_id: str, credentials_path: str = ""):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.credentials_path = credentials_path
        self.client = None

    async def initialize(self):
        """Inicializar cliente BigQuery"""
        try:
            if self.credentials_path:
                self.client = bigquery.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id
                )
            else:
                self.client = bigquery.Client(project=self.project_id)

            # Criar dataset se não existir
            await self._create_dataset_if_not_exists()

            logger.info("✅ BigQuery client initialized")
        except Exception as e:
            logger.error(f"❌ BigQuery initialization failed: {e}")

    async def _create_dataset_if_not_exists(self):
        """Criar dataset se não existir"""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)

            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"BigQuery dataset exists: {self.dataset_id}")
            except Exception:
                # Criar dataset
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "Trading Bot Analytics Data"

                self.client.create_dataset(dataset)
                logger.info(f"✅ BigQuery dataset created: {self.dataset_id}")

        except Exception as e:
            logger.error(f"❌ Error creating BigQuery dataset: {e}")

    async def create_tables(self):
        """Criar tabelas de analytics"""
        tables_schema = {
            "trading_metrics": [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("trade_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("position_size", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("entry_price", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("exit_price", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("profit_loss", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("duration_minutes", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("strategy_name", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE")
            ],

            "user_analytics": [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("event_data", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("ip_address", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("user_agent", "STRING", mode="NULLABLE")
            ],

            "system_metrics": [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("metric_value", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("metric_unit", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("component", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("dimensions", "JSON", mode="NULLABLE")
            ],

            "financial_data": [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("price", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("volume", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("bid", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("ask", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("spread", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("market_data", "JSON", mode="NULLABLE")
            ]
        }

        for table_name, schema in tables_schema.items():
            await self._create_table(table_name, schema)

    async def _create_table(self, table_name: str, schema: List):
        """Criar tabela individual"""
        try:
            table_ref = self.client.dataset(self.dataset_id).table(table_name)

            try:
                self.client.get_table(table_ref)
                logger.info(f"BigQuery table exists: {table_name}")
            except Exception:
                # Criar tabela
                table = bigquery.Table(table_ref, schema=schema)

                # Configurar particionamento por timestamp
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="timestamp"
                )

                self.client.create_table(table)
                logger.info(f"✅ BigQuery table created: {table_name}")

        except Exception as e:
            logger.error(f"❌ Error creating BigQuery table {table_name}: {e}")

    async def insert_data(self, table_name: str, data: List[Dict]) -> bool:
        """Inserir dados na tabela"""
        try:
            table_ref = self.client.dataset(self.dataset_id).table(table_name)
            table = self.client.get_table(table_ref)

            # Inserir dados
            errors = self.client.insert_rows_json(table, data)

            if errors:
                logger.error(f"❌ BigQuery insert errors: {errors}")
                return False

            logger.info(f"✅ Inserted {len(data)} rows into {table_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inserting data into BigQuery: {e}")
            return False

    async def execute_query(self, query: str) -> pd.DataFrame:
        """Executar query e retornar DataFrame"""
        try:
            query_job = self.client.query(query)
            results = query_job.result()

            # Converter para DataFrame
            df = results.to_dataframe()

            logger.info(f"✅ Query executed successfully, {len(df)} rows returned")
            return df

        except Exception as e:
            logger.error(f"❌ Error executing BigQuery query: {e}")
            return pd.DataFrame()


class SnowflakeManager:
    """❄️ Gerenciador Snowflake"""

    def __init__(self, account: str, user: str, password: str, warehouse: str, database: str, schema: str = "ANALYTICS"):
        self.connection_params = {
            'account': account,
            'user': user,
            'password': password,
            'warehouse': warehouse,
            'database': database,
            'schema': schema
        }
        self.connection = None

    async def initialize(self):
        """Inicializar conexão Snowflake"""
        try:
            self.connection = snowflake.connector.connect(**self.connection_params)
            logger.info("✅ Snowflake connection established")

            # Criar schema se não existir
            await self._create_schema_if_not_exists()

        except Exception as e:
            logger.error(f"❌ Snowflake initialization failed: {e}")

    async def _create_schema_if_not_exists(self):
        """Criar schema se não existir"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.connection_params['schema']}")
            cursor.close()
            logger.info(f"✅ Snowflake schema ready: {self.connection_params['schema']}")

        except Exception as e:
            logger.error(f"❌ Error creating Snowflake schema: {e}")

    async def create_tables(self):
        """Criar tabelas de analytics no Snowflake"""
        tables_ddl = {
            "TRADING_METRICS": """
                CREATE TABLE IF NOT EXISTS TRADING_METRICS (
                    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                    USER_ID VARCHAR(255) NOT NULL,
                    SYMBOL VARCHAR(50) NOT NULL,
                    TRADE_TYPE VARCHAR(20) NOT NULL,
                    POSITION_SIZE FLOAT NOT NULL,
                    ENTRY_PRICE FLOAT NOT NULL,
                    EXIT_PRICE FLOAT,
                    PROFIT_LOSS FLOAT,
                    DURATION_MINUTES INTEGER,
                    STRATEGY_NAME VARCHAR(100),
                    METADATA VARIANT
                )
            """,

            "USER_ANALYTICS": """
                CREATE TABLE IF NOT EXISTS USER_ANALYTICS (
                    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                    USER_ID VARCHAR(255) NOT NULL,
                    EVENT_TYPE VARCHAR(100) NOT NULL,
                    EVENT_DATA VARIANT,
                    SESSION_ID VARCHAR(255),
                    IP_ADDRESS VARCHAR(45),
                    USER_AGENT TEXT
                )
            """,

            "SYSTEM_METRICS": """
                CREATE TABLE IF NOT EXISTS SYSTEM_METRICS (
                    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                    METRIC_NAME VARCHAR(100) NOT NULL,
                    METRIC_VALUE FLOAT NOT NULL,
                    METRIC_UNIT VARCHAR(20),
                    COMPONENT VARCHAR(100) NOT NULL,
                    DIMENSIONS VARIANT
                )
            """,

            "FINANCIAL_DATA": """
                CREATE TABLE IF NOT EXISTS FINANCIAL_DATA (
                    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
                    SYMBOL VARCHAR(50) NOT NULL,
                    PRICE FLOAT NOT NULL,
                    VOLUME FLOAT,
                    BID FLOAT,
                    ASK FLOAT,
                    SPREAD FLOAT,
                    MARKET_DATA VARIANT
                )
            """
        }

        cursor = self.connection.cursor()

        for table_name, ddl in tables_ddl.items():
            try:
                cursor.execute(ddl)
                logger.info(f"✅ Snowflake table ready: {table_name}")
            except Exception as e:
                logger.error(f"❌ Error creating Snowflake table {table_name}: {e}")

        cursor.close()

    async def execute_query(self, query: str) -> pd.DataFrame:
        """Executar query no Snowflake"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)

            # Converter para DataFrame
            data = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            cursor.close()

            logger.info(f"✅ Snowflake query executed, {len(df)} rows returned")
            return df

        except Exception as e:
            logger.error(f"❌ Error executing Snowflake query: {e}")
            return pd.DataFrame()


class ETLPipeline:
    """🔄 Pipeline ETL"""

    def __init__(self, source_db_url: str, warehouse_manager):
        self.source_db_url = source_db_url
        self.warehouse_manager = warehouse_manager
        self.source_pool = None

    async def initialize(self):
        """Inicializar pipeline ETL"""
        try:
            self.source_pool = await asyncpg.create_pool(self.source_db_url)
            logger.info("✅ ETL Pipeline initialized")
        except Exception as e:
            logger.error(f"❌ ETL Pipeline initialization failed: {e}")

    async def extract_trading_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extrair dados de trading"""
        try:
            async with self.source_pool.acquire() as conn:
                query = """
                    SELECT
                        created_at as timestamp,
                        user_id,
                        symbol,
                        trade_type,
                        position_size,
                        entry_price,
                        exit_price,
                        profit_loss,
                        EXTRACT(EPOCH FROM (closed_at - created_at))/60 as duration_minutes,
                        strategy_name,
                        metadata
                    FROM trading_positions
                    WHERE created_at >= $1 AND created_at <= $2
                """

                rows = await conn.fetch(query, start_date, end_date)

                data = []
                for row in rows:
                    data.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'user_id': str(row['user_id']),
                        'symbol': row['symbol'],
                        'trade_type': row['trade_type'],
                        'position_size': float(row['position_size']),
                        'entry_price': float(row['entry_price']),
                        'exit_price': float(row['exit_price']) if row['exit_price'] else None,
                        'profit_loss': float(row['profit_loss']) if row['profit_loss'] else None,
                        'duration_minutes': int(row['duration_minutes']) if row['duration_minutes'] else None,
                        'strategy_name': row['strategy_name'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    })

                logger.info(f"✅ Extracted {len(data)} trading records")
                return data

        except Exception as e:
            logger.error(f"❌ Error extracting trading data: {e}")
            return []

    async def extract_user_analytics(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extrair dados de analytics de usuários"""
        try:
            async with self.source_pool.acquire() as conn:
                query = """
                    SELECT
                        timestamp,
                        user_id,
                        event_type,
                        event_data,
                        session_id,
                        ip_address,
                        user_agent
                    FROM user_events
                    WHERE timestamp >= $1 AND timestamp <= $2
                """

                rows = await conn.fetch(query, start_date, end_date)

                data = []
                for row in rows:
                    data.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'user_id': str(row['user_id']),
                        'event_type': row['event_type'],
                        'event_data': json.loads(row['event_data']) if row['event_data'] else {},
                        'session_id': row['session_id'],
                        'ip_address': row['ip_address'],
                        'user_agent': row['user_agent']
                    })

                logger.info(f"✅ Extracted {len(data)} user analytics records")
                return data

        except Exception as e:
            logger.error(f"❌ Error extracting user analytics: {e}")
            return []

    async def load_to_warehouse(self, table_name: str, data: List[Dict]) -> bool:
        """Carregar dados para o data warehouse"""
        try:
            if isinstance(self.warehouse_manager, BigQueryManager):
                return await self.warehouse_manager.insert_data(table_name, data)
            elif isinstance(self.warehouse_manager, SnowflakeManager):
                # Implementar inserção no Snowflake
                return await self._load_to_snowflake(table_name, data)

            return False

        except Exception as e:
            logger.error(f"❌ Error loading data to warehouse: {e}")
            return False

    async def _load_to_snowflake(self, table_name: str, data: List[Dict]) -> bool:
        """Carregar dados no Snowflake"""
        try:
            if not data:
                return True

            cursor = self.warehouse_manager.connection.cursor()

            # Preparar dados para inserção
            columns = list(data[0].keys())
            placeholders = ", ".join(["%s"] * len(columns))

            insert_query = f"""
                INSERT INTO {table_name.upper()} ({", ".join(columns).upper()})
                VALUES ({placeholders})
            """

            # Converter dados para tuplas
            values = []
            for row in data:
                values.append(tuple(row.values()))

            # Inserir em batch
            cursor.executemany(insert_query, values)
            cursor.close()

            logger.info(f"✅ Loaded {len(data)} rows into Snowflake table {table_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading data to Snowflake: {e}")
            return False

    async def run_daily_etl(self):
        """Executar ETL diário"""
        try:
            # Data de ontem
            yesterday = datetime.now() - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

            logger.info(f"🔄 Starting daily ETL for {start_date.date()}")

            # Extrair e carregar dados de trading
            trading_data = await self.extract_trading_data(start_date, end_date)
            if trading_data:
                await self.load_to_warehouse("trading_metrics", trading_data)

            # Extrair e carregar analytics de usuários
            user_data = await self.extract_user_analytics(start_date, end_date)
            if user_data:
                await self.load_to_warehouse("user_analytics", user_data)

            logger.info("✅ Daily ETL completed successfully")

        except Exception as e:
            logger.error(f"❌ Daily ETL failed: {e}")


class RealtimeAnalytics:
    """⚡ Analytics em tempo real"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.metrics_buffer = []
        self.buffer_size = 1000

    async def initialize(self):
        """Inicializar sistema de analytics em tempo real"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Real-time analytics initialized")
        except Exception as e:
            logger.error(f"❌ Real-time analytics initialization failed: {e}")

    async def track_metric(self, metric: AnalyticsMetric):
        """Rastrear métrica em tempo real"""
        try:
            # Adicionar ao buffer
            metric_data = {
                'id': metric.id,
                'name': metric.name,
                'category': metric.category,
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat(),
                'dimensions': metric.dimensions,
                'metadata': metric.metadata
            }

            self.metrics_buffer.append(metric_data)

            # Armazenar no Redis para acesso rápido
            key = f"metric:{metric.name}:{metric.timestamp.strftime('%Y%m%d%H%M')}"
            await self.redis_client.setex(key, 3600, json.dumps(metric_data))  # TTL 1 hora

            # Flush buffer se necessário
            if len(self.metrics_buffer) >= self.buffer_size:
                await self.flush_metrics()

        except Exception as e:
            logger.error(f"❌ Error tracking metric: {e}")

    async def flush_metrics(self):
        """Flush métricas para armazenamento permanente"""
        try:
            if not self.metrics_buffer:
                return

            # Aqui você pode implementar o envio para o data warehouse
            logger.info(f"🔄 Flushing {len(self.metrics_buffer)} metrics")

            # Limpar buffer
            self.metrics_buffer.clear()

        except Exception as e:
            logger.error(f"❌ Error flushing metrics: {e}")

    async def get_realtime_metrics(self, metric_name: str, minutes_back: int = 60) -> List[Dict]:
        """Obter métricas em tempo real"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes_back)

            metrics = []
            current_time = start_time

            while current_time <= end_time:
                key = f"metric:{metric_name}:{current_time.strftime('%Y%m%d%H%M')}"
                data = await self.redis_client.get(key)

                if data:
                    metrics.append(json.loads(data))

                current_time += timedelta(minutes=1)

            return metrics

        except Exception as e:
            logger.error(f"❌ Error getting real-time metrics: {e}")
            return []


class BusinessIntelligenceDashboard:
    """📊 Dashboard de Business Intelligence"""

    def __init__(self, warehouse_manager):
        self.warehouse_manager = warehouse_manager

    async def generate_trading_performance_report(self, user_id: str = None, days: int = 30) -> Dict:
        """Gerar relatório de performance de trading"""
        try:
            # Query base
            where_clause = "WHERE timestamp >= CURRENT_DATE() - INTERVAL %d DAY" % days
            if user_id:
                where_clause += f" AND user_id = '{user_id}'"

            query = f"""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(profit_loss) as total_pnl,
                    AVG(profit_loss) as avg_pnl,
                    MAX(profit_loss) as max_profit,
                    MIN(profit_loss) as max_loss,
                    AVG(duration_minutes) as avg_duration
                FROM trading_metrics
                {where_clause}
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """

            df = await self.warehouse_manager.execute_query(query)

            if df.empty:
                return {"error": "No trading data found"}

            # Calcular métricas agregadas
            total_trades = df['total_trades'].sum()
            winning_trades = df['winning_trades'].sum()
            total_pnl = df['total_pnl'].sum()

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            sharpe_ratio = self._calculate_sharpe_ratio(df['total_pnl'].tolist())

            return {
                'summary': {
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(float(total_pnl), 2),
                    'sharpe_ratio': round(sharpe_ratio, 3),
                    'avg_daily_pnl': round(float(df['total_pnl'].mean()), 2)
                },
                'daily_data': df.to_dict('records'),
                'chart_data': self._create_pnl_chart(df)
            }

        except Exception as e:
            logger.error(f"❌ Error generating trading performance report: {e}")
            return {"error": str(e)}

    async def generate_user_analytics_report(self, days: int = 30) -> Dict:
        """Gerar relatório de analytics de usuários"""
        try:
            query = f"""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_events,
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(events_per_user) as avg_events_per_user
                FROM (
                    SELECT
                        timestamp,
                        user_id,
                        session_id,
                        COUNT(*) OVER (PARTITION BY user_id, DATE(timestamp)) as events_per_user
                    FROM user_analytics
                    WHERE timestamp >= CURRENT_DATE() - INTERVAL {days} DAY
                ) subquery
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """

            df = await self.warehouse_manager.execute_query(query)

            if df.empty:
                return {"error": "No user analytics data found"}

            return {
                'summary': {
                    'total_active_users': int(df['active_users'].sum()),
                    'avg_daily_users': round(float(df['active_users'].mean()), 1),
                    'total_events': int(df['total_events'].sum()),
                    'total_sessions': int(df['total_sessions'].sum())
                },
                'daily_data': df.to_dict('records'),
                'chart_data': self._create_user_activity_chart(df)
            }

        except Exception as e:
            logger.error(f"❌ Error generating user analytics report: {e}")
            return {"error": str(e)}

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calcular Sharpe Ratio"""
        try:
            if not returns or len(returns) < 2:
                return 0.0

            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

            if np.std(excess_returns) == 0:
                return 0.0

            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return float(sharpe)

        except Exception:
            return 0.0

    def _create_pnl_chart(self, df: pd.DataFrame) -> Dict:
        """Criar gráfico de P&L"""
        try:
            fig = go.Figure()

            # Adicionar linha de P&L cumulativo
            cumulative_pnl = df['total_pnl'].cumsum()

            fig.add_trace(go.Scatter(
                x=df['date'],
                y=cumulative_pnl,
                mode='lines+markers',
                name='P&L Cumulativo',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title='Performance de Trading - P&L Cumulativo',
                xaxis_title='Data',
                yaxis_title='P&L ($)',
                hovermode='x unified'
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"❌ Error creating P&L chart: {e}")
            return {}

    def _create_user_activity_chart(self, df: pd.DataFrame) -> Dict:
        """Criar gráfico de atividade de usuários"""
        try:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['active_users'],
                name='Usuários Ativos',
                marker_color='green'
            ))

            fig.update_layout(
                title='Atividade de Usuários',
                xaxis_title='Data',
                yaxis_title='Usuários Ativos',
                hovermode='x unified'
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"❌ Error creating user activity chart: {e}")
            return {}


class AlertSystem:
    """🚨 Sistema de alertas"""

    def __init__(self, warehouse_manager, notification_config: Dict = None):
        self.warehouse_manager = warehouse_manager
        self.notification_config = notification_config or {}
        self.alerts = {}  # {alert_id: Alert}

    async def add_alert(self, alert: Alert):
        """Adicionar alerta"""
        self.alerts[alert.id] = alert
        logger.info(f"✅ Alert added: {alert.name}")

    async def check_alerts(self):
        """Verificar todos os alertas"""
        for alert in self.alerts.values():
            if alert.is_active:
                await self._check_single_alert(alert)

    async def _check_single_alert(self, alert: Alert):
        """Verificar alerta individual"""
        try:
            # Obter valor atual da métrica
            query = f"""
                SELECT {alert.metric_name} as value
                FROM system_metrics
                WHERE metric_name = '{alert.metric_name}'
                ORDER BY timestamp DESC
                LIMIT 1
            """

            df = await self.warehouse_manager.execute_query(query)

            if df.empty:
                return

            current_value = float(df.iloc[0]['value'])

            # Verificar condição
            triggered = self._evaluate_condition(current_value, alert.condition, alert.threshold_value)

            if triggered:
                await self._trigger_alert(alert, current_value)

        except Exception as e:
            logger.error(f"❌ Error checking alert {alert.name}: {e}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Avaliar condição do alerta"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            else:
                return False
        except Exception:
            return False

    async def _trigger_alert(self, alert: Alert, current_value: float):
        """Disparar alerta"""
        try:
            alert.trigger_count += 1
            alert.last_triggered_at = datetime.now()

            message = f"""
            🚨 ALERTA: {alert.name}

            Descrição: {alert.description}
            Métrica: {alert.metric_name}
            Valor atual: {current_value}
            Condição: {alert.condition} {alert.threshold_value}
            Severidade: {alert.severity.value.upper()}

            Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            # Enviar notificações
            for recipient in alert.recipients:
                await self._send_notification(recipient, f"Alerta: {alert.name}", message, alert.severity)

            logger.warning(f"🚨 Alert triggered: {alert.name} (value: {current_value})")

        except Exception as e:
            logger.error(f"❌ Error triggering alert: {e}")

    async def _send_notification(self, recipient: str, subject: str, message: str, severity: AlertSeverity):
        """Enviar notificação"""
        try:
            if "@" in recipient:
                # Email
                await self._send_email_notification(recipient, subject, message)
            elif recipient.startswith("slack:"):
                # Slack
                await self._send_slack_notification(recipient[6:], message)
            elif recipient.startswith("webhook:"):
                # Webhook
                await self._send_webhook_notification(recipient[8:], message)

        except Exception as e:
            logger.error(f"❌ Error sending notification: {e}")

    async def _send_email_notification(self, email: str, subject: str, message: str):
        """Enviar notificação por email"""
        try:
            if not self.notification_config.get('smtp_server'):
                logger.warning("SMTP not configured for email notifications")
                return

            msg = MimeMultipart()
            msg['From'] = self.notification_config['from_email']
            msg['To'] = email
            msg['Subject'] = subject

            msg.attach(MimeText(message, 'plain'))

            server = smtplib.SMTP(
                self.notification_config['smtp_server'],
                self.notification_config['smtp_port']
            )
            server.starttls()
            server.login(
                self.notification_config['smtp_user'],
                self.notification_config['smtp_password']
            )
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Email notification sent to: {email}")

        except Exception as e:
            logger.error(f"❌ Error sending email notification: {e}")

    async def _send_slack_notification(self, channel: str, message: str):
        """Enviar notificação para Slack"""
        # Implementar integração com Slack
        logger.info(f"📱 Slack notification sent to: {channel}")

    async def _send_webhook_notification(self, webhook_url: str, message: str):
        """Enviar notificação via webhook"""
        # Implementar webhook
        logger.info(f"🔗 Webhook notification sent to: {webhook_url}")


class RealAnalyticsSystem:
    """📊 Sistema principal de analytics"""

    def __init__(self, config: DataWarehouseConfig, source_db_url: str):
        self.config = config
        self.warehouse_manager = None
        self.etl_pipeline = None
        self.realtime_analytics = None
        self.dashboard = None
        self.alert_system = None

    async def initialize(self):
        """Inicializar sistema completo"""
        try:
            # Inicializar warehouse manager
            if self.config.provider == DataWarehouseProvider.BIGQUERY:
                self.warehouse_manager = BigQueryManager(
                    self.config.project_id,
                    self.config.dataset_id
                )
            elif self.config.provider == DataWarehouseProvider.SNOWFLAKE:
                account, user, password = self.config.connection_string.split(":")
                self.warehouse_manager = SnowflakeManager(
                    account, user, password,
                    self.config.warehouse_name,
                    self.config.project_id,
                    self.config.schema_name
                )

            await self.warehouse_manager.initialize()
            await self.warehouse_manager.create_tables()

            # Inicializar componentes
            self.etl_pipeline = ETLPipeline(source_db_url, self.warehouse_manager)
            await self.etl_pipeline.initialize()

            self.realtime_analytics = RealtimeAnalytics()
            await self.realtime_analytics.initialize()

            self.dashboard = BusinessIntelligenceDashboard(self.warehouse_manager)
            self.alert_system = AlertSystem(self.warehouse_manager)

            # Configurar alertas padrão
            await self._setup_default_alerts()

            logger.info("✅ Real Analytics System fully initialized")

        except Exception as e:
            logger.error(f"❌ Real Analytics System initialization failed: {e}")

    async def _setup_default_alerts(self):
        """Configurar alertas padrão"""
        default_alerts = [
            Alert(
                name="High Error Rate",
                description="Taxa de erro do sistema acima de 5%",
                metric_name="error_rate",
                condition=">",
                threshold_value=5.0,
                severity=AlertSeverity.ERROR,
                recipients=["admin@tradingbot.com"]
            ),
            Alert(
                name="Low Win Rate",
                description="Taxa de vitória abaixo de 40%",
                metric_name="win_rate",
                condition="<",
                threshold_value=40.0,
                severity=AlertSeverity.WARNING,
                recipients=["trading@tradingbot.com"]
            ),
            Alert(
                name="High Latency",
                description="Latência acima de 100ms",
                metric_name="avg_latency_ms",
                condition=">",
                threshold_value=100.0,
                severity=AlertSeverity.WARNING,
                recipients=["ops@tradingbot.com"]
            )
        ]

        for alert in default_alerts:
            await self.alert_system.add_alert(alert)

    async def run_daily_analytics(self):
        """Executar analytics diário"""
        try:
            logger.info("🔄 Starting daily analytics processing")

            # Executar ETL
            await self.etl_pipeline.run_daily_etl()

            # Verificar alertas
            await self.alert_system.check_alerts()

            # Flush métricas em tempo real
            await self.realtime_analytics.flush_metrics()

            logger.info("✅ Daily analytics processing completed")

        except Exception as e:
            logger.error(f"❌ Daily analytics processing failed: {e}")

    async def get_analytics_report(self, report_type: ReportType, **kwargs) -> Dict:
        """Obter relatório de analytics"""
        try:
            if report_type == ReportType.TRADING_PERFORMANCE:
                return await self.dashboard.generate_trading_performance_report(**kwargs)
            elif report_type == ReportType.USER_ANALYTICS:
                return await self.dashboard.generate_user_analytics_report(**kwargs)
            else:
                return {"error": f"Report type {report_type.value} not implemented"}

        except Exception as e:
            logger.error(f"❌ Error generating analytics report: {e}")
            return {"error": str(e)}


# 🧪 Função de teste
async def test_real_analytics_system():
    """Testar sistema de analytics real"""
    # Configuração do BigQuery (exemplo)
    config = DataWarehouseConfig(
        provider=DataWarehouseProvider.BIGQUERY,
        project_id="trading-bot-analytics",
        dataset_id="trading_data",
        connection_string=""
    )

    source_db_url = "postgresql://trading_user:password@localhost:5432/trading_db"

    # Inicializar sistema
    analytics_system = RealAnalyticsSystem(config, source_db_url)
    await analytics_system.initialize()

    print("\n" + "="*80)
    print("📊 REAL ANALYTICS & REPORTING SYSTEM TEST")
    print("="*80)

    # 1. Testar métricas em tempo real
    print("\n⚡ TESTING REAL-TIME METRICS...")
    metric = AnalyticsMetric(
        name="trading_profit",
        category="trading",
        value=150.75,
        unit="USD",
        dimensions={"user_id": "test_user", "symbol": "EURUSD"}
    )

    await analytics_system.realtime_analytics.track_metric(metric)
    print(f"✅ Real-time metric tracked: {metric.name} = {metric.value}")

    # 2. Gerar relatório de performance
    print("\n📈 GENERATING TRADING PERFORMANCE REPORT...")
    trading_report = await analytics_system.get_analytics_report(
        ReportType.TRADING_PERFORMANCE,
        days=30
    )

    if "error" not in trading_report:
        print(f"✅ Trading report generated:")
        print(f"   Total Trades: {trading_report['summary']['total_trades']}")
        print(f"   Win Rate: {trading_report['summary']['win_rate']}%")
        print(f"   Total P&L: ${trading_report['summary']['total_pnl']}")
        print(f"   Sharpe Ratio: {trading_report['summary']['sharpe_ratio']}")
    else:
        print(f"⚠️ Trading report: {trading_report['error']}")

    # 3. Gerar relatório de usuários
    print("\n👥 GENERATING USER ANALYTICS REPORT...")
    user_report = await analytics_system.get_analytics_report(
        ReportType.USER_ANALYTICS,
        days=30
    )

    if "error" not in user_report:
        print(f"✅ User analytics report generated:")
        print(f"   Total Active Users: {user_report['summary']['total_active_users']}")
        print(f"   Avg Daily Users: {user_report['summary']['avg_daily_users']}")
        print(f"   Total Events: {user_report['summary']['total_events']}")
        print(f"   Total Sessions: {user_report['summary']['total_sessions']}")
    else:
        print(f"⚠️ User report: {user_report['error']}")

    # 4. Executar processamento diário
    print("\n🔄 TESTING DAILY ANALYTICS PROCESSING...")
    await analytics_system.run_daily_analytics()
    print("✅ Daily analytics processing completed")

    print("\n" + "="*80)
    print("✅ REAL ANALYTICS & REPORTING SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_real_analytics_system())
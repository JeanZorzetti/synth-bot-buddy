"""
Database Configuration - Phase 12 Real Infrastructure
Configuração completa de banco PostgreSQL e Redis para produção
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncpg
import redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

# Database Models Base
Base = declarative_base()

class TradingSession(Base):
    __tablename__ = 'trading_sessions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    status = Column(String(20), default='active')
    ai_model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradingPosition(Base):
    __tablename__ = 'trading_positions'

    id = Column(Integer, primary_key=True)
    position_id = Column(String(50), unique=True, nullable=False)
    session_id = Column(String(50), ForeignKey('trading_sessions.session_id'))
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'CALL' or 'PUT'
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    amount = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    duration_seconds = Column(Integer)
    pnl = Column(Float, default=0.0)
    status = Column(String(20), default='open')
    ai_confidence = Column(Float)
    ai_signal_strength = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    bid = Column(Float)
    ask = Column(Float)
    spread = Column(Float)

class ProcessedFeatures(Base):
    __tablename__ = 'processed_features'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    features_json = Column(JSON, nullable=False)  # All calculated features
    rsi_14 = Column(Float)
    macd_signal = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    sma_20 = Column(Float)
    ema_12 = Column(Float)
    volume_sma = Column(Float)
    atr_14 = Column(Float)
    momentum_10 = Column(Float)
    stochastic_k = Column(Float)
    williams_r = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class AIModelMetrics(Base):
    __tablename__ = 'ai_model_metrics'

    id = Column(Integer, primary_key=True)
    model_id = Column(String(50), nullable=False)
    model_type = Column(String(20), nullable=False)  # 'lstm', 'transformer', etc.
    version = Column(String(20), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    training_start = Column(DateTime)
    training_end = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserActivity(Base):
    __tablename__ = 'user_activities'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    username = Column(String(100), nullable=False)
    activity_type = Column(String(50), nullable=False)
    description = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(100))
    api_endpoint = Column(String(200))
    response_status = Column(Integer)
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    timestamp = Column(DateTime, nullable=False)
    component = Column(String(50))  # 'api', 'database', 'ai_engine', etc.
    environment = Column(String(20), default='production')
    additional_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql+asyncpg://trading_user:trading_password@localhost:5432/trading_bot_db'
        )
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

        # AsyncPG connection for high-performance operations
        self.postgres_pool: Optional[asyncpg.Pool] = None

        # SQLAlchemy async engine for ORM operations
        self.async_engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )

        # Session factory
        self.async_session = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Redis connection
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Logger
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database connections and create tables"""
        try:
            # Create PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                self.database_url.replace('+asyncpg', ''),
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Create tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Test Redis connection
            await self._test_redis_connection()

            self.logger.info("Database connections initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False

    async def _test_redis_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise

    async def close(self):
        """Close all database connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.async_engine:
            await self.async_engine.dispose()
        self.redis_client.close()

    # Trading Sessions Management
    async def create_trading_session(self, session_data: Dict[str, Any]) -> str:
        """Create new trading session"""
        async with self.async_session() as session:
            trading_session = TradingSession(**session_data)
            session.add(trading_session)
            await session.commit()
            await session.refresh(trading_session)
            return trading_session.session_id

    async def get_active_trading_sessions(self) -> List[Dict[str, Any]]:
        """Get all active trading sessions"""
        async with self.postgres_pool.acquire() as conn:
            query = """
                SELECT session_id, start_time, total_trades, total_pnl,
                       status, ai_model_version, created_at
                FROM trading_sessions
                WHERE status = 'active'
                ORDER BY start_time DESC
            """
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def update_trading_session_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """Update trading session metrics"""
        async with self.postgres_pool.acquire() as conn:
            query = """
                UPDATE trading_sessions
                SET total_trades = $1, winning_trades = $2, losing_trades = $3,
                    total_pnl = $4, max_drawdown = $5, sharpe_ratio = $6,
                    updated_at = $7
                WHERE session_id = $8
            """
            await conn.execute(
                query,
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                metrics.get('total_pnl', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                datetime.utcnow(),
                session_id
            )

    # Market Data Management
    async def store_market_data(self, market_data: List[Dict[str, Any]]):
        """Store market data efficiently"""
        if not market_data:
            return

        async with self.postgres_pool.acquire() as conn:
            query = """
                INSERT INTO market_data
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume, bid, ask, spread)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    spread = EXCLUDED.spread
            """

            await conn.executemany(query, [
                (
                    data['symbol'], data['timestamp'], data['open_price'],
                    data['high_price'], data['low_price'], data['close_price'],
                    data.get('volume', 0), data.get('bid'), data.get('ask'),
                    data.get('spread')
                )
                for data in market_data
            ])

    async def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest market data for symbol"""
        async with self.postgres_pool.acquire() as conn:
            query = """
                SELECT * FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            rows = await conn.fetch(query, symbol, limit)
            return [dict(row) for row in rows]

    # Caching Methods (Redis)
    async def cache_set(self, key: str, value: str, expire_seconds: int = 3600):
        """Set cache value with expiration"""
        try:
            self.redis_client.setex(key, expire_seconds, value)
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")

    async def cache_get(self, key: str) -> Optional[str]:
        """Get cache value"""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    async def cache_delete(self, key: str):
        """Delete cache key"""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")

    # System Metrics
    async def record_system_metric(self, name: str, value: float, component: str = 'system', unit: str = None):
        """Record system metric"""
        async with self.postgres_pool.acquire() as conn:
            query = """
                INSERT INTO system_metrics (metric_name, metric_value, metric_unit, timestamp, component)
                VALUES ($1, $2, $3, $4, $5)
            """
            await conn.execute(query, name, value, unit, datetime.utcnow(), component)

    async def get_system_metrics(self, component: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics for time period"""
        async with self.postgres_pool.acquire() as conn:
            where_clause = "WHERE timestamp >= $1"
            params = [datetime.utcnow() - timedelta(hours=hours)]

            if component:
                where_clause += " AND component = $2"
                params.append(component)

            query = f"""
                SELECT metric_name, metric_value, metric_unit, timestamp, component
                FROM system_metrics
                {where_clause}
                ORDER BY timestamp DESC
            """

            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

# Global database manager instance
db_manager = DatabaseManager()

async def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    if not db_manager.postgres_pool:
        await db_manager.initialize()
    return db_manager

# Database dependency for FastAPI
async def get_db_session():
    """Get database session for FastAPI dependency injection"""
    async with db_manager.async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Health check
async def check_database_health() -> Dict[str, Any]:
    """Check database and Redis health"""
    health_status = {
        'postgresql': False,
        'redis': False,
        'timestamp': datetime.utcnow().isoformat()
    }

    try:
        # Check PostgreSQL
        async with db_manager.postgres_pool.acquire() as conn:
            result = await conn.fetchval('SELECT 1')
            health_status['postgresql'] = result == 1
    except Exception as e:
        health_status['postgresql_error'] = str(e)

    try:
        # Check Redis
        db_manager.redis_client.ping()
        health_status['redis'] = True
    except Exception as e:
        health_status['redis_error'] = str(e)

    return health_status
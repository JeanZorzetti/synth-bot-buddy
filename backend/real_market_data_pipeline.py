"""
🔄 REAL MARKET DATA PIPELINE
Complete pipeline for real-time market data aggregation and processing
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd
from enum import Enum
import aioredis
import aiofiles
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

from real_deriv_client import RealDerivWebSocketClient, RealTickData
from real_tick_processor import TickProcessor, ProcessedTickData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Qualidade dos dados"""
    EXCELLENT = "excellent"    # < 1% missing, < 10ms latency
    GOOD = "good"             # < 5% missing, < 50ms latency
    FAIR = "fair"             # < 10% missing, < 100ms latency
    POOR = "poor"             # > 10% missing or > 100ms latency


@dataclass
class MarketDataMetrics:
    """Métricas de qualidade dos dados de mercado"""
    symbol: str
    last_update: datetime
    tick_count_1m: int = 0
    tick_count_5m: int = 0
    tick_count_1h: int = 0
    missing_ticks_pct: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    data_quality: DataQuality = DataQuality.FAIR
    last_price: float = 0.0
    price_change_24h: float = 0.0
    volatility_24h: float = 0.0


@dataclass
class PipelineConfig:
    """Configuração do pipeline de dados"""
    symbols: List[str] = field(default_factory=list)
    enable_redis_cache: bool = True
    enable_file_backup: bool = True
    enable_data_validation: bool = True
    max_latency_ms: float = 100.0
    quality_threshold: float = 0.95  # 95% quality threshold
    buffer_size: int = 10000
    cache_ttl_seconds: int = 3600
    backup_interval_minutes: int = 15


class DataValidator:
    """Validador de qualidade dos dados"""

    def __init__(self):
        self.validation_rules = {
            'price_range': (0.1, 10000.0),     # Price should be reasonable
            'price_change_max': 0.10,           # Max 10% change in single tick
            'timestamp_tolerance': 30,          # Max 30 seconds old
            'spread_max': 0.001                 # Max 0.1% spread
        }

    def validate_tick(self, tick: RealTickData, previous_tick: Optional[RealTickData] = None) -> bool:
        """Validar tick recebido"""
        try:
            # Check price range
            if not (self.validation_rules['price_range'][0] <= tick.tick <= self.validation_rules['price_range'][1]):
                logger.warning(f"Price out of range: {tick.tick} for {tick.symbol}")
                return False

            # Check timestamp
            age_seconds = (datetime.now() - tick.timestamp).total_seconds()
            if age_seconds > self.validation_rules['timestamp_tolerance']:
                logger.warning(f"Tick too old: {age_seconds}s for {tick.symbol}")
                return False

            # Check price change if previous tick available
            if previous_tick:
                price_change = abs(tick.tick - previous_tick.tick) / previous_tick.tick
                if price_change > self.validation_rules['price_change_max']:
                    logger.warning(f"Large price change: {price_change:.4f} for {tick.symbol}")
                    return False

            # Check spread if available
            if tick.spread and tick.spread > self.validation_rules['spread_max']:
                logger.warning(f"Large spread: {tick.spread} for {tick.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating tick: {e}")
            return False

    def calculate_data_quality(self, metrics: MarketDataMetrics) -> DataQuality:
        """Calcular qualidade dos dados baseado nas métricas"""
        quality_score = 0.0

        # Missing ticks penalty
        if metrics.missing_ticks_pct < 0.01:
            quality_score += 0.4
        elif metrics.missing_ticks_pct < 0.05:
            quality_score += 0.3
        elif metrics.missing_ticks_pct < 0.10:
            quality_score += 0.2
        else:
            quality_score += 0.1

        # Latency penalty
        if metrics.avg_latency_ms < 10:
            quality_score += 0.4
        elif metrics.avg_latency_ms < 50:
            quality_score += 0.3
        elif metrics.avg_latency_ms < 100:
            quality_score += 0.2
        else:
            quality_score += 0.1

        # Tick frequency bonus
        if metrics.tick_count_1m > 50:  # High frequency
            quality_score += 0.2
        elif metrics.tick_count_1m > 20:  # Medium frequency
            quality_score += 0.15
        else:
            quality_score += 0.1

        # Determine quality level
        if quality_score >= 0.9:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.7:
            return DataQuality.GOOD
        elif quality_score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR


class DataCache:
    """Cache de dados com Redis"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_enabled = True

    async def connect(self):
        """Conectar ao Redis"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
            self.cache_enabled = False
            return False

    async def cache_tick(self, tick: ProcessedTickData, ttl: int = 3600):
        """Cachear tick processado"""
        if not self.cache_enabled or not self.redis_client:
            return

        try:
            key = f"tick:{tick.symbol}:{int(tick.timestamp.timestamp())}"
            data = {
                'symbol': tick.symbol,
                'price': tick.price,
                'timestamp': tick.timestamp.isoformat(),
                'rsi': tick.rsi,
                'macd': tick.macd,
                'volatility': tick.volatility_1m,
                'momentum': tick.momentum_score
            }

            await self.redis_client.setex(key, ttl, json.dumps(data))

        except Exception as e:
            logger.error(f"Error caching tick: {e}")

    async def get_cached_ticks(self, symbol: str, minutes: int = 60) -> List[Dict]:
        """Obter ticks cacheados"""
        if not self.cache_enabled or not self.redis_client:
            return []

        try:
            start_time = datetime.now() - timedelta(minutes=minutes)
            pattern = f"tick:{symbol}:*"

            keys = await self.redis_client.keys(pattern)
            ticks = []

            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    tick_data = json.loads(data)
                    tick_time = datetime.fromisoformat(tick_data['timestamp'])

                    if tick_time >= start_time:
                        ticks.append(tick_data)

            # Sort by timestamp
            ticks.sort(key=lambda x: x['timestamp'])
            return ticks

        except Exception as e:
            logger.error(f"Error getting cached ticks: {e}")
            return []

    async def cache_metrics(self, metrics: MarketDataMetrics):
        """Cachear métricas de mercado"""
        if not self.cache_enabled or not self.redis_client:
            return

        try:
            key = f"metrics:{metrics.symbol}"
            data = {
                'symbol': metrics.symbol,
                'last_update': metrics.last_update.isoformat(),
                'tick_count_1m': metrics.tick_count_1m,
                'tick_count_5m': metrics.tick_count_5m,
                'missing_ticks_pct': metrics.missing_ticks_pct,
                'avg_latency_ms': metrics.avg_latency_ms,
                'data_quality': metrics.data_quality.value,
                'last_price': metrics.last_price,
                'volatility_24h': metrics.volatility_24h
            }

            await self.redis_client.setex(key, 300, json.dumps(data))  # 5 minutes TTL

        except Exception as e:
            logger.error(f"Error caching metrics: {e}")

    async def disconnect(self):
        """Desconectar do Redis"""
        if self.redis_client:
            await self.redis_client.close()


class DataBackup:
    """Sistema de backup de dados"""

    def __init__(self, backup_dir: str = "data_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_enabled = True

    async def backup_ticks(self, symbol: str, ticks: List[ProcessedTickData]):
        """Fazer backup dos ticks"""
        if not self.backup_enabled or not ticks:
            return

        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.backup_dir / f"{symbol}_ticks_{timestamp}.json"

            # Convert ticks to dict
            data = []
            for tick in ticks:
                tick_dict = {
                    'symbol': tick.symbol,
                    'timestamp': tick.timestamp.isoformat(),
                    'price': tick.price,
                    'price_change': tick.price_change,
                    'price_velocity': tick.price_velocity,
                    'volatility_1m': tick.volatility_1m,
                    'rsi': tick.rsi,
                    'macd': tick.macd,
                    'momentum_score': tick.momentum_score,
                    'trend_strength': tick.trend_strength
                }
                data.append(tick_dict)

            # Write to file
            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(data, indent=2))

            logger.info(f"Backed up {len(ticks)} ticks for {symbol} to {filename}")

        except Exception as e:
            logger.error(f"Error backing up ticks: {e}")

    async def load_backup(self, symbol: str, date: str) -> List[Dict]:
        """Carregar backup de data específica"""
        try:
            pattern = f"{symbol}_ticks_{date}*.json"
            backup_files = list(self.backup_dir.glob(pattern))

            if not backup_files:
                logger.warning(f"No backup files found for {symbol} on {date}")
                return []

            # Load most recent file
            latest_file = max(backup_files, key=lambda f: f.stat().st_mtime)

            async with aiofiles.open(latest_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            logger.info(f"Loaded {len(data)} ticks from backup: {latest_file}")
            return data

        except Exception as e:
            logger.error(f"Error loading backup: {e}")
            return []


class RealMarketDataPipeline:
    """Pipeline completo de dados de mercado em tempo real"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Core components
        self.deriv_client = None
        self.tick_processor = TickProcessor(buffer_size=config.buffer_size)
        self.data_validator = DataValidator()
        self.data_cache = DataCache() if config.enable_redis_cache else None
        self.data_backup = DataBackup() if config.enable_file_backup else None

        # Data tracking
        self.metrics: Dict[str, MarketDataMetrics] = {}
        self.last_ticks: Dict[str, RealTickData] = {}
        self.latency_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Pipeline state
        self.is_running = False
        self.start_time = None
        self.processed_tick_count = 0
        self.error_count = 0

        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        # Thread pool for heavy computations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def initialize(self, app_id: str, api_token: str = None) -> bool:
        """Inicializar pipeline"""
        try:
            logger.info("Initializing Real Market Data Pipeline...")

            # Initialize Deriv client
            self.deriv_client = RealDerivWebSocketClient(app_id, api_token)

            # Setup callbacks
            self.deriv_client.add_tick_callback(self._on_raw_tick)
            self.tick_processor.add_processed_tick_callback(self._on_processed_tick)

            # Connect to Redis cache
            if self.data_cache:
                await self.data_cache.connect()

            # Connect to Deriv
            connected = await self.deriv_client.connect()
            if not connected:
                logger.error("Failed to connect to Deriv WebSocket")
                return False

            # Subscribe to symbols
            for symbol in self.config.symbols:
                success = await self.deriv_client.subscribe_ticks(symbol)
                if success:
                    logger.info(f"Subscribed to {symbol}")
                    # Initialize metrics
                    self.metrics[symbol] = MarketDataMetrics(
                        symbol=symbol,
                        last_update=datetime.now()
                    )
                else:
                    logger.error(f"Failed to subscribe to {symbol}")

            # Start background tasks
            await self._start_background_tasks()

            self.is_running = True
            self.start_time = datetime.now()

            logger.info("Real Market Data Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            return False

    async def _start_background_tasks(self):
        """Iniciar tarefas em background"""
        # Metrics calculation task
        task1 = asyncio.create_task(self._metrics_calculator())
        self.background_tasks.add(task1)

        # Data backup task
        if self.data_backup:
            task2 = asyncio.create_task(self._backup_scheduler())
            self.background_tasks.add(task2)

        # Health monitoring task
        task3 = asyncio.create_task(self._health_monitor())
        self.background_tasks.add(task3)

        # Quality monitoring task
        task4 = asyncio.create_task(self._quality_monitor())
        self.background_tasks.add(task4)

    async def _on_raw_tick(self, tick: RealTickData):
        """Processar tick raw recebido"""
        try:
            receive_time = datetime.now()
            latency_ms = (receive_time - tick.timestamp).total_seconds() * 1000

            # Track latency
            self.latency_tracking[tick.symbol].append(latency_ms)

            # Validate data if enabled
            if self.config.enable_data_validation:
                previous_tick = self.last_ticks.get(tick.symbol)
                if not self.data_validator.validate_tick(tick, previous_tick):
                    logger.warning(f"Invalid tick rejected for {tick.symbol}")
                    return

            # Store last tick
            self.last_ticks[tick.symbol] = tick

            # Process tick
            await self.tick_processor.process_real_tick(tick)

        except Exception as e:
            logger.error(f"Error processing raw tick: {e}")
            self.error_count += 1

    async def _on_processed_tick(self, processed_tick: ProcessedTickData):
        """Processar tick já processado"""
        try:
            self.processed_tick_count += 1

            # Cache processed tick
            if self.data_cache:
                await self.data_cache.cache_tick(processed_tick, self.config.cache_ttl_seconds)

            # Update metrics
            if processed_tick.symbol in self.metrics:
                self.metrics[processed_tick.symbol].last_update = processed_tick.timestamp
                self.metrics[processed_tick.symbol].last_price = processed_tick.price

            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(processed_tick)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")

        except Exception as e:
            logger.error(f"Error handling processed tick: {e}")
            self.error_count += 1

    async def _metrics_calculator(self):
        """Calcular métricas de qualidade dos dados"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Calculate every minute

                for symbol in self.config.symbols:
                    if symbol not in self.metrics:
                        continue

                    metrics = self.metrics[symbol]

                    # Get tick counts
                    processed_ticks = self.tick_processor.get_processed_history(symbol, 3600)  # Last hour
                    now = datetime.now()

                    # Count ticks in different windows
                    ticks_1m = len([t for t in processed_ticks if (now - t.timestamp).total_seconds() <= 60])
                    ticks_5m = len([t for t in processed_ticks if (now - t.timestamp).total_seconds() <= 300])
                    ticks_1h = len(processed_ticks)

                    # Calculate latency stats
                    if symbol in self.latency_tracking and self.latency_tracking[symbol]:
                        latencies = list(self.latency_tracking[symbol])
                        avg_latency = np.mean(latencies)
                        max_latency = np.max(latencies)
                    else:
                        avg_latency = max_latency = 0

                    # Estimate missing ticks (assume 1 tick per second ideal)
                    expected_ticks_1m = 60
                    missing_pct = max(0, (expected_ticks_1m - ticks_1m) / expected_ticks_1m)

                    # Update metrics
                    metrics.tick_count_1m = ticks_1m
                    metrics.tick_count_5m = ticks_5m
                    metrics.tick_count_1h = ticks_1h
                    metrics.missing_ticks_pct = missing_pct
                    metrics.avg_latency_ms = avg_latency
                    metrics.max_latency_ms = max_latency

                    # Calculate data quality
                    metrics.data_quality = self.data_validator.calculate_data_quality(metrics)

                    # Cache metrics
                    if self.data_cache:
                        await self.data_cache.cache_metrics(metrics)

            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")

    async def _backup_scheduler(self):
        """Agendar backups automáticos"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.backup_interval_minutes * 60)

                for symbol in self.config.symbols:
                    # Get recent ticks for backup
                    recent_ticks = self.tick_processor.get_processed_history(symbol, 1000)
                    if recent_ticks:
                        await self.data_backup.backup_ticks(symbol, recent_ticks)

                logger.info(f"Completed scheduled backup for {len(self.config.symbols)} symbols")

            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")

    async def _health_monitor(self):
        """Monitorar saúde do pipeline"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check if we're receiving data
                now = datetime.now()
                stale_symbols = []

                for symbol, metrics in self.metrics.items():
                    time_since_last = (now - metrics.last_update).total_seconds()
                    if time_since_last > 60:  # No data for 1 minute
                        stale_symbols.append(symbol)

                if stale_symbols:
                    logger.warning(f"Stale data detected for symbols: {stale_symbols}")

                # Check system resources
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()

                if memory_usage > 90:
                    logger.warning(f"High memory usage: {memory_usage}%")

                if cpu_usage > 90:
                    logger.warning(f"High CPU usage: {cpu_usage}%")

                # Check error rate
                if self.processed_tick_count > 0:
                    error_rate = self.error_count / self.processed_tick_count
                    if error_rate > 0.05:  # More than 5% errors
                        logger.warning(f"High error rate: {error_rate:.2%}")

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _quality_monitor(self):
        """Monitorar qualidade dos dados"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                poor_quality_symbols = []

                for symbol, metrics in self.metrics.items():
                    if metrics.data_quality in [DataQuality.POOR, DataQuality.FAIR]:
                        poor_quality_symbols.append((symbol, metrics.data_quality))

                if poor_quality_symbols:
                    logger.warning(f"Poor data quality detected: {poor_quality_symbols}")

                    # Notify quality callbacks
                    for callback in self.quality_callbacks:
                        try:
                            await callback(poor_quality_symbols)
                        except Exception as e:
                            logger.error(f"Error in quality callback: {e}")

            except Exception as e:
                logger.error(f"Error in quality monitor: {e}")

    # Public API Methods

    def get_latest_data(self, symbol: str) -> Optional[ProcessedTickData]:
        """Obter dados mais recentes de um símbolo"""
        return self.tick_processor.get_latest_processed_tick(symbol)

    def get_historical_data(self, symbol: str, minutes: int = 60) -> List[ProcessedTickData]:
        """Obter dados históricos"""
        all_ticks = self.tick_processor.get_processed_history(symbol, count=minutes*60)
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [tick for tick in all_ticks if tick.timestamp >= cutoff_time]

    def get_dataframe(self, symbol: str, minutes: int = 60) -> pd.DataFrame:
        """Obter dados como DataFrame"""
        return self.tick_processor.get_feature_dataframe(symbol, count=minutes*60)

    def get_pipeline_metrics(self) -> Dict:
        """Obter métricas do pipeline"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'symbols_tracked': len(self.config.symbols),
            'processed_tick_count': self.processed_tick_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.processed_tick_count),
            'processing_stats': self.tick_processor.get_processing_stats(),
            'symbol_metrics': {symbol: {
                'data_quality': metrics.data_quality.value,
                'tick_count_1m': metrics.tick_count_1m,
                'avg_latency_ms': metrics.avg_latency_ms,
                'last_price': metrics.last_price
            } for symbol, metrics in self.metrics.items()}
        }

    def add_data_callback(self, callback: Callable[[ProcessedTickData], None]):
        """Adicionar callback para novos dados"""
        self.data_callbacks.append(callback)

    def add_quality_callback(self, callback: Callable[[List], None]):
        """Adicionar callback para alertas de qualidade"""
        self.quality_callbacks.append(callback)

    async def stop(self):
        """Parar pipeline"""
        logger.info("Stopping Real Market Data Pipeline...")

        self.is_running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Disconnect from services
        if self.deriv_client:
            await self.deriv_client.disconnect()

        if self.data_cache:
            await self.data_cache.disconnect()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Real Market Data Pipeline stopped")


# 🧪 Função de teste
async def test_market_data_pipeline():
    """Testar pipeline de dados de mercado"""
    config = PipelineConfig(
        symbols=["R_100", "R_50", "R_25"],
        enable_redis_cache=False,  # Disable for testing
        enable_file_backup=True,
        enable_data_validation=True,
        max_latency_ms=100.0
    )

    pipeline = RealMarketDataPipeline(config)

    # Callback para novos dados
    async def on_new_data(processed_tick: ProcessedTickData):
        print(f"📊 New data: {processed_tick.symbol} = {processed_tick.price:.5f}")
        print(f"   Quality metrics: RSI={processed_tick.rsi:.2f}, Vol={processed_tick.volatility_1m:.4f}")

    # Callback para qualidade
    async def on_quality_alert(poor_quality_symbols):
        print(f"⚠️  Quality alert: {poor_quality_symbols}")

    pipeline.add_data_callback(on_new_data)
    pipeline.add_quality_callback(on_quality_alert)

    # Initialize pipeline
    APP_ID = "1089"  # Use your app ID
    success = await pipeline.initialize(APP_ID)

    if success:
        print("✅ Pipeline initialized successfully")

        # Run for 5 minutes
        await asyncio.sleep(300)

        # Get metrics
        metrics = pipeline.get_pipeline_metrics()
        print(f"\n📈 Pipeline Metrics:")
        print(f"   Processed ticks: {metrics['processed_tick_count']}")
        print(f"   Error rate: {metrics['error_rate']:.2%}")
        print(f"   Uptime: {metrics['uptime_seconds']:.0f}s")

        # Get data for analysis
        for symbol in config.symbols:
            df = pipeline.get_dataframe(symbol, minutes=5)
            print(f"\n📊 {symbol} Data: {len(df)} ticks")
            if not df.empty:
                print(df.tail(3))

        await pipeline.stop()
    else:
        print("❌ Failed to initialize pipeline")


if __name__ == "__main__":
    print("🔄 TESTING REAL MARKET DATA PIPELINE")
    print("=" * 45)
    print("⚠️  This will connect to real Deriv API")
    print("=" * 45)

    # Uncomment to run test
    # asyncio.run(test_market_data_pipeline())
"""
Real Logging and Monitoring System - Phase 12 Real Infrastructure
Sistema completo de logging e monitoramento para produção
"""

import os
import json
import asyncio
import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import psutil
import traceback
from pathlib import Path

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogComponent(Enum):
    API = "api"
    DATABASE = "database"
    AI_ENGINE = "ai_engine"
    TRADING = "trading"
    WEBSOCKET = "websocket"
    AUTHENTICATION = "authentication"
    STRATEGY = "strategy"
    PERFORMANCE = "performance"
    SYSTEM = "system"

class RealLoggingSystem:
    def __init__(self):
        self.log_dir = Path(os.getenv('LOG_DIR', './logs'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.max_log_size = int(os.getenv('MAX_LOG_SIZE_MB', '100')) * 1024 * 1024  # 100MB
        self.backup_count = int(os.getenv('LOG_BACKUP_COUNT', '10'))

        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.loggers = {}
        self.system_metrics_logger = None
        self.trading_logger = None
        self.error_logger = None

        # Initialize logging system
        self._setup_logging_system()

    def _setup_logging_system(self):
        """Setup comprehensive logging system"""

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.handlers.RotatingFileHandler(
                    self.log_dir / 'application.log',
                    maxBytes=self.max_log_size,
                    backupCount=self.backup_count
                )
            ]
        )

        # Setup component-specific loggers
        for component in LogComponent:
            logger = self._create_component_logger(component)
            self.loggers[component.value] = logger

        # Setup specialized loggers
        self.system_metrics_logger = self._create_metrics_logger()
        self.trading_logger = self._create_trading_logger()
        self.error_logger = self._create_error_logger()

    def _create_component_logger(self, component: LogComponent) -> logging.Logger:
        """Create logger for specific component"""
        logger_name = f"trading_bot.{component.value}"
        logger = logging.getLogger(logger_name)

        # File handler for component-specific logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{component.value}.log",
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )

        # JSON formatter for structured logging
        formatter = JsonFormatter()
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(getattr(logging, self.log_level))

        return logger

    def _create_metrics_logger(self) -> logging.Logger:
        """Create system metrics logger"""
        logger = logging.getLogger("trading_bot.metrics")

        # Metrics-specific handler
        metrics_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / "system_metrics.log",
            when='h',  # Rotate hourly
            interval=1,
            backupCount=24  # Keep 24 hours
        )

        metrics_formatter = MetricsFormatter()
        metrics_handler.setFormatter(metrics_formatter)

        logger.addHandler(metrics_handler)
        logger.setLevel(logging.INFO)

        return logger

    def _create_trading_logger(self) -> logging.Logger:
        """Create trading-specific logger"""
        logger = logging.getLogger("trading_bot.trades")

        # Trading-specific handler
        trading_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading_activity.log",
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )

        trading_formatter = TradingFormatter()
        trading_handler.setFormatter(trading_formatter)

        logger.addHandler(trading_handler)
        logger.setLevel(logging.INFO)

        return logger

    def _create_error_logger(self) -> logging.Logger:
        """Create error-specific logger"""
        logger = logging.getLogger("trading_bot.errors")

        # Error-specific handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )

        error_formatter = ErrorFormatter()
        error_handler.setFormatter(error_formatter)

        logger.addHandler(error_handler)
        logger.setLevel(logging.WARNING)

        return logger

    def log(
        self,
        component: LogComponent,
        level: LogLevel,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log message with structured data"""
        logger = self.loggers.get(component.value)
        if not logger:
            logger = logging.getLogger("trading_bot.general")

        # Prepare log data
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component.value,
            'level': level.value,
            'message': message,
            'user_id': user_id,
            'session_id': session_id,
            'extra_data': extra_data or {}
        }

        # Log with appropriate level
        log_level = getattr(logging, level.value)
        logger.log(log_level, message, extra=log_data)

    def log_trading_activity(
        self,
        activity_type: str,
        symbol: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Log trading-specific activity"""
        trading_data = {
            'activity_type': activity_type,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data': data
        }

        self.trading_logger.info(f"Trading activity: {activity_type}", extra=trading_data)

    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics"""
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }

        self.system_metrics_logger.info("System metrics", extra=metrics_data)

    def log_error(
        self,
        component: LogComponent,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """Log error with full context"""
        error_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'user_id': user_id
        }

        self.error_logger.error(f"Error in {component.value}: {str(error)}", extra=error_data)

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time_ms: float,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ):
        """Log API request details"""
        api_data = {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time_ms': response_time_ms,
            'user_id': user_id,
            'request_data': request_data or {}
        }

        level = LogLevel.INFO if status_code < 400 else LogLevel.WARNING
        self.log(
            LogComponent.API,
            level,
            f"{method} {endpoint} - {status_code} ({response_time_ms:.2f}ms)",
            api_data,
            user_id
        )

    def log_ai_prediction(
        self,
        model_id: str,
        symbol: str,
        prediction: Dict[str, Any],
        confidence: float,
        execution_time_ms: float
    ):
        """Log AI prediction details"""
        ai_data = {
            'model_id': model_id,
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'execution_time_ms': execution_time_ms
        }

        self.log(
            LogComponent.AI_ENGINE,
            LogLevel.INFO,
            f"AI prediction for {symbol}: {prediction.get('signal', 'unknown')}",
            ai_data
        )

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)

            # Network metrics (if available)
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)

            # Database metrics (if connected)
            db_metrics = await self._get_database_metrics()

            # Redis metrics (if connected)
            redis_metrics = await self._get_redis_metrics()

            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_gb': round(memory_available_gb, 2)
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': round(disk_free_gb, 2)
                },
                'network': {
                    'sent_mb': round(network_sent_mb, 2),
                    'received_mb': round(network_recv_mb, 2)
                },
                'database': db_metrics,
                'redis': redis_metrics
            }

            # Log metrics
            self.log_system_metrics(metrics)

            return metrics

        except Exception as e:
            self.log_error(LogComponent.SYSTEM, e, {'action': 'collect_system_metrics'})
            return {}

    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        try:
            from database_config import get_db_manager

            db_manager = await get_db_manager()

            # Get connection pool stats
            pool = db_manager.postgres_pool
            if pool:
                return {
                    'pool_size': pool.get_size(),
                    'pool_max_size': pool.get_max_size(),
                    'pool_min_size': pool.get_min_size(),
                    'connections_in_use': pool.get_size() - pool.get_idle_size(),
                    'idle_connections': pool.get_idle_size()
                }

        except Exception as e:
            self.log_error(LogComponent.DATABASE, e, {'action': 'get_database_metrics'})

        return {'status': 'unavailable'}

    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis performance metrics"""
        try:
            from redis_cache_manager import get_cache_manager

            cache_manager = await get_cache_manager()
            return await cache_manager.get_cache_stats()

        except Exception as e:
            self.log_error(LogComponent.DATABASE, e, {'action': 'get_redis_metrics'})

        return {'status': 'unavailable'}

    def get_recent_logs(
        self,
        component: Optional[LogComponent] = None,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            log_file = self.log_dir / "application.log"
            if component:
                log_file = self.log_dir / f"{component.value}.log"

            recent_logs = []
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-limit:]):
                        try:
                            log_entry = json.loads(line.strip())
                            if level is None or log_entry.get('level') == level.value:
                                recent_logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue

            return recent_logs[:limit]

        except Exception as e:
            self.log_error(LogComponent.SYSTEM, e, {'action': 'get_recent_logs'})
            return []

    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            for log_file in self.log_dir.glob('*.log*'):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        self.log(
                            LogComponent.SYSTEM,
                            LogLevel.INFO,
                            f"Deleted old log file: {log_file.name}"
                        )
                except Exception as e:
                    self.log_error(LogComponent.SYSTEM, e, {'file': str(log_file)})

        except Exception as e:
            self.log_error(LogComponent.SYSTEM, e, {'action': 'cleanup_old_logs'})

# Custom Formatters

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra data if available
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, default=str)

class MetricsFormatter(logging.Formatter):
    """Specialized formatter for metrics logging"""

    def format(self, record):
        if hasattr(record, 'metrics'):
            return json.dumps(record.metrics, default=str)
        return record.getMessage()

class TradingFormatter(logging.Formatter):
    """Specialized formatter for trading activity"""

    def format(self, record):
        trading_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'message': record.getMessage()
        }

        # Add trading-specific data
        for attr in ['activity_type', 'symbol', 'user_id', 'data']:
            if hasattr(record, attr):
                trading_entry[attr] = getattr(record, attr)

        return json.dumps(trading_entry, default=str)

class ErrorFormatter(logging.Formatter):
    """Specialized formatter for error logging"""

    def format(self, record):
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add error-specific data
        for attr in ['component', 'error_type', 'error_message', 'traceback', 'context', 'user_id']:
            if hasattr(record, attr):
                error_entry[attr] = getattr(record, attr)

        return json.dumps(error_entry, default=str)

# Global logging system instance
logging_system = RealLoggingSystem()

def get_logger(component: LogComponent) -> logging.Logger:
    """Get component-specific logger"""
    return logging_system.loggers.get(component.value, logging.getLogger("trading_bot.general"))

# Async monitoring task
async def start_system_monitoring(interval_seconds: int = 60):
    """Start continuous system monitoring"""
    while True:
        try:
            await logging_system.collect_system_metrics()
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            logging_system.log_error(LogComponent.SYSTEM, e, {'task': 'system_monitoring'})
            await asyncio.sleep(interval_seconds)

# Context manager for performance logging
class PerformanceLogger:
    def __init__(self, component: LogComponent, operation: str, user_id: Optional[str] = None):
        self.component = component
        self.operation = operation
        self.user_id = user_id
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000

            if exc_type is None:
                # Operation completed successfully
                logging_system.log(
                    self.component,
                    LogLevel.INFO,
                    f"Operation completed: {self.operation}",
                    {'duration_ms': duration_ms, 'operation': self.operation},
                    self.user_id
                )
            else:
                # Operation failed
                logging_system.log_error(
                    self.component,
                    exc_val,
                    {'operation': self.operation, 'duration_ms': duration_ms},
                    self.user_id
                )
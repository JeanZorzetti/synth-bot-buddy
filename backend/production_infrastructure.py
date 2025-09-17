"""
Production Infrastructure Manager - Phase 12 Real Infrastructure
Sistema completo de infraestrutura para ambiente de produção
"""

import os
import asyncio
import signal
import uvloop
from typing import Dict, List, Any, Optional
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import aiofiles
import yaml
from pathlib import Path

# FastAPI and dependencies
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Custom imports
from database_config import get_db_manager, check_database_health
from redis_cache_manager import get_cache_manager
from real_logging_system import logging_system, LogComponent, LogLevel, start_system_monitoring

class ProductionConfig:
    """Production environment configuration"""

    def __init__(self):
        # Environment
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'

        # Server configuration
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '8000'))
        self.workers = int(os.getenv('WORKERS', str(multiprocessing.cpu_count())))

        # Security
        self.allowed_hosts = os.getenv('ALLOWED_HOSTS', '*').split(',')
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

        # Performance
        self.max_connections = int(os.getenv('MAX_CONNECTIONS', '1000'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.keep_alive_timeout = int(os.getenv('KEEP_ALIVE_TIMEOUT', '5'))

        # Monitoring
        self.metrics_enabled = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))

        # Rate limiting
        self.rate_limit_per_minute = int(os.getenv('RATE_LIMIT_PER_MINUTE', '1000'))
        self.rate_limit_burst = int(os.getenv('RATE_LIMIT_BURST', '100'))

class ProductionApp:
    """Production-ready FastAPI application"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.app = FastAPI(
            title="AI Trading Bot API",
            description="Production API for AI Trading Bot",
            version="1.0.0",
            debug=config.debug,
            docs_url="/docs" if config.debug else None,
            redoc_url="/redoc" if config.debug else None
        )

        # Rate limiting storage
        self.rate_limit_storage = {}

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_security()
        self._setup_routes()

    def _setup_middleware(self):
        """Configure production middleware"""

        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Trusted hosts
        if '*' not in self.config.allowed_hosts:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.allowed_hosts
            )

        # Request logging middleware
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            start_time = datetime.utcnow()

            try:
                response = await call_next(request)
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                logging_system.log_api_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    response_time_ms=duration_ms,
                    user_id=getattr(request.state, 'user_id', None)
                )

                return response

            except Exception as e:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                logging_system.log_error(
                    LogComponent.API,
                    e,
                    {
                        'method': request.method,
                        'endpoint': request.url.path,
                        'duration_ms': duration_ms
                    }
                )
                raise

        # Rate limiting middleware
        @self.app.middleware("http")
        async def rate_limiting_middleware(request: Request, call_next):
            if not await self._check_rate_limit(request):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            return await call_next(request)

    def _setup_security(self):
        """Configure security measures"""
        self.security = HTTPBearer(auto_error=False)

    def _setup_routes(self):
        """Setup production API routes"""

        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check endpoint"""
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'environment': self.config.environment,
                'version': '1.0.0'
            }

            # Check database health
            try:
                db_health = await check_database_health()
                health_status['database'] = db_health
            except Exception as e:
                health_status['database'] = {'status': 'unhealthy', 'error': str(e)}
                health_status['status'] = 'degraded'

            # Check Redis health
            try:
                cache_manager = await get_cache_manager()
                redis_health = await cache_manager.health_check()
                health_status['redis'] = redis_health
            except Exception as e:
                health_status['redis'] = {'status': 'unhealthy', 'error': str(e)}
                health_status['status'] = 'degraded'

            # Check system metrics
            try:
                system_metrics = await logging_system.collect_system_metrics()
                health_status['system'] = {
                    'cpu_percent': system_metrics.get('cpu', {}).get('percent', 0),
                    'memory_percent': system_metrics.get('memory', {}).get('percent', 0),
                    'disk_percent': system_metrics.get('disk', {}).get('percent', 0)
                }

                # Check if system is under stress
                if (system_metrics.get('cpu', {}).get('percent', 0) > 90 or
                    system_metrics.get('memory', {}).get('percent', 0) > 90):
                    health_status['status'] = 'degraded'

            except Exception as e:
                health_status['system'] = {'status': 'unhealthy', 'error': str(e)}

            status_code = 200 if health_status['status'] == 'healthy' else 503
            return health_status

        @self.app.get("/routes")
        async def list_routes():
            """List all available API routes for debugging"""
            if not self.config.debug:
                raise HTTPException(status_code=404, detail="Not found")

            routes = []
            for route in self.app.routes:
                if hasattr(route, 'methods'):
                    routes.append({
                        'path': route.path,
                        'methods': list(route.methods),
                        'name': route.name
                    })

            return {'routes': routes}

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics (protected endpoint)"""
            if not self.config.metrics_enabled:
                raise HTTPException(status_code=404, detail="Not found")

            try:
                metrics = await logging_system.collect_system_metrics()
                return metrics
            except Exception as e:
                logging_system.log_error(LogComponent.API, e, {'endpoint': '/metrics'})
                raise HTTPException(status_code=500, detail="Failed to collect metrics")

        @self.app.get("/logs")
        async def get_recent_logs(
            component: Optional[str] = None,
            level: Optional[str] = None,
            limit: int = 100
        ):
            """Get recent log entries (debug only)"""
            if not self.config.debug:
                raise HTTPException(status_code=404, detail="Not found")

            try:
                from real_logging_system import LogComponent, LogLevel

                log_component = None
                if component:
                    try:
                        log_component = LogComponent(component)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

                log_level = None
                if level:
                    try:
                        log_level = LogLevel(level.upper())
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid level: {level}")

                logs = logging_system.get_recent_logs(log_component, log_level, limit)
                return {'logs': logs}

            except Exception as e:
                logging_system.log_error(LogComponent.API, e, {'endpoint': '/logs'})
                raise HTTPException(status_code=500, detail="Failed to get logs")

    async def _check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limits"""
        client_ip = request.client.host
        current_time = datetime.utcnow()

        # Simple in-memory rate limiting (in production, use Redis)
        if client_ip not in self.rate_limit_storage:
            self.rate_limit_storage[client_ip] = []

        # Clean old requests
        self.rate_limit_storage[client_ip] = [
            req_time for req_time in self.rate_limit_storage[client_ip]
            if (current_time - req_time).seconds < 60
        ]

        # Check rate limit
        if len(self.rate_limit_storage[client_ip]) >= self.config.rate_limit_per_minute:
            return False

        # Add current request
        self.rate_limit_storage[client_ip].append(current_time)
        return True

class ProductionManager:
    """Production environment manager"""

    def __init__(self):
        self.config = ProductionConfig()
        self.app_instance = ProductionApp(self.config)
        self.app = self.app_instance.app

        # Background tasks
        self.background_tasks = []
        self.shutdown_event = asyncio.Event()

    async def startup(self):
        """Initialize production environment"""
        try:
            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Starting production environment initialization"
            )

            # Initialize database
            db_manager = await get_db_manager()
            if not db_manager:
                raise RuntimeError("Failed to initialize database")

            # Initialize Redis cache
            cache_manager = await get_cache_manager()
            if not cache_manager:
                raise RuntimeError("Failed to initialize Redis cache")

            # Start background monitoring
            if self.config.metrics_enabled:
                monitoring_task = asyncio.create_task(
                    start_system_monitoring(self.config.health_check_interval)
                )
                self.background_tasks.append(monitoring_task)

            # Start log cleanup task
            cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self.background_tasks.append(cleanup_task)

            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Production environment initialized successfully"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'startup'}
            )
            raise

    async def shutdown(self):
        """Graceful shutdown of production environment"""
        try:
            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Initiating graceful shutdown"
            )

            # Signal shutdown to background tasks
            self.shutdown_event.set()

            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Close database connections
            try:
                db_manager = await get_db_manager()
                await db_manager.close()
            except Exception as e:
                logging_system.log_error(LogComponent.DATABASE, e, {'action': 'shutdown'})

            # Close Redis connections
            try:
                cache_manager = await get_cache_manager()
                await cache_manager.close()
            except Exception as e:
                logging_system.log_error(LogComponent.DATABASE, e, {'action': 'shutdown'})

            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Graceful shutdown completed"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'shutdown'}
            )

    async def _periodic_cleanup(self):
        """Periodic maintenance tasks"""
        while not self.shutdown_event.is_set():
            try:
                # Clean old logs every 24 hours
                await logging_system.cleanup_old_logs()

                # Clear rate limit storage
                current_time = datetime.utcnow()
                for client_ip in list(self.app_instance.rate_limit_storage.keys()):
                    self.app_instance.rate_limit_storage[client_ip] = [
                        req_time for req_time in self.app_instance.rate_limit_storage[client_ip]
                        if (current_time - req_time).seconds < 60
                    ]

                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.SYSTEM,
                    e,
                    {'task': 'periodic_cleanup'}
                )
                await asyncio.sleep(3600)  # Continue after error

# Signal handlers for graceful shutdown
def setup_signal_handlers(production_manager: ProductionManager):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        asyncio.create_task(production_manager.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# Production server runner
async def run_production_server():
    """Run production server with optimal configuration"""

    # Use uvloop for better performance on Linux
    if hasattr(asyncio, 'set_event_loop_policy'):
        try:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass  # uvloop not available on this platform

    # Initialize production manager
    production_manager = ProductionManager()

    # Setup signal handlers
    setup_signal_handlers(production_manager)

    # Startup
    await production_manager.startup()

    try:
        # Configure uvicorn
        config = uvicorn.Config(
            app=production_manager.app,
            host=production_manager.config.host,
            port=production_manager.config.port,
            workers=1,  # Use 1 worker with async, scale with Docker/K8s
            loop="asyncio",
            http="httptools",
            ws="websockets",
            lifespan="on",
            access_log=not production_manager.config.debug,
            use_colors=False,
            timeout_keep_alive=production_manager.config.keep_alive_timeout,
            timeout_graceful_shutdown=30
        )

        server = uvicorn.Server(config)

        logging_system.log(
            LogComponent.SYSTEM,
            LogLevel.INFO,
            f"Starting production server on {production_manager.config.host}:{production_manager.config.port}"
        )

        await server.serve()

    except Exception as e:
        logging_system.log_error(
            LogComponent.SYSTEM,
            e,
            {'action': 'run_server'}
        )
        raise
    finally:
        await production_manager.shutdown()

# Dockerfile generator
def generate_dockerfile():
    """Generate production Dockerfile"""
    dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "production_infrastructure:production_manager.app", "--host", "0.0.0.0", "--port", "8000"]
'''

    return dockerfile_content

# Docker Compose generator
def generate_docker_compose():
    """Generate production docker-compose.yml"""
    compose_content = '''version: '3.8'

services:
  trading-bot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://trading_user:trading_password@postgres:5432/trading_bot_db
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - CORS_ORIGINS=${CORS_ORIGINS}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=trading_bot_db
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD=trading_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
'''

    return compose_content

# Global production manager instance
production_manager = ProductionManager()

if __name__ == "__main__":
    # Run production server
    asyncio.run(run_production_server())
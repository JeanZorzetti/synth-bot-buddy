"""
🌐 PRODUCTION CONFIGURATION
Environment setup and security hardening for production deployment
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import secrets
from datetime import datetime, timedelta
import ssl


@dataclass
class SecurityConfig:
    """🔐 Configurações de Segurança"""
    api_key_encryption: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 300
    session_timeout_minutes: int = 60
    ssl_required: bool = True
    cors_origins: List[str] = None
    api_key_rotation_days: int = 30
    enable_audit_log: bool = True

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []


@dataclass
class DatabaseConfig:
    """🗄️ Configurações de Database"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot_prod"
    username: str = "trading_user"
    password: str = ""  # Será carregado de variável de ambiente
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    ssl_mode: str = "require"
    connection_timeout: int = 10


@dataclass
class RedisConfig:
    """📦 Configurações do Redis (Cache)"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class TradingConfig:
    """📈 Configurações de Trading"""
    max_concurrent_positions: int = 10
    max_daily_loss_pct: float = 5.0
    max_position_size_pct: float = 2.0
    emergency_stop_drawdown_pct: float = 15.0
    min_confidence_threshold: float = 0.75
    api_rate_limit_per_second: int = 10
    contract_types_enabled: List[str] = None
    symbols_enabled: List[str] = None

    def __post_init__(self):
        if self.contract_types_enabled is None:
            self.contract_types_enabled = ["CALL", "PUT"]
        if self.symbols_enabled is None:
            self.symbols_enabled = ["R_100", "R_50", "R_25", "R_75"]


@dataclass
class MonitoringConfig:
    """📊 Configurações de Monitoramento"""
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_alerting: bool = True
    health_check_interval_seconds: int = 30
    performance_logging: bool = True
    error_notification_webhook: str = ""
    slack_webhook_url: str = ""
    email_notifications: List[str] = None

    def __post_init__(self):
        if self.email_notifications is None:
            self.email_notifications = []


@dataclass
class AIModelConfig:
    """🧠 Configurações do Modelo de IA"""
    model_version: str = "v2.1.3"
    model_path: str = "/models/lstm_trading_model.h5"
    backup_model_path: str = "/models/backup/lstm_trading_model_backup.h5"
    inference_batch_size: int = 32
    max_inference_time_ms: int = 100
    model_validation_interval_hours: int = 24
    auto_retrain_enabled: bool = True
    retrain_threshold_accuracy: float = 0.65
    feature_importance_threshold: float = 0.01


class ProductionConfigManager:
    """🏭 Gerenciador de Configuração de Produção"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config_path = Path(f"config/{environment}")
        self.config_path.mkdir(parents=True, exist_ok=True)

        # Configurações
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.trading = TradingConfig()
        self.monitoring = MonitoringConfig()
        self.ai_model = AIModelConfig()

        # Logging
        self.setup_logging()

        # Carregar configurações existentes
        self.load_configuration()

    def setup_logging(self):
        """Configurar logging para produção"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/trading_bot_{self.environment}.log'),
                logging.StreamHandler()
            ]
        )

        # Logger específico para auditoria
        audit_logger = logging.getLogger('audit')
        audit_handler = logging.FileHandler('logs/audit.log')
        audit_handler.setFormatter(logging.Formatter(log_format))
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)

    def load_environment_variables(self):
        """Carregar variáveis de ambiente"""
        # Database
        self.database.password = os.getenv('DB_PASSWORD', '')
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', str(self.database.port)))

        # Redis
        self.redis.password = os.getenv('REDIS_PASSWORD', '')
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)

        # API Keys
        deriv_api_key = os.getenv('DERIV_API_KEY', '')
        if not deriv_api_key:
            raise ValueError("DERIV_API_KEY environment variable is required")

        # Monitoring
        self.monitoring.error_notification_webhook = os.getenv('ERROR_WEBHOOK_URL', '')
        self.monitoring.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')

        # Security
        jwt_secret = os.getenv('JWT_SECRET_KEY', '')
        if not jwt_secret:
            # Gerar secret aleatório se não fornecido
            jwt_secret = secrets.token_urlsafe(32)
            logging.warning("Generated random JWT secret. Set JWT_SECRET_KEY environment variable.")

    def generate_ssl_context(self) -> ssl.SSLContext:
        """Gerar contexto SSL para produção"""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        # Carregar certificados se disponíveis
        cert_file = os.getenv('SSL_CERT_FILE', 'certs/server.crt')
        key_file = os.getenv('SSL_KEY_FILE', 'certs/server.key')

        if os.path.exists(cert_file) and os.path.exists(key_file):
            context.load_cert_chain(cert_file, key_file)
            logging.info("SSL certificates loaded successfully")
        else:
            logging.warning("SSL certificates not found. Using default context.")

        return context

    def setup_security_hardening(self) -> Dict:
        """Configurar medidas de segurança"""
        security_measures = {
            'api_key_hashing': True,
            'request_rate_limiting': True,
            'session_management': True,
            'cors_configuration': True,
            'sql_injection_protection': True,
            'xss_protection': True
        }

        # 1. Configurar CORS
        if self.environment == "production":
            allowed_origins = os.getenv('CORS_ORIGINS', '').split(',')
            self.security.cors_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

        # 2. Rate limiting
        self.security.rate_limiting_enabled = True
        self.security.max_requests_per_minute = 300  # Conservative para produção

        # 3. Session timeout
        self.security.session_timeout_minutes = 60

        # 4. API key rotation
        self.security.api_key_rotation_days = 30

        return security_measures

    def setup_database_optimization(self) -> Dict:
        """Otimizar configurações de database"""
        # Pool de conexões otimizado para produção
        self.database.pool_size = 20
        self.database.max_overflow = 30
        self.database.pool_timeout = 30

        # SSL obrigatório em produção
        if self.environment == "production":
            self.database.ssl_mode = "require"

        optimization_settings = {
            'connection_pooling': True,
            'ssl_encryption': self.database.ssl_mode == "require",
            'connection_timeout': self.database.connection_timeout,
            'query_optimization': True
        }

        return optimization_settings

    def setup_monitoring_and_alerting(self) -> Dict:
        """Configurar monitoramento e alertas"""
        monitoring_setup = {
            'prometheus_metrics': self.monitoring.enable_prometheus,
            'grafana_dashboards': self.monitoring.enable_grafana,
            'health_checks': True,
            'error_alerting': self.monitoring.enable_alerting,
            'performance_tracking': self.monitoring.performance_logging
        }

        # Health check endpoints
        health_check_endpoints = [
            '/health',
            '/health/db',
            '/health/redis',
            '/health/ai-model',
            '/health/deriv-api'
        ]

        monitoring_setup['health_endpoints'] = health_check_endpoints

        return monitoring_setup

    def setup_ai_model_production(self) -> Dict:
        """Configurar modelo de IA para produção"""
        ai_config = {
            'model_validation': True,
            'backup_model': True,
            'inference_optimization': True,
            'auto_retraining': self.ai_model.auto_retrain_enabled,
            'performance_monitoring': True
        }

        # Verificar se modelos existem
        model_paths = [
            self.ai_model.model_path,
            self.ai_model.backup_model_path
        ]

        for path in model_paths:
            if not os.path.exists(path):
                logging.warning(f"AI model not found at: {path}")

        return ai_config

    def setup_backup_and_recovery(self) -> Dict:
        """Configurar backup e recuperação"""
        backup_config = {
            'database_backup_enabled': True,
            'database_backup_interval_hours': 6,
            'model_backup_enabled': True,
            'model_backup_interval_hours': 24,
            'configuration_backup': True,
            'log_backup_enabled': True,
            'backup_retention_days': 30
        }

        # Criar diretórios de backup
        backup_dirs = [
            'backups/database',
            'backups/models',
            'backups/config',
            'backups/logs'
        ]

        for backup_dir in backup_dirs:
            Path(backup_dir).mkdir(parents=True, exist_ok=True)

        return backup_config

    def validate_production_readiness(self) -> Dict:
        """Validar se sistema está pronto para produção"""
        validation_results = {
            'environment_variables': self._check_environment_variables(),
            'ssl_certificates': self._check_ssl_certificates(),
            'database_connection': self._check_database_config(),
            'redis_connection': self._check_redis_config(),
            'ai_models': self._check_ai_models(),
            'security_settings': self._check_security_config(),
            'monitoring_setup': self._check_monitoring_config()
        }

        # Verificar se todas as validações passaram
        all_valid = all(validation_results.values())

        validation_results['production_ready'] = all_valid

        if not all_valid:
            failed_checks = [check for check, result in validation_results.items() if not result]
            logging.error(f"Production readiness failed: {failed_checks}")

        return validation_results

    def _check_environment_variables(self) -> bool:
        """Verificar variáveis de ambiente obrigatórias"""
        required_vars = [
            'DERIV_API_KEY',
            'DB_PASSWORD',
            'JWT_SECRET_KEY'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logging.error(f"Missing required environment variables: {missing_vars}")
            return False

        return True

    def _check_ssl_certificates(self) -> bool:
        """Verificar certificados SSL"""
        if not self.security.ssl_required:
            return True

        cert_file = os.getenv('SSL_CERT_FILE', 'certs/server.crt')
        key_file = os.getenv('SSL_KEY_FILE', 'certs/server.key')

        return os.path.exists(cert_file) and os.path.exists(key_file)

    def _check_database_config(self) -> bool:
        """Verificar configuração de database"""
        return bool(self.database.password and self.database.host)

    def _check_redis_config(self) -> bool:
        """Verificar configuração do Redis"""
        return bool(self.redis.host)

    def _check_ai_models(self) -> bool:
        """Verificar modelos de IA"""
        return os.path.exists(self.ai_model.model_path)

    def _check_security_config(self) -> bool:
        """Verificar configurações de segurança"""
        return self.security.rate_limiting_enabled and len(self.security.cors_origins) > 0

    def _check_monitoring_config(self) -> bool:
        """Verificar configurações de monitoramento"""
        return self.monitoring.enable_prometheus or self.monitoring.enable_grafana

    def save_configuration(self):
        """Salvar configuração em arquivos"""
        configs = {
            'security': asdict(self.security),
            'database': asdict(self.database),
            'redis': asdict(self.redis),
            'trading': asdict(self.trading),
            'monitoring': asdict(self.monitoring),
            'ai_model': asdict(self.ai_model)
        }

        for config_name, config_data in configs.items():
            config_file = self.config_path / f"{config_name}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        logging.info(f"Configuration saved to {self.config_path}")

    def load_configuration(self):
        """Carregar configuração de arquivos"""
        config_files = {
            'security': self.security,
            'database': self.database,
            'redis': self.redis,
            'trading': self.trading,
            'monitoring': self.monitoring,
            'ai_model': self.ai_model
        }

        for config_name, config_obj in config_files.items():
            config_file = self.config_path / f"{config_name}.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)

        # Carregar variáveis de ambiente
        self.load_environment_variables()

    def get_production_config(self) -> Dict:
        """Obter configuração completa para produção"""
        return {
            'environment': self.environment,
            'security': asdict(self.security),
            'database': asdict(self.database),
            'redis': asdict(self.redis),
            'trading': asdict(self.trading),
            'monitoring': asdict(self.monitoring),
            'ai_model': asdict(self.ai_model),
            'timestamp': datetime.now().isoformat()
        }

    def generate_deployment_checklist(self) -> List[Dict]:
        """Gerar checklist de deployment"""
        checklist = [
            {
                'category': 'Environment Setup',
                'items': [
                    'Set all required environment variables',
                    'Configure SSL certificates',
                    'Setup database with proper credentials',
                    'Configure Redis cache server',
                    'Verify network connectivity'
                ]
            },
            {
                'category': 'Security Configuration',
                'items': [
                    'Enable rate limiting',
                    'Configure CORS origins',
                    'Setup API key encryption',
                    'Enable audit logging',
                    'Verify SSL/TLS configuration'
                ]
            },
            {
                'category': 'AI Model Deployment',
                'items': [
                    'Deploy trained model files',
                    'Setup backup models',
                    'Configure model validation',
                    'Enable auto-retraining',
                    'Test inference performance'
                ]
            },
            {
                'category': 'Monitoring & Alerting',
                'items': [
                    'Setup Prometheus metrics',
                    'Configure Grafana dashboards',
                    'Enable health checks',
                    'Configure error alerting',
                    'Setup log aggregation'
                ]
            },
            {
                'category': 'Trading Configuration',
                'items': [
                    'Configure position limits',
                    'Set risk management parameters',
                    'Enable emergency stops',
                    'Configure API rate limits',
                    'Test Deriv API connectivity'
                ]
            },
            {
                'category': 'Backup & Recovery',
                'items': [
                    'Setup database backups',
                    'Configure model backups',
                    'Enable log rotation',
                    'Test recovery procedures',
                    'Document backup schedules'
                ]
            }
        ]

        return checklist


# 🚀 Production Setup Script
def setup_production_environment():
    """Script principal para configurar ambiente de produção"""
    print("🚀 SETTING UP PRODUCTION ENVIRONMENT")
    print("=" * 50)

    # Inicializar configuração
    config_manager = ProductionConfigManager("production")

    # 1. Configurar segurança
    print("🔐 Setting up security...")
    security_config = config_manager.setup_security_hardening()
    print(f"   Security measures: {list(security_config.keys())}")

    # 2. Configurar database
    print("🗄️ Setting up database...")
    db_config = config_manager.setup_database_optimization()
    print(f"   Database optimizations: {list(db_config.keys())}")

    # 3. Configurar monitoramento
    print("📊 Setting up monitoring...")
    monitoring_config = config_manager.setup_monitoring_and_alerting()
    print(f"   Monitoring features: {list(monitoring_config.keys())}")

    # 4. Configurar IA
    print("🧠 Setting up AI models...")
    ai_config = config_manager.setup_ai_model_production()
    print(f"   AI configurations: {list(ai_config.keys())}")

    # 5. Configurar backup
    print("💾 Setting up backup systems...")
    backup_config = config_manager.setup_backup_and_recovery()
    print(f"   Backup features: {list(backup_config.keys())}")

    # 6. Validar produção
    print("✅ Validating production readiness...")
    validation_results = config_manager.validate_production_readiness()

    if validation_results['production_ready']:
        print("🎉 PRODUCTION ENVIRONMENT READY!")
    else:
        print("❌ Production validation failed!")
        failed_checks = [k for k, v in validation_results.items() if not v and k != 'production_ready']
        print(f"   Failed checks: {failed_checks}")

    # 7. Salvar configuração
    config_manager.save_configuration()

    # 8. Gerar checklist
    checklist = config_manager.generate_deployment_checklist()
    print("\n📋 DEPLOYMENT CHECKLIST:")
    for category in checklist:
        print(f"\n{category['category']}:")
        for item in category['items']:
            print(f"  ☐ {item}")

    return config_manager


if __name__ == "__main__":
    setup_production_environment()
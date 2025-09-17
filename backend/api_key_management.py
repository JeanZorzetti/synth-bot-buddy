"""
🔑 API KEY MANAGEMENT SYSTEM
Sistema completo de gerenciamento de chaves API para usuários
"""

import asyncio
import hashlib
import secrets
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncpg
import aiofiles
from pathlib import Path
import ipaddress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """🔐 Status das API Keys"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class APIKeyPermission(Enum):
    """🛡️ Permissões das API Keys"""
    READ_MARKET_DATA = "read_market_data"
    READ_PORTFOLIO = "read_portfolio"
    READ_ORDERS = "read_orders"
    WRITE_ORDERS = "write_orders"
    WRITE_PORTFOLIO = "write_portfolio"
    READ_ANALYTICS = "read_analytics"
    ADMIN_ACCESS = "admin_access"
    FULL_ACCESS = "full_access"


@dataclass
class APIKey:
    """🔑 Modelo de API Key"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    key_id: str = ""  # Parte pública da chave
    key_secret: str = ""  # Hash da parte secreta
    name: str = ""
    description: str = ""
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    permissions: Set[APIKeyPermission] = field(default_factory=set)
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 1000
    rate_limit_per_hour: int = 10000
    rate_limit_per_day: int = 100000
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_used_ip: str = ""
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class APIKeyUsage:
    """📊 Uso da API Key"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    api_key_id: str = ""
    user_id: str = ""
    endpoint: str = ""
    method: str = ""
    ip_address: str = ""
    user_agent: str = ""
    response_status: int = 0
    response_time_ms: float = 0.0
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class RateLimiter:
    """⚡ Rate Limiter para API Keys"""

    def __init__(self):
        self.usage_tracking = {}  # {api_key_id: {minute: count, hour: count, day: count}}

    def check_rate_limit(self, api_key: APIKey, current_time: datetime = None) -> Dict[str, bool]:
        """Verificar rate limits"""
        if current_time is None:
            current_time = datetime.now()

        key_id = api_key.key_id

        # Inicializar tracking se não existir
        if key_id not in self.usage_tracking:
            self.usage_tracking[key_id] = {
                'minute': {'count': 0, 'reset_time': current_time + timedelta(minutes=1)},
                'hour': {'count': 0, 'reset_time': current_time + timedelta(hours=1)},
                'day': {'count': 0, 'reset_time': current_time + timedelta(days=1)}
            }

        tracking = self.usage_tracking[key_id]

        # Reset contadores se necessário
        for period in ['minute', 'hour', 'day']:
            if current_time >= tracking[period]['reset_time']:
                tracking[period]['count'] = 0
                if period == 'minute':
                    tracking[period]['reset_time'] = current_time + timedelta(minutes=1)
                elif period == 'hour':
                    tracking[period]['reset_time'] = current_time + timedelta(hours=1)
                elif period == 'day':
                    tracking[period]['reset_time'] = current_time + timedelta(days=1)

        # Verificar limites
        limits_status = {
            'minute_ok': tracking['minute']['count'] < api_key.rate_limit_per_minute,
            'hour_ok': tracking['hour']['count'] < api_key.rate_limit_per_hour,
            'day_ok': tracking['day']['count'] < api_key.rate_limit_per_day,
            'remaining_minute': max(0, api_key.rate_limit_per_minute - tracking['minute']['count']),
            'remaining_hour': max(0, api_key.rate_limit_per_hour - tracking['hour']['count']),
            'remaining_day': max(0, api_key.rate_limit_per_day - tracking['day']['count']),
            'reset_times': {
                'minute': tracking['minute']['reset_time'].isoformat(),
                'hour': tracking['hour']['reset_time'].isoformat(),
                'day': tracking['day']['reset_time'].isoformat()
            }
        }

        limits_status['allowed'] = all([
            limits_status['minute_ok'],
            limits_status['hour_ok'],
            limits_status['day_ok']
        ])

        return limits_status

    def record_usage(self, api_key_id: str):
        """Registrar uso da API"""
        if api_key_id in self.usage_tracking:
            tracking = self.usage_tracking[api_key_id]
            tracking['minute']['count'] += 1
            tracking['hour']['count'] += 1
            tracking['day']['count'] += 1


class IPWhitelistManager:
    """🛡️ Gerenciador de whitelist de IPs"""

    def __init__(self):
        pass

    def is_ip_allowed(self, api_key: APIKey, client_ip: str) -> bool:
        """Verificar se IP está na whitelist"""
        if not api_key.allowed_ips:
            return True  # Sem restrições de IP

        try:
            client_addr = ipaddress.ip_address(client_ip)

            for allowed_ip in api_key.allowed_ips:
                try:
                    # Suporte para CIDR notation
                    if '/' in allowed_ip:
                        allowed_network = ipaddress.ip_network(allowed_ip, strict=False)
                        if client_addr in allowed_network:
                            return True
                    else:
                        allowed_addr = ipaddress.ip_address(allowed_ip)
                        if client_addr == allowed_addr:
                            return True
                except ValueError:
                    logger.warning(f"Invalid IP format in whitelist: {allowed_ip}")
                    continue

            return False

        except ValueError:
            logger.warning(f"Invalid client IP: {client_ip}")
            return False

    def add_ip_to_whitelist(self, api_key: APIKey, ip_address: str) -> bool:
        """Adicionar IP à whitelist"""
        try:
            # Validar formato do IP
            ipaddress.ip_address(ip_address.split('/')[0])  # Remove CIDR se presente

            if ip_address not in api_key.allowed_ips:
                api_key.allowed_ips.append(ip_address)
                return True
            return False

        except ValueError:
            logger.error(f"Invalid IP format: {ip_address}")
            return False

    def remove_ip_from_whitelist(self, api_key: APIKey, ip_address: str) -> bool:
        """Remover IP da whitelist"""
        try:
            api_key.allowed_ips.remove(ip_address)
            return True
        except ValueError:
            return False


class APIKeyManager:
    """🔑 Gerenciador principal de API Keys"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.rate_limiter = RateLimiter()
        self.ip_manager = IPWhitelistManager()

    async def initialize(self):
        """Inicializar sistema"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            await self._create_tables()
            logger.info("✅ API Key Management System initialized")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            await self._initialize_file_storage()

    async def _create_tables(self):
        """Criar tabelas necessárias"""
        async with self.pool.acquire() as conn:
            # Tabela de API Keys
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    key_id VARCHAR UNIQUE NOT NULL,
                    key_secret VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    description TEXT,
                    status VARCHAR DEFAULT 'active',
                    permissions JSONB DEFAULT '[]',
                    allowed_ips JSONB DEFAULT '[]',
                    rate_limit_per_minute INTEGER DEFAULT 1000,
                    rate_limit_per_hour INTEGER DEFAULT 10000,
                    rate_limit_per_day INTEGER DEFAULT 100000,
                    expires_at TIMESTAMP,
                    last_used_at TIMESTAMP,
                    last_used_ip VARCHAR,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Tabela de uso das API Keys
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_key_usage (
                    id UUID PRIMARY KEY,
                    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL,
                    endpoint VARCHAR NOT NULL,
                    method VARCHAR NOT NULL,
                    ip_address VARCHAR NOT NULL,
                    user_agent TEXT,
                    response_status INTEGER,
                    response_time_ms DECIMAL,
                    request_size_bytes INTEGER,
                    response_size_bytes INTEGER,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Índices para performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_key_id ON api_key_usage(api_key_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_timestamp ON api_key_usage(timestamp)")

    async def _initialize_file_storage(self):
        """Inicializar armazenamento em arquivo"""
        self.storage_path = Path("data/api_keys")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Using file-based storage for API keys")

    def generate_api_key_pair(self) -> tuple[str, str, str]:
        """Gerar par de chaves API (key_id, key_secret, full_key)"""
        # Gerar key_id (parte pública)
        key_id = f"tb_{secrets.token_urlsafe(16)}"

        # Gerar key_secret (parte privada)
        key_secret = secrets.token_urlsafe(32)

        # Chave completa para o usuário
        full_key = f"{key_id}.{key_secret}"

        # Hash do secret para armazenamento
        secret_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        return key_id, secret_hash, full_key

    def verify_api_key(self, full_key: str) -> tuple[str, str]:
        """Verificar e extrair componentes da API key"""
        try:
            if '.' not in full_key:
                raise ValueError("Invalid API key format")

            key_id, key_secret = full_key.split('.', 1)
            secret_hash = hashlib.sha256(key_secret.encode()).hexdigest()

            return key_id, secret_hash
        except Exception:
            raise ValueError("Invalid API key format")

    async def create_api_key(self, user_id: str, name: str, description: str = "", permissions: List[APIKeyPermission] = None, expires_in_days: int = None) -> Optional[Dict]:
        """Criar nova API key"""
        try:
            # Gerar chaves
            key_id, secret_hash, full_key = self.generate_api_key_pair()

            # Definir expiração
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)

            # Criar objeto API Key
            api_key = APIKey(
                user_id=user_id,
                key_id=key_id,
                key_secret=secret_hash,
                name=name,
                description=description,
                permissions=set(permissions or [APIKeyPermission.READ_MARKET_DATA]),
                expires_at=expires_at
            )

            # Salvar no banco
            success = await self._save_api_key(api_key)

            if success:
                logger.info(f"✅ API key created: {key_id} for user {user_id}")
                return {
                    'api_key_id': api_key.id,
                    'key_id': key_id,
                    'full_key': full_key,  # Retornar apenas uma vez
                    'name': name,
                    'permissions': [p.value for p in api_key.permissions],
                    'expires_at': expires_at.isoformat() if expires_at else None,
                    'rate_limits': {
                        'per_minute': api_key.rate_limit_per_minute,
                        'per_hour': api_key.rate_limit_per_hour,
                        'per_day': api_key.rate_limit_per_day
                    }
                }

        except Exception as e:
            logger.error(f"❌ Error creating API key: {e}")

        return None

    async def validate_api_key(self, full_key: str, client_ip: str = "", endpoint: str = "", method: str = "") -> Optional[Dict]:
        """Validar API key e verificar permissões"""
        try:
            # Extrair componentes
            key_id, secret_hash = self.verify_api_key(full_key)

            # Buscar API key
            api_key = await self._get_api_key_by_id(key_id)
            if not api_key:
                logger.warning(f"API key not found: {key_id}")
                return None

            # Verificar status
            if api_key.status != APIKeyStatus.ACTIVE:
                logger.warning(f"API key not active: {key_id} (status: {api_key.status.value})")
                return None

            # Verificar expiração
            if api_key.expires_at and datetime.now() > api_key.expires_at:
                logger.warning(f"API key expired: {key_id}")
                await self._update_api_key_status(api_key.id, APIKeyStatus.EXPIRED)
                return None

            # Verificar hash do secret
            if api_key.key_secret != secret_hash:
                logger.warning(f"Invalid API key secret: {key_id}")
                return None

            # Verificar IP whitelist
            if client_ip and not self.ip_manager.is_ip_allowed(api_key, client_ip):
                logger.warning(f"IP not allowed for API key {key_id}: {client_ip}")
                return None

            # Verificar rate limits
            rate_limit_status = self.rate_limiter.check_rate_limit(api_key)
            if not rate_limit_status['allowed']:
                logger.warning(f"Rate limit exceeded for API key: {key_id}")
                return {
                    'valid': False,
                    'error': 'rate_limit_exceeded',
                    'rate_limit_status': rate_limit_status
                }

            # Registrar uso
            self.rate_limiter.record_usage(api_key.key_id)
            await self._record_usage(api_key, client_ip, endpoint, method)

            return {
                'valid': True,
                'api_key': api_key,
                'user_id': api_key.user_id,
                'permissions': [p.value for p in api_key.permissions],
                'rate_limit_status': rate_limit_status
            }

        except ValueError as e:
            logger.warning(f"API key validation error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error validating API key: {e}")
            return None

    async def list_user_api_keys(self, user_id: str) -> List[Dict]:
        """Listar API keys do usuário"""
        try:
            api_keys = []

            if self.pool:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC",
                        user_id
                    )

                    for row in rows:
                        api_key = self._row_to_api_key(row)
                        api_keys.append({
                            'id': api_key.id,
                            'key_id': api_key.key_id,
                            'name': api_key.name,
                            'description': api_key.description,
                            'status': api_key.status.value,
                            'permissions': [p.value for p in api_key.permissions],
                            'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                            'last_used_at': api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                            'usage_count': api_key.usage_count,
                            'created_at': api_key.created_at.isoformat(),
                            'rate_limits': {
                                'per_minute': api_key.rate_limit_per_minute,
                                'per_hour': api_key.rate_limit_per_hour,
                                'per_day': api_key.rate_limit_per_day
                            }
                        })

            return api_keys

        except Exception as e:
            logger.error(f"❌ Error listing user API keys: {e}")
            return []

    async def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Revogar API key"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    result = await conn.execute(
                        "UPDATE api_keys SET status = 'revoked', updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        api_key_id, user_id
                    )

                    if result == "UPDATE 1":
                        logger.info(f"✅ API key revoked: {api_key_id}")
                        return True

        except Exception as e:
            logger.error(f"❌ Error revoking API key: {e}")

        return False

    async def update_api_key_permissions(self, api_key_id: str, user_id: str, permissions: List[APIKeyPermission]) -> bool:
        """Atualizar permissões da API key"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    permissions_json = json.dumps([p.value for p in permissions])
                    result = await conn.execute(
                        "UPDATE api_keys SET permissions = $1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
                        permissions_json, api_key_id, user_id
                    )

                    if result == "UPDATE 1":
                        logger.info(f"✅ API key permissions updated: {api_key_id}")
                        return True

        except Exception as e:
            logger.error(f"❌ Error updating API key permissions: {e}")

        return False

    async def get_api_key_usage_stats(self, api_key_id: str, days: int = 30) -> Dict:
        """Obter estatísticas de uso da API key"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    # Estatísticas gerais
                    stats_row = await conn.fetchrow("""
                        SELECT
                            COUNT(*) as total_requests,
                            COUNT(DISTINCT DATE(timestamp)) as active_days,
                            AVG(response_time_ms) as avg_response_time,
                            COUNT(*) FILTER (WHERE response_status >= 400) as error_count,
                            SUM(request_size_bytes) as total_request_bytes,
                            SUM(response_size_bytes) as total_response_bytes
                        FROM api_key_usage
                        WHERE api_key_id = $1 AND timestamp >= NOW() - INTERVAL '%s days'
                    """, api_key_id, days)

                    # Endpoints mais usados
                    endpoints_rows = await conn.fetch("""
                        SELECT endpoint, method, COUNT(*) as count
                        FROM api_key_usage
                        WHERE api_key_id = $1 AND timestamp >= NOW() - INTERVAL '%s days'
                        GROUP BY endpoint, method
                        ORDER BY count DESC
                        LIMIT 10
                    """, api_key_id, days)

                    # Uso por dia
                    daily_usage_rows = await conn.fetch("""
                        SELECT DATE(timestamp) as date, COUNT(*) as requests
                        FROM api_key_usage
                        WHERE api_key_id = $1 AND timestamp >= NOW() - INTERVAL '%s days'
                        GROUP BY DATE(timestamp)
                        ORDER BY date DESC
                    """, api_key_id, days)

                    return {
                        'period_days': days,
                        'total_requests': stats_row['total_requests'] or 0,
                        'active_days': stats_row['active_days'] or 0,
                        'avg_response_time_ms': float(stats_row['avg_response_time'] or 0),
                        'error_count': stats_row['error_count'] or 0,
                        'error_rate': (stats_row['error_count'] / max(stats_row['total_requests'], 1)) * 100,
                        'total_request_bytes': stats_row['total_request_bytes'] or 0,
                        'total_response_bytes': stats_row['total_response_bytes'] or 0,
                        'top_endpoints': [
                            {'endpoint': row['endpoint'], 'method': row['method'], 'count': row['count']}
                            for row in endpoints_rows
                        ],
                        'daily_usage': [
                            {'date': row['date'].isoformat(), 'requests': row['requests']}
                            for row in daily_usage_rows
                        ]
                    }

        except Exception as e:
            logger.error(f"❌ Error getting API key usage stats: {e}")

        return {}

    async def _save_api_key(self, api_key: APIKey) -> bool:
        """Salvar API key no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    permissions_json = json.dumps([p.value for p in api_key.permissions])
                    allowed_ips_json = json.dumps(api_key.allowed_ips)

                    await conn.execute("""
                        INSERT INTO api_keys (
                            id, user_id, key_id, key_secret, name, description, status,
                            permissions, allowed_ips, rate_limit_per_minute, rate_limit_per_hour,
                            rate_limit_per_day, expires_at, last_used_at, last_used_ip,
                            usage_count, created_at, updated_at, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    """,
                        api_key.id, api_key.user_id, api_key.key_id, api_key.key_secret,
                        api_key.name, api_key.description, api_key.status.value,
                        permissions_json, allowed_ips_json, api_key.rate_limit_per_minute,
                        api_key.rate_limit_per_hour, api_key.rate_limit_per_day,
                        api_key.expires_at, api_key.last_used_at, api_key.last_used_ip,
                        api_key.usage_count, api_key.created_at, api_key.updated_at,
                        json.dumps(api_key.metadata)
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving API key: {e}")

        return False

    async def _get_api_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """Buscar API key por key_id"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM api_keys WHERE key_id = $1",
                        key_id
                    )
                    if row:
                        return self._row_to_api_key(row)

        except Exception as e:
            logger.error(f"❌ Error getting API key: {e}")

        return None

    async def _update_api_key_status(self, api_key_id: str, status: APIKeyStatus) -> bool:
        """Atualizar status da API key"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE api_keys SET status = $1, updated_at = NOW() WHERE id = $2",
                        status.value, api_key_id
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error updating API key status: {e}")

        return False

    async def _record_usage(self, api_key: APIKey, client_ip: str, endpoint: str, method: str) -> bool:
        """Registrar uso da API key"""
        try:
            if self.pool:
                usage = APIKeyUsage(
                    api_key_id=api_key.id,
                    user_id=api_key.user_id,
                    endpoint=endpoint,
                    method=method,
                    ip_address=client_ip
                )

                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO api_key_usage (
                            id, api_key_id, user_id, endpoint, method, ip_address,
                            user_agent, response_status, response_time_ms,
                            request_size_bytes, response_size_bytes, timestamp, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        usage.id, usage.api_key_id, usage.user_id, usage.endpoint,
                        usage.method, usage.ip_address, usage.user_agent,
                        usage.response_status, usage.response_time_ms,
                        usage.request_size_bytes, usage.response_size_bytes,
                        usage.timestamp, json.dumps(usage.metadata)
                    )

                    # Atualizar estatísticas da API key
                    await conn.execute("""
                        UPDATE api_keys
                        SET usage_count = usage_count + 1,
                            last_used_at = NOW(),
                            last_used_ip = $1
                        WHERE id = $2
                    """, client_ip, api_key.id)

                return True

        except Exception as e:
            logger.error(f"❌ Error recording API key usage: {e}")

        return False

    def _row_to_api_key(self, row) -> APIKey:
        """Converter row do banco para APIKey"""
        permissions = set()
        if row['permissions']:
            perm_list = json.loads(row['permissions'])
            permissions = {APIKeyPermission(p) for p in perm_list if p in [e.value for e in APIKeyPermission]}

        allowed_ips = []
        if row['allowed_ips']:
            allowed_ips = json.loads(row['allowed_ips'])

        metadata = {}
        if row['metadata']:
            metadata = json.loads(row['metadata'])

        return APIKey(
            id=str(row['id']),
            user_id=str(row['user_id']),
            key_id=row['key_id'],
            key_secret=row['key_secret'],
            name=row['name'],
            description=row['description'] or '',
            status=APIKeyStatus(row['status']),
            permissions=permissions,
            allowed_ips=allowed_ips,
            rate_limit_per_minute=row['rate_limit_per_minute'],
            rate_limit_per_hour=row['rate_limit_per_hour'],
            rate_limit_per_day=row['rate_limit_per_day'],
            expires_at=row['expires_at'],
            last_used_at=row['last_used_at'],
            last_used_ip=row['last_used_ip'] or '',
            usage_count=row['usage_count'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=metadata
        )


# 🧪 Função de teste
async def test_api_key_management():
    """Testar sistema de gerenciamento de API keys"""
    database_url = "postgresql://trading_user:password@localhost:5432/trading_db"

    # Inicializar sistema
    api_manager = APIKeyManager(database_url)
    await api_manager.initialize()

    print("\n" + "="*80)
    print("🔑 API KEY MANAGEMENT SYSTEM TEST")
    print("="*80)

    test_user_id = str(uuid.uuid4())

    # 1. Criar API key
    print(f"\n🔑 CREATING API KEY for user: {test_user_id}")
    api_key_result = await api_manager.create_api_key(
        user_id=test_user_id,
        name="Test API Key",
        description="Chave de teste para desenvolvimento",
        permissions=[APIKeyPermission.READ_MARKET_DATA, APIKeyPermission.READ_PORTFOLIO],
        expires_in_days=30
    )

    if api_key_result:
        print(f"✅ API Key created: {api_key_result['key_id']}")
        print(f"   Full Key: {api_key_result['full_key']}")
        print(f"   Permissions: {api_key_result['permissions']}")
        print(f"   Rate Limits: {api_key_result['rate_limits']}")

        full_key = api_key_result['full_key']
    else:
        print("❌ Failed to create API key")
        return

    # 2. Validar API key
    print(f"\n🔐 VALIDATING API KEY...")
    validation_result = await api_manager.validate_api_key(
        full_key=full_key,
        client_ip="127.0.0.1",
        endpoint="/api/market-data",
        method="GET"
    )

    if validation_result and validation_result['valid']:
        print(f"✅ API Key validation successful")
        print(f"   User ID: {validation_result['user_id']}")
        print(f"   Permissions: {validation_result['permissions']}")
        print(f"   Rate Limit Status: {validation_result['rate_limit_status']['allowed']}")
    else:
        print(f"❌ API Key validation failed")

    # 3. Testar rate limiting
    print(f"\n⚡ TESTING RATE LIMITING...")
    for i in range(3):
        validation_result = await api_manager.validate_api_key(
            full_key=full_key,
            client_ip="127.0.0.1",
            endpoint=f"/api/test-{i}",
            method="GET"
        )

        if validation_result:
            rate_status = validation_result['rate_limit_status']
            print(f"   Request {i+1}: Remaining minute: {rate_status['remaining_minute']}")

    # 4. Listar API keys do usuário
    print(f"\n📋 LISTING USER API KEYS...")
    user_keys = await api_manager.list_user_api_keys(test_user_id)

    for key in user_keys:
        print(f"   🔑 {key['name']} ({key['key_id']})")
        print(f"      Status: {key['status']}")
        print(f"      Usage: {key['usage_count']} requests")
        print(f"      Last used: {key['last_used_at'] or 'Never'}")

    # 5. Estatísticas de uso
    if user_keys:
        print(f"\n📊 USAGE STATISTICS...")
        stats = await api_manager.get_api_key_usage_stats(user_keys[0]['id'], days=7)

        print(f"   Total Requests: {stats.get('total_requests', 0)}")
        print(f"   Active Days: {stats.get('active_days', 0)}")
        print(f"   Error Rate: {stats.get('error_rate', 0):.2f}%")
        print(f"   Avg Response Time: {stats.get('avg_response_time_ms', 0):.2f}ms")

    print("\n" + "="*80)
    print("✅ API KEY MANAGEMENT SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_api_key_management())
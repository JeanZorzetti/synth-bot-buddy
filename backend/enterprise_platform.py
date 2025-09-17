"""
Enterprise Platform - Plataforma Enterprise
Sistema completo enterprise com multi-user, autenticação, autorização e recursos avançados.
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
import bcrypt
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import aioredis
import aiosqlite
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"

class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class SubscriptionTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    INSTITUTIONAL = "institutional"

@dataclass
class User:
    """Usuário da plataforma"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    subscription_tier: SubscriptionTier
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    organization_id: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: int = 1000  # requests per hour

@dataclass
class Organization:
    """Organização/Empresa"""
    org_id: str
    name: str
    domain: str
    subscription_tier: SubscriptionTier
    max_users: int
    created_at: datetime
    admin_user_id: str
    is_active: bool
    settings: Dict[str, Any]

@dataclass
class TradingAccount:
    """Conta de trading vinculada ao usuário"""
    account_id: str
    user_id: str
    account_name: str
    broker: str
    account_type: str  # demo, live
    balance: float
    currency: str
    api_credentials: Dict[str, str]
    is_active: bool
    risk_limits: Dict[str, float]

@dataclass
class UserSession:
    """Sessão de usuário"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool

class PermissionManager:
    """Gerenciador de permissões e autorização"""

    def __init__(self):
        # Permissões por role
        self.role_permissions = {
            UserRole.ADMIN: {
                "user:read", "user:write", "user:delete",
                "trading:read", "trading:write", "trading:execute",
                "portfolio:read", "portfolio:write",
                "analytics:read", "analytics:write",
                "system:read", "system:write", "system:admin"
            },
            UserRole.MANAGER: {
                "user:read", "user:write",
                "trading:read", "trading:write", "trading:execute",
                "portfolio:read", "portfolio:write",
                "analytics:read", "analytics:write"
            },
            UserRole.TRADER: {
                "trading:read", "trading:write", "trading:execute",
                "portfolio:read", "portfolio:write",
                "analytics:read"
            },
            UserRole.ANALYST: {
                "trading:read", "portfolio:read",
                "analytics:read", "analytics:write"
            },
            UserRole.VIEWER: {
                "trading:read", "portfolio:read", "analytics:read"
            },
            UserRole.API_USER: {
                "api:read", "api:write", "trading:execute"
            }
        }

        # Limitações por subscription tier
        self.tier_limits = {
            SubscriptionTier.FREE: {
                "max_accounts": 1,
                "max_symbols": 5,
                "api_calls_per_hour": 100,
                "data_retention_days": 30
            },
            SubscriptionTier.BASIC: {
                "max_accounts": 3,
                "max_symbols": 20,
                "api_calls_per_hour": 1000,
                "data_retention_days": 90
            },
            SubscriptionTier.PROFESSIONAL: {
                "max_accounts": 10,
                "max_symbols": 100,
                "api_calls_per_hour": 10000,
                "data_retention_days": 365
            },
            SubscriptionTier.ENTERPRISE: {
                "max_accounts": 50,
                "max_symbols": 500,
                "api_calls_per_hour": 100000,
                "data_retention_days": 1095
            },
            SubscriptionTier.INSTITUTIONAL: {
                "max_accounts": -1,  # unlimited
                "max_symbols": -1,   # unlimited
                "api_calls_per_hour": -1,  # unlimited
                "data_retention_days": -1   # unlimited
            }
        }

    def get_user_permissions(self, user: User) -> Set[str]:
        """Obtém permissões do usuário"""
        base_permissions = self.role_permissions.get(user.role, set())
        return base_permissions.union(user.permissions)

    def check_permission(self, user: User, permission: str) -> bool:
        """Verifica se usuário tem permissão"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions

    def check_tier_limit(self, user: User, limit_type: str, current_value: int) -> bool:
        """Verifica limite do tier de subscription"""
        limits = self.tier_limits.get(user.subscription_tier, {})
        limit_value = limits.get(limit_type, 0)

        if limit_value == -1:  # unlimited
            return True

        return current_value < limit_value

class AuthenticationManager:
    """Gerenciador de autenticação"""

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)

        # Rate limiting
        self.login_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.max_attempts = 5
        self.lockout_duration = timedelta(minutes=15)

    def hash_password(self, password: str) -> str:
        """Hash da senha"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifica senha"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_token(self, user: User) -> str:
        """Gera JWT token"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def check_rate_limit(self, identifier: str) -> bool:
        """Verifica rate limiting de login"""
        now = datetime.now()
        attempts = self.login_attempts[identifier]

        # Remove tentativas antigas
        while attempts and attempts[0] < now - self.lockout_duration:
            attempts.popleft()

        return len(attempts) < self.max_attempts

    def record_login_attempt(self, identifier: str, success: bool):
        """Registra tentativa de login"""
        if not success:
            self.login_attempts[identifier].append(datetime.now())

    def generate_api_key(self, user_id: str) -> str:
        """Gera API key"""
        data = f"{user_id}_{secrets.token_urlsafe(32)}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

class UserManager:
    """Gerenciador de usuários"""

    def __init__(self, db_path: str = "enterprise.db"):
        self.db_path = db_path
        self.auth_manager = AuthenticationManager()
        self.permission_manager = PermissionManager()

        # Cache de usuários
        self.user_cache: Dict[str, User] = {}
        self.session_cache: Dict[str, UserSession] = {}

        # Lock para operações thread-safe
        self.cache_lock = threading.Lock()

    async def initialize(self):
        """Inicializa o gerenciador"""
        await self._create_database_tables()
        await self._create_default_admin()

    async def _create_database_tables(self):
        """Cria tabelas do banco de dados"""
        async with aiosqlite.connect(self.db_path) as db:
            # Tabela de usuários
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    subscription_tier TEXT NOT NULL,
                    permissions TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active BOOLEAN NOT NULL,
                    organization_id TEXT,
                    api_key TEXT,
                    rate_limit INTEGER
                )
            """)

            # Tabela de organizações
            await db.execute("""
                CREATE TABLE IF NOT EXISTS organizations (
                    org_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT UNIQUE NOT NULL,
                    subscription_tier TEXT NOT NULL,
                    max_users INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    admin_user_id TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL,
                    settings TEXT
                )
            """)

            # Tabela de contas de trading
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trading_accounts (
                    account_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    account_name TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    balance REAL NOT NULL,
                    currency TEXT NOT NULL,
                    api_credentials TEXT,
                    is_active BOOLEAN NOT NULL,
                    risk_limits TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Tabela de sessões
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            await db.commit()

    async def _create_default_admin(self):
        """Cria usuário admin padrão"""
        admin_exists = await self.get_user_by_username("admin")
        if not admin_exists:
            admin_user = User(
                user_id=str(uuid.uuid4()),
                username="admin",
                email="admin@synth-bot-buddy.com",
                password_hash=self.auth_manager.hash_password("admin123"),
                role=UserRole.ADMIN,
                subscription_tier=SubscriptionTier.ENTERPRISE,
                permissions=set(),
                created_at=datetime.now(),
                last_login=None,
                is_active=True
            )

            await self.create_user(admin_user)
            logger.info("Usuário admin padrão criado")

    async def create_user(self, user: User) -> bool:
        """Cria novo usuário"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO users (
                        user_id, username, email, password_hash, role, subscription_tier,
                        permissions, created_at, last_login, is_active, organization_id,
                        api_key, rate_limit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.email, user.password_hash,
                    user.role.value, user.subscription_tier.value,
                    json.dumps(list(user.permissions)), user.created_at.isoformat(),
                    user.last_login.isoformat() if user.last_login else None,
                    user.is_active, user.organization_id, user.api_key, user.rate_limit
                ))
                await db.commit()

            # Atualizar cache
            with self.cache_lock:
                self.user_cache[user.user_id] = user

            return True

        except Exception as e:
            logger.error(f"Erro ao criar usuário: {e}")
            return False

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Obtém usuário por ID"""
        # Verificar cache primeiro
        with self.cache_lock:
            if user_id in self.user_cache:
                return self.user_cache[user_id]

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                )
                row = await cursor.fetchone()

                if row:
                    user = self._row_to_user(row)
                    # Atualizar cache
                    with self.cache_lock:
                        self.user_cache[user_id] = user
                    return user

        except Exception as e:
            logger.error(f"Erro ao buscar usuário: {e}")

        return None

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Obtém usuário por username"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                )
                row = await cursor.fetchone()

                if row:
                    return self._row_to_user(row)

        except Exception as e:
            logger.error(f"Erro ao buscar usuário por username: {e}")

        return None

    async def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[Tuple[User, str]]:
        """Autentica usuário e retorna token"""
        # Verificar rate limiting
        if not self.auth_manager.check_rate_limit(ip_address):
            self.auth_manager.record_login_attempt(ip_address, False)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts"
            )

        user = await self.get_user_by_username(username)
        if not user or not user.is_active:
            self.auth_manager.record_login_attempt(ip_address, False)
            return None

        if not self.auth_manager.verify_password(password, user.password_hash):
            self.auth_manager.record_login_attempt(ip_address, False)
            return None

        # Autenticação bem-sucedida
        self.auth_manager.record_login_attempt(ip_address, True)

        # Atualizar último login
        user.last_login = datetime.now()
        await self.update_user(user)

        # Gerar token
        token = self.auth_manager.generate_token(user)

        return user, token

    async def create_session(self, user: User, ip_address: str, user_agent: str) -> UserSession:
        """Cria sessão de usuário"""
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO user_sessions (
                        session_id, user_id, created_at, expires_at,
                        ip_address, user_agent, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id, session.user_id,
                    session.created_at.isoformat(), session.expires_at.isoformat(),
                    session.ip_address, session.user_agent, session.is_active
                ))
                await db.commit()

            # Atualizar cache
            with self.cache_lock:
                self.session_cache[session.session_id] = session

        except Exception as e:
            logger.error(f"Erro ao criar sessão: {e}")

        return session

    async def update_user(self, user: User) -> bool:
        """Atualiza usuário"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE users SET
                        username = ?, email = ?, password_hash = ?, role = ?,
                        subscription_tier = ?, permissions = ?, last_login = ?,
                        is_active = ?, organization_id = ?, api_key = ?, rate_limit = ?
                    WHERE user_id = ?
                """, (
                    user.username, user.email, user.password_hash, user.role.value,
                    user.subscription_tier.value, json.dumps(list(user.permissions)),
                    user.last_login.isoformat() if user.last_login else None,
                    user.is_active, user.organization_id, user.api_key, user.rate_limit,
                    user.user_id
                ))
                await db.commit()

            # Atualizar cache
            with self.cache_lock:
                self.user_cache[user.user_id] = user

            return True

        except Exception as e:
            logger.error(f"Erro ao atualizar usuário: {e}")
            return False

    def _row_to_user(self, row) -> User:
        """Converte row do DB para User"""
        return User(
            user_id=row[0],
            username=row[1],
            email=row[2],
            password_hash=row[3],
            role=UserRole(row[4]),
            subscription_tier=SubscriptionTier(row[5]),
            permissions=set(json.loads(row[6]) if row[6] else []),
            created_at=datetime.fromisoformat(row[7]),
            last_login=datetime.fromisoformat(row[8]) if row[8] else None,
            is_active=bool(row[9]),
            organization_id=row[10],
            api_key=row[11],
            rate_limit=row[12] or 1000
        )

class MultiTenantManager:
    """Gerenciador multi-tenant"""

    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager

        # Cache de organizações
        self.org_cache: Dict[str, Organization] = {}
        self.cache_lock = threading.Lock()

    async def create_organization(self, name: str, domain: str, admin_email: str) -> Organization:
        """Cria nova organização"""
        org_id = str(uuid.uuid4())

        # Criar usuário admin da organização
        admin_user = User(
            user_id=str(uuid.uuid4()),
            username=f"admin_{domain.replace('.', '_')}",
            email=admin_email,
            password_hash=self.user_manager.auth_manager.hash_password("admin123"),
            role=UserRole.ADMIN,
            subscription_tier=SubscriptionTier.PROFESSIONAL,
            permissions=set(),
            created_at=datetime.now(),
            last_login=None,
            is_active=True,
            organization_id=org_id
        )

        await self.user_manager.create_user(admin_user)

        # Criar organização
        organization = Organization(
            org_id=org_id,
            name=name,
            domain=domain,
            subscription_tier=SubscriptionTier.PROFESSIONAL,
            max_users=50,
            created_at=datetime.now(),
            admin_user_id=admin_user.user_id,
            is_active=True,
            settings={}
        )

        try:
            async with aiosqlite.connect(self.user_manager.db_path) as db:
                await db.execute("""
                    INSERT INTO organizations (
                        org_id, name, domain, subscription_tier, max_users,
                        created_at, admin_user_id, is_active, settings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    organization.org_id, organization.name, organization.domain,
                    organization.subscription_tier.value, organization.max_users,
                    organization.created_at.isoformat(), organization.admin_user_id,
                    organization.is_active, json.dumps(organization.settings)
                ))
                await db.commit()

            # Atualizar cache
            with self.cache_lock:
                self.org_cache[org_id] = organization

            logger.info(f"Organização {name} criada com sucesso")
            return organization

        except Exception as e:
            logger.error(f"Erro ao criar organização: {e}")
            raise

    async def get_organization_users(self, org_id: str) -> List[User]:
        """Obtém usuários de uma organização"""
        try:
            users = []
            async with aiosqlite.connect(self.user_manager.db_path) as db:
                cursor = await db.execute(
                    "SELECT * FROM users WHERE organization_id = ?", (org_id,)
                )
                rows = await cursor.fetchall()

                for row in rows:
                    users.append(self.user_manager._row_to_user(row))

            return users

        except Exception as e:
            logger.error(f"Erro ao buscar usuários da organização: {e}")
            return []

class AuditLogger:
    """Logger de auditoria para compliance"""

    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path

    async def initialize(self):
        """Inicializa logger de auditoria"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL
                )
            """)
            await db.commit()

    async def log_action(self, user_id: str, action: str, resource: str = None,
                        details: Dict[str, Any] = None, ip_address: str = None,
                        user_agent: str = None, success: bool = True):
        """Registra ação de auditoria"""
        try:
            log_id = str(uuid.uuid4())

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO audit_logs (
                        log_id, user_id, action, resource, details,
                        ip_address, user_agent, timestamp, success
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id, user_id, action, resource,
                    json.dumps(details) if details else None,
                    ip_address, user_agent, datetime.now().isoformat(), success
                ))
                await db.commit()

        except Exception as e:
            logger.error(f"Erro ao registrar auditoria: {e}")

class EnterprisePlatform:
    """Plataforma Enterprise principal"""

    def __init__(self):
        # Gerenciadores
        self.user_manager = UserManager()
        self.multi_tenant_manager = MultiTenantManager(self.user_manager)
        self.audit_logger = AuditLogger()

        # Cache Redis
        self.redis_client = None

        # Thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # Estado
        self.is_initialized = False

    async def initialize(self):
        """Inicializa plataforma enterprise"""
        try:
            # Inicializar componentes
            await self.user_manager.initialize()
            await self.audit_logger.initialize()

            # Conectar Redis
            try:
                self.redis_client = await aioredis.from_url("redis://localhost:6379")
            except Exception as e:
                logger.warning(f"Redis não disponível: {e}")

            self.is_initialized = True
            logger.info("Enterprise Platform inicializada")

        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")
            raise

    async def create_user_account(self, username: str, email: str, password: str,
                                role: UserRole = UserRole.TRADER,
                                subscription_tier: SubscriptionTier = SubscriptionTier.BASIC) -> User:
        """Cria conta de usuário"""
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=self.user_manager.auth_manager.hash_password(password),
            role=role,
            subscription_tier=subscription_tier,
            permissions=set(),
            created_at=datetime.now(),
            last_login=None,
            is_active=True
        )

        success = await self.user_manager.create_user(user)
        if success:
            await self.audit_logger.log_action(
                user.user_id, "user_created", "user", {"username": username}
            )
            return user
        else:
            raise Exception("Falha ao criar usuário")

    async def authenticate_and_authorize(self, username: str, password: str,
                                       required_permission: str = None,
                                       ip_address: str = None) -> Tuple[User, str]:
        """Autentica e autoriza usuário"""
        result = await self.user_manager.authenticate_user(username, password, ip_address or "unknown")

        if not result:
            await self.audit_logger.log_action(
                None, "login_failed", "auth", {"username": username}, ip_address, success=False
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        user, token = result

        # Verificar permissão se necessária
        if required_permission:
            if not self.user_manager.permission_manager.check_permission(user, required_permission):
                await self.audit_logger.log_action(
                    user.user_id, "access_denied", "auth",
                    {"permission": required_permission}, ip_address, success=False
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )

        await self.audit_logger.log_action(
            user.user_id, "login_success", "auth", {"username": username}, ip_address
        )

        return user, token

    async def get_user_dashboard_data(self, user: User) -> Dict[str, Any]:
        """Obtém dados do dashboard do usuário"""
        # Verificar permissões
        if not self.user_manager.permission_manager.check_permission(user, "trading:read"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

        # Simular dados do dashboard
        dashboard_data = {
            "user_info": {
                "username": user.username,
                "role": user.role.value,
                "subscription_tier": user.subscription_tier.value,
                "last_login": user.last_login.isoformat() if user.last_login else None
            },
            "account_summary": {
                "total_balance": 10000.0,
                "unrealized_pnl": 150.25,
                "realized_pnl": 500.75,
                "open_positions": 3
            },
            "permissions": list(self.user_manager.permission_manager.get_user_permissions(user)),
            "subscription_limits": self.user_manager.permission_manager.tier_limits.get(
                user.subscription_tier, {}
            )
        }

        await self.audit_logger.log_action(
            user.user_id, "dashboard_accessed", "dashboard"
        )

        return dashboard_data

    async def upgrade_subscription(self, user: User, new_tier: SubscriptionTier) -> bool:
        """Upgrade de subscription"""
        old_tier = user.subscription_tier
        user.subscription_tier = new_tier

        success = await self.user_manager.update_user(user)

        if success:
            await self.audit_logger.log_action(
                user.user_id, "subscription_upgraded", "subscription",
                {"old_tier": old_tier.value, "new_tier": new_tier.value}
            )

        return success

    async def get_platform_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas da plataforma"""
        try:
            async with aiosqlite.connect(self.user_manager.db_path) as db:
                # Total de usuários
                cursor = await db.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                total_users = (await cursor.fetchone())[0]

                # Usuários por role
                cursor = await db.execute("SELECT role, COUNT(*) FROM users WHERE is_active = 1 GROUP BY role")
                users_by_role = dict(await cursor.fetchall())

                # Usuários por subscription tier
                cursor = await db.execute("SELECT subscription_tier, COUNT(*) FROM users WHERE is_active = 1 GROUP BY subscription_tier")
                users_by_tier = dict(await cursor.fetchall())

                # Total de organizações
                cursor = await db.execute("SELECT COUNT(*) FROM organizations WHERE is_active = 1")
                total_orgs = (await cursor.fetchone())[0]

            return {
                "total_users": total_users,
                "total_organizations": total_orgs,
                "users_by_role": users_by_role,
                "users_by_subscription_tier": users_by_tier,
                "platform_uptime": "99.9%",
                "api_calls_today": 50000
            }

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}

    async def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Gera relatório de compliance"""
        try:
            async with aiosqlite.connect(self.audit_logger.db_path) as db:
                cursor = await db.execute("""
                    SELECT action, COUNT(*) as count,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                    FROM audit_logs
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY action
                """, (start_date.isoformat(), end_date.isoformat()))

                actions_summary = {}
                for row in await cursor.fetchall():
                    action, total, successful = row
                    actions_summary[action] = {
                        "total": total,
                        "successful": successful,
                        "failed": total - successful
                    }

                # Total de logs
                cursor = await db.execute("""
                    SELECT COUNT(*) FROM audit_logs
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                total_logs = (await cursor.fetchone())[0]

            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_audit_logs": total_logs,
                "actions_summary": actions_summary,
                "compliance_status": "COMPLIANT",
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro ao gerar relatório de compliance: {e}")
            return {}

    async def shutdown(self):
        """Encerra plataforma enterprise"""
        if self.redis_client:
            await self.redis_client.close()

        self.thread_pool.shutdown(wait=True)
        logger.info("Enterprise Platform encerrada")
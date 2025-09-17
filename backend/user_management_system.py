"""
👥 REAL USER MANAGEMENT SYSTEM
Sistema completo de gerenciamento de usuários com autenticação real
"""

import asyncio
import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import logging
from pathlib import Path
import aiofiles
import asyncpg
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """👤 Tipos de usuários"""
    ADMIN = "admin"
    PREMIUM = "premium"
    BASIC = "basic"
    TRIAL = "trial"
    SUSPENDED = "suspended"


class SubscriptionStatus(Enum):
    """💳 Status de assinatura"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    TRIAL = "trial"
    PENDING = "pending"


@dataclass
class User:
    """👤 Modelo de usuário completo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    username: str = ""
    password_hash: str = ""
    full_name: str = ""
    phone: str = ""
    country: str = ""
    role: UserRole = UserRole.TRIAL
    subscription_status: SubscriptionStatus = SubscriptionStatus.TRIAL
    subscription_plan: str = "trial"
    subscription_expires: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    verification_token: Optional[str] = None
    reset_token: Optional[str] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    preferences: Dict = field(default_factory=dict)
    trading_settings: Dict = field(default_factory=dict)
    risk_profile: Dict = field(default_factory=dict)
    kyc_status: str = "pending"
    kyc_documents: List[str] = field(default_factory=list)
    total_trades: int = 0
    total_profit_loss: float = 0.0
    last_activity: Optional[datetime] = None


@dataclass
class Session:
    """🔐 Sessão de usuário"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    token: str = ""
    refresh_token: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True


class DatabaseManager:
    """🗄️ Gerenciador de banco de dados para usuários"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None

    async def initialize(self):
        """Inicializar conexão com banco"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            await self._create_tables()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            # Fallback to file-based storage
            await self._initialize_file_storage()

    async def _create_tables(self):
        """Criar tabelas necessárias"""
        async with self.pool.acquire() as conn:
            # Tabela de usuários
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    email VARCHAR UNIQUE NOT NULL,
                    username VARCHAR UNIQUE NOT NULL,
                    password_hash VARCHAR NOT NULL,
                    full_name VARCHAR,
                    phone VARCHAR,
                    country VARCHAR,
                    role VARCHAR DEFAULT 'trial',
                    subscription_status VARCHAR DEFAULT 'trial',
                    subscription_plan VARCHAR DEFAULT 'trial',
                    subscription_expires TIMESTAMP,
                    api_keys JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verification_token VARCHAR,
                    reset_token VARCHAR,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    preferences JSONB DEFAULT '{}',
                    trading_settings JSONB DEFAULT '{}',
                    risk_profile JSONB DEFAULT '{}',
                    kyc_status VARCHAR DEFAULT 'pending',
                    kyc_documents JSONB DEFAULT '[]',
                    total_trades INTEGER DEFAULT 0,
                    total_profit_loss DECIMAL DEFAULT 0.0,
                    last_activity TIMESTAMP
                )
            """)

            # Tabela de sessões
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id UUID PRIMARY KEY,
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    token VARCHAR NOT NULL,
                    refresh_token VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    ip_address VARCHAR,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Índices para performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")

    async def _initialize_file_storage(self):
        """Inicializar armazenamento em arquivo como fallback"""
        self.storage_path = Path("data/users")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Using file-based storage as fallback")

    async def save_user(self, user: User) -> bool:
        """Salvar usuário no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO users (
                            id, email, username, password_hash, full_name, phone, country,
                            role, subscription_status, subscription_plan, subscription_expires,
                            api_keys, created_at, last_login, is_active, is_verified,
                            verification_token, reset_token, login_attempts, locked_until,
                            preferences, trading_settings, risk_profile, kyc_status,
                            kyc_documents, total_trades, total_profit_loss, last_activity
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                            $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
                        ) ON CONFLICT (id) DO UPDATE SET
                            email = EXCLUDED.email,
                            username = EXCLUDED.username,
                            password_hash = EXCLUDED.password_hash,
                            full_name = EXCLUDED.full_name,
                            phone = EXCLUDED.phone,
                            country = EXCLUDED.country,
                            role = EXCLUDED.role,
                            subscription_status = EXCLUDED.subscription_status,
                            subscription_plan = EXCLUDED.subscription_plan,
                            subscription_expires = EXCLUDED.subscription_expires,
                            api_keys = EXCLUDED.api_keys,
                            last_login = EXCLUDED.last_login,
                            is_active = EXCLUDED.is_active,
                            is_verified = EXCLUDED.is_verified,
                            verification_token = EXCLUDED.verification_token,
                            reset_token = EXCLUDED.reset_token,
                            login_attempts = EXCLUDED.login_attempts,
                            locked_until = EXCLUDED.locked_until,
                            preferences = EXCLUDED.preferences,
                            trading_settings = EXCLUDED.trading_settings,
                            risk_profile = EXCLUDED.risk_profile,
                            kyc_status = EXCLUDED.kyc_status,
                            kyc_documents = EXCLUDED.kyc_documents,
                            total_trades = EXCLUDED.total_trades,
                            total_profit_loss = EXCLUDED.total_profit_loss,
                            last_activity = EXCLUDED.last_activity
                    """,
                        user.id, user.email, user.username, user.password_hash,
                        user.full_name, user.phone, user.country, user.role.value,
                        user.subscription_status.value, user.subscription_plan,
                        user.subscription_expires, json.dumps(user.api_keys),
                        user.created_at, user.last_login, user.is_active, user.is_verified,
                        user.verification_token, user.reset_token, user.login_attempts,
                        user.locked_until, json.dumps(user.preferences),
                        json.dumps(user.trading_settings), json.dumps(user.risk_profile),
                        user.kyc_status, json.dumps(user.kyc_documents),
                        user.total_trades, user.total_profit_loss, user.last_activity
                    )
                return True
            else:
                # Fallback para arquivo
                user_file = self.storage_path / f"{user.id}.json"
                user_data = {
                    'id': user.id,
                    'email': user.email,
                    'username': user.username,
                    'password_hash': user.password_hash,
                    'full_name': user.full_name,
                    'phone': user.phone,
                    'country': user.country,
                    'role': user.role.value,
                    'subscription_status': user.subscription_status.value,
                    'subscription_plan': user.subscription_plan,
                    'subscription_expires': user.subscription_expires.isoformat() if user.subscription_expires else None,
                    'api_keys': user.api_keys,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_active': user.is_active,
                    'is_verified': user.is_verified,
                    'verification_token': user.verification_token,
                    'reset_token': user.reset_token,
                    'login_attempts': user.login_attempts,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None,
                    'preferences': user.preferences,
                    'trading_settings': user.trading_settings,
                    'risk_profile': user.risk_profile,
                    'kyc_status': user.kyc_status,
                    'kyc_documents': user.kyc_documents,
                    'total_trades': user.total_trades,
                    'total_profit_loss': user.total_profit_loss,
                    'last_activity': user.last_activity.isoformat() if user.last_activity else None
                }

                async with aiofiles.open(user_file, 'w') as f:
                    await f.write(json.dumps(user_data, indent=2))
                return True

        except Exception as e:
            logger.error(f"❌ Error saving user {user.id}: {e}")
            return False

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Buscar usuário por email"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
                    if row:
                        return self._row_to_user(row)
            else:
                # Busca em arquivos
                for user_file in self.storage_path.glob("*.json"):
                    async with aiofiles.open(user_file, 'r') as f:
                        user_data = json.loads(await f.read())
                        if user_data.get('email') == email:
                            return self._dict_to_user(user_data)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting user by email {email}: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Buscar usuário por ID"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
                    if row:
                        return self._row_to_user(row)
            else:
                user_file = self.storage_path / f"{user_id}.json"
                if user_file.exists():
                    async with aiofiles.open(user_file, 'r') as f:
                        user_data = json.loads(await f.read())
                        return self._dict_to_user(user_data)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting user by ID {user_id}: {e}")
            return None

    def _row_to_user(self, row) -> User:
        """Converter row do banco para User"""
        return User(
            id=str(row['id']),
            email=row['email'],
            username=row['username'],
            password_hash=row['password_hash'],
            full_name=row['full_name'] or '',
            phone=row['phone'] or '',
            country=row['country'] or '',
            role=UserRole(row['role']),
            subscription_status=SubscriptionStatus(row['subscription_status']),
            subscription_plan=row['subscription_plan'],
            subscription_expires=row['subscription_expires'],
            api_keys=json.loads(row['api_keys']) if row['api_keys'] else [],
            created_at=row['created_at'],
            last_login=row['last_login'],
            is_active=row['is_active'],
            is_verified=row['is_verified'],
            verification_token=row['verification_token'],
            reset_token=row['reset_token'],
            login_attempts=row['login_attempts'],
            locked_until=row['locked_until'],
            preferences=json.loads(row['preferences']) if row['preferences'] else {},
            trading_settings=json.loads(row['trading_settings']) if row['trading_settings'] else {},
            risk_profile=json.loads(row['risk_profile']) if row['risk_profile'] else {},
            kyc_status=row['kyc_status'],
            kyc_documents=json.loads(row['kyc_documents']) if row['kyc_documents'] else [],
            total_trades=row['total_trades'],
            total_profit_loss=float(row['total_profit_loss']),
            last_activity=row['last_activity']
        )

    def _dict_to_user(self, data: Dict) -> User:
        """Converter dict para User"""
        return User(
            id=data['id'],
            email=data['email'],
            username=data['username'],
            password_hash=data['password_hash'],
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            country=data.get('country', ''),
            role=UserRole(data.get('role', 'trial')),
            subscription_status=SubscriptionStatus(data.get('subscription_status', 'trial')),
            subscription_plan=data.get('subscription_plan', 'trial'),
            subscription_expires=datetime.fromisoformat(data['subscription_expires']) if data.get('subscription_expires') else None,
            api_keys=data.get('api_keys', []),
            created_at=datetime.fromisoformat(data['created_at']),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            is_active=data.get('is_active', True),
            is_verified=data.get('is_verified', False),
            verification_token=data.get('verification_token'),
            reset_token=data.get('reset_token'),
            login_attempts=data.get('login_attempts', 0),
            locked_until=datetime.fromisoformat(data['locked_until']) if data.get('locked_until') else None,
            preferences=data.get('preferences', {}),
            trading_settings=data.get('trading_settings', {}),
            risk_profile=data.get('risk_profile', {}),
            kyc_status=data.get('kyc_status', 'pending'),
            kyc_documents=data.get('kyc_documents', []),
            total_trades=data.get('total_trades', 0),
            total_profit_loss=data.get('total_profit_loss', 0.0),
            last_activity=datetime.fromisoformat(data['last_activity']) if data.get('last_activity') else None
        )


class AuthenticationManager:
    """🔐 Gerenciador de autenticação"""

    def __init__(self, secret_key: str, db_manager: DatabaseManager):
        self.secret_key = secret_key
        self.db_manager = db_manager
        self.algorithm = "HS256"
        self.token_expire_hours = 24
        self.refresh_token_expire_days = 30
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30

    def hash_password(self, password: str) -> str:
        """Hash da senha"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verificar senha"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_token(self, user: User) -> Dict[str, str]:
        """Gerar JWT token"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expire_hours),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Refresh token
        refresh_payload = {
            'user_id': user.id,
            'type': 'refresh',
            'exp': datetime.utcnow() + timedelta(days=self.refresh_token_expire_days),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }

        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

        return {
            'access_token': token,
            'refresh_token': refresh_token,
            'token_type': 'bearer',
            'expires_in': self.token_expire_hours * 3600
        }

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verificar token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    async def authenticate_user(self, email: str, password: str, ip_address: str = "", user_agent: str = "") -> Optional[Dict]:
        """Autenticar usuário"""
        user = await self.db_manager.get_user_by_email(email)

        if not user:
            logger.warning(f"Authentication failed: user not found for {email}")
            return None

        # Verificar se conta está bloqueada
        if user.locked_until and datetime.now() < user.locked_until:
            logger.warning(f"Account locked for user {email}")
            return None

        # Verificar senha
        if not self.verify_password(password, user.password_hash):
            # Incrementar tentativas de login
            user.login_attempts += 1

            if user.login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.lockout_duration_minutes)
                logger.warning(f"Account locked due to too many failed attempts: {email}")

            await self.db_manager.save_user(user)
            return None

        # Reset tentativas de login em caso de sucesso
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        user.last_activity = datetime.now()

        await self.db_manager.save_user(user)

        # Gerar tokens
        tokens = self.generate_token(user)

        # Criar sessão
        session = Session(
            user_id=user.id,
            token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            ip_address=ip_address,
            user_agent=user_agent
        )

        logger.info(f"✅ User authenticated successfully: {email}")

        return {
            'user': user,
            'tokens': tokens,
            'session': session
        }


class UserManager:
    """👥 Gerenciador principal de usuários"""

    def __init__(self, database_url: str, secret_key: str):
        self.db_manager = DatabaseManager(database_url)
        self.auth_manager = AuthenticationManager(secret_key, self.db_manager)

    async def initialize(self):
        """Inicializar sistema"""
        await self.db_manager.initialize()
        logger.info("✅ User Management System initialized")

    async def register_user(self, email: str, username: str, password: str, full_name: str = "", phone: str = "", country: str = "") -> Optional[User]:
        """Registrar novo usuário"""
        try:
            # Verificar se usuário já existe
            existing_user = await self.db_manager.get_user_by_email(email)
            if existing_user:
                logger.warning(f"User already exists: {email}")
                return None

            # Criar usuário
            user = User(
                email=email,
                username=username,
                password_hash=self.auth_manager.hash_password(password),
                full_name=full_name,
                phone=phone,
                country=country,
                role=UserRole.TRIAL,
                subscription_status=SubscriptionStatus.TRIAL,
                subscription_expires=datetime.now() + timedelta(days=14),  # 14 dias de trial
                verification_token=secrets.token_urlsafe(32),
                trading_settings={
                    'max_risk_per_trade': 0.02,  # 2%
                    'max_daily_risk': 0.05,      # 5%
                    'auto_trading': False,
                    'default_position_size': 0.01,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04
                },
                risk_profile={
                    'risk_tolerance': 'medium',
                    'trading_experience': 'beginner',
                    'max_drawdown': 0.10,
                    'preferred_assets': ['forex'],
                    'trading_hours': {'start': '09:00', 'end': '17:00'}
                }
            )

            # Salvar usuário
            success = await self.db_manager.save_user(user)
            if success:
                logger.info(f"✅ User registered successfully: {email}")
                return user
            else:
                logger.error(f"❌ Failed to save user: {email}")
                return None

        except Exception as e:
            logger.error(f"❌ Error registering user {email}: {e}")
            return None

    async def login_user(self, email: str, password: str, ip_address: str = "", user_agent: str = "") -> Optional[Dict]:
        """Login de usuário"""
        return await self.auth_manager.authenticate_user(email, password, ip_address, user_agent)

    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Obter perfil completo do usuário"""
        user = await self.db_manager.get_user_by_id(user_id)
        if not user:
            return None

        return {
            'id': user.id,
            'email': user.email,
            'username': user.username,
            'full_name': user.full_name,
            'phone': user.phone,
            'country': user.country,
            'role': user.role.value,
            'subscription_status': user.subscription_status.value,
            'subscription_plan': user.subscription_plan,
            'subscription_expires': user.subscription_expires.isoformat() if user.subscription_expires else None,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'is_active': user.is_active,
            'is_verified': user.is_verified,
            'kyc_status': user.kyc_status,
            'total_trades': user.total_trades,
            'total_profit_loss': user.total_profit_loss,
            'trading_settings': user.trading_settings,
            'risk_profile': user.risk_profile,
            'preferences': user.preferences
        }

    async def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """Atualizar perfil do usuário"""
        user = await self.db_manager.get_user_by_id(user_id)
        if not user:
            return False

        # Atualizar campos permitidos
        allowed_fields = ['full_name', 'phone', 'country', 'preferences', 'trading_settings', 'risk_profile']

        for field, value in updates.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)

        user.last_activity = datetime.now()
        return await self.db_manager.save_user(user)

    async def upgrade_subscription(self, user_id: str, plan: str, duration_months: int = 1) -> bool:
        """Upgrade de assinatura"""
        user = await self.db_manager.get_user_by_id(user_id)
        if not user:
            return False

        # Atualizar plano
        user.subscription_plan = plan
        user.subscription_status = SubscriptionStatus.ACTIVE
        user.subscription_expires = datetime.now() + timedelta(days=30 * duration_months)

        # Atualizar role baseado no plano
        if plan == "premium":
            user.role = UserRole.PREMIUM
        elif plan == "basic":
            user.role = UserRole.BASIC

        return await self.db_manager.save_user(user)

    async def generate_api_key(self, user_id: str, name: str = "") -> Optional[str]:
        """Gerar nova API key para usuário"""
        user = await self.db_manager.get_user_by_id(user_id)
        if not user:
            return None

        # Gerar API key
        api_key = f"tb_{secrets.token_urlsafe(32)}"

        # Adicionar à lista de API keys
        user.api_keys.append({
            'key': api_key,
            'name': name or f"API Key {len(user.api_keys) + 1}",
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'is_active': True
        })

        success = await self.db_manager.save_user(user)
        return api_key if success else None

    async def get_user_stats(self) -> Dict:
        """Estatísticas de usuários"""
        try:
            stats = {
                'total_users': 0,
                'active_users': 0,
                'premium_users': 0,
                'trial_users': 0,
                'verified_users': 0,
                'new_users_today': 0,
                'total_trades': 0,
                'total_profit_loss': 0.0
            }

            if self.db_manager.pool:
                async with self.db_manager.pool.acquire() as conn:
                    # Total users
                    total = await conn.fetchval("SELECT COUNT(*) FROM users")
                    stats['total_users'] = total

                    # Active users
                    active = await conn.fetchval("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
                    stats['active_users'] = active

                    # Premium users
                    premium = await conn.fetchval("SELECT COUNT(*) FROM users WHERE role = 'premium'")
                    stats['premium_users'] = premium

                    # Trial users
                    trial = await conn.fetchval("SELECT COUNT(*) FROM users WHERE role = 'trial'")
                    stats['trial_users'] = trial

                    # Verified users
                    verified = await conn.fetchval("SELECT COUNT(*) FROM users WHERE is_verified = TRUE")
                    stats['verified_users'] = verified

                    # New users today
                    new_today = await conn.fetchval(
                        "SELECT COUNT(*) FROM users WHERE created_at >= CURRENT_DATE"
                    )
                    stats['new_users_today'] = new_today

                    # Trading stats
                    trading_stats = await conn.fetchrow(
                        "SELECT SUM(total_trades), SUM(total_profit_loss) FROM users"
                    )
                    stats['total_trades'] = trading_stats[0] or 0
                    stats['total_profit_loss'] = float(trading_stats[1] or 0.0)

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting user stats: {e}")
            return {'error': str(e)}


# 🧪 Função de teste
async def test_user_management_system():
    """Testar sistema de gerenciamento de usuários"""
    # Configurações
    database_url = "postgresql://trading_user:password@localhost:5432/trading_db"
    secret_key = "your-secret-key-here-must-be-very-secure"

    # Inicializar sistema
    user_manager = UserManager(database_url, secret_key)
    await user_manager.initialize()

    print("\n" + "="*80)
    print("👥 USER MANAGEMENT SYSTEM TEST")
    print("="*80)

    # 1. Registrar usuário de teste
    print("\n📝 REGISTERING TEST USER...")
    test_user = await user_manager.register_user(
        email="test@tradingbot.com",
        username="testuser",
        password="SecurePassword123!",
        full_name="Test User",
        phone="+1234567890",
        country="Brazil"
    )

    if test_user:
        print(f"✅ User registered: {test_user.email}")
        print(f"   User ID: {test_user.id}")
        print(f"   Role: {test_user.role.value}")
        print(f"   Subscription: {test_user.subscription_status.value}")
        print(f"   Trial expires: {test_user.subscription_expires}")
    else:
        print("❌ Failed to register user")

    # 2. Login do usuário
    print("\n🔐 LOGIN TEST...")
    login_result = await user_manager.login_user(
        email="test@tradingbot.com",
        password="SecurePassword123!",
        ip_address="127.0.0.1",
        user_agent="Test Agent"
    )

    if login_result:
        print("✅ Login successful")
        print(f"   Access Token: {login_result['tokens']['access_token'][:50]}...")
        print(f"   Token Type: {login_result['tokens']['token_type']}")
        print(f"   Expires In: {login_result['tokens']['expires_in']} seconds")
    else:
        print("❌ Login failed")

    # 3. Obter perfil do usuário
    if test_user:
        print("\n👤 USER PROFILE...")
        profile = await user_manager.get_user_profile(test_user.id)
        if profile:
            print(f"✅ Profile retrieved:")
            print(f"   Email: {profile['email']}")
            print(f"   Full Name: {profile['full_name']}")
            print(f"   Country: {profile['country']}")
            print(f"   KYC Status: {profile['kyc_status']}")
            print(f"   Trading Settings: {profile['trading_settings']}")

    # 4. Gerar API Key
    if test_user:
        print("\n🔑 GENERATING API KEY...")
        api_key = await user_manager.generate_api_key(test_user.id, "Test API Key")
        if api_key:
            print(f"✅ API Key generated: {api_key}")
        else:
            print("❌ Failed to generate API key")

    # 5. Estatísticas do sistema
    print("\n📊 SYSTEM STATISTICS...")
    stats = await user_manager.get_user_stats()
    print(f"   Total Users: {stats.get('total_users', 0)}")
    print(f"   Active Users: {stats.get('active_users', 0)}")
    print(f"   Premium Users: {stats.get('premium_users', 0)}")
    print(f"   Trial Users: {stats.get('trial_users', 0)}")
    print(f"   Verified Users: {stats.get('verified_users', 0)}")
    print(f"   New Users Today: {stats.get('new_users_today', 0)}")

    print("\n" + "="*80)
    print("✅ USER MANAGEMENT SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_user_management_system())
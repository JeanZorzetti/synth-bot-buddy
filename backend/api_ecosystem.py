"""
API Ecosystem - Ecossistema de APIs
Sistema completo de APIs RESTful, WebSockets e SDK para desenvolvedores terceiros.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import base64
from collections import defaultdict, deque
from fastapi import FastAPI, HTTPException, Depends, status, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aioredis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from enterprise_platform import EnterprisePlatform, User, UserRole

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"

class EndpointCategory(Enum):
    TRADING = "trading"
    PORTFOLIO = "portfolio"
    ANALYTICS = "analytics"
    MARKET_DATA = "market_data"
    USER = "user"
    ADMIN = "admin"

@dataclass
class APIKey:
    """Chave de API"""
    key_id: str
    user_id: str
    api_key: str
    secret_key: str
    name: str
    permissions: List[str]
    rate_limit: int
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]

@dataclass
class APIUsage:
    """Uso da API"""
    api_key: str
    endpoint: str
    method: str
    timestamp: datetime
    response_time_ms: float
    status_code: int
    request_size: int
    response_size: int

@dataclass
class WebhookConfig:
    """Configuração de webhook"""
    webhook_id: str
    user_id: str
    url: str
    events: List[str]
    secret: str
    is_active: bool
    retry_config: Dict[str, Any]

# Modelos Pydantic para API
class TradingSignalRequest(BaseModel):
    symbol: str
    action: str = Field(..., regex="^(buy|sell|hold)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_size: Optional[float] = None
    timeframe: str = "1m"

class TradingSignalResponse(BaseModel):
    signal_id: str
    symbol: str
    action: str
    confidence: float
    timestamp: datetime
    execution_price: Optional[float] = None
    status: str

class PortfolioSummaryResponse(BaseModel):
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    positions_count: int
    cash_balance: float
    allocation: Dict[str, float]

class MarketDataRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1m"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)

class MarketDataResponse(BaseModel):
    symbol: str
    data: List[Dict[str, Any]]
    timeframe: str
    count: int

class RateLimiter:
    """Rate limiter customizado"""

    def __init__(self):
        self.redis_client = None
        self.local_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Inicializa rate limiter"""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Redis não disponível para rate limiting: {e}")

    async def check_rate_limit(self, key: str, limit: int, window: int = 3600) -> bool:
        """Verifica rate limit"""
        now = time.time()

        if self.redis_client:
            return await self._check_redis_rate_limit(key, limit, window, now)
        else:
            return await self._check_local_rate_limit(key, limit, window, now)

    async def _check_redis_rate_limit(self, key: str, limit: int, window: int, now: float) -> bool:
        """Rate limit usando Redis"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.zadd(key, {str(uuid.uuid4()): now})
            pipe.expire(key, window)

            results = await pipe.execute()
            current_count = results[1]

            return current_count < limit

        except Exception as e:
            logger.error(f"Erro no Redis rate limit: {e}")
            return True  # Allow on error

    async def _check_local_rate_limit(self, key: str, limit: int, window: int, now: float) -> bool:
        """Rate limit usando cache local"""
        timestamps = self.local_cache[key]

        # Remove timestamps antigos
        while timestamps and timestamps[0] < now - window:
            timestamps.popleft()

        if len(timestamps) < limit:
            timestamps.append(now)
            return True

        return False

class APIKeyManager:
    """Gerenciador de chaves de API"""

    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self.usage_logs: List[APIUsage] = []

    def generate_api_key(self, user_id: str, name: str, permissions: List[str],
                        rate_limit: int = 1000) -> APIKey:
        """Gera nova chave de API"""
        key_id = str(uuid.uuid4())
        api_key = self._generate_key()
        secret_key = self._generate_secret()

        api_key_obj = APIKey(
            key_id=key_id,
            user_id=user_id,
            api_key=api_key,
            secret_key=secret_key,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            is_active=True,
            created_at=datetime.now(),
            expires_at=None,
            last_used=None
        )

        self.api_keys[api_key] = api_key_obj
        return api_key_obj

    def _generate_key(self) -> str:
        """Gera chave de API"""
        return f"sbk_{secrets.token_urlsafe(32)}"

    def _generate_secret(self) -> str:
        """Gera chave secreta"""
        return secrets.token_urlsafe(64)

    def verify_api_signature(self, api_key: str, signature: str, timestamp: str,
                           body: str) -> bool:
        """Verifica assinatura HMAC da requisição"""
        api_key_obj = self.api_keys.get(api_key)
        if not api_key_obj or not api_key_obj.is_active:
            return False

        # Verificar timestamp (máximo 5 minutos)
        try:
            req_time = datetime.fromisoformat(timestamp)
            if abs((datetime.now() - req_time).total_seconds()) > 300:
                return False
        except:
            return False

        # Calcular assinatura esperada
        message = f"{api_key}{timestamp}{body}"
        expected_signature = hmac.new(
            api_key_obj.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def log_usage(self, api_key: str, endpoint: str, method: str,
                 response_time_ms: float, status_code: int,
                 request_size: int, response_size: int):
        """Registra uso da API"""
        usage = APIUsage(
            api_key=api_key,
            endpoint=endpoint,
            method=method,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            status_code=status_code,
            request_size=request_size,
            response_size=response_size
        )

        self.usage_logs.append(usage)

        # Atualizar último uso
        if api_key in self.api_keys:
            self.api_keys[api_key].last_used = datetime.now()

        # Manter apenas últimos 10000 logs
        if len(self.usage_logs) > 10000:
            self.usage_logs = self.usage_logs[-8000:]

class WebSocketManager:
    """Gerenciador de conexões WebSocket"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> channels
        self.channels: Dict[str, Set[str]] = defaultdict(set)       # channel -> user_ids

    async def connect(self, websocket: WebSocket, user_id: str):
        """Conecta usuário via WebSocket"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket conectado para usuário {user_id}")

    def disconnect(self, user_id: str):
        """Desconecta usuário"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]

            # Remove das subscrições
            for channel in list(self.subscriptions[user_id]):
                self.unsubscribe(user_id, channel)

            del self.subscriptions[user_id]

        logger.info(f"WebSocket desconectado para usuário {user_id}")

    def subscribe(self, user_id: str, channel: str):
        """Subscreve usuário a um canal"""
        self.subscriptions[user_id].add(channel)
        self.channels[channel].add(user_id)

    def unsubscribe(self, user_id: str, channel: str):
        """Remove subscrição de usuário"""
        self.subscriptions[user_id].discard(channel)
        self.channels[channel].discard(user_id)

    async def send_personal_message(self, user_id: str, message: Dict[str, Any]):
        """Envia mensagem para usuário específico"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {user_id}: {e}")
                self.disconnect(user_id)

    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Faz broadcast para todos os usuários de um canal"""
        user_ids = list(self.channels[channel])

        for user_id in user_ids:
            await self.send_personal_message(user_id, message)

class WebhookManager:
    """Gerenciador de webhooks"""

    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task = None

    async def start_webhook_worker(self):
        """Inicia worker de webhooks"""
        self.worker_task = asyncio.create_task(self._webhook_worker())

    async def stop_webhook_worker(self):
        """Para worker de webhooks"""
        if self.worker_task:
            self.worker_task.cancel()

    async def _webhook_worker(self):
        """Worker que processa eventos de webhook"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._process_webhook_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no worker de webhook: {e}")

    async def register_webhook(self, user_id: str, url: str, events: List[str],
                             retry_config: Dict[str, Any] = None) -> str:
        """Registra novo webhook"""
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)

        webhook = WebhookConfig(
            webhook_id=webhook_id,
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
            is_active=True,
            retry_config=retry_config or {"max_retries": 3, "backoff": "exponential"}
        )

        self.webhooks[webhook_id] = webhook
        return webhook_id

    async def trigger_webhook(self, event_type: str, data: Dict[str, Any], user_id: str = None):
        """Dispara webhook para um evento"""
        event = {
            "event_type": event_type,
            "data": data,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

        await self.event_queue.put(event)

    async def _process_webhook_event(self, event: Dict[str, Any]):
        """Processa evento de webhook"""
        event_type = event["event_type"]
        user_id = event.get("user_id")

        # Encontrar webhooks interessados neste evento
        relevant_webhooks = [
            webhook for webhook in self.webhooks.values()
            if webhook.is_active and event_type in webhook.events and
            (user_id is None or webhook.user_id == user_id)
        ]

        for webhook in relevant_webhooks:
            await self._send_webhook(webhook, event)

    async def _send_webhook(self, webhook: WebhookConfig, event: Dict[str, Any]):
        """Envia webhook"""
        try:
            import aiohttp

            # Preparar payload
            payload = {
                "webhook_id": webhook.webhook_id,
                "event": event,
                "timestamp": datetime.now().isoformat()
            }

            # Calcular assinatura
            payload_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                webhook.secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()

            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": f"sha256={signature}",
                "X-Webhook-Id": webhook.webhook_id
            }

            # Enviar webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook {webhook.webhook_id} enviado com sucesso")
                    else:
                        logger.warning(f"Webhook {webhook.webhook_id} falhou: {response.status}")

        except Exception as e:
            logger.error(f"Erro ao enviar webhook {webhook.webhook_id}: {e}")

class APIDocumentationGenerator:
    """Gerador de documentação de API"""

    def __init__(self, app: FastAPI):
        self.app = app

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Gera especificação OpenAPI"""
        return {
            "openapi": "3.0.2",
            "info": {
                "title": "Synth Bot Buddy API",
                "description": "API completa para trading autônomo com IA",
                "version": "2.0.0",
                "contact": {
                    "name": "Synth Bot Buddy",
                    "email": "api@synth-bot-buddy.com"
                }
            },
            "servers": [
                {"url": "https://api.synth-bot-buddy.com/v1", "description": "Production"},
                {"url": "https://api-staging.synth-bot-buddy.com/v1", "description": "Staging"}
            ],
            "paths": self._generate_paths(),
            "components": self._generate_components()
        }

    def _generate_paths(self) -> Dict[str, Any]:
        """Gera paths da API"""
        return {
            "/trading/signals": {
                "post": {
                    "summary": "Create Trading Signal",
                    "description": "Cria um sinal de trading com base na análise da IA",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TradingSignalRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Sinal criado com sucesso",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TradingSignalResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/portfolio/summary": {
                "get": {
                    "summary": "Get Portfolio Summary",
                    "description": "Obtém resumo do portfólio do usuário",
                    "responses": {
                        "200": {
                            "description": "Resumo do portfólio",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PortfolioSummaryResponse"}
                                }
                            }
                        }
                    }
                }
            }
        }

    def _generate_components(self) -> Dict[str, Any]:
        """Gera componentes da API"""
        return {
            "schemas": {
                "TradingSignalRequest": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "example": "EUR/USD"},
                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "position_size": {"type": "number", "minimum": 0}
                    },
                    "required": ["symbol", "action", "confidence"]
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "SignatureAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-Signature"
                }
            }
        }

class APIEcosystem:
    """Ecossistema de APIs principal"""

    def __init__(self, enterprise_platform: EnterprisePlatform):
        self.enterprise_platform = enterprise_platform

        # Componentes
        self.rate_limiter = RateLimiter()
        self.api_key_manager = APIKeyManager()
        self.websocket_manager = WebSocketManager()
        self.webhook_manager = WebhookManager()

        # FastAPI app
        self.app = FastAPI(
            title="Synth Bot Buddy API",
            description="API Enterprise para Trading Autônomo com IA",
            version="2.0.0"
        )

        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Rate limiting
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        # Configurar rotas
        self._setup_routes()

        # Documentação
        self.doc_generator = APIDocumentationGenerator(self.app)

    async def initialize(self):
        """Inicializa ecossistema de APIs"""
        await self.rate_limiter.initialize()
        await self.webhook_manager.start_webhook_worker()
        logger.info("API Ecosystem inicializado")

    def _setup_routes(self):
        """Configura rotas da API"""
        # Authentication
        @self.app.post("/auth/login")
        async def login(request: Request, username: str, password: str):
            """Login de usuário"""
            try:
                user, token = await self.enterprise_platform.authenticate_and_authorize(
                    username, password, ip_address=request.client.host
                )

                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "user": {
                        "user_id": user.user_id,
                        "username": user.username,
                        "role": user.role.value
                    }
                }
            except HTTPException as e:
                raise e
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # API Key Management
        @self.app.post("/api-keys")
        async def create_api_key(request: Request, name: str, permissions: List[str],
                               current_user: User = Depends(self._get_current_user)):
            """Cria nova API key"""
            api_key = self.api_key_manager.generate_api_key(
                current_user.user_id, name, permissions
            )

            return {
                "key_id": api_key.key_id,
                "api_key": api_key.api_key,
                "secret_key": api_key.secret_key,
                "permissions": api_key.permissions
            }

        # Trading Signals
        @self.app.post("/v1/trading/signals", response_model=TradingSignalResponse)
        async def create_trading_signal(signal_request: TradingSignalRequest,
                                      current_user: User = Depends(self._get_current_user_api)):
            """Cria sinal de trading"""
            # Verificar permissões
            if not self.enterprise_platform.user_manager.permission_manager.check_permission(
                current_user, "trading:write"
            ):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            # Simular criação de sinal
            signal_id = str(uuid.uuid4())

            response = TradingSignalResponse(
                signal_id=signal_id,
                symbol=signal_request.symbol,
                action=signal_request.action,
                confidence=signal_request.confidence,
                timestamp=datetime.now(),
                status="created"
            )

            # Trigger webhook
            await self.webhook_manager.trigger_webhook(
                "trading_signal_created",
                asdict(response),
                current_user.user_id
            )

            return response

        # Portfolio Summary
        @self.app.get("/v1/portfolio/summary", response_model=PortfolioSummaryResponse)
        async def get_portfolio_summary(current_user: User = Depends(self._get_current_user_api)):
            """Obtém resumo do portfólio"""
            # Verificar permissões
            if not self.enterprise_platform.user_manager.permission_manager.check_permission(
                current_user, "portfolio:read"
            ):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            # Simular dados do portfólio
            return PortfolioSummaryResponse(
                total_value=10000.0,
                unrealized_pnl=150.25,
                realized_pnl=500.75,
                positions_count=3,
                cash_balance=2000.0,
                allocation={"EUR/USD": 0.4, "GBP/USD": 0.3, "USD/JPY": 0.3}
            )

        # Market Data
        @self.app.post("/v1/market-data", response_model=List[MarketDataResponse])
        async def get_market_data(data_request: MarketDataRequest,
                                current_user: User = Depends(self._get_current_user_api)):
            """Obtém dados de mercado"""
            # Verificar permissões
            if not self.enterprise_platform.user_manager.permission_manager.check_permission(
                current_user, "analytics:read"
            ):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            responses = []
            for symbol in data_request.symbols:
                # Simular dados de mercado
                mock_data = [
                    {
                        "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                        "open": 1.1000 + i * 0.0001,
                        "high": 1.1005 + i * 0.0001,
                        "low": 1.0995 + i * 0.0001,
                        "close": 1.1002 + i * 0.0001,
                        "volume": 1000 + i * 10
                    }
                    for i in range(min(data_request.limit, 100))
                ]

                responses.append(MarketDataResponse(
                    symbol=symbol,
                    data=mock_data,
                    timeframe=data_request.timeframe,
                    count=len(mock_data)
                ))

            return responses

        # WebSocket endpoint
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            """Endpoint WebSocket"""
            await self.websocket_manager.connect(websocket, user_id)

            try:
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(user_id, data)

            except WebSocketDisconnect:
                self.websocket_manager.disconnect(user_id)

        # API Documentation
        @self.app.get("/docs/openapi.json")
        async def get_openapi_spec():
            """Especificação OpenAPI"""
            return self.doc_generator.generate_openapi_spec()

        # Health Check
        @self.app.get("/health")
        async def health_check():
            """Health check do sistema"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }

    async def _get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Obtém usuário atual via JWT"""
        token = credentials.credentials
        payload = self.enterprise_platform.user_manager.auth_manager.verify_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = await self.enterprise_platform.user_manager.get_user_by_id(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    async def _get_current_user_api(self, request: Request):
        """Obtém usuário atual via API Key"""
        api_key = request.headers.get("X-API-Key")
        signature = request.headers.get("X-Signature")
        timestamp = request.headers.get("X-Timestamp")

        if not api_key:
            raise HTTPException(status_code=401, detail="API Key required")

        # Verificar rate limit
        if not await self.rate_limiter.check_rate_limit(api_key, 1000):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Verificar assinatura se fornecida
        if signature and timestamp:
            body = await request.body()
            if not self.api_key_manager.verify_api_signature(
                api_key, signature, timestamp, body.decode()
            ):
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Buscar API key
        api_key_obj = self.api_key_manager.api_keys.get(api_key)
        if not api_key_obj or not api_key_obj.is_active:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Buscar usuário
        user = await self.enterprise_platform.user_manager.get_user_by_id(api_key_obj.user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    async def _handle_websocket_message(self, user_id: str, data: Dict[str, Any]):
        """Processa mensagem WebSocket"""
        message_type = data.get("type")

        if message_type == "subscribe":
            channel = data.get("channel")
            if channel:
                self.websocket_manager.subscribe(user_id, channel)
                await self.websocket_manager.send_personal_message(user_id, {
                    "type": "subscription_confirmed",
                    "channel": channel
                })

        elif message_type == "unsubscribe":
            channel = data.get("channel")
            if channel:
                self.websocket_manager.unsubscribe(user_id, channel)
                await self.websocket_manager.send_personal_message(user_id, {
                    "type": "unsubscription_confirmed",
                    "channel": channel
                })

    async def broadcast_market_update(self, symbol: str, data: Dict[str, Any]):
        """Faz broadcast de atualização de mercado"""
        message = {
            "type": "market_update",
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_manager.broadcast_to_channel(f"market_{symbol}", message)

    async def get_api_usage_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """Obtém estatísticas de uso da API"""
        # Filtrar logs por usuário se especificado
        relevant_logs = self.api_key_manager.usage_logs

        if user_id:
            user_api_keys = [
                key for key, api_key_obj in self.api_key_manager.api_keys.items()
                if api_key_obj.user_id == user_id
            ]
            relevant_logs = [
                log for log in relevant_logs
                if log.api_key in user_api_keys
            ]

        # Calcular estatísticas
        total_requests = len(relevant_logs)
        if total_requests == 0:
            return {"total_requests": 0}

        avg_response_time = sum(log.response_time_ms for log in relevant_logs) / total_requests
        error_rate = sum(1 for log in relevant_logs if log.status_code >= 400) / total_requests

        # Endpoints mais usados
        endpoint_usage = defaultdict(int)
        for log in relevant_logs:
            endpoint_usage[log.endpoint] += 1

        return {
            "total_requests": total_requests,
            "average_response_time_ms": avg_response_time,
            "error_rate": error_rate,
            "top_endpoints": dict(sorted(endpoint_usage.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    async def shutdown(self):
        """Encerra ecossistema de APIs"""
        await self.webhook_manager.stop_webhook_worker()
        logger.info("API Ecosystem encerrado")
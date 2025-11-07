# 🏗️ ARQUITETURA: Sistema de Execução de Ordens Deriv

**Documento Técnico Complementar ao Plano de Execução**

---

## 📐 VISÃO GERAL DA ARQUITETURA

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUÁRIO FINAL                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React/TypeScript)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ OrderForm    │  │ OrderService │  │ OrderHistory │         │
│  │ Component    │─→│   (API)      │  │  Component   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP POST /api/order/execute
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI/Python)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  REST API    │  │ OrderManager │  │  Validator   │         │
│  │  Endpoint    │─→│   Service    │─→│   Service    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           DerivAPI Client (WebSocket)               │       │
│  │  • connect()  • authorize()  • buy()  • sell()      │       │
│  └─────────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────────┘
                         │ WebSocket (wss://)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DERIV API (WebSocket Server)                  │
│  wss://ws.derivws.com/websockets/v3?app_id=1089                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FLUXO DE DADOS DETALHADO

### 1. Fluxo de Execução de Ordem (Happy Path)

```
USUÁRIO                FRONTEND              BACKEND              DERIV API
   │                      │                     │                      │
   │  Preenche Form       │                     │                      │
   ├─────────────────────►│                     │                      │
   │                      │                     │                      │
   │  Clica "Executar"    │                     │                      │
   ├─────────────────────►│                     │                      │
   │                      │                     │                      │
   │                      │ POST /api/order/    │                      │
   │                      │      execute        │                      │
   │                      ├────────────────────►│                      │
   │                      │                     │                      │
   │                      │                     │  WebSocket Connect   │
   │                      │                     ├─────────────────────►│
   │                      │                     │◄─────────────────────┤
   │                      │                     │  Connected (OK)      │
   │                      │                     │                      │
   │                      │                     │  authorize(token)    │
   │                      │                     ├─────────────────────►│
   │                      │                     │◄─────────────────────┤
   │                      │                     │  Authorized (LoginID)│
   │                      │                     │                      │
   │                      │                     │  get_proposal(params)│
   │                      │                     ├─────────────────────►│
   │                      │                     │◄─────────────────────┤
   │                      │                     │  Proposal (price)    │
   │                      │                     │                      │
   │                      │                     │  buy(proposal_id)    │
   │                      │                     ├─────────────────────►│
   │                      │                     │◄─────────────────────┤
   │                      │                     │  Buy Response (ID)   │
   │                      │                     │                      │
   │                      │  Response {success} │                      │
   │                      │◄────────────────────┤                      │
   │                      │                     │                      │
   │  Exibe Resultado     │                     │                      │
   │◄─────────────────────┤                     │                      │
   │  "Ordem #12345678"   │                     │                      │
   │                      │                     │                      │
```

### 2. Fluxo de Erro (Error Handling)

```
Possíveis Pontos de Falha:

1. VALIDAÇÃO FRONTEND
   ├─ Token vazio → Alerta imediato
   ├─ Valor inválido → Validação de campo
   └─ Campos obrigatórios → Desabilita botão

2. VALIDAÇÃO BACKEND
   ├─ Token inválido → HTTP 401 Unauthorized
   ├─ Parâmetros inválidos → HTTP 400 Bad Request
   └─ Servidor indisponível → HTTP 503 Service Unavailable

3. CONEXÃO WEBSOCKET
   ├─ Timeout → Retry (3x) → Falha
   ├─ Conexão recusada → Erro de rede
   └─ Disconnected → Reconexão automática

4. API DERIV
   ├─ Autenticação falhou → "Token inválido"
   ├─ Saldo insuficiente → "Insufficient balance"
   ├─ Mercado fechado → "Market closed"
   └─ Proposta rejeitada → "Invalid proposal"

5. EXECUÇÃO DA ORDEM
   ├─ Timeout na compra → Rollback
   ├─ Preço mudou → Tentar novamente
   └─ Erro desconhecido → Log + Alerta admin
```

---

## 🗂️ ESTRUTURA DE ARQUIVOS

```
synth-bot-buddy-main/
│
├── backend/
│   ├── main.py                      # FastAPI app principal
│   ├── deriv_api.py                 # Cliente WebSocket Deriv (JÁ EXISTE)
│   ├── test_simple_order.py         # 🆕 Script de teste isolado
│   │
│   ├── services/                    # 🆕 Camada de serviços
│   │   ├── order_service.py         # Lógica de execução de ordem
│   │   ├── validation_service.py    # Validações de negócio
│   │   └── logging_service.py       # Sistema de logs
│   │
│   ├── models/                      # 🆕 Modelos de dados
│   │   ├── order_models.py          # Pydantic models para ordem
│   │   └── response_models.py       # Modelos de resposta API
│   │
│   └── routes/                      # 🆕 Rotas da API
│       └── order_routes.py          # Endpoints de ordem
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── orders/              # 🆕 Componentes de ordem
│   │   │       ├── OrderExecutor.tsx
│   │   │       ├── OrderForm.tsx
│   │   │       ├── OrderResult.tsx
│   │   │       └── OrderHistory.tsx
│   │   │
│   │   ├── services/
│   │   │   └── orderService.ts      # 🆕 API client para ordens
│   │   │
│   │   ├── hooks/
│   │   │   └── useOrder.ts          # 🆕 Hook customizado
│   │   │
│   │   └── types/
│   │       └── order.types.ts       # 🆕 TypeScript types
│   │
├── docs/
│   ├── PLANO-EXECUCAO-ORDEM-DERIV.md        # ✅ Plano estratégico
│   ├── ARQUITETURA-EXECUCAO-ORDEM.md        # ✅ Este documento
│   └── API-ENDPOINT-DOCUMENTATION.md        # 🆕 Docs da API
│
└── tests/                           # 🆕 Testes automatizados
    ├── test_order_service.py
    ├── test_deriv_api.py
    └── test_integration.py
```

---

## 📦 MODELOS DE DADOS

### Backend (Pydantic Models)

```python
# backend/models/order_models.py

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

class OrderRequest(BaseModel):
    """Request para executar ordem"""
    token: str = Field(..., min_length=10, description="Token API Deriv")
    contract_type: Literal["CALL", "PUT"] = Field(..., description="Tipo de contrato")
    symbol: str = Field(default="R_75", description="Símbolo do ativo")
    amount: float = Field(..., gt=0, le=100, description="Valor da aposta em USD")
    duration: int = Field(..., gt=0, le=60, description="Duração em minutos")
    duration_unit: Literal["m", "h", "d"] = Field(default="m", description="Unidade de duração")

    @validator('amount')
    def validate_amount(cls, v):
        if v < 0.35:  # Mínimo da Deriv
            raise ValueError("Valor mínimo: $0.35")
        return round(v, 2)

class OrderResponse(BaseModel):
    """Response da execução de ordem"""
    success: bool
    contract_id: Optional[int] = None
    buy_price: Optional[float] = None
    payout: Optional[float] = None
    longcode: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[dict] = None

class ProposalData(BaseModel):
    """Dados da proposta de contrato"""
    id: str
    ask_price: float
    payout: float
    spot: float
    spot_time: int
    display_value: str
```

### Frontend (TypeScript Types)

```typescript
// frontend/src/types/order.types.ts

export interface OrderParams {
  token: string;
  contractType: 'CALL' | 'PUT';
  symbol: string;
  amount: number;
  duration: number;
  durationUnit?: 'm' | 'h' | 'd';
}

export interface OrderResult {
  success: boolean;
  contractId?: number;
  buyPrice?: number;
  payout?: number;
  longcode?: string;
  status?: string;
  error?: string;
  errorDetails?: Record<string, any>;
}

export interface OrderHistoryItem {
  id: string;
  timestamp: Date;
  contractId: number;
  contractType: 'CALL' | 'PUT';
  symbol: string;
  amount: number;
  payout: number;
  result?: 'win' | 'loss' | 'pending';
}
```

---

## 🔐 SEGURANÇA E VALIDAÇÕES

### Camadas de Validação

```
┌───────────────────────────────────────────────────────────┐
│ CAMADA 1: Frontend Validation (Imediata)                 │
├───────────────────────────────────────────────────────────┤
│ • Campos obrigatórios preenchidos                         │
│ • Formato de token (min 10 chars)                         │
│ • Valor entre $0.35 - $100                                │
│ • Duração entre 1-60 minutos                              │
│ • Símbolo válido (select list)                            │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ CAMADA 2: Backend Input Validation (Request)             │
├───────────────────────────────────────────────────────────┤
│ • Pydantic model validation                               │
│ • Type checking automático                                │
│ • Range validation                                        │
│ • Sanitização de inputs                                   │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ CAMADA 3: Business Logic Validation (Service)            │
├───────────────────────────────────────────────────────────┤
│ • Token válido (test authorize)                           │
│ • Saldo suficiente (get_balance)                          │
│ • Mercado aberto (is_market_open)                         │
│ • Rate limit (max 10 orders/min)                          │
│ • Blacklist de símbolos                                   │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ CAMADA 4: Deriv API Validation (External)                │
├───────────────────────────────────────────────────────────┤
│ • Token scopes (Read + Trade)                             │
│ • Proposta válida                                         │
│ • Preço aceito                                            │
│ • Execução confirmada                                     │
└───────────────────────────────────────────────────────────┘
```

### Rate Limiting

```python
# Implementação simples de rate limiting
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        now = datetime.now()
        user_requests = self.requests[user_id]

        # Limpar requisições antigas
        user_requests[:] = [req for req in user_requests if now - req < self.window]

        if len(user_requests) >= self.max_requests:
            return False

        user_requests.append(now)
        return True
```

---

## 📊 MONITORAMENTO E LOGGING

### Níveis de Log

```python
# Estrutura de logging

import logging
from datetime import datetime

# Configuração
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orders.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Eventos a logar:

# 1. INFO: Operações normais
logger.info(f"Order received: {order_id}")
logger.info(f"Proposal obtained: ${proposal_price}")
logger.info(f"Order executed: Contract #{contract_id}")

# 2. WARNING: Situações suspeitas
logger.warning(f"Low balance warning: ${balance}")
logger.warning(f"Rate limit approaching: {requests}/10")

# 3. ERROR: Erros recuperáveis
logger.error(f"Order failed: {error_message}")
logger.error(f"WebSocket timeout, retrying...")

# 4. CRITICAL: Erros graves
logger.critical(f"Cannot connect to Deriv API")
logger.critical(f"Database connection lost")
```

### Métricas a Monitorar

```
┌──────────────────────────────────────────────────────────┐
│ MÉTRICAS DE PERFORMANCE                                  │
├──────────────────────────────────────────────────────────┤
│ • Tempo de resposta por endpoint (avg, p95, p99)         │
│ • Taxa de sucesso/falha de ordens                        │
│ • Tempo de execução WebSocket                            │
│ • Latência até Deriv API                                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ MÉTRICAS DE NEGÓCIO                                      │
├──────────────────────────────────────────────────────────┤
│ • Número de ordens executadas/dia                        │
│ • Volume total negociado                                 │
│ • Taxa de win/loss                                       │
│ • Símbolos mais negociados                               │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ MÉTRICAS DE ERRO                                         │
├──────────────────────────────────────────────────────────┤
│ • Tipos de erro mais comuns                              │
│ • Taxa de timeout                                        │
│ • Taxa de autenticação falhada                           │
│ • Tentativas de uso após rate limit                      │
└──────────────────────────────────────────────────────────┘
```

---

## 🧪 ESTRATÉGIA DE TESTES

### Pirâmide de Testes

```
                    ┌────────────┐
                    │   E2E      │  ← Poucos, caros
                    │  (1 teste) │
                  ┌─┴────────────┴─┐
                  │  Integration   │  ← Moderados
                  │   (5 testes)   │
              ┌───┴────────────────┴───┐
              │       Unit Tests       │  ← Muitos, rápidos
              │      (20+ testes)      │
              └────────────────────────┘
```

### Testes Unitários

```python
# tests/test_order_service.py

import pytest
from services.order_service import OrderService
from models.order_models import OrderRequest

def test_validate_order_request():
    """Testa validação de request de ordem"""
    request = OrderRequest(
        token="test_token_123",
        contract_type="CALL",
        symbol="R_75",
        amount=1.0,
        duration=5
    )
    assert request.amount == 1.0
    assert request.contract_type == "CALL"

def test_order_request_validation_fails():
    """Testa que validação falha com dados inválidos"""
    with pytest.raises(ValueError):
        OrderRequest(
            token="test",
            contract_type="INVALID",  # ← Tipo inválido
            symbol="R_75",
            amount=-1.0,  # ← Valor negativo
            duration=5
        )
```

### Testes de Integração

```python
# tests/test_integration.py

import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_execute_order_endpoint():
    """Testa endpoint de execução de ordem"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/order/execute", json={
            "token": "test_token",
            "contract_type": "CALL",
            "symbol": "R_75",
            "amount": 1.0,
            "duration": 5
        })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] in [True, False]
```

### Teste E2E (Manual)

```
1. Iniciar backend (python start.py)
2. Iniciar frontend (npm run dev)
3. Abrir navegador (http://localhost:5173)
4. Navegar para "Executar Ordem"
5. Preencher formulário:
   - Token: [seu token demo]
   - Tipo: CALL
   - Símbolo: R_75
   - Valor: $1.00
   - Duração: 5 min
6. Clicar "Executar Ordem"
7. Verificar:
   ✓ Loading aparecer
   ✓ Sucesso ou erro claro
   ✓ Contract ID exibido
   ✓ Link para Deriv
8. Abrir plataforma Deriv
9. Verificar contrato aparece
10. Aguardar resultado
```

---

## 🚀 OTIMIZAÇÕES FUTURAS

### Performance

1. **Connection Pooling**
   - Manter WebSocket persistente
   - Reusar conexão para múltiplas ordens
   - Reduzir latência de conexão

2. **Caching**
   - Cache de símbolos ativos (TTL: 5min)
   - Cache de proposals (TTL: 10s)
   - Redis para cache distribuído

3. **Async Processing**
   - Queue de ordens (Celery/Redis)
   - Processamento em background
   - Notificações via WebSocket

### Escalabilidade

```
┌─────────────────────────────────────────────────────────┐
│ LOAD BALANCER (Nginx)                                   │
└───┬─────────────────────────────────────────────────┬───┘
    │                                                 │
    ▼                                                 ▼
┌────────────────┐                            ┌────────────────┐
│  Backend #1    │                            │  Backend #2    │
│  (FastAPI)     │                            │  (FastAPI)     │
└────────────────┘                            └────────────────┘
    │                                                 │
    └─────────────────────┬───────────────────────────┘
                          ▼
                  ┌────────────────┐
                  │  Redis Cache   │
                  │  + Queue       │
                  └────────────────┘
```

---

## 📚 REFERÊNCIAS TÉCNICAS

### Deriv API

- **Documentação Oficial:** https://api.deriv.com/docs/
- **API Explorer:** https://api.deriv.com/api-explorer
- **WebSocket Endpoint:** wss://ws.derivws.com/websockets/v3
- **App ID Demo:** 1089

### Tecnologias Utilizadas

- **Backend:** Python 3.11, FastAPI, websockets, pydantic
- **Frontend:** React 18, TypeScript, Vite
- **Comunicação:** REST API (HTTP), WebSocket (WSS)
- **Validação:** Pydantic (backend), Zod (frontend opcional)

### Padrões Implementados

- **Repository Pattern:** Separação de lógica de acesso a dados
- **Service Layer:** Lógica de negócio isolada
- **DTO (Data Transfer Objects):** Pydantic models
- **Error Handling:** Try-catch com logging
- **Dependency Injection:** FastAPI dependencies

---

## ✅ CONCLUSÃO

Esta arquitetura foi desenhada para ser:

- **Simples:** Fácil de entender e manter
- **Escalável:** Preparada para crescimento
- **Segura:** Múltiplas camadas de validação
- **Testável:** Estrutura que facilita testes
- **Documentada:** Código auto-explicativo

**Próximo Passo:** Executar o plano documentado em `PLANO-EXECUCAO-ORDEM-DERIV.md`

---

**Documento criado em:** 2025-11-06
**Versão:** 1.0
**Status:** ✅ Completo

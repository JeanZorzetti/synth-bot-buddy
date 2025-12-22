# ABUTRE - REFATORAÇÃO PARA DASHBOARD DE MONITORAMENTO

**Data:** 2025-12-22
**Objetivo:** Transformar Abutre de bot executor para dashboard de visualização de dados do Deriv Bot XML

---

## 🎯 Visão Geral

### Arquitetura Atual (Problemática)
```
┌─────────────────┐
│  Abutre Bot     │  ← Executa trades via Python/Deriv API
│  (Python)       │  ← Alta latência (WebSocket + processamento)
└────────┬────────┘
         │
         ▼
   ┌────────────┐
   │ Deriv API  │
   └────────────┘
```

**Problemas:**
- ❌ Latência de rede (Python ↔ Deriv API)
- ❌ Latência de processamento (event loop, DB writes)
- ❌ Complexidade de manutenção (API client, websocket, error handling)
- ❌ Rate limits da API

### Arquitetura Nova (Proposta)
```
┌──────────────────┐
│   Deriv Bot      │  ← Executa trades via XML (zero latência)
│   (XML/Blockly)  │  ← Roda no próprio browser do Deriv
└────────┬─────────┘
         │ HTTP POST
         ▼
   ┌──────────────────┐
   │  Abutre API      │  ← Recebe logs/eventos
   │  (FastAPI)       │  ← Armazena no database
   └────────┬─────────┘
            │ WebSocket
            ▼
   ┌──────────────────┐
   │  Dashboard       │  ← Visualiza dados em tempo real
   │  (React/Next.js) │  ← Equity curve, trades, metrics
   └──────────────────┘
```

**Vantagens:**
- ✅ **Zero latência** - XML roda direto no Deriv Bot
- ✅ **Simplicidade** - Dashboard apenas visualiza dados
- ✅ **Confiabilidade** - Deriv Bot é testado e estável
- ✅ **Escalabilidade** - API recebe dados de múltiplos bots
- ✅ **No rate limits** - XML não tem limites da API

---

## 📦 Componentes da Nova Arquitetura

### 1. Deriv Bot XML (Executor)
**Responsabilidade:** Executar a estratégia Abutre e enviar eventos para API

**Funcionalidades:**
- Monitorar candles de V100
- Detectar streaks de 8+ velas
- Executar Martingale (até Level 10)
- Enviar eventos via HTTP POST:
  - `candle_closed`
  - `trigger_detected`
  - `trade_opened`
  - `trade_closed`
  - `balance_update`

**Localização:** `backend/bots/abutre/deriv_bot_xml/abutre_strategy.xml`

---

### 2. Abutre API (Ingestão de Dados)
**Responsabilidade:** Receber eventos do XML e persistir no database

**Endpoints:**

#### `POST /api/abutre/events/candle`
```json
{
  "timestamp": "2025-12-22T18:30:00Z",
  "symbol": "1HZ100V",
  "open": 663.59,
  "high": 663.92,
  "low": 663.12,
  "close": 663.60,
  "color": 1  // 1 = green, -1 = red
}
```

#### `POST /api/abutre/events/trigger`
```json
{
  "timestamp": "2025-12-22T18:30:00Z",
  "streak_count": 8,
  "direction": "GREEN"
}
```

#### `POST /api/abutre/events/trade_opened`
```json
{
  "timestamp": "2025-12-22T18:31:00Z",
  "trade_id": "abc123",
  "direction": "PUT",  // Betting AGAINST the streak
  "stake": 1.0,
  "level": 1,
  "contract_id": "12345678"
}
```

#### `POST /api/abutre/events/trade_closed`
```json
{
  "timestamp": "2025-12-22T18:32:00Z",
  "trade_id": "abc123",
  "result": "WIN",  // WIN, LOSS, STOP_LOSS
  "profit": 0.95,
  "balance": 10031.49,
  "max_level_reached": 1
}
```

#### `POST /api/abutre/events/balance`
```json
{
  "timestamp": "2025-12-22T18:32:00Z",
  "balance": 10031.49
}
```

**Arquivo:** `backend/api/routes/abutre_events.py`

---

### 3. Database Schema

#### Tabela: `abutre_candles`
```sql
CREATE TABLE abutre_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL DEFAULT '1HZ100V',
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    color INTEGER NOT NULL,  -- 1 (green), -1 (red)
    source TEXT DEFAULT 'deriv_bot_xml'
);
```

#### Tabela: `abutre_triggers`
```sql
CREATE TABLE abutre_triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    streak_count INTEGER NOT NULL,
    direction TEXT NOT NULL  -- 'GREEN' ou 'RED'
);
```

#### Tabela: `abutre_trades`
```sql
CREATE TABLE abutre_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    contract_id TEXT,

    -- Entry
    entry_time DATETIME NOT NULL,
    direction TEXT NOT NULL,  -- 'CALL' ou 'PUT'
    initial_stake REAL NOT NULL,

    -- Progression
    max_level_reached INTEGER NOT NULL,
    total_staked REAL NOT NULL,  -- Soma de todos os stakes

    -- Exit
    exit_time DATETIME,
    result TEXT,  -- 'WIN', 'LOSS', 'STOP_LOSS'
    profit REAL,
    balance_after REAL,

    -- Metadata
    source TEXT DEFAULT 'deriv_bot_xml'
);
```

#### Tabela: `abutre_balance_history`
```sql
CREATE TABLE abutre_balance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    balance REAL NOT NULL,
    peak_balance REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    roi_pct REAL NOT NULL
);
```

**Arquivo:** `backend/database/abutre_schema.sql`

---

### 4. Dashboard (Visualização)

**Componentes React:**

#### `AbutreDashboard.tsx`
- Metrics cards (Balance, ROI, Win Rate, Drawdown)
- Equity Curve
- Recent Trades Table
- Market Monitor (current streak)

**Dados em tempo real via WebSocket:**
- `candle_closed` → Atualiza Market Monitor
- `trade_closed` → Atualiza Trades Table + Metrics
- `balance_update` → Atualiza Balance Card

**Dados históricos via API REST:**
- `GET /api/abutre/stats` → Métricas agregadas
- `GET /api/abutre/trades?limit=50` → Últimos trades
- `GET /api/abutre/balance_history` → Equity curve

**Arquivo:** `frontend/src/pages/AbutreDashboard.tsx`

---

## 🚧 Plano de Implementação

### Fase 1: Backend (Ingestão de Dados)
- [ ] Criar `backend/api/routes/abutre_events.py`
- [ ] Implementar endpoints POST para cada tipo de evento
- [ ] Criar `backend/database/abutre_repository.py` para persistência
- [ ] Adicionar validação de dados (Pydantic schemas)
- [ ] Broadcast de eventos via WebSocket para dashboard

### Fase 2: Database
- [ ] Criar migration script `backend/database/migrations/003_abutre_events.sql`
- [ ] Executar migration no database local
- [ ] Testar CRUD de todos os eventos

### Fase 3: Deriv Bot XML
- [ ] Criar `backend/bots/abutre/deriv_bot_xml/abutre_strategy.xml`
- [ ] Implementar lógica de streak detection
- [ ] Implementar Martingale com 10 níveis
- [ ] Adicionar HTTP POST para cada evento
- [ ] Testar no Deriv Bot sandbox

### Fase 4: Dashboard (Frontend)
- [ ] Manter componentes atuais (AbutreDashboard, EquityCurve, etc)
- [ ] Modificar `useWebSocket` para escutar eventos do XML
- [ ] Criar `useDashboard` hooks para queries REST
- [ ] Adicionar indicador "Live" quando bot XML está rodando
- [ ] Remover botões Start/Stop (não aplicável)

### Fase 5: Documentação
- [ ] Guia de setup do XML no Deriv Bot
- [ ] Documentação da API de eventos
- [ ] Exemplos de payload para cada endpoint
- [ ] Troubleshooting common issues

---

## 🔧 Mudanças Necessárias

### Arquivos a MANTER:
- ✅ `frontend/src/pages/AbutreDashboard.tsx` (adaptar)
- ✅ `frontend/src/components/abutre/*` (manter visualização)
- ✅ `backend/database/abutre.db` (schema adaptado)
- ✅ `backend/bots/abutre/config.py` (configs da estratégia)

### Arquivos a REMOVER/ARQUIVAR:
- ❌ `backend/bots/abutre/main.py` → Arquivar em `_archive/`
- ❌ `backend/bots/abutre/core/deriv_api_client.py` → Não precisa mais
- ❌ `backend/bots/abutre/core/market_data_handler.py` → Não precisa mais
- ❌ `backend/bots/abutre/core/order_executor.py` → Não precisa mais
- ❌ `backend/bots/abutre/core/websocket_server.py` → Substituir por eventos HTTP

### Arquivos NOVOS:
- ✨ `backend/api/routes/abutre_events.py`
- ✨ `backend/database/abutre_repository.py`
- ✨ `backend/bots/abutre/deriv_bot_xml/abutre_strategy.xml`
- ✨ `backend/bots/abutre/deriv_bot_xml/README.md`
- ✨ `backend/database/migrations/003_abutre_events.sql`

---

## 📊 Fluxo de Dados

### 1. Execução no Deriv Bot
```
Deriv Bot XML (Browser)
  ↓ Detecta candle fechado
  ↓ HTTP POST /api/abutre/events/candle
FastAPI
  ↓ Valida payload
  ↓ Persiste no DB (abutre_candles)
  ↓ Broadcast via WebSocket
Dashboard
  ↓ Atualiza Market Monitor
```

### 2. Trade Lifecycle
```
Deriv Bot XML
  ↓ Streak >= 8 detectado
  ↓ POST /api/abutre/events/trigger
  ↓ Abre trade (PUT/CALL)
  ↓ POST /api/abutre/events/trade_opened
  ↓ Trade finaliza (WIN/LOSS)
  ↓ POST /api/abutre/events/trade_closed
  ↓ POST /api/abutre/events/balance
FastAPI
  ↓ Persiste trades + balance
  ↓ Calcula métricas (ROI, Win Rate, DD)
  ↓ Broadcast via WebSocket
Dashboard
  ↓ Atualiza Equity Curve
  ↓ Adiciona linha na Trades Table
  ↓ Atualiza Metrics Cards
```

---

## 🧪 Exemplo de Payload Completo

### Cenário: Trade WIN no Level 1

#### 1. Candle fechado (8ª vela verde)
```bash
curl -X POST http://localhost:8000/api/abutre/events/candle \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:30:00Z",
    "symbol": "1HZ100V",
    "open": 663.50,
    "high": 663.92,
    "low": 663.12,
    "close": 663.60,
    "color": 1
  }'
```

#### 2. Trigger detectado
```bash
curl -X POST http://localhost:8000/api/abutre/events/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:30:05Z",
    "streak_count": 8,
    "direction": "GREEN"
  }'
```

#### 3. Trade aberto (PUT - contra a tendência)
```bash
curl -X POST http://localhost:8000/api/abutre/events/trade_opened \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:31:00Z",
    "trade_id": "trade_1703271060",
    "direction": "PUT",
    "stake": 1.0,
    "level": 1,
    "contract_id": "12345678"
  }'
```

#### 4. Próximo candle fecha VERMELHO → WIN!
```bash
curl -X POST http://localhost:8000/api/abutre/events/candle \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:31:00Z",
    "symbol": "1HZ100V",
    "open": 663.60,
    "high": 663.70,
    "low": 662.90,
    "close": 663.10,
    "color": -1
  }'
```

#### 5. Trade fechado com WIN
```bash
curl -X POST http://localhost:8000/api/abutre/events/trade_closed \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:32:00Z",
    "trade_id": "trade_1703271060",
    "result": "WIN",
    "profit": 0.95,
    "balance": 10001.95,
    "max_level_reached": 1
  }'
```

#### 6. Balance update
```bash
curl -X POST http://localhost:8000/api/abutre/events/balance \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:32:00Z",
    "balance": 10001.95
  }'
```

---

## 📈 Vantagens da Nova Arquitetura

| Aspecto | Antes (Python Bot) | Depois (XML + Dashboard) |
|---------|-------------------|--------------------------|
| **Latência** | ~200-500ms | ~5-10ms (XML nativo) |
| **Manutenção** | Alta (API client, WS, errors) | Baixa (apenas API de ingest) |
| **Confiabilidade** | Dependente de Python/network | Deriv Bot testado e estável |
| **Escalabilidade** | 1 bot por processo | N bots → 1 dashboard |
| **Rate Limits** | Sim (API limits) | Não (XML não tem limits) |
| **Deploy** | Servidor Python 24/7 | XML roda no browser do usuário |
| **Custo** | Servidor dedicado | Apenas hosting do dashboard |

---

## 🎯 Próximos Passos

1. ✅ Criar este documento de arquitetura
2. ⏳ Implementar endpoints de eventos no backend
3. ⏳ Criar schema do database
4. ⏳ Desenvolver XML do Deriv Bot
5. ⏳ Adaptar dashboard para consumir eventos
6. ⏳ Testar integração end-to-end
7. ⏳ Deploy em produção

---

**Criado por:** Claude Sonnet 4.5
**Status:** 📝 Planejamento

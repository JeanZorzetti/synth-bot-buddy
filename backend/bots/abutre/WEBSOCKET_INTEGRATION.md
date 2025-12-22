# WebSocket Integration - Abutre Bot + Dashboard

## 📡 Visão Geral

O sistema Abutre possui integração real-time bidirecional entre backend (bot) e frontend (dashboard) usando Socket.IO.

- **Backend:** Python + Socket.IO (server)
- **Frontend:** Next.js + Socket.IO Client
- **Porta padrão:** 8000
- **Protocol:** WebSocket (Socket.IO)

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Components (Dashboard, Settings, etc.)              │   │
│  │       ↓                                              │   │
│  │  useWebSocket Hook (React)                          │   │
│  │       ↓                                              │   │
│  │  WebSocketClient (Socket.IO Client)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│                    Socket.IO Connection                     │
│                          ↓                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Backend (Python)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  WebSocketServer (Socket.IO Server)                 │   │
│  │       ↓                                              │   │
│  │  AbutreBot (Main Bot)                               │   │
│  │       ↓                                              │   │
│  │  Event Emitters (balance, trades, etc.)            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📤 Eventos Emitidos pelo Backend

### 1. `balance_update`
Emitido quando o saldo da conta muda.

```python
await ws_server.emit_balance_update(
    balance=2050.00,
    peak=2100.00,
    drawdown_pct=2.38
)
```

**Payload:**
```json
{
  "balance": 2050.00,
  "peak": 2100.00,
  "drawdown_pct": 2.38,
  "timestamp": "2025-01-21T14:30:00Z"
}
```

---

### 2. `new_candle`
Emitido quando uma nova vela M1 é fechada.

```python
await ws_server.emit_new_candle({
    'timestamp': '2025-01-21T14:30:00Z',
    'open': 1000.50,
    'high': 1001.20,
    'low': 1000.30,
    'close': 1000.80,
    'color': 'GREEN'
})
```

---

### 3. `trigger_detected`
Emitido quando um streak >= 8 é detectado (gatilho de entrada).

```python
await ws_server.emit_trigger_detected(
    streak_count=8,
    direction='RED'
)
```

**Payload:**
```json
{
  "streak_count": 8,
  "direction": "RED",
  "timestamp": "2025-01-21T14:30:00Z"
}
```

---

### 4. `trade_opened`
Emitido quando um trade é iniciado.

```python
await ws_server.emit_trade_opened({
    'trade_id': 'T12345',
    'entry_time': '2025-01-21T14:30:00Z',
    'direction': 'CALL',
    'level': 1,
    'stake': 1.00,
    'entry_streak_size': 8
})
```

---

### 5. `trade_closed`
Emitido quando um trade é finalizado.

```python
await ws_server.emit_trade_closed({
    'trade_id': 'T12345',
    'exit_time': '2025-01-21T14:31:00Z',
    'result': 'WIN',
    'profit': 0.95,
    'final_level': 1,
    'balance': 2000.95
})
```

---

### 6. `position_update`
Emitido quando o estado da posição Martingale muda.

```python
await ws_server.emit_position_update({
    'in_position': True,
    'direction': 'CALL',
    'entry_timestamp': '2025-01-21T14:30:00Z',
    'entry_streak_size': 8,
    'current_level': 2,
    'current_stake': 2.00,
    'total_loss': -1.00,
    'next_stake': 4.00
})
```

**Quando não há posição:**
```python
await ws_server.emit_position_update({
    'in_position': False,
    'direction': None,
    'entry_timestamp': None,
    'entry_streak_size': 0,
    'current_level': 0,
    'current_stake': 0,
    'total_loss': 0,
    'next_stake': 0
})
```

---

### 7. `market_data`
Emitido a cada vela fechada com dados do mercado.

```python
await ws_server.emit_market_data(
    symbol='V100',
    price=1000.80,
    streak_count=5,
    streak_direction='GREEN'
)
```

---

### 8. `risk_stats`
Emitido após cada trade fechado com estatísticas atualizadas.

```python
await ws_server.emit_risk_stats({
    'total_trades': 150,
    'wins': 145,
    'losses': 5,
    'win_rate': 96.67,
    'roi': 15.25
})
```

---

### 9. `bot_status`
Emitido quando o status do bot muda.

**Possíveis status:**
- `RUNNING` - Bot ativo e executando trades
- `PAUSED` - Bot pausado (paper trading mode)
- `STOPPED` - Bot desligado
- `STOPPING` - Bot em processo de shutdown

```python
await ws_server.emit_bot_status('RUNNING', 'Bot started successfully')
```

---

### 10. `system_alert`
Emitido para notificações e alertas do sistema.

**Levels:**
- `success` - Operação bem-sucedida
- `warning` - Aviso (trigger detected, etc.)
- `error` - Erro (risk violation, etc.)
- `info` - Informação geral

```python
await ws_server.emit_system_alert('warning', 'Max drawdown approaching 25%')
```

---

### 11. `connection_ack`
Emitido automaticamente quando um cliente conecta.

```json
{
  "status": "connected"
}
```

---

## 📥 Comandos Recebidos do Frontend

### 1. `bot_command`
Controlar o bot (start/pause/stop).

**Frontend envia:**
```typescript
socket.emit('bot_command', { command: 'start' })
socket.emit('bot_command', { command: 'pause' })
socket.emit('bot_command', { command: 'stop' })
```

**Backend responde:**
- Emite `bot_status` com novo status
- Emite `system_alert` confirmando ação

---

### 2. `update_settings`
Atualizar parâmetros da estratégia.

**Frontend envia:**
```typescript
socket.emit('update_settings', {
  delay_threshold: 10,
  max_level: 12,
  initial_stake: 2.00,
  multiplier: 2.0,
  max_drawdown: 25,
  auto_trading: true
})
```

**Backend responde:**
- Emite `settings_updated` com `{ success: true }`
- Emite `system_alert` confirmando atualização

---

## 🚀 Como Iniciar

### Backend

1. Instalar dependências:
```bash
cd backend/bots/abutre
pip install -r requirements.txt
```

2. Configurar .env:
```bash
cp .env.example .env
# Editar DERIV_API_TOKEN
```

3. Rodar bot:
```bash
python -m backend.bots.abutre.main --demo --paper-trading
```

O WebSocket server inicia automaticamente na porta 8000.

---

### Frontend

1. Instalar dependências:
```bash
cd frontend/abutre-dashboard
npm install
```

2. Configurar .env:
```bash
cp .env.example .env.local
# NEXT_PUBLIC_WS_URL=http://localhost:8000
```

3. Rodar dashboard:
```bash
npm run dev
```

Dashboard disponível em: http://localhost:3000

---

## 🔧 Troubleshooting

### Erro: "Connection refused" no frontend

**Causa:** Backend não está rodando ou porta diferente.

**Solução:**
1. Verificar se backend está ativo: `netstat -an | findstr 8000`
2. Verificar NEXT_PUBLIC_WS_URL no frontend
3. Verificar firewall/antivírus bloqueando porta 8000

---

### Erro: "Module socketio not found"

**Causa:** Dependências do backend não instaladas.

**Solução:**
```bash
pip install python-socketio==5.10.0 python-engineio==4.8.0
```

---

### Frontend não recebe eventos

**Causa:** WebSocket não conectou corretamente.

**Solução:**
1. Abrir DevTools → Network → WS
2. Verificar se há conexão Socket.IO ativa
3. Ver logs do backend para confirmar conexão: `[WS] Client connected: <sid>`

---

## 📊 Fluxo de Dados Completo

### Cenário: Trade WIN

1. **Nova vela fecha** → Backend emite `new_candle`
2. **Streak detectado (8+)** → Backend emite `trigger_detected` + `system_alert`
3. **Trade aberto** → Backend emite `trade_opened` + `position_update`
4. **Vela seguinte fecha** → Backend emite `new_candle` + `market_data`
5. **Trade fecha (WIN)** → Backend emite:
   - `trade_closed` (result: WIN, profit: +$0.95)
   - `position_update` (in_position: false)
   - `balance_update` (balance aumenta)
   - `risk_stats` (win_rate atualizado)
   - `system_alert` (success, "Trade closed: WIN")

Frontend atualiza:
- MetricsCard (Balance, ROI, Win Rate)
- EquityCurve (novo ponto no gráfico)
- CurrentPosition (volta para "Waiting for Signal")
- TradesTable (adiciona novo trade)
- Toast notification (sucesso)

---

### Cenário: Trade LOSS (Martingale Level Up)

1. **Trade aberto Level 1** → `trade_opened`, `position_update` (level: 1, stake: $1.00)
2. **Vela fecha contra** → `new_candle`, `market_data`
3. **Level UP para 2** → Backend emite:
   - `position_update` (level: 2, stake: $2.00, total_loss: -$1.00)
   - `system_alert` (warning, "Martingale Level 2")
4. **Vela fecha a favor** → `new_candle`
5. **Trade fecha WIN** → `trade_closed` (profit recupera loss), `position_update` (in_position: false)

Frontend mostra:
- Progress bar do level (2/10)
- Warning quando level >= 7
- Total loss acumulado
- Next stake calculado

---

## 🔒 Segurança

### CORS
Atualmente configurado para aceitar todas as origens (`cors_allowed_origins='*'`).

**Em produção, restringir para:**
```python
self.sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins=['https://seu-dominio.com'],
    logger=False,
    engineio_logger=False
)
```

### Autenticação
Atualmente não há autenticação no WebSocket.

**Para produção, adicionar:**
1. JWT token validation no evento `connect`
2. Verificar token antes de emitir dados sensíveis
3. Rate limiting para comandos

---

## 📝 Notas

- Todos os timestamps são ISO 8601 format (UTC)
- Valores monetários em float (2 decimais)
- Percentagens em float (0-100)
- Direções: 'CALL' ou 'PUT'
- Cores de vela: 'GREEN', 'RED', 'DOJI'
- O servidor WebSocket inicia automaticamente com o bot
- Reconexão automática no frontend (exponential backoff)

---

**Última atualização:** 2025-01-21
**Autor:** Claude Code (modo AUTO-PILOT)

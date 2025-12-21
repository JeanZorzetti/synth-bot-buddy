# ROADMAP - SISTEMA ABUTRE (Delayed Martingale)

**Objetivo:** Implementar sistema completo de trading automatizado baseado na estratégia "Abutre" validada em backtest (+40.25% ROI em 180 dias)

**Status:** 🟢 Em Desenvolvimento

---

## 📊 RESULTADOS DO BACKTEST (Validação)

```
Dataset: V100 M1 (180 dias, 258,086 candles)
Banca: $2,000 → $2,805.10 (+40.25% ROI)
Win Rate: 100% (1,018 trades, 0 busts)
Max Drawdown: 24.81%
Lucro/Trade: $0.79
```

**Parâmetros Validados:**
- Delay Threshold: 8 velas consecutivas
- Max Level: 10 (capacidade $1,023)
- Initial Stake: $1.00
- Multiplier: 2.0x

---

## 🎯 FASES DO PROJETO

### ✅ FASE 0: Pesquisa e Validação (CONCLUÍDA)
- [x] Análise de risco Martingale tradicional
- [x] Descoberta da "Sequência da Morte" (18 velas)
- [x] Implementação do Delayed Martingale
- [x] Backtest com 180 dias de dados
- [x] Validação matemática (+40.25% ROI, 100% win rate)

---

### 🔄 FASE 1: Backend - Core Engine (EM ANDAMENTO)

#### 1.1. Infraestrutura Base
- [ ] Criar estrutura de diretórios do bot
  - `backend/bots/abutre/`
  - `backend/bots/abutre/core/`
  - `backend/bots/abutre/strategies/`
  - `backend/bots/abutre/utils/`
  - `backend/bots/abutre/tests/`

- [ ] Configuração e ambiente
  - [ ] `config.py` - Parâmetros do Abutre (Delay, Max Level, etc.)
  - [ ] `logger.py` - Sistema de logging estruturado
  - [ ] `.env.example` - Template de variáveis de ambiente
  - [ ] `requirements.txt` - Dependências do bot

#### 1.2. Conexão com Deriv API
- [ ] `deriv_api_client.py` - Cliente WebSocket Deriv
  - [ ] Autenticação (API token)
  - [ ] Subscribe to tick stream (V100 M1)
  - [ ] Subscribe to balance updates
  - [ ] Reconnect automático em caso de queda
  - [ ] Rate limiting (evitar ban)

- [ ] `market_data_handler.py` - Processamento de dados
  - [ ] Receber ticks em tempo real
  - [ ] Construir candles M1 a partir dos ticks
  - [ ] Detectar cor da vela (verde/vermelha)
  - [ ] Calcular streak count (sequências)
  - [ ] Buffer de histórico (últimas 100 velas)

#### 1.3. Estratégia Abutre
- [ ] `abutre_strategy.py` - Lógica da estratégia
  - [ ] `detect_trigger()` - Detecta Delay >= 8 velas
  - [ ] `calculate_position_size()` - Martingale sizing
  - [ ] `get_bet_direction()` - Contra-tendência
  - [ ] `check_max_level()` - Safety limit (Nível 10)
  - [ ] `calculate_profit_loss()` - P&L por trade

- [ ] `risk_manager.py` - Gestão de risco
  - [ ] Verificar saldo disponível
  - [ ] Stop Loss automático (Nível 10)
  - [ ] Max Drawdown killer (25%)
  - [ ] Emergency shutdown (conexão perdida, etc.)

#### 1.4. Execução de Ordens
- [ ] `order_executor.py` - Interface com Deriv
  - [ ] `place_order()` - Executar ordem BUY/SELL
  - [ ] `close_position()` - Fechar posição manualmente
  - [ ] Retry logic (falha de execução)
  - [ ] Slippage monitoring

- [ ] `position_tracker.py` - Tracking de posições
  - [ ] Estado atual (in_position, current_level, etc.)
  - [ ] Histórico de trades (DB local)
  - [ ] Cálculo de métricas (win rate, ROI, etc.)

#### 1.5. Persistência de Dados
- [ ] `database.py` - SQLite local
  - [ ] Tabela: `trades` (histórico completo)
  - [ ] Tabela: `candles` (buffer M1)
  - [ ] Tabela: `balance_history` (equity curve)
  - [ ] Tabela: `system_events` (logs críticos)

- [ ] `models.py` - Schemas SQLAlchemy
  - [ ] Trade model
  - [ ] Candle model
  - [ ] Balance model
  - [ ] Event model

#### 1.6. Testes Unitários
- [ ] `test_abutre_strategy.py`
  - [ ] Test trigger detection
  - [ ] Test position sizing
  - [ ] Test profit calculation
  - [ ] Test edge cases (Nível 10+)

- [ ] `test_risk_manager.py`
  - [ ] Test max level enforcement
  - [ ] Test drawdown killer
  - [ ] Test balance checks

---

### 🎨 FASE 2: Frontend - Dashboard (PRÓXIMA)

#### 2.1. Arquitetura Frontend
- [ ] Escolher stack: **Next.js 14 + TypeScript + Tailwind**
  - [ ] Setup projeto Next.js
  - [ ] Configurar TypeScript
  - [ ] Instalar Tailwind CSS
  - [ ] Configurar shadcn/ui (componentes)

#### 2.2. Layout Principal
- [ ] `app/layout.tsx` - Layout base
  - [ ] Header (logo, user info, saldo)
  - [ ] Sidebar (navegação)
  - [ ] Main content area
  - [ ] Footer (status, versão)

- [ ] `app/page.tsx` - Dashboard principal
  - [ ] Grid responsivo (4 colunas em desktop)
  - [ ] Cards de métricas principais
  - [ ] Gráfico de equity curve
  - [ ] Lista de trades recentes

#### 2.3. Componentes de Métricas
- [ ] `components/MetricsCard.tsx`
  - [ ] Saldo atual
  - [ ] ROI (%)
  - [ ] Total trades (hoje/total)
  - [ ] Win rate
  - [ ] Max drawdown

- [ ] `components/EquityCurve.tsx`
  - [ ] Chart.js ou Recharts
  - [ ] Linha de equity
  - [ ] Linha de baseline ($2,000)
  - [ ] Tooltips com detalhes

#### 2.4. Componentes de Trading
- [ ] `components/CurrentPosition.tsx`
  - [ ] Status: "Waiting" / "In Position"
  - [ ] Se in_position:
    - Current level
    - Current stake
    - Unrealized P&L
    - Entry streak size
  - [ ] Botão: "Close Position" (manual override)

- [ ] `components/MarketMonitor.tsx`
  - [ ] Preço atual V100
  - [ ] Cor da última vela
  - [ ] Streak count atual (ex: "🟢 5 velas verdes")
  - [ ] Countdown até gatilho (ex: "Faltam 3 para Delay 8")

- [ ] `components/TradesTable.tsx`
  - [ ] Tabela com últimos 50 trades
  - [ ] Colunas: ID, Entry Time, Exit Time, Direction, Level, P&L, Result
  - [ ] Filtros: Today, This Week, All Time
  - [ ] Export CSV

#### 2.5. Página de Configuração
- [ ] `app/settings/page.tsx`
  - [ ] Form para ajustar parâmetros:
    - Delay Threshold (6-12)
    - Max Level (8-12)
    - Initial Stake ($0.50-$5.00)
    - Auto-trading ON/OFF
  - [ ] Botão: "Save & Restart Bot"
  - [ ] Warning: "Changing parameters requires backtest validation"

#### 2.6. Sistema de Alertas
- [ ] `components/AlertSystem.tsx`
  - [ ] Toast notifications (shadcn/ui)
  - [ ] Tipos de alerta:
    - 🟢 "Gatilho detectado! Entrando SELL..."
    - 🟡 "Subindo para Nível 3..."
    - ✅ "WIN! +$7.60"
    - 🔴 "Max Drawdown atingido! Bot pausado"
  - [ ] Som opcional (beep no trigger)

#### 2.7. WebSocket Real-Time
- [ ] `lib/websocket-client.ts`
  - [ ] Conexão com backend via Socket.IO
  - [ ] Eventos:
    - `balance_update`
    - `new_candle`
    - `trigger_detected`
    - `trade_opened`
    - `trade_closed`
    - `system_alert`
  - [ ] Auto-reconnect

---

### 🧪 FASE 3: Validação (CRÍTICA)

#### 3.1. Forward Test (30 dias)
- [ ] Rodar bot em **DEMO account**
  - [ ] Configurar API token de demo
  - [ ] Iniciar bot com banca virtual $2,000
  - [ ] Monitorar por 30 dias

- [ ] Métricas de validação:
  - [ ] ROI esperado: +6-7% (40%/6 meses)
  - [ ] Win rate esperado: > 95%
  - [ ] Max drawdown: < 30%
  - [ ] Total trades: ~170 (1018 / 6 meses)

- [ ] **Critério de Aprovação:**
  - ✅ Se ROI > 5% E Win Rate > 90% → Avançar Fase 3.2
  - ❌ Se ROI < 0% OU Bust → Aumentar Delay para 10 e repetir

#### 3.2. Paper Trading (60 dias)
- [ ] Monitoramento sem execução
  - [ ] Bot detecta gatilhos mas NÃO executa
  - [ ] Registra em planilha:
    - Timestamp do gatilho
    - Direção (BUY/SELL)
    - Resultado simulado
    - Spread real observado

- [ ] Análise de divergências:
  - [ ] Sinais gerados = backtest?
  - [ ] Spread real vs assumido (5%)
  - [ ] Slippage em níveis altos

#### 3.3. Live Trading Micro (30 dias)
- [ ] **CONTA REAL** com capital reduzido
  - [ ] Capital: $200 (10% da banca final)
  - [ ] Stake inicial: $0.10 (escala 1:10)
  - [ ] Max Level: 10 (mesma proporção)

- [ ] Resultado esperado:
  - [ ] $200 → $240 (+20% em 1 mês)
  - [ ] Se alcançar: Escalar para $2,000
  - [ ] Se bust: Perda máxima $200 (aceitável)

---

### 🚀 FASE 4: Deploy em Produção

#### 4.1. Infraestrutura
- [ ] Escolher hosting: **VPS** (DigitalOcean, AWS EC2, Vultr)
  - [ ] Setup Ubuntu 22.04
  - [ ] Instalar Python 3.11+
  - [ ] Instalar Node.js 20+
  - [ ] Configurar PM2 (process manager)

- [ ] Banco de dados:
  - [ ] PostgreSQL (migrar de SQLite)
  - [ ] Backup automático (daily)

- [ ] Segurança:
  - [ ] SSL/TLS (Let's Encrypt)
  - [ ] API keys em .env (não commitar)
  - [ ] Firewall (UFW)
  - [ ] Fail2ban (proteção SSH)

#### 4.2. CI/CD
- [ ] GitHub Actions
  - [ ] Workflow: Test → Build → Deploy
  - [ ] Auto-deploy em push para `main`
  - [ ] Rollback automático se testes falharem

#### 4.3. Monitoramento
- [ ] Logs centralizados (Winston + CloudWatch)
- [ ] Métricas (Prometheus + Grafana)
- [ ] Alertas (Email/Telegram em eventos críticos):
  - Max Drawdown > 25%
  - Bot offline > 5 minutos
  - Saldo < $1,000

---

### 📈 FASE 5: Otimização e Escala

#### 5.1. Análise de Sensibilidade
- [ ] Testar variações de Delay (6, 7, 9, 10, 11, 12)
- [ ] Testar variações de Max Level (8, 9, 11, 12)
- [ ] Identificar configuração ótima para diferentes períodos

#### 5.2. Multi-Asset
- [ ] Expandir para V75 (mesmo algoritmo)
- [ ] Expandir para V50 (menor volatilidade)
- [ ] Diversificação de risco

#### 5.3. Machine Learning Opcional
- [ ] LSTM para prever DURAÇÃO de streaks
  - Se LSTM prever "streak vai durar 12+ velas"
  - Delay pode ser reduzido para 6-7 (mais agressivo)
- [ ] Feature: Horário do dia, dia da semana
  - Streaks longas ocorrem mais em horários específicos?

---

## 🛠️ STACK TECNOLÓGICO

### Backend
```python
# Core
Python 3.11+
asyncio (WebSocket handling)

# API & Data
python-deriv-api    # Deriv WebSocket
websockets          # WS client
aiohttp             # Async HTTP

# Database
SQLAlchemy          # ORM
alembic             # Migrations
PostgreSQL / SQLite # Database

# Utilities
pydantic            # Data validation
python-dotenv       # Environment vars
loguru              # Logging
pytest              # Testing
```

### Frontend
```javascript
// Framework
Next.js 14          // React framework
TypeScript          // Type safety

// UI
Tailwind CSS        // Styling
shadcn/ui           // Components
Radix UI            // Primitives

// Charts & Viz
Recharts            // Charts
framer-motion       // Animations

// State & Data
Zustand             // State management
Socket.IO Client    // WebSocket
TanStack Query      // Data fetching
```

### DevOps
```bash
# Deployment
PM2                 # Process manager
Nginx               # Reverse proxy
Docker (optional)   # Containerization

# Monitoring
Winston             # Logging
Prometheus          # Metrics
Grafana             # Visualization

# CI/CD
GitHub Actions      # Automation
```

---

## 📋 CHECKLIST DE SEGURANÇA

- [ ] API keys em variáveis de ambiente (nunca hardcoded)
- [ ] .env adicionado ao .gitignore
- [ ] Rate limiting na API Deriv (evitar ban)
- [ ] Validação de saldo antes de cada ordem
- [ ] Max Drawdown killer (emergency stop)
- [ ] Logs de todas as operações críticas
- [ ] Backup diário do banco de dados
- [ ] Alertas de sistema (email/telegram)
- [ ] SSL/TLS em produção
- [ ] Autenticação no frontend (opcional: OAuth)

---

## 📊 CRITÉRIOS DE SUCESSO

### Forward Test (30 dias DEMO)
- ✅ ROI > 5%
- ✅ Win Rate > 90%
- ✅ Max DD < 30%
- ✅ 0 busts

### Paper Trading (60 dias)
- ✅ Sinais replicam backtest
- ✅ Spread real < 7%
- ✅ Slippage aceitável

### Live Micro (30 dias REAL)
- ✅ $200 → $240 (+20%)
- ✅ 0 busts

### Produção (6 meses REAL)
- ✅ $2,000 → $2,800+ (+40%)
- ✅ Win Rate > 95%
- ✅ Max DD < 30%
- ✅ Sistema estável (uptime > 99%)

---

## 🎯 MILESTONES

| Milestone | Data Alvo | Status |
|-----------|-----------|--------|
| M1: Backend Core Completo | Semana 1 | 🔄 Em andamento |
| M2: Frontend Dashboard | Semana 2 | ⏳ Pendente |
| M3: Forward Test (Demo) | Semana 3-6 | ⏳ Pendente |
| M4: Paper Trading | Semana 7-14 | ⏳ Pendente |
| M5: Live Micro | Semana 15-18 | ⏳ Pendente |
| M6: Deploy Produção | Semana 19 | ⏳ Pendente |

---

## 📝 NOTAS IMPORTANTES

### Riscos Conhecidos
1. **Cisne Negro:** Sequência > 18 velas quebraria o sistema
   - Mitigação: Aumentar Delay para 10 (margem +2)

2. **Spread Real:** Simulação assumiu 5%
   - Validar spread real da corretora em paper trading

3. **Slippage:** Níveis altos ($512) podem ter slippage
   - Testar em horários de alta liquidez

4. **Overfitting:** Backtest pode não se repetir
   - Forward test é CRÍTICO para validação

### Premissas
- V100 continua seguindo distribuição de streaks observada
- Deriv API permanece estável
- Spread/comissões não mudam drasticamente
- Lei dos Grandes Números se aplica (reversão à média)

---

**🚀 INÍCIO DA IMPLEMENTAÇÃO: AGORA**

**Próximo passo:** Implementar FASE 1.1 - Infraestrutura Base

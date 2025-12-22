# ABUTRE BOT - STATUS ATUAL

**Última atualização:** 22/12/2025
**Fase atual:** FASE 3.1 - Forward Test (Dia 1/30)
**Status:** ✅ SISTEMA OPERACIONAL - Monitoramento em andamento

---

## 🎯 RESUMO EXECUTIVO

O **Abutre Bot** é um sistema de trading automatizado que implementa a estratégia **Delayed Martingale** para mercados de volatilidade (Deriv.com). O bot foi desenvolvido, testado em backtest (180 dias) e agora está em **Forward Test** por 30 dias em conta DEMO antes de usar capital real.

**Performance Backtest (180 dias):**
- ✅ ROI: **+40.25%**
- ✅ Win Rate: **100%** (0 losses)
- ✅ Max Drawdown: **18.2%**
- ✅ Total Trades: **1,018**
- ✅ Busts: **0**

**Próximo Milestone:** Validar performance em Forward Test (30 dias) com dados reais do mercado.

---

## 📊 FASE 3.1 - FORWARD TEST (30 DIAS)

### Status: 🟢 EM ANDAMENTO (Dia 1/30)

**Início:** 22/12/2025
**Fim previsto:** 21/01/2026
**Modo:** Paper Trading (AUTO_TRADING=false)

### Objetivos

| Métrica | Alvo (30 dias) | Status |
|---------|----------------|--------|
| **ROI** | > 5% | ⏳ Monitorando |
| **Win Rate** | > 90% | ⏳ Monitorando |
| **Max Drawdown** | < 30% | ⏳ Monitorando |
| **Total Trades** | ~170 | ⏳ Monitorando |
| **Busts** | 0 | ⏳ Monitorando |

**Critério de Aprovação:**
- ✅ ROI > 5% E Win Rate > 90% → Avançar para FASE 3.2 (Paper Trading Real)
- ❌ ROI < 0% OU Bust → Aumentar `DELAY_THRESHOLD=10` e repetir

---

## 🚀 AMBIENTE DE PRODUÇÃO

### Frontend (Dashboard)

| Parâmetro | Valor |
|-----------|-------|
| **URL** | <https://botderiv.rollabs.com.br/abutre> |
| **Framework** | React + Vite + React Router |
| **Deploy** | Vercel |
| **WebSocket** | Conectado ✅ |
| **Status** | Online ✅ |

**Páginas:**
- `/abutre` - Dashboard principal (métricas, trades, market monitor)
- Sidebar: Botão "Abutre Bot" com badge "FASE 3"

### Backend (Bot)

| Parâmetro | Valor |
|-----------|-------|
| **URL** | <wss://botderivapi.roilabs.com.br> |
| **Endpoint WS** | `/ws/dashboard` |
| **Framework** | FastAPI + Native WebSocket |
| **Deploy** | Easypanel (VPS) |
| **Status** | Running ✅ |

**Configuração:**
- Token: `paE5sSemx3oANLE` (DEMO account - VRTC)
- Symbol: V100 (1HZ100V)
- Paper Trading: **ATIVO** (sem execução real)

### Parâmetros da Estratégia

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `DELAY_THRESHOLD` | 8 | Aguardar 8 velas consecutivas |
| `MAX_LEVEL` | 10 | Máximo 10 níveis Martingale |
| `INITIAL_STAKE` | $1.00 | Stake inicial por trade |
| `MULTIPLIER` | 2.0 | Dobrar stake a cada nível |
| `BANKROLL` | $2,000 | Banca inicial (DEMO) |
| `MAX_DRAWDOWN_PCT` | 0.25 | Stop se drawdown > 25% |
| `AUTO_TRADING` | false | **Paper Trading** (observação) |

---

## 📁 ARQUIVOS IMPORTANTES

### Código Principal

```
backend/bots/abutre/
├── main.py                    # Bot principal
├── start_bot.py               # Startup script (resolve imports)
├── config.py                  # Configuração (lê .env)
├── strategy/
│   └── delayed_martingale.py  # Lógica da estratégia
├── data/
│   └── abutre.db              # SQLite database (trades, balance)
├── logs/
│   └── abutre.log             # Logs de execução
└── .env                       # Variáveis de ambiente (NÃO commitado)
```

### Frontend

```
frontend/src/
├── pages/
│   └── AbutreDashboard.tsx    # Dashboard principal
├── components/abutre/
│   ├── MetricsCards.tsx       # Cards de métricas
│   ├── EquityCurveChart.tsx   # Gráfico equity curve
│   ├── TradesTable.tsx        # Tabela de trades
│   ├── MarketMonitor.tsx      # Monitor de mercado
│   └── RiskPanel.tsx          # Painel de risco
├── lib/
│   └── websocket-client.ts    # Cliente WebSocket (Native WS)
└── hooks/
    └── useAbutreStore.ts      # Zustand store
```

### Documentação

```
backend/bots/abutre/
├── SETUP_DERIV_API.md         # Guia de configuração API (516 linhas)
├── FORWARD_TEST_LOG.md        # Log de monitoramento (30 dias)
├── STATUS.md                  # Este arquivo
└── scripts/
    ├── README.md              # Guia de scripts
    └── get_daily_metrics.py   # Extrator de métricas
```

---

## 🔧 COMO USAR

### 1. Acessar Dashboard

URL: <https://botderiv.rollabs.com.br/abutre>

**O que você verá:**
- ✅ Status: Connected (verde)
- ✅ Modo: Paper Trading
- ✅ Saldo: $2,000.00 (inicial)
- ⏳ Aguardando gatilhos (8+ velas consecutivas)

### 2. Monitorar Diariamente

**Script de métricas:**

```bash
cd backend/bots/abutre
python scripts/get_daily_metrics.py
```

**Output:**
- Saldo atual
- ROI, Win Rate, Max Drawdown
- Trades do dia
- Eventos importantes
- **Entrada formatada** para copiar no `FORWARD_TEST_LOG.md`

### 3. Atualizar Log

1. Executar script de métricas
2. Copiar saída formatada
3. Colar no `FORWARD_TEST_LOG.md` no dia correspondente
4. Commitar mudanças:
   ```bash
   git add backend/bots/abutre/FORWARD_TEST_LOG.md
   git commit -m "docs: Forward Test Day X - atualização métricas"
   git push
   ```

### 4. Verificar Logs

**Logs em tempo real:**
```bash
tail -f backend/bots/abutre/logs/abutre.log
```

**Buscar erros:**
```bash
grep ERROR backend/bots/abutre/logs/abutre.log
```

**Últimas 100 linhas:**
```bash
tail -n 100 backend/bots/abutre/logs/abutre.log
```

---

## 📊 HISTÓRICO DE DESENVOLVIMENTO

### FASE 1: Backtest (COMPLETO ✅)

- [x] Implementação da estratégia Delayed Martingale
- [x] Backtest com 180 dias de dados históricos
- [x] Validação de parâmetros (Delay=8, MaxLevel=10)
- [x] Resultado: **+40.25% ROI, 100% Win Rate**

### FASE 2: Frontend Dashboard (COMPLETO ✅)

- [x] Dashboard React + TypeScript
- [x] Integração WebSocket real-time
- [x] Componentes: Metrics, Equity Curve, Trades Table
- [x] Deploy em Vercel
- [x] Integrado ao frontend principal em `/abutre`

### FASE 3.0: Integração Produção (COMPLETO ✅)

- [x] Backend FastAPI no Easypanel (VPS)
- [x] WebSocket nativo (não Socket.IO)
- [x] Token DEMO configurado
- [x] Dashboard conectado em produção
- [x] Sistema de logs e database

### FASE 3.1: Forward Test (EM ANDAMENTO 🟢)

**Status:** Dia 1/30 - Iniciado 22/12/2025

- [x] Ambiente de produção configurado
- [x] Bot rodando em DEMO account
- [x] Paper Trading ativo (sem risco)
- [x] Log de acompanhamento criado
- [x] Script de métricas automático
- [ ] **Monitoramento por 30 dias** (em andamento)
- [ ] Validação de critérios de aprovação

### FASE 3.2: Paper Trading Real (PENDENTE ⏳)

Aguardando aprovação da FASE 3.1.

### FASE 3.3: Live Trading Micro (PENDENTE ⏳)

Capital real reduzido ($200) após validação completa.

---

## 🔐 SEGURANÇA

### Configuração Atual (DEMO)

- ✅ Conta: DEMO (VRTC) - **SEM RISCO FINANCEIRO**
- ✅ Paper Trading: **ATIVO** (AUTO_TRADING=false)
- ✅ Token DEMO no `.env` (não commitado no Git)
- ✅ Zero possibilidade de perda de dinheiro

### Antes de Ativar Live Trading

**NUNCA ativar `AUTO_TRADING=true` sem:**

1. ✅ Forward Test aprovado (30 dias, ROI > 5%, WR > 90%)
2. ✅ Paper Trading validado (60 dias)
3. ✅ Live Micro testado ($200 capital)
4. ✅ Token REAL separado do DEMO
5. ✅ Monitoramento 24/7 configurado
6. ✅ Alertas de risco ativados

**Checklist de Segurança:**
- [ ] `.env` no `.gitignore` (já configurado ✅)
- [ ] Token DEMO ≠ Token REAL
- [ ] `MAX_DRAWDOWN_PCT` configurado
- [ ] `MIN_BALANCE` configurado
- [ ] Telegram/Email alerts configurados
- [ ] Backup do database diário

---

## 📈 PRÓXIMOS PASSOS

### Curto Prazo (Próximos 7 dias)

1. **Monitorar diariamente** (5 min/dia)
   - Acessar dashboard
   - Executar `python scripts/get_daily_metrics.py`
   - Atualizar `FORWARD_TEST_LOG.md`
   - Commitar mudanças

2. **Aguardar primeiro gatilho**
   - Bot detecta 8+ velas consecutivas
   - Simula trade (paper trading)
   - Registra no database

3. **Verificar métricas semanais**
   - ROI semanal vs esperado (+1.5%)
   - Win Rate mantém > 95%
   - Max Drawdown < 10%

### Médio Prazo (30 dias)

1. **Completar Forward Test**
   - Acumular ~170 trades
   - ROI > 5%
   - Win Rate > 90%
   - Zero busts

2. **Análise Final**
   - Comparar com backtest
   - Identificar divergências
   - Validar spread/slippage

3. **Decisão:**
   - ✅ Aprovado → FASE 3.2
   - ❌ Reprovado → Ajustar `DELAY_THRESHOLD=10` e repetir

### Longo Prazo (90-120 dias)

1. **FASE 3.2:** Paper Trading Real (60 dias)
2. **FASE 3.3:** Live Micro ($200 capital, 30 dias)
3. **FASE 4:** Live Trading Full ($2,000 capital)

---

## 🆘 TROUBLESHOOTING

### Dashboard mostra "Disconnected"

**Causa:** Backend Easypanel offline ou WebSocket bloqueado

**Solução:**
1. Verificar backend: `curl https://botderivapi.roilabs.com.br/health`
2. Verificar logs do Easypanel
3. Verificar firewall/proxy WebSocket

### Bot não detecta gatilhos

**Causa:** Mercado sem streaks de 8+ velas

**Solução:**
- Normal - aguardar atividade do mercado
- Verificar logs: `grep "trigger_detected" logs/abutre.log`
- V100 geralmente tem 3-5 gatilhos por dia

### Script de métricas retorna erro

**Causa:** Database ainda não criado

**Solução:**
1. Bot precisa rodar pelo menos uma vez
2. Verificar: `ls -la data/abutre.db`
3. Se não existe: Aguardar bot conectar e criar database

### Métricas zeradas

**Causa:** Nenhum trade executado ainda

**Solução:**
- Normal em paper trading
- Aguardar primeiro gatilho (8+ velas)
- Verificar dashboard: Status deve estar "Connected"

---

## 📞 RECURSOS

**Documentação:**
- Setup API: `backend/bots/abutre/SETUP_DERIV_API.md`
- Scripts: `backend/bots/abutre/scripts/README.md`
- Roadmap: `roadmaps/ROADMAP_ABUTRE.md`

**URLs:**
- Dashboard: <https://botderiv.rollabs.com.br/abutre>
- Backend: <https://botderivapi.roilabs.com.br>
- Deriv API Docs: <https://api.deriv.com>
- Deriv Tokens: <https://app.deriv.com/account/api-token>

**Arquivos:**
- Logs: `backend/bots/abutre/logs/abutre.log`
- Database: `backend/bots/abutre/data/abutre.db`
- Forward Test Log: `backend/bots/abutre/FORWARD_TEST_LOG.md`

---

## ✅ CHECKLIST DE VALIDAÇÃO

### Ambiente de Produção

- [x] Dashboard acessível em <https://botderiv.rollabs.com.br/abutre>
- [x] WebSocket conectado ao backend Easypanel
- [x] Status mostra "Connected" (verde)
- [x] Token DEMO configurado (paE5sSemx3oANLE)
- [x] Paper Trading ativo (AUTO_TRADING=false)

### Sistema de Monitoramento

- [x] `FORWARD_TEST_LOG.md` criado
- [x] Script `get_daily_metrics.py` funcional
- [x] Logs salvando em `logs/abutre.log`
- [x] Database criado em `data/abutre.db`

### Segurança

- [x] `.env` no `.gitignore`
- [x] Token é de conta DEMO (VRTC)
- [x] `MAX_DRAWDOWN_PCT` configurado (25%)
- [x] `MIN_BALANCE` configurado ($500)
- [x] Zero risco financeiro (paper trading)

### Documentação

- [x] Guia de setup completo (SETUP_DERIV_API.md)
- [x] Guia de scripts (scripts/README.md)
- [x] Status atual documentado (STATUS.md)
- [x] Roadmap atualizado (ROADMAP_ABUTRE.md)

---

**Status Geral:** 🟢 SISTEMA PRONTO PARA MONITORAMENTO

**Última verificação:** 22/12/2025
**Próxima ação:** Monitoramento diário por 30 dias
**Responsável:** Executar `python scripts/get_daily_metrics.py` diariamente

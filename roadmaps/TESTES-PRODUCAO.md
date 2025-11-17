# 🧪 Guia de Testes em Produção - Deriv Bot Inteligente

## 📋 Visão Geral

Este documento descreve **como testar cada funcionalidade** do bot em produção e os **resultados esperados** para validação.

---

## **FASE 4: Gestão de Risco** 🛡️

### 🔍 Testes

#### 1. Position Sizing Kelly Criterion

**Endpoint:**
```bash
POST /api/risk/position-size
```

**Request:**
```json
{
  "capital": 10000,
  "win_rate": 0.65,
  "avg_win": 50,
  "avg_loss": 25,
  "entry_price": 12.50,
  "stop_loss": 12.30
}
```

**Resultado Esperado:**
```json
{
  "recommended_size": 234.5,
  "risk_amount": 46.90,
  "risk_percentage": 0.47,
  "kelly_fraction": 0.25,
  "max_loss_if_stopped": 46.90
}
```

#### 2. Stop Loss Dinâmico (ATR)

**Endpoint:**
```bash
GET /api/risk/stop-loss/atr/{symbol}?position=long
```

**Resultado Esperado:**
```json
{
  "symbol": "1HZ75V",
  "current_price": 12.50,
  "atr_14": 0.15,
  "stop_loss": 12.20,
  "distance_pct": 2.4,
  "type": "atr_based"
}
```

#### 3. Trailing Stop Update

**Endpoint:**
```bash
POST /api/risk/trailing-stop/update
```

**Request:**
```json
{
  "position_id": "abc123",
  "current_price": 12.75,
  "trailing_percent": 2.0
}
```

**Resultado Esperado:**
```json
{
  "position_id": "abc123",
  "previous_stop": 12.20,
  "new_stop": 12.50,
  "moved_up": true,
  "locked_profit": 0.30
}
```

#### 4. Validação de Risk/Reward

**Endpoint:**
```bash
POST /api/risk/validate-trade
```

**Request:**
```json
{
  "entry": 12.50,
  "stop_loss": 12.30,
  "take_profit": 12.90,
  "min_rr": 2.0
}
```

**Resultado Esperado:**
```json
{
  "valid": true,
  "risk": 0.20,
  "reward": 0.40,
  "rr_ratio": 2.0,
  "recommendation": "Trade aprovado - R:R adequado"
}
```

#### 5. Circuit Breaker

**Endpoint:**
```bash
GET /api/risk/circuit-breaker/status
```

**Resultado Esperado (Normal):**
```json
{
  "status": "active",
  "daily_loss": 250.00,
  "daily_loss_limit": 500.00,
  "utilization": 50.0,
  "trades_today": 15,
  "max_trades_daily": 30,
  "can_trade": true
}
```

**Resultado Esperado (Bloqueado):**
```json
{
  "status": "triggered",
  "reason": "Daily loss limit reached",
  "daily_loss": 520.00,
  "daily_loss_limit": 500.00,
  "can_trade": false,
  "resume_at": "2025-11-08T00:00:00Z"
}
```

### ✅ Critérios de Aceitação - Fase 4

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Position sizing calcula corretamente | Segue Kelly Criterion com fração 0.25 | ⏳ |
| ATR stop loss adapta à volatilidade | Distância proporcional ao ATR | ⏳ |
| Trailing stop move apenas para cima | Nunca diminui em posições long | ⏳ |
| R:R validation rejeita trades ruins | Apenas trades com R:R > 2.0 passam | ⏳ |
| Circuit breaker para trading | Após 5% de perda diária | ⏳ |
| Max 3 trades simultâneos | Rejeita 4º trade | ⏳ |

---

## **FASE 5: Order Flow Analysis** 💹

### 🔍 Testes

#### 1. Order Book Depth Analysis

**Endpoint:**
```bash
GET /api/orderflow/depth/{symbol}
```

**Resultado Esperado:**
```json
{
  "symbol": "1HZ75V",
  "timestamp": "2025-11-07T20:15:00Z",
  "bid_volume": 15420,
  "ask_volume": 12380,
  "bid_pressure": 55.5,
  "imbalance": "bullish",
  "bid_walls": [
    {"price": 12.20, "size": 5000, "strength": "strong"}
  ],
  "ask_walls": [
    {"price": 12.60, "size": 3500, "strength": "moderate"}
  ]
}
```

#### 2. Aggressive Orders Detection

**Endpoint:**
```bash
GET /api/orderflow/aggressive-orders/{symbol}
```

**Resultado Esperado:**
```json
{
  "symbol": "1HZ75V",
  "window": "5min",
  "aggressive_buys": 25,
  "aggressive_sells": 12,
  "delta": 13,
  "sentiment": "bullish",
  "large_orders": [
    {"side": "buy", "size": 1500, "price": 12.45, "timestamp": "2025-11-07T20:14:30Z"}
  ]
}
```

#### 3. Volume Profile (POC/VAH/VAL)

**Endpoint:**
```bash
GET /api/orderflow/volume-profile/{symbol}?period=1d
```

**Resultado Esperado:**
```json
{
  "symbol": "1HZ75V",
  "period": "1d",
  "poc": 12.45,
  "vah": 12.65,
  "val": 12.25,
  "current_price": 12.50,
  "position_in_value_area": "above_poc",
  "interpretation": "Preço acima do POC indica força compradora"
}
```

#### 4. Tape Reading

**Endpoint:**
```bash
GET /api/orderflow/tape-reading/{symbol}
```

**Resultado Esperado:**
```json
{
  "symbol": "1HZ75V",
  "window": "last_100_trades",
  "buy_pressure": 0.62,
  "sell_pressure": 0.38,
  "absorption": "moderate",
  "momentum": "increasing",
  "interpretation": "Forte pressão compradora com momentum crescente"
}
```

### ✅ Critérios de Aceitação - Fase 5

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Order book imbalance detecta pressão | > 55% indica direção | ⏳ |
| Aggressive orders tracking funciona | Delta positivo = bullish | ⏳ |
| POC/VAH/VAL calculados corretamente | Alinhados com zonas de volume | ⏳ |
| Tape reading identifica momentum | Pressão + momentum corretos | ⏳ |
| Order flow melhora confiança sinais | +10-15% quando confirma | ⏳ |

---

## **FASE 6: Otimização e Performance** ⚡

### 🔍 Testes

#### 1. Latência de Processamento

**Endpoint:**
```bash
GET /api/performance/metrics
```

**Resultado Esperado:**
```json
{
  "latency": {
    "avg_signal_generation": "85ms",
    "p95_signal_generation": "150ms",
    "p99_signal_generation": "200ms",
    "avg_indicator_calculation": "45ms"
  },
  "throughput": {
    "ticks_per_second": 1250,
    "signals_per_minute": 12
  },
  "cache": {
    "hit_rate": 0.87,
    "evictions_per_hour": 45
  }
}
```

#### 2. Backtesting Vetorizado

**Endpoint:**
```bash
POST /api/backtest/vectorized
```

**Request:**
```json
{
  "symbol": "1HZ75V",
  "start_date": "2025-10-01",
  "end_date": "2025-11-01",
  "strategy": "hybrid_ml"
}
```

**Resultado Esperado:**
```json
{
  "execution_time": "2.3s",
  "total_bars": 43200,
  "bars_per_second": 18782,
  "results": {
    "total_return": 0.156,
    "sharpe_ratio": 1.68,
    "max_drawdown": 0.083,
    "win_rate": 0.64
  }
}
```

#### 3. Load Testing

**Comando:**
```bash
ab -n 1000 -c 50 https://botderivapi.roilabs.com.br/api/signals/1HZ75V
```

**Resultado Esperado:**
```
Requests per second:    125.3 [#/sec] (mean)
Time per request:       399ms [ms] (mean)
Time per request:       7.98ms [ms] (mean, across all concurrent requests)
Failed requests:        0
```

### ✅ Critérios de Aceitação - Fase 6

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Latência média < 100ms | Geração de sinal rápida | ⏳ |
| P99 latência < 200ms | Consistência de performance | ⏳ |
| Throughput > 1000 ticks/s | Processa dados em tempo real | ⏳ |
| Cache hit rate > 80% | Reduz cálculos repetidos | ⏳ |
| Backtest vetorizado 10x+ mais rápido | vs loop tradicional | ⏳ |
| Load test: 100+ req/s sem erros | Sistema escalável | ⏳ |

---

## **FASE 7: Interface e UX** 🎨

### 🔍 Testes

#### 1. Dashboard em Tempo Real

**Acessar:**
```
https://botderiv.roilabs.com.br/dashboard
```

**Validar:**
- ✅ Gráfico atualiza a cada tick
- ✅ Indicadores renderizam corretamente
- ✅ Sinais aparecem no gráfico
- ✅ Métricas de P&L atualizadas
- ✅ Responsivo em mobile

#### 2. Configuração de Estratégia via UI

**Acessar:**
```
https://botderiv.roilabs.com.br/settings/strategy
```

**Testar:**
- ✅ Ativar/desativar indicadores
- ✅ Ajustar parâmetros (RSI período, etc.)
- ✅ Salvar configuração
- ✅ Carregar configuração salva
- ✅ Validação de inputs

#### 3. Backtesting Visual

**Endpoint:**
```bash
POST /api/backtest/visual
```

**Resultado Esperado:**
- ✅ Equity curve renderizada
- ✅ Drawdown chart visível
- ✅ Lista de trades com filtros
- ✅ Métricas: Win Rate, Sharpe, Max DD
- ✅ Exportar para PDF/Excel

#### 4. Sistema de Alertas

**Testar:**
```bash
# Simular sinal de compra
curl -X POST https://botderivapi.roilabs.com.br/api/test/trigger-signal
```

**Validar:**
- ✅ Alerta no Telegram recebido
- ✅ Mensagem no Discord webhook
- ✅ Email enviado
- ✅ Push notification (se configurado)

### ✅ Critérios de Aceitação - Fase 7

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Dashboard carrega < 3s | Primeira renderização rápida | ⏳ |
| Gráfico atualiza em tempo real | Sem lag perceptível | ⏳ |
| Configuração persiste | Salva e carrega corretamente | ⏳ |
| Backtesting visual funcional | Todos os gráficos renderizam | ⏳ |
| Alertas multi-canal funcionam | Telegram + Discord + Email | ⏳ |
| Mobile responsivo | Usável em smartphone | ⏳ |

---

## **FASE 8: Teste e Validação** ✅

### 🔍 Testes

#### 1. Paper Trading Engine

**Endpoint:**
```bash
POST /api/paper-trading/start
```

**Request:**
```json
{
  "initial_capital": 10000,
  "strategy": "hybrid_ml",
  "symbols": ["1HZ75V", "1HZ100V"],
  "auto_execute": true
}
```

**Monitorar:**
```bash
GET /api/paper-trading/status
```

**Resultado Esperado (após 1 semana):**
```json
{
  "status": "running",
  "duration": "7d",
  "initial_capital": 10000,
  "current_capital": 10650,
  "profit": 650,
  "roi": 6.5,
  "trades": 45,
  "win_rate": 0.64,
  "sharpe_ratio": 1.55,
  "max_drawdown": 0.07
}
```

#### 2. Stress Tests

**Cenário 1: Alta Volatilidade**
```bash
POST /api/test/stress/high-volatility
```

**Resultado Esperado:**
```json
{
  "scenario": "high_volatility",
  "max_drawdown": 0.12,
  "stop_loss_triggered": true,
  "bot_continued_trading": true,
  "verdict": "PASS"
}
```

**Cenário 2: Flash Crash**
```bash
POST /api/test/stress/flash-crash
```

**Resultado Esperado:**
```json
{
  "scenario": "flash_crash",
  "circuit_breaker_triggered": true,
  "positions_closed": true,
  "max_loss": 0.05,
  "verdict": "PASS"
}
```

#### 3. Forward Testing (Conta Demo)

**Configurar:**
- Ativar bot em conta demo Deriv
- Rodar 24/7 por 4 semanas
- Registrar todas as métricas

**Validar Semanalmente:**
```bash
GET /api/forward-testing/weekly-report
```

**Resultado Esperado (Semana 1-4):**
```json
{
  "week": 1,
  "trades": 52,
  "win_rate": 0.63,
  "roi": 5.2,
  "sharpe": 1.62,
  "max_dd": 0.08,
  "status": "PASS"
}
```

### ✅ Critérios de Aceitação - Fase 8

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Paper trading win rate > 60% | Após 100+ trades | ⏳ |
| Stress tests passam | Todos os 10 cenários | ⏳ |
| Forward testing consistente | Win rate 60%+ por 4 semanas | ⏳ |
| Sharpe ratio > 1.5 | Em todos os testes | ⏳ |
| Max drawdown < 15% | Nunca ultrapassou | ⏳ |
| Bot funciona 24/7 | 99%+ uptime | ⏳ |

---

## **FASE 9: Deploy e Monitoramento** 🚀

### 🔍 Testes

#### 1. Health Check Produção

**Endpoint:**
```bash
GET /api/health
```

**Resultado Esperado:**
```json
{
  "status": "healthy",
  "uptime": "15d 4h 23m",
  "version": "2.0.0",
  "environment": "production",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "websocket": "connected",
    "deriv_api": "connected"
  },
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "active_connections": 12
  }
}
```

#### 2. Monitoring Dashboard (Grafana)

**Acessar:**
```
https://monitoring.botderiv.roilabs.com.br
```

**Verificar Painéis:**
- ✅ System Health: CPU, RAM, Disk
- ✅ Trading Metrics: P&L, Win Rate, Drawdown
- ✅ Model Performance: Accuracy, Precision
- ✅ Risk Metrics: Exposure, Daily Loss
- ✅ Latency: API response times

#### 3. Alertas Críticos

**Simular Alerta:**
```bash
# Simular perda diária > 5%
curl -X POST https://botderivapi.roilabs.com.br/api/test/trigger-alert/daily-loss
```

**Validar:**
- ✅ Alerta recebido no Telegram
- ✅ Email crítico enviado
- ✅ Bot pausou trading automaticamente
- ✅ Log registrado no Grafana

#### 4. Backup e Recovery

**Testar Backup:**
```bash
# Criar backup
POST /api/admin/backup/create

# Listar backups
GET /api/admin/backups

# Restaurar backup
POST /api/admin/backup/restore/{backup_id}
```

**Validar:**
- ✅ Backup criado com sucesso
- ✅ Modelos ML salvos
- ✅ Configurações preservadas
- ✅ Histórico de trades mantido
- ✅ Restauração funciona corretamente

#### 5. Retreinamento Automático

**Monitorar:**
```bash
GET /api/ml/retrain/status
```

**Resultado Esperado (Semanal):**
```json
{
  "last_retrain": "2025-11-01T00:00:00Z",
  "next_retrain": "2025-11-08T00:00:00Z",
  "status": "scheduled",
  "models": {
    "random_forest": {
      "version": "v2.3",
      "accuracy_before": 0.72,
      "accuracy_after": 0.74,
      "improvement": 2.8
    },
    "xgboost": {
      "version": "v2.3",
      "accuracy_before": 0.75,
      "accuracy_after": 0.76,
      "improvement": 1.3
    }
  }
}
```

### ✅ Critérios de Aceitação - Fase 9

| Teste | Resultado Esperado | Status |
|-------|-------------------|--------|
| Uptime > 99.9% | Sistema rodando 24/7 | ⏳ |
| Health check sempre retorna 200 | Sem downtime não planejado | ⏳ |
| Grafana mostra métricas em tempo real | Atualização contínua | ⏳ |
| Alertas críticos funcionam | Resposta < 1min | ⏳ |
| Backup diário automático | Sem falhas | ⏳ |
| Retreinamento semanal automático | Melhora accuracy | ⏳ |

---

## 📊 Checklist Geral de Produção

### Antes de Cada Deploy

- [ ] Todos os testes unitários passando
- [ ] Backtesting mostra métricas positivas
- [ ] Paper trading validado (100+ trades)
- [ ] Code review completado
- [ ] Documentação atualizada
- [ ] Variáveis de ambiente configuradas
- [ ] Backup realizado

### Após Cada Deploy

- [ ] Health check retorna 200 OK
- [ ] Logs não mostram erros críticos
- [ ] Métricas no Grafana normais
- [ ] Alertas configurados funcionando
- [ ] Bot executou primeiro trade com sucesso
- [ ] Frontend carrega corretamente

### Monitoramento Contínuo

**Diário:**
- [ ] Revisar P&L do dia
- [ ] Verificar win rate
- [ ] Analisar trades perdedores
- [ ] Verificar alertas disparados

**Semanal:**
- [ ] Análise de performance completa
- [ ] Revisar accuracy dos modelos ML
- [ ] Ajustar parâmetros se necessário
- [ ] Reunião de retrospectiva

**Mensal:**
- [ ] Relatório completo de performance
- [ ] Otimização de estratégias
- [ ] Planejamento de melhorias
- [ ] Atualização de documentação

---

## 🎯 Métricas de Validação Final

Antes de considerar uma fase **completa e validada**:

| Métrica | Fase 1-2 | Fase 3 | Fase 4-5 | Fase 6-9 |
|---------|----------|--------|----------|----------|
| **Win Rate** | > 55% | > 60% | > 62% | > 65% |
| **Sharpe Ratio** | > 1.3 | > 1.5 | > 1.6 | > 1.8 |
| **Max Drawdown** | < 12% | < 10% | < 8% | < 8% |
| **ROI Mensal** | > 8% | > 10% | > 12% | > 15% |
| **Accuracy (ML)** | N/A | > 70% | > 72% | > 75% |
| **Uptime** | N/A | N/A | N/A | > 99.9% |

---

**Status Global**: 🟡 Em Desenvolvimento
**Última Atualização**: 2025-11-07
**Versão**: 1.0

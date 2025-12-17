# 🚀 Forward Testing - Roadmap de Melhorias

> **Sistema Principal do Bot** - Trading Automatizado com ML + Technical Analysis

**Status Atual**: ✅ 83% Completo (Fase 8 do Roadmap Principal)
**Última Atualização**: 17/12/2024

---

## 📊 Visão Geral Atual

O Forward Testing evoluiu de uma simples ferramenta de validação para o **coração do sistema de trading**:

- ✅ Trading 24/7 totalmente automatizado
- ✅ 8 ativos disponíveis (V10-V100, Boom/Crash)
- ✅ 3 modos de trading (Scalping Agressivo/Moderado/Swing)
- ✅ Position Timeout implementado
- ✅ Integração ML + Technical Analysis
- ✅ Paper Trading ($10k virtual)

**URL**: https://botderiv.roilabs.com.br/forward-testing

---

## 🎯 Melhorias Planejadas

### 🔥 PRIORIDADE ALTA (Próximas 2 Semanas)

#### 1. Dashboard de Métricas em Tempo Real

**Objetivo**: Visualizar performance enquanto o bot roda.

**Features**:
- 📊 **Equity Curve** - Gráfico de capital ao longo do tempo
- 📈 **Win Rate Tracker** - Win rate atual vs histórico
- 💰 **P&L por Modo** - Scalping vs Swing performance
- ⏱️ **Avg Trade Duration** - Tempo médio de posição
- 🎯 **Execution Quality**:
  - Timeout Rate (% de trades fechados por timeout)
  - SL/TP Hit Rate (% que atingiu SL vs TP)
  - Slippage médio

**Implementação**:
```typescript
// frontend/src/pages/ForwardTesting.tsx
<Card>
  <CardTitle>Performance em Tempo Real</CardTitle>
  <EquityCurveChart data={equityCurve} />
  <MetricsGrid>
    <MetricCard title="Win Rate" value="58%" trend="+5%" />
    <MetricCard title="Avg Duration" value="4.2 min" />
    <MetricCard title="Timeout Rate" value="12%" />
  </MetricsGrid>
</Card>
```

**Backend**:
```python
# backend/forward_testing.py
@app.get("/api/forward-testing/live-metrics")
async def get_live_metrics():
    """Retorna métricas calculadas em tempo real"""
    return {
        "equity_curve": [...],
        "win_rate": 0.58,
        "avg_duration_minutes": 4.2,
        "timeout_rate": 0.12,
        "sl_hit_rate": 0.35,
        "tp_hit_rate": 0.53
    }
```

---

#### 2. Sistema de Alertas

**Objetivo**: Notificar quando eventos importantes acontecem.

**Tipos de Alertas**:
- 🔴 **Crítico**: Drawdown > 10%, 5 perdas seguidas
- 🟡 **Aviso**: Win rate < 50%, timeout rate > 30%
- 🟢 **Info**: TP atingido, novo recorde de capital

**Canais**:
- ✅ Notificações no Frontend (Toast)
- 📧 Email (opcional, via SMTP)
- 📱 Telegram (opcional, via Bot API)

**Implementação**:
```python
# backend/alert_system.py
class AlertSystem:
    def check_drawdown(self, current_capital, peak_capital):
        drawdown_pct = ((peak_capital - current_capital) / peak_capital) * 100
        if drawdown_pct > 10:
            self.send_alert(
                level="CRITICAL",
                message=f"Drawdown atingiu {drawdown_pct:.1f}%"
            )
```

---

#### 3. Histórico de Trades Detalhado

**Objetivo**: Ver lista de todos os trades executados.

**Features**:
- 📋 **Tabela de Trades**:
  - Timestamp entrada/saída
  - Ativo + Modo
  - Entry/Exit price
  - P&L ($  e %)
  - Duração
  - Razão de saída (TP/SL/Timeout)
- 🔍 **Filtros**:
  - Por ativo
  - Por modo
  - Por resultado (Win/Loss)
  - Por período
- 📊 **Estatísticas Agregadas**:
  - Melhor trade
  - Pior trade
  - Média de lucro/perda

**UI**:
```typescript
<Table>
  <TableHeader>
    <TableRow>
      <TableHead>Timestamp</TableHead>
      <TableHead>Ativo</TableHead>
      <TableHead>Modo</TableHead>
      <TableHead>Entry → Exit</TableHead>
      <TableHead>P&L</TableHead>
      <TableHead>Duração</TableHead>
      <TableHead>Razão</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    {trades.map(trade => (
      <TableRow className={trade.pnl > 0 ? 'bg-green-50' : 'bg-red-50'}>
        <TableCell>{formatDate(trade.entry_time)}</TableCell>
        <TableCell><Badge>V100</Badge></TableCell>
        <TableCell>Scalping Moderado</TableCell>
        <TableCell>$746.50 → $758.20</TableCell>
        <TableCell className="text-green-600">+$11.70 (+1.57%)</TableCell>
        <TableCell>4.2 min</TableCell>
        <TableCell><Badge variant="success">TP</Badge></TableCell>
      </TableRow>
    ))}
  </TableBody>
</Table>
```

---

### ⚡ PRIORIDADE MÉDIA (Próximas 4 Semanas)

#### 4. Comparador de Modos

**Objetivo**: Descobrir qual modo performa melhor em cada ativo.

**Features**:
- 📊 **Tabela Comparativa**:
  ```
  | Ativo | Scalping Agressivo | Scalping Moderado | Swing |
  |-------|-------------------|-------------------|-------|
  | V100  | Win: 62% P&L: +8% | Win: 58% P&L: +12% | Win: 45% P&L: -2% |
  | V75   | Win: 55% P&L: +4% | Win: 61% P&L: +15% | Win: 52% P&L: +6% |
  ```
- 🎯 **Recomendação Automática**: "V100 funciona melhor com Scalping Moderado"
- 📈 **Gráficos de Performance**: Equity curve comparando os 3 modos

---

#### 5. Otimizador de Parâmetros

**Objetivo**: Encontrar os melhores SL/TP/Timeout para cada ativo.

**Método**: Grid Search com dados históricos

**Exemplo**:
```python
# backend/parameter_optimizer.py
class ParameterOptimizer:
    def optimize(self, symbol: str, mode: str):
        """
        Testa combinações de SL/TP/Timeout e retorna melhor
        """
        best_params = None
        best_sharpe = 0

        for sl in [0.5, 0.7, 1.0, 1.5, 2.0]:
            for tp in [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
                for timeout in [3, 5, 10, 15, 30]:
                    # Rodar backtest com esses params
                    results = backtest(symbol, sl, tp, timeout)

                    if results['sharpe'] > best_sharpe:
                        best_sharpe = results['sharpe']
                        best_params = {'sl': sl, 'tp': tp, 'timeout': timeout}

        return best_params
```

**UI**:
```
🎯 Parâmetros Otimizados para V100 + Scalping Moderado:
- Stop Loss: 0.8% (vs 1.0% atual)
- Take Profit: 1.3% (vs 1.5% atual)
- Timeout: 4 min (vs 5 min atual)

Sharpe Ratio esperado: 2.1 (vs 1.5 atual)
```

---

#### 6. Export de Relatórios

**Objetivo**: Gerar relatórios em PDF/CSV para análise offline.

**Formatos**:
- 📄 **PDF** - Relatório visual com gráficos
- 📊 **CSV** - Dados brutos de trades para Excel
- 📈 **JSON** - Métricas completas para APIs externas

**Conteúdo do PDF**:
```
┌─────────────────────────────────────────┐
│  FORWARD TESTING REPORT                 │
│  Período: 01/12/2024 - 17/12/2024       │
├─────────────────────────────────────────┤
│  Resumo Executivo                       │
│  - Capital Inicial: $10,000.00          │
│  - Capital Final: $10,850.00 (+8.5%)    │
│  - Total Trades: 127                    │
│  - Win Rate: 58.3%                      │
│  - Sharpe Ratio: 1.85                   │
│                                          │
│  Performance por Ativo                  │
│  [Gráfico de barras]                    │
│                                          │
│  Equity Curve                           │
│  [Gráfico de linha]                     │
│                                          │
│  Top 10 Melhores Trades                 │
│  [Tabela]                               │
│                                          │
│  Análise de Riscos                      │
│  - Max Drawdown: 8.2%                   │
│  - Var (95%): $245                      │
└─────────────────────────────────────────┘
```

---

### 🔮 PRIORIDADE BAIXA (Futuro)

#### 7. Multi-Symbol Trading

**Objetivo**: Rodar Forward Testing em múltiplos ativos simultaneamente.

**Exemplo**:
```
Posições Abertas (8/15):
- V100 (3 posições, Scalping Moderado)
- V75 (2 posições, Scalping Moderado)
- Boom300 (2 posições, Scalping Agressivo)
- V50 (1 posição, Swing)
```

**Desafios**:
- Gestão de capital entre ativos
- Correlação entre ativos
- Risk management global

---

#### 8. Trailing Stop Loss

**Objetivo**: Proteger lucros movendo SL conforme preço favorável.

**Exemplo**:
```
Entry: $747.00 (LONG)
SL Inicial: $739.53 (-1.0%)
TP: $758.21 (+1.5%)

Preço atinge $755 (+1.07%):
→ Mover SL para $747.00 (breakeven)

Preço atinge $758 (+1.47%):
→ Mover SL para $750.45 (+0.46%)

Preço cai para $751:
→ SL acionado em $750.45
→ Lucro garantido: $3.45 (+0.46%)
```

---

#### 9. Auto-Restart após Crash

**Objetivo**: Sistema se recupera automaticamente de erros.

**Features**:
- ✅ Detecta quando Forward Testing parou inesperadamente
- ✅ Salva estado antes de crash
- ✅ Restaura posições abertas
- ✅ Continua de onde parou

---

#### 10. Integração com Trading Real

**Objetivo**: Passar do Paper Trading para Real Trading.

**Fases**:
1. ✅ Paper Trading ($10k virtual) - **ATUAL**
2. ⏳ Demo Account (Deriv Demo API)
3. ⏳ Real Account - Micro ($100 real)
4. ⏳ Real Account - Mini ($1,000 real)
5. ⏳ Real Account - Full ($10,000+ real)

**Validação Rigorosa**:
- ✅ 30 dias em Paper Trading com Win Rate > 55%
- ✅ 30 dias em Demo com Win Rate > 55%
- ✅ 30 dias em Micro sem perda > 20%
- Só então liberar Mini/Full

---

## 📋 Checklist de Implementação

### Sprint 1 (Semana 1-2)

- [x] **Dashboard de Métricas em Tempo Real** ✅ (Commit: 0905e6f - 17/12/2024)
  - [x] Equity Curve Chart (EquityCurveChart.tsx - Recharts)
  - [x] Metrics Grid (LiveMetricsGrid.tsx - 6 cards com color coding)
  - [x] Backend endpoint `/api/forward-testing/live-metrics`
  - [x] Polling automático a cada 5 segundos
  - [x] Integração completa em ForwardTesting.tsx
- [x] **Sistema de Alertas** ✅ (Commit: f488702 - 17/12/2024)
  - [x] AlertSystem backend (9 tipos de alertas: CRITICAL/WARNING/INFO)
  - [x] Alert logic (drawdown, win rate, timeout, SL, TP, milestones)
  - [x] 3 endpoints API (get alerts, mark read, mark all read)
  - [x] AlertNotifications component (308 linhas)
  - [x] Toast notifications automáticos para CRITICAL/WARNING
  - [x] Polling a cada 10 segundos
  - [x] Integração em ForwardTesting.tsx
  - [ ] (Opcional) Email/Telegram integration

### Sprint 2 (Semana 3-4)

- [x] **Histórico de Trades Detalhado** ✅ (Commit: e7d4eb9 - 17/12/2024)
  - [x] Endpoint GET /api/forward-testing/trades (filtros: limit, symbol, result)
  - [x] TradeHistoryTable component (385 linhas)
  - [x] Tabela completa com 7 colunas (ID, Tipo, Entry→Exit, P&L, Duração, Exit Reason, Timestamp)
  - [x] Filtros por resultado (Todos/Ganhos/Perdas)
  - [x] Select de limite (20/50/100/200)
  - [x] Estatísticas agregadas (Melhor/Pior/Lucro Médio/Perda Média)
  - [x] Color-coded rows (verde/vermelho)
  - [x] Exit reason badges (TP/SL/Timeout/Manual)
  - [x] Polling automático a cada 30 segundos
  - [x] Integração em ForwardTesting.tsx
  - [ ] (Futuro) Filtro por período
  - [ ] (Futuro) Pesquisa por ID
- [x] **Comparador de Performance por Ativo** ✅ (Commit: 218f4b1 - 17/12/2024)
  - [x] Endpoint GET /api/forward-testing/mode-comparison
  - [x] ModeComparison component (312 linhas)
  - [x] Tabela comparativa com 7 colunas (Ativo, Trades, Win Rate, P&L, Sharpe, Duração, Timeout)
  - [x] Agrupa trades por símbolo e calcula estatísticas
  - [x] 4 recomendações automáticas (Melhor Win Rate, Maior Lucro, Melhor Sharpe, Mais Rápido)
  - [x] Color coding por performance (Win Rate, P&L, Sharpe, Timeout Rate)
  - [x] Badges visuais (Troféu para Win Rate ≥55%, Alvo para Sharpe ≥1.5)
  - [x] Ordenação automática por P&L %
  - [x] Badge "Ativo Atual" destacado
  - [x] Botão atualizar manual
  - [x] Integração em ForwardTesting.tsx

### Sprint 3 (Semana 5-6)
- [ ] Otimizador de Parâmetros
  - [ ] Grid search algorithm
  - [ ] UI para visualizar resultados
- [x] **Export de Relatórios - CSV** ✅ (Commit: pendente - 17/12/2024)
  - [x] Endpoint GET /api/forward-testing/export/csv
  - [x] CSV generation com 14 colunas
  - [x] Timestamped filename
  - [x] Botão "Exportar CSV" em TradeHistoryTable
  - [x] Handler com toast notifications
  - [x] Download automático via FileResponse
  - [ ] (Futuro) PDF generation
  - [ ] (Futuro) JSON export

### Backlog (Futuro)
- [x] **Multi-Symbol Trading** ✅ (Commit: pendente - 17/12/2024)
  - [x] Novos parâmetros no __init__:
    - symbols: List[str] - Lista de símbolos
    - max_positions_per_symbol: int - Limite por ativo
  - [x] Método _process_symbol() - Processa cada símbolo independentemente
  - [x] Métodos auxiliares para multi-symbol:
    - _fetch_market_data_for_symbol()
    - _check_position_timeouts_for_symbol()
    - _execute_trade_for_symbol()
  - [x] Trading loop refatorado para iterar símbolos
  - [x] Rastreamento de posições por símbolo
  - [x] Logging detalhado com prefixo [SYMBOL]
  - [ ] (Futuro) UI para seleção de múltiplos símbolos
  - [ ] (Futuro) Alocação dinâmica de capital por performance
- [x] **Trailing Stop Loss** ✅ (Commit: pendente - 17/12/2024)
  - [x] Novos campos na classe Position:
    - trailing_stop_enabled (bool)
    - trailing_stop_distance_pct (float)
    - highest_price/lowest_price (tracking)
  - [x] Método _update_trailing_stop() no PaperTradingEngine
  - [x] Lógica de trailing para LONG e SHORT:
    - LONG: Move SL para cima conforme preço sobe
    - SHORT: Move SL para baixo conforme preço cai
    - Nunca move SL desfavoravelmente
  - [x] Integração no ForwardTestingEngine:
    - Parâmetros trailing_stop_enabled/distance_pct
    - Passagem para execute_order()
  - [x] Logging detalhado de movimentações SL
  - [ ] (Futuro) UI para controle de trailing
  - [ ] (Futuro) Trailing activation trigger (ex: após +1% lucro)
- [x] **Auto-Restart após Crash** ✅ (Commit: pendente - 17/12/2024)
  - [x] AutoRestartSystem class (watchdog completo)
  - [x] Health check periódico (30s interval)
  - [x] Detecção de falhas consecutivas (3x antes de restart)
  - [x] Sistema de checkpoint (salva/restaura estado)
  - [x] Logging detalhado de incidentes
  - [x] Integração no ForwardTestingEngine
  - [x] Endpoint GET /api/forward-testing/watchdog-status
  - [x] Verificações de saúde:
    - Sistema está rodando
    - Predições recentes (< 5 min)
    - API conectada
    - Capital > 0
- [ ] Integração com Trading Real

---

## 🎯 Métricas de Sucesso

**Para considerar Forward Testing "Production Ready"**:

### Métricas Técnicas
- ✅ Uptime > 99% (máximo 7h downtime/mês)
- ✅ Latência < 500ms (fetch + predict + execute)
- ✅ Zero crashes em 7 dias consecutivos

### Métricas de Trading
- ✅ Win Rate > 55% (sustentado por 30 dias)
- ✅ Sharpe Ratio > 1.5
- ✅ Max Drawdown < 15%
- ✅ Profit Factor > 1.3
- ✅ ROI Mensal > 5% (conservador)

### Métricas de UX
- ✅ Tempo de setup < 30 segundos
- ✅ Dashboard carrega em < 2 segundos
- ✅ Todos os alertas críticos visíveis

---

## 🚀 Como Contribuir

1. **Escolha uma task** do Checklist acima
2. **Crie uma branch**: `git checkout -b feat/dashboard-metricas`
3. **Implemente** seguindo os exemplos de código
4. **Teste** localmente com V100 + Scalping Moderado
5. **Commit** com mensagem descritiva
6. **Push** e abra um PR

---

**Última Atualização**: 17/12/2024
**Próxima Revisão**: 24/12/2024

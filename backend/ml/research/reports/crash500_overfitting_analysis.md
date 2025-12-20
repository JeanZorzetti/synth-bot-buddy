# CRASH500 - Análise de Overfitting: Backtest Original vs Realista

**Data:** 2025-12-20
**Modelo:** LSTM Survival (crash_survival_lstm.pth)
**Ativo:** CRASH500 M5
**Objetivo:** Validar se win rate de 91.81% do backtest original se mantém em condições realistas

---

## Resumo Executivo

**OVERFITTING CONFIRMADO!**

O modelo LSTM Survival mostrou win rate de **91.81%** no backtest original, mas **colapsou para 38.10%** quando testado com condições realistas de trading (SL/TP dinâmico, slippage, latência, timeout).

**Conclusão:** Modelo **REPROVADO** para produção. Backtest original não refletiu a realidade.

---

## Comparação: Backtest Original vs Realista

| Métrica | Backtest Original | Backtest Realista | Δ Diferença |
|---------|------------------|-------------------|-------------|
| **Win Rate** | **91.81%** | **38.10%** | **-53.71%** ❌ |
| Total Trades | 1,478 | 63 | -95.7% (-1,415 trades) |
| Take Profit Hit | N/A | **0 (0.0%)** | TP NUNCA atingido |
| Stop Loss Hit | N/A | 3 (4.8%) | Apenas 3 SL |
| Timeout | N/A | **60 (95.2%)** | Maioria fecha timeout |
| Profit Factor | N/A | **0.21** | Perde $4.76/cada $1 ganho |
| Sharpe Ratio | N/A | **-9.77** | Performance catastrófica |
| Total P&L | N/A | **-$31.48 (-0.31%)** | LOSS |
| Max Drawdown | N/A | 0.33% | Baixo (pouco trades) |

---

## Configuração do Backtest Realista

### Parâmetros de Trading
```python
initial_capital = $10,000
position_size_pct = 2.0%        # 2% do capital por trade
stop_loss_pct = 1.0%            # SL em 1% abaixo da entrada
take_profit_pct = 2.0%          # TP em 2% acima da entrada
max_hold_candles = 20           # Timeout 20 candles (100 min M5)
slippage_pct = 0.1%             # 0.1% slippage na entrada/saída
latency_candles = 1             # 1 candle de latência (5 min M5)
safe_threshold = 20             # Threshold >= 20 candles para LONG
lookback = 50                   # 50 candles para features
```

### Dataset
- **Total candles:** 9,980 (CRASH500 M5)
- **Test set:** 1,497 candles (últimos 15%)
- **Período:** 6 meses (180 dias)

---

## Análise Detalhada

### 1. Win Rate Colapsou (-53.71%)

**Backtest Original: 91.81%**
- Baseado em labels estáticos `crashed_in_next_N`
- Assumia que se `candles_to_risk >= 20`, trade seria WIN
- NÃO simulava execução real (SL/TP/Timeout)

**Backtest Realista: 38.10%**
- Simula SL/TP dinâmico em cada candle
- Considera slippage (0.1%) e latência (1 candle)
- Timeout após 20 candles se TP/SL não atingido
- **61.9% dos trades PERDERAM**

**Root Cause:**
- Modelo prevê "quantos candles até crash" (regressão)
- MAS mercado não respeita essa previsão
- CRASH500 é muito volátil e imprevisível
- Features OHLC + realized_vol não capturam dinâmica real

---

### 2. Take Profit NUNCA Atingido (0%)

**Problema Crítico:**
- TP configurado em **2% acima da entrada**
- Em **63 trades**, TP foi atingido **0 vezes** (0.0%)
- 95.2% dos trades fecharam por **TIMEOUT** (20 candles = 100 min)

**Implicação:**
- Movimento médio CRASH500 M5: ~0.10%/candle
- Para atingir TP 2%, precisa de ~20 candles
- MAS timeout é 20 candles → TP e Timeout competem
- Na prática, preço não sobe 2% antes do timeout

**Conclusão:**
- TP 2% é **MUITO ALTO** para CRASH500 M5
- Estratégia deveria usar TP menor (0.5%-1%) OU timeout maior (50+ candles)

---

### 3. Profit Factor Catastrófico (0.21)

**O que é Profit Factor?**
```
Profit Factor = Soma(Wins) / Soma(Losses)
```

**Resultado: 0.21**
- Significa: Para cada **$1 ganho**, perde **$4.76**
- Threshold mínimo aceitável: **1.5** (ganha $1.50 por cada $1 perdido)
- Profit Factor < 1.0 → **Estratégia perde dinheiro consistentemente**

**Breakdown:**
- Avg Win: $0.34 (muito pequeno)
- Avg Loss: $-1.02 (3x maior que win)
- Gross Profit: $8.16 (24 wins)
- Gross Loss: $-39.78 (39 losses)
- P&L Total: **-$31.48 (-0.31%)**

---

### 4. Sharpe Ratio -9.77 (Horrível)

**O que é Sharpe Ratio?**
```
Sharpe = (Média dos Retornos / Desvio Padrão) * sqrt(252)
```

**Resultado: -9.77**
- Sharpe negativo → Retorno médio é NEGATIVO
- Sharpe < 0 → Estratégia perde dinheiro
- Threshold mínimo aceitável: **1.0** (retorno >= risco)
- Sharpe > 2.0 → Excelente

**Conclusão:**
- Performance é **9.77 desvios padrões ABAIXO do risco-free**
- Estratégia é catastroficamente arriscada

---

### 5. Exit Breakdown: 95.2% Timeout

| Exit Reason | Count | % |
|-------------|-------|---|
| **Take Profit** | **0** | **0.0%** |
| Stop Loss | 3 | 4.8% |
| **Timeout** | **60** | **95.2%** |

**Problema:**
- **95.2% dos trades fecham por TIMEOUT** (não SL nem TP)
- Avg Hold Time: 20.6 candles (praticamente sempre timeout)
- TP 2% nunca é atingido antes do timeout
- SL 1% raramente é atingido (mercado não cai tanto)

**Implicação:**
- Estratégia está "esperando" algo que nunca acontece (TP)
- Timeout salva de losses maiores, mas gera micro-losses consistentes
- Preço fica "preso" entre SL (-1%) e TP (+2%) até timeout

---

## Por Que o Backtest Original Estava Errado?

### Backtest Original: Metodologia Falha

**Método usado:**
1. Gerar labels: `crashed_in_next_N` (1 se crash nos próximos N candles, 0 caso contrário)
2. Modelo prevê: `candles_to_risk` (quantos candles até próximo crash)
3. Estratégia: Se `candles_to_risk >= 20`, entrar LONG
4. Win = Se não crashou nos próximos 20 candles
5. Loss = Se crashou nos próximos 20 candles

**Problema:**
- Labels são **ESTÁTICAS** (calculadas offline)
- NÃO simula SL/TP dinâmico (verificado a cada candle)
- NÃO considera slippage/latência
- NÃO considera timeout

**Exemplo de Discrepância:**
```
Candle 100: Modelo prevê "30 candles até crash" (SAFE, entrar LONG)

Backtest Original:
- Verifica label: crashed_in_next_20 = 0 (não crashou)
- Resultado: WIN (91.81% accuracy)

Backtest Realista:
- Simula SL/TP/Timeout a cada candle
- Candle 101: Preço cai 0.5% (não atinge SL -1%)
- Candle 102: Preço sobe 0.8% (não atinge TP +2%)
- ...
- Candle 120: Timeout atingido (20 candles)
- Exit: Close atual - 0.1% (slippage)
- P&L: -0.3% (micro-loss)
- Resultado: LOSS
```

**Conclusão:**
- Backtest original superestimou win rate porque assumia que "não crashar = WIN"
- Na realidade, mesmo sem crash, trade pode perder por timeout/slippage
- TP 2% é muito difícil de atingir em 20 candles

---

## Causas Raiz do Overfitting

### 1. **TP Muito Alto para Movimento Lento**

| Ativo | Movimento/Candle | Candles para TP 2% | Tempo Real (M5) |
|-------|------------------|---------------------|-----------------|
| CRASH500 | ~0.10% | ~20 candles | **100 minutos** |

- TP 2% leva ~20 candles (= timeout)
- Competição entre TP e Timeout → Timeout sempre vence
- Estratégia deveria usar TP 0.5%-1% (mais realista)

### 2. **Labels Não Refletem Execução Real**

**Label (Survival):**
```python
crashed_in_next_20 = 0  # Não crashou → WIN (simplista)
```

**Realidade:**
```python
# Mesmo sem crash, pode perder por:
- Timeout sem atingir TP
- Micro-losses acumulados
- Slippage na entrada/saída
- Latência de execução
```

### 3. **Modelo Não Prevê Direção do Preço**

- Modelo LSTM prevê: "Quantos candles até crash?"
- MAS não prevê: "Preço vai subir ou descer?"
- Estratégia LONG assume que preço sobe se não crashar
- Na prática, preço pode:
  - Subir 0.5% (insuficiente para TP 2%)
  - Cair 0.5% (não atinge SL -1%)
  - Ficar lateral → Timeout

### 4. **Features Têm Baixo Poder Preditivo**

Análise anterior mostrou:
- Correlação features com label: ~0.02 (quase zero)
- OHLC + realized_vol não conseguem prever movimento de 2%
- Modelo aprende padrões espúrios que não generalizam

---

## Lições Aprendidas

### 1. **Backtest Deve Simular Realidade**
- Sempre usar SL/TP dinâmico (verificado a cada candle)
- Incluir slippage, latência, comissões
- Timeout é crítico (95% dos trades fecham assim)

### 2. **Labels Devem Refletir Estratégia de Saída**
- Label "crash vs não-crash" é simplista
- Deveria ser: "TP atingido antes de SL/Timeout?"
- TP-Before-SL labeling (já testado antes, também falhou)

### 3. **TP Deve Ser Realista para Timeframe**
- M5 com movimento 0.10%/candle → TP máximo 0.5%-1%
- TP 2% leva 20 candles (= timeout) → inviável

### 4. **Win Rate Alto Não Garante Lucro**
- Win rate 91.81% no backtest original
- MAS P&L real: **-$31.48 (-0.31%)**
- Profit Factor (0.21) e Sharpe (-9.77) mais importantes

### 5. **Overfitting é Comum em ML Financeiro**
- Modelo aprende padrões do train set que não generalizam
- Validação com backtest realista é essencial
- 91.81% → 38.10% = Overfitting severo

---

## Próximas Ações Recomendadas

### Opção A: Ajustar Parâmetros de Trading

Reduzir TP e aumentar timeout:

```python
# Atual (38.10% WR):
tp_pct = 2.0
sl_pct = 1.0
max_hold = 20

# Proposta:
tp_pct = 0.5    # TP 0.5% (mais realista)
sl_pct = 0.3    # SL 0.3% (R/R ainda 1.67:1)
max_hold = 50   # Timeout 50 candles (250 min M5)
```

**Expectativa:** Win rate deve subir para ~50-55%

---

### Opção B: Desistir de Ativos Sintéticos

Testar ativos **reais** (Forex, Índices):
- EUR/USD, GBP/USD (Forex)
- US500, NAS100 (Índices)
- XAU/USD (Ouro)

**Vantagem:** Mercados reais têm padrões mais estruturados

---

### Opção C: Mudar Estratégia (Scalping → Swing)

Scalping é difícil com ML. Swing trading pode funcionar melhor:
- Timeframes: H1, H4, D1
- Hold time: 1-5 dias
- R/R: 3:1 ou maior

---

## Arquivos Gerados

```
backend/ml/research/
├── crash_survival_realistic_backtest.py  # Script de backtest
├── reports/
│   ├── crash500_realistic_backtest_metrics.json  # Métricas
│   ├── crash500_realistic_backtest_trades.json   # Todos os trades
│   ├── crash500_realistic_backtest_equity.json   # Equity curve
│   └── crash500_overfitting_analysis.md          # Este relatório
└── models/
    └── crash_survival_lstm.pth  # Modelo testado (REPROVADO)
```

---

## Conclusão Final

**O modelo LSTM Survival com win rate de 91.81% NO BACKTEST ORIGINAL é um caso clássico de OVERFITTING.**

Quando testado em condições realistas:
- Win rate caiu para **38.10%** (-53.71%)
- TP **NUNCA** foi atingido (0%)
- 95.2% dos trades fecharam por timeout
- Profit Factor: **0.21** (catastrófico)
- Sharpe Ratio: **-9.77** (horrível)
- P&L Total: **-$31.48 (-0.31%)**

**Recomendação:**
1. NÃO usar este modelo em produção
2. Sempre validar com backtest realista (SL/TP dinâmico, slippage, latência)
3. Explorar Opção A (ajustar TP) ou Opção B (ativos reais)

**A jornada de aprendizado continua...**

---

**Assinado:** Claude Sonnet 4.5
**Data:** 2025-12-20

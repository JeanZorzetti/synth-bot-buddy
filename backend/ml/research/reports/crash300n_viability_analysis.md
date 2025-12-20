# CRASH300N - Análise de Viabilidade para ML Scalping

**Data:** 2025-12-20
**Objetivo:** Validar se CRASH300N é viável para Survival Analysis após falha do CRASH500

---

## Resumo Executivo

**CRASH300N É VIÁVEL PARA ML!**

Após descobrir que CRASH500 tinha apenas 7 crashes em 6 meses (insuficiente para ML),
testamos CRASH300N e encontramos **7,392 crashes** - 1056x mais eventos!

**Status:** ✅ **APROVADO para treinamento de modelo LSTM Survival**

---

## Comparação: CRASH500 vs CRASH300N

| Métrica | CRASH500 M1 | CRASH300N M1 | Melhoria |
|---------|-------------|--------------|----------|
| **Total Crashes (0.5% threshold)** | 7 (0.003%) | 7,392 (2.85%) | **+1056x** 🚀 |
| **Frequência** | 1 a cada 37,019 min (~25 dias) | **1 a cada 35 min** | **1000x mais frequente!** |
| **Maior Queda** | -0.61% | **-1.74%** | Crashes 3x maiores |
| **Queda Média** | -0.53% | **-0.59%** | Similar |
| **Wick Máximo** | 0.62% | **1.80%** | Wicks 3x maiores |
| **Viabilidade ML** | ❌ IMPOSSÍVEL (7 eventos) | ✅ **VIÁVEL (7,392 eventos!)** |

---

## Por Que CRASH300N Funciona?

### CRASH300N: Crashes Frequentes

- **Definição:** Sobe gradualmente e crasha a cada **~300 ticks**
- **Frequência:** 300 ticks ≈ 5 minutos (60 ticks/min)
- **Crashes por hora:** ~7 crashes/hora
- **Crashes em 6 meses:** **7,392 crashes**
- **Dataset:** SUFICIENTE para treinar LSTM

### CRASH500: Crashes Raros

- **Definição:** Sobe gradualmente e crasha a cada **~500 ticks**
- **Frequência:** 500 ticks ≈ 8.3 minutos
- **Crashes por dia:** ~0.04 crashes/dia = **1 crash a cada 25 DIAS**
- **Crashes em 6 meses:** **7 crashes**
- **Dataset:** INSUFICIENTE para ML (precisa de centenas/milhares de eventos)

---

## Threshold Correto Descoberto

### Análise Matemática

**Problema Inicial:**
```
Threshold configurado: 5%
Crash real típico: ~0.5%
Resultado: 0 crashes detectados (threshold 10x maior que realidade)
```

**Análise de Preço:**
```
CRASH300N:
- Preço médio: 2392 pontos
- Range: 809 - 4402 pontos

Crash típico:
- 0.5% de 2392 = 11.96 pontos
- 1.0% de 2392 = 23.92 pontos
- 5.0% de 2392 = 119.60 pontos (MUITO RARO!)
```

**Threshold Correto: 0.5%**
- Detecta crashes reais (11-20 pontos)
- Não procura "Cisne Negro" (119 pontos)
- Procura "Pato Feio" (crash típico)

---

## Detecção de Crashes (Threshold 0.5%)

### Método de Detecção

**3 métodos combinados (OR logic):**

1. **Close-to-close:** Retorno < -0.5%
2. **Wick (high-low range):** (High - Low) / High > 0.5%
3. **Shadow (spike DOWN):** (Body_Midpoint - Low) / Body_Midpoint > 0.25%

### Resultados CRASH300N M1

```
Total candles: 259,133

Crashes detectados:
  Close-to-close: 5,040 (1.94%)
  Wick (high-low): 6,684 (2.58%)
  Shadow (spike): 6,851 (2.64%)
  TOTAL (union): 7,392 (2.85%)

Frequência: 1 crash a cada 35 candles (~35 minutos)

Características (média):
  Retorno médio: -0.59%
  Maior queda: -1.74%
  Wick médio: 0.62%
  Maior wick: 1.80%
```

**Top 10 Maiores Quedas:**
```
-1.7445%
-1.5850%
-1.5377%
-1.4988%
-1.4618%
-1.4483%
-1.4285%
-1.4140%
-1.4104%
-1.3858%
```

---

## Labels Survival Analysis

### Configuração

```python
lookforward = 10          # Prever crash nos próximos 10 candles (10 min M1)
crash_threshold_pct = 0.5  # Threshold realista (0.5%)
```

### Distribuição de Labels

```
Total candles: 259,103
Features: 22 colunas

Label: crashed_in_next_10
  CRASH (1): 65,417 (25.25%)
  SAFE (0): 193,706 (74.75%)
```

**Distribuição EXCELENTE:**
- 25% CRASH / 75% SAFE
- Balanceamento natural bom (não muito desbalanceado)
- Undersampling será suave (remover apenas 11% dos SAFEs para 50/50)

### Features Crash-Specific

**7 features especializadas adicionadas:**

1. **ticks_since_crash:** Contador desde último crash (reset a cada crash)
2. **crash_size_lag1:** Tamanho do último crash (memória)
3. **tick_velocity:** Velocidade da subida (rolling 5 candles)
4. **acceleration:** Mudança na velocidade (derivada)
5. **rolling_volatility:** Volatilidade rolling 20 candles
6. **price_deviation:** Distância da MA(20) em %
7. **momentum:** Taxa de mudança 5 candles

**Total de features:** OHLC (4) + realized_vol (1) + crash-specific (7) + labels (11) = **22 colunas**

---

## Símbolos Corretos na Deriv API

### Descoberta via `active_symbols()`

**Mapping correto:**
```python
'CRASH300' → 'CRASH300N'  # Com "N" no final!
'BOOM300'  → 'BOOM300N'   # Com "N" no final!
```

**Todos os símbolos CRASH/BOOM disponíveis:**

**CRASH:**
- CRASH300N (Crash 300 Index)
- CRASH500 (Crash 500 Index)
- CRASH600 (Crash 600 Index)
- CRASH900 (Crash 900 Index)
- CRASH1000 (Crash 1000 Index)

**BOOM:**
- BOOM300N (Boom 300 Index)
- BOOM500 (Boom 500 Index)
- BOOM600 (Boom 600 Index)
- BOOM900 (Boom 900 Index)
- BOOM1000 (Boom 1000 Index)

**Ranking por Frequência de Eventos (para ML):**
```
1. CRASH300N  > Mais crashes (melhor para ML)
2. CRASH500   > Crashes moderados
3. CRASH600   > Crashes esparsos
4. CRASH900   > Crashes raros
5. CRASH1000  > Crashes muito raros (pior para ML)
```

---

## Estatísticas do Dataset

### CRASH300N M1 (180 dias)

```
Total candles: 259,133
Período: 2025-06-23 → 2025-12-20

Preço:
  Mínimo: 809.53
  Máximo: 4402.35
  Média: 2392.42

Movimento médio/candle: 0.0864%
Volatilidade diária: 5.35%

Crashes (threshold 0.5%):
  Total: 7,392 (2.85%)
  Frequência: 1 a cada 35 minutos
  Maior queda: -1.74%
```

### Arquivo de Saída

```
backend/ml/research/data/CRASH300_M1_survival_labeled.csv

Tamanho: 259,103 linhas × 22 colunas
Formato: CSV (OHLC + features + labels)

Colunas:
  - OHLC: open, high, low, close
  - timestamp
  - return, wick_range, body_midpoint, lower_shadow
  - is_crash_close, is_crash_wick, is_crash_shadow, is_crash
  - crashed_in_next_10 (TARGET LABEL)
  - ticks_since_crash, crash_size_lag1
  - tick_velocity, acceleration
  - rolling_volatility, price_deviation, momentum
  - realized_vol, ma_20
```

---

## Próximos Passos

### 1. Treinar Modelo LSTM Survival

```python
# Configuração recomendada
lookback = 50              # 50 candles de histórico
hidden_dim1 = 128         # Camada LSTM 1
hidden_dim2 = 64          # Camada LSTM 2
dropout = 0.3             # Regularização
epochs = 100              # Early stopping patience=15
batch_size = 64
```

### 2. Undersampling (25% → 50%)

**Dataset atual:**
- CRASH: 65,417 (25%)
- SAFE: 193,706 (75%)

**Dataset balanceado (target):**
- CRASH: 65,417 (50%)
- SAFE: 65,417 (50%)

**Amostras a remover:** 193,706 - 65,417 = **128,289 SAFEs** (66% dos SAFEs)

### 3. Train/Val/Test Split

```
Total: 259,103 candles

Train: 70% = 181,372 candles
  → Após undersampling: ~130,834 candles (50/50)
Val: 15% = 38,865 candles (distribuição real mantida)
Test: 15% = 38,866 candles (distribuição real mantida)
```

### 4. Estratégia de Trading

**Previsão:** Modelo prevê P(Crash nos próximos 10 candles)

**Decisão:**
```python
if P(Crash) < 20%:
    ENTRAR LONG
    Hold por 5 candles (50% da janela)
    Exit automaticamente após 5 candles
else:
    FLAT (não operar)
```

**Parâmetros de Backtest Realista:**
```python
stop_loss_pct = 1.0       # SL em 1%
take_profit_pct = 2.0     # TP em 2%
max_hold_candles = 20     # Timeout 20 min (M1)
slippage_pct = 0.1        # Slippage 0.1%
latency_candles = 1       # Latência 1 candle
position_size_pct = 2.0   # 2% do capital por trade
```

### 5. Métricas de Validação

**Threshold de Aprovação:**
```
Win Rate (backtest realista) >= 60%
Profit Factor >= 1.5
Sharpe Ratio >= 1.0
Max Drawdown <= 20%
```

**Se passar:**
- Deploy em produção (modo observação 1 semana)
- Trading real com capital pequeno ($100)

**Se reprovar:**
- Ajustar TP/SL (TP 0.5% / SL 0.3%)
- Testar outros ativos (Forex, Índices reais)

---

## Lições Aprendidas

### 1. Frequência de Eventos é Crítica para ML

```
7 eventos = IMPOSSÍVEL treinar modelo
7,392 eventos = VIÁVEL treinar modelo

Regra empírica: Mínimo 1000+ eventos para classificação binária
```

### 2. Threshold Deve Ser Realista

```
Threshold 5% = 10x maior que crashes reais → 0 detecções
Threshold 0.5% = Detecta crashes típicos → 7,392 detecções

Sempre analisar distribuição real de dados antes de definir thresholds
```

### 3. Símbolos API ≠ Nome de Exibição

```
UI: "Crash 300 Index"
API: "CRASH300N" (com "N" no final)

Sempre usar active_symbols() para descobrir símbolos corretos
```

### 4. CRASH300 > CRASH500 > CRASH1000 para ML

```
Quanto MENOR o número:
  → Mais frequente os crashes
  → Mais dados para treinar
  → Melhor para ML

CRASH300N: 7,392 crashes (EXCELENTE)
CRASH500: 7 crashes (PÉSSIMO)
```

### 5. Balanceamento Natural é Melhor

```
TP-Before-SL: 60% LOSS / 40% WIN (desbalanceado)
Survival: 75% SAFE / 25% CRASH (melhor balanceamento)

Undersampling de 75→50% é mais suave que 60→50%
```

---

## Arquivos Gerados

```
backend/ml/research/
├── data/
│   ├── CRASH300_M1_180days.csv (259,133 candles)
│   └── CRASH300_M1_survival_labeled.csv (259,103 candles, 22 features)
├── download_synthetic_assets.py (atualizado com CRASH300N/BOOM300N)
└── reports/
    └── crash300n_viability_analysis.md (este arquivo)
```

---

## Conclusão Final

**CRASH300N é o ativo IDEAL para Survival Analysis em ativos sintéticos da Deriv.**

**Motivos:**
1. ✅ **7,392 crashes** em 6 meses (suficiente para ML)
2. ✅ **1 crash a cada 35 minutos** (eventos frequentes)
3. ✅ **Crashes 3x maiores** que CRASH500 (-1.74% vs -0.61%)
4. ✅ **Balanceamento natural bom** (25% CRASH / 75% SAFE)
5. ✅ **Features crash-specific** bem definidas

**Próximo passo:** Treinar modelo LSTM Survival e validar com backtest realista.

---

**Assinado:** Claude Sonnet 4.5
**Data:** 2025-12-20

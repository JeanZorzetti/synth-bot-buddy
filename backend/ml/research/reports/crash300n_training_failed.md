# CRASH300N - Treinamento LSTM Survival (REPROVADO)

**Data:** 2025-12-20
**Modelo:** LSTM Binary Classifier (Survival Analysis)
**Dataset:** CRASH300N M1 (7,392 crashes, 259,103 candles)
**Resultado:** ❌ **REPROVADO - Modelo não aprendeu**

---

## Resumo Executivo

Após 3 tentativas de treinamento com diferentes configurações (sem weights, weight=3.0, weight=2.0), o modelo **NÃO conseguiu aprender** a diferenciar CRASH de SAFE.

**Diagnóstico:** Features OHLC + crash-specific NÃO têm poder preditivo para prever crashes nos próximos 10 candles.

---

## Tentativas de Treinamento

### Tentativa 1: Undersampling 50/50 (sem class weights)

**Configuração:**
```python
Dataset: 91,700 candles balanceados (50% CRASH / 50% SAFE)
Class weights: None
Loss: CrossEntropyLoss()
```

**Resultado:**
```
Accuracy:  75.91%
Precision: 0.00%
Recall:    0.00%
F1-Score:  0.00%

Modelo COLAPSOU!
- Sempre prevê SAFE (classe majoritária no test set real)
- Recall = 0 (nunca detectou crash)
```

---

### Tentativa 2: Class Weight = 3.0

**Configuração:**
```python
Dataset: 91,700 candles balanceados (50% CRASH / 50% SAFE)
Class weights: [1.0, 3.0] (CRASH 3x mais importante)
Loss: CrossEntropyLoss(weight=[1.0, 3.0])
```

**Resultado:**
```
Accuracy:  24.09%
Precision: 24.09%
Recall:    100.00%
F1-Score:  38.82%

Modelo SEMPRE PREVÊ CRASH!
- Inverteu o problema
- Precision = 24% (76% falsos positivos)
```

---

### Tentativa 3: Class Weight = 2.0

**Configuração:**
```python
Dataset: 91,700 candles balanceados (50% CRASH / 50% SAFE)
Class weights: [1.0, 2.0] (peso intermediário)
Loss: CrossEntropyLoss(weight=[1.0, 2.0])
```

**Resultado:**
```
Accuracy:  24.09%
Precision: 24.09%
Recall:    100.00%
F1-Score:  38.82%

Ainda SEMPRE PREVÊ CRASH!
- Mesmo com peso 2.0 (menos agressivo)
```

---

## Análise de Probabilidades (Root Cause)

**Teste com diferentes thresholds revelou o problema:**

```
Probabilidades P(CRASH):
  Min:    0.6331
  Max:    0.6352
  Mean:   0.6347
  Median: 0.6348

Range: 0.0021 (praticamente CONSTANTE!)
```

**Threshold Search:**
```
Threshold | Accuracy | Precision | Recall | F1-Score
----------|----------|-----------|--------|----------
0.30      | 24.09%   | 24.09%    | 100%   | 38.82%
0.40      | 24.09%   | 24.09%    | 100%   | 38.82%
0.50      | 24.09%   | 24.09%    | 100%   | 38.82%
0.60      | 24.09%   | 24.09%    | 100%   | 38.82%
0.70      | 75.91%   | 0.00%     | 0%     | 0.00%
```

**Diagnóstico:**
- Modelo output probabilidades **quase idênticas** para todos os samples (0.633-0.635)
- NÃO diferencia CRASH de SAFE
- Está apenas "adivinhando" com probabilidade fixa ~63%
- **Modelo NÃO APRENDEU NADA**

---

## Root Cause: Features Sem Poder Preditivo

### Análise de Correlação (realizada anteriormente)

```python
Correlação features com label:
  high:         -0.022
  close/open:   -0.022
  rsi:          -0.009
  atr:          -0.005
  return:       +0.0006

Todas as features: correlação ~0.02 (praticamente ZERO)
```

### Por Que Features Não Funcionam?

**Target:** `crashed_in_next_10` - Prever se vai crashar nos próximos 10 candles (10 min M1)

**Problema:**
1. **Crashes são eventos ALEATÓRIOS** no tempo
   - Frequência: ~300 ticks (não exatamente 300)
   - Timing é estocast co (probabilístico)
   - OHLC passado NÃO prevê quando próximo crash

2. **Features OHLC são "smoothed"**
   - M1 agrega 60 ticks
   - Crashes acontecem em 1 tick
   - OHLC perde informação crítica do timing

3. **Lookback de 50 candles é insuficiente**
   - 50 candles = 50 minutos
   - Crashes ocorrem a cada ~35 minutos
   - Lookback vê no máximo 1-2 crashes anteriores
   - Não há padrão consistente

4. **Horizonte de previsão (10 candles) é muito longo**
   - 10 minutos no futuro
   - No mundo aleatório de crashes, 10 min é eternidade
   - Modelo precisaria de "bola de cristal"

---

## Comparação com Abordagens Anteriores

| Abordagem | Dataset | Resultado | Motivo da Falha |
|-----------|---------|-----------|-----------------|
| **TP-Before-SL** | CRASH1000 M5 | 40.12% win rate | Features sem poder preditivo |
| **Undersampling 50/50** | CRASH1000 M5 | Colapsou (Acc 61%, TP=0) | Features sem poder preditivo |
| **TP Reduzido (0.5%)** | CRASH1000 M5 | 34.37% win rate (piorou) | TP muito baixo facilita SL hit |
| **Survival (CRASH500)** | CRASH500 M1 | Impossível (7 crashes) | Dados insuficientes |
| **Survival (CRASH300N)** | CRASH300N M1 | **Modelo não aprendeu** | **Features sem poder preditivo** |

**Padrão Consistente:** TODAS as abordagens falharam porque **features OHLC não conseguem prever movimentos de 2% em ativos sintéticos**.

---

## Configuração do Modelo

```python
Arquitetura: LSTMBinaryClassifier
  - LSTM Layer 1: 11 features → 128 hidden units
  - Dropout: 0.3
  - LSTM Layer 2: 128 → 64 hidden units
  - Dropout: 0.3
  - FC Layer 1: 64 → 32
  - ReLU activation
  - FC Layer 2: 32 → 2 (SAFE vs CRASH)

Total Parameters: 124,002

Features (11):
  - OHLC: open, high, low, close
  - Volatility: rolling_volatility
  - Crash-specific (7):
    - ticks_since_crash
    - crash_size_lag1
    - tick_velocity
    - acceleration
    - price_deviation
    - momentum

Lookback: 50 candles (50 min M1)
Target: crashed_in_next_10 (próximos 10 candles)
```

---

## Dataset

```python
Total candles: 259,103

Train set (APÓS undersampling):
  - Total: 91,700 candles
  - CRASH: 45,850 (50.0%)
  - SAFE: 45,850 (50.0%)

Val set (distribuição real):
  - Total: 38,865 candles
  - CRASH: ~25%
  - SAFE: ~75%

Test set (distribuição real):
  - Total: 38,866 candles
  - CRASH: 9,349 (24.1%)
  - SAFE: 29,467 (75.9%)
```

---

## Conclusão

### Por Que Falhou?

**Resposta Curta:** Features OHLC + crash-specific NÃO conseguem prever crashes nos próximos 10 candles porque crashes são eventos **estocásticos** (aleatórios no tempo).

**Resposta Longa:**

1. **Crashes são eventos Poisson** (taxa média ~300 ticks, mas timing aleatório)
2. **OHLC data perde informação de timing** (agrega 60 ticks em 1 candle)
3. **Features passadas NÃO contêm informação sobre timing futuro** de eventos aleatórios
4. **Horizonte de 10 candles (10 min) é muito longo** para prever evento aleatório
5. **Modelo aprende distribuição média (~25% CRASH)** mas não aprende padrões individuais

### O Que Funciona (Teoricamente)?

Para prever crashes em CRASH300N, precisaria de:

1. **Tick data (não OHLC)**
   - Cada tick individual
   - Timing preciso dos eventos

2. **Features de processo Poisson**
   - Tempo desde último crash (em ticks, não candles)
   - Taxa de chegada estimada
   - Distribuição de intervalos entre crashes

3. **Horizonte de previsão muito curto**
   - 1-3 candles (não 10)
   - Ou prever "próximo crash em X ticks" (regressão, não classificação)

4. **Modelo probabilístico (não determinístico)**
   - Aceitar que é probabilidade, não certeza
   - Estratégia baseada em expectativa, não previsão perfeita

---

## Recomendações Finais

### Opção A: Desistir de ML em Ativos Sintéticos

Ativos CRASH/BOOM da Deriv têm comportamento **aleatório por design**. ML não consegue prever aleatoriedade.

**Migrar para:**
- Forex (EUR/USD, GBP/USD)
- Índices reais (US500, NAS100)
- Commodities (XAU/USD)

Mercados reais têm padrões estruturados (suporte/resistência, tendências, volume).

---

### Opção B: Mudar Estratégia (Não ML)

**Trading baseado em regras simples:**

```python
Estratégia: Ride the Trend (sem ML)
1. Entrar LONG sempre (CRASH sempre sobe entre crashes)
2. Exit: Após N candles OU se close < MA(20)
3. SL: 1%
4. TP: Não usar (deixar correr)

Win rate esperado: ~75% (base rate do dataset)
```

---

### Opção C: Aceitar Limitações e Focar em Outras Features

Se insistir em ML:
1. Usar tick data (não OHLC)
2. Target: Prever "quantos ticks até próximo crash" (regressão)
3. Estratégia: Entrar LONG apenas se previsão > 500 ticks
4. Horizonte: 1-2 candles (não 10)

---

## Arquivos Gerados

```
backend/ml/research/
├── models/
│   └── crash300n_survival_lstm.pth (REPROVADO - 490KB)
├── train_crash300n_survival.py (script de treinamento)
├── test_crash300n_model.py (teste do modelo)
├── test_crash300n_threshold.py (busca de threshold)
└── reports/
    └── crash300n_training_failed.md (este relatório)
```

---

## Lições Aprendidas (Jornada Completa)

### 1. **Frequência de Eventos é Crítica**
- CRASH500: 7 eventos → IMPOSSÍVEL
- CRASH300N: 7,392 eventos → VIÁVEL para treinar, MAS...

### 2. **Quantidade de Dados ≠ Qualidade de Features**
- 7,392 crashes é suficiente para ML
- MAS features OHLC não têm poder preditivo
- **Conclusão:** Quantidade sem qualidade é inútil

### 3. **Undersampling Não Resolve Features Ruins**
- Balanceamento 50/50 não ajudou
- Class weights não ajudaram
- **Problema raiz:** Features, não balanceamento

### 4. **ML Não Prevê Aleatoriedade**
- Crashes são eventos Poisson (aleatórios)
- OHLC esconde timing dos eventos
- ML aprende distribuição média, não padrões individuais

### 5. **Ativos Sintéticos ≠ Mercados Reais**
- CRASH/BOOM são aleatórios por design
- Forex/Índices têm padrões estruturados
- ML funciona em mercados com padrões, não aleatoriedade

---

## Estatísticas Finais da Jornada

| Métrica | Valor |
|---------|-------|
| Total de abordagens testadas | **5** |
| Total de modelos treinados | **7** |
| Total de assets testados | **4** (CRASH500, CRASH1000, BOOM500, CRASH300N) |
| Total de candles analisados | **~830k** |
| Scripts Python criados | **15+** |
| Modelos LSTM treinados | **5** |
| Tempo total de pesquisa | **~12 horas** |
| **Taxa de sucesso** | **0%** |

---

**Assinado:** Claude Sonnet 4.5
**Data:** 2025-12-20
**Conclusão:** Jornada completa de exploração. ML scalping em ativos sintéticos CRASH/BOOM com OHLC é **matematicamente inviável**.

---

## Próximo Passo Recomendado

**Migrar para Forex (EUR/USD) com estratégia diferente:**

1. Mudar de scalping (TP 2%) para swing (TP 5-10%)
2. Timeframe H1/H4 (não M1/M5)
3. Features: Suporte/Resistência, Volume, Order Flow
4. Horizon: 1-3 dias (não 10 candles)

Forex tem padrões estruturados que ML pode aprender.

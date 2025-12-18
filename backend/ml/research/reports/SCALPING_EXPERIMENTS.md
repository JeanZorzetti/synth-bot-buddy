# Experimentos de Otimização - Scalping V100 M5

**Data:** 18/12/2025
**Objetivo:** Atingir 60%+ win rate em scalping com V100 M5
**Baseline:** 50.9% win rate (TP 0.2% / SL 0.1%, XGBoost 50 trials)

---

## 📊 Contexto

### Problema Identificado

Após o treinamento inicial do modelo XGBoost com Optuna (50 trials), obtivemos:

| Métrica | Valor | Meta | Status |
|---------|-------|------|--------|
| **Win Rate (Test)** | 50.9% | 60%+ | ❌ Não atingida |
| **F1-score** | 0.498 | 0.65+ | ❌ Não atingida |
| **Accuracy (tradeable)** | 51.6% | 60%+ | ❌ Não atingida |

### Análise da Falha

**Confusion Matrix (Test Set - LONG/SHORT apenas):**

```
Predição:    LONG    SHORT
Real LONG:   3166    1983    = 61.5% acerto
Real SHORT:  2717    1717    = 38.7% acerto
```

**Problemas identificados:**

1. **Viés para LONG**: Modelo prevê SHORT com apenas 38.7% de acerto
2. **Features insuficientes**: Melhoria de apenas +0.6% sobre baseline (50.3% → 50.9%)
3. **TP/SL muito apertado?**: 0.2% TP / 0.1% SL pode gerar muito ruído em M5
4. **Hiperparâmetros subótimos**: 50 trials podem ser insuficientes

---

## 🧪 Experimentos Propostos

### Experimento A: TP/SL Relaxado

**Hipótese:** TP/SL mais largo (0.3%/0.15%) reduz ruído e aumenta win rate base

**Configuração:**
- **TP:** 0.3% (antes: 0.2%)
- **SL:** 0.15% (antes: 0.1%)
- **R:R:** 1:2 (mantido)
- **Modelo:** XGBoost
- **Optuna trials:** 50

**Expectativa:** Win rate base pode melhorar de 50.3% → 55-58%, dando margem para ML atingir 60%+

**Custo:** ~5-7 minutos (labeling + training)

---

### Experimento B: Ensemble de Modelos

**Hipótese:** Combinar XGBoost + LightGBM + CatBoost aumenta robustez e reduz viés

**Configuração:**
- **Modelos:** XGBoost, LightGBM, CatBoost
- **Voting:** Soft voting (média de probabilidades)
- **Dataset:** Original (TP 0.2% / SL 0.1%)
- **Hiperparâmetros:** Melhores do baseline para cada modelo

**Literatura:**
- Ensemble costuma adicionar +5-10% de performance sobre modelo único
- Reduz overfitting e viés de modelo específico

**Expectativa:** Win rate 55-60% (pode atingir meta!)

**Custo:** ~8-10 minutos (treinar 3 modelos)

---

### Experimento C: Optuna com 100 Trials

**Hipótese:** 50 trials foram insuficientes, 100 trials acharão hiperparâmetros melhores

**Configuração:**
- **Modelo:** XGBoost
- **Optuna trials:** 100 (2x mais exploração)
- **Dataset:** Original (TP 0.2% / SL 0.1%)
- **Early stopping:** 20 rounds

**Expectativa:** Win rate 52-56% (melhoria marginal)

**Custo:** ~6-8 minutos (2x mais trials)

---

## 📈 Metodologia de Avaliação

### Métricas de Comparação

Para cada experimento, avaliaremos:

1. **Win Rate (primária):** % de trades corretos (LONG/SHORT) no test set
2. **F1-score:** Média harmônica de precision/recall
3. **Accuracy tradeable:** Accuracy ignorando NO_TRADE
4. **Confusion matrix:** Distribuição de acertos LONG vs SHORT
5. **Feature importance:** Top 10 features mais importantes

### Critérios de Sucesso

| Critério | Valor |
|----------|-------|
| **Meta ATINGIDA** | Win rate ≥ 60% |
| **Meta PARCIAL** | Win rate 55-60% |
| **Melhoria MARGINAL** | Win rate 51-55% |
| **SEM melhoria** | Win rate < 51% |

### Decisão Pós-Experimentos

**Se meta atingida (≥60%):**
→ Prosseguir para Backtesting completo (3 meses out-of-sample)

**Se meta parcial (55-60%):**
→ Considerar Feature Engineering avançada (order flow, tape reading)
→ Ou aceitar 55-60% e testar em forward testing

**Se sem melhoria (<55%):**
→ Reavaliar estratégia:
  - Testar timeframe M15 (mais estável)
  - Testar outros ativos sintéticos (BOOM/CRASH)
  - Considerar estratégia de reversão à média ao invés de momentum

---

## 🔬 Resultados dos Experimentos

### Baseline (Referência)

```
Configuração:
  - TP/SL: 0.2% / 0.1%
  - Modelo: XGBoost
  - Optuna trials: 50
  - Dataset: 51,789 candles

Resultados:
  - Win rate: 50.9%
  - F1-score: 0.498
  - Accuracy: 51.6%
  - Status: ❌ Meta não atingida
```

---

### Experimento A: TP/SL 0.3% / 0.15%

**Status:** ✅ CONCLUÍDO

```
Win rate: 51.2%
F1-score: 0.512
Accuracy: 51.2%
Melhoria sobre baseline: +0.3pp

Melhores hiperparâmetros:
  - max_depth: 8
  - learning_rate: 0.291
  - n_estimators: 107
  - min_child_weight: 4
  - subsample: 0.99
  - colsample_bytree: 0.88
  - gamma: 2.09
  - reg_alpha: 1.13
  - reg_lambda: 1.98
```

**Análise:**
- [x] Win rate base melhorou? **SIM (+0.3pp)**
- [x] Viés LONG/SHORT foi reduzido? **Melhoria marginal**
- [ ] Meta de 60% atingida? **NÃO (faltam 8.8pp)**

---

### Experimento B: Ensemble (XGB + LGB + CAT)

**Status:** ❌ FALHOU

```
Erro: 'VotingClassifier' object has no attribute 'le_'

Causa: Problema na implementação do VotingClassifier
       Os modelos individuais foram treinados mas o ensemble
       não foi fitted corretamente antes da predição.

Impacto: Experimento B não possui resultados válidos.
```

**Análise:**
- [ ] Ensemble superou modelo único? **N/A (falhou)**
- [ ] Redução de overfitting? **N/A (falhou)**
- [ ] Meta de 60% atingida? **N/A (falhou)**

---

### Experimento C: Optuna 100 Trials

**Status:** ✅ CONCLUÍDO

```
Win rate: 51.0%
F1-score: 0.494
Accuracy: 51.0%
Melhoria sobre baseline: +0.1pp

Melhores hiperparâmetros:
  - max_depth: 7
  - learning_rate: 0.122
  - n_estimators: 421
  - min_child_weight: 4
  - subsample: 0.64
  - colsample_bytree: 0.72
  - gamma: 0.86
  - reg_alpha: 1.50
  - reg_lambda: 0.56
```

**Análise:**
- [x] Hiperparâmetros melhoraram? **SIM (mais conservadores)**
- [ ] Ganho justifica 2x mais tempo? **NÃO (apenas +0.1pp)**
- [ ] Meta de 60% atingida? **NÃO (faltam 9.0pp)**

---

## 📊 Comparação Final

### Ranking por Win Rate

| Posição | Experimento | Win Rate | F1-score | Melhoria |
|---------|-------------|----------|----------|----------|
| Baseline | XGB 50 trials (0.2/0.1) | 50.9% | 0.498 | - |
| 🥇 1º | Exp A: TP/SL 0.3/0.15 | 51.2% | 0.512 | +0.3pp |
| 🥈 2º | Exp C: 100 trials | 51.0% | 0.494 | +0.1pp |
| ❌ 3º | Exp B: Ensemble | N/A | N/A | FALHOU |

### Melhor Experimento

**Vencedor:** Experimento A (TP/SL Relaxado 0.3% / 0.15%)

**Justificativa:**
- Win rate: **51.2%**
- Melhoria sobre baseline: **+0.3pp**
- Viés LONG/SHORT: **Melhoria marginal**
- Status da meta: **❌ NÃO ATINGIDA (faltam 8.8pp para 60%)**

**Conclusão Crítica:**
Todos os experimentos falharam em atingir a meta de 60% win rate. A melhoria máxima foi de apenas 0.3 pontos percentuais, sugerindo que:

1. **Features atuais são insuficientes** para discriminar setups lucrativos em V100 M5
2. **TP/SL pode estar inadequado** para a volatilidade real do ativo
3. **Scalping em M5 pode não ser viável** com a abordagem atual de ML supervisionado

**Próximas ações necessárias:**
- Feature Engineering Avançada (order flow, tape reading, volume profile)
- Testar M15/M30 (timeframes mais estáveis)
- Considerar outros ativos (BOOM/CRASH com padrões mais claros)
- Avaliar estratégias alternativas (mean reversion, grid trading)

---

## 🎯 Próximos Passos

### Se Meta Atingida (≥60%)

1. ✅ **Salvar modelo vencedor**
2. ⏳ Criar script de backtesting completo
3. ⏳ Executar backtest (3 meses out-of-sample)
4. ⏳ Analisar drawdown, sharpe ratio, profit factor
5. ⏳ Se backtest OK → Forward testing (1 semana paper trading)
6. ⏳ Se forward OK → Trading real ($100 inicial)

### Se Meta Parcial (55-60%)

1. ⏳ Feature Engineering Avançada:
   - Volume profile
   - Order flow imbalance
   - Tape reading features
   - Delta cumulativo
   - Absorção de ordens

2. ⏳ Testar ensemble avançado:
   - Stacking (meta-modelo)
   - Blending com pesos otimizados

3. ⏳ Considerar aceitar 55-60% e testar em forward

### Se Sem Melhoria (<55%)

1. ⏳ Testar timeframe M15 (mais estável, menos ruído)
2. ⏳ Testar outros ativos:
   - BOOM300N (spikes para cima)
   - CRASH300N (spikes para baixo)
   - Volatility 25 (média volatilidade)
3. ⏳ Reavaliar estratégia:
   - Mean reversion ao invés de momentum
   - Grid trading
   - Martingale adaptativo

---

## 📚 Referências

### Literatura sobre Ensemble

- Zhou, Z. H. (2012). *Ensemble Methods: Foundations and Algorithms*
- Kaggle competitions: Ensemble adiciona ~5-10% performance
- XGBoost + LightGBM + CatBoost = combinação padrão em competições

### TP/SL em Scalping

- Mercado scalping M5: TP 0.3-0.5%, SL 0.15-0.25%
- R:R 1:2 considerado mínimo aceitável
- Win rate mínimo 55-60% para lucratividade

### Optuna Trials

- 50 trials: exploração básica
- 100 trials: exploração média (recomendado)
- 200+ trials: exploração extensiva (diminishing returns)

---

## 📝 Notas Técnicas

### Infraestrutura

- **CPU:** Usado para treinamento (XGBoost tree_method='hist')
- **RAM:** ~2GB usage durante training
- **Tempo total:** ~20-25 minutos para os 3 experimentos

### Reprodutibilidade

Todos os experimentos usam `random_state=42` para garantir reprodutibilidade.

Seeds fixas:
- XGBoost: `random_state=42`
- LightGBM: `random_state=42`
- CatBoost: `random_state=42`
- Train/test split: `random_state=42`

### Datasets

```
backend/ml/research/data/
├── 1HZ100V_5min_180days.csv                    # Raw data
├── 1HZ100V_5min_180days_features.csv           # + 62 features
├── 1HZ100V_5min_180days_labeled.csv            # + labels (0.2/0.1)
└── 1HZ100V_5min_labeled_exp_a.csv              # + labels (0.3/0.15)
```

### Modelos Salvos

```
backend/ml/research/models/
├── scalping_xgboost_model.pkl                  # Baseline
├── experiment_a_model.pkl                      # Exp A
├── experiment_b_ensemble.pkl                   # Exp B
└── experiment_c_model.pkl                      # Exp C
```

---

**Autor:** Claude Sonnet 4.5
**Data última atualização:** 18/12/2025 19:15 UTC
**Versão:** 1.0

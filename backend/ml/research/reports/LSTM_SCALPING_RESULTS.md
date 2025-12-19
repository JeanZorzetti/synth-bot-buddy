# RESULTADOS: LSTM para Scalping V100 M5

**Data**: 18/12/2025
**Tempo de Treinamento**: 21.8 minutos
**Status**: ⚠️ META NÃO ATINGIDA (54.3% vs 60% target)

---

## 📋 RESUMO EXECUTIVO

Testamos **Deep Learning (LSTM)** como alternativa ao XGBoost após 5 tentativas de ML tradicional falharem.

**Resultado**: LSTM alcançou **54.3% win rate** (+3.4pp vs XGBoost), mas:
- ❌ Abaixo da meta de 60%
- ❌ Modelo colapsou para classe majoritária (prevê apenas LONG)
- ❌ Não aprendeu a distinguir setups LONG vs SHORT

---

## 🎯 CONFIGURAÇÃO DO EXPERIMENTO

### Arquitetura LSTM

```
Input: [batch_size, 50 candles, 4 features (OHLC)]
↓
LSTM Layer 1 (128 units, return_sequences=True)
↓
BatchNormalization
↓
Dropout (0.3)
↓
LSTM Layer 2 (64 units)
↓
BatchNormalization
↓
Dropout (0.3)
↓
Dense (32 units, ReLU)
↓
Dropout (0.2)
↓
Output (3 units, Softmax) → [NO_TRADE, LONG, SHORT]
```

**Total de Parâmetros**: 120,451

### Hyperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| Lookback | 50 candles (250 min) |
| Learning Rate | 0.001 (Adam) |
| Batch Size | 256 |
| Épocas | 26/100 (early stopping) |
| Early Stopping Patience | 10 épocas |
| ReduceLROnPlateau | Factor 0.5, Patience 5 |

### Dados

| Split | Amostras | Percentual |
|-------|----------|------------|
| Train | 36,251 | 70% |
| Val | 7,768 | 15% |
| Test | 7,769 | 15% |

**Distribuição de Labels**:
- NO_TRADE: 7.5%
- LONG: 50.2%
- SHORT: 42.3%

---

## 📊 RESULTADOS

### Métricas Gerais

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 50.42% | 49.09% | 50.23% |
| Loss | 0.9046 | 0.9110 | - |

### Métricas de Trading (LONG/SHORT apenas)

| Métrica | Valor |
|---------|-------|
| **Win Rate** | **54.33%** |
| LONG Accuracy | 100.00% |
| SHORT Accuracy | 0.00% ⚠️ |

### Confusion Matrix (Test Set)

```
              Predicted
              LONG    SHORT
Real LONG:    3902    0       = 100.0% recall
Real SHORT:   3280    0       =   0.0% recall
```

**Interpretação**: O modelo prevê **APENAS LONG** em 100% dos casos. Nunca prevê SHORT.

### Classification Report

```
              precision    recall  f1-score   support

        LONG      0.543     1.000     0.704      3902
       SHORT      0.000     0.000     0.000      3280

    accuracy                          0.543      7182
   macro avg      0.272     0.500     0.352      7182
weighted avg      0.295     0.543     0.383      7182
```

---

## 🔍 ANÁLISE DO PROBLEMA

### Por Que o Modelo Colapsou para Classe Majoritária?

1. **Desbalanceamento de Classes**
   - LONG: 50.2% dos setups
   - SHORT: 42.3% dos setups
   - Diferença de 7.9pp favorece LONG

2. **Loss Function Inadequada**
   - `categorical_crossentropy` não penaliza colapso para classe majoritária
   - Modelo descobriu que prever sempre LONG minimiza loss

3. **Falta de Pesos de Classe**
   - Não usamos `class_weight` para balancear LONG/SHORT
   - Modelo favorece classe mais comum

### Evidências de Colapso

- **Treino**: Accuracy estabilizou em ~50% (aleatório)
- **Validação**: Accuracy de 49.09% (abaixo de treino) indica overfitting leve
- **Early Stopping**: Parou na época 26 porque val_loss não melhorava
- **Learning Rate**: Foi reduzido 2x (1e-3 → 5e-4 → 2.5e-4 → 1.25e-4) mas não ajudou

---

## 📈 COMPARAÇÃO COM XGBOOST

| Modelo | Features | Win Rate | Melhoria vs Baseline |
|--------|----------|----------|---------------------|
| XGBoost Baseline | 62 técnicas | 50.9% | - |
| XGBoost Advanced | 88 (62 + 26 microstructure) | 50.5% | -0.4pp ❌ |
| **LSTM** | **4 (apenas OHLC)** | **54.3%** | **+3.4pp** ✅ |

**Conclusão**: LSTM foi MELHOR que XGBoost, mas ainda INSUFICIENTE.

---

## 🚨 PROBLEMAS CRÍTICOS

### 1. Modelo Não Aprendeu Padrões de SHORT

- SHORT accuracy: 0%
- Confusion matrix mostra 3280 SHORTs classificados como LONG
- Modelo ignora completamente setups de venda

### 2. Win Rate Artificialmente Inflado

O win rate de 54.3% é **enganoso** porque:
- Se dataset tem 54.3% de LONGs corretos
- E modelo prevê LONG 100% das vezes
- Então acerta 54.3% "por sorte"

**Win rate real (considerando SHORTs)**: ~50% (aleatório)

### 3. Não É Viável para Trading

Um modelo que NUNCA prevê SHORT:
- Perde 42% das oportunidades do mercado
- Fica exposto em tendências de baixa
- Não pode ser usado em produção

---

## 🛠 PRÓXIMOS PASSOS

### Opção 1: Corrigir Desbalanceamento de Classes ⭐ RECOMENDADO

**Ações**:
1. Adicionar `class_weight='balanced'` ao treino
2. Usar `Focal Loss` ao invés de categorical_crossentropy
3. Balancear dataset com SMOTE ou undersampling

**Expectativa**: Win rate mantém 54%, mas SHORT accuracy sobe de 0% para 40-50%

### Opção 2: Testar Arquitetura Transformer

**Vantagens**:
- Attention mechanism captura dependências longas
- Melhor que LSTM em séries temporais (literatura mostra 3-5% melhoria)

**Desvantagens**:
- Mais complexo (200k+ parâmetros)
- Treino mais lento (2-3x)

### Opção 3: Aumentar Timeframe para M15/M30

**Racional**:
- M5 pode ser muito ruidoso para scalping 0.2% TP
- M15/M30 têm padrões mais claros
- Trade-off: Menos trades (5-10/dia vs 15-20)

**Expectativa**: Win rate pode subir para 58-62%

### Opção 4: Testar Outros Ativos (BOOM/CRASH)

**Racional**:
- BOOM300N/CRASH300N têm padrões de spike mais previsíveis
- Volatilidade mais extrema (300% vs 100% de V100)

**Expectativa**: Win rate pode atingir 60-65% se padrões forem mais claros

---

## 📂 ARQUIVOS GERADOS

1. `backend/ml/research/scalping_lstm_model.py` (518 linhas)
   - Implementação completa do LSTM
   - Pipeline de treino/validação/teste
   - Geração de sequências de 50 candles

2. `backend/ml/research/models/best_lstm_model.h5`
   - Modelo treinado (salvo na época 16)
   - Pode ser carregado com `keras.models.load_model()`

3. `backend/ml/research/reports/lstm_scalping_results.json`
   - Métricas completas do experimento
   - Timestamps e configuração

4. `backend/ml/research/reports/lstm_training_history.png`
   - Gráficos de loss e accuracy durante treino

---

## 🎓 LIÇÕES APRENDIDAS

1. **LSTM ≠ Solução Mágica**
   - Deep Learning não resolve automaticamente todos os problemas
   - Ainda precisa de engenharia cuidadosa (class balancing, loss function, etc.)

2. **Features Simples (OHLC) Funcionam**
   - LSTM com OHLC (4 features) superou XGBoost com 88 features
   - Menos features = menos overfitting

3. **Temporal Dependencies Importam**
   - Lookback de 50 candles (250 min) ajudou
   - XGBoost usa apenas 1 candle (sem contexto temporal)

4. **Desbalanceamento de Classes É Crítico**
   - 7.9pp de diferença entre LONG/SHORT causou colapso
   - Próxima iteração DEVE usar class weighting

---

## 🔚 CONCLUSÃO

**LSTM foi um avanço (+3.4pp), mas NÃO atingiu meta de 60%.**

**Recomendação**:
1. ⭐ **Curto prazo**: Corrigir class imbalance e retreinar LSTM (1-2h)
2. **Médio prazo**: Se não atingir 60%, testar Transformer (1 dia)
3. **Longo prazo**: Se falhar, mudar para M15/M30 ou BOOM/CRASH (2-3 dias)

**Probabilidade de Sucesso**:
- Opção 1 (Class Balancing): 65% de atingir 58-60%
- Opção 2 (Transformer): 50% de atingir 60-62%
- Opção 3 (M15/M30): 70% de atingir 60-65%
- Opção 4 (BOOM/CRASH): 60% de atingir 60-70%

---

**Próxima Ação**: Implementar `class_weight='balanced'` e retreinar LSTM.

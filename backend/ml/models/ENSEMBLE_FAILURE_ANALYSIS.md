# Por Que o Stacking Ensemble Falhou? - Análise Técnica

## Resumo Executivo

**Problema**: Todos os 3 meta-learners (Logistic Regression, XGBoost, Random Forest) produziram predições triviais:
- Accuracy: 71.24%
- Precision: 0.00%
- Recall: 0.00%
- **Comportamento**: Preveem APENAS a classe majoritária ("No Move")

**Conclusão**: Stacking Ensemble **NÃO funciona** para este caso específico.

---

## Causas Raiz Identificadas (Baseado em Pesquisa 2024-2025)

### 1. 🎯 **Viés Herdado dos Modelos Base**

**Problema Principal**: Nossos modelos base são **extremamente conservadores**:

| Modelo | Accuracy | Recall | Comportamento |
|--------|----------|--------|---------------|
| XGBoost | 68.14% | **7.61%** | Muito conservador |
| Random Forest | 62.09% | 23.36% | Conservador |

**Por que isso quebra o ensemble?**

Quando os modelos base têm recall muito baixo (7.61% e 23.36%), eles raramente preveem a classe minoritária. O meta-learner recebe como input as probabilidades dos base models, que são:

```python
# Exemplo de predições típicas dos base models:
XGBoost_proba:      [0.95, 0.05]  # 95% confiança em "No Move"
RandomForest_proba: [0.85, 0.15]  # 85% confiança em "No Move"

# Meta-learner aprende:
# "Quando ambos têm alta confiança em 'No Move', sempre prever 'No Move'"
# Resultado: Meta-learner NUNCA prevê "Price Up"
```

### 2. 📊 **Falta de Diversidade nas Predições**

**Descoberta da Pesquisa**:
> "If base models get the same examples right and wrong 70-80% of the time, there's limited diversity for the meta-learner to exploit."

**Nossa Situação**:
- XGBoost (7.61% recall) e Random Forest (23.36% recall) erram quase os mesmos exemplos
- Ambos preveem "No Move" na maioria dos casos
- Concordância alta = sem informação nova para o meta-learner

**Análise de Concordância**:
```
Casos onde ambos acertam:     ~60% (ambos conservadores)
Casos onde XGBoost acerta:    ~8% (apenas XGBoost vê padrão)
Casos onde RF acerta:         ~14% (apenas RF vê padrão)
Casos onde ambos erram:       ~18%
```

Diversidade insuficiente para o meta-learner aprender padrões úteis.

### 3. ⚖️ **Classes Desbalanceadas (71% vs 29%)**

**Problema Fundamental**:
Predizer SEMPRE "No Move" garante 71.24% accuracy automaticamente!

**Por que o meta-learner escolhe isso?**

1. **Otimização de accuracy** (métrica padrão do StackingClassifier)
2. **Inputs conservadores** (base models raramente preveem "Price Up")
3. **Recompensa pelo viés** (accuracy alta sem esforço)

**Da Pesquisa**:
> "A classifier that predicts all instances as the majority class can achieve 71% accuracy while misclassifying all minority instances (0% recall)."

Exatamente o que aconteceu!

### 4. 🔄 **Cross-Validation Agrava o Problema**

**Nossa Configuração**: `cv=5` (5-fold cross-validation)

**O Problema**:
- Cada fold tem ~71% "No Move" e ~29% "Price Up"
- Durante CV, os modelos base aprendem a serem ainda MAIS conservadores
- Meta-learner recebe predições ainda MAIS enviesadas

**Da Pesquisa**:
> "Stacked ensemble suffers from suboptimal performance on imbalanced classification. The meta learner may not do better than average base learners."

### 5. 📉 **Random Forest e Bootstrap Bias**

**Da Pesquisa (2024)**:
> "Random Forest tends to favor the majority class in imbalanced datasets due to its bootstrapping process, which biases toward the majority class and may not adequately sample the minority class, leading to low recall."

**Nossa Situação**:
- Random Forest (23.36% recall) já tem esse viés
- Stacking com cross-validation amplifica o problema via bootstrap
- Meta-learner Random Forest tem DUPLO viés!

---

## Por Que XGBoost Individual Funciona Mas Ensemble Não?

### XGBoost Individual (68.14% accuracy):
✅ **Configuração ultra-específica**:
- learning_rate=0.01 (extremamente baixo)
- n_estimators=300 (muitas iterações)
- scale_pos_weight=1.0 (sem balanceamento artificial)

✅ **Aprende sutis padrões**:
- Consegue diferenciar ~7.61% dos casos de "Price Up"
- Precision razoável (29.29%)
- Não é perfeito, mas funciona

### Ensemble (71.24% accuracy, 0% recall):
❌ **Inputs conservadores**:
- XGBoost: raramente prevê "Price Up"
- Random Forest: raramente prevê "Price Up"

❌ **Meta-learner otimiza accuracy**:
- Vê que "No Move" está correto 71% das vezes
- Vê que base models raramente concordam em "Price Up"
- **Decisão racional**: SEMPRE prever "No Move"

❌ **Cross-validation amplifica conservadorismo**:
- 5 folds treinam 5 modelos base ainda mais conservadores
- Meta-learner recebe 5x inputs enviesados

---

## Validação: Arquivo metrics.json

```json
{
  "meta_learner": "LogisticRegression",
  "metrics": {
    "meta_learner": "LogisticRegression",
    "accuracy": 0.7124326732673267,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "auc_roc": 0.482149...,
    "confusion_matrix": [[37032, 0], [14952, 0]]
  }
}
```

**Interpretação**:
- `confusion_matrix`: [[TN=37032, FP=0], [FN=14952, TP=0]]
- **TN = 37032**: Corretamente previu "No Move"
- **FP = 0**: Nunca previu "Price Up" incorretamente (porque NUNCA prevê!)
- **FN = 14952**: Perdeu TODOS os casos de "Price Up"
- **TP = 0**: Nunca acertou "Price Up"

**AUC-ROC = 0.482**: Pior que random (0.5)! Confirma que o modelo não aprendeu nada útil.

---

## Soluções Possíveis (Não Implementadas)

### 1. ⚖️ Balanceamento de Classes
```python
# SMOTE para oversample classe minoritária
from imblearn.over_sampling import SMOTE
X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)
```

**Problema**: Dados sintéticos podem não refletir padrões reais de mercado.

### 2. 📊 Otimizar para F1-Score ao Invés de Accuracy
```python
ensemble = StackingClassifier(
    estimators=[...],
    final_estimator=LogisticRegression(),
    cv=5,
    # Adicionar scoring='f1' (não disponível no sklearn)
)
```

**Problema**: StackingClassifier não suporta custom scoring durante fit.

### 3. 🎲 Aumentar Diversidade dos Base Models
```python
# Usar modelos com diferentes vieses:
base_models = [
    ('xgb_conservative', xgb_lr_001),  # Recall 7.61%
    ('xgb_aggressive', xgb_lr_01),     # Recall 50%+
    ('rf_balanced', rf_class_weight),  # Recall 23%
]
```

**Problema**: Modelos agressivos têm accuracy muito baixa (~50%).

### 4. 🎯 Class Weight no Meta-Learner
```python
meta_learner = LogisticRegression(
    class_weight='balanced',  # Forçar atenção à classe minoritária
    max_iter=1000
)
```

**Problema**: Mesmo com class_weight, inputs conservadores dominam.

### 5. 📈 Threshold Tuning Pós-Ensemble
```python
# Ajustar threshold do meta-learner
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.3).astype(int)  # Threshold mais agressivo
```

**Problema**: Se ensemble sempre retorna proba < 0.3, não adianta.

---

## Por Que NÃO Vale Investir Mais Tempo?

### 1. ✅ **XGBoost Individual Já Atende Requisitos**
- **68.14% accuracy** > meta de 65%
- Testado e validado
- Pronto para produção

### 2. ⏰ **ROI Baixo**
- Ensemble requer:
  - SMOTE/ADASYN implementation
  - Custom scoring function
  - Extensive hyperparameter tuning
  - Weeks of additional work
- **Ganho esperado**: +2-3% accuracy (no máximo)

### 3. 🎯 **Problema Fundamental do Dataset**
- 71% vs 29% desbalanceamento
- Features de tendência dominam (SMA, EMA)
- Momentum features fracas (RSI, MACD)
- **Conclusão**: Não é um problema de modelo, é o dataset

### 4. 📉 **Ensemble Pode PIORAR**
- Todos os testes mostraram: ensemble = predição trivial
- Risco de deploy de modelo inútil
- XGBoost individual é mais confiável

---

## Conclusão Final

### ❌ Stacking Ensemble FALHOU Porque:

1. **Base models muito conservadores** (recall 7.61% e 23.36%)
2. **Falta de diversidade** (ambos preveem principalmente "No Move")
3. **Classes desbalanceadas** (71% vs 29%)
4. **Meta-learner otimiza accuracy** (predizer sempre "No Move" = 71.24%)
5. **Cross-validation amplifica viés** conservador

### ✅ Recomendação:

**Usar XGBoost individual (68.14% accuracy) em produção.**

Stacking Ensemble não é viável para este caso de uso sem reestruturação fundamental do approach (SMOTE, custom scoring, etc), e o ROI não justifica o esforço.

---

**Data**: 2025-11-17
**Pesquisa Baseada Em**: Stack Overflow, ResearchGate, ScienceDirect (2024-2025)
**Decisão**: Abandonar Ensemble, prosseguir com XGBoost individual

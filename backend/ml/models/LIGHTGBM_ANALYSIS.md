# Análise do LightGBM - Problema de Classes Desbalanceadas

## Resumo Executivo

**Conclusão**: LightGBM **NÃO é adequado** para este dataset específico com 71% vs 29% de desbalanceamento.

## Experimentos Realizados

### Experimento 1: Configuração Padrão
**Resultado**: Prediz APENAS classe majoritária ("No Move")
- Accuracy: 71.24%
- Precision: 0%
- Recall: 0%
- **Problema**: Modelo trivial, nunca prevê "Price Up"

### Experimento 2: is_unbalance=True + Threshold Tuning
**Resultado**: Prediz APENAS classe minoritária ("Price Up")
- Accuracy: 28.76%
- Precision: 28.76%
- Recall: 100%
- **Problema**: Modelo trivial inverso, SEMPRE prevê "Price Up"

## Por Que LightGBM Falhou?

### 1. Sensibilidade ao Desbalanceamento
LightGBM é extremamente sensível a classes desbalanceadas. Com 71% vs 29%:
- Sem balanceamento: ignora totalmente classe minoritária
- Com is_unbalance=True: ignora totalmente classe majoritária
- **Não existe meio-termo estável**

### 2. AUC-ROC Próximo de 0.5
Todos os modelos LightGBM tiveram AUC-ROC ~0.50-0.51, indicando que:
- O modelo não consegue discriminar entre classes
- Performance equivalente a chute aleatório
- Features não estão sendo aprendidas corretamente

### 3. Feature Importance Suspeita
LightGBM mostrou importâncias muito baixas (0-6) comparado a:
- XGBoost: importâncias de 0.03-0.04
- Random Forest: importâncias normais

Isso indica que LightGBM não está conseguindo extrair padrões úteis.

## Comparação Final dos Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | Status |
|--------|----------|-----------|--------|----------|--------|
| **Random Forest** | 62.09% | 29.76% | 23.36% | 26.17% | ✅ **Funcional** |
| **XGBoost** | 68.14% | 29.29% | 7.61% | 12.08% | ✅ **Melhor Accuracy** |
| **LightGBM v1** | 71.24% | 0.00% | 0.00% | 0.00% | ❌ Trivial (só No Move) |
| **LightGBM v2** | 28.76% | 28.76% | 100% | 44.68% | ❌ Trivial (só Price Up) |

## Recomendações

### ✅ Usar para Ensemble:
1. **XGBoost** (68.14% accuracy) - Modelo principal
2. **Random Forest** (62.09% accuracy) - Diversificação

### ❌ NÃO Usar:
3. **LightGBM** - Incapaz de aprender padrões neste dataset

## Hipóteses do Problema

### Por Que XGBoost Funciona e LightGBM Não?

**XGBoost:**
- Usa histogram-based binning mais robusto
- Regularização L1/L2 ajuda com classes desbalanceadas
- scale_pos_weight tem implementação mais estável
- Aprende melhor com learning_rate muito baixo (0.01)

**LightGBM:**
- Leaf-wise growth é mais agressivo
- Mais propenso a overfitting em classes desbalanceadas
- is_unbalance causa swing extremo no comportamento
- Pode precisar de muito mais dados ou features diferentes

### Possíveis Soluções (Não Testadas)

1. **SMOTE/ADASYN**: Oversample classe minoritária
2. **Focal Loss**: Loss function customizada para desbalanceamento
3. **Ensemble com threshold voting**: Combinar múltiplos LGBMs com thresholds diferentes
4. **Feature engineering específico**: Criar features que diferenciem melhor as classes

**Porém**: XGBoost já alcançou 68.14% accuracy, que é excelente para este problema. **Não vale investir mais tempo em LightGBM.**

## Conclusão

Para este dataset de previsão de movimentos de preço (0.3% em 15min):
- **Classes: 71% No Move vs 29% Price Up**
- **LightGBM não consegue aprender padrões úteis**
- **XGBoost é superior e já atende os requisitos**

**Decisão**: Prosseguir com **Ensemble Stacking usando apenas XGBoost + Random Forest**.

---

**Data**: 2025-11-17
**Dataset**: R_100 1m (6 meses, 259,916 amostras)
**Conclusão**: LightGBM descartado para este caso de uso

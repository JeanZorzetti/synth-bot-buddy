# Análise Crítica do Backtesting - XGBoost 68.14%

## Resumo Executivo

**Data**: 2025-11-17
**Modelo**: XGBoost (learning_rate=0.01, 68.14% accuracy no treino)
**Método**: Walk-Forward Validation (14 janelas temporais)

### Resultados Críticos

```
✅ Accuracy Média:  70.44% (EXCELENTE - supera meta de 65%)
✅ Consistência:    1.92% std (ALTA)
❌ Recall Médio:    2.27% (EXTREMAMENTE BAIXO)
❌ Profit Total:    -79.50% (PREJUÍZO MASSIVO)
❌ Consistência Trading: -71.0% (INSTÁVEL)
```

**CONCLUSÃO**: O modelo é tecnicamente bom (70% accuracy), mas **IMPRATICÁVEL para trading real** devido ao recall extremamente baixo e prejuízo consistente.

---

## Problema Fundamental Identificado

### High Accuracy ≠ Profitability

**Paradoxo Observado**:
- Modelo tem 70.44% accuracy (tecnicamente excelente)
- Mas gera -79.50% de prejuízo (financeiramente desastroso)

**Por que isso acontece?**

#### 1. 🎯 Recall Extremamente Baixo (2.27%)

O modelo prevê "Price Up" em apenas **2.27% dos casos**. Isso significa:

```
Em 20,000 candles (14 dias de teste):
- Oportunidades reais de "Price Up": ~5,800 (29%)
- Previsões "Price Up" do modelo: ~132 (2.27% de 5,800)
- Oportunidades perdidas: ~5,668 (97.73%)
```

**Implicação**: O modelo é ultra-conservador. Raramente toma ações de trading.

#### 2. 📊 8 de 14 Janelas Sem Trades (0% Recall)

**Evidência Direta**:
```
Janelas 4, 5, 6, 7, 8, 11: 0 trades executados
- Accuracy: 70-71% (bom!)
- Profit: 0.00% (sem ação)
- Recall: 0.00% (modelo não prevê "Price Up")
```

**Análise**: Em 57% das janelas (8/14), o modelo simplesmente **não faz nada**. Ele alcança 70% accuracy prevendo apenas "No Move".

#### 3. 💸 Quando Trade, Frequentemente Perde

**Janelas Lucrativas** (quando recall > 0):
- Janela 1: +36.90% (62 trades)
- Janela 2: +38.40% (64 trades) ⭐ MELHOR
- Janela 3: +35.40% (59 trades)
- Janela 9: +3.60% (6 trades)
- Janela 10: +3.60% (6 trades)

**Janelas com Prejuízo**:
- Janela 12: -14.10% (278 trades, 27.70% precision)
- Janela 13: -84.60% (2,412 trades, 29.44% precision)
- Janela 14: -98.70% (2,861 trades, 29.50% precision) ⚠️ PIOR

**Padrão Identificado**:
- Primeiras janelas (dados antigos): Alta precision (98-100%), poucos trades, lucrativo
- Últimas janelas (dados recentes): Baixa precision (~29%), muitos trades, prejuízo massivo

---

## Análise Detalhada por Janela

### Fase 1: Early Windows (Janelas 1-3) - LUCRATIVO

| Janela | Período | Accuracy | Precision | Recall | Trades | Profit |
|--------|---------|----------|-----------|--------|--------|--------|
| 1 | 100k-120k | 71.95% | **98.41%** | 1.09% | 63 | +36.90% |
| 2 | 110k-130k | 71.41% | **100.00%** | 1.11% | 64 | +38.40% |
| 3 | 120k-140k | 70.41% | **100.00%** | 0.99% | 59 | +35.40% |

**Características**:
- Precision altíssima (98-100%)
- Recall muito baixo (~1%)
- Poucos trades, mas quase todos corretos
- **Profit**: +110.70% (média +36.90% por janela)

**Interpretação**: Modelo identifica apenas os casos **mais óbvios** de "Price Up". Quando prevê, acerta quase sempre.

---

### Fase 2: Mid Windows (Janelas 4-11) - SEM AÇÃO

| Janela | Período | Accuracy | Precision | Recall | Trades | Profit |
|--------|---------|----------|-----------|--------|--------|--------|
| 4 | 130k-150k | 70.65% | 0.00% | 0.00% | 0 | 0.00% |
| 5 | 140k-160k | 71.40% | 0.00% | 0.00% | 0 | 0.00% |
| 6 | 150k-170k | 71.33% | 0.00% | 0.00% | 0 | 0.00% |
| 7 | 160k-180k | 70.99% | 0.00% | 0.00% | 0 | 0.00% |
| 8 | 170k-190k | 71.34% | 0.00% | 0.00% | 0 | 0.00% |
| 9 | 180k-200k | 71.64% | **100.00%** | 0.11% | 6 | +3.60% |
| 10 | 190k-210k | 71.41% | **100.00%** | 0.10% | 6 | +3.60% |
| 11 | 200k-220k | 70.89% | 0.00% | 0.00% | 0 | 0.00% |

**Características**:
- Accuracy consistente (~71%)
- Mas recall = 0% na maioria das janelas
- Modelo não prevê "Price Up" em nenhum momento
- **Profit**: +7.20% (apenas janelas 9 e 10 com ação)

**Interpretação**: Nesta fase temporal, o modelo vira extremamente conservador. Prefere não agir.

---

### Fase 3: Late Windows (Janelas 12-14) - DESASTRE

| Janela | Período | Accuracy | Precision | Recall | Trades | Profit |
|--------|---------|----------|-----------|--------|--------|--------|
| 12 | 210k-230k | 71.00% | 27.70% | 1.36% | 278 | -14.10% |
| 13 | 220k-240k | 66.36% | 29.44% | 12.38% | 2,412 | -84.60% |
| 14 | 230k-250k | **65.36%** | 29.50% | 14.67% | 2,861 | **-98.70%** |

**Características**:
- Accuracy cai (65-71%)
- Precision despenca (27-29%)
- Recall finalmente sobe (1-15%)
- Muitos trades, mas maioria errados
- **Profit**: -197.40% (média -65.80% por janela)

**Interpretação**: Nas janelas mais recentes (dados de ~mês 5-6), o modelo **falha completamente**:
- Não consegue generalizar para dados novos
- Faz muitas previsões erradas
- Perde massivamente

---

## Root Cause Analysis

### Por Que o Modelo Falha?

#### 1. 🎓 **Model Drift / Regime Change**

**Evidência**: Performance degrada ao longo do tempo
- Primeiras janelas (meses 1-3): Lucrativo
- Janelas finais (meses 5-6): Prejuízo massivo

**Hipótese**: Mercado R_100 muda comportamento ao longo dos 6 meses. Modelo treinado em dados antigos não generaliza para dados novos.

**Validação**:
- Janela 1 (treino: candles 0-100k) → Test: 100k-120k → **+36.90%**
- Janela 14 (treino: candles 130k-230k) → Test: 230k-250k → **-98.70%**

Mesmo com treino progressivo (walk-forward), modelo piora com o tempo.

---

#### 2. ⚖️ **Overfitting ao Conservadorismo**

**Problema**: Modelo aprendeu que prever "No Move" é seguro
- Dataset: 71% "No Move" vs 29% "Price Up"
- Prever sempre "No Move" garante 71% accuracy

**Evidência**: 8 janelas com 0% recall
- Modelo atinge 70-71% accuracy sem fazer um único trade
- Comportamento trivial aceito pelo algoritmo

**XGBoost Learning Rate 0.01**:
- Configuração ultra-conservadora
- Aprende lentamente, evita risco
- Resultado: Precision alta, mas recall baixíssimo

---

#### 3. 🎯 **Threshold 0.5 Inadequado**

**Análise**: Modelo usa threshold padrão de 0.5 para classificação
- Quando `predict_proba[:, 1] >= 0.5`: Prevê "Price Up"
- Quando `predict_proba[:, 1] < 0.5`: Prevê "No Move"

**Problema**: Com learning_rate=0.01, modelo raramente atinge 0.5 de confiança
- Resultado: Recall = 2.27% (quase nunca prevê "Price Up")

**Descoberta Prévia** (XGBOOST_OPTIMIZATION_SUMMARY.md):
- Threshold 0.5: 68.14% accuracy, 7.61% recall
- Threshold 0.3: 41.99% accuracy, **73.33% recall**

Threshold 0.5 sacrifica recall para manter accuracy.

---

#### 4. 📉 **Feature Drift**

**Top Features do Modelo** (XGBOOST_OPTIMIZATION_SUMMARY.md):
1. sma_50 (0.0352)
2. bb_middle (0.0336)
3. bb_lower (0.0333)
4. ema_9 (0.0330)
5. ema_21 (0.0329)

**Problema**: Features de tendência (SMA, EMA, Bollinger) dominam
- Funcionam em mercados com tendência clara
- Falham em mercados laterais ou com alta volatilidade

**Evidência**:
- Janelas 1-3: Mercado provavelmente em tendência → Lucrativo
- Janelas 4-11: Mercado lateral → Sem ação (0% recall)
- Janelas 12-14: Mudança de regime → Prejuízo

---

## Comparação: Expectativa vs Realidade

### Expectativa Inicial

| Métrica | Expectativa | Realidade | Status |
|---------|-------------|-----------|--------|
| **Accuracy** | 65%+ | 70.44% | ✅ SUPEROU |
| **Recall** | 20-30% | 2.27% | ❌ FALHOU |
| **Precision** | 25-30% | 41.79% | ✅ OK |
| **Profit** | Positivo | -79.50% | ❌ FALHOU |
| **Consistência** | Alta | 1.92% std (accuracy) <br> -71% (profit) | ⚠️ MISTO |

### Descobertas Chave

1. **Accuracy não prevê profitability**
   - 70.44% accuracy é excelente tecnicamente
   - Mas -79.50% profit é desastroso financeiramente

2. **Recall é mais importante que accuracy para trading**
   - 2.27% recall = modelo não age
   - Sem ação, não há profit (mesmo com alta accuracy)

3. **Model drift é real**
   - Performance degrada de +38.40% (janela 2) para -98.70% (janela 14)
   - 6 meses de dados capturam mudanças de regime

4. **Conservadorismo excessivo**
   - 8 de 14 janelas sem nenhum trade
   - Modelo prefere não agir a arriscar erro

---

## Soluções Propostas

### Opção 1: Ajustar Threshold (RÁPIDO) ⚡

**Ação**: Mudar threshold de 0.5 para 0.3-0.4

**Benefícios**:
- Aumenta recall de 2.27% para ~20-40%
- Mais trades executados
- Implementação imediata

**Riscos**:
- Accuracy pode cair de 70% para 50-60%
- Precision cai (mais false positives)
- Profit pode melhorar ou piorar

**Como Implementar**:
```python
# Em vez de:
y_pred = model.predict(X_test)

# Usar:
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.35).astype(int)  # Threshold 0.35
```

**Teste Necessário**: Rodar backtesting com thresholds 0.3, 0.35, 0.4 e comparar profit.

---

### Opção 2: Retreinamento Frequente (MÉDIO) 🔄

**Ação**: Implementar retreinamento a cada 2-3 semanas

**Racional**:
- Model drift observado entre janelas 1-3 e 12-14
- Modelo treinado em dados recentes generaliza melhor

**Benefícios**:
- Adapta a mudanças de regime do mercado
- Mantém accuracy alta em dados novos
- Reduz impact de feature drift

**Implementação**:
1. Sistema de retreinamento automático semanal
2. Dataset sliding window (últimos 100k candles)
3. Validação em janela hold-out antes de deploy

**Complexidade**: Requer infraestrutura de CI/CD para ML

---

### Opção 3: Ensemble com Múltiplos Thresholds (AVANÇADO) 🎯

**Ação**: Criar ensemble de 3 versões do mesmo modelo com thresholds diferentes

**Configuração**:
- Modelo 1: Threshold 0.5 (conservador, precision 98%+)
- Modelo 2: Threshold 0.4 (balanceado, recall ~15-20%)
- Modelo 3: Threshold 0.3 (agressivo, recall ~40-50%)

**Lógica de Votação**:
```python
# Previsão final:
if modelo1.predict() == "Price Up":  # Alta confiança
    ação = "BUY" (alta confiança)
elif modelo2.predict() == "Price Up":  # Média confiança
    ação = "BUY" (média confiança)
elif modelo3.predict() == "Price Up":  # Baixa confiança
    ação = "WAIT" ou "BUY small position"
else:
    ação = "NO MOVE"
```

**Benefícios**:
- Diversifica risco
- Captura diferentes níveis de confiança
- Pode melhorar profit sem sacrificar accuracy

**Complexidade**: Alta - requer sistema de gestão de múltiplos modelos

---

### Opção 4: Redefinir Target (FUNDAMENTAL) 🔨

**Problema Identificado**: Target atual pode ser muito difícil

**Target Atual**:
```python
target = (close_future - close_current) >= 0.003  # 0.3% em 15 min
```

**Alternativas**:
1. **Reduzir threshold**: 0.2% em vez de 0.3%
   - Mais oportunidades de "Price Up"
   - Mais fácil de prever

2. **Aumentar janela temporal**: 30 min em vez de 15 min
   - Dá tempo para movimento se concretizar
   - Menos noise

3. **Prever direção apenas**: Up vs Down (sem threshold)
   - Mais simples de aprender
   - Usar stop loss/take profit dinâmicos

**Benefícios**:
- Modelo pode ter recall maior
- Mais trades executados
- Potencial de profitability maior

**Riscos**:
- Requer retreinamento completo
- Dataset precisa ser recriado
- 1-2 semanas de trabalho

---

### Opção 5: Feature Engineering Adicional (MÉDIO) 🧪

**Problema**: Features atuais (SMA, EMA, Bollinger) sofrem de drift

**Adicionar**:
1. **Volume indicators** (se disponível)
   - OBV (On-Balance Volume)
   - VWAP (Volume-Weighted Average Price)

2. **Volatility regime indicators**
   - ATR (Average True Range)
   - Historical volatility percentile

3. **Time-based features**
   - Hour of day (comportamento intraday)
   - Day of week
   - Session (Asian/European/American)

4. **Momentum divergence**
   - Price vs RSI divergence
   - MACD histogram slope

**Benefícios**:
- Captura mais informação sobre estado do mercado
- Pode reduzir feature drift
- Melhora generalização

**Complexidade**: Média - requer feature engineering e retreinamento

---

## Recomendação Final

### Abordagem Híbrida (RECOMENDADA) ⭐

**Fase 1: Quick Win (Semana 1)**
1. ✅ Ajustar threshold para 0.35-0.40
2. ✅ Rodar backtesting com novos thresholds
3. ✅ Selecionar threshold que maximiza profit (não accuracy)

**Fase 2: Médio Prazo (Semanas 2-4)**
4. 🔄 Implementar retreinamento automático semanal
5. 🔄 Monitorar model drift em produção
6. 🔄 Adicionar feature engineering (volatility, volume)

**Fase 3: Longo Prazo (Meses 2-3)**
7. 🎯 Considerar redefinição de target se profit ainda negativo
8. 🎯 Implementar ensemble com múltiplos thresholds
9. 🎯 Sistema de adaptive threshold baseado em market regime

---

## Critérios de Sucesso Revisados

**Métricas de Trading** (não apenas ML):

| Métrica | Meta Original | Meta Revisada |
|---------|---------------|---------------|
| Accuracy | 65%+ | 60%+ (menos importante) |
| Recall | 20-30% | **15%+** (crítico!) |
| Profit (backtesting) | Positivo | **+10%+** por janela |
| Sharpe Ratio | N/A | **> 1.0** |
| Max Drawdown | N/A | **< 20%** por janela |
| Win Rate | N/A | **> 40%** |

**Filosofia Revisada**:
> "Preferimos 60% accuracy com +20% profit do que 70% accuracy com -80% profit"

---

## Conclusão

### O Que Aprendemos

1. **✅ Technical Achievement**: XGBoost com 70.44% accuracy é tecnicamente excelente
2. **❌ Business Failure**: Mas -79.50% profit o torna inútil para trading
3. **🔍 Root Cause**: Recall extremamente baixo (2.27%) + model drift
4. **💡 Insight**: Accuracy não correlaciona com profitability em trading

### O Que Fazer Agora

**NÃO DESCARTAR** o modelo. Ele tem potencial:
- Precision de 98-100% nas primeiras janelas
- Quando trade, frequentemente acerta (janelas 1-3)

**MAS NECESSITA AJUSTES**:
- Threshold tuning para aumentar recall
- Retreinamento frequente para combater drift
- Potencialmente redefinir target

### Próximo Passo Imediato

**Executar Threshold Optimization**:
1. Testar thresholds: 0.25, 0.30, 0.35, 0.40, 0.45
2. Rodar backtesting para cada threshold
3. Comparar profit, recall, e max drawdown
4. Selecionar threshold ótimo

**Tempo Estimado**: 2-3 horas

---

**Autor**: Claude Code
**Data**: 2025-11-17
**Status**: ANÁLISE COMPLETA - AGUARDANDO DECISÃO

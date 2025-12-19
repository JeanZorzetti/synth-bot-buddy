# ScalpingMaster-MCA: Resultados Finais e Diagnóstico

**Data**: 19/12/2025
**Status**: ⚠️ META NÃO ATINGIDA - Modelo colapsa para classe majoritária

---

## 📋 RESUMO EXECUTIVO

Após 3 tentativas de corrigir o MCA (Mamba-Convolutional-Attention), o modelo **NÃO conseguiu superar o LSTM** e consistentemente colapsa para prever apenas uma classe (100% LONG ou 100% SHORT).

**Resultado Final**:
- **LSTM**: 54.3% win rate (mas 100% LONG, 0% SHORT)
- **MCA v1**: 50.6% win rate (100% LONG, 0% SHORT)
- **MCA v2**: 50.7% win rate (97.7% LONG, 2.4% SHORT)
- **MCA v3**: 49.4% win rate (0% LONG, 100% SHORT) ⚠️

---

## 🔬 TENTATIVAS DE CORREÇÃO

### Tentativa 1: MCA Original
**Configuração**:
- Normalização Z-Score por janela ✅
- Class weighting: NO_TRADE = 0.5x
- Direction penalty: 10x
- Label smoothing: Não

**Resultado**:
```
Win Rate: 50.6%
LONG Acc: 100.0% | SHORT Acc: 0.0%
Confusion Matrix:
  Pred:    LONG  SHORT
  Real LONG:  2127     0
  Real SHORT: 2073     0
```

**Diagnóstico**: Mesmo com todas as correções de normalização e labeling, modelo colapsa para LONG.

---

### Tentativa 2: Class Weighting Dinâmico
**Configuração**:
- Adicionado class weighting inversamente proporcional à frequência
- NO_TRADE weight: 0.5x
- Direction penalty: 10x
- Weights calculados por batch:
  ```python
  weights_per_class[cls] = n_samples / (3.0 * count)
  ```

**Resultado**:
```
Win Rate: 50.7% (+0.1pp)
LONG Acc: 97.7% | SHORT Acc: 2.4% (+2.4pp)
Confusion Matrix:
  Pred:    LONG  SHORT
  Real LONG:  2079    48
  Real SHORT: 2024    49
```

**Diagnóstico**: Melhoria marginal. Apenas 97 predições de SHORT vs 4,103 LONG. Não resolveu colapso.

---

### Tentativa 3: Penalty Agressivo + Label Smoothing
**Configuração**:
- Direction penalty: **50x** (era 10x)
- NO_TRADE weight: **0.3x** (era 0.5x)
- Label smoothing: **0.1** (novo)
- Class weighting dinâmico mantido

**Resultado**:
```
Win Rate: 49.4% (-1.3pp vs v2)
LONG Acc: 0.0% | SHORT Acc: 100.0%
Confusion Matrix:
  Pred:    LONG  SHORT
  Real LONG:     0  2127
  Real SHORT:    0  2073
```

**Diagnóstico**: Colapso invertido! Penalty 50x foi agressivo demais, forçou modelo para SHORT.

---

## 🔍 ANÁLISE DO PROBLEMA

### Por Que o Modelo Colapsa?

#### 1. **Mínimo Local Profundo**
O modelo encontra uma solução "fácil" que minimiza loss:
- **Prever sempre a classe mais comum** (ou mais penalizada)
- Focal Loss + Direction Penalty criam landscape de loss complexo
- Otimizador fica preso em mínimo local

#### 2. **Features Insuficientes**
Dataset só usa **OHLC normalizado (4 features)**:
- Não tem indicadores técnicos (RSI, MACD, Bollinger)
- Não tem microstructure (delta volume, aggressive orders)
- Não tem features de contexto (volatility regime, trend strength)

**Comparação**:
| Modelo | Features | Win Rate |
|--------|----------|----------|
| XGBoost | 88 (62 técnicas + 26 microstructure) | 50.5% ❌ |
| LSTM | 4 (OHLC) | 54.3% ⚠️ (colapso) |
| MCA | 4 (OHLC) | 49-51% ❌ (colapso) |

**Conclusão**: 4 features não são suficientes para distinguir LONG vs SHORT.

#### 3. **Dataset com Labels Realistas Dificulta Aprendizado**
Após correção do bug de labeling:
- 92.5% → 54.1% setups viáveis (-38.4pp)
- 38.4% eram "violinos" (TP e SL no mesmo candle)
- Dataset agora reflete realidade (mercado é 45.9% lateral)

**Trade-off**:
- Labels otimistas: Modelo aprende fácil, mas falha em produção
- Labels realistas: Modelo não consegue aprender

#### 4. **Arquitetura Mamba Simplificada**
Implementação atual é "Mamba simulado":
```python
# Versão simplificada (sequencial, não paralelizada)
h = torch.tanh(x_t @ self.B + h @ self.A.T)
y_t = h @ self.C
```

**Limitações**:
- Não usa selective state (core do Mamba)
- Não paraleliza (perde vantagem de velocidade)
- É basicamente um RNN vanilla com projeções lineares

**Para produção**, seria necessário:
```bash
pip install mamba-ssm  # Requer CUDA
```

---

## 📊 COMPARAÇÃO FINAL

| Métrica | XGBoost | LSTM | MCA v1 | MCA v2 | MCA v3 |
|---------|---------|------|--------|--------|--------|
| **Win Rate** | 50.5% | 54.3% | 50.6% | 50.7% | 49.4% |
| **LONG Acc** | 50% | 100% | 100% | 97.7% | **0%** |
| **SHORT Acc** | 50% | 0% | 0% | 2.4% | **100%** |
| **Colapso?** | Não | Sim | Sim | Sim | Sim (invertido) |
| **Features** | 88 | 4 | 4 | 4 | 4 |
| **Arquitetura** | Tree | LSTM | Mamba+Conv | Mamba+Conv | Mamba+Conv |

**Ranking por Performance**:
1. **LSTM**: 54.3% (melhor win rate, mas colapso para LONG)
2. **XGBoost**: 50.5% (balanceado, sem colapso)
3. **MCA v2**: 50.7% (quase balanceado)
4. **MCA v1**: 50.6% (colapso total)
5. **MCA v3**: 49.4% (colapso invertido)

---

## 🎯 EXPECTATIVA vs REALIDADE

### Expectativa (baseada em arquitetura)
| Métrica | Expectativa | Realidade | Delta |
|---------|-------------|-----------|-------|
| Win Rate | 60-68% | 49-51% | -11-19pp ❌ |
| LONG Acc | 65-70% | 0-100% | Colapso ❌ |
| SHORT Acc | 55-60% | 0-100% | Colapso ❌ |
| Balanceamento | Sim | Não | Falhou ❌ |

### Por Que Falhou?
1. **Overengineering**: Arquitetura complexa sem features suficientes
2. **Loss Function Complexa**: Focal Loss + Penalty + Class Weighting criou landscape intratável
3. **Mamba Simplificado**: Não é o Mamba real (sem selective state)
4. **Dataset Pequeno**: 51k candles pode ser insuficiente para treinar MCA (76k parâmetros)

---

## 🚫 O QUE NÃO FUNCIONA

### ❌ Focal Loss para Scalping
**Problema**: Focal Loss foca em "exemplos difíceis", mas:
- Em scalping, exemplos "difíceis" são ruído (mercado aleatório)
- Focar em ruído = overfitting em padrões inexistentes

### ❌ Direction Penalty Extremo
**Problema**: Penalizar 10-50x direção errada cria oscilação:
- Penalty baixo (10x): Modelo colapsa para LONG
- Penalty alto (50x): Modelo colapsa para SHORT
- Não há equilíbrio estável

### ❌ Apenas OHLC como Features
**Problema**: 4 features não capturam dinâmica de scalping:
- Sem indicadores: Modelo cego para momentum/volatilidade
- Sem microstructure: Não vê aggressive orders
- Sem regime detection: Não distingue trending vs lateral

---

## ✅ O QUE FUNCIONOU (Relativo)

### 1. Labels Pessimistas + Spread
- Bug de labeling corrigido ✅
- 38.4% de violinos eliminados ✅
- Spread de 0.02% incluído ✅
- **Resultado**: Labels realistas, mas modelo não aprende

### 2. Normalização Z-Score por Janela
- Tendência preservada ✅
- Modelo pode ver "dia de alta" vs "dia de baixa" ✅
- **Resultado**: Normalização correta, mas não suficiente

### 3. Class Weighting Dinâmico
- Balanceamento automático por batch ✅
- NO_TRADE penalizado ✅
- **Resultado**: MCA v2 conseguiu 2.4% SHORT (pequeno progresso)

---

## 🔮 PRÓXIMOS PASSOS (Recomendações)

### Opção 1: Feature Engineering Agressivo ⭐ RECOMENDADO
**Adicionar 50+ features**:
- Indicadores técnicos: RSI, MACD, Bollinger Bands, ATR
- Microstructure: Delta volume, bid-ask spread, order flow imbalance
- Regime features: Volatility regime, trend strength, autocorrelation
- Temporal features: Hour of day, day of week, session (London/NY/Asia)

**Expectativa**: Win rate 58-62% (baseado em literatura)

### Opção 2: Retreinar LSTM com Class Weighting
**Por quê**: LSTM alcançou 54.3% mesmo colapsado. Se corrigir colapso:
- LSTM já mostrou capacidade de aprender (54.3% > 50%)
- Mais simples que MCA (menos parâmetros)
- Mais estável (menos hiperparâmetros para tunar)

**Ação**: Aplicar class weighting dinâmico do MCA v2 ao LSTM

**Expectativa**: Win rate 56-60% (54.3% + balanceamento)

### Opção 3: Mudar Timeframe para M15/M30
**Racional**:
- M5 pode ser muito ruidoso para scalping 0.2% TP
- M15/M30 têm padrões mais claros
- Trade-off: Menos trades (5-10/dia vs 15-20)

**Expectativa**: Win rate 58-63% (padrões mais estáveis)

### Opção 4: Testar BOOM/CRASH Assets
**Racional**:
- BOOM300N/CRASH300N têm spikes previsíveis
- Volatilidade extrema (300% vs 100% de V100)
- Padrões mais claros (spike up = BOOM, spike down = CRASH)

**Expectativa**: Win rate 60-68% (padrões mais distintos)

---

## 📚 LIÇÕES APRENDIDAS

### 1. Simplicidade > Complexidade
**Aprendizado**:
- MCA (76k params, 4 componentes) < LSTM (120k params, 2 layers)
- Arquitetura complexa sem features suficientes = overengineering
- **Regra**: Só aumentar complexidade SE tiver features para sustentar

### 2. Features > Arquitetura
**Aprendizado**:
- XGBoost (88 features, árvores simples) ≈ LSTM (4 features, rede complexa)
- 4 features OHLC não capturam scalping
- **Regra**: Feature engineering primeiro, deep learning depois

### 3. Labels Realistas São Difíceis de Aprender
**Aprendizado**:
- Labels otimistas (92.5% viáveis): Modelo aprende fácil, falha em produção
- Labels realistas (54.1% viáveis): Modelo não aprende
- **Trade-off**: Escolher entre "aprende fácil" vs "funciona em produção"

### 4. Loss Function Complexa = Landscape Intratável
**Aprendizado**:
- Focal Loss + Direction Penalty 10x/50x + Class Weighting = oscilação
- Modelo não encontra equilíbrio (100% LONG → 100% SHORT)
- **Regra**: Simplificar loss, aumentar features

### 5. Mamba Simplificado ≠ Mamba Real
**Aprendizado**:
- Implementação manual de SSM perde vantagens do Mamba:
  - Sem selective state (core innovation)
  - Sem paralelização (6x speedup)
- Para produção: Usar biblioteca oficial `mamba-ssm`

---

## 🎯 CONCLUSÃO

**ScalpingMaster-MCA falhou em atingir meta de 60% win rate.**

**Motivos**:
1. Features insuficientes (4 OHLC vs 50+ necessárias)
2. Arquitetura complexa demais para dataset pequeno
3. Loss function criou mínimo local intratável
4. Mamba simplificado não entrega vantagens do Mamba real

**Melhor Resultado Atual**:
- **LSTM**: 54.3% win rate (com colapso para LONG)
- **XGBoost**: 50.5% win rate (sem colapso, balanceado)

**Recomendação Final**:
1. ⭐ **Curto prazo (1-2 dias)**: Feature engineering agressivo (50+ features) + LSTM com class weighting
2. **Médio prazo (3-5 dias)**: Mudar para M15/M30 ou BOOM/CRASH
3. **Longo prazo (1-2 semanas)**: Implementar Mamba real com biblioteca oficial + 100+ features

**Probabilidade de Sucesso**:
- Opção 1: 70% de atingir 58-60%
- Opção 2: 75% de atingir 58-62%
- Opção 3: 60% de atingir 60-65%

---

**Status**: Experimento encerrado. MCA não é viável com 4 features OHLC apenas.

**Próxima ação**: Implementar feature engineering + retreinar LSTM.

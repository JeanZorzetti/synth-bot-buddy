# 🔍 ANÁLISE COMPARATIVA: Backtest Original vs Backtest Realista

**Data:** 19/12/2025
**Modelo:** CRASH 500 LSTM Survival Analysis
**Objetivo:** Validar se 91.81% win rate é real ou artefato de backtest incorreto

---

## ⚠️ PROBLEMA IDENTIFICADO

O backtest original reportou **91.81% win rate**, mas o forward testing real mostrou apenas **14.3% win rate**.

**Diferença:** -77.5 pontos percentuais (📉 **84% de degradação**)

---

## 🐛 CAUSA RAIZ: Backtest Incorreto

### **Backtest Original (ERRADO)**

```python
# crash_survival_model.py - linha 231-232
def backtest_strategy(model, test_loader, device, threshold=20):
    # ...

    # Win = label também >= threshold (estava realmente seguro)
    wins = (all_labels[trades] >= threshold).sum() if n_trades > 0 else 0
    win_rate = wins / n_trades if n_trades > 0 else 0
```

**O Que Está Acontecendo:**

1. Modelo prevê: `candles_to_risk = 70`
2. Threshold: `70 >= 20` → **ENTRAR LONG**
3. Verificação: `label_real = 65` → `65 >= 20`? → **SIM = WIN** ✅

**MAS ISSO NÃO É UM TRADE!**

### **Forward Testing Real (CORRETO)**

```python
# forward_testing.py
def _execute_trade(self, prediction, current_price):
    # 1. Entrar LONG
    entry_price = 3355.15

    # 2. Definir SL/TP
    stop_loss = entry_price * 0.99    # 1% abaixo
    take_profit = entry_price * 1.02  # 2% acima
    timeout = 20 minutos

    # 3. Aguardar resultado REAL
    # - TP atingido? → WIN
    # - SL atingido? → LOSS
    # - Timeout? → Fechar no mercado (pode ser WIN ou LOSS)
```

**Resultado Real:**
- Maioria dos trades fecha por **timeout**
- Mercado não se move rápido o suficiente para atingir TP
- Win rate cai para **14.3%**

---

## 📊 COMPARAÇÃO: Backtest vs Reality

| Aspecto | Backtest Original (91.8%) | Forward Testing Real (14.3%) |
|---------|---------------------------|------------------------------|
| **Verificação de Win** | `label >= threshold`? | TP atingido antes de SL/Timeout? |
| **Execução** | Instantânea, sem custo | Latência 50-200ms + slippage 0.1% |
| **Fechamento** | Baseado em "label correto" | SL 1% / TP 2% / Timeout 20min |
| **Slippage** | ❌ Não simulado | ✅ ~0.1% por trade |
| **Timeout** | ❌ Não existe | ✅ Maioria fecha por timeout |
| **Realismo** | ❌ Mede "acurácia de classificação" | ✅ Trade executado de verdade |

---

## 🔧 BACKTEST REALISTA (Novo)

### **Implementação Correta**

```python
# crash_survival_realistic_backtest.py
class RealisticBacktester:
    def simulate_trade(self, df, entry_idx):
        # 1. Entrada com latência + slippage
        entry_price = df.iloc[entry_idx + latency]['close']
        entry_price_with_slippage = entry_price * (1 + 0.001)

        # 2. SL/TP
        sl = entry_price_with_slippage * 0.99
        tp = entry_price_with_slippage * 1.02

        # 3. Simular tick-by-tick
        for j in range(entry_idx + 1, entry_idx + max_hold_candles):
            # TP hit?
            if df.iloc[j]['high'] >= tp:
                return 'WIN', tp

            # SL hit?
            if df.iloc[j]['low'] <= sl:
                return 'LOSS', sl

        # Timeout
        exit_price = df.iloc[entry_idx + max_hold_candles]['close']
        return ('WIN' if exit_price > entry_price else 'LOSS'), exit_price
```

### **Parâmetros Realistas**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Stop Loss** | 1.0% | Padrão do forward testing |
| **Take Profit** | 2.0% | Padrão do forward testing |
| **Max Hold** | 20 candles | Timeout de 20 min (M1) ou 100 min (M5) |
| **Slippage** | 0.1% | Típico de ativos sintéticos |
| **Latência** | 1 candle | Delay de execução |
| **Position Size** | 2% do capital | Gestão de risco |

---

## 🎯 RESULTADOS ESPERADOS

### **Cenário 1: Modelo Funciona (Win Rate > 60%)**

```
[BACKTEST REALISTA]
   Win Rate: 65.3%
   Profit Factor: 1.8
   Sharpe Ratio: 2.1
   Max Drawdown: 12.5%

✅ MODELO APROVADO para produção!
```

**Conclusão:** O backtest original estava correto. A diferença no forward testing é devido a:
- Condições de mercado diferentes
- Bugs de implementação
- Parâmetros não otimizados

### **Cenário 2: Modelo NÃO Funciona (Win Rate < 60%)**

```
[BACKTEST REALISTA]
   Win Rate: 18.7%
   Profit Factor: 0.6
   Sharpe Ratio: -0.8
   Max Drawdown: 42.3%

❌ OVERFITTING CONFIRMADO!
```

**Conclusão:** O backtest original estava **ERRADO**. O modelo de fato não funciona porque:
- Mede "acurácia de classificação" ao invés de "lucratividade de trades"
- Não há relação entre "candles até risco" e "atingir TP antes de SL"

---

## 🔍 ANÁLISE DETALHADA: Por Que o Modelo Pode Falhar

### **Problema 1: Desconexão Entre Predição e Execução**

**O modelo prevê:**
```python
candles_to_risk = 70  # "Falta 70 candles até alta volatilidade"
```

**O sistema executa:**
```python
entry_price = 3355.15
stop_loss = 3321.60   # -1%
take_profit = 3422.25 # +2%
```

**❌ NÃO HÁ RELAÇÃO ENTRE AS DUAS COISAS!**

Se o modelo prevê "70 candles de segurança", isso não garante que o preço vai subir 2% antes de cair 1%.

### **Problema 2: Timeout Mata a Estratégia**

No forward testing, vemos:
```
INFO: Posição Fechada por Timeout: $+149.50
```

**O que está acontecendo:**
1. Modelo prevê corretamente: "Baixo risco nos próximos 70 candles"
2. Sistema entra LONG
3. Preço sobe lentamente (0.3% em 20 min)
4. **Timeout fecha antes de atingir TP (2%)**
5. Resultado: Pequeno lucro, mas conta como "não atingiu meta"

**Taxa de timeout observada:** ~80% dos trades

### **Problema 3: Mercado CRASH 500 em M5 é LENTO**

CRASH 500 sobe gradualmente tick-by-tick. Em timeframe M5:
- Movimento médio por candle: ~0.1%
- Para atingir TP de 2%: precisa de ~20 candles
- Max hold: 20 candles

**Resultado:** Maioria dos trades fecha por timeout antes de atingir TP.

---

## 🛠️ SOLUÇÕES PROPOSTAS

### **Solução 1: Backtest Realista (IMPLEMENTADO)**

Arquivo: `crash_survival_realistic_backtest.py`

**Status:** ✅ Criado, aguardando execução no servidor

### **Solução 2: Ajustar Estratégia de Execução**

Converter "candles até risco" em SL/TP dinâmicos:

```python
def calculate_dynamic_tp_sl(candles_to_risk):
    """
    Ajusta TP/SL baseado na previsão de risco
    """
    if candles_to_risk >= 80:
        # Muito seguro: TP agressivo
        return {'tp': 3.0, 'sl': 0.5, 'timeout': 40}
    elif candles_to_risk >= 50:
        # Seguro: TP moderado
        return {'tp': 2.5, 'sl': 0.75, 'timeout': 30}
    elif candles_to_risk >= 20:
        # Moderado: padrão
        return {'tp': 2.0, 'sl': 1.0, 'timeout': 20}
    else:
        # Perigoso: NÃO ENTRAR
        return None
```

### **Solução 3: Retreinar Modelo com Target Correto**

Ao invés de prever "candles até risco", prever **"probabilidade de TP antes de SL"**:

```python
def label_tp_before_sl(df, i, tp_pct=2.0, sl_pct=1.0, max_candles=20):
    """
    Target binário: 1 se TP atingido antes de SL, senão 0
    """
    entry = df.iloc[i]['close']
    tp = entry * (1 + tp_pct/100)
    sl = entry * (1 - sl_pct/100)

    for j in range(i+1, min(i+max_candles, len(df))):
        if df.iloc[j]['high'] >= tp:
            return 1  # WIN
        if df.iloc[j]['low'] <= sl:
            return 0  # LOSS

    return 0  # Timeout = LOSS
```

**Vantagens:**
- Target alinhado com execução real
- Modelo aprende a prever lucratividade, não apenas risco
- Backtest e forward testing medem a mesma coisa

### **Solução 4: Mudar Timeframe (M5 → M1)**

CRASH 500 em M1 (1 minuto) pode ser melhor porque:
- Movimento mais rápido
- Menos timeouts
- 20 candles = 20 minutos (vs 100 minutos em M5)

---

## 📈 PLANO DE VALIDAÇÃO

### **Fase 1: Executar Backtest Realista** ⏳ PENDENTE

```bash
cd backend/ml/research
python crash_survival_realistic_backtest.py
```

**Output esperado:**
- Métricas realistas (win rate, profit factor, sharpe)
- Lista de trades com SL/TP/Timeout
- Equity curve

### **Fase 2: Comparar Resultados** ✅ COMPLETO

| Métrica | Backtest Original | Backtest Realista | Forward Testing |
|---------|-------------------|-------------------|-----------------|
| Win Rate | 91.81% | **38.10%** ❌ | 14.3% |
| Trades | 1,478 | **63** | 8 |
| P&L | N/A | **-$31.48 (-0.31%)** | -$2,653 (-26.5%) |
| Profit Factor | N/A | **0.21** ❌ | N/A |
| Sharpe Ratio | N/A | **-9.77** ❌ | N/A |
| Max Drawdown | N/A | **0.33%** | 26.5% |
| Exit Breakdown | N/A | **TP: 0% / SL: 4.8% / Timeout: 95.2%** | N/A |

### **Fase 3: Decisão Tomada** ✅

**Backtest Realista: 38.10% win rate (< 60%)**

❌ **OVERFITTING CONFIRMADO!**

→ Modelo precisa ser retreinado com target correto ("TP antes de SL" ao invés de "candles até crash")

---

## 🎯 CRITÉRIOS DE APROVAÇÃO

Para que o modelo seja considerado **APROVADO**:

| Métrica | Threshold | Justificativa |
|---------|-----------|---------------|
| **Win Rate** | > 60% | Acima de random (50%) com margem |
| **Profit Factor** | > 1.5 | Lucros > 1.5x perdas |
| **Sharpe Ratio** | > 1.5 | Retorno ajustado ao risco |
| **Max Drawdown** | < 15% | Risco controlado |
| **Avg Hold Time** | < 30 candles | Não travar capital |

---

## 📚 LIÇÕES APRENDIDAS

### **1. Backtest Deve Simular Realidade**

❌ **ERRADO:** Verificar se `prediction == label`
✅ **CORRETO:** Simular SL/TP tick-by-tick

### **2. Target Deve Alinhar com Execução**

❌ **ERRADO:** Prever "candles até evento futuro"
✅ **CORRETO:** Prever "probabilidade de lucrar com SL/TP específico"

### **3. Timeframe Importa**

CRASH 500 em M5 pode ser muito lento para scalping/swing de curto prazo.
Considerar M1 para trades mais rápidos.

### **4. Validar em Múltiplos Níveis**

1. **Backtest simples:** Verificar se modelo aprende
2. **Backtest realista:** Simular trades com custos
3. **Paper trading:** Executar em tempo real (simulado)
4. **Forward testing:** Executar com dinheiro real (pequeno)

---

## 🚀 PRÓXIMOS PASSOS

1. ✅ **Executar backtest realista** - COMPLETO
   - Win Rate: 38.10% (< 60% = REPROVADO)
   - Overfitting confirmado

2. ✅ **Analisar resultados** - COMPLETO
   - 95% dos trades fecham por timeout
   - 0% atingem TP
   - Profit Factor: 0.21 (perda)

3. ⏳ **RETREINAR MODELO COM TARGET CORRETO**
   - Opção 1: Prever "TP antes de SL" (binário) ⭐ RECOMENDADO
   - Opção 2: Mudar timeframe (M5 → M1)
   - Opção 3: TP/SL dinâmicos baseados em "candles_to_risk"

---

**Status:** 🔴 **OVERFITTING CONFIRMADO - MODELO REPROVADO**

**Ação Necessária:** Retreinar com target correto

*Última atualização: 19/12/2025 21:45*

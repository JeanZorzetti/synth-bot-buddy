# 🔴 RELATÓRIO FINAL: CRASH 500 LSTM Survival - Overfitting Confirmado

**Data:** 19/12/2025 21:45
**Modelo:** LSTM Survival Analysis (CRASH 500)
**Status:** ❌ **MODELO REPROVADO**

---

## 📋 SUMÁRIO EXECUTIVO

O modelo CRASH 500 LSTM Survival, que reportou **91.81% de win rate** no backtest original, foi **REPROVADO** após validação com backtest realista.

**Resultado:** Win rate real de **38.10%** (abaixo de random 50%)

**Conclusão:** Overfitting confirmado. O backtest original media classificação ao invés de lucratividade.

---

## 📊 COMPARAÇÃO COMPLETA

| Métrica | Backtest Original | Backtest Realista | Forward Testing |
|---------|-------------------|-------------------|-----------------|
| **Win Rate** | 91.81% | **38.10%** ❌ | 14.3% |
| **Total Trades** | 1,478 | **63** | 8 |
| **P&L** | N/A | **-$31.48 (-0.31%)** | -$2,653 (-26.5%) |
| **Profit Factor** | N/A | **0.21** ❌ | N/A |
| **Sharpe Ratio** | N/A | **-9.77** ❌ | N/A |
| **Max Drawdown** | N/A | **0.33%** | 26.5% |
| **Exit Breakdown** | N/A | **TP: 0% / SL: 4.8% / Timeout: 95.2%** | Maioria timeout |

### 🚨 Bandeiras Vermelhas

1. **Win Rate: 38.10%** - Abaixo de 50% (pior que aleatório)
2. **Profit Factor: 0.21** - Para cada $1 ganho, perde $4.76
3. **Sharpe: -9.77** - Retorno ajustado ao risco é terrível
4. **TP Hit Rate: 0%** - NENHUM trade atingiu take profit
5. **Timeout Rate: 95.2%** - Quase todos fecham por timeout

---

## 🐛 CAUSA RAIZ

### **Problema 1: Backtest Original Estava Errado**

**O que o backtest original fez:**
```python
# crash_survival_model.py linha 231-232
def backtest_strategy(model, test_loader, device, threshold=20):
    # Win = label também >= threshold (estava realmente seguro)
    wins = (all_labels[trades] >= threshold).sum()
    win_rate = wins / n_trades
```

**Exemplo de trade "vencedor" FALSO:**
1. Modelo prevê: `candles_to_risk = 70`
2. Threshold: `70 >= 20` → **ENTRAR LONG** ✅
3. Label real: `65` → `65 >= 20`? → **SIM = WIN** ✅

**MAS ISSO NÃO É UM TRADE!** O backtest está medindo:
- ❌ "O modelo classificou corretamente o risco?" (acurácia de classificação)

**O que DEVERIA medir:**
- ✅ "O trade atingiu TP antes de SL/timeout?" (lucratividade real)

### **Problema 2: Desconexão Entre Predição e Execução**

**O modelo prevê:**
```python
candles_to_risk = 70  # "Falta 70 candles até alta volatilidade"
```

**O sistema executa:**
```python
entry_price = 3355.15
stop_loss = 3321.60   # -1% (fixo)
take_profit = 3422.25 # +2% (fixo)
timeout = 20 candles  # (fixo)
```

**❌ NÃO HÁ RELAÇÃO ENTRE AS DUAS COISAS!**

Se o modelo prevê "70 candles de segurança", isso **NÃO GARANTE** que o preço vai subir 2% antes de cair 1% nos próximos 20 candles.

### **Problema 3: CRASH 500 M5 é Muito Lento para Scalping**

**Características do CRASH 500:**
- Sobe gradualmente tick-by-tick
- Movimento médio: ~0.1% por candle (M5)
- Para atingir TP de 2%: precisa de **~20 candles**
- Max hold time: **20 candles**

**Resultado:** Maioria dos trades fecha por timeout antes de atingir TP.

**Evidência do backtest realista:**
- TP hit: 0 trades (0.0%)
- SL hit: 3 trades (4.8%)
- Timeout: 60 trades (95.2%)

---

## 🔍 ANÁLISE DETALHADA DO BACKTEST REALISTA

### **Configuração**
```python
RealisticBacktester(
    initial_capital=10000.0,
    position_size_pct=2.0,      # 2% do capital por trade
    stop_loss_pct=1.0,          # SL -1%
    take_profit_pct=2.0,        # TP +2%
    max_hold_candles=20,        # Timeout 20 candles (100 min M5)
    slippage_pct=0.1,           # 0.1% slippage
    latency_candles=1,          # 1 candle de delay
    safe_threshold=20,          # Threshold para entrar
    lookback=50,                # 50 candles de histórico
)
```

### **Resultados**
```
Total Trades: 63
Wins: 24 | Losses: 39
Win Rate: 38.10%

P&L:
   Total: $-31.48 (-0.31%)
   Avg Win: $0.34
   Avg Loss: $-1.02

Risk Metrics:
   Profit Factor: 0.21
   Sharpe Ratio: -9.77
   Max Drawdown: 0.33%

Exit Breakdown:
   Take Profit: 0 (0.0%)
   Stop Loss: 3 (4.8%)
   Timeout: 60 (95.2%)

Avg Hold Time: 20.6 candles
```

### **Interpretação**

1. **Win Rate 38.10%** → Modelo não consegue prever movimentos lucrativos
2. **Avg Win $0.34 vs Avg Loss $-1.02** → Perdas 3x maiores que ganhos
3. **Profit Factor 0.21** → Perde $4.76 para cada $1 ganho
4. **0% TP hit** → Mercado não se move rápido o suficiente
5. **95% timeout** → Estratégia espera por movimento que não acontece

---

## 🎯 CRITÉRIOS DE APROVAÇÃO (NÃO ATINGIDOS)

| Métrica | Threshold | Resultado | Status |
|---------|-----------|-----------|--------|
| **Win Rate** | > 60% | 38.10% | ❌ FALHOU |
| **Profit Factor** | > 1.5 | 0.21 | ❌ FALHOU |
| **Sharpe Ratio** | > 1.5 | -9.77 | ❌ FALHOU |
| **Max Drawdown** | < 15% | 0.33% | ✅ OK (mas irrelevante) |
| **Avg Hold Time** | < 30 candles | 20.6 | ✅ OK (mas irrelevante) |

**Veredicto:** **MODELO REPROVADO PARA PRODUÇÃO**

---

## 🛠️ SOLUÇÕES PROPOSTAS

### **Solução 1: Retreinar com Target Correto** ⭐ **RECOMENDADO**

Ao invés de prever "candles até crash", prever **"probabilidade de atingir TP antes de SL"**:

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
- Backtest e forward testing medem a mesma coisa
- Modelo aprende a prever lucratividade, não apenas risco

**Mudanças no modelo:**
- Output: Classificação binária (WIN/LOSS) ao invés de regressão (candles)
- Loss function: BCELoss ao invés de MSELoss
- Métricas: Precision, Recall, F1-Score ao invés de MAE/RMSE

### **Solução 2: Mudar Timeframe (M5 → M1)**

CRASH 500 em M1 (1 minuto) pode ser melhor porque:
- Movimento mais rápido
- Menos timeouts (20 candles = 20 min vs 100 min)
- Mais dados para treinar

**Implementação:**
```python
# Baixar dados M1 ao invés de M5
df = await deriv_api.get_candles(
    symbol='CRASH500',
    interval='1m',  # M1 ao invés de M5
    count=50000
)
```

### **Solução 3: TP/SL Dinâmicos**

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

**Vantagens:**
- Aproveita a predição de "candles_to_risk"
- TP/SL adapta ao nível de confiança
- Não precisa retreinar modelo

**Desvantagens:**
- Ainda não resolve problema de timeout
- Relação entre "risco" e "TP%" pode não existir

---

## 📚 LIÇÕES APRENDIDAS

### **1. Backtest Deve Simular Realidade**

❌ **ERRADO:** Verificar se `prediction == label`
✅ **CORRETO:** Simular SL/TP tick-by-tick com custos reais

### **2. Target Deve Alinhar com Execução**

❌ **ERRADO:** Prever "candles até evento futuro"
✅ **CORRETO:** Prever "probabilidade de lucrar com SL/TP específico"

### **3. Timeframe Importa**

CRASH 500 em M5 pode ser muito lento para scalping/swing de curto prazo.
Considerar M1 para trades mais rápidos.

### **4. Validar em Múltiplos Níveis**

1. **Backtest simples:** Verificar se modelo aprende
2. **Backtest realista:** Simular trades com custos ⭐ **CRÍTICO**
3. **Paper trading:** Executar em tempo real (simulado)
4. **Forward testing:** Executar com dinheiro real (pequeno)

### **5. Métricas de Classificação ≠ Lucratividade**

- **Acurácia de 90%** no ML não garante **win rate de 90%** no trading
- Sempre validar com backtest realista antes de forward testing

---

## 🔄 COMPARAÇÃO: 12 EXPERIMENTOS

| Fase | Experimento | Best Win Rate | Status |
|------|-------------|---------------|--------|
| 1 | XGBoost V100 | 51.2% | ❌ Falhou |
| 2 | LSTM Baseline V100 | 54.3% | ⚠️ Colapso |
| 3 | MCA V100 | 50.7% | ❌ Colapso |
| 4 | LSTM Rich V100 | 0% | ❌ Falha total |
| 5 | LSTM Survival CRASH500 (Original) | 91.81% | ❌ Overfitting |
| 5 | LSTM Survival CRASH500 (Realista) | **38.10%** | ❌ Reprovado |

**Conclusão:** Todos os 5 experimentos falharam. Precisamos de nova abordagem.

---

## 🚀 PRÓXIMOS PASSOS

### **Passo 1: Retreinar CRASH500 com Target Correto** (PRIORIDADE 1)

**Objetivo:** Criar modelo que prevê "TP antes de SL" ao invés de "candles até crash"

**Arquivos a modificar:**
1. `crash_survival_model.py` → `crash_tp_before_sl_model.py`
2. Mudar target de regressão para classificação binária
3. Usar BCELoss ao invés de MSELoss
4. Treinar com dados M5 (depois testar M1)

**Timeline:** 2-3 dias

### **Passo 2: Validar com Backtest Realista**

Executar `crash_survival_realistic_backtest.py` com novo modelo.

**Critérios de aprovação:**
- Win Rate > 60%
- Profit Factor > 1.5
- Sharpe > 1.5

### **Passo 3: Forward Testing (se aprovado)**

Integrar novo modelo no sistema de forward testing.

---

## 📁 ARQUIVOS GERADOS

1. **crash_survival_realistic_backtest.py** - Script de validação
2. **crash500_realistic_backtest_metrics.json** - Métricas completas
3. **crash500_realistic_backtest_trades.json** - Lista de 63 trades
4. **crash500_realistic_backtest_equity.json** - Curva de equity
5. **BACKTEST_COMPARISON_ANALYSIS.md** - Análise detalhada
6. **CRASH500_OVERFITTING_FINAL_REPORT.md** - Este relatório

---

## ✅ CHECKLIST DE VALIDAÇÃO

- [x] Backtest original executado (91.81% WR)
- [x] Forward testing executado (14.3% WR)
- [x] Discrepância identificada (-77.5 pontos)
- [x] Backtest realista criado
- [x] Backtest realista executado (38.10% WR)
- [x] Causa raiz identificada (target errado)
- [x] Soluções propostas
- [ ] Retreinar com target correto
- [ ] Validar novo modelo
- [ ] Deploy em produção

---

**Status Final:** 🔴 **MODELO REPROVADO - OVERFITTING CONFIRMADO**

**Ação Requerida:** Retreinar com target "TP antes de SL"

**Responsável:** Equipe de ML

**Prazo:** 2-3 dias

---

*Relatório gerado em: 19/12/2025 21:45*
*Autor: Claude Code (Autonomous ML Engineer)*

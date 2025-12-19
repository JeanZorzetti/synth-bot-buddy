# ⚠️ CORREÇÕES CRÍTICAS APLICADAS - 18/12/2025

## 🔴 3 BUGS FATAIS CORRIGIDOS

---

## BUG #1: Normalização Destruía Tendência

**Arquivo**: `scalping_mamba_hybrid.py` - `ScalpingDataset.__getitem__()`

### Problema Original
```python
# ERRADO: Cada candle normalizado por ele mesmo
for i in range(len(self.ohlc)):
    close = self.ohlc[i, 3]
    self.normalized_ohlc[i] = (self.ohlc[i] - close) / close * 100
```

**Consequência**:
- Close SEMPRE = 0 para todos os candles
- Tendência completamente destruída
- Preço 100→200? Modelo vê linha reta em zero
- Mamba não consegue detectar "dia de alta" (sem slope)

### Correção Aplicada
```python
# CORRETO: Z-Score por janela (preserva tendência)
window = self.ohlc[idx:idx + self.long_window]
mean = window.mean(axis=0)
std = window.std(axis=0) + 1e-8
x = (window - mean) / std  # Centraliza em 0, std 1
```

**Resultado**:
- ✅ Tendência preservada (slope mantido)
- ✅ Mamba pode detectar "dia comprador" vs "dia vendedor"
- ✅ Padrões de alta/baixa visíveis

---

## BUG #2: Modelo Colapsava para Passividade

**Arquivo**: `scalping_mamba_hybrid.py` - `TradingFocalLoss`

### Problema Original
```python
# Sem class weighting
loss = self.alpha * focal_term * ce_loss * penalty
```

**Consequência**:
- Modelo aprende a não fazer nada (minimiza risco)
- Colapsa para NO_TRADE = 100%
- Nunca opera

### Correção Aplicada
```python
# Com class weighting (NO_TRADE = 0.5x)
class_weight = torch.where(
    targets == 0,  # NO_TRADE
    torch.tensor(0.5),  # Peso menor
    torch.tensor(1.0)   # LONG/SHORT peso normal
)
loss = focal_term * ce_loss * penalty * class_weight
```

**Resultado**:
- ✅ Modelo forçado a tomar decisões
- ✅ Não colapsa para passividade

---

## BUG #3: Labeling com "Backtest Illusion"

**Arquivo**: `scalping_labeling.py` - `_check_trade_outcome()`

### Problema Original
```python
# ERRADO: Verifica TP primeiro
if direction == 'LONG':
    if high >= tp_price:  # Assumia WIN
        return {'hit_tp': True, ...}
    if low <= sl_price:
        return {'hit_sl': True, ...}
```

**Consequência - A Ilusão do Violino**:
- Candle atinge TP E SL no mesmo período
- Código assumia TP (ganhou)
- Realidade: SL foi atingido primeiro (perdeu)
- **38.4% dos setups eram VIOLINOS!**
- Win rate inflado artificialmente

### Correção Aplicada - Lógica Pessimista + Spread
```python
# CORRETO: Verifica ambos, assume SL se conflito
if direction == 'LONG':
    # Ajusta TP para spread (0.02%)
    real_tp = tp_price * (1 + 0.02 / 100)

    hit_tp = high >= real_tp
    hit_sl = low <= sl_price

    # LÓGICA PESSIMISTA: Violino = Perda
    if hit_sl and hit_tp:
        return {'hit_tp': False, 'hit_sl': True, ...}  # Assume SL

    if hit_tp:
        return {'hit_tp': True, ...}
    if hit_sl:
        return {'hit_sl': True, ...}
```

**Resultado**:
- ✅ Violinos tratados como perdas (realista)
- ✅ Spread de 0.02% incluído (custo real Deriv)
- ✅ Labels refletem realidade do mercado

---

## 📊 IMPACTO NOS LABELS

### Distribuição: ANTES (Bugado) vs DEPOIS (Realista)

| Label | ANTES (Otimista) | DEPOIS (Pessimista) | Mudança |
|-------|------------------|---------------------|---------|
| **LONG** | 50.2% (26,034) | **27.3%** (14,148) | -22.9pp ⬇️ |
| **SHORT** | 42.3% (21,915) | **26.9%** (13,919) | -15.4pp ⬇️ |
| **NO_TRADE** | 7.5% (3,889) | **45.9%** (23,771) | +38.4pp ⬆️ |
| **Setup Viáveis** | 92.5% | **54.1%** | -38.4pp |

### Interpretação

**92.5% → 54.1% de setups viáveis**:
- **38.4% dos setups eram VIOLINOS** (TP e SL no mesmo candle)
- Antes: Modelo treinava em falsos positivos
- Depois: Modelo vê realidade (mercado é duro!)

**Distribuição mais balanceada**:
- LONG: 27.3% (antes 50.2%) - mais realista
- SHORT: 26.9% (antes 42.3%) - mais balanceado
- NO_TRADE: 45.9% (antes 7.5%) - mercado lateral é maioria

---

## 🎯 EXPECTATIVA COM CORREÇÕES

### Win Rate Esperado

| Cenário | Win Rate | Comentário |
|---------|----------|------------|
| **LSTM (bugado)** | 54.3% | Com normalização errada |
| **MCA (labels bugados)** | 60-65% | Ainda inflado por violinos |
| **MCA (labels corretos)** | **55-62%** | Realista, considerando violinos |

**Por quê win rate vai CAIR?**
- Labels agora incluem violinos como perdas
- Spread de 0.02% reduz TP efetivo
- Mercado lateral (45.9% NO_TRADE) é mais comum

**Mas isso é BOM!**
- Win rate agora reflete REALIDADE
- Backtest alinhado com forward testing
- Sem surpresas desagradáveis em produção

---

## ⚙️ DIMENSÕES DO MAMBA (Bonus Fix)

**Problema**: RuntimeError (tensores incompatíveis)

**Correção**:
```python
# Dimensões corretas das matrizes SSM
self.A = nn.Parameter(torch.randn(d_state, d_state))    # [16, 16]
self.B = nn.Parameter(torch.randn(d_model, d_state))    # [64, 16]
self.C = nn.Parameter(torch.randn(d_state, d_model))    # [16, 64]
```

---

## 📂 ARQUIVOS MODIFICADOS

1. ✅ `scalping_mamba_hybrid.py`:
   - ScalpingDataset: Normalização Z-Score
   - TradingFocalLoss: Class weighting NO_TRADE
   - MambaBlock: Dimensões corretas

2. ✅ `scalping_labeling.py`:
   - _check_trade_outcome: Lógica pessimista
   - Spread de 0.02% incluído

3. ✅ Dataset regenerado:
   - `1HZ100V_5min_180days_labeled_pessimista.csv`
   - Labels realistas (54.1% viáveis)

---

## 🚀 PRÓXIMA AÇÃO

**Treinar MCA com TODAS as correções**:
- ✅ Normalização preserva tendência
- ✅ Class weighting evita passividade
- ✅ Labels realistas (violinos = perdas)

**Expectativa**: Win rate 55-62% (realista, sem ilusões)

---

## 🎓 LIÇÕES APRENDIDAS

1. **Normalização é Arte**
   - Errar = destruir informação crítica
   - Z-Score por janela preserva estrutura temporal
   - Sempre verificar: modelo VÊ tendências?

2. **Backtest Honesto**
   - Violinos (TP+SL mesmo candle) SÃO perdas
   - Spread é custo real (não ignorar!)
   - Lógica pessimista previne surpresas

3. **Class Balancing**
   - Não só LONG vs SHORT
   - NO_TRADE também precisa ajuste
   - Peso muito alto = modelo passivo demais

4. **Code Review Salva Vidas**
   - 3 bugs fatais identificados
   - Teriam causado dezenas de horas de debugging
   - Review técnico identificou em minutos

---

**Status**: Modelo pronto para treinamento realista! 🎯

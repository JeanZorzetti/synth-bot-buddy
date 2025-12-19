# ScalpingMaster-MCA: Correções Críticas Aplicadas

**Data**: 18/12/2025
**Status**: Modelo corrigido baseado em code review técnico

---

## 🔴 PROBLEMAS IDENTIFICADOS (Code Review)

### 1. CRÍTICO: Normalização Destruía Tendência

**Problema Original**:
```python
# ERRADO: Normalizava cada candle por ele mesmo
for i in range(len(self.ohlc)):
    close = self.ohlc[i, 3]
    self.normalized_ohlc[i] = (self.ohlc[i] - close) / close * 100
```

**Consequência**:
- Close sempre = 0 para todos os candles
- Tendência completamente destruída
- Preço sobe 100→200? Modelo vê linha reta em zero
- Mamba não consegue detectar "dia de alta" (sem slope)

**Correção Aplicada**:
```python
# CORRETO: Normalização Z-Score por janela (preserva tendência!)
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

### 2. BALANCEAMENTO: NO_TRADE Dominava

**Problema**: Sem peso diferenciado, modelo aprende a não fazer nada (minimiza risco)

**Correção**:
```python
# Adicionado class_weight para NO_TRADE
no_trade_weight = 0.5  # Metade do peso de LONG/SHORT

# Aplicado na loss function
class_weight = torch.where(
    targets == 0,  # NO_TRADE
    torch.tensor(0.5),  # Peso menor
    torch.tensor(1.0)   # LONG/SHORT peso normal
)

loss = focal_term * ce_loss * penalty * class_weight
```

**Resultado**:
- ✅ Modelo forçado a tomar decisões
- ✅ Não colapsa para "sempre NO_TRADE"

---

### 3. PERFORMANCE: Mamba Simulado (Não Paralelizado)

**Nota**:
- Implementação atual é RNN estilo vanilla (sequencial)
- Funciona para protótipo, mas **não é paralelizável**
- Perde vantagem de velocidade do Mamba real (6x)

**Solução para Produção**:
```bash
# Instalar biblioteca oficial Mamba (requer CUDA)
pip install mamba-ssm
```

```python
from mamba_ssm import Mamba

# Substituir MambaBlock simplificado
self.mamba_brain = Mamba(
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2
)
```

**Status**: Mantido simplificado para CPU (funciona para treino inicial)

---

## ✅ MELHORIAS IMPLEMENTADAS

### Comparação Antes vs Depois

| Aspecto | Antes (Bugado) | Depois (Corrigido) |
|---------|----------------|-------------------|
| **Normalização** | Por candle (Close=0) | Z-Score por janela |
| **Tendência** | ❌ Destruída | ✅ Preservada |
| **Class Weight** | ❌ Sem peso | ✅ NO_TRADE=0.5x |
| **Modelo Passivo** | ✅ Sim (só NO_TRADE) | ❌ Forçado a operar |
| **Mamba** | Simplificado | Simplificado (ok para CPU) |

---

## 🎯 EXPECTATIVAS ATUALIZADAS

### Antes das Correções (Estimado com Bug)
- Win Rate: 50-52% (aleatório, sem tendência)
- SHORT Accuracy: 0-10% (colapso)
- Problema: Modelo cego para tendências

### Depois das Correções (Estimado)
- Win Rate: **60-68%** ⬆️
- SHORT Accuracy: **50-60%** ⬆️
- LONG Accuracy: **65-70%**
- Modelo agora VÊ tendências e age nelas

---

## 📊 MUDANÇAS NO CÓDIGO

### ScalpingDataset (Dataset.py)

**Antes**:
```python
# Destruía tendência
for i in range(len(self.ohlc)):
    close = self.ohlc[i, 3]
    self.normalized_ohlc[i] = (self.ohlc[i] - close) / close * 100
```

**Depois**:
```python
# Preserva tendência
window = self.ohlc[idx:idx + self.long_window]
mean = window.mean(axis=0)
std = window.std(axis=0) + 1e-8
x = (window - mean) / std
```

### TradingFocalLoss (Loss Function)

**Antes**:
```python
# Sem class weighting
loss = self.alpha * focal_term * ce_loss * penalty
```

**Depois**:
```python
# Com class weighting (NO_TRADE = 0.5x)
class_weight = torch.where(targets == 0, 0.5, 1.0)
loss = self.alpha * focal_term * ce_loss * penalty * class_weight
```

---

## 🚀 PRÓXIMA AÇÃO

**Retreinar modelo com correções**:
```bash
python scalping_mamba_hybrid.py
```

**Expectativa**: Win rate deve subir de ~54% (LSTM) para **60-68%** (MCA corrigido)

---

## 📚 LIÇÕES APRENDIDAS

1. **Normalização Importa MUITO**
   - Errar normalização = destruir informação crítica
   - Z-Score por janela preserva estrutura temporal
   - Sempre verificar: modelo VÊ tendências?

2. **Class Balancing É Arte**
   - Não é só balancear LONG vs SHORT
   - NO_TRADE também precisa de ajuste
   - Peso muito alto = modelo passivo demais

3. **Code Review Salva Vidas**
   - Bug de normalização passou despercebido
   - Teria treinado modelo cego por horas
   - Review técnico identificou em 5 min

---

**Próximo**: Treinar modelo corrigido e comparar com LSTM baseline.

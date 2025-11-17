# 🔍 Como Ver Logs de Produção - Análise de Sinais

## Objetivo
Ver os logs detalhados da análise técnica para entender por que os sinais estão sendo gerados como NEUTRAL, BUY ou SELL.

---

## 📋 Passo a Passo - EasyPanel

### 1. Acessar Logs do Backend

1. Entre no EasyPanel: https://easypanel.io
2. Selecione o projeto do bot
3. Clique no serviço **backend**
4. Vá em **Logs** (ícone de terminal/console)

### 2. Fazer Request para Gerar Logs

Abra um terminal e execute:

```bash
curl https://botderivapi.roilabs.com.br/api/signals/1HZ75V
```

### 3. Ver Análise Detalhada nos Logs

Nos logs do EasyPanel você verá algo assim:

```
============================================================
ANÁLISE DE SINAIS PARA 1HZ75V
Preço atual: 102.78324
============================================================

[TENDÊNCIA - EMA Crossover]
  EMA 9: 102.78324
  EMA 21: 102.82573
  Sinal: NEUTRAL
  Strength: 0.00
  - NEUTRAL

[RSI]
  Valor: 33.18
  Sinal: NEUTRAL
  Condição: neutral
  Strength: 0.00
  - NEUTRAL

[MACD]
  MACD Line: -0.02704
  Signal Line: 0.00384
  Histogram: -0.03087 (anterior: -0.02500)
  Sinal: SELL
  Strength: 25.00
  ✓ VOTO SELL adicionado

[BOLLINGER BANDS]
  Preço: 102.78324
  Upper: 103.13264
  Middle: 102.81545
  Lower: 102.49827
  Width: 0.00617
  Sinal: NEUTRAL
  Condição: middle_zone
  Strength: 0.00
  - NEUTRAL

[STOCHASTIC]
  %K: 23.99 (anterior: 25.50)
  %D: 36.27 (anterior: 38.10)
  Sinal: NEUTRAL
  Strength: 0.00
  - NEUTRAL

============================================================
RESUMO DOS VOTOS:
  BUY signals: 0 (strength total: 0.00)
  SELL signals: 1 (strength total: 25.00)
    • MACD negativo e se afastando (bearish)
  Total signals: 1
============================================================

============================================================
DECISÃO FINAL:
  Requisitos para BUY: >= 3 sinais BUY E buy_strength > sell_strength
  Requisitos para SELL: >= 3 sinais SELL E sell_strength > buy_strength
  BUY signals: 0 | BUY strength: 0.00
  SELL signals: 1 | SELL strength: 25.00
  ⊘ SINAL: NEUTRAL
  Motivo: Menos de 3 sinais em ambas direções
============================================================
```

---

## 🔍 Como Interpretar os Logs

### Indicadores que Votam

Cada indicador pode votar em:
- **BUY** - sinal de compra
- **SELL** - sinal de venda
- **NEUTRAL** - sem sinal claro

### Critérios para Gerar Sinal

Para gerar sinal **BUY**:
- Precisa de **3 ou mais** indicadores votando BUY
- **E** `buy_strength > sell_strength`

Para gerar sinal **SELL**:
- Precisa de **3 ou mais** indicadores votando SELL
- **E** `sell_strength > buy_strength`

Caso contrário: **NEUTRAL**

### Exemplo de Análise

No exemplo acima, temos:
- ✅ RSI em 33.18 (próximo de sobrevenda, mas não < 30)
- ❌ EMA 9 abaixo da EMA 21 (sem crossover)
- ✅ MACD negativo (voto SELL)
- ❌ Bollinger na zona central (neutral)
- ❌ Stochastic em 23.99 (não < 20)

**Resultado:** Apenas 1 voto (SELL), precisa de 3 → **NEUTRAL**

---

## 💡 Dicas

### Forçar Sinal BUY para Teste

Se quiser testar com sinal BUY garantido, você pode:

1. Modificar temporariamente os thresholds em `momentum_indicators.py`:
```python
# RSI
if rsi_value <= 40:  # era 30
    return {'signal': 'BUY', ...}

# Stochastic
if k_current <= 30:  # era 20
    return {'signal': 'BUY', ...}
```

2. Fazer commit e push
3. Aguardar redeploy
4. Testar novamente

### Ver Logs em Tempo Real

No EasyPanel, ative "Auto-scroll" nos logs para ver em tempo real enquanto faz requests.

---

## 🚀 Próximos Passos

Após ver os logs, você pode:

1. **Ajustar thresholds** - se muitos NEUTRAL, tornar critérios mais lenientes
2. **Testar com dados reais** - integrar Deriv API em vez de dados sintéticos
3. **Implementar backtesting** - validar estratégia em dados históricos

---

## 📊 Comandos Úteis

```bash
# Testar indicadores
curl https://botderivapi.roilabs.com.br/api/indicators/1HZ75V

# Testar sinais (gera logs)
curl https://botderivapi.roilabs.com.br/api/signals/1HZ75V

# Health check
curl https://botderivapi.roilabs.com.br/health
```

**Nota:** Cada chamada ao `/api/signals` gera logs completos no EasyPanel.

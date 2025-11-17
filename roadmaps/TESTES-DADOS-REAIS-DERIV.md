# 🔌 Testes com Dados Reais do Deriv API

## ✅ Implementado

A integração com Deriv API está completa! Os endpoints agora:
1. **Tentam buscar dados reais** do Deriv se WebSocket estiver conectado
2. **Fazem fallback para dados sintéticos** se não houver conexão
3. **Informam a fonte dos dados** na resposta (`data_source`)

---

## 📋 Como Testar

### **Cenário 1: Sem Conexão (Dados Sintéticos)**

Atualmente os endpoints estão usando dados sintéticos porque o WebSocket não está conectado.

```bash
# Testar indicadores
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V"

# Resposta esperada:
{
  "symbol": "1HZ75V",
  "timeframe": "1m",
  "data_source": "synthetic_no_connection",  // ← Indica dados sintéticos
  "candles_analyzed": 500,
  "indicators": {...}
}
```

### **Cenário 2: Com Conexão (Dados Reais)**

Para usar dados reais do Deriv:

#### **Passo 1: Conectar WebSocket**

No frontend ou via API:

```bash
curl -X POST https://botderivapi.roilabs.com.br/connect \
  -H "Content-Type: application/json" \
  -d '{"api_token": "SEU_TOKEN_DERIV"}'
```

#### **Passo 2: Verificar Conexão**

```bash
curl https://botderivapi.roilabs.com.br/health

# Deve mostrar:
{
  "websocket_manager": {
    "initialized": true,
    "state": "authenticated"  // ← Importante!
  }
}
```

#### **Passo 3: Testar com Dados Reais**

```bash
# Indicadores com dados reais
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?timeframe=5m"

# Resposta:
{
  "data_source": "deriv_api",  // ← Dados reais! 🎉
  "candles_analyzed": 500,
  "indicators": {
    "rsi": 45.23,  // ← RSI calculado com dados reais
    "macd_histogram": 0.012,
    ...
  }
}
```

---

## 🚀 Novos Recursos

### **1. Suporte a Múltiplos Timeframes**

```bash
# 1 minuto
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=1m"

# 5 minutos
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=5m"

# 15 minutos
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=15m"

# 1 hora
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=1h"

# 4 horas
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=4h"
```

### **2. Análise de Múltiplos Símbolos**

```bash
# Comparar sinais de vários ativos
curl "https://botderivapi.roilabs.com.br/api/signals/multi?symbols=1HZ75V,1HZ100V,R_100,BOOM1000&timeframe=5m"

# Resposta:
{
  "timestamp": "2025-11-17T16:00:00Z",
  "timeframe": "5m",
  "total_symbols": 4,
  "summary": {
    "buy_signals": 1,
    "sell_signals": 2,
    "neutral_signals": 1
  },
  "signals": [
    {
      "symbol": "1HZ75V",
      "signal_type": "BUY",
      "strength": 75,
      "confidence": 85,
      ...
    },
    {
      "symbol": "1HZ100V",
      "signal_type": "SELL",
      ...
    },
    ...
  ]
}
```

### **3. Controle de Quantidade de Candles**

```bash
# Análise rápida (200 candles)
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?count=200"

# Análise profunda (1000 candles)
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?count=1000"
```

---

## 🔍 Símbolos Suportados

### **Volatility Indices (Synthetic)**

- `1HZ75V` - Volatility 75 (1s)
- `1HZ100V` - Volatility 100 (1s)
- `1HZ150V` - Volatility 150 (1s)
- `1HZ200V` - Volatility 200 (1s)
- `1HZ250V` - Volatility 250 (1s)

### **Crash/Boom**

- `BOOM1000` - Boom 1000
- `BOOM500` - Boom 500
- `CRASH1000` - Crash 1000
- `CRASH500` - Crash 500

### **Continuous Indices**

- `R_100` - Volatility 100 Index
- `R_75` - Volatility 75 Index
- `R_50` - Volatility 50 Index
- `R_25` - Volatility 25 Index
- `R_10` - Volatility 10 Index

---

## 📊 Validação dos Dados Reais

### **Comparar com TradingView**

1. Abra TradingView: https://www.tradingview.com/chart/
2. Selecione o símbolo (ex: Volatility 75 Index)
3. Configure timeframe (ex: 5 minutos)
4. Compare os valores de RSI, MACD, Bollinger Bands

**Exemplo:**

```bash
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?timeframe=5m"
```

Compare:
- **RSI** deve estar próximo do valor no TradingView
- **MACD Histogram** deve ter mesmo sinal (positivo/negativo)
- **Bollinger Bands** devem estar nas mesmas faixas

### **Verificar Consistência Temporal**

```bash
# Fazer 3 requests com 1 minuto de intervalo
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?timeframe=1m"
# Esperar 1 minuto
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?timeframe=1m"
# Esperar 1 minuto
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V?timeframe=1m"
```

**Validação:**
- Valores devem mudar gradualmente
- RSI não deve variar mais de 5-10 pontos entre requests
- Tendência deve ser consistente

---

## 🐛 Troubleshooting

### **Erro: "DerivAPI não inicializado"**

**Causa:** WebSocket não está conectado.

**Solução:**
1. Conecte via `/connect` endpoint com token válido
2. Verifique status em `/health`
3. O sistema fará fallback para dados sintéticos

### **Erro: "Símbolo não encontrado"**

**Causa:** Símbolo inválido ou não disponível.

**Solução:**
- Use símbolos válidos da lista acima
- Verifique ortografia (case-sensitive)

### **data_source: "synthetic_fallback"**

**Causa:** WebSocket conectado, mas erro ao buscar dados.

**Motivos comuns:**
- Símbolo temporariamente indisponível
- Limite de rate do Deriv API
- Timeframe incompatível

**Ação:** Logs do EasyPanel mostrarão o erro específico.

---

## 📈 Próximos Passos

### **1. Validar Precisão dos Indicadores**

- [ ] Comparar RSI com TradingView (diferença < 1%)
- [ ] Validar MACD com MT5
- [ ] Conferir Bollinger Bands

### **2. Testar Diferentes Condições de Mercado**

- [ ] Alta volatilidade (BOOM1000, CRASH1000)
- [ ] Baixa volatilidade (R_10, R_25)
- [ ] Mercado lateral
- [ ] Tendência forte

### **3. Backtesting com Dados Reais**

- [ ] Implementar endpoint `/api/backtest`
- [ ] Testar estratégia em 1 mês de dados históricos
- [ ] Calcular win rate real
- [ ] Validar sharpe ratio > 1.3

### **4. Paper Trading**

- [ ] Monitorar sinais por 1 semana
- [ ] Registrar todos os sinais gerados
- [ ] Comparar com resultado real do mercado
- [ ] Ajustar thresholds se necessário

---

## 🎯 Comandos Rápidos para Testar

```bash
# Health check
curl https://botderivapi.roilabs.com.br/health

# Indicadores (dados sintéticos por enquanto)
curl "https://botderivapi.roilabs.com.br/api/indicators/1HZ75V"

# Sinais (dados sintéticos por enquanto)
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V"

# Múltiplos símbolos
curl "https://botderivapi.roilabs.com.br/api/signals/multi?symbols=1HZ75V,R_100,BOOM1000"

# Diferentes timeframes
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=5m"
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=15m"
curl "https://botderivapi.roilabs.com.br/api/signals/1HZ75V?timeframe=1h"
```

---

## ✅ Critérios de Sucesso

| Critério | Status | Observações |
|----------|--------|-------------|
| **Endpoints retornam 200 OK** | ⏳ | Aguardar redeploy |
| **data_source correto** | ⏳ | synthetic_no_connection por enquanto |
| **Indicadores calculados** | ✅ | Funcionando com dados sintéticos |
| **Múltiplos símbolos** | ⏳ | Testar após redeploy |
| **Diferentes timeframes** | ⏳ | Testar após redeploy |
| **Fallback funciona** | ⏳ | Testar com e sem conexão |

**Status após redeploy será atualizado! 🚀**

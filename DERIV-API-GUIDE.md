# 🚀 Guia de Uso da API Deriv - 16 Funcionalidades Essenciais

Este guia documenta as 16 funcionalidades essenciais da API do Deriv implementadas no **Synth Bot Buddy** para operação real em contas fictícias.

## 📋 Sumário das Funcionalidades

### **🔐 Autenticação & Conexão**
1. **`POST /deriv/connect`** - Conectar e autenticar
2. **`POST /deriv/disconnect`** - Desconectar
3. **`GET /deriv/status`** - Status da conexão
4. **`GET /deriv/health`** - Health check

### **💰 Informações da Conta**
5. **`GET /deriv/balance`** - Saldo da conta
6. **`GET /deriv/portfolio`** - Contratos abertos
7. **`GET /deriv/history`** - Histórico de trades

### **📊 Dados de Mercado**
8. **`GET /deriv/symbols`** - Símbolos disponíveis
9. **`GET /deriv/symbols/{symbol}/info`** - Info do símbolo
10. **`POST /deriv/subscribe/ticks/{symbol}`** - Subscrever ticks
11. **`GET /deriv/ticks/{symbol}/last`** - Último tick

### **💹 Operações de Trading**
12. **`POST /deriv/buy`** - Comprar contratos
13. **`POST /deriv/sell`** - Vender contratos

---

## 🔧 Como Usar

### 1. **Conectar à API Deriv**

```bash
curl -X POST "http://localhost:8000/deriv/connect" \
-H "Content-Type: application/json" \
-d '{
  "api_token": "seu_token_aqui",
  "demo": true
}'
```

**Resposta:**
```json
{
  "status": "success",
  "message": "Conectado e autenticado com sucesso na Deriv API",
  "connection_info": {
    "is_connected": true,
    "is_authenticated": true,
    "balance": 10000.0,
    "loginid": "VRTC12345"
  }
}
```

### 2. **Verificar Saldo**

```bash
curl -X GET "http://localhost:8000/deriv/balance"
```

**Resposta:**
```json
{
  "status": "success",
  "balance": 10000.0,
  "currency": "USD",
  "loginid": "VRTC12345"
}
```

### 3. **Obter Símbolos Disponíveis**

```bash
curl -X GET "http://localhost:8000/deriv/symbols"
```

**Resposta:**
```json
{
  "status": "success",
  "symbols": [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"
  ],
  "count": 10
}
```

### 4. **Subscrever a Ticks de um Símbolo**

```bash
curl -X POST "http://localhost:8000/deriv/subscribe/ticks/R_50"
```

**Resposta:**
```json
{
  "status": "success",
  "message": "Subscrito a ticks de R_50",
  "symbol": "R_50"
}
```

### 5. **Obter Último Tick**

```bash
curl -X GET "http://localhost:8000/deriv/ticks/R_50/last"
```

**Resposta:**
```json
{
  "status": "success",
  "symbol": "R_50",
  "tick": {
    "price": 234.567,
    "timestamp": 1641234567,
    "epoch": 1641234567
  }
}
```

### 6. **Comprar Contrato**

```bash
curl -X POST "http://localhost:8000/deriv/buy" \
-H "Content-Type: application/json" \
-d '{
  "contract_type": "CALL",
  "symbol": "R_50",
  "amount": 10.0,
  "duration": 5,
  "duration_unit": "m"
}'
```

**Resposta:**
```json
{
  "status": "success",
  "message": "Contrato comprado com sucesso",
  "contract": {
    "contract_id": 123456789,
    "buy_price": 10.0,
    "payout": 19.50,
    "longcode": "Win payout if Volatility 50 Index is strictly higher than entry spot at 5 minutes after contract start time."
  }
}
```

### 7. **Vender Contrato**

```bash
curl -X POST "http://localhost:8000/deriv/sell" \
-H "Content-Type: application/json" \
-d '{
  "contract_id": 123456789
}'
```

**Resposta:**
```json
{
  "status": "success",
  "message": "Contrato vendido com sucesso",
  "sale": {
    "sold_for": 15.25,
    "transaction_id": 987654321
  }
}
```

### 8. **Verificar Portfólio**

```bash
curl -X GET "http://localhost:8000/deriv/portfolio"
```

**Resposta:**
```json
{
  "status": "success",
  "contracts": [
    {
      "contract_id": 123456789,
      "symbol": "R_50",
      "contract_type": "CALL",
      "buy_price": 10.0,
      "current_spot": 234.890,
      "profit": 5.25,
      "payout": 19.50,
      "is_sold": false
    }
  ],
  "count": 1
}
```

### 9. **Obter Histórico de Trades**

```bash
curl -X GET "http://localhost:8000/deriv/history?limit=10"
```

**Resposta:**
```json
{
  "status": "success",
  "transactions": [
    {
      "transaction_id": 987654321,
      "contract_id": 123456789,
      "symbol": "R_50",
      "contract_type": "CALL",
      "buy_price": 10.0,
      "sell_price": 15.25,
      "profit": 5.25,
      "duration": 300,
      "purchase_time": 1641234567,
      "sell_time": 1641234867
    }
  ],
  "count": 1
}
```

---

## 🎯 **Exemplos de Uso Avançado**

### **Trading Automatizado**

```python
import aiohttp
import asyncio

async def automated_trading():
    """Exemplo de trading automatizado"""
    
    # 1. Conectar
    async with aiohttp.ClientSession() as session:
        # Conectar
        connect_data = {
            "api_token": "seu_token_demo",
            "demo": True
        }
        
        async with session.post("http://localhost:8000/deriv/connect", 
                              json=connect_data) as response:
            result = await response.json()
            print("Conectado:", result)
        
        # 2. Obter símbolos
        async with session.get("http://localhost:8000/deriv/symbols") as response:
            symbols = await response.json()
            print("Símbolos disponíveis:", symbols)
        
        # 3. Subscrever a ticks
        symbol = "R_50"
        async with session.post(f"http://localhost:8000/deriv/subscribe/ticks/{symbol}") as response:
            subscription = await response.json()
            print("Subscrito:", subscription)
        
        # 4. Comprar contrato
        trade_data = {
            "contract_type": "CALL",
            "symbol": symbol,
            "amount": 10.0,
            "duration": 5,
            "duration_unit": "m"
        }
        
        async with session.post("http://localhost:8000/deriv/buy", 
                              json=trade_data) as response:
            trade_result = await response.json()
            print("Trade executado:", trade_result)
        
        # 5. Monitorar portfólio
        await asyncio.sleep(60)  # Aguardar 1 minuto
        
        async with session.get("http://localhost:8000/deriv/portfolio") as response:
            portfolio = await response.json()
            print("Portfólio:", portfolio)

# Executar
asyncio.run(automated_trading())
```

### **Monitoramento em Tempo Real**

```python
import websockets
import json
import asyncio

async def monitor_deriv_connection():
    """Monitorar status da conexão Deriv"""
    
    while True:
        try:
            # Health check
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/deriv/health") as response:
                    health = await response.json()
                    
                    if health.get("is_healthy"):
                        print("✅ Conexão Deriv saudável")
                    else:
                        print("❌ Problema na conexão Deriv")
                        print("Detalhes:", health)
            
            await asyncio.sleep(30)  # Check a cada 30 segundos
            
        except Exception as e:
            print(f"Erro no monitoramento: {e}")
            await asyncio.sleep(60)
```

---

## 🔒 **Segurança e Tokens**

### **Obtendo Token da API Deriv**

1. Acesse [app.deriv.com](https://app.deriv.com)
2. Login na sua conta (use conta demo para testes)
3. Vá em **Settings** → **API token**
4. Crie um novo token com scopes:
   - ✅ **Read** - Para leitura de dados
   - ✅ **Trade** - Para execução de trades
   - ✅ **Payments** - Para informações de pagamento

### **Configuração de Segurança**

```bash
# Definir token como variável de ambiente
export DERIV_API_TOKEN="seu_token_aqui"

# Usar em produção
curl -X POST "http://localhost:8000/deriv/connect" \
-H "Content-Type: application/json" \
-d '{
  "api_token": "'$DERIV_API_TOKEN'",
  "demo": true
}'
```

---

## 📊 **Códigos de Status HTTP**

| Código | Significado | Ação |
|--------|-------------|------|
| **200** | Sucesso | Operação realizada com sucesso |
| **400** | Erro de parâmetros | Verificar dados enviados |
| **401** | Não autenticado | Conectar primeiro com `/deriv/connect` |
| **404** | Não encontrado | Verificar símbolo ou ID do contrato |
| **500** | Erro interno | Verificar logs do servidor |

---

## 🎛️ **Tipos de Contratos Disponíveis**

### **Contratos Básicos**
- **`CALL`** - Rise (preço sobe)
- **`PUT`** - Fall (preço desce)

### **Símbolos Populares**
- **`R_10`** - Volatility 10 Index
- **`R_25`** - Volatility 25 Index  
- **`R_50`** - Volatility 50 Index
- **`R_75`** - Volatility 75 Index
- **`R_100`** - Volatility 100 Index

### **Durações Suportadas**
- **Minutos**: 1-60 (`duration_unit: "m"`)
- **Segundos**: 15-3600 (`duration_unit: "s"`)
- **Horas**: 1-24 (`duration_unit: "h"`)

---

## 🚨 **Troubleshooting**

### **Erro: "Token inválido"**
```bash
# Verificar se token está correto
curl -X GET "http://localhost:8000/deriv/status"
```

### **Erro: "Símbolo não encontrado"**
```bash
# Listar símbolos disponíveis
curl -X GET "http://localhost:8000/deriv/symbols"
```

### **Erro: "Conexão perdida"**
```bash
# Reconectar
curl -X POST "http://localhost:8000/deriv/connect" \
-H "Content-Type: application/json" \
-d '{"api_token": "seu_token", "demo": true}'
```

### **Health Check Completo**
```bash
curl -X GET "http://localhost:8000/deriv/health"
```

---

## 🎯 **Próximos Passos**

1. **✅ Implementação Concluída** - 16 funcionalidades da API Deriv
2. **🧪 Testes** - Validar operações em conta demo
3. **🔄 Integração** - Conectar com sistema de trading existente
4. **📊 Dashboard** - Interface visual para monitoramento
5. **🤖 Automação** - Bot de trading com as APIs

---

## 📞 **Suporte**

- **Logs**: Verificar `backend/logs/` para depuração
- **Documentação Deriv**: [developers.deriv.com](https://developers.deriv.com)
- **Status da API**: `GET /deriv/health`
- **Endpoints disponíveis**: `GET /routes`

---

**🎉 Pronto! Agora você tem acesso completo à API do Deriv para operações reais em conta fictícia!**
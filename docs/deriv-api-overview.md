# Deriv API - Overview Completo

## Introdução

A API da Deriv permite que desenvolvedores criem aplicações de trading personalizadas usando WebSockets para comunicação em tempo real. Esta documentação cobre os principais endpoints e funcionalidades.

**Base URL:** `wss://ws.binaryws.com/websockets/v3?app_id=YOUR_APP_ID`

## Autenticação

### OAuth 2.0 (Recomendado)

```javascript
// 1. Redirecionar para OAuth
const authUrl = 'https://oauth.deriv.com/oauth2/authorize?app_id=99188';
window.location.href = authUrl;

// 2. Processar callback com parâmetros
const params = new URLSearchParams(window.location.search);
const token = params.get('token1');
const account = params.get('acct1');
```

### Token de API (Legacy)

```json
{
  "authorize": "YOUR_API_TOKEN"
}
```

## Principais Endpoints

### 1. Buy (Comprar Contrato)

**Propósito:** Comprar contratos de trading

```json
{
  "buy": 1,
  "subscribe": 1,
  "price": 10,
  "parameters": {
    "amount": 10,
    "basis": "stake",
    "contract_type": "CALL",
    "currency": "USD",
    "duration": 5,
    "duration_unit": "m",
    "symbol": "R_100"
  }
}
```

### 2. Sell (Vender Contrato)

**Propósito:** Vender contratos antes do vencimento

```json
{
  "sell": 1,
  "price": 15
}
```

### 3. Portfolio (Portfólio)

**Propósito:** Obter contratos em aberto

```json
{
  "portfolio": 1
}
```

### 4. Balance (Saldo)

**Propósito:** Obter saldo da conta

```json
{
  "balance": 1,
  "subscribe": 1
}
```

### 5. Ticks (Preços em Tempo Real)

**Propósito:** Subscrever a preços em tempo real

```json
{
  "ticks": "R_100",
  "subscribe": 1
}
```

### 6. History (Histórico)

**Propósito:** Obter histórico de transações

```json
{
  "statement": 1,
  "limit": 50
}
```

### 7. Active Symbols (Símbolos Ativos)

**Propósito:** Obter lista de símbolos disponíveis

```json
{
  "active_symbols": "brief"
}
```

### 8. Profit Table (Tabela de Lucros)

**Propósito:** Obter relatório de lucros/perdas

```json
{
  "profit_table": 1,
  "limit": 50
}
```

## Tipos de Contrato

### Rise/Fall (Alta/Baixa)

```json
{
  "contract_type": "CALL",  // Alta
  "contract_type": "PUT"    // Baixa
}
```

### Higher/Lower (Maior/Menor)

```json
{
  "contract_type": "CALLE", // Maior que
  "contract_type": "PUTE"   // Menor que
}
```

### Touch/No Touch (Toque/Sem Toque)

```json
{
  "contract_type": "ONETOUCH", // Toque
  "contract_type": "NOTOUCH",  // Sem toque
  "barrier": "100.50"
}
```

### Matches/Differs (Igual/Diferente)

```json
{
  "contract_type": "DIGITMATCH",  // Igual
  "contract_type": "DIGITDIFF",   // Diferente
  "barrier": "5"
}
```

### Even/Odd (Par/Ímpar)

```json
{
  "contract_type": "DIGITEVEN", // Par
  "contract_type": "DIGITODD"   // Ímpar
}
```

### Over/Under (Acima/Abaixo)

```json
{
  "contract_type": "DIGITOVER",  // Acima
  "contract_type": "DIGITUNDER", // Abaixo
  "barrier": "5"
}
```

## Símbolos Disponíveis

### Índices Sintéticos

| Símbolo | Nome | Volatilidade |
|---------|------|--------------|
| `R_10` | Volatility 10 Index | ~10% |
| `R_25` | Volatility 25 Index | ~25% |
| `R_50` | Volatility 50 Index | ~50% |
| `R_75` | Volatility 75 Index | ~75% |
| `R_100` | Volatility 100 Index | ~100% |

### Índices de Salto

| Símbolo | Nome | Característica |
|---------|------|----------------|
| `JD10` | Jump 10 Index | Saltos de 10% |
| `JD25` | Jump 25 Index | Saltos de 25% |
| `JD50` | Jump 50 Index | Saltos de 50% |
| `JD75` | Jump 75 Index | Saltos de 75% |
| `JD100` | Jump 100 Index | Saltos de 100% |

### Índices Boom e Crash

| Símbolo | Nome | Característica |
|---------|------|----------------|
| `BOOM1000` | Boom 1000 Index | Picos de alta |
| `CRASH1000` | Crash 1000 Index | Quedas bruscas |

## Estrutura de Resposta

### Resposta de Sucesso

```json
{
  "buy": {
    "balance_after": 9990,
    "buy_price": 10,
    "contract_id": 123456789,
    "longcode": "Win payout if...",
    "payout": 19.54,
    "purchase_time": 1640995200,
    "shortcode": "CALL_R_100_10...",
    "start_time": 1640995200,
    "transaction_id": 987654321
  },
  "echo_req": { /* request original */ },
  "msg_type": "buy",
  "req_id": 1
}
```

### Resposta de Erro

```json
{
  "error": {
    "code": "InvalidContractType",
    "message": "Invalid contract type.",
    "details": {
      "field": "contract_type"
    }
  },
  "echo_req": { /* request original */ },
  "msg_type": "buy",
  "req_id": 1
}
```

## Códigos de Erro Comuns

| Código | Descrição | Solução |
|--------|-----------|---------|
| `AuthorizationRequired` | Token necessário | Incluir token válido |
| `InvalidToken` | Token inválido | Verificar validade do token |
| `InvalidContractType` | Tipo de contrato inválido | Usar tipos suportados |
| `InvalidSymbol` | Símbolo inválido | Usar símbolos válidos |
| `InvalidDuration` | Duração inválida | Verificar limites de duração |
| `InsufficientBalance` | Saldo insuficiente | Verificar saldo da conta |
| `MarketClosed` | Mercado fechado | Negociar em horário de funcionamento |
| `PriceChanged` | Preço alterado | Tentar novamente com preço atualizado |

## Limites e Restrições

### Limites de Duração

| Unidade | Mínimo | Máximo |
|---------|--------|--------|
| Segundos | 5 | 600 |
| Minutos | 1 | 60 |
| Horas | 1 | 24 |
| Dias | 1 | 365 |

### Limites de Stake

| Tipo de Conta | Mínimo | Máximo |
|---------------|--------|--------|
| Demo | $0.35 | $50,000 |
| Real | $0.35 | $50,000 |

### Rate Limits

- **Máximo:** 150 requests por minuto
- **Burst:** 50 requests por 10 segundos
- **Conexões:** 5 conexões simultâneas por conta

## Exemplo Completo - Bot de Trading

```javascript
const WebSocket = require('ws');

class DerivBot {
  constructor(appId, token) {
    this.appId = appId;
    this.token = token;
    this.ws = null;
    this.isAuthenticated = false;
  }

  connect() {
    this.ws = new WebSocket(`wss://ws.binaryws.com/websockets/v3?app_id=${this.appId}`);

    this.ws.onopen = () => {
      console.log('Connected to Deriv');
      this.authenticate();
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };
  }

  authenticate() {
    this.send({ authorize: this.token });
  }

  handleMessage(data) {
    switch (data.msg_type) {
      case 'authorize':
        if (!data.error) {
          this.isAuthenticated = true;
          console.log('Authenticated successfully');
          this.subscribeTicks('R_100');
        }
        break;

      case 'tick':
        console.log(`Tick for ${data.tick.symbol}: ${data.tick.quote}`);
        this.analyzeAndTrade(data.tick);
        break;

      case 'buy':
        if (!data.error) {
          console.log(`Contract purchased: ${data.buy.contract_id}`);
        } else {
          console.error('Buy error:', data.error);
        }
        break;
    }
  }

  subscribeTicks(symbol) {
    this.send({
      ticks: symbol,
      subscribe: 1
    });
  }

  buyContract(symbol, contractType, amount, duration) {
    this.send({
      buy: 1,
      subscribe: 1,
      price: amount * 2, // Max price willing to pay
      parameters: {
        amount: amount,
        basis: 'stake',
        contract_type: contractType,
        currency: 'USD',
        duration: duration,
        duration_unit: 'm',
        symbol: symbol
      }
    });
  }

  analyzeAndTrade(tick) {
    // Exemplo: estratégia simples baseada em preço
    const price = tick.quote;
    const symbol = tick.symbol;

    // Buy CALL se preço termina em número par
    if (Math.floor(price * 100) % 2 === 0) {
      this.buyContract(symbol, 'CALL', 1, 1);
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

// Uso do bot
const bot = new DerivBot(1089, 'YOUR_TOKEN_HERE');
bot.connect();
```

## Best Practices

### 1. Gestão de Conexão

```javascript
// Implementar reconexão automática
ws.onclose = () => {
  setTimeout(() => {
    console.log('Reconnecting...');
    connect();
  }, 5000);
};
```

### 2. Gestão de Erro

```javascript
// Sempre verificar erros
if (data.error) {
  console.error(`Error [${data.error.code}]: ${data.error.message}`);
  return;
}
```

### 3. Rate Limiting

```javascript
// Implementar queue para requests
class RequestQueue {
  constructor(maxPerMinute = 150) {
    this.queue = [];
    this.maxPerMinute = maxPerMinute;
    this.timestamps = [];
  }

  canSend() {
    const now = Date.now();
    this.timestamps = this.timestamps.filter(t => now - t < 60000);
    return this.timestamps.length < this.maxPerMinute;
  }

  send(data) {
    if (this.canSend()) {
      this.timestamps.push(Date.now());
      ws.send(JSON.stringify(data));
    } else {
      this.queue.push(data);
    }
  }
}
```

### 4. Gestão de Capital

```javascript
class CapitalManager {
  constructor(initialCapital, maxRisk = 0.02) {
    this.initialCapital = initialCapital;
    this.currentCapital = initialCapital;
    this.maxRisk = maxRisk;
  }

  getStakeAmount() {
    // Nunca arriscar mais que 2% do capital
    return Math.min(
      this.currentCapital * this.maxRisk,
      50 // Máximo absoluto
    );
  }

  updateBalance(newBalance) {
    this.currentCapital = newBalance;
  }
}
```

## Recursos Adicionais

- **API Explorer:** https://api.deriv.com/api-explorer/
- **WebSocket Playground:** https://api.deriv.com/playground/
- **Developer Community:** https://community.deriv.com/
- **GitHub Examples:** https://github.com/deriv-com/
- **Official Documentation:** https://developers.deriv.com/

---

**Versão:** 1.0
**Última Atualização:** Janeiro 2025
**Autor:** Synth Bot Buddy Documentation
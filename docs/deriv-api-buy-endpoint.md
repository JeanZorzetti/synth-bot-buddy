# Deriv API - Buy Endpoint Documentation

## Visão Geral

O endpoint `buy` da API Deriv é usado para comprar contratos de trading. Este documento fornece informações detalhadas sobre como usar este endpoint, incluindo parâmetros, exemplos e estruturas de response.

**Fonte:** https://api.deriv.com/api-explorer/#buy

## Endpoint Information

- **Method:** WebSocket Message
- **Endpoint:** `buy`
- **Send capability:** ✅ Yes
- **Subscribe capability:** ❌ No (but can subscribe to updates)
- **Scope:** Trade
- **Authentication:** Required

## Request Structure

### Basic Request Format

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
    "duration": 1,
    "duration_unit": "m",
    "symbol": "R_100"
  }
}
```

## Parameters

### Main Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `buy` | integer | ✅ | Must be `1` to indicate buy request |
| `subscribe` | integer | ❌ | Set to `1` to receive contract updates |
| `price` | number | ✅ | Maximum price willing to pay for the contract |
| `parameters` | object | ✅ | Contract parameters object |

### Parameters Object

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `amount` | number | ✅ | Contract amount |
| `basis` | string | ✅ | `"stake"` or `"payout"` |
| `contract_type` | string | ✅ | Contract type (e.g., "CALL", "PUT") |
| `currency` | string | ✅ | Currency code (e.g., "USD", "EUR") |
| `duration` | integer | ✅ | Contract duration |
| `duration_unit` | string | ✅ | Time unit: `"s"` (seconds), `"m"` (minutes), `"h"` (hours), `"d"` (days) |
| `symbol` | string | ✅ | Trading symbol (e.g., "R_100", "R_75", "R_50") |
| `barrier` | string | ❌ | Barrier level for certain contract types |
| `barrier2` | string | ❌ | Second barrier for range contracts |

## Contract Types

### Supported Contract Types

| Type | Description |
|------|-------------|
| `CALL` | Higher/Call option |
| `PUT` | Lower/Put option |
| `DIGITEVEN` | Even/Odd - Even |
| `DIGITODD` | Even/Odd - Odd |
| `DIGITOVER` | Digit Over |
| `DIGITUNDER` | Digit Under |
| `ONETOUCH` | Touch/No Touch - Touch |
| `NOTOUCH` | Touch/No Touch - No Touch |
| `RANGE` | In/Out - Stays Between |
| `UPORDOWN` | In/Out - Goes Outside |

## Symbols

### Synthetic Indices

| Symbol | Description |
|--------|-------------|
| `R_10` | Volatility 10 Index |
| `R_25` | Volatility 25 Index |
| `R_50` | Volatility 50 Index |
| `R_75` | Volatility 75 Index |
| `R_100` | Volatility 100 Index |
| `JD10` | Jump 10 Index |
| `JD25` | Jump 25 Index |
| `RDBULL` | Bull Market Index |
| `RDBEAR` | Bear Market Index |

## Examples

### Example 1: Simple Call Contract

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

### Example 2: Digit Contract

```json
{
  "buy": 1,
  "subscribe": 1,
  "price": 5,
  "parameters": {
    "amount": 5,
    "basis": "stake",
    "contract_type": "DIGITOVER",
    "currency": "USD",
    "duration": 1,
    "duration_unit": "m",
    "symbol": "R_75",
    "barrier": "5"
  }
}
```

## Response Structure

### Successful Response

```json
{
  "buy": {
    "balance_after": 9990,
    "buy_price": 10,
    "contract_id": 123456789,
    "longcode": "Win payout if Volatility 100 Index is strictly higher than entry spot at 5 minutes after contract start time.",
    "payout": 19.54,
    "purchase_time": 1640995200,
    "shortcode": "CALL_R_100_10_1640995200_1640995500_S0P_0",
    "start_time": 1640995200,
    "transaction_id": 987654321
  },
  "echo_req": {
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
  },
  "msg_type": "buy",
  "req_id": 1
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `balance_after` | number | Account balance after purchase |
| `buy_price` | number | Actual price paid for the contract |
| `contract_id` | integer | Unique contract identifier |
| `longcode` | string | Human-readable contract description |
| `payout` | number | Potential payout amount |
| `purchase_time` | integer | Unix timestamp of purchase |
| `shortcode` | string | Short contract identifier |
| `start_time` | integer | Unix timestamp when contract starts |
| `transaction_id` | integer | Transaction identifier |

### Error Response

```json
{
  "error": {
    "code": "InvalidContractType",
    "message": "Invalid contract type."
  },
  "echo_req": {
    "buy": 1,
    "parameters": {
      "contract_type": "INVALID"
    }
  },
  "msg_type": "buy",
  "req_id": 1
}
```

## Code Examples

### JavaScript/WebSocket

```javascript
const WebSocket = require('ws');
const app_id = 1089; // Replace with your app_id
const ws = new WebSocket(`wss://ws.binaryws.com/websockets/v3?app_id=${app_id}`);

ws.onopen = function(evt) {
  // Authenticate first
  ws.send(JSON.stringify({
    "authorize": "YOUR_API_TOKEN"
  }));
};

ws.onmessage = function(evt) {
  const data = JSON.parse(evt.data);

  if (data.msg_type === 'authorize' && !data.error) {
    // Buy contract after successful authentication
    ws.send(JSON.stringify({
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
    }));
  }

  if (data.msg_type === 'buy') {
    console.log('Contract purchased:', data.buy);
  }
};
```

### Python

```python
import json
import websocket

def on_open(ws):
    # Authenticate first
    auth_data = json.dumps({"authorize": "YOUR_API_TOKEN"})
    ws.send(auth_data)

def on_message(ws, message):
    data = json.loads(message)

    if data.get('msg_type') == 'authorize' and not data.get('error'):
        # Buy contract after successful authentication
        buy_data = json.dumps({
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
        })
        ws.send(buy_data)

    if data.get('msg_type') == 'buy':
        print(f"Contract purchased: {data['buy']}")

app_id = 1089  # Replace with your app_id
ws = websocket.WebSocketApp(
    f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}",
    on_open=on_open,
    on_message=on_message
)

ws.run_forever()
```

## Error Codes

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `AuthorizationRequired` | API token not provided | Include valid API token |
| `InvalidToken` | Invalid API token | Check token validity |
| `InvalidContractType` | Invalid contract type | Use supported contract types |
| `InvalidSymbol` | Invalid trading symbol | Use valid symbol codes |
| `InvalidDuration` | Invalid duration value | Check duration limits |
| `InsufficientBalance` | Not enough balance | Ensure sufficient account balance |
| `MarketClosed` | Market is closed | Trade during market hours |
| `PriceChanged` | Price has changed | Retry with updated price |

## Best Practices

1. **Authentication**: Always authenticate before making buy requests
2. **Price Validation**: Check current prices before submitting buy orders
3. **Error Handling**: Implement proper error handling for all scenarios
4. **Subscription**: Use subscribe parameter to receive real-time updates
5. **Balance Check**: Verify account balance before placing orders
6. **Rate Limiting**: Respect API rate limits to avoid blocking

## Notes

- All monetary values are in the account's base currency
- Contract purchase is immediate and irreversible
- Always test with demo accounts before using real money
- Market conditions affect contract availability and pricing
- Some contract types may not be available for all symbols

---

**Documentation Version:** 1.0
**Last Updated:** January 2025
**Source:** https://api.deriv.com/api-explorer/#buy
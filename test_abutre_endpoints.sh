#!/bin/bash

# ABUTRE BOT - API ENDPOINTS TEST SUITE
# Testa todos os endpoints de eventos do Deriv Bot XML

API_BASE="http://localhost:8000/api/abutre/events"
# Para produção: API_BASE="https://botderivapi.roilabs.com.br/api/abutre/events"

echo "=========================================="
echo "ABUTRE BOT - API ENDPOINTS TEST"
echo "=========================================="
echo ""

# Test 1: POST Candle Event
echo "1️⃣  Testing POST /candle"
curl -X POST "${API_BASE}/candle" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:30:00Z",
    "symbol": "1HZ100V",
    "open": 663.50,
    "high": 663.92,
    "low": 663.12,
    "close": 663.60,
    "color": 1
  }' \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 2: POST Trigger Event
echo "2️⃣  Testing POST /trigger"
curl -X POST "${API_BASE}/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:30:05Z",
    "streak_count": 8,
    "direction": "GREEN"
  }' \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 3: POST Trade Opened Event
echo "3️⃣  Testing POST /trade_opened"
curl -X POST "${API_BASE}/trade_opened" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:31:00Z",
    "trade_id": "trade_1703271060",
    "direction": "PUT",
    "stake": 1.0,
    "level": 1,
    "contract_id": "12345678"
  }' \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 4: POST Trade Closed Event (WIN)
echo "4️⃣  Testing POST /trade_closed (WIN)"
curl -X POST "${API_BASE}/trade_closed" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:32:00Z",
    "trade_id": "trade_1703271060",
    "result": "WIN",
    "profit": 0.95,
    "balance": 10001.95,
    "max_level_reached": 1
  }' \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 5: POST Balance Event
echo "5️⃣  Testing POST /balance"
curl -X POST "${API_BASE}/balance" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-12-22T18:32:00Z",
    "balance": 10001.95
  }' \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 6: GET Stats
echo "6️⃣  Testing GET /stats"
curl -X GET "${API_BASE}/stats" \
  -H "Accept: application/json" \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 7: GET Trades
echo "7️⃣  Testing GET /trades?limit=10"
curl -X GET "${API_BASE}/trades?limit=10" \
  -H "Accept: application/json" \
  -w "\nStatus: %{http_code}\n\n"

echo "---"

# Test 8: GET Balance History
echo "8️⃣  Testing GET /balance_history?limit=100"
curl -X GET "${API_BASE}/balance_history?limit=100" \
  -H "Accept: application/json" \
  -w "\nStatus: %{http_code}\n\n"

echo "=========================================="
echo "✅ All tests completed!"
echo "=========================================="

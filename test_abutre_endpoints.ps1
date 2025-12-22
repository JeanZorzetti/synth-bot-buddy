# ABUTRE BOT - API ENDPOINTS TEST SUITE (PowerShell)
# Testa todos os endpoints de eventos do Deriv Bot XML

$API_BASE = "http://localhost:8000/api/abutre/events"
# Para produção: $API_BASE = "https://botderivapi.roilabs.com.br/api/abutre/events"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ABUTRE BOT - API ENDPOINTS TEST" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: POST Candle Event
Write-Host "1️⃣  Testing POST /candle" -ForegroundColor Yellow
$candleBody = @{
    timestamp = "2025-12-22T18:30:00Z"
    symbol = "1HZ100V"
    open = 663.50
    high = 663.92
    low = 663.12
    close = 663.60
    color = 1
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/candle" -Method Post -Body $candleBody -ContentType "application/json"
    Write-Host "✅ Success: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 2: POST Trigger Event
Write-Host "2️⃣  Testing POST /trigger" -ForegroundColor Yellow
$triggerBody = @{
    timestamp = "2025-12-22T18:30:05Z"
    streak_count = 8
    direction = "GREEN"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/trigger" -Method Post -Body $triggerBody -ContentType "application/json"
    Write-Host "✅ Success: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 3: POST Trade Opened Event
Write-Host "3️⃣  Testing POST /trade_opened" -ForegroundColor Yellow
$tradeOpenedBody = @{
    timestamp = "2025-12-22T18:31:00Z"
    trade_id = "trade_1703271060"
    direction = "PUT"
    stake = 1.0
    level = 1
    contract_id = "12345678"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/trade_opened" -Method Post -Body $tradeOpenedBody -ContentType "application/json"
    Write-Host "✅ Success: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 4: POST Trade Closed Event (WIN)
Write-Host "4️⃣  Testing POST /trade_closed (WIN)" -ForegroundColor Yellow
$tradeClosedBody = @{
    timestamp = "2025-12-22T18:32:00Z"
    trade_id = "trade_1703271060"
    result = "WIN"
    profit = 0.95
    balance = 10001.95
    max_level_reached = 1
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/trade_closed" -Method Post -Body $tradeClosedBody -ContentType "application/json"
    Write-Host "✅ Success: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 5: POST Balance Event
Write-Host "5️⃣  Testing POST /balance" -ForegroundColor Yellow
$balanceBody = @{
    timestamp = "2025-12-22T18:32:00Z"
    balance = 10001.95
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/balance" -Method Post -Body $balanceBody -ContentType "application/json"
    Write-Host "✅ Success: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 6: GET Stats
Write-Host "6️⃣  Testing GET /stats" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/stats" -Method Get
    Write-Host "✅ Success: $($response.data | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 7: GET Trades
Write-Host "7️⃣  Testing GET /trades?limit=10" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/trades?limit=10" -Method Get
    Write-Host "✅ Success: Found $($response.data.Count) trades" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

# Test 8: GET Balance History
Write-Host "8️⃣  Testing GET /balance_history?limit=100" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/balance_history?limit=100" -Method Get
    Write-Host "✅ Success: Found $($response.data.Count) balance snapshots" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "---`n"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ All tests completed!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

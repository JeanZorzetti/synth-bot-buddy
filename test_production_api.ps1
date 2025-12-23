# ABUTRE BOT - PRODUCTION API TEST
# Popula dados de teste no servidor de produção

$API_BASE = "https://botderivapi.roilabs.com.br/api/abutre/events"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "ABUTRE BOT - PRODUCTION TEST" -ForegroundColor Cyan
Write-Host "API: $API_BASE" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Enviar 1 trade de exemplo completo
Write-Host "1. Enviando candle de exemplo..." -ForegroundColor Yellow
$candle = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    symbol = "1HZ100V"
    open = 663.50
    high = 663.92
    low = 663.12
    close = 663.60
    color = 1
} | ConvertTo-Json

try {
    $r1 = Invoke-RestMethod -Uri "$API_BASE/candle" -Method Post -Body $candle -ContentType "application/json"
    Write-Host "   ✅ $($r1.message)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Erro: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2. Enviando trigger (8 velas verdes)..." -ForegroundColor Yellow
$trigger = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    streak_count = 8
    direction = "GREEN"
} | ConvertTo-Json

try {
    $r2 = Invoke-RestMethod -Uri "$API_BASE/trigger" -Method Post -Body $trigger -ContentType "application/json"
    Write-Host "   ✅ $($r2.message)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Erro: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Adicionando 10 trades de teste..." -ForegroundColor Yellow

for ($i = 1; $i -le 10; $i++) {
    $timestamp = (Get-Date).AddMinutes(-$i * 2).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    # Trade Opened
    $opened = @{
        timestamp = $timestamp
        trade_id = "prod_trade_$i"
        direction = if ($i % 2 -eq 0) { "CALL" } else { "PUT" }
        stake = 1.0
        level = 1
        contract_id = "contract_$i"
    } | ConvertTo-Json

    $null = Invoke-RestMethod -Uri "$API_BASE/trade_opened" -Method Post -Body $opened -ContentType "application/json"

    # Trade Closed
    $result = if ($i % 3 -eq 0) { "LOSS" } else { "WIN" }
    $profit = if ($result -eq "WIN") { 0.95 } else { -1.0 }
    $balance = 10000.0 + ($i * 0.50) + $profit

    $closedTime = (Get-Date).AddMinutes(-$i * 2 + 0.5).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    $closed = @{
        timestamp = $closedTime
        trade_id = "prod_trade_$i"
        result = $result
        profit = $profit
        balance = $balance
        max_level_reached = 1
    } | ConvertTo-Json

    $null = Invoke-RestMethod -Uri "$API_BASE/trade_closed" -Method Post -Body $closed -ContentType "application/json"

    $emoji = if ($result -eq "WIN") { "✅" } else { "❌" }
    Write-Host "   Trade $i`: $emoji $result" -ForegroundColor $(if ($result -eq "WIN") { "Green" } else { "Red" })
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "4. Verificando estatísticas..." -ForegroundColor Yellow

try {
    $stats = Invoke-RestMethod -Uri "$API_BASE/stats"

    Write-Host ""
    Write-Host "📊 ESTATÍSTICAS:" -ForegroundColor Cyan
    Write-Host "   Total Trades: $($stats.data.total_trades)" -ForegroundColor White
    Write-Host "   Wins: $($stats.data.wins) | Losses: $($stats.data.losses)" -ForegroundColor White
    Write-Host "   Win Rate: $([math]::Round($stats.data.win_rate_pct, 2))%" -ForegroundColor $(if ($stats.data.win_rate_pct -ge 50) { "Green" } else { "Red" })
    Write-Host "   Balance: `$$($stats.data.current_balance)" -ForegroundColor Cyan
    Write-Host "   ROI: $([math]::Round($stats.data.roi_pct, 2))%" -ForegroundColor $(if ($stats.data.roi_pct -gt 0) { "Green" } else { "Red" })

} catch {
    Write-Host "   ❌ Erro ao buscar stats: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "✅ DADOS POPULADOS COM SUCESSO!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Acesse agora: https://botderiv.roilabs.com.br/abutre" -ForegroundColor Cyan
Write-Host ""

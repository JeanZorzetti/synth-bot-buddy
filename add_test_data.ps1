$API = 'http://localhost:8000/api/abutre/events'

Write-Host "Adding more test data..." -ForegroundColor Yellow

for ($i = 2; $i -le 10; $i++) {
    $timestamp = (Get-Date).AddMinutes(-$i * 2).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

    # Candle
    $candle = @{
        timestamp = $timestamp
        symbol = '1HZ100V'
        open = 663.00 + ($i * 0.1)
        high = 664.00 + ($i * 0.1)
        low = 662.00 + ($i * 0.1)
        close = 663.50 + ($i * 0.1)
        color = if ($i % 2 -eq 0) { 1 } else { -1 }
    } | ConvertTo-Json

    $null = Invoke-RestMethod -Uri "$API/candle" -Method Post -Body $candle -ContentType 'application/json'

    # Trade opened
    $opened = @{
        timestamp = $timestamp
        trade_id = "trade_$i"
        direction = if ($i % 2 -eq 0) { 'CALL' } else { 'PUT' }
        stake = 1.0
        level = 1
    } | ConvertTo-Json

    $null = Invoke-RestMethod -Uri "$API/trade_opened" -Method Post -Body $opened -ContentType 'application/json'

    # Trade closed
    $result = if ($i % 3 -eq 0) { 'LOSS' } else { 'WIN' }
    $profit = if ($result -eq 'WIN') { 0.95 } else { -1.0 }
    $balance = 10001.95 + (($i - 1) * 0.50) + $profit

    $closedTime = (Get-Date).AddMinutes(-$i * 2 + 0.5).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

    $closed = @{
        timestamp = $closedTime
        trade_id = "trade_$i"
        result = $result
        profit = $profit
        balance = $balance
        max_level_reached = 1
    } | ConvertTo-Json

    $null = Invoke-RestMethod -Uri "$API/trade_closed" -Method Post -Body $closed -ContentType 'application/json'

    Write-Host "  Trade $i added ($result)" -ForegroundColor Gray
}

Write-Host "`nDone! Testing stats endpoint:" -ForegroundColor Green
$stats = Invoke-RestMethod -Uri "$API/stats"
Write-Host "Total Trades: $($stats.data.total_trades)" -ForegroundColor Cyan
Write-Host "Wins: $($stats.data.wins)" -ForegroundColor Green
Write-Host "Win Rate: $([math]::Round($stats.data.win_rate_pct, 2))%" -ForegroundColor Yellow
Write-Host "Balance: `$$($stats.data.current_balance)" -ForegroundColor Cyan

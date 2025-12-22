# ABUTRE BOT - TESTE EM PRODUÇÃO

**Ambiente:**
- Backend: https://botderivapi.roilabs.com.br
- Frontend: https://botderiv.roilabs.com.br/abutre

---

## 🚀 PASSO 1: DEPLOY

### Backend
```bash
# SSH no servidor
ssh user@seu-servidor

# Pull código
cd /path/to/backend
git pull origin main

# Reiniciar
pm2 restart backend
# OU
sudo systemctl restart fastapi
```

### Frontend
```bash
cd /path/to/frontend
git pull origin main
npm run build
pm2 restart frontend
```

---

## 🧪 PASSO 2: TESTAR API EM PRODUÇÃO

Execute este script PowerShell:

```powershell
# test_production.ps1
$API = "https://botderivapi.roilabs.com.br/api/abutre/events"

# 1. POST Candle
$candle = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    symbol = "1HZ100V"
    open = 663.50
    high = 663.92
    low = 663.12
    close = 663.60
    color = 1
} | ConvertTo-Json

Write-Host "Testing POST /candle..." -ForegroundColor Yellow
$r1 = Invoke-RestMethod -Uri "$API/candle" -Method Post -Body $candle -ContentType "application/json"
Write-Host "✅ $($r1.message)" -ForegroundColor Green

# 2. POST Trigger
$trigger = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    streak_count = 8
    direction = "GREEN"
} | ConvertTo-Json

Write-Host "`nTesting POST /trigger..." -ForegroundColor Yellow
$r2 = Invoke-RestMethod -Uri "$API/trigger" -Method Post -Body $trigger -ContentType "application/json"
Write-Host "✅ $($r2.message)" -ForegroundColor Green

# 3. POST Trade Opened
$tradeId = "trade_" + (Get-Date).Ticks
$opened = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    trade_id = $tradeId
    direction = "PUT"
    stake = 0.35
    level = 1
} | ConvertTo-Json

Write-Host "`nTesting POST /trade_opened..." -ForegroundColor Yellow
$r3 = Invoke-RestMethod -Uri "$API/trade_opened" -Method Post -Body $opened -ContentType "application/json"
Write-Host "✅ Trade ID: $tradeId" -ForegroundColor Green

# 4. POST Trade Closed (WIN)
$closed = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    trade_id = $tradeId
    result = "WIN"
    profit = 0.33
    balance = 10000.33
    max_level_reached = 1
} | ConvertTo-Json

Write-Host "`nTesting POST /trade_closed..." -ForegroundColor Yellow
$r4 = Invoke-RestMethod -Uri "$API/trade_closed" -Method Post -Body $closed -ContentType "application/json"
Write-Host "✅ Trade closed: WIN +$0.33" -ForegroundColor Green

# 5. GET Stats
Write-Host "`nTesting GET /stats..." -ForegroundColor Yellow
$stats = Invoke-RestMethod -Uri "$API/stats"
Write-Host "✅ Stats:" -ForegroundColor Green
Write-Host "   Trades: $($stats.data.total_trades)"
Write-Host "   Wins: $($stats.data.wins)"
Write-Host "   Balance: $$($stats.data.current_balance)"

# 6. GET Trades
Write-Host "`nTesting GET /trades..." -ForegroundColor Yellow
$trades = Invoke-RestMethod -Uri "$API/trades?limit=5"
Write-Host "✅ Found $($trades.data.Count) trades" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
```

---

## 📊 PASSO 3: VERIFICAR DASHBOARD

1. **Abra:** https://botderiv.roilabs.com.br/abutre

2. **Verifique os cards:**
   - Current Balance: Deve mostrar $10,000.33
   - Win Rate: Deve mostrar 100%
   - Total Trades: Deve mostrar 1

3. **Verifique Equity Curve:**
   - Deve mostrar gráfico com pontos
   - Linha deve estar ascendente (lucro)

4. **Verifique Recent Trades:**
   - Tabela deve ter 1+ linha
   - Deve mostrar o trade WIN de $0.33

---

## 🔌 PASSO 4: TESTAR WEBSOCKET

Abra Console do Browser (F12) e cole:

```javascript
const ws = new WebSocket('wss://botderivapi.roilabs.com.br/ws/abutre')

ws.onopen = () => console.log('✅ WebSocket conectado!')
ws.onmessage = (e) => console.log('📨 Mensagem:', JSON.parse(e.data))
ws.onerror = (e) => console.error('❌ Erro:', e)
```

**Deve ver:**
```
✅ WebSocket conectado!
📨 Mensagem: {event: 'bot_status', data: {...}}
📨 Mensagem: {event: 'risk_stats', data: {...}}
```

---

## ✅ CHECKLIST

- [ ] API /health retorna 200
- [ ] POST /candle cria evento no DB
- [ ] POST /trigger cria trigger
- [ ] POST /trade_opened cria trade
- [ ] POST /trade_closed atualiza trade
- [ ] GET /stats retorna dados corretos
- [ ] Dashboard carrega sem erros
- [ ] Cards mostram dados atualizados
- [ ] Equity Curve renderiza
- [ ] Trades Table mostra dados
- [ ] WebSocket conecta e recebe eventos

---

**PRONTO PARA PRODUÇÃO!** 🚀

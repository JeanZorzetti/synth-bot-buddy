# 🚀 ABUTRE BOT - QUICK START PRODUÇÃO

## Status Atual
✅ Backend API implementado (8 endpoints)
✅ Frontend dashboard atualizado
✅ Scripts de teste criados
✅ XML bot corrigido e simplificado
✅ Código commitado e enviado para GitHub

---

## 🎯 PRÓXIMOS PASSOS (EXECUTAR AGORA)

### 1. Deploy Backend
```bash
ssh user@botderivapi.roilabs.com.br
cd /path/to/backend
git pull origin main
pm2 restart backend
# OU
sudo systemctl restart fastapi
```

### 2. Deploy Frontend
```bash
ssh user@botderiv.roilabs.com.br
cd /path/to/frontend
git pull origin main
npm run build
pm2 restart frontend
```

### 3. Testar API em Produção

**Windows PowerShell:**
```powershell
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main
.\test_abutre_endpoints.ps1
```

**Linux/Mac:**
```bash
cd ~/synth-bot-buddy-main
chmod +x test_abutre_endpoints.sh
./test_abutre_endpoints.sh
```

### 4. Verificar Dashboard
Acesse: https://botderiv.roilabs.com.br/abutre

**Deve ver:**
- ✅ Cards atualizados (Balance, Win Rate, Total Trades)
- ✅ Equity Curve com dados
- ✅ Tabela de trades populada
- ✅ WebSocket conectado (Console do browser: F12)

---

## 📊 ENDPOINTS DISPONÍVEIS

### Base URL
```
https://botderivapi.roilabs.com.br/api/abutre/events
```

### POST Endpoints (XML Bot → Backend)
| Endpoint | Descrição | Body Example |
|----------|-----------|--------------|
| `/candle` | Novo candle fechado | `{timestamp, symbol, open, high, low, close, color}` |
| `/trigger` | Trigger de Abutre detectado | `{timestamp, streak_count, direction}` |
| `/trade_opened` | Trade aberto (martingale level) | `{timestamp, trade_id, direction, stake, level}` |
| `/trade_closed` | Trade fechado | `{timestamp, trade_id, result, profit, balance}` |
| `/balance` | Atualização de saldo | `{timestamp, balance}` |

### GET Endpoints (Dashboard → Backend)
| Endpoint | Descrição | Retorno |
|----------|-----------|---------|
| `/stats` | Estatísticas gerais | Win rate, P&L, total trades, etc |
| `/trades?limit=50` | Últimas trades | Array de trades |
| `/balance_history?limit=1000` | Histórico de saldo | Equity curve data |

---

## 🧪 TESTE RÁPIDO (1 minuto)

```powershell
# 1. Health check
curl https://botderivapi.roilabs.com.br/health

# 2. Post candle de teste
$body = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    symbol = "1HZ100V"
    open = 663.50
    high = 663.92
    low = 663.12
    close = 663.60
    color = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://botderivapi.roilabs.com.br/api/abutre/events/candle" `
  -Method Post -Body $body -ContentType "application/json"

# 3. Verificar stats
Invoke-RestMethod -Uri "https://botderivapi.roilabs.com.br/api/abutre/events/stats"
```

**Resultado esperado:**
```json
{
  "status": "success",
  "message": "Candle event received",
  "data": { "candle_id": 1 }
}
```

---

## 🔴 TROUBLESHOOTING

### Erro: 404 Not Found
**Causa:** Backend não deployado ou rota incorreta
**Fix:** Verificar `git pull` e restart do backend

### Erro: 500 Internal Server Error
**Causa:** Database não inicializado ou variáveis de ambiente faltando
**Fix:**
```bash
# No servidor backend
cd /path/to/backend
python -c "from database.init_db import init_database; init_database()"
```

### Dashboard não atualiza
**Causa:** WebSocket desconectado
**Fix:** Abrir Console (F12) e verificar:
```javascript
const ws = new WebSocket('wss://botderivapi.roilabs.com.br/ws/abutre')
ws.onopen = () => console.log('✅ Conectado')
ws.onmessage = (e) => console.log('📨', JSON.parse(e.data))
```

### XML Bot não está enviando eventos
**Causa:** HTTP POST ainda não implementado no XML (requer Tampermonkey)
**Solução temporária:** Usar scripts de teste para simular eventos

---

## 📁 ARQUIVOS IMPORTANTES

| Arquivo | Descrição |
|---------|-----------|
| `TEST_PRODUCTION.md` | Guia completo de testes |
| `test_abutre_endpoints.ps1` | Script PowerShell de testes |
| `test_abutre_endpoints.sh` | Script Bash de testes |
| `bot_abutre_v100_fixed.xml` | XML corrigido do bot |
| `ABUTRE_DASHBOARD_REFACTOR.md` | Arquitetura completa |
| `backend/api/routes/abutre_events.py` | API endpoints |
| `frontend/src/hooks/useAbutreEvents.ts` | React hook |

---

## ✅ CHECKLIST PRÉ-PRODUÇÃO

- [ ] Deploy backend (`git pull` + restart)
- [ ] Deploy frontend (`git pull` + `npm run build` + restart)
- [ ] Testar `/health` endpoint (200 OK)
- [ ] Rodar `test_abutre_endpoints.ps1` (8/8 testes passam)
- [ ] Dashboard carrega sem erros
- [ ] WebSocket conecta (F12 console)
- [ ] Cards mostram dados de teste
- [ ] Equity curve renderiza

---

## 🎯 PRÓXIMO PASSO: INTEGRAÇÃO COM XML

**Opção 1: Manual (Curto Prazo)**
Usar scripts de teste para simular eventos enquanto roda bot XML no Deriv

**Opção 2: Tampermonkey (Longo Prazo)**
Criar userscript para interceptar eventos do Deriv Bot e enviar HTTP POST
(Ver: `ABUTRE_XML_CHANGELOG.md` seção "4. Tampermonkey Integration")

---

**PRONTO PARA TESTE EM PRODUÇÃO! 🚀**

Execute os 3 passos acima e reporte qualquer erro.

# 🧪 ABUTRE BOT - GUIA DE TESTE LOCAL

## Status Atual ✅
- ✅ Backend rodando em `http://localhost:8000`
- ✅ API com 10 trades de teste
- ✅ Todos endpoints funcionando
- ✅ Dados populados no banco SQLite

---

## 📊 Dados Atuais no Sistema

```
Total Trades: 10
Wins: 7 (70%)
Losses: 3 (30%)
Balance: $10,003.40
```

---

## 🎯 COMO TESTAR O DASHBOARD LOCALMENTE

### Opção 1: Frontend em Modo Dev (RECOMENDADO)

```bash
# 1. Abrir novo terminal
cd frontend

# 2. Instalar dependências (se necessário)
npm install

# 3. Iniciar frontend dev
npm run dev
```

**Resultado esperado:**
```
  VITE v4.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**4. Acessar no navegador:**
```
http://localhost:5173/abutre
```

### Opção 2: Testar API Diretamente (sem frontend)

```powershell
# Stats
curl http://localhost:8000/api/abutre/events/stats

# Trades
curl "http://localhost:8000/api/abutre/events/trades?limit=10"

# Balance History
curl "http://localhost:8000/api/abutre/events/balance_history?limit=100"
```

---

## 🔍 VERIFICAR SE BACKEND ESTÁ RODANDO

```bash
# Testar health
curl http://localhost:8000/health

# Ver stats
curl http://localhost:8000/api/abutre/events/stats
```

**Resultado esperado:**
```json
{
  "status": "success",
  "data": {
    "total_trades": 10,
    "wins": 7,
    "win_rate_pct": 70.0,
    "current_balance": 10003.40,
    ...
  }
}
```

---

## 🧹 LIMPAR DADOS DE TESTE

Se quiser começar do zero:

```bash
cd backend
rm abutre_events.db
```

Depois rode novamente:
```bash
.\test_abutre_endpoints.ps1
```

---

## ❌ TROUBLESHOOTING

### Frontend não está carregando dados

**Problema:** Dashboard mostra "Waiting for Data"

**Causa:** Frontend está configurado para produção (`https://botderivapi.roilabs.com.br`)

**Solução:** Iniciar frontend em modo dev (Opção 1 acima)

### Backend não responde

**Sintoma:** `curl: Failed to connect`

**Verificar:**
```bash
netstat -ano | findstr :8000
```

**Se não aparecer nada, iniciar backend:**
```bash
cd backend
../.venv/Scripts/python.exe -m uvicorn main:app --reload
```

### CORS Error no navegador

**Sintoma:** Console mostra "CORS policy blocked"

**Solução:** Backend já tem CORS habilitado. Certifique-se de acessar via `http://localhost:5173` (não abrir `index.html` diretamente)

---

## 📁 ESTRUTURA DE ARQUIVOS

```
backend/
  ├── abutre_events.db         ← SQLite com dados de teste
  ├── database/
  │   └── abutre_repository.py ← Acesso ao banco
  └── api/
      ├── routes/
      │   └── abutre_events.py ← API endpoints
      └── schemas/
          └── abutre_events.py ← Validação Pydantic

frontend/
  ├── .env.local               ← VITE_API_URL=http://localhost:8000
  └── src/
      ├── hooks/
      │   └── useAbutreEvents.ts ← React hook
      └── pages/
          └── AbutreDashboard.tsx
```

---

## ✅ CHECKLIST

- [x] Backend rodando em localhost:8000
- [x] API retornando dados (/stats, /trades, /balance_history)
- [x] 10 trades de teste populados
- [ ] Frontend rodando em localhost:5173
- [ ] Dashboard mostrando cards atualizados
- [ ] Equity Curve renderizando
- [ ] Trades Table populada

---

## 🚀 PRÓXIMO PASSO

Uma vez validado localmente, siga o [PRODUCTION_QUICK_START.md](PRODUCTION_QUICK_START.md) para deploy em produção.

---

**BACKEND JÁ ESTÁ RODANDO! 🎉**

Agora basta iniciar o frontend:
```bash
cd frontend
npm run dev
```

E acessar: http://localhost:5173/abutre

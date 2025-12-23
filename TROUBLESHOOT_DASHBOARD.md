# TROUBLESHOOTING - Dashboard Não Mostra Trades

## ✅ O QUE JÁ ESTÁ FUNCIONANDO

1. **API Backend**: ✅ Retornando 15 trades
   ```bash
   curl https://botderivapi.roilabs.com.br/api/abutre/events/trades
   # Response: 15 trades
   ```

2. **Frontend Simplificado**: ✅ Deploy feito
   - Commit: `4e16e44`
   - Build: Compilado sem erros

3. **Banco de Dados**: ✅ Populado com dados de teste
   - 15 trades
   - 10 wins, 5 losses
   - Win Rate: 66.67%

---

## ❌ PROBLEMA: "Nenhum trade encontrado"

Possíveis causas:

### 1. Cache do Browser (MAIS PROVÁVEL)

**Solução:**

```
1. Pressione CTRL + SHIFT + R (Windows)
   ou CMD + SHIFT + R (Mac)

2. Ou abra em aba anônima:
   - Chrome: CTRL + SHIFT + N
   - Firefox: CTRL + SHIFT + P

3. Ou limpe o cache:
   - Chrome: Settings → Privacy → Clear browsing data
   - Selecione "Cached images and files"
   - Time range: "Last hour"
```

---

### 2. CORS (Cross-Origin Resource Sharing)

**Verificar no Console do Browser:**

1. Abra https://botderiv.roilabs.com.br/abutre
2. Pressione F12 (Developer Tools)
3. Vá na aba "Console"
4. Procure por erros vermelhos com "CORS"

**Se ver erro de CORS:**

```
Access to fetch at 'https://botderivapi.roilabs.com.br/...'
from origin 'https://botderiv.roilabs.com.br' has been blocked by CORS policy
```

**Solução**: Verificar configuração CORS no backend (ver abaixo)

---

### 3. Frontend Não Rebuild (Build Antigo)

**Verificar:**

Abra o código-fonte da página:
- Botão direito → "View Page Source"
- Procure por "Abutre Bot - Histórico de Trades" no HTML

**Se NÃO encontrar**, o frontend não foi buildado corretamente.

**Solução:**

```bash
# SSH no servidor
ssh user@botderiv.roilabs.com.br

# Ir para pasta do frontend
cd /path/to/frontend

# Pull do código novo
git pull origin main

# LIMPAR build antigo
rm -rf dist/
rm -rf node_modules/.vite/

# Rebuild
npm run build

# Reiniciar servidor
pm2 restart frontend
```

---

### 4. Variável de Ambiente Errada

**Verificar no Console:**

```javascript
// Cole no console do browser (F12)
console.log(import.meta.env.VITE_API_URL)
// Deve mostrar: https://botderivapi.roilabs.com.br
```

**Se mostrar `undefined` ou URL errada:**

1. Verificar se existe `.env.production` no servidor:
   ```bash
   cat /path/to/frontend/.env.production
   # Deve conter: VITE_API_URL=https://botderivapi.roilabs.com.br
   ```

2. Se não existir, criar:
   ```bash
   echo "VITE_API_URL=https://botderivapi.roilabs.com.br" > .env.production
   npm run build
   pm2 restart frontend
   ```

---

### 5. API Request Falhando (Network Error)

**Verificar no Console (F12 → Network):**

1. Abra https://botderiv.roilabs.com.br/abutre
2. Pressione F12
3. Vá na aba "Network"
4. Clique no botão "Atualizar" da página
5. Procure por requisições para `/api/abutre/events/trades`

**Cenários:**

#### ✅ Request 200 OK com dados:
```json
{
  "status": "success",
  "data": [...]
}
```
→ API está funcionando! Problema é no frontend.

#### ❌ Request 404 Not Found:
→ URL errada no frontend ou rota não existe no backend

#### ❌ Request Failed / CORS Error:
→ Problema de CORS no backend

#### ❌ Request nunca acontece:
→ Frontend não está tentando buscar (código não executou)

---

## 🔧 TESTE RÁPIDO - HTML STANDALONE

Baixe e abra este arquivo no browser:
**test_frontend_api.html**

```bash
# No seu PC
start test_frontend_api.html

# Ou abra manualmente no Chrome/Firefox
```

Este arquivo testa a API DIRETAMENTE, sem React/Vite.

**Se funcionar aqui mas não no dashboard:**
→ Problema é no código React (useAbutreEvents hook)

**Se NÃO funcionar aqui:**
→ Problema é na API ou CORS

---

## 🩺 DIAGNÓSTICO COMPLETO

Execute estes comandos no terminal:

```bash
# 1. Verificar se API está retornando trades
curl -s https://botderivapi.roilabs.com.br/api/abutre/events/trades | python -m json.tool | findstr "trade_id"

# Deve mostrar: "trade_id": "trade_1", "trade_id": "trade_2", ...

# 2. Verificar stats
curl -s https://botderivapi.roilabs.com.br/api/abutre/events/stats

# Deve mostrar: "total_trades": 15

# 3. Verificar CORS headers
curl -I https://botderivapi.roilabs.com.br/api/abutre/events/trades

# Procure por: Access-Control-Allow-Origin
```

---

## 🛠️ SOLUÇÕES DEFINITIVAS

### Solução 1: Hard Refresh do Browser

```
CTRL + SHIFT + R (Windows)
CMD + SHIFT + R (Mac)
```

### Solução 2: Limpar Cache + Aba Anônima

```
1. CTRL + SHIFT + DELETE (abrir limpeza de cache)
2. Selecionar "Cached images and files"
3. Limpar
4. Abrir aba anônima: CTRL + SHIFT + N
5. Acessar: https://botderiv.roilabs.com.br/abutre
```

### Solução 3: Rebuild Completo no Servidor

```bash
ssh user@botderiv.roilabs.com.br
cd /path/to/frontend

# Limpar tudo
rm -rf dist/
rm -rf node_modules/.vite/

# Pull do código
git pull origin main

# Rebuild
npm run build

# Restart
pm2 restart frontend
pm2 logs frontend  # Ver logs
```

### Solução 4: Verificar Logs do Frontend

```bash
# No servidor
pm2 logs frontend

# Procurar por erros:
# - "Failed to fetch"
# - "CORS error"
# - "404 Not Found"
```

### Solução 5: Popular o Banco Novamente

Se o banco foi resetado:

```bash
# No seu PC
powershell -File add_test_data.ps1

# Verificar se populou
curl https://botderivapi.roilabs.com.br/api/abutre/events/stats
```

---

## ✅ CHECKLIST DE VERIFICAÇÃO

Execute na ordem:

- [ ] **1. API está retornando dados?**
  ```bash
  curl https://botderivapi.roilabs.com.br/api/abutre/events/trades
  ```
  Deve retornar JSON com `"data": [...]`

- [ ] **2. Frontend simplificado está deployed?**
  - Abrir: https://botderiv.roilabs.com.br/abutre
  - Ver título: "Abutre Bot - Histórico de Trades"
  - Ver botão: "Atualizar"

- [ ] **3. Cache do browser limpo?**
  - CTRL + SHIFT + R
  - Ou aba anônima

- [ ] **4. Console do browser sem erros?**
  - F12 → Console
  - Sem erros vermelhos

- [ ] **5. Network tab mostra requisição para API?**
  - F12 → Network
  - Clicar "Atualizar"
  - Ver request para `/api/abutre/events/trades`
  - Status: 200 OK

- [ ] **6. Resposta da API tem dados?**
  - Clicar na requisição no Network tab
  - Ver "Response" tab
  - JSON deve ter `"data": [...]` com trades

---

## 🎯 PRÓXIMO PASSO

Depois de testar tudo acima, me informe:

1. **Teste HTML funciona?** (test_frontend_api.html)
2. **Console do browser tem erros?** (F12 → Console)
3. **Network tab mostra requisição?** (F12 → Network)
4. **Requisição retorna dados?** (Response tab)

Com essas informações, consigo identificar exatamente onde está o problema!

---

**Atualizado**: 2025-12-23 10:45 GMT

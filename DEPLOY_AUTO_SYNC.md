# DEPLOY COM AUTO-SYNC + POSTGRESQL

## ✅ O QUE FOI IMPLEMENTADO

### 1. Auto-Sync no Startup
- Toda vez que o servidor backend inicia, os trades são sincronizados automaticamente
- Verifica se o banco está vazio antes de sincronizar
- Não bloqueia o startup do servidor (roda em background)

### 2. Suporte a PostgreSQL
- Repository completo para PostgreSQL
- Auto-detecção: usa PostgreSQL se `DATABASE_URL` estiver configurado, senão usa SQLite
- Schema idêntico ao SQLite

### 3. Dados Persistentes
- PostgreSQL garante que dados nunca são perdidos
- Mesmo se o container reiniciar, dados permanecem

---

## 🚀 COMO FAZER DEPLOY NO EASYPANEL

### Passo 1: Configurar Variáveis de Ambiente

No Easypanel, vá em **Settings → Environment Variables** e adicione:

```bash
# PostgreSQL (já configurado no Easypanel)
DATABASE_URL=postgresql://botderiv:PAz0I8**@dados_botderiv:5432/botderiv

# Deriv API
DERIV_API_TOKEN=paE5sSemx3oANLE
DERIV_APP_ID=1089

# Auto-Sync
AUTO_SYNC_ON_STARTUP=true
ABUTRE_API_URL=http://localhost:8000/api/abutre/events

# Outros
INITIAL_CAPITAL=10.0
ENVIRONMENT=production
```

### Passo 2: Instalar Dependências

No terminal do Easypanel:

```bash
cd /app/backend
pip install psycopg2-binary websockets
```

### Passo 3: Fazer Deploy

```bash
# Pull do código
git pull origin main

# Restart do serviço
pm2 restart backend
# OU
systemctl restart backend
```

### Passo 4: Verificar Logs

```bash
# Ver logs do auto-sync
pm2 logs backend | grep "AUTO SYNC"

# Deve aparecer:
# [INFO] AUTO SYNC DERIV - STARTUP
# [INFO] Banco vazio detectado! Iniciando sincronizacao automatica...
# [INFO] Login OK - Conta: VRTC14275364 | Balance: $9919.25
# [INFO] 100 trades encontrados. Sincronizando...
# [INFO] Sincronizacao concluida! Enviados: 100 | Erros: 0
```

---

## 📊 COMO FUNCIONA

### No Startup do Servidor:

```python
# backend/main.py - lifespan function

async def lifespan(app: FastAPI):
    # Servidor inicia
    logger.info("Iniciando aplicação...")

    # Auto-sync roda em background
    from auto_sync_deriv import auto_sync_on_startup
    asyncio.create_task(auto_sync_on_startup())

    # Servidor continua inicializando normalmente
    # ...
```

### Lógica do Auto-Sync:

```python
# backend/auto_sync_deriv.py

async def auto_sync_on_startup():
    # 1. Aguarda 3s para API estar pronta
    await asyncio.sleep(3)

    # 2. Verifica se banco está vazio
    response = requests.get(f"{API_URL}/stats")
    total_trades = response.json()["data"]["total_trades"]

    # 3. Se vazio, sincroniza
    if total_trades == 0:
        await sync_deriv_history()  # Busca últimos 100 trades
```

---

## 🔍 VERIFICAR SE FUNCIONOU

### 1. Verificar Banco PostgreSQL

```bash
# Conectar no PostgreSQL
psql postgresql://botderiv:PAz0I8**@dados_botderiv:5432/botderiv

# Verificar trades
SELECT COUNT(*) FROM abutre_trades;

# Deve retornar: 100
```

### 2. Verificar API

```bash
curl https://botderivapi.roilabs.com.br/api/abutre/events/stats

# Resposta esperada:
{
  "status": "success",
  "data": {
    "total_trades": 100,
    "wins": 49,
    "win_rate_pct": 49.0,
    ...
  }
}
```

### 3. Verificar Dashboard

Acesse: **https://botderiv.roilabs.com.br/abutre**

Deve mostrar **100 trades** automaticamente!

---

## 🛠️ TROUBLESHOOTING

### Problema: Auto-sync não roda

**Solução**: Verificar logs

```bash
pm2 logs backend | grep "auto_sync"

# Se não aparecer nada, verificar se import está correto:
python -c "from backend.auto_sync_deriv import auto_sync_on_startup; print('OK')"
```

### Problema: Erro de conexão PostgreSQL

**Solução**: Verificar DATABASE_URL

```bash
echo $DATABASE_URL
# Deve retornar: postgresql://botderiv:PAz0I8**@dados_botderiv:5432/botderiv

# Testar conexão
psql $DATABASE_URL -c "SELECT 1"
```

### Problema: psycopg2 não instalado

**Solução**:

```bash
pip install psycopg2-binary
# OU se falhar:
apt-get install libpq-dev
pip install psycopg2
```

### Problema: Trades duplicados

**Solução**: Limpar banco antes de sincronizar

```bash
# Conectar no PostgreSQL
psql $DATABASE_URL

# Limpar trades
DELETE FROM abutre_trades;

# Restart do servidor (auto-sync vai popular novamente)
pm2 restart backend
```

---

## 📝 ARQUIVOS CRIADOS

| Arquivo | Descrição |
|---------|-----------|
| `backend/auto_sync_deriv.py` | Script de sincronização automática |
| `backend/database/abutre_repository_postgres.py` | Repository PostgreSQL |
| `backend/database/__init__.py` | Auto-detecção SQLite/PostgreSQL |
| `backend/.env.production` | Variáveis de ambiente de produção |
| `backend/requirements.txt` | Adicionado psycopg2-binary |

---

## ✅ CHECKLIST DE DEPLOY

- [ ] Variáveis de ambiente configuradas no Easypanel
- [ ] DATABASE_URL apontando para PostgreSQL
- [ ] psycopg2-binary instalado
- [ ] Git pull feito (commit `9d90d6f`)
- [ ] Backend reiniciado
- [ ] Logs verificados (auto-sync rodou)
- [ ] API retornando 100 trades
- [ ] Dashboard mostrando 100 trades
- [ ] Testar restart do servidor (trades devem persistir)

---

## 🎯 RESULTADO FINAL

**Antes**:
- Servidor reinicia → Banco SQLite perde dados → Dashboard vazio

**Depois**:
- Servidor reinicia → Auto-sync detecta banco vazio → Sincroniza 100 trades → Dashboard cheio!
- PostgreSQL garante persistência → Dados nunca perdidos!

---

**Última atualização**: 2025-12-23 12:00 GMT
**Commit**: `9d90d6f`
**Status**: ✅ Pronto para deploy

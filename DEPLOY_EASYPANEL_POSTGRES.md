# 🚀 Deploy Automático no Easypanel com PostgreSQL

## 📋 O que acontece no deploy

Quando você fizer deploy no Easypanel, o seguinte processo acontece **automaticamente**:

### 1️⃣ Build do Container
```bash
docker build -t botderiv-backend .
```

### 2️⃣ Instalação de Dependências
```bash
pip install -r requirements.txt
# Inclui psycopg2-binary para PostgreSQL
```

### 3️⃣ Startup do Servidor FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4️⃣ Auto-Sync com Migração Automática

No startup, o sistema executa **automaticamente** (via `main.py`):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(auto_sync_on_startup())
```

Que por sua vez executa **4 passos sequenciais**:

#### **PASSO 1: Migração de Banco de Dados** ✅
```python
from migrate import run_migrations
migration_success = run_migrations()
```

Isso cria **automaticamente** as 4 tabelas no PostgreSQL:
- `abutre_candles` - Histórico de candles
- `abutre_triggers` - Gatilhos de entrada
- `abutre_trades` - Histórico de trades ⭐
- `abutre_balance_history` - Evolução do saldo

**Importante**: Usa `CREATE TABLE IF NOT EXISTS`, então é seguro rodar múltiplas vezes.

#### **PASSO 2: Aguardar API** ⏳
```python
await asyncio.sleep(3)
```

Espera 3 segundos para garantir que todos os endpoints da API estão prontos.

#### **PASSO 3: Verificar se Precisa Sincronizar** 🔍
```python
response = requests.get(f"{ABUTRE_API_URL}/stats")
total_trades = data.get("data", {}).get("total_trades", 0)

if total_trades == 0:
    # Banco vazio, precisa sincronizar!
```

Se `total_trades == 0`, significa que o banco está vazio e precisa ser populado.

#### **PASSO 4: Sincronizar com Deriv API** 🔄
```python
async with websockets.connect(DERIV_WS_URL) as ws:
    # 1. Login com token
    await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))

    # 2. Buscar últimos 100 trades
    await ws.send(json.dumps({
        "profit_table": 1,
        "limit": 100,
        "sort": "DESC"
    }))

    # 3. Enviar cada trade para API
    for tx in transactions:
        requests.post(f"{ABUTRE_API_URL}/trade_opened", json=trade_opened)
        requests.post(f"{ABUTRE_API_URL}/trade_closed", json=trade_closed)
```

## 🔧 Variáveis de Ambiente Necessárias

Configure no Easypanel:

```bash
# PostgreSQL (Conexão Interna)
DATABASE_URL=postgresql://botderiv:PAzoI8**@dados_botderiv:5432/botderiv

# Deriv API
DERIV_API_TOKEN=paE5sSemx3oANLE
DERIV_APP_ID=99188

# Auto-Sync Config (usar 127.0.0.1 para evitar problemas DNS)
ABUTRE_API_URL=http://127.0.0.1:8000/api/abutre/events
AUTO_SYNC_ON_STARTUP=true
```

## 📊 Verificação Pós-Deploy

### 1. Verificar Logs do Container

No Easypanel, vá em **Logs** e procure por:

```
============================================================
AUTO SYNC DERIV - STARTUP
============================================================
PASSO 1: Verificando/criando tabelas do banco de dados...
✅ Tabelas verificadas/criadas com sucesso!
PASSO 2: Aguardando API ficar pronta...
PASSO 3: Verificando se banco precisa de sincronização...
Banco vazio detectado! Iniciando sincronizacao automatica...
PASSO 4: Sincronizando histórico da Deriv...
Login OK - Conta: VRTC14275364 | Balance: $9919.25
100 trades encontrados. Sincronizando...
Sincronizacao concluida! Enviados: 100 | Erros: 0
✅ Sincronização automática completada com sucesso!
============================================================
```

### 2. Verificar Tabelas no PostgreSQL

Conecte no PostgreSQL externo:

```bash
psql postgresql://botderiv:PAz0I8**@31.97.23.166:5439/botderiv
```

Verifique se as tabelas foram criadas:

```sql
\dt abutre_*

-- Saída esperada:
-- abutre_candles
-- abutre_triggers
-- abutre_trades
-- abutre_balance_history
```

Conte quantos trades foram importados:

```sql
SELECT COUNT(*) as total_trades FROM abutre_trades;

-- Saída esperada:
-- total_trades
-- ------------
-- 100
```

### 3. Verificar Dashboard

Acesse: https://botderiv.roilabs.com.br/abutre

**Deve mostrar**:
- ✅ Tabela com 100 trades reais
- ✅ Dados da sua conta Deriv (VRTC14275364)
- ✅ Sem mensagem "Nenhum trade encontrado"
- ✅ Sem estado de loading infinito

## 🐛 Troubleshooting

### Problema: "Nenhum trade encontrado"

**Causa**: Auto-sync não rodou ou falhou

**Solução**:
1. Verifique logs do container
2. Verifique se `DERIV_API_TOKEN` está correto
3. Force restart do container no Easypanel

### Problema: "Error: relation 'abutre_trades' does not exist"

**Causa**: Migração não rodou

**Solução**:
1. Verifique se `DATABASE_URL` está correto
2. Verifique se PostgreSQL está acessível
3. Rode migração manual:

```bash
# Dentro do container
python migrate.py
```

### Problema: "Loading..." infinito

**Causa**: Bug no frontend (já corrigido)

**Solução**:
1. Fazer rebuild do frontend
2. Limpar cache do browser (Ctrl+Shift+R)

## 📝 Arquivos Importantes

| Arquivo | Função |
|---------|--------|
| `migrate.py` | Cria tabelas automaticamente |
| `auto_sync_deriv.py` | Sincroniza histórico Deriv no startup |
| `database/abutre_repository_postgres.py` | Repository PostgreSQL |
| `main.py` | Chama auto-sync no startup via `lifespan()` |

## ✅ Checklist de Deploy

- [ ] Variáveis de ambiente configuradas no Easypanel
- [ ] PostgreSQL criado e porta 5439 exposta
- [ ] Código commitado no GitHub
- [ ] Deploy feito no Easypanel
- [ ] Logs verificados (auto-sync executou?)
- [ ] Tabelas criadas no PostgreSQL
- [ ] 100 trades importados
- [ ] Dashboard mostrando dados reais

## 🎯 Resultado Esperado

Após o deploy completo:

1. ✅ **Backend inicia** sem erros
2. ✅ **Tabelas criadas** automaticamente no PostgreSQL
3. ✅ **100 trades importados** da Deriv API
4. ✅ **Dashboard** mostra histórico completo
5. ✅ **Dados persistem** mesmo após restart do servidor
6. ✅ **Próximos restarts** não duplicam dados (verifica antes de sincronizar)

## 🔄 Comportamento em Restarts

**Primeiro Startup** (banco vazio):
```
PASSO 1: Criar tabelas ✅
PASSO 2: Aguardar API ✅
PASSO 3: Banco vazio? SIM ✅
PASSO 4: Sincronizar 100 trades ✅
```

**Próximos Startups** (banco com dados):
```
PASSO 1: Tabelas já existem ✅
PASSO 2: Aguardar API ✅
PASSO 3: Banco vazio? NÃO ❌
PASSO 4: SKIP (não precisa sincronizar) ⏭️
```

Isso garante que:
- ✅ Não duplica dados
- ✅ Não faz requests desnecessários à Deriv API
- ✅ Startup é rápido quando banco já tem dados

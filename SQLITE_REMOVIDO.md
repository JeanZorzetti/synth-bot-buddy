# 🔒 SQLite Removido - PostgreSQL Obrigatório

## 🎯 Problema Resolvido

**Antes**: Sistema tinha fallback para SQLite quando `DATABASE_URL` não estava configurada
- ❌ Servidor iniciava com SQLite silenciosamente
- ❌ Tabelas PostgreSQL ficavam vazias
- ❌ Dados salvos em SQLite temporário (perdidos no restart)
- ❌ Difícil diagnosticar o problema

**Agora**: PostgreSQL é OBRIGATÓRIO
- ✅ Servidor **NÃO inicia** sem `DATABASE_URL` configurada
- ✅ Erro claro e explícito no log
- ✅ Impossível rodar acidentalmente com SQLite

---

## 🔧 Mudanças Implementadas

### 1. `backend/database/__init__.py`

**Antes**:
```python
if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    logger.info("Using PostgreSQL database")
    from .abutre_repository_postgres import get_abutre_repository
else:
    logger.info("Using SQLite database")  # ❌ FALLBACK PERIGOSO
    from .abutre_repository import get_abutre_repository
```

**Depois**:
```python
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    logger.error("❌ DATABASE_URL não configurada!")
    raise RuntimeError("DATABASE_URL environment variable is required.")

if not DATABASE_URL.startswith("postgresql"):
    logger.error(f"❌ DATABASE_URL inválida: {DATABASE_URL}")
    raise RuntimeError("Only PostgreSQL is supported.")

logger.info(f"Using PostgreSQL database: {DATABASE_URL.split('@')[1]}")
from .abutre_repository_postgres import get_abutre_repository
```

### 2. `backend/migrate.py`

**Antes**:
```python
if not DATABASE_URL:
    logger.warning("DATABASE_URL não configurada, pulando migrações")
    return False

if not DATABASE_URL.startswith("postgresql"):
    logger.info("Usando SQLite, não precisa de migrações")  # ❌ SILENCIOSO
    return True
```

**Depois**:
```python
if not DATABASE_URL:
    logger.error("❌ DATABASE_URL não configurada!")
    logger.error("Configure a variável de ambiente DATABASE_URL.")
    logger.error("Exemplo: DATABASE_URL=postgresql://user:pass@host:5432/database")
    return False

if not DATABASE_URL.startswith("postgresql"):
    logger.error(f"❌ Apenas PostgreSQL é suportado!")
    logger.error(f"Recebido: {DATABASE_URL}")
    return False
```

### 3. `backend/api/routes/abutre_events.py`

**Antes**:
```python
from database.abutre_repository import get_abutre_repository  # Import direto SQLite
```

**Depois**:
```python
from database import get_abutre_repository  # Import do __init__ (PostgreSQL)
```

---

## 🚨 Comportamento Agora

### Se `DATABASE_URL` não estiver configurada:

**Log**:
```
ERROR:database:❌ DATABASE_URL não configurada! Configure a variável de ambiente.
Traceback (most recent call last):
  File "backend/database/__init__.py", line 14, in <module>
    raise RuntimeError("DATABASE_URL environment variable is required.")
RuntimeError: DATABASE_URL environment variable is required. Please configure PostgreSQL connection.
```

**Resultado**: Servidor **NÃO inicia**

### Se `DATABASE_URL` não for PostgreSQL:

**Log**:
```
ERROR:database:❌ DATABASE_URL inválida: sqlite:///data.db
RuntimeError: Only PostgreSQL is supported. DATABASE_URL must start with 'postgresql://'
```

**Resultado**: Servidor **NÃO inicia**

---

## ✅ Como Configurar Corretamente

### Local (`.env`)
```bash
DATABASE_URL=postgresql://botderiv:PAzo18**@31.97.23.166:5439/botderiv
```

### Easypanel (Environment Variables)
```bash
DATABASE_URL=postgresql://botderiv:PAzo18**@dados_botderiv:5432/botderiv
DERIV_API_TOKEN=paE5sSemx3oANLE
DERIV_APP_ID=99188
ABUTRE_API_URL=http://127.0.0.1:8000/api/abutre/events
AUTO_SYNC_ON_STARTUP=true
ENVIRONMENT=production
```

---

## 🧪 Testar Localmente

### 1. Sem DATABASE_URL (deve falhar)
```bash
cd backend
unset DATABASE_URL  # Linux/Mac
# ou
$env:DATABASE_URL="" # PowerShell

python main.py
```

**Esperado**:
```
ERROR:database:❌ DATABASE_URL não configurada!
RuntimeError: DATABASE_URL environment variable is required.
```

### 2. Com PostgreSQL (deve funcionar)
```bash
export DATABASE_URL="postgresql://botderiv:PAzo18**@31.97.23.166:5439/botderiv"
python main.py
```

**Esperado**:
```
INFO:database:Using PostgreSQL database: 31.97.23.166:5439/botderiv
INFO:migrate:INICIANDO MIGRAÇÕES DO BANCO DE DADOS
INFO:migrate:✅ Migrações completadas com sucesso!
```

---

## 📊 Arquivos Afetados

| Arquivo | Mudança | Status |
|---------|---------|--------|
| `backend/database/__init__.py` | Removido fallback SQLite | ✅ |
| `backend/migrate.py` | Erros claros se não PostgreSQL | ✅ |
| `backend/api/routes/abutre_events.py` | Import correto | ✅ |
| `backend/database/abutre_repository.py` | **NÃO REMOVIDO** (ainda existe) | ⚠️ |

**Nota**: O arquivo `abutre_repository.py` (SQLite) ainda existe no código, mas **não é mais usado**.
Pode ser removido em limpeza futura se necessário.

---

## 🎯 Benefícios

1. ✅ **Falha Rápida**: Erro explícito no startup se configuração errada
2. ✅ **Impossível Usar SQLite**: Não há mais fallback silencioso
3. ✅ **Mensagens Claras**: Log mostra exatamente o que está errado
4. ✅ **Força Boas Práticas**: Deve configurar PostgreSQL no Easypanel
5. ✅ **Evita Perda de Dados**: Não salva em banco temporário por engano

---

## 🔗 Próximo Passo

Agora que SQLite foi removido, no **próximo deploy no Easypanel**:

1. Se `DATABASE_URL` **não** estiver configurada:
   - ❌ Servidor não vai iniciar
   - ❌ Logs vão mostrar erro claro
   - ✅ Você saberá imediatamente que precisa configurar

2. Depois de configurar `DATABASE_URL`:
   - ✅ Servidor inicia normalmente
   - ✅ Tabelas PostgreSQL criadas automaticamente
   - ✅ 100 trades sincronizados da Deriv
   - ✅ Dashboard funcionando

---

**Commit**: `2eb7fd9` - feat: Remover suporte SQLite - PostgreSQL obrigatório

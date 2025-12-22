# ABUTRE BOT - TROUBLESHOOTING

Soluções para problemas comuns durante o setup e execução.

---

## ❌ Erro: `HTTP 401 - Unauthorized`

**Sintoma:**
```
ERROR: Failed to connect: server rejected WebSocket connection: HTTP 401
CRITICAL: Failed to connect to Deriv API!
```

### Causa 1: `app_id` ausente na URL ✅ **RESOLVIDO**

**Problema:**
A Deriv API **EXIGE** o parâmetro `app_id` na URL WebSocket.

**Solução:**
- ✅ Já corrigido no código (commit `e2406b8`)
- URL agora é: `wss://ws.derivws.com/websockets/v3?app_id=1089`
- `app_id=1089` é o ID público de teste da Deriv

**Como verificar:**
```python
from bots.abutre.core.deriv_api_client import DerivAPIClient
from bots.abutre.config import config

print(config.DERIV_APP_ID)  # Deve mostrar: 1089
```

### Causa 2: Token inválido ou expirado

**Sintomas:**
- Erro 401 mesmo após fix do app_id
- Token diferente do esperado

**Soluções:**

#### Opção 1: Gerar novo token DEMO
1. Acessar: https://app.deriv.com/account/api-token
2. Login com sua conta Deriv
3. **IMPORTANTE:** Mudar para conta **DEMO (VRTC)** no topo
4. Criar novo token:
   - Name: `Abutre Bot Demo`
   - Scopes: ✅ Read, ✅ Trade, ✅ Payments, ✅ Trading Information
5. Copiar o token (40+ caracteres)
6. Atualizar `.env`:
   ```bash
   DERIV_API_TOKEN=seu_token_completo_aqui
   ```

#### Opção 2: Variáveis de ambiente no Easypanel
1. Acessar: Easypanel → Seu App → Settings → Environment Variables
2. Adicionar/Atualizar:
   ```
   DERIV_API_TOKEN=seu_token_aqui
   DERIV_APP_ID=1089
   ```
3. Rebuild do container

---

## ❌ Erro: `No module named 'sqlalchemy'`

**Sintoma:**
```
ERROR: No module named 'sqlalchemy'
```

**Causa:**
Dependências do Abutre não instaladas.

**Solução:**
✅ Já corrigido em `backend/requirements.txt`

Dependências adicionadas:
- SQLAlchemy>=2.0.23
- python-socketio>=5.10.0
- python-engineio>=4.8.0
- alembic>=1.13.0

**Se persistir:**
```bash
# No servidor
pip install -r backend/requirements.txt
```

---

## ❌ Erro: `'NoneType' object has no attribute 'listen'`

**Sintoma:**
```
ERROR: 'NoneType' object has no attribute 'listen'
INFO: WebSocket server initialized on port 8000
```

**Causa:**
Conflito de porta WebSocket (bot tentava criar servidor na porta 8000).

**Solução:**
✅ Já corrigido (commit `7804d67`)

Bot agora usa `disable_ws=True` quando gerenciado pelo FastAPI.

---

## ❌ Erro: `ResolutionImpossible` - Conflito de dependências

**Sintoma:**
```
ERROR: Cannot install websockets>=12.0 and python-deriv-api 0.1.6
The conflict is caused by: python-deriv-api 0.1.6 depends on websockets==10.3
```

**Solução:**
✅ Já corrigido em `backend/requirements.txt`

Removido `websockets>=12.0` (instalado automaticamente pelo `python-deriv-api`).

---

## 🔧 Verificação de Health

### 1. Verificar configuração

```python
# No servidor ou localmente
python -c "
from backend.bots.abutre.config import config
print(f'Token: {config.DERIV_API_TOKEN[:5]}...')
print(f'App ID: {config.DERIV_APP_ID}')
print(f'WS URL: {config.DERIV_WS_URL}')
"
```

**Output esperado:**
```
Token: paE5s...
App ID: 1089
WS URL: wss://ws.derivws.com/websockets/v3
```

### 2. Testar conexão manual

```python
import asyncio
from backend.bots.abutre.core.deriv_api_client import DerivAPIClient

async def test():
    client = DerivAPIClient()
    connected = await client.connect()
    print(f"Connected: {connected}")
    if connected:
        print("✅ Conexão OK!")
    await client.disconnect()

asyncio.run(test())
```

### 3. Verificar logs do bot

```bash
# Via API
curl https://botderivapi.roilabs.com.br/api/abutre/status

# Ou acessar dashboard
https://botderiv.rollabs.com.br/abutre
```

---

## 📚 Referências

### Documentação Oficial Deriv API
- **WebSocket API:** https://developers.deriv.com/docs/websockets
- **Python Deriv API:** https://github.com/deriv-com/python-deriv-api
- **Obter API Token:** https://app.deriv.com/account/api-token
- **Registrar App ID:** https://api.deriv.com

### Issues Relacionados
- [WebSocket 401 Unauthorized Error](https://github.com/orgs/deepgram/discussions/1140)
- [Api login authentication issue error 403](https://community.deriv.com/t/api-login-authentication-issue-error-403/88124)
- [Handshake status 401](https://github.com/websocket-client/websocket-client/issues/385)

---

## 🆘 Ainda com problemas?

1. Verificar logs detalhados em `backend/bots/abutre/logs/abutre.log`
2. Executar script de diagnóstico: `python backend/bots/abutre/scripts/test_connection.py`
3. Abrir issue no GitHub com logs completos

---

**Última atualização:** 22/12/2025
**Versão:** 1.0.0

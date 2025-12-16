# 🔍 DEBUG: Por que o código não está em produção?

**Data**: 2025-12-16
**Status**: Investigação ativa

---

## 🎯 Problema

Após 5 commits com correções críticas (41debb3 → ada46ef), o código **não está rodando em produção**:

**Evidências:**
- ✅ Código commitado e pushed para GitHub
- ❌ Endpoint `/api/forward-testing/status` retorna 404
- ❌ Logs não mostram atividade de Forward Testing
- ❌ Apenas logs HTTP aparecem (requisições do frontend)

**Última versão esperada**: `9ec01f0` (inclui health check com git_commit)

---

## 🔎 Passo 1: Verificar Qual Código Está Rodando

Acessar no navegador:
```
https://botderiv.roilabs.com.br/health
```

**O que verificar:**

```json
{
  "status": "healthy",
  "git_commit": "???"  // <- ESTE É O CAMPO CRÍTICO
}
```

### Cenário 1: `git_commit` não existe
**Significa**: Código antigo (antes do commit 9ec01f0)
**Ação**: Deploy não aconteceu, ir para Passo 2

### Cenário 2: `git_commit: "9ec01f0"` ou superior
**Significa**: Código novo está rodando!
**Ação**: Problema não é deployment, ir para Passo 3

### Cenário 3: `git_commit: "f2b2eca"` ou anterior
**Significa**: Deploy parcial, Easypanel pegou commit vazio mas não os anteriores
**Ação**: Force push ou rebuild manual (Passo 2)

---

## 🔧 Passo 2: Forçar Deploy no Easypanel

### Opção A: Rebuild via Easypanel UI

1. Acessar https://easypanel.io
2. Ir em **Projects** → **synth-bot-buddy** (ou nome do projeto)
3. Clicar em **Backend Service**
4. Clicar em **Rebuild** (botão no canto superior direito)
5. Aguardar build logs:
   ```
   ✅ Cloning repository...
   ✅ Pulling latest changes...
   ✅ Building Docker image...
   ✅ Deploying container...
   ```

### Opção B: Verificar Configuração de Auto-Deploy

1. No Easypanel, ir em **Backend Service** → **Settings**
2. Verificar:
   - **Git Branch**: Deve ser `main`
   - **Auto Deploy**: Deve estar ✅ habilitado
   - **Webhook**: Deve ter uma URL `https://easypanel.io/webhooks/...`

3. Se auto-deploy estiver desabilitado:
   - Habilitar
   - Clicar em "Save"
   - Fazer um novo commit dummy para testar

### Opção C: Verificar Webhook no GitHub

1. Acessar https://github.com/JeanZorzetti/synth-bot-buddy/settings/hooks
2. Deve ter um webhook apontando para Easypanel
3. Verificar **Recent Deliveries**:
   - ✅ Status 200: Webhook funcionando
   - ❌ Status 4xx/5xx: Webhook quebrado

Se webhook estiver quebrado:
- Copiar URL do webhook no Easypanel (Settings → Webhook URL)
- Adicionar novo webhook no GitHub com essa URL
- Fazer commit teste

---

## 🐛 Passo 3: Se Código Novo Está Rodando Mas Forward Testing Não

Se `/health` mostrar `git_commit: "9ec01f0"` mas Forward Testing continuar sem funcionar:

### 3.1 Verificar se Forward Testing Está Iniciado

```bash
curl https://botderiv.roilabs.com.br/api/forward-testing/status
```

**Resultado esperado:**
```json
{
  "running": true,
  "symbol": "R_100",
  "start_time": "2025-12-16T..."
}
```

**Se retornar 404:**
- Endpoint não foi registrado (verificar se `main.py` tem o endpoint)
- Backend rodando código antigo (voltar ao Passo 1)

**Se retornar `"running": false`:**
- Forward Testing não foi iniciado
- Fazer POST para iniciar:
  ```bash
  curl -X POST https://botderiv.roilabs.com.br/api/forward-testing/start
  ```

### 3.2 Verificar Logs do Container

No Easypanel Console (ou Easypanel UI → Logs):

```bash
# Logs das últimas 100 linhas
docker logs <container_id> --tail 100

# Ou via Easypanel UI: Backend Service → Logs
```

**O que procurar:**

✅ **Logs saudáveis:**
```
INFO:     Application startup complete.
INFO:     Forward Testing iniciado para R_100
DEBUG:    📊 Solicitando último tick para R_100
DEBUG:    ⏳ Warm-up: Aguardando histórico (50/200)
```

❌ **Logs problemáticos:**
```
ERROR:    Exception in ASGI application
ERROR:    ModuleNotFoundError: No module named 'deriv_api_legacy'
ERROR:    NameError: name 'tick' is not defined
```

### 3.3 Verificar Token Deriv API

```bash
# No Easypanel Console
echo $DERIV_API_TOKEN
```

**Deve retornar**: Token válido começando com `aBcD...`

**Se retornar vazio:**
- Configurar variável de ambiente no Easypanel:
  - Settings → Environment Variables
  - Adicionar `DERIV_API_TOKEN=<seu_token>`
  - Rebuild container

---

## 🧪 Passo 4: Teste Rápido de Conectividade

Se tudo acima estiver OK mas ainda não funcionar:

### 4.1 Testar Deriv API Diretamente

```bash
# No Easypanel Console
cd /app/backend
python3 -c "
import asyncio
from deriv_api_legacy import DerivAPILegacy

async def test():
    api = DerivAPILegacy()
    await api.connect()
    print('✅ Deriv API conectada')
    response = await api.get_latest_tick('R_100')
    print(f'✅ Último tick: {response}')
    await api.disconnect()

asyncio.run(test())
"
```

**Resultado esperado:**
```
✅ Deriv API conectada
✅ Último tick: {'history': {'prices': [105.234], ...}}
```

### 4.2 Testar ML Predictor

```bash
cd /app/backend
python3 -c "
from ml_predictor import MLPredictor
predictor = MLPredictor()
print(f'✅ ML Predictor carregado: {predictor.model is not None}')
"
```

**Resultado esperado:**
```
✅ ML Predictor carregado: True
```

---

## 📊 Diagnóstico Rápido - Checklist

Execute esta verificação na ordem:

- [ ] **1. Código em produção**
  - [ ] `/health` retorna `git_commit: "9ec01f0"` ou superior

- [ ] **2. Forward Testing rodando**
  - [ ] `/api/forward-testing/status` retorna 200 (não 404)
  - [ ] `"running": true` no JSON

- [ ] **3. Logs aparecem**
  - [ ] Logs mostram "Forward Testing iniciado"
  - [ ] Logs mostram ticks sendo coletados

- [ ] **4. Token configurado**
  - [ ] `echo $DERIV_API_TOKEN` retorna valor
  - [ ] `/health` mostra `"deriv_token_configured": true`

**Se TODOS os itens estiverem ✅**: Sistema funcionando, aguardar 33 minutos para warm-up

**Se ALGUM item estiver ❌**: Seguir o passo correspondente acima

---

## 🚨 Solução Emergencial: Deploy Manual

Se nada funcionar, fazer deploy manual:

### No Easypanel Console:

```bash
# 1. Ir para o diretório do app
cd /app

# 2. Puxar código mais recente
git fetch origin main
git reset --hard origin/main

# 3. Reiniciar backend
# (Via Easypanel UI: Backend Service → Restart)
```

---

## 📝 Próximos Passos Após Deploy Bem-Sucedido

1. ✅ Confirmar `/health` mostra `git_commit: "9ec01f0"`
2. ✅ Iniciar Forward Testing: `POST /api/forward-testing/start`
3. ✅ Monitorar logs por 5 minutos (deve mostrar ticks)
4. ⏳ Aguardar 33 minutos (warm-up de 200 ticks)
5. ✅ Validar primeira previsão ML aparece
6. ✅ Validar primeiro trade executa (quando confidence ≥ 60%)

---

**Status Atual**: Aguardando verificação do Passo 1 - Qual código está em produção?

**Última ação**: Commit 9ec01f0 pushed para GitHub com health check atualizado

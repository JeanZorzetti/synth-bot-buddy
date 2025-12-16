# ⚡ RESOLVER DEPLOYMENT AGORA - Guia Executivo

**Status**: Código corrigido e pronto, mas NÃO está em produção
**Objetivo**: Forçar deployment e verificar que está funcionando

---

## 🎯 O QUE FAZER (3 opções, escolha a mais fácil)

### OPÇÃO 1: Força Update via Script (RECOMENDADO)

**No Easypanel Console**, executar:

```bash
cd /app
bash backend/force_update.sh
```

Esse script vai:
- ✅ Fazer `git reset --hard origin/main` (forçar código novo)
- ✅ Verificar se todos os fixes estão presentes
- ✅ Te dizer para reiniciar o backend

Depois:
- Reiniciar backend via Easypanel UI (Services → Backend → Restart)

---

### OPÇÃO 2: Diagnóstico Completo via Python

**No Easypanel Console**, executar:

```bash
cd /app
python backend/check_deployment.py
```

Esse script vai:
- 🔍 Verificar qual commit está rodando
- 🔍 Verificar se fixes estão presentes nos arquivos
- 🔍 Verificar se token Deriv está configurado
- 📋 Te dar ações específicas baseadas no que encontrar

---

### OPÇÃO 3: Manual (se scripts não funcionarem)

**No Easypanel Console**:

```bash
# 1. Forçar código novo
cd /app
git fetch origin main
git reset --hard origin/main

# 2. Verificar commit
git log -1 --format='%h - %s'
# Deve mostrar: 3bd2f36 - feat: Adicionar verificação de versão...

# 3. Reiniciar (via Easypanel UI ou supervisorctl se disponível)
```

**Via Easypanel UI**:
- Services → Backend → Restart

---

## ✅ COMO VERIFICAR QUE FUNCIONOU

Após deployment + restart, abrir no navegador:

### 1. Health Check
```
https://botderiv.roilabs.com.br/health
```

**Procurar:**
```json
{
  "git_commit": "9ec01f0"  // <- ou superior (3bd2f36)
}
```

❌ Se `git_commit` não existir → Código antigo ainda rodando
✅ Se `git_commit: "9ec01f0"` ou superior → Código novo rodando!

### 2. Forward Testing Status
```
https://botderiv.roilabs.com.br/api/forward-testing/status
```

**Procurar:**
```json
{
  "status": "success",
  "data": {
    "code_version": {
      "ticks_history_fix": true,
      "warm_up_filter_fix": true,
      "commit": "9ec01f0"
    }
  }
}
```

❌ Se retornar 404 → Endpoint não existe, código antigo
❌ Se não tiver `code_version` → Código parcialmente atualizado
✅ Se tiver `code_version` com os 2 fixes true → TUDO CERTO!

---

## 🚀 DEPOIS QUE CONFIRMAR QUE ESTÁ RODANDO

### 1. Iniciar Forward Testing

```bash
curl -X POST https://botderiv.roilabs.com.br/api/forward-testing/start
```

**Ou via frontend**: https://botderiv.roilabs.com.br/forward-testing → Clicar "Start"

### 2. Monitorar Logs (primeiros 5 minutos)

Easypanel UI → Services → Backend → Logs

**O que deve aparecer:**
```
INFO:     Forward Testing iniciado para R_100
DEBUG:    📊 Solicitando último tick para R_100
DEBUG:    ⏳ Warm-up: Aguardando histórico (1/200)
DEBUG:    ⏳ Warm-up: Aguardando histórico (2/200)
...
```

✅ Se aparecer isso → FUNCIONA!
❌ Se aparecer "already subscribed" → Código antigo ainda rodando
❌ Se aparecer "name 'tick' is not defined" → Código antigo

### 3. Aguardar Warm-up (33 minutos)

O ML precisa coletar 200 ticks (10 segundos cada) = 33 minutos

Depois:
```
DEBUG:    ✅ Previsão ML: PRICE_UP (confidence: 75%)
INFO:     📈 Trade executado: LONG @ 105.234
```

---

## 🐛 SE AINDA NÃO FUNCIONAR

### Problema: Mesmo após força update, código antigo roda

**Causa provável**: Easypanel usa imagem Docker em cache

**Solução**:
1. Easypanel UI → Services → Backend
2. Clicar em **"Rebuild"** (não só Restart)
3. Aguardar build completo (~2-5 minutos)
4. Verificar `/health` novamente

### Problema: Build falha no Easypanel

**Ver logs do build**:
- Easypanel UI → Services → Backend → Build Logs

**Erros comuns**:
- `ModuleNotFoundError` → requirements.txt desatualizado
- `git error` → Problema de permissão/webhook
- `Dockerfile not found` → Configuração errada no Easypanel

---

## 📊 RESUMO DOS COMMITS

| Commit | Descrição | Status |
|--------|-----------|--------|
| `41debb3` | Fix: Filtrar previsões de warm-up | ✅ Pushed |
| `e493849` | Fix: Rate limiting (remover forget_all loop) | ✅ Pushed |
| `89010a1` | Fix: Subscrição ao conectar | ✅ Pushed |
| `75a1b8e` | Fix: Usar ticks_history (CRÍTICO) | ✅ Pushed |
| `ada46ef` | Fix: NameError tick['symbol'] | ✅ Pushed |
| `f2b2eca` | Trigger redeploy (empty commit) | ✅ Pushed |
| `9ec01f0` | Feat: Health check com git_commit | ✅ Pushed |
| `3bd2f36` | Feat: Verificação de versão em /status | ✅ Pushed |

**Versão esperada em produção**: `3bd2f36` (ou qualquer superior)

---

## 🎯 AÇÃO IMEDIATA

**Escolher uma das 3 opções acima e executar AGORA.**

Depois de executar, me mostrar o resultado de:
```
curl https://botderiv.roilabs.com.br/health | jq '.git_commit'
```

Se retornar `"9ec01f0"` ou `"3bd2f36"` → SUCESSO, código novo rodando!

Se retornar `null` ou error → Deployment não funcionou, tentar Rebuild

---

**Última atualização**: 2025-12-16 (após commits 9ec01f0 e 3bd2f36)
**Scripts criados**: `backend/force_update.sh`, `backend/check_deployment.py`

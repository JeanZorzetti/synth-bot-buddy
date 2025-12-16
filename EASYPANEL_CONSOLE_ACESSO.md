# 🖥️ Como Acessar o Easypanel Console

**Objetivo**: Executar comandos diretamente no servidor onde o backend está rodando

---

## 📍 Passo 1: Acessar o Easypanel

1. Ir para: https://easypanel.io
2. Fazer login
3. Selecionar seu projeto (ex: `synth-bot-buddy` ou `botderiv`)

---

## 📍 Passo 2: Abrir o Console do Backend

### Método 1: Via Services List (mais comum)

1. No menu lateral esquerdo, clicar em **"Services"**
2. Localizar o serviço **"Backend"** (ou nome que você deu)
3. Clicar no serviço Backend
4. No topo da página, clicar na aba **"Console"** ou **"Terminal"**

### Método 2: Via Overview

1. No menu lateral esquerdo, clicar em **"Overview"**
2. Procurar card do **"Backend"**
3. No card, clicar no ícone de terminal (🖥️) ou três pontinhos (⋮) → **"Console"**

---

## 📍 Passo 3: Executar Comandos

Uma vez no console, você verá um terminal preto com prompt tipo:

```
root@abc123:/app#
```

Agora pode executar qualquer um destes comandos:

### Opção A: Force Update (RECOMENDADO)

```bash
bash backend/force_update.sh
```

### Opção B: Diagnóstico Completo

```bash
python backend/check_deployment.py
```

### Opção C: Manual

```bash
git fetch origin main
git reset --hard origin/main
git log -1 --format='%h - %s'
```

---

## 📍 Passo 4: Reiniciar o Backend

**IMPORTANTE**: Após forçar update, você PRECISA reiniciar o backend!

### Via UI (mais fácil)

1. Voltar para página do serviço Backend
2. Clicar no botão **"Restart"** (geralmente no canto superior direito)
3. Aguardar ~10-30 segundos

### Via Console (se disponível)

```bash
supervisorctl restart backend
```

Ou:

```bash
# Se usar PM2
pm2 restart backend

# Se usar systemd
systemctl restart backend
```

---

## 📍 Passo 5: Verificar que Funcionou

Abrir no navegador:

```
https://botderiv.roilabs.com.br/health
```

**Procurar:**
```json
{
  "git_commit": "9ec01f0"  // ou "3bd2f36" ou "1bd1493"
}
```

✅ Se aparecer `git_commit` com valor → **SUCESSO!**
❌ Se não aparecer `git_commit` → Código antigo ainda rodando, tentar Rebuild (próximo passo)

---

## 🔄 ALTERNATIVA: Rebuild Completo (se Force Update não funcionar)

Se após força update + restart o `git_commit` ainda não aparecer:

### Via Easypanel UI

1. Services → Backend
2. Clicar em **"Rebuild"** (pode estar em menu ⋮ ou botão separado)
3. Aguardar build completo (~2-5 minutos)
4. Acompanhar logs do build na aba **"Build Logs"**
5. Quando completar, verificar `/health` novamente

### Possíveis problemas no Build:

**"git: permission denied"**
- Verificar SSH keys/deploy keys configuradas
- Verificar se webhook do GitHub está funcionando

**"requirements.txt: No such file"**
- Dockerfile está apontando para diretório errado
- Verificar configuração do service no Easypanel

**"Module not found"**
- requirements.txt desatualizado
- Adicionar módulo faltando e commitar

---

## 🆘 Se Nada Funcionar

### Verificar Configuração do Service

1. Services → Backend → **"Settings"**
2. Verificar:
   - **Repository**: `https://github.com/JeanZorzetti/synth-bot-buddy`
   - **Branch**: `main`
   - **Auto Deploy**: ✅ habilitado
   - **Build Path**: `/` ou `/backend`
   - **Start Command**: Algo como `uvicorn main:app --host 0.0.0.0 --port 8000`

### Verificar Logs de Deploy

1. Services → Backend → **"Deployments"** ou **"Deploy Logs"**
2. Verificar o último deploy:
   - ✅ Status "Success" → Deploy funcionou
   - ❌ Status "Failed" → Ver mensagem de erro

### Verificar Webhook GitHub

1. GitHub: https://github.com/JeanZorzetti/synth-bot-buddy/settings/hooks
2. Clicar no webhook do Easypanel
3. Ver **"Recent Deliveries"**:
   - ✅ Status 200 → Webhook funcionando
   - ❌ Status 4xx/5xx → Webhook quebrado, recriar

---

## 📞 Resumo Visual

```
1. Easypanel.io
   ↓
2. Login → Projeto
   ↓
3. Services → Backend → Console
   ↓
4. bash backend/force_update.sh
   ↓
5. Services → Backend → Restart
   ↓
6. Verificar: https://botderiv.roilabs.com.br/health
   ↓
7. ✅ git_commit aparece? SUCESSO!
   ❌ git_commit não aparece? Rebuild (Services → Backend → Rebuild)
```

---

**Última atualização**: 2025-12-16
**Versão esperada**: `1bd1493` (ou superior)
**Scripts disponíveis**: `force_update.sh`, `check_deployment.py`

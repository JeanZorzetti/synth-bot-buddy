# 🚀 DEPLOY STATUS & NEXT STEPS

## ✅ FRONTEND - JÁ DEPLOYADO (Vercel)

### URL de Produção
https://botderiv.roilabs.com.br/abutre/history

### Funcionalidades LIVE:
- ✅ Seletor de Período (Presets + Customizado)
- ✅ Paginação (50 trades por página)
- ✅ Ordenação (Mais recentes primeiro)
- ✅ Exportação CSV
- ✅ Sincronização automática ao selecionar período

### Commits Deployed:
- 53cc72d - feat: Add pagination to history page
- d14d9ca - feat: Make custom period button sync before fetch

## ⚠️ BACKEND - PENDENTE (Easypanel)

### Status Atual:
**CÓDIGO PRONTO ✅** | **DEPLOY BLOQUEADO ⚠️**

### Problema:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

Docker build falha ao instalar dependências grandes:
- torch (899 MB)
- nvidia-cudnn-cu12 (706 MB)
- numpy (20 MB)
- scipy, xgboost, etc.

### Commit Pendente Deploy:
- 1bcad67 - fix: Change Deriv API limit from 1000 to 999

---

## 🔧 COMO RESOLVER - 3 OPÇÕES

### OPÇÃO 1: Limpar Cache Docker (MAIS RÁPIDO) ⚡

1. Acessar Easypanel Dashboard
2. Ir em "Services" > "synth-bot-backend"
3. Abrir "Console" ou "Terminal"
4. Executar:

```bash
# Limpar TUDO do Docker (cuidado!)
docker system prune -a -f --volumes

# OU limpar só o necessário:
docker builder prune -a -f    # Remove build cache
docker image prune -a -f      # Remove imagens não usadas
docker container prune -f     # Remove containers parados
docker volume prune -f        # Remove volumes não usados

# Verificar espaço liberado
df -h
```

5. Ir em "Deploy" > "Redeploy" no Easypanel
6. Aguardar build (~5-10 min)

**Espaço esperado a liberar**: 2-5 GB

---

### OPÇÃO 2: Remover ML Dependencies (PERMANENTE) 🔥

Se você **NÃO está usando** as predições de ML em produção, pode simplificar o backend:

1. Editar `backend/requirements.txt`
2. Comentar ou remover estas linhas:

```txt
# ML Dependencies (COMENTAR SE NÃO USAR)
# torch==2.1.1
# xgboost==2.0.3
# scikit-learn==1.3.2
# nvidia-cudnn-cu12==8.9.7.29
# nvidia-cublas-cu12==12.1.3.1
# nvidia-cuda-cupti-cu12==12.1.105
# nvidia-cuda-nvrtc-cu12==12.1.105
# nvidia-cuda-runtime-cu12==12.1.105
# nvidia-cufft-cu12==11.0.2.54
# nvidia-curand-cu12==10.3.2.106
# nvidia-cusolver-cu12==11.4.5.107
# nvidia-cusparse-cu12==12.1.0.106
# nvidia-nccl-cu12==2.20.5
# nvidia-nvjitlink-cu12==12.3.101
# nvidia-nvtx-cu12==12.1.105
# triton==2.1.0
```

3. Commit:
```bash
git add backend/requirements.txt
git commit -m "chore: Remove ML dependencies to reduce build size"
git push
```

4. Easypanel vai rebuildar automaticamente

**Economia de espaço**: ~3 GB

**ATENÇÃO**: Isso desabilita o endpoint `/api/ml/predict`. Se você usa, NÃO faça isso.

---

### OPÇÃO 3: Upgrade Server (MAIS CARO) 💰

1. Ir em Easypanel Dashboard
2. Ir em "Settings" > "Server"
3. Aumentar o disco (ex: de 20GB para 40GB)
4. Aplicar mudanças
5. Redeploy do backend

**Custo extra**: Depende do provedor (AWS/DO/Hetzner)

---

## 📋 CHECKLIST PÓS-DEPLOY

Após resolver o problema de disco e o backend fazer rebuild:

### 1. Verificar Backend Health
```bash
curl https://synth-bot-backend.roilabs.com.br/health
# Deve retornar: {"status": "ok"}
```

### 2. Testar Endpoint de Sync
```bash
curl "https://synth-bot-backend.roilabs.com.br/api/abutre/sync/quick/7"
# Deve retornar JSON com trades_synced
```

### 3. Testar no Frontend
1. Acessar: https://botderiv.roilabs.com.br/abutre/history
2. Clicar em "Última Semana"
3. Verificar se trades aparecem
4. Tentar período 20/12/2025 - 23/12/2025
5. Verificar se trades de 20/12 aparecem agora

### 4. Verificar Logs
No Easypanel Console:
```bash
# Ver logs do backend
docker logs -f <container-name> --tail 100

# Procurar por:
# ✅ "Login OK - Conta: ..."
# ✅ "X trades encontrados no período"
# ❌ "Input validation failed: limit" (não deve mais aparecer!)
```

---

## 🎯 RESULTADO ESPERADO

Após deploy bem-sucedido:

| Funcionalidade | Status Antes | Status Depois |
|---------------|--------------|---------------|
| Frontend Pagination | ✅ OK | ✅ OK |
| Frontend Period Selector | ✅ OK | ✅ OK |
| Backend Sync (limit 999) | ❌ 1000 (erro) | ✅ 999 (OK) |
| Trades de 20/12/2025 | ❌ Não sincroniza | ✅ Sincroniza |
| Warning de período antigo | ❌ Sem aviso | ✅ Com aviso |

---

## 🆘 SE AINDA DER ERRO

### Se o erro persistir após limpar cache:

1. Verificar espaço total do disco:
```bash
df -h /
# Se "Use%" estiver > 90%, precisa de mais espaço
```

2. Verificar tamanho das imagens Docker:
```bash
docker images
docker system df -v
```

3. Última opção: Multi-stage build
   - Criar `Dockerfile` otimizado com multi-stage
   - Reduzir imagem final para apenas runtime (sem build tools)
   - Economia: ~50% do tamanho

### Contato de Suporte:
Se nenhuma opção funcionar:
- Abrir ticket no Easypanel
- Ou considerar migrar para Railway/Render (tem free tier maior)

---

**Criado em**: 2024-12-23  
**Recomendação**: Tentar **OPÇÃO 1** primeiro (mais rápido e sem side effects)

# 🚨 PROBLEMA: Logs em DEBUG (Invisível em Produção)

**Status**: 🔴 Sistema travado mas logs críticos estavam invisíveis
**Fix aplicado**: Commit `44a0283` - Mudado logger.debug → logger.info
**Ação necessária**: REBUILD + RESTART

---

## 📊 O Que Aconteceu

Forward Testing iniciou corretamente:
- ✅ Token Deriv configurado: SIM
- ✅ Modelo ML carregado: xgboost_improved_learning_rate_20251117_160409.pkl
- ✅ Conectado à Deriv API
- ✅ Autenticado (LoginID: VRTC14275364)
- ✅ Subscrições antigas canceladas

MAS depois só apareceram logs HTTP (polling do frontend).

**Causa**: Logs críticos estavam em `logger.debug()` (invisível em produção)

**Evidência**: Não apareceu nos logs:
```
📊 Solicitando último tick para R_100
✅ Resposta recebida da Deriv API
⏳ Warm-up: Aguardando histórico (1/200)
```

---

## 🔧 Ação Necessária

### Opção 1: Deploy Automático (se configurado)

Se você tem deploy automático configurado (GitHub Actions, Easypanel, Railway, Render):

1. **Verificar se já deployou**:
   ```bash
   # Acessar logs do servidor
   # Procurar por: "✅ Conectado e autenticado na Deriv API para dados reais"
   ```

2. **Se NÃO deployou automaticamente**:
   - Acesse o painel do seu provedor (Easypanel/Railway/Render)
   - Clique em "Redeploy" ou "Restart"
   - Aguarde 2-3 minutos

### Opção 2: Deploy Manual (SSH)

Se você tem acesso SSH ao servidor:

```bash
# 1. Conectar ao servidor
ssh usuario@seu-servidor.com

# 2. Navegar para o diretório do projeto
cd /app  # ou path do seu projeto

# 3. Pull das mudanças
git pull origin main

# 4. Verificar commit atual
git log -1 --oneline
# Deve mostrar: e19f5ed fix: Adicionar token ao authorize() em Forward Testing

# 5. Reiniciar backend
# Opção A: Se usando systemd
sudo systemctl restart trading-bot

# Opção B: Se usando Docker
docker-compose restart backend

# Opção C: Se usando uvicorn diretamente
pkill -f "uvicorn main:app"
uvicorn main:app --host 0.0.0.0 --port 8000 &
```

### Opção 3: Plataformas Específicas

#### Easypanel
1. Acessar https://easypanel.io/
2. Ir para o seu projeto "trading-bot"
3. Clicar em "Rebuild & Restart"
4. Aguardar deploy (2-3 minutos)

#### Railway
1. Acessar https://railway.app/
2. Ir para o seu projeto
3. Aba "Deployments"
4. Clicar em "Redeploy" no último deployment

#### Render
1. Acessar https://render.com/
2. Ir para o seu Web Service
3. Clicar em "Manual Deploy" → "Deploy latest commit"

---

## ✅ Como Verificar se Funcionou

### 1. Verificar Logs em Tempo Real

Acesse a página: https://botderiv.roilabs.com.br/forward-testing

**ANTES (erro):**
```
ERROR: DerivAPI.authorize() missing 1 required positional argument: 'token'
WARNING: ⚠️ Usando dados mock como fallback temporário
```

**DEPOIS (sucesso):**
```
INFO: ✅ Conectado e autenticado na Deriv API para dados reais
INFO: Tick recebido: R_100 @ $100.0829 (epoch: 1734360708)
```

### 2. Verificar Bugs Registrados

Na mesma página, seção "Bugs Registrados":

**ANTES:**
- 1 bug: `market_data_fetch_error` - "DerivAPI.authorize() missing..."

**DEPOIS:**
- 0 bugs (lista vazia) ✅

### 3. Verificar Previsões ML

Seção "Previsões ML Recentes":

**ANTES:**
- Preço: `$100.9615` (mock - sempre ~100)
- Confidence: `0.0%` (mock)

**DEPOIS:**
- Preço: `$100.0829` (real - varia naturalmente)
- Confidence: `> 0.0%` (calculado do modelo real)

### 4. API Endpoint

Teste via curl/browser:
```bash
curl https://botderivapi.roilabs.com.br/api/forward-testing/status
```

**Resposta esperada:**
```json
{
  "is_running": true,
  "total_bugs": 0,  // ✅ Deve ser 0
  "total_predictions": 1,
  "duration_hours": 0.1,
  "paper_trading_metrics": {
    "capital": 10000.0,
    "total_trades": 0
  }
}
```

---

## 🔍 Troubleshooting

### Problema: Deploy automático não aconteceu

**Causa**: Webhook do GitHub não configurado ou falhou

**Solução**:
1. Acesse Settings → Webhooks no GitHub
2. Verifique se há webhook para seu provedor
3. Se não houver, faça deploy manual

### Problema: "DERIV_API_TOKEN não configurado"

**Causa**: Variável de ambiente faltando no servidor

**Solução**:
1. Acesse painel do provedor
2. Vá em Environment Variables
3. Adicione:
   ```
   DERIV_API_TOKEN=paE5sSemx3oANLE
   ```
4. Reinicie o serviço

### Problema: Backend não reinicia

**Causa**: Erro de sintaxe ou import

**Solução**:
```bash
# Ver logs de erro
docker logs trading-bot-backend
# ou
journalctl -u trading-bot -n 50
```

---

## 📞 Status de Deploy por Plataforma

Marque aqui após executar:

- [ ] **Easypanel**: Deploy iniciado em ___:___ (horário)
- [ ] **Railway**: Deploy iniciado em ___:___ (horário)
- [ ] **Render**: Deploy iniciado em ___:___ (horário)
- [ ] **VPS/SSH**: Restart executado em ___:___ (horário)

- [ ] **Verificação**: Logs mostram "✅ Conectado e autenticado"
- [ ] **Validação**: Bugs Registrados = 0
- [ ] **Confirmação**: Preço real sendo coletado ($100.08+)

---

## 🎯 Resultado Esperado Final

Após reiniciar:
- ✅ Forward Testing conectado à Deriv API real
- ✅ Dados de mercado REAIS (não mock)
- ✅ Previsões ML com confidence > 0%
- ✅ Zero bugs de "authorize"
- ✅ Logs: "✅ Conectado e autenticado na Deriv API"

**Tempo estimado**: 2-5 minutos para deploy + restart

---

**Criado**: 2025-12-16 13:55 BRT
**Urgência**: 🔴 ALTA - Sistema usando mock data até restart

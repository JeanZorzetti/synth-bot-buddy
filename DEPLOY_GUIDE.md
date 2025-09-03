# 🚀 Guia de Deploy e Troubleshooting - EasyPanel

## 📋 URLs de Produção
- **Frontend**: https://botderiv.roilabs.com.br
- **Backend API**: https://botderivapi.roilabs.com.br
- **API Docs**: https://botderivapi.roilabs.com.br/docs
- **Health Check**: https://botderivapi.roilabs.com.br/health

## ✅ Status Checks

### 1. Verificar Backend Online
```bash
curl https://botderivapi.roilabs.com.br/health
```

**Resposta Esperada**:
```json
{
  "status": "healthy",
  "environment": "production",
  "websocket_manager": {
    "initialized": true,
    "state": "disconnected"
  },
  "dependencies": {
    "deriv_token_configured": true
  }
}
```

### 2. Verificar Rotas Disponíveis
```bash
curl https://botderivapi.roilabs.com.br/routes
```

### 3. Testar Conexão Deriv
```bash
curl -X POST https://botderivapi.roilabs.com.br/connect
```

## 🔧 Configuração EasyPanel

### Variáveis de Ambiente Necessárias:
```env
DERIV_API_TOKEN=FFJjPKCm9wnktDA
DERIV_APP_ID=1089
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Build Settings:
- **Build Path**: `/backend`
- **Dockerfile**: `Dockerfile`
- **Port**: `8000`

## 🐛 Problemas Comuns e Soluções

### ❌ Error 502 - Bad Gateway
**Causa**: Aplicação não está iniciando corretamente

**Soluções**:
1. Verificar logs do container no EasyPanel
2. Confirmar que todas as dependências estão no requirements.txt
3. Verificar se o Dockerfile está correto
4. Testar localmente: `docker build -t test . && docker run -p 8000:8000 test`

### ❌ CORS Errors
**Causa**: Frontend não consegue acessar backend

**Soluções**:
1. Verificar se URLs do frontend estão no CORS do backend
2. Confirmar HTTPS/HTTP matching
3. Testar com curl primeiro

### ❌ WebSocket Connection Failed
**Causa**: Token inválido ou problemas de rede

**Soluções**:
1. Verificar se `DERIV_API_TOKEN` está configurado
2. Testar token em https://api.deriv.com/
3. Verificar logs do WebSocket manager

### ❌ Health Check Failing
**Causa**: Aplicação não responde na porta correta

**Soluções**:
1. Verificar se porta 8000 está exposta
2. Confirmar que uvicorn está binding em 0.0.0.0
3. Testar health endpoint manualmente

## 📊 Monitoramento

### Logs Importantes:
```bash
# Ver logs do container
docker logs <container_id>

# Logs específicos do WebSocket
grep "WebSocket" /var/log/app.log

# Logs de conexão Deriv
grep "Deriv" /var/log/app.log
```

### Métricas de Saúde:
- ✅ Health endpoint responde em < 1s
- ✅ WebSocket conecta em < 5s  
- ✅ Ticks são recebidos em < 10s
- ✅ CPU < 50%, Memoria < 512MB

## 🔄 Processo de Deploy

### 1. Desenvolvimento Local:
```bash
# Backend
cd backend
python test_connection.py
python start.py

# Frontend  
cd frontend
npm run dev
```

### 2. Commit & Push:
```bash
git add .
git commit -m "fix: production improvements"
git push origin main
```

### 3. Deploy no EasyPanel:
- Auto-deploy via GitHub webhook
- Verificar build logs
- Testar endpoints

### 4. Validação Produção:
```bash
# Health check
curl https://botderivapi.roilabs.com.br/health

# Frontend test  
open https://botderiv.roilabs.com.br
```

## 🚨 Emergência - Rollback

Se algo der errado:

1. **EasyPanel**: Fazer rollback para commit anterior
2. **Logs**: Verificar o que causou o problema
3. **Fix**: Corrigir localmente e fazer novo deploy
4. **Monitor**: Acompanhar métricas pós-deploy

## 📞 Suporte

- **API Status**: https://botderivapi.roilabs.com.br/health
- **Documentation**: https://botderivapi.roilabs.com.br/docs  
- **Routes**: https://botderivapi.roilabs.com.br/routes
- **GitHub Issues**: Para reportar problemas
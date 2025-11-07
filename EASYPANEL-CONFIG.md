# ⚙️ Configuração EasyPanel - Passo a Passo

## 🚨 Situação Atual

✅ **Backend inicia corretamente** dentro do container:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

❌ **Mas retorna 502** quando acessado externamente:
```
curl https://botderivapi.roilabs.com.br/health
error code: 502
```

**Problema**: O proxy do EasyPanel não está conseguindo se comunicar com o container.

---

## 🔧 Configurações Necessárias no EasyPanel

### 1. **Configuração de Porta**

No EasyPanel, configuração do serviço:

**Port Mapping**:
- **Container Port**: `8000`
- **Public Port**: `80` ou `443` (automático)
- **Protocol**: `HTTP`

### 2. **Health Check Path**

Configure o health check:

- **Path**: `/health`
- **Port**: `8000`
- **Interval**: `30s`
- **Timeout**: `10s`
- **Retries**: `3`

### 3. **Environment Variables**

Já configuradas ✅:
```env
APP_ID=99188
ENVIRONMENT=production
INITIAL_CAPITAL=10.0
WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3
SECRET_KEY=sua_chave_secreta
```

### 4. **Domain Configuration**

Verifique se o domínio está configurado:

- **Domain**: `botderivapi.roilabs.com.br`
- **SSL**: Habilitado
- **Force HTTPS**: Habilitado

---

## 🐛 Troubleshooting - Passo a Passo

### Passo 1: Verificar Container

No EasyPanel → Serviço → **Logs**

Procure por:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

✅ Se aparecer: Container está OK
❌ Se não aparecer: Verificar erros de inicialização

### Passo 2: Verificar Porta Interna

No EasyPanel → Serviço → **Console** (se disponível)

Dentro do container, teste:
```bash
curl http://localhost:8000/health
```

Deve retornar:
```json
{"status": "healthy", ...}
```

### Passo 3: Verificar Porta Exposta

No EasyPanel → Serviço → **Settings** → **Ports**

Deve ter:
```
Container: 8000 → Public: 80/443
```

**IMPORTANTE**: Certifique-se que a porta está configurada como **HTTP** (não TCP)

### Passo 4: Verificar Proxy/Load Balancer

O EasyPanel usa Caddy ou Traefik como proxy reverso.

**Configuração necessária**:
- Proxy deve encaminhar para `http://container:8000`
- Headers HTTP devem ser preservados
- Timeout adequado (pelo menos 60s)

---

## 🔍 Configurações Específicas do EasyPanel

### Opção A: Via Interface Web

1. **Services** → Seu serviço → **Settings**
2. **Ports**:
   - Add Port Mapping
   - Container Port: `8000`
   - Protocol: `HTTP`
3. **Domains**:
   - Add Domain: `botderivapi.roilabs.com.br`
   - Enable SSL: ✅
4. **Deploy**:
   - Clique em **Redeploy**

### Opção B: Via Docker Compose (se suportado)

```yaml
version: '3.8'

services:
  backend:
    image: seu-registry/botderiv-backend:latest
    ports:
      - "8000:8000"
    environment:
      - APP_ID=99188
      - ENVIRONMENT=production
      - INITIAL_CAPITAL=10.0
      - WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3
      - SECRET_KEY=sua_chave_secreta
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.botderiv.rule=Host(`botderivapi.roilabs.com.br`)"
      - "traefik.http.services.botderiv.loadbalancer.server.port=8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

---

## 🎯 Checklist de Configuração

Verifique cada item:

- [ ] Container está iniciando corretamente (logs mostram "Application startup complete")
- [ ] Porta 8000 está exposta no container
- [ ] Porta 8000 está mapeada para porta pública no EasyPanel
- [ ] Protocol configurado como **HTTP** (não TCP ou WebSocket)
- [ ] Domínio `botderivapi.roilabs.com.br` apontando para o serviço
- [ ] SSL/HTTPS habilitado
- [ ] Health check configurado para `/health` na porta 8000
- [ ] Environment variables configuradas
- [ ] Proxy reverso encaminhando corretamente

---

## 🔧 Possíveis Soluções

### Solução 1: Verificar Target Port

No EasyPanel, alguns serviços precisam de uma configuração explícita de "Target Port".

Procure por configuração como:
- **Target Port**: `8000`
- **Service Port**: `80`

### Solução 2: Desabilitar Health Check Temporariamente

Se o health check está falhando e reiniciando o container:

1. Desabilite o health check temporariamente
2. Verifique se o serviço fica acessível
3. Se funcionar, ajuste o health check para timeout maior

### Solução 3: Verificar Logs do Proxy

No EasyPanel, procure por logs do proxy/load balancer:
- Traefik logs
- Caddy logs
- Nginx logs

Procure por erros como:
```
dial tcp: connection refused
upstream request timeout
502 bad gateway
```

### Solução 4: Testar com Porta Diferente

Temporariamente, teste com porta 8080:

No Dockerfile:
```dockerfile
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

No EasyPanel:
- Container Port: `8080`

---

## 🆘 Se Nada Funcionar

### Alternativa 1: Usar Railway, Render ou Fly.io

Essas plataformas têm configuração mais simples:

**Railway**:
```bash
railway login
railway init
railway up
```

**Render**:
- Conectar repositório GitHub
- Auto-deploy configurado
- CORS funciona out-of-the-box

**Fly.io**:
```bash
fly launch
fly deploy
```

### Alternativa 2: VPS Simples (DigitalOcean, Hetzner)

Com docker-compose em VPS, você tem controle total:

```bash
# No VPS
git clone https://github.com/JeanZorzetti/synth-bot-buddy.git
cd synth-bot-buddy
docker-compose up -d
```

Configure nginx:
```nginx
server {
    listen 80;
    server_name botderivapi.roilabs.com.br;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 Status de Debug

**Última verificação**: 2025-11-07 19:50 GMT

- ✅ Container inicia corretamente
- ✅ Uvicorn rodando na porta 8000
- ✅ Health check interno funciona (`127.0.0.1:43330 - "GET /health HTTP/1.1" 200 OK`)
- ❌ Health check externo retorna 502
- ❌ CORS ainda bloqueando requisições

**Conclusão**: Problema está na camada de **rede/proxy do EasyPanel**, não no código.

---

## 📞 Próximos Passos

1. Verificar configuração de porta no EasyPanel
2. Confirmar que proxy está encaminhando para porta 8000
3. Verificar logs do proxy/load balancer
4. Se não resolver, considerar plataforma alternativa

---

**Documentação EasyPanel**: https://easypanel.io/docs
**Repositório**: https://github.com/JeanZorzetti/synth-bot-buddy

# 🚀 Deploy Imediato - Correção CORS

## ✅ O Que Foi Feito

Implementado um **middleware CORS customizado** que resolve o erro:
```
Access to fetch at 'https://botderivapi.roilabs.com.br/api/order/execute'
from origin 'https://botderiv.roilabs.com.br' has been blocked by CORS policy
```

### Commits Aplicados:
1. **4458228** - Melhorias iniciais CORS + endpoint debug
2. **60cc8a3** - Middleware customizado (PRINCIPAL)
3. **5f8845c** - Documentação atualizada

---

## 🔧 Como Aplicar em Produção

### Opção 1: Pull + Restart (Mais Simples)

Se o backend está rodando direto no servidor:

```bash
# No servidor de produção
cd /caminho/do/projeto/synth-bot-buddy

# Pull das mudanças
git pull origin main

# Reiniciar backend
cd backend
source ../.venv/bin/activate  # Linux/Mac
# ou
..\.venv\Scripts\activate     # Windows

# Matar processo atual (se estiver rodando)
pkill -f "uvicorn main:app"

# Iniciar novamente
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Opção 2: Docker Rebuild

Se está usando Docker:

```bash
# No servidor de produção
cd /caminho/do/projeto/synth-bot-buddy

# Pull das mudanças
git pull origin main

# Rebuild do container
cd backend
docker build -t synth-bot-backend .

# Parar container antigo
docker stop synth-backend
docker rm synth-backend

# Iniciar novo container
docker run -d \
  --name synth-backend \
  -p 8000:8000 \
  --env-file .env \
  synth-bot-backend
```

### Opção 3: Systemd Service

Se está usando systemd:

```bash
# Pull das mudanças
cd /caminho/do/projeto/synth-bot-buddy
git pull origin main

# Reiniciar serviço
sudo systemctl restart synth-backend
sudo systemctl status synth-backend
```

---

## 🧪 Como Testar Após Deploy

### 1. Testar Backend Diretamente

```bash
# Health check
curl https://botderivapi.roilabs.com.br/health

# CORS test
curl https://botderivapi.roilabs.com.br/cors-test \
  -H "Origin: https://botderiv.roilabs.com.br" \
  -v
```

**Resposta esperada do /cors-test**:
```json
{
  "status": "CORS is working",
  "origin": "https://botderiv.roilabs.com.br",
  "headers": {...},
  "method": "GET",
  "url": "..."
}
```

### 2. Testar Preflight OPTIONS

```bash
curl -X OPTIONS https://botderivapi.roilabs.com.br/api/order/execute \
  -H "Origin: https://botderiv.roilabs.com.br" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v
```

**Headers esperados na resposta**:
```
< HTTP/1.1 200 OK
< Access-Control-Allow-Origin: https://botderiv.roilabs.com.br
< Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, PATCH
< Access-Control-Allow-Headers: *
< Access-Control-Allow-Credentials: true
< Access-Control-Max-Age: 3600
```

### 3. Testar no Frontend

1. Acesse: https://botderiv.roilabs.com.br
2. Abra Developer Tools (F12)
3. Vá para a aba **Network**
4. Clique em "Execute Order"
5. Verifique:
   - ✅ Aparece uma requisição OPTIONS (preflight) com status 200
   - ✅ Aparece uma requisição POST com status 200
   - ✅ Nenhum erro CORS no console

---

## 📊 Logs para Monitorar

### Docker Logs
```bash
docker logs -f synth-backend
```

### Systemd Logs
```bash
journalctl -u synth-backend -f
```

### Uvicorn Direto
Se rodando em terminal, os logs aparecem diretamente.

**O que procurar**:
- ✅ Servidor inicia sem erros
- ✅ Requisições OPTIONS aparecem com status 200
- ✅ Requisições POST aparecem logo após OPTIONS

---

## ❌ Se Ainda Não Funcionar

### 1. Verificar Reverse Proxy (Nginx)

Se usa nginx na frente do backend, verifique a configuração:

```nginx
location /api {
    proxy_pass http://localhost:8000;

    # NÃO adicione headers CORS aqui!
    # Deixe o FastAPI cuidar disso

    # Apenas passe os headers através
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Recarregar nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 2. Verificar Cloudflare

Se usa Cloudflare:

1. Vá em **SSL/TLS** → **Overview**
2. Certifique-se que está em **Full** ou **Full (strict)**

3. Vá em **Network**
4. Desative **HTTP/2 to Origin** temporariamente para testar

5. Vá em **Rules** → **Page Rules**
6. Adicione regra para `botderivapi.roilabs.com.br/*`:
   - Cache Level: Bypass
   - Disable Performance (temporariamente)

### 3. Verificar Origem no Código

Verifique no frontend se a URL está correta:

**Arquivo**: `frontend/.env.production`
```env
VITE_API_URL=https://botderivapi.roilabs.com.br
```

**IMPORTANTE**: Após mudar `.env.production`, é necessário rebuild:
```bash
cd frontend
npm run build
```

### 4. Testar com Allow All (Debug)

Temporariamente, para isolar o problema, você pode testar com `allow_origins=["*"]`:

**Arquivo**: `backend/main.py` (linha 229)
```python
allowed_origins = ["*"]  # APENAS PARA TESTE!
```

Se funcionar com `["*"]`, o problema é na validação de origem. Verifique:
- Se a origem tem `https` ou `http`
- Se não há barra `/` no final
- Se não há espaços

**LEMBRE-SE**: Reverter para a whitelist após o teste!

---

## ✅ Checklist Final

Após deploy, verifique:

- [ ] Backend iniciou sem erros
- [ ] `/health` retorna status 200
- [ ] `/cors-test` retorna informações corretas
- [ ] Requisição OPTIONS retorna status 200 com headers CORS
- [ ] Frontend não mostra erros CORS no console
- [ ] Order execution funciona e retorna Contract ID
- [ ] Logs não mostram erros críticos

---

## 📞 Próximos Passos

Após CORS funcionar:

1. ✅ Testar execução de ordem completa
2. ✅ Verificar se Contract ID é salvo
3. ✅ Confirmar ordem na plataforma Deriv
4. 🎉 Marcar Objetivo 1 como CONCLUÍDO!

---

## 🆘 Suporte

Se continuar com problemas:

1. **Envie logs do backend** durante a requisição
2. **Screenshot do console** do navegador (F12)
3. **Screenshot da aba Network** mostrando a requisição OPTIONS

**Documentação Completa**: Ver [CORS-FIX.md](CORS-FIX.md)

---

**Última atualização**: 2025-11-07 (Commit: 5f8845c)

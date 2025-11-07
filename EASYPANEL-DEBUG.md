# 🔍 EasyPanel - Guia de Debug (502 Bad Gateway)

## 🚨 Problema Identificado

O backend está retornando **502 Bad Gateway** no EasyPanel, o que significa que o servidor não está iniciando corretamente.

```
< HTTP/1.1 502 Bad Gateway
```

O erro CORS é apenas um **sintoma secundário** - o problema real é que o backend não está funcionando.

---

## ✅ Solução Aplicada

**Commit**: `927a3e1` - "fix: Simplify CORS to allow all origins temporarily"

### Mudanças:
1. ❌ **Removido** `CustomCORSMiddleware` (estava causando erro de inicialização)
2. ❌ **Removido** import `BaseHTTPMiddleware`
3. ✅ **Simplificado** para usar apenas `CORSMiddleware` padrão
4. ⚠️ **Temporário**: Usando `allow_origins=["*"]` para debug

---

## 🚀 Como Aplicar no EasyPanel

### 1. Fazer Deploy do Novo Commit

No EasyPanel:
1. Vá para o serviço **botderiv**
2. Na aba **Deploy**, clique em **Redeploy**
3. Ou configure para auto-deploy do branch `main`

### 2. Verificar Logs em Tempo Real

No EasyPanel:
1. Vá para o serviço **botderiv**
2. Clique na aba **Logs**
3. Observe a inicialização do container

**Logs esperados (sucesso)**:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Logs de erro (problema)**:
```
ModuleNotFoundError: No module named 'xxx'
ImportError: cannot import name 'xxx'
SyntaxError: invalid syntax
```

### 3. Testar Endpoints Após Deploy

```bash
# 1. Health check (deve retornar 200, não 502)
curl https://botderivapi.roilabs.com.br/health

# 2. CORS test
curl https://botderivapi.roilabs.com.br/cors-test

# 3. Routes list
curl https://botderivapi.roilabs.com.br/routes
```

**Resposta esperada do /health**:
```json
{
  "status": "healthy",
  "timestamp": 1731010000.0,
  "version": "0.1.0",
  "environment": "production",
  ...
}
```

---

## 🐛 Possíveis Causas do 502

### 1. **Dependências Faltando**

Verificar se `requirements.txt` está completo:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0
requests==2.31.0
...
```

### 2. **Porta Incorreta**

Verificar se o EasyPanel está configurado para a porta correta:
- **Backend expõe**: Porta 8000
- **EasyPanel deve mapear**: Porta 8000

### 3. **Variáveis de Ambiente Faltando**

Verificar se as env vars estão configuradas no EasyPanel:
```env
APP_ID=99188
ENVIRONMENT=production
INITIAL_CAPITAL=10.0
WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3
SECRET_KEY=sua_chave_secreta
```

### 4. **Health Check Falhando**

O Dockerfile tem um health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1
```

Se o health check falhar 3 vezes, o container é reiniciado.

---

## 🔧 Comandos de Debug no EasyPanel

### Ver Logs do Container

No EasyPanel → Serviço → Logs

Ou via CLI (se tiver acesso SSH):
```bash
docker logs -f <container-id>
```

### Entrar no Container (se possível)

```bash
docker exec -it <container-id> /bin/bash

# Dentro do container, testar:
cd /app
python -c "import fastapi; print('OK')"
python -c "from main import app; print('OK')"
```

### Testar Manualmente

```bash
# Dentro do container
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📊 Checklist de Verificação

Após fazer deploy do commit `927a3e1`:

- [ ] Deploy concluído no EasyPanel sem erros
- [ ] Logs mostram "Application startup complete"
- [ ] `/health` retorna status 200 (não 502)
- [ ] `/cors-test` retorna dados JSON
- [ ] Frontend consegue fazer requisições (sem erro CORS)
- [ ] Order execution funciona
- [ ] Contract ID é retornado corretamente

---

## ⚠️ Próximos Passos (Após Confirmar Funcionamento)

### 1. Restaurar Whitelist de Origens

Depois que o backend estiver funcionando com `allow_origins=["*"]`, vamos restaurar a whitelist para segurança:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://botderiv.roilabs.com.br",
        "http://botderiv.roilabs.com.br"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```

### 2. Testar com Whitelist

Fazer novo commit, deploy e testar se CORS ainda funciona.

### 3. Marcar Objetivo 1 como Concluído

Quando tudo estiver funcionando:
- ✅ Backend responde sem 502
- ✅ CORS funciona em produção
- ✅ Order execution retorna Contract ID
- 🎉 Objetivo 1: 100% COMPLETO!

---

## 🆘 Se Ainda Não Funcionar

### Opção 1: Verificar Proxy/Load Balancer

EasyPanel pode estar usando nginx ou Caddy na frente. Verificar se:
- Proxy está passando requisições para porta 8000
- Timeout do proxy não é muito curto
- Headers não estão sendo removidos

### Opção 2: Testar Localmente com Docker

```bash
# No seu computador
cd backend
docker build -t test-backend .
docker run -p 8000:8000 --env-file .env test-backend

# Testar
curl http://localhost:8000/health
```

Se funcionar localmente mas não no EasyPanel, o problema é de infraestrutura.

### Opção 3: Simplificar Ainda Mais

Criar um `main_simple.py` minimal para testar:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

Se isso funcionar, o problema está em alguma dependência ou import.

---

## 📞 Informações Importantes

- **Commit Atual**: `927a3e1`
- **Branches**: `main`
- **URL Backend**: https://botderivapi.roilabs.com.br
- **URL Frontend**: https://botderiv.roilabs.com.br
- **Repositório**: https://github.com/JeanZorzetti/synth-bot-buddy

---

**Última atualização**: 2025-11-07 19:45 GMT

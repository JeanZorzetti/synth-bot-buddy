# 🔧 Correção CORS - Produção

## 📋 Mudanças Implementadas

### 1. Melhorias no Middleware CORS

**Arquivo**: `backend/main.py` (linhas 224-243)

**Alterações**:
- ✅ Adicionado `expose_headers=["*"]` para expor headers na resposta
- ✅ Especificados métodos HTTP explicitamente incluindo `OPTIONS`
- ✅ Adicionado comentário indicando que CORS deve estar antes das rotas

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "https://botderiv.roilabs.com.br",  # Production frontend
        "http://botderiv.roilabs.com.br"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```

### 2. Endpoint de Teste CORS

**Novo endpoint**: `GET /cors-test`

```bash
curl https://botderivapi.roilabs.com.br/cors-test
```

Retorna informações sobre headers e origem da requisição para debug.

---

## 🧪 Como Testar

### 1. Teste Local (antes de fazer deploy)

```bash
# No diretório backend
cd backend

# Ativar ambiente virtual
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac

# Iniciar servidor
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

No navegador, acesse:
- http://localhost:5173 (frontend dev)
- Tente executar uma ordem
- Verifique console do navegador para erros CORS

### 2. Teste de CORS via cURL

```bash
# Testar preflight OPTIONS request
curl -X OPTIONS https://botderivapi.roilabs.com.br/api/order/execute \
  -H "Origin: https://botderiv.roilabs.com.br" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v

# Deve retornar headers:
# Access-Control-Allow-Origin: https://botderiv.roilabs.com.br
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, PATCH
# Access-Control-Allow-Headers: *
```

### 3. Teste do Endpoint CORS

```bash
# Teste simples
curl https://botderivapi.roilabs.com.br/cors-test

# Teste com Origin header
curl https://botderivapi.roilabs.com.br/cors-test \
  -H "Origin: https://botderiv.roilabs.com.br" \
  -v
```

### 4. Teste Completo de Produção

1. **Build do Frontend**:
```bash
cd frontend
npm run build
```

2. **Deploy do Backend** (conforme método usado):
```bash
# Se usando Docker
docker build -t synth-bot-backend ./backend
docker run -d -p 8000:8000 synth-bot-backend

# Se usando uvicorn direto
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

3. **Testar no Navegador**:
   - Acesse: https://botderiv.roilabs.com.br
   - Abra Developer Tools (F12) → Network
   - Tente executar uma ordem
   - Verifique:
     - ✅ Requisição OPTIONS retorna status 200
     - ✅ Headers CORS presentes na resposta
     - ✅ Requisição POST executada com sucesso

---

## 🐛 Troubleshooting

### Erro: "No 'Access-Control-Allow-Origin' header"

**Possíveis causas**:

1. **Reverse Proxy removendo headers**
   - Se usando nginx, adicionar:
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;

       # Importante: não sobrescrever headers CORS
       proxy_pass_header Access-Control-Allow-Origin;
       proxy_pass_header Access-Control-Allow-Methods;
       proxy_pass_header Access-Control-Allow-Headers;
   }
   ```

2. **Cloudflare ou CDN intermediário**
   - Verificar se Cloudflare está em modo "proxy" (nuvem laranja)
   - Temporariamente mudar para "DNS only" (nuvem cinza) para testar
   - Se for a causa, ajustar regras de Page Rules do Cloudflare

3. **Backend não reiniciado após mudanças**
   ```bash
   # Reiniciar serviço
   sudo systemctl restart synth-backend

   # Ou se usando Docker
   docker restart <container-id>
   ```

4. **Origin não está na lista allow_origins**
   - Verificar se `https://botderiv.roilabs.com.br` está na lista
   - Atenção para `http` vs `https`
   - Verificar se não há espaços ou caracteres extras

### Erro: "Response to preflight request doesn't pass"

**Solução**: FastAPI deve tratar OPTIONS automaticamente com o middleware CORS. Se ainda falhar:

1. Verificar ordem do middleware (CORS deve ser primeiro)
2. Verificar se há outro middleware bloqueando OPTIONS
3. Testar com `allow_origins=["*"]` temporariamente para isolar o problema

### Logs não mostram requisições OPTIONS

**Causa**: Requisição OPTIONS pode estar sendo bloqueada antes de chegar ao FastAPI.

**Verificar**:
```bash
# Monitorar logs do nginx (se aplicável)
sudo tail -f /var/log/nginx/access.log

# Monitorar logs do backend
journalctl -u synth-backend -f
```

---

## 📊 Endpoints de Debug

### `/health` - Health Check Detalhado
```bash
curl https://botderivapi.roilabs.com.br/health
```

Retorna:
- Status do servidor
- Estado do WebSocket Manager
- Variáveis de ambiente
- Dependências instaladas

### `/routes` - Lista de Rotas
```bash
curl https://botderivapi.roilabs.com.br/routes
```

Retorna todas as rotas disponíveis com métodos HTTP.

### `/cors-test` - Teste de CORS
```bash
curl https://botderivapi.roilabs.com.br/cors-test \
  -H "Origin: https://botderiv.roilabs.com.br"
```

Retorna headers da requisição para verificar CORS.

---

## ✅ Checklist de Verificação

Antes de considerar o problema resolvido:

- [ ] Requisição OPTIONS retorna status 200
- [ ] Header `Access-Control-Allow-Origin` presente na resposta OPTIONS
- [ ] Header `Access-Control-Allow-Methods` inclui POST
- [ ] Header `Access-Control-Allow-Headers` presente
- [ ] Requisição POST executa após OPTIONS bem-sucedido
- [ ] Frontend recebe resposta sem erros CORS
- [ ] Console do navegador não mostra erros CORS
- [ ] Ordem é executada com sucesso e Contract ID retornado

---

## 🔄 Próximos Passos

Após corrigir CORS:

1. ✅ Testar execução de ordem completa em produção
2. ✅ Verificar se Contract ID é retornado corretamente
3. ✅ Confirmar que ordem aparece na plataforma Deriv
4. ✅ Marcar Objetivo 1 como 100% concluído
5. 📝 Documentar URLs de produção funcionais

---

## 📞 Suporte

- **Documentação CORS FastAPI**: https://fastapi.tiangolo.com/tutorial/cors/
- **Repositório**: https://github.com/JeanZorzetti/synth-bot-buddy

---

**Última atualização**: 2025-11-07

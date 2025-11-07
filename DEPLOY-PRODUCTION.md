# 🚀 Guia de Deploy - Produção

## 📋 Pré-requisitos

- Servidor com Python 3.11+
- Node.js 18+ (para build do frontend)
- Domínios configurados:
  - Frontend: `botderiv.roilabs.com.br`
  - Backend API: `botderivapi.roilabs.com.br`

---

## 🔧 Configuração

### 1. Backend (API)

#### 1.1 Preparar Ambiente

```bash
# Clonar repositório
git clone https://github.com/JeanZorzetti/synth-bot-buddy.git
cd synth-bot-buddy

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instalar dependências
cd backend
pip install -r requirements.txt
```

#### 1.2 Variáveis de Ambiente

Criar arquivo `.env` no diretório `backend/`:

```env
# Backend Environment Variables
APP_ID=99188
ENVIRONMENT=production
INITIAL_CAPITAL=10.0
WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3

# Security
SECRET_KEY=sua_chave_secreta_aqui_gere_uma_forte
```

#### 1.3 Iniciar Backend

**Opção 1: Uvicorn Direto**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Opção 2: Com Gunicorn (Recomendado para Produção)**
```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Opção 3: Docker**
```bash
# Criar Dockerfile
docker build -t synth-bot-backend .
docker run -d -p 8000:8000 --name synth-backend synth-bot-backend
```

---

### 2. Frontend

#### 2.1 Build para Produção

```bash
cd frontend

# Instalar dependências
npm install

# Build
npm run build
```

O build será gerado em `frontend/dist/`

#### 2.2 Deploy do Frontend

**Opção 1: Nginx**

```nginx
server {
    listen 80;
    server_name botderiv.roilabs.com.br;

    root /var/www/synth-bot-buddy/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy para API
    location /api {
        proxy_pass http://botderivapi.roilabs.com.br;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**Opção 2: Vercel**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel --prod
```

**Opção 3: Netlify**

```bash
# Install Netlify CLI
npm install netlify-cli -g

# Deploy
cd frontend
netlify deploy --prod --dir=dist
```

---

## 🔒 Configuração de Segurança

### SSL/HTTPS

**Certbot (Let's Encrypt)**

```bash
# Backend
sudo certbot --nginx -d botderivapi.roilabs.com.br

# Frontend
sudo certbot --nginx -d botderiv.roilabs.com.br
```

---

## 📊 Monitoramento

### Health Check

Backend fornece endpoint de health check:

```bash
curl https://botderivapi.roilabs.com.br/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-07T00:00:00Z",
  "version": "1.0.0"
}
```

---

## 🐳 Docker Compose (Recomendado)

Criar `docker-compose.yml` na raiz:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - APP_ID=99188
      - ENVIRONMENT=production
    restart: unless-stopped
    volumes:
      - ./backend:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
    restart: unless-stopped
```

Iniciar:
```bash
docker-compose up -d
```

---

## 🔄 Atualização/Deploy Contínuo

### Script de Deploy

Criar `deploy.sh`:

```bash
#!/bin/bash

echo "🚀 Iniciando deploy..."

# Pull latest changes
git pull origin main

# Backend
echo "📦 Atualizando backend..."
cd backend
source ../.venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart synth-backend

# Frontend
echo "🎨 Atualizando frontend..."
cd ../frontend
npm install
npm run build
sudo cp -r dist/* /var/www/synth-bot-buddy/frontend/dist/
sudo systemctl reload nginx

echo "✅ Deploy concluído!"
```

---

## 📝 Checklist de Deploy

### Antes do Deploy:
- [ ] Build do frontend testado localmente
- [ ] Backend testado com `pytest`
- [ ] Variáveis de ambiente configuradas
- [ ] SSL/HTTPS configurado
- [ ] CORS configurado com domínios corretos
- [ ] Backup do banco de dados (se aplicável)

### Após o Deploy:
- [ ] Health check funcionando
- [ ] Frontend carrega corretamente
- [ ] API responde corretamente
- [ ] Testar execução de ordem end-to-end
- [ ] Logs não mostram erros críticos
- [ ] Monitoramento ativo

---

## 🆘 Troubleshooting

### Backend não inicia

```bash
# Verificar logs
journalctl -u synth-backend -f

# Testar manualmente
cd backend
source ../.venv/bin/activate
python main.py
```

### Frontend com erro 404

```bash
# Verificar nginx
sudo nginx -t
sudo systemctl status nginx

# Verificar arquivos
ls -la /var/www/synth-bot-buddy/frontend/dist/
```

### CORS Error

- Verificar `main.py` linha 227-238
- Confirmar que domínio está na lista `allow_origins`
- Verificar SSL (https vs http)

---

## 📞 Suporte

- **Repositório**: https://github.com/JeanZorzetti/synth-bot-buddy
- **Issues**: https://github.com/JeanZorzetti/synth-bot-buddy/issues

---

## 🎯 URLs de Produção

- **Frontend**: https://botderiv.roilabs.com.br
- **Backend API**: https://botderivapi.roilabs.com.br
- **API Docs**: https://botderivapi.roilabs.com.br/docs
- **Health Check**: https://botderivapi.roilabs.com.br/health

---

**Deploy com sucesso! 🚀**

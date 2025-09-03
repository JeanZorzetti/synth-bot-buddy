# 🚀 Setup do Backend - Synth Bot Buddy

## 📋 Pré-requisitos
- Python 3.8+ instalado
- pip funcionando

## 🔧 Instalação Rápida

### 1. Instalar dependências
```bash
cd backend
python install_deps.py
```

Ou manualmente:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Executar o servidor
```bash
# Opção 1: Script simples
python run.py

# Opção 2: Script completo com verificações
python start.py

# Opção 3: Direto com uvicorn
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Endpoints Principais
- **Servidor**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs
- **Saúde**: http://localhost:8000/health
- **Status**: http://localhost:8000/status

## ⚠️ Solução de Problemas

### "ModuleNotFoundError: No module named 'fastapi'"
```bash
pip install fastapi uvicorn[standard] websockets pydantic python-dotenv requests
```

### Python não encontrado
- Windows: Instale do python.org ou Microsoft Store
- Configure PATH corretamente
- Use `python` ou `python3`

### Porta 8000 ocupada
```bash
# Use porta diferente
python -m uvicorn main:app --reload --port 8001
```

## 🔑 Configuração de Token
- ❌ **NÃO** precisa configurar .env
- ✅ **Tokens são configurados via frontend**
- ⚡ **Sistema funciona dinamicamente**

## 📝 Logs
- Logs aparecerão no terminal
- Nível INFO habilitado
- Debugging detalhado para autenticação

## 🐛 Debug
Se houver erro 401 no frontend:
1. Verifique logs do backend no terminal
2. Use token válido da Deriv
3. Teste endpoint `/validate-token`
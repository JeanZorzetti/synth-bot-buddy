# INSTALAR DEPENDÊNCIAS DO ABUTRE BOT

**Erro comum:** `No module named 'sqlalchemy'`

**Causa:** As dependências do bot Abutre não estão instaladas no ambiente Python do backend.

---

## 🔧 Solução Rápida (Servidor Easypanel)

### Opção 1: SSH no servidor e instalar

```bash
# 1. SSH no servidor Easypanel
ssh user@seu-servidor

# 2. Entrar no diretório do backend
cd /app/backend  # (ou onde o backend está rodando)

# 3. Instalar dependências do Abutre
pip install -r bots/abutre/requirements.txt

# 4. Reiniciar backend
# (Easypanel faz isso automaticamente em alguns casos)
```

### Opção 2: Adicionar ao requirements.txt principal

Copiar as dependências essenciais de `bots/abutre/requirements.txt` para `backend/requirements.txt`:

```txt
# Abutre Bot dependencies
SQLAlchemy==2.0.23
python-deriv-api==0.1.6
websockets==12.0
python-socketio==5.10.0
python-engineio==4.8.0
```

Depois fazer push e o Easypanel reinstala automaticamente.

---

## 📦 Dependências Necessárias

**Essenciais (mínimo para rodar):**
- `SQLAlchemy==2.0.23` - Database ORM
- `python-deriv-api==0.1.6` - Deriv API
- `websockets==12.0` - WebSocket client
- `python-dotenv==1.0.0` - Variáveis .env

**Opcionais (melhorias):**
- `python-socketio==5.10.0` - Socket.IO server
- `pandas==2.1.4` - Data processing
- `pytest==7.4.3` - Testing

---

## ✅ Verificar Instalação

```bash
# Testar import
python -c "import sqlalchemy; print('✅ SQLAlchemy OK')"
python -c "from deriv_api import DerivAPI; print('✅ Deriv API OK')"

# Se der erro, instalar:
pip install sqlalchemy python-deriv-api websockets python-dotenv
```

---

## 🚀 Após Instalar

1. Reiniciar backend (ou deixar Easypanel reiniciar)
2. Acessar dashboard: https://botderiv.rollabs.com.br/abutre
3. Clicar "Iniciar Bot"
4. Verificar se mostra "🟢 Bot Rodando"

---

## 🆘 Troubleshooting

**Erro persiste após instalar?**
- Verificar se está usando o ambiente Python correto
- Verificar logs do Easypanel
- Tentar instalar globalmente: `pip install --user sqlalchemy`

**Como saber qual Python está sendo usado?**
```bash
which python
python --version
pip list | grep -i sqlalchemy
```

---

**Solução alternativa:** Merge as requirements:

```bash
cd backend
cat bots/abutre/requirements.txt >> requirements.txt
```

Depois commit e push → Easypanel reinstala tudo.

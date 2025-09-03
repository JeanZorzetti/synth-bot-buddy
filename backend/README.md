# Backend - Synth Bot Buddy

O "Cérebro" do sistema de trading automatizado para a plataforma Deriv.

## 🚀 Configuração Inicial

### 1. Instalação de Dependências

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configuração de Ambiente

1. Copie o arquivo de exemplo:
   ```bash
   cp .env.example .env
   ```

2. Configure suas credenciais no arquivo `.env`:
   ```env
   DERIV_API_TOKEN=seu_token_aqui
   DERIV_APP_ID=1089
   ```

### 3. Obter Token da API Deriv

1. Acesse: https://app.deriv.com/account/api-token
2. Gere um novo token com os escopos:
   - ✅ **Read** (para ler dados da conta)
   - ✅ **Trade** (para executar ordens)
   - ❌ **Payments** (desnecessário - risco de segurança)
   - ❌ **Admin** (desnecessário - risco de segurança)

⚠️ **IMPORTANTE**: Mantenha seu token seguro e nunca o compartilhe!

## 🧪 Testando a Conexão

Antes de executar o servidor, teste sua conexão:

```bash
python test_connection.py
```

Este teste vai verificar:
- ✅ Conexão WebSocket
- ✅ Autenticação com seu token
- ✅ Recebimento de dados de saldo
- ✅ Subscrição a ticks em tempo real

## 🖥️ Executando o Servidor

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

O servidor estará disponível em: http://localhost:8000

## 📡 API Endpoints

### Status e Conexão
- `GET /` - Status do servidor
- `GET /status` - Status detalhado do bot
- `POST /connect` - Conectar à API Deriv
- `POST /disconnect` - Desconectar da API Deriv

### Controle do Bot
- `POST /start` - Iniciar o bot de trading
- `POST /stop` - Parar o bot de trading

### Trading (Requer autenticação)
- `POST /buy` - Executar uma ordem de compra
  - Parâmetros: `contract_type`, `amount`, `duration`, `symbol`

## 🔧 Arquitetura

### WebSocket Manager (`websocket_manager.py`)
- Gerencia conexão persistente com API Deriv
- Handlers para diferentes tipos de mensagem
- Auto-reconexão e heartbeat
- Gestão de estados de conexão

### Estados de Conexão
1. `DISCONNECTED` - Desconectado
2. `CONNECTING` - Conectando
3. `CONNECTED` - Conectado (sem autenticação)
4. `AUTHENTICATED` - Autenticado e pronto para trading
5. `ERROR` - Estado de erro

### Event Handlers
- `handle_tick_data` - Processa ticks em tempo real
- `handle_balance_update` - Atualiza saldo da conta
- `handle_trade_result` - Processa resultados de trades
- `handle_connection_status` - Monitora status da conexão

## 🛡️ Segurança

- Tokens API são carregados via variáveis de ambiente
- CORS configurado apenas para frontend local
- Logs detalhados para auditoria
- Validação de estados antes de executar ordens

## 🐛 Debugging

### Logs
O sistema usa logging nativo do Python. Para debug detalhado:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Problemas Comuns

1. **Token inválido**: Verifique se o token está correto e tem os escopos necessários
2. **Conexão falhando**: Verifique sua conexão de internet
3. **Timeout na autenticação**: Token pode estar expirado ou inválido
4. **CORS errors**: Certifique-se que o frontend está rodando na porta correta

## 📈 Próximos Passos

- [ ] Implementar estratégias de trading
- [ ] Sistema de logging avançado
- [ ] Backtesting engine
- [ ] Risk management
- [ ] Performance analytics
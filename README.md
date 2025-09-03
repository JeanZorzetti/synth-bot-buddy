# 🤖 Synth Bot Buddy

Sistema de trading automatizado para a plataforma Deriv com interface web moderna e backend inteligente.

## 🚀 **Status do Projeto**

✅ **Backend WebSocket** - Conexão completa com API Deriv  
✅ **Frontend React** - Interface moderna e responsiva  
✅ **Integração Real-time** - Dashboard com dados reais  
🚧 **Estratégias de Trading** - Em desenvolvimento  
🚧 **Sistema de Logging** - Em desenvolvimento  

## 📋 **Pré-requisitos**

- **Python 3.11+** (para backend)
- **Node.js 18+** (para frontend)  
- **Conta Deriv** com token API

## ⚡ **Início Rápido**

### 1. **Backend (API + WebSocket)**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Configure seu token Deriv no .env
python start.py
```

### 2. **Frontend (Interface)**
```bash
cd frontend
npm install
npm run dev
```

### 3. **Acessar Sistema**
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs

## 🔧 **Configuração Detalhada**

### Backend Configuration
Edite o arquivo `backend/.env`:
```env
DERIV_API_TOKEN=seu_token_aqui
DERIV_APP_ID=1089
ENVIRONMENT=development
```

### Frontend Configuration  
Edite o arquivo `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

## 🧪 **Como Testar**

1. **Testar Conexão Backend**:
   ```bash
   cd backend
   python test_connection.py
   ```

2. **Verificar Frontend**:
   - Abra http://localhost:5173
   - Verifique status "Backend Online"
   - Clique "Conectar" para conectar à API Deriv
   - Clique "Iniciar Bot" para começar a receber dados

## 📱 **Funcionalidades**

### ✅ **Implementado**
- 🔗 Conexão WebSocket segura com Deriv API
- 🔐 Autenticação automática com token
- 📊 Recebimento de ticks em tempo real  
- 💰 Monitoramento de saldo da conta
- 🎯 Execução de ordens de compra/venda
- 📱 Interface web responsiva
- 🔄 Auto-reconexão e heartbeat
- 🚨 Notificações e alertas
- 📈 Dashboard em tempo real

### 🚧 **Em Desenvolvimento** 
- 🧠 Estratégias de trading inteligentes
- 📝 Sistema de logging avançado
- 📊 Backtesting engine
- ⚖️ Risk management
- 📈 Análise de performance

## 📡 **API Endpoints**

- `GET /` - Status do servidor
- `GET /status` - Status detalhado do bot
- `POST /connect` - Conectar à API Deriv  
- `POST /start` - Iniciar bot de trading
- `POST /stop` - Parar bot de trading
- `POST /buy` - Executar ordem de compra

## 🛡️ **Segurança**

- ✅ Tokens API via variáveis de ambiente
- ✅ CORS configurado para frontend local
- ✅ Logs detalhados para auditoria
- ✅ Validação de estados antes de ordens
- ✅ Escopos mínimos de permissão (Read + Trade)

## 🐳 **Docker**

```bash
cd backend
docker build -t synth-bot-buddy .
docker run -p 8000:8000 --env-file .env synth-bot-buddy
```

## 📚 **Arquitetura**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Deriv API     │
│   (React)       │◄──►│   (FastAPI)     │◄──►│  (WebSocket)    │
│   Dashboard     │    │   WebSocket     │    │   Real-time     │
│   Controls      │    │   Manager       │    │   Trading       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🆘 **Solução de Problemas**

### Python não encontrado
- Reinstale Python de https://python.org
- Marque "Add to PATH" durante instalação

### Backend não conecta
- Verifique se o token está correto no .env
- Confirme que tem permissões Read + Trade
- Teste com `python test_connection.py`

### Frontend não carrega dados  
- Confirme que backend está rodando (porta 8000)
- Verifique configuração VITE_API_URL no .env

## 📞 **Suporte**

- 📧 **Issues**: Use o sistema de issues do GitHub
- 📖 **Documentação**: Veja /backend/README.md
- 🔧 **API Deriv**: https://api.deriv.com/

---

## **Guia Técnico - API Deriv**

Informações técnicas essenciais para integração com a API da Deriv.

### **1. Arquitetura e Conexão Principal**

A API da Deriv é construída sobre o protocolo **WebSocket**, priorizando comunicação de baixa latência e em tempo real, essencial para aplicações de negociação.

*   **Endpoint de Conexão:** Todas as comunicações devem ser estabelecidas através do seguinte endpoint WebSocket seguro:
    ```
    wss://ws.derivws.com/websockets/v3?app_id={app_id}
    ```
*   **Parâmetro `app_id`:** É obrigatório e serve para identificar sua aplicação. Você deve obter um `app_id` ao registrar seu aplicativo na plataforma Deriv.
*   **Criptografia:** A conexão `wss://` é protegida com criptografia TLS, garantindo a confidencialidade dos dados trocados.

### **2. Segurança e Autenticação: Tokens API**

O acesso à API é controlado por **Tokens API** gerados pelo usuário. Sua aplicação nunca deve solicitar ou armazenar o nome de usuário e senha do cliente.

#### **2.1. Escopos de Permissão (Scopes)**

Ao criar um token, são atribuídos escopos que definem o que sua aplicação pode fazer. É crucial seguir o **Princípio do Menor Privilégio**, solicitando apenas as permissões estritamente necessárias.

| Escopo | Descrição Funcional | Vetor de Risco Associado |
| :--- | :--- | :--- |
| **Read** | Visualizar atividade da conta, configurações, saldos, histórico. | Exposição de dados financeiros confidenciais e estratégias de negociação. |
| **Trade** | Comprar e vender contratos, gerenciar posições. | Execução de negociações não autorizadas ou errôneas, resultando em perdas financeiras diretas. |
| **Payments** | Sacar para agentes de pagamento, transferir fundos. | **Risco elevado de roubo de fundos da conta.** |
| **Admin** | Abrir contas, gerenciar configurações, gerenciar outros tokens. | **Risco de comprometimento total da conta, incluindo a criação de tokens maliciosos.** |
| **Trading info** | Visualizar o histórico de negociações. | Exposição do histórico de performance e estratégias de negociação. |

**AVISO:** A concessão dos escopos `Payments` e `Admin` transfere um risco imenso para o usuário. Use-os com extrema cautela e apenas se for absolutamente essencial para a funcionalidade da sua aplicação.

#### **2.2. Ciclo de Vida do Token (Responsabilidade do Usuário)**

O usuário tem controle total sobre o ciclo de vida do token:
1.  **Autorização:** O usuário gera o token em sua conta e o fornece à sua aplicação.
2.  **Monitoramento:** O usuário pode ver todos os aplicativos conectados e suas permissões.
3.  **Revogação:** O usuário pode revogar o acesso a qualquer momento, invalidando o token imediatamente.

### **3. Gerenciamento da Conexão WebSocket**

A natureza persistente do WebSocket exige um gerenciamento ativo da conexão.

#### **3.1. Fluxo de Comunicação (Eventos e Métodos)**

*   **Eventos (Reações do Cliente):**
    *   `OnOpen`: Disparado quando a conexão é estabelecida. Use este evento para autenticar a sessão.
    *   `OnMessage`: Disparado a cada nova mensagem do servidor (preços, confirmações, etc.). É aqui que a lógica principal da sua aplicação reside.
    *   `OnClose`: Disparado quando a conexão é encerrada.
    *   `OnError`: Disparado em caso de erro de comunicação.
*   **Métodos (Ações do Cliente):**
    *   `Send()`: Envia uma requisição em formato JSON para o servidor.
    *   `Close()`: Encerra a conexão de forma ordenada.

#### **3.2. Manutenção da Conexão (Heartbeat)**

*   **Timeout:** A sessão WebSocket expirará após **2 minutos de inatividade**.
*   **Solução:** Para manter a conexão ativa, sua aplicação **deve** implementar uma estratégia de "heartbeat", enviando uma requisição leve (como `ping`) em intervalos regulares (ex: a cada 30 segundos).

#### **3.3. Limites de Requisição (Rate Limits)**

*   A API impõe limites no número de chamadas por período.
*   **Instrução:** Em vez de usar valores fixos, sua aplicação deve consultar dinamicamente os limites atuais através da chamada `website_status` e verificar o campo `api_call_limits` para se adaptar a mudanças.

### **4. Principais Funcionalidades da API**

A API é modular e cobre uma vasta gama de funcionalidades:

*   **APIs de Dados de Mercado:** Para obter dados de mercado em tempo real e informações sobre contratos disponíveis para negociação.
*   **APIs de Negociação:** Para executar ordens de compra e venda em diversos instrumentos (Opções Digitais, Accumulators, Vanilla, Turbo, Multipliers).
*   **APIs de Gestão de Conta e Caixa:** Para acessar saldos, extratos, histórico e realizar depósitos/saques (requer escopo `Payments`).
*   **APIs Auxiliares:** Para integração com MT5, funcionalidades P2P, e outras utilidades.

### **5. Ferramentas e Recursos para Desenvolvedores**

*   **API Explorer:** Uma ferramenta online interativa ("Playground") fornecida pela Deriv. É essencial para:
    *   Aprender a estrutura das chamadas de API.
    *   Prototipar e testar requisições sem escrever código.
    *   Depurar chamadas que não funcionam como esperado.
*   **Suporte:**
    *   **Email:** Contato direto com a equipe de suporte para questões técnicas.
    *   **Comunidade Telegram:** Canal para discutir ideias e interagir com outros desenvolvedores.
# 🎯 PLANO: Executar Ordem na Deriv API

**Objetivo Principal:** Fazer a aplicação executar uma ordem de compra real na plataforma Deriv.

**Data:** 2025-11-06
**Status:** 📝 Planejamento

---

## 📊 ANÁLISE DA SITUAÇÃO ATUAL

### ✅ O que JÁ EXISTE:

1. **Backend Python (FastAPI)**
   - Localização: `backend/main.py`
   - WebSocket Manager implementado
   - Endpoints REST disponíveis

2. **Cliente Deriv API**
   - Localização: `backend/deriv_api.py`
   - Classe `DerivAPI` completa
   - Método `buy()` implementado (linha 344)
   - 16 funcionalidades essenciais da API

3. **Frontend React**
   - Localização: `frontend/`
   - Interface dashboard existente
   - Hooks e serviços prontos

### ❌ O que FALTA:

1. **Teste funcional isolado** - Não há script simples para testar ordem
2. **Endpoint específico para ordem** - Precisa ser exposto no backend
3. **Interface para executar ordem** - Botão/formulário no frontend
4. **Validação end-to-end** - Teste completo do fluxo

---

## 🎯 ESTRATÉGIA DE IMPLEMENTAÇÃO

### Abordagem: **Bottom-Up (Base → Topo)**

Vamos construir de baixo para cima, testando cada camada antes de subir:

```
┌─────────────────────────────────────┐
│   CAMADA 4: Interface Frontend      │ ← Último
├─────────────────────────────────────┤
│   CAMADA 3: Endpoint Backend        │ ← Terceiro
├─────────────────────────────────────┤
│   CAMADA 2: Cliente Deriv API       │ ← Segundo (já existe)
├─────────────────────────────────────┤
│   CAMADA 1: Teste Isolado/Prova    │ ← Primeiro
└─────────────────────────────────────┘
```

---

## 📋 PLANO DE EXECUÇÃO DETALHADO

### **FASE 1: PROVA DE CONCEITO (POC)**

**Objetivo:** Validar que conseguimos executar uma ordem via código

#### Passo 1.1: Criar Script de Teste Isolado
- **Arquivo:** `backend/test_simple_order.py`
- **Função:** Script standalone que executa uma ordem completa
- **Não depende de:** Frontend, servidor rodando, banco de dados

**Fluxo do Script:**
```python
1. Configurar parâmetros (token, símbolo, valor)
2. Conectar WebSocket Deriv
3. Autenticar com token API
4. Obter proposta de contrato (proposal)
5. Validar proposta (preço, payout)
6. Executar compra (buy)
7. Exibir resultado (contract_id, status)
8. Desconectar
```

**Validações:**
- [ ] Conexão estabelecida
- [ ] Autenticação OK
- [ ] Proposta recebida
- [ ] Ordem executada
- [ ] Contract ID retornado

#### Passo 1.2: Executar Teste Manual
```bash
cd backend
python test_simple_order.py
```

**Resultado Esperado:**
```
✅ Conectado à Deriv API
✅ Autenticado (LoginID: CR123456)
✅ Proposta obtida (Payout: $1.85)
✅ Ordem executada (Contract ID: 12345678)
📊 Status: ATIVO
💰 Resultado aguardando...
```

---

### **FASE 2: INTEGRAÇÃO BACKEND**

**Objetivo:** Expor funcionalidade via API REST

#### Passo 2.1: Criar Endpoint `/api/order/execute`
- **Arquivo:** `backend/main.py`
- **Método:** POST
- **Autenticação:** Token via header ou body

**Request Body:**
```json
{
  "token": "seu_token_deriv",
  "contract_type": "CALL",
  "symbol": "R_75",
  "amount": 1.0,
  "duration": 5,
  "duration_unit": "m"
}
```

**Response Success:**
```json
{
  "success": true,
  "contract_id": 12345678,
  "buy_price": 1.00,
  "payout": 1.85,
  "longcode": "Win payout if Volatility 75 Index is strictly higher than...",
  "status": "active"
}
```

**Response Error:**
```json
{
  "success": false,
  "error": "Insufficient balance",
  "details": "..."
}
```

#### Passo 2.2: Adicionar Tratamento de Erros
- Validar token antes de executar
- Verificar saldo disponível
- Validar parâmetros da ordem
- Timeout de 30s para execução
- Log detalhado de todas as operações

#### Passo 2.3: Testar Endpoint via cURL/Postman
```bash
curl -X POST http://localhost:8000/api/order/execute \
  -H "Content-Type: application/json" \
  -d '{
    "token": "...",
    "contract_type": "CALL",
    "symbol": "R_75",
    "amount": 1.0,
    "duration": 5
  }'
```

**Validações:**
- [ ] Endpoint responde
- [ ] Valida token inválido
- [ ] Executa ordem real
- [ ] Retorna dados corretos
- [ ] Loga operação

---

### **FASE 3: INTERFACE FRONTEND**

**Objetivo:** Criar interface para usuário executar ordens

#### Passo 3.1: Criar Componente `OrderExecutor`
- **Arquivo:** `frontend/src/components/OrderExecutor.tsx`
- **Localização na UI:** Dashboard principal ou página dedicada

**Elementos do Formulário:**
- Campo: Token API (password/text)
- Select: Tipo de contrato (CALL/PUT)
- Select: Símbolo (R_75, R_100, etc)
- Input: Valor da aposta (USD)
- Input: Duração (minutos)
- Botão: "Executar Ordem"
- Display: Resultado da operação

#### Passo 3.2: Criar Serviço `orderService.ts`
```typescript
// frontend/src/services/orderService.ts
export const executeOrder = async (params: OrderParams) => {
  const response = await fetch('/api/order/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return response.json();
};
```

#### Passo 3.3: Integrar no Dashboard
- Adicionar rota `/order/execute`
- Adicionar item no menu
- Feedback visual (loading, success, error)
- Histórico de ordens executadas

**Validações:**
- [ ] Formulário valida campos
- [ ] Loading state durante execução
- [ ] Exibe sucesso com contract ID
- [ ] Exibe erro amigável
- [ ] Limpa formulário após sucesso

---

### **FASE 4: VALIDAÇÃO END-TO-END**

**Objetivo:** Testar fluxo completo em ambiente real

#### Passo 4.1: Teste em Conta Demo
1. Abrir frontend (http://localhost:5173)
2. Navegar para "Executar Ordem"
3. Preencher formulário com token DEMO
4. Clicar "Executar Ordem"
5. Verificar contrato na plataforma Deriv
6. Aguardar resultado do contrato

#### Passo 4.2: Validações de Segurança
- [ ] Token não é exposto nos logs
- [ ] Validação de saldo antes da ordem
- [ ] Rate limiting (máx 10 ordens/min)
- [ ] Confirmação antes de executar
- [ ] Histórico auditável

#### Passo 4.3: Testes de Erro
- Token inválido
- Saldo insuficiente
- Símbolo indisponível
- Mercado fechado
- Timeout de rede

---

## 🔧 REQUISITOS TÉCNICOS

### Backend:
- Python 3.11+
- FastAPI
- websockets
- python-deriv-api (ou implementação própria)

### Frontend:
- React 18+
- TypeScript
- Fetch API / Axios

### Deriv API:
- **Token API:** Necessário (criar em https://app.deriv.com/account/api-token)
- **Scopes necessários:** `Read` + `Trade`
- **Ambiente:** Demo (app_id: 1089) ou Real (app_id próprio)

---

## 📝 CHECKLIST DE IMPLEMENTAÇÃO

### Fase 1: POC
- [ ] Criar `test_simple_order.py`
- [ ] Testar conexão WebSocket
- [ ] Testar autenticação
- [ ] Testar obtenção de proposta
- [ ] Testar execução de ordem
- [ ] Validar resultado

### Fase 2: Backend
- [ ] Criar endpoint `/api/order/execute`
- [ ] Implementar validações
- [ ] Adicionar tratamento de erros
- [ ] Adicionar logging
- [ ] Testar via cURL/Postman
- [ ] Documentar API (OpenAPI/Swagger)

### Fase 3: Frontend
- [ ] Criar componente `OrderExecutor`
- [ ] Criar serviço `orderService`
- [ ] Integrar no dashboard
- [ ] Adicionar validações de UI
- [ ] Adicionar feedback visual
- [ ] Testar responsividade

### Fase 4: Validação
- [ ] Teste end-to-end em Demo
- [ ] Validações de segurança
- [ ] Testes de erro
- [ ] Performance test
- [ ] Documentação de uso

---

## ⚠️ CONSIDERAÇÕES DE SEGURANÇA

### 🔒 Token API:
- **NUNCA** commitar tokens no código
- Usar variáveis de ambiente (.env)
- Token configurado pelo usuário via UI
- Não logar tokens completos

### 💰 Gestão de Risco:
- Limite máximo por ordem (ex: $10)
- Confirmação para ordens > $5
- Histórico completo de ordens
- Alerta de saldo baixo

### 🛡️ Validações:
- Verificar saldo antes da ordem
- Validar parâmetros da ordem
- Timeout para prevenir travamentos
- Rate limiting para evitar spam

---

## 📊 CRITÉRIOS DE SUCESSO

### ✅ Objetivo Cumprido Quando:

1. **Script de teste executa ordem com sucesso**
   - Conecta → Autentica → Proposta → Compra → Resultado

2. **Endpoint backend funcional**
   - Recebe requisição → Executa → Retorna resultado

3. **Interface frontend operacional**
   - Usuário preenche → Clica → Vê resultado

4. **Validação end-to-end**
   - Ordem aparece na plataforma Deriv
   - Resultado é retornado corretamente

---

## 🚀 PRÓXIMOS PASSOS (Após Objetivo 1)

Após validar a execução de ordens, podemos evoluir para:

1. **Estratégias Automatizadas**
   - Bot que executa ordens baseado em sinais
   - Análise técnica automática

2. **Gestão de Portfolio**
   - Múltiplas ordens simultâneas
   - Diversificação automática

3. **Backtesting**
   - Testar estratégias com dados históricos

4. **Machine Learning**
   - Predição de movimentos
   - Otimização de parâmetros

---

## 📚 REFERÊNCIAS

- [Deriv API Documentation](https://api.deriv.com/docs/)
- [Deriv API Playground](https://api.deriv.com/api-explorer)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- Arquivo local: `backend/deriv_api.py` (implementação atual)

---

## 📞 SUPORTE

Em caso de dúvidas técnicas:
- Documentação: `docs/deriv-api-buy-endpoint.md`
- README principal: `README.md`
- Issues: GitHub Issues

---

**Documento criado em:** 2025-11-06
**Última atualização:** 2025-11-06
**Versão:** 1.0
**Status:** 📝 Pronto para execução

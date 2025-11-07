# 🧪 Como Testar Execução de Ordem

**Script:** `test_simple_order.py`
**Objetivo:** Validar execução de ordem na Deriv API

---

## ⚡ QUICK START (3 minutos)

### 0. Instalar Dependências (PRIMEIRO!)

**Se você receber erro `ModuleNotFoundError: No module named 'websockets'`:**

**Windows:**
```bash
cd backend
install_dependencies.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Ou manualmente:**
```bash
pip install websockets ujson fastapi uvicorn pydantic python-dotenv
```

### 1. Obter Token API

```
1. Acesse: https://app.deriv.com/account/api-token
2. Clique "Create new token"
3. Nome: "Synth Bot Test"
4. Scopes: ☑️ Read, ☑️ Trade
5. Copie o token gerado
```

### 2. Configurar Token

**Opção A: Editar arquivo**
```bash
# Abrir arquivo
notepad test_simple_order.py

# Linha 15, trocar:
TOKEN = "SEU_TOKEN_AQUI"
# Por:
TOKEN = "seu_token_copiado"
```

**Opção B: Variável de ambiente (recomendado)**
```bash
# Windows
set DERIV_TOKEN=seu_token_aqui

# Linux/Mac
export DERIV_TOKEN=seu_token_aqui
```

### 3. Executar Teste

```bash
cd backend
python test_simple_order.py
```

---

## 📊 RESULTADO ESPERADO

### ✅ Sucesso

```
🚀 TESTE DE EXECUÇÃO DE ORDEM NA DERIV
============================================================

1️⃣ Conectando à Deriv API...
✅ Conectado com sucesso

2️⃣ Autenticando com token...
✅ Autenticado
   LoginID: VRTC12345
   Saldo: 10000.00 USD

3️⃣ Obtendo proposta...
✅ Proposta obtida
   Preço: $1.00
   Payout: $1.85

👉 Deseja continuar? (sim/não): sim

4️⃣ Executando ordem...
✅ ORDEM EXECUTADA COM SUCESSO!

📊 DETALHES DA ORDEM:
   Contract ID: 123456789
   Preço pago: $1.00
   Descrição: Win payout if Volatility 75 Index...

🔗 Ver contrato na plataforma:
   https://app.deriv.com/contract/123456789

============================================================
✅ TESTE CONCLUÍDO COM SUCESSO
============================================================
```

### ❌ Erros Comuns

#### Erro: "Token não configurado"
```
❌ ERRO: Token não configurado!
```
**Solução:** Configure o token (veja Passo 2)

#### Erro: "Autenticação falhou"
```
❌ Erro de autenticação: InvalidToken
```
**Solução:**
- Verifique se copiou o token corretamente
- Confirme que o token tem scopes Read + Trade
- Crie um novo token se necessário

#### Erro: "Saldo insuficiente"
```
⚠️  AVISO: Saldo insuficiente!
```
**Solução:**
- Use conta Demo (saldo virtual ilimitado)
- Ou reduza o valor: `AMOUNT = 0.35` (mínimo)

---

## 🔧 CONFIGURAÇÕES AVANÇADAS

### Alterar Parâmetros

Edite as linhas 15-19 em `test_simple_order.py`:

```python
TOKEN = "seu_token"          # Seu token API
SYMBOL = "R_75"              # R_75, R_100, R_50, etc
CONTRACT_TYPE = "CALL"       # CALL (Rise) ou PUT (Fall)
AMOUNT = 1.0                 # Valor em USD (mín: 0.35)
DURATION = 5                 # Duração em minutos (1-60)
```

### Símbolos Disponíveis

| Código | Nome | Volatilidade |
|--------|------|--------------|
| R_75 | Volatility 75 Index | Alta |
| R_100 | Volatility 100 Index | Muito Alta |
| R_50 | Volatility 50 Index | Média |
| R_25 | Volatility 25 Index | Baixa |

### Modo Automático (Sem Confirmação)

```bash
# Windows
set AUTO_CONFIRM=true
python test_simple_order.py

# Linux/Mac
AUTO_CONFIRM=true python test_simple_order.py
```

---

## 🐛 TROUBLESHOOTING

### Python não encontrado
```bash
# Verificar instalação
python --version
# ou
python3 --version

# Instalar se necessário
# Windows: https://python.org
# Ubuntu: sudo apt install python3
# Mac: brew install python3
```

### Módulo deriv_api não encontrado
```bash
# Verificar se arquivo existe
ls deriv_api.py

# Se não existir, você está no diretório errado
cd backend
```

### Timeout / Conexão falhou
```
❌ TIMEOUT: Operação demorou muito tempo
```
**Soluções:**
1. Verificar conexão com internet
2. Desabilitar VPN/Proxy
3. Verificar firewall
4. Tentar novamente

### WebSocket não conecta
```
❌ Falha na conexão
```
**Soluções:**
1. Verificar se wss://ws.derivws.com está acessível
2. Testar em: https://api.deriv.com/api-explorer
3. Verificar bloqueio de firewall/antivírus

---

## ✅ VALIDAÇÃO DO TESTE

### Checklist de Sucesso

- [ ] Script executou sem erros
- [ ] Autenticação bem-sucedida
- [ ] Proposta obtida com preço válido
- [ ] Ordem executada (Contract ID recebido)
- [ ] Link do contrato funciona
- [ ] Contrato aparece na plataforma Deriv

### Verificar no Deriv

1. Abra o link do contrato: `https://app.deriv.com/contract/SEU_CONTRACT_ID`
2. Ou acesse: https://app.deriv.com/reports/positions
3. Confirme que o contrato aparece
4. Aguarde 5 minutos para resultado

---

## 📝 PRÓXIMOS PASSOS

### ✅ Teste Bem-Sucedido?

**Parabéns! Fase 1 completa. Agora:**

1. **Fase 2:** Criar endpoint backend
   - Arquivo: `backend/models/order_models.py`
   - Endpoint: `POST /api/order/execute`
   - Guia: [GUIA-RAPIDO-IMPLEMENTACAO.md](../docs/GUIA-RAPIDO-IMPLEMENTACAO.md#fase-2-endpoint-backend-45-min)

2. **Fase 3:** Criar interface frontend
   - Componente: `OrderExecutor.tsx`
   - Serviço: `orderService.ts`
   - Guia: [GUIA-RAPIDO-IMPLEMENTACAO.md](../docs/GUIA-RAPIDO-IMPLEMENTACAO.md#fase-3-interface-frontend-60-min)

3. **Fase 4:** Teste end-to-end
   - Frontend → Backend → Deriv API
   - Validação completa

---

## 📞 SUPORTE

### Ainda com problemas?

1. **Consulte documentação completa:**
   - [GUIA-RAPIDO-IMPLEMENTACAO.md](../docs/GUIA-RAPIDO-IMPLEMENTACAO.md)
   - [ARQUITETURA-EXECUCAO-ORDEM.md](../docs/ARQUITETURA-EXECUCAO-ORDEM.md)

2. **Verifique troubleshooting:**
   - [GUIA-RAPIDO → Troubleshooting](../docs/GUIA-RAPIDO-IMPLEMENTACAO.md#troubleshooting)

3. **Crie issue no GitHub:**
   - Inclua output completo do erro
   - Sistema operacional
   - Versão do Python

---

**Boa sorte! 🚀**

*Criado: 2025-11-06*
*Objetivo 1 - Fase 1*

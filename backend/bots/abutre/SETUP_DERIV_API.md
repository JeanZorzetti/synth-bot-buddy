# GUIA DE CONFIGURAÇÃO - DERIV API TOKEN

**Objetivo:** Configurar acesso à Deriv API para o Abutre Bot em modo DEMO (Paper Trading)

---

## 📋 PRÉ-REQUISITOS

1. **Conta Deriv criada** (https://deriv.com)
2. **Conta DEMO ativada** (é automático ao criar conta)
3. **Acesso ao painel de API Tokens**

---

## 🔑 PASSO 1: OBTER TOKEN DA DERIV API

### 1.1. Acessar Painel de API Tokens

1. Acesse: **https://app.deriv.com/account/api-token**
2. Faça login na sua conta Deriv
3. Você verá a página "API Token Manager"

### 1.2. Criar Novo Token (DEMO)

1. Clique em **"Create new token"**
2. Configure as permissões:

   ```
   Token name: Abutre Bot Demo

   Scopes (permissões):
   ✅ Read           - Ler dados da conta
   ✅ Trade          - Executar trades
   ✅ Payments       - Ver transações
   ✅ Trading information - Ver informações de trading
   ✅ Admin          - Gerenciar conta

   Account: [Selecione sua conta DEMO]
   - Procure por conta que contém "VRTC" (Virtual)
   - Exemplo: VRTC12345678
   ```

3. Clique em **"Create"**
4. **COPIE O TOKEN GERADO** (você só verá UMA VEZ!)
   - Exemplo: `a1-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890AbCdEfGh`

### 1.3. Verificar Tipo de Conta

**IMPORTANTE:** Use APENAS token de conta **DEMO (VRTC)**

```
✅ CORRETO: Token de conta DEMO (VRTC12345678)
❌ ERRADO:  Token de conta REAL (CR12345678)
```

**Como identificar:**
- Conta DEMO: Login ID começa com **"VRTC"** (Virtual)
- Conta REAL: Login ID começa com **"CR"** (Cash Real)

---

## ⚙️ PASSO 2: CONFIGURAR VARIÁVEIS DE AMBIENTE

### 2.1. Localizar Arquivo .env

```bash
cd backend/bots/abutre
```

### 2.2. Criar .env (se não existe)

```bash
# Copiar template
cp .env.example .env
```

### 2.3. Editar .env

Abra o arquivo `.env` e configure:

```bash
# ==================== DERIV API ====================

# Cole aqui o token DEMO que você copiou
DERIV_API_TOKEN=a1-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890AbCdEfGh

# URL do WebSocket (NÃO ALTERE)
DERIV_WS_URL=wss://ws.derivws.com/websockets/v3

# Símbolo para trading (V100 = Volatility 100 Index)
SYMBOL=1HZ100V

# Tipo de contrato (CALL = comprar, PUT = vender)
CONTRACT_TYPE=CALL

# Duração (1 tick = 1 segundo)
DURATION=1
DURATION_UNIT=t

# ==================== STRATEGY PARAMETERS ====================

# ATENÇÃO: Estes parâmetros foram validados em backtest
# Só altere se souber o que está fazendo!

DELAY_THRESHOLD=8          # Aguardar 8 velas consecutivas
MAX_LEVEL=10               # Máximo de 10 níveis Martingale
INITIAL_STAKE=1.0          # Apostar $1.00 por trade
MULTIPLIER=2.0             # Dobrar stake a cada nível
BANKROLL=2000.0            # Banca inicial $2000
MAX_DRAWDOWN_PCT=0.25      # Parar se drawdown > 25%

# ==================== RISK MANAGEMENT ====================

# MODO PAPER TRADING (SEGURANÇA)
AUTO_TRADING=false         # false = Apenas observar, NÃO executar trades

# Spread por trade (5%)
SPREAD_PCT=0.05

# Saldo mínimo de segurança
MIN_BALANCE=500.0

# ==================== SYSTEM ====================

DB_PATH=backend/bots/abutre/data/abutre.db
LOG_LEVEL=INFO
LOG_FILE=backend/bots/abutre/logs/abutre.log
```

### 2.4. Salvar e Fechar

---

## ✅ PASSO 3: VALIDAR CONFIGURAÇÃO

### 3.1. Verificar Token Carregado

Execute o bot em modo de teste:

```bash
cd backend/bots/abutre
python main.py --demo --paper-trading
```

**Saída esperada:**

```
[INFO] Abutre Bot v1.0.0 - Delayed Martingale System
[INFO] Mode: DEMO | Paper Trading: ON
[INFO] Connecting to Deriv API...
[INFO] ✓ Connected to Deriv WebSocket
[INFO] ✓ Authorized account: VRTC12345678
[INFO] ✓ Balance: $10,000.00 (DEMO)
[INFO] Symbol: V100 (1HZ100V)
[INFO] Strategy: Delay=8, MaxLevel=10, Stake=$1.00
[INFO] WebSocket server started on port 8000
[INFO] Bot ready. Waiting for signals...
```

### 3.2. Verificar Erros Comuns

**Erro: "DERIV_API_TOKEN is required"**
```
Causa: Token não configurado no .env
Solução: Verifique se você colou o token corretamente
```

**Erro: "Invalid token"**
```
Causa: Token expirado ou inválido
Solução: Gere um novo token no painel da Deriv
```

**Erro: "Authorization failed"**
```
Causa: Token sem permissões corretas
Solução: Crie novo token com scopes: Read, Trade, Payments, Trading information
```

**Erro: "Account is not virtual"**
```
Causa: Token é de conta REAL, não DEMO
Solução: Use token de conta VRTC (Virtual)
```

---

## 🔄 PASSO 4: INTEGRAÇÃO COM FRONTEND

### 4.1. Iniciar Backend

```bash
# Terminal 1 - Backend Abutre Bot
cd backend/bots/abutre
python main.py --demo --paper-trading
```

**Aguarde até ver:**
```
[INFO] WebSocket server started on port 8000
[INFO] Bot ready. Waiting for signals...
```

### 4.2. Acessar Dashboard

1. Acesse: **http://localhost:8080/abutre** (ou URL do Vercel)
2. Faça login com suas credenciais
3. Você verá o dashboard em tempo real:
   - ✅ Conexão: Verde (conectado ao backend)
   - ✅ Saldo: $10,000.00 (DEMO)
   - ✅ Bot Status: RUNNING
   - ✅ Modo: Paper Trading (observação)

### 4.3. Verificar Comunicação WebSocket

No dashboard, você deve ver:

```
System Status: Online
WebSocket: Connected ✓
Bot Mode: Paper Trading
Current Balance: $10,000.00
```

**Eventos em tempo real:**
- ✅ `balance_update` - Atualização de saldo
- ✅ `market_data` - Dados do mercado (preço, streak)
- ✅ `trigger_detected` - Gatilho detectado (8+ velas)
- ✅ `trade_opened` - Trade simulado aberto
- ✅ `trade_closed` - Trade simulado fechado

---

## 🧪 PASSO 5: TESTAR PAPER TRADING

### 5.1. Modo Observação (Padrão)

Com `AUTO_TRADING=false`, o bot irá:

1. ✅ Conectar à Deriv API
2. ✅ Receber ticks em tempo real
3. ✅ Detectar gatilhos (8+ velas consecutivas)
4. ✅ Calcular sinais de entrada
5. ✅ **SIMULAR** trades (NÃO executa de verdade)
6. ✅ Registrar resultados no banco de dados

**Segurança:**
- ❌ NENHUM trade real é executado
- ✅ Zero risco financeiro
- ✅ Validação da estratégia com dados reais

### 5.2. Monitorar Logs

```bash
# Terminal 2 - Logs em tempo real
tail -f backend/bots/abutre/logs/abutre.log
```

**Eventos esperados:**

```
[INFO] New candle: V100 | Close: 1234.56 | Color: RED
[INFO] Streak detected: 8 consecutive RED candles
[INFO] Trigger activated! Direction: CALL (buy)
[INFO] [PAPER TRADING] Trade opened: Level 1, Stake: $1.00
[INFO] [PAPER TRADING] Trade closed: WIN | Profit: +$0.95
[INFO] Balance updated: $10,000.95
```

### 5.3. Verificar Dashboard

No frontend, você verá:

**Metrics Cards:**
- Current Balance: $10,000.95 (+0.01%)
- Win Rate: 100% (1/1)
- Max Drawdown: 0%

**Market Monitor:**
- Current Streak: 8 RED ⬇️
- Status: TRIGGERED ⚡
- Next Signal: CALL

**Trades Table:**
```
Time       | Direction | Level | Result | P&L    | Balance
10:30:45   | CALL      | 1     | WIN    | +$0.95 | $10,000.95
```

---

## 📊 PASSO 6: FORWARD TEST (30 DIAS)

### 6.1. Objetivo

Validar estratégia com **dados reais** antes de usar dinheiro real.

**Métricas alvo (30 dias):**
- ✅ ROI > +5%
- ✅ Win Rate > 90%
- ✅ Max Drawdown < 30%
- ✅ Zero busts

### 6.2. Configuração

```bash
# .env
AUTO_TRADING=false        # Manter em Paper Trading
BANKROLL=2000.0           # Simular banca de $2000
INITIAL_STAKE=1.0         # Stake inicial $1.00
DELAY_THRESHOLD=8         # Parâmetros validados
MAX_LEVEL=10
```

### 6.3. Monitoramento

**Diário:**
1. Verificar dashboard: `/abutre`
2. Conferir logs: `backend/bots/abutre/logs/`
3. Registrar métricas:
   - Balance atual
   - Total trades
   - Win rate
   - Max drawdown

**Semanal:**
1. Exportar dados: `backend/bots/abutre/data/abutre.db`
2. Analisar gráfico de equity
3. Comparar com backtest
4. Ajustar se necessário

### 6.4. Critérios de Aprovação

**Se após 30 dias:**

✅ **APROVADO** (pode avançar para Live Micro):
- ROI > +5%
- Win Rate > 90%
- Max DD < 30%
- 0 busts

❌ **REPROVADO** (ajustar parâmetros):
- ROI < 0%
- Win Rate < 80%
- Busts ocorreram
- **Ação:** Aumentar DELAY_THRESHOLD para 10 e repetir

---

## 🔐 SEGURANÇA

### Boas Práticas

1. **NUNCA commitar .env**
   ```bash
   # Verificar se .env está no .gitignore
   grep ".env" .gitignore

   # Se não estiver, adicionar:
   echo ".env" >> .gitignore
   ```

2. **Usar tokens diferentes para DEMO e REAL**
   ```
   .env.demo   - Token VRTC (demo)
   .env.prod   - Token CR (real) - NUNCA commitar!
   ```

3. **Rotacionar tokens periodicamente**
   - Criar novo token a cada 90 dias
   - Revogar tokens antigos

4. **Limitar permissões**
   - Token DEMO: Todas as permissões OK
   - Token REAL: Apenas Read + Trade (sem Admin)

### Checklist de Segurança

- [ ] Token é de conta DEMO (VRTC)
- [ ] `.env` está no `.gitignore`
- [ ] `AUTO_TRADING=false` (paper trading)
- [ ] `BANKROLL` configurado corretamente
- [ ] `MAX_DRAWDOWN_PCT` configurado (25%)
- [ ] Logs estão sendo salvos
- [ ] Dashboard mostra "Paper Trading" mode

---

## 🆘 TROUBLESHOOTING

### Problema: Bot não conecta

**Sintomas:**
```
[ERROR] Failed to connect to Deriv API
[ERROR] WebSocket connection failed
```

**Soluções:**
1. Verificar internet
2. Verificar se token está correto
3. Testar conexão manual:
   ```bash
   curl -X POST https://ws.derivws.com/websockets/v3 \
     -d '{"authorize":"SEU_TOKEN"}'
   ```

### Problema: Token inválido

**Sintomas:**
```
[ERROR] Authorization failed: Invalid token
```

**Soluções:**
1. Gerar novo token na Deriv
2. Verificar se copiou token completo
3. Verificar se token não expirou

### Problema: Trades não aparecem

**Sintomas:**
- Bot roda sem erros
- Dashboard não mostra trades

**Soluções:**
1. Verificar se `AUTO_TRADING=false` (esperado em paper trading)
2. Aguardar gatilho (8+ velas consecutivas)
3. Verificar logs: `tail -f logs/abutre.log`
4. Verificar WebSocket: Dashboard deve mostrar "Connected"

---

## 📞 SUPORTE

### Recursos Oficiais

- **Documentação Deriv API:** https://api.deriv.com/
- **Painel de API Tokens:** https://app.deriv.com/account/api-token
- **WebSocket Playground:** https://api.deriv.com/api-explorer
- **Suporte Deriv:** https://deriv.com/contact-us

### Logs Úteis

```bash
# Ver logs em tempo real
tail -f backend/bots/abutre/logs/abutre.log

# Filtrar apenas erros
grep ERROR backend/bots/abutre/logs/abutre.log

# Ver últimas 100 linhas
tail -n 100 backend/bots/abutre/logs/abutre.log

# Buscar por "trade"
grep -i "trade" backend/bots/abutre/logs/abutre.log
```

### Verificar Configuração

```bash
# Mostrar variáveis carregadas (SEM mostrar token completo)
cd backend/bots/abutre
python -c "
from config import AbutreConfig
config = AbutreConfig()
print(f'Token configurado: {bool(config.DERIV_API_TOKEN)}')
print(f'Symbol: {config.SYMBOL}')
print(f'Delay: {config.DELAY_THRESHOLD}')
print(f'Max Level: {config.MAX_LEVEL}')
print(f'Auto Trading: {config.AUTO_TRADING}')
"
```

---

## ✅ CHECKLIST FINAL

Antes de iniciar Forward Test de 30 dias:

### Configuração
- [ ] Token DEMO obtido da Deriv
- [ ] `.env` criado e configurado
- [ ] `AUTO_TRADING=false` (Paper Trading)
- [ ] `DERIV_API_TOKEN` preenchido
- [ ] Parâmetros validados configurados

### Testes
- [ ] Bot conecta à Deriv API
- [ ] Autorização bem-sucedida
- [ ] Balance mostra $10,000 (DEMO)
- [ ] WebSocket server iniciado
- [ ] Dashboard acessível em `/abutre`
- [ ] Conexão WebSocket funciona

### Segurança
- [ ] `.env` no `.gitignore`
- [ ] Token é de conta DEMO (VRTC)
- [ ] Logs estão sendo salvos
- [ ] Sem erros no console

### Monitoramento
- [ ] Dashboard mostra métricas em tempo real
- [ ] Logs salvando eventos
- [ ] Banco de dados salvando trades
- [ ] Equity curve atualizando

---

**Status:** ✅ Configuração completa - Pronto para Forward Test

**Próximo passo:** Deixar bot rodando por 30 dias e monitorar métricas diariamente.

---

*Última atualização: 2025-12-22*

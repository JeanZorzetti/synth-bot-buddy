# SINCRONIZAR HISTÓRICO REAL DA DERIV

## 🎯 O QUE ISSO FAZ

Busca os **últimos 100 trades REAIS** da sua conta Deriv e mostra no dashboard.

**SEM DADOS MOCK, SEM FAKE, DADOS 100% REAIS DA SUA CONTA!**

---

## ⚡ PASSO A PASSO (5 MINUTOS)

### 1. Criar API Token na Deriv

1. Acesse: **https://app.deriv.com/account/api-token**

2. Clique em **"Create new token"**

3. Configure as permissões:
   - ✅ **Read** (obrigatório)
   - ✅ **Trade** (obrigatório)
   - ✅ **Trading information** (obrigatório)
   - ❌ Payments (NÃO precisa)
   - ❌ Admin (NÃO precisa)

4. Nome do token: "Abutre Dashboard"

5. Clique em **"Create"**

6. **COPIE O TOKEN** (ele aparece UMA VEZ SÓ!)
   - Exemplo: `a1-bC3dE5fG7hI9jK0lM2nO4pQ6rS8tU1vW3xY5zA7bC9dE`

---

### 2. Executar o Script

**Windows (PowerShell):**

```powershell
# Configurar token
$env:DERIV_API_TOKEN="seu_token_aqui"

# Executar script
python sync_deriv_history.py
```

**Linux/Mac:**

```bash
# Configurar token
export DERIV_API_TOKEN='seu_token_aqui'

# Executar script
python sync_deriv_history.py
```

---

### 3. Ver Resultado

Você vai ver no terminal:

```
============================================================
SYNC DERIV HISTORY → ABUTRE DASHBOARD
============================================================

🔐 Fazendo login na Deriv...
✅ Login bem-sucedido!
   Conta: CR1234567
   Balance: $10,523.45
   Currency: USD

📊 Buscando histórico de trades...
✅ 87 trades encontrados!

📤 Enviando trades para dashboard...

  ✅ 45678901... | CALL | WIN  |    +$0.95
  ✅ 45678902... | PUT  | LOSS |    -$1.00
  ✅ 45678903... | CALL | WIN  |    +$0.95
  ...

============================================================
✅ Sincronização concluída!
   Trades enviados: 87
   Trades com erro: 0

🌐 Acesse: https://botderiv.roilabs.com.br/abutre
============================================================
```

---

### 4. Abrir Dashboard

Acesse: **https://botderiv.roilabs.com.br/abutre**

Pressione **CTRL + SHIFT + R** (limpar cache)

Você vai ver **TODOS OS SEUS TRADES REAIS** na tabela!

---

## 🔄 LIMPAR DADOS MOCK PRIMEIRO (OPCIONAL)

Se você quer **DELETAR** os 15 trades de teste antes de sincronizar os reais:

```python
# criar clear_database.py
import requests

API = "https://botderivapi.roilabs.com.br/api/abutre"

# ATENÇÃO: Isso vai DELETAR TUDO!
response = requests.delete(f"{API}/clear")
print(f"Database cleared: {response.status_code}")
```

Depois execute:

```bash
python clear_database.py
python sync_deriv_history.py
```

---

## 📊 O QUE SERÁ SINCRONIZADO

Para cada trade real da Deriv:

| Campo Deriv          | Campo Dashboard     | Exemplo            |
|---------------------|---------------------|-------------------|
| contract_id         | trade_id            | 123456789         |
| purchase_time       | entry_time          | 2025-12-23 10:30  |
| contract_type       | direction           | CALL / PUT        |
| buy_price           | initial_stake       | $1.00             |
| sell_price - buy_price | profit           | +$0.95 / -$1.00   |
| profit > 0 ? WIN : LOSS | result          | WIN / LOSS        |
| sell_time           | exit_time           | 2025-12-23 10:31  |

---

## ⚙️ AUTOMATIZAR (RODAR A CADA 5 MINUTOS)

**Windows (Task Scheduler):**

1. Criar arquivo `sync_loop.bat`:

```batch
@echo off
:loop
python sync_deriv_history.py
timeout /t 300
goto loop
```

2. Executar: `sync_loop.bat`

**Linux/Mac (cron):**

```bash
# Editar crontab
crontab -e

# Adicionar linha (rodar a cada 5 minutos)
*/5 * * * * cd /path/to/project && python sync_deriv_history.py >> sync.log 2>&1
```

---

## 🛠️ TROUBLESHOOTING

### Erro: "DERIV_API_TOKEN não configurado"

**Solução**: Você esqueceu de configurar a variável de ambiente.

```powershell
$env:DERIV_API_TOKEN="seu_token_aqui"
```

### Erro: "Invalid token"

**Solução**: Token inválido ou expirado. Crie um novo em https://app.deriv.com/account/api-token

### Erro: "Connection timeout"

**Solução**: Problema de rede. Verifique se consegue acessar https://ws.derivws.com

### Dashboard ainda mostra "Nenhum trade encontrado"

**Solução**: Limpe cache do browser (CTRL + SHIFT + R)

---

## 🔐 SEGURANÇA

**NUNCA COMPARTILHE SEU API TOKEN!**

- ❌ Não commite no Git
- ❌ Não compartilhe em screenshots
- ❌ Não envie por email/chat

Se vazou, revogue imediatamente em: https://app.deriv.com/account/api-token

---

## 🎯 PRÓXIMOS PASSOS

Depois de sincronizar o histórico:

1. ✅ Ver todos os trades reais no dashboard
2. ⏳ Configurar auto-sync a cada 5 minutos
3. ⏳ Adicionar filtros (data, direção, resultado)
4. ⏳ Gráfico de equity com dados reais

---

**IMPORTANTE**: Este script é **READ-ONLY**. Ele apenas **LÊ** o histórico, **NÃO EXECUTA** nenhum trade novo!

Para executar trades automaticamente, use o `deriv_to_abutre_bridge.py` junto com a estratégia Abutre.

---

**Atualizado**: 2025-12-23 10:50 GMT

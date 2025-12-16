# 🗄️ Como Criar Database trades.db em Produção

**Status**: ⚠️ Database trades.db não existe no servidor
**Impacto**: Trade History page vazia (não mostra os 3 trades de exemplo)
**Solução**: Executar script Python no servidor

---

## 🚀 Opção 1: Via Easypanel Console (Recomendado)

### Passo 1: Acessar Console do Container

1. Entre no **Easypanel**: https://easypanel.io/
2. Vá para o projeto **botderiv**
3. Clique no serviço backend
4. Clique em **"Console"** ou **"Terminal"**

### Passo 2: Executar Script

No terminal que abrir, execute:

```bash
cd /app
python backend/database/setup.py
```

**Saída esperada:**
```
✅ Criando database em: /app/backend/trades.db
✅ Tabela trades_history criada
✅ Índice 1/5 criado
✅ Índice 2/5 criado
✅ Índice 3/5 criado
✅ Índice 4/5 criado
✅ Índice 5/5 criado
✅ Trade de exemplo 1/3 inserido
✅ Trade de exemplo 2/3 inserido
✅ Trade de exemplo 3/3 inserido

✅ Setup completo! Database tem 3 trades de exemplo
```

### Passo 3: Verificar Criação

```bash
ls -lh /app/backend/trades.db
# Deve mostrar: -rw-r--r-- ... 32K ... trades.db
```

### Passo 4: Testar API

```bash
curl http://localhost:8000/api/trades/stats
```

**Resposta esperada:**
```json
{
  "overall": {
    "total_trades": 3,
    "wins": 2,
    "losses": 1,
    "win_rate": 66.67,
    "total_pnl": 5.0,
    "avg_profit": 2.5,
    "avg_loss": -5.0
  },
  "by_symbol": [...],
  "by_strategy": [...],
  "recent_performance": [...]
}
```

---

## 🚀 Opção 2: Via SSH (Se Disponível)

Se você tem acesso SSH direto ao servidor:

```bash
# Conectar
ssh usuario@seu-servidor.com

# Encontrar container
docker ps | grep botderiv

# Entrar no container
docker exec -it <container-id> bash

# Executar script
cd /app
python backend/database/setup.py

# Verificar
ls -lh backend/trades.db
curl http://localhost:8000/api/trades/stats
```

---

## 🚀 Opção 3: Criar Manualmente via Python (Fallback)

Se as opções acima não funcionarem, crie diretamente no console Python:

```bash
# No terminal do container
cd /app
python3
```

Depois cole este código:

```python
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("/app/backend/trades.db")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Criar tabela
cursor.execute("""
CREATE TABLE IF NOT EXISTS trades_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL DEFAULT 1.0,
    position_size REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    profit_loss REAL,
    profit_loss_pct REAL,
    result TEXT,
    strategy TEXT,
    confidence REAL,
    ml_prediction TEXT,
    indicators TEXT,
    notes TEXT,
    closed_at TEXT,
    duration_seconds INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
""")

# Criar índices
for idx in ['timestamp', 'symbol', 'result', 'strategy', 'created_at']:
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{idx} ON trades_history({idx});")

# Inserir trade de exemplo
now = datetime.now()
cursor.execute("""
    INSERT INTO trades_history (
        timestamp, symbol, direction, entry_price, exit_price, quantity,
        position_size, stop_loss, take_profit, profit_loss, profit_loss_pct,
        result, strategy, confidence, ml_prediction, indicators, notes,
        closed_at, duration_seconds
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    (now - timedelta(days=1)).isoformat(), 'R_100', 'UP', 100.50, 101.25, 1.0,
    1000, 99.50, 102.50, 7.5, 0.75, 'win', 'ML_Predictor', 0.75, 'UP',
    json.dumps({'rsi': 65}), 'Trade de exemplo', (now - timedelta(hours=23)).isoformat(), 3600
))

conn.commit()
print(f"✅ Database criado com sucesso!")
conn.close()
exit()
```

---

## ✅ Como Verificar que Funcionou

### 1. Via API
```bash
curl https://botderivapi.roilabs.com.br/api/trades/stats
```

Deve retornar JSON com:
- `total_trades: 3`
- `wins: 2`
- `losses: 1`
- `win_rate: 66.67`

### 2. Via Frontend

Acesse: https://botderiv.roilabs.com.br/trade-history

Deve mostrar:
- **3 trades na tabela**
- **Win Rate**: 66.67%
- **Total P&L**: +$5.00
- **Gráfico** com dados

### 3. Via Logs do Backend

No Easypanel, vá em **Logs** e procure por:
```
INFO: "GET /api/trades/stats HTTP/1.1" 200 OK
```

---

## 🐛 Troubleshooting

### Erro: "No such file or directory: python"

Use `python3` em vez de `python`:
```bash
python3 backend/database/setup.py
```

### Erro: "Permission denied"

Adicione sudo (se disponível):
```bash
sudo python backend/database/setup.py
```

Ou execute como root:
```bash
docker exec -u root -it <container-id> python /app/backend/database/setup.py
```

### Erro: "Database já existe"

O database foi criado! Para recriar:
```bash
rm /app/backend/trades.db
python backend/database/setup.py
```

### Frontend ainda vazio após criar database

1. **Restart do backend**:
   ```bash
   # No Easypanel, clique em "Restart"
   ```

2. **Clear cache do navegador**:
   - Ctrl+Shift+R (Windows/Linux)
   - Cmd+Shift+R (Mac)

3. **Verificar API diretamente**:
   ```bash
   curl https://botderivapi.roilabs.com.br/api/trades/stats
   ```

---

## 📊 Resultado Esperado

Após executar o script:

| Componente | ANTES | DEPOIS |
|------------|-------|--------|
| **Database** | ❌ Não existe | ✅ 32KB com 3 trades |
| **Trade History** | 🔴 Vazio | ✅ Mostra 3 trades |
| **API /trades/stats** | 🔴 Empty | ✅ JSON com stats |
| **Win Rate Card** | 🔴 0% | ✅ 66.67% |
| **Total P&L Card** | 🔴 $0 | ✅ +$5.00 |

---

**Tempo estimado**: 2 minutos
**Dificuldade**: ⭐ Fácil

🚀 Execute e veja a Trade History page ganhar vida!

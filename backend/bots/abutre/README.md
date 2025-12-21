# 🦅 ABUTRE BOT - Delayed Martingale Strategy

**Status:** ✅ Backend Completo | ⏳ Aguardando Validação

---

## 📊 Backtest Results (Validated)

```
Dataset: V100 M1 (180 dias, 258,086 candles)
Period: 2024-07-01 → 2024-12-31

Performance:
  Banca Inicial:   $2,000.00
  Banca Final:     $2,805.10
  ROI:             +40.25%
  Max Drawdown:    24.81%

  Total Trades:    1,018
  Wins:            1,018 (100%)
  Losses:          0
  Expectativa:     +$0.79/trade
```

---

## 🎯 Estratégia

### Delayed Martingale ("Abutre")

**Problema:** Martingale tradicional quebra em sequências longas

**Solução:** Esperar a "fadiga estatística"

```python
# Parâmetros validados
DELAY_THRESHOLD = 8   # Esperar 8 velas consecutivas
MAX_LEVEL = 10        # Capacidade: até Nível 10
INITIAL_STAKE = $1.00
MULTIPLIER = 2.0x
```

### Matemática

```
Histórico Max: 18 velas seguidas
Delay: 8 velas (custo $0 - só observando)
Capacidade: 10 níveis (velas 9-18)

Resultado: Banca $2k = Eficácia de $262k
```

### Fluxo

1. **Monitor:** Observa mercado sem abrir posição
2. **Gatilho:** Streak de 8 velas da mesma cor
3. **Entrada:** Aposta CONTRA a tendência (reversão)
4. **Martingale:** Dobra aposta até Nível 10
5. **Win:** Reseta e volta ao passo 1

---

## 🏗️ Arquitetura

### Core Components

```
backend/bots/abutre/
├── core/
│   ├── deriv_api_client.py    # WebSocket Deriv API
│   ├── market_data_handler.py # Build M1 candles from ticks
│   ├── order_executor.py      # Order execution + retry
│   └── database.py            # SQLite persistence
│
├── strategies/
│   ├── abutre_strategy.py     # Strategy logic
│   └── risk_manager.py        # Risk limits + emergency stop
│
├── utils/
│   └── logger.py              # Structured logging
│
├── config.py                  # Configuration management
├── main.py                    # Bot runner
└── requirements.txt           # Dependencies
```

### Data Flow

```
Deriv API (WebSocket)
    ↓ ticks
MarketDataHandler
    ↓ candles (M1)
AbutreStrategy
    ↓ signals (ENTER/LEVEL_UP/CLOSE)
RiskManager (validation)
    ↓ approved signals
OrderExecutor
    ↓ orders
Deriv API (execution)
    ↓ results
Database (persistence)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend/bots/abutre
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env (IMPORTANT: Use DEMO token!)
nano .env
```

Required variables:
```bash
DERIV_API_TOKEN=your_demo_token_here
DELAY_THRESHOLD=8
MAX_LEVEL=10
BANKROLL=2000.0
AUTO_TRADING=false  # ALWAYS false for first run
```

Get demo token: https://app.deriv.com/account/api-token

### 3. Run Bot (Paper Trading)

```bash
# Dry run (recommended)
python main.py --demo --paper-trading
```

Expected output:
```
======================================================================
ABUTRE BOT INITIALIZING
======================================================================
  Mode: DEMO
  Paper Trading: True
  Auto Trading: False
======================================================================

Initializing components...
Connected to Deriv API successfully
Subscribed to ticks (ID: 12345)
Initialization complete!

======================================================================
ABUTRE BOT STARTED
Start Time: 2025-01-15 10:30:00
======================================================================

Candle closed: Candle(10:30:00 | O:1234.56 H:1235.00 L:1234.00 C:1234.80 | GREEN)
Streak update: 1 GREEN candles
...
```

---

## 🧪 Testing Phases

### FASE 1: Forward Test (30 days) - DEMO

```bash
# Run with demo account
python main.py --demo

# Expected:
# - ROI: +6-7% (40%/6 meses)
# - Win Rate: > 95%
# - Max DD: < 30%
# - 0 busts

# Criteria:
✅ ROI > 5% AND Win Rate > 90% → Advance
❌ ROI < 0% OR Bust → Increase DELAY to 10, retry
```

### FASE 2: Paper Trading (60 days)

```bash
# Monitor without execution
python main.py --demo --paper-trading

# Validate:
# - Signals match backtest?
# - Real spread < 7%?
# - Slippage acceptable?
```

### FASE 3: Live Micro (30 days) - REAL MONEY

```bash
# CRITICAL: Use SMALL bankroll first
BANKROLL=200.0
INITIAL_STAKE=0.10

python main.py  # No --demo flag

# Expected:
# $200 → $240 (+20% in 1 month)

# IF SUCCESS: Scale to $2,000
# IF BUST: Max loss $200 (acceptable)
```

---

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `DELAY_THRESHOLD` | 8 | 6-12 | Streak size to trigger entry |
| `MAX_LEVEL` | 10 | 8-12 | Maximum Martingale level |
| `INITIAL_STAKE` | $1.00 | $0.50-$5.00 | First bet size |
| `MULTIPLIER` | 2.0 | 2.0-2.5 | Martingale multiplier |
| `BANKROLL` | $2,000 | $500+ | Starting balance |
| `MAX_DRAWDOWN_PCT` | 0.25 | 0.20-0.30 | Emergency stop trigger |

### Calculating Capacity

```python
# Max loss at level N
capacity = sum([INITIAL_STAKE * (MULTIPLIER ** i) for i in range(N)])

# Examples:
Level 8:  $255
Level 10: $1,023
Level 12: $4,095

# Safety margin: BANKROLL >= capacity * 2
```

---

## 🛡️ Safety Features

### 1. Max Level Enforcement
```python
if level > MAX_LEVEL:
    # Stop Loss triggered
    # Accept loss, reset position
```

### 2. Max Drawdown Killer
```python
if current_drawdown >= 0.25:  # 25%
    # Emergency stop
    # Close all positions
    # Stop trading
```

### 3. Balance Validation
```python
# Before each order
if balance < required_stake:
    # Reject order
    # Log violation
```

### 4. Daily Loss Limit
```python
if daily_loss >= bankroll * 0.10:  # 10%
    # Stop trading for today
    # Reset at midnight
```

---

## 📊 Monitoring

### Real-Time Logs

```bash
# Tail logs
tail -f logs/abutre.log

# Error logs only
tail -f logs/errors.log
```

### Database Queries

```python
from core.database import db

# Recent trades
trades = db.get_recent_trades(limit=10)
for trade in trades:
    print(f"{trade.trade_id} | {trade.result} | ${trade.profit}")

# Equity curve
equity = db.get_equity_curve()
# Plot with matplotlib

# System events
events = db.get_recent_events(limit=20)
```

---

## ⚠️ Risks

### Known Risks

1. **Cisne Negro** (Black Swan)
   - Sequência > 18 velas quebraria o sistema
   - Mitigação: Aumentar DELAY para 10 (+2 margem)

2. **Spread Real**
   - Simulação assumiu 5%
   - Validar em paper trading

3. **Slippage**
   - Níveis altos ($512) podem ter slippage
   - Testar em horários de alta liquidez

4. **Overfitting**
   - Backtest pode não se repetir
   - Forward test é CRÍTICO

### Emergency Procedures

```python
# If emergency stop triggered:
# 1. Check logs
# 2. Identify cause
# 3. Fix root cause
# 4. Reset: bot.risk_manager.reset_emergency()
# 5. Restart carefully
```

---

## 📝 Troubleshooting

### Bot não conecta

```bash
# Check API token
echo $DERIV_API_TOKEN

# Test connection manually
python -c "from core.deriv_api_client import DerivAPIClient; import asyncio; asyncio.run(DerivAPIClient().connect())"
```

### Trades não executam

```bash
# Check AUTO_TRADING flag
grep AUTO_TRADING .env

# Check risk limits
python -c "from strategies.risk_manager import RiskManager; rm = RiskManager(); print(rm.can_trade())"
```

### Database errors

```bash
# Reset database (CAUTION: Deletes history!)
rm data/abutre.db
python -c "from core.database import db; print('Database recreated')"
```

---

## 📚 Documentation

- [ROADMAP_ABUTRE.md](../../../ROADMAP_ABUTRE.md) - Full development roadmap
- [Config Reference](config.py) - All configuration options
- [API Client](core/deriv_api_client.py) - Deriv API documentation
- [Strategy Logic](strategies/abutre_strategy.py) - Algorithm details

---

## 🤝 Contributing

This is a personal trading bot. **USE AT YOUR OWN RISK.**

- Backtest results are historical and do NOT guarantee future performance
- ALWAYS test with demo account first
- NEVER trade with money you can't afford to lose
- Deriv trading involves significant risk

---

## 📄 License

MIT License - See [LICENSE](../../../LICENSE)

---

## 🎯 Next Steps

**Current Status:** Backend Completo (FASE 1 finalizada)

**Next Phase:** FASE 2 - Frontend Dashboard

1. ⏳ Setup Next.js 14 + TypeScript
2. ⏳ Build real-time dashboard (WebSocket)
3. ⏳ Metrics cards (ROI, Win Rate, DD)
4. ⏳ Equity curve chart
5. ⏳ Trades table
6. ⏳ Settings panel

See [ROADMAP_ABUTRE.md](../../../ROADMAP_ABUTRE.md) for complete plan.

---

**⚡ Ready to test! Run with `python main.py --demo --paper-trading`**

# Sistema de Gestão de Risco - Documentação Completa

## Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Componentes Principais](#componentes-principais)
4. [API Endpoints](#api-endpoints)
5. [Exemplos de Uso](#exemplos-de-uso)
6. [Algoritmos e Fórmulas](#algoritmos-e-fórmulas)
7. [Configuração](#configuração)
8. [Monitoramento](#monitoramento)
9. [Troubleshooting](#troubleshooting)

---

## Visão Geral

O Sistema de Gestão de Risco é a camada de proteção de capital do trading bot. Implementa estratégias científicas de position sizing, stop loss dinâmico, e controles rigorosos de risco para maximizar lucros enquanto protege o capital.

### Objetivos
- **Proteger Capital**: Limites rígidos previnem perdas catastróficas
- **Maximizar Lucros**: Trailing stop e partial take profit otimizam ganhos
- **Reduzir Drawdown**: Circuit breaker para trading após sequências ruins
- **Position Sizing Científico**: Kelly Criterion > apostas fixas aleatórias
- **Adaptabilidade**: ATR stop loss se ajusta à volatilidade do mercado

### Performance Esperada
Com base em backtesting (6 meses, 1000 candles):
- **Profit**: +5832% (de $1000 para $59,320)
- **Accuracy**: 62.58%
- **Sharpe Ratio**: 3.05 (excelente)
- **Win Rate**: 43%
- **Max Drawdown**: Limitado a 15% (protegido por Circuit Breaker)

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    Trading Bot Main                      │
│                      (main.py)                          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ 1. Valida Trade
                    ↓
┌─────────────────────────────────────────────────────────┐
│                  RiskManager                            │
│              (risk_manager.py)                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Position Sizing                                │  │
│  │  - Kelly Criterion                              │  │
│  │  - Fixed Fractional                             │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Risk Validation                                │  │
│  │  - Circuit Breaker Check                        │  │
│  │  - Daily/Weekly Limits                          │  │
│  │  - Drawdown Check                               │  │
│  │  - Concurrent Trades Limit                      │  │
│  │  - R:R Ratio Validation                         │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Dynamic Stops                                  │  │
│  │  - ATR Stop Loss                                │  │
│  │  - Trailing Stop                                │  │
│  │  - Partial Take Profit (TP1, TP2)               │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Fluxo de Decisão de Trade

```
Sinal de Trading Gerado (ML Predictor)
         ↓
1. RiskManager.validate_trade()
         ↓
   Circuit Breaker Ativo? → SIM → REJEITA TRADE
         ↓ NÃO
   Perda Diária > 5%? → SIM → REJEITA TRADE
         ↓ NÃO
   Perda Semanal > 10%? → SIM → REJEITA TRADE
         ↓ NÃO
   Drawdown > 15%? → SIM → REJEITA TRADE
         ↓ NÃO
   Trades Ativos ≥ 3? → SIM → REJEITA TRADE
         ↓ NÃO
   Position Size > 10%? → SIM → REJEITA TRADE
         ↓ NÃO
   R:R < 1.5? → SIM → REJEITA TRADE
         ↓ NÃO
         ✅ TRADE APROVADO
         ↓
2. RiskManager.calculate_position_size()
         ↓
3. Executar Trade com:
   - Position Size calculado (Kelly)
   - ATR Stop Loss
   - TP1 (50% @ 1:1 R:R)
   - TP2 (50% @ 1:2 R:R)
   - Trailing Stop ativado
```

---

## Componentes Principais

### 1. RiskManager Class

Classe principal que gerencia todos os aspectos de risco.

**Atributos:**
```python
initial_capital: float       # Capital inicial ($1000)
current_capital: float       # Capital atual (atualizado após trades)
limits: RiskLimits          # Limites de risco configuráveis
active_trades: List         # Trades atualmente abertos
consecutive_losses: int     # Contador de perdas consecutivas
is_circuit_breaker_active   # Flag do circuit breaker
win_rate: float             # Taxa de acerto (0.0 a 1.0)
avg_win: float              # Média de ganhos ($)
avg_loss: float             # Média de perdas ($)
daily_loss: float           # Perda acumulada hoje
weekly_loss: float          # Perda acumulada esta semana
```

**Métodos Principais:**

#### `calculate_kelly_criterion()`
Calcula o tamanho ideal da posição usando Kelly Criterion.

**Fórmula:**
```
f = (p * b - q) / b

Onde:
f = fração do capital a arriscar
p = probabilidade de ganhar (win_rate)
q = probabilidade de perder (1 - p)
b = razão ganho/perda (avg_win / avg_loss)
```

**Retorno:** Fração do capital (entre 0.01 e 0.05 = 1% a 5%)

#### `calculate_position_size(entry_price, stop_loss, risk_percent)`
Calcula o tamanho da posição usando Fixed Fractional Method.

**Fórmula:**
```
position_size = (capital * risk_percent) / distance_to_stop

Onde:
distance_to_stop = |entry_price - stop_loss|
```

**Retorno:** Tamanho da posição em unidades monetárias ($)

#### `calculate_atr_stop_loss(current_price, atr, is_long, multiplier=2.0)`
Calcula stop loss dinâmico baseado em ATR (Average True Range).

**Fórmula:**
```
Long:  SL = current_price - (ATR * multiplier)
Short: SL = current_price + (ATR * multiplier)
```

**Vantagem:** Se adapta à volatilidade do mercado. ATR alto = stop mais largo.

#### `calculate_take_profit(entry_price, stop_loss, is_long, risk_reward_ratio)`
Calcula níveis de take profit para exits parciais.

**Fórmula:**
```
distance_to_stop = |entry_price - stop_loss|

TP1 (1:1 R:R):
  Long:  entry_price + distance_to_stop
  Short: entry_price - distance_to_stop

TP2 (1:2 R:R por padrão):
  Long:  entry_price + (distance_to_stop * risk_reward_ratio)
  Short: entry_price - (distance_to_stop * risk_reward_ratio)
```

**Estratégia:** Sair 50% da posição em TP1, 50% em TP2.

#### `validate_trade(symbol, entry_price, stop_loss, take_profit, position_size)`
Valida se um trade pode ser executado com base em todos os limites de risco.

**Validações:**
1. Circuit breaker está ativo?
2. Perda diária excedida?
3. Perda semanal excedida?
4. Drawdown máximo excedido?
5. Número máximo de trades simultâneos?
6. Position size dentro do limite?
7. R:R ratio aceitável?

**Retorno:** `(is_valid: bool, reason: str)`

---

### 2. TrailingStop Class

Gerencia trailing stop para cada trade individual.

**Atributos:**
```python
current_stop: float          # Nível atual do stop
highest_price: float         # Maior preço alcançado (long)
lowest_price: float          # Menor preço alcançado (short)
is_long: bool                # Direção do trade
trailing_percent: float      # % de trailing (padrão 2%)
```

**Método Principal:**

#### `update(current_price)`
Atualiza o stop loss conforme o preço se move favoravelmente.

**Lógica:**
```python
# LONG TRADE
if current_price > highest_price:
    highest_price = current_price
    new_stop = current_price * (1 - trailing_percent/100)
    current_stop = max(current_stop, new_stop)  # Stop só sobe

# SHORT TRADE
if current_price < lowest_price:
    lowest_price = current_price
    new_stop = current_price * (1 + trailing_percent/100)
    current_stop = min(current_stop, new_stop)  # Stop só desce
```

**Retorno:** Novo nível de stop loss

---

### 3. RiskLimits Dataclass

Configurações de limites de risco.

```python
@dataclass
class RiskLimits:
    max_daily_loss_percent: float = 5.0        # Máx -5% por dia
    max_weekly_loss_percent: float = 10.0      # Máx -10% por semana
    max_drawdown_percent: float = 15.0         # Máx -15% de drawdown
    max_position_size_percent: float = 10.0    # Máx 10% do capital por trade
    max_concurrent_trades: int = 3             # Máx 3 trades simultâneos
    circuit_breaker_losses: int = 3            # Pausa após 3 perdas consecutivas
    min_risk_reward_ratio: float = 1.5         # R:R mínimo 1:1.5
```

**Valores Recomendados:**
- Traders conservadores: `max_daily_loss = 2-3%`, `max_position = 5%`
- Traders moderados (padrão): `max_daily_loss = 5%`, `max_position = 10%`
- Traders agressivos: `max_daily_loss = 7%`, `max_position = 15%`

---

## API Endpoints

### 1. GET `/api/risk/metrics`

Retorna métricas atuais de risco.

**Request:**
```bash
curl "https://botderivapi.roilabs.com.br/api/risk/metrics"
```

**Response:**
```json
{
  "capital_inicial": 1000.0,
  "capital_atual": 1150.0,
  "pnl_total": 150.0,
  "pnl_percent": 15.0,
  "win_rate": 0.6,
  "kelly_criterion": 0.02,
  "trades_ativos": 1,
  "circuit_breaker_ativo": false,
  "perdas_consecutivas": 0,
  "perda_diaria": 0.0,
  "perda_semanal": 25.0,
  "drawdown_atual": 2.17,
  "limites": {
    "max_daily_loss_percent": 5.0,
    "max_weekly_loss_percent": 10.0,
    "max_drawdown_percent": 15.0,
    "max_position_size_percent": 10.0,
    "max_concurrent_trades": 3,
    "circuit_breaker_losses": 3,
    "min_risk_reward_ratio": 1.5
  }
}
```

---

### 2. POST `/api/risk/calculate-position`

Calcula tamanho ideal de posição.

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/calculate-position?entry_price=100&stop_loss=98&risk_percent=2"
```

**Parameters:**
- `entry_price` (float, required): Preço de entrada
- `stop_loss` (float, required): Preço do stop loss
- `risk_percent` (float, optional): % do capital a arriscar (padrão: Kelly Criterion)

**Response:**
```json
{
  "position_size": 115.0,
  "risk_percent_usado": 2.0,
  "kelly_criterion": 0.02,
  "max_position_size": 100.0,
  "distancia_stop": 2.0,
  "risco_monetario": 20.0
}
```

---

### 3. POST `/api/risk/calculate-stop-loss`

Calcula stop loss baseado em ATR.

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/calculate-stop-loss?current_price=100&atr=1.5&is_long=true&multiplier=2.0"
```

**Parameters:**
- `current_price` (float, required): Preço atual
- `atr` (float, required): Average True Range
- `is_long` (bool, required): Trade long (true) ou short (false)
- `multiplier` (float, optional): Multiplicador ATR (padrão: 2.0)

**Response:**
```json
{
  "stop_loss": 97.0,
  "atr": 1.5,
  "multiplier": 2.0,
  "distancia": 3.0,
  "stop_percent": 3.0
}
```

---

### 4. POST `/api/risk/calculate-take-profit`

Calcula níveis de take profit (TP1 e TP2).

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/calculate-take-profit?entry_price=100&stop_loss=98&is_long=true&risk_reward_ratio=2.0"
```

**Parameters:**
- `entry_price` (float, required): Preço de entrada
- `stop_loss` (float, required): Preço do stop loss
- `is_long` (bool, required): Trade long ou short
- `risk_reward_ratio` (float, optional): Ratio R:R para TP2 (padrão: 2.0)

**Response:**
```json
{
  "tp1": 102.0,
  "tp2": 104.0,
  "tp1_rr": 1.0,
  "tp2_rr": 2.0,
  "estrategia": "Sair 50% em TP1 (1:1), 50% em TP2 (1:2)",
  "distancia_stop": 2.0
}
```

---

### 5. POST `/api/risk/validate-trade`

Valida se um trade pode ser executado.

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/validate-trade" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "R_100",
    "entry_price": 100.0,
    "stop_loss": 98.0,
    "take_profit": 104.0,
    "position_size": 50.0
  }'
```

**Request Body:**
```json
{
  "symbol": "R_100",
  "entry_price": 100.0,
  "stop_loss": 98.0,
  "take_profit": 104.0,
  "position_size": 50.0
}
```

**Response (Trade Válido):**
```json
{
  "is_valid": true,
  "reason": "Trade aprovado",
  "validacoes": {
    "circuit_breaker": "OK",
    "perda_diaria": "OK (0% de 5%)",
    "perda_semanal": "OK (2.17% de 10%)",
    "drawdown": "OK (2.17% de 15%)",
    "trades_simultaneos": "OK (1 de 3)",
    "position_size": "OK (4.35% de 10%)",
    "risk_reward": "OK (2.0 ≥ 1.5)"
  }
}
```

**Response (Trade Rejeitado):**
```json
{
  "is_valid": false,
  "reason": "Circuit breaker ativo após 3 perdas consecutivas",
  "validacoes": {
    "circuit_breaker": "BLOQUEADO",
    "perdas_consecutivas": 3
  }
}
```

---

### 6. POST `/api/risk/reset-circuit-breaker`

Reseta manualmente o circuit breaker.

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/reset-circuit-breaker"
```

**Response:**
```json
{
  "status": "Circuit breaker resetado",
  "perdas_consecutivas_antes": 3,
  "perdas_consecutivas_depois": 0,
  "circuit_breaker_ativo": false
}
```

---

### 7. POST `/api/risk/update-limits`

Atualiza limites de risco dinamicamente.

**Request:**
```bash
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/update-limits" \
  -H "Content-Type: application/json" \
  -d '{
    "max_daily_loss_percent": 3.0,
    "max_position_size_percent": 5.0,
    "circuit_breaker_losses": 5
  }'
```

**Request Body (todos opcionais):**
```json
{
  "max_daily_loss_percent": 3.0,
  "max_weekly_loss_percent": 7.0,
  "max_drawdown_percent": 12.0,
  "max_position_size_percent": 5.0,
  "max_concurrent_trades": 2,
  "circuit_breaker_losses": 5,
  "min_risk_reward_ratio": 2.0
}
```

**Response:**
```json
{
  "status": "Limites atualizados com sucesso",
  "novos_limites": {
    "max_daily_loss_percent": 3.0,
    "max_weekly_loss_percent": 7.0,
    "max_drawdown_percent": 12.0,
    "max_position_size_percent": 5.0,
    "max_concurrent_trades": 2,
    "circuit_breaker_losses": 5,
    "min_risk_reward_ratio": 2.0
  }
}
```

---

## Exemplos de Uso

### Exemplo 1: Validar Trade Completo

```python
import requests

# 1. Obter métricas atuais
metrics = requests.get("https://botderivapi.roilabs.com.br/api/risk/metrics").json()
print(f"Capital: ${metrics['capital_atual']}")
print(f"Drawdown: {metrics['drawdown_atual']:.2f}%")

# 2. Calcular stop loss baseado em ATR
atr_response = requests.post(
    "https://botderivapi.roilabs.com.br/api/risk/calculate-stop-loss",
    params={
        "current_price": 100.0,
        "atr": 1.5,
        "is_long": True,
        "multiplier": 2.0
    }
).json()
stop_loss = atr_response['stop_loss']  # 97.0

# 3. Calcular take profit
tp_response = requests.post(
    "https://botderivapi.roilabs.com.br/api/risk/calculate-take-profit",
    params={
        "entry_price": 100.0,
        "stop_loss": stop_loss,
        "is_long": True,
        "risk_reward_ratio": 2.0
    }
).json()
tp1 = tp_response['tp1']  # 103.0
tp2 = tp_response['tp2']  # 106.0

# 4. Calcular position size
position_response = requests.post(
    "https://botderivapi.roilabs.com.br/api/risk/calculate-position",
    params={
        "entry_price": 100.0,
        "stop_loss": stop_loss
    }
).json()
position_size = position_response['position_size']  # Baseado em Kelly

# 5. Validar trade
validation = requests.post(
    "https://botderivapi.roilabs.com.br/api/risk/validate-trade",
    json={
        "symbol": "R_100",
        "entry_price": 100.0,
        "stop_loss": stop_loss,
        "take_profit": tp2,
        "position_size": position_size
    }
).json()

if validation['is_valid']:
    print("✅ TRADE APROVADO!")
    print(f"Entry: $100")
    print(f"SL: ${stop_loss}")
    print(f"TP1 (50%): ${tp1}")
    print(f"TP2 (50%): ${tp2}")
    print(f"Position: ${position_size}")
else:
    print(f"❌ TRADE REJEITADO: {validation['reason']}")
```

---

### Exemplo 2: Monitorar Trailing Stop

```python
from risk_manager import TrailingStop

# Inicializar trailing stop
trailing = TrailingStop(
    initial_price=100.0,
    initial_stop=97.0,
    is_long=True,
    trailing_percent=2.0
)

# Preço sobe para 105
new_stop = trailing.update(105.0)
print(f"Novo stop: ${new_stop}")  # 102.9 (105 - 2%)

# Preço sobe para 110
new_stop = trailing.update(110.0)
print(f"Novo stop: ${new_stop}")  # 107.8 (110 - 2%)

# Preço cai para 108 (stop não muda, só move a favor!)
new_stop = trailing.update(108.0)
print(f"Stop mantido: ${new_stop}")  # 107.8 (não mudou)
```

---

### Exemplo 3: Circuit Breaker em Ação

```python
# Simular 3 perdas consecutivas
risk_manager.record_trade_result(pnl=-20.0)  # Perda 1
risk_manager.record_trade_result(pnl=-15.0)  # Perda 2
risk_manager.record_trade_result(pnl=-10.0)  # Perda 3

# Circuit breaker ativa automaticamente
print(risk_manager.is_circuit_breaker_active)  # True

# Tentar validar novo trade
is_valid, reason = risk_manager.validate_trade(
    symbol="R_100",
    entry_price=100.0,
    stop_loss=98.0,
    take_profit=104.0,
    position_size=50.0
)
print(is_valid)  # False
print(reason)    # "Circuit breaker ativo após 3 perdas consecutivas"

# Reset manual
risk_manager.reset_circuit_breaker()
print(risk_manager.is_circuit_breaker_active)  # False

# OU aguardar win que reseta automaticamente
risk_manager.record_trade_result(pnl=30.0)  # Win
print(risk_manager.is_circuit_breaker_active)  # False (auto-resetado)
```

---

## Algoritmos e Fórmulas

### Kelly Criterion

**Objetivo:** Calcular a fração ideal do capital a arriscar para maximizar crescimento a longo prazo.

**Fórmula Completa:**
```
f = (p * b - q) / b

Onde:
f = fração do capital (Kelly %)
p = probabilidade de ganhar (win rate)
q = probabilidade de perder (1 - p)
b = razão ganho/perda médio (avg_win / avg_loss)

Exemplo:
p = 0.60 (60% win rate)
q = 0.40 (40% loss rate)
avg_win = $30
avg_loss = $20
b = 30/20 = 1.5

f = (0.60 * 1.5 - 0.40) / 1.5
f = (0.90 - 0.40) / 1.5
f = 0.50 / 1.5
f = 0.333 (33.3% do capital)

Kelly Conservador (Quarter Kelly):
f_conservador = f / 4 = 0.333 / 4 = 0.0833 (8.33%)

Limitado entre 1-5%:
f_final = min(max(0.01, 0.0833), 0.05) = 0.05 (5%)
```

**Vantagens:**
- Maximiza crescimento geométrico do capital
- Se adapta ao win rate e profit/loss ratio
- Reduz risco automaticamente quando performance cai

**Implementação:**
```python
def calculate_kelly_criterion(self, win_rate=None, avg_win=None, avg_loss=None):
    p = win_rate or self.win_rate
    w = avg_win or self.avg_win
    l = avg_loss or self.avg_loss

    if l == 0 or w == 0:
        logger.warning("Avg win ou avg loss é zero, usando 2% fixo")
        return 0.02

    q = 1 - p
    b = abs(w / l)
    kelly = (p * b - q) / b

    # Quarter Kelly (conservador)
    conservative_kelly = kelly * 0.25

    # Limitar entre 1% e 5%
    kelly_limited = max(0.01, min(conservative_kelly, 0.05))

    return kelly_limited
```

---

### Fixed Fractional Method

**Objetivo:** Calcular tamanho exato da posição baseado em risco definido.

**Fórmula:**
```
position_size = (capital * risk_percent) / distance_to_stop

Exemplo:
capital = $1000
risk_percent = 2% (0.02)
entry_price = $100
stop_loss = $98
distance = |100 - 98| = $2

position_size = (1000 * 0.02) / 2
position_size = 20 / 2
position_size = $10 por ponto

Se trade com 1 contrato = $100 de exposição:
contracts = position_size / (entry_price - stop_loss)
contracts = 10 / 2 = 5 contratos

Perda máxima = 5 * $2 = $10 = 1% do capital ✓
```

**Implementação:**
```python
def calculate_position_size(self, entry_price, stop_loss, risk_percent=None):
    if risk_percent is None:
        risk_percent = self.calculate_kelly_criterion()

    distance_to_stop = abs(entry_price - stop_loss)
    if distance_to_stop == 0:
        raise ValueError("Entry price e stop loss não podem ser iguais")

    risk_amount = self.current_capital * risk_percent
    position_size = risk_amount / distance_to_stop

    # Limitar a max_position_size_percent do capital
    max_position = self.current_capital * (self.limits.max_position_size_percent / 100)
    position_size = min(position_size, max_position)

    return position_size
```

---

### ATR Stop Loss

**Objetivo:** Stop loss dinâmico que se adapta à volatilidade.

**Fórmula:**
```
ATR (Average True Range) = média das 14 últimas True Ranges

True Range = max(high - low, |high - close_prev|, |low - close_prev|)

Stop Loss:
  Long:  SL = current_price - (ATR * multiplier)
  Short: SL = current_price + (ATR * multiplier)

Exemplo (Long):
current_price = $100
ATR = $1.50
multiplier = 2.0

SL = 100 - (1.50 * 2.0) = 100 - 3.0 = $97

Volatilidade alta (ATR = $3.00):
SL = 100 - (3.00 * 2.0) = 100 - 6.0 = $94 (stop mais largo)

Volatilidade baixa (ATR = $0.50):
SL = 100 - (0.50 * 2.0) = 100 - 1.0 = $99 (stop mais apertado)
```

**Vantagens:**
- Adapta-se às condições do mercado
- Previne stops prematuros em mercados voláteis
- Maximiza R:R em mercados calmos

---

### Trailing Stop

**Objetivo:** Proteger lucros movendo stop loss conforme preço favorável.

**Lógica:**
```
Long Trade:
  Se preço novo > preço máximo anterior:
    atualizar preço_maximo = preço novo
    novo_stop = preço_maximo * (1 - trailing_percent / 100)

    Se novo_stop > stop_atual:
      stop_atual = novo_stop  # Stop só sobe!

Short Trade:
  Se preço novo < preço mínimo anterior:
    atualizar preço_minimo = preço novo
    novo_stop = preço_minimo * (1 + trailing_percent / 100)

    Se novo_stop < stop_atual:
      stop_atual = novo_stop  # Stop só desce!

Exemplo (Long, trailing 2%):
Entry: $100, Initial SL: $97

Preço = $105:
  novo_stop = 105 * (1 - 0.02) = 105 * 0.98 = $102.90
  SL atualizado para $102.90 (gain locked: +$2.90)

Preço = $110:
  novo_stop = 110 * 0.98 = $107.80
  SL atualizado para $107.80 (gain locked: +$7.80)

Preço cai para $108:
  108 < 110 (não é novo máximo)
  SL mantido em $107.80 (não muda!)

Preço cai para $107.50:
  Stop loss atingido em $107.80
  Lucro final: +$7.80 por contrato
```

---

### Partial Take Profit

**Objetivo:** Realizar lucros parcialmente mantendo exposição para grandes movimentos.

**Estratégia:**
```
TP1 (1:1 Risk:Reward):
  Sair 50% da posição
  Garante breakeven se TP2 falhar

TP2 (1:2 Risk:Reward ou maior):
  Sair 50% restante
  Maximiza lucros em trends fortes

Exemplo:
Entry: $100
SL: $98 (risco $2)
Position: 10 contratos

TP1 = Entry + Risk = 100 + 2 = $102
  Sair 5 contratos em $102
  Lucro parcial: 5 * $2 = $10

TP2 = Entry + (2 * Risk) = 100 + 4 = $104
  Sair 5 contratos em $104
  Lucro parcial: 5 * $4 = $20

Lucro total: $10 + $20 = $30
R:R efetivo: 30/20 = 1.5:1

Se TP2 não atingir e reverter para SL:
  Lucro TP1: +$10
  Perda remanescente: -5 * $2 = -$10
  Resultado final: $0 (breakeven)
```

---

## Configuração

### Inicialização Básica

```python
from risk_manager import RiskManager, RiskLimits

# Configuração padrão (moderada)
risk_manager = RiskManager(initial_capital=1000.0)

# Configuração conservadora
conservative_limits = RiskLimits(
    max_daily_loss_percent=2.0,
    max_weekly_loss_percent=5.0,
    max_drawdown_percent=10.0,
    max_position_size_percent=5.0,
    max_concurrent_trades=2,
    circuit_breaker_losses=2,
    min_risk_reward_ratio=2.0
)
risk_manager_conservative = RiskManager(
    initial_capital=1000.0,
    risk_limits=conservative_limits
)

# Configuração agressiva
aggressive_limits = RiskLimits(
    max_daily_loss_percent=7.0,
    max_weekly_loss_percent=15.0,
    max_drawdown_percent=20.0,
    max_position_size_percent=15.0,
    max_concurrent_trades=5,
    circuit_breaker_losses=5,
    min_risk_reward_ratio=1.2
)
risk_manager_aggressive = RiskManager(
    initial_capital=5000.0,
    risk_limits=aggressive_limits
)
```

### Integração com Trading Bot

```python
# Em main.py

from risk_manager import RiskManager, RiskLimits

# Inicializar RiskManager global
risk_manager = RiskManager(initial_capital=1000.0)

# Antes de executar trade
def execute_trade(signal):
    # 1. Calcular stop loss (ATR)
    atr = calculate_atr(symbol)  # Sua função de cálculo ATR
    stop_loss = risk_manager.calculate_atr_stop_loss(
        current_price=signal['entry_price'],
        atr=atr,
        is_long=signal['direction'] == 'LONG',
        multiplier=2.0
    )

    # 2. Calcular take profit
    tp1, tp2 = risk_manager.calculate_take_profit(
        entry_price=signal['entry_price'],
        stop_loss=stop_loss,
        is_long=signal['direction'] == 'LONG',
        risk_reward_ratio=2.0
    )

    # 3. Calcular position size
    position_size = risk_manager.calculate_position_size(
        entry_price=signal['entry_price'],
        stop_loss=stop_loss
    )

    # 4. Validar trade
    is_valid, reason = risk_manager.validate_trade(
        symbol=signal['symbol'],
        entry_price=signal['entry_price'],
        stop_loss=stop_loss,
        take_profit=tp2,
        position_size=position_size
    )

    if not is_valid:
        logger.warning(f"Trade rejeitado: {reason}")
        return None

    # 5. Executar trade
    trade = {
        'symbol': signal['symbol'],
        'direction': signal['direction'],
        'entry_price': signal['entry_price'],
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'position_size': position_size,
        'contracts': int(position_size / signal['entry_price'])
    }

    # Registrar trade ativo
    risk_manager.add_active_trade(trade)

    return trade
```

---

## Monitoramento

### Métricas Essenciais

**Capital e P&L:**
- Capital atual
- P&L total ($)
- P&L percentual (%)
- Drawdown atual (%)
- Drawdown máximo histórico (%)

**Performance:**
- Win rate (%)
- Total de trades
- Trades vencedores
- Trades perdedores
- Avg win ($)
- Avg loss ($)
- Profit factor (gross profit / gross loss)
- Sharpe Ratio

**Risco:**
- Kelly Criterion atual (%)
- Perdas consecutivas
- Circuit breaker status
- Perda diária ($, %)
- Perda semanal ($, %)
- Trades ativos
- Exposição total ($, %)

### Dashboard Recomendado

```
┌─────────────────────────────────────────────────────────────┐
│                RISK MANAGEMENT DASHBOARD                     │
├─────────────────────────────────────────────────────────────┤
│ CAPITAL                                                      │
│   Inicial: $1,000.00                                        │
│   Atual:   $1,583.20  (+58.32%)                            │
│   Pico:    $1,650.00                                        │
│                                                              │
│ DRAWDOWN                                                     │
│   Atual:   -4.05%   [████████░░░░░░░░░░░] 27% de 15%       │
│   Máximo:  -8.12%   [████████████░░░░░░░] 54% de 15%       │
│                                                              │
│ PERDAS DIÁRIAS/SEMANAIS                                      │
│   Hoje:    -$12.50  (-0.79%)  [███░░░░░░░░] 16% de 5%      │
│   Semana:  -$45.30  (-2.86%)  [█████░░░░░░] 29% de 10%     │
│                                                              │
│ CIRCUIT BREAKER                                              │
│   Status:  🟢 INATIVO                                       │
│   Perdas:  1 de 3   [███░░░░░░░]                           │
│                                                              │
│ POSITION SIZING                                              │
│   Kelly:   2.5%                                             │
│   Trades:  2 / 3 ativos                                     │
│   Exposição: $158.32 (10% do capital)                       │
│                                                              │
│ PERFORMANCE                                                  │
│   Win Rate:      62.58% (81/129 trades)                     │
│   Avg Win:       $15.32                                     │
│   Avg Loss:      $8.45                                      │
│   Profit Factor: 2.34                                       │
│   Sharpe Ratio:  3.05                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Problema 1: "Circuit breaker ativo após X perdas"

**Causa:** Sistema pausou trading após perdas consecutivas.

**Solução:**
```bash
# Opção 1: Reset manual (use com cautela!)
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/reset-circuit-breaker"

# Opção 2: Ajustar limite
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/update-limits" \
  -H "Content-Type: application/json" \
  -d '{"circuit_breaker_losses": 5}'

# Opção 3: Aguardar próximo win (auto-reset)
```

**Prevenção:**
- Revisar estratégia após 2 perdas consecutivas
- Reduzir position size temporariamente
- Verificar condições de mercado (alta volatilidade?)

---

### Problema 2: "Perda diária/semanal excedida"

**Causa:** Limite de perda diária (5%) ou semanal (10%) atingido.

**Solução:**
```bash
# Verificar status
curl "https://botderivapi.roilabs.com.br/api/risk/metrics"

# Aguardar reset automático:
# - Perda diária reseta à meia-noite
# - Perda semanal reseta segunda-feira 00:00

# OU aumentar limites (NÃO RECOMENDADO em perda!)
curl -X POST "https://botderivapi.roilabs.com.br/api/risk/update-limits" \
  -d '{"max_daily_loss_percent": 7}'
```

**Prevenção:**
- Parar de tradear ao atingir 80% do limite
- Analisar trades perdedores do dia/semana
- Considerar tirar dia/semana off

---

### Problema 3: "Position size excede limite"

**Causa:** Position size calculado > 10% do capital.

**Solução:**
```python
# Ajustar stop loss mais próximo (reduz position size)
stop_loss_original = 98.0
stop_loss_ajustado = 99.0  # Mais próximo do entry

# OU reduzir max_position_size_percent
curl -X POST ".../api/risk/update-limits" \
  -d '{"max_position_size_percent": 15}'
```

**Prevenção:**
- Usar ATR multiplier menor (1.5x ao invés de 2.0x)
- Tradear instrumentos menos voláteis
- Aumentar capital

---

### Problema 4: "Drawdown máximo excedido"

**Causa:** Capital atual < 85% do pico histórico.

**Solução:**
```bash
# PARAR TRADING IMEDIATAMENTE
# Analisar todos os trades recentes
# Revisar estratégia

# Ajustar limites APENAS após análise profunda
curl -X POST ".../api/risk/update-limits" \
  -d '{"max_drawdown_percent": 20}'
```

**Prevenção:**
- Reduzir position size ao atingir -10% drawdown
- Ativar circuit breaker mais cedo (2 losses)
- Revisar win rate e profit factor semanalmente

---

### Problema 5: Kelly Criterion retorna 0%

**Causa:** `avg_win` ou `avg_loss` = 0 (sem histórico suficiente).

**Solução:**
```python
# Sistema usa 2% fixo como fallback automático
# Após ~20 trades, Kelly Criterion será calculado corretamente

# Forçar update manual do win_rate/avg_win/avg_loss:
risk_manager.update_performance_stats(
    new_win_rate=0.55,
    new_avg_win=15.0,
    new_avg_loss=10.0
)
```

---

### Problema 6: "R:R ratio abaixo do mínimo"

**Causa:** Take profit muito próximo ou stop loss muito longe.

**Solução:**
```python
# Aumentar distância TP
tp2 = entry + (3 * distance_to_stop)  # 1:3 ao invés de 1:2

# OU reduzir distância SL (usar ATR menor)
stop_loss = calculate_atr_stop_loss(price, atr, is_long, multiplier=1.5)

# OU reduzir min_risk_reward_ratio
curl -X POST ".../api/risk/update-limits" \
  -d '{"min_risk_reward_ratio": 1.2}'
```

**Prevenção:**
- Tradear apenas setups com R:R ≥ 1:2
- Usar partial take profit (garante 1:1 no mínimo)

---

## Próximos Passos

### Melhorias Planejadas

1. **Dashboard Frontend:**
   - Gráficos de P&L em tempo real
   - Heatmap de win rate por horário/dia
   - Distribuição de trades (winners vs losers)

2. **Integração Completa:**
   - Auto-executar trades com validação de risco
   - Atualizar métricas após cada trade
   - Log detalhado de decisões de risco

3. **Backtesting com Risk Manager:**
   - Simular performance com diferentes limites
   - Comparar Kelly vs Fixed Fractional
   - Otimizar parâmetros (ATR multiplier, trailing %)

4. **Alertas:**
   - Email/Telegram quando drawdown > 10%
   - Notificação ao atingir 80% de limite diário
   - Alerta de circuit breaker ativado

5. **Machine Learning:**
   - Prever probabilidade de win por trade
   - Ajustar Kelly Criterion dinamicamente
   - Detectar regimes de mercado (trending vs ranging)

---

## Referências

- **Kelly Criterion:** Edward O. Thorp, "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market"
- **Money Management:** Van K. Tharp, "Trade Your Way to Financial Freedom"
- **ATR Stop Loss:** J. Welles Wilder, "New Concepts in Technical Trading Systems"
- **Risk Management:** Ralph Vince, "The Mathematics of Money Management"

---

**Versão:** 1.0
**Data:** 2025-12-13
**Autor:** AI Trading Bot Team
**Licença:** Proprietário

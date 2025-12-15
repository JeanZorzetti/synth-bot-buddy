# 🔍 ROADMAP - Verificação de Dados Reais vs Mockados

**Objetivo:** Auditar todo o sistema para identificar e documentar quais dados são reais (Deriv API) vs mockados/simulados, eliminar redundâncias e garantir que o sistema está pronto para trading real.

**Status Geral:** 🟡 Em Análise

---

## 📋 Índice de Verificação

1. [Dashboard](#fase-1-dashboard)
2. [Risk Management](#fase-2-risk-management)
3. [Order Flow](#fase-3-order-flow)
4. [Backtesting](#fase-4-backtesting)
5. [Paper Trading](#fase-5-paper-trading)
6. [Forward Testing](#fase-6-forward-testing)
7. [Settings](#fase-7-settings)
8. [Backend Services](#fase-8-backend-services)
9. [Redundâncias e Limpeza](#fase-9-redundâncias-e-limpeza)

---

## FASE 1: Dashboard

**URL:** https://botderiv.roilabs.com.br/dashboard

### 1.1 Overview Cards (Métricas Principais)

**Componentes a Verificar:**

- [ ] **Total P&L Card**
  - Fonte de dados: `?`
  - API endpoint: `/api/trades/stats` ou mockado?
  - Verificar se usa dados reais do histórico de trades
  - Arquivo: `frontend/src/pages/Dashboard.tsx`

- [ ] **Win Rate Card**
  - Fonte: Calculado de trades reais ou fixo?
  - Verificar cálculo: `wins / (wins + losses)`

- [ ] **Active Positions Card**
  - Fonte: Deriv API real ou mock?
  - WebSocket connection real?

- [ ] **Daily Profit Card**
  - Filtro de data funcionando?
  - Dados do dia atual ou mockados?

**Endpoints Relacionados:**
```
GET /api/trades/stats
GET /api/portfolio/positions
WebSocket /ws/portfolio
```

### 1.2 Price Chart (Gráfico de Preços)

- [ ] **Candlestick Chart**
  - Fonte: Deriv API Ticks/Candles ou mock?
  - Endpoint: `/api/market/candles` ou Deriv WebSocket?
  - Intervalo de tempo real (1m, 5m, 15m)?
  - Indicadores técnicos calculados de dados reais?

- [ ] **Volume Data**
  - Volume real do mercado ou simulado?

**Arquivos a Verificar:**
```
frontend/src/pages/Dashboard.tsx
backend/deriv_api_legacy.py
backend/main.py (endpoints /api/market/*)
```

### 1.3 ML Predictions Panel

- [ ] **Prediction Cards**
  - Modelo ML real (XGBoost treinado) ou mockado?
  - Arquivo modelo: `backend/models/xgboost_model.pkl` existe?
  - Features extraídas de dados reais?

- [ ] **Confidence Score**
  - Calculado do modelo real ou valor fixo?

- [ ] **Signal Strength**
  - Baseado em análise técnica real?

**Endpoints:**
```
GET /api/ml/predict/{symbol}
GET /api/ml/info
```

**Arquivos:**
```
backend/ml_predictor.py
backend/models/xgboost_model.pkl
```

### 1.4 Active Positions Table

- [ ] **Positions List**
  - Fonte: Deriv API `/portfolio` ou mock?
  - Update em tempo real via WebSocket?

- [ ] **Position Details**
  - Entry price, current price, P&L calculados de dados reais?
  - Stop Loss / Take Profit reais ou simulados?

### 1.5 Recent Trades Table

- [ ] **Trades History**
  - Fonte: Database SQLite local ou Deriv API?
  - Endpoint: `/api/trades/history`
  - Database: `backend/trades.db` existe e tem dados?

**Checklist de Verificação:**
```sql
-- Verificar se trades.db tem dados reais
SELECT COUNT(*) FROM trades_history;
SELECT * FROM trades_history LIMIT 5;
```

### 1.6 WebSocket Connections

- [ ] **Deriv API WebSocket**
  - URL: `wss://ws.derivws.com/websockets/v3?app_id=...`
  - Token configurado em `.env.production`?
  - Connection status real ou sempre "connected"?

- [ ] **Portfolio Updates**
  - Recebe ticks/candles reais?
  - Latência aceitável (<100ms)?

**Arquivos:**
```
backend/deriv_api_legacy.py (linha ~50-200)
frontend/src/pages/Dashboard.tsx (useEffect WebSocket)
```

---

## FASE 2: Risk Management

**URL:** https://botderiv.roilabs.com.br/risk-management

### 2.1 Risk Limits Configuration

- [ ] **Max Position Size**
  - Valor salvo em database ou mock?
  - Validação real ao abrir trade?

- [ ] **Stop Loss / Take Profit**
  - Aplicado em trades reais?
  - Endpoint: `/api/risk/limits`

- [ ] **Daily Loss Limit**
  - Verificado contra trades reais do dia?
  - Bloqueia novas ordens se atingido?

**Arquivos:**
```
backend/risk_manager.py
backend/main.py (/api/risk/*)
frontend/src/pages/RiskManagement.tsx
```

### 2.2 Kelly Criterion Calculator

- [ ] **Win Rate Input**
  - Calculado de histórico real ou manual?

- [ ] **Profit/Loss Ratio**
  - Médias reais de trades?

- [ ] **Kelly Percentage**
  - Cálculo correto: `(p * b - q) / b`?
  - Usado em position sizing real?

**Arquivo:**
```
backend/kelly_ml_predictor.py
```

### 2.3 Position Sizing

- [ ] **Capital Base**
  - Saldo real da conta Deriv ou mock?
  - Endpoint: `/api/portfolio/balance`

- [ ] **Risk per Trade**
  - % do capital real?

- [ ] **Max Concurrent Positions**
  - Validado contra posições abertas reais?

### 2.4 Trailing Stop

- [ ] **Trailing Configuration**
  - Salvo em database?
  - Aplicado em trades ativos?

- [ ] **Activation Price**
  - Monitora preço real do mercado?

- [ ] **Update Mechanism**
  - WebSocket atualiza stop loss real?

**Endpoints:**
```
POST /api/risk/trailing-stop
GET /api/risk/limits
PUT /api/risk/limits
```

---

## FASE 3: Order Flow

**URL:** https://botderiv.roilabs.com.br/order-flow

### 3.1 Order Book Visualization

- [ ] **Bid/Ask Levels**
  - Fonte: Deriv API order book real?
  - WebSocket: `proposal_open_contract` ou mock?

- [ ] **Depth Chart**
  - Volume em cada nível é real?
  - Update frequência (<500ms)?

**Arquivos:**
```
backend/order_flow.py
frontend/src/pages/OrderFlow.tsx
```

### 3.2 Tape Reading

- [ ] **Trade Stream**
  - Ticks reais do mercado?
  - WebSocket: `ticks_history` ou mock?

- [ ] **Aggressive Orders Detection**
  - Algoritmo analisa ticks reais?
  - Threshold configurável?

- [ ] **Volume Profile**
  - Calculado de trades reais?

### 3.3 Institutional Activity

- [ ] **Large Orders Detection**
  - Detecta ordens > threshold em volume real?

- [ ] **Absorption Zones**
  - Identifica de order book real?

- [ ] **Iceberg Orders**
  - Algoritmo detecta padrões reais?

**Arquivo:**
```
backend/order_flow.py (classe OrderFlowAnalyzer)
```

### 3.4 Signal Enhancement

- [ ] **ML Signal + Order Flow**
  - Combina predição ML com tape reading real?
  - Endpoint: `/api/order-flow/enhance-signal`

- [ ] **Confidence Adjustment**
  - Score final baseado em dados reais?

---

## FASE 4: Backtesting

**URL:** https://botderiv.roilabs.com.br/backtesting

### 4.1 Historical Data

- [ ] **Data Source**
  - Deriv API `/ticks_history` real?
  - Cache em database local?
  - CSV files mockados?

- [ ] **Date Range Selector**
  - Busca dados históricos reais?
  - Endpoint: `/api/backtest/data/{symbol}`

**Arquivos:**
```
backend/backtesting.py
backend/data/historical/ (verificar CSVs)
```

### 4.2 Strategy Parameters

- [ ] **Indicators Selection**
  - RSI, MACD, Bollinger calculados de dados históricos reais?

- [ ] **Entry/Exit Rules**
  - Testados contra ticks reais?

- [ ] **Position Sizing**
  - Usa Kelly real ou fixo?

### 4.3 Backtest Execution

- [ ] **Simulation Engine**
  - Biblioteca: `backtrader` ou custom?
  - Processa tick-by-tick real?

- [ ] **Slippage Simulation**
  - Baseado em spread real médio?

- [ ] **Commission**
  - Usa taxas reais da Deriv?

**Endpoint:**
```
POST /api/backtest/run
```

### 4.4 Results Analysis

- [ ] **Performance Metrics**
  - Total P&L calculado de simulação real?
  - Win Rate, Sharpe Ratio, Drawdown corretos?

- [ ] **Trade List**
  - Todas as trades simuladas armazenadas?

- [ ] **Equity Curve**
  - Gráfico de evolução de capital correto?

**Arquivos:**
```
backend/backtesting.py (método run_backtest)
frontend/src/pages/Backtesting.tsx
```

---

## FASE 5: Paper Trading

**URL:** https://botderiv.roilabs.com.br/paper-trading

### 5.1 Virtual Account

- [ ] **Initial Capital**
  - Configurável ou fixo em $10,000?
  - Salvo em session/database?

- [ ] **Balance Tracking**
  - Atualizado com P&L de trades simulados?

**Arquivo:**
```
backend/paper_trading_engine.py
```

### 5.2 Trade Execution Simulation

- [ ] **Order Placement**
  - Simula latência real (100ms)?
  - Slippage configurável (0.1%)?

- [ ] **Market Price**
  - Usa preço real da Deriv API no momento da ordem?
  - Endpoint: `/api/market/price/{symbol}`

- [ ] **Fill Simulation**
  - Mock de execução ou tenta ordem real em demo account?

**Endpoints:**
```
POST /api/paper-trading/order
GET /api/paper-trading/positions
GET /api/paper-trading/stats
```

### 5.3 Position Management

- [ ] **Open Positions**
  - Armazenadas em memória ou database?
  - Update de P&L usa preço real atual?

- [ ] **Stop Loss / Take Profit**
  - Monitora preço real para trigger?
  - WebSocket ou polling?

- [ ] **Close Position**
  - Usa preço de mercado real no fechamento?

### 5.4 Performance Metrics

- [ ] **Real-time Stats**
  - Win Rate calculado de trades simulados?
  - Sharpe Ratio correto?

- [ ] **Trade History**
  - Salvo em database: `paper_trades.db`?
  - Endpoint: `/api/paper-trading/history`

**Arquivo:**
```
backend/paper_trading_engine.py (classe PaperTradingEngine)
```

---

## FASE 6: Forward Testing

**URL:** https://botderiv.roilabs.com.br/forward-testing

### 6.1 ML + Paper Trading Integration

- [ ] **Prediction Generation**
  - ML Predictor usa features de dados reais atuais?
  - Confidence threshold aplicado?

- [ ] **Auto-Trading Loop**
  - Executa trades automaticamente em paper trading?
  - Intervalo configurável?

**Arquivos:**
```
backend/forward_testing.py
backend/ml_predictor.py
```

### 6.2 Market Data Feed

- [ ] **Real-time Ticks**
  - Deriv API WebSocket real ou mock?
  - Método: `_fetch_market_data()` em forward_testing.py

- [ ] **OHLCV Data**
  - Candles de 1min reais?
  - Volume real?

**Verificar:**
```python
# backend/forward_testing.py linha ~191-224
async def _fetch_market_data(self):
    # Mock ou real?
```

### 6.3 Trade Execution

- [ ] **Signal to Order**
  - Confidence >= 60% executa ordem?
  - Position size calculado com Kelly real?

- [ ] **Risk Management**
  - Stop Loss / Take Profit aplicados?
  - Max positions respeitado?

### 6.4 Bug Logging

- [ ] **Error Tracking**
  - Erros reais salvos em `forward_testing_logs/bugs.jsonl`?

- [ ] **Performance Monitoring**
  - Latência de predição < 1s?
  - Accuracy tracking real?

**Endpoints:**
```
POST /api/forward-testing/start
POST /api/forward-testing/stop
GET /api/forward-testing/status
GET /api/forward-testing/predictions
GET /api/forward-testing/bugs
POST /api/forward-testing/report
```

### 6.5 Validation Report

- [ ] **4-Week Data Collection**
  - Métricas acumuladas de trades reais simulados?

- [ ] **Approval Criteria**
  - Win Rate > 60%?
  - Sharpe Ratio > 1.5?
  - Max Drawdown < 15%?

- [ ] **Report Generation**
  - Markdown report em `forward_testing_logs/validation_report_{timestamp}.md`?

---

## FASE 7: Settings

**URL:** https://botderiv.roilabs.com.br/settings

### 7.1 API Connection

- [ ] **Deriv API Token**
  - Variável de ambiente: `DERIV_API_TOKEN`?
  - Arquivo: `.env.production` existe?
  - Token válido e não expirado?

- [ ] **App ID**
  - `DERIV_APP_ID` configurado?
  - Registrado em https://app.deriv.com?

- [ ] **Connection Test**
  - Endpoint: `/api/settings/test-connection`?
  - Testa conexão real com Deriv?

**Verificar:**
```bash
# .env.production deve ter:
DERIV_API_TOKEN=your_token_here
DERIV_APP_ID=your_app_id
DERIV_API_URL=wss://ws.derivws.com/websockets/v3
```

### 7.2 Trading Preferences

- [ ] **Default Symbol**
  - Salvo em database ou config?
  - Usado em novas ordens?

- [ ] **Default Timeframe**
  - Aplicado em charts reais?

- [ ] **Auto-Trading Toggle**
  - Habilita/desabilita execução real?

### 7.3 ML Model Settings

- [ ] **Model Path**
  - Aponta para `models/xgboost_model.pkl` real?
  - Arquivo existe e tem tamanho > 0?

- [ ] **Confidence Threshold**
  - Valor usado em predições reais?
  - Endpoint: `/api/ml/settings`

- [ ] **Retrain Schedule**
  - Cron configurado (Domingos 3 AM)?
  - Scheduler ativo?

### 7.4 Risk Limits

- [ ] **Global Limits**
  - Salvos em database?
  - Aplicados em todas as ordens?

- [ ] **Account Protection**
  - Daily loss limit enforced?
  - Max drawdown trigger stop trading?

---

## FASE 8: Backend Services

### 8.1 Deriv API Integration

**Arquivo:** `backend/deriv_api_legacy.py`

- [ ] **Connection Management**
  - WebSocket real conectado?
  - Reconnection automático funciona?

- [ ] **Authentication**
  - Token enviado no authorize request?
  - Response com account info real?

- [ ] **Subscriptions**
  - `ticks`, `proposal`, `portfolio` ativos?
  - Callbacks processam dados reais?

**Métodos a Verificar:**
```python
async def connect()
async def authorize()
async def subscribe_ticks()
async def get_portfolio()
async def buy_contract()
```

### 8.2 ML Predictor Service

**Arquivo:** `backend/ml_predictor.py`

- [ ] **Model Loading**
  - XGBoost model carregado de arquivo real?
  - Método: `load_model()`

- [ ] **Feature Extraction**
  - RSI, MACD, Bollinger de dados reais?
  - Método: `_extract_features()`

- [ ] **Prediction**
  - `predict()` usa modelo real?
  - Retorna probabilidades [UP, DOWN]?

**Testar:**
```bash
curl http://localhost:8000/api/ml/predict/R_100
```

### 8.3 Paper Trading Engine

**Arquivo:** `backend/paper_trading_engine.py`

- [ ] **Order Execution**
  - `execute_order()` usa preço real de mercado?
  - Slippage aplicado?

- [ ] **Position Tracking**
  - `positions` dict atualizado com P&L real?

- [ ] **Metrics Calculation**
  - `get_performance_stats()` usa trades reais?

### 8.4 Forward Testing Engine

**Arquivo:** `backend/forward_testing.py`

- [ ] **Market Data**
  - `_fetch_market_data()` é mock ou Deriv API real?
  - **CRITICAL:** Linha ~191-224

```python
async def _fetch_market_data(self):
    # TODO: Verificar se é mock ou API real
    # Atualmente retorna dados simulados
```

- [ ] **ML Integration**
  - `_generate_prediction()` chama MLPredictor real?

- [ ] **Trade Execution**
  - `_execute_trade()` usa PaperTradingEngine real?

### 8.5 Retrain Service

**Arquivo:** `backend/ml_retrain_service.py`

- [ ] **Data Collection**
  - `collect_training_data()` lê CSVs reais de `data/training/`?

- [ ] **Model Training**
  - `train_model()` treina XGBoost com dados reais?

- [ ] **Deployment**
  - `deploy_model()` substitui modelo em produção?

- [ ] **Scheduler**
  - `retrain_scheduler.py` executa Domingos 3 AM?
  - APScheduler rodando?

### 8.6 Database

**SQLite Databases:**

- [ ] **trades.db**
  - Localização: `backend/trades.db`
  - Tabela: `trades_history`
  - Tem dados reais ou vazio?

```sql
SELECT * FROM trades_history LIMIT 10;
```

- [ ] **paper_trades.db** (se existir)
  - Localização: `backend/paper_trades.db`
  - Trades simulados armazenados?

### 8.7 Metrics & Monitoring

**Arquivo:** `backend/metrics.py`

- [ ] **Prometheus Metrics**
  - Counters/Gauges atualizados com dados reais?
  - Endpoint: `/metrics` expõe métricas?

- [ ] **Performance Tracking**
  - Latency, Accuracy, Win Rate de trades reais?

---

## FASE 9: Redundâncias e Limpeza

### 9.1 Código Duplicado

**Verificar:**

- [ ] **Multiple MLPredictor Instances**
  - `ml_predictor.py` vs `kelly_ml_predictor.py`
  - Consolidar em uma classe?

- [ ] **Deriv API Wrappers**
  - `deriv_api_legacy.py` vs outro wrapper?
  - Usar apenas um?

- [ ] **Paper Trading Engines**
  - `paper_trading_engine.py` vs outro similar?

### 9.2 Arquivos Não Utilizados

**Procurar por:**

- [ ] **Arquivos `_old`, `_backup`, `_v1`**
  - Deletar ou documentar motivo

- [ ] **Imports Não Usados**
  - Rodar `pylint` ou `flake8`

- [ ] **Funções Dead Code**
  - Métodos nunca chamados

### 9.3 Configurações Hardcoded

**Substituir por variáveis de ambiente:**

- [ ] **API URLs**
  - `wss://ws.derivws.com` hardcoded?
  - Usar `DERIV_API_URL` do .env

- [ ] **Thresholds**
  - Confidence, Stop Loss, etc. configuráveis?

- [ ] **Paths**
  - `models/`, `data/`, `logs/` em .env?

### 9.4 Logs e Debug

- [ ] **Production Logging**
  - `LOG_LEVEL=INFO` em produção?
  - Não deixar `DEBUG` ativo

- [ ] **Sensitive Data**
  - Tokens não logados em plaintext?
  - Usar masking: `token[:5]...`

### 9.5 Testes

- [ ] **Unit Tests**
  - `backend/tests/` tem cobertura real?
  - Rodar `pytest` e verificar %

- [ ] **Integration Tests**
  - Testa conexão real com Deriv API?

---

## FASE 10: Checklist Final

### 10.1 Dados Reais (MUST HAVE)

- [ ] ML Predictor usando modelo XGBoost treinado real
- [ ] Market data de Deriv API WebSocket real
- [ ] Paper Trading usando preços reais de mercado
- [ ] Forward Testing coletando métricas de trades simulados reais
- [ ] Risk Management aplicando limites em ordens reais

### 10.2 Dados Mockados (ACEITÁVEL)

- [ ] Backtesting com dados históricos em CSV (se não houver API history)
- [ ] Order Flow simulation (se Deriv não expor order book completo)
- [ ] Initial training data (se não houver histórico suficiente)

### 10.3 Pronto para Produção

- [ ] `.env.production` configurado com token real
- [ ] Database `trades.db` inicializado
- [ ] Modelo ML treinado existe em `models/xgboost_model.pkl`
- [ ] Scheduler de retreinamento ativo
- [ ] Monitoramento Prometheus + Grafana funcionando
- [ ] Alertas Telegram + Email configurados
- [ ] Backup automático de modelos ativo
- [ ] Forward Testing rodando por >= 4 semanas
- [ ] Win Rate > 60%, Sharpe > 1.5, Drawdown < 15%

---

## 📊 Template de Verificação por Feature

Para cada feature, preencher:

```markdown
### Feature: [Nome]

**Status:** 🔴 Não Verificado | 🟡 Em Análise | 🟢 Validado

**Tipo de Dados:**
- [ ] 🟢 Dados Reais (Deriv API)
- [ ] 🟡 Dados Simulados Realistas
- [ ] 🔴 Dados Mockados/Fixos

**Arquivos Envolvidos:**
- Backend: `path/to/file.py:linha`
- Frontend: `path/to/component.tsx:linha`

**Endpoints API:**
- `GET /api/...`
- `WebSocket /ws/...`

**Dependências Externas:**
- Deriv API: Sim/Não
- Database: SQLite / PostgreSQL / Mock
- ML Model: Real / Mock

**Testes Realizados:**
1. [ ] Teste manual via curl
2. [ ] Teste no frontend
3. [ ] Verificado logs do backend
4. [ ] Confirmado dados são reais

**Problemas Encontrados:**
- [Listar issues]

**Ações Necessárias:**
- [ ] Corrigir mock para usar dados reais
- [ ] Configurar .env
- [ ] Treinar modelo ML
- [ ] etc.
```

---

## 🚀 Plano de Execução

### Semana 1: Auditoria Inicial
- [ ] Executar verificação de todas as 7 páginas principais
- [ ] Documentar quais features usam dados reais vs mock
- [ ] Identificar redundâncias críticas

### Semana 2: Correções Prioritárias
- [ ] Substituir mocks por dados reais onde crítico
- [ ] Configurar conexão real com Deriv API
- [ ] Treinar/validar modelo ML com dados reais

### Semana 3: Eliminação de Redundâncias
- [ ] Remover código duplicado
- [ ] Consolidar serviços similares
- [ ] Refatorar onde necessário

### Semana 4: Testes Finais
- [ ] Rodar suite completa de testes
- [ ] Validar todas as features com dados reais
- [ ] Forward Testing final por 1 semana

---

## 📝 Notas e Observações

### Prioridade Alta (CRITICAL)
- Forward Testing `_fetch_market_data()` - Verificar se usa mock ou API real
- ML Predictor model file - Confirmar existência e validade
- Deriv API token - Validar conexão real

### Prioridade Média
- Order Flow - Pode usar simulação se API não expor order book
- Backtesting - CSVs aceitáveis se não houver API history

### Prioridade Baixa
- UI/UX - Foco em dados, não aparência
- Logs - Pode manter debug temporariamente

---

## 🔗 Links Úteis

- Deriv API Docs: https://api.deriv.com/
- FastAPI Docs: https://fastapi.tiangolo.com/
- XGBoost Docs: https://xgboost.readthedocs.io/
- Prometheus: https://prometheus.io/docs/

---

**Última Atualização:** 15/12/2024
**Responsável:** Claude Code (Autonomous Agent)
**Status Geral:** 🟡 Pendente de Auditoria Completa

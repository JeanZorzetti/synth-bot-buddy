# 🤖 ROADMAP - IA Scalping para Ativos Sintéticos

> **Objetivo:** Criar um modelo de IA especializado em **scalping de ativos sintéticos** (CRASH/BOOM/Volatility Indices) com interface web completa para treinamento, monitoramento e deploy.

---

## 📋 ÍNDICE

- [Fase 1: Arquitetura e Planejamento](#fase-1-arquitetura-e-planejamento)
- [Fase 2: Coleta de Dados e Feature Engineering](#fase-2-coleta-de-dados-e-feature-engineering)
- [Fase 3: Desenvolvimento do Modelo IA](#fase-3-desenvolvimento-do-modelo-ia)
- [Fase 4: Interface Web de Treinamento](#fase-4-interface-web-de-treinamento)
- [Fase 5: Sistema de Backtesting](#fase-5-sistema-de-backtesting)
- [Fase 6: Deploy e Monitoramento](#fase-6-deploy-e-monitoramento)
- [Fase 7: Otimização e Produção](#fase-7-otimização-e-produção)

---

## 🎯 VISÃO GERAL

### **O Que é Scalping em Ativos Sintéticos?**

**Scalping** é uma estratégia de trading de alta frequência que visa capturar pequenos movimentos de preço em períodos muito curtos (segundos a minutos).

**Ativos Sintéticos** (Deriv):
- **CRASH 500/1000**: Sobe continuamente com crashes periódicos
- **BOOM 500/1000**: Desce continuamente com spikes periódicos
- **Volatility Indices (V10/V25/V50/V75/V100)**: Volatilidade controlada
- **Step Indices**: Preços em degraus fixos

### **Características da IA para Scalping**

| Característica | Descrição |
|----------------|-----------|
| **Timeframe** | 1s, 5s, 15s, 30s, 1min |
| **Hold Time** | 10s - 5min (máximo) |
| **Win Rate Target** | 65-75% |
| **Risk:Reward** | 1:1 ou 1:1.5 |
| **Trades/Dia** | 50-200 trades |
| **Tipo de Modelo** | LSTM + CNN + Attention (ensemble) |
| **Features** | Microestrutura de mercado, order flow, volatility clusters |

---

## 🏗️ FASE 1: Arquitetura e Planejamento

### ✅ **1.1. Definir Arquitetura do Sistema**

**Objetivo:** Planejar stack tecnológico e fluxo de dados

**Componentes:**

```
┌─────────────────────────────────────────────────────────────┐
│                  FRONTEND (Dashboard IA)                     │
│  React + TypeScript + TailwindCSS + Charts                   │
└───────────────────┬─────────────────────────────────────────┘
                    │ HTTP/WebSocket
┌───────────────────▼─────────────────────────────────────────┐
│              BACKEND (FastAPI + Python)                      │
│  - API de Treinamento                                        │
│  - API de Inferência                                         │
│  - WebSocket para streaming de predições                     │
└───────┬───────────┬───────────┬──────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌──────────┐
   │ Data   │  │ Model  │  │ Trading  │
   │Storage │  │Training│  │ Engine   │
   └────────┘  └────────┘  └──────────┘
        │
        ▼
   ┌────────────────────┐
   │  Deriv API         │
   │  (Tick Stream)     │
   └────────────────────┘
```

**Stack Tecnológico:**

| Camada | Tecnologia | Justificativa |
|--------|------------|---------------|
| **Model** | PyTorch | Flexibilidade para LSTM/CNN/Transformers |
| **Feature Store** | Redis + PostgreSQL | Cache rápido + persistência |
| **Tick Data** | InfluxDB / TimescaleDB | Time-series otimizado |
| **Training** | Ray/Dask | Treinamento distribuído |
| **API** | FastAPI | Async, WebSocket nativo |
| **Frontend** | React + Recharts | Dashboard interativo |

**Tarefas:**
- [ ] Criar diagrama de arquitetura detalhado
- [ ] Definir schema de banco de dados (ticks, features, models, experiments)
- [ ] Escolher infraestrutura (local vs cloud)
- [ ] Documentar decisões técnicas

---

### ✅ **1.2. Setup do Ambiente de Desenvolvimento**

**Objetivo:** Preparar ambiente para desenvolvimento e treinamento

**Estrutura de Pastas:**

```
backend/
├── ml/
│   ├── scalping/                    # Novo módulo de scalping
│   │   ├── __init__.py
│   │   ├── data_collector.py        # Coleta ticks da Deriv
│   │   ├── feature_engineering.py   # Cria features de scalping
│   │   ├── models/
│   │   │   ├── lstm_scalper.py
│   │   │   ├── cnn_microstructure.py
│   │   │   ├── transformer_attention.py
│   │   │   └── ensemble.py
│   │   ├── trainer.py               # Pipeline de treinamento
│   │   ├── backtester.py            # Backtest engine
│   │   └── config.py                # Configurações
│   └── research/                     # Notebooks de pesquisa
│       └── scalping_experiments.ipynb
├── api/
│   └── scalping_routes.py           # Endpoints da IA Scalping
└── database/
    └── migrations/                   # Migrações de schema

frontend/
└── src/
    └── pages/
        └── IAScalping.tsx           # Nova aba no dashboard
```

**Dependências Python:**

```txt
# ML/DL
torch>=2.0.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Time Series
statsmodels>=0.14.0
arch>=6.0.0  # GARCH models

# Feature Engineering
ta>=0.11.0  # Technical Analysis
pandas-ta>=0.3.14

# Data Storage
influxdb-client>=1.38.0
psycopg2-binary>=2.9.0
redis>=5.0.0

# Distributed Computing
ray[default]>=2.7.0
dask[complete]>=2023.9.0

# API
fastapi[all]>=0.103.0
websockets>=11.0

# Visualization
plotly>=5.17.0
seaborn>=0.12.0
```

**Tarefas:**
- [ ] Criar estrutura de pastas
- [ ] Instalar dependências
- [ ] Configurar InfluxDB/TimescaleDB para tick data
- [ ] Setup Redis para cache de features
- [ ] Criar notebooks de pesquisa

---

## 📊 FASE 2: Coleta de Dados e Feature Engineering

### ✅ **2.1. Sistema de Coleta de Ticks**

**Objetivo:** Coletar tick-by-tick data de alta frequência dos ativos sintéticos

**Módulo:** `backend/ml/scalping/data_collector.py`

```python
class ScalpingDataCollector:
    """
    Coleta ticks em tempo real da Deriv API
    e armazena em banco time-series
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols  # ['CRASH500', 'BOOM1000', etc]
        self.deriv_api = DerivAPI()
        self.influxdb = InfluxDBClient()

    async def stream_ticks(self, symbol: str):
        """Stream contínuo de ticks"""
        async for tick in self.deriv_api.subscribe_ticks(symbol):
            await self._process_tick(tick)

    async def _process_tick(self, tick: Dict):
        """Processa e armazena tick"""
        # 1. Salvar tick raw
        await self.influxdb.write_tick(tick)

        # 2. Calcular microfeatures
        features = self.compute_microfeatures(tick)

        # 3. Cachear em Redis para acesso rápido
        await self.redis.set(f"latest_tick:{symbol}", features)

    def compute_microfeatures(self, tick: Dict) -> Dict:
        """
        Features de microestrutura:
        - Tick direction (uptick/downtick)
        - Volume imbalance
        - Spread
        - Tick velocity
        """
        pass
```

**Features de Tick-Level:**

| Feature | Descrição | Cálculo |
|---------|-----------|---------|
| **tick_direction** | Direção do movimento | +1 (up), -1 (down), 0 (flat) |
| **tick_size** | Tamanho do movimento | abs(price - prev_price) |
| **tick_velocity** | Velocidade de chegada | ticks_per_second |
| **bid_ask_spread** | Spread normalizado | (ask - bid) / mid_price |
| **tick_imbalance** | Desbalanceamento | (upticks - downticks) / total_ticks |

**Tarefas:**
- [ ] Implementar `ScalpingDataCollector`
- [ ] Conectar à Deriv API para streaming
- [ ] Setup InfluxDB schema para ticks
- [ ] Criar pipeline de processamento em tempo real
- [ ] Coletar **1 milhão de ticks** de cada símbolo (mínimo)

---

### ✅ **2.2. Feature Engineering para Scalping**

**Objetivo:** Criar features preditivas para movimentos de curto prazo

**Módulo:** `backend/ml/scalping/feature_engineering.py`

**Categorias de Features:**

#### **A. Microestrutura de Mercado**

```python
def compute_microstructure_features(ticks: pd.DataFrame, window: int = 100):
    """
    Features de microestrutura (últimos 100 ticks)
    """
    features = {}

    # 1. Order Flow
    features['tick_imbalance'] = (
        (ticks['direction'] == 1).sum() -
        (ticks['direction'] == -1).sum()
    ) / len(ticks)

    # 2. Volume Profile
    features['volume_ratio'] = ticks['volume'].tail(20).sum() / ticks['volume'].mean()

    # 3. Trade Intensity
    features['ticks_per_second'] = len(ticks) / (ticks['timestamp'].max() - ticks['timestamp'].min()).total_seconds()

    # 4. Price Impact
    features['price_impact'] = ticks['price'].pct_change().abs().mean()

    return features
```

#### **B. Volatilidade Realizada**

```python
def compute_volatility_features(ticks: pd.DataFrame):
    """
    Volatilidade em múltiplas escalas
    """
    returns = ticks['price'].pct_change()

    features = {
        # Realized Volatility (Parkinson estimator)
        'rv_parkinson': np.sqrt(
            ((np.log(ticks['high'] / ticks['low'])) ** 2) / (4 * np.log(2))
        ).mean(),

        # GARCH(1,1) forecast
        'garch_forecast': fit_garch(returns).forecast(horizon=1),

        # Volatility clusters (regime detection)
        'vol_regime': classify_volatility_regime(returns),
    }

    return features
```

#### **C. Pattern Recognition**

```python
def detect_scalping_patterns(ticks: pd.DataFrame):
    """
    Padrões de curto prazo
    """
    features = {}

    # 1. Micro Support/Resistance
    features['near_support'] = is_near_level(ticks['price'], 'support', tolerance=0.001)
    features['near_resistance'] = is_near_level(ticks['price'], 'resistance', tolerance=0.001)

    # 2. Momentum bursts (aceleração súbita)
    features['momentum_burst'] = detect_momentum_burst(ticks['price'])

    # 3. Mean reversion signal
    features['mean_reversion'] = ticks['price'].iloc[-1] / ticks['price'].rolling(50).mean().iloc[-1] - 1

    return features
```

#### **D. Features Específicas de CRASH/BOOM**

```python
def compute_crash_boom_features(ticks: pd.DataFrame, symbol: str):
    """
    Features específicas para CRASH/BOOM
    """
    features = {}

    if 'CRASH' in symbol:
        # Prever proximidade do crash
        features['ticks_since_last_crash'] = count_ticks_since_event(ticks, event='crash')
        features['crash_probability'] = estimate_crash_probability(ticks)
        features['uptrend_strength'] = measure_trend_strength(ticks, direction='up')

    elif 'BOOM' in symbol:
        # Prever proximidade do boom
        features['ticks_since_last_boom'] = count_ticks_since_event(ticks, event='boom')
        features['boom_probability'] = estimate_boom_probability(ticks)
        features['downtrend_strength'] = measure_trend_strength(ticks, direction='down')

    return features
```

**Dataset Final:**

```python
# Estrutura do dataset de treinamento
X = pd.DataFrame({
    # Microestrutura (10 features)
    'tick_imbalance', 'volume_ratio', 'ticks_per_second', 'price_impact', ...,

    # Volatilidade (8 features)
    'rv_parkinson', 'garch_forecast', 'vol_regime', ...,

    # Patterns (5 features)
    'near_support', 'near_resistance', 'momentum_burst', ...,

    # CRASH/BOOM específicas (6 features)
    'ticks_since_last_crash', 'crash_probability', ...,

    # Time features (4 features)
    'hour_of_day', 'day_of_week', 'market_session', 'is_high_volume_hour',
})

# Target: movimento nos próximos N ticks
y = (ticks['price'].shift(-N) / ticks['price'] - 1 > threshold).astype(int)
```

**Tarefas:**
- [ ] Implementar todas as categorias de features
- [ ] Criar pipeline de feature engineering
- [ ] Validar features (correlação, importância)
- [ ] Otimizar performance (caching, vectorização)
- [ ] Documentar cada feature

---

## 🤖 FASE 3: Desenvolvimento do Modelo IA

### ✅ **3.1. Modelo Base - LSTM para Sequências de Ticks**

**Objetivo:** Modelo recorrente para capturar dependências temporais

**Arquitetura:**

```python
class LSTMScalper(nn.Module):
    """
    LSTM para predição de movimento nos próximos N ticks
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        # LSTM Stack
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=False  # Só forward para real-time
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.2
        )

        # Classificador
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 classes: DOWN, NEUTRAL, UP
        )

    def forward(self, x):
        # x: [batch, sequence_length, features]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Pegar último timestep
        final_hidden = attn_out[:, -1, :]

        # Classificação
        output = self.fc(final_hidden)

        return output, attn_weights
```

**Configuração de Treinamento:**

```python
config = {
    'sequence_length': 100,  # Últimos 100 ticks
    'prediction_horizon': 10,  # Prever movimento em 10 ticks
    'batch_size': 256,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 5,
    'loss_function': 'focal_loss',  # Para desbalanceamento de classes
    'optimizer': 'AdamW',
    'scheduler': 'OneCycleLR',
}
```

**Tarefas:**
- [ ] Implementar arquitetura LSTM
- [ ] Criar data loaders otimizados
- [ ] Implementar Focal Loss para desbalanceamento
- [ ] Treinar modelo baseline
- [ ] Avaliar performance (accuracy, precision, recall, F1)

---

### ✅ **3.2. Modelo Avançado - CNN para Microestrutura**

**Objetivo:** CNN 1D para capturar padrões locais no order flow

```python
class CNNMicrostructure(nn.Module):
    """
    CNN 1D para detectar padrões de microestrutura
    """
    def __init__(self, input_dim: int):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        # x: [batch, features, sequence_length]

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        # Global average pooling
        x = x.mean(dim=2)

        # Classificação
        output = self.fc(x)

        return output
```

**Tarefas:**
- [ ] Implementar CNN 1D
- [ ] Treinar modelo
- [ ] Comparar com LSTM

---

### ✅ **3.3. Ensemble - Combinar LSTM + CNN + XGBoost**

**Objetivo:** Meta-model que combina os melhores aspectos de cada arquitetura

```python
class ScalpingEnsemble:
    """
    Ensemble de LSTM + CNN + XGBoost
    """
    def __init__(self):
        self.lstm = LSTMScalper()
        self.cnn = CNNMicrostructure()
        self.xgboost = xgb.XGBClassifier()

        # Meta-learner
        self.meta_model = LogisticRegression()

    def fit(self, X_train, y_train):
        # Treinar modelos base
        self.lstm.fit(X_train)
        self.cnn.fit(X_train)
        self.xgboost.fit(X_train)

        # Gerar predições dos modelos base
        lstm_pred = self.lstm.predict_proba(X_train)
        cnn_pred = self.cnn.predict_proba(X_train)
        xgb_pred = self.xgboost.predict_proba(X_train)

        # Stacking
        meta_features = np.hstack([lstm_pred, cnn_pred, xgb_pred])

        # Treinar meta-model
        self.meta_model.fit(meta_features, y_train)

    def predict(self, X):
        # Predições dos modelos base
        lstm_pred = self.lstm.predict_proba(X)
        cnn_pred = self.cnn.predict_proba(X)
        xgb_pred = self.xgboost.predict_proba(X)

        # Combinar
        meta_features = np.hstack([lstm_pred, cnn_pred, xgb_pred])

        # Predição final
        return self.meta_model.predict(meta_features)
```

**Tarefas:**
- [ ] Implementar ensemble
- [ ] Otimizar pesos via grid search
- [ ] Validar performance no conjunto de teste

---

## 🖥️ FASE 4: Interface Web de Treinamento

### ✅ **4.1. Página IAScalping.tsx**

**Objetivo:** Dashboard interativo para controlar treinamento de modelos

**Estrutura:**

```tsx
// frontend/src/pages/IAScalping.tsx

export default function IAScalping() {
    return (
        <div className="p-6">
            {/* Header */}
            <h1>🤖 IA Scalping - Ativos Sintéticos</h1>

            {/* Seção 1: Coleta de Dados */}
            <section className="mb-8">
                <h2>📊 Coleta de Dados</h2>
                <DataCollectionPanel />
            </section>

            {/* Seção 2: Feature Engineering */}
            <section className="mb-8">
                <h2>🔧 Feature Engineering</h2>
                <FeatureEngineeringPanel />
            </section>

            {/* Seção 3: Treinamento */}
            <section className="mb-8">
                <h2>🧠 Treinamento do Modelo</h2>
                <TrainingPanel />
            </section>

            {/* Seção 4: Backtesting */}
            <section className="mb-8">
                <h2>📈 Backtesting</h2>
                <BacktestingPanel />
            </section>

            {/* Seção 5: Deploy */}
            <section className="mb-8">
                <h2>🚀 Deploy</h2>
                <DeploymentPanel />
            </section>
        </div>
    );
}
```

**Componentes:**

#### **A. DataCollectionPanel**

```tsx
function DataCollectionPanel() {
    const [symbols, setSymbols] = useState(['CRASH500', 'BOOM1000']);
    const [isCollecting, setIsCollecting] = useState(false);
    const [stats, setStats] = useState(null);

    const startCollection = async () => {
        await api.post('/api/scalping/data/start', { symbols });
        setIsCollecting(true);
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            {/* Seletor de símbolos */}
            <SymbolSelector value={symbols} onChange={setSymbols} />

            {/* Botão de início */}
            <button onClick={startCollection}>
                {isCollecting ? 'Coletando...' : 'Iniciar Coleta'}
            </button>

            {/* Estatísticas em tempo real */}
            <CollectionStats stats={stats} />

            {/* Gráfico de ticks/segundo */}
            <TicksPerSecondChart data={stats?.ticks_per_second} />
        </div>
    );
}
```

#### **B. TrainingPanel**

```tsx
function TrainingPanel() {
    const [config, setConfig] = useState({
        model_type: 'ensemble',
        sequence_length: 100,
        prediction_horizon: 10,
        epochs: 50,
        batch_size: 256,
    });
    const [trainingStatus, setTrainingStatus] = useState(null);

    const startTraining = async () => {
        const response = await api.post('/api/scalping/train/start', config);

        // WebSocket para acompanhar progresso
        const ws = new WebSocket(`ws://localhost:8000/ws/training/${response.data.job_id}`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setTrainingStatus(data);
        };
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            {/* Configurações */}
            <ConfigForm config={config} onChange={setConfig} />

            {/* Botão de treinamento */}
            <button onClick={startTraining}>Iniciar Treinamento</button>

            {/* Progresso em tempo real */}
            <TrainingProgress status={trainingStatus} />

            {/* Métricas */}
            <MetricsChart
                accuracy={trainingStatus?.accuracy}
                loss={trainingStatus?.loss}
            />
        </div>
    );
}
```

#### **C. BacktestingPanel**

```tsx
function BacktestingPanel() {
    const [model, setModel] = useState(null);
    const [results, setResults] = useState(null);

    const runBacktest = async () => {
        const response = await api.post('/api/scalping/backtest/run', {
            model_id: model.id,
            start_date: '2024-01-01',
            end_date: '2024-12-19',
            initial_capital: 10000,
        });
        setResults(response.data);
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            {/* Seletor de modelo */}
            <ModelSelector value={model} onChange={setModel} />

            {/* Configurações de backtest */}
            <BacktestConfig />

            {/* Resultados */}
            {results && (
                <>
                    <EquityCurveChart data={results.equity_curve} />
                    <MetricsTable metrics={results.metrics} />
                    <TradesTable trades={results.trades} />
                </>
            )}
        </div>
    );
}
```

**Tarefas:**
- [ ] Criar página `IAScalping.tsx`
- [ ] Implementar todos os componentes
- [ ] Integrar com backend via API
- [ ] Adicionar WebSocket para updates em tempo real
- [ ] Criar gráficos interativos (Recharts/Plotly)

---

### ✅ **4.2. Backend API Endpoints**

**Módulo:** `backend/api/scalping_routes.py`

```python
from fastapi import APIRouter, WebSocket, BackgroundTasks
from typing import List, Dict

router = APIRouter(prefix="/api/scalping", tags=["IA Scalping"])

# ==================== DATA COLLECTION ====================

@router.post("/data/start")
async def start_data_collection(
    symbols: List[str],
    background_tasks: BackgroundTasks
):
    """Inicia coleta de ticks em background"""
    collector = ScalpingDataCollector(symbols)
    background_tasks.add_task(collector.start_streaming)

    return {"status": "started", "symbols": symbols}

@router.get("/data/stats")
async def get_collection_stats():
    """Retorna estatísticas da coleta"""
    stats = await db.get_collection_stats()
    return stats

# ==================== FEATURE ENGINEERING ====================

@router.post("/features/compute")
async def compute_features(
    symbol: str,
    start_date: str,
    end_date: str
):
    """Computa features para período específico"""
    engine = FeatureEngineeringEngine()
    features = await engine.compute(symbol, start_date, end_date)

    return {"total_features": len(features), "columns": list(features.columns)}

# ==================== TRAINING ====================

@router.post("/train/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """Inicia treinamento de modelo"""
    job_id = generate_job_id()

    trainer = ModelTrainer(config)
    background_tasks.add_task(trainer.train, job_id)

    return {"job_id": job_id, "status": "pending"}

@router.websocket("/ws/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """WebSocket para acompanhar progresso do treinamento"""
    await websocket.accept()

    # Stream de updates
    async for update in training_progress_stream(job_id):
        await websocket.send_json(update)

@router.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Status do treinamento"""
    status = await db.get_training_job(job_id)
    return status

# ==================== BACKTESTING ====================

@router.post("/backtest/run")
async def run_backtest(
    model_id: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000
):
    """Executa backtest do modelo"""
    backtester = ScalpingBacktester()
    results = await backtester.run(
        model_id=model_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    return results

# ==================== DEPLOYMENT ====================

@router.post("/deploy")
async def deploy_model(model_id: str, environment: str = "staging"):
    """Deploy modelo para produção/staging"""
    deployer = ModelDeployer()
    deployment_info = await deployer.deploy(model_id, environment)

    return deployment_info

@router.get("/models")
async def list_models():
    """Lista todos os modelos treinados"""
    models = await db.get_all_models()
    return models
```

**Tarefas:**
- [ ] Implementar todos os endpoints
- [ ] Adicionar autenticação
- [ ] Criar WebSocket para streaming
- [ ] Documentação Swagger
- [ ] Testes unitários

---

## 📈 FASE 5: Sistema de Backtesting

### ✅ **5.1. Backtest Engine para Scalping**

**Objetivo:** Simular estratégia de scalping com precisão tick-by-tick

**Módulo:** `backend/ml/scalping/backtester.py`

```python
class ScalpingBacktester:
    """
    Backtest engine otimizado para scalping
    """
    def __init__(self, model, config: BacktestConfig):
        self.model = model
        self.config = config

        # Parâmetros de scalping
        self.take_profit_pips = config.take_profit_pips  # Ex: 5 pips
        self.stop_loss_pips = config.stop_loss_pips      # Ex: 5 pips
        self.max_hold_time = config.max_hold_time        # Ex: 300 segundos
        self.max_trades_per_hour = config.max_trades_per_hour

        # Estado
        self.positions = []
        self.trades = []
        self.equity_curve = []

    async def run(self, ticks: pd.DataFrame, initial_capital: float):
        """
        Executa backtest tick-by-tick
        """
        capital = initial_capital

        for i in range(self.config.sequence_length, len(ticks)):
            # 1. Pegar janela de ticks
            window = ticks.iloc[i - self.config.sequence_length:i]

            # 2. Gerar predição
            prediction = self.model.predict(window)

            # 3. Verificar se pode abrir nova posição
            if self._can_open_position(ticks.iloc[i]):
                if prediction == 'UP' and confidence > self.config.min_confidence:
                    self._open_position('LONG', ticks.iloc[i], capital)
                elif prediction == 'DOWN' and confidence > self.config.min_confidence:
                    self._open_position('SHORT', ticks.iloc[i], capital)

            # 4. Atualizar posições abertas
            self._update_positions(ticks.iloc[i])

            # 5. Registrar equity
            self.equity_curve.append({
                'timestamp': ticks.iloc[i]['timestamp'],
                'equity': capital + self._calculate_unrealized_pnl()
            })

        # Calcular métricas
        metrics = self._calculate_metrics()

        return {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'metrics': metrics
        }

    def _update_positions(self, current_tick: Dict):
        """Atualiza posições abertas (SL/TP/Timeout)"""
        for position in self.positions[:]:
            # Take Profit
            if self._hit_take_profit(position, current_tick):
                self._close_position(position, current_tick, reason='take_profit')

            # Stop Loss
            elif self._hit_stop_loss(position, current_tick):
                self._close_position(position, current_tick, reason='stop_loss')

            # Timeout
            elif self._exceeded_max_hold_time(position, current_tick):
                self._close_position(position, current_tick, reason='timeout')

    def _calculate_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'avg_profit': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': (
                sum([t['pnl'] for t in winning_trades]) /
                abs(sum([t['pnl'] for t in losing_trades]))
            ) if losing_trades else float('inf'),
            'sharpe_ratio': self._calculate_sharpe(),
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_hold_time': np.mean([t['hold_time'] for t in self.trades]),
        }
```

**Métricas Específicas de Scalping:**

| Métrica | Objetivo | Descrição |
|---------|----------|-----------|
| **Win Rate** | > 65% | % de trades vencedores |
| **Profit Factor** | > 1.5 | Total profit / Total loss |
| **Avg Hold Time** | < 300s | Tempo médio de posição (5min) |
| **Trades/Day** | 50-200 | Frequência de trades |
| **Max Consecutive Losses** | < 5 | Sequência máxima de perdas |
| **Slippage Impact** | < 2% | Impacto do slippage no P&L |

**Tarefas:**
- [ ] Implementar `ScalpingBacktester`
- [ ] Adicionar simulação de slippage/comissões
- [ ] Criar visualizações de resultados
- [ ] Validar contra dados out-of-sample

---

## 🚀 FASE 6: Deploy e Monitoramento

### ✅ **6.1. Sistema de Deploy Automatizado**

**Objetivo:** Deploy de modelos treinados para produção

```python
class ModelDeployer:
    """
    Deploy de modelos para ambiente de produção
    """
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.docker_client = docker.from_env()

    async def deploy(self, model_id: str, environment: str):
        """
        Deploy de modelo

        Steps:
        1. Validar modelo (performance mínima)
        2. Criar container Docker
        3. Deploy em Kubernetes/Docker Swarm
        4. Health checks
        5. Gradual rollout (canary deployment)
        """

        # 1. Validar
        model = await self.model_registry.get(model_id)
        if not self._meets_production_criteria(model):
            raise ValueError("Modelo não atende critérios de produção")

        # 2. Criar imagem Docker
        image = self._build_docker_image(model)

        # 3. Deploy
        if environment == 'production':
            deployment = await self._deploy_to_production(image)
        else:
            deployment = await self._deploy_to_staging(image)

        return deployment

    def _meets_production_criteria(self, model) -> bool:
        """
        Critérios mínimos para produção:
        - Win rate > 65%
        - Sharpe ratio > 1.5
        - Max drawdown < 15%
        - Testado em out-of-sample data
        """
        metrics = model.backtest_metrics

        return (
            metrics['win_rate'] > 0.65 and
            metrics['sharpe_ratio'] > 1.5 and
            metrics['max_drawdown'] < 0.15
        )
```

**Tarefas:**
- [ ] Implementar `ModelDeployer`
- [ ] Criar Dockerfile para modelo
- [ ] Setup Kubernetes manifests
- [ ] Implementar canary deployment
- [ ] Configurar monitoring (Prometheus/Grafana)

---

### ✅ **6.2. Monitoramento de Modelo em Produção**

**Objetivo:** Detectar model drift e performance degradation

```python
class ModelMonitor:
    """
    Monitora modelo em produção
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.drift_detector = DriftDetector()

    async def check_health(self):
        """
        Health checks:
        1. Win rate (últimas 24h vs baseline)
        2. Prediction distribution (drift)
        3. Feature drift
        4. Latência de inferência
        """
        metrics = await self._get_recent_metrics(hours=24)

        health = {
            'win_rate_degradation': self._check_win_rate(metrics),
            'prediction_drift': self.drift_detector.check_predictions(metrics),
            'feature_drift': self.drift_detector.check_features(metrics),
            'latency': metrics['avg_inference_latency_ms'],
        }

        # Alertar se algo estiver fora do normal
        if any([
            health['win_rate_degradation'] > 0.1,  # 10% drop
            health['prediction_drift'] > 0.05,
            health['feature_drift'] > 0.05,
            health['latency'] > 100  # > 100ms
        ]):
            await self._send_alert(health)

        return health
```

**Dashboard de Monitoramento:**

```tsx
function ModelMonitoringDashboard({ model_id }) {
    const [health, setHealth] = useState(null);

    useEffect(() => {
        // Poll health a cada 1 minuto
        const interval = setInterval(async () => {
            const response = await api.get(`/api/scalping/models/${model_id}/health`);
            setHealth(response.data);
        }, 60000);

        return () => clearInterval(interval);
    }, [model_id]);

    return (
        <div className="grid grid-cols-2 gap-4">
            {/* Win Rate */}
            <MetricCard
                title="Win Rate (24h)"
                value={health?.win_rate}
                baseline={0.65}
                status={health?.win_rate_status}
            />

            {/* Drift */}
            <MetricCard
                title="Prediction Drift"
                value={health?.prediction_drift}
                threshold={0.05}
                status={health?.drift_status}
            />

            {/* Latência */}
            <MetricCard
                title="Latência"
                value={`${health?.latency}ms`}
                threshold={100}
            />

            {/* Gráficos */}
            <WinRateChart data={health?.win_rate_history} />
            <DriftChart data={health?.drift_history} />
        </div>
    );
}
```

**Tarefas:**
- [ ] Implementar `ModelMonitor`
- [ ] Configurar alertas (email/Slack/Discord)
- [ ] Criar dashboard de monitoramento
- [ ] Setup logs estruturados (ELK stack)

---

## ⚡ FASE 7: Otimização e Produção

### ✅ **7.1. Otimização de Performance**

**Objetivos:**
- Reduzir latência de inferência para < 10ms
- Aumentar throughput para > 1000 predições/segundo

**Técnicas:**

#### **A. Model Quantization**

```python
import torch.quantization as quantization

# Quantizar modelo para INT8
model_quantized = quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# Reduz tamanho do modelo em ~4x
# Aumenta velocidade em ~2-3x
```

#### **B. ONNX Runtime**

```python
import onnxruntime as ort

# Converter para ONNX
torch.onnx.export(model, dummy_input, "scalper.onnx")

# Inferência otimizada
session = ort.InferenceSession("scalper.onnx")
prediction = session.run(None, {input_name: input_data})
```

#### **C. Batch Inference**

```python
class BatchInferenceEngine:
    """
    Processa múltiplas predições em batch
    """
    def __init__(self, model, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.queue = asyncio.Queue()

    async def predict(self, features):
        """
        Adiciona à fila e aguarda resultado
        """
        future = asyncio.Future()
        await self.queue.put((features, future))
        return await future

    async def _batch_worker(self):
        """
        Worker que processa batches
        """
        while True:
            batch = []

            # Coletar até max_batch_size
            for _ in range(self.max_batch_size):
                if not self.queue.empty():
                    batch.append(await self.queue.get())

            if batch:
                # Inferência em batch
                features_batch = [item[0] for item in batch]
                predictions = self.model.predict_batch(features_batch)

                # Retornar resultados
                for (_, future), pred in zip(batch, predictions):
                    future.set_result(pred)
```

**Tarefas:**
- [ ] Implementar quantização de modelo
- [ ] Converter para ONNX
- [ ] Setup batch inference
- [ ] Benchmark de performance

---

### ✅ **7.2. A/B Testing de Modelos**

**Objetivo:** Testar novos modelos em produção sem risco total

```python
class ABTestingManager:
    """
    A/B testing de modelos em produção
    """
    def __init__(self):
        self.models = {}
        self.traffic_split = {}  # Ex: {'model_v1': 0.8, 'model_v2': 0.2}

    def route_request(self, request):
        """
        Roteia request para modelo baseado em split
        """
        import random

        rand = random.random()
        cumulative = 0

        for model_id, percentage in self.traffic_split.items():
            cumulative += percentage
            if rand < cumulative:
                return self.models[model_id]

        return self.models[list(self.models.keys())[0]]

    async def compare_models(self, duration_hours: int = 24):
        """
        Compara performance de modelos em A/B test
        """
        metrics = await self._collect_metrics(duration_hours)

        results = {
            model_id: {
                'win_rate': metrics[model_id]['win_rate'],
                'sharpe_ratio': metrics[model_id]['sharpe_ratio'],
                'total_trades': metrics[model_id]['total_trades'],
            }
            for model_id in self.models.keys()
        }

        # Teste estatístico (t-test)
        winner = self._statistical_test(results)

        return winner
```

**Tarefas:**
- [ ] Implementar A/B testing framework
- [ ] Criar dashboard de comparação
- [ ] Automatizar rollout do vencedor

---

## 📝 CHECKLIST DE VALIDAÇÃO

Antes de colocar em produção, validar:

### **Dados**
- [ ] Coletou > 1 milhão de ticks por símbolo
- [ ] Dataset split: 70% train / 15% validation / 15% test
- [ ] Validação cruzada temporal (não misturar futuro com passado)
- [ ] Features normalizadas e sem leakage

### **Modelo**
- [ ] Win rate > 65% em out-of-sample
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 15%
- [ ] Testado em múltiplos períodos de mercado
- [ ] Latência de inferência < 10ms

### **Sistema**
- [ ] Backtest realista (slippage, comissões, latência)
- [ ] Monitoramento em tempo real funcionando
- [ ] Alertas configurados
- [ ] Logs estruturados
- [ ] Documentação completa

### **Produção**
- [ ] Deploy em staging testado
- [ ] Canary deployment configurado
- [ ] Rollback automático em caso de falha
- [ ] A/B testing funcionando

---

## 🎯 MÉTRICAS DE SUCESSO

### **Fase de Desenvolvimento**
- [ ] 5+ modelos treinados e comparados
- [ ] Ensemble com performance superior aos modelos base
- [ ] Features validadas (importância, correlação)

### **Fase de Teste**
- [ ] Backtest lucrativo em período de 6+ meses
- [ ] Win rate consistente > 65%
- [ ] Profit factor > 1.5

### **Fase de Produção**
- [ ] 30 dias de paper trading sem bugs críticos
- [ ] Performance em produção ≥ 90% da performance em backtest
- [ ] Latência média < 10ms
- [ ] Uptime > 99.9%

---

## 🚀 ROADMAP DE EXECUÇÃO

| Fase | Duração Estimada | Prioridade |
|------|------------------|------------|
| **Fase 1** - Arquitetura | 1 semana | 🔴 CRÍTICA |
| **Fase 2** - Dados + Features | 2 semanas | 🔴 CRÍTICA |
| **Fase 3** - Modelo IA | 3 semanas | 🔴 CRÍTICA |
| **Fase 4** - Interface Web | 2 semanas | 🟡 ALTA |
| **Fase 5** - Backtesting | 1 semana | 🟡 ALTA |
| **Fase 6** - Deploy | 1 semana | 🟢 MÉDIA |
| **Fase 7** - Otimização | 2 semanas | 🟢 MÉDIA |

**Total:** ~12 semanas (3 meses)

---

## 📚 REFERÊNCIAS TÉCNICAS

### **Papers Acadêmicos**
- "Deep Learning for High-Frequency Trading" (Tsantekidis et al., 2017)
- "LSTM for Limit Order Book Prediction" (Zhang et al., 2019)
- "Market Microstructure in Practice" (Lehalle & Laruelle, 2018)

### **Livros**
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Algorithmic Trading: Winning Strategies" - Ernie Chan

### **Datasets**
- Deriv API Historical Data
- Tick-by-tick data from InfluxDB

---

**🎯 OBJETIVO FINAL:** Sistema de IA autônomo capaz de operar scalping em ativos sintéticos com win rate > 65%, operando 24/7 com monitoramento completo e deploy automatizado.

**📊 ROI Esperado:** Se alcançar 65% win rate com 100 trades/dia @ $10/trade = **+$2,000/mês** (após custos)

---

*Roadmap criado em: 2025-12-19*
*Versão: 1.0*
*Próxima revisão: Após conclusão de cada fase*

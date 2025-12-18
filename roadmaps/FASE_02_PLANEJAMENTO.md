# FASE 0.2 - ANÁLISE DE FEATURES PARA SCALPING (PLANEJAMENTO)

**Data**: 18/12/2025
**Status**: PLANEJADO (aguardando aprovação para iniciar)
**Ativos**: V75 (1HZ75V) + V100 (1HZ100V)
**Timeframe**: M5 (5 minutos)

---

## 🎯 OBJETIVO

Validar se V75 e/ou V100 são viáveis para scalping usando:
1. Timeframe M5 (em vez de M1)
2. Features técnicas (RSI, Bollinger Bands, Stochastic, MACD)
3. Modelo XGBoost para filtrar setups de alta probabilidade

**Meta**: Atingir 55-65% win rate com profit factor > 1.5

---

## 📊 POR QUE TESTAR V75 E V100?

### Resultado Fase 0.1 (M1)
- V75 em M1: 2.7% success rate (NÃO VIÁVEL)
- Motivo: M1 é muito ruidoso, sem filtros técnicos

### Descoberta da Análise Comparativa
- Mercado usa M5-M15 + filtros técnicos → 55-79% win rate
- V100 tem swings 30% maiores que V75
- Ambos podem ser viáveis com metodologia correta

### Hipóteses a Validar

**V75**:
- Win rate esperado: 55-65% (estrutura mais limpa)
- Swings: Moderados (~0.5% por trade)
- Risk-adjusted return: ALTO (Sharpe ratio melhor)

**V100**:
- Win rate esperado: 50-60% (mais caótico)
- Swings: Grandes (~0.65% por trade, +30%)
- Risk-adjusted return: MÉDIO (Sharpe ratio menor, mas lucros brutos maiores)

---

## 🔧 IMPLEMENTAÇÃO

### Etapa 1: Coleta de Dados M5 (1 dia)

#### Script a Modificar
`backend/ml/research/scalping_volatility_analysis.py`

**Mudanças necessárias**:

```python
# Linha 74: Mudar granularidade de 60 (1min) para 300 (5min)
"granularity": 300,  # 5 minutos (M5)

# Linha 537: Adicionar V100 à lista de símbolos
symbols = [
    '1HZ75V',   # Volatility 75
    '1HZ100V',  # Volatility 100
]
```

**Dados a coletar**:
- 6 meses de histórico (mesmo período da Fase 0.1)
- Timeframe: M5 (5 minutos)
- ~52,000 candles por ativo (vs 259,000 no M1)

**Saída esperada**:
```
backend/ml/research/data/
├── 1HZ75V_5min_180days.csv
└── 1HZ100V_5min_180days.csv
```

### Etapa 2: Cálculo de Features Técnicas (1 dia)

#### Features a Implementar

**Arquivo**: `backend/ml/research/scalping_feature_engineering.py` (NOVO)

**Grupo 1: Indicadores Clássicos** (adaptados para M5):

```python
def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # RSI (Relative Strength Index)
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)

    # EMA (Exponential Moving Averages)
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)

    return df
```

**Grupo 2: Candlestick Patterns**:

```python
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # Engulfing patterns
    df['bullish_engulfing'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &  # Previous bearish
        (df['close'] > df['open']) &  # Current bullish
        (df['open'] < df['close'].shift(1)) &  # Opens below prev close
        (df['close'] > df['open'].shift(1))  # Closes above prev open
    ).astype(int)

    df['bearish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous bullish
        (df['close'] < df['open']) &  # Current bearish
        (df['open'] > df['close'].shift(1)) &  # Opens above prev close
        (df['close'] < df['open'].shift(1))  # Closes below prev open
    ).astype(int)

    # Hammer / Shooting Star
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    df['hammer'] = (
        (df['lower_shadow'] > df['body'] * 2) &
        (df['upper_shadow'] < df['body'] * 0.3)
    ).astype(int)

    df['shooting_star'] = (
        (df['upper_shadow'] > df['body'] * 2) &
        (df['lower_shadow'] < df['body'] * 0.3)
    ).astype(int)

    return df
```

**Grupo 3: Price Action**:

```python
def calculate_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    # Higher highs / Lower lows (trend detection)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    # Support / Resistance proximity
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    df['dist_to_support'] = (df['close'] - df['support']) / df['close']
    df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']

    # Volatility squeeze (Bollinger Bands width)
    df['volatility_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)

    return df
```

**Total de Features**: ~30 features técnicas

### Etapa 3: Criação do Dataset de Treinamento (1 dia)

#### Labeling Strategy

**Arquivo**: `backend/ml/research/scalping_labeling.py` (NOVO)

```python
def label_scalping_setup(df: pd.DataFrame, tp_pct: float = 0.5, sl_pct: float = 0.25, max_candles: int = 20) -> pd.DataFrame:
    """
    Labels:
    - 1 (LONG): TP atingido antes de SL
    - -1 (SHORT): TP atingido antes de SL (direção inversa)
    - 0 (NO_TRADE): Nem TP nem SL atingido, ou setup inválido
    """

    labels = []

    for i in range(len(df) - max_candles):
        entry_price = df.iloc[i]['close']

        # Targets para LONG
        tp_long = entry_price * (1 + tp_pct / 100)
        sl_long = entry_price * (1 - sl_pct / 100)

        # Targets para SHORT
        tp_short = entry_price * (1 - tp_pct / 100)
        sl_short = entry_price * (1 + sl_pct / 100)

        hit_tp_long = False
        hit_sl_long = False
        hit_tp_short = False
        hit_sl_short = False

        # Verificar próximos candles
        for j in range(i + 1, min(i + max_candles + 1, len(df))):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']

            # Verificar LONG
            if high >= tp_long:
                hit_tp_long = True
                break
            if low <= sl_long:
                hit_sl_long = True
                break

            # Verificar SHORT
            if low <= tp_short:
                hit_tp_short = True
                break
            if high >= sl_short:
                hit_sl_short = True
                break

        # Decidir label
        if hit_tp_long and not hit_sl_long:
            labels.append(1)  # LONG setup válido
        elif hit_tp_short and not hit_sl_short:
            labels.append(-1)  # SHORT setup válido
        else:
            labels.append(0)  # NO_TRADE

    # Preencher últimos candles com 0
    labels.extend([0] * max_candles)

    df['label'] = labels
    return df
```

**Configurações a testar**:
1. Conservative: TP 0.3% / SL 0.15% (R:R 1:2)
2. Moderate: TP 0.5% / SL 0.25% (R:R 1:2)
3. Aggressive: TP 0.75% / SL 0.35% (R:R 1:2.14)

### Etapa 4: Treinamento de Modelos XGBoost (1 dia)

#### Arquivo: `backend/ml/research/scalping_model_training.py` (NOVO)

**Estratégia**:
- Train/Validation/Test split: 60% / 20% / 20%
- Cross-validation: 5-fold time-series split
- Otimização de hiperparâmetros: Optuna (50 trials)

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import optuna

def train_scalping_model(X_train, y_train, X_val, y_val):
    """Treina modelo XGBoost para scalping"""

    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,  # LONG, SHORT, NO_TRADE
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        }

        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Otimizar para F1-score das classes 1 e -1 (ignorar NO_TRADE)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred, average='macro', labels=[1, -1])

        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Treinar modelo final com melhores hiperparâmetros
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    return model, best_params
```

**Métricas a avaliar**:
- Accuracy geral
- F1-score para LONG e SHORT
- Precision/Recall por classe
- Confusion matrix
- Feature importance

### Etapa 5: Backtesting em Out-of-Sample (1-2 dias)

#### Arquivo: `backend/ml/research/scalping_backtest.py` (NOVO)

**Estratégia**:
- Período: 3 meses out-of-sample (mais recentes)
- Simular trades reais usando previsões do modelo
- Calcular métricas de trading

```python
def backtest_scalping_model(model, X_test, y_test, df_test, tp_pct=0.5, sl_pct=0.25):
    """Backtest do modelo em dados out-of-sample"""

    predictions = model.predict(X_test)

    trades = []

    for i in range(len(predictions)):
        pred = predictions[i]

        if pred == 0:  # NO_TRADE
            continue

        entry_price = df_test.iloc[i]['close']
        entry_time = df_test.iloc[i]['timestamp']
        direction = 'LONG' if pred == 1 else 'SHORT'

        # Calcular TP e SL
        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

        # Simular execução
        for j in range(i + 1, min(i + 21, len(df_test))):  # Max 20 candles
            high = df_test.iloc[j]['high']
            low = df_test.iloc[j]['low']
            exit_time = df_test.iloc[j]['timestamp']

            # Verificar TP
            if (direction == 'LONG' and high >= tp_price) or (direction == 'SHORT' and low <= tp_price):
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': tp_price,
                    'result': 'WIN',
                    'pnl_pct': tp_pct if direction == 'LONG' else -tp_pct,
                    'duration_candles': j - i
                })
                break

            # Verificar SL
            if (direction == 'LONG' and low <= sl_price) or (direction == 'SHORT' and high >= sl_price):
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': sl_price,
                    'result': 'LOSS',
                    'pnl_pct': -sl_pct if direction == 'LONG' else sl_pct,
                    'duration_candles': j - i
                })
                break

    # Calcular métricas
    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_pnl = trades_df['pnl_pct'].sum()
    avg_win = trades_df[trades_df['result'] == 'WIN']['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl_pct'].mean() if losses > 0 else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 else 0

    # Sharpe ratio (anualizado)
    returns = trades_df['pnl_pct'].values
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else 0

    # Max drawdown
    cumulative = (1 + trades_df['pnl_pct'] / 100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl_pct': total_pnl,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'avg_duration_candles': trades_df['duration_candles'].mean(),
        'trades_df': trades_df
    }
```

**Métricas de Aprovação**:
- Win rate > 55%
- Profit factor > 1.5
- Sharpe ratio > 1.0
- Max drawdown < 20%

---

## 📋 CRONOGRAMA DETALHADO

### Dia 1: Coleta de Dados M5
- [  ] Modificar `scalping_volatility_analysis.py` (granularidade 300)
- [  ] Coletar 1HZ75V em M5 (6 meses)
- [  ] Coletar 1HZ100V em M5 (6 meses)
- [  ] Validar dados (sem missing values, timestamps corretos)

### Dia 2: Feature Engineering
- [  ] Criar `scalping_feature_engineering.py`
- [  ] Implementar features técnicas (RSI, BB, Stoch, MACD, EMA)
- [  ] Implementar candlestick patterns
- [  ] Implementar price action features
- [  ] Validar features (sem NaN, sem leakage)

### Dia 3: Labeling e Dataset
- [  ] Criar `scalping_labeling.py`
- [  ] Gerar labels para 3 configs (Conservative/Moderate/Aggressive)
- [  ] Split train/val/test (60/20/20)
- [  ] Análise exploratória (distribuição de labels, correlação de features)

### Dia 4: Treinamento de Modelos
- [  ] Criar `scalping_model_training.py`
- [  ] Treinar modelo V75 (Optuna 50 trials)
- [  ] Treinar modelo V100 (Optuna 50 trials)
- [  ] Avaliar em validation set
- [  ] Salvar modelos e hiperparâmetros

### Dia 5-6: Backtesting Comparativo
- [  ] Criar `scalping_backtest.py`
- [  ] Backtest V75 em 3 meses out-of-sample
- [  ] Backtest V100 em 3 meses out-of-sample
- [  ] Comparar métricas (win rate, profit factor, Sharpe, drawdown)
- [  ] Gerar relatórios individuais
- [  ] Gerar relatório comparativo V75 vs V100

### Dia 7: Decisão e Documentação
- [  ] Analisar resultados comparativos
- [  ] Decidir: V75, V100, ou ambos (50/50)
- [  ] Atualizar `SCALPING_RESEARCH_ROADMAP.md`
- [  ] Criar `FASE_02_RESULTS.md` com conclusões
- [  ] Commit e push para repositório

**Tempo total estimado**: 6-7 dias

---

## 🎯 CRITÉRIOS DE APROVAÇÃO

### Para um ativo ser aprovado para Fase 1 (Forward Testing):

| Métrica | Mínimo | Ideal |
|---------|--------|-------|
| Win Rate | > 55% | > 60% |
| Profit Factor | > 1.5 | > 2.0 |
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 20% | < 15% |
| Total Trades (3 meses) | > 100 | > 200 |

### Decisão Final:

**Se AMBOS aprovados**:
- Usar portfólio 50/50 (diversificação)
- Implementar gerenciamento de risco por ativo

**Se APENAS V75 aprovado**:
- Focar apenas em V75
- Usar 100% do capital

**Se APENAS V100 aprovado**:
- Focar apenas em V100
- Usar 100% do capital (com position sizing conservador)

**Se NENHUM aprovado**:
- **DESISTIR** de scalping
- **FOCAR** em R_100 swing (já validado: 62.58% accuracy)

---

## 📊 COMPARAÇÃO ESPERADA

### V75 (Hipótese)
- ✅ Win rate: 55-65% (estrutura mais limpa)
- ⚠️ Swings: 0.5% por trade
- ✅ Sharpe: 1.2-1.8 (menos volátil)
- ✅ Trades/dia: 5-10
- **Perfil**: Consistência, menor risco

### V100 (Hipótese)
- ⚠️ Win rate: 50-60% (mais caótico)
- ✅ Swings: 0.65% por trade (+30%)
- ⚠️ Sharpe: 1.0-1.5 (mais volátil)
- ✅ Trades/dia: 5-10
- **Perfil**: Lucros maiores, maior risco

### Portfólio 50/50 (Hipótese)
- Win rate: 52-62% (média ponderada)
- Swings: 0.58% por trade
- Sharpe: 1.1-1.6 (diversificação reduz volatilidade)
- Trades/dia: 10-20
- **Perfil**: MELHOR opção (diversificação + swings aceitáveis)

---

## 🚀 PRÓXIMOS PASSOS APÓS FASE 0.2

### Se >= 1 Ativo Aprovado:

**Fase 1: Forward Testing** (1-2 semanas)
- Paper trading em produção
- 100 trades mínimo por ativo
- Validar win rate real vs backtest
- Ajustar hiperparâmetros se necessário

**Fase 2: Trading Real** (gradual)
- Semana 1: $100 capital, 0.01 lote
- Semana 2: Se win rate > 55% → $500, 0.05 lote
- Semana 3: Se win rate > 55% → $2000, 0.2 lote

### Se Nenhum Ativo Aprovado:

**Path B: Focar em R_100 Swing**
- Otimizar modelo existente (62.58% accuracy)
- Implementar trailing stop
- Position sizing dinâmico
- Deploy em produção (1 semana)

---

## 📚 REFERÊNCIAS TÉCNICAS

### Libraries Python Necessárias:
```bash
pip install ta  # Technical Analysis library
pip install optuna  # Hyperparameter optimization
pip install shap  # Feature importance
```

### Papers e Recursos:
1. [Technical Analysis Library Documentation](https://technical-analysis-library-in-python.readthedocs.io/)
2. [XGBoost Hyperparameter Tuning Guide](https://xgboost.readthedocs.io/en/stable/parameter.html)
3. [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

**Criado por**: Claude Sonnet 4.5
**Data**: 18/12/2025
**Versão**: 1.0

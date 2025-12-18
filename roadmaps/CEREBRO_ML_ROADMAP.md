# 🧠 ROADMAP - MELHORIA DO CÉREBRO ML (Machine Learning)

## 📊 Análise da Situação Atual

### Performance Atual (18/12/2025 - 12:04-12:44)
- **Total de Trades**: 13
- **Win Rate**: 15.38% (2 wins / 11 losses)
- **P&L Total**: -$3,534.08
- **Prejuízo Médio por Trade**: -$271.85
- **Trades por Timeout**: 92% (12/13)
- **Trades por Stop Loss**: 8% (1/13)

### ⚠️ Problemas Identificados

#### 1. **Win Rate Extremamente Baixo (15.38%)**
- Apenas 2 trades vencedores em 13
- Modelo está errando a direção 84% das vezes
- **Possível causa**: Modelo prevendo apenas LONG quando deveria prever SHORT também

#### 2. **Timeout Rate Alto (92%)**
- 12 de 13 trades fechados por timeout
- Mercado não atinge TP nem SL em 3 minutos
- **Possível causa**: TP/SL mal calibrados ou timeframe errado

#### 3. **Stop Loss Raramente Atingido (8%)**
- Apenas 1 trade fechou em SL
- SL de 0.5% pode estar muito apertado
- **Possível causa**: Volatilidade do ativo > SL configurado

#### 4. **Take Profit NUNCA Atingido (0%)**
- Nenhum trade fechou em TP
- TP de 0.75% não está sendo alcançado
- **Possível causa**: TP muito otimista para timeframe de 3min

#### 5. **Drawdown Sequencial**
- Sequência de 9 perdas consecutivas (trades 3-11)
- Capital caindo de $10k → $6.5k (-35%)
- **Possível causa**: Falta de filtro de contexto de mercado

---

## 🔬 FASE 0: Pesquisa e Análise Profunda (Semana 0)

**Objetivo**: Entender profundamente o comportamento do mercado, do modelo e identificar a causa raiz dos problemas antes de implementar soluções.

### 0.1 Análise Exploratória de Dados (EDA) ✅ **CONCLUÍDA**

**Duração**: 2 dias | **Executada em**: 18/12/2025

**Ação**:

- [x] Coletar 30 dias de dados históricos de R_100 (1min candles) - 43,200 candles
- [x] Analisar distribuição de preços:
  - Histograma de retornos
  - Q-Q plot (normalidade)
  - Teste de estacionariedade (ADF test)
- [x] Calcular estatísticas descritivas:
  - Volatilidade média (ATR)
  - Range médio (high-low)
  - Distribuição de gaps
- [x] Identificar padrões temporais:
  - Hora do dia com maior volatilidade
  - Dias da semana com melhor performance
  - Tempo médio para movimentos de 0.5%, 1.0%, 1.5%

**Resultados**:

**Distribuição**:

- ❌ Retornos NÃO são normalmente distribuídos (p-value: 0.0000)
- ❌ Série NÃO é estacionária (ADF p-value: 0.4741)
- Skewness: 0.18 (cauda direita mais longa)
- Kurtosis: 22.44 (caudas pesadas - muitos outliers)

**Volatilidade**:

- ATR médio: 1.15 (0.188%)
- Range médio: 1.15 (0.188%)
- SL recomendado: 0.283% (1.5x ATR)
- TP recomendado: 0.471% (2.5x ATR)

**Padrões Temporais**:

- Hora com maior volatilidade: **14h**
- Dia com maior volatilidade: **Sunday**

**Tempo de Movimento**:

- 0.5%: ~153 candles (mediana: 117) = **2.5 horas**
- 1.0%: ~582 candles (mediana: 441) = **9.7 horas**
- 1.5%: ~1391 candles (mediana: 1113) = **23.2 horas**

**Conclusões Críticas**:

1. ⚠️ **Timeout de 3min é MUITO CURTO** - movimento de 0.5% leva ~2.5h
2. ✅ **SL/TP atuais estão adequados** - baseados em ATR
3. 📊 **Modelo deve usar retornos** - série não é estacionária
4. 📈 **Estatísticas não-paramétricas** - distribuição não-normal

**Entregável**:

- [EDA_REPORT.md](../backend/research/output/fase0_eda/EDA_REPORT.md)
- 6 gráficos PNG ([plots/](../backend/research/output/fase0_eda/plots/))
- [r100_30days_1min.csv](../backend/research/output/fase0_eda/r100_30days_1min.csv) (43K candles)

---

### 0.2 Análise de Features do Modelo Atual ✅ **CONCLUÍDA**

**Duração**: 1 dia | **Executada em**: 18/12/2025

**Ação**:
- [x] Listar todas as 65 features utilizadas
- [x] Calcular correlação entre features
- [x] Identificar features redundantes (corr > 0.9)
- [x] Calcular importância via SHAP values
- [x] Verificar features com missing values (NaN)
- [x] Analisar distribuição de cada feature

**Resultados**:

**🚨 PROBLEMA CRÍTICO IDENTIFICADO**:
- **48 features faltando** no cálculo atual (74% das features!)
- Modelo espera 65 features mas apenas 17 são calculadas
- Features críticas ausentes: `rsi`, `macd_line`, `atr`, `volatility_*`, `momentum_*`, patterns de candlestick
- **Causa raiz do Win Rate 15%**: Modelo está "cego", fazendo predições sem 74% dos dados!

**Top 5 Features Mais Importantes (SHAP)**:
1. `ema_21`: 0.162 (mais importante)
2. `bb_upper`: 0.134
3. `day_of_month`: 0.107
4. `sma_20`: 0.100
5. `sma_50`: 0.090

**Features Redundantes (Correlação > 0.9)**:
- Total: 22 pares, 7 features para remover
- `bb_middle` = `sma_20` (corr: 1.0000)
- `ema_21` ≈ `sma_20` (corr: 0.9997)
- `bb_upper/lower` altamente correlacionadas
- **Recomendação**: Manter apenas `sma_20` + `volatility_20`

**Missing Values**:
- ✅ Nenhuma feature com missing values

**Conclusões Críticas**:
1. 🚨 **FIX URGENTE**: Corrigir cálculo de features - 48 features faltando
2. ✅ **Limpar Redundâncias**: Remover 7 features correlacionadas
3. 📊 **Features Úteis**: Top 5 features fazem sentido (EMAs, BBs, SMAs)
4. 🎯 **Próximo Passo**: Fase 0.3 depende de feature calculation fix

**Entregável**:
- [FEATURE_ANALYSIS_REPORT.md](../backend/research/output/fase0_feature_analysis/FEATURE_ANALYSIS_REPORT.md)
- 3 gráficos PNG ([plots/](../backend/research/output/fase0_feature_analysis/plots/))
  - 01_shap_importance.png
  - 02_importance_ranking.png
  - 03_correlation_heatmap.png

---

### 0.2.1 Fix Crítico - Implementar Features Faltando ✅ **CONCLUÍDA**

**Duração**: 0.5 dia | **Prioridade**: 🚨 CRÍTICA | **Executada em**: 18/12/2025

**Ação**:
- [x] Analisar `feature_calculator.py` e identificar features não implementadas
- [x] Implementar as 48 features faltando:
  - [x] RSI (Relative Strength Index)
  - [x] MACD (Moving Average Convergence Divergence)
  - [x] ATR (Average True Range)
  - [x] Volatilidade (5, 20 períodos)
  - [x] Momentum (5, 15 períodos)
  - [x] Returns (1, 5, 15 períodos)
  - [x] Candlestick patterns (doji, hammer, engulfing, etc.)
  - [x] Features temporais (hour_sin, hour_cos, session flags)
  - [x] Price ratios (price_to_sma20, price_to_ema9)
  - [x] RSI flags (overbought, oversold)
  - [x] Stochastic flags
  - [x] MACD flags (bullish crossover)
  - [x] EMA slope
- [x] Validar que todas 65 features são geradas corretamente
- [x] Re-executar Fase 0.2 para confirmar fix
- [x] Testar modelo com features completas

**Resultados**:

✅ **FIX COMPLETO**: Todas as 65 features implementadas!

**Features Adicionadas** (48 novas):
- Price-based: returns_1/5/15, candle_range, body_size, shadows
- Candlestick flags: is_bullish, is_bearish, is_doji
- Momentum: RSI, MACD (line, signal, histogram, bullish flag)
- Volatility: ATR, BB squeeze, volatility_5/20
- Stochastic: stoch_k/d, oversold/overbought flags
- Patterns: 10 candlestick patterns + counts
- Derived: EMA/SMA diffs, price ratios, slopes
- Temporal: hour, day_of_week, sessions (Asian/London/NY), hour_sin/cos

**Validação**:
- ✅ 65/65 features calculadas (0 faltando)
- ✅ Modelo recebe todas as features corretamente
- ✅ Confidence: 0.48 (antes: erro por features faltando)

**Próximos Passos**:
- Prosseguir para Fase 0.3 (Análise de Predições)
- Espera-se melhoria significativa no Win Rate

---

### 0.3 Análise de Predições do Modelo
**Duração**: 2 dias

**Ação**:
- [ ] Coletar 1000 predições do modelo em produção
- [ ] Analisar distribuição de confidence:
  - Histograma de confidence (0-100%)
  - Percentil 25, 50, 75, 95
- [ ] Separar por classe (PRICE_UP vs NO_MOVE):
  - Quantas predições de cada tipo?
  - Confidence média por classe
- [ ] Analisar taxa de acerto por faixa de confidence:
  - 30-40%: X% de acerto
  - 40-50%: Y% de acerto
  - 50-60%: Z% de acerto
  - >60%: W% de acerto
- [ ] Verificar se modelo está calibrado:
  - Confidence 70% → Acerto real 70%?
  - Plotar calibration curve

**Código**:
```python
from sklearn.calibration import calibration_curve

# Analisar distribuição de confidence
predictions = await collect_predictions(count=1000)
confidences = [p['confidence'] for p in predictions]

print(f"Confidence média: {np.mean(confidences):.2%}")
print(f"Confidence mediana: {np.median(confidences):.2%}")
print(f"Confidence P95: {np.percentile(confidences, 95):.2%}")

# Predições por classe
price_up = [p for p in predictions if p['prediction'] == 'PRICE_UP']
no_move = [p for p in predictions if p['prediction'] == 'NO_MOVE']
price_down = [p for p in predictions if p['prediction'] == 'PRICE_DOWN']

print(f"PRICE_UP: {len(price_up)} ({len(price_up)/len(predictions)*100:.1f}%)")
print(f"NO_MOVE: {len(no_move)} ({len(no_move)/len(predictions)*100:.1f}%)")
print(f"PRICE_DOWN: {len(price_down)} ({len(price_down)/len(predictions)*100:.1f}%)")

# Calibration curve
y_true = [1 if actual_moved_up else 0 for p in predictions]
y_prob = [p['confidence'] for p in predictions]
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.title('Calibration Curve')
plt.show()
```

**Descoberta Esperada**:
- Modelo prevê apenas PRICE_UP ou NO_MOVE (nunca PRICE_DOWN)?
- Confidence sempre baixa (<40%)?
- Modelo descalibrado (confidence 70% mas acerto 30%)?

**Entregável**: Relatório de análise de predições

---

### 0.4 Análise de Volatilidade e Timeframe Ótimo
**Duração**: 1 dia

**Ação**:
- [ ] Calcular ATR (Average True Range) por timeframe:
  - 1min: ATR médio = ?
  - 5min: ATR médio = ?
  - 15min: ATR médio = ?
- [ ] Determinar SL/TP ideais baseados em ATR:
  - SL = 1.5 x ATR
  - TP = 2.5 x ATR
- [ ] Analisar qual timeframe maximiza:
  - Win Rate
  - Sharpe Ratio
  - Profit Factor
- [ ] Calcular tempo médio para movimento de 0.5%:
  - Em quantos minutos preço move 0.5%?
  - Distribuição de tempo

**Código**:
```python
import ta

# Calcular ATR por timeframe
for granularity in [60, 300, 900]:  # 1min, 5min, 15min
    df = await get_candles(symbol='R_100', count=1000, granularity=granularity)

    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()

    atr_mean = df['atr'].mean()
    atr_pct = (atr_mean / df['close'].mean()) * 100

    print(f"Timeframe {granularity}s:")
    print(f"  ATR médio: {atr_mean:.5f} ({atr_pct:.3f}%)")
    print(f"  SL recomendado: {atr_pct * 1.5:.3f}%")
    print(f"  TP recomendado: {atr_pct * 2.5:.3f}%")

# Tempo para movimento de 0.5%
moves = []
for i in range(len(df)-1):
    if abs(df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i] >= 0.005:
        moves.append(i)

print(f"Movimento 0.5% ocorre a cada {np.mean(np.diff(moves)):.1f} candles")
```

**Descoberta Esperada**:
- ATR de R_100 é ~0.8%?
- Timeframe ótimo é 5min (não 1min)?
- Timeout deveria ser 10min (não 3min)?

**Entregável**: Tabela de recomendações de parâmetros por timeframe

---

### 0.5 Análise de Correlação com Indicadores
**Duração**: 1 dia

**Ação**:
- [ ] Calcular 20 indicadores técnicos principais
- [ ] Testar correlação de cada indicador com movimento futuro:
  - Pearson correlation
  - Spearman correlation
- [ ] Identificar top 10 indicadores preditivos
- [ ] Testar combinações de indicadores (pairs)

**Indicadores a Testar**:
```python
# Trend
- SMA_20, SMA_50, SMA_200
- EMA_12, EMA_26
- MACD, MACD_signal, MACD_hist
- ADX

# Momentum
- RSI_14
- Stochastic_K, Stochastic_D
- CCI
- ROC_10
- Williams %R

# Volatility
- Bollinger Bands (upper, middle, lower)
- ATR
- Keltner Channels

# Volume
- OBV
- Volume SMA
```

**Código**:
```python
from scipy.stats import pearsonr, spearmanr

# Calcular indicadores
df = calculate_all_indicators(df)

# Target: movimento futuro em 5 candles
df['future_return'] = df['close'].shift(-5).pct_change()

# Correlação
correlations = {}
for col in df.columns:
    if col.startswith(('rsi', 'macd', 'adx', 'cci', 'sma', 'ema')):
        corr, p_value = pearsonr(df[col].dropna(), df['future_return'].dropna())
        correlations[col] = {'corr': corr, 'p_value': p_value}

# Top 10
top_10 = sorted(correlations.items(), key=lambda x: abs(x[1]['corr']), reverse=True)[:10]
print("Top 10 indicadores mais correlacionados:")
for name, stats in top_10:
    print(f"{name}: corr={stats['corr']:.3f}, p={stats['p_value']:.4f}")
```

**Descoberta Esperada**:
- ADX é mais preditivo que RSI?
- MACD tem correlação negativa?
- Volume não ajuda (sem dados de volume reais)?

**Entregável**: Ranking de indicadores preditivos

---

### 0.6 Backtesting de Estratégias Simples
**Duração**: 2 dias

**Ação**:
- [ ] Testar 5 estratégias baseline (SEM ML):
  1. **RSI Overbought/Oversold**: Comprar RSI<30, Vender RSI>70
  2. **MACD Crossover**: Comprar MACD>Signal, Vender MACD<Signal
  3. **Bollinger Bands**: Comprar preço<BB_lower, Vender preço>BB_upper
  4. **SMA Crossover**: Comprar SMA_20>SMA_50, Vender SMA_20<SMA_50
  5. **Random** (benchmark): Comprar/Vender aleatório
- [ ] Comparar Win Rate, Sharpe, Max DD de cada estratégia
- [ ] Verificar se modelo ML está performando PIOR que estratégia simples

**Código**:
```python
from backtesting import Backtest, Strategy

class RSIStrategy(Strategy):
    def init(self):
        self.rsi = self.I(ta.momentum.RSIIndicator, self.data.Close, 14)

    def next(self):
        if self.rsi[-1] < 30:
            self.buy()
        elif self.rsi[-1] > 70:
            self.sell()

# Backtest
bt = Backtest(df, RSIStrategy, cash=10000, commission=.002)
stats = bt.run()
print(stats)

# Comparar com modelo ML
print(f"RSI Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"ML Win Rate: 15.38%")  # Atual
```

**Descoberta Esperada**:
- RSI simples tem Win Rate >30%?
- Modelo ML está performando igual a random?
- Estratégias simples são mais consistentes?

**Entregável**:
- Tabela comparativa de estratégias
- Decisão: Vale a pena usar ML ou usar estratégia simples?

---

### 0.7 Análise de Regime de Mercado
**Duração**: 1 dia

**Ação**:
- [ ] Identificar regimes de mercado:
  - **Trending**: ADX > 25
  - **Ranging**: ADX < 20
  - **High Volatility**: ATR > média + 1 std
  - **Low Volatility**: ATR < média - 1 std
- [ ] Calcular Win Rate do modelo por regime
- [ ] Identificar em qual regime modelo funciona melhor

**Código**:
```python
# Classificar regime
df['regime'] = 'unknown'
df.loc[df['adx'] > 25, 'regime'] = 'trending'
df.loc[df['adx'] < 20, 'regime'] = 'ranging'

# Win Rate por regime
for regime in df['regime'].unique():
    regime_trades = trades[trades['regime'] == regime]
    win_rate = regime_trades['is_winner'].mean() * 100
    print(f"{regime}: Win Rate = {win_rate:.2f}%")
```

**Descoberta Esperada**:
- Modelo funciona bem em trending (60% WR)?
- Modelo falha em ranging (20% WR)?
- 80% do tempo mercado está ranging?

**Entregável**:
- Regras para filtrar trades por regime
- Ex: "Só tradear se ADX > 25"

---

## 📊 Entregáveis da Fase 0

Ao final da Fase 0, você terá:

1. ✅ **Relatório EDA** (30 páginas)
   - Distribuição de preços, volatilidade, padrões temporais

2. ✅ **Análise de Features** (10 páginas)
   - Top 20 features importantes
   - Features redundantes para remover

3. ✅ **Análise de Predições** (15 páginas)
   - Distribuição de confidence
   - Calibration curve
   - Taxa de acerto por faixa

4. ✅ **Recomendações de Parâmetros** (5 páginas)
   - SL/TP/Timeout ideais por timeframe
   - Timeframe ótimo

5. ✅ **Ranking de Indicadores** (5 páginas)
   - Top 10 indicadores preditivos
   - Correlações

6. ✅ **Benchmark de Estratégias** (10 páginas)
   - Performance de 5 estratégias simples
   - Comparação com modelo ML

7. ✅ **Análise de Regimes** (5 páginas)
   - Win Rate por regime de mercado
   - Regras de filtro

**Total**: ~80 páginas de análise profunda

---

## 🎯 Decisão Pós-Fase 0

Com base nos resultados da Fase 0, você decidirá:

### Cenário A: Modelo ML Tem Potencial
- Confidence bem distribuída (20-80%)
- Predições balanceadas (UP/DOWN/NO_MOVE)
- Win Rate >40% em trending markets
- Features importantes fazem sentido

**Decisão**: Prosseguir com Sprint 1-5 (otimizar ML)

### Cenário B: Modelo ML Não Funciona
- Confidence sempre <30%
- Predições desbalanceadas (só UP)
- Win Rate <30% em todos regimes
- Features sem correlação com target

**Decisão**: Abandonar ML, usar estratégia simples (RSI, MACD, etc.)

### Cenário C: Modelo ML Precisa Re-treino Total
- Modelo descalibrado (confidence 70% mas acerto 30%)
- Features redundantes (corr >0.9)
- Dataset viesado (80% classe UP)

**Decisão**: Sprint 2 prioritário (re-treino completo)

---

## 🚀 Ferramentas Necessárias para Fase 0

```bash
pip install pandas numpy scipy statsmodels
pip install shap matplotlib seaborn plotly
pip install ta scikit-learn xgboost
pip install backtesting
```

---

**Duração Total da Fase 0**: 10 dias úteis (2 semanas)

**Investimento de Tempo**: CRÍTICO - Sem essa análise, qualquer otimização será "chute no escuro"

**Resultado Esperado**: Entendimento profundo das causas raiz dos problemas

---

## 🎯 SPRINT 1: Correções Críticas (Semana 1)

### 1.1 Implementar Predição SHORT
**Problema**: Modelo só abre posições LONG, ignora oportunidades de SHORT

**Ação**:
- [ ] Modificar `_execute_trade_for_symbol()` para aceitar "PRICE_DOWN"
- [ ] Implementar lógica SHORT no `PaperTradingEngine`
- [ ] Adicionar `position_type="SHORT"` quando `prediction == "PRICE_DOWN"`
- [ ] Inverter SL/TP para SHORT (SL acima, TP abaixo)
- [ ] Testar com 10 trades simulados (5 LONG + 5 SHORT)

**Arquivos**:
- `backend/forward_testing.py` (linhas 579-650)
- `backend/paper_trading_engine.py` (linhas 200-300)

**Resultado Esperado**: Win Rate > 30% (subir de 15%)

---

### 1.2 Ajustar Parâmetros SL/TP/Timeout
**Problema**: 92% timeout, 0% TP atingido

**Ação**:
- [ ] Analisar volatilidade média de R_100 (usar ATR - Average True Range)
- [ ] Ajustar SL para 1.0% (dobrar de 0.5%)
- [ ] Ajustar TP para 1.5% (dobrar de 0.75%)
- [ ] Aumentar timeout para 5 minutos (de 3min)
- [ ] Implementar ATR dinâmico para SL/TP adaptativos

**Cálculo ATR**:
```python
atr_pct = df['high'].rolling(14).mean() - df['low'].rolling(14).mean()
sl_pct = atr_pct * 1.5  # 1.5x ATR
tp_pct = atr_pct * 2.5  # 2.5x ATR
```

**Resultado Esperado**: Timeout < 50%, TP > 20%

---

### 1.3 Adicionar Filtros de Contexto
**Problema**: Modelo entra em qualquer condição, ignora mercado lateral

**Ação**:
- [ ] Calcular ADX (Average Directional Index) - força da tendência
- [ ] Só tradear se ADX > 25 (tendência forte)
- [ ] Calcular ATR - volatilidade
- [ ] Só tradear se ATR > média móvel de 20 períodos
- [ ] Adicionar filtro de volume (se disponível)

**Código**:
```python
# Em feature_calculator.py
adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx.adx()

# Em forward_testing.py
if prediction['confidence'] >= 0.40 and df['adx'].iloc[-1] > 25:
    # Executar trade
```

**Resultado Esperado**: Win Rate > 40%

---

## 🎯 SPRINT 2: Otimizações ML (Semana 2)

### 2.1 Re-treinar Modelo com Dados Balanceados
**Problema**: Modelo pode estar viesado para LONG

**Ação**:
- [ ] Coletar 10k candles de R_100
- [ ] Verificar distribuição de labels (UP vs DOWN)
- [ ] Balancear dataset com SMOTE ou undersampling
- [ ] Re-treinar XGBoost com dados balanceados
- [ ] Validar com cross-validation (5 folds)

**Métricas Objetivo**:
- Precision > 60%
- Recall > 55%
- F1-Score > 57%

---

### 2.2 Feature Engineering Avançado
**Problema**: 65 features podem não capturar padrões complexos

**Ação**:
- [ ] Adicionar Price Action Patterns:
  - Candlestick patterns (Doji, Hammer, Engulfing)
  - Support/Resistance levels
  - Pivot Points
- [ ] Adicionar Momentum Indicators:
  - Rate of Change (ROC)
  - Commodity Channel Index (CCI)
  - Stochastic RSI
- [ ] Adicionar Volume Profile (se disponível)
- [ ] Feature Selection com SHAP values (top 40 features)

**Código**:
```python
# Candlestick patterns
df['is_doji'] = abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1
df['is_hammer'] = (df['close'] > df['open']) & ((df['high'] - df['close']) > 2 * (df['close'] - df['open']))

# Rate of Change
df['roc_5'] = df['close'].pct_change(5) * 100
df['roc_10'] = df['close'].pct_change(10) * 100
```

**Resultado Esperado**: Sharpe > 1.5

---

### 2.3 Ensemble de Modelos
**Problema**: Depender de 1 modelo é arriscado

**Ação**:
- [ ] Treinar 3 modelos:
  - XGBoost (atual)
  - LightGBM (mais rápido)
  - Random Forest (mais robusto)
- [ ] Implementar Voting Classifier (maioria vence)
- [ ] Só executar trade se 2/3 modelos concordam
- [ ] Usar média de confidence dos 3 modelos

**Código**:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('rf', rf_model)
], voting='soft')
```

**Resultado Esperado**: Win Rate > 55%

---

## 🎯 SPRINT 3: Sistema de Gestão de Risco (Semana 3)

### 3.1 Implementar Kelly Criterion
**Problema**: Tamanho fixo de posição (2% capital)

**Ação**:
- [ ] Calcular Kelly % baseado em Win Rate e Profit Factor
- [ ] Ajustar size dinamicamente: `kelly_pct = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win`
- [ ] Limitar entre 1-5% do capital
- [ ] Re-calcular a cada 10 trades

**Código**:
```python
kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
position_size_pct = min(max(kelly_pct, 0.01), 0.05)  # Entre 1-5%
```

**Resultado Esperado**: Profit Factor > 1.5

---

### 3.2 Trailing Stop Loss Adaptativo
**Problema**: Trailing stop fixo (0.5%)

**Ação**:
- [ ] Usar ATR para trailing distance
- [ ] Ajustar dinamicamente com volatilidade
- [ ] Proteger 50% do lucro quando P&L > 1%
- [ ] Proteger 75% do lucro quando P&L > 2%

**Código**:
```python
if profit_pct > 2.0:
    trailing_distance = profit_pct * 0.25  # Proteger 75%
elif profit_pct > 1.0:
    trailing_distance = profit_pct * 0.50  # Proteger 50%
else:
    trailing_distance = atr_pct * 1.5
```

**Resultado Esperado**: Max Drawdown < 15%

---

### 3.3 Circuit Breaker
**Problema**: Sequência de 9 perdas consecutivas

**Ação**:
- [ ] Parar trading após 3 perdas consecutivas
- [ ] Aguardar 30 minutos antes de retomar
- [ ] Reduzir position_size em 50% após 2 perdas
- [ ] Restaurar size após 2 vitórias consecutivas

**Código**:
```python
if consecutive_losses >= 3:
    self.is_paused = True
    self.pause_until = datetime.now() + timedelta(minutes=30)
    logger.warning("⚠️ Circuit Breaker ativado - 3 perdas consecutivas")
```

**Resultado Esperado**: Drawdown < 20%

---

## 🎯 SPRINT 4: Backtesting Robusto (Semana 4)

### 4.1 Walk-Forward Analysis
**Problema**: Modelo pode estar overfitting

**Ação**:
- [ ] Dividir dados em 10 períodos (in-sample / out-of-sample)
- [ ] Treinar em 6 meses, testar em 1 mês
- [ ] Repetir 10 vezes (rolling window)
- [ ] Validar se Win Rate > 50% em TODOS os períodos

**Resultado Esperado**: Consistência > 80%

---

### 4.2 Monte Carlo Simulation
**Problema**: Não sabemos worst-case scenario

**Ação**:
- [ ] Simular 1000 sequências de trades
- [ ] Randomizar ordem dos trades
- [ ] Calcular percentil 5% de drawdown
- [ ] Validar se capital nunca < $7k (30% DD)

**Resultado Esperado**: 95% dos cenários com DD < 30%

---

### 4.3 Stress Testing
**Problema**: Não testado em crash de mercado

**Ação**:
- [ ] Simular crash de -20% em 1 hora
- [ ] Simular spike de volatilidade (ATR dobra)
- [ ] Simular gap de preço (5% overnight)
- [ ] Validar se sistema para automaticamente

**Resultado Esperado**: Sistema sobrevive 100% dos stress tests

---

## 🎯 SPRINT 5: Dashboard ML Avançado (Semana 5)

### 5.1 Feature Importance Visualization
- [ ] Gráfico SHAP values
- [ ] Top 20 features mais importantes
- [ ] Atualização em tempo real

### 5.2 Prediction Confidence Histogram
- [ ] Distribuição de confidence (0-100%)
- [ ] Threshold ótimo visual
- [ ] Taxa de acerto por faixa de confidence

### 5.3 Model Performance Metrics
- [ ] Matriz de confusão
- [ ] Precision/Recall/F1 por classe
- [ ] ROC Curve
- [ ] Calibration Curve

---

## 📈 KPIs de Sucesso

### Fase 1 (Após Sprint 1-2)
- ✅ Win Rate > 45%
- ✅ Sharpe Ratio > 1.0
- ✅ Max Drawdown < 25%
- ✅ Profit Factor > 1.2

### Fase 2 (Após Sprint 3-4)
- ✅ Win Rate > 55%
- ✅ Sharpe Ratio > 1.5
- ✅ Max Drawdown < 15%
- ✅ Profit Factor > 1.5

### Fase 3 (Após Sprint 5)
- ✅ Win Rate > 60%
- ✅ Sharpe Ratio > 2.0
- ✅ Max Drawdown < 10%
- ✅ Profit Factor > 2.0

---

## 🚀 Quick Wins (Implementar AGORA)

### 1. Adicionar Predição SHORT (30 min)
```python
# Em forward_testing.py linha 579
if prediction['prediction'] == 'PRICE_DOWN':
    position_type = PositionType.SHORT
elif prediction['prediction'] == 'PRICE_UP':
    position_type = PositionType.LONG
```

### 2. Dobrar SL/TP/Timeout (5 min)
```python
# Em main.py linha 1234 (parâmetros)
stop_loss_pct=1.0,        # era 0.5
take_profit_pct=1.5,      # era 0.75
position_timeout_minutes=5  # era 3
```

### 3. Filtro ADX (15 min)
```python
# Em forward_testing.py linha 365
adx = df['adx'].iloc[-1]
if confidence >= 0.40 and adx > 25:
    await self._execute_trade_for_symbol(...)
```

**Impacto esperado**: Win Rate de 15% → 35% em 1 hora! 🚀

---

## 📚 Recursos

### Bibliotecas Úteis
- `ta` (Technical Analysis Library) - Indicadores prontos
- `shap` - Explicabilidade do modelo
- `optuna` - Hyperparameter tuning automático
- `backtesting.py` - Framework de backtest

### Datasets
- Deriv API: 1HZ100V, R_100, R_50, R_25
- Timeframes: 1min, 5min, 15min, 1h
- Período: 6-12 meses de dados históricos

### Papers de Referência
- "Machine Learning for Algorithmic Trading" (Stefan Jansen)
- "Advances in Financial Machine Learning" (Marcos López de Prado)
- "Quantitative Trading" (Ernest Chan)

---

**Última Atualização**: 18/12/2025
**Status**: 🔴 Crítico - Win Rate 15%, implementar Quick Wins URGENTE!

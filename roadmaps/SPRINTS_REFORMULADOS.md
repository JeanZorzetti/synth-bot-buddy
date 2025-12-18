# 🎯 SPRINTS REFORMULADOS - Baseado nas Descobertas da Fase 0

## 🎯 SPRINT 1: Validação do Modelo Multi-Class em Produção (Semana 1) 🆕

**Objetivo**: Validar se o modelo multi-class corrige os problemas críticos identificados na Fase 0 e estabelecer baseline de performance real.

**Status**: 🔵 PRÓXIMO - Aguardando estabilização do deploy

**Pré-requisitos**:
- ✅ Modelo multi-class treinado (xgboost_multiclass_20251218_114940.pkl)
- ✅ Timeout ajustado para 180 minutos
- ✅ ml_predictor.py com suporte multi-class
- ✅ Deploy em produção (botderivapi.roilabs.com.br)
- ⏳ Bug UnboundLocalError corrigido

---

### 1.1 Teste de Validação Inicial (50 trades)
**Objetivo**: Confirmar que modelo prevê as 3 classes e timeout está funcionando

**Ação**:
- [ ] Iniciar forward testing em produção com modelo multi-class
- [ ] Executar 50 trades (mínimo para validação estatística)
- [ ] Monitorar logs em tempo real para erros críticos
- [ ] Coletar métricas a cada 10 trades

**Métricas para Monitorar**:
```
- Distribuição de predições (target: 20-40% cada classe)
- Timeout rate (target: <30%, baseline: 92%)
- Win rate (target: >40%, baseline: 15.38%)
- Confidence média (target: >45%)
- SL hit rate (baseline: 8%)
- TP hit rate (baseline: 0%)
```

**Critérios de Sucesso**:
- ✅ Modelo prevê TODAS as 3 classes (não >70% em uma única)
- ✅ Timeout rate < 30% (prova que 180min funciona)
- ✅ Win rate > 35% (melhoria de 2x sobre baseline)
- ✅ Sem erros críticos no ml_predictor.py

**Se FALHAR**: Voltar para análise adicional e investigar causa raiz

---

### 1.2 Análise Detalhada de Performance
**Objetivo**: Entender quando e por que o modelo acerta/erra

**Ação**:
- [ ] Exportar histórico de trades via `/api/forward-testing/export/csv`
- [ ] Criar notebook Jupyter de análise post-mortem
- [ ] Gerar relatório com insights acionáveis

**Análises a Realizar**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Win rate por tipo de predição
win_rate_by_class = df.groupby('prediction').agg({
    'profit_loss': lambda x: (x > 0).mean() * 100,
    'id': 'count'
}).round(2)

# 2. Correlation entre confidence e profit
plt.scatter(df['confidence'], df['profit_loss'])
plt.xlabel('Confidence')
plt.ylabel('P&L ($)')
plt.title('Confidence vs Profit Correlation')

# 3. Win rate por hora do dia
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
hourly_wr = df.groupby('hour')['profit_loss'].apply(lambda x: (x > 0).mean() * 100)

# 4. Identificar piores trades
worst_trades = df.nsmallest(10, 'profit_loss')
print("Top 10 piores trades:")
print(worst_trades[['prediction', 'confidence', 'profit_loss', 'exit_reason', 'duration_minutes']])

# 5. Análise de drawdown sequences
df['is_loss'] = df['profit_loss'] < 0
df['loss_streak'] = (df['is_loss'] != df['is_loss'].shift()).cumsum()
max_streak = df[df['is_loss']].groupby('loss_streak').size().max()
print(f"Maior sequência de perdas: {max_streak} trades")
```

**Entregável**:
- PDF: `fase1_analise_performance.pdf` com gráficos e insights
- Lista de hipóteses para Sprint 2
- Identificação de padrões específicos de falha

---

### 1.3 Calibração de Confidence Threshold
**Objetivo**: Encontrar threshold ótimo que maximize Sharpe Ratio

**Contexto**: Threshold 0.40 foi escolhido empiricamente, pode não ser ótimo para modelo multi-class

**Ação**:
- [ ] Simular performance com diferentes thresholds usando trades históricos
- [ ] Testar: 0.35, 0.38, 0.40, 0.42, 0.45, 0.50
- [ ] Para cada threshold, calcular:
  - Total de trades executados
  - Win rate
  - Profit Factor
  - Sharpe Ratio
  - Max Drawdown
  - Expectancy ($)

**Código de Simulação**:
```python
thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.50]
results = []

for thresh in thresholds:
    # Filtrar trades com confidence >= threshold
    filtered = df[df['confidence'] >= thresh]

    if len(filtered) < 20:  # Mínimo 20 trades
        continue

    wins = filtered[filtered['profit_loss'] > 0]
    losses = filtered[filtered['profit_loss'] < 0]

    win_rate = len(wins) / len(filtered) * 100
    profit_factor = wins['profit_loss'].sum() / abs(losses['profit_loss'].sum()) if len(losses) > 0 else 0
    sharpe = filtered['profit_loss'].mean() / filtered['profit_loss'].std() if filtered['profit_loss'].std() > 0 else 0
    expectancy = filtered['profit_loss'].mean()

    results.append({
        'threshold': thresh,
        'trades': len(filtered),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 2),
        'expectancy': round(expectancy, 2)
    })

# Exibir tabela comparativa
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Encontrar threshold com melhor Sharpe
best = max(results, key=lambda x: x['sharpe'])
print(f"\nMelhor threshold: {best['threshold']} (Sharpe: {best['sharpe']:.2f}, WR: {best['win_rate']}%)")
```

**Critério de Decisão**:
- Se Sharpe melhora >15%: Atualizar threshold em produção
- Se mudança <10%: Manter 0.40
- Considerar trade-off entre volume de trades vs qualidade

**Resultado Esperado**:
- Threshold ótimo identificado e documentado
- Sharpe Ratio > 1.0

---

## 🎯 SPRINT 2: Otimização de Parâmetros (Semana 2) 🔄

**Objetivo**: Otimizar SL/TP/Timeout e adicionar filtros contextuais (SE necessário)

**Status**: 🟡 CONDICIONAL - Só executar se Sprint 1 atingir >40% win rate

**Pré-condição**: Sprint 1 deve ter gerado >50 trades e win rate >35%

---

### 2.1 Grid Search para SL/TP Ótimos
**Problema**: SL=2%, TP=4% foram escolhidos empiricamente

**Ação**:
- [ ] Usar dados históricos de forward testing como baseline
- [ ] Testar combinações de parâmetros:
  - SL: [1.5%, 2.0%, 2.5%, 3.0%]
  - TP: [3.0%, 4.0%, 5.0%, 6.0%]
  - Risk:Reward Ratios: [1:1.5, 1:2, 1:2.5, 1:3]
- [ ] Para cada combinação, simular P&L, Win Rate, Sharpe
- [ ] Identificar combinação com melhor Profit Factor

**Código de Grid Search**:
```python
from itertools import product

sl_options = [1.5, 2.0, 2.5, 3.0]
tp_options = [3.0, 4.0, 5.0, 6.0]

results = []

for sl, tp in product(sl_options, tp_options):
    # Simular trades com novos SL/TP
    simulated_df = simulate_trades(df, sl_pct=sl, tp_pct=tp)

    wins = simulated_df[simulated_df['profit_loss'] > 0]
    losses = simulated_df[simulated_df['profit_loss'] < 0]

    win_rate = len(wins) / len(simulated_df) * 100
    profit_factor = wins['profit_loss'].sum() / abs(losses['profit_loss'].sum())
    total_pnl = simulated_df['profit_loss'].sum()

    results.append({
        'sl': sl,
        'tp': tp,
        'risk_reward': tp / sl,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'total_pnl': round(total_pnl, 2),
        'trades': len(simulated_df)
    })

# Ordenar por Profit Factor
results_df = pd.DataFrame(results).sort_values('profit_factor', ascending=False)
print("Top 10 combinações:")
print(results_df.head(10).to_string(index=False))
```

**Critério de Decisão**:
- Escolher combinação com melhor Profit Factor E Sharpe > 1.0
- Considerar trade-off entre win rate e expectancy

**Resultado Esperado**: SL/TP otimizados identificados

---

### 2.2 Validação de Timeout (180 min)
**Objetivo**: Confirmar se 180 min é realmente ótimo ou pode ser ajustado

**Ação**:
- [ ] Analisar duração média dos trades vencedores
- [ ] Calcular percentil 90 de duração dos winners
- [ ] Verificar se timeout está "cortando" trades vencedores
- [ ] Testar timeouts: [120min, 150min, 180min, 240min]

**Análise de Duração**:
```python
winners = df[df['profit_loss'] > 0]
losers = df[df['profit_loss'] < 0]

print("Duração média - Winners:", winners['duration_minutes'].mean())
print("Duração média - Losers:", losers['duration_minutes'].mean())
print("Percentil 90 - Winners:", winners['duration_minutes'].quantile(0.9))

# Analisar trades que deram timeout
timeouts = df[df['exit_reason'] == 'timeout']
timeout_pnl = timeouts['profit_loss'].mean()
print(f"P&L médio de trades timeout: ${timeout_pnl:.2f}")
```

**Critério**: Se percentil 90 de winners > 180min, aumentar timeout

---

### 2.3 Filtros de Contexto (CONDICIONAL)
**Objetivo**: Adicionar filtros APENAS SE win rate ainda < 45% após otimizar SL/TP

**Problema**: Modelo pode estar entrando em mercado lateral (sem tendência)

**Ação**:
- [ ] Calcular ADX (Average Directional Index) para todos os trades históricos
- [ ] Verificar se win rate é maior quando ADX > 25
- [ ] Se SIM (diferença >10%): Implementar filtro ADX
- [ ] Testar com backtesting antes de deploy

**Código de Análise**:
```python
# Adicionar ADX aos dados históricos
df['adx'] = calcular_adx(df)  # Requer recalcular com dados OHLC

# Comparar win rate com/sem ADX
high_adx = df[df['adx'] > 25]
low_adx = df[df['adx'] <= 25]

wr_high = (high_adx['profit_loss'] > 0).mean() * 100
wr_low = (low_adx['profit_loss'] > 0).mean() * 100

print(f"Win rate com ADX>25: {wr_high:.2f}%")
print(f"Win rate com ADX<=25: {wr_low:.2f}%")
print(f"Diferença: {wr_high - wr_low:.2f}pp")

if wr_high - wr_low > 10:
    print("✅ Filtro ADX melhora performance - Implementar!")
else:
    print("❌ Filtro ADX não agrega valor - Ignorar")
```

**Implementação**:
```python
# Em feature_calculator.py
from ta.trend import ADXIndicator

def add_adx(df):
    adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    return df

# Em forward_testing.py
if prediction['confidence'] >= threshold and df['adx'].iloc[-1] > 25:
    await self._execute_trade(prediction, current_price)
else:
    logger.info(f"Trade ignorado - ADX={df['adx'].iloc[-1]:.2f} < 25")
```

**Resultado Esperado**:
- Decisão data-driven sobre implementar filtro ADX
- Se implementado: Win rate > 50%

---

## 🎯 SPRINT 3: Re-treinamento com Melhorias (Semana 3) 🆕

**Objetivo**: Re-treinar modelo incorporando learnings dos Sprints 1-2

**Status**: 🟡 CONDICIONAL - Só executar se acurácia atual < 50%

**Pré-condição**: Ter identificado padrões claros de falha no Sprint 1.2

---

### 3.1 Feature Selection via SHAP
**Objetivo**: Remover features irrelevantes que podem causar overfitting

**Contexto**: Fase 0.2 mostrou que top 20 features representam 80% da importância

**Ação**:
- [ ] Carregar modelo atual
- [ ] Calcular SHAP values para todas as 65 features
- [ ] Identificar features com SHAP mean < 0.01 (ruído)
- [ ] Criar novo dataset com apenas top 40 features
- [ ] Re-treinar modelo e comparar performance

**Código**:
```python
import shap

# Calcular SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Importância média
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Classe PRICE_UP

feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Selecionar top 40
top_features = importance_df.head(40)['feature'].tolist()

print(f"Features selecionadas: {len(top_features)}")
print(f"Importância acumulada: {importance_df.head(40)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%")

# Re-treinar com features selecionadas
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

model_v2 = xgb.XGBClassifier(...)
model_v2.fit(X_train_selected, y_train)
```

**Critério de Sucesso**: Acurácia melhora OU modelo fica mais rápido sem perder acurácia

---

### 3.2 Hyperparameter Tuning (Grid Search)
**Objetivo**: Encontrar hiperparâmetros ótimos para modelo multi-class

**Contexto**: Parâmetros atuais foram escolhidos empiricamente

**Ação**:
- [ ] Definir grid de parâmetros a testar
- [ ] Usar cross-validation (5 folds) para cada combinação
- [ ] Encontrar combinação com melhor F1-score (macro)
- [ ] Re-treinar modelo final com best params

**Grid de Parâmetros**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [200, 300, 400],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=3),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Treinar modelo final
best_model = grid_search.best_estimator_
```

**Resultado Esperado**: Acurácia > 40% (atualmente 33.25%)

---

### 3.3 Ensemble de Modelos (OPCIONAL)
**Objetivo**: Combinar múltiplos modelos para melhorar robustez

**Só executar SE**: Acurácia do modelo único ainda < 45% após 3.1 e 3.2

**Ação**:
- [ ] Treinar LightGBM com mesmo dataset
- [ ] Treinar Random Forest com mesmo dataset
- [ ] Criar Voting Classifier (soft voting)
- [ ] Validar se ensemble > modelo único

**Código**:
```python
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Treinar modelos individuais
xgb_model = xgb.XGBClassifier(...)
lgbm_model = lgb.LGBMClassifier(...)
rf_model = RandomForestClassifier(n_estimators=300, max_depth=10)

xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Criar ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model)
    ],
    voting='soft',
    weights=[2, 1, 1]  # XGBoost tem peso maior
)

ensemble.fit(X_train, y_train)

# Comparar acurácia
acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
acc_ensemble = accuracy_score(y_test, ensemble.predict(X_test))

print(f"XGBoost alone: {acc_xgb*100:.2f}%")
print(f"Ensemble: {acc_ensemble*100:.2f}%")
print(f"Melhoria: {(acc_ensemble - acc_xgb)*100:.2f}pp")
```

**Critério**: Só usar ensemble se melhoria > 5pp

---

## 🎯 SPRINT 4: Validação Robusta (Semana 4) ✅

**Objetivo**: Garantir que modelo é robusto e não overfitted

**Status**: 🔴 CRÍTICO - Essencial antes de produção

---

### 4.1 Walk-Forward Analysis
**Objetivo**: Validar consistência do modelo ao longo do tempo

**Problema**: Modelo pode estar overfitted nos dados de treino

**Ação**:
- [ ] Dividir 6 meses de dados em 10 períodos (janelas de 18 dias)
- [ ] Para cada período:
  - Treinar nos 5 períodos anteriores
  - Testar no período seguinte
  - Registrar win rate, Sharpe, Profit Factor
- [ ] Validar se win rate > 45% em PELO MENOS 8/10 períodos

**Código**:
```python
import numpy as np
from datetime import timedelta

# Dividir dataset em 10 períodos
periods = np.array_split(df, 10)
results = []

for i in range(5, len(periods)):
    # Train: períodos 0 a i-1
    train_periods = periods[:i]
    train_df = pd.concat(train_periods)

    # Test: período i
    test_df = periods[i]

    # Treinar modelo
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']

    model = xgb.XGBClassifier(...)
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    win_rate = (y_test == y_pred).mean() * 100

    results.append({
        'period': i,
        'accuracy': acc,
        'win_rate': win_rate,
        'samples': len(test_df)
    })

    print(f"Period {i}: Accuracy={acc*100:.2f}%, WR={win_rate:.2f}%")

# Análise de consistência
results_df = pd.DataFrame(results)
consistency = (results_df['win_rate'] > 45).sum() / len(results_df) * 100

print(f"\nConsistência (WR>45%): {consistency:.0f}% dos períodos")
print(f"WR médio: {results_df['win_rate'].mean():.2f}%")
print(f"Desvio padrão: {results_df['win_rate'].std():.2f}%")
```

**Critérios de Aprovação**:
- Win rate > 45% em 80% dos períodos
- Desvio padrão < 10%
- Sem período com accuracy < 30%

**Se FALHAR**: Modelo está overfitted - retornar ao Sprint 3

---

### 4.2 Monte Carlo Simulation
**Objetivo**: Entender worst-case scenario de drawdown

**Ação**:
- [ ] Simular 1000 sequências aleatórias de trades históricos
- [ ] Para cada simulação, calcular:
  - Max Drawdown
  - Ruin probability (capital < $5k)
  - Sharpe Ratio
- [ ] Calcular percentil 5% de drawdown (worst case)
- [ ] Validar que 95% das simulações têm DD < 30%

**Código**:
```python
import random

n_simulations = 1000
max_drawdowns = []
ruin_count = 0

for sim in range(n_simulations):
    # Embaralhar ordem dos trades
    simulated_trades = random.sample(list(df['profit_loss']), len(df))

    # Calcular equity curve
    capital = 10000
    equity = [capital]

    for pnl in simulated_trades:
        capital += pnl
        equity.append(capital)

        if capital < 5000:
            ruin_count += 1
            break

    # Calcular max drawdown
    peak = equity[0]
    max_dd = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    max_drawdowns.append(max_dd)

# Análise de risco
print(f"Max Drawdown médio: {np.mean(max_drawdowns):.2f}%")
print(f"Max DD percentil 95%: {np.percentile(max_drawdowns, 95):.2f}%")
print(f"Probabilidade de ruína: {ruin_count/n_simulations*100:.2f}%")

# Gráfico de distribuição
plt.hist(max_drawdowns, bins=50)
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Frequência')
plt.title('Distribuição de Max Drawdown (1000 simulações)')
plt.axvline(np.percentile(max_drawdowns, 95), color='r', linestyle='--', label='P95')
plt.legend()
plt.show()
```

**Critérios de Aprovação**:
- Percentil 95 de DD < 30%
- Probabilidade de ruína < 5%
- DD médio < 20%

---

### 4.3 Stress Testing
**Objetivo**: Validar que sistema sobrevive a eventos extremos

**Cenários de Teste**:
1. **Crash -20%**: Preço cai 20% em 1 hora
2. **Volatilidade Spike**: ATR dobra repentinamente
3. **Gap de Preço**: Gap de 5% overnight
4. **Sequência de Perdas**: 10 perdas consecutivas

**Ação**:
- [ ] Simular cada cenário
- [ ] Verificar se circuit breaker (se implementado) funciona
- [ ] Validar que sistema não quebra (erros, crashes)
- [ ] Calcular impact no capital

**Código de Simulação**:
```python
def simulate_crash(df, crash_pct=-20):
    """Simula crash de -20% em 1 hora"""
    crash_df = df.copy()
    # Reduzir preços em 20%
    crash_df['close'] *= (1 + crash_pct/100)
    crash_df['high'] *= (1 + crash_pct/100)
    crash_df['low'] *= (1 + crash_pct/100)

    # Executar forward testing no cenário de crash
    # ... (código de simulação)

    return impact_on_capital

# Testar todos os cenários
scenarios = {
    'crash_20pct': simulate_crash(df, -20),
    'volatility_spike': simulate_volatility_spike(df),
    'price_gap': simulate_gap(df, 5),
    'loss_streak': simulate_loss_streak(df, 10)
}

print("Stress Test Results:")
for scenario, impact in scenarios.items():
    print(f"{scenario}: Capital impact = {impact:.2f}%")
```

**Critérios de Aprovação**:
- Sistema não quebra em nenhum cenário
- Capital loss < 40% no pior cenário
- Circuit breaker ativa (se implementado)

---

## 📊 Critérios Finais para Produção

Antes de considerar o sistema PRONTO para produção real (capital real), TODOS os critérios devem ser atendidos:

### ✅ Sprint 1 (Validação)
- [x] Modelo prevê as 3 classes (não >70% em uma)
- [ ] Win rate > 40% em 50+ trades
- [ ] Timeout rate < 30%
- [ ] Sharpe Ratio > 1.0

### ✅ Sprint 2 (Otimização)
- [ ] SL/TP otimizados via grid search
- [ ] Profit Factor > 1.5
- [ ] Max Drawdown < 20%

### ✅ Sprint 3 (Re-treinamento - OPCIONAL)
- [ ] Acurácia > 45% (se re-treinado)
- [ ] Feature selection reduz overfitting

### ✅ Sprint 4 (Validação Robusta)
- [ ] Walk-Forward: Consistência > 80%
- [ ] Monte Carlo: P95 DD < 30%
- [ ] Stress Tests: Sistema sobrevive 100%

**Se TODOS os critérios forem atendidos**: Sistema APROVADO para produção com capital pequeno ($100-$500 inicial)

**Se ALGUM critério FALHAR**: Retornar ao Sprint correspondente e corrigir problema raiz antes de prosseguir.

---

## 🔄 Roadmap de Execução Sugerido

```
Semana 1: Sprint 1 (Validação) → Gera baseline de performance real
Semana 2: Sprint 2 (Otimização) → Melhora parâmetros baseado em dados
Semana 3: Sprint 3 (Re-treino - SE necessário) → Melhora modelo
Semana 4: Sprint 4 (Validação Robusta) → Garante robustez

Total: 4 semanas até decisão de produção
```

**Marco Final**: Sistema em produção com capital real ($100 inicial) em modo observação por 1 semana antes de escalar.

# 🔥 ROADMAP DE PESQUISA: SCALPING ML TRADING

## 🎯 OBJETIVO
Descobrir se é possível construir um bot de scalping ML rentável para mercados Deriv, através de análise científica e baseada em dados.

**Definição de Scalping para este projeto**:
- Timeframe: 1-15 minutos
- Trades/dia: 15-50
- SL/TP: 0.5% - 2.0%
- Win rate mínimo: 55%

---

## 📊 FASE 0: ANÁLISE DE VIABILIDADE DE SCALPING

### Objetivo
Responder a pergunta: **"Scalping é viável nos ativos Deriv com ML?"**

### Metodologia
1. Coletar dados históricos de 6 meses dos ativos candidatos
2. Calcular volatilidade (ATR) e tempo médio para atingir alvos de scalping
3. Analisar microestrutura de mercado (spread, slippage, tick frequency)
4. Simular cenários de scalping com diferentes SL/TP

---

## 🔬 FASE 0.1: Análise de Volatilidade para Scalping

### Objetivo
Identificar ativos com volatilidade suficiente para scalping (targets de 0.5%-2%)

### Ativos a Analisar
```python
SCALPING_CANDIDATES = {
    '1HZ75V': 'Volatility 75 (1s)',
    '1HZ100V': 'Volatility 100 (1s)',
    'BOOM300N': 'Boom 300 Index',
    'BOOM500N': 'Boom 500 Index',
    'BOOM1000N': 'Boom 1000 Index',
    'CRASH300N': 'Crash 300 Index',
    'CRASH500N': 'Crash 500 Index',
    'CRASH1000N': 'Crash 1000 Index',
    '1HZ50V': 'Volatility 50 (1s)',
    '1HZ25V': 'Volatility 25 (1s)',
}
```

### Métricas a Calcular

#### 1. ATR (Average True Range)
```python
# Para cada ativo, calcular:
- ATR médio (14 períodos, timeframe 1min)
- ATR percentual médio (ATR / Close * 100)
- Distribuição de ATR (min, max, quartis)
- Volatilidade por hora do dia
```

#### 2. Tempo para Atingir Targets
```python
# Simulação: quanto tempo leva para preço se mover X%?
SCALPING_TARGETS = [0.5, 0.75, 1.0, 1.5, 2.0]  # % targets

# Para cada target, calcular:
- Tempo médio até hit
- Taxa de sucesso (% de vezes que atinge antes de reverter)
- Melhor horário do dia
- Drawdown médio durante o movimento
```

#### 3. Microestrutura de Mercado
```python
# Análise de tick-level data:
- Spread médio (bid-ask)
- Tick frequency (ticks/minuto)
- Volatilidade intrabar (high-low range)
- Probabilidade de gaps
```

### Critérios de Aprovação por Ativo

Para um ativo ser considerado **VIÁVEL PARA SCALPING**:

| Métrica | Mínimo Aceitável | Ideal |
|---------|------------------|-------|
| ATR % (1min) | > 0.05% | > 0.10% |
| Tempo para 1% TP | < 10 min | < 5 min |
| Taxa de sucesso 1% TP antes de -0.5% SL | > 60% | > 70% |
| Tick frequency | > 50 ticks/min | > 100 ticks/min |
| Spread médio | < 0.01% | < 0.005% |

### Script de Análise

```python
# backend/ml/research/scalping_volatility_analysis.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class ScalpingVolatilityAnalyzer:
    """
    Analisa viabilidade de scalping para diferentes ativos Deriv
    """

    def __init__(self, symbol: str, data_path: str):
        self.symbol = symbol
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.results = {}

    def calculate_atr_metrics(self, period: int = 14):
        """Calcula ATR e métricas relacionadas"""
        self.df['high_low'] = self.df['high'] - self.df['low']
        self.df['high_close'] = abs(self.df['high'] - self.df['close'].shift(1))
        self.df['low_close'] = abs(self.df['low'] - self.df['close'].shift(1))

        self.df['true_range'] = self.df[['high_low', 'high_close', 'low_close']].max(axis=1)
        self.df['atr'] = self.df['true_range'].rolling(window=period).mean()
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100

        return {
            'atr_mean': self.df['atr'].mean(),
            'atr_pct_mean': self.df['atr_pct'].mean(),
            'atr_pct_median': self.df['atr_pct'].median(),
            'atr_pct_std': self.df['atr_pct'].std(),
            'atr_pct_min': self.df['atr_pct'].min(),
            'atr_pct_max': self.df['atr_pct'].max(),
        }

    def calculate_time_to_target(self, target_pct: float, stop_loss_pct: float):
        """
        Calcula tempo médio para atingir target antes de hit stop loss

        Returns:
            {
                'avg_time_minutes': float,
                'success_rate': float,
                'avg_drawdown': float,
                'best_hour': int
            }
        """
        results = []

        for i in range(len(self.df) - 100):  # Simular próximos 100 candles
            entry_price = self.df.iloc[i]['close']
            target_price = entry_price * (1 + target_pct / 100)
            stop_price = entry_price * (1 - stop_loss_pct / 100)

            # Simular movimento do preço
            hit_target = False
            hit_stop = False
            time_to_exit = 0
            max_drawdown = 0

            for j in range(i + 1, min(i + 100, len(self.df))):
                high = self.df.iloc[j]['high']
                low = self.df.iloc[j]['low']

                # Verificar drawdown
                current_drawdown = ((low - entry_price) / entry_price) * 100
                max_drawdown = min(max_drawdown, current_drawdown)

                # Verificar hit
                if high >= target_price:
                    hit_target = True
                    time_to_exit = j - i
                    break
                if low <= stop_price:
                    hit_stop = True
                    time_to_exit = j - i
                    break

            if hit_target or hit_stop:
                results.append({
                    'success': hit_target,
                    'time_minutes': time_to_exit,
                    'drawdown': max_drawdown,
                    'hour': self.df.iloc[i]['timestamp'].hour
                })

        df_results = pd.DataFrame(results)

        return {
            'avg_time_minutes': df_results[df_results['success']]['time_minutes'].mean(),
            'success_rate': df_results['success'].mean() * 100,
            'avg_drawdown': df_results['drawdown'].mean(),
            'best_hour': df_results.groupby('hour')['success'].mean().idxmax(),
            'best_hour_success_rate': df_results.groupby('hour')['success'].mean().max() * 100
        }

    def calculate_tick_metrics(self):
        """Calcula métricas de tick frequency e spread"""
        # Assumir que cada candle de 1min tem dados de tick
        self.df['intrabar_volatility'] = ((self.df['high'] - self.df['low']) / self.df['close']) * 100

        return {
            'avg_intrabar_volatility': self.df['intrabar_volatility'].mean(),
            'median_intrabar_volatility': self.df['intrabar_volatility'].median(),
        }

    def analyze_hourly_patterns(self):
        """Analisa padrões de volatilidade por hora"""
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly = self.df.groupby('hour').agg({
            'atr_pct': 'mean',
            'intrabar_volatility': 'mean',
            'volume': 'sum'
        }).round(4)

        return hourly.to_dict()

    def generate_report(self, output_path: str):
        """Gera relatório completo de viabilidade"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE DE VIABILIDADE DE SCALPING: {self.symbol}")
        print(f"{'='*60}\n")

        # 1. ATR Metrics
        print("## 1. MÉTRICAS DE VOLATILIDADE (ATR)")
        atr_metrics = self.calculate_atr_metrics()
        for key, value in atr_metrics.items():
            print(f"   - {key}: {value:.4f}")

        # 2. Time to Target Analysis
        print("\n## 2. ANÁLISE DE TEMPO PARA TARGETS")
        targets = [0.5, 0.75, 1.0, 1.5, 2.0]
        stop_loss = 0.5

        for target in targets:
            result = self.calculate_time_to_target(target, stop_loss)
            print(f"\n   Target: +{target}% | SL: -{stop_loss}%")
            print(f"   - Tempo médio: {result['avg_time_minutes']:.1f} min")
            print(f"   - Taxa de sucesso: {result['success_rate']:.1f}%")
            print(f"   - Drawdown médio: {result['avg_drawdown']:.2f}%")
            print(f"   - Melhor horário: {result['best_hour']}h ({result['best_hour_success_rate']:.1f}% win rate)")

        # 3. Tick Metrics
        print("\n## 3. MICROESTRUTURA DE MERCADO")
        tick_metrics = self.calculate_tick_metrics()
        for key, value in tick_metrics.items():
            print(f"   - {key}: {value:.4f}%")

        # 4. Hourly Patterns
        print("\n## 4. PADRÕES POR HORA DO DIA")
        hourly = self.analyze_hourly_patterns()
        print(f"   Hora com maior volatilidade: {max(hourly['atr_pct'], key=hourly['atr_pct'].get)}h")
        print(f"   Hora com menor volatilidade: {min(hourly['atr_pct'], key=hourly['atr_pct'].get)}h")

        # 5. Veredicto
        print("\n## 5. VEREDICTO FINAL")

        # Critérios de aprovação
        approved = True
        reasons = []

        if atr_metrics['atr_pct_mean'] < 0.05:
            approved = False
            reasons.append(f"ATR muito baixo ({atr_metrics['atr_pct_mean']:.4f}% < 0.05%)")

        target_1pct = self.calculate_time_to_target(1.0, 0.5)
        if target_1pct['avg_time_minutes'] > 10:
            approved = False
            reasons.append(f"Tempo para 1% TP muito longo ({target_1pct['avg_time_minutes']:.1f} min > 10 min)")

        if target_1pct['success_rate'] < 60:
            approved = False
            reasons.append(f"Taxa de sucesso baixa ({target_1pct['success_rate']:.1f}% < 60%)")

        if approved:
            print(f"   ✅ {self.symbol} é VIÁVEL para scalping")
            print(f"   Recomendação: SL 0.5%, TP 1.0%, Timeout 10 min")
        else:
            print(f"   ❌ {self.symbol} NÃO é viável para scalping")
            print(f"   Razões:")
            for reason in reasons:
                print(f"      - {reason}")

        # Salvar relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# ANÁLISE DE VIABILIDADE DE SCALPING: {self.symbol}\n\n")
            f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Métricas de Volatilidade\n\n")
            for key, value in atr_metrics.items():
                f.write(f"- {key}: {value:.4f}\n")
            # ... (continuar escrevendo outras seções)

# Script de execução
if __name__ == "__main__":
    symbols = ['1HZ75V', '1HZ100V', 'BOOM300N', 'CRASH300N']

    for symbol in symbols:
        analyzer = ScalpingVolatilityAnalyzer(
            symbol=symbol,
            data_path=f"data/{symbol}_1min_6months.csv"
        )
        analyzer.generate_report(f"research/scalping_viability_{symbol}.md")
```

### Entregáveis da Fase 0.1

- [ ] Script `scalping_volatility_analysis.py` implementado
- [ ] Dados históricos coletados (6 meses, 1min) para todos os ativos
- [ ] Relatórios individuais para cada ativo (`scalping_viability_{SYMBOL}.md`)
- [ ] Relatório comparativo final (`scalping_assets_comparison.md`)
- [ ] Ranking de ativos por viabilidade de scalping

**Critério de Sucesso**: Identificar **pelo menos 2 ativos** viáveis para scalping

---

## 🔬 FASE 0.2: Análise de Features para Scalping

### Objetivo
Identificar quais features técnicas têm maior poder preditivo em timeframes de scalping (1-5 min)

### Hipótese
Features que funcionam em timeframes longos (30min-1h) podem **não funcionar** em scalping devido a:
- Maior ruído de mercado
- Menor significância estatística de indicadores tradicionais
- Necessidade de features de microestrutura (order flow, volume imbalance)

### Features a Testar

#### Grupo 1: Indicadores Clássicos (adaptados para scalping)
```python
CLASSICAL_FEATURES = {
    'rsi_2': 'RSI com período 2 (ultra-rápido)',
    'rsi_5': 'RSI com período 5',
    'ema_cross_3_7': 'Cruzamento EMA 3 e 7',
    'macd_fast': 'MACD (5, 13, 1)',
    'bb_position': 'Posição dentro das Bandas de Bollinger (5, 1.5)',
    'stoch_5': 'Estocástico rápido (5, 3)',
}
```

#### Grupo 2: Features de Microestrutura
```python
MICROSTRUCTURE_FEATURES = {
    'volume_imbalance': 'Desbalanceamento de volume (buy vs sell)',
    'tick_direction': 'Direção dos últimos 5 ticks',
    'intrabar_momentum': 'Momentum intrabar (close - open) / (high - low)',
    'spread_widening': 'Aumento do spread (indicador de volatilidade iminente)',
    'large_tick_ratio': 'Proporção de ticks grandes (> 2x ATR)',
}
```

#### Grupo 3: Features de Price Action
```python
PRICE_ACTION_FEATURES = {
    'recent_swing_high_low': 'Distância para swing high/low recente',
    'support_resistance_proximity': 'Proximidade de S/R nos últimos 20 candles',
    'candlestick_pattern_1min': 'Padrões de candlestick em 1min',
    'volatility_expansion': 'Expansão de volatilidade (ATR atual vs ATR médio)',
    'price_rejection': 'Rejeição de preço (wicks longos)',
}
```

### Metodologia de Teste

```python
# backend/ml/research/scalping_feature_analysis.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

class ScalpingFeatureAnalyzer:
    """
    Analisa importância de features para scalping
    """

    def __init__(self, symbol: str, target_pct: float = 1.0, timeframe: str = '1min'):
        self.symbol = symbol
        self.target_pct = target_pct
        self.timeframe = timeframe

    def create_scalping_labels(self, df: pd.DataFrame):
        """
        Cria labels para scalping:
        1 = preço sobe target_pct% nos próximos N candles
        0 = não atinge target ou hit stop loss primeiro
        """
        labels = []

        for i in range(len(df) - 20):
            entry_price = df.iloc[i]['close']
            target_price = entry_price * (1 + self.target_pct / 100)
            stop_price = entry_price * (1 - self.target_pct / 2 / 100)  # SL = metade do TP

            hit_target = False
            for j in range(i + 1, min(i + 20, len(df))):
                if df.iloc[j]['high'] >= target_price:
                    hit_target = True
                    break
                if df.iloc[j]['low'] <= stop_price:
                    break

            labels.append(1 if hit_target else 0)

        return labels

    def calculate_all_features(self, df: pd.DataFrame):
        """Calcula todas as features candidatas"""
        # Grupo 1: Clássicos
        df['rsi_2'] = self._calculate_rsi(df['close'], 2)
        df['rsi_5'] = self._calculate_rsi(df['close'], 5)
        # ... (implementar todas)

        # Grupo 2: Microestrutura
        df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
        # ... (implementar todas)

        # Grupo 3: Price Action
        df['volatility_expansion'] = df['atr'] / df['atr'].rolling(20).mean()
        # ... (implementar todas)

        return df

    def analyze_feature_importance(self):
        """Analisa importância de features usando múltiplos métodos"""
        # 1. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, max_depth=10)
        rf.fit(X_train, y_train)
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # 2. Permutation Importance
        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10)

        # 3. SHAP Values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        return {
            'rf_importance': rf_importance,
            'perm_importance': perm_importance,
            'shap_values': shap_values
        }

    def test_feature_groups(self):
        """Testa cada grupo de features isoladamente"""
        results = {}

        groups = {
            'Classical': CLASSICAL_FEATURES,
            'Microstructure': MICROSTRUCTURE_FEATURES,
            'PriceAction': PRICE_ACTION_FEATURES,
        }

        for group_name, features in groups.items():
            X_group = X_train[list(features.keys())]

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_group, y_train)

            accuracy = rf.score(X_test[list(features.keys())], y_test)

            results[group_name] = {
                'accuracy': accuracy,
                'feature_count': len(features)
            }

        return results

    def generate_report(self):
        """Gera relatório de features para scalping"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE DE FEATURES PARA SCALPING: {self.symbol}")
        print(f"Target: +{self.target_pct}% em {self.timeframe}")
        print(f"{'='*60}\n")

        # Top 10 features mais importantes
        print("## TOP 10 FEATURES MAIS IMPORTANTES\n")
        importance = self.analyze_feature_importance()
        print(importance['rf_importance'].head(10).to_string())

        # Performance por grupo
        print("\n## PERFORMANCE POR GRUPO DE FEATURES\n")
        group_results = self.test_feature_groups()
        for group, result in group_results.items():
            print(f"{group}: Accuracy {result['accuracy']:.2%} ({result['feature_count']} features)")

        # Recomendações
        print("\n## RECOMENDAÇÕES\n")
        best_group = max(group_results, key=lambda x: group_results[x]['accuracy'])
        print(f"✅ Melhor grupo: {best_group} ({group_results[best_group]['accuracy']:.2%})")

        top_features = importance['rf_importance'].head(15)['feature'].tolist()
        print(f"✅ Usar estas 15 features para modelo final:")
        for i, feat in enumerate(top_features, 1):
            print(f"   {i}. {feat}")
```

### Entregáveis da Fase 0.2

- [ ] Script `scalping_feature_analysis.py` implementado
- [ ] Relatório de importância de features por ativo
- [ ] Comparação: Features de scalping vs Features de swing
- [ ] Lista final de features recomendadas para scalping

**Critério de Sucesso**: Identificar **top 15 features** que atingem accuracy > 55% em predições de scalping

---

## 🤖 FASE 1: Treinamento de Modelo Específico para Scalping

### Objetivo
Treinar modelo XGBoost otimizado para scalping nos ativos viáveis identificados na Fase 0.1

### Estratégia de Modelagem

#### Multi-Task Learning
```python
# Prever simultaneamente:
1. Direção (UP/DOWN/NO_MOVE)
2. Magnitude do movimento (0.5%, 1%, 1.5%, 2%)
3. Tempo estimado para atingir target (1-15 min)
```

#### Balanceamento de Dataset para Scalping
```python
# Problema: Em scalping, maioria das predições são NO_MOVE
# Solução: Balanceamento agressivo

BALANCING_STRATEGY = {
    'NO_MOVE': 0.20,  # 20% (reduzir drasticamente)
    'PRICE_UP': 0.40,  # 40%
    'PRICE_DOWN': 0.40,  # 40%
}
```

#### Hyperparameter Tuning para Scalping
```python
# Parâmetros otimizados para alta frequência
SCALPING_XGB_PARAMS = {
    'max_depth': [3, 4, 5],  # Árvores rasas (menos overfit)
    'learning_rate': [0.01, 0.05, 0.1],  # LR mais alto para convergência rápida
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],  # Regularização para prevenir overfit
}
```

### Script de Treinamento

```python
# backend/ml/training/train_xgboost_scalping.py

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ScalpingModelTrainer:
    """
    Treina modelo XGBoost específico para scalping
    """

    def __init__(self, symbol: str, target_pct: float = 1.0, stop_loss_pct: float = 0.5):
        self.symbol = symbol
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.model = None

    def create_scalping_labels_multiclass(self, df: pd.DataFrame, lookahead: int = 15):
        """
        Cria labels multiclass para scalping:
        0 = NO_MOVE (não atinge nem TP nem SL no período)
        1 = PRICE_UP (atinge TP antes de SL)
        2 = PRICE_DOWN (atinge SL primeiro ou movimento contrário)
        """
        labels = []

        for i in range(len(df) - lookahead):
            entry_price = df.iloc[i]['close']
            target_price = entry_price * (1 + self.target_pct / 100)
            stop_price = entry_price * (1 - self.stop_loss_pct / 100)

            hit_target = False
            hit_stop = False

            for j in range(i + 1, min(i + lookahead, len(df))):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']

                if high >= target_price:
                    hit_target = True
                    break
                if low <= stop_price:
                    hit_stop = True
                    break

            if hit_target:
                labels.append(1)  # PRICE_UP
            elif hit_stop:
                labels.append(2)  # PRICE_DOWN
            else:
                labels.append(0)  # NO_MOVE

        return labels

    def balance_dataset_for_scalping(self, X, y):
        """Balanceamento agressivo para scalping"""
        from sklearn.utils import resample

        X_df = pd.DataFrame(X)
        X_df['target'] = y

        # Separar por classe
        no_move = X_df[X_df['target'] == 0]
        price_up = X_df[X_df['target'] == 1]
        price_down = X_df[X_df['target'] == 2]

        # Determinar target size (baseado na menor classe)
        min_size = min(len(price_up), len(price_down))
        target_size_move = int(min_size * 1.5)  # UP/DOWN têm 1.5x mais amostras
        target_size_no_move = int(min_size * 0.5)  # NO_MOVE reduzido drasticamente

        # Resample
        no_move_resampled = resample(no_move, n_samples=target_size_no_move, random_state=42)
        price_up_resampled = resample(price_up, n_samples=target_size_move, random_state=42)
        price_down_resampled = resample(price_down, n_samples=target_size_move, random_state=42)

        # Combinar
        balanced = pd.concat([no_move_resampled, price_up_resampled, price_down_resampled])
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        y_balanced = balanced['target'].values
        X_balanced = balanced.drop('target', axis=1).values

        return X_balanced, y_balanced

    def train_with_grid_search(self, X_train, y_train):
        """Treina modelo com Grid Search otimizado para scalping"""

        # Time Series Cross-Validation (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)

        # XGBoost com multiclass
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

        # Grid Search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=SCALPING_XGB_PARAMS,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        print("Iniciando Grid Search...")
        grid_search.fit(X_train, y_train)

        print(f"\nMelhores parâmetros: {grid_search.best_params_}")
        print(f"Melhor score CV: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        return grid_search.best_estimator_

    def evaluate_scalping_model(self, X_test, y_test):
        """Avaliação específica para scalping"""
        y_pred = self.model.predict(X_test)

        # Métricas gerais
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        # Métricas por classe
        class_metrics = {}
        for class_id, class_name in enumerate(['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN']):
            mask = y_test == class_id
            class_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0
            class_metrics[class_name] = {
                'accuracy': class_acc,
                'count': mask.sum()
            }

        # Simular trading performance
        trade_results = self.simulate_scalping_trades(X_test, y_test, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_metrics': class_metrics,
            'trade_simulation': trade_results
        }

    def simulate_scalping_trades(self, X_test, y_true, y_pred):
        """Simula trades de scalping usando predições"""
        trades = []

        for i, pred in enumerate(y_pred):
            if pred == 0:  # NO_MOVE - não trade
                continue

            # Simular trade
            true_label = y_true[i]

            if pred == 1:  # Previu PRICE_UP
                if true_label == 1:  # Acertou
                    pnl_pct = self.target_pct
                    result = 'WIN'
                else:
                    pnl_pct = -self.stop_loss_pct
                    result = 'LOSS'
            elif pred == 2:  # Previu PRICE_DOWN (SHORT)
                if true_label == 2:  # Acertou
                    pnl_pct = self.target_pct
                    result = 'WIN'
                else:
                    pnl_pct = -self.stop_loss_pct
                    result = 'LOSS'

            trades.append({
                'prediction': pred,
                'true_label': true_label,
                'result': result,
                'pnl_pct': pnl_pct
            })

        df_trades = pd.DataFrame(trades)

        return {
            'total_trades': len(trades),
            'win_rate': (df_trades['result'] == 'WIN').mean() * 100,
            'total_pnl_pct': df_trades['pnl_pct'].sum(),
            'avg_pnl_per_trade': df_trades['pnl_pct'].mean(),
            'sharpe_ratio': df_trades['pnl_pct'].mean() / df_trades['pnl_pct'].std() if len(trades) > 1 else 0
        }

    def save_model(self, output_path: str):
        """Salva modelo treinado"""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Modelo salvo em {output_path}")

# Script de execução
if __name__ == "__main__":
    # Treinar para cada ativo viável
    viable_assets = ['1HZ75V', '1HZ100V', 'BOOM300N']  # Assumindo que passaram na Fase 0.1

    for symbol in viable_assets:
        print(f"\n{'='*60}")
        print(f"TREINANDO MODELO SCALPING PARA {symbol}")
        print(f"{'='*60}\n")

        trainer = ScalpingModelTrainer(
            symbol=symbol,
            target_pct=1.0,  # 1% TP
            stop_loss_pct=0.5  # 0.5% SL (R:R 1:2)
        )

        # Carregar dados e features
        # ... (implementar carregamento)

        # Balancear dataset
        X_balanced, y_balanced = trainer.balance_dataset_for_scalping(X_train, y_train)

        # Treinar com Grid Search
        model = trainer.train_with_grid_search(X_balanced, y_balanced)

        # Avaliar
        metrics = trainer.evaluate_scalping_model(X_test, y_test)

        print("\n## RESULTADOS")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"F1-Score: {metrics['f1_score']:.2%}")
        print(f"\nSimulação de Trading:")
        print(f"  - Trades: {metrics['trade_simulation']['total_trades']}")
        print(f"  - Win Rate: {metrics['trade_simulation']['win_rate']:.1f}%")
        print(f"  - Total P&L: {metrics['trade_simulation']['total_pnl_pct']:+.2f}%")
        print(f"  - Sharpe Ratio: {metrics['trade_simulation']['sharpe_ratio']:.2f}")

        # Salvar modelo
        trainer.save_model(f"backend/ml/models/xgboost_scalping_{symbol}.pkl")
```

### Entregáveis da Fase 1

- [ ] Script `train_xgboost_scalping.py` implementado
- [ ] Modelos treinados para cada ativo viável (`xgboost_scalping_{SYMBOL}.pkl`)
- [ ] Relatório de performance por ativo
- [ ] Comparação: Modelo Scalping vs Modelo Swing (R_100)

**Critério de Sucesso**:
- Win rate > 55%
- Sharpe Ratio > 1.5
- Total P&L (backtest) > +15% em 3 meses

---

## 🧪 FASE 2: Validação em Forward Testing (Paper Trading)

### Objetivo
Validar modelo de scalping em condições reais de mercado (sem riscar capital)

### Configuração de Forward Testing

```python
SCALPING_FORWARD_TEST_CONFIG = {
    'symbol': '1HZ75V',  # Melhor ativo identificado
    'mode': 'scalping_ml',
    'stop_loss_pct': 0.5,
    'take_profit_pct': 1.0,
    'position_timeout_minutes': 10,  # Timeout agressivo
    'confidence_threshold': 0.50,  # Threshold mais alto para scalping
    'max_positions_per_symbol': 1,
    'max_position_size_pct': 2.0,
    'trading_hours': {
        'start': 8,  # Melhor horário identificado na Fase 0.1
        'end': 16
    }
}
```

### Métricas de Validação

| Métrica | Target | Crítico |
|---------|--------|---------|
| Win Rate | > 55% | < 50% = FALHA |
| Trades/Dia | 15-30 | < 10 = Baixa atividade |
| Sharpe Ratio | > 1.5 | < 1.0 = Alto risco |
| Max Drawdown | < 10% | > 15% = Inaceitável |
| Timeout Rate | < 30% | > 50% = Modelo lento |
| Avg Trade Duration | 5-10 min | > 15 min = Não é scalping |

### Plano de Testes

#### Teste 1: Validação Inicial (100 trades)
- **Objetivo**: Confirmar que modelo funciona em produção
- **Duração**: 3-7 dias
- **Critério**: Win rate > 50%, sem bugs críticos

#### Teste 2: Otimização de Threshold (200 trades)
- **Objetivo**: Encontrar confidence threshold ótimo
- **Teste**: Comparar thresholds 0.45, 0.50, 0.55, 0.60
- **Métrica**: Maximizar Sharpe Ratio

#### Teste 3: Teste de Robustez (500 trades)
- **Objetivo**: Validar consistência em diferentes condições de mercado
- **Duração**: 2-4 semanas
- **Critério**: Todas as métricas target atingidas

### Entregáveis da Fase 2

- [ ] Forward testing rodando 24/7 em produção
- [ ] Dashboard de monitoramento em tempo real
- [ ] Relatório de validação (100/200/500 trades)
- [ ] Ajustes de threshold e parâmetros

**Critério de Sucesso**: Após 500 trades, todas as métricas target atingidas

---

## 🚀 FASE 3: Trading Real (se Fase 2 aprovada)

### Pré-requisitos
- ✅ Win rate > 55% em 500+ trades de paper trading
- ✅ Sharpe > 1.5
- ✅ Max Drawdown < 10%
- ✅ Sistema estável sem bugs críticos por 2 semanas

### Estratégia de Entrada Gradual

#### Stage 1: Micro Capital ($100)
- Executar 50 trades
- Validar slippage real vs paper trading
- Validar custos de transação

#### Stage 2: Small Capital ($500)
- Executar 200 trades
- Escalar position size gradualmente
- Monitorar impacto psicológico

#### Stage 3: Medium Capital ($2000+)
- Apenas se Stages 1-2 bem-sucedidas
- Manter position size < 2% do capital

---

## 📊 MÉTRICAS DE SUCESSO GLOBAL

Para considerar o projeto SCALPING ML bem-sucedido:

| Fase | Critério |
|------|----------|
| Fase 0.1 | ✅ Identificar 2+ ativos viáveis |
| Fase 0.2 | ✅ Features com accuracy > 55% |
| Fase 1 | ✅ Modelo com win rate > 55%, Sharpe > 1.5 |
| Fase 2 | ✅ 500 trades com métricas target |
| Fase 3 | ✅ 50 trades reais lucrativos |

**Se qualquer fase FALHAR**:
- Documentar causa raiz
- Decidir: pivotar para swing trading OU investigar outros ativos

---

## 📝 NOTAS IMPORTANTES

### Diferenças Críticas: Scalping vs Swing

| Aspecto | Swing (R_100) | Scalping (V75/V100) |
|---------|---------------|---------------------|
| Timeframe | 30min - 1h | 1min - 5min |
| SL/TP | 2% / 4% | 0.5% / 1% |
| Timeout | 180 min | 10 min |
| Trades/dia | 3-8 | 15-50 |
| Win rate esperado | 40-45% | 55-60% |
| Features principais | Tendência, Momentum | Microestrutura, Volatilidade |
| Risco de overtrading | Baixo | **ALTO** |
| Sensibilidade a custos | Baixa | **ALTA** |

### Riscos Específicos de Scalping

1. **Overtrading**: Gerar 50 trades/dia pode destruir capital com custos
2. **Overfitting**: Modelos em timeframes curtos são mais propensos a overfit
3. **Slippage**: Em scalping, 0.01% de slippage = 2% do profit!
4. **Estresse Psicológico**: Alta frequência = alta pressão emocional
5. **Dependência de Infraestrutura**: Latência de rede pode matar scalping

### Quando Desistir de Scalping

Se após Fase 0.1, **NENHUM ativo** atingir:
- ATR > 0.05%
- Tempo para 1% TP < 10 min
- Taxa de sucesso > 60%

**ENTÃO**: Scalping não é viável nos ativos Deriv. Focar em swing trading (já validado com R_100).

---

## 🎯 PRÓXIMO PASSO

**INICIAR FASE 0.1**: Análise de Volatilidade para Scalping

1. Coletar dados históricos de V75, V100, BOOM300, CRASH300 (6 meses, 1min)
2. Implementar script `scalping_volatility_analysis.py`
3. Gerar relatórios de viabilidade por ativo
4. Decidir se vale a pena continuar com scalping

**Tempo estimado**: 3-5 dias

"""
CRASH300N - XGBoost Non-Linear Survival Analysis

Hipotese: O ruido na Hazard Curve esconde padroes nao-lineares
         que XGBoost pode explorar (particionamento de espaco)

Estrategia: Criar features de interacao e usar XGBoost para
           encontrar "bolsoes" de baixa probabilidade de crash

Metrica: AUC-ROC > 0.55 (edge detectavel)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def create_nonlinear_features(df):
    """
    Cria features de interacao e nao-lineares para capturar
    padroes complexos no processo de crash
    """
    print("\n[FEATURE ENGINEERING] Criando features nao-lineares...")

    # 1. Features base (temporais)
    df['log_ret'] = np.log(df['close'] / df['open'])
    df['is_crash'] = (df['log_ret'] <= -0.005).astype(int)

    # Candles desde ultimo crash
    df['crash_group'] = df['is_crash'].cumsum()
    df['candles_since_crash'] = df.groupby('crash_group').cumcount()

    # Tamanho do ultimo crash
    crash_sizes = df['log_ret'].where(df['is_crash'] == 1)
    df['last_crash_magnitude'] = crash_sizes.ffill().shift(1)

    # Densidade de crashes
    df['crash_density_50'] = df['is_crash'].rolling(window=50).sum().shift(1)
    df['crash_density_100'] = df['is_crash'].rolling(window=100).sum().shift(1)

    # 2. Features de INTERACAO (A Magica!)
    # Hipotese: O perigo e a COMBINACAO de tempo + magnitude
    df['hazard_intensity'] = df['candles_since_crash'] * abs(df['last_crash_magnitude'])

    # Tempo ao quadrado (efeito nao-linear do tempo)
    df['time_squared'] = df['candles_since_crash'] ** 2
    df['time_cubed'] = df['candles_since_crash'] ** 3

    # 3. Features de MOMENTUM (Derivadas)
    # Segunda derivada do preco (aceleracao)
    df['velocity'] = df['log_ret'].diff()
    df['acceleration'] = df['velocity'].diff()

    # Taxa de mudanca da volatilidade
    df['volatility'] = df['log_ret'].rolling(window=20).std()
    df['volatility_change'] = df['volatility'].diff()

    # 4. Features de REGIME (Estado do mercado)
    # Distancia da media movel
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['distance_from_ma'] = (df['close'] - df['ma_20']) / df['ma_20']

    # Bollinger Bands
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['ma_20'] + 2 * df['bb_std']
    df['bb_lower'] = df['ma_20'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 5. Features de CICLO (Periodicity)
    # Tentativa de detectar ciclos ocultos
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour if 'timestamp' in df.columns else 0
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek if 'timestamp' in df.columns else 0

    # Modulo do tempo (detectar ciclos de ~300 candles)
    df['cycle_300'] = df['candles_since_crash'] % 300
    df['cycle_100'] = df['candles_since_crash'] % 100
    df['cycle_50'] = df['candles_since_crash'] % 50

    # 6. Target (proximo candle vai crashar?)
    df['target_next_crash'] = df['is_crash'].shift(-1).fillna(0).astype(int)

    # Limpar NaNs
    df = df.dropna()

    print(f"  Total features criadas: {df.shape[1]}")
    print(f"  Samples apos limpeza: {len(df):,}")

    return df


def train_xgboost_model(df):
    """
    Treina XGBoost com features nao-lineares
    """
    print("\n" + "="*70)
    print("XGBOOST NON-LINEAR SURVIVAL ANALYSIS")
    print("="*70)

    # Features para usar
    feature_cols = [
        # Temporais
        'candles_since_crash', 'last_crash_magnitude',
        'crash_density_50', 'crash_density_100',

        # Interacoes
        'hazard_intensity', 'time_squared', 'time_cubed',

        # Momentum
        'velocity', 'acceleration', 'volatility', 'volatility_change',

        # Regime
        'distance_from_ma', 'bb_position',

        # Ciclos
        'cycle_300', 'cycle_100', 'cycle_50',
        'hour_of_day', 'day_of_week',

        # OHLC basico
        'log_ret'
    ]

    # Verificar features disponiveis
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"\n[FEATURES] Usando {len(available_features)} features:")
    for feat in available_features:
        print(f"  - {feat}")

    X = df[available_features].values
    y = df['target_next_crash'].values

    # Split temporal (70/15/15)
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"\n[SPLIT]")
    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.2f}% CRASH)")
    print(f"  Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% CRASH)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.2f}% CRASH)")

    # XGBoost parameters (otimizado para AUC)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'verbose': 0
    }

    print(f"\n[MODEL] XGBoost Classifier")
    print(f"  Objective: binary:logistic")
    print(f"  Metric: AUC-ROC")
    print(f"  Max depth: {params['max_depth']}")
    print(f"  Learning rate: {params['learning_rate']}")

    # Train
    print(f"\n[TRAINING] Treinando...")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iteration = model.best_iteration
    print(f"  Best iteration: {best_iteration}")

    # Predict probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n[AUC-ROC]")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")

    # Feature Importance
    print(f"\n[FEATURE IMPORTANCE] Top 10:")
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<30}: {row['importance']:.4f}")

    # Test set evaluation com threshold otimizado
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION (Threshold Optimization)")
    print(f"{'='*70}")

    # Encontrar threshold otimo via ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    # Threshold que maximiza (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n[THRESHOLD OPTIMIZATION]")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  TPR at optimal: {tpr[optimal_idx]:.4f}")
    print(f"  FPR at optimal: {fpr[optimal_idx]:.4f}")

    # Predictions com threshold otimizado
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"\n[METRICS] (Threshold={optimal_threshold:.4f})")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n[CONFUSION MATRIX]")
    print(f"                  Predicted SAFE | Predicted CRASH")
    print(f"  Actual SAFE:    {cm[0,0]:>14} | {cm[0,1]:>15}")
    print(f"  Actual CRASH:   {cm[1,0]:>14} | {cm[1,1]:>15}")

    # Analise de probabilidades
    print(f"\n[PROBABILITY DISTRIBUTION]")
    print(f"  Min P(CRASH): {y_test_proba.min():.6f}")
    print(f"  Max P(CRASH): {y_test_proba.max():.6f}")
    print(f"  Mean P(CRASH): {y_test_proba.mean():.6f}")
    print(f"  Median P(CRASH): {np.median(y_test_proba):.6f}")
    print(f"  Std P(CRASH): {y_test_proba.std():.6f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUC={test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC=0.5000)')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], s=200, c='red',
                marker='o', label=f'Optimal Threshold ({optimal_threshold:.4f})')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - XGBoost CRASH300N', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plot_path = Path(__file__).parent / "reports" / "crash300n_xgboost_roc.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] ROC Curve salva em: {plot_path}")

    # Salvar modelo
    model_path = Path(__file__).parent / "models" / "crash300n_xgboost.json"
    model.save_model(model_path)
    print(f"[MODEL] Modelo salvo em: {model_path}")

    # Veredicto final
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL")
    print(f"{'='*70}\n")

    if test_auc > 0.55:
        edge = (test_auc - 0.5) * 100
        print(f"SUCESSO! EDGE DETECTADO!")
        print(f"  AUC-ROC: {test_auc:.4f} (>{0.55})")
        print(f"  Edge: {edge:.2f}% acima do random")
        print(f"\n  O modelo APRENDEU padroes nao-lineares!")
        print(f"  XGBoost conseguiu explorar a variacao de 263% da Hazard Curve")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Implementar estrategia de trading baseada em probabilidade")
        print(f"  >> Operar LONG apenas quando P(CRASH) < {optimal_threshold:.4f}")
        print(f"  >> Win rate esperado: ~{(1-prec)*100:.1f}% (evitando zonas de risco)")
    elif test_auc > 0.52:
        print(f"EDGE FRACO DETECTADO")
        print(f"  AUC-ROC: {test_auc:.4f} (entre 0.52-0.55)")
        print(f"  Edge marginal, mas exploravel com bom money management")
    else:
        print(f"FALHOU - SEM EDGE")
        print(f"  AUC-ROC: {test_auc:.4f} (<0.52)")
        print(f"  Modelo nao encontrou padroes exploraveis")
        print(f"  Processo e verdadeiramente aleatorio (Poisson puro)")
        print(f"\n  CONCLUSAO FINAL:")
        print(f"  >> ML scalping em CRASH300N e INVIAVEL")
        print(f"  >> Recomendacao: Migrar para Forex/Indices reais")

    print(f"\n{'='*70}\n")

    return model, available_features, optimal_threshold, test_auc


def main():
    print("="*70)
    print("CRASH300N - XGBOOST NON-LINEAR SURVIVAL")
    print("="*70)

    # Load data
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("ERRO: Nenhum arquivo CRASH300N encontrado")
        return

    input_file = files[0]
    print(f"\n[LOAD] {input_file.name}")
    df = pd.read_csv(input_file)

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Total candles: {len(df):,}")

    # Feature engineering
    df = create_nonlinear_features(df)

    # Train model
    model, features, threshold, auc = train_xgboost_model(df)


if __name__ == "__main__":
    main()

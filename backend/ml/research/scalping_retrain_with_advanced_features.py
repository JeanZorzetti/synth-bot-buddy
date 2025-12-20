"""
Retreinamento do Modelo XGBoost com Features Avançadas

CONTEXTO:
- Baseline: 50.9% win rate (62 features básicas)
- Experimento A: 51.2% win rate (TP/SL 0.3%/0.15%)
- Experimentos falharam em atingir 60% win rate

SOLUÇÃO:
Retreinar com 88 features (62 básicas + 26 avançadas)

EXPECTATIVA:
Win rate 51.2% -> 60-65%

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

# Import das classes de treinamento
from scalping_model_training import ScalpingModelTrainer
from scalping_labeling import ScalpingLabeler


def main():
    """
    Pipeline de retreinamento com features avançadas
    """
    print("="*70)
    print("RETREINAMENTO COM FEATURES AVANCADAS - V100 M5")
    print("="*70)

    data_dir = Path(__file__).parent / "data"
    models_dir = Path(__file__).parent / "models"
    reports_dir = Path(__file__).parent / "reports"

    # 1. Carregar dataset com features avançadas
    advanced_features_path = data_dir / "1HZ100V_5min_180days_features_advanced.csv"

    if not advanced_features_path.exists():
        print(f"\n[ERRO] Dataset com features avancadas nao encontrado!")
        print(f"Execute primeiro: python scalping_advanced_features.py")
        return

    print(f"\n[LOAD] Carregando dataset: {advanced_features_path.name}")
    df_features = pd.read_csv(advanced_features_path)
    print(f"   Dataset: {len(df_features)} candles, {len(df_features.columns)} colunas")

    # 2. Gerar labels (TP 0.2% / SL 0.1% - baseline)
    print("\n[LABEL] Gerando labels LONG/SHORT/NO_TRADE...")
    print("   Configuracao: TP 0.2% / SL 0.1% / R:R 1:2")

    labeler = ScalpingLabeler(
        df_features,
        tp_pct=0.2,
        sl_pct=0.1,
        max_candles=20
    )
    df_labeled = labeler.generate_labels()

    # Estatísticas de labels
    label_dist = df_labeled['label'].value_counts()
    print(f"\n   Distribuicao de labels:")
    print(f"      NO_TRADE (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df_labeled)*100:.1f}%)")
    print(f"      LONG (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df_labeled)*100:.1f}%)")
    print(f"      SHORT (2): {label_dist.get(2, 0)} ({label_dist.get(2, 0)/len(df_labeled)*100:.1f}%)")

    # 3. Treinar modelo com Optuna (50 trials)
    print("\n[TRAIN] Iniciando treinamento com Optuna...")
    print("   Modelo: XGBoost")
    print("   Optuna trials: 50")
    print("   Split: 60% train / 20% val / 20% test")

    trainer = ScalpingModelTrainer(df_labeled, test_size=0.2, val_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()

    print(f"\n   Train set: {len(X_train)} samples")
    print(f"   Val set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Total features: {X_train.shape[1]}")

    # Otimizar hiperparâmetros
    print("\n[OPTUNA] Otimizando hiperparametros...")
    start_time = time.time()

    best_params = trainer.optimize_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=50
    )

    elapsed_time = time.time() - start_time
    print(f"\n   Tempo de otimizacao: {elapsed_time/60:.1f} min")
    print(f"   Melhores hiperparametros:")
    for key, value in best_params.items():
        if key not in ['objective', 'num_class', 'random_state', 'tree_method']:
            print(f"      {key}: {value}")

    # 4. Treinar modelo final
    print("\n[FINAL] Treinando modelo final...")
    model = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)

    # 5. Avaliar em Test Set
    print("\n[EVAL] Avaliando em Test Set...")
    metrics = trainer.evaluate_model(model, X_test, y_test, split_name='Test')

    # 6. Análise de Feature Importance
    print("\n[IMPORTANCE] Analisando feature importance...")
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # Separar features básicas vs avançadas
    advanced_features_names = [
        'buy_candles_count', 'sell_candles_count', 'order_flow_imbalance',
        'imbalance_ma_10', 'imbalance_strength',
        'aggressive_buy', 'aggressive_sell', 'aggr_buy_ma_5',
        'aggr_sell_ma_5', 'aggr_net', 'aggr_streak',
        'range_ma_20', 'range_std_20', 'range_zscore',
        'is_high_range', 'is_low_range', 'range_spike',
        'hrn_distance', 'lrn_distance',
        'delta_body', 'cumulative_delta_20', 'delta_trend', 'delta_strength',
        'consolidation_ratio', 'is_consolidation', 'consolidation_direction'
    ]

    advanced_in_top10 = sum(1 for feat in feature_importance.head(10).index if feat in advanced_features_names)
    print(f"\n   Features avancadas no TOP 10: {advanced_in_top10}/10")

    # 7. Comparar com baseline
    print("\n" + "="*70)
    print("COMPARACAO COM BASELINE")
    print("="*70)

    baseline_win_rate = 0.509  # Baseline do treinamento inicial
    exp_a_win_rate = 0.512     # Melhor experimento (A)
    current_win_rate = metrics['win_rate']

    improvement_vs_baseline = (current_win_rate - baseline_win_rate) * 100
    improvement_vs_exp_a = (current_win_rate - exp_a_win_rate) * 100

    print(f"\nBaseline (62 features basicas):")
    print(f"   Win rate: {baseline_win_rate*100:.1f}%")
    print(f"\nExperimento A (TP/SL 0.3/0.15):")
    print(f"   Win rate: {exp_a_win_rate*100:.1f}%")
    print(f"\nModelo Atual (88 features - 62 basicas + 26 avancadas):")
    print(f"   Win rate: {current_win_rate*100:.1f}%")
    print(f"   Melhoria vs baseline: {improvement_vs_baseline:+.1f}pp")
    print(f"   Melhoria vs Exp A: {improvement_vs_exp_a:+.1f}pp")

    # 8. Verificar se atingiu meta
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)

    meta_atingida = current_win_rate >= 0.60

    if meta_atingida:
        print(f"\n[SUCESSO] META DE 60% ATINGIDA!")
        print(f"   Win rate: {current_win_rate*100:.1f}%")
        print(f"   Status: PRONTO PARA BACKTESTING")
        print(f"\nProximo passo: Backtesting completo (3 meses out-of-sample)")
    else:
        gap_to_goal = (0.60 - current_win_rate) * 100
        print(f"\n[AVISO] Meta nao atingida")
        print(f"   Win rate: {current_win_rate*100:.1f}%")
        print(f"   Faltam: {gap_to_goal:.1f}pp para 60%")

        if current_win_rate >= 0.55:
            print(f"\n   Status: PARCIAL (55-60%)")
            print(f"   Proximos passos:")
            print(f"   1. Testar em backtesting mesmo assim")
            print(f"   2. Se backtest OK, considerar forward testing")
            print(f"   3. Ou adicionar mais features (order book depth, etc)")
        elif current_win_rate >= 0.52:
            print(f"\n   Status: MELHORIA MARGINAL (52-55%)")
            print(f"   Proximos passos:")
            print(f"   1. Testar M15/M30 (timeframes mais estaveis)")
            print(f"   2. Testar BOOM/CRASH (padroes mais claros)")
        else:
            print(f"\n   Status: SEM MELHORIA SIGNIFICATIVA (<52%)")
            print(f"   Proximos passos:")
            print(f"   1. Reavaliar estrategia (mean reversion, grid trading)")
            print(f"   2. Considerar que scalping pode nao ser viavel em V100 M5")

    # 9. Salvar modelo e resultados
    print("\n[SAVE] Salvando modelo e resultados...")

    # Salvar modelo
    model_path = models_dir / 'scalping_xgboost_advanced_features.pkl'
    import joblib
    joblib.dump(model, model_path)
    print(f"   Modelo salvo: {model_path}")

    # Salvar resultados
    results = {
        'experiment': 'retrain_with_advanced_features',
        'dataset': '1HZ100V_5min_180days_features_advanced.csv',
        'total_features': X_train.shape[1],
        'basic_features': 62,
        'advanced_features': 26,
        'tp_pct': 0.2,
        'sl_pct': 0.1,
        'n_trials': 50,
        'win_rate': float(metrics['win_rate']),
        'f1_score': float(metrics['f1_tradeable']),
        'accuracy': float(metrics['accuracy_tradeable']),
        'improvement_vs_baseline': float(improvement_vs_baseline),
        'improvement_vs_exp_a': float(improvement_vs_exp_a),
        'goal_achieved': meta_atingida,
        'best_params': best_params,
        'top_10_features': feature_importance.head(10).index.tolist(),
        'advanced_in_top10': advanced_in_top10,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    results_path = reports_dir / 'retrain_advanced_features_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Resultados salvos: {results_path}")

    print("\n" + "="*70)
    print("RETREINAMENTO CONCLUIDO!")
    print("="*70)


if __name__ == "__main__":
    main()

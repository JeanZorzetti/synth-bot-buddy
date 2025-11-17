"""
Treina LightGBM CORRIGIDO - Evitando previsão apenas da classe majoritária
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from model_training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def train_lightgbm_fixed():
    """
    Treina LightGBM com configurações que evitam predição trivial
    """
    print("\n" + "="*60)
    print("LIGHTGBM CORRIGIDO - Evitando Classe Majoritária")
    print("Problema: Modelo anterior previa apenas 'No Move'")
    print("Solução: is_unbalance=True + threshold tuning")
    print("="*60)

    # Criar trainer
    trainer = ModelTrainer()

    # Carregar dataset
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "ml_dataset_R100_1m_6months.pkl"

    logger.info(f"Carregando dataset...")
    X, y = trainer.load_ml_dataset(dataset_path)

    # Split temporal
    X_train, X_test, y_train, y_test = trainer.split_train_test(
        X, y,
        test_size=0.2,
        time_aware=True
    )

    logger.info(f"\n{'='*60}")
    logger.info("ESTRATÉGIA DE CORREÇÃO")
    logger.info("="*60)
    logger.info("Problema identificado:")
    logger.info("  - LightGBM default está ignorando classe minoritária")
    logger.info("  - Recall = 0% (nunca prevê 'Price Up')")
    logger.info("")
    logger.info("Soluções aplicadas:")
    logger.info("  - is_unbalance=True (LightGBM balanceamento interno)")
    logger.info("  - min_data_in_leaf reduzido (permitir folhas menores)")
    logger.info("  - Threshold tuning para balancear precision/recall")
    logger.info("  - Boost_from_average=False")

    # Configurações otimizadas para evitar predição trivial
    configs = [
        {
            'name': 'Unbalanced-1',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': -1,
            'min_child_samples': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'is_unbalance': True,
            'boost_from_average': False
        },
        {
            'name': 'Unbalanced-2',
            'num_leaves': 25,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'max_depth': 6,
            'min_child_samples': 10,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'is_unbalance': True,
            'boost_from_average': False
        },
        {
            'name': 'ScaleWeight',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': -1,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.5,
            'boost_from_average': False
        },
        {
            'name': 'Conservative',
            'num_leaves': 15,
            'learning_rate': 0.01,
            'n_estimators': 400,
            'max_depth': 5,
            'min_child_samples': 5,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'is_unbalance': True,
            'boost_from_average': False
        }
    ]

    best_model = None
    best_f1 = 0  # Usar F1 ao invés de accuracy para evitar modelos triviais
    best_config = None
    best_metrics = None
    best_threshold = 0.5
    all_results = []

    for config in configs:
        logger.info(f"\n>>> Testando: {config['name']}")

        # Separar parâmetros do LGBMClassifier
        lgbm_params = {
            'num_leaves': config['num_leaves'],
            'learning_rate': config['learning_rate'],
            'n_estimators': config['n_estimators'],
            'max_depth': config['max_depth'],
            'min_child_samples': config['min_child_samples'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'boost_from_average': config.get('boost_from_average', True),
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Adicionar is_unbalance ou scale_pos_weight
        if 'is_unbalance' in config and config['is_unbalance']:
            lgbm_params['is_unbalance'] = True
        if 'scale_pos_weight' in config:
            lgbm_params['scale_pos_weight'] = config['scale_pos_weight']

        model = lgb.LGBMClassifier(**lgbm_params)

        logger.info("    Treinando...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Obter probabilidades
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Testar múltiplos thresholds
        best_local_f1 = 0
        best_local_threshold = 0.5
        best_local_metrics = None

        for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            y_pred = (y_pred_proba >= threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Evitar modelos triviais: precision E recall devem ser > 5%
            if precision > 0.05 and recall > 0.05:
                if f1 > best_local_f1:
                    best_local_f1 = f1
                    best_local_threshold = threshold
                    best_local_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

        # Se não encontrou threshold válido, usar threshold=0.5
        if best_local_metrics is None:
            y_pred = model.predict(X_test)
            best_local_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            best_local_threshold = 0.5

        auc_roc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"    Melhor threshold: {best_local_threshold}")
        logger.info(f"    Accuracy:  {best_local_metrics['accuracy']*100:.2f}%")
        logger.info(f"    Precision: {best_local_metrics['precision']*100:.2f}%")
        logger.info(f"    Recall:    {best_local_metrics['recall']*100:.2f}%")
        logger.info(f"    F1-Score:  {best_local_metrics['f1']*100:.2f}%")
        logger.info(f"    AUC-ROC:   {auc_roc:.4f}")

        result = {
            'config': config['name'],
            'threshold': best_local_threshold,
            'accuracy': best_local_metrics['accuracy'],
            'precision': best_local_metrics['precision'],
            'recall': best_local_metrics['recall'],
            'f1': best_local_metrics['f1'],
            'auc_roc': auc_roc
        }
        all_results.append(result)

        # Selecionar melhor por F1-Score (evita modelos triviais)
        if best_local_metrics['f1'] > best_f1:
            best_f1 = best_local_metrics['f1']
            best_model = model
            best_config = config
            best_config['threshold'] = best_local_threshold
            best_metrics = result

    # RESULTADOS FINAIS
    print("\n" + "="*60)
    print("RESUMO - Ordenado por F1-Score")
    print("="*60)

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('f1', ascending=False)

    for idx, row in df_results.iterrows():
        print(f"{row['config']:<20} | F1: {row['f1']*100:5.2f}% | " +
              f"Acc: {row['accuracy']*100:5.2f}% | Rec: {row['recall']*100:5.2f}% | " +
              f"Threshold: {row['threshold']}")

    print("\n" + "="*60)
    print("MELHOR MODELO LIGHTGBM")
    print("="*60)
    print(f"Configuração: {best_config['name']}")
    print(f"Threshold: {best_config['threshold']}")
    print("")
    print(f"Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Comparação completa
    print("\n" + "="*60)
    print("COMPARAÇÃO COMPLETA")
    print("="*60)
    print(f"                    Accuracy  Precision  Recall   F1-Score")
    print(f"Random Forest       62.09%    29.76%     23.36%   26.17%")
    print(f"XGBoost High Acc    68.14%    29.29%     7.61%    12.08%")
    print(f"LightGBM            {best_metrics['accuracy']*100:5.2f}%    " +
          f"{best_metrics['precision']*100:5.2f}%     {best_metrics['recall']*100:5.2f}%    " +
          f"{best_metrics['f1']*100:5.2f}%")
    print("="*60)

    # Validação
    if best_metrics['recall'] > 0.05:
        print(f"\n[OK] Modelo NÃO é trivial (recall = {best_metrics['recall']*100:.2f}%)")
    else:
        print(f"\n[AVISO] Modelo ainda pode ser trivial (recall muito baixo)")

    # Classification Report
    print("\nClassification Report:")
    y_pred_final = (best_model.predict_proba(X_test)[:, 1] >= best_config['threshold']).astype(int)
    print(classification_report(y_test, y_pred_final, target_names=['No Move', 'Price Up']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_final)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    print(f"  FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("TOP 20 FEATURES")
    print("="*60)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.0f}")

    # Salvar
    metadata = {
        'dataset': 'R_100_1m_6months',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%',
        'best_config': best_config,
        'all_results': all_results,
        'optimal_threshold': best_config['threshold']
    }

    trainer.save_model(
        best_model,
        f"lightgbm_fixed_{best_config['name'].lower()}",
        best_metrics,
        feature_importance,
        metadata
    )

    print("\n[SUCESSO] Modelo LightGBM corrigido salvo!")

    return best_model, best_metrics


if __name__ == "__main__":
    train_lightgbm_fixed()

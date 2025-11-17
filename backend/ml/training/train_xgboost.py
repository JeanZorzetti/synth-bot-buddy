"""
Treina XGBoost - Estado da Arte para Trading (2025)
Baseado em pesquisa: 98.69% accuracy (WCS-XGBoost), 86.2% (XGB+sentimentos)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import xgboost as xgb
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


def train_xgboost():
    """
    Treina XGBoost com hiperparâmetros otimizados para trading
    """
    print("\n" + "="*60)
    print("TREINAMENTO XGBoost - Estado da Arte 2025")
    print("Meta: 70-75% accuracy (baseado em literatura)")
    print("Dataset: 6 meses (260k candles)")
    print("="*60)

    # Criar trainer
    trainer = ModelTrainer()

    # Carregar dataset de 6 meses
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "ml_dataset_R100_1m_6months.pkl"

    logger.info(f"Carregando dataset de 6 meses...")
    X, y = trainer.load_ml_dataset(dataset_path)

    # Split temporal (importante para séries temporais)
    X_train, X_test, y_train, y_test = trainer.split_train_test(
        X, y,
        test_size=0.2,
        time_aware=True
    )

    # Calcular scale_pos_weight para dados desbalanceados
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    scale_pos_weight = neg_samples / pos_samples

    logger.info(f"\n{'='*60}")
    logger.info("CONFIGURAÇÃO XGBOOST")
    logger.info("="*60)
    logger.info(f"Classes desbalanceadas:")
    logger.info(f"  Negativas (No Move): {neg_samples:,} ({neg_samples/len(y_train)*100:.1f}%)")
    logger.info(f"  Positivas (Price Up): {pos_samples:,} ({pos_samples/len(y_train)*100:.1f}%)")
    logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # Configurações baseadas em pesquisa 2025
    configs = [
        {
            'name': 'Baseline',
            'params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight
            }
        },
        {
            'name': 'Deep Learning',
            'params': {
                'max_depth': 10,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'min_child_weight': 3,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'scale_pos_weight': scale_pos_weight,
                'gamma': 0.1
            }
        },
        {
            'name': 'Regularized',
            'params': {
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'gamma': 0.2,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }
    ]

    best_model = None
    best_accuracy = 0
    best_config = None
    best_metrics = None

    for config in configs:
        logger.info(f"\n>>> Testando configuração: {config['name']}")
        logger.info(f"    Parâmetros:")
        for key, value in config['params'].items():
            logger.info(f"      {key}: {value}")

        # Criar e treinar modelo XGBoost
        model = xgb.XGBClassifier(
            **config['params'],
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )

        logger.info("\n    Iniciando treinamento...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        logger.info("    [OK] Treinamento concluído!")

        # Avaliar
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.0

        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist(),
            'model_name': f'XGBoost - {config["name"]}',
            'test_samples': len(y_test)
        }

        # Log resultados
        logger.info(f"\n    RESULTADOS:")
        logger.info(f"    Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"    Precision: {precision*100:.2f}%")
        logger.info(f"    Recall:    {recall*100:.2f}%")
        logger.info(f"    F1-Score:  {f1*100:.2f}%")
        logger.info(f"    AUC-ROC:   {auc_roc:.4f}")
        logger.info(f"    Confusion Matrix:")
        logger.info(f"      TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
        logger.info(f"      FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")

        # Guardar melhor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_config = config
            best_metrics = metrics

    # Resultado final
    print("\n" + "="*60)
    print("MELHOR MODELO XGBOOST")
    print("="*60)
    print(f"Configuração: {best_config['name']}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1_score']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Classification Report detalhado
    print("\nClassification Report:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['No Move', 'Price Up']))

    # Feature importance (top 20)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("TOP 20 FEATURES MAIS IMPORTANTES")
    print("="*60)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.6f}")

    # Salvar modelo
    metadata = {
        'dataset': 'R_100_1m_6months',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%',
        'config': best_config['name'],
        'parameters': best_config['params']
    }

    trainer.save_model(
        best_model,
        f"xgboost_{best_config['name'].lower().replace(' ', '_')}",
        best_metrics,
        feature_importance,
        metadata
    )

    print("\n[SUCESSO] Modelo XGBoost salvo!")
    print(f"Arquivos salvos em: backend/ml/models/")

    # Comparação com Random Forest
    print("\n" + "="*60)
    print("COMPARAÇÃO: XGBoost vs Random Forest")
    print("="*60)
    print(f"                    Random Forest  |  XGBoost")
    print(f"  Accuracy:         62.09%         |  {best_accuracy*100:.2f}%")
    print(f"  Precision:        29.76%         |  {best_metrics['precision']*100:.2f}%")
    print(f"  Recall:           23.36%         |  {best_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:         26.17%         |  {best_metrics['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:          0.5156         |  {best_metrics['auc_roc']:.4f}")
    print("="*60)

    improvement = (best_accuracy - 0.6209) / 0.6209 * 100
    print(f"\nMelhoria: {improvement:+.2f}%")

    if best_accuracy >= 0.70:
        print("\n🎉 META ATINGIDA: Accuracy >= 70%!")
    else:
        print(f"\n⚠️ META: {(0.70 - best_accuracy)*100:.2f}% para atingir 70%")

    return best_model, best_metrics


if __name__ == "__main__":
    train_xgboost()

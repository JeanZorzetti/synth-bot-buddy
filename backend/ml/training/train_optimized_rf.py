"""
Treina Random Forest Otimizado com 6 meses de dados
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from model_training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def train_optimized_rf():
    """
    Treina Random Forest com hiperparâmetros otimizados
    """
    print("\n" + "="*60)
    print("TREINAMENTO OTIMIZADO - RANDOM FOREST")
    print("Dataset: 6 meses de dados históricos")
    print("="*60)

    # Criar trainer
    trainer = ModelTrainer()

    # Carregar dataset de 6 meses
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "ml_dataset_R100_1m_6months.pkl"

    logger.info(f"Carregando dataset de 6 meses...")
    X, y = trainer.load_ml_dataset(dataset_path)

    # Split temporal
    X_train, X_test, y_train, y_test = trainer.split_train_test(
        X, y,
        test_size=0.2,
        time_aware=True
    )

    # Hiperparâmetros otimizados (baseado em experiência)
    logger.info("\n" + "="*60)
    logger.info("CONFIGURAÇÃO OTIMIZADA")
    logger.info("="*60)

    # Testar diferentes configurações
    configs = [
        {
            'name': 'Baseline',
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        },
        {
            'name': 'Deep Trees',
            'n_estimators': 200,
            'max_depth': 30,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        {
            'name': 'Many Trees',
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 15,
            'min_samples_leaf': 8
        }
    ]

    best_model = None
    best_accuracy = 0
    best_config = None
    best_metrics = None

    for config in configs:
        logger.info(f"\n>>> Testando configuração: {config['name']}")
        logger.info(f"    n_estimators: {config['n_estimators']}")
        logger.info(f"    max_depth: {config['max_depth']}")
        logger.info(f"    min_samples_split: {config['min_samples_split']}")
        logger.info(f"    min_samples_leaf: {config['min_samples_leaf']}")

        # Treinar
        model = trainer.train_random_forest(
            X_train, y_train,
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            class_weight='balanced'
        )

        # Avaliar
        metrics = trainer.evaluate_model(
            model, X_test, y_test,
            model_name=f"RF - {config['name']}"
        )

        # Guardar melhor
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = model
            best_config = config
            best_metrics = metrics

        logger.info(f">>> Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f">>> F1-Score: {metrics['f1_score']*100:.2f}%")

    # Resultado final
    print("\n" + "="*60)
    print("MELHOR MODELO ENCONTRADO")
    print("="*60)
    print(f"Configuração: {best_config['name']}")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall: {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score: {best_metrics['f1_score']*100:.2f}%")
    print(f"AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Feature importance
    feature_importance = trainer.get_feature_importance(
        best_model,
        X.columns.tolist(),
        top_n=20
    )

    # Salvar melhor modelo
    metadata = {
        'dataset': 'R_100_1m_6months',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%',
        'config': best_config
    }

    trainer.save_model(
        best_model,
        f"random_forest_optimized_{best_config['name'].lower().replace(' ', '_')}",
        best_metrics,
        feature_importance,
        metadata
    )

    print("\n[SUCESSO] Modelo otimizado salvo!")
    return best_model, best_metrics


if __name__ == "__main__":
    train_optimized_rf()

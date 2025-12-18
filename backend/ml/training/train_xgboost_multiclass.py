"""
Treina XGBoost MULTI-CLASS - Corrige bias de predição

PROBLEMA CRÍTICO:
- Modelo atual (binary) prevê APENAS PRICE_UP (100% das vezes)
- Acurácia de 15.38% (pior que random)

SOLUÇÃO:
- Mudar de binary:logistic para multi:softmax (3 classes)
- Balancear dataset (33% UP / 33% DOWN / 33% NO_MOVE)
- Remover threshold e usar predict() direto
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from model_training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def balance_multiclass_dataset(X, y, target_samples_per_class=None):
    """
    Balanceia dataset para ter distribuição uniforme entre as 3 classes

    Args:
        X: Features DataFrame
        y: Labels Series (0=NO_MOVE, 1=PRICE_UP, 2=PRICE_DOWN)
        target_samples_per_class: Número de amostras por classe (None = usar mínimo)

    Returns:
        X_balanced, y_balanced
    """
    logger.info("\n" + "="*60)
    logger.info("BALANCEAMENTO DE DATASET MULTI-CLASS")
    logger.info("="*60)

    # Contar amostras por classe
    class_counts = y.value_counts().sort_index()
    logger.info(f"\nDistribuição ANTES do balanceamento:")
    for class_id, count in class_counts.items():
        class_name = ['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'][class_id]
        logger.info(f"  Classe {class_id} ({class_name}): {count:,} ({count/len(y)*100:.1f}%)")

    # Determinar tamanho alvo
    if target_samples_per_class is None:
        target_samples_per_class = int(class_counts.min())

    logger.info(f"\nTarget: {target_samples_per_class:,} amostras por classe")

    # Balancear usando undersampling (manter proporções temporais)
    balanced_indices = []

    for class_id in range(3):
        class_indices = y[y == class_id].index.tolist()

        # Se temos mais amostras que o target, fazer undersampling
        if len(class_indices) > target_samples_per_class:
            # Manter distribuição temporal - pegar amostras espaçadas
            step = len(class_indices) / target_samples_per_class
            sampled_indices = [class_indices[int(i * step)] for i in range(target_samples_per_class)]
        else:
            # Se temos menos, manter todas
            sampled_indices = class_indices

        balanced_indices.extend(sampled_indices)

    # Criar dataset balanceado
    X_balanced = X.loc[balanced_indices]
    y_balanced = y.loc[balanced_indices]

    # Verificar resultado
    balanced_counts = y_balanced.value_counts().sort_index()
    logger.info(f"\nDistribuição DEPOIS do balanceamento:")
    for class_id, count in balanced_counts.items():
        class_name = ['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'][class_id]
        logger.info(f"  Classe {class_id} ({class_name}): {count:,} ({count/len(y_balanced)*100:.1f}%)")

    logger.info(f"\nDataset reduzido: {len(y):,} -> {len(y_balanced):,} amostras")
    logger.info("="*60)

    return X_balanced, y_balanced


def train_xgboost_multiclass():
    """
    Treina XGBoost com 3 classes: NO_MOVE, PRICE_UP, PRICE_DOWN
    """
    print("\n" + "="*60)
    print("XGBOOST MULTI-CLASS - Correção de Bias")
    print("="*60)

    # Criar trainer
    trainer = ModelTrainer()

    # Carregar dataset de 6 meses
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "ml_dataset_R100_1m_6months.pkl"

    logger.info(f"Carregando dataset de 6 meses...")
    X, y = trainer.load_ml_dataset(dataset_path)

    # CONVERTER DE BINARY PARA MULTI-CLASS
    # Atualmente: 0=NO_MOVE, 1=PRICE_UP
    # Precisamos adicionar: 2=PRICE_DOWN

    # Recarregar dataset raw para recalcular labels
    logger.info("\nRecalculando labels para multi-class...")
    df_raw = pd.read_pickle(dataset_path)

    # Calcular retornos futuros (15 minutos)
    df_raw['future_return'] = df_raw['close'].shift(-15) / df_raw['close'] - 1

    # Criar labels multi-class com threshold de 0.3%
    threshold = 0.003
    y_multiclass = pd.Series(0, index=df_raw.index)  # Default: NO_MOVE
    y_multiclass[df_raw['future_return'] > threshold] = 1  # PRICE_UP
    y_multiclass[df_raw['future_return'] < -threshold] = 2  # PRICE_DOWN

    # Remover últimas 15 linhas (sem label futuro)
    valid_indices = y_multiclass.index[:-15]
    X = X.loc[valid_indices]
    y_multiclass = y_multiclass.loc[valid_indices]

    logger.info(f"\nDistribuição ORIGINAL das classes:")
    for class_id, count in y_multiclass.value_counts().sort_index().items():
        class_name = ['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'][class_id]
        logger.info(f"  {class_name}: {count:,} ({count/len(y_multiclass)*100:.1f}%)")

    # Split temporal ANTES de balancear (para manter distribuição realista no test)
    X_train, X_test, y_train, y_test = trainer.split_train_test(
        X, y_multiclass,
        test_size=0.2,
        time_aware=True
    )

    # Balancear APENAS o conjunto de treino
    X_train_balanced, y_train_balanced = balance_multiclass_dataset(X_train, y_train)

    logger.info(f"\nTrain samples (balanced): {len(X_train_balanced):,}")
    logger.info(f"Test samples (original distribution): {len(X_test):,}")

    # Verificar distribuição do test set
    logger.info(f"\nDistribuição do TEST set:")
    for class_id, count in y_test.value_counts().sort_index().items():
        class_name = ['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'][class_id]
        logger.info(f"  {class_name}: {count:,} ({count/len(y_test)*100:.1f}%)")

    # Treinar XGBoost MULTI-CLASS
    print("\n" + "="*60)
    print("TREINANDO XGBOOST MULTI-CLASS")
    print("="*60)

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',  # MULTI-CLASS!!!
        num_class=3,  # 3 classes
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )

    logger.info(f"Treinando com objective='multi:softmax' e num_class=3...")

    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # Avaliar
    print("\n" + "="*60)
    print("AVALIAÇÃO DO MODELO MULTI-CLASS")
    print("="*60)

    y_pred = model.predict(X_test)

    # Métricas gerais
    accuracy = accuracy_score(y_test, y_pred)

    # Métricas por classe (macro average)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    logger.info(f"\nMétricas Globais:")
    logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
    logger.info(f"  Precision (macro): {precision*100:.2f}%")
    logger.info(f"  Recall (macro):    {recall*100:.2f}%")
    logger.info(f"  F1-Score (macro):  {f1*100:.2f}%")

    # Validar distribuição de predições
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    logger.info(f"\nDistribuição das PREDIÇÕES:")
    for class_id, count in pred_counts.items():
        class_name = ['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'][class_id]
        logger.info(f"  {class_name}: {count:,} ({count/len(y_pred)*100:.1f}%)")

    # VALIDACAO CRITICA: Modelo preve todas as 3 classes?
    unique_predictions = set(y_pred)
    if len(unique_predictions) == 3:
        logger.info("\nSUCESSO: Modelo preve TODAS as 3 classes!")
    else:
        logger.error(f"\nFALHA: Modelo so preve {len(unique_predictions)} classes: {unique_predictions}")
        logger.error("   Modelo ainda esta com bias!")

    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test, y_pred,
        target_names=['NO_MOVE', 'PRICE_UP', 'PRICE_DOWN'],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info("                 Predicted")
    logger.info("               NO  UP  DOWN")
    logger.info(f"Actual NO    {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
    logger.info(f"Actual UP    {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
    logger.info(f"Actual DOWN  {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n" + "="*60)
        print("TOP 20 FEATURES MAIS IMPORTANTES")
        print("="*60)
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.6f}")

    # Salvar modelo
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_classes': 3,
        'classes_predicted': len(unique_predictions)
    }

    metadata = {
        'dataset': 'R_100_1m_6months_balanced',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%',
        'num_class': 3,
        'objective': 'multi:softmax',
        'balanced': True
    }

    trainer.save_model(
        model,
        "xgboost_multiclass",
        metrics,
        feature_importance if 'feature_importance' in locals() else None,
        metadata
    )

    print("\n" + "="*60)
    print("COMPARAÇÃO COM MODELO ANTERIOR")
    print("="*60)
    print(f"Modelo BINARY (com bias):     15.38% accuracy, 100% PRICE_UP")
    print(f"Modelo MULTI-CLASS (novo):    {accuracy*100:.2f}% accuracy, {len(unique_predictions)}/3 classes")

    improvement = accuracy - 0.1538
    print(f"\nMelhoria: {improvement*100:+.2f} pontos percentuais")

    if accuracy >= 0.45:
        print("\nOBJETIVO ATINGIDO: Acuracia >= 45%")
    else:
        print(f"\nFaltam {(0.45-accuracy)*100:.2f}pp para atingir 45%")

    print("\n[SUCESSO] Modelo XGBoost Multi-Class salvo!")
    print("="*60)

    return model, metrics, feature_importance


if __name__ == "__main__":
    train_xgboost_multiclass()

"""
Treina XGBoost MELHORADO - Diagnóstico e Otimização
Investiga: scale_pos_weight, feature scaling, thresholds, hyperparameters
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
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


def train_xgboost_improved():
    """
    Investiga e otimiza XGBoost com múltiplas estratégias
    """
    print("\n" + "="*60)
    print("XGBOOST MELHORADO - Diagnóstico de Performance")
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

    # Calcular scale_pos_weight
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    natural_scale = neg_samples / pos_samples

    logger.info(f"\n{'='*60}")
    logger.info("ANÁLISE DE CLASSES")
    logger.info("="*60)
    logger.info(f"Negativas: {neg_samples:,} ({neg_samples/len(y_train)*100:.1f}%)")
    logger.info(f"Positivas: {pos_samples:,} ({pos_samples/len(y_train)*100:.1f}%)")
    logger.info(f"Natural scale_pos_weight: {natural_scale:.2f}")

    # EXPERIMENTO 1: Testar diferentes scale_pos_weight
    print("\n" + "="*60)
    print("EXPERIMENTO 1: scale_pos_weight")
    print("="*60)

    scale_weights = [
        ('Sem balanceamento', 1.0),
        ('Balanceamento leve', 1.5),
        ('Balanceamento médio', 2.0),
        ('Balanceamento natural', natural_scale)
    ]

    best_model = None
    best_accuracy = 0
    best_config = None
    best_metrics = None
    all_results = []

    for name, scale_weight in scale_weights:
        logger.info(f"\n>>> Testando: {name} (scale_pos_weight={scale_weight:.2f})")

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Avaliar
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

        logger.info(f"    Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"    Precision: {precision*100:.2f}%")
        logger.info(f"    Recall:    {recall*100:.2f}%")
        logger.info(f"    F1-Score:  {f1*100:.2f}%")

        result = {
            'experiment': 'scale_pos_weight',
            'config': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
        all_results.append(result)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_config = {'experiment': 'scale_pos_weight', 'name': name, 'scale_weight': scale_weight}
            best_metrics = result

    # EXPERIMENTO 2: Feature Scaling
    print("\n" + "="*60)
    print("EXPERIMENTO 2: Feature Scaling")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("\n>>> Testando: XGBoost com StandardScaler")

    model_scaled = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # Sem balanceamento artificial
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model_scaled.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    y_pred_scaled = model_scaled.predict(X_test_scaled)
    y_pred_proba_scaled = model_scaled.predict_proba(X_test_scaled)[:, 1]

    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
    precision_scaled = precision_score(y_test, y_pred_scaled, zero_division=0)
    recall_scaled = recall_score(y_test, y_pred_scaled, zero_division=0)
    f1_scaled = f1_score(y_test, y_pred_scaled, zero_division=0)
    auc_roc_scaled = roc_auc_score(y_test, y_pred_proba_scaled) if len(np.unique(y_test)) > 1 else 0.0

    logger.info(f"    Accuracy:  {accuracy_scaled*100:.2f}%")
    logger.info(f"    Precision: {precision_scaled*100:.2f}%")
    logger.info(f"    Recall:    {recall_scaled*100:.2f}%")
    logger.info(f"    F1-Score:  {f1_scaled*100:.2f}%")

    result_scaled = {
        'experiment': 'feature_scaling',
        'config': 'StandardScaler',
        'accuracy': accuracy_scaled,
        'precision': precision_scaled,
        'recall': recall_scaled,
        'f1': f1_scaled,
        'auc_roc': auc_roc_scaled
    }
    all_results.append(result_scaled)

    if accuracy_scaled > best_accuracy:
        best_accuracy = accuracy_scaled
        best_model = model_scaled
        best_config = {'experiment': 'feature_scaling', 'name': 'StandardScaler'}
        best_metrics = result_scaled

    # EXPERIMENTO 3: Diferentes learning rates
    print("\n" + "="*60)
    print("EXPERIMENTO 3: Learning Rates")
    print("="*60)

    learning_rates = [
        ('Muito lento', 0.01),
        ('Lento', 0.03),
        ('Médio', 0.05),
        ('Padrão', 0.1)
    ]

    for name, lr in learning_rates:
        logger.info(f"\n>>> Testando: {name} (learning_rate={lr})")

        model_lr = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=lr,
            n_estimators=300,  # Mais árvores para learning rates baixas
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model_lr.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred_lr = model_lr.predict(X_test)
        y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr, zero_division=0)
        recall_lr = recall_score(y_test, y_pred_lr, zero_division=0)
        f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
        auc_roc_lr = roc_auc_score(y_test, y_pred_proba_lr) if len(np.unique(y_test)) > 1 else 0.0

        logger.info(f"    Accuracy:  {accuracy_lr*100:.2f}%")
        logger.info(f"    Precision: {precision_lr*100:.2f}%")
        logger.info(f"    Recall:    {recall_lr*100:.2f}%")
        logger.info(f"    F1-Score:  {f1_lr*100:.2f}%")

        result_lr = {
            'experiment': 'learning_rate',
            'config': name,
            'accuracy': accuracy_lr,
            'precision': precision_lr,
            'recall': recall_lr,
            'f1': f1_lr,
            'auc_roc': auc_roc_lr
        }
        all_results.append(result_lr)

        if accuracy_lr > best_accuracy:
            best_accuracy = accuracy_lr
            best_model = model_lr
            best_config = {'experiment': 'learning_rate', 'name': name, 'lr': lr}
            best_metrics = result_lr

    # EXPERIMENTO 4: Max depth
    print("\n" + "="*60)
    print("EXPERIMENTO 4: Max Depth")
    print("="*60)

    depths = [
        ('Raso', 3),
        ('Médio', 5),
        ('Profundo', 8),
        ('Muito profundo', 12)
    ]

    for name, depth in depths:
        logger.info(f"\n>>> Testando: {name} (max_depth={depth})")

        model_depth = xgb.XGBClassifier(
            max_depth=depth,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model_depth.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred_depth = model_depth.predict(X_test)
        y_pred_proba_depth = model_depth.predict_proba(X_test)[:, 1]

        accuracy_depth = accuracy_score(y_test, y_pred_depth)
        precision_depth = precision_score(y_test, y_pred_depth, zero_division=0)
        recall_depth = recall_score(y_test, y_pred_depth, zero_division=0)
        f1_depth = f1_score(y_test, y_pred_depth, zero_division=0)
        auc_roc_depth = roc_auc_score(y_test, y_pred_proba_depth) if len(np.unique(y_test)) > 1 else 0.0

        logger.info(f"    Accuracy:  {accuracy_depth*100:.2f}%")
        logger.info(f"    Precision: {precision_depth*100:.2f}%")
        logger.info(f"    Recall:    {recall_depth*100:.2f}%")
        logger.info(f"    F1-Score:  {f1_depth*100:.2f}%")

        result_depth = {
            'experiment': 'max_depth',
            'config': name,
            'accuracy': accuracy_depth,
            'precision': precision_depth,
            'recall': recall_depth,
            'f1': f1_depth,
            'auc_roc': auc_roc_depth
        }
        all_results.append(result_depth)

        if accuracy_depth > best_accuracy:
            best_accuracy = accuracy_depth
            best_model = model_depth
            best_config = {'experiment': 'max_depth', 'name': name, 'depth': depth}
            best_metrics = result_depth

    # RESULTADOS FINAIS
    print("\n" + "="*60)
    print("RESUMO DE TODOS OS EXPERIMENTOS")
    print("="*60)

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    print("\nTop 5 Melhores Configurações:")
    print(df_results.head(5).to_string(index=False))

    print("\n" + "="*60)
    print("MELHOR MODELO ENCONTRADO")
    print("="*60)
    print(f"Experimento: {best_config['experiment']}")
    print(f"Configuração: {best_config['name']}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Comparação com Random Forest
    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL")
    print("="*60)
    print(f"Random Forest:        62.09% accuracy")
    print(f"XGBoost Original:     50.26% accuracy")
    print(f"XGBoost Melhorado:    {best_accuracy*100:.2f}% accuracy")
    improvement = (best_accuracy - 0.5026) / 0.5026 * 100
    print(f"Melhoria XGBoost:     {improvement:+.2f}%")

    if best_accuracy > 0.6209:
        print(f"\nSUCESSO: XGBoost superou Random Forest!")
    else:
        gap = (0.6209 - best_accuracy) / 0.6209 * 100
        print(f"\nAinda faltam {gap:.2f}% para superar Random Forest")

    # Classification Report
    print("\nClassification Report (Melhor Modelo):")
    if hasattr(best_model, 'predict'):
        y_pred_best = best_model.predict(X_test if best_config['experiment'] != 'feature_scaling' else X_test_scaled)
        print(classification_report(y_test, y_pred_best, target_names=['No Move', 'Price Up']))

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
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
            'best_config': best_config,
            'all_experiments': all_results
        }

        trainer.save_model(
            best_model,
            f"xgboost_improved_{best_config['experiment']}",
            best_metrics,
            feature_importance,
            metadata
        )

        print("\n[SUCESSO] Modelo XGBoost melhorado salvo!")

    return best_model, best_metrics, df_results


if __name__ == "__main__":
    train_xgboost_improved()

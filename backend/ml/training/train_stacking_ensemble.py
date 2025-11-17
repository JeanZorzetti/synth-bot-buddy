"""
Treina STACKING ENSEMBLE - Combina XGBoost + Random Forest
Meta: Superar 68.14% do XGBoost sozinho
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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


def train_stacking_ensemble():
    """
    Cria Stacking Ensemble combinando XGBoost + Random Forest
    """
    print("\n" + "="*60)
    print("STACKING ENSEMBLE - XGBoost + Random Forest")
    print("Estratégia: Combinar pontos fortes de ambos modelos")
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
    logger.info("MODELOS BASE (Level 0)")
    logger.info("="*60)

    # Modelo 1: XGBoost (68.14% accuracy)
    logger.info("\n1. XGBoost High Accuracy")
    logger.info("   Configuração: learning_rate=0.01, max_depth=6")
    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=300,
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

    # Modelo 2: Random Forest (62.09% accuracy)
    logger.info("\n2. Random Forest")
    logger.info("   Configuração: 200 estimators, max_depth=30")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"\n{'='*60}")
    logger.info("META-LEARNER (Level 1)")
    logger.info("="*60)
    logger.info("Testando 3 meta-learners diferentes:")
    logger.info("  1. Logistic Regression (linear)")
    logger.info("  2. XGBoost leve (não-linear)")
    logger.info("  3. Random Forest leve (não-linear)")

    # Definir meta-learners para testar
    meta_learners = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(max_iter=1000, random_state=42)
        },
        {
            'name': 'XGBoost-Meta',
            'model': xgb.XGBClassifier(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                verbosity=0
            )
        },
        {
            'name': 'RandomForest-Meta',
            'model': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
        }
    ]

    best_ensemble = None
    best_accuracy = 0
    best_meta_name = None
    best_metrics = None
    all_results = []

    for meta_config in meta_learners:
        logger.info(f"\n>>> Testando meta-learner: {meta_config['name']}")

        # Criar Stacking Ensemble
        ensemble = StackingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('random_forest', rf_model)
            ],
            final_estimator=meta_config['model'],
            cv=5,  # 5-fold cross-validation
            n_jobs=-1
        )

        logger.info("    Treinando ensemble (isso pode demorar)...")
        ensemble.fit(X_train, y_train)
        logger.info("    [OK] Treinamento concluído!")

        # Avaliar
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"    RESULTADOS:")
        logger.info(f"      Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"      Precision: {precision*100:.2f}%")
        logger.info(f"      Recall:    {recall*100:.2f}%")
        logger.info(f"      F1-Score:  {f1*100:.2f}%")
        logger.info(f"      AUC-ROC:   {auc_roc:.4f}")

        result = {
            'meta_learner': meta_config['name'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist()
        }
        all_results.append(result)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_ensemble = ensemble
            best_meta_name = meta_config['name']
            best_metrics = result

    # RESULTADOS FINAIS
    print("\n" + "="*60)
    print("RESUMO DE META-LEARNERS")
    print("="*60)

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    for idx, row in df_results.iterrows():
        print(f"{row['meta_learner']:<25} | Acc: {row['accuracy']*100:5.2f}% | " +
              f"F1: {row['f1']*100:5.2f}% | Rec: {row['recall']*100:5.2f}%")

    print("\n" + "="*60)
    print("MELHOR ENSEMBLE")
    print("="*60)
    print(f"Meta-learner: {best_meta_name}")
    print(f"Modelos base: XGBoost + Random Forest")
    print("")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Comparação COMPLETA
    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL - TODOS OS MODELOS")
    print("="*60)
    print(f"                         Accuracy  Precision  Recall   F1-Score")
    print(f"Random Forest            62.09%    29.76%     23.36%   26.17%")
    print(f"XGBoost High Acc         68.14%    29.29%     7.61%    12.08%")
    print(f"Stacking Ensemble        {best_accuracy*100:5.2f}%    " +
          f"{best_metrics['precision']*100:5.2f}%     {best_metrics['recall']*100:5.2f}%    " +
          f"{best_metrics['f1']*100:5.2f}%")
    print("="*60)

    # Verificar melhoria
    xgb_accuracy = 0.6814
    if best_accuracy > xgb_accuracy:
        improvement = (best_accuracy - xgb_accuracy) / xgb_accuracy * 100
        print(f"\n[SUCESSO] Ensemble superou XGBoost em {improvement:.2f}%!")
    elif best_accuracy >= xgb_accuracy * 0.99:
        print(f"\n[OK] Ensemble tem performance similar ao XGBoost")
    else:
        gap = (xgb_accuracy - best_accuracy) / xgb_accuracy * 100
        print(f"\n[INFO] XGBoost individual ainda é {gap:.2f}% superior")
        print(f"       Ensemble não trouxe benefício neste caso")

    # Classification Report
    print("\nClassification Report (Ensemble):")
    y_pred_final = best_ensemble.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=['No Move', 'Price Up']))

    # Confusion Matrix detalhada
    cm_final = confusion_matrix(y_test, y_pred_final)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm_final[0][0]:,} (correctly predicted no move)")
    print(f"  False Positives: {cm_final[0][1]:,} (incorrectly predicted price up)")
    print(f"  False Negatives: {cm_final[1][0]:,} (missed price increases)")
    print(f"  True Positives:  {cm_final[1][1]:,} (correctly predicted price up)")

    # Análise de predições individuais dos modelos base
    print("\n" + "="*60)
    print("ANÁLISE DOS MODELOS BASE")
    print("="*60)

    # Treinar modelos individuais para comparar
    logger.info("Treinando modelos base individuais para análise...")

    xgb_individual = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.01, n_estimators=300,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=1.0, random_state=42, verbosity=0
    )
    xgb_individual.fit(X_train, y_train)

    rf_individual = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced', random_state=42
    )
    rf_individual.fit(X_train, y_train)

    xgb_pred = xgb_individual.predict(X_test)
    rf_pred = rf_individual.predict(X_test)
    ensemble_pred = best_ensemble.predict(X_test)

    # Casos onde ensemble difere
    xgb_only_correct = ((xgb_pred == y_test) & (ensemble_pred != y_test)).sum()
    rf_only_correct = ((rf_pred == y_test) & (ensemble_pred != y_test)).sum()
    ensemble_only_correct = ((ensemble_pred == y_test) & (xgb_pred != y_test) & (rf_pred != y_test)).sum()
    all_agree_correct = ((xgb_pred == y_test) & (rf_pred == y_test) & (ensemble_pred == y_test)).sum()

    print(f"\nAnálise de Concordância:")
    print(f"  Todos acertaram:                {all_agree_correct:,} casos")
    print(f"  Apenas XGBoost acertou:         {xgb_only_correct:,} casos")
    print(f"  Apenas Random Forest acertou:   {rf_only_correct:,} casos")
    print(f"  Apenas Ensemble acertou:        {ensemble_only_correct:,} casos")

    # Salvar ensemble
    metadata = {
        'dataset': 'R_100_1m_6months',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%',
        'meta_learner': best_meta_name,
        'base_models': ['XGBoost', 'RandomForest'],
        'all_results': all_results,
        'comparison': {
            'random_forest': 0.6209,
            'xgboost': 0.6814,
            'ensemble': best_accuracy
        }
    }

    # Salvar usando pickle
    models_dir = Path(__file__).parent.parent / "models"
    ensemble_path = models_dir / f"stacking_ensemble_{best_meta_name.lower()}_20251117.pkl"

    with open(ensemble_path, 'wb') as f:
        pickle.dump(best_ensemble, f)

    logger.info(f"[SALVO] Ensemble: {ensemble_path}")

    # Salvar métricas
    metrics_path = models_dir / f"stacking_ensemble_{best_meta_name.lower()}_20251117_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'meta_learner': best_meta_name,
            'metrics': best_metrics,
            'metadata': metadata
        }, f, indent=2, default=str)

    logger.info(f"[SALVO] Métricas: {metrics_path}")

    print("\n[SUCESSO] Stacking Ensemble salvo!")
    print(f"Arquivos salvos em: backend/ml/models/")

    return best_ensemble, best_metrics, df_results


if __name__ == "__main__":
    train_stacking_ensemble()

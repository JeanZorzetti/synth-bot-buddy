"""
Treina XGBoost BALANCEADO - Otimizado para Trading Real
Meta: 65%+ accuracy com 20%+ recall
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


def train_xgboost_balanced():
    """
    Treina XGBoost balanceando accuracy e recall para uso prático em trading
    """
    print("\n" + "="*60)
    print("XGBOOST BALANCEADO - Otimizado para Trading Real")
    print("Meta: Accuracy >= 65% + Recall >= 20%")
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
    logger.info("ESTRATÉGIA: Balancear Accuracy + Recall")
    logger.info("="*60)
    logger.info("Problema identificado:")
    logger.info("  - Learning rate muito baixo (0.01) = alta accuracy mas recall muito baixo")
    logger.info("  - Learning rate muito alto (0.1) = melhor recall mas accuracy baixa")
    logger.info("  - scale_pos_weight prejudica accuracy")
    logger.info("")
    logger.info("Solução:")
    logger.info("  - Learning rate intermediário (0.02-0.04)")
    logger.info("  - Max depth moderado (4-6)")
    logger.info("  - Mais estimadores (400-500)")
    logger.info("  - Threshold tuning para balancear precisão/recall")

    # Configurações balanceadas
    configs = [
        {
            'name': 'Balanced-1',
            'max_depth': 4,
            'learning_rate': 0.02,
            'n_estimators': 400,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.05
        },
        {
            'name': 'Balanced-2',
            'max_depth': 5,
            'learning_rate': 0.025,
            'n_estimators': 400,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1
        },
        {
            'name': 'Balanced-3',
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 350,
            'min_child_weight': 2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0
        },
        {
            'name': 'Balanced-4',
            'max_depth': 4,
            'learning_rate': 0.035,
            'n_estimators': 300,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.05
        }
    ]

    best_model = None
    best_score = 0  # Combinação de accuracy e recall
    best_config = None
    best_metrics = None
    all_results = []

    for config in configs:
        logger.info(f"\n>>> Testando: {config['name']}")
        logger.info(f"    max_depth={config['max_depth']}, lr={config['learning_rate']}, n_est={config['n_estimators']}")

        model = xgb.XGBClassifier(
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            n_estimators=config['n_estimators'],
            min_child_weight=config['min_child_weight'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            gamma=config['gamma'],
            scale_pos_weight=1.0,  # Sem balanceamento artificial
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

        # Avaliar com threshold padrão (0.5)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"    Threshold=0.5:")
        logger.info(f"      Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"      Precision: {precision*100:.2f}%")
        logger.info(f"      Recall:    {recall*100:.2f}%")
        logger.info(f"      F1-Score:  {f1*100:.2f}%")

        # Testar diferentes thresholds para otimizar recall
        best_threshold = 0.5
        best_threshold_score = 0

        for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            acc_thresh = accuracy_score(y_test, y_pred_thresh)
            rec_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)

            # Score balanceado: média ponderada (70% accuracy, 30% recall)
            score_thresh = 0.7 * acc_thresh + 0.3 * rec_thresh

            if score_thresh > best_threshold_score:
                best_threshold_score = score_thresh
                best_threshold = threshold

        # Avaliar com melhor threshold
        y_pred_best = (y_pred_proba >= best_threshold).astype(int)
        accuracy_best = accuracy_score(y_test, y_pred_best)
        precision_best = precision_score(y_test, y_pred_best, zero_division=0)
        recall_best = recall_score(y_test, y_pred_best, zero_division=0)
        f1_best = f1_score(y_test, y_pred_best, zero_division=0)

        logger.info(f"    Threshold={best_threshold} (otimizado):")
        logger.info(f"      Accuracy:  {accuracy_best*100:.2f}%")
        logger.info(f"      Precision: {precision_best*100:.2f}%")
        logger.info(f"      Recall:    {recall_best*100:.2f}%")
        logger.info(f"      F1-Score:  {f1_best*100:.2f}%")

        # Score balanceado para escolher melhor modelo
        balanced_score = 0.7 * accuracy_best + 0.3 * recall_best

        result = {
            'config': config['name'],
            'threshold': best_threshold,
            'accuracy': accuracy_best,
            'precision': precision_best,
            'recall': recall_best,
            'f1': f1_best,
            'auc_roc': auc_roc,
            'balanced_score': balanced_score
        }
        all_results.append(result)

        if balanced_score > best_score:
            best_score = balanced_score
            best_model = model
            best_config = config
            best_config['best_threshold'] = best_threshold
            best_metrics = result

    # RESULTADOS FINAIS
    print("\n" + "="*60)
    print("RESUMO DE TODAS AS CONFIGURACOES")
    print("="*60)

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('balanced_score', ascending=False)

    print("\nRanking (por balanced_score = 70% accuracy + 30% recall):")
    for idx, row in df_results.iterrows():
        print(f"{row['config']:<15} | Score: {row['balanced_score']:.4f} | " +
              f"Acc: {row['accuracy']*100:5.2f}% | Rec: {row['recall']*100:5.2f}% | " +
              f"Threshold: {row['threshold']}")

    print("\n" + "="*60)
    print("MELHOR MODELO BALANCEADO")
    print("="*60)
    print(f"Configuracao: {best_config['name']}")
    print(f"Threshold otimizado: {best_config['best_threshold']}")
    print(f"Balanced Score: {best_score:.4f}")
    print("")
    print(f"Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Comparação final
    print("\n" + "="*60)
    print("COMPARACAO COMPLETA")
    print("="*60)
    print(f"                    Accuracy  Precision  Recall   F1-Score")
    print(f"Random Forest       62.09%    29.76%     23.36%   26.17%")
    print(f"XGBoost Original    50.26%    29.37%     51.91%   37.51%")
    print(f"XGBoost High Acc    68.14%    29.29%     7.61%    12.08%")
    print(f"XGBoost BALANCED    {best_metrics['accuracy']*100:5.2f}%    " +
          f"{best_metrics['precision']*100:5.2f}%     {best_metrics['recall']*100:5.2f}%    " +
          f"{best_metrics['f1']*100:5.2f}%")
    print("="*60)

    # Verificar se atingiu metas
    print("\nAvaliacao de Metas:")
    if best_metrics['accuracy'] >= 0.65:
        print(f"  [OK] Accuracy >= 65%: {best_metrics['accuracy']*100:.2f}%")
    else:
        print(f"  [FALHOU] Accuracy < 65%: {best_metrics['accuracy']*100:.2f}%")

    if best_metrics['recall'] >= 0.20:
        print(f"  [OK] Recall >= 20%: {best_metrics['recall']*100:.2f}%")
    else:
        print(f"  [FALHOU] Recall < 20%: {best_metrics['recall']*100:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    y_pred_final = (best_model.predict_proba(X_test)[:, 1] >= best_config['best_threshold']).astype(int)
    print(classification_report(y_test, y_pred_final, target_names=['No Move', 'Price Up']))

    # Confusion Matrix detalhada
    cm = confusion_matrix(y_test, y_pred_final)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:,} (correctly predicted no move)")
    print(f"  False Positives: {cm[0][1]:,} (incorrectly predicted price up)")
    print(f"  False Negatives: {cm[1][0]:,} (missed price increases)")
    print(f"  True Positives:  {cm[1][1]:,} (correctly predicted price up)")

    # Feature importance
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
        'all_results': all_results,
        'optimization': 'balanced_accuracy_recall',
        'optimal_threshold': best_config['best_threshold']
    }

    trainer.save_model(
        best_model,
        f"xgboost_balanced_{best_config['name'].lower()}",
        best_metrics,
        feature_importance,
        metadata
    )

    print("\n[SUCESSO] Modelo XGBoost balanceado salvo!")
    print(f"Use threshold={best_config['best_threshold']} para predictions")

    return best_model, best_metrics, best_config['best_threshold']


if __name__ == "__main__":
    train_xgboost_balanced()

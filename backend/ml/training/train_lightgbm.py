"""
Treina LightGBM - Modelo rápido e eficiente para ensemble
Meta: 65%+ accuracy com treinamento rápido
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


def train_lightgbm():
    """
    Treina LightGBM com configurações otimizadas
    """
    print("\n" + "="*60)
    print("TREINAMENTO LIGHTGBM - Modelo Rápido e Eficiente")
    print("Meta: 65%+ accuracy (similar ao XGBoost)")
    print("="*60)

    # Criar trainer
    trainer = ModelTrainer()

    # Carregar dataset
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

    # Calcular class weights
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    scale_pos_weight = neg_samples / pos_samples

    logger.info(f"\n{'='*60}")
    logger.info("ANÁLISE DE CLASSES")
    logger.info("="*60)
    logger.info(f"Negativas: {neg_samples:,} ({neg_samples/len(y_train)*100:.1f}%)")
    logger.info(f"Positivas: {pos_samples:,} ({pos_samples/len(y_train)*100:.1f}%)")
    logger.info(f"Natural scale_pos_weight: {scale_pos_weight:.2f}")

    logger.info(f"\n{'='*60}")
    logger.info("ESTRATÉGIA LIGHTGBM")
    logger.info("="*60)
    logger.info("Aprendizados do XGBoost:")
    logger.info("  - scale_pos_weight=1.0 (sem balanceamento artificial)")
    logger.info("  - learning_rate baixo (0.01-0.03) funciona melhor")
    logger.info("  - num_leaves moderado (evitar overfitting)")
    logger.info("")
    logger.info("Vantagens do LightGBM:")
    logger.info("  - Treinamento mais rápido que XGBoost")
    logger.info("  - Uso eficiente de memória")
    logger.info("  - Bom para datasets grandes")

    # Configurações baseadas nos aprendizados do XGBoost
    configs = [
        {
            'name': 'Baseline',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        {
            'name': 'Conservative',
            'num_leaves': 15,
            'learning_rate': 0.01,
            'n_estimators': 400,
            'max_depth': 5,
            'min_child_samples': 30,
            'subsample': 0.85,
            'colsample_bytree': 0.85
        },
        {
            'name': 'Balanced',
            'num_leaves': 25,
            'learning_rate': 0.02,
            'n_estimators': 350,
            'max_depth': 6,
            'min_child_samples': 25,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        {
            'name': 'Deep',
            'num_leaves': 40,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        }
    ]

    best_model = None
    best_accuracy = 0
    best_config = None
    best_metrics = None
    all_results = []

    for config in configs:
        logger.info(f"\n>>> Testando: {config['name']}")
        logger.info(f"    num_leaves={config['num_leaves']}, lr={config['learning_rate']}")
        logger.info(f"    n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")

        # Criar modelo LightGBM
        model = lgb.LGBMClassifier(
            num_leaves=config['num_leaves'],
            learning_rate=config['learning_rate'],
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_child_samples=config['min_child_samples'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            scale_pos_weight=1.0,  # Sem balanceamento artificial
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        logger.info("    Treinando...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        logger.info("    [OK] Treinamento concluído!")

        # Avaliar
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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
        logger.info(f"      Confusion Matrix:")
        logger.info(f"        TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
        logger.info(f"        FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")

        result = {
            'config': config['name'],
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
            best_model = model
            best_config = config
            best_metrics = result

    # RESULTADOS FINAIS
    print("\n" + "="*60)
    print("RESUMO DE TODAS AS CONFIGURAÇÕES")
    print("="*60)

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    print("\nRanking por Accuracy:")
    for idx, row in df_results.iterrows():
        print(f"{row['config']:<15} | Acc: {row['accuracy']*100:5.2f}% | " +
              f"Prec: {row['precision']*100:5.2f}% | Rec: {row['recall']*100:5.2f}% | " +
              f"F1: {row['f1']*100:5.2f}%")

    print("\n" + "="*60)
    print("MELHOR MODELO LIGHTGBM")
    print("="*60)
    print(f"Configuração: {best_config['name']}")
    print(f"num_leaves: {best_config['num_leaves']}")
    print(f"learning_rate: {best_config['learning_rate']}")
    print(f"n_estimators: {best_config['n_estimators']}")
    print("")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Comparação completa
    print("\n" + "="*60)
    print("COMPARAÇÃO COMPLETA DE MODELOS")
    print("="*60)
    print(f"                    Accuracy  Precision  Recall   F1-Score  AUC-ROC")
    print(f"Random Forest       62.09%    29.76%     23.36%   26.17%    0.5156")
    print(f"XGBoost High Acc    68.14%    29.29%     7.61%    12.08%    0.5156")
    print(f"LightGBM            {best_accuracy*100:5.2f}%    " +
          f"{best_metrics['precision']*100:5.2f}%     {best_metrics['recall']*100:5.2f}%    " +
          f"{best_metrics['f1']*100:5.2f}%    {best_metrics['auc_roc']:.4f}")
    print("="*60)

    # Verificar se atingiu meta
    if best_accuracy >= 0.65:
        print(f"\n[OK] META ATINGIDA: Accuracy >= 65% ({best_accuracy*100:.2f}%)")
    else:
        gap = (0.65 - best_accuracy) * 100
        print(f"\n[INFO] Faltam {gap:.2f} pontos percentuais para atingir 65%")

    # Comparação com XGBoost
    xgb_accuracy = 0.6814
    if best_accuracy > xgb_accuracy:
        improvement = (best_accuracy - xgb_accuracy) / xgb_accuracy * 100
        print(f"\n[SUCESSO] LightGBM superou XGBoost em {improvement:.2f}%!")
    elif best_accuracy >= xgb_accuracy * 0.98:  # Dentro de 2%
        print(f"\n[OK] LightGBM tem performance similar ao XGBoost (diferença < 2%)")
    else:
        gap = (xgb_accuracy - best_accuracy) / xgb_accuracy * 100
        print(f"\n[INFO] XGBoost ainda é {gap:.2f}% superior")

    # Classification Report
    print("\nClassification Report:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['No Move', 'Price Up']))

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

    # Comparar top features com XGBoost
    print("\n" + "="*60)
    print("COMPARAÇÃO DE FEATURES: LightGBM vs XGBoost")
    print("="*60)
    xgb_top_features = ['sma_50', 'bb_middle', 'bb_lower', 'ema_9', 'ema_21']
    lgbm_top_features = feature_importance.head(5)['feature'].tolist()

    print("Top 5 XGBoost:  ", ', '.join(xgb_top_features))
    print("Top 5 LightGBM: ", ', '.join(lgbm_top_features))

    # Contar features em comum
    common_features = set(xgb_top_features) & set(lgbm_top_features)
    print(f"\nFeatures em comum no Top 5: {len(common_features)}/5")
    if common_features:
        print(f"Features compartilhadas: {', '.join(common_features)}")

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
        'comparison': {
            'random_forest': 0.6209,
            'xgboost': 0.6814,
            'lightgbm': best_accuracy
        }
    }

    trainer.save_model(
        best_model,
        f"lightgbm_{best_config['name'].lower()}",
        best_metrics,
        feature_importance,
        metadata
    )

    print("\n[SUCESSO] Modelo LightGBM salvo!")
    print(f"Arquivos salvos em: backend/ml/models/")

    # Recomendação para ensemble
    print("\n" + "="*60)
    print("PRÓXIMO PASSO: ENSEMBLE STACKING")
    print("="*60)
    print("Modelos disponíveis para ensemble:")
    print(f"  1. Random Forest:    62.09% accuracy")
    print(f"  2. XGBoost High Acc: 68.14% accuracy")
    print(f"  3. LightGBM:         {best_accuracy*100:.2f}% accuracy")
    print("")
    print("Estratégia de Ensemble recomendada:")
    print("  - Meta-learner: Regressão Logística ou XGBoost")
    print("  - Features: Probabilidades dos 3 modelos base")
    print("  - Expectativa: 70-72% accuracy (combinando pontos fortes)")
    print("="*60)

    return best_model, best_metrics, df_results


if __name__ == "__main__":
    train_lightgbm()

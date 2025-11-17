"""
Model Training - Random Forest, XGBoost, LSTM
Treina modelos de Machine Learning para previsão de mercado
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Treina e avalia modelos de Machine Learning
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Args:
            models_dir: Diretório para salvar modelos treinados
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()

    def load_ml_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carrega dataset ML preparado

        Args:
            dataset_path: Caminho para o arquivo .pkl

        Returns:
            (X, y) - Features e target
        """
        logger.info(f"Carregando dataset de {dataset_path}")
        df = pd.read_pickle(dataset_path)

        # Separar features e target
        non_feature_cols = [
            'timestamp', 'symbol', 'timeframe',
            'open', 'high', 'low', 'close', 'volume',
            'target', 'target_continuous'
        ]

        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols]
        y = df['target']

        logger.info(f"Dataset carregado: X={X.shape}, y={y.shape}")
        logger.info(f"Distribuição de classes: {y.value_counts().to_dict()}")

        return X, y

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        time_aware: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino e teste

        Args:
            X: Features
            y: Target
            test_size: Proporção de teste (0.2 = 20%)
            time_aware: Se True, usa split temporal (últimos dados = teste)

        Returns:
            X_train, X_test, y_train, y_test
        """
        if time_aware:
            # Split temporal - últimos 20% são teste
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            logger.info("Split temporal realizado")
        else:
            # Split aleatório
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("Split aleatório realizado")

        logger.info(f"Treino: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Teste: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 20,
        class_weight: str = 'balanced'
    ) -> RandomForestClassifier:
        """
        Treina modelo Random Forest

        Args:
            X_train: Features de treino
            y_train: Target de treino
            n_estimators: Número de árvores
            max_depth: Profundidade máxima das árvores
            min_samples_split: Mínimo de amostras para split
            class_weight: 'balanced' para ajustar desbalanceamento

        Returns:
            Modelo treinado
        """
        logger.info("\n" + "="*60)
        logger.info("TREINANDO RANDOM FOREST")
        logger.info("="*60)
        logger.info(f"Hiperparâmetros:")
        logger.info(f"  n_estimators: {n_estimators}")
        logger.info(f"  max_depth: {max_depth}")
        logger.info(f"  min_samples_split: {min_samples_split}")
        logger.info(f"  class_weight: {class_weight}")

        # Criar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,  # Usar todos os cores
            verbose=1
        )

        # Treinar
        logger.info("\nIniciando treinamento...")
        model.fit(X_train, y_train)

        logger.info("[OK] Random Forest treinado com sucesso!")

        return model

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Dict:
        """
        Avalia performance do modelo

        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            model_name: Nome do modelo para logging

        Returns:
            Dicionário com métricas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AVALIANDO {model_name}")
        logger.info("="*60)

        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist(),
            'model_name': model_name,
            'test_samples': len(y_test)
        }

        # Logging
        logger.info(f"\n{'='*60}")
        logger.info("MÉTRICAS DE PERFORMANCE")
        logger.info("="*60)
        logger.info(f"Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"Precision: {precision*100:.2f}%")
        logger.info(f"Recall:    {recall*100:.2f}%")
        logger.info(f"F1-Score:  {f1*100:.2f}%")
        logger.info(f"AUC-ROC:   {auc_roc:.4f}")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
        logger.info(f"  FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")

        # Classification Report
        logger.info(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Move', 'Price Up']))

        return metrics

    def get_feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: list,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Retorna importância das features

        Args:
            model: Modelo Random Forest treinado
            feature_names: Nomes das features
            top_n: Número de features mais importantes

        Returns:
            DataFrame com importâncias
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE IMPORTANCE - TOP {top_n}")
        logger.info("="*60)

        # Obter importâncias
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Top N
        top_features = feature_imp_df.head(top_n)

        logger.info("\nTop features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']:<30} {row['importance']:.6f}")

        return feature_imp_df

    def save_model(
        self,
        model,
        model_name: str,
        metrics: Dict,
        feature_importance: pd.DataFrame,
        metadata: Optional[Dict] = None
    ):
        """
        Salva modelo treinado e metadados

        Args:
            model: Modelo treinado
            model_name: Nome do modelo
            metrics: Métricas de avaliação
            feature_importance: DataFrame com importâncias
            metadata: Metadados adicionais
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{timestamp}"

        # Salvar modelo (pickle)
        model_path = self.models_dir / f"{base_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"[SALVO] Modelo: {model_path}")

        # Salvar métricas
        metrics_data = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()},
            'metadata': metadata or {}
        }

        metrics_path = self.models_dir / f"{base_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"[SALVO] Métricas: {metrics_path}")

        # Salvar feature importance
        fi_path = self.models_dir / f"{base_name}_feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        logger.info(f"[SALVO] Feature Importance: {fi_path}")

    def load_model(self, model_path: str):
        """
        Carrega modelo salvo

        Args:
            model_path: Caminho para o arquivo .pkl

        Returns:
            Modelo carregado
        """
        logger.info(f"Carregando modelo de {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("[OK] Modelo carregado")
        return model


def main():
    """
    Pipeline completo de treinamento
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )

    # Criar trainer
    trainer = ModelTrainer()

    # 1. Carregar dataset
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "ml_dataset_R100_1m.pkl"

    X, y = trainer.load_ml_dataset(dataset_path)

    # 2. Split train/test (temporal)
    X_train, X_test, y_train, y_test = trainer.split_train_test(
        X, y,
        test_size=0.2,
        time_aware=True
    )

    # 3. Treinar Random Forest
    rf_model = trainer.train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=20,
        min_samples_split=20,
        class_weight='balanced'
    )

    # 4. Avaliar modelo
    metrics = trainer.evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # 5. Feature Importance
    feature_importance = trainer.get_feature_importance(
        rf_model,
        X.columns.tolist(),
        top_n=20
    )

    # 6. Salvar modelo
    metadata = {
        'dataset': 'R_100_1m',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns),
        'prediction_horizon': '15min',
        'threshold': '0.3%'
    }

    trainer.save_model(
        rf_model,
        'random_forest_R100',
        metrics,
        feature_importance,
        metadata
    )

    # Resumo final
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO")
    print("="*60)
    print(f"Modelo: Random Forest")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("\nModelo salvo em: backend/ml/models/")
    print("="*60)


if __name__ == "__main__":
    main()

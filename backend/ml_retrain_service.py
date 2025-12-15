"""
Sistema de Retreinamento Automático de Modelos ML

Features:
- Retreinamento periódico (semanal/mensal)
- Avaliação automática de performance
- Versionamento de modelos
- Rollback automático se performance degradar
- Logging detalhado de métricas
- Backup de modelos antigos
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ml_predictor import MLPredictor

logger = logging.getLogger(__name__)


class ModelVersion:
    """Representa uma versão de modelo ML"""

    def __init__(
        self,
        version: str,
        created_at: str,
        metrics: Dict[str, float],
        model_path: str,
        status: str = "active"
    ):
        self.version = version
        self.created_at = created_at
        self.metrics = metrics
        self.model_path = model_path
        self.status = status  # active, archived, rolled_back

    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'created_at': self.created_at,
            'metrics': self.metrics,
            'model_path': self.model_path,
            'status': self.status
        }


class MLRetrainService:
    """
    Serviço de Retreinamento Automático de Modelos ML

    - Coleta dados de trades históricos
    - Retreina modelo XGBoost
    - Avalia performance (accuracy, precision, recall, f1)
    - Compara com modelo atual
    - Deploy automático se performance melhorar
    - Rollback se performance degradar
    """

    def __init__(
        self,
        models_dir: str = "models",
        data_dir: str = "data/training",
        min_samples: int = 1000,
        min_accuracy_threshold: float = 0.60,
        min_improvement_pct: float = 2.0,  # Mínimo 2% de melhoria
        retrain_interval_days: int = 7,
        backup_retention_days: int = 30
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.min_samples = min_samples
        self.min_accuracy_threshold = min_accuracy_threshold
        self.min_improvement_pct = min_improvement_pct
        self.retrain_interval_days = retrain_interval_days
        self.backup_retention_days = backup_retention_days

        # Criar diretórios
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "backups").mkdir(exist_ok=True)
        (self.models_dir / "versions").mkdir(exist_ok=True)

        # Carregar histórico de versões
        self.versions_file = self.models_dir / "versions" / "history.json"
        self.versions_history = self._load_versions_history()

        logger.info(f"MLRetrainService inicializado: {len(self.versions_history)} versões no histórico")

    def _load_versions_history(self) -> List[ModelVersion]:
        """Carrega histórico de versões de modelos"""
        if not self.versions_file.exists():
            return []

        try:
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                return [
                    ModelVersion(
                        version=v['version'],
                        created_at=v['created_at'],
                        metrics=v['metrics'],
                        model_path=v['model_path'],
                        status=v.get('status', 'archived')
                    )
                    for v in data
                ]
        except Exception as e:
            logger.error(f"Erro ao carregar histórico de versões: {e}")
            return []

    def _save_versions_history(self):
        """Salva histórico de versões"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(
                    [v.to_dict() for v in self.versions_history],
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Erro ao salvar histórico de versões: {e}")

    def _generate_version_name(self) -> str:
        """Gera nome de versão: v{MAJOR}.{MINOR}.{YYYYMMDD}"""
        # Última versão
        if not self.versions_history:
            return "v1.0.20241215"

        last_version = self.versions_history[-1].version
        parts = last_version.replace('v', '').split('.')
        major, minor = int(parts[0]), int(parts[1])

        # Incrementar minor version
        new_version = f"v{major}.{minor + 1}.{datetime.now().strftime('%Y%m%d')}"
        return new_version

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Verifica se deve retreinar o modelo

        Returns:
            (should_retrain, reason)
        """
        # Sem versões, precisa treinar
        if not self.versions_history:
            return True, "Nenhum modelo treinado ainda"

        # Última versão
        last_version = self.versions_history[-1]
        last_trained = datetime.fromisoformat(last_version.created_at)
        days_since_training = (datetime.now() - last_trained).days

        # Verificar intervalo
        if days_since_training >= self.retrain_interval_days:
            return True, f"Último treino há {days_since_training} dias (limite: {self.retrain_interval_days})"

        return False, f"Último treino há {days_since_training} dias (OK)"

    def collect_training_data(self) -> Optional[pd.DataFrame]:
        """
        Coleta dados de treino de trades históricos

        Returns:
            DataFrame com features e target (direction: UP=1, DOWN=0)
        """
        try:
            # Buscar arquivos CSV no data_dir
            csv_files = list(self.data_dir.glob("*.csv"))

            if not csv_files:
                logger.warning(f"Nenhum arquivo CSV encontrado em {self.data_dir}")
                return None

            # Carregar e concatenar
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                dfs.append(df)

            data = pd.concat(dfs, ignore_index=True)

            # Verificar quantidade mínima
            if len(data) < self.min_samples:
                logger.warning(f"Apenas {len(data)} amostras (mínimo: {self.min_samples})")
                return None

            logger.info(f"Coletados {len(data)} samples de {len(csv_files)} arquivos")
            return data

        except Exception as e:
            logger.error(f"Erro ao coletar dados de treino: {e}")
            return None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara features e target para treino

        Assumes df has columns:
        - close, high, low, volume (OHLCV data)
        - direction (target: 'UP' or 'DOWN')
        """
        # Calcular features técnicas
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()

        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)

        # Target: UP=1, DOWN=0
        df['target'] = (df['direction'] == 'UP').astype(int)

        # Features selecionadas
        feature_cols = [
            'rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'atr', 'volume_ratio',
            'price_change_1', 'price_change_5', 'price_change_10'
        ]

        # Drop NaN
        df = df.dropna()

        X = df[feature_cols].values
        y = df['target'].values

        logger.info(f"Features preparadas: {X.shape}, Target: {y.shape}")
        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """
        Treina novo modelo XGBoost e avalia performance

        Returns:
            (model, metrics)
        """
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # XGBoost Classifier
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1
        )

        # Treinar
        logger.info("Treinando modelo XGBoost...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Avaliar
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'samples_train': int(len(X_train)),
            'samples_test': int(len(X_test)),
            'positive_rate': float(y.mean())
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        logger.info(f"Modelo treinado - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

        return model, metrics

    def should_deploy_model(self, new_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Decide se deve fazer deploy do novo modelo

        Critérios:
        - Accuracy >= threshold mínimo
        - Melhoria >= X% vs modelo atual
        """
        accuracy = new_metrics['accuracy']

        # Verificar threshold mínimo
        if accuracy < self.min_accuracy_threshold:
            return False, f"Accuracy {accuracy:.2%} abaixo do mínimo ({self.min_accuracy_threshold:.2%})"

        # Se não tem modelo atual, fazer deploy
        if not self.versions_history:
            return True, "Primeiro modelo, fazendo deploy"

        # Comparar com modelo atual
        current_version = self.versions_history[-1]
        current_accuracy = current_version.metrics.get('accuracy', 0)

        improvement_pct = ((accuracy - current_accuracy) / current_accuracy) * 100

        if improvement_pct >= self.min_improvement_pct:
            return True, f"Melhoria de {improvement_pct:.2f}% (mínimo: {self.min_improvement_pct}%)"
        else:
            return False, f"Melhoria de {improvement_pct:.2f}% insuficiente (mínimo: {self.min_improvement_pct}%)"

    def backup_current_model(self):
        """Faz backup do modelo atual antes de substituir"""
        current_model_path = self.models_dir / "xgboost_model.pkl"

        if not current_model_path.exists():
            logger.info("Nenhum modelo atual para backup")
            return

        # Backup com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.models_dir / "backups" / f"xgboost_model_{timestamp}.pkl"

        shutil.copy2(current_model_path, backup_path)
        logger.info(f"Backup do modelo atual salvo: {backup_path}")

        # Limpar backups antigos
        self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        """Remove backups com mais de X dias"""
        backups_dir = self.models_dir / "backups"
        cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)

        for backup_file in backups_dir.glob("*.pkl"):
            file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                backup_file.unlink()
                logger.info(f"Backup antigo removido: {backup_file}")

    def deploy_model(self, model: xgb.XGBClassifier, metrics: Dict[str, float]) -> str:
        """
        Faz deploy do novo modelo

        Returns:
            version_name
        """
        # Gerar versão
        version_name = self._generate_version_name()

        # Backup do modelo atual
        self.backup_current_model()

        # Salvar novo modelo
        model_path = self.models_dir / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Salvar versão
        version_path = self.models_dir / "versions" / f"{version_name}.pkl"
        shutil.copy2(model_path, version_path)

        # Registrar versão
        new_version = ModelVersion(
            version=version_name,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            model_path=str(version_path),
            status='active'
        )

        # Marcar versões antigas como archived
        for v in self.versions_history:
            v.status = 'archived'

        self.versions_history.append(new_version)
        self._save_versions_history()

        logger.info(f"✅ Modelo {version_name} deployed com sucesso!")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.4f}")

        return version_name

    def rollback_to_version(self, version_name: str) -> bool:
        """
        Faz rollback para uma versão específica
        """
        # Buscar versão
        target_version = None
        for v in self.versions_history:
            if v.version == version_name:
                target_version = v
                break

        if not target_version:
            logger.error(f"Versão {version_name} não encontrada")
            return False

        # Backup do modelo atual
        self.backup_current_model()

        # Restaurar versão
        model_path = self.models_dir / "xgboost_model.pkl"
        shutil.copy2(target_version.model_path, model_path)

        # Atualizar status
        for v in self.versions_history:
            v.status = 'archived'
        target_version.status = 'active'
        self._save_versions_history()

        logger.info(f"✅ Rollback para {version_name} realizado com sucesso")
        return True

    def execute_retrain(self, force: bool = False) -> Dict:
        """
        Executa o processo completo de retreinamento

        Returns:
            result dict com status e detalhes
        """
        result = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'action': None,
            'version': None,
            'metrics': None,
            'message': None
        }

        try:
            # Verificar se deve retreinar
            if not force:
                should, reason = self.should_retrain()
                if not should:
                    result['action'] = 'skipped'
                    result['message'] = f"Retreinamento não necessário: {reason}"
                    return result

            # Coletar dados
            logger.info("📊 Coletando dados de treino...")
            df = self.collect_training_data()
            if df is None:
                result['message'] = "Dados insuficientes para retreinamento"
                return result

            # Preparar features
            logger.info("🔧 Preparando features...")
            X, y = self.prepare_features(df)

            # Treinar modelo
            logger.info("🤖 Treinando novo modelo...")
            model, metrics = self.train_model(X, y)

            # Verificar se deve fazer deploy
            should_deploy, deploy_reason = self.should_deploy_model(metrics)

            if should_deploy:
                logger.info(f"✅ {deploy_reason}")
                version = self.deploy_model(model, metrics)

                result['success'] = True
                result['action'] = 'deployed'
                result['version'] = version
                result['metrics'] = metrics
                result['message'] = f"Modelo {version} deployed: {deploy_reason}"
            else:
                logger.warning(f"⚠️ {deploy_reason}")
                result['action'] = 'rejected'
                result['metrics'] = metrics
                result['message'] = f"Modelo não deployed: {deploy_reason}"

        except Exception as e:
            logger.error(f"❌ Erro no retreinamento: {e}", exc_info=True)
            result['message'] = f"Erro: {str(e)}"

        return result

    def get_versions_history(self) -> List[Dict]:
        """Retorna histórico de versões"""
        return [v.to_dict() for v in self.versions_history]

    def get_current_version(self) -> Optional[Dict]:
        """Retorna versão ativa atual"""
        for v in reversed(self.versions_history):
            if v.status == 'active':
                return v.to_dict()
        return None


# Singleton global
_retrain_service = None

def get_retrain_service() -> MLRetrainService:
    """Retorna instância singleton do MLRetrainService"""
    global _retrain_service
    if _retrain_service is None:
        _retrain_service = MLRetrainService()
    return _retrain_service

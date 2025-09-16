"""
Advanced Model Ensemble - Sistema Avançado de Ensemble de Modelos
Combina múltiplos modelos de ML/DL para predições mais robustas e precisas.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"

class PredictionType(Enum):
    PRICE_DIRECTION = "price_direction"
    PRICE_TARGET = "price_target"
    VOLATILITY = "volatility"
    SIGNAL_STRENGTH = "signal_strength"

@dataclass
class ModelPrediction:
    """Predição de um modelo individual"""
    model_type: ModelType
    prediction_type: PredictionType
    value: float
    confidence: float
    timestamp: datetime
    features_used: List[str]
    model_version: str

@dataclass
class EnsemblePrediction:
    """Predição do ensemble"""
    final_prediction: float
    confidence: float
    individual_predictions: List[ModelPrediction]
    weight_distribution: Dict[str, float]
    consensus_level: float
    prediction_type: PredictionType
    timestamp: datetime

@dataclass
class ModelPerformance:
    """Performance de um modelo"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    win_rate: float
    total_predictions: int
    last_updated: datetime

class AdvancedLSTMModel(nn.Module):
    """LSTM avançado com atenção e múltiplas camadas"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.2, bidirectional: bool = True):
        super(AdvancedLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Camadas LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Camada de atenção
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout
        )

        # Camadas densas
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        # Ativações
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Atenção (self-attention)
        attended_out, attention_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attended_out = attended_out.transpose(0, 1)

        # Pegar último timestep
        final_out = attended_out[:, -1, :]

        # Camadas densas
        out = self.relu(self.fc1(final_out))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.fc3(out))

        return out

class TransformerModel(nn.Module):
    """Modelo Transformer para séries temporais"""

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Camadas de saída
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)

        # Projeção da entrada
        x = self.input_projection(x)

        # Adicionar encoding posicional
        if seq_len <= self.positional_encoding.size(1):
            x += self.positional_encoding[:, :seq_len, :].to(x.device)

        # Transformer
        transformer_out = self.transformer(x)

        # Pegar último timestep
        out = transformer_out[:, -1, :]

        # Camadas de saída
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))

        return out

class PatternCNN(nn.Module):
    """CNN para detecção de padrões em séries temporais"""

    def __init__(self, input_channels: int, sequence_length: int):
        super(PatternCNN, self).__init__()

        # Camadas convolucionais
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Camadas densas
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape para Conv1d: (batch, channels, sequence)
        x = x.transpose(1, 2)

        # Camadas convolucionais
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Camadas densas
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x

class AdvancedModelEnsemble:
    """Sistema avançado de ensemble de modelos de ML/DL"""

    def __init__(self, feature_size: int = 50, sequence_length: int = 60):
        self.feature_size = feature_size
        self.sequence_length = sequence_length

        # Modelos
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[ModelType, Any] = {}
        self.model_weights: Dict[ModelType, float] = {}
        self.model_performance: Dict[ModelType, ModelPerformance] = {}

        # Estado de treinamento
        self.is_trained = False
        self.training_data: List[Tuple[np.ndarray, float]] = []
        self.validation_data: List[Tuple[np.ndarray, float]] = []

        # Configurações
        self.min_confidence_threshold = 0.6
        self.weight_decay_factor = 0.95
        self.performance_window = 1000  # Últimas N predições para performance

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.prediction_lock = threading.Lock()

        # Device para PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"AdvancedModelEnsemble inicializado no device: {self.device}")

    async def initialize_models(self):
        """Inicializa todos os modelos do ensemble"""
        try:
            # LSTM avançado
            self.models[ModelType.LSTM] = AdvancedLSTMModel(
                input_size=self.feature_size,
                hidden_size=128,
                num_layers=3,
                bidirectional=True
            ).to(self.device)

            # Transformer
            self.models[ModelType.TRANSFORMER] = TransformerModel(
                input_size=self.feature_size,
                d_model=128,
                nhead=8,
                num_layers=6
            ).to(self.device)

            # CNN para padrões
            self.models[ModelType.CNN] = PatternCNN(
                input_channels=self.feature_size,
                sequence_length=self.sequence_length
            ).to(self.device)

            # Random Forest
            self.models[ModelType.RANDOM_FOREST] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # Gradient Boosting
            self.models[ModelType.GRADIENT_BOOSTING] = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )

            # Inicializar scalers
            for model_type in self.models.keys():
                if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                    self.scalers[model_type] = RobustScaler()
                else:
                    self.scalers[model_type] = StandardScaler()

            # Pesos iniciais iguais
            num_models = len(self.models)
            initial_weight = 1.0 / num_models
            for model_type in self.models.keys():
                self.model_weights[model_type] = initial_weight

            logger.info(f"Inicializados {num_models} modelos no ensemble")

        except Exception as e:
            logger.error(f"Erro ao inicializar modelos: {e}")
            raise

    async def add_training_data(self, features: np.ndarray, target: float):
        """Adiciona dados de treinamento"""
        if features.shape[0] != self.sequence_length or features.shape[1] != self.feature_size:
            logger.warning(f"Shape incorreto de features: {features.shape}")
            return

        self.training_data.append((features, target))

        # Limitar tamanho dos dados de treinamento
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-8000:]  # Manter últimos 8k

    async def train_ensemble(self, validation_split: float = 0.2):
        """Treina todos os modelos do ensemble"""
        if len(self.training_data) < 100:
            logger.warning("Dados insuficientes para treinamento")
            return False

        try:
            # Dividir dados
            split_idx = int(len(self.training_data) * (1 - validation_split))
            train_data = self.training_data[:split_idx]
            val_data = self.training_data[split_idx:]

            self.validation_data = val_data

            # Preparar dados
            X_train = np.array([data[0] for data in train_data])
            y_train = np.array([data[1] for data in train_data])

            X_val = np.array([data[0] for data in val_data])
            y_val = np.array([data[1] for data in val_data])

            # Treinar cada modelo
            training_tasks = []
            for model_type in self.models.keys():
                task = asyncio.create_task(
                    self._train_single_model(model_type, X_train, y_train, X_val, y_val)
                )
                training_tasks.append(task)

            # Aguardar conclusão de todos os treinamentos
            results = await asyncio.gather(*training_tasks, return_exceptions=True)

            # Verificar resultados
            successful_models = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    model_type = list(self.models.keys())[i]
                    logger.error(f"Erro ao treinar {model_type}: {result}")
                else:
                    successful_models += 1

            if successful_models > 0:
                self.is_trained = True
                await self._update_model_weights()
                logger.info(f"Ensemble treinado com {successful_models} modelos")
                return True
            else:
                logger.error("Falha no treinamento de todos os modelos")
                return False

        except Exception as e:
            logger.error(f"Erro no treinamento do ensemble: {e}")
            return False

    async def _train_single_model(self, model_type: ModelType,
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray):
        """Treina um modelo individual"""
        try:
            model = self.models[model_type]
            scaler = self.scalers[model_type]

            if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                # Modelos sklearn - features flattenadas
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)

                # Normalizar
                X_train_scaled = scaler.fit_transform(X_train_flat)
                X_val_scaled = scaler.transform(X_val_flat)

                # Treinar
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    model.fit,
                    X_train_scaled, y_train
                )

                # Validar
                y_pred = model.predict(X_val_scaled)

            else:
                # Modelos PyTorch
                X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                y_train_tensor = torch.FloatTensor(y_train).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)

                # Normalizar (por batch)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)

                scaler.fit(X_train_flat)

                # Treinar modelo PyTorch
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                criterion = nn.MSELoss()

                model.train()
                num_epochs = 50
                batch_size = 32

                for epoch in range(num_epochs):
                    total_loss = 0
                    num_batches = (len(X_train_tensor) + batch_size - 1) // batch_size

                    for i in range(0, len(X_train_tensor), batch_size):
                        batch_X = X_train_tensor[i:i+batch_size]
                        batch_y = y_train_tensor[i:i+batch_size]

                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    if epoch % 10 == 0:
                        avg_loss = total_loss / num_batches
                        logger.debug(f"{model_type} Epoch {epoch}, Loss: {avg_loss:.4f}")

                # Validar
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val_tensor).squeeze().cpu().numpy()

            # Calcular métricas de performance
            performance = self._calculate_performance_metrics(y_val, y_pred, model_type)
            self.model_performance[model_type] = performance

            logger.info(f"{model_type} treinado - R²: {performance.r2_score:.3f}, MSE: {performance.mse:.4f}")

        except Exception as e:
            logger.error(f"Erro ao treinar modelo {model_type}: {e}")
            raise

    def _calculate_performance_metrics(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     model_type: ModelType) -> ModelPerformance:
        """Calcula métricas de performance"""
        try:
            # Métricas básicas de regressão
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Métricas de classificação (direção)
            y_true_direction = (y_true > 0.5).astype(int)
            y_pred_direction = (y_pred > 0.5).astype(int)

            accuracy = np.mean(y_true_direction == y_pred_direction)

            # Precision, Recall, F1
            tp = np.sum((y_true_direction == 1) & (y_pred_direction == 1))
            fp = np.sum((y_true_direction == 0) & (y_pred_direction == 1))
            fn = np.sum((y_true_direction == 1) & (y_pred_direction == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Sharpe ratio simplificado (para predictions binárias)
            returns = y_pred - 0.5  # Centro em zero
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

            # Win rate
            win_rate = np.mean(y_pred > 0.5)

            return ModelPerformance(
                model_type=model_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                mse=mse,
                mae=mae,
                r2_score=r2,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                total_predictions=len(y_true),
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return ModelPerformance(
                model_type=model_type,
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                mse=1.0, mae=1.0, r2_score=0.0, sharpe_ratio=0.0,
                win_rate=0.5, total_predictions=0,
                last_updated=datetime.now()
            )

    async def predict(self, features: np.ndarray,
                     prediction_type: PredictionType = PredictionType.PRICE_DIRECTION) -> EnsemblePrediction:
        """Gera predição do ensemble"""
        if not self.is_trained:
            raise ValueError("Ensemble não foi treinado ainda")

        if features.shape != (self.sequence_length, self.feature_size):
            raise ValueError(f"Shape incorreto: esperado {(self.sequence_length, self.feature_size)}, recebido {features.shape}")

        async with asyncio.Lock():
            try:
                # Coletar predições de todos os modelos
                prediction_tasks = []
                for model_type in self.models.keys():
                    task = asyncio.create_task(
                        self._predict_single_model(model_type, features, prediction_type)
                    )
                    prediction_tasks.append(task)

                individual_predictions = await asyncio.gather(*prediction_tasks)

                # Filtrar predições válidas
                valid_predictions = [p for p in individual_predictions if p is not None]

                if not valid_predictions:
                    raise ValueError("Nenhuma predição válida obtida")

                # Calcular predição final ponderada
                weighted_sum = 0.0
                total_weight = 0.0

                for pred in valid_predictions:
                    weight = self.model_weights.get(pred.model_type, 0.0)
                    weighted_sum += pred.value * weight * pred.confidence
                    total_weight += weight * pred.confidence

                if total_weight == 0:
                    final_prediction = np.mean([p.value for p in valid_predictions])
                    confidence = np.mean([p.confidence for p in valid_predictions])
                else:
                    final_prediction = weighted_sum / total_weight
                    confidence = total_weight / len(valid_predictions)

                # Calcular consenso
                predictions_array = np.array([p.value for p in valid_predictions])
                consensus_level = 1.0 - (np.std(predictions_array) / np.mean(predictions_array)) if np.mean(predictions_array) > 0 else 0.5

                # Distribuição de pesos
                weight_distribution = {
                    pred.model_type.value: self.model_weights.get(pred.model_type, 0.0)
                    for pred in valid_predictions
                }

                return EnsemblePrediction(
                    final_prediction=final_prediction,
                    confidence=min(confidence, 1.0),
                    individual_predictions=valid_predictions,
                    weight_distribution=weight_distribution,
                    consensus_level=consensus_level,
                    prediction_type=prediction_type,
                    timestamp=datetime.now()
                )

            except Exception as e:
                logger.error(f"Erro na predição do ensemble: {e}")
                raise

    async def _predict_single_model(self, model_type: ModelType,
                                  features: np.ndarray,
                                  prediction_type: PredictionType) -> Optional[ModelPrediction]:
        """Gera predição de um modelo individual"""
        try:
            model = self.models[model_type]
            scaler = self.scalers[model_type]

            if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                # Modelos sklearn
                features_flat = features.reshape(1, -1)
                features_scaled = scaler.transform(features_flat)

                prediction = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    model.predict,
                    features_scaled
                )
                value = float(prediction[0])

                # Confidence baseada na performance do modelo
                performance = self.model_performance.get(model_type)
                confidence = performance.accuracy if performance else 0.5

            else:
                # Modelos PyTorch
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    prediction = model(features_tensor).cpu().numpy()
                    value = float(prediction[0])

                    # Confidence baseada na performance
                    performance = self.model_performance.get(model_type)
                    confidence = performance.r2_score if performance and performance.r2_score > 0 else 0.5

            # Aplicar threshold de confiança
            if confidence < self.min_confidence_threshold:
                confidence *= 0.5  # Reduzir confiança se abaixo do threshold

            return ModelPrediction(
                model_type=model_type,
                prediction_type=prediction_type,
                value=value,
                confidence=confidence,
                timestamp=datetime.now(),
                features_used=[f"feature_{i}" for i in range(self.feature_size)],
                model_version="1.0"
            )

        except Exception as e:
            logger.error(f"Erro na predição do modelo {model_type}: {e}")
            return None

    async def _update_model_weights(self):
        """Atualiza pesos dos modelos baseado na performance"""
        try:
            if not self.model_performance:
                return

            # Calcular score combinado para cada modelo
            scores = {}
            for model_type, performance in self.model_performance.items():
                # Score combinado: accuracy + r2_score + f1_score - mse
                score = (
                    performance.accuracy * 0.3 +
                    max(performance.r2_score, 0) * 0.3 +
                    performance.f1_score * 0.2 +
                    (1 - min(performance.mse, 1)) * 0.2
                )
                scores[model_type] = max(score, 0.1)  # Mínimo 0.1

            # Normalizar pesos
            total_score = sum(scores.values())
            if total_score > 0:
                for model_type in self.model_weights:
                    self.model_weights[model_type] = scores.get(model_type, 0.1) / total_score

            logger.info(f"Pesos atualizados: {self.model_weights}")

        except Exception as e:
            logger.error(f"Erro ao atualizar pesos: {e}")

    async def update_performance(self, prediction: EnsemblePrediction, actual_result: float):
        """Atualiza performance dos modelos com resultado real"""
        try:
            for individual_pred in prediction.individual_predictions:
                model_type = individual_pred.model_type

                # Calcular erro
                error = abs(individual_pred.value - actual_result)

                # Atualizar performance (média móvel simples)
                if model_type in self.model_performance:
                    current_perf = self.model_performance[model_type]

                    # Atualizar métricas
                    alpha = 0.1  # Fator de learning
                    current_perf.mae = current_perf.mae * (1 - alpha) + error * alpha
                    current_perf.accuracy = current_perf.accuracy * (1 - alpha) + (1 - error) * alpha
                    current_perf.total_predictions += 1
                    current_perf.last_updated = datetime.now()

            # Atualizar pesos baseado na nova performance
            await self._update_model_weights()

        except Exception as e:
            logger.error(f"Erro ao atualizar performance: {e}")

    async def get_ensemble_status(self) -> Dict[str, Any]:
        """Obtém status completo do ensemble"""
        return {
            "is_trained": self.is_trained,
            "num_models": len(self.models),
            "training_samples": len(self.training_data),
            "validation_samples": len(self.validation_data),
            "model_weights": self.model_weights,
            "model_performance": {
                model_type.value: asdict(perf)
                for model_type, perf in self.model_performance.items()
            },
            "device": str(self.device),
            "feature_size": self.feature_size,
            "sequence_length": self.sequence_length
        }

    async def save_ensemble(self, filepath: str):
        """Salva o ensemble treinado"""
        try:
            ensemble_data = {
                "feature_size": self.feature_size,
                "sequence_length": self.sequence_length,
                "model_weights": self.model_weights,
                "model_performance": {
                    model_type.value: asdict(perf)
                    for model_type, perf in self.model_performance.items()
                },
                "is_trained": self.is_trained
            }

            # Salvar dados do ensemble
            with open(f"{filepath}_ensemble.json", 'w') as f:
                json.dump(ensemble_data, f, default=str)

            # Salvar modelos individuais
            for model_type, model in self.models.items():
                model_path = f"{filepath}_{model_type.value}"

                if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                    joblib.dump(model, f"{model_path}.joblib")
                else:
                    torch.save(model.state_dict(), f"{model_path}.pt")

                # Salvar scaler
                joblib.dump(self.scalers[model_type], f"{model_path}_scaler.joblib")

            logger.info(f"Ensemble salvo em {filepath}")

        except Exception as e:
            logger.error(f"Erro ao salvar ensemble: {e}")

    async def load_ensemble(self, filepath: str):
        """Carrega ensemble salvo"""
        try:
            # Carregar dados do ensemble
            with open(f"{filepath}_ensemble.json", 'r') as f:
                ensemble_data = json.load(f)

            self.feature_size = ensemble_data["feature_size"]
            self.sequence_length = ensemble_data["sequence_length"]
            self.model_weights = ensemble_data["model_weights"]
            self.is_trained = ensemble_data["is_trained"]

            # Recriar modelos
            await self.initialize_models()

            # Carregar modelos individuais
            for model_type in self.models.keys():
                model_path = f"{filepath}_{model_type.value}"

                if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                    self.models[model_type] = joblib.load(f"{model_path}.joblib")
                else:
                    self.models[model_type].load_state_dict(
                        torch.load(f"{model_path}.pt", map_location=self.device)
                    )

                # Carregar scaler
                self.scalers[model_type] = joblib.load(f"{model_path}_scaler.joblib")

            logger.info(f"Ensemble carregado de {filepath}")

        except Exception as e:
            logger.error(f"Erro ao carregar ensemble: {e}")

    async def shutdown(self):
        """Encerra o ensemble"""
        self.thread_pool.shutdown(wait=True)
        logger.info("AdvancedModelEnsemble encerrado")
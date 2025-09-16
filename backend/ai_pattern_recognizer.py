"""
🤖 AI Trading Bot - Pattern Recognition System
Advanced LSTM neural network for tick pattern analysis

Author: Claude Code
Created: 2025-01-16
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Attention, Input, Bidirectional, TimeDistributed,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import time
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from queue import Queue

from .tick_data_collector import TickData
from .feature_engine import FeatureEngine, MarketFeatures

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for LSTM model"""
    sequence_length: int = 60  # Number of ticks to look back
    feature_dim: int = 22     # Number of features per tick
    lstm_units: int = 128     # LSTM hidden units
    attention_heads: int = 8  # Multi-head attention
    dropout_rate: float = 0.3 # Dropout rate
    l2_reg: float = 0.001    # L2 regularization
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2

@dataclass
class PredictionResult:
    """Result from model prediction"""
    signal_strength: float    # [-1, 1] sell to buy signal
    confidence_score: float   # [0, 1] prediction confidence
    probability_up: float     # Probability of price going up
    probability_down: float   # Probability of price going down
    timestamp: float
    features_used: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrainingMetrics:
    """Training session metrics"""
    accuracy: float
    val_accuracy: float
    loss: float
    val_loss: float
    auc_score: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    epochs_trained: int
    model_version: str

class AttentionLSTMModel:
    """Advanced LSTM with Attention mechanism for pattern recognition"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False
        self.version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        logger.info(f"AttentionLSTMModel initialized with version {self.version}")

    def build_model(self) -> Model:
        """Build the LSTM model with attention mechanism"""

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.config.feature_dim))

        # First Bidirectional LSTM layer
        lstm1 = Bidirectional(
            LSTM(self.config.lstm_units,
                 return_sequences=True,
                 kernel_regularizer=l2(self.config.l2_reg))
        )(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(self.config.dropout_rate)(lstm1)

        # Second Bidirectional LSTM layer
        lstm2 = Bidirectional(
            LSTM(self.config.lstm_units // 2,
                 return_sequences=True,
                 kernel_regularizer=l2(self.config.l2_reg))
        )(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(self.config.dropout_rate)(lstm2)

        # Multi-Head Attention layer
        attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=self.config.lstm_units
        )(lstm2, lstm2)

        # Add & Norm (Residual connection)
        attention_output = Add()([lstm2, attention])
        attention_output = LayerNormalization()(attention_output)

        # Global Average Pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)

        # Dense layers for classification
        dense1 = Dense(64, activation='relu',
                      kernel_regularizer=l2(self.config.l2_reg))(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config.dropout_rate)(dense1)

        dense2 = Dense(32, activation='relu',
                      kernel_regularizer=l2(self.config.l2_reg))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(self.config.dropout_rate / 2)(dense2)

        # Output layer (binary classification)
        outputs = Dense(1, activation='sigmoid')(dense2)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        self.model = model
        logger.info(f"Model built successfully. Total parameters: {model.count_params():,}")

        return model

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""

        if len(features) < self.config.sequence_length:
            logger.warning(f"Not enough data for sequences. Need {self.config.sequence_length}, got {len(features)}")
            return np.array([]), np.array([])

        X_sequences = []
        y_sequences = []

        for i in range(len(features) - self.config.sequence_length):
            # Get sequence of features
            sequence = features[i:i + self.config.sequence_length]
            target = targets[i + self.config.sequence_length]

            X_sequences.append(sequence)
            y_sequences.append(target)

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        logger.info(f"Prepared {len(X_sequences)} sequences of length {self.config.sequence_length}")

        return X_sequences, y_sequences

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingMetrics:
        """Train the LSTM model"""

        start_time = time.time()

        try:
            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(X, y)

            if len(X_seq) == 0:
                raise ValueError("No sequences prepared for training")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y_seq
            )

            logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

            # Build model if not exists
            if self.model is None:
                self.build_model()

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.patience // 2,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'models/lstm_model_{self.version}.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)

            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Calculate training time
            training_time = time.time() - start_time

            # Evaluate model
            val_predictions = self.model.predict(X_val)
            val_predictions_binary = (val_predictions > 0.5).astype(int)

            # Calculate metrics
            accuracy = self.history.history['accuracy'][-1]
            val_accuracy = self.history.history['val_accuracy'][-1]
            loss = self.history.history['loss'][-1]
            val_loss = self.history.history['val_loss'][-1]

            auc_score = roc_auc_score(y_val, val_predictions)

            # Classification report
            report = classification_report(y_val, val_predictions_binary, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            metrics = TrainingMetrics(
                accuracy=accuracy,
                val_accuracy=val_accuracy,
                loss=loss,
                val_loss=val_loss,
                auc_score=auc_score,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                training_time=training_time,
                epochs_trained=len(self.history.history['loss']),
                model_version=self.version
            )

            self.is_trained = True

            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Final metrics - Accuracy: {val_accuracy:.4f}, AUC: {auc_score:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, features_sequence: np.ndarray) -> PredictionResult:
        """Make prediction on feature sequence"""

        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if len(features_sequence.shape) == 2:
            # Single sequence, add batch dimension
            features_sequence = np.expand_dims(features_sequence, axis=0)

        if features_sequence.shape[1] != self.config.sequence_length:
            raise ValueError(f"Expected sequence length {self.config.sequence_length}, got {features_sequence.shape[1]}")

        try:
            # Get prediction probability
            probability = float(self.model.predict(features_sequence, verbose=0)[0][0])

            # Convert to signal strength [-1, 1]
            signal_strength = (probability - 0.5) * 2

            # Confidence score (how far from 0.5)
            confidence_score = abs(probability - 0.5) * 2

            result = PredictionResult(
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                probability_up=probability,
                probability_down=1.0 - probability,
                timestamp=time.time(),
                features_used=features_sequence.shape[2]
            )

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save_model(self, filepath: str) -> bool:
        """Save model and configuration"""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False

            # Save model
            self.model.save(filepath)

            # Save configuration
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)

            # Save training history if available
            if self.history:
                history_path = filepath.replace('.h5', '_history.json')
                with open(history_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    history_dict = {}
                    for key, values in self.history.history.items():
                        history_dict[key] = [float(v) for v in values]
                    json.dump(history_dict, f, indent=2)

            logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model and configuration"""
        try:
            # Load model
            self.model = load_model(filepath)

            # Load configuration
            config_path = filepath.replace('.h5', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    self.config = ModelConfig(**config_dict)

            # Load training history if available
            history_path = filepath.replace('.h5', '_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_dict = json.load(f)
                    # Convert back to format expected by Keras
                    self.history = type('History', (), {'history': history_dict})()

            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class ModelEnsemble:
    """Ensemble of multiple LSTM models for robust predictions"""

    def __init__(self):
        self.models: List[AttentionLSTMModel] = []
        self.weights: List[float] = []

    def add_model(self, model: AttentionLSTMModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"Added model to ensemble. Total models: {len(self.models)}")

    def predict(self, features_sequence: np.ndarray) -> PredictionResult:
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        confidences = []

        for model, weight in zip(self.models, self.weights):
            result = model.predict(features_sequence)
            predictions.append(result.signal_strength * weight)
            confidences.append(result.confidence_score * weight)

        # Weighted average
        total_weight = sum(self.weights)
        avg_signal = sum(predictions) / total_weight
        avg_confidence = sum(confidences) / total_weight

        # Convert back to probability
        probability_up = (avg_signal + 1) / 2

        return PredictionResult(
            signal_strength=avg_signal,
            confidence_score=avg_confidence,
            probability_up=probability_up,
            probability_down=1.0 - probability_up,
            timestamp=time.time(),
            features_used=features_sequence.shape[2] if len(features_sequence.shape) > 2 else 0
        )

class AIPatternRecognizer:
    """Main AI Pattern Recognition System"""

    def __init__(self, feature_engine: FeatureEngine, config: Optional[ModelConfig] = None):
        self.feature_engine = feature_engine
        self.config = config or ModelConfig()
        self.model = AttentionLSTMModel(self.config)
        self.ensemble = ModelEnsemble()

        # Real-time prediction
        self.sequence_buffer = []
        self.is_predicting = False

        # Training state
        self.training_thread = None
        self.is_training = False
        self.training_progress = 0.0

        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.total_prediction_time = 0.0

        logger.info("AIPatternRecognizer initialized")

    def train_model(self, ticks: List[TickData], retrain: bool = False) -> TrainingMetrics:
        """Train the main LSTM model"""

        if self.is_training:
            raise RuntimeError("Training already in progress")

        try:
            self.is_training = True
            logger.info(f"Starting model training with {len(ticks)} ticks")

            # Prepare training data using feature engine
            X, y = self.feature_engine.prepare_training_data(ticks)

            if len(X) == 0:
                raise ValueError("No training data prepared")

            # Train model
            if retrain or not self.model.is_trained:
                metrics = self.model.train(X, y)
                logger.info(f"Training completed with accuracy: {metrics.val_accuracy:.4f}")
                return metrics
            else:
                raise ValueError("Model already trained. Use retrain=True to retrain")

        finally:
            self.is_training = False

    def train_model_async(self, ticks: List[TickData], callback: Optional[callable] = None):
        """Train model asynchronously"""

        def training_worker():
            try:
                metrics = self.train_model(ticks)
                if callback:
                    callback(metrics, None)
            except Exception as e:
                if callback:
                    callback(None, e)

        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()

    def add_tick_for_prediction(self, tick: TickData):
        """Add tick to real-time prediction buffer"""

        # Extract features
        features = self.feature_engine.extract_features(tick)
        if features is None:
            return

        # Normalize features
        normalized_features = self.feature_engine.normalize_features(features)

        # Add to sequence buffer
        self.sequence_buffer.append(normalized_features)

        # Keep only required sequence length
        if len(self.sequence_buffer) > self.config.sequence_length:
            self.sequence_buffer.pop(0)

    def predict_realtime(self) -> Optional[PredictionResult]:
        """Make real-time prediction from current buffer"""

        if not self.model.is_trained:
            return None

        if len(self.sequence_buffer) < self.config.sequence_length:
            return None

        try:
            start_time = time.time()

            # Prepare sequence
            sequence = np.array(self.sequence_buffer[-self.config.sequence_length:])

            # Make prediction
            if self.ensemble.models:
                result = self.ensemble.predict(sequence)
            else:
                result = self.model.predict(sequence)

            # Update performance metrics
            prediction_time = time.time() - start_time
            self.total_prediction_time += prediction_time
            self.predictions_made += 1

            return result

        except Exception as e:
            logger.error(f"Error in real-time prediction: {e}")
            return None

    def evaluate_predictions(self, test_ticks: List[TickData]) -> Dict[str, float]:
        """Evaluate model performance on test data"""

        if not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation")

        try:
            # Prepare test data
            X_test, y_test = self.feature_engine.prepare_training_data(test_ticks)

            if len(X_test) == 0:
                raise ValueError("No test data prepared")

            # Prepare sequences
            X_seq, y_seq = self.model.prepare_sequences(X_test, y_test)

            # Make predictions
            predictions = self.model.model.predict(X_seq)
            predictions_binary = (predictions > 0.5).astype(int)

            # Calculate metrics
            accuracy = np.mean(predictions_binary.flatten() == y_seq)
            auc_score = roc_auc_score(y_seq, predictions)

            # Additional metrics
            tp = np.sum((predictions_binary == 1) & (y_seq.reshape(-1, 1) == 1))
            fp = np.sum((predictions_binary == 1) & (y_seq.reshape(-1, 1) == 0))
            tn = np.sum((predictions_binary == 0) & (y_seq.reshape(-1, 1) == 0))
            fn = np.sum((predictions_binary == 0) & (y_seq.reshape(-1, 1) == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'total_predictions': len(y_seq)
            }

        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {}

    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""

        avg_prediction_time = (self.total_prediction_time / self.predictions_made
                             if self.predictions_made > 0 else 0)

        return {
            'is_trained': self.model.is_trained,
            'is_training': self.is_training,
            'model_version': self.model.version,
            'config': asdict(self.config),
            'predictions_made': self.predictions_made,
            'avg_prediction_time_ms': avg_prediction_time * 1000,
            'sequence_buffer_size': len(self.sequence_buffer),
            'ensemble_models': len(self.ensemble.models),
            'training_history_available': self.model.history is not None
        }

    def save_models(self, base_path: str = "models/") -> bool:
        """Save all models and configurations"""
        try:
            os.makedirs(base_path, exist_ok=True)

            # Save main model
            main_model_path = f"{base_path}main_model_{self.model.version}.h5"
            success = self.model.save_model(main_model_path)

            # Save ensemble models
            for i, model in enumerate(self.ensemble.models):
                ensemble_path = f"{base_path}ensemble_model_{i}_{model.version}.h5"
                model.save_model(ensemble_path)

            # Save recognizer state
            state_path = f"{base_path}recognizer_state.json"
            state = {
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'total_prediction_time': self.total_prediction_time,
                'model_version': self.model.version
            }

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Models saved to {base_path}")
            return success

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False

    def load_models(self, base_path: str = "models/") -> bool:
        """Load models and configurations"""
        try:
            # Find latest main model
            model_files = [f for f in os.listdir(base_path) if f.startswith('main_model_') and f.endswith('.h5')]

            if not model_files:
                logger.warning("No main model found")
                return False

            # Load latest model
            latest_model = sorted(model_files)[-1]
            main_model_path = os.path.join(base_path, latest_model)

            success = self.model.load_model(main_model_path)

            # Load recognizer state if available
            state_path = os.path.join(base_path, 'recognizer_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.predictions_made = state.get('predictions_made', 0)
                    self.correct_predictions = state.get('correct_predictions', 0)
                    self.total_prediction_time = state.get('total_prediction_time', 0.0)

            logger.info(f"Models loaded from {base_path}")
            return success

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
"""
🤖 AI Trading Bot - Training Pipeline
Automated model training with cross-validation and performance tracking

Author: Claude Code
Created: 2025-01-16
"""

import numpy as np
import pandas as pd
import logging
import time
import os
import json
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

from .tick_data_collector import TickData
from .feature_engine import FeatureEngine
from .ai_pattern_recognizer import AIPatternRecognizer, ModelConfig, TrainingMetrics
from .data_storage import TimeSeriesDB

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data preparation
    min_training_samples: int = 10000
    validation_split: float = 0.2
    test_split: float = 0.1

    # Cross-validation
    cv_folds: int = 5
    use_time_series_cv: bool = True

    # Training schedule
    retrain_interval_hours: int = 24
    auto_retrain: bool = True
    retrain_threshold_accuracy: float = 0.6

    # Model selection
    max_models_to_keep: int = 5
    ensemble_size: int = 3

    # Performance monitoring
    performance_window_hours: int = 48
    min_prediction_confidence: float = 0.7

@dataclass
class TrainingSession:
    """Information about a training session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    symbol: str = ""
    total_samples: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    metrics: Optional[TrainingMetrics] = None
    cross_validation_scores: List[float] = None
    status: str = "starting"  # starting, training, validating, completed, failed
    error_message: str = ""

@dataclass
class ModelPerformance:
    """Real-time model performance tracking"""
    model_version: str
    accuracy_24h: float
    accuracy_7d: float
    total_predictions: int
    correct_predictions: int
    false_positives: int
    false_negatives: int
    avg_confidence: float
    last_updated: float

class CrossValidator:
    """Advanced cross-validation for time series data"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def time_series_split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time series aware cross-validation split"""

        if self.config.use_time_series_cv:
            # Use TimeSeriesSplit to respect temporal order
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            splits = []

            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]

                splits.append((
                    (X_train_fold, y_train_fold),
                    (X_val_fold, y_val_fold)
                ))

            return splits
        else:
            # Use stratified K-fold for better class balance
            skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            splits = []

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]

                splits.append((
                    (X_train_fold, y_train_fold),
                    (X_val_fold, y_val_fold)
                ))

            return splits

    def validate_model(self, model_class, config: ModelConfig, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation on model"""

        try:
            logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")

            splits = self.time_series_split(X, y)
            fold_scores = []
            fold_metrics = []

            for fold_idx, ((X_train, y_train), (X_val, y_val)) in enumerate(splits):
                logger.info(f"Training fold {fold_idx + 1}/{self.config.cv_folds}")

                # Create fresh model for this fold
                model = model_class(config)

                # Train on fold
                try:
                    metrics = model.train(X_train, y_train)

                    # Evaluate on validation set
                    X_val_seq, y_val_seq = model.prepare_sequences(X_val, y_val)

                    if len(X_val_seq) > 0:
                        val_predictions = model.model.predict(X_val_seq)
                        val_accuracy = np.mean((val_predictions > 0.5).astype(int).flatten() == y_val_seq)

                        fold_scores.append(val_accuracy)
                        fold_metrics.append(metrics)

                        logger.info(f"Fold {fold_idx + 1} accuracy: {val_accuracy:.4f}")
                    else:
                        logger.warning(f"No validation sequences for fold {fold_idx + 1}")

                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}")
                    continue

            if not fold_scores:
                return {'error': 'No successful folds completed'}

            # Calculate cross-validation statistics
            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)

            result = {
                'cv_scores': fold_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_confidence_interval': (cv_mean - 1.96 * cv_std, cv_mean + 1.96 * cv_std),
                'fold_metrics': [asdict(m) for m in fold_metrics],
                'n_folds_completed': len(fold_scores)
            }

            logger.info(f"Cross-validation completed: {cv_mean:.4f} ± {cv_std:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}

class ModelVersionManager:
    """Manage multiple model versions and performance tracking"""

    def __init__(self, base_path: str = "models/"):
        self.base_path = base_path
        self.models_registry = {}
        self.performance_tracker = {}

        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)

        # Load existing registry
        self.load_registry()

    def register_model(self, model_version: str, metrics: TrainingMetrics, cv_results: Dict[str, Any]):
        """Register a new trained model"""

        model_info = {
            'version': model_version,
            'created_at': time.time(),
            'training_metrics': asdict(metrics),
            'cv_results': cv_results,
            'model_path': f"{self.base_path}model_{model_version}.h5",
            'is_active': False,
            'performance_history': []
        }

        self.models_registry[model_version] = model_info
        self.save_registry()

        logger.info(f"Registered model {model_version} with accuracy {metrics.val_accuracy:.4f}")

    def activate_model(self, model_version: str):
        """Activate a specific model version"""

        if model_version not in self.models_registry:
            raise ValueError(f"Model version {model_version} not found")

        # Deactivate all models
        for version in self.models_registry:
            self.models_registry[version]['is_active'] = False

        # Activate selected model
        self.models_registry[model_version]['is_active'] = True
        self.save_registry()

        logger.info(f"Activated model {model_version}")

    def get_active_model(self) -> Optional[str]:
        """Get currently active model version"""

        for version, info in self.models_registry.items():
            if info['is_active']:
                return version
        return None

    def get_best_models(self, n: int = 3, metric: str = 'val_accuracy') -> List[str]:
        """Get N best models by specified metric"""

        model_scores = []

        for version, info in self.models_registry.items():
            if 'training_metrics' in info:
                score = info['training_metrics'].get(metric, 0)
                model_scores.append((version, score))

        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        return [version for version, _ in model_scores[:n]]

    def cleanup_old_models(self, keep_count: int = 5):
        """Remove old model files, keeping only the best ones"""

        try:
            best_models = self.get_best_models(keep_count)

            # Get all model versions
            all_versions = list(self.models_registry.keys())

            # Remove old models
            for version in all_versions:
                if version not in best_models:
                    model_info = self.models_registry[version]
                    model_path = model_info.get('model_path', '')

                    # Remove model file
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Removed old model file: {model_path}")

                    # Remove from registry
                    del self.models_registry[version]

            self.save_registry()
            logger.info(f"Cleaned up models, kept {len(best_models)} best models")

        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")

    def update_performance(self, model_version: str, accuracy: float, total_predictions: int):
        """Update real-time performance for a model"""

        if model_version not in self.models_registry:
            return

        performance_entry = {
            'timestamp': time.time(),
            'accuracy': accuracy,
            'total_predictions': total_predictions
        }

        self.models_registry[model_version]['performance_history'].append(performance_entry)

        # Keep only recent performance data (last 100 entries)
        history = self.models_registry[model_version]['performance_history']
        if len(history) > 100:
            self.models_registry[model_version]['performance_history'] = history[-100:]

        self.save_registry()

    def save_registry(self):
        """Save models registry to file"""
        try:
            registry_path = os.path.join(self.base_path, 'models_registry.json')
            with open(registry_path, 'w') as f:
                json.dump(self.models_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def load_registry(self):
        """Load models registry from file"""
        try:
            registry_path = os.path.join(self.base_path, 'models_registry.json')
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.models_registry = json.load(f)
                logger.info(f"Loaded {len(self.models_registry)} models from registry")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")

class TrainingPipeline:
    """Main training pipeline for AI model"""

    def __init__(self,
                 database: TimeSeriesDB,
                 feature_engine: FeatureEngine,
                 config: Optional[TrainingConfig] = None):

        self.database = database
        self.feature_engine = feature_engine
        self.config = config or TrainingConfig()

        # Components
        self.cross_validator = CrossValidator(self.config)
        self.version_manager = ModelVersionManager()

        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False

        # Auto-retrain scheduler
        self.last_training_time = 0
        self.auto_retrain_enabled = self.config.auto_retrain

        # Performance monitoring
        self.performance_monitors = {}

        logger.info("TrainingPipeline initialized")

    def prepare_training_data(self, symbol: str, hours: int = 168) -> Tuple[np.ndarray, np.ndarray, List[TickData]]:
        """Prepare training data from database"""

        try:
            # Get recent tick data
            end_time = time.time()
            start_time = end_time - (hours * 3600)

            ticks = await self.database.get_ticks(symbol, start_time, end_time)

            if len(ticks) < self.config.min_training_samples:
                raise ValueError(f"Insufficient data: {len(ticks)} ticks, need {self.config.min_training_samples}")

            logger.info(f"Loaded {len(ticks)} ticks for {symbol}")

            # Prepare features and targets
            X, y = self.feature_engine.prepare_training_data(ticks)

            if len(X) < self.config.min_training_samples:
                raise ValueError(f"Insufficient processed samples: {len(X)}")

            logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")

            return X, y, ticks

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def train_model(self, symbol: str, model_config: Optional[ModelConfig] = None) -> TrainingSession:
        """Train a new model"""

        if self.is_training:
            raise RuntimeError("Training already in progress")

        # Create training session
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = TrainingSession(
            session_id=session_id,
            start_time=time.time(),
            symbol=symbol,
            status="starting"
        )

        self.current_session = session

        try:
            self.is_training = True
            logger.info(f"Starting training session {session_id} for {symbol}")

            # Prepare data
            session.status = "preparing_data"
            X, y, ticks = self.prepare_training_data(symbol)

            session.total_samples = len(X)

            # Split data
            test_size = int(len(X) * self.config.test_split)
            val_size = int(len(X) * self.config.validation_split)
            train_size = len(X) - test_size - val_size

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            session.training_samples = len(X_train)
            session.validation_samples = len(X_val)
            session.test_samples = len(X_test)

            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Cross-validation
            session.status = "cross_validation"
            logger.info("Starting cross-validation")

            cv_results = self.cross_validator.validate_model(
                AIPatternRecognizer(self.feature_engine).__class__,
                model_config or ModelConfig(),
                X_train,
                y_train
            )

            if 'error' in cv_results:
                raise ValueError(f"Cross-validation failed: {cv_results['error']}")

            session.cross_validation_scores = cv_results['cv_scores']

            # Train final model
            session.status = "training"
            logger.info("Training final model")

            recognizer = AIPatternRecognizer(self.feature_engine, model_config)

            # Combine train and validation data for final training
            X_final = np.vstack([X_train, X_val])
            y_final = np.hstack([y_train, y_val])

            metrics = recognizer.train_model(ticks[:train_size + val_size])

            # Test model performance
            session.status = "testing"
            if len(X_test) > 0:
                test_results = recognizer.evaluate_predictions(ticks[train_size + val_size:])
                logger.info(f"Test accuracy: {test_results.get('accuracy', 0):.4f}")

            # Save model
            model_version = recognizer.model.version
            recognizer.save_models()

            # Register model
            self.version_manager.register_model(model_version, metrics, cv_results)

            # Activate if it's the best model
            best_models = self.version_manager.get_best_models(1)
            if not best_models or model_version == best_models[0]:
                self.version_manager.activate_model(model_version)

            # Complete session
            session.status = "completed"
            session.end_time = time.time()
            session.metrics = metrics

            self.last_training_time = time.time()

            logger.info(f"Training session {session_id} completed successfully")
            logger.info(f"Model {model_version} - Accuracy: {metrics.val_accuracy:.4f}, AUC: {metrics.auc_score:.4f}")

            return session

        except Exception as e:
            session.status = "failed"
            session.error_message = str(e)
            session.end_time = time.time()

            logger.error(f"Training session {session_id} failed: {e}")
            raise

        finally:
            self.is_training = False

    def train_model_async(self, symbol: str, model_config: Optional[ModelConfig] = None,
                         callback: Optional[Callable] = None):
        """Train model asynchronously"""

        def training_worker():
            try:
                session = self.train_model(symbol, model_config)
                if callback:
                    callback(session, None)
            except Exception as e:
                if callback:
                    callback(None, e)

        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()

    def should_retrain(self, symbol: str) -> bool:
        """Check if model should be retrained"""

        if not self.auto_retrain_enabled:
            return False

        # Check time since last training
        time_since_training = time.time() - self.last_training_time
        if time_since_training < (self.config.retrain_interval_hours * 3600):
            return False

        # Check current model performance
        active_model = self.version_manager.get_active_model()
        if active_model:
            model_info = self.version_manager.models_registry.get(active_model, {})
            performance_history = model_info.get('performance_history', [])

            # Check recent performance
            if performance_history:
                recent_performance = [p['accuracy'] for p in performance_history[-10:]]
                avg_recent_accuracy = np.mean(recent_performance)

                if avg_recent_accuracy < self.config.retrain_threshold_accuracy:
                    logger.info(f"Model performance below threshold: {avg_recent_accuracy:.4f}")
                    return True

        # Check data availability
        try:
            end_time = time.time()
            start_time = end_time - (24 * 3600)  # Last 24 hours

            recent_ticks = await self.database.get_ticks(symbol, start_time, end_time, limit=1000)

            if len(recent_ticks) > self.config.min_training_samples // 10:
                logger.info("Sufficient new data available for retraining")
                return True

        except Exception as e:
            logger.error(f"Error checking data availability: {e}")

        return False

    def auto_retrain_check(self, symbol: str):
        """Automatic retraining check (to be called periodically)"""

        if self.is_training:
            return

        if self.should_retrain(symbol):
            logger.info(f"Starting automatic retraining for {symbol}")
            self.train_model_async(symbol)

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""

        status = {
            'is_training': self.is_training,
            'current_session': asdict(self.current_session) if self.current_session else None,
            'last_training_time': self.last_training_time,
            'auto_retrain_enabled': self.auto_retrain_enabled,
            'models_in_registry': len(self.version_manager.models_registry),
            'active_model': self.version_manager.get_active_model()
        }

        # Add model performance if available
        active_model = self.version_manager.get_active_model()
        if active_model and active_model in self.version_manager.models_registry:
            model_info = self.version_manager.models_registry[active_model]
            if 'training_metrics' in model_info:
                status['active_model_metrics'] = model_info['training_metrics']

        return status

    def get_model_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard of all models by performance"""

        leaderboard = []

        for version, info in self.version_manager.models_registry.items():
            if 'training_metrics' in info:
                metrics = info['training_metrics']
                cv_results = info.get('cv_results', {})

                entry = {
                    'version': version,
                    'accuracy': metrics.get('val_accuracy', 0),
                    'auc_score': metrics.get('auc_score', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'cv_mean': cv_results.get('cv_mean', 0),
                    'cv_std': cv_results.get('cv_std', 0),
                    'training_time': metrics.get('training_time', 0),
                    'created_at': info.get('created_at', 0),
                    'is_active': info.get('is_active', False)
                }

                leaderboard.append(entry)

        # Sort by accuracy (descending)
        leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)

        return leaderboard

    def export_training_history(self, filepath: str = None) -> str:
        """Export complete training history"""

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"training_history_{timestamp}.json"

        try:
            export_data = {
                'config': asdict(self.config),
                'models_registry': self.version_manager.models_registry,
                'training_sessions': [],
                'export_timestamp': time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Training history exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting training history: {e}")
            raise
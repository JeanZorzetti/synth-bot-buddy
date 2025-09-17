"""
Real Model Training System - Phase 13 Real-Time Data Pipeline
Sistema de treinamento de modelos IA com dados históricos reais
"""

import asyncio
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from influxdb_timeseries import get_influxdb_manager
from realtime_feature_processor import get_feature_processor
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    LSTM = "lstm"
    TRANSFORMER = "transformer"

class TrainingStatus(Enum):
    IDLE = "idle"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ModelPerformance:
    model_id: str
    model_type: ModelType
    training_timestamp: datetime

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float

    # Trading-specific metrics
    predicted_direction_accuracy: float
    profitable_trades_percentage: float
    sharpe_ratio: float
    max_drawdown: float

    # Training details
    training_samples: int
    validation_samples: int
    training_time_seconds: float
    feature_count: int

    # Model configuration
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    cross_validation_score: float

@dataclass
class TrainingSession:
    session_id: str
    symbols: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    status: TrainingStatus

    # Data configuration
    lookback_days: int
    prediction_horizon_minutes: int
    train_test_split_ratio: float

    # Model configuration
    models_to_train: List[ModelType]
    use_cross_validation: bool
    hyperparameter_tuning: bool

    # Results
    trained_models: Dict[str, ModelPerformance]
    best_model_id: Optional[str]
    training_logs: List[str]
    error_message: Optional[str]

class RealModelTrainer:
    """Advanced model training system using real historical data"""

    def __init__(self):
        # Model storage
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.scalers_dir = Path("scalers")
        self.scalers_dir.mkdir(exist_ok=True)

        # Active training sessions
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.trained_models: Dict[str, Dict[str, Any]] = {}  # model_id -> model_info

        # Training configuration
        self.default_config = {
            'lookback_days': 30,
            'prediction_horizon_minutes': 5,
            'train_test_split': 0.8,
            'min_samples_for_training': 1000,
            'max_training_time_hours': 6,
            'cross_validation_folds': 5
        }

        # Feature engineering
        self.feature_columns = [
            'price', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'momentum_10', 'roc_10', 'atr_14',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'bollinger_width', 'bollinger_position',
            'stoch_k', 'stoch_d', 'williams_r',
            'volume_sma_10', 'volume_ratio', 'vwap',
            'price_change_1', 'price_change_5', 'price_change_10',
            'volatility_5', 'volatility_10',
            'bid_ask_spread', 'spread_ratio', 'tick_direction', 'tick_intensity',
            'hurst_exponent', 'fractal_dimension', 'entropy',
            'autocorr_1', 'autocorr_5'
        ]

        # Dependencies
        self.influxdb_manager = None
        self.feature_processor = None
        self.cache_manager = None

        # Logging
        self.logger = logging_system.loggers.get('ai_engine', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize model trainer"""
        try:
            # Initialize dependencies
            self.influxdb_manager = await get_influxdb_manager()
            self.feature_processor = await get_feature_processor()
            self.cache_manager = await get_cache_manager()

            # Load existing models
            await self._load_existing_models()

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                "Real model trainer initialized",
                {'existing_models': len(self.trained_models)}
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'initialize_trainer'}
            )
            raise

    async def start_training_session(
        self,
        symbols: List[str],
        models_to_train: List[ModelType],
        lookback_days: int = 30,
        prediction_horizon_minutes: int = 5,
        hyperparameter_tuning: bool = True,
        use_cross_validation: bool = True
    ) -> str:
        """Start a new training session"""
        try:
            # Generate session ID
            session_id = f"training_{int(datetime.utcnow().timestamp())}_{len(symbols)}symbols"

            # Create training session
            session = TrainingSession(
                session_id=session_id,
                symbols=symbols,
                start_time=datetime.utcnow(),
                end_time=None,
                status=TrainingStatus.IDLE,
                lookback_days=lookback_days,
                prediction_horizon_minutes=prediction_horizon_minutes,
                train_test_split_ratio=0.8,
                models_to_train=models_to_train,
                use_cross_validation=use_cross_validation,
                hyperparameter_tuning=hyperparameter_tuning,
                trained_models={},
                best_model_id=None,
                training_logs=[],
                error_message=None
            )

            self.active_sessions[session_id] = session

            # Start training asynchronously
            asyncio.create_task(self._execute_training_session(session_id))

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Started training session: {session_id}",
                {'symbols': symbols, 'models': [m.value for m in models_to_train]}
            )

            return session_id

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'start_training_session', 'symbols': symbols}
            )
            raise

    async def _execute_training_session(self, session_id: str):
        """Execute complete training session"""
        try:
            session = self.active_sessions[session_id]
            session.status = TrainingStatus.PREPARING_DATA

            # Log training start
            session.training_logs.append(f"Training session started: {datetime.utcnow().isoformat()}")

            # Prepare training data
            training_data = await self._prepare_training_data(session)
            if not training_data:
                raise ValueError("No training data available")

            session.training_logs.append(f"Prepared training data: {len(training_data)} samples")

            # Train models
            session.status = TrainingStatus.TRAINING

            for model_type in session.models_to_train:
                try:
                    session.training_logs.append(f"Training {model_type.value} model...")

                    model_performance = await self._train_single_model(
                        session, training_data, model_type
                    )

                    if model_performance:
                        model_id = f"{session_id}_{model_type.value}"
                        session.trained_models[model_id] = model_performance
                        session.training_logs.append(f"Completed {model_type.value}: accuracy={model_performance.accuracy:.3f}")

                except Exception as model_error:
                    session.training_logs.append(f"Failed to train {model_type.value}: {str(model_error)}")
                    logging_system.log_error(
                        LogComponent.AI_ENGINE,
                        model_error,
                        {'session_id': session_id, 'model_type': model_type.value}
                    )

            # Select best model
            session.status = TrainingStatus.VALIDATING
            session.best_model_id = self._select_best_model(session.trained_models)

            # Save session results
            session.status = TrainingStatus.SAVING
            await self._save_training_results(session)

            # Complete session
            session.status = TrainingStatus.COMPLETED
            session.end_time = datetime.utcnow()
            session.training_logs.append(f"Training session completed: {session.end_time.isoformat()}")

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Training session completed: {session_id}",
                {'models_trained': len(session.trained_models), 'best_model': session.best_model_id}
            )

        except Exception as e:
            session.status = TrainingStatus.FAILED
            session.error_message = str(e)
            session.end_time = datetime.utcnow()
            session.training_logs.append(f"Training session failed: {str(e)}")

            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'execute_training_session', 'session_id': session_id}
            )

    async def _prepare_training_data(self, session: TrainingSession) -> Optional[pd.DataFrame]:
        """Prepare training data from historical features"""
        try:
            all_data = []

            # Get data for each symbol
            for symbol in session.symbols:
                session.training_logs.append(f"Preparing data for {symbol}...")

                # Calculate date range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=session.lookback_days)

                # Get historical features from InfluxDB
                features_df = await self.influxdb_manager.query_features(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time
                )

                if features_df.empty:
                    session.training_logs.append(f"No feature data found for {symbol}")
                    continue

                # Sort by timestamp
                features_df = features_df.sort_values('timestamp')

                # Add symbol column
                features_df['symbol'] = symbol

                # Create target variable (future price direction)
                features_df = self._create_target_variable(features_df, session.prediction_horizon_minutes)

                all_data.append(features_df)
                session.training_logs.append(f"Added {len(features_df)} samples for {symbol}")

            if not all_data:
                return None

            # Combine all symbol data
            combined_data = pd.concat(all_data, ignore_index=True)

            # Clean and validate data
            combined_data = self._clean_training_data(combined_data)

            if len(combined_data) < self.default_config['min_samples_for_training']:
                session.training_logs.append(f"Insufficient training data: {len(combined_data)} samples")
                return None

            session.training_logs.append(f"Final training dataset: {len(combined_data)} samples")
            return combined_data

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'prepare_training_data', 'session_id': session.session_id}
            )
            return None

    def _create_target_variable(self, df: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
        """Create target variable for supervised learning"""
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp').copy()

            # Calculate future price change
            df['future_price'] = df['price'].shift(-horizon_minutes)  # Assuming 1-minute data
            df['price_change'] = (df['future_price'] - df['price']) / df['price']

            # Create binary target (1 = up, 0 = down)
            df['target'] = (df['price_change'] > 0).astype(int)

            # Create multi-class target for more nuanced predictions
            df['target_multiclass'] = pd.cut(
                df['price_change'],
                bins=[-np.inf, -0.001, 0.001, np.inf],
                labels=[0, 1, 2]  # 0=down, 1=flat, 2=up
            ).astype(int)

            # Remove rows with missing target
            df = df.dropna(subset=['target', 'future_price'])

            return df

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'create_target_variable'}
            )
            return df

    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess training data"""
        try:
            # Remove rows with missing values in feature columns
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            df = df.dropna(subset=feature_cols)

            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            # Remove outliers (optional, using IQR method)
            for col in feature_cols:
                if df[col].dtype in [np.float64, np.int64]:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)

            return df

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'clean_training_data'}
            )
            return df

    async def _train_single_model(
        self,
        session: TrainingSession,
        training_data: pd.DataFrame,
        model_type: ModelType
    ) -> Optional[ModelPerformance]:
        """Train a single model"""
        try:
            start_time = datetime.utcnow()

            # Prepare features and target
            feature_cols = [col for col in self.feature_columns if col in training_data.columns]
            X = training_data[feature_cols].values
            y = training_data['target'].values

            # Split data (time-aware split)
            split_idx = int(len(X) * session.train_test_split_ratio)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create and train model
            model, hyperparameters = self._create_model(model_type, session.hyperparameter_tuning)

            if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] and TF_AVAILABLE:
                # Train deep learning model
                model, training_history = await self._train_deep_model(
                    model, X_train_scaled, y_train, X_test_scaled, y_test
                )
            else:
                # Train traditional ML model
                if session.use_cross_validation:
                    # Use time series cross validation
                    tscv = TimeSeriesSplit(n_splits=self.default_config['cross_validation_folds'])
                    cv_scores = []

                    for train_idx, val_idx in tscv.split(X_train_scaled):
                        X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                        model_copy = self._create_model(model_type, False)[0]
                        model_copy.fit(X_train_cv, y_train_cv)
                        cv_score = model_copy.score(X_val_cv, y_val_cv)
                        cv_scores.append(cv_score)

                    cv_score = np.mean(cv_scores)
                else:
                    cv_score = 0.0

                # Train final model
                model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = None

            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate performance metrics
            performance = self._calculate_model_performance(
                model_type, y_test, y_pred, y_pred_proba,
                training_data[split_idx:], feature_cols, hyperparameters,
                len(X_train), len(X_test),
                (datetime.utcnow() - start_time).total_seconds(),
                cv_score if 'cv_score' in locals() else 0.0
            )

            # Save model and scaler
            model_id = f"{session.session_id}_{model_type.value}"
            await self._save_model(model_id, model, scaler, performance)

            return performance

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'train_single_model', 'model_type': model_type.value}
            )
            return None

    def _create_model(self, model_type: ModelType, hyperparameter_tuning: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """Create and configure model"""
        hyperparams = {}

        if model_type == ModelType.RANDOM_FOREST:
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                model = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy'
                )
            else:
                hyperparams = {'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
                model = RandomForestClassifier(**hyperparams)

        elif model_type == ModelType.XGBOOST:
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1]
                }
                model = GridSearchCV(
                    xgb.XGBClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy'
                )
            else:
                hyperparams = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
                model = xgb.XGBClassifier(**hyperparams)

        elif model_type == ModelType.LIGHTGBM:
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1]
                }
                model = GridSearchCV(
                    lgb.LGBMClassifier(random_state=42, verbose=-1),
                    param_grid, cv=3, scoring='accuracy'
                )
            else:
                hyperparams = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
                model = lgb.LGBMClassifier(**hyperparams, verbose=-1)

        elif model_type == ModelType.GRADIENT_BOOSTING:
            hyperparams = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
            model = GradientBoostingClassifier(**hyperparams)

        elif model_type == ModelType.LOGISTIC_REGRESSION:
            hyperparams = {'C': 1.0, 'random_state': 42}
            model = LogisticRegression(**hyperparams)

        elif model_type == ModelType.SVM:
            hyperparams = {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
            model = SVC(**hyperparams, probability=True)

        elif model_type == ModelType.LSTM and TF_AVAILABLE:
            model = self._create_lstm_model()
            hyperparams = {'lstm_units': 50, 'dropout': 0.2, 'optimizer': 'adam'}

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model, hyperparams

    def _create_lstm_model(self, input_shape: Optional[Tuple] = None) -> tf.keras.Model:
        """Create LSTM model for time series prediction"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for LSTM training")

        if input_shape is None:
            input_shape = (60, len(self.feature_columns))  # 60 time steps

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    async def _train_deep_model(
        self, model, X_train, y_train, X_test, y_test
    ) -> Tuple[Any, Dict]:
        """Train deep learning model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")

        # Reshape for LSTM (samples, time_steps, features)
        # For simplicity, we'll create sequences from the feature data
        sequence_length = 60

        def create_sequences(data, labels, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(data)):
                X_seq.append(data[i-seq_length:i])
                y_seq.append(labels[i])
            return np.array(X_seq), np.array(y_seq)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks,
            verbose=0
        )

        return model, history.history

    def _calculate_model_performance(
        self, model_type: ModelType, y_true, y_pred, y_pred_proba,
        test_data: pd.DataFrame, feature_cols: List[str], hyperparams: Dict[str, Any],
        train_samples: int, test_samples: int, training_time: float, cv_score: float
    ) -> ModelPerformance:
        """Calculate comprehensive model performance metrics"""

        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        # AUC score (if probabilities available)
        auc_score = 0.0
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(y_true, y_pred_proba)

        # Trading-specific metrics
        predicted_direction_accuracy = accuracy  # Same as accuracy for binary classification

        # Simulate trading performance
        profitable_trades_pct = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0

        if 'price_change' in test_data.columns:
            # Calculate trading simulation metrics
            test_data = test_data.copy()
            test_data['predicted'] = y_pred
            test_data['actual_return'] = test_data['price_change']

            # Calculate returns when following predictions
            test_data['strategy_return'] = np.where(
                test_data['predicted'] == 1,
                test_data['actual_return'],
                -test_data['actual_return']
            )

            profitable_trades = (test_data['strategy_return'] > 0).sum()
            total_trades = len(test_data)
            profitable_trades_pct = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

            # Sharpe ratio
            if len(test_data['strategy_return']) > 1:
                returns_std = test_data['strategy_return'].std()
                if returns_std > 0:
                    sharpe_ratio = test_data['strategy_return'].mean() / returns_std

            # Max drawdown
            cumulative_returns = (1 + test_data['strategy_return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100  # Convert to percentage

        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(model_type, 'feature_importances_') and len(feature_cols) > 0:
            try:
                importances = model_type.feature_importances_
                feature_importance = dict(zip(feature_cols, importances.tolist()))
            except:
                pass

        # Create performance object
        model_id = f"model_{model_type.value}_{int(datetime.utcnow().timestamp())}"

        performance = ModelPerformance(
            model_id=model_id,
            model_type=model_type,
            training_timestamp=datetime.utcnow(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            predicted_direction_accuracy=predicted_direction_accuracy,
            profitable_trades_percentage=profitable_trades_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            training_samples=train_samples,
            validation_samples=test_samples,
            training_time_seconds=training_time,
            feature_count=len(feature_cols),
            hyperparameters=hyperparams,
            feature_importance=feature_importance,
            cross_validation_score=cv_score
        )

        return performance

    def _select_best_model(self, trained_models: Dict[str, ModelPerformance]) -> Optional[str]:
        """Select best model based on comprehensive scoring"""
        if not trained_models:
            return None

        best_model_id = None
        best_score = -1

        for model_id, performance in trained_models.items():
            # Composite score considering multiple metrics
            score = (
                performance.accuracy * 0.3 +
                performance.f1_score * 0.2 +
                performance.auc_score * 0.2 +
                (performance.profitable_trades_percentage / 100) * 0.15 +
                max(0, performance.sharpe_ratio / 2) * 0.1 +
                max(0, (100 + performance.max_drawdown) / 100) * 0.05  # Less negative drawdown is better
            )

            if score > best_score:
                best_score = score
                best_model_id = model_id

        return best_model_id

    async def _save_model(self, model_id: str, model: Any, scaler: Any, performance: ModelPerformance):
        """Save trained model and associated data"""
        try:
            # Save model
            model_path = self.models_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path)

            # Save scaler
            scaler_path = self.scalers_dir / f"{model_id}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

            # Save performance metadata
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(performance), f, indent=2, default=str)

            # Store in memory
            self.trained_models[model_id] = {
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'metadata': performance,
                'created_at': datetime.utcnow()
            }

            # Cache model info
            await self.cache_manager.set(
                CacheNamespace.MODEL_CACHE,
                f"model_info:{model_id}",
                asdict(performance),
                ttl=7200  # 2 hours
            )

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Model saved: {model_id}",
                {'accuracy': performance.accuracy, 'f1_score': performance.f1_score}
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'save_model', 'model_id': model_id}
            )

    async def _save_training_results(self, session: TrainingSession):
        """Save training session results"""
        try:
            results_path = self.models_dir / f"{session.session_id}_results.json"

            # Prepare session data for serialization
            session_data = asdict(session)
            session_data['models_to_train'] = [m.value for m in session.models_to_train]
            session_data['status'] = session.status.value

            # Convert ModelPerformance objects to dict
            trained_models_data = {}
            for model_id, performance in session.trained_models.items():
                trained_models_data[model_id] = asdict(performance)

            session_data['trained_models'] = trained_models_data

            with open(results_path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            # Cache session results
            await self.cache_manager.set(
                CacheNamespace.MODEL_CACHE,
                f"training_session:{session.session_id}",
                session_data,
                ttl=86400  # 24 hours
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'save_training_results', 'session_id': session.session_id}
            )

    async def _load_existing_models(self):
        """Load existing models from disk"""
        try:
            for model_file in self.models_dir.glob("*.joblib"):
                if "_scaler" not in model_file.name:
                    model_id = model_file.stem

                    # Check for metadata file
                    metadata_file = self.models_dir / f"{model_id}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        # Find corresponding scaler
                        scaler_path = self.scalers_dir / f"{model_id}_scaler.joblib"

                        self.trained_models[model_id] = {
                            'model_path': str(model_file),
                            'scaler_path': str(scaler_path) if scaler_path.exists() else None,
                            'metadata': metadata,
                            'loaded_from_disk': True
                        }

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Loaded {len(self.trained_models)} existing models"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'load_existing_models'}
            )

    async def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session status"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                'session_id': session_id,
                'status': session.status.value,
                'progress': len(session.trained_models) / len(session.models_to_train) if session.models_to_train else 0,
                'models_trained': len(session.trained_models),
                'total_models': len(session.models_to_train),
                'best_model_id': session.best_model_id,
                'training_logs': session.training_logs[-10:],  # Last 10 logs
                'error_message': session.error_message
            }
        return None

    async def get_model_performance(self, model_id: str) -> Optional[ModelPerformance]:
        """Get model performance metrics"""
        if model_id in self.trained_models:
            metadata = self.trained_models[model_id]['metadata']
            if isinstance(metadata, dict):
                return ModelPerformance(**metadata)
            return metadata
        return None

    def get_trainer_status(self) -> Dict[str, Any]:
        """Get trainer status"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_models': len(self.trained_models),
            'session_statuses': {
                session_id: session.status.value
                for session_id, session in self.active_sessions.items()
            },
            'available_model_types': [mt.value for mt in ModelType],
            'tensorflow_available': TF_AVAILABLE
        }

# Global model trainer instance
model_trainer = RealModelTrainer()

async def get_model_trainer() -> RealModelTrainer:
    """Get initialized model trainer"""
    if not model_trainer.influxdb_manager:
        await model_trainer.initialize()
    return model_trainer
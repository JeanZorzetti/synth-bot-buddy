"""
Model Drift Detection and Auto-Retraining System - Phase 13 Real-Time Data Pipeline
Sistema de detecção de drift e retreinamento automático de modelos
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque
import joblib

# Drift detection libraries
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from real_model_trainer import get_model_trainer, ModelType, TrainingStatus
from influxdb_timeseries import get_influxdb_manager
from realtime_feature_processor import get_feature_processor, ProcessedFeatures
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

class DriftType(Enum):
    PERFORMANCE_DRIFT = "performance_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"

class DriftSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetrainingTrigger(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DISTRIBUTION_CHANGE = "data_distribution_change"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED = "scheduled"

@dataclass
class DriftAlert:
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    timestamp: datetime
    description: str
    metrics: Dict[str, float]
    threshold_breached: float
    current_value: float
    retraining_triggered: bool = False

@dataclass
class ModelMonitoringMetrics:
    model_id: str
    timestamp: datetime

    # Performance metrics
    recent_accuracy: float
    recent_f1_score: float
    baseline_accuracy: float
    baseline_f1_score: float
    performance_degradation_pct: float

    # Data drift metrics
    feature_drift_scores: Dict[str, float]
    overall_drift_score: float
    data_quality_score: float

    # Prediction drift metrics
    prediction_distribution_shift: float
    confidence_drift_score: float

    # Volume metrics
    predictions_count_24h: int
    predictions_count_7d: int

    # Drift status
    drift_detected: bool
    drift_types: List[DriftType]
    requires_retraining: bool

class ModelDriftDetector:
    """Advanced model drift detection and auto-retraining system"""

    def __init__(self):
        # Drift detection configuration
        self.drift_thresholds = {
            'performance_degradation_pct': 10.0,  # 10% performance drop
            'feature_drift_threshold': 0.1,       # PSI threshold
            'prediction_drift_threshold': 0.05,   # Distribution shift
            'confidence_drift_threshold': 0.15,   # Confidence change
            'data_quality_min': 80.0,            # Minimum data quality
            'min_predictions_for_drift': 100,     # Minimum samples for drift detection
        }

        # Retraining configuration
        self.retraining_config = {
            'auto_retrain_enabled': True,
            'min_hours_between_retraining': 24,
            'max_training_attempts_per_day': 3,
            'performance_improvement_threshold': 2.0,  # 2% improvement required
            'retraining_data_days': 30,
        }

        # Monitoring data
        self.model_baselines: Dict[str, Dict[str, Any]] = {}
        self.recent_predictions: Dict[str, deque] = {}  # model_id -> predictions
        self.recent_features: Dict[str, deque] = {}     # model_id -> features
        self.drift_history: Dict[str, List[DriftAlert]] = {}
        self.retraining_history: Dict[str, List[Dict[str, Any]]] = {}

        # Active monitoring
        self.monitored_models: set = set()
        self.monitoring_active = False

        # Dependencies
        self.model_trainer = None
        self.influxdb_manager = None
        self.feature_processor = None
        self.cache_manager = None

        # Logging
        self.logger = logging_system.loggers.get('ai_engine', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize drift detector"""
        try:
            # Initialize dependencies
            self.model_trainer = await get_model_trainer()
            self.influxdb_manager = await get_influxdb_manager()
            self.feature_processor = await get_feature_processor()
            self.cache_manager = await get_cache_manager()

            # Load existing model baselines
            await self._load_model_baselines()

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                "Model drift detector initialized",
                {'monitored_models': len(self.model_baselines)}
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'initialize_drift_detector'}
            )
            raise

    async def start_monitoring(self, model_ids: List[str]):
        """Start drift monitoring for models"""
        try:
            self.monitoring_active = True

            for model_id in model_ids:
                if model_id not in self.monitored_models:
                    # Initialize monitoring structures
                    self.monitored_models.add(model_id)
                    self.recent_predictions[model_id] = deque(maxlen=10000)
                    self.recent_features[model_id] = deque(maxlen=10000)
                    self.drift_history[model_id] = []
                    self.retraining_history[model_id] = []

                    # Load or create baseline
                    await self._establish_model_baseline(model_id)

            # Start periodic monitoring
            asyncio.create_task(self._periodic_drift_monitoring())

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Started drift monitoring for {len(self.monitored_models)} models"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'start_monitoring', 'model_ids': model_ids}
            )

    async def stop_monitoring(self):
        """Stop drift monitoring"""
        try:
            self.monitoring_active = False
            self.monitored_models.clear()

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                "Drift monitoring stopped"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'stop_monitoring'}
            )

    async def record_prediction(self, model_id: str, features: ProcessedFeatures,
                              prediction: Dict[str, Any]):
        """Record model prediction for drift monitoring"""
        try:
            if model_id not in self.monitored_models:
                return

            # Record prediction
            prediction_record = {
                'timestamp': datetime.utcnow(),
                'prediction': prediction.get('signal', 0),
                'confidence': prediction.get('confidence', 0.5),
                'probability': prediction.get('probability_up', 0.5)
            }

            self.recent_predictions[model_id].append(prediction_record)

            # Record features
            feature_dict = asdict(features)
            feature_dict.pop('symbol', None)
            feature_dict.pop('timestamp', None)

            self.recent_features[model_id].append({
                'timestamp': features.timestamp,
                'features': feature_dict
            })

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'record_prediction', 'model_id': model_id}
            )

    async def _periodic_drift_monitoring(self):
        """Perform periodic drift monitoring"""
        while self.monitoring_active:
            try:
                for model_id in list(self.monitored_models):
                    await self._monitor_model_drift(model_id)

                # Wait 1 hour before next check
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.AI_ENGINE,
                    e,
                    {'action': 'periodic_drift_monitoring'}
                )
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _monitor_model_drift(self, model_id: str):
        """Monitor drift for specific model"""
        try:
            # Check if we have enough data
            if (len(self.recent_predictions[model_id]) < self.drift_thresholds['min_predictions_for_drift']):
                return

            # Calculate monitoring metrics
            metrics = await self._calculate_monitoring_metrics(model_id)
            if not metrics:
                return

            # Store metrics
            await self._store_monitoring_metrics(metrics)

            # Check for drift
            drift_alerts = await self._detect_drift(metrics)

            # Process alerts
            for alert in drift_alerts:
                await self._process_drift_alert(alert)

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'monitor_model_drift', 'model_id': model_id}
            )

    async def _calculate_monitoring_metrics(self, model_id: str) -> Optional[ModelMonitoringMetrics]:
        """Calculate comprehensive monitoring metrics"""
        try:
            current_time = datetime.utcnow()
            baseline = self.model_baselines.get(model_id, {})

            if not baseline:
                return None

            # Get recent predictions and features
            recent_preds = list(self.recent_predictions[model_id])
            recent_features = list(self.recent_features[model_id])

            # Filter last 24 hours and 7 days
            day_ago = current_time - timedelta(hours=24)
            week_ago = current_time - timedelta(days=7)

            preds_24h = [p for p in recent_preds if p['timestamp'] >= day_ago]
            preds_7d = [p for p in recent_preds if p['timestamp'] >= week_ago]
            features_24h = [f for f in recent_features if f['timestamp'] >= day_ago]

            if not preds_24h or not features_24h:
                return None

            # Calculate performance metrics (simulated - in real scenario, you'd have actual outcomes)
            recent_accuracy = self._estimate_recent_accuracy(preds_24h)
            recent_f1 = self._estimate_recent_f1(preds_24h)

            baseline_accuracy = baseline.get('baseline_accuracy', 0.5)
            baseline_f1 = baseline.get('baseline_f1', 0.5)

            performance_degradation = ((baseline_accuracy - recent_accuracy) / baseline_accuracy) * 100

            # Calculate feature drift
            feature_drift_scores = self._calculate_feature_drift(features_24h, baseline)
            overall_drift_score = np.mean(list(feature_drift_scores.values())) if feature_drift_scores else 0.0

            # Calculate prediction drift
            prediction_shift = self._calculate_prediction_drift(preds_24h, baseline)
            confidence_drift = self._calculate_confidence_drift(preds_24h, baseline)

            # Data quality score (simplified)
            data_quality_score = self._estimate_data_quality(features_24h)

            # Determine drift status
            drift_detected = (
                performance_degradation > self.drift_thresholds['performance_degradation_pct'] or
                overall_drift_score > self.drift_thresholds['feature_drift_threshold'] or
                prediction_shift > self.drift_thresholds['prediction_drift_threshold'] or
                data_quality_score < self.drift_thresholds['data_quality_min']
            )

            drift_types = []
            if performance_degradation > self.drift_thresholds['performance_degradation_pct']:
                drift_types.append(DriftType.PERFORMANCE_DRIFT)
            if overall_drift_score > self.drift_thresholds['feature_drift_threshold']:
                drift_types.append(DriftType.DATA_DRIFT)
            if prediction_shift > self.drift_thresholds['prediction_drift_threshold']:
                drift_types.append(DriftType.CONCEPT_DRIFT)

            requires_retraining = drift_detected and len(drift_types) > 0

            metrics = ModelMonitoringMetrics(
                model_id=model_id,
                timestamp=current_time,
                recent_accuracy=recent_accuracy,
                recent_f1_score=recent_f1,
                baseline_accuracy=baseline_accuracy,
                baseline_f1_score=baseline_f1,
                performance_degradation_pct=performance_degradation,
                feature_drift_scores=feature_drift_scores,
                overall_drift_score=overall_drift_score,
                data_quality_score=data_quality_score,
                prediction_distribution_shift=prediction_shift,
                confidence_drift_score=confidence_drift,
                predictions_count_24h=len(preds_24h),
                predictions_count_7d=len(preds_7d),
                drift_detected=drift_detected,
                drift_types=drift_types,
                requires_retraining=requires_retraining
            )

            return metrics

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'calculate_monitoring_metrics', 'model_id': model_id}
            )
            return None

    def _estimate_recent_accuracy(self, predictions: List[Dict]) -> float:
        """Estimate recent accuracy (simplified simulation)"""
        # In a real system, you'd compare with actual outcomes
        # Here we simulate based on confidence scores
        if not predictions:
            return 0.5

        # Simulate accuracy based on average confidence
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        # Higher confidence generally correlates with better accuracy
        estimated_accuracy = 0.4 + (avg_confidence * 0.4)  # Range 0.4-0.8

        return min(1.0, max(0.0, estimated_accuracy))

    def _estimate_recent_f1(self, predictions: List[Dict]) -> float:
        """Estimate recent F1 score (simplified simulation)"""
        # Similar simulation approach
        if not predictions:
            return 0.5

        avg_confidence = np.mean([p['confidence'] for p in predictions])
        estimated_f1 = 0.35 + (avg_confidence * 0.45)  # Range 0.35-0.8

        return min(1.0, max(0.0, estimated_f1))

    def _calculate_feature_drift(self, recent_features: List[Dict], baseline: Dict) -> Dict[str, float]:
        """Calculate feature drift using Population Stability Index (PSI)"""
        try:
            if not recent_features or 'baseline_features' not in baseline:
                return {}

            drift_scores = {}
            baseline_features = baseline['baseline_features']

            # Extract recent feature values
            recent_feature_dict = {}
            for feature_record in recent_features:
                for feature_name, value in feature_record['features'].items():
                    if feature_name not in recent_feature_dict:
                        recent_feature_dict[feature_name] = []
                    recent_feature_dict[feature_name].append(value)

            # Calculate PSI for each feature
            for feature_name in baseline_features:
                if feature_name in recent_feature_dict:
                    baseline_values = baseline_features[feature_name]
                    recent_values = recent_feature_dict[feature_name]

                    psi = self._calculate_psi(baseline_values, recent_values)
                    drift_scores[feature_name] = psi

            return drift_scores

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'calculate_feature_drift'}
            )
            return {}

    def _calculate_psi(self, baseline_values: List[float], recent_values: List[float]) -> float:
        """Calculate Population Stability Index"""
        try:
            if not baseline_values or not recent_values:
                return 0.0

            # Create bins based on baseline distribution
            baseline_array = np.array(baseline_values)
            recent_array = np.array(recent_values)

            # Use quantile-based binning
            bins = np.quantile(baseline_array, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            bins = np.unique(bins)  # Remove duplicates

            if len(bins) < 2:
                return 0.0

            # Calculate distributions
            baseline_hist, _ = np.histogram(baseline_array, bins=bins)
            recent_hist, _ = np.histogram(recent_array, bins=bins)

            # Convert to probabilities (add small epsilon to avoid log(0))
            epsilon = 1e-10
            baseline_prob = (baseline_hist + epsilon) / (baseline_hist.sum() + epsilon * len(baseline_hist))
            recent_prob = (recent_hist + epsilon) / (recent_hist.sum() + epsilon * len(recent_hist))

            # Calculate PSI
            psi = np.sum((recent_prob - baseline_prob) * np.log(recent_prob / baseline_prob))

            return abs(psi)

        except Exception:
            return 0.0

    def _calculate_prediction_drift(self, predictions: List[Dict], baseline: Dict) -> float:
        """Calculate drift in prediction distribution"""
        try:
            if not predictions or 'baseline_prediction_dist' not in baseline:
                return 0.0

            # Get recent prediction distribution
            recent_preds = [p['prediction'] for p in predictions]
            recent_dist = np.histogram(recent_preds, bins=[-1, -0.5, 0, 0.5, 1])[0]
            recent_dist = recent_dist / recent_dist.sum() if recent_dist.sum() > 0 else recent_dist

            baseline_dist = np.array(baseline['baseline_prediction_dist'])

            # Calculate KL divergence
            epsilon = 1e-10
            recent_dist = recent_dist + epsilon
            baseline_dist = baseline_dist + epsilon

            kl_div = stats.entropy(recent_dist, baseline_dist)

            return kl_div

        except Exception:
            return 0.0

    def _calculate_confidence_drift(self, predictions: List[Dict], baseline: Dict) -> float:
        """Calculate drift in confidence scores"""
        try:
            if not predictions or 'baseline_confidence_mean' not in baseline:
                return 0.0

            recent_confidences = [p['confidence'] for p in predictions]
            recent_mean = np.mean(recent_confidences)
            baseline_mean = baseline['baseline_confidence_mean']

            # Calculate relative change
            drift = abs(recent_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0

            return drift

        except Exception:
            return 0.0

    def _estimate_data_quality(self, features: List[Dict]) -> float:
        """Estimate data quality score"""
        try:
            if not features:
                return 0.0

            total_features = 0
            valid_features = 0

            for feature_record in features:
                for feature_name, value in feature_record['features'].items():
                    total_features += 1
                    if not (np.isnan(value) or np.isinf(value) or value is None):
                        valid_features += 1

            quality_score = (valid_features / total_features) * 100 if total_features > 0 else 0

            return quality_score

        except Exception:
            return 0.0

    async def _detect_drift(self, metrics: ModelMonitoringMetrics) -> List[DriftAlert]:
        """Detect drift based on metrics"""
        try:
            alerts = []

            # Performance drift
            if metrics.performance_degradation_pct > self.drift_thresholds['performance_degradation_pct']:
                alert = DriftAlert(
                    alert_id=f"drift_{int(datetime.utcnow().timestamp())}",
                    model_id=metrics.model_id,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=self._determine_severity(metrics.performance_degradation_pct, 10, 20, 30),
                    timestamp=datetime.utcnow(),
                    description=f"Model performance degraded by {metrics.performance_degradation_pct:.1f}%",
                    metrics={'degradation_pct': metrics.performance_degradation_pct},
                    threshold_breached=self.drift_thresholds['performance_degradation_pct'],
                    current_value=metrics.performance_degradation_pct
                )
                alerts.append(alert)

            # Feature drift
            if metrics.overall_drift_score > self.drift_thresholds['feature_drift_threshold']:
                alert = DriftAlert(
                    alert_id=f"drift_{int(datetime.utcnow().timestamp())}_feature",
                    model_id=metrics.model_id,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=self._determine_severity(metrics.overall_drift_score, 0.1, 0.2, 0.5),
                    timestamp=datetime.utcnow(),
                    description=f"Feature distribution drift detected: {metrics.overall_drift_score:.3f}",
                    metrics={'drift_score': metrics.overall_drift_score},
                    threshold_breached=self.drift_thresholds['feature_drift_threshold'],
                    current_value=metrics.overall_drift_score
                )
                alerts.append(alert)

            # Prediction drift
            if metrics.prediction_distribution_shift > self.drift_thresholds['prediction_drift_threshold']:
                alert = DriftAlert(
                    alert_id=f"drift_{int(datetime.utcnow().timestamp())}_pred",
                    model_id=metrics.model_id,
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity=self._determine_severity(metrics.prediction_distribution_shift, 0.05, 0.1, 0.2),
                    timestamp=datetime.utcnow(),
                    description=f"Prediction distribution shift: {metrics.prediction_distribution_shift:.3f}",
                    metrics={'shift': metrics.prediction_distribution_shift},
                    threshold_breached=self.drift_thresholds['prediction_drift_threshold'],
                    current_value=metrics.prediction_distribution_shift
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'detect_drift', 'model_id': metrics.model_id}
            )
            return []

    def _determine_severity(self, value: float, low_thresh: float,
                          medium_thresh: float, high_thresh: float) -> DriftSeverity:
        """Determine drift severity based on value and thresholds"""
        if value >= high_thresh:
            return DriftSeverity.CRITICAL
        elif value >= medium_thresh:
            return DriftSeverity.HIGH
        elif value >= low_thresh:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW

    async def _process_drift_alert(self, alert: DriftAlert):
        """Process drift alert and trigger actions"""
        try:
            # Store alert
            if alert.model_id not in self.drift_history:
                self.drift_history[alert.model_id] = []
            self.drift_history[alert.model_id].append(alert)

            # Log alert
            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.WARNING if alert.severity in [DriftSeverity.LOW, DriftSeverity.MEDIUM] else LogLevel.ERROR,
                f"Model drift detected: {alert.description}",
                {
                    'model_id': alert.model_id,
                    'drift_type': alert.drift_type.value,
                    'severity': alert.severity.value,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold_breached
                }
            )

            # Cache alert
            await self.cache_manager.set(
                CacheNamespace.SYSTEM_METRICS,
                f"drift_alert:{alert.model_id}:{alert.alert_id}",
                asdict(alert),
                ttl=7200  # 2 hours
            )

            # Trigger retraining if conditions are met
            if (alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] and
                self.retraining_config['auto_retrain_enabled']):

                await self._trigger_auto_retraining(alert)

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'process_drift_alert', 'alert_id': alert.alert_id}
            )

    async def _trigger_auto_retraining(self, alert: DriftAlert):
        """Trigger automatic model retraining"""
        try:
            model_id = alert.model_id

            # Check retraining constraints
            if not await self._can_retrain_model(model_id):
                logging_system.log(
                    LogComponent.AI_ENGINE,
                    LogLevel.INFO,
                    f"Retraining skipped for {model_id}: constraints not met"
                )
                return

            # Extract symbol from model_id (assuming format contains symbol)
            # This is a simplified approach - in practice, you'd store this mapping
            symbols = ['R_10', 'R_25']  # Default symbols

            # Start retraining session
            retraining_session_id = await self.model_trainer.start_training_session(
                symbols=symbols,
                models_to_train=[ModelType.XGBOOST, ModelType.RANDOM_FOREST],
                lookback_days=self.retraining_config['retraining_data_days'],
                prediction_horizon_minutes=5,
                hyperparameter_tuning=True,
                use_cross_validation=True
            )

            # Record retraining event
            retraining_record = {
                'timestamp': datetime.utcnow(),
                'trigger': RetrainingTrigger.PERFORMANCE_DEGRADATION.value,
                'alert_id': alert.alert_id,
                'session_id': retraining_session_id,
                'drift_type': alert.drift_type.value,
                'severity': alert.severity.value
            }

            self.retraining_history[model_id].append(retraining_record)

            # Mark alert as triggering retraining
            alert.retraining_triggered = True

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Auto-retraining triggered for {model_id}",
                {
                    'session_id': retraining_session_id,
                    'trigger_alert': alert.alert_id,
                    'drift_type': alert.drift_type.value
                }
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'trigger_auto_retraining', 'model_id': alert.model_id}
            )

    async def _can_retrain_model(self, model_id: str) -> bool:
        """Check if model can be retrained based on constraints"""
        try:
            current_time = datetime.utcnow()

            # Check minimum hours between retraining
            recent_retraining = [
                r for r in self.retraining_history.get(model_id, [])
                if (current_time - r['timestamp']).total_seconds() <
                   self.retraining_config['min_hours_between_retraining'] * 3600
            ]

            if recent_retraining:
                return False

            # Check maximum attempts per day
            day_ago = current_time - timedelta(hours=24)
            daily_attempts = [
                r for r in self.retraining_history.get(model_id, [])
                if r['timestamp'] >= day_ago
            ]

            if len(daily_attempts) >= self.retraining_config['max_training_attempts_per_day']:
                return False

            return True

        except Exception:
            return False

    async def _establish_model_baseline(self, model_id: str):
        """Establish baseline metrics for model"""
        try:
            # Try to load existing baseline
            cached_baseline = await self.cache_manager.get(
                CacheNamespace.MODEL_CACHE,
                f"model_baseline:{model_id}"
            )

            if cached_baseline:
                self.model_baselines[model_id] = cached_baseline
                return

            # Create new baseline (simplified - would use historical data in practice)
            baseline = {
                'created_at': datetime.utcnow().isoformat(),
                'baseline_accuracy': 0.65,  # Default baseline
                'baseline_f1': 0.62,
                'baseline_confidence_mean': 0.7,
                'baseline_prediction_dist': [0.3, 0.4, 0.3],  # [down, neutral, up]
                'baseline_features': {}  # Would be populated with historical feature distributions
            }

            self.model_baselines[model_id] = baseline

            # Cache baseline
            await self.cache_manager.set(
                CacheNamespace.MODEL_CACHE,
                f"model_baseline:{model_id}",
                baseline,
                ttl=86400 * 7  # 1 week
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'establish_model_baseline', 'model_id': model_id}
            )

    async def _load_model_baselines(self):
        """Load existing model baselines"""
        try:
            # In practice, you'd load from persistent storage
            # For now, we'll use cache-based loading
            pass

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'load_model_baselines'}
            )

    async def _store_monitoring_metrics(self, metrics: ModelMonitoringMetrics):
        """Store monitoring metrics"""
        try:
            # Store in InfluxDB
            monitoring_data = {
                'recent_accuracy': metrics.recent_accuracy,
                'performance_degradation_pct': metrics.performance_degradation_pct,
                'overall_drift_score': metrics.overall_drift_score,
                'data_quality_score': metrics.data_quality_score,
                'prediction_distribution_shift': metrics.prediction_distribution_shift,
                'confidence_drift_score': metrics.confidence_drift_score,
                'predictions_count_24h': metrics.predictions_count_24h
            }

            await self.influxdb_manager.write_feature_data(
                symbol=f"model_{metrics.model_id}",
                features=monitoring_data,
                timestamp=metrics.timestamp
            )

            # Cache metrics
            await self.cache_manager.set(
                CacheNamespace.SYSTEM_METRICS,
                f"model_monitoring:{metrics.model_id}",
                asdict(metrics),
                ttl=3600  # 1 hour
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'store_monitoring_metrics', 'model_id': metrics.model_id}
            )

    async def get_drift_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get drift status for model"""
        try:
            # Get recent metrics
            cached_metrics = await self.cache_manager.get(
                CacheNamespace.SYSTEM_METRICS,
                f"model_monitoring:{model_id}"
            )

            if not cached_metrics:
                return None

            # Get recent alerts
            recent_alerts = []
            if model_id in self.drift_history:
                recent_alerts = [
                    asdict(alert) for alert in self.drift_history[model_id][-5:]
                ]

            # Get retraining history
            recent_retraining = []
            if model_id in self.retraining_history:
                recent_retraining = self.retraining_history[model_id][-3:]

            return {
                'model_id': model_id,
                'monitoring_metrics': cached_metrics,
                'recent_alerts': recent_alerts,
                'retraining_history': recent_retraining,
                'monitoring_active': model_id in self.monitored_models,
                'can_retrain': await self._can_retrain_model(model_id)
            }

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'get_drift_status', 'model_id': model_id}
            )
            return None

    def get_detector_status(self) -> Dict[str, Any]:
        """Get drift detector status"""
        return {
            'monitoring_active': self.monitoring_active,
            'monitored_models': list(self.monitored_models),
            'thresholds': self.drift_thresholds,
            'retraining_config': self.retraining_config,
            'total_alerts': sum(len(alerts) for alerts in self.drift_history.values()),
            'total_retraining_sessions': sum(len(sessions) for sessions in self.retraining_history.values()),
            'recent_predictions_count': {
                model_id: len(predictions)
                for model_id, predictions in self.recent_predictions.items()
            }
        }

# Global drift detector instance
drift_detector = ModelDriftDetector()

async def get_drift_detector() -> ModelDriftDetector:
    """Get initialized drift detector"""
    if not drift_detector.model_trainer:
        await drift_detector.initialize()
    return drift_detector
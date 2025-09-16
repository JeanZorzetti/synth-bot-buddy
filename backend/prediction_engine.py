"""
🤖 AI Trading Bot - Real-time Prediction Engine
High-performance real-time signal generation and confidence scoring

Author: Claude Code
Created: 2025-01-16
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque
from queue import Queue, Empty
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

from .tick_data_collector import TickData
from .feature_engine import FeatureEngine, MarketFeatures
from .ai_pattern_recognizer import AIPatternRecognizer, PredictionResult, ModelEnsemble
from .training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    min_confidence_threshold: float = 0.7
    signal_strength_threshold: float = 0.6
    ensemble_agreement_threshold: float = 0.8
    signal_cooldown_seconds: float = 30.0
    max_signals_per_minute: int = 5

    # Risk filters
    max_consecutive_signals: int = 3
    min_time_between_signals: float = 10.0
    volatility_filter_enabled: bool = True
    max_volatility_threshold: float = 0.05

@dataclass
class TradingSignal:
    """Trading signal with full context"""
    signal_id: str
    timestamp: float
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    signal_strength: float  # [-1, 1]
    confidence_score: float  # [0, 1]

    # Prediction details
    probability_up: float
    probability_down: float

    # Risk assessment
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    volatility_score: float

    # Model information
    model_version: str
    models_agreement: float  # For ensemble predictions

    # Features context
    key_features: Dict[str, float]
    market_context: Dict[str, Any]

    # Execution guidance
    suggested_position_size: float
    suggested_duration: int  # seconds
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    signals_last_hour: int = 0
    signals_last_day: int = 0

    avg_confidence: float = 0.0
    avg_accuracy: float = 0.0
    win_rate: float = 0.0

    last_signal_time: float = 0.0
    processing_time_avg: float = 0.0
    predictions_per_second: float = 0.0

class SignalValidator:
    """Validate signals before emission"""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.recent_signals = deque(maxlen=100)
        self.consecutive_signals = 0
        self.last_signal_type = None
        self.last_signal_time = 0

    def validate_signal(self, signal: TradingSignal, current_volatility: float) -> Tuple[bool, str]:
        """Validate if signal should be emitted"""

        current_time = time.time()

        # Confidence threshold
        if signal.confidence_score < self.config.min_confidence_threshold:
            return False, f"Confidence {signal.confidence_score:.3f} below threshold {self.config.min_confidence_threshold}"

        # Signal strength threshold
        if abs(signal.signal_strength) < self.config.signal_strength_threshold:
            return False, f"Signal strength {signal.signal_strength:.3f} below threshold {self.config.signal_strength_threshold}"

        # Time-based filters
        time_since_last = current_time - self.last_signal_time
        if time_since_last < self.config.min_time_between_signals:
            return False, f"Too soon after last signal: {time_since_last:.1f}s"

        # Rate limiting
        recent_signals_count = len([s for s in self.recent_signals
                                  if current_time - s.timestamp < 60])
        if recent_signals_count >= self.config.max_signals_per_minute:
            return False, f"Rate limit exceeded: {recent_signals_count} signals in last minute"

        # Consecutive signals limit
        if (signal.signal_type == self.last_signal_type and
            self.consecutive_signals >= self.config.max_consecutive_signals):
            return False, f"Too many consecutive {signal.signal_type} signals: {self.consecutive_signals}"

        # Volatility filter
        if (self.config.volatility_filter_enabled and
            current_volatility > self.config.max_volatility_threshold):
            return False, f"High volatility {current_volatility:.4f} above threshold {self.config.max_volatility_threshold}"

        return True, "Signal validated"

    def record_signal(self, signal: TradingSignal):
        """Record emitted signal for tracking"""
        self.recent_signals.append(signal)

        # Update consecutive counter
        if signal.signal_type == self.last_signal_type:
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1

        self.last_signal_type = signal.signal_type
        self.last_signal_time = signal.timestamp

class RiskAssessment:
    """Assess risk levels for trading signals"""

    @staticmethod
    def assess_signal_risk(prediction: PredictionResult, features: MarketFeatures,
                          recent_volatility: float) -> Tuple[str, float]:
        """Assess risk level of a trading signal"""

        risk_factors = []

        # Confidence-based risk
        if prediction.confidence_score < 0.8:
            risk_factors.append(1 - prediction.confidence_score)

        # Volatility-based risk
        if recent_volatility > 0.03:
            risk_factors.append(recent_volatility * 10)

        # Technical indicator divergence
        if abs(features.rsi_fast - features.rsi_slow) > 20:
            risk_factors.append(0.3)

        # Trend strength
        if abs(features.trend_strength) < 0.3:
            risk_factors.append(0.4)  # Weak trend = higher risk

        # Time-based risk (market hours, etc.)
        if features.time_of_day < 0.2 or features.time_of_day > 0.8:
            risk_factors.append(0.2)  # Off-hours trading

        # Calculate overall risk score
        if not risk_factors:
            risk_score = 0.1  # Minimum risk
        else:
            risk_score = min(sum(risk_factors) / len(risk_factors), 1.0)

        # Categorize risk
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return risk_level, risk_score

    @staticmethod
    def calculate_position_size(prediction: PredictionResult, risk_score: float,
                              base_position_size: float = 1.0) -> float:
        """Calculate suggested position size based on risk"""

        # Kelly Criterion adaptation
        win_probability = prediction.confidence_score

        # Conservative Kelly fraction
        if win_probability > 0.5:
            kelly_fraction = ((win_probability * 2) - 1) / 1  # Assume 1:1 risk/reward
            kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        else:
            kelly_fraction = 0.0

        # Adjust for risk score
        risk_adjustment = 1.0 - risk_score

        # Final position size
        position_size = base_position_size * kelly_fraction * risk_adjustment

        return max(position_size, 0.01)  # Minimum position size

class PredictionEngine:
    """Real-time prediction and signal generation engine"""

    def __init__(self,
                 recognizer: AIPatternRecognizer,
                 feature_engine: FeatureEngine,
                 config: Optional[SignalConfig] = None):

        self.recognizer = recognizer
        self.feature_engine = feature_engine
        self.config = config or SignalConfig()

        # Components
        self.validator = SignalValidator(self.config)
        self.risk_assessor = RiskAssessment()

        # Real-time processing
        self.tick_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=100)
        self.is_running = False
        self.processing_thread = None

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.processing_times = deque(maxlen=1000)

        # Signal callbacks
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []

        # Feature buffer for real-time processing
        self.feature_buffer = deque(maxlen=100)

        logger.info("PredictionEngine initialized")

    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Add callback for when signals are generated"""
        self.signal_callbacks.append(callback)

    def start_processing(self):
        """Start real-time tick processing"""
        if self.is_running:
            logger.warning("PredictionEngine already running")
            return

        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        logger.info("PredictionEngine started")

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

        logger.info("PredictionEngine stopped")

    def add_tick(self, tick: TickData):
        """Add tick for real-time processing"""
        try:
            self.tick_queue.put_nowait(tick)
        except:
            logger.warning("Tick queue full, dropping tick")

    def _processing_loop(self):
        """Main processing loop"""
        logger.info("Starting prediction processing loop")

        while self.is_running:
            try:
                # Get tick from queue
                try:
                    tick = self.tick_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Process tick
                start_time = time.time()
                self._process_tick(tick)
                processing_time = time.time() - start_time

                # Update performance metrics
                self.processing_times.append(processing_time)
                self.metrics.processing_time_avg = statistics.mean(self.processing_times)
                self.metrics.predictions_per_second = 1.0 / processing_time if processing_time > 0 else 0

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue

    def _process_tick(self, tick: TickData):
        """Process single tick and generate prediction"""

        try:
            # Extract features
            features = self.feature_engine.extract_features(tick)
            if features is None:
                return

            # Add to feature buffer
            self.feature_buffer.append(features)

            # Check if we have enough features for prediction
            if len(self.feature_buffer) < self.recognizer.config.sequence_length:
                return

            # Prepare sequence for prediction
            sequence = []
            for feature in list(self.feature_buffer)[-self.recognizer.config.sequence_length:]:
                normalized = self.feature_engine.normalize_features(feature)
                sequence.append(normalized)

            sequence_array = np.array(sequence).reshape(1, len(sequence), -1)

            # Make prediction
            prediction = self.recognizer.model.predict(sequence_array)

            # Create trading signal
            signal = self._create_trading_signal(tick, prediction, features)

            if signal:
                # Validate signal
                current_volatility = features.volatility_rolling
                is_valid, reason = self.validator.validate_signal(signal, current_volatility)

                if is_valid:
                    # Record and emit signal
                    self.validator.record_signal(signal)
                    self._emit_signal(signal)

                    logger.info(f"Signal emitted: {signal.signal_type} for {signal.symbol} "
                              f"(strength: {signal.signal_strength:.3f}, confidence: {signal.confidence_score:.3f})")
                else:
                    logger.debug(f"Signal rejected: {reason}")

        except Exception as e:
            logger.error(f"Error processing tick: {e}")

    def _create_trading_signal(self, tick: TickData, prediction: PredictionResult,
                              features: MarketFeatures) -> Optional[TradingSignal]:
        """Create trading signal from prediction"""

        try:
            # Determine signal type
            if prediction.signal_strength > 0:
                signal_type = "BUY"
            elif prediction.signal_strength < 0:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"

            # Skip HOLD signals
            if signal_type == "HOLD":
                return None

            # Risk assessment
            risk_level, risk_score = self.risk_assessor.assess_signal_risk(
                prediction, features, features.volatility_rolling
            )

            # Position sizing
            suggested_size = self.risk_assessor.calculate_position_size(
                prediction, risk_score
            )

            # Key features for context
            key_features = {
                'price': features.price,
                'velocity_1s': features.velocity_1s,
                'volatility_rolling': features.volatility_rolling,
                'rsi_fast': features.rsi_fast,
                'trend_strength': features.trend_strength,
                'support_resistance': features.support_resistance
            }

            # Market context
            market_context = {
                'time_of_day': features.time_of_day,
                'price_z_score': features.price_z_score,
                'volatility_percentile': features.volatility_percentile,
                'momentum_5': features.momentum_5
            }

            # Create signal
            signal = TradingSignal(
                signal_id=f"signal_{int(time.time() * 1000)}",
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                signal_type=signal_type,
                signal_strength=prediction.signal_strength,
                confidence_score=prediction.confidence_score,
                probability_up=prediction.probability_up,
                probability_down=prediction.probability_down,
                risk_level=risk_level,
                volatility_score=risk_score,
                model_version=self.recognizer.model.version,
                models_agreement=1.0,  # Single model for now
                key_features=key_features,
                market_context=market_context,
                suggested_position_size=suggested_size,
                suggested_duration=300,  # 5 minutes default
            )

            return signal

        except Exception as e:
            logger.error(f"Error creating trading signal: {e}")
            return None

    def _emit_signal(self, signal: TradingSignal):
        """Emit signal to all registered callbacks"""

        # Update metrics
        self.metrics.total_signals += 1
        self.metrics.last_signal_time = signal.timestamp

        # Add to signal queue
        try:
            self.signal_queue.put_nowait(signal)
        except:
            logger.warning("Signal queue full")

        # Call all callbacks
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def get_recent_signals(self, count: int = 10) -> List[TradingSignal]:
        """Get recent signals"""
        return list(self.validator.recent_signals)[-count:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""

        current_time = time.time()

        # Count signals in time windows
        signals_last_hour = len([s for s in self.validator.recent_signals
                               if current_time - s.timestamp < 3600])
        signals_last_day = len([s for s in self.validator.recent_signals
                              if current_time - s.timestamp < 86400])

        # Calculate average confidence
        if self.validator.recent_signals:
            avg_confidence = statistics.mean([s.confidence_score for s in self.validator.recent_signals])
        else:
            avg_confidence = 0.0

        return {
            'is_running': self.is_running,
            'total_signals': self.metrics.total_signals,
            'signals_last_hour': signals_last_hour,
            'signals_last_day': signals_last_day,
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': self.metrics.processing_time_avg * 1000,
            'predictions_per_second': self.metrics.predictions_per_second,
            'tick_queue_size': self.tick_queue.qsize(),
            'signal_queue_size': self.signal_queue.qsize(),
            'feature_buffer_size': len(self.feature_buffer),
            'last_signal_time': self.metrics.last_signal_time
        }

    def update_signal_outcome(self, signal_id: str, was_successful: bool):
        """Update signal outcome for performance tracking"""

        # Find signal
        for signal in self.validator.recent_signals:
            if signal.signal_id == signal_id:
                if was_successful:
                    self.metrics.successful_signals += 1
                else:
                    self.metrics.failed_signals += 1

                # Update win rate
                total_outcomes = self.metrics.successful_signals + self.metrics.failed_signals
                if total_outcomes > 0:
                    self.metrics.win_rate = self.metrics.successful_signals / total_outcomes

                logger.info(f"Signal {signal_id} outcome: {'SUCCESS' if was_successful else 'FAILURE'}")
                break

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get detailed signal statistics"""

        if not self.validator.recent_signals:
            return {'error': 'No signals generated yet'}

        signals = list(self.validator.recent_signals)

        # Signal type distribution
        buy_signals = len([s for s in signals if s.signal_type == "BUY"])
        sell_signals = len([s for s in signals if s.signal_type == "SELL"])

        # Confidence distribution
        confidences = [s.confidence_score for s in signals]

        # Risk distribution
        risk_levels = [s.risk_level for s in signals]
        low_risk = len([r for r in risk_levels if r == "LOW"])
        medium_risk = len([r for r in risk_levels if r == "MEDIUM"])
        high_risk = len([r for r in risk_levels if r == "HIGH"])

        return {
            'total_signals': len(signals),
            'signal_distribution': {
                'BUY': buy_signals,
                'SELL': sell_signals,
                'buy_percentage': (buy_signals / len(signals)) * 100
            },
            'confidence_stats': {
                'min': min(confidences),
                'max': max(confidences),
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences)
            },
            'risk_distribution': {
                'LOW': low_risk,
                'MEDIUM': medium_risk,
                'HIGH': high_risk,
                'low_percentage': (low_risk / len(signals)) * 100
            },
            'time_range': {
                'oldest_signal': min(s.timestamp for s in signals),
                'newest_signal': max(s.timestamp for s in signals)
            }
        }

    def export_signals(self, filepath: str = None) -> str:
        """Export recent signals to file"""

        if filepath is None:
            timestamp = time.time()
            filepath = f"signals_export_{int(timestamp)}.json"

        try:
            signals_data = [asdict(signal) for signal in self.validator.recent_signals]

            export_data = {
                'export_timestamp': time.time(),
                'total_signals': len(signals_data),
                'config': asdict(self.config),
                'performance_metrics': self.get_performance_metrics(),
                'signals': signals_data
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Signals exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            raise
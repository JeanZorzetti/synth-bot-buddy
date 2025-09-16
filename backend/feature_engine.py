"""
🤖 AI Trading Bot - Feature Engineering Engine
Advanced feature extraction for ML model training

Author: Claude Code
Created: 2025-01-16
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
from collections import deque
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib

from .tick_data_collector import TickData, TickSequence

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    """Complete set of market features for ML"""
    # Basic price features
    price: float
    price_change: float
    price_change_pct: float

    # Velocity features
    velocity_1s: float
    velocity_5s: float
    velocity_30s: float
    velocity_60s: float

    # Volatility features
    volatility_5tick: float
    volatility_20tick: float
    volatility_rolling: float

    # Momentum features
    rsi_fast: float
    rsi_slow: float
    momentum_5: float
    momentum_20: float

    # Pattern features
    trend_strength: float
    reversal_signal: float
    support_resistance: float

    # Time features
    timestamp: float
    time_of_day: float
    time_since_last: float

    # Statistical features
    price_z_score: float
    volatility_percentile: float
    volume_intensity: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array([
            self.price, self.price_change, self.price_change_pct,
            self.velocity_1s, self.velocity_5s, self.velocity_30s, self.velocity_60s,
            self.volatility_5tick, self.volatility_20tick, self.volatility_rolling,
            self.rsi_fast, self.rsi_slow, self.momentum_5, self.momentum_20,
            self.trend_strength, self.reversal_signal, self.support_resistance,
            self.time_of_day, self.time_since_last,
            self.price_z_score, self.volatility_percentile, self.volume_intensity
        ])

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names"""
        return [
            'price', 'price_change', 'price_change_pct',
            'velocity_1s', 'velocity_5s', 'velocity_30s', 'velocity_60s',
            'volatility_5tick', 'volatility_20tick', 'volatility_rolling',
            'rsi_fast', 'rsi_slow', 'momentum_5', 'momentum_20',
            'trend_strength', 'reversal_signal', 'support_resistance',
            'time_of_day', 'time_since_last',
            'price_z_score', 'volatility_percentile', 'volume_intensity'
        ]

class PriceVelocityCalculator:
    """Calculate price velocity across multiple timeframes"""

    def __init__(self, max_history: int = 1000):
        self.price_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)

    def add_tick(self, tick: TickData):
        """Add new tick for velocity calculation"""
        self.price_history.append(tick.price)
        self.timestamp_history.append(tick.timestamp)

    def calculate_velocity(self, timeframe_seconds: float) -> float:
        """Calculate price velocity over specified timeframe"""
        if len(self.price_history) < 2:
            return 0.0

        current_time = self.timestamp_history[-1]
        cutoff_time = current_time - timeframe_seconds

        # Find prices within timeframe
        relevant_prices = []
        relevant_times = []

        for i in range(len(self.timestamp_history) - 1, -1, -1):
            if self.timestamp_history[i] >= cutoff_time:
                relevant_prices.append(self.price_history[i])
                relevant_times.append(self.timestamp_history[i])
            else:
                break

        if len(relevant_prices) < 2:
            return 0.0

        # Calculate velocity as price change per second
        price_change = relevant_prices[0] - relevant_prices[-1]  # Most recent - oldest
        time_change = relevant_times[0] - relevant_times[-1]

        if time_change == 0:
            return 0.0

        return price_change / time_change

    def get_all_velocities(self) -> Dict[str, float]:
        """Get velocities for all standard timeframes"""
        return {
            'velocity_1s': self.calculate_velocity(1.0),
            'velocity_5s': self.calculate_velocity(5.0),
            'velocity_30s': self.calculate_velocity(30.0),
            'velocity_60s': self.calculate_velocity(60.0)
        }

class VolatilityAnalyzer:
    """Advanced volatility calculation and analysis"""

    def __init__(self, max_history: int = 1000):
        self.price_history = deque(maxlen=max_history)
        self.returns_history = deque(maxlen=max_history)
        self.volatility_history = deque(maxlen=100)

    def add_tick(self, tick: TickData):
        """Add tick and calculate returns"""
        if self.price_history:
            last_price = self.price_history[-1]
            return_pct = (tick.price - last_price) / last_price
            self.returns_history.append(return_pct)

        self.price_history.append(tick.price)

    def calculate_volatility_rolling(self, window: int = 20) -> float:
        """Calculate rolling volatility"""
        if len(self.returns_history) < window:
            return 0.0

        recent_returns = list(self.returns_history)[-window:]
        return float(np.std(recent_returns))

    def calculate_volatility_tick_based(self, tick_count: int) -> float:
        """Calculate volatility based on recent tick count"""
        if len(self.price_history) < tick_count:
            return 0.0

        recent_prices = list(self.price_history)[-tick_count:]
        price_changes = [recent_prices[i] - recent_prices[i-1]
                        for i in range(1, len(recent_prices))]

        if not price_changes:
            return 0.0

        return float(np.std(price_changes))

    def get_volatility_percentile(self) -> float:
        """Get current volatility percentile"""
        current_vol = self.calculate_volatility_rolling()

        if len(self.volatility_history) < 10:
            self.volatility_history.append(current_vol)
            return 50.0  # Default to median

        self.volatility_history.append(current_vol)

        # Calculate percentile
        sorted_vols = sorted(self.volatility_history)
        position = sorted_vols.index(current_vol)
        percentile = (position / len(sorted_vols)) * 100

        return percentile

    def get_all_volatilities(self) -> Dict[str, float]:
        """Get all volatility measures"""
        return {
            'volatility_5tick': self.calculate_volatility_tick_based(5),
            'volatility_20tick': self.calculate_volatility_tick_based(20),
            'volatility_rolling': self.calculate_volatility_rolling(),
            'volatility_percentile': self.get_volatility_percentile()
        }

class MomentumIndicators:
    """Technical momentum indicators for tick data"""

    def __init__(self, max_history: int = 1000):
        self.price_history = deque(maxlen=max_history)
        self.rsi_calculator = RSICalculator()

    def add_tick(self, tick: TickData):
        """Add tick for momentum calculation"""
        self.price_history.append(tick.price)
        self.rsi_calculator.add_price(tick.price)

    def calculate_momentum(self, period: int) -> float:
        """Calculate momentum over specified period"""
        if len(self.price_history) < period + 1:
            return 0.0

        current_price = self.price_history[-1]
        period_ago_price = self.price_history[-period-1]

        if period_ago_price == 0:
            return 0.0

        return (current_price - period_ago_price) / period_ago_price

    def calculate_trend_strength(self) -> float:
        """Calculate overall trend strength"""
        if len(self.price_history) < 20:
            return 0.0

        recent_prices = list(self.price_history)[-20:]

        # Linear regression to measure trend
        x = np.arange(len(recent_prices))
        slope, _, r_value, _, _ = stats.linregress(x, recent_prices)

        # Trend strength is R-squared weighted by slope
        trend_strength = (r_value ** 2) * np.sign(slope)
        return float(trend_strength)

    def detect_reversal_signal(self) -> float:
        """Detect potential reversal patterns"""
        if len(self.price_history) < 10:
            return 0.0

        recent_prices = list(self.price_history)[-10:]

        # Simple reversal detection: check for V or Λ patterns
        mid_point = len(recent_prices) // 2
        first_half = recent_prices[:mid_point]
        second_half = recent_prices[mid_point:]

        first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
        second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]

        # Reversal strength
        reversal_strength = abs(first_trend - second_trend)

        # Direction: positive for bullish reversal, negative for bearish
        if first_trend < 0 and second_trend > 0:
            return reversal_strength  # Bullish reversal
        elif first_trend > 0 and second_trend < 0:
            return -reversal_strength  # Bearish reversal

        return 0.0

    def get_all_momentum(self) -> Dict[str, float]:
        """Get all momentum indicators"""
        return {
            'rsi_fast': self.rsi_calculator.get_rsi(5),
            'rsi_slow': self.rsi_calculator.get_rsi(14),
            'momentum_5': self.calculate_momentum(5),
            'momentum_20': self.calculate_momentum(20),
            'trend_strength': self.calculate_trend_strength(),
            'reversal_signal': self.detect_reversal_signal()
        }

class RSICalculator:
    """Optimized RSI calculation for real-time ticks"""

    def __init__(self, max_history: int = 100):
        self.price_history = deque(maxlen=max_history)
        self.gains = deque(maxlen=max_history)
        self.losses = deque(maxlen=max_history)

    def add_price(self, price: float):
        """Add new price and calculate gain/loss"""
        if self.price_history:
            change = price - self.price_history[-1]
            gain = max(change, 0)
            loss = max(-change, 0)

            self.gains.append(gain)
            self.losses.append(loss)

        self.price_history.append(price)

    def get_rsi(self, period: int = 14) -> float:
        """Calculate RSI for specified period"""
        if len(self.gains) < period:
            return 50.0  # Neutral RSI

        recent_gains = list(self.gains)[-period:]
        recent_losses = list(self.losses)[-period:]

        avg_gain = sum(recent_gains) / len(recent_gains)
        avg_loss = sum(recent_losses) / len(recent_losses)

        if avg_loss == 0:
            return 100.0  # No losses, maximum RSI

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

class SupportResistanceDetector:
    """Detect support and resistance levels from tick data"""

    def __init__(self, max_history: int = 500):
        self.price_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        self.levels = {'support': [], 'resistance': []}

    def add_tick(self, tick: TickData):
        """Add tick for support/resistance analysis"""
        self.price_history.append(tick.price)
        self.timestamp_history.append(tick.timestamp)

        # Update levels periodically
        if len(self.price_history) % 50 == 0:
            self._update_levels()

    def _update_levels(self):
        """Update support and resistance levels"""
        if len(self.price_history) < 50:
            return

        prices = list(self.price_history)

        # Find local peaks and valleys
        peaks = []
        valleys = []

        for i in range(2, len(prices) - 2):
            # Peak detection
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                peaks.append(prices[i])

            # Valley detection
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                valleys.append(prices[i])

        # Cluster similar levels
        self.levels['resistance'] = self._cluster_levels(peaks)
        self.levels['support'] = self._cluster_levels(valleys)

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.001) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []

        clustered = []
        sorted_levels = sorted(levels)

        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) <= tolerance * current_cluster[-1]:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add final cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered

    def get_nearest_level(self, current_price: float) -> float:
        """Get distance to nearest support/resistance level"""
        all_levels = self.levels['support'] + self.levels['resistance']

        if not all_levels:
            return 0.0

        distances = [abs(current_price - level) for level in all_levels]
        min_distance = min(distances)

        # Normalize by current price
        return min_distance / current_price

class FeatureEngine:
    """Main feature engineering engine for AI model"""

    def __init__(self, max_history: int = 2000):
        self.max_history = max_history

        # Feature calculators
        self.velocity_calc = PriceVelocityCalculator(max_history)
        self.volatility_analyzer = VolatilityAnalyzer(max_history)
        self.momentum_indicators = MomentumIndicators(max_history)
        self.support_resistance = SupportResistanceDetector(max_history)

        # Price statistics
        self.price_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)

        # Scalers for normalization
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.is_fitted = False

        # Performance metrics
        self.features_generated = 0
        self.processing_time_total = 0.0

        logger.info("FeatureEngine initialized")

    def add_tick(self, tick: TickData):
        """Add tick to all feature calculators"""
        self.velocity_calc.add_tick(tick)
        self.volatility_analyzer.add_tick(tick)
        self.momentum_indicators.add_tick(tick)
        self.support_resistance.add_tick(tick)

        self.price_history.append(tick.price)
        self.timestamp_history.append(tick.timestamp)

    def extract_features(self, tick: TickData) -> Optional[MarketFeatures]:
        """Extract complete feature set for a tick"""
        start_time = time.time()

        try:
            # Add tick to calculators
            self.add_tick(tick)

            if len(self.price_history) < 2:
                return None

            # Basic price features
            prev_price = self.price_history[-2]
            price_change = tick.price - prev_price
            price_change_pct = (price_change / prev_price) if prev_price != 0 else 0.0

            # Get velocity features
            velocity_features = self.velocity_calc.get_all_velocities()

            # Get volatility features
            volatility_features = self.volatility_analyzer.get_all_volatilities()

            # Get momentum features
            momentum_features = self.momentum_indicators.get_all_momentum()

            # Time features
            time_of_day = (tick.timestamp % 86400) / 86400  # Normalized time of day
            time_since_last = tick.timestamp - self.timestamp_history[-2] if len(self.timestamp_history) > 1 else 0

            # Statistical features
            price_z_score = self._calculate_price_z_score(tick.price)
            volume_intensity = tick.volume if tick.volume else 0.0

            # Support/resistance
            support_resistance = self.support_resistance.get_nearest_level(tick.price)

            # Create feature object
            features = MarketFeatures(
                price=tick.price,
                price_change=price_change,
                price_change_pct=price_change_pct,
                velocity_1s=velocity_features['velocity_1s'],
                velocity_5s=velocity_features['velocity_5s'],
                velocity_30s=velocity_features['velocity_30s'],
                velocity_60s=velocity_features['velocity_60s'],
                volatility_5tick=volatility_features['volatility_5tick'],
                volatility_20tick=volatility_features['volatility_20tick'],
                volatility_rolling=volatility_features['volatility_rolling'],
                rsi_fast=momentum_features['rsi_fast'],
                rsi_slow=momentum_features['rsi_slow'],
                momentum_5=momentum_features['momentum_5'],
                momentum_20=momentum_features['momentum_20'],
                trend_strength=momentum_features['trend_strength'],
                reversal_signal=momentum_features['reversal_signal'],
                support_resistance=support_resistance,
                timestamp=tick.timestamp,
                time_of_day=time_of_day,
                time_since_last=time_since_last,
                price_z_score=price_z_score,
                volatility_percentile=volatility_features['volatility_percentile'],
                volume_intensity=volume_intensity
            )

            # Update metrics
            self.features_generated += 1
            self.processing_time_total += time.time() - start_time

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def _calculate_price_z_score(self, current_price: float) -> float:
        """Calculate z-score for current price"""
        if len(self.price_history) < 20:
            return 0.0

        recent_prices = list(self.price_history)[-20:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)

        if std_price == 0:
            return 0.0

        return (current_price - mean_price) / std_price

    def prepare_training_data(self, ticks: List[TickData], target_lookahead: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from tick sequence"""
        try:
            features_list = []
            targets_list = []

            # Reset calculators for clean feature extraction
            self._reset_calculators()

            for i, tick in enumerate(ticks):
                features = self.extract_features(tick)

                if features is None:
                    continue

                # Calculate target (future price direction)
                if i + target_lookahead < len(ticks):
                    future_price = ticks[i + target_lookahead].price
                    current_price = tick.price

                    # Binary classification: 1 for price increase, 0 for decrease
                    target = 1 if future_price > current_price else 0

                    features_list.append(features.to_array())
                    targets_list.append(target)

            if not features_list:
                return np.array([]), np.array([])

            X = np.array(features_list)
            y = np.array(targets_list)

            # Fit scalers if not already fitted
            if not self.is_fitted:
                self.feature_scaler.fit(X)
                self.is_fitted = True

            # Normalize features
            X_scaled = self.feature_scaler.transform(X)

            logger.info(f"Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

            return X_scaled, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def _reset_calculators(self):
        """Reset all calculators for clean feature extraction"""
        self.velocity_calc = PriceVelocityCalculator(self.max_history)
        self.volatility_analyzer = VolatilityAnalyzer(self.max_history)
        self.momentum_indicators = MomentumIndicators(self.max_history)
        self.support_resistance = SupportResistanceDetector(self.max_history)

        self.price_history.clear()
        self.timestamp_history.clear()

    def normalize_features(self, features: MarketFeatures) -> np.ndarray:
        """Normalize features for model input"""
        if not self.is_fitted:
            logger.warning("Feature scaler not fitted. Returning raw features.")
            return features.to_array()

        features_array = features.to_array().reshape(1, -1)
        return self.feature_scaler.transform(features_array).flatten()

    def get_feature_importance_stats(self) -> Dict[str, Any]:
        """Get statistics about feature generation"""
        avg_processing_time = (self.processing_time_total / self.features_generated
                             if self.features_generated > 0 else 0)

        return {
            'features_generated': self.features_generated,
            'total_processing_time': self.processing_time_total,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'features_per_second': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'scaler_fitted': self.is_fitted,
            'feature_count': len(MarketFeatures.get_feature_names()),
            'feature_names': MarketFeatures.get_feature_names()
        }

    def save_scaler(self, filepath: str) -> bool:
        """Save fitted scaler to file"""
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'feature_scaler': self.feature_scaler,
                    'is_fitted': self.is_fitted
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
            return False

    def load_scaler(self, filepath: str) -> bool:
        """Load fitted scaler from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.feature_scaler = data['feature_scaler']
                self.is_fitted = data['is_fitted']
            return True
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return False
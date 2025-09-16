"""
🤖 AI Trading Bot - Tick Data Collector
Real-time tick-by-tick data collection and processing for ML training

Author: Claude Code
Created: 2025-01-16
"""

import asyncio
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Structure for individual tick data"""
    symbol: str
    price: float
    timestamp: float  # Unix timestamp with milliseconds
    epoch: int       # Deriv epoch
    ask: Optional[float] = None
    bid: Optional[float] = None
    spread: Optional[float] = None
    volume: Optional[int] = None

    def __post_init__(self):
        """Calculate derived fields"""
        if self.ask and self.bid:
            self.spread = round(self.ask - self.bid, 5)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def to_ml_features(self) -> Dict[str, float]:
        """Convert to ML-ready features"""
        return {
            'price': self.price,
            'timestamp': self.timestamp,
            'spread': self.spread or 0.0,
            'volume': float(self.volume or 0)
        }

@dataclass
class TickSequence:
    """Sequence of ticks for pattern analysis"""
    symbol: str
    ticks: List[TickData]
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return self.end_time - self.start_time

    @property
    def tick_count(self) -> int:
        """Number of ticks in sequence"""
        return len(self.ticks)

    @property
    def price_range(self) -> Dict[str, float]:
        """Price statistics for the sequence"""
        if not self.ticks:
            return {'min': 0, 'max': 0, 'avg': 0, 'volatility': 0}

        prices = [tick.price for tick in self.ticks]
        return {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices),
            'volatility': float(np.std(prices)) if len(prices) > 1 else 0.0
        }

class CircularTickBuffer:
    """High-performance circular buffer for tick storage"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
        self._total_ticks = 0

    def add_tick(self, tick: TickData) -> None:
        """Add tick to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(tick)
            self._total_ticks += 1

    def get_recent_ticks(self, count: int) -> List[TickData]:
        """Get most recent N ticks"""
        with self.lock:
            if count >= len(self.buffer):
                return list(self.buffer)
            return list(self.buffer)[-count:]

    def get_ticks_in_timeframe(self, seconds: float) -> List[TickData]:
        """Get ticks from last N seconds"""
        current_time = time.time()
        cutoff_time = current_time - seconds

        with self.lock:
            return [tick for tick in self.buffer
                   if tick.timestamp >= cutoff_time]

    def get_tick_sequence(self, start_time: float, end_time: float) -> TickSequence:
        """Get tick sequence for specific time range"""
        with self.lock:
            filtered_ticks = [tick for tick in self.buffer
                            if start_time <= tick.timestamp <= end_time]

            return TickSequence(
                symbol=filtered_ticks[0].symbol if filtered_ticks else "UNKNOWN",
                ticks=filtered_ticks,
                start_time=start_time,
                end_time=end_time
            )

    @property
    def size(self) -> int:
        """Current buffer size"""
        return len(self.buffer)

    @property
    def total_ticks_processed(self) -> int:
        """Total ticks processed since start"""
        return self._total_ticks

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            if not self.buffer:
                return {
                    'size': 0,
                    'total_processed': self._total_ticks,
                    'oldest_tick': None,
                    'newest_tick': None,
                    'timespan_seconds': 0
                }

            oldest = self.buffer[0]
            newest = self.buffer[-1]

            return {
                'size': len(self.buffer),
                'total_processed': self._total_ticks,
                'oldest_tick': oldest.timestamp,
                'newest_tick': newest.timestamp,
                'timespan_seconds': newest.timestamp - oldest.timestamp,
                'symbols': list(set(tick.symbol for tick in self.buffer))
            }

class TickDataValidator:
    """Data validation and cleaning for tick data"""

    def __init__(self):
        self.validation_stats = {
            'total_ticks': 0,
            'valid_ticks': 0,
            'invalid_ticks': 0,
            'outliers_removed': 0,
            'errors': []
        }

    def validate_tick(self, tick_data: Dict[str, Any]) -> Optional[TickData]:
        """Validate and clean tick data"""
        self.validation_stats['total_ticks'] += 1

        try:
            # Basic validation
            if not tick_data.get('quote'):
                self._log_error("Missing quote data")
                return None

            quote = tick_data['quote']
            price = float(quote)

            # Price validation
            if price <= 0:
                self._log_error(f"Invalid price: {price}")
                return None

            # Extract symbol
            symbol = tick_data.get('symbol', 'UNKNOWN')

            # Extract timestamp
            timestamp = time.time()  # Current time if not provided
            if 'epoch' in tick_data:
                timestamp = float(tick_data['epoch'])

            # Create tick data
            tick = TickData(
                symbol=symbol,
                price=price,
                timestamp=timestamp,
                epoch=tick_data.get('epoch', int(timestamp)),
                ask=tick_data.get('ask'),
                bid=tick_data.get('bid')
            )

            # Outlier detection
            if self._is_outlier(tick):
                self.validation_stats['outliers_removed'] += 1
                self._log_error(f"Outlier detected: {price}")
                return None

            self.validation_stats['valid_ticks'] += 1
            return tick

        except Exception as e:
            self._log_error(f"Validation error: {str(e)}")
            return None

    def _is_outlier(self, tick: TickData) -> bool:
        """Simple outlier detection"""
        # Basic range check - adjust based on asset
        if tick.symbol.startswith('R_'):  # Synthetic indices
            return tick.price < 0.1 or tick.price > 10000
        return False

    def _log_error(self, error: str) -> None:
        """Log validation error"""
        self.validation_stats['invalid_ticks'] += 1
        self.validation_stats['errors'].append({
            'timestamp': time.time(),
            'error': error
        })
        logger.warning(f"Tick validation error: {error}")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats['total_ticks'] > 0:
            stats['success_rate'] = stats['valid_ticks'] / stats['total_ticks']
        else:
            stats['success_rate'] = 0.0
        return stats

class TickDataCollector:
    """Main tick data collector for AI training"""

    def __init__(self, deriv_api, buffer_size: int = 50000):
        self.deriv_api = deriv_api
        self.buffers: Dict[str, CircularTickBuffer] = {}
        self.validator = TickDataValidator()
        self.buffer_size = buffer_size

        # Streaming control
        self.is_streaming = False
        self.subscribed_symbols: List[str] = []

        # Callbacks for real-time processing
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.sequence_callbacks: List[Callable[[TickSequence], None]] = []

        # Performance metrics
        self.metrics = {
            'ticks_per_second': 0,
            'last_tick_time': 0,
            'total_symbols': 0,
            'streaming_duration': 0
        }

        logger.info("TickDataCollector initialized")

    def add_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """Add callback for each new tick"""
        self.tick_callbacks.append(callback)

    def add_sequence_callback(self, callback: Callable[[TickSequence], None]) -> None:
        """Add callback for tick sequences"""
        self.sequence_callbacks.append(callback)

    async def start_streaming(self, symbols: List[str]) -> bool:
        """Start real-time tick streaming"""
        try:
            logger.info(f"Starting tick streaming for symbols: {symbols}")

            self.subscribed_symbols = symbols
            self.is_streaming = True

            # Initialize buffers for each symbol
            for symbol in symbols:
                if symbol not in self.buffers:
                    self.buffers[symbol] = CircularTickBuffer(self.buffer_size)

            # Subscribe to tick streams
            for symbol in symbols:
                success = await self._subscribe_to_symbol(symbol)
                if not success:
                    logger.error(f"Failed to subscribe to {symbol}")
                    return False

            logger.info(f"Successfully started streaming for {len(symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"Error starting tick streaming: {e}")
            return False

    async def stop_streaming(self) -> None:
        """Stop tick streaming"""
        logger.info("Stopping tick streaming...")
        self.is_streaming = False

        # Unsubscribe from all symbols
        for symbol in self.subscribed_symbols:
            try:
                await self._unsubscribe_from_symbol(symbol)
            except Exception as e:
                logger.error(f"Error unsubscribing from {symbol}: {e}")

        self.subscribed_symbols.clear()
        logger.info("Tick streaming stopped")

    async def _subscribe_to_symbol(self, symbol: str) -> bool:
        """Subscribe to tick stream for specific symbol"""
        try:
            # Use deriv_api to subscribe to ticks
            request = {
                "ticks": symbol,
                "subscribe": 1
            }

            response = await self.deriv_api._send_request(request)

            if response and 'tick' in response:
                # Process initial tick
                await self._process_tick_response(response)
                return True

            return 'error' not in response

        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False

    async def _unsubscribe_from_symbol(self, symbol: str) -> None:
        """Unsubscribe from tick stream"""
        try:
            request = {
                "forget_all": "ticks"
            }
            await self.deriv_api._send_request(request)
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")

    async def _process_tick_response(self, response: Dict[str, Any]) -> None:
        """Process incoming tick data"""
        try:
            if 'tick' not in response:
                return

            tick_data = response['tick']

            # Validate and clean tick data
            validated_tick = self.validator.validate_tick(tick_data)
            if not validated_tick:
                return

            # Store in appropriate buffer
            symbol = validated_tick.symbol
            if symbol not in self.buffers:
                self.buffers[symbol] = CircularTickBuffer(self.buffer_size)

            self.buffers[symbol].add_tick(validated_tick)

            # Update metrics
            self._update_metrics()

            # Call registered callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(validated_tick)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

            logger.debug(f"Processed tick for {symbol}: {validated_tick.price}")

        except Exception as e:
            logger.error(f"Error processing tick response: {e}")

    def _update_metrics(self) -> None:
        """Update performance metrics"""
        current_time = time.time()

        if self.metrics['last_tick_time'] > 0:
            time_diff = current_time - self.metrics['last_tick_time']
            if time_diff > 0:
                self.metrics['ticks_per_second'] = 1.0 / time_diff

        self.metrics['last_tick_time'] = current_time
        self.metrics['total_symbols'] = len(self.buffers)

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[TickData]:
        """Get recent ticks for symbol"""
        if symbol not in self.buffers:
            return []
        return self.buffers[symbol].get_recent_ticks(count)

    def get_ticks_timeframe(self, symbol: str, seconds: float) -> List[TickData]:
        """Get ticks from last N seconds"""
        if symbol not in self.buffers:
            return []
        return self.buffers[symbol].get_ticks_in_timeframe(seconds)

    def prepare_training_dataset(self, symbol: str, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Prepare dataset for ML training"""
        try:
            if symbol not in self.buffers:
                return {'error': f'No data for symbol {symbol}'}

            # Get ticks from specified timeframe
            seconds = timeframe_hours * 3600
            ticks = self.buffers[symbol].get_ticks_in_timeframe(seconds)

            if len(ticks) < 100:
                return {'error': f'Insufficient data: {len(ticks)} ticks'}

            # Convert to ML features
            features = []
            targets = []

            for i in range(len(ticks) - 1):
                current_tick = ticks[i]
                next_tick = ticks[i + 1]

                # Feature vector
                feature = current_tick.to_ml_features()
                features.append(feature)

                # Target (price direction)
                price_change = next_tick.price - current_tick.price
                target = 1 if price_change > 0 else 0  # Binary classification
                targets.append(target)

            return {
                'symbol': symbol,
                'features': features,
                'targets': targets,
                'timeframe_hours': timeframe_hours,
                'total_samples': len(features),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error preparing training dataset: {e}")
            return {'error': str(e)}

    def get_collector_stats(self) -> Dict[str, Any]:
        """Get comprehensive collector statistics"""
        stats = {
            'streaming': self.is_streaming,
            'subscribed_symbols': self.subscribed_symbols.copy(),
            'metrics': self.metrics.copy(),
            'validation': self.validator.get_validation_stats(),
            'buffers': {}
        }

        # Buffer statistics for each symbol
        for symbol, buffer in self.buffers.items():
            stats['buffers'][symbol] = buffer.get_stats()

        return stats

    def normalize_tick_data(self, ticks: List[TickData]) -> List[Dict[str, float]]:
        """Normalize tick data for ML processing"""
        if not ticks:
            return []

        try:
            # Extract prices for normalization
            prices = [tick.price for tick in ticks]

            if len(prices) < 2:
                return [tick.to_ml_features() for tick in ticks]

            # Calculate normalization parameters
            price_mean = np.mean(prices)
            price_std = np.std(prices)

            if price_std == 0:
                price_std = 1  # Avoid division by zero

            # Normalize and return
            normalized = []
            for tick in ticks:
                features = tick.to_ml_features()
                features['price_normalized'] = (tick.price - price_mean) / price_std
                normalized.append(features)

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing tick data: {e}")
            return [tick.to_ml_features() for tick in ticks]
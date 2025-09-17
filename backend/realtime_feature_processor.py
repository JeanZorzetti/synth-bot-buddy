"""
Real-Time Feature Processing Pipeline - Phase 13 Real-Time Data Pipeline
Sistema de processamento de features em tempo real com dados de mercado reais
"""

import asyncio
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import logging

from real_deriv_websocket import RealDerivWebSocket, TickData, CandleData, get_deriv_websocket
from influxdb_timeseries import get_influxdb_manager
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

@dataclass
class ProcessedFeatures:
    symbol: str
    timestamp: datetime

    # Price-based features
    price: float
    sma_5: float
    sma_10: float
    sma_20: float
    sma_50: float
    ema_5: float
    ema_10: float
    ema_20: float
    ema_50: float

    # Momentum indicators
    rsi_14: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    momentum_10: float
    roc_10: float

    # Volatility indicators
    atr_14: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_width: float
    bollinger_position: float

    # Volume indicators (when available)
    volume_sma_10: float
    volume_ratio: float
    vwap: float

    # Stochastic indicators
    stoch_k: float
    stoch_d: float
    williams_r: float

    # Price patterns
    price_change_1: float
    price_change_5: float
    price_change_10: float
    volatility_5: float
    volatility_10: float

    # Market microstructure
    bid_ask_spread: float
    spread_ratio: float
    tick_direction: float
    tick_intensity: float

    # Advanced features
    hurst_exponent: float
    fractal_dimension: float
    entropy: float
    autocorr_1: float
    autocorr_5: float

class RealTimeFeatureProcessor:
    """Real-time feature processing pipeline for market data"""

    def __init__(self):
        # Data storage
        self.tick_buffers: Dict[str, deque] = {}  # symbol -> deque of ticks
        self.candle_buffers: Dict[str, deque] = {}  # symbol -> deque of candles
        self.feature_history: Dict[str, deque] = {}  # symbol -> deque of features

        # Buffer configuration
        self.max_tick_buffer_size = 1000
        self.max_candle_buffer_size = 200
        self.max_feature_history_size = 100

        # Feature calculation parameters
        self.min_ticks_for_features = 50
        self.min_candles_for_features = 30

        # WebSocket and storage
        self.websocket_client: Optional[RealDerivWebSocket] = None
        self.influxdb_manager = None
        self.cache_manager = None

        # Feature callbacks
        self.feature_callbacks: List[Callable[[ProcessedFeatures], None]] = []

        # Processing status
        self.processing_active = False
        self.symbols_subscribed: set = set()

        # Logging
        self.logger = logging_system.loggers.get('ai_engine', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize feature processor"""
        try:
            # Initialize dependencies
            self.websocket_client = await get_deriv_websocket()
            self.influxdb_manager = await get_influxdb_manager()
            self.cache_manager = await get_cache_manager()

            # Register callbacks for real-time data
            self.websocket_client.add_tick_callback(self._process_tick_data)
            self.websocket_client.add_candle_callback(self._process_candle_data)

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                "Real-time feature processor initialized"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'initialize_feature_processor'}
            )
            raise

    async def start_processing(self, symbols: List[str]):
        """Start real-time feature processing for symbols"""
        try:
            self.processing_active = True

            # Initialize buffers for symbols
            for symbol in symbols:
                if symbol not in self.tick_buffers:
                    self.tick_buffers[symbol] = deque(maxlen=self.max_tick_buffer_size)
                    self.candle_buffers[symbol] = deque(maxlen=self.max_candle_buffer_size)
                    self.feature_history[symbol] = deque(maxlen=self.max_feature_history_size)

                # Subscribe to real-time data
                success = await self.websocket_client.subscribe_ticks(symbol)
                if success:
                    self.symbols_subscribed.add(symbol)
                    logging_system.log(
                        LogComponent.AI_ENGINE,
                        LogLevel.INFO,
                        f"Started feature processing for {symbol}"
                    )

                # Load historical data to populate buffers
                await self._load_historical_data(symbol)

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                f"Feature processing started for {len(self.symbols_subscribed)} symbols"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'start_processing', 'symbols': symbols}
            )
            raise

    async def stop_processing(self):
        """Stop real-time feature processing"""
        try:
            self.processing_active = False

            # Unsubscribe from all symbols
            for symbol in list(self.symbols_subscribed):
                await self.websocket_client.unsubscribe_ticks(symbol)

            self.symbols_subscribed.clear()

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.INFO,
                "Feature processing stopped"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'stop_processing'}
            )

    async def _load_historical_data(self, symbol: str):
        """Load historical data to populate buffers"""
        try:
            # Load recent candle data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)

            # Get historical candles from InfluxDB
            candles_df = await self.influxdb_manager.query_candle_data(
                symbol=symbol,
                granularity='1m',
                start_time=start_time,
                end_time=end_time,
                limit=200
            )

            if not candles_df.empty:
                # Convert DataFrame to CandleData objects
                for _, row in candles_df.iterrows():
                    candle = CandleData(
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=int(row['volume']) if pd.notna(row['volume']) else 0,
                        epoch=int(row['timestamp'].timestamp())
                    )
                    self.candle_buffers[symbol].append(candle)

                logging_system.log(
                    LogComponent.AI_ENGINE,
                    LogLevel.INFO,
                    f"Loaded {len(candles_df)} historical candles for {symbol}"
                )

            # Get historical tick data (last hour)
            tick_start_time = end_time - timedelta(hours=1)
            ticks_df = await self.influxdb_manager.query_tick_data(
                symbol=symbol,
                start_time=tick_start_time,
                end_time=end_time,
                limit=1000
            )

            if not ticks_df.empty:
                # Convert DataFrame to TickData objects
                for _, row in ticks_df.iterrows():
                    tick = TickData(
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        bid=row['bid'],
                        ask=row['ask'],
                        price=row['price'],
                        spread=row['spread'],
                        pip_size=row['pip_size'],
                        quote_id='',
                        epoch=int(row['timestamp'].timestamp())
                    )
                    self.tick_buffers[symbol].append(tick)

                logging_system.log(
                    LogComponent.AI_ENGINE,
                    LogLevel.INFO,
                    f"Loaded {len(ticks_df)} historical ticks for {symbol}"
                )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'load_historical_data', 'symbol': symbol}
            )

    async def _process_tick_data(self, tick: TickData):
        """Process incoming tick data"""
        try:
            if not self.processing_active or tick.symbol not in self.symbols_subscribed:
                return

            # Add to buffer
            self.tick_buffers[tick.symbol].append(tick)

            # Check if we have enough data for feature calculation
            if len(self.tick_buffers[tick.symbol]) >= self.min_ticks_for_features:
                # Calculate features asynchronously
                asyncio.create_task(self._calculate_tick_features(tick.symbol))

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'process_tick_data', 'symbol': tick.symbol}
            )

    async def _process_candle_data(self, candle: CandleData):
        """Process incoming candle data"""
        try:
            if not self.processing_active or candle.symbol not in self.symbols_subscribed:
                return

            # Add to buffer
            self.candle_buffers[candle.symbol].append(candle)

            # Calculate features with candle data
            if len(self.candle_buffers[candle.symbol]) >= self.min_candles_for_features:
                asyncio.create_task(self._calculate_candle_features(candle.symbol))

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'process_candle_data', 'symbol': candle.symbol}
            )

    async def _calculate_tick_features(self, symbol: str):
        """Calculate features from tick data"""
        try:
            ticks = list(self.tick_buffers[symbol])
            if len(ticks) < self.min_ticks_for_features:
                return

            # Convert to numpy arrays for efficient calculation
            prices = np.array([tick.price for tick in ticks])
            timestamps = np.array([tick.timestamp for tick in ticks])
            spreads = np.array([tick.spread for tick in ticks])
            bids = np.array([tick.bid for tick in ticks])
            asks = np.array([tick.ask for tick in ticks])

            # Calculate microstructure features
            latest_tick = ticks[-1]

            # Price changes
            price_changes_1 = np.diff(prices[-2:])
            price_changes_5 = np.diff(prices[-6:-1]) if len(prices) >= 6 else np.array([0])
            price_changes_10 = np.diff(prices[-11:-1]) if len(prices) >= 11 else np.array([0])

            # Volatility
            volatility_5 = np.std(prices[-5:]) if len(prices) >= 5 else 0
            volatility_10 = np.std(prices[-10:]) if len(prices) >= 10 else 0

            # Tick direction and intensity
            tick_direction = 1 if len(price_changes_1) > 0 and price_changes_1[0] > 0 else (-1 if len(price_changes_1) > 0 and price_changes_1[0] < 0 else 0)
            tick_intensity = np.abs(price_changes_1[0]) if len(price_changes_1) > 0 else 0

            # Spread analysis
            avg_spread = np.mean(spreads[-10:]) if len(spreads) >= 10 else spreads[-1]
            spread_ratio = latest_tick.spread / avg_spread if avg_spread > 0 else 1

            # Advanced features
            hurst_exponent = self._calculate_hurst_exponent(prices[-100:] if len(prices) >= 100 else prices)
            entropy = self._calculate_entropy(prices[-50:] if len(prices) >= 50 else prices)
            autocorr_1 = self._calculate_autocorrelation(prices, lag=1)
            autocorr_5 = self._calculate_autocorrelation(prices, lag=5)

            # Create partial features (will be completed with candle features)
            tick_features = {
                'bid_ask_spread': latest_tick.spread,
                'spread_ratio': spread_ratio,
                'tick_direction': tick_direction,
                'tick_intensity': tick_intensity,
                'price_change_1': price_changes_1[0] if len(price_changes_1) > 0 else 0,
                'price_change_5': np.mean(price_changes_5),
                'price_change_10': np.mean(price_changes_10),
                'volatility_5': volatility_5,
                'volatility_10': volatility_10,
                'hurst_exponent': hurst_exponent,
                'entropy': entropy,
                'autocorr_1': autocorr_1,
                'autocorr_5': autocorr_5
            }

            # Cache tick features
            await self.cache_manager.set(
                CacheNamespace.FEATURE_CACHE,
                f"{symbol}:tick_features",
                tick_features,
                ttl=60
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'calculate_tick_features', 'symbol': symbol}
            )

    async def _calculate_candle_features(self, symbol: str):
        """Calculate features from candle data"""
        try:
            candles = list(self.candle_buffers[symbol])
            if len(candles) < self.min_candles_for_features:
                return

            # Convert to pandas DataFrame for TA-Lib
            df = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'open': candle.open_price,
                'high': candle.high_price,
                'low': candle.low_price,
                'close': candle.close_price,
                'volume': candle.volume
            } for candle in candles])

            df = df.sort_values('timestamp')

            # Extract price arrays
            opens = df['open'].values.astype(float)
            highs = df['high'].values.astype(float)
            lows = df['low'].values.astype(float)
            closes = df['close'].values.astype(float)
            volumes = df['volume'].values.astype(float)

            # Calculate technical indicators
            latest_candle = candles[-1]
            features = {}

            try:
                # Moving averages
                features['sma_5'] = talib.SMA(closes, timeperiod=5)[-1] if len(closes) >= 5 else closes[-1]
                features['sma_10'] = talib.SMA(closes, timeperiod=10)[-1] if len(closes) >= 10 else closes[-1]
                features['sma_20'] = talib.SMA(closes, timeperiod=20)[-1] if len(closes) >= 20 else closes[-1]
                features['sma_50'] = talib.SMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else closes[-1]

                features['ema_5'] = talib.EMA(closes, timeperiod=5)[-1] if len(closes) >= 5 else closes[-1]
                features['ema_10'] = talib.EMA(closes, timeperiod=10)[-1] if len(closes) >= 10 else closes[-1]
                features['ema_20'] = talib.EMA(closes, timeperiod=20)[-1] if len(closes) >= 20 else closes[-1]
                features['ema_50'] = talib.EMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else closes[-1]

                # Momentum indicators
                features['rsi_14'] = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50.0
                features['rsi_21'] = talib.RSI(closes, timeperiod=21)[-1] if len(closes) >= 21 else 50.0

                macd, macd_signal, macd_hist = talib.MACD(closes)
                features['macd'] = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0.0
                features['macd_signal'] = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0.0
                features['macd_histogram'] = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0.0

                features['momentum_10'] = talib.MOM(closes, timeperiod=10)[-1] if len(closes) >= 10 else 0.0
                features['roc_10'] = talib.ROC(closes, timeperiod=10)[-1] if len(closes) >= 10 else 0.0

                # Volatility indicators
                features['atr_14'] = talib.ATR(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else 0.0

                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
                features['bollinger_upper'] = bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else closes[-1]
                features['bollinger_middle'] = bb_middle[-1] if len(bb_middle) > 0 and not np.isnan(bb_middle[-1]) else closes[-1]
                features['bollinger_lower'] = bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else closes[-1]

                features['bollinger_width'] = (features['bollinger_upper'] - features['bollinger_lower']) / features['bollinger_middle']
                features['bollinger_position'] = (closes[-1] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])

                # Stochastic indicators
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
                features['stoch_k'] = stoch_k[-1] if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else 50.0
                features['stoch_d'] = stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else 50.0

                features['williams_r'] = talib.WILLR(highs, lows, closes)[-1] if len(closes) >= 14 else -50.0

                # Volume indicators (if volume data is available)
                if volumes.sum() > 0:
                    features['volume_sma_10'] = talib.SMA(volumes, timeperiod=10)[-1] if len(volumes) >= 10 else volumes[-1]
                    features['volume_ratio'] = volumes[-1] / features['volume_sma_10'] if features['volume_sma_10'] > 0 else 1.0
                    features['vwap'] = np.average(closes[-10:], weights=volumes[-10:]) if len(volumes) >= 10 else closes[-1]
                else:
                    features['volume_sma_10'] = 0.0
                    features['volume_ratio'] = 1.0
                    features['vwap'] = closes[-1]

            except Exception as ta_error:
                logging_system.log_error(
                    LogComponent.AI_ENGINE,
                    ta_error,
                    {'action': 'calculate_talib_features', 'symbol': symbol}
                )
                # Set default values if TA-Lib fails
                features.update({
                    'sma_5': closes[-1], 'sma_10': closes[-1], 'sma_20': closes[-1], 'sma_50': closes[-1],
                    'ema_5': closes[-1], 'ema_10': closes[-1], 'ema_20': closes[-1], 'ema_50': closes[-1],
                    'rsi_14': 50.0, 'rsi_21': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                    'momentum_10': 0.0, 'roc_10': 0.0, 'atr_14': 0.0,
                    'bollinger_upper': closes[-1], 'bollinger_middle': closes[-1], 'bollinger_lower': closes[-1],
                    'bollinger_width': 0.0, 'bollinger_position': 0.5,
                    'stoch_k': 50.0, 'stoch_d': 50.0, 'williams_r': -50.0,
                    'volume_sma_10': 0.0, 'volume_ratio': 1.0, 'vwap': closes[-1]
                })

            # Get tick features from cache
            tick_features = await self.cache_manager.get(
                CacheNamespace.FEATURE_CACHE,
                f"{symbol}:tick_features"
            ) or {}

            # Combine all features
            features.update(tick_features)

            # Add fractal dimension
            features['fractal_dimension'] = self._calculate_fractal_dimension(closes[-50:] if len(closes) >= 50 else closes)

            # Create ProcessedFeatures object
            processed_features = ProcessedFeatures(
                symbol=symbol,
                timestamp=latest_candle.timestamp,
                price=latest_candle.close_price,
                **{k: float(v) if not np.isnan(float(v)) else 0.0 for k, v in features.items()}
            )

            # Store features
            await self._store_features(processed_features)

            # Add to history
            self.feature_history[symbol].append(processed_features)

            # Notify callbacks
            for callback in self.feature_callbacks:
                try:
                    await callback(processed_features)
                except Exception as callback_error:
                    logging_system.log_error(
                        LogComponent.AI_ENGINE,
                        callback_error,
                        {'callback': str(callback), 'symbol': symbol}
                    )

            logging_system.log(
                LogComponent.AI_ENGINE,
                LogLevel.DEBUG,
                f"Features calculated for {symbol}",
                {'feature_count': len(asdict(processed_features)), 'timestamp': processed_features.timestamp.isoformat()}
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'calculate_candle_features', 'symbol': symbol}
            )

    async def _store_features(self, features: ProcessedFeatures):
        """Store processed features in InfluxDB"""
        try:
            feature_dict = asdict(features)
            # Remove non-numeric fields
            feature_dict.pop('symbol')
            feature_dict.pop('timestamp')

            await self.influxdb_manager.write_feature_data(
                symbol=features.symbol,
                features=feature_dict,
                timestamp=features.timestamp
            )

            # Also store in Redis cache
            await self.cache_manager.cache_features(
                features.symbol,
                features.timestamp.isoformat(),
                feature_dict
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'store_features', 'symbol': features.symbol}
            )

    def _calculate_hurst_exponent(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent for trend analysis"""
        try:
            if len(prices) < max_lag * 2:
                return 0.5  # Random walk default

            lags = range(2, min(max_lag, len(prices) // 2))
            tau = [np.std(np.diff(prices, n=lag)) for lag in lags]

            # Linear regression in log space
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = reg[0]

            return max(0.0, min(1.0, hurst))  # Clamp between 0 and 1

        except Exception:
            return 0.5

    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension"""
        try:
            if len(prices) < 10:
                return 1.5  # Default value

            # Higuchi's method
            k_max = min(10, len(prices) // 4)
            lk = []

            for k in range(1, k_max + 1):
                lm = []
                for m in range(k):
                    n_max = int((len(prices) - m - 1) / k)
                    if n_max > 0:
                        lmk = sum(abs(prices[m + i*k] - prices[m + (i-1)*k])
                                for i in range(1, n_max + 1))
                        lmk = lmk * (len(prices) - 1) / (n_max * k)
                        lm.append(lmk)

                if lm:
                    lk.append(np.mean(lm))

            if len(lk) > 1:
                # Linear regression to find fractal dimension
                ln_k = np.log(range(1, len(lk) + 1))
                ln_lk = np.log(lk)
                slope, _ = np.polyfit(ln_k, ln_lk, 1)
                fractal_dim = 2 - slope
                return max(1.0, min(2.0, fractal_dim))  # Clamp between 1 and 2

            return 1.5

        except Exception:
            return 1.5

    def _calculate_entropy(self, prices: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy of price distribution"""
        try:
            if len(prices) < 2:
                return 0.0

            # Create histogram
            hist, _ = np.histogram(prices, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero bins

            if len(hist) == 0:
                return 0.0

            # Calculate Shannon entropy
            entropy = -np.sum(hist * np.log2(hist))
            return max(0.0, entropy)

        except Exception:
            return 0.0

    def _calculate_autocorrelation(self, prices: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        try:
            if len(prices) < lag + 1:
                return 0.0

            returns = np.diff(prices)
            if len(returns) < lag + 1:
                return 0.0

            corr_matrix = np.corrcoef(returns[:-lag], returns[lag:])
            if corr_matrix.shape == (2, 2):
                return corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

            return 0.0

        except Exception:
            return 0.0

    def add_feature_callback(self, callback: Callable[[ProcessedFeatures], None]):
        """Add callback for processed features"""
        self.feature_callbacks.append(callback)

    def remove_feature_callback(self, callback: Callable[[ProcessedFeatures], None]):
        """Remove feature callback"""
        if callback in self.feature_callbacks:
            self.feature_callbacks.remove(callback)

    async def get_latest_features(self, symbol: str) -> Optional[ProcessedFeatures]:
        """Get latest processed features for symbol"""
        try:
            if symbol in self.feature_history and self.feature_history[symbol]:
                return self.feature_history[symbol][-1]

            # Try to get from cache
            cached_features = await self.cache_manager.get_cached_features(
                symbol,
                datetime.utcnow().isoformat()
            )

            if cached_features:
                return ProcessedFeatures(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    **cached_features
                )

            return None

        except Exception as e:
            logging_system.log_error(
                LogComponent.AI_ENGINE,
                e,
                {'action': 'get_latest_features', 'symbol': symbol}
            )
            return None

    def get_processing_status(self) -> Dict[str, Any]:
        """Get feature processing status"""
        return {
            'processing_active': self.processing_active,
            'symbols_subscribed': list(self.symbols_subscribed),
            'buffer_sizes': {
                symbol: {
                    'ticks': len(self.tick_buffers.get(symbol, [])),
                    'candles': len(self.candle_buffers.get(symbol, [])),
                    'features': len(self.feature_history.get(symbol, []))
                }
                for symbol in self.symbols_subscribed
            },
            'callbacks_registered': len(self.feature_callbacks)
        }

# Global feature processor instance
feature_processor = RealTimeFeatureProcessor()

async def get_feature_processor() -> RealTimeFeatureProcessor:
    """Get initialized feature processor"""
    if not feature_processor.websocket_client:
        await feature_processor.initialize()
    return feature_processor
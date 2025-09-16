"""
📊 REAL TICK DATA PROCESSOR
Real-time processing of live market tick data from Deriv API
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from enum import Enum
import threading
import queue
import statistics
from concurrent.futures import ThreadPoolExecutor
import ta  # Technical Analysis library

from real_deriv_client import RealTickData, RealDerivWebSocketClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedTickData:
    """Dados de tick processados com features calculadas"""
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[float] = None

    # Basic price features
    price_change: float = 0.0
    price_change_pct: float = 0.0
    price_velocity: float = 0.0
    price_acceleration: float = 0.0

    # Volatility features
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    volatility_15m: float = 0.0

    # Technical indicators
    sma_5: float = 0.0
    sma_20: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_position: float = 0.0

    # Market microstructure
    spread: Optional[float] = None
    tick_direction: int = 0  # 1=up, -1=down, 0=unchanged
    momentum_score: float = 0.0

    # Derived features
    price_position_sma: float = 0.0  # Position relative to SMA
    volatility_rank: float = 0.0     # Volatility percentile
    trend_strength: float = 0.0       # Trend strength indicator


class TickProcessor:
    """Processador de ticks em tempo real"""

    def __init__(self, buffer_size: int = 5000):
        self.buffer_size = buffer_size

        # Data buffers por símbolo
        self.tick_buffers: Dict[str, Deque[RealTickData]] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.processed_buffers: Dict[str, Deque[ProcessedTickData]] = defaultdict(lambda: deque(maxlen=buffer_size))

        # Feature calculation state
        self.ema_12_state: Dict[str, float] = {}
        self.ema_26_state: Dict[str, float] = {}
        self.rsi_state: Dict[str, Dict] = {}

        # Processing statistics
        self.processing_stats = {
            'ticks_processed': 0,
            'processing_time_ms': deque(maxlen=1000),
            'errors': 0,
            'last_processed': None
        }

        # Thread pool for CPU-intensive calculations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Callbacks
        self.processed_tick_callbacks: List[Callable] = []

    async def process_real_tick(self, tick_data: RealTickData) -> ProcessedTickData:
        """Processar tick real recebido"""
        start_time = time.perf_counter()

        try:
            # Adicionar ao buffer
            self.tick_buffers[tick_data.symbol].append(tick_data)

            # Processar features
            processed_tick = await self._calculate_features(tick_data)

            # Adicionar ao buffer processado
            self.processed_buffers[tick_data.symbol].append(processed_tick)

            # Atualizar estatísticas
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_stats['processing_time_ms'].append(processing_time)
            self.processing_stats['ticks_processed'] += 1
            self.processing_stats['last_processed'] = datetime.now()

            # Notificar callbacks
            for callback in self.processed_tick_callbacks:
                try:
                    await callback(processed_tick)
                except Exception as e:
                    logger.error(f"Error in processed tick callback: {e}")

            return processed_tick

        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.processing_stats['errors'] += 1
            raise

    async def _calculate_features(self, tick_data: RealTickData) -> ProcessedTickData:
        """Calcular features técnicas para o tick"""
        symbol = tick_data.symbol
        current_price = tick_data.tick

        # Obter histórico de preços
        price_history = [t.tick for t in self.tick_buffers[symbol]]

        if len(price_history) < 2:
            # Não há histórico suficiente
            return ProcessedTickData(
                symbol=symbol,
                timestamp=tick_data.timestamp,
                price=current_price,
                spread=tick_data.spread
            )

        # Executar cálculos em thread pool para não bloquear
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self.thread_pool,
            self._calculate_features_sync,
            symbol,
            current_price,
            price_history,
            tick_data
        )

        return features

    def _calculate_features_sync(self, symbol: str, current_price: float,
                                price_history: List[float], tick_data: RealTickData) -> ProcessedTickData:
        """Calcular features de forma síncrona (CPU-intensivo)"""

        # Convert to numpy for faster calculations
        prices = np.array(price_history)

        # Basic price features
        price_change = current_price - price_history[-2] if len(price_history) >= 2 else 0.0
        price_change_pct = (price_change / price_history[-2] * 100) if price_history[-2] != 0 else 0.0

        # Price velocity (rate of change)
        price_velocity = self._calculate_price_velocity(prices)

        # Price acceleration (change in velocity)
        price_acceleration = self._calculate_price_acceleration(prices)

        # Volatility calculations
        volatility_1m = self._calculate_volatility(prices, window=60)   # 1 minute
        volatility_5m = self._calculate_volatility(prices, window=300)  # 5 minutes
        volatility_15m = self._calculate_volatility(prices, window=900) # 15 minutes

        # Technical indicators
        sma_5 = self._calculate_sma(prices, 5)
        sma_20 = self._calculate_sma(prices, 20)

        # EMA with state tracking
        ema_12 = self._calculate_ema_with_state(symbol, current_price, 12, '12')
        ema_26 = self._calculate_ema_with_state(symbol, current_price, 26, '26')

        # RSI calculation
        rsi = self._calculate_rsi(symbol, prices)

        # MACD
        macd = ema_12 - ema_26
        macd_signal = self._calculate_ema_signal(symbol, macd)

        # Bollinger Bands
        bollinger_upper, bollinger_lower, bollinger_position = self._calculate_bollinger_bands(prices, current_price)

        # Tick direction
        tick_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0

        # Momentum score
        momentum_score = self._calculate_momentum_score(prices)

        # Derived features
        price_position_sma = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0.0
        volatility_rank = self._calculate_volatility_rank(symbol, volatility_1m)
        trend_strength = self._calculate_trend_strength(prices)

        return ProcessedTickData(
            symbol=symbol,
            timestamp=tick_data.timestamp,
            price=current_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            price_velocity=price_velocity,
            price_acceleration=price_acceleration,
            volatility_1m=volatility_1m,
            volatility_5m=volatility_5m,
            volatility_15m=volatility_15m,
            sma_5=sma_5,
            sma_20=sma_20,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower,
            bollinger_position=bollinger_position,
            spread=tick_data.spread,
            tick_direction=tick_direction,
            momentum_score=momentum_score,
            price_position_sma=price_position_sma,
            volatility_rank=volatility_rank,
            trend_strength=trend_strength
        )

    def _calculate_price_velocity(self, prices: np.ndarray) -> float:
        """Calcular velocidade do preço (taxa de mudança)"""
        if len(prices) < 3:
            return 0.0

        # Usar regressão linear para calcular velocidade
        x = np.arange(len(prices))
        coeffs = np.polyfit(x[-10:], prices[-10:], 1)  # Últimos 10 pontos
        return coeffs[0]  # Slope

    def _calculate_price_acceleration(self, prices: np.ndarray) -> float:
        """Calcular aceleração do preço (mudança na velocidade)"""
        if len(prices) < 6:
            return 0.0

        # Calcular velocidades em janelas
        velocity1 = self._calculate_price_velocity(prices[-10:])
        velocity2 = self._calculate_price_velocity(prices[-15:-5])

        return velocity1 - velocity2

    def _calculate_volatility(self, prices: np.ndarray, window: int) -> float:
        """Calcular volatilidade em janela específica"""
        if len(prices) < window:
            window = len(prices)

        if window < 2:
            return 0.0

        recent_prices = prices[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) * 100  # Em percentual

    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calcular Simple Moving Average"""
        if len(prices) < period:
            period = len(prices)

        return np.mean(prices[-period:]) if period > 0 else 0.0

    def _calculate_ema_with_state(self, symbol: str, current_price: float, period: int, key: str) -> float:
        """Calcular EMA mantendo estado"""
        alpha = 2.0 / (period + 1)
        state_key = f"{symbol}_ema_{key}"

        if state_key not in self.ema_12_state and state_key not in self.ema_26_state:
            # Initialize with current price
            if key == '12':
                self.ema_12_state[state_key] = current_price
            else:
                self.ema_26_state[state_key] = current_price
            return current_price

        # Update EMA
        if key == '12':
            prev_ema = self.ema_12_state[state_key]
            new_ema = alpha * current_price + (1 - alpha) * prev_ema
            self.ema_12_state[state_key] = new_ema
        else:
            prev_ema = self.ema_26_state[state_key]
            new_ema = alpha * current_price + (1 - alpha) * prev_ema
            self.ema_26_state[state_key] = new_ema

        return new_ema

    def _calculate_rsi(self, symbol: str, prices: np.ndarray) -> float:
        """Calcular RSI com estado"""
        if len(prices) < 14:
            return 50.0

        # Calculate price changes
        changes = np.diff(prices[-15:])  # 14 periods + 1
        gains = changes.copy()
        losses = changes.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)

        # Average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema_signal(self, symbol: str, macd_value: float) -> float:
        """Calcular EMA signal line do MACD"""
        # Simplified signal line calculation
        signal_key = f"{symbol}_macd_signal"
        alpha = 2.0 / (9 + 1)  # 9-period EMA

        if signal_key not in self.ema_12_state:
            self.ema_12_state[signal_key] = macd_value
            return macd_value

        prev_signal = self.ema_12_state[signal_key]
        new_signal = alpha * macd_value + (1 - alpha) * prev_signal
        self.ema_12_state[signal_key] = new_signal

        return new_signal

    def _calculate_bollinger_bands(self, prices: np.ndarray, current_price: float) -> tuple:
        """Calcular Bollinger Bands"""
        if len(prices) < 20:
            return current_price, current_price, 0.5

        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])

        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)

        # Position within bands (0 = lower band, 1 = upper band)
        if upper_band > lower_band:
            position = (current_price - lower_band) / (upper_band - lower_band)
            position = max(0, min(1, position))  # Clamp between 0 and 1
        else:
            position = 0.5

        return upper_band, lower_band, position

    def _calculate_momentum_score(self, prices: np.ndarray) -> float:
        """Calcular score de momentum"""
        if len(prices) < 10:
            return 0.0

        # ROC (Rate of Change) over different periods
        roc_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        roc_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0

        # Weighted momentum score
        momentum = (roc_5 * 0.7) + (roc_10 * 0.3)

        # Normalize to [-1, 1] range
        return np.tanh(momentum * 100)

    def _calculate_volatility_rank(self, symbol: str, current_volatility: float) -> float:
        """Calcular rank de volatilidade (percentil)"""
        # Get historical volatility values
        volatility_history = []
        for processed_tick in self.processed_buffers[symbol]:
            if processed_tick.volatility_1m > 0:
                volatility_history.append(processed_tick.volatility_1m)

        if len(volatility_history) < 10:
            return 0.5  # Default to 50th percentile

        # Calculate percentile rank
        rank = sum(1 for v in volatility_history if v <= current_volatility)
        return rank / len(volatility_history)

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcular força da tendência"""
        if len(prices) < 20:
            return 0.0

        # Calculate directional movement
        ups = 0
        downs = 0

        for i in range(1, min(20, len(prices))):
            if prices[-i] > prices[-i-1]:
                ups += 1
            elif prices[-i] < prices[-i-1]:
                downs += 1

        total_moves = ups + downs
        if total_moves == 0:
            return 0.0

        # Trend strength based on directional bias
        strength = abs(ups - downs) / total_moves
        direction = 1 if ups > downs else -1

        return strength * direction

    def get_latest_processed_tick(self, symbol: str) -> Optional[ProcessedTickData]:
        """Obter último tick processado de um símbolo"""
        if symbol in self.processed_buffers and self.processed_buffers[symbol]:
            return self.processed_buffers[symbol][-1]
        return None

    def get_processed_history(self, symbol: str, count: int = 100) -> List[ProcessedTickData]:
        """Obter histórico de ticks processados"""
        if symbol in self.processed_buffers:
            return list(self.processed_buffers[symbol])[-count:]
        return []

    def get_feature_dataframe(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Obter features como DataFrame para análise"""
        processed_ticks = self.get_processed_history(symbol, count)

        if not processed_ticks:
            return pd.DataFrame()

        # Convert to dict list
        data = []
        for tick in processed_ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'price': tick.price,
                'price_change': tick.price_change,
                'price_change_pct': tick.price_change_pct,
                'price_velocity': tick.price_velocity,
                'price_acceleration': tick.price_acceleration,
                'volatility_1m': tick.volatility_1m,
                'volatility_5m': tick.volatility_5m,
                'volatility_15m': tick.volatility_15m,
                'sma_5': tick.sma_5,
                'sma_20': tick.sma_20,
                'ema_12': tick.ema_12,
                'ema_26': tick.ema_26,
                'rsi': tick.rsi,
                'macd': tick.macd,
                'macd_signal': tick.macd_signal,
                'bollinger_upper': tick.bollinger_upper,
                'bollinger_lower': tick.bollinger_lower,
                'bollinger_position': tick.bollinger_position,
                'spread': tick.spread,
                'tick_direction': tick.tick_direction,
                'momentum_score': tick.momentum_score,
                'price_position_sma': tick.price_position_sma,
                'volatility_rank': tick.volatility_rank,
                'trend_strength': tick.trend_strength
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        return df

    def get_processing_stats(self) -> Dict:
        """Obter estatísticas de processamento"""
        if self.processing_stats['processing_time_ms']:
            avg_processing_time = statistics.mean(self.processing_stats['processing_time_ms'])
            max_processing_time = max(self.processing_stats['processing_time_ms'])
            min_processing_time = min(self.processing_stats['processing_time_ms'])
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0

        return {
            'ticks_processed': self.processing_stats['ticks_processed'],
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'min_processing_time_ms': min_processing_time,
            'errors': self.processing_stats['errors'],
            'last_processed': self.processing_stats['last_processed'],
            'symbols_active': len(self.tick_buffers),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.tick_buffers.items()}
        }

    def add_processed_tick_callback(self, callback: Callable[[ProcessedTickData], None]):
        """Adicionar callback para ticks processados"""
        self.processed_tick_callbacks.append(callback)

    def clear_buffers(self, symbol: str = None):
        """Limpar buffers de dados"""
        if symbol:
            if symbol in self.tick_buffers:
                self.tick_buffers[symbol].clear()
            if symbol in self.processed_buffers:
                self.processed_buffers[symbol].clear()
        else:
            self.tick_buffers.clear()
            self.processed_buffers.clear()
            self.ema_12_state.clear()
            self.ema_26_state.clear()
            self.rsi_state.clear()


# 🧪 Função de teste
async def test_tick_processor():
    """Testar processador de ticks"""
    processor = TickProcessor()

    # Callback para ticks processados
    async def on_processed_tick(processed_tick: ProcessedTickData):
        print(f"📊 Processed tick: {processed_tick.symbol} = {processed_tick.price:.5f}")
        print(f"   RSI: {processed_tick.rsi:.2f}, MACD: {processed_tick.macd:.5f}")
        print(f"   Volatility: {processed_tick.volatility_1m:.4f}, Momentum: {processed_tick.momentum_score:.4f}")

    processor.add_processed_tick_callback(on_processed_tick)

    # Simular ticks
    symbol = "R_100"
    base_price = 245.67

    for i in range(100):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01)
        new_price = base_price + price_change
        base_price = new_price

        # Create tick data
        tick_data = RealTickData(
            symbol=symbol,
            tick=new_price,
            epoch=int(time.time()) + i,
            quote=new_price,
            pip_size=0.01,
            timestamp=datetime.now(),
            spread=0.001
        )

        # Process tick
        processed = await processor.process_real_tick(tick_data)

        # Small delay
        await asyncio.sleep(0.1)

    # Get stats
    stats = processor.get_processing_stats()
    print(f"\n📈 Processing Stats:")
    print(f"   Ticks processed: {stats['ticks_processed']}")
    print(f"   Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")
    print(f"   Errors: {stats['errors']}")

    # Get DataFrame
    df = processor.get_feature_dataframe(symbol)
    print(f"\n📊 Feature DataFrame shape: {df.shape}")
    print(df.tail())


if __name__ == "__main__":
    print("📊 TESTING REAL TICK PROCESSOR")
    print("=" * 40)

    asyncio.run(test_tick_processor())
"""
Advanced Feature Engineering - Sistema Avançado de Engenharia de Features
Sistema completo para geração de features sofisticadas incluindo microestrutura de mercado e dados alternativos.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import talib
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

from real_tick_processor import ProcessedTickData

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureCategory(Enum):
    TECHNICAL = "technical"
    MICROSTRUCTURE = "microstructure"
    STATISTICAL = "statistical"
    SENTIMENT = "sentiment"
    VOLUME_PROFILE = "volume_profile"
    PATTERN = "pattern"
    CROSS_ASSET = "cross_asset"
    ALTERNATIVE = "alternative"

class FeatureImportanceMethod(Enum):
    MUTUAL_INFO = "mutual_info"
    F_REGRESSION = "f_regression"
    CORRELATION = "correlation"
    PERMUTATION = "permutation"

@dataclass
class FeatureMetadata:
    """Metadados de uma feature"""
    name: str
    category: FeatureCategory
    description: str
    calculation_method: str
    lookback_period: int
    stability_score: float
    importance_score: float
    last_updated: datetime

@dataclass
class MarketMicrostructureData:
    """Dados de microestrutura de mercado"""
    bid_ask_spread: float
    market_impact: float
    order_flow_imbalance: float
    volume_weighted_price: float
    realized_volatility: float
    tick_direction: int
    trade_intensity: float
    price_improvement: float

@dataclass
class VolumeProfileData:
    """Dados de perfil de volume"""
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_at_price: Dict[float, float]
    volume_delta: float
    cumulative_volume_delta: float

@dataclass
class PatternFeatures:
    """Features de padrões técnicos"""
    trend_strength: float
    support_resistance_strength: float
    breakout_probability: float
    reversal_signals: int
    continuation_signals: int
    chart_pattern_score: float

class AdvancedFeatureEngine:
    """Motor avançado de engenharia de features"""

    def __init__(self, symbols: List[str], lookback_periods: List[int] = None):
        self.symbols = symbols
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]

        # Buffers de dados por símbolo
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.tick_buffers: Dict[str, deque] = {}

        # Features calculadas
        self.feature_cache: Dict[str, Dict[str, float]] = {}
        self.feature_metadata: Dict[str, FeatureMetadata] = {}

        # Configurações
        self.max_buffer_size = 1000
        self.feature_update_frequency = 1  # segundos
        self.importance_update_frequency = 300  # 5 minutos

        # Scalers e transformadores
        self.scalers: Dict[str, Any] = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.pca = PCA(n_components=0.95)  # 95% da variância
        self.ica = FastICA(n_components=20, random_state=42)

        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=50)

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.calculation_lock = threading.Lock()

        # Estado
        self.is_initialized = False
        self.last_feature_update = {}

        self._initialize_buffers()
        logger.info("AdvancedFeatureEngine inicializado")

    def _initialize_buffers(self):
        """Inicializa buffers de dados"""
        for symbol in self.symbols:
            self.price_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.volume_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.tick_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.feature_cache[symbol] = {}
            self.last_feature_update[symbol] = datetime.now()

    async def add_tick_data(self, symbol: str, tick_data: ProcessedTickData):
        """Adiciona dados de tick para um símbolo"""
        if symbol not in self.symbols:
            return

        # Adicionar aos buffers
        self.price_buffers[symbol].append(tick_data.price)
        self.volume_buffers[symbol].append(getattr(tick_data, 'volume', 1.0))
        self.tick_buffers[symbol].append(tick_data)

        # Verificar se deve atualizar features
        time_since_update = (datetime.now() - self.last_feature_update[symbol]).total_seconds()
        if time_since_update >= self.feature_update_frequency:
            await self.update_features(symbol)

    async def update_features(self, symbol: str):
        """Atualiza todas as features para um símbolo"""
        if len(self.price_buffers[symbol]) < min(self.lookback_periods):
            return

        async with asyncio.Lock():
            try:
                # Executar cálculos em thread pool
                features = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self._calculate_all_features_sync,
                    symbol
                )

                # Atualizar cache
                self.feature_cache[symbol].update(features)
                self.last_feature_update[symbol] = datetime.now()

            except Exception as e:
                logger.error(f"Erro ao atualizar features para {symbol}: {e}")

    def _calculate_all_features_sync(self, symbol: str) -> Dict[str, float]:
        """Calcula todas as features sincronamente"""
        features = {}

        prices = np.array(list(self.price_buffers[symbol]))
        volumes = np.array(list(self.volume_buffers[symbol]))
        ticks = list(self.tick_buffers[symbol])

        # Features técnicas
        features.update(self._calculate_technical_features(prices, volumes))

        # Features estatísticas
        features.update(self._calculate_statistical_features(prices))

        # Features de microestrutura
        features.update(self._calculate_microstructure_features(ticks))

        # Features de volume profile
        features.update(self._calculate_volume_profile_features(prices, volumes))

        # Features de padrões
        features.update(self._calculate_pattern_features(prices))

        # Features de sentimento (simplificado)
        features.update(self._calculate_sentiment_features(prices))

        return features

    def _calculate_technical_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calcula features técnicas"""
        features = {}

        try:
            # Indicadores básicos para cada período
            for period in self.lookback_periods:
                if len(prices) < period:
                    continue

                recent_prices = prices[-period:]
                recent_volumes = volumes[-period:]

                # Moving averages
                sma = np.mean(recent_prices)
                ema = talib.EMA(recent_prices, timeperiod=min(period, len(recent_prices)))[-1]

                features[f'sma_{period}'] = sma
                features[f'ema_{period}'] = ema if not np.isnan(ema) else sma
                features[f'price_to_sma_{period}'] = prices[-1] / sma if sma > 0 else 1.0

                # Bollinger Bands
                if len(recent_prices) >= 20:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(recent_prices)
                    features[f'bb_position_{period}'] = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
                    features[f'bb_width_{period}'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] > 0 else 0

                # RSI
                if len(recent_prices) >= 14:
                    rsi = talib.RSI(recent_prices)
                    features[f'rsi_{period}'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0

                # MACD
                if len(recent_prices) >= 26:
                    macd, macd_signal, macd_hist = talib.MACD(recent_prices)
                    features[f'macd_{period}'] = macd[-1] if not np.isnan(macd[-1]) else 0.0
                    features[f'macd_signal_{period}'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
                    features[f'macd_histogram_{period}'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0

                # Stochastic
                if len(prices) >= period:
                    high_prices = np.maximum.accumulate(recent_prices)
                    low_prices = np.minimum.accumulate(recent_prices)
                    stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, recent_prices)
                    features[f'stoch_k_{period}'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50.0
                    features[f'stoch_d_{period}'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50.0

                # Volume indicators
                features[f'volume_sma_{period}'] = np.mean(recent_volumes)
                features[f'volume_ratio_{period}'] = recent_volumes[-1] / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1.0

                # Volatility
                returns = np.diff(recent_prices) / recent_prices[:-1]
                features[f'volatility_{period}'] = np.std(returns) * np.sqrt(252)  # Anualizada
                features[f'realized_vol_{period}'] = np.sqrt(np.sum(returns**2))

            # Indicators avançados
            if len(prices) >= 50:
                # ADX (Average Directional Index)
                high_prices = np.maximum.accumulate(prices[-50:])
                low_prices = np.minimum.accumulate(prices[-50:])
                adx = talib.ADX(high_prices, low_prices, prices[-50:])
                features['adx'] = adx[-1] if not np.isnan(adx[-1]) else 25.0

                # Williams %R
                willr = talib.WILLR(high_prices, low_prices, prices[-50:])
                features['williams_r'] = willr[-1] if not np.isnan(willr[-1]) else -50.0

                # CCI (Commodity Channel Index)
                cci = talib.CCI(high_prices, low_prices, prices[-50:])
                features['cci'] = cci[-1] if not np.isnan(cci[-1]) else 0.0

        except Exception as e:
            logger.error(f"Erro no cálculo de features técnicas: {e}")

        return features

    def _calculate_statistical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calcula features estatísticas avançadas"""
        features = {}

        try:
            returns = np.diff(prices) / prices[:-1]

            # Estatísticas básicas
            features['returns_mean'] = np.mean(returns)
            features['returns_std'] = np.std(returns)
            features['returns_skew'] = stats.skew(returns)
            features['returns_kurtosis'] = stats.kurtosis(returns)

            # Estatísticas de ordem superior
            features['returns_var'] = np.var(returns)
            features['returns_min'] = np.min(returns)
            features['returns_max'] = np.max(returns)
            features['returns_range'] = np.max(returns) - np.min(returns)

            # Percentis
            for percentile in [5, 25, 75, 95]:
                features[f'returns_p{percentile}'] = np.percentile(returns, percentile)

            # Autocorrelação
            for lag in [1, 5, 10]:
                if len(returns) > lag:
                    autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    features[f'autocorr_lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0

            # Testes estatísticos
            # Jarque-Bera test for normality
            if len(returns) > 8:
                jb_stat, jb_pvalue = stats.jarque_bera(returns)
                features['jarque_bera_stat'] = jb_stat
                features['jarque_bera_pvalue'] = jb_pvalue

            # Ljung-Box test for autocorrelation
            if len(returns) > 10:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat = acorr_ljungbox(returns, lags=min(10, len(returns)//4), return_df=True)
                features['ljung_box_stat'] = lb_stat['lb_stat'].iloc[-1]
                features['ljung_box_pvalue'] = lb_stat['lb_pvalue'].iloc[-1]

            # Hurst exponent (simplified)
            if len(prices) >= 50:
                features['hurst_exponent'] = self._calculate_hurst_exponent(prices)

            # Fractal dimension
            features['fractal_dimension'] = self._calculate_fractal_dimension(prices)

        except Exception as e:
            logger.error(f"Erro no cálculo de features estatísticas: {e}")

        return features

    def _calculate_microstructure_features(self, ticks: List[ProcessedTickData]) -> Dict[str, float]:
        """Calcula features de microestrutura de mercado"""
        features = {}

        try:
            if len(ticks) < 2:
                return features

            # Tick direction (Lee-Ready algorithm simplificado)
            tick_directions = []
            for i in range(1, len(ticks)):
                if ticks[i].price > ticks[i-1].price:
                    tick_directions.append(1)
                elif ticks[i].price < ticks[i-1].price:
                    tick_directions.append(-1)
                else:
                    tick_directions.append(0)

            if tick_directions:
                features['tick_direction_mean'] = np.mean(tick_directions)
                features['tick_direction_imbalance'] = np.sum(tick_directions) / len(tick_directions)

            # Trade intensity
            time_diffs = []
            for i in range(1, len(ticks)):
                time_diff = (ticks[i].timestamp - ticks[i-1].timestamp).total_seconds()
                time_diffs.append(time_diff)

            if time_diffs:
                features['trade_intensity'] = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0.0
                features['trade_intensity_std'] = np.std(time_diffs)

            # Price jumps
            price_changes = []
            for i in range(1, len(ticks)):
                change = abs(ticks[i].price - ticks[i-1].price) / ticks[i-1].price
                price_changes.append(change)

            if price_changes:
                features['avg_price_jump'] = np.mean(price_changes)
                features['max_price_jump'] = np.max(price_changes)
                features['price_jump_frequency'] = np.sum(np.array(price_changes) > np.std(price_changes)) / len(price_changes)

            # Volume-weighted features
            prices = [tick.price for tick in ticks]
            volumes = [getattr(tick, 'volume', 1.0) for tick in ticks]

            if len(prices) == len(volumes):
                vwap = np.average(prices, weights=volumes)
                features['vwap'] = vwap
                features['price_to_vwap'] = prices[-1] / vwap if vwap > 0 else 1.0

                # Volume distribution
                features['volume_mean'] = np.mean(volumes)
                features['volume_std'] = np.std(volumes)
                features['volume_imbalance'] = (volumes[-1] - np.mean(volumes)) / np.std(volumes) if np.std(volumes) > 0 else 0.0

        except Exception as e:
            logger.error(f"Erro no cálculo de features de microestrutura: {e}")

        return features

    def _calculate_volume_profile_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calcula features de perfil de volume"""
        features = {}

        try:
            if len(prices) != len(volumes) or len(prices) < 10:
                return features

            # Discretizar preços em bins
            num_bins = min(20, len(prices) // 5)
            price_bins = np.linspace(np.min(prices), np.max(prices), num_bins)

            # Volume por faixa de preço
            volume_at_price = np.zeros(num_bins - 1)
            for i in range(len(prices)):
                bin_idx = np.digitize(prices[i], price_bins) - 1
                if 0 <= bin_idx < len(volume_at_price):
                    volume_at_price[bin_idx] += volumes[i]

            # Point of Control (POC) - preço com maior volume
            poc_idx = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            features['poc_price'] = poc_price
            features['price_to_poc'] = prices[-1] / poc_price if poc_price > 0 else 1.0

            # Value Area (70% do volume)
            total_volume = np.sum(volume_at_price)
            cumulative_volume = np.cumsum(volume_at_price)

            value_area_start_idx = np.where(cumulative_volume >= total_volume * 0.15)[0]
            value_area_end_idx = np.where(cumulative_volume >= total_volume * 0.85)[0]

            if len(value_area_start_idx) > 0 and len(value_area_end_idx) > 0:
                value_area_low = price_bins[value_area_start_idx[0]]
                value_area_high = price_bins[value_area_end_idx[0] + 1]

                features['value_area_low'] = value_area_low
                features['value_area_high'] = value_area_high
                features['value_area_width'] = value_area_high - value_area_low

                # Posição do preço atual na value area
                if value_area_high != value_area_low:
                    features['price_position_in_va'] = (prices[-1] - value_area_low) / (value_area_high - value_area_low)
                else:
                    features['price_position_in_va'] = 0.5

            # Volume delta (diferença entre compras e vendas estimada)
            buy_volume = 0
            sell_volume = 0

            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    buy_volume += volumes[i]
                elif prices[i] < prices[i-1]:
                    sell_volume += volumes[i]

            total_directional_volume = buy_volume + sell_volume
            if total_directional_volume > 0:
                features['volume_delta'] = (buy_volume - sell_volume) / total_directional_volume
            else:
                features['volume_delta'] = 0.0

            features['buy_volume_ratio'] = buy_volume / np.sum(volumes) if np.sum(volumes) > 0 else 0.5

        except Exception as e:
            logger.error(f"Erro no cálculo de features de volume profile: {e}")

        return features

    def _calculate_pattern_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calcula features de padrões técnicos"""
        features = {}

        try:
            if len(prices) < 20:
                return features

            # Trend strength
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)

            features['trend_slope'] = slope
            features['trend_strength'] = abs(r_value)
            features['trend_direction'] = 1 if slope > 0 else -1
            features['trend_significance'] = 1 - p_value if p_value < 1 else 0

            # Support and resistance levels
            peaks, _ = find_peaks(prices, distance=5)
            troughs, _ = find_peaks(-prices, distance=5)

            if len(peaks) > 0:
                resistance_level = np.mean(prices[peaks])
                features['resistance_level'] = resistance_level
                features['distance_to_resistance'] = (resistance_level - prices[-1]) / prices[-1]

            if len(troughs) > 0:
                support_level = np.mean(prices[troughs])
                features['support_level'] = support_level
                features['distance_to_support'] = (prices[-1] - support_level) / prices[-1]

            # Breakout detection
            recent_high = np.max(prices[-20:])
            recent_low = np.min(prices[-20:])
            historical_high = np.max(prices[:-20]) if len(prices) > 20 else recent_high
            historical_low = np.min(prices[:-20]) if len(prices) > 20 else recent_low

            features['breakout_upward'] = 1 if prices[-1] > historical_high else 0
            features['breakout_downward'] = 1 if prices[-1] < historical_low else 0
            features['range_position'] = (prices[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

            # Volatility clustering
            returns = np.diff(prices) / prices[:-1]
            vol_window = 10
            if len(returns) >= vol_window * 2:
                recent_vol = np.std(returns[-vol_window:])
                historical_vol = np.std(returns[:-vol_window])
                features['volatility_regime'] = recent_vol / historical_vol if historical_vol > 0 else 1.0

            # Price momentum
            for period in [5, 10, 20]:
                if len(prices) > period:
                    momentum = (prices[-1] - prices[-period]) / prices[-period]
                    features[f'momentum_{period}'] = momentum

            # Mean reversion signals
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            features['z_score'] = (prices[-1] - mean_price) / std_price if std_price > 0 else 0.0
            features['mean_reversion_signal'] = abs(features['z_score']) > 2

        except Exception as e:
            logger.error(f"Erro no cálculo de features de padrões: {e}")

        return features

    def _calculate_sentiment_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calcula features de sentimento (simplificado)"""
        features = {}

        try:
            if len(prices) < 10:
                return features

            # Fear & Greed Index simplificado
            returns = np.diff(prices) / prices[:-1]

            # Volatilidade como proxy para medo
            volatility = np.std(returns)
            features['fear_index'] = min(volatility * 100, 100)  # 0-100 scale

            # Momentum como proxy para ganância
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['greed_index'] = max(min((momentum + 0.01) * 5000, 100), 0)  # 0-100 scale

            # Sentiment score combinado
            features['sentiment_score'] = (100 - features['fear_index'] + features['greed_index']) / 2

            # Market stress indicator
            price_changes = np.abs(np.diff(prices) / prices[:-1])
            stress_threshold = np.percentile(price_changes, 95)
            recent_stress = np.mean(price_changes[-5:] > stress_threshold) if len(price_changes) >= 5 else 0
            features['market_stress'] = recent_stress

        except Exception as e:
            logger.error(f"Erro no cálculo de features de sentimento: {e}")

        return features

    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calcula expoente de Hurst (simplificado)"""
        try:
            n = len(prices)
            if n < 20:
                return 0.5

            # Método R/S simplificado
            lags = [5, 10, 20, min(50, n//4)]
            rs_values = []

            for lag in lags:
                if lag >= n:
                    continue

                returns = np.diff(prices[-lag:]) / prices[-lag:-1]
                mean_return = np.mean(returns)

                # Desvios cumulativos
                cumulative_deviations = np.cumsum(returns - mean_return)
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                S = np.std(returns)

                if S > 0 and R > 0:
                    rs_values.append(np.log(R/S))

            if len(rs_values) >= 2:
                # Regressão linear para estimar H
                log_lags = [np.log(lag) for lag in lags[:len(rs_values)]]
                slope, _, _, _, _ = stats.linregress(log_lags, rs_values)
                return max(min(slope, 1.0), 0.0)  # Limitar entre 0 e 1

        except Exception:
            pass

        return 0.5  # Valor neutro

    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calcula dimensão fractal simplificada"""
        try:
            if len(prices) < 10:
                return 1.5

            # Método box-counting simplificado
            scales = [2, 4, 8, min(16, len(prices)//4)]
            counts = []

            for scale in scales:
                if scale >= len(prices):
                    continue

                # Dividir em boxes e contar ocupados
                n_boxes = len(prices) // scale
                occupied_boxes = 0

                for i in range(n_boxes):
                    box_start = i * scale
                    box_end = (i + 1) * scale
                    box_data = prices[box_start:box_end]

                    if len(box_data) > 1 and np.max(box_data) != np.min(box_data):
                        occupied_boxes += 1

                if occupied_boxes > 0:
                    counts.append(np.log(occupied_boxes))

            if len(counts) >= 2:
                log_scales = [np.log(1.0/scale) for scale in scales[:len(counts)]]
                slope, _, _, _, _ = stats.linregress(log_scales, counts)
                return max(min(abs(slope), 2.0), 1.0)  # Limitar entre 1 e 2

        except Exception:
            pass

        return 1.5  # Valor neutro

    async def get_feature_vector(self, symbol: str, include_metadata: bool = False) -> Optional[Dict[str, Any]]:
        """Obtém vetor de features para um símbolo"""
        if symbol not in self.feature_cache:
            return None

        features = self.feature_cache[symbol].copy()

        if include_metadata:
            return {
                "features": features,
                "metadata": {
                    "symbol": symbol,
                    "last_update": self.last_feature_update[symbol].isoformat(),
                    "feature_count": len(features),
                    "buffer_size": len(self.price_buffers[symbol])
                }
            }

        return features

    async def get_feature_importance(self, target_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calcula importância das features"""
        try:
            # Combinar features de todos os símbolos
            all_features = []
            all_targets = []

            for symbol in self.symbols:
                if symbol in target_data and symbol in self.feature_cache:
                    features = self.feature_cache[symbol]
                    targets = target_data[symbol]

                    if features and targets:
                        feature_vector = list(features.values())
                        all_features.append(feature_vector)
                        all_targets.extend(targets[-len(feature_vector):])

            if len(all_features) < 10:  # Mínimo de dados
                return {}

            # Preparar dados
            X = np.array(all_features)
            y = np.array(all_targets[:len(all_features)])

            # Calcular importância
            importance_scores = {}
            feature_names = list(self.feature_cache[self.symbols[0]].keys())

            # Mutual information
            if len(X) > 0 and len(y) > 0:
                mi_scores = mutual_info_regression(X, y)
                for i, name in enumerate(feature_names):
                    importance_scores[name] = float(mi_scores[i]) if i < len(mi_scores) else 0.0

            return importance_scores

        except Exception as e:
            logger.error(f"Erro no cálculo de importância: {e}")
            return {}

    async def apply_feature_selection(self, target_data: Dict[str, List[float]], n_features: int = 50):
        """Aplica seleção de features"""
        try:
            importance_scores = await self.get_feature_importance(target_data)

            if not importance_scores:
                return

            # Selecionar top N features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, score in sorted_features[:n_features]]

            # Filtrar cache de features
            for symbol in self.symbols:
                if symbol in self.feature_cache:
                    filtered_features = {
                        name: value for name, value in self.feature_cache[symbol].items()
                        if name in selected_features
                    }
                    self.feature_cache[symbol] = filtered_features

            logger.info(f"Selecionadas {len(selected_features)} features de {len(importance_scores)}")

        except Exception as e:
            logger.error(f"Erro na seleção de features: {e}")

    async def get_feature_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas das features"""
        stats = {
            "total_symbols": len(self.symbols),
            "features_per_symbol": {},
            "avg_features_per_symbol": 0,
            "feature_categories": {},
            "buffer_sizes": {},
            "last_updates": {}
        }

        total_features = 0
        for symbol in self.symbols:
            if symbol in self.feature_cache:
                n_features = len(self.feature_cache[symbol])
                stats["features_per_symbol"][symbol] = n_features
                total_features += n_features

                stats["buffer_sizes"][symbol] = len(self.price_buffers[symbol])
                stats["last_updates"][symbol] = self.last_feature_update[symbol].isoformat()

        if len(self.symbols) > 0:
            stats["avg_features_per_symbol"] = total_features / len(self.symbols)

        return stats

    async def export_features(self, filepath: str, include_raw_data: bool = False):
        """Exporta features para arquivo"""
        try:
            export_data = {
                "feature_cache": self.feature_cache,
                "feature_metadata": {name: asdict(metadata) for name, metadata in self.feature_metadata.items()},
                "statistics": await self.get_feature_statistics(),
                "export_timestamp": datetime.now().isoformat()
            }

            if include_raw_data:
                export_data["raw_data"] = {
                    "price_buffers": {symbol: list(buffer) for symbol, buffer in self.price_buffers.items()},
                    "volume_buffers": {symbol: list(buffer) for symbol, buffer in self.volume_buffers.items()}
                }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, default=str, indent=2)

            logger.info(f"Features exportadas para: {filepath}")

        except Exception as e:
            logger.error(f"Erro ao exportar features: {e}")

    async def shutdown(self):
        """Encerra o motor de features"""
        self.thread_pool.shutdown(wait=True)
        logger.info("AdvancedFeatureEngine encerrado")
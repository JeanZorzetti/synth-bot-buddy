"""
Multi-Asset Manager - Gerenciador de Múltiplos Ativos
Sistema completo para trading simultâneo em múltiplos ativos com correlação e análise cross-asset.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx

from real_tick_processor import ProcessedTickData
from real_trading_executor import RealTradeRequest, TradeType, ContractType
from advanced_model_ensemble import AdvancedModelEnsemble, EnsemblePrediction
from real_portfolio_optimizer import RealPortfolioOptimizer, OptimizationResult

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    INDICES = "indices"
    STOCKS = "stocks"
    BONDS = "bonds"

class CorrelationRegime(Enum):
    LOW = "low"           # < 0.3
    MODERATE = "moderate" # 0.3 - 0.6
    HIGH = "high"         # 0.6 - 0.8
    EXTREME = "extreme"   # > 0.8

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class AssetInfo:
    """Informações de um ativo"""
    symbol: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    tick_size: float
    min_trade_size: float
    trading_hours: Dict[str, str]
    volatility_regime: str
    liquidity_score: float
    is_active: bool = True

@dataclass
class CrossAssetSignal:
    """Sinal cross-asset"""
    source_symbol: str
    target_symbol: str
    signal_type: str
    strength: float
    confidence: float
    timeframe: str
    timestamp: datetime
    description: str

@dataclass
class CorrelationMatrix:
    """Matriz de correlação entre ativos"""
    symbols: List[str]
    correlation_matrix: np.ndarray
    correlation_regime: CorrelationRegime
    calculation_timestamp: datetime
    lookback_period: int
    stability_score: float

@dataclass
class AssetCluster:
    """Cluster de ativos correlacionados"""
    cluster_id: int
    symbols: List[str]
    cluster_center: np.ndarray
    avg_correlation: float
    volatility_level: str
    representative_symbol: str

@dataclass
class MultiAssetSignal:
    """Sinal agregado de múltiplos ativos"""
    primary_symbol: str
    supporting_symbols: List[str]
    signal_direction: int  # 1, 0, -1
    confidence: float
    cross_asset_strength: float
    regime_consistency: float
    timestamp: datetime

class SymbolSelector:
    """Seletor dinâmico de símbolos baseado em volatilidade e liquidez"""

    def __init__(self, min_volatility: float = 0.01, min_liquidity: float = 0.3):
        self.min_volatility = min_volatility
        self.min_liquidity = min_liquidity

        # Universo de símbolos por classe de ativo
        self.symbol_universe = {
            AssetClass.FOREX: [
                "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
                "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY"
            ],
            AssetClass.CRYPTO: [
                "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LTC/USD",
                "XRP/USD", "LINK/USD", "BCH/USD", "EOS/USD", "XLM/USD"
            ],
            AssetClass.COMMODITIES: [
                "Gold", "Silver", "Oil", "Gas", "Copper",
                "Platinum", "Palladium", "Wheat", "Corn", "Coffee"
            ],
            AssetClass.INDICES: [
                "SPX500", "NAS100", "US30", "UK100", "GER40",
                "FRA40", "AUS200", "JPN225", "HKG33", "EUSTX50"
            ]
        }

        # Configurações por classe
        self.asset_configs = {
            AssetClass.FOREX: {"max_symbols": 6, "volatility_weight": 0.4},
            AssetClass.CRYPTO: {"max_symbols": 4, "volatility_weight": 0.6},
            AssetClass.COMMODITIES: {"max_symbols": 3, "volatility_weight": 0.5},
            AssetClass.INDICES: {"max_symbols": 4, "volatility_weight": 0.3}
        }

    async def select_optimal_symbols(self,
                                   volatility_data: Dict[str, float],
                                   liquidity_data: Dict[str, float],
                                   max_total_symbols: int = 20) -> List[str]:
        """Seleciona símbolos ótimos baseado em critérios"""
        selected_symbols = []

        for asset_class in AssetClass:
            if asset_class not in self.symbol_universe:
                continue

            class_symbols = self.symbol_universe[asset_class]
            config = self.asset_configs.get(asset_class, {"max_symbols": 3, "volatility_weight": 0.5})

            # Calcular scores para símbolos da classe
            symbol_scores = []
            for symbol in class_symbols:
                volatility = volatility_data.get(symbol, 0.02)
                liquidity = liquidity_data.get(symbol, 0.5)

                # Filtros mínimos
                if volatility < self.min_volatility or liquidity < self.min_liquidity:
                    continue

                # Score combinado
                vol_weight = config["volatility_weight"]
                score = (volatility * vol_weight + liquidity * (1 - vol_weight))
                symbol_scores.append((symbol, score))

            # Selecionar top símbolos da classe
            symbol_scores.sort(key=lambda x: x[1], reverse=True)
            max_symbols = min(config["max_symbols"], len(symbol_scores))

            for symbol, score in symbol_scores[:max_symbols]:
                selected_symbols.append(symbol)

        # Limitar total
        if len(selected_symbols) > max_total_symbols:
            # Recalcular scores globais e selecionar top
            global_scores = []
            for symbol in selected_symbols:
                volatility = volatility_data.get(symbol, 0.02)
                liquidity = liquidity_data.get(symbol, 0.5)
                score = volatility * 0.5 + liquidity * 0.5
                global_scores.append((symbol, score))

            global_scores.sort(key=lambda x: x[1], reverse=True)
            selected_symbols = [symbol for symbol, score in global_scores[:max_total_symbols]]

        return selected_symbols

class CorrelationAnalyzer:
    """Analisador de correlações cross-asset"""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [50, 100, 200]

        # Buffers de preços por símbolo
        self.price_buffers: Dict[str, deque] = {}
        self.return_buffers: Dict[str, deque] = {}

        # Matrizes de correlação
        self.correlation_matrices: Dict[int, CorrelationMatrix] = {}

        # Configurações
        self.max_buffer_size = 500
        self.correlation_threshold = 0.7
        self.regime_change_threshold = 0.15

    async def add_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Adiciona dados de preço para análise"""
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.return_buffers[symbol] = deque(maxlen=self.max_buffer_size)

        # Adicionar preço
        self.price_buffers[symbol].append((timestamp, price))

        # Calcular retorno se possível
        if len(self.price_buffers[symbol]) >= 2:
            prev_price = self.price_buffers[symbol][-2][1]
            return_val = (price - prev_price) / prev_price
            self.return_buffers[symbol].append((timestamp, return_val))

    async def calculate_correlation_matrix(self, symbols: List[str],
                                         lookback_period: int) -> Optional[CorrelationMatrix]:
        """Calcula matriz de correlação para um período"""
        if len(symbols) < 2:
            return None

        try:
            # Coletar retornos alinhados temporalmente
            aligned_returns = self._align_returns(symbols, lookback_period)

            if len(aligned_returns) < lookback_period // 2:
                return None

            # Calcular matriz de correlação
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr().values

            # Classificar regime de correlação
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            regime = self._classify_correlation_regime(avg_correlation)

            # Calcular estabilidade (variância das correlações ao longo do tempo)
            stability_score = self._calculate_correlation_stability(symbols, lookback_period)

            return CorrelationMatrix(
                symbols=symbols,
                correlation_matrix=correlation_matrix,
                correlation_regime=regime,
                calculation_timestamp=datetime.now(),
                lookback_period=lookback_period,
                stability_score=stability_score
            )

        except Exception as e:
            logger.error(f"Erro ao calcular matriz de correlação: {e}")
            return None

    def _align_returns(self, symbols: List[str], lookback_period: int) -> Dict[str, List[float]]:
        """Alinha retornos temporalmente entre símbolos"""
        aligned_returns = {symbol: [] for symbol in symbols}

        # Encontrar timestamps comuns
        common_timestamps = None
        for symbol in symbols:
            if symbol not in self.return_buffers or len(self.return_buffers[symbol]) < lookback_period:
                continue

            symbol_timestamps = set(ts for ts, _ in list(self.return_buffers[symbol])[-lookback_period:])

            if common_timestamps is None:
                common_timestamps = symbol_timestamps
            else:
                common_timestamps = common_timestamps.intersection(symbol_timestamps)

        if not common_timestamps:
            return aligned_returns

        # Extrair retornos para timestamps comuns
        sorted_timestamps = sorted(common_timestamps)

        for symbol in symbols:
            if symbol not in self.return_buffers:
                continue

            symbol_data = {ts: ret for ts, ret in self.return_buffers[symbol]}

            for ts in sorted_timestamps:
                if ts in symbol_data:
                    aligned_returns[symbol].append(symbol_data[ts])

        return aligned_returns

    def _classify_correlation_regime(self, avg_correlation: float) -> CorrelationRegime:
        """Classifica regime de correlação"""
        if avg_correlation < 0.3:
            return CorrelationRegime.LOW
        elif avg_correlation < 0.6:
            return CorrelationRegime.MODERATE
        elif avg_correlation < 0.8:
            return CorrelationRegime.HIGH
        else:
            return CorrelationRegime.EXTREME

    def _calculate_correlation_stability(self, symbols: List[str], lookback_period: int) -> float:
        """Calcula estabilidade das correlações"""
        try:
            # Calcular correlações em janelas deslizantes
            window_size = lookback_period // 4
            correlations_over_time = []

            for i in range(window_size, lookback_period, window_size // 2):
                window_returns = self._align_returns(symbols, i)

                if len(window_returns[symbols[0]]) < window_size // 2:
                    continue

                window_df = pd.DataFrame(window_returns)
                window_corr = window_df.corr().values

                # Extrair correlações únicas (triângulo superior)
                triu_indices = np.triu_indices_from(window_corr, k=1)
                correlations_over_time.append(window_corr[triu_indices])

            if len(correlations_over_time) < 2:
                return 0.5

            # Calcular estabilidade como 1 - variância_normalizada
            correlations_array = np.array(correlations_over_time)
            variance = np.var(correlations_array, axis=0)
            stability = 1.0 - np.mean(variance)

            return max(min(stability, 1.0), 0.0)

        except Exception as e:
            logger.error(f"Erro no cálculo de estabilidade: {e}")
            return 0.5

    async def detect_correlation_regime_change(self, symbols: List[str]) -> bool:
        """Detecta mudança no regime de correlação"""
        if len(self.correlation_matrices) < 2:
            return False

        # Comparar matrizes mais recentes
        periods = sorted(self.correlation_matrices.keys())
        current_matrix = self.correlation_matrices[periods[-1]]
        previous_matrix = self.correlation_matrices[periods[-2]]

        # Calcular diferença nas correlações médias
        current_avg = np.mean(np.abs(current_matrix.correlation_matrix[np.triu_indices_from(current_matrix.correlation_matrix, k=1)]))
        previous_avg = np.mean(np.abs(previous_matrix.correlation_matrix[np.triu_indices_from(previous_matrix.correlation_matrix, k=1)]))

        change = abs(current_avg - previous_avg)

        return change > self.regime_change_threshold

class AssetClusterAnalyzer:
    """Analisador de clusters de ativos"""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.pca = PCA(n_components=0.95)  # 95% da variância

        self.clusters: List[AssetCluster] = []
        self.last_clustering_time: Optional[datetime] = None

    async def perform_clustering(self, correlation_matrix: CorrelationMatrix) -> List[AssetCluster]:
        """Realiza clustering dos ativos"""
        try:
            if correlation_matrix.correlation_matrix.shape[0] < self.n_clusters:
                # Não há ativos suficientes para clustering
                return []

            # Usar correlações como features para clustering
            features = correlation_matrix.correlation_matrix

            # Aplicar PCA se necessário
            if features.shape[1] > 10:
                features = self.pca.fit_transform(features)

            # Clustering
            cluster_labels = self.kmeans.fit_predict(features)

            # Criar objetos de cluster
            clusters = []
            for cluster_id in range(self.n_clusters):
                cluster_symbols = [
                    correlation_matrix.symbols[i]
                    for i in range(len(correlation_matrix.symbols))
                    if cluster_labels[i] == cluster_id
                ]

                if not cluster_symbols:
                    continue

                # Calcular correlação média do cluster
                cluster_indices = [correlation_matrix.symbols.index(s) for s in cluster_symbols]
                cluster_corr_matrix = correlation_matrix.correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                avg_correlation = np.mean(np.abs(cluster_corr_matrix[np.triu_indices_from(cluster_corr_matrix, k=1)]))

                # Símbolo representativo (com maior correlação média no cluster)
                representative_idx = 0
                max_avg_corr = 0
                for i, idx in enumerate(cluster_indices):
                    symbol_corr = np.mean(np.abs(correlation_matrix.correlation_matrix[idx, cluster_indices]))
                    if symbol_corr > max_avg_corr:
                        max_avg_corr = symbol_corr
                        representative_idx = i

                cluster = AssetCluster(
                    cluster_id=cluster_id,
                    symbols=cluster_symbols,
                    cluster_center=self.kmeans.cluster_centers_[cluster_id],
                    avg_correlation=avg_correlation,
                    volatility_level=self._classify_volatility_level(avg_correlation),
                    representative_symbol=cluster_symbols[representative_idx]
                )

                clusters.append(cluster)

            self.clusters = clusters
            self.last_clustering_time = datetime.now()

            logger.info(f"Clustering realizado: {len(clusters)} clusters criados")
            return clusters

        except Exception as e:
            logger.error(f"Erro no clustering: {e}")
            return []

    def _classify_volatility_level(self, avg_correlation: float) -> str:
        """Classifica nível de volatilidade baseado na correlação"""
        if avg_correlation > 0.7:
            return "high"
        elif avg_correlation > 0.4:
            return "medium"
        else:
            return "low"

class CrossAssetSignalGenerator:
    """Gerador de sinais cross-asset"""

    def __init__(self, correlation_analyzer: CorrelationAnalyzer):
        self.correlation_analyzer = correlation_analyzer

        # Tipos de sinais cross-asset
        self.signal_types = [
            "lead_lag",           # Um ativo lidera outro
            "divergence",         # Divergência entre ativos correlacionados
            "convergence",        # Convergência após divergência
            "sector_rotation",    # Rotação entre setores
            "risk_on_off",       # Risk-on/Risk-off
            "flight_to_quality"   # Fuga para qualidade
        ]

    async def generate_cross_asset_signals(self,
                                         predictions: Dict[str, EnsemblePrediction],
                                         correlation_matrix: CorrelationMatrix) -> List[CrossAssetSignal]:
        """Gera sinais cross-asset"""
        signals = []

        try:
            symbols = list(predictions.keys())

            # Lead-lag signals
            signals.extend(await self._detect_lead_lag_signals(predictions, correlation_matrix))

            # Divergence signals
            signals.extend(await self._detect_divergence_signals(predictions, correlation_matrix))

            # Risk-on/Risk-off signals
            signals.extend(await self._detect_risk_on_off_signals(predictions))

        except Exception as e:
            logger.error(f"Erro na geração de sinais cross-asset: {e}")

        return signals

    async def _detect_lead_lag_signals(self,
                                     predictions: Dict[str, EnsemblePrediction],
                                     correlation_matrix: CorrelationMatrix) -> List[CrossAssetSignal]:
        """Detecta sinais de lead-lag entre ativos"""
        signals = []

        try:
            symbols = correlation_matrix.symbols
            corr_matrix = correlation_matrix.correlation_matrix

            for i, source_symbol in enumerate(symbols):
                if source_symbol not in predictions:
                    continue

                source_pred = predictions[source_symbol]

                for j, target_symbol in enumerate(symbols):
                    if i >= j or target_symbol not in predictions:
                        continue

                    target_pred = predictions[target_symbol]
                    correlation = corr_matrix[i, j]

                    # Detectar lead-lag baseado em diferença de confiança
                    confidence_diff = source_pred.confidence - target_pred.confidence
                    prediction_diff = source_pred.final_prediction - target_pred.final_prediction

                    # Sinal válido se há alta correlação mas diferença significativa nas predições
                    if abs(correlation) > 0.6 and abs(confidence_diff) > 0.2:
                        strength = abs(confidence_diff) * abs(correlation)

                        signal = CrossAssetSignal(
                            source_symbol=source_symbol,
                            target_symbol=target_symbol,
                            signal_type="lead_lag",
                            strength=strength,
                            confidence=max(source_pred.confidence, target_pred.confidence),
                            timeframe="short",
                            timestamp=datetime.now(),
                            description=f"{source_symbol} leading {target_symbol} with correlation {correlation:.2f}"
                        )

                        signals.append(signal)

        except Exception as e:
            logger.error(f"Erro na detecção de lead-lag: {e}")

        return signals

    async def _detect_divergence_signals(self,
                                       predictions: Dict[str, EnsemblePrediction],
                                       correlation_matrix: CorrelationMatrix) -> List[CrossAssetSignal]:
        """Detecta sinais de divergência"""
        signals = []

        try:
            symbols = correlation_matrix.symbols
            corr_matrix = correlation_matrix.correlation_matrix

            for i, symbol1 in enumerate(symbols):
                if symbol1 not in predictions:
                    continue

                pred1 = predictions[symbol1]

                for j, symbol2 in enumerate(symbols):
                    if i >= j or symbol2 not in predictions:
                        continue

                    pred2 = predictions[symbol2]
                    correlation = corr_matrix[i, j]

                    # Divergência: alta correlação histórica mas sinais opostos
                    if abs(correlation) > 0.6:
                        signal_divergence = abs(pred1.final_prediction - pred2.final_prediction)

                        if signal_divergence > 0.3:  # Sinais divergentes
                            strength = signal_divergence * abs(correlation)

                            signal = CrossAssetSignal(
                                source_symbol=symbol1,
                                target_symbol=symbol2,
                                signal_type="divergence",
                                strength=strength,
                                confidence=min(pred1.confidence, pred2.confidence),
                                timeframe="medium",
                                timestamp=datetime.now(),
                                description=f"Divergência entre {symbol1} e {symbol2} (corr: {correlation:.2f})"
                            )

                            signals.append(signal)

        except Exception as e:
            logger.error(f"Erro na detecção de divergência: {e}")

        return signals

    async def _detect_risk_on_off_signals(self,
                                        predictions: Dict[str, EnsemblePrediction]) -> List[CrossAssetSignal]:
        """Detecta sinais de risk-on/risk-off"""
        signals = []

        try:
            # Classificar ativos por risco
            risk_on_assets = []  # Ações, crypto, commodities
            risk_off_assets = []  # Bonds, JPY, CHF

            for symbol, prediction in predictions.items():
                if any(x in symbol.upper() for x in ["SPX", "NAS", "BTC", "ETH", "OIL", "GOLD"]):
                    risk_on_assets.append((symbol, prediction))
                elif any(x in symbol.upper() for x in ["JPY", "CHF", "BOND"]):
                    risk_off_assets.append((symbol, prediction))

            # Calcular sentimento agregado
            if risk_on_assets and risk_off_assets:
                risk_on_sentiment = np.mean([pred.final_prediction for _, pred in risk_on_assets])
                risk_off_sentiment = np.mean([pred.final_prediction for _, pred in risk_off_assets])

                sentiment_diff = risk_on_sentiment - risk_off_sentiment

                if abs(sentiment_diff) > 0.2:
                    signal_type = "risk_on" if sentiment_diff > 0 else "risk_off"

                    for symbol, prediction in risk_on_assets + risk_off_assets:
                        signal = CrossAssetSignal(
                            source_symbol="MARKET_SENTIMENT",
                            target_symbol=symbol,
                            signal_type=signal_type,
                            strength=abs(sentiment_diff),
                            confidence=prediction.confidence,
                            timeframe="medium",
                            timestamp=datetime.now(),
                            description=f"Regime {signal_type} detectado para {symbol}"
                        )

                        signals.append(signal)

        except Exception as e:
            logger.error(f"Erro na detecção de risk-on/off: {e}")

        return signals

class MultiAssetManager:
    """Gerenciador principal de múltiplos ativos"""

    def __init__(self, max_symbols: int = 20):
        self.max_symbols = max_symbols

        # Componentes
        self.symbol_selector = SymbolSelector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.cluster_analyzer = AssetClusterAnalyzer()
        self.signal_generator = CrossAssetSignalGenerator(self.correlation_analyzer)

        # Estado
        self.active_symbols: Set[str] = set()
        self.asset_info: Dict[str, AssetInfo] = {}
        self.cross_asset_signals: List[CrossAssetSignal] = []
        self.market_regime = MarketRegime.SIDEWAYS

        # Ensembles por símbolo
        self.symbol_ensembles: Dict[str, AdvancedModelEnsemble] = {}

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=6)
        self.update_lock = threading.Lock()

        # Tarefas de monitoramento
        self.monitoring_task = None
        self.correlation_task = None

        logger.info("MultiAssetManager inicializado")

    async def initialize(self, initial_symbols: List[str]):
        """Inicializa o gerenciador com símbolos iniciais"""
        try:
            # Validar e adicionar símbolos
            for symbol in initial_symbols[:self.max_symbols]:
                await self.add_symbol(symbol)

            # Iniciar tarefas de monitoramento
            await self._start_monitoring_tasks()

            logger.info(f"MultiAssetManager inicializado com {len(self.active_symbols)} símbolos")

        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")

    async def add_symbol(self, symbol: str, asset_class: AssetClass = AssetClass.FOREX):
        """Adiciona um símbolo para gerenciamento"""
        try:
            if symbol in self.active_symbols:
                return

            # Criar informações do ativo
            asset_info = AssetInfo(
                symbol=symbol,
                asset_class=asset_class,
                base_currency=symbol.split('/')[0] if '/' in symbol else symbol[:3],
                quote_currency=symbol.split('/')[1] if '/' in symbol else "USD",
                tick_size=0.00001,  # Simplificado
                min_trade_size=0.01,
                trading_hours={"start": "00:00", "end": "24:00"},
                volatility_regime="normal",
                liquidity_score=0.8
            )

            self.asset_info[symbol] = asset_info
            self.active_symbols.add(symbol)

            # Criar ensemble para o símbolo
            ensemble = AdvancedModelEnsemble(feature_size=50, sequence_length=60)
            await ensemble.initialize_models()
            self.symbol_ensembles[symbol] = ensemble

            logger.info(f"Símbolo {symbol} adicionado ao gerenciamento")

        except Exception as e:
            logger.error(f"Erro ao adicionar símbolo {symbol}: {e}")

    async def remove_symbol(self, symbol: str):
        """Remove um símbolo do gerenciamento"""
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            self.asset_info.pop(symbol, None)

            if symbol in self.symbol_ensembles:
                await self.symbol_ensembles[symbol].shutdown()
                del self.symbol_ensembles[symbol]

            logger.info(f"Símbolo {symbol} removido do gerenciamento")

    async def process_multi_asset_tick(self, symbol: str, tick_data: ProcessedTickData):
        """Processa tick de dados multi-asset"""
        if symbol not in self.active_symbols:
            return

        # Adicionar dados ao analisador de correlação
        await self.correlation_analyzer.add_price_data(
            symbol, tick_data.price, tick_data.timestamp
        )

        # Processar ensemble do símbolo se disponível
        if symbol in self.symbol_ensembles:
            # Simular features para o exemplo
            features = np.random.randn(60, 50)  # Seria obtido do feature engine
            await self.symbol_ensembles[symbol].add_training_data(features, tick_data.price)

    async def generate_multi_asset_signals(self) -> List[MultiAssetSignal]:
        """Gera sinais agregados multi-asset"""
        signals = []

        try:
            # Obter predições de todos os ensembles
            predictions = {}
            for symbol, ensemble in self.symbol_ensembles.items():
                if ensemble.is_trained:
                    # Simular features para predição
                    features = np.random.randn(60, 50)
                    prediction = await ensemble.predict(features)
                    predictions[symbol] = prediction

            if len(predictions) < 2:
                return signals

            # Calcular matriz de correlação mais recente
            symbols = list(predictions.keys())
            correlation_matrix = await self.correlation_analyzer.calculate_correlation_matrix(
                symbols, lookback_period=100
            )

            if not correlation_matrix:
                return signals

            # Gerar sinais cross-asset
            cross_signals = await self.signal_generator.generate_cross_asset_signals(
                predictions, correlation_matrix
            )

            # Converter para sinais multi-asset
            for cross_signal in cross_signals:
                if cross_signal.target_symbol in predictions:
                    target_prediction = predictions[cross_signal.target_symbol]

                    # Determinar direção baseada na predição
                    signal_direction = 1 if target_prediction.final_prediction > 0.5 else -1
                    if abs(target_prediction.final_prediction - 0.5) < 0.1:
                        signal_direction = 0

                    # Encontrar símbolos de suporte (correlacionados)
                    supporting_symbols = []
                    if correlation_matrix:
                        target_idx = symbols.index(cross_signal.target_symbol)
                        for i, symbol in enumerate(symbols):
                            if symbol != cross_signal.target_symbol:
                                correlation = correlation_matrix.correlation_matrix[target_idx, i]
                                if abs(correlation) > 0.5:
                                    supporting_symbols.append(symbol)

                    multi_signal = MultiAssetSignal(
                        primary_symbol=cross_signal.target_symbol,
                        supporting_symbols=supporting_symbols,
                        signal_direction=signal_direction,
                        confidence=cross_signal.confidence,
                        cross_asset_strength=cross_signal.strength,
                        regime_consistency=self._calculate_regime_consistency(correlation_matrix),
                        timestamp=datetime.now()
                    )

                    signals.append(multi_signal)

        except Exception as e:
            logger.error(f"Erro na geração de sinais multi-asset: {e}")

        return signals

    def _calculate_regime_consistency(self, correlation_matrix: CorrelationMatrix) -> float:
        """Calcula consistência do regime de mercado"""
        try:
            # Baseado na estabilidade das correlações
            return correlation_matrix.stability_score

        except Exception:
            return 0.5

    async def optimize_symbol_allocation(self) -> Dict[str, float]:
        """Otimiza alocação entre símbolos"""
        try:
            if len(self.active_symbols) < 2:
                return {symbol: 1.0 for symbol in self.active_symbols}

            # Simular dados de retorno para otimização
            return_data = {}
            for symbol in self.active_symbols:
                # Em produção, seria obtido dos dados reais
                returns = np.random.normal(0.001, 0.02, 100)  # Simular retornos
                return_data[symbol] = returns.tolist()

            # Usar portfolio optimizer
            optimizer = RealPortfolioOptimizer()

            # Adicionar ativos simulados
            for symbol in self.active_symbols:
                # Simular dados de tick
                simulated_ticks = []
                for i in range(100):
                    tick = ProcessedTickData(
                        timestamp=datetime.now() - timedelta(hours=i),
                        price=100 + np.random.normal(0, 2),
                        volume=1000,
                        symbol=symbol
                    )
                    simulated_ticks.append(tick)

                await optimizer.add_asset(symbol, simulated_ticks)

            # Otimizar
            from real_portfolio_optimizer import OptimizationMethod, RiskConstraint
            result = await optimizer.optimize_portfolio(OptimizationMethod.RISK_PARITY)

            return result.weights

        except Exception as e:
            logger.error(f"Erro na otimização de alocação: {e}")
            # Retornar alocação igual em caso de erro
            equal_weight = 1.0 / len(self.active_symbols)
            return {symbol: equal_weight for symbol in self.active_symbols}

    async def _start_monitoring_tasks(self):
        """Inicia tarefas de monitoramento"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.correlation_task = asyncio.create_task(self._correlation_loop())

    async def _monitoring_loop(self):
        """Loop de monitoramento principal"""
        while True:
            try:
                # Atualizar regime de mercado
                await self._update_market_regime()

                # Gerenciar símbolos dinamicamente
                await self._dynamic_symbol_management()

                await asyncio.sleep(300)  # 5 minutos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                await asyncio.sleep(60)

    async def _correlation_loop(self):
        """Loop de análise de correlação"""
        while True:
            try:
                symbols = list(self.active_symbols)

                # Calcular matrizes para diferentes períodos
                for period in self.correlation_analyzer.lookback_periods:
                    correlation_matrix = await self.correlation_analyzer.calculate_correlation_matrix(
                        symbols, period
                    )

                    if correlation_matrix:
                        self.correlation_analyzer.correlation_matrices[period] = correlation_matrix

                # Clustering
                if symbols and len(symbols) >= 5:
                    latest_matrix = list(self.correlation_analyzer.correlation_matrices.values())[-1]
                    await self.cluster_analyzer.perform_clustering(latest_matrix)

                await asyncio.sleep(600)  # 10 minutos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no loop de correlação: {e}")
                await asyncio.sleep(300)

    async def _update_market_regime(self):
        """Atualiza regime de mercado"""
        try:
            # Analisar correlações médias
            if self.correlation_analyzer.correlation_matrices:
                latest_matrix = list(self.correlation_analyzer.correlation_matrices.values())[-1]

                if latest_matrix.correlation_regime == CorrelationRegime.HIGH:
                    self.market_regime = MarketRegime.VOLATILE
                elif latest_matrix.correlation_regime == CorrelationRegime.LOW:
                    self.market_regime = MarketRegime.SIDEWAYS
                else:
                    # Manter regime atual se moderado
                    pass

        except Exception as e:
            logger.error(f"Erro na atualização do regime: {e}")

    async def _dynamic_symbol_management(self):
        """Gerenciamento dinâmico de símbolos"""
        try:
            # Simular dados de volatilidade e liquidez
            volatility_data = {symbol: np.random.uniform(0.01, 0.05) for symbol in self.active_symbols}
            liquidity_data = {symbol: np.random.uniform(0.3, 1.0) for symbol in self.active_symbols}

            # Selecionar símbolos ótimos
            optimal_symbols = await self.symbol_selector.select_optimal_symbols(
                volatility_data, liquidity_data, self.max_symbols
            )

            # Adicionar novos símbolos
            for symbol in optimal_symbols:
                if symbol not in self.active_symbols and len(self.active_symbols) < self.max_symbols:
                    await self.add_symbol(symbol)

            # Remover símbolos com baixa performance
            current_symbols = list(self.active_symbols)
            for symbol in current_symbols:
                if symbol not in optimal_symbols and len(self.active_symbols) > 5:
                    await self.remove_symbol(symbol)

        except Exception as e:
            logger.error(f"Erro no gerenciamento dinâmico: {e}")

    async def get_multi_asset_status(self) -> Dict[str, Any]:
        """Obtém status completo multi-asset"""
        return {
            "active_symbols": list(self.active_symbols),
            "total_symbols": len(self.active_symbols),
            "market_regime": self.market_regime.value,
            "correlation_matrices": len(self.correlation_analyzer.correlation_matrices),
            "clusters": len(self.cluster_analyzer.clusters),
            "cross_asset_signals": len(self.cross_asset_signals),
            "trained_ensembles": sum(1 for e in self.symbol_ensembles.values() if e.is_trained),
            "asset_classes": {
                ac.value: sum(1 for info in self.asset_info.values() if info.asset_class == ac)
                for ac in AssetClass
            }
        }

    async def shutdown(self):
        """Encerra o gerenciador multi-asset"""
        # Parar tarefas
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.correlation_task:
            self.correlation_task.cancel()

        # Encerrar ensembles
        for ensemble in self.symbol_ensembles.values():
            await ensemble.shutdown()

        self.thread_pool.shutdown(wait=True)
        logger.info("MultiAssetManager encerrado")
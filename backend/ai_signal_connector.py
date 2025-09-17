"""
AI Signal Connector - Conecta Modelos AI aos Sinais de Trading
Sistema que integra previsões dos modelos AI com execução de trading em tempo real
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import json

from real_model_trainer import ModelType, TrainingSession
from realtime_feature_processor import RealtimeFeatureProcessor
from real_trading_executor import RealTradingExecutor, TradingMode, TradeType
from real_position_manager import RealPositionManager
from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem

class SignalStrength(Enum):
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class SignalDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class AISignal:
    """Sinal gerado pelos modelos AI"""
    symbol: str
    direction: SignalDirection
    strength: float
    confidence: float
    timestamp: datetime
    model_predictions: Dict[str, float]
    features_used: Dict[str, float]
    expected_return: float
    risk_score: float
    holding_period: int  # em minutos
    stop_loss_pct: float
    take_profit_pct: float
    metadata: Dict[str, Any]

@dataclass
class ModelEnsemble:
    """Conjunto de modelos AI"""
    models: Dict[ModelType, Any]
    weights: Dict[ModelType, float]
    last_updated: datetime
    performance_scores: Dict[ModelType, float]
    is_ready: bool = False

class AISignalConnector:
    """Conecta modelos AI aos sinais de trading"""

    def __init__(self):
        # Componentes principais
        self.feature_processor = RealtimeFeatureProcessor()
        self.trading_executor = RealTradingExecutor()
        self.position_manager = RealPositionManager(None)  # Will be set in initialize
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()

        # Ensemble de modelos
        self.model_ensemble = ModelEnsemble(
            models={},
            weights={},
            last_updated=datetime.utcnow(),
            performance_scores={}
        )

        # Configurações de sinal
        self.signal_threshold = 0.6  # Threshold mínimo para gerar sinal
        self.max_signals_per_hour = 10
        self.min_signal_gap_minutes = 5
        self.symbols_monitored = [
            "R_10", "R_25", "R_50", "R_75", "R_100",
            "FRXEURUSD", "FRXGBPUSD", "FRXUSDJPY",
            "FRXAUDUSD", "FRXUSDCAD"
        ]

        # Estado do sistema
        self.is_active = False
        self.signals_generated_hour = 0
        self.last_signal_time = {}
        self.signal_history = []
        self.performance_metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "avg_return": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0
        }

        # Tasks assíncronas
        self.signal_generation_task = None
        self.model_update_task = None

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializa o conector de sinais AI"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.feature_processor.initialize()
            await self.trading_executor.initialize()
            await self.logger.initialize()

            # Carregar modelos treinados
            await self._load_trained_models()

            # Verificar se ensemble está pronto
            if len(self.model_ensemble.models) > 0:
                self.model_ensemble.is_ready = True

            await self.logger.log_activity("ai_signal_connector_initialized", {
                "models_loaded": len(self.model_ensemble.models),
                "symbols_monitored": len(self.symbols_monitored),
                "ensemble_ready": self.model_ensemble.is_ready
            })

            print("✅ AI Signal Connector inicializado com sucesso")

        except Exception as e:
            await self.logger.log_error("ai_signal_connector_init_error", str(e))
            raise

    async def start_signal_generation(self, trading_mode: TradingMode = TradingMode.PAPER):
        """Inicia geração de sinais AI"""
        if self.is_active:
            return

        if not self.model_ensemble.is_ready:
            print("❌ Ensemble de modelos não está pronto. Carregue os modelos primeiro.")
            return

        self.is_active = True

        # Iniciar feature processor
        await self.feature_processor.start_processing(self.symbols_monitored)

        # Iniciar trading executor
        await self.trading_executor.start_trading(self.symbols_monitored, trading_mode)

        # Iniciar tasks
        self.signal_generation_task = asyncio.create_task(self._signal_generation_loop())
        self.model_update_task = asyncio.create_task(self._model_performance_update_loop())

        await self.logger.log_activity("signal_generation_started", {
            "trading_mode": trading_mode.value,
            "symbols": self.symbols_monitored
        })

        print(f"🤖 Geração de sinais AI iniciada - Modo: {trading_mode.value}")

    async def stop_signal_generation(self):
        """Para a geração de sinais"""
        self.is_active = False

        if self.signal_generation_task:
            self.signal_generation_task.cancel()

        if self.model_update_task:
            self.model_update_task.cancel()

        await self.feature_processor.stop_processing()
        await self.trading_executor.stop_trading()

        await self.logger.log_activity("signal_generation_stopped", {})
        print("⏹️ Geração de sinais AI parada")

    async def _signal_generation_loop(self):
        """Loop principal de geração de sinais"""
        while self.is_active:
            try:
                current_hour = datetime.utcnow().hour

                # Reset contador a cada hora
                if current_hour != getattr(self, '_last_hour', -1):
                    self.signals_generated_hour = 0
                    self._last_hour = current_hour

                # Verificar limite de sinais por hora
                if self.signals_generated_hour >= self.max_signals_per_hour:
                    await asyncio.sleep(60)  # Aguardar 1 minuto
                    continue

                # Gerar sinais para cada símbolo
                for symbol in self.symbols_monitored:
                    await self._process_symbol_for_signals(symbol)

                await asyncio.sleep(10)  # Verificar a cada 10 segundos

            except Exception as e:
                await self.logger.log_error("signal_generation_error", str(e))
                await asyncio.sleep(30)

    async def _process_symbol_for_signals(self, symbol: str):
        """Processa um símbolo para gerar sinais"""
        try:
            # Verificar gap mínimo entre sinais
            last_signal = self.last_signal_time.get(symbol)
            if last_signal:
                time_since_last = (datetime.utcnow() - last_signal).total_seconds() / 60
                if time_since_last < self.min_signal_gap_minutes:
                    return

            # Obter features do processador
            features = await self.feature_processor.get_latest_features(symbol)
            if not features or len(features) < 50:  # Mínimo de features necessárias
                return

            # Gerar previsões dos modelos
            predictions = await self._generate_model_predictions(symbol, features)
            if not predictions:
                return

            # Calcular ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(predictions)

            # Gerar sinal se passar no threshold
            if abs(ensemble_prediction) >= self.signal_threshold:
                signal = await self._create_ai_signal(symbol, features, predictions, ensemble_prediction)

                if signal:
                    await self._execute_signal(signal)
                    self.signals_generated_hour += 1
                    self.last_signal_time[symbol] = datetime.utcnow()

        except Exception as e:
            await self.logger.log_error("symbol_signal_processing_error", f"{symbol}: {str(e)}")

    async def _generate_model_predictions(self, symbol: str, features: Dict[str, float]) -> Dict[ModelType, float]:
        """Gera previsões de todos os modelos"""
        predictions = {}

        try:
            # Converter features para array numpy
            feature_array = np.array(list(features.values())).reshape(1, -1)

            # Normalizar features
            feature_array = self._normalize_features(feature_array)

            for model_type, model in self.model_ensemble.models.items():
                try:
                    if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING,
                                    ModelType.XGBOOST, ModelType.LIGHTGBM]:
                        # Modelos sklearn
                        prediction = model.predict(feature_array)[0]

                    elif model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
                        # Modelos PyTorch
                        model.eval()
                        with torch.no_grad():
                            tensor_input = torch.FloatTensor(feature_array)
                            prediction = model(tensor_input).item()

                    else:
                        # Outros modelos
                        prediction = model.predict(feature_array)[0]

                    predictions[model_type] = float(prediction)

                except Exception as e:
                    await self.logger.log_error("model_prediction_error", f"{model_type.value}: {str(e)}")

            return predictions

        except Exception as e:
            await self.logger.log_error("predictions_generation_error", f"{symbol}: {str(e)}")
            return {}

    def _calculate_ensemble_prediction(self, predictions: Dict[ModelType, float]) -> float:
        """Calcula previsão do ensemble"""
        if not predictions:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for model_type, prediction in predictions.items():
            weight = self.model_ensemble.weights.get(model_type, 1.0)
            performance = self.model_ensemble.performance_scores.get(model_type, 0.5)

            # Ajustar peso baseado na performance
            adjusted_weight = weight * performance

            weighted_sum += prediction * adjusted_weight
            total_weight += adjusted_weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _create_ai_signal(
        self,
        symbol: str,
        features: Dict[str, float],
        predictions: Dict[ModelType, float],
        ensemble_prediction: float
    ) -> Optional[AISignal]:
        """Cria um sinal AI baseado nas previsões"""
        try:
            # Determinar direção
            direction = SignalDirection.BUY if ensemble_prediction > 0 else SignalDirection.SELL

            # Calcular força do sinal
            strength = min(abs(ensemble_prediction), 1.0)

            # Calcular confiança baseada na concordância dos modelos
            prediction_values = list(predictions.values())
            agreement = self._calculate_model_agreement(prediction_values)
            confidence = agreement * strength

            # Verificar se sinal é forte o suficiente
            if confidence < self.signal_threshold:
                return None

            # Calcular parâmetros de risco
            volatility = features.get('volatility_20', 0.02)
            rsi = features.get('rsi_14', 50)

            # Ajustar stop loss e take profit baseado na volatilidade
            base_sl = 0.01  # 1%
            base_tp = 0.02  # 2%

            stop_loss_pct = base_sl + (volatility * 2)  # Ajustar por volatilidade
            take_profit_pct = base_tp + (volatility * 1.5)

            # Ajustar por RSI (evitar sobrecompra/sobrevenda extrema)
            if direction == SignalDirection.BUY and rsi > 80:
                confidence *= 0.7  # Reduzir confiança
            elif direction == SignalDirection.SELL and rsi < 20:
                confidence *= 0.7

            # Calcular período de holding esperado
            holding_period = self._calculate_holding_period(features, strength)

            # Calcular retorno esperado
            expected_return = ensemble_prediction * strength

            # Calcular score de risco
            risk_score = self._calculate_risk_score(features, volatility, strength)

            signal = AISignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                model_predictions=predictions,
                features_used=features,
                expected_return=expected_return,
                risk_score=risk_score,
                holding_period=holding_period,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                metadata={
                    "ensemble_prediction": ensemble_prediction,
                    "model_agreement": agreement,
                    "volatility": volatility,
                    "rsi": rsi
                }
            )

            return signal

        except Exception as e:
            await self.logger.log_error("signal_creation_error", f"{symbol}: {str(e)}")
            return None

    def _calculate_model_agreement(self, predictions: List[float]) -> float:
        """Calcula o nível de concordância entre modelos"""
        if len(predictions) < 2:
            return 0.5

        # Normalizar previsões para [-1, 1]
        normalized = [max(-1, min(1, p)) for p in predictions]

        # Calcular direções
        directions = [1 if p > 0 else -1 for p in normalized]

        # Percentual de modelos que concordam com a direção majoritária
        majority_direction = 1 if sum(directions) > 0 else -1
        agreement_count = sum(1 for d in directions if d == majority_direction)

        agreement_ratio = agreement_count / len(directions)

        # Ajustar pela dispersão das previsões
        std_dev = np.std(normalized)
        dispersion_penalty = min(std_dev, 0.5)  # Máximo 50% de penalidade

        return max(0.1, agreement_ratio - dispersion_penalty)

    def _calculate_holding_period(self, features: Dict[str, float], strength: float) -> int:
        """Calcula período esperado de holding"""
        base_period = 15  # 15 minutos base

        # Ajustar por volatilidade
        volatility = features.get('volatility_20', 0.02)
        volatility_factor = 1 / (1 + volatility * 10)  # Menor período para alta volatilidade

        # Ajustar por força do sinal
        strength_factor = 1 + strength  # Sinais mais fortes podem durar mais

        period = int(base_period * volatility_factor * strength_factor)
        return max(5, min(period, 60))  # Entre 5 e 60 minutos

    def _calculate_risk_score(self, features: Dict[str, float], volatility: float, strength: float) -> float:
        """Calcula score de risco do sinal"""
        base_risk = 0.3

        # Ajustar por volatilidade
        volatility_risk = volatility * 5  # Volatilidade contribui para risco

        # Ajustar por indicadores técnicos
        rsi = features.get('rsi_14', 50)
        rsi_risk = 0.0
        if rsi > 80 or rsi < 20:  # Condições extremas
            rsi_risk = 0.2

        # Sinais mais fortes podem ter menor risco (mais confiáveis)
        strength_risk_reduction = strength * 0.1

        total_risk = base_risk + volatility_risk + rsi_risk - strength_risk_reduction
        return max(0.1, min(total_risk, 1.0))

    async def _execute_signal(self, signal: AISignal):
        """Executa um sinal AI"""
        try:
            # Determinar tipo de trade
            trade_type = TradeType.BUY if signal.direction == SignalDirection.BUY else TradeType.SELL

            # Calcular tamanho da posição baseado na confiança
            base_amount = 10.0  # Valor base
            position_size = base_amount * signal.confidence

            # Executar trade
            result = await self.trading_executor.execute_trade(
                symbol=signal.symbol,
                trade_type=trade_type,
                amount=position_size,
                duration_minutes=signal.holding_period,
                stop_loss_pct=signal.stop_loss_pct,
                take_profit_pct=signal.take_profit_pct
            )

            if result and result.success:
                # Salvar sinal executado
                self.signal_history.append(signal)
                self.performance_metrics["total_signals"] += 1

                # Cache do sinal
                await self.cache_manager.set(
                    CacheNamespace.AI_PREDICTIONS,
                    f"executed_signal_{signal.symbol}_{int(signal.timestamp.timestamp())}",
                    asdict(signal),
                    ttl=3600
                )

                # Log da execução
                await self.logger.log_activity("ai_signal_executed", {
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "amount": position_size,
                    "contract_id": result.contract_id
                })

                print(f"🎯 Sinal AI executado: {signal.symbol} {signal.direction.value} - Conf: {signal.confidence:.2f}")

            else:
                await self.logger.log_error("signal_execution_failed", f"{signal.symbol}: Falha na execução")

        except Exception as e:
            await self.logger.log_error("signal_execution_error", f"{signal.symbol}: {str(e)}")

    async def _load_trained_models(self):
        """Carrega modelos treinados"""
        try:
            # Carregar modelos do cache ou arquivo
            for model_type in ModelType:
                try:
                    # Tentar carregar do cache primeiro
                    cached_model = await self.cache_manager.get(
                        CacheNamespace.AI_PREDICTIONS,
                        f"trained_model_{model_type.value}"
                    )

                    if cached_model:
                        # Deserializar modelo
                        if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                            model = joblib.loads(cached_model)
                        else:
                            # Para modelos torch, seria necessário carregar estado
                            continue

                        self.model_ensemble.models[model_type] = model
                        self.model_ensemble.weights[model_type] = 1.0
                        self.model_ensemble.performance_scores[model_type] = 0.7  # Score padrão

                except Exception as e:
                    self.logger_py.warning(f"Falha ao carregar modelo {model_type.value}: {e}")

            # Se não há modelos, criar modelos simples para demonstração
            if len(self.model_ensemble.models) == 0:
                await self._create_demo_models()

            self.model_ensemble.last_updated = datetime.utcnow()

            print(f"📚 Modelos carregados: {list(self.model_ensemble.models.keys())}")

        except Exception as e:
            await self.logger.log_error("model_loading_error", str(e))

    async def _create_demo_models(self):
        """Cria modelos simples para demonstração"""
        try:
            # Modelo Random Forest simples
            rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

            # Dados de exemplo para treinar
            X_demo = np.random.randn(100, 20)
            y_demo = np.random.randn(100)

            rf_model.fit(X_demo, y_demo)

            self.model_ensemble.models[ModelType.RANDOM_FOREST] = rf_model
            self.model_ensemble.weights[ModelType.RANDOM_FOREST] = 1.0
            self.model_ensemble.performance_scores[ModelType.RANDOM_FOREST] = 0.6

            print("🎲 Modelo demo criado para demonstração")

        except Exception as e:
            await self.logger.log_error("demo_model_creation_error", str(e))

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normaliza features para modelos"""
        # Normalização Z-score simples
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Evitar divisão por zero

        return (features - mean) / std

    async def _model_performance_update_loop(self):
        """Loop para atualizar performance dos modelos"""
        while self.is_active:
            try:
                await self._update_model_performance()
                await asyncio.sleep(300)  # A cada 5 minutos

            except Exception as e:
                await self.logger.log_error("model_performance_update_error", str(e))
                await asyncio.sleep(600)

    async def _update_model_performance(self):
        """Atualiza scores de performance dos modelos"""
        try:
            # Analisar sinais recentes e sua performance
            recent_signals = [s for s in self.signal_history[-50:]]  # Últimos 50 sinais

            if len(recent_signals) < 10:
                return

            # Calcular performance por modelo
            # (Implementação simplificada - em produção seria mais complexa)
            for model_type in self.model_ensemble.models.keys():
                # Simular score de performance
                base_score = self.model_ensemble.performance_scores.get(model_type, 0.5)
                noise = np.random.normal(0, 0.05)  # Pequena variação
                new_score = max(0.1, min(0.9, base_score + noise))

                self.model_ensemble.performance_scores[model_type] = new_score

        except Exception as e:
            await self.logger.log_error("performance_update_calculation_error", str(e))

    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos sinais"""
        try:
            total_signals = len(self.signal_history)

            if total_signals == 0:
                return {
                    "total_signals": 0,
                    "signals_today": 0,
                    "avg_confidence": 0.0,
                    "symbol_distribution": {},
                    "direction_distribution": {},
                    "performance_metrics": self.performance_metrics
                }

            # Sinais de hoje
            today = datetime.utcnow().date()
            signals_today = sum(1 for s in self.signal_history if s.timestamp.date() == today)

            # Confiança média
            avg_confidence = np.mean([s.confidence for s in self.signal_history])

            # Distribuição por símbolo
            symbol_dist = {}
            for signal in self.signal_history:
                symbol_dist[signal.symbol] = symbol_dist.get(signal.symbol, 0) + 1

            # Distribuição por direção
            direction_dist = {}
            for signal in self.signal_history:
                dir_val = signal.direction.value
                direction_dist[dir_val] = direction_dist.get(dir_val, 0) + 1

            return {
                "total_signals": total_signals,
                "signals_today": signals_today,
                "signals_this_hour": self.signals_generated_hour,
                "avg_confidence": float(avg_confidence),
                "symbol_distribution": symbol_dist,
                "direction_distribution": direction_dist,
                "performance_metrics": self.performance_metrics,
                "model_ensemble_status": {
                    "models_loaded": len(self.model_ensemble.models),
                    "is_ready": self.model_ensemble.is_ready,
                    "last_updated": self.model_ensemble.last_updated.isoformat()
                }
            }

        except Exception as e:
            await self.logger.log_error("signal_statistics_error", str(e))
            return {}

    async def shutdown(self):
        """Encerra o conector de sinais"""
        await self.stop_signal_generation()
        await self.logger.log_activity("ai_signal_connector_shutdown", {})
        print("🔌 AI Signal Connector encerrado")
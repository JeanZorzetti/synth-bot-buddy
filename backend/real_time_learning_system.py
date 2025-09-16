"""
Real-Time Learning System - Sistema de Aprendizado em Tempo Real
Sistema avançado de aprendizado online que adapta modelos continuamente com novos dados.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from river import linear_model, ensemble, metrics, drift
import torch
import torch.nn as nn
import torch.optim as optim

from advanced_model_ensemble import AdvancedModelEnsemble, ModelPrediction, EnsemblePrediction
from real_trading_executor import RealTradeResult

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    BATCH = "batch"
    ONLINE = "online"
    MINI_BATCH = "mini_batch"
    ACTIVE = "active"

class DriftDetectionMethod(Enum):
    ADWIN = "adwin"
    DDM = "ddm"
    EDDM = "eddm"
    PAGE_HINKLEY = "page_hinkley"

@dataclass
class PerformanceDrift:
    """Detecção de degradação de performance"""
    model_id: str
    detection_time: datetime
    drift_type: str
    severity: float  # 0-1
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    recommendation: str

@dataclass
class LearningEvent:
    """Evento de aprendizado"""
    timestamp: datetime
    features: np.ndarray
    target: float
    prediction: Optional[float]
    loss: Optional[float]
    learning_rate: float
    model_updated: bool

@dataclass
class AdaptationMetrics:
    """Métricas de adaptação do modelo"""
    total_updates: int
    successful_adaptations: int
    drift_detections: int
    average_learning_time: float
    performance_improvement: float
    data_efficiency: float  # Learning per sample
    adaptation_rate: float

class OnlineLearner:
    """Aprendiz online para modelos de River/sklearn"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

        # Modelos online
        self.linear_model = linear_model.LinearRegression()
        self.pa_model = linear_model.PassiveAggressiveRegressor()
        self.ensemble_model = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            max_depth=10,
            seed=42
        )

        # Métricas online
        self.mse_metric = metrics.MSE()
        self.mae_metric = metrics.MAE()

        # Estado
        self.total_samples = 0
        self.last_update = datetime.now()

    async def learn_sample(self, features: Dict[str, float], target: float) -> float:
        """Aprende com uma amostra individual"""
        try:
            # Fazer predição primeiro
            pred_linear = self.linear_model.predict_one(features)
            pred_pa = self.pa_model.predict_one(features)
            pred_ensemble = self.ensemble_model.predict_one(features)

            # Média das predições
            prediction = (pred_linear + pred_pa + pred_ensemble) / 3

            # Atualizar modelos
            self.linear_model.learn_one(features, target)
            self.pa_model.learn_one(features, target)
            self.ensemble_model.learn_one(features, target)

            # Atualizar métricas
            self.mse_metric.update(target, prediction)
            self.mae_metric.update(target, prediction)

            self.total_samples += 1
            self.last_update = datetime.now()

            return prediction

        except Exception as e:
            logger.error(f"Erro no aprendizado online: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Obtém métricas atuais"""
        return {
            "mse": self.mse_metric.get(),
            "mae": self.mae_metric.get(),
            "total_samples": self.total_samples,
            "last_update": self.last_update.isoformat()
        }

class ModelPerformanceTracker:
    """Rastreador de performance de modelos"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Histórico de performance
        self.predictions_history: deque = deque(maxlen=window_size)
        self.targets_history: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

        # Métricas atuais
        self.current_accuracy = 0.0
        self.current_mse = 0.0
        self.current_sharpe = 0.0

        # Baseline para detecção de drift
        self.baseline_metrics = {
            "accuracy": 0.6,
            "mse": 0.1,
            "sharpe": 0.5
        }

        # Detectores de drift
        self.drift_detectors = self._initialize_drift_detectors()

    def _initialize_drift_detectors(self):
        """Inicializa detectores de drift"""
        return {
            DriftDetectionMethod.ADWIN: drift.ADWIN(delta=0.002),
            DriftDetectionMethod.DDM: drift.DDM(),
            DriftDetectionMethod.EDDM: drift.EDDM()
        }

    async def add_prediction(self, prediction: float, target: float):
        """Adiciona predição e alvo para tracking"""
        self.predictions_history.append(prediction)
        self.targets_history.append(target)
        self.timestamps.append(datetime.now())

        # Atualizar métricas
        await self._update_metrics()

        # Verificar drift
        await self._check_drift(prediction, target)

    async def _update_metrics(self):
        """Atualiza métricas de performance"""
        if len(self.predictions_history) < 10:
            return

        predictions = np.array(list(self.predictions_history))
        targets = np.array(list(self.targets_history))

        # Accuracy (direção)
        pred_direction = (predictions > 0.5).astype(int)
        target_direction = (targets > 0.5).astype(int)
        self.current_accuracy = accuracy_score(target_direction, pred_direction)

        # MSE
        self.current_mse = mean_squared_error(targets, predictions)

        # Sharpe ratio simplificado
        returns = predictions - 0.5
        if np.std(returns) > 0:
            self.current_sharpe = np.mean(returns) / np.std(returns)
        else:
            self.current_sharpe = 0.0

    async def _check_drift(self, prediction: float, target: float):
        """Verifica drift na performance"""
        error = abs(prediction - target)

        # Atualizar detectores
        for method, detector in self.drift_detectors.items():
            try:
                in_drift, in_warning = detector.update(error)

                if in_drift:
                    logger.warning(f"Drift detectado pelo método {method.value}")
                    await self._handle_drift_detection(method, detector)

            except Exception as e:
                logger.error(f"Erro no detector {method.value}: {e}")

    async def _handle_drift_detection(self, method: DriftDetectionMethod, detector):
        """Lida com detecção de drift"""
        # Implementar ação de response ao drift
        logger.info(f"Ação necessária devido ao drift detectado por {method.value}")

    def detect_performance_degradation(self) -> Optional[PerformanceDrift]:
        """Detecta degradação de performance"""
        if len(self.predictions_history) < self.window_size // 2:
            return None

        # Comparar performance atual com baseline
        degradation_threshold = 0.1  # 10% de degradação

        current_metrics = {
            "accuracy": self.current_accuracy,
            "mse": self.current_mse,
            "sharpe": self.current_sharpe
        }

        degradation_detected = False
        severity = 0.0

        for metric, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics[metric]

            if metric == "mse":
                # Para MSE, menor é melhor
                if current_value > baseline_value * (1 + degradation_threshold):
                    degradation_detected = True
                    severity = max(severity, (current_value - baseline_value) / baseline_value)
            else:
                # Para accuracy e sharpe, maior é melhor
                if current_value < baseline_value * (1 - degradation_threshold):
                    degradation_detected = True
                    severity = max(severity, (baseline_value - current_value) / baseline_value)

        if degradation_detected:
            return PerformanceDrift(
                model_id="ensemble",
                detection_time=datetime.now(),
                drift_type="performance_degradation",
                severity=min(severity, 1.0),
                metrics_before=self.baseline_metrics.copy(),
                metrics_after=current_metrics,
                recommendation="Retreinar modelo com dados recentes"
            )

        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtém resumo de performance"""
        return {
            "current_metrics": {
                "accuracy": self.current_accuracy,
                "mse": self.current_mse,
                "sharpe": self.current_sharpe
            },
            "baseline_metrics": self.baseline_metrics,
            "samples_tracked": len(self.predictions_history),
            "window_size": self.window_size,
            "last_update": self.timestamps[-1].isoformat() if self.timestamps else None
        }

class RetrainingScheduler:
    """Agendador de retreinamento de modelos"""

    def __init__(self, ensemble: AdvancedModelEnsemble):
        self.ensemble = ensemble

        # Configurações de retreinamento
        self.min_samples_for_retrain = 500
        self.max_time_without_retrain = timedelta(hours=24)
        self.performance_threshold = 0.1  # 10% degradação

        # Estado
        self.last_retrain_time = datetime.now()
        self.pending_samples = []
        self.is_retraining = False

        # Scheduler task
        self.scheduler_task = None

    async def start_scheduler(self):
        """Inicia o agendador"""
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Agendador de retreinamento iniciado")

    async def stop_scheduler(self):
        """Para o agendador"""
        if self.scheduler_task:
            self.scheduler_task.cancel()
        logger.info("Agendador de retreinamento parado")

    async def _scheduler_loop(self):
        """Loop principal do agendador"""
        while True:
            try:
                await self._check_retrain_conditions()
                await asyncio.sleep(300)  # Verificar a cada 5 minutos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no agendador: {e}")
                await asyncio.sleep(60)

    async def _check_retrain_conditions(self):
        """Verifica condições para retreinamento"""
        if self.is_retraining:
            return

        should_retrain = False
        retrain_reason = ""

        # Verificar tempo desde último retreinamento
        time_since_retrain = datetime.now() - self.last_retrain_time
        if time_since_retrain > self.max_time_without_retrain:
            should_retrain = True
            retrain_reason = "Tempo máximo atingido"

        # Verificar quantidade de amostras pendentes
        if len(self.pending_samples) >= self.min_samples_for_retrain:
            should_retrain = True
            retrain_reason = "Amostras suficientes acumuladas"

        if should_retrain:
            logger.info(f"Iniciando retreinamento: {retrain_reason}")
            await self._trigger_retrain()

    async def _trigger_retrain(self):
        """Dispara retreinamento"""
        self.is_retraining = True

        try:
            # Adicionar amostras pendentes ao ensemble
            for features, target in self.pending_samples:
                await self.ensemble.add_training_data(features, target)

            # Retreinar ensemble
            success = await self.ensemble.train_ensemble()

            if success:
                logger.info("Retreinamento concluído com sucesso")
                self.last_retrain_time = datetime.now()
                self.pending_samples.clear()
            else:
                logger.warning("Retreinamento falhou")

        except Exception as e:
            logger.error(f"Erro durante retreinamento: {e}")
        finally:
            self.is_retraining = False

    async def add_training_sample(self, features: np.ndarray, target: float):
        """Adiciona amostra para retreinamento futuro"""
        self.pending_samples.append((features, target))

        # Limitar tamanho da fila
        if len(self.pending_samples) > self.min_samples_for_retrain * 2:
            self.pending_samples = self.pending_samples[-self.min_samples_for_retrain:]

class RealTimeLearningSystem:
    """Sistema completo de aprendizado em tempo real"""

    def __init__(self, ensemble: AdvancedModelEnsemble):
        self.ensemble = ensemble

        # Componentes
        self.online_learner = OnlineLearner()
        self.performance_tracker = ModelPerformanceTracker()
        self.retrain_scheduler = RetrainingScheduler(ensemble)

        # Estado
        self.learning_events: List[LearningEvent] = []
        self.adaptation_metrics = AdaptationMetrics(
            total_updates=0,
            successful_adaptations=0,
            drift_detections=0,
            average_learning_time=0.0,
            performance_improvement=0.0,
            data_efficiency=0.0,
            adaptation_rate=0.0
        )

        # Configurações
        self.learning_mode = LearningMode.ONLINE
        self.min_confidence_for_learning = 0.5
        self.learning_rate_decay = 0.999

        # Threading
        self.learning_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        logger.info("RealTimeLearningSystem inicializado")

    async def initialize(self):
        """Inicializa o sistema"""
        await self.retrain_scheduler.start_scheduler()
        logger.info("Sistema de aprendizado em tempo real iniciado")

    async def learn_from_trade_result(self, trade_result: RealTradeResult,
                                    prediction: EnsemblePrediction,
                                    market_features: np.ndarray):
        """Aprende com resultado de trade real"""
        try:
            start_time = datetime.now()

            # Converter resultado de trade para target
            target = self._convert_trade_result_to_target(trade_result)

            # Verificar se deve aprender com esta amostra
            if not self._should_learn_from_sample(prediction, target):
                return

            # Aprendizado online
            if self.learning_mode in [LearningMode.ONLINE, LearningMode.ACTIVE]:
                await self._online_learning_step(market_features, target, prediction)

            # Adicionar para retreinamento futuro
            await self.retrain_scheduler.add_training_sample(market_features, target)

            # Atualizar tracker de performance
            await self.performance_tracker.add_prediction(prediction.final_prediction, target)

            # Atualizar performance do ensemble
            await self.ensemble.update_performance(prediction, target)

            # Registrar evento de aprendizado
            learning_time = (datetime.now() - start_time).total_seconds()
            await self._record_learning_event(market_features, target, prediction, learning_time)

            # Verificar drift de performance
            drift = self.performance_tracker.detect_performance_degradation()
            if drift:
                await self._handle_performance_drift(drift)

            logger.debug(f"Aprendizado concluído em {learning_time:.3f}s")

        except Exception as e:
            logger.error(f"Erro no aprendizado: {e}")

    def _convert_trade_result_to_target(self, trade_result: RealTradeResult) -> float:
        """Converte resultado de trade para target de aprendizado"""
        if trade_result.profit_loss is None:
            return 0.5  # Neutro se não há P&L

        # Normalizar P&L para [0, 1]
        # Positivo = 1, Negativo = 0, Neutro = 0.5
        if trade_result.profit_loss > 0:
            return min(0.5 + (trade_result.profit_loss / trade_result.amount) * 0.5, 1.0)
        elif trade_result.profit_loss < 0:
            return max(0.5 + (trade_result.profit_loss / trade_result.amount) * 0.5, 0.0)
        else:
            return 0.5

    def _should_learn_from_sample(self, prediction: EnsemblePrediction, target: float) -> bool:
        """Determina se deve aprender com esta amostra"""
        # Não aprender com predições de baixa confiança
        if prediction.confidence < self.min_confidence_for_learning:
            return False

        # Aprender sempre com resultados extremos (muito bons ou ruins)
        if target <= 0.2 or target >= 0.8:
            return True

        # Para casos intermediários, aprender com probabilidade baseada na confiança
        return np.random.random() < prediction.confidence

    async def _online_learning_step(self, features: np.ndarray, target: float,
                                  prediction: EnsemblePrediction):
        """Executa passo de aprendizado online"""
        try:
            # Converter features para dict (necessário para River)
            features_dict = {f"feature_{i}": float(features.flatten()[i])
                           for i in range(len(features.flatten()))}

            # Aprendizado online
            online_prediction = await self.online_learner.learn_sample(features_dict, target)

            # Atualizar métricas de adaptação
            self.adaptation_metrics.total_updates += 1

            # Verificar se houve melhoria
            ensemble_error = abs(prediction.final_prediction - target)
            online_error = abs(online_prediction - target)

            if online_error < ensemble_error:
                self.adaptation_metrics.successful_adaptations += 1

        except Exception as e:
            logger.error(f"Erro no aprendizado online: {e}")

    async def _record_learning_event(self, features: np.ndarray, target: float,
                                   prediction: EnsemblePrediction, learning_time: float):
        """Registra evento de aprendizado"""
        event = LearningEvent(
            timestamp=datetime.now(),
            features=features,
            target=target,
            prediction=prediction.final_prediction,
            loss=abs(prediction.final_prediction - target),
            learning_rate=self.online_learner.learning_rate,
            model_updated=True
        )

        self.learning_events.append(event)

        # Manter apenas últimos 1000 eventos
        if len(self.learning_events) > 1000:
            self.learning_events = self.learning_events[-800:]

        # Atualizar métricas
        self.adaptation_metrics.average_learning_time = (
            self.adaptation_metrics.average_learning_time * 0.9 + learning_time * 0.1
        )

    async def _handle_performance_drift(self, drift: PerformanceDrift):
        """Lida com drift de performance detectado"""
        logger.warning(f"Drift de performance detectado: {drift.drift_type}, severidade: {drift.severity:.2f}")

        self.adaptation_metrics.drift_detections += 1

        # Ações baseadas na severidade
        if drift.severity > 0.3:  # Drift severo
            logger.info("Disparando retreinamento imediato devido a drift severo")
            await self.retrain_scheduler._trigger_retrain()

        elif drift.severity > 0.15:  # Drift moderado
            # Aumentar taxa de aprendizado online
            self.online_learner.learning_rate *= 1.5
            logger.info("Taxa de aprendizado online aumentada")

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Obtém insights do sistema de aprendizado"""
        # Calcular métricas derivadas
        if self.adaptation_metrics.total_updates > 0:
            self.adaptation_metrics.adaptation_rate = (
                self.adaptation_metrics.successful_adaptations /
                self.adaptation_metrics.total_updates
            )

        # Eficiência de dados (aprendizado por amostra)
        if len(self.learning_events) > 0:
            recent_losses = [event.loss for event in self.learning_events[-100:]]
            if len(recent_losses) > 1:
                self.adaptation_metrics.data_efficiency = 1.0 - np.mean(recent_losses)

        return {
            "adaptation_metrics": asdict(self.adaptation_metrics),
            "online_learner_metrics": self.online_learner.get_metrics(),
            "performance_summary": self.performance_tracker.get_performance_summary(),
            "recent_learning_events": len(self.learning_events),
            "learning_mode": self.learning_mode.value,
            "is_retraining": self.retrain_scheduler.is_retraining,
            "pending_samples": len(self.retrain_scheduler.pending_samples)
        }

    async def set_learning_mode(self, mode: LearningMode):
        """Define modo de aprendizado"""
        self.learning_mode = mode
        logger.info(f"Modo de aprendizado alterado para: {mode.value}")

    async def trigger_manual_retrain(self):
        """Dispara retreinamento manual"""
        logger.info("Retreinamento manual solicitado")
        await self.retrain_scheduler._trigger_retrain()

    async def export_learning_data(self, filepath: str):
        """Exporta dados de aprendizado"""
        try:
            export_data = {
                "learning_events": [asdict(event) for event in self.learning_events],
                "adaptation_metrics": asdict(self.adaptation_metrics),
                "online_metrics": self.online_learner.get_metrics(),
                "performance_data": self.performance_tracker.get_performance_summary()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, default=str, indent=2)

            logger.info(f"Dados de aprendizado exportados para: {filepath}")

        except Exception as e:
            logger.error(f"Erro ao exportar dados: {e}")

    async def shutdown(self):
        """Encerra o sistema"""
        await self.retrain_scheduler.stop_scheduler()
        self.thread_pool.shutdown(wait=True)
        logger.info("RealTimeLearningSystem encerrado")
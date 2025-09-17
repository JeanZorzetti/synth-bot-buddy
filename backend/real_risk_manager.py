"""
Real Risk Manager - Gerenciamento de Risco com Capital Real
Sistema avançado de gestão de risco para trading automatizado com capital real
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math
from scipy import stats

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem
from real_position_manager import RealPositionManager, Position, PositionType

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskAlertType(Enum):
    ACCOUNT_BALANCE = "account_balance"
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MARGIN = "margin"
    DAILY_LOSS = "daily_loss"
    CONCENTRATION = "concentration"

@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_daily_loss_pct: float = 5.0  # 5% máximo por dia
    max_position_size_pct: float = 10.0  # 10% do capital por posição
    max_portfolio_risk_pct: float = 20.0  # 20% de risco total
    max_correlation: float = 0.7  # Correlação máxima entre posições
    max_drawdown_pct: float = 15.0  # 15% de drawdown máximo
    min_account_balance: float = 1000.0  # Balance mínimo
    max_leverage: float = 100.0  # Leverage máximo
    max_positions_per_symbol: int = 3
    max_total_positions: int = 20
    var_confidence_level: float = 0.95  # 95% VaR
    stress_test_scenarios: int = 1000

@dataclass
class RiskMetrics:
    """Métricas de risco calculadas"""
    current_drawdown_pct: float
    daily_pnl_pct: float
    portfolio_var: float
    portfolio_volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    max_correlation: float
    concentration_risk: float
    margin_utilization_pct: float
    risk_level: RiskLevel
    timestamp: datetime

@dataclass
class RiskAlert:
    """Alerta de risco"""
    alert_type: RiskAlertType
    risk_level: RiskLevel
    message: str
    current_value: float
    limit_value: float
    recommended_action: str
    timestamp: datetime
    symbol: Optional[str] = None

class PositionSizer:
    """Calculador de tamanho de posição"""

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calcula tamanho ótimo da posição usando Kelly Criterion"""
        if avg_loss <= 0 or win_rate <= 0:
            return 0.01  # 1% mínimo

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        return max(0.01, min(kelly_fraction, 0.25))  # Entre 1% e 25%

    @staticmethod
    def fixed_fractional(risk_per_trade: float = 0.02) -> float:
        """Método de fração fixa"""
        return max(0.005, min(risk_per_trade, 0.05))  # Entre 0.5% e 5%

    @staticmethod
    def volatility_adjusted(base_size: float, volatility: float, target_vol: float = 0.15) -> float:
        """Ajusta tamanho baseado na volatilidade"""
        if volatility <= 0:
            return base_size

        adjustment = target_vol / volatility
        return base_size * max(0.1, min(adjustment, 3.0))

    @staticmethod
    def correlation_adjusted(base_size: float, correlation: float, max_correlation: float = 0.7) -> float:
        """Ajusta tamanho baseado na correlação com portfólio"""
        if correlation <= max_correlation:
            return base_size

        reduction_factor = 1 - (correlation - max_correlation) / (1 - max_correlation)
        return base_size * max(0.1, reduction_factor)

class RealRiskManager:
    """Gerenciador de risco para capital real"""

    def __init__(self, initial_capital: float = 10000.0):
        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()
        self.position_manager = None  # Será definido na inicialização

        # Capital e configurações
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.available_margin = initial_capital
        self.used_margin = 0.0

        # Limites de risco
        self.risk_limits = RiskLimits()

        # Estado do sistema
        self.is_monitoring = False
        self.risk_monitoring_task = None
        self.daily_pnl = 0.0
        self.peak_balance = initial_capital
        self.current_drawdown = 0.0

        # Métricas históricas
        self.balance_history = [initial_capital]
        self.pnl_history = []
        self.drawdown_history = []
        self.risk_metrics_history = []

        # Alertas ativos
        self.active_alerts = []
        self.alert_history = []

        # Performance tracking
        self.daily_returns = []
        self.monthly_returns = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0

        # Estado de emergência
        self.emergency_stop = False
        self.emergency_reason = None

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self, position_manager: RealPositionManager):
        """Inicializa o gerenciador de risco"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.logger.initialize()

            self.position_manager = position_manager

            # Carregar dados históricos
            await self._load_historical_data()

            await self.logger.log_activity("risk_manager_initialized", {
                "initial_capital": self.initial_capital,
                "risk_limits": asdict(self.risk_limits)
            })

            print("✅ Risk Manager inicializado com sucesso")

        except Exception as e:
            await self.logger.log_error("risk_manager_init_error", str(e))
            raise

    async def start_monitoring(self):
        """Inicia monitoramento de risco"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.risk_monitoring_task = asyncio.create_task(self._risk_monitoring_loop())

        await self.logger.log_activity("risk_monitoring_started", {})
        print("🛡️ Monitoramento de risco iniciado")

    async def stop_monitoring(self):
        """Para o monitoramento de risco"""
        self.is_monitoring = False

        if self.risk_monitoring_task:
            self.risk_monitoring_task.cancel()

        await self.logger.log_activity("risk_monitoring_stopped", {})
        print("⏹️ Monitoramento de risco parado")

    async def validate_new_position(
        self,
        symbol: str,
        position_type: PositionType,
        amount: float,
        entry_price: float
    ) -> Tuple[bool, Optional[str], float]:
        """Valida se uma nova posição pode ser aberta"""
        try:
            # 1. Verificar parada de emergência
            if self.emergency_stop:
                return False, f"Trading em emergência: {self.emergency_reason}", 0.0

            # 2. Verificar balance mínimo
            if self.current_balance < self.risk_limits.min_account_balance:
                return False, "Balance abaixo do mínimo permitido", 0.0

            # 3. Verificar perda diária máxima
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100
            if self.daily_pnl < 0 and daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
                return False, f"Perda diária máxima atingida: {daily_loss_pct:.1f}%", 0.0

            # 4. Calcular tamanho sugerido da posição
            suggested_amount = await self._calculate_optimal_position_size(
                symbol, position_type, amount, entry_price
            )

            # 5. Verificar tamanho máximo da posição
            position_value = suggested_amount * entry_price
            position_pct = position_value / self.current_balance * 100

            if position_pct > self.risk_limits.max_position_size_pct:
                max_amount = (self.current_balance * self.risk_limits.max_position_size_pct / 100) / entry_price
                suggested_amount = max_amount

            # 6. Verificar margem disponível
            required_margin = position_value / self.risk_limits.max_leverage
            if required_margin > self.available_margin:
                return False, "Margem insuficiente", 0.0

            # 7. Verificar número máximo de posições
            if not await self._check_position_limits(symbol):
                return False, "Limite de posições atingido", 0.0

            # 8. Verificar correlação
            correlation_risk = await self._calculate_correlation_risk(symbol, suggested_amount)
            if correlation_risk > self.risk_limits.max_correlation:
                # Reduzir tamanho baseado na correlação
                suggested_amount = PositionSizer.correlation_adjusted(
                    suggested_amount, correlation_risk, self.risk_limits.max_correlation
                )

            # 9. Verificar drawdown atual
            if self.current_drawdown >= self.risk_limits.max_drawdown_pct:
                # Reduzir tamanho em situação de drawdown
                suggested_amount *= 0.5

            return True, None, suggested_amount

        except Exception as e:
            await self.logger.log_error("position_validation_error", f"{symbol}: {str(e)}")
            return False, "Erro na validação de risco", 0.0

    async def _calculate_optimal_position_size(
        self,
        symbol: str,
        position_type: PositionType,
        requested_amount: float,
        entry_price: float
    ) -> float:
        """Calcula tamanho ótimo da posição"""
        try:
            # 1. Tamanho base usando Kelly Criterion
            win_rate = self.winning_trades / max(self.total_trades, 1)
            kelly_size = PositionSizer.kelly_criterion(win_rate, self.avg_win, abs(self.avg_loss))
            base_amount = self.current_balance * kelly_size / entry_price

            # 2. Ajustar por volatilidade
            volatility = await self._get_symbol_volatility(symbol)
            vol_adjusted_amount = PositionSizer.volatility_adjusted(base_amount, volatility)

            # 3. Ajustar por fração fixa (backup)
            fixed_fraction_amount = self.current_balance * 0.02 / entry_price  # 2%

            # 4. Usar o menor entre os métodos
            calculated_amount = min(vol_adjusted_amount, fixed_fraction_amount, requested_amount)

            # 5. Limitar ao máximo permitido
            max_position_value = self.current_balance * self.risk_limits.max_position_size_pct / 100
            max_amount = max_position_value / entry_price

            return min(calculated_amount, max_amount)

        except Exception as e:
            await self.logger.log_error("position_size_calculation_error", f"{symbol}: {str(e)}")
            # Fallback para 1% do capital
            return self.current_balance * 0.01 / entry_price

    async def _risk_monitoring_loop(self):
        """Loop principal de monitoramento de risco"""
        while self.is_monitoring:
            try:
                # Atualizar métricas de risco
                await self._update_risk_metrics()

                # Verificar alertas
                await self._check_risk_alerts()

                # Atualizar balance e margem
                await self._update_account_metrics()

                # Verificar condições de emergência
                await self._check_emergency_conditions()

                await asyncio.sleep(10)  # A cada 10 segundos

            except Exception as e:
                await self.logger.log_error("risk_monitoring_error", str(e))
                await asyncio.sleep(30)

    async def _update_risk_metrics(self):
        """Atualiza métricas de risco"""
        try:
            if not self.position_manager:
                return

            # Obter resumo das posições
            portfolio_summary = await self.position_manager.get_position_summary()

            # Atualizar PnL não realizado
            unrealized_pnl = portfolio_summary.get("total_unrealized_pnl", 0.0)

            # Calcular balance atual
            self.current_balance = self.initial_capital + self.daily_pnl + unrealized_pnl

            # Atualizar peak e drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance

            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100

            # Calcular métricas avançadas
            portfolio_var = await self._calculate_portfolio_var()
            portfolio_volatility = await self._calculate_portfolio_volatility()
            sharpe_ratio = await self._calculate_sharpe_ratio()
            calmar_ratio = await self._calculate_calmar_ratio()
            max_correlation = await self._calculate_max_correlation()
            concentration_risk = await self._calculate_concentration_risk()
            margin_utilization = (self.used_margin / self.available_margin * 100) if self.available_margin > 0 else 0

            # Determinar nível de risco
            risk_level = self._determine_risk_level()

            # Criar métricas
            metrics = RiskMetrics(
                current_drawdown_pct=self.current_drawdown,
                daily_pnl_pct=self.daily_pnl / self.initial_capital * 100,
                portfolio_var=portfolio_var,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_correlation=max_correlation,
                concentration_risk=concentration_risk,
                margin_utilization_pct=margin_utilization,
                risk_level=risk_level,
                timestamp=datetime.utcnow()
            )

            # Armazenar métricas
            self.risk_metrics_history.append(metrics)

            # Manter apenas últimas 1000 entradas
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]

            # Cache das métricas
            await self.cache_manager.set(
                CacheNamespace.USER_SESSIONS,
                "current_risk_metrics",
                asdict(metrics),
                ttl=60
            )

        except Exception as e:
            await self.logger.log_error("risk_metrics_update_error", str(e))

    async def _check_risk_alerts(self):
        """Verifica e gera alertas de risco"""
        try:
            new_alerts = []

            # 1. Verificar perda diária
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100
            if self.daily_pnl < 0 and daily_loss_pct >= self.risk_limits.max_daily_loss_pct * 0.8:
                alert = RiskAlert(
                    alert_type=RiskAlertType.DAILY_LOSS,
                    risk_level=RiskLevel.HIGH if daily_loss_pct >= self.risk_limits.max_daily_loss_pct else RiskLevel.MODERATE,
                    message=f"Perda diária próxima do limite: {daily_loss_pct:.1f}%",
                    current_value=daily_loss_pct,
                    limit_value=self.risk_limits.max_daily_loss_pct,
                    recommended_action="Considere reduzir exposição ou parar trading",
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)

            # 2. Verificar drawdown
            if self.current_drawdown >= self.risk_limits.max_drawdown_pct * 0.7:
                alert = RiskAlert(
                    alert_type=RiskAlertType.DRAWDOWN,
                    risk_level=RiskLevel.CRITICAL if self.current_drawdown >= self.risk_limits.max_drawdown_pct else RiskLevel.HIGH,
                    message=f"Drawdown elevado: {self.current_drawdown:.1f}%",
                    current_value=self.current_drawdown,
                    limit_value=self.risk_limits.max_drawdown_pct,
                    recommended_action="Revisar estratégia e reduzir risco",
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)

            # 3. Verificar balance mínimo
            if self.current_balance <= self.risk_limits.min_account_balance * 1.2:
                alert = RiskAlert(
                    alert_type=RiskAlertType.ACCOUNT_BALANCE,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Balance próximo do mínimo: ${self.current_balance:.2f}",
                    current_value=self.current_balance,
                    limit_value=self.risk_limits.min_account_balance,
                    recommended_action="Depositar fundos ou parar trading",
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)

            # 4. Verificar utilização de margem
            margin_utilization = (self.used_margin / self.available_margin * 100) if self.available_margin > 0 else 0
            if margin_utilization >= 80:
                alert = RiskAlert(
                    alert_type=RiskAlertType.MARGIN,
                    risk_level=RiskLevel.HIGH if margin_utilization >= 90 else RiskLevel.MODERATE,
                    message=f"Alta utilização de margem: {margin_utilization:.1f}%",
                    current_value=margin_utilization,
                    limit_value=100.0,
                    recommended_action="Fechar algumas posições para liberar margem",
                    timestamp=datetime.utcnow()
                )
                new_alerts.append(alert)

            # Adicionar novos alertas
            for alert in new_alerts:
                if not self._alert_already_active(alert):
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)

                    # Log do alerta
                    await self.logger.log_activity("risk_alert_generated", {
                        "alert_type": alert.alert_type.value,
                        "risk_level": alert.risk_level.value,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "limit_value": alert.limit_value
                    })

            # Limpar alertas antigos (mais de 1 hora)
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self.active_alerts = [a for a in self.active_alerts if a.timestamp > cutoff_time]

        except Exception as e:
            await self.logger.log_error("risk_alerts_check_error", str(e))

    def _alert_already_active(self, new_alert: RiskAlert) -> bool:
        """Verifica se alerta similar já está ativo"""
        for active_alert in self.active_alerts:
            if (active_alert.alert_type == new_alert.alert_type and
                active_alert.symbol == new_alert.symbol):
                return True
        return False

    async def _check_emergency_conditions(self):
        """Verifica condições de parada de emergência"""
        try:
            should_stop = False
            reason = None

            # 1. Perda diária crítica
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100
            if self.daily_pnl < 0 and daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
                should_stop = True
                reason = f"Perda diária máxima atingida: {daily_loss_pct:.1f}%"

            # 2. Drawdown crítico
            if self.current_drawdown >= self.risk_limits.max_drawdown_pct:
                should_stop = True
                reason = f"Drawdown máximo atingido: {self.current_drawdown:.1f}%"

            # 3. Balance crítico
            if self.current_balance <= self.risk_limits.min_account_balance:
                should_stop = True
                reason = f"Balance abaixo do mínimo: ${self.current_balance:.2f}"

            # 4. Margem esgotada
            if self.available_margin <= 0:
                should_stop = True
                reason = "Margem esgotada"

            if should_stop and not self.emergency_stop:
                await self._trigger_emergency_stop(reason)

        except Exception as e:
            await self.logger.log_error("emergency_conditions_check_error", str(e))

    async def _trigger_emergency_stop(self, reason: str):
        """Aciona parada de emergência"""
        try:
            self.emergency_stop = True
            self.emergency_reason = reason

            # Fechar todas as posições
            if self.position_manager:
                closed_positions = await self.position_manager.force_close_all_positions()

            # Alerta crítico
            emergency_alert = RiskAlert(
                alert_type=RiskAlertType.ACCOUNT_BALANCE,
                risk_level=RiskLevel.EMERGENCY,
                message=f"PARADA DE EMERGÊNCIA: {reason}",
                current_value=0.0,
                limit_value=0.0,
                recommended_action="Trading interrompido. Revisar estratégia.",
                timestamp=datetime.utcnow()
            )

            self.active_alerts.append(emergency_alert)

            # Log da emergência
            await self.logger.log_activity("emergency_stop_triggered", {
                "reason": reason,
                "current_balance": self.current_balance,
                "current_drawdown": self.current_drawdown,
                "daily_pnl": self.daily_pnl,
                "positions_closed": closed_positions if 'closed_positions' in locals() else 0
            })

            print(f"🚨 EMERGÊNCIA: {reason}")

        except Exception as e:
            await self.logger.log_error("emergency_stop_trigger_error", str(e))

    def _determine_risk_level(self) -> RiskLevel:
        """Determina nível de risco geral"""
        if self.emergency_stop:
            return RiskLevel.EMERGENCY

        # Pontuação de risco baseada em múltiplos fatores
        risk_score = 0

        # Drawdown (peso 30%)
        drawdown_score = (self.current_drawdown / self.risk_limits.max_drawdown_pct) * 30

        # Perda diária (peso 25%)
        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100 if self.daily_pnl < 0 else 0
        daily_loss_score = (daily_loss_pct / self.risk_limits.max_daily_loss_pct) * 25

        # Utilização de margem (peso 20%)
        margin_util = (self.used_margin / self.available_margin * 100) if self.available_margin > 0 else 0
        margin_score = (margin_util / 100) * 20

        # Balance (peso 15%)
        balance_ratio = self.current_balance / self.initial_capital
        balance_score = (1 - balance_ratio) * 15 if balance_ratio < 1 else 0

        # Volatilidade (peso 10%)
        volatility_score = 0  # Simplificado

        risk_score = drawdown_score + daily_loss_score + margin_score + balance_score + volatility_score

        if risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    async def get_risk_summary(self) -> Dict[str, Any]:
        """Retorna resumo do estado de risco"""
        try:
            current_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None

            return {
                "account_status": {
                    "current_balance": self.current_balance,
                    "initial_capital": self.initial_capital,
                    "daily_pnl": self.daily_pnl,
                    "daily_pnl_pct": self.daily_pnl / self.initial_capital * 100,
                    "available_margin": self.available_margin,
                    "used_margin": self.used_margin,
                    "margin_utilization_pct": (self.used_margin / self.available_margin * 100) if self.available_margin > 0 else 0
                },
                "risk_metrics": {
                    "current_drawdown_pct": self.current_drawdown,
                    "peak_balance": self.peak_balance,
                    "risk_level": current_metrics.risk_level.value if current_metrics else "unknown",
                    "portfolio_var": current_metrics.portfolio_var if current_metrics else 0.0,
                    "sharpe_ratio": current_metrics.sharpe_ratio if current_metrics else 0.0
                },
                "limits": asdict(self.risk_limits),
                "active_alerts": [asdict(alert) for alert in self.active_alerts],
                "emergency_status": {
                    "emergency_stop": self.emergency_stop,
                    "emergency_reason": self.emergency_reason
                },
                "performance": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "win_rate": (self.winning_trades / max(self.total_trades, 1)) * 100,
                    "avg_win": self.avg_win,
                    "avg_loss": self.avg_loss
                }
            }

        except Exception as e:
            await self.logger.log_error("risk_summary_error", str(e))
            return {}

    # Métodos auxiliares para cálculos avançados
    async def _calculate_portfolio_var(self) -> float:
        """Calcula Value at Risk do portfólio"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0

            returns = np.array(self.daily_returns[-30:])
            var_percentile = (1 - self.risk_limits.var_confidence_level) * 100
            var = np.percentile(returns, var_percentile)
            return abs(var) * self.current_balance

        except Exception:
            return 0.0

    async def _calculate_portfolio_volatility(self) -> float:
        """Calcula volatilidade do portfólio"""
        try:
            if len(self.daily_returns) < 10:
                return 0.0

            returns = np.array(self.daily_returns[-30:])
            return np.std(returns) * np.sqrt(252)  # Anualizada

        except Exception:
            return 0.0

    async def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe ratio"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0

            returns = np.array(self.daily_returns[-30:])
            excess_returns = returns - 0.02/252  # Risk-free rate 2% anual
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0

        except Exception:
            return 0.0

    async def _calculate_calmar_ratio(self) -> float:
        """Calcula Calmar ratio"""
        try:
            if len(self.daily_returns) < 30 or self.current_drawdown == 0:
                return 0.0

            annual_return = np.mean(self.daily_returns[-30:]) * 252
            return annual_return / (self.current_drawdown / 100)

        except Exception:
            return 0.0

    async def _calculate_max_correlation(self) -> float:
        """Calcula correlação máxima entre posições"""
        # Implementação simplificada
        return 0.0

    async def _calculate_concentration_risk(self) -> float:
        """Calcula risco de concentração"""
        # Implementação simplificada
        return 0.0

    async def _calculate_correlation_risk(self, symbol: str, amount: float) -> float:
        """Calcula risco de correlação para nova posição"""
        # Implementação simplificada
        return 0.3

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Obtém volatilidade do símbolo"""
        try:
            # Tentar obter do cache
            cached_vol = await self.cache_manager.get(
                CacheNamespace.MARKET_DATA,
                f"{symbol}:volatility"
            )

            if cached_vol:
                return float(cached_vol)

            # Volatilidade padrão por tipo de ativo
            volatility_map = {
                "R_": 0.15,  # Volatility indices
                "FRX": 0.08,  # Forex
                "cry": 0.25,  # Crypto
            }

            for prefix, vol in volatility_map.items():
                if symbol.startswith(prefix):
                    return vol

            return 0.12  # Padrão

        except Exception:
            return 0.12

    async def _check_position_limits(self, symbol: str) -> bool:
        """Verifica limites de posições"""
        if not self.position_manager:
            return True

        summary = await self.position_manager.get_position_summary()

        # Verificar total de posições
        if summary.get("active_positions_count", 0) >= self.risk_limits.max_total_positions:
            return False

        # Verificar posições por símbolo
        symbols_count = summary.get("symbols_count", {})
        if symbols_count.get(symbol, 0) >= self.risk_limits.max_positions_per_symbol:
            return False

        return True

    async def _update_account_metrics(self):
        """Atualiza métricas da conta"""
        try:
            # Simular atualização de margem e balance
            # Em produção, isso viria da API do broker

            if self.position_manager:
                summary = await self.position_manager.get_position_summary()
                self.used_margin = summary.get("total_margin_used", 0.0)
                unrealized_pnl = summary.get("total_unrealized_pnl", 0.0)

                # Atualizar balance atual
                self.current_balance = self.initial_capital + self.daily_pnl + unrealized_pnl
                self.available_margin = max(0, self.current_balance - self.used_margin)

        except Exception as e:
            await self.logger.log_error("account_metrics_update_error", str(e))

    async def _load_historical_data(self):
        """Carrega dados históricos"""
        try:
            # Carregar do cache ou banco de dados
            # Implementação simplificada
            pass
        except Exception as e:
            await self.logger.log_error("historical_data_load_error", str(e))

    async def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Atualiza métricas de performance após um trade"""
        try:
            self.total_trades += 1
            pnl = trade_result.get("realized_pnl", 0.0)
            self.daily_pnl += pnl

            if pnl > 0:
                self.winning_trades += 1
                self.avg_win = (self.avg_win * (self.winning_trades - 1) + pnl) / self.winning_trades
            else:
                self.losing_trades += 1
                self.avg_loss = (self.avg_loss * (self.losing_trades - 1) + abs(pnl)) / self.losing_trades

            # Adicionar retorno diário
            daily_return = pnl / self.current_balance
            self.daily_returns.append(daily_return)

            # Manter apenas últimos 252 dias (1 ano)
            if len(self.daily_returns) > 252:
                self.daily_returns = self.daily_returns[-252:]

            await self.logger.log_activity("performance_updated", {
                "total_trades": self.total_trades,
                "win_rate": (self.winning_trades / self.total_trades) * 100,
                "daily_pnl": self.daily_pnl
            })

        except Exception as e:
            await self.logger.log_error("performance_update_error", str(e))

    async def reset_emergency_stop(self):
        """Reset do estado de emergência (apenas com confirmação manual)"""
        self.emergency_stop = False
        self.emergency_reason = None
        await self.logger.log_activity("emergency_stop_reset", {})
        print("✅ Estado de emergência resetado")

    async def shutdown(self):
        """Encerra o gerenciador de risco"""
        await self.stop_monitoring()
        await self.logger.log_activity("risk_manager_shutdown", {})
        print("🔌 Risk Manager encerrado")
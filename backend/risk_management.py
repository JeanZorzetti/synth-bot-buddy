"""
🛡️ RISK MANAGEMENT SYSTEM
Sistema inteligente de gestão de risco para trading autônomo
Protege o capital e otimiza exposição baseado em IA/ML
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np

from autonomous_trading_engine import TradingDecision, Position, PortfolioState, TradeAction
from prediction_engine import TradingSignal


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    CRITICAL = "CRITICAL"


class RiskAction(Enum):
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK = "BLOCK"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class RiskMetrics:
    """Métricas de risco do portfólio"""
    current_drawdown: float
    max_drawdown: float
    daily_var: float  # Value at Risk diário
    sharpe_ratio: float
    volatility: float
    concentration_risk: float
    correlation_risk: float
    leverage_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_var': self.daily_var,
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'leverage_ratio': self.leverage_ratio
        }


@dataclass
class RiskAssessment:
    """Avaliação de risco para uma operação"""
    risk_level: RiskLevel
    risk_score: float
    recommended_action: RiskAction
    position_size_adjustment: float
    reasons: List[str]
    max_allowed_size: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'recommended_action': self.recommended_action.value,
            'position_size_adjustment': self.position_size_adjustment,
            'reasons': self.reasons,
            'max_allowed_size': self.max_allowed_size
        }


@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_daily_loss: float = 0.05  # 5%
    max_drawdown: float = 0.15   # 15%
    max_position_size: float = 0.10  # 10%
    max_correlation: float = 0.70   # 70%
    max_concentration: float = 0.30  # 30% em um ativo
    max_leverage: float = 2.0      # 2x
    max_daily_trades: int = 100
    min_confidence: float = 0.70   # 70%
    var_limit: float = 0.03       # 3% VaR diário


class IntelligentRiskManager:
    """
    🛡️ Gerenciador Inteligente de Risco

    Funcionalidades:
    1. Análise de risco em tempo real
    2. Ajuste dinâmico de tamanho de posição
    3. Monitoramento de drawdown
    4. Correlação entre ativos
    5. Paradas de emergência automáticas
    6. Value at Risk (VaR) calculations
    7. Concentration risk analysis
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 risk_limits: Optional[RiskLimits] = None):

        self.logger = logging.getLogger(__name__)

        # Configurações
        self.initial_balance = initial_balance
        self.risk_limits = risk_limits or RiskLimits()

        # Estado do sistema
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.now().date()

        # Histórico para análises
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []

        # Estado de emergência
        self.emergency_mode = False
        self.risk_breaches = 0

        # Correlações entre ativos
        self.asset_correlations: Dict[str, Dict[str, float]] = {}
        self.asset_exposures: Dict[str, float] = {}

        # Cache de cálculos
        self._metrics_cache: Optional[RiskMetrics] = None
        self._cache_timestamp = datetime.now()

    async def assess_trade_risk(self,
                              decision: TradingDecision,
                              portfolio: PortfolioState,
                              current_signal: TradingSignal) -> RiskAssessment:
        """
        🔍 Avalia risco de uma operação específica

        Retorna recomendação completa de risco para a IA
        """

        reasons = []
        risk_score = 0.0

        # 1. Verificar limites básicos
        basic_checks = await self._check_basic_limits(decision, portfolio)
        risk_score += basic_checks['score']
        reasons.extend(basic_checks['reasons'])

        # 2. Análise de drawdown
        drawdown_check = await self._check_drawdown_risk(portfolio)
        risk_score += drawdown_check['score']
        reasons.extend(drawdown_check['reasons'])

        # 3. Análise de concentração
        concentration_check = await self._check_concentration_risk(decision, portfolio)
        risk_score += concentration_check['score']
        reasons.extend(concentration_check['reasons'])

        # 4. Análise de correlação
        correlation_check = await self._check_correlation_risk(decision, portfolio)
        risk_score += correlation_check['score']
        reasons.extend(correlation_check['reasons'])

        # 5. Análise de volatilidade do mercado
        volatility_check = await self._check_market_volatility(current_signal)
        risk_score += volatility_check['score']
        reasons.extend(volatility_check['reasons'])

        # 6. Análise de frequência de trading
        frequency_check = await self._check_trading_frequency()
        risk_score += frequency_check['score']
        reasons.extend(frequency_check['reasons'])

        # 7. Análise de qualidade do sinal
        signal_check = await self._check_signal_quality(current_signal)
        risk_score += signal_check['score']
        reasons.extend(signal_check['reasons'])

        # Determinar nível de risco
        risk_level = self._calculate_risk_level(risk_score)

        # Determinar ação recomendada
        recommended_action, size_adjustment = self._determine_risk_action(
            risk_level, risk_score, decision
        )

        # Calcular tamanho máximo permitido
        max_allowed_size = await self._calculate_max_position_size(
            decision, portfolio, risk_level
        )

        assessment = RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            recommended_action=recommended_action,
            position_size_adjustment=size_adjustment,
            reasons=reasons,
            max_allowed_size=max_allowed_size
        )

        # Log da avaliação
        self.logger.info(f"🛡️ Avaliação de Risco: {risk_level.value} | "
                        f"Score: {risk_score:.2f} | "
                        f"Ação: {recommended_action.value}")

        if reasons:
            self.logger.info(f"📋 Motivos: {', '.join(reasons[:3])}")

        return assessment

    async def _check_basic_limits(self,
                                decision: TradingDecision,
                                portfolio: PortfolioState) -> Dict[str, Any]:
        """Verifica limites básicos de trading"""

        score = 0.0
        reasons = []

        # Tamanho da posição vs saldo
        position_ratio = decision.position_size / portfolio.total_balance
        if position_ratio > self.risk_limits.max_position_size:
            score += 3.0
            reasons.append(f"Posição muito grande: {position_ratio:.1%} > {self.risk_limits.max_position_size:.1%}")

        # Confiança do sinal
        if decision.confidence < self.risk_limits.min_confidence:
            score += 2.0
            reasons.append(f"Confiança baixa: {decision.confidence:.1%} < {self.risk_limits.min_confidence:.1%}")

        # Saldo disponível
        if portfolio.available_balance < decision.position_size:
            score += 5.0
            reasons.append("Saldo insuficiente")

        # Verificar se já atingiu limite diário de trades
        if self.daily_trades >= self.risk_limits.max_daily_trades:
            score += 4.0
            reasons.append(f"Limite diário de trades atingido: {self.daily_trades}")

        return {'score': score, 'reasons': reasons}

    async def _check_drawdown_risk(self, portfolio: PortfolioState) -> Dict[str, Any]:
        """Verifica risco de drawdown"""

        score = 0.0
        reasons = []

        # Atualizar peak balance
        if portfolio.total_balance > self.peak_balance:
            self.peak_balance = portfolio.total_balance

        # Calcular drawdown atual
        current_drawdown = (self.peak_balance - portfolio.total_balance) / self.peak_balance

        # Verificar drawdown diário
        daily_return = (portfolio.total_balance - self.current_balance) / self.current_balance

        if current_drawdown > self.risk_limits.max_drawdown:
            score += 10.0  # Critical
            reasons.append(f"Drawdown crítico: {current_drawdown:.1%}")
        elif current_drawdown > self.risk_limits.max_drawdown * 0.8:
            score += 5.0
            reasons.append(f"Drawdown elevado: {current_drawdown:.1%}")
        elif current_drawdown > self.risk_limits.max_drawdown * 0.6:
            score += 2.0
            reasons.append(f"Drawdown moderado: {current_drawdown:.1%}")

        # Verificar perda diária
        if daily_return < -self.risk_limits.max_daily_loss:
            score += 6.0
            reasons.append(f"Perda diária excessiva: {daily_return:.1%}")

        return {'score': score, 'reasons': reasons}

    async def _check_concentration_risk(self,
                                      decision: TradingDecision,
                                      portfolio: PortfolioState) -> Dict[str, Any]:
        """Verifica risco de concentração"""

        score = 0.0
        reasons = []

        # Calcular exposição atual por ativo
        symbol_exposure = 0.0
        for position in portfolio.open_positions:
            if position.symbol == decision.symbol:
                symbol_exposure += position.position_size

        # Adicionar nova posição
        total_symbol_exposure = symbol_exposure + decision.position_size
        concentration_ratio = total_symbol_exposure / portfolio.total_balance

        if concentration_ratio > self.risk_limits.max_concentration:
            score += 4.0
            reasons.append(f"Concentração excessiva em {decision.symbol}: {concentration_ratio:.1%}")
        elif concentration_ratio > self.risk_limits.max_concentration * 0.8:
            score += 2.0
            reasons.append(f"Alta concentração em {decision.symbol}: {concentration_ratio:.1%}")

        # Verificar número de posições abertas
        if len(portfolio.open_positions) >= 10:  # Muitas posições
            score += 1.0
            reasons.append(f"Muitas posições abertas: {len(portfolio.open_positions)}")

        return {'score': score, 'reasons': reasons}

    async def _check_correlation_risk(self,
                                    decision: TradingDecision,
                                    portfolio: PortfolioState) -> Dict[str, Any]:
        """Verifica risco de correlação entre ativos"""

        score = 0.0
        reasons = []

        # Calcular correlação com posições existentes
        high_correlation_exposure = 0.0

        for position in portfolio.open_positions:
            correlation = self._get_asset_correlation(decision.symbol, position.symbol)

            if abs(correlation) > self.risk_limits.max_correlation:
                if (decision.action == position.action):  # Mesma direção
                    high_correlation_exposure += position.position_size

        if high_correlation_exposure > portfolio.total_balance * 0.2:  # 20%
            score += 3.0
            reasons.append(f"Alta correlação com posições existentes")

        return {'score': score, 'reasons': reasons}

    async def _check_market_volatility(self, signal: TradingSignal) -> Dict[str, Any]:
        """Verifica risco baseado na volatilidade do mercado"""

        score = 0.0
        reasons = []

        if signal.features:
            volatility = signal.features.price_volatility

            if volatility > 0.05:  # 5% volatilidade
                score += 3.0
                reasons.append(f"Volatilidade extrema: {volatility:.1%}")
            elif volatility > 0.03:  # 3% volatilidade
                score += 1.5
                reasons.append(f"Alta volatilidade: {volatility:.1%}")

            # Verificar momentum anormal
            momentum = abs(signal.features.momentum_5min)
            if momentum > 0.03:  # 3% momentum
                score += 1.0
                reasons.append(f"Momentum anormal detectado")

        return {'score': score, 'reasons': reasons}

    async def _check_trading_frequency(self) -> Dict[str, Any]:
        """Verifica frequência de trading"""

        score = 0.0
        reasons = []

        # Reset diário
        today = datetime.now().date()
        if today > self.last_daily_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_daily_reset = today

        # Verificar frequência
        frequency_ratio = self.daily_trades / self.risk_limits.max_daily_trades

        if frequency_ratio > 0.9:
            score += 2.0
            reasons.append(f"Frequência de trading muito alta: {self.daily_trades}")
        elif frequency_ratio > 0.7:
            score += 1.0
            reasons.append(f"Frequência de trading elevada")

        return {'score': score, 'reasons': reasons}

    async def _check_signal_quality(self, signal: TradingSignal) -> Dict[str, Any]:
        """Verifica qualidade do sinal de trading"""

        score = 0.0
        reasons = []

        # Força do sinal
        if abs(signal.signal_strength) < 0.5:
            score += 2.0
            reasons.append(f"Sinal fraco: {signal.signal_strength:.2f}")

        # Confiança
        if signal.confidence < 0.8:
            score += 1.0
            reasons.append(f"Confiança moderada: {signal.confidence:.1%}")

        # Verificar se o sinal é muito recente (pode ser ruído)
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age < 30:  # Menos de 30 segundos
            score += 0.5
            reasons.append("Sinal muito recente - possível ruído")

        return {'score': score, 'reasons': reasons}

    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Determina nível de risco baseado no score"""

        if risk_score >= 10.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 6.0:
            return RiskLevel.EXTREME
        elif risk_score >= 4.0:
            return RiskLevel.HIGH
        elif risk_score >= 2.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_risk_action(self,
                             risk_level: RiskLevel,
                             risk_score: float,
                             decision: TradingDecision) -> Tuple[RiskAction, float]:
        """Determina ação recomendada baseada no risco"""

        if risk_level == RiskLevel.CRITICAL:
            return RiskAction.EMERGENCY_STOP, 0.0

        elif risk_level == RiskLevel.EXTREME:
            return RiskAction.BLOCK, 0.0

        elif risk_level == RiskLevel.HIGH:
            # Reduzir tamanho baseado no score
            reduction = min(0.8, risk_score / 10.0)  # Até 80% de redução
            return RiskAction.REDUCE_SIZE, 1.0 - reduction

        elif risk_level == RiskLevel.MEDIUM:
            # Redução leve
            reduction = min(0.3, risk_score / 20.0)  # Até 30% de redução
            return RiskAction.REDUCE_SIZE, 1.0 - reduction

        else:  # LOW risk
            return RiskAction.ALLOW, 1.0

    async def _calculate_max_position_size(self,
                                         decision: TradingDecision,
                                         portfolio: PortfolioState,
                                         risk_level: RiskLevel) -> float:
        """Calcula tamanho máximo permitido para posição"""

        # Tamanho base baseado no saldo
        base_size = portfolio.available_balance * self.risk_limits.max_position_size

        # Ajustar baseado no nível de risco
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.EXTREME: 0.1,
            RiskLevel.CRITICAL: 0.0
        }

        max_size = base_size * risk_multipliers[risk_level]

        # Considerar concentração por ativo
        current_exposure = sum(
            pos.position_size for pos in portfolio.open_positions
            if pos.symbol == decision.symbol
        )

        max_concentration = portfolio.total_balance * self.risk_limits.max_concentration
        remaining_capacity = max_concentration - current_exposure

        return min(max_size, remaining_capacity, decision.position_size)

    def _get_asset_correlation(self, symbol1: str, symbol2: str) -> float:
        """Obtém correlação entre dois ativos"""

        if symbol1 == symbol2:
            return 1.0

        # Correlações conhecidas para pares de forex
        forex_correlations = {
            ('EURUSD', 'GBPUSD'): 0.85,
            ('EURUSD', 'EURGBP'): 0.45,
            ('GBPUSD', 'EURGBP'): -0.65,
            ('USDJPY', 'USDCHF'): 0.75,
            ('AUDUSD', 'NZDUSD'): 0.90,
            ('EURUSD', 'USDCHF'): -0.80,
            ('GBPUSD', 'USDCHF'): -0.75
        }

        # Verificar correlação direta
        pair = (symbol1, symbol2)
        if pair in forex_correlations:
            return forex_correlations[pair]

        # Verificar correlação reversa
        reverse_pair = (symbol2, symbol1)
        if reverse_pair in forex_correlations:
            return forex_correlations[reverse_pair]

        # Correlação padrão baixa para ativos diferentes
        return 0.1

    async def calculate_portfolio_metrics(self, portfolio: PortfolioState) -> RiskMetrics:
        """Calcula métricas de risco do portfólio"""

        # Usar cache se recente (< 1 minuto)
        if (self._metrics_cache and
            (datetime.now() - self._cache_timestamp).total_seconds() < 60):
            return self._metrics_cache

        # Calcular drawdown atual
        current_drawdown = (self.peak_balance - portfolio.total_balance) / self.peak_balance

        # Calcular volatilidade dos retornos
        volatility = self._calculate_returns_volatility()

        # Calcular Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Calcular VaR diário (95% confidence)
        daily_var = self._calculate_var()

        # Calcular risco de concentração
        concentration_risk = self._calculate_concentration_risk(portfolio)

        # Calcular risco de correlação
        correlation_risk = self._calculate_correlation_risk(portfolio)

        # Calcular leverage ratio
        leverage_ratio = portfolio.get_total_exposure() / portfolio.total_balance

        metrics = RiskMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=self._get_max_drawdown(),
            daily_var=daily_var,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            leverage_ratio=leverage_ratio
        )

        # Atualizar cache
        self._metrics_cache = metrics
        self._cache_timestamp = datetime.now()

        return metrics

    def _calculate_returns_volatility(self) -> float:
        """Calcula volatilidade dos retornos"""
        if len(self.daily_returns) < 2:
            return 0.0

        return statistics.stdev(self.daily_returns)

    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe ratio"""
        if len(self.daily_returns) < 2:
            return 0.0

        mean_return = statistics.mean(self.daily_returns)
        volatility = statistics.stdev(self.daily_returns)

        if volatility == 0:
            return 0.0

        # Assumir risk-free rate de 0 para simplificar
        return mean_return / volatility * np.sqrt(252)  # Annualized

    def _calculate_var(self) -> float:
        """Calcula Value at Risk (95% confidence)"""
        if len(self.daily_returns) < 10:
            return 0.0

        # VaR paramétrico (95% confidence = 1.65 standard deviations)
        mean_return = statistics.mean(self.daily_returns)
        volatility = statistics.stdev(self.daily_returns)

        var_95 = -(mean_return - 1.65 * volatility)
        return max(0, var_95)

    def _calculate_concentration_risk(self, portfolio: PortfolioState) -> float:
        """Calcula risco de concentração"""
        if not portfolio.open_positions:
            return 0.0

        # Calcular exposição por ativo
        exposures = {}
        for position in portfolio.open_positions:
            symbol = position.symbol
            exposures[symbol] = exposures.get(symbol, 0) + position.position_size

        # Calcular concentração máxima
        max_exposure = max(exposures.values()) if exposures else 0
        concentration = max_exposure / portfolio.total_balance

        return concentration

    def _calculate_correlation_risk(self, portfolio: PortfolioState) -> float:
        """Calcula risco de correlação"""
        if len(portfolio.open_positions) < 2:
            return 0.0

        # Calcular correlação ponderada média
        total_correlation = 0.0
        total_weight = 0.0

        positions = portfolio.open_positions
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                correlation = abs(self._get_asset_correlation(pos1.symbol, pos2.symbol))
                weight = (pos1.position_size + pos2.position_size) / 2
                total_correlation += correlation * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_correlation / total_weight

    def _get_max_drawdown(self) -> float:
        """Calcula drawdown máximo histórico"""
        if not self.portfolio_values:
            return 0.0

        peak = self.initial_balance
        max_dd = 0.0

        for _, value in self.portfolio_values:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    async def update_trade_record(self,
                                trade_info: Dict[str, Any],
                                portfolio_balance: float):
        """Atualiza registros para análise de risco"""

        # Adicionar ao histórico
        self.trade_history.append({
            'timestamp': datetime.now(),
            'trade_info': trade_info,
            'balance_after': portfolio_balance
        })

        # Atualizar portfolio values
        self.portfolio_values.append((datetime.now(), portfolio_balance))

        # Calcular retorno diário
        if self.current_balance > 0:
            daily_return = (portfolio_balance - self.current_balance) / self.current_balance
            self.daily_returns.append(daily_return)

            # Manter apenas últimos 252 dias (1 ano)
            if len(self.daily_returns) > 252:
                self.daily_returns = self.daily_returns[-252:]

        self.current_balance = portfolio_balance
        self.daily_trades += 1

    async def should_halt_trading(self, portfolio: PortfolioState) -> bool:
        """Verifica se deve parar trading por segurança"""

        metrics = await self.calculate_portfolio_metrics(portfolio)

        # Critérios para parada de emergência
        emergency_conditions = [
            metrics.current_drawdown > self.risk_limits.max_drawdown,
            metrics.daily_var > self.risk_limits.var_limit,
            self.daily_pnl < -self.risk_limits.max_daily_loss * portfolio.total_balance,
            metrics.leverage_ratio > self.risk_limits.max_leverage
        ]

        if any(emergency_conditions):
            if not self.emergency_mode:
                self.emergency_mode = True
                self.logger.warning("🚨 MODO DE EMERGÊNCIA ATIVADO - Trading pausado por segurança")
            return True

        return False

    def get_risk_report(self) -> Dict[str, Any]:
        """Gera relatório completo de risco"""

        return {
            'risk_limits': {
                'max_daily_loss': self.risk_limits.max_daily_loss,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_position_size': self.risk_limits.max_position_size,
                'var_limit': self.risk_limits.var_limit
            },
            'current_status': {
                'emergency_mode': self.emergency_mode,
                'daily_trades': self.daily_trades,
                'daily_pnl': self.daily_pnl,
                'risk_breaches': self.risk_breaches
            },
            'metrics': self._metrics_cache.to_dict() if self._metrics_cache else None,
            'trade_statistics': {
                'total_trades': len(self.trade_history),
                'returns_tracked': len(self.daily_returns),
                'portfolio_values_tracked': len(self.portfolio_values)
            }
        }
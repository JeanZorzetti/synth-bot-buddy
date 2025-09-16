"""
Real Portfolio Optimizer - Otimização Real de Portfólio
Sistema avançado de otimização de portfólio com algoritmos modernos de alocação de capital.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from real_tick_processor import ProcessedTickData
from real_trading_executor import RealTradeRequest, TradeType, ContractType

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    KELLY_CRITERION = "kelly_criterion"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    HIERARCHICAL_RISK_PARITY = "hrp"

class RiskConstraintType(Enum):
    MAX_PORTFOLIO_VAR = "max_var"
    MAX_INDIVIDUAL_WEIGHT = "max_weight"
    MAX_SECTOR_CONCENTRATION = "max_sector"
    MIN_DIVERSIFICATION = "min_diversification"
    MAX_TURNOVER = "max_turnover"

@dataclass
class AssetCharacteristics:
    """Características de um ativo"""
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation_to_market: float
    liquidity_score: float
    sector: str = "unknown"
    market_cap: float = 0.0

@dataclass
class RiskConstraint:
    """Restrição de risco"""
    constraint_type: RiskConstraintType
    value: float
    weight: float = 1.0

@dataclass
class OptimizationResult:
    """Resultado da otimização"""
    method: OptimizationMethod
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    diversification_ratio: float
    turnover: float
    optimization_time: float
    convergence_status: str
    objective_value: float

@dataclass
class PortfolioMetrics:
    """Métricas do portfólio"""
    total_value: float
    number_of_positions: int
    concentration_hhi: float  # Herfindahl-Hirschman Index
    diversification_ratio: float
    risk_adjusted_return: float
    maximum_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95_1d: float
    expected_shortfall: float
    beta_to_market: float

class RealPortfolioOptimizer:
    """Otimizador real de portfólio com algoritmos avançados"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

        # Estado do otimizador
        self.assets: Dict[str, AssetCharacteristics] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None

        # Configurações de otimização
        self.optimization_window_days = 252  # 1 ano
        self.rebalancing_frequency_days = 30  # Rebalanceamento mensal
        self.min_history_points = 50

        # Restrições padrão
        self.default_constraints = [
            RiskConstraint(RiskConstraintType.MAX_INDIVIDUAL_WEIGHT, 0.4),
            RiskConstraint(RiskConstraintType.MAX_PORTFOLIO_VAR, 0.05),
            RiskConstraint(RiskConstraintType.MIN_DIVERSIFICATION, 0.5)
        ]

        # Cache de resultados
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        self.last_optimization_time: Optional[datetime] = None

        logger.info("RealPortfolioOptimizer inicializado")

    async def add_asset(self, symbol: str, price_data: List[ProcessedTickData]):
        """Adiciona um ativo ao universo de investimento"""
        try:
            # Converter dados de tick para formato adequado
            prices = [(tick.timestamp, tick.price) for tick in price_data]
            self.price_history[symbol] = prices

            # Calcular características do ativo
            characteristics = await self._calculate_asset_characteristics(symbol, price_data)
            self.assets[symbol] = characteristics

            # Invalidar cache
            self.optimization_cache.clear()

            logger.info(f"Ativo {symbol} adicionado ao otimizador")

        except Exception as e:
            logger.error(f"Erro ao adicionar ativo {symbol}: {e}")

    async def _calculate_asset_characteristics(self, symbol: str,
                                            price_data: List[ProcessedTickData]) -> AssetCharacteristics:
        """Calcula características estatísticas de um ativo"""
        if len(price_data) < self.min_history_points:
            # Características padrão para ativos com pouco histórico
            return AssetCharacteristics(
                symbol=symbol,
                expected_return=0.0,
                volatility=0.2,
                sharpe_ratio=0.0,
                max_drawdown=0.1,
                beta=1.0,
                correlation_to_market=0.5,
                liquidity_score=0.5
            )

        prices = np.array([tick.price for tick in price_data])
        returns = np.diff(prices) / prices[:-1]

        # Retorno esperado (anualizado)
        expected_return = np.mean(returns) * 252

        # Volatilidade (anualizada)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Máximo drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Score de liquidez baseado em volume (simplificado)
        volumes = [tick.volume for tick in price_data if hasattr(tick, 'volume')]
        liquidity_score = min(np.mean(volumes) / 1000000 if volumes else 0.5, 1.0)

        return AssetCharacteristics(
            symbol=symbol,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            beta=1.0,  # Simplificado - seria calculado contra benchmark
            correlation_to_market=0.5,  # Simplificado
            liquidity_score=liquidity_score
        )

    async def optimize_portfolio(self, method: OptimizationMethod,
                               target_return: Optional[float] = None,
                               risk_constraints: List[RiskConstraint] = None,
                               current_weights: Dict[str, float] = None) -> OptimizationResult:
        """Executa otimização de portfólio"""
        start_time = datetime.now()

        try:
            # Validar entrada
            if len(self.assets) < 2:
                raise ValueError("Mínimo 2 ativos necessários para otimização")

            # Usar restrições padrão se não fornecidas
            constraints = risk_constraints or self.default_constraints

            # Preparar dados para otimização
            symbols = list(self.assets.keys())
            returns_matrix = await self._build_returns_matrix(symbols)
            expected_returns = np.array([self.assets[s].expected_return for s in symbols])

            # Calcular matriz de covariância
            self.covariance_matrix = await self._calculate_covariance_matrix(returns_matrix)
            self.correlation_matrix = await self._calculate_correlation_matrix(returns_matrix)

            # Executar otimização baseada no método
            if method == OptimizationMethod.MEAN_VARIANCE:
                result = await self._optimize_mean_variance(
                    expected_returns, symbols, target_return, constraints, current_weights
                )
            elif method == OptimizationMethod.RISK_PARITY:
                result = await self._optimize_risk_parity(symbols, constraints)
            elif method == OptimizationMethod.KELLY_CRITERION:
                result = await self._optimize_kelly_criterion(
                    expected_returns, symbols, constraints
                )
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                result = await self._optimize_minimum_variance(symbols, constraints)
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                result = await self._optimize_maximum_sharpe(
                    expected_returns, symbols, constraints
                )
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                result = await self._optimize_hrp(symbols)
            else:
                raise ValueError(f"Método {method} não implementado")

            # Calcular tempo de otimização
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time

            # Armazenar no cache
            cache_key = f"{method.value}_{hash(str(symbols))}"
            self.optimization_cache[cache_key] = result
            self.last_optimization_time = datetime.now()

            logger.info(f"Otimização {method.value} concluída em {optimization_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Erro na otimização {method.value}: {e}")
            raise

    async def _build_returns_matrix(self, symbols: List[str]) -> np.ndarray:
        """Constrói matriz de retornos dos ativos"""
        returns_data = []

        for symbol in symbols:
            if symbol not in self.price_history:
                continue

            prices = [price for _, price in self.price_history[symbol]]
            returns = np.diff(prices) / np.array(prices[:-1])
            returns_data.append(returns)

        # Encontrar tamanho mínimo comum
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[-min_length:] for r in returns_data]).T

        return returns_matrix

    async def _calculate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calcula matriz de covariância com shrinkage Ledoit-Wolf"""
        try:
            # Usar estimador Ledoit-Wolf para melhor estabilidade
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_matrix).covariance_

            # Anualizar
            return cov_matrix * 252

        except Exception:
            # Fallback para covariância sample
            return np.cov(returns_matrix.T) * 252

    async def _calculate_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calcula matriz de correlação"""
        return np.corrcoef(returns_matrix.T)

    async def _optimize_mean_variance(self, expected_returns: np.ndarray,
                                    symbols: List[str],
                                    target_return: Optional[float],
                                    constraints: List[RiskConstraint],
                                    current_weights: Dict[str, float]) -> OptimizationResult:
        """Otimização Média-Variância de Markowitz"""
        n_assets = len(symbols)

        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            return portfolio_variance

        # Restrições
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Soma = 1
        ]

        if target_return:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })

        # Bounds para pesos
        bounds = [(0, 1) for _ in range(n_assets)]

        # Aplicar restrições de risco
        for constraint in constraints:
            if constraint.constraint_type == RiskConstraintType.MAX_INDIVIDUAL_WEIGHT:
                bounds = [(0, constraint.value) for _ in range(n_assets)]

        # Pesos iniciais
        x0 = np.array([1/n_assets] * n_assets)
        if current_weights:
            x0 = np.array([current_weights.get(s, 1/n_assets) for s in symbols])

        # Otimização
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Otimização não convergiu: {result.message}")

        weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(self.covariance_matrix, result.x)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.MEAN_VARIANCE,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,  # Será calculado posteriormente
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(result.x),
            turnover=self._calculate_turnover(weights_dict, current_weights),
            optimization_time=0.0,
            convergence_status="success" if result.success else "failed",
            objective_value=result.fun
        )

    async def _optimize_risk_parity(self, symbols: List[str],
                                  constraints: List[RiskConstraint]) -> OptimizationResult:
        """Otimização Risk Parity"""
        n_assets = len(symbols)

        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
            marginal_contribs = np.dot(self.covariance_matrix, weights) / portfolio_vol
            risk_contribs = weights * marginal_contribs

            # Minimizar diferença entre contribuições de risco
            target_contrib = 1.0 / n_assets
            return np.sum((risk_contribs / np.sum(risk_contribs) - target_contrib) ** 2)

        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0.01, 1) for _ in range(n_assets)]  # Min 1% por ativo
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}
        expected_returns = np.array([self.assets[s].expected_return for s in symbols])
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(self.covariance_matrix, result.x)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.RISK_PARITY,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(result.x),
            turnover=0.0,
            optimization_time=0.0,
            convergence_status="success" if result.success else "failed",
            objective_value=result.fun
        )

    async def _optimize_kelly_criterion(self, expected_returns: np.ndarray,
                                      symbols: List[str],
                                      constraints: List[RiskConstraint]) -> OptimizationResult:
        """Otimização Kelly Criterion"""
        n_assets = len(symbols)

        def objective(weights):
            # Kelly ótimo: f = (μ - r) / σ²
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))

            if portfolio_variance <= 0:
                return 1e6

            # Maximizar log do crescimento esperado
            excess_return = portfolio_return - self.risk_free_rate
            growth_rate = excess_return - 0.5 * portfolio_variance

            return -growth_rate  # Minimizar negativo = maximizar

        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, 0.5) for _ in range(n_assets)]  # Max 50% por ativo (Kelly conservador)
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(self.covariance_matrix, result.x)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.KELLY_CRITERION,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(result.x),
            turnover=0.0,
            optimization_time=0.0,
            convergence_status="success" if result.success else "failed",
            objective_value=result.fun
        )

    async def _optimize_minimum_variance(self, symbols: List[str],
                                       constraints: List[RiskConstraint]) -> OptimizationResult:
        """Otimização Mínima Variância"""
        n_assets = len(symbols)

        def objective(weights):
            return np.dot(weights, np.dot(self.covariance_matrix, weights))

        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}
        expected_returns = np.array([self.assets[s].expected_return for s in symbols])
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(result.fun)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.MINIMUM_VARIANCE,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(result.x),
            turnover=0.0,
            optimization_time=0.0,
            convergence_status="success" if result.success else "failed",
            objective_value=result.fun
        )

    async def _optimize_maximum_sharpe(self, expected_returns: np.ndarray,
                                     symbols: List[str],
                                     constraints: List[RiskConstraint]) -> OptimizationResult:
        """Otimização Máximo Sharpe Ratio"""
        n_assets = len(symbols)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))

            if portfolio_vol <= 0:
                return -1e6

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimizar negativo = maximizar

        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(self.covariance_matrix, result.x)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.MAXIMUM_SHARPE,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(result.x),
            turnover=0.0,
            optimization_time=0.0,
            convergence_status="success" if result.success else "failed",
            objective_value=result.fun
        )

    async def _optimize_hrp(self, symbols: List[str]) -> OptimizationResult:
        """Hierarchical Risk Parity (HRP)"""
        # Implementação simplificada do HRP
        n_assets = len(symbols)

        # Usar matriz de correlação para clustering
        distance_matrix = np.sqrt(0.5 * (1 - self.correlation_matrix))

        # Alocação igual como simplificação
        # (implementação completa do HRP requereria scipy.cluster.hierarchy)
        equal_weights = np.array([1/n_assets] * n_assets)

        weights_dict = {symbols[i]: equal_weights[i] for i in range(n_assets)}
        expected_returns = np.array([self.assets[s].expected_return for s in symbols])
        portfolio_return = np.dot(equal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(equal_weights, np.dot(self.covariance_matrix, equal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return OptimizationResult(
            method=OptimizationMethod.HIERARCHICAL_RISK_PARITY,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=0.0,
            expected_shortfall=0.0,
            diversification_ratio=self._calculate_diversification_ratio(equal_weights),
            turnover=0.0,
            optimization_time=0.0,
            convergence_status="success",
            objective_value=0.0
        )

    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calcula ratio de diversificação"""
        individual_vols = np.sqrt(np.diag(self.covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))

        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    def _calculate_turnover(self, new_weights: Dict[str, float],
                          current_weights: Dict[str, float]) -> float:
        """Calcula turnover do portfólio"""
        if not current_weights:
            return 1.0  # 100% turnover se não há posições atuais

        turnover = 0.0
        for symbol in new_weights:
            current_weight = current_weights.get(symbol, 0.0)
            turnover += abs(new_weights[symbol] - current_weight)

        return turnover / 2  # Dividir por 2 (convenção padrão)

    async def generate_rebalancing_trades(self, optimization_result: OptimizationResult,
                                        current_portfolio_value: float,
                                        current_positions: Dict[str, float]) -> List[RealTradeRequest]:
        """Gera trades para rebalanceamento do portfólio"""
        trades = []

        try:
            target_weights = optimization_result.weights

            for symbol, target_weight in target_weights.items():
                current_weight = current_positions.get(symbol, 0.0) / current_portfolio_value

                # Calcular diferença de alocação
                weight_diff = target_weight - current_weight
                trade_amount = abs(weight_diff * current_portfolio_value)

                # Filtro de trades pequenos (< 1% do portfólio)
                if trade_amount < current_portfolio_value * 0.01:
                    continue

                # Determinar tipo de trade
                if weight_diff > 0:
                    trade_type = TradeType.BUY
                    contract_type = ContractType.CALL
                else:
                    trade_type = TradeType.SELL
                    contract_type = ContractType.PUT
                    trade_amount = abs(weight_diff * current_portfolio_value)

                # Criar requisição de trade
                trade_request = RealTradeRequest(
                    symbol=symbol,
                    trade_type=trade_type,
                    contract_type=contract_type,
                    amount=trade_amount,
                    duration=300,  # 5 minutos
                    duration_unit="s"
                )

                trades.append(trade_request)

            logger.info(f"Gerados {len(trades)} trades para rebalanceamento")
            return trades

        except Exception as e:
            logger.error(f"Erro ao gerar trades de rebalanceamento: {e}")
            return []

    async def calculate_portfolio_metrics(self, current_weights: Dict[str, float],
                                        current_values: Dict[str, float]) -> PortfolioMetrics:
        """Calcula métricas completas do portfólio"""
        try:
            total_value = sum(current_values.values())
            n_positions = len([w for w in current_weights.values() if w > 0.001])

            # HHI de concentração
            weights_array = np.array(list(current_weights.values()))
            concentration_hhi = np.sum(weights_array ** 2)

            # Ratio de diversificação
            diversification_ratio = self._calculate_diversification_ratio(weights_array)

            # Retorno ajustado ao risco (Sharpe)
            expected_returns = np.array([self.assets[s].expected_return for s in current_weights.keys()])
            portfolio_return = np.dot(weights_array, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(self.covariance_matrix, weights_array)))
            risk_adjusted_return = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

            # VaR 95% (simplificado)
            var_95_1d = portfolio_vol / np.sqrt(252) * 1.65  # 95% confiança

            return PortfolioMetrics(
                total_value=total_value,
                number_of_positions=n_positions,
                concentration_hhi=concentration_hhi,
                diversification_ratio=diversification_ratio,
                risk_adjusted_return=risk_adjusted_return,
                maximum_drawdown=0.0,  # Seria calculado com histórico
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                var_95_1d=var_95_1d,
                expected_shortfall=var_95_1d * 1.3,  # Aproximação
                beta_to_market=1.0  # Simplificado
            )

        except Exception as e:
            logger.error(f"Erro ao calcular métricas do portfólio: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)

    async def get_optimization_recommendation(self, current_portfolio: Dict[str, float],
                                            market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Obtém recomendação de otimização baseada nas condições atuais"""
        recommendations = {
            "suggested_method": OptimizationMethod.RISK_PARITY,
            "rebalancing_needed": False,
            "risk_level": "medium",
            "expected_improvement": 0.0,
            "reasoning": []
        }

        try:
            # Analisar volatilidade do mercado
            market_volatility = market_conditions.get("volatility", 0.2)
            if market_volatility > 0.3:
                recommendations["suggested_method"] = OptimizationMethod.MINIMUM_VARIANCE
                recommendations["risk_level"] = "low"
                recommendations["reasoning"].append("Alta volatilidade detectada - foco em redução de risco")

            # Analisar concentração atual
            if current_portfolio:
                weights = np.array(list(current_portfolio.values()))
                hhi = np.sum(weights ** 2)
                if hhi > 0.5:  # Alta concentração
                    recommendations["rebalancing_needed"] = True
                    recommendations["reasoning"].append("Portfólio muito concentrado - diversificação necessária")

            # Analisar tendência de mercado
            market_trend = market_conditions.get("trend", "neutral")
            if market_trend == "bullish":
                recommendations["suggested_method"] = OptimizationMethod.MAXIMUM_SHARPE
                recommendations["reasoning"].append("Mercado em alta - foco em maximização de retorno")
            elif market_trend == "bearish":
                recommendations["suggested_method"] = OptimizationMethod.MINIMUM_VARIANCE
                recommendations["reasoning"].append("Mercado em baixa - foco em preservação de capital")

            return recommendations

        except Exception as e:
            logger.error(f"Erro ao gerar recomendação: {e}")
            return recommendations

    async def backtest_strategy(self, method: OptimizationMethod,
                              start_date: datetime,
                              end_date: datetime) -> Dict[str, Any]:
        """Realiza backtest de uma estratégia de otimização"""
        # Implementação simplificada
        # Em produção, seria necessário dados históricos completos
        return {
            "total_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades_count": 0
        }

    async def get_risk_attribution(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Calcula atribuição de risco do portfólio"""
        try:
            weights_array = np.array(list(weights.values()))
            symbols = list(weights.keys())

            portfolio_variance = np.dot(weights_array, np.dot(self.covariance_matrix, weights_array))
            marginal_contributions = np.dot(self.covariance_matrix, weights_array)
            component_contributions = weights_array * marginal_contributions

            risk_attribution = {}
            for i, symbol in enumerate(symbols):
                risk_attribution[symbol] = {
                    "weight": weights[symbol],
                    "marginal_contribution": marginal_contributions[i],
                    "component_contribution": component_contributions[i],
                    "risk_contribution_pct": component_contributions[i] / portfolio_variance * 100
                }

            return {
                "portfolio_variance": portfolio_variance,
                "total_risk": np.sqrt(portfolio_variance),
                "risk_attribution": risk_attribution
            }

        except Exception as e:
            logger.error(f"Erro na atribuição de risco: {e}")
            return {}

    async def shutdown(self):
        """Encerra o otimizador"""
        self.optimization_cache.clear()
        logger.info("RealPortfolioOptimizer encerrado")
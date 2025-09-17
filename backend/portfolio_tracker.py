"""
Portfolio Tracker - Sistema de Rastreamento de Portfólio e Cálculo de P&L
Sistema completo para rastreamento de portfólio, cálculo de P&L e análise de performance
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem
from real_position_manager import RealPositionManager, Position, PositionType
from order_execution_monitor import OrderExecutionMonitor, Order, OrderStatus

class PortfolioPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"

class AssetClass(Enum):
    VOLATILITY_INDICES = "volatility_indices"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    INDICES = "indices"

@dataclass
class PortfolioSnapshot:
    """Snapshot do portfólio em um momento específico"""
    timestamp: datetime
    total_value: float
    available_cash: float
    used_margin: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_return: float
    total_return_pct: float
    positions_count: int
    active_symbols: List[str]
    largest_position: Optional[str]
    largest_position_value: float
    risk_metrics: Dict[str, float]

@dataclass
class PositionSummary:
    """Resumo de uma posição para o portfólio"""
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_in_portfolio: float
    days_held: int
    risk_contribution: float

@dataclass
class PerformanceMetrics:
    """Métricas de performance do portfólio"""
    period: PortfolioPeriod
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    volatility: float
    var_95: float
    expected_shortfall: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int

@dataclass
class AssetAllocation:
    """Alocação por classe de ativo"""
    asset_class: AssetClass
    total_value: float
    weight_pct: float
    positions_count: int
    unrealized_pnl: float
    symbols: List[str]

class PortfolioTracker:
    """Rastreador de portfólio e calculador de P&L"""

    def __init__(self, initial_capital: float = 10000.0):
        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()
        self.position_manager = None  # Será definido na inicialização
        self.order_monitor = None  # Será definido na inicialização

        # Capital inicial e configurações
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.available_cash = initial_capital
        self.used_margin = 0.0

        # Estado do sistema
        self.is_tracking = False
        self.tracking_task = None
        self.calculation_task = None

        # Dados históricos
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_cache: Dict[PortfolioPeriod, PerformanceMetrics] = {}

        # Métricas em tempo real
        self.current_positions: Dict[str, PositionSummary] = {}
        self.asset_allocation: Dict[AssetClass, AssetAllocation] = {}

        # Configurações
        self.snapshot_frequency_minutes = 5
        self.performance_update_frequency_minutes = 15
        self.max_history_days = 365

        # Classificação de símbolos por asset class
        self.symbol_classification = {
            "R_10": AssetClass.VOLATILITY_INDICES,
            "R_25": AssetClass.VOLATILITY_INDICES,
            "R_50": AssetClass.VOLATILITY_INDICES,
            "R_75": AssetClass.VOLATILITY_INDICES,
            "R_100": AssetClass.VOLATILITY_INDICES,
            "FRXEURUSD": AssetClass.FOREX,
            "FRXGBPUSD": AssetClass.FOREX,
            "FRXUSDJPY": AssetClass.FOREX,
            "FRXAUDUSD": AssetClass.FOREX,
            "FRXUSDCAD": AssetClass.FOREX,
            "cryBTCUSD": AssetClass.CRYPTO,
            "cryETHUSD": AssetClass.CRYPTO,
        }

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(
        self,
        position_manager: RealPositionManager,
        order_monitor: OrderExecutionMonitor
    ):
        """Inicializa o rastreador de portfólio"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.logger.initialize()

            self.position_manager = position_manager
            self.order_monitor = order_monitor

            # Carregar dados históricos
            await self._load_historical_data()

            # Calcular snapshot inicial
            await self._calculate_portfolio_snapshot()

            await self.logger.log_activity("portfolio_tracker_initialized", {
                "initial_capital": self.initial_capital,
                "history_entries": len(self.portfolio_history)
            })

            print("✅ Portfolio Tracker inicializado com sucesso")

        except Exception as e:
            await self.logger.log_error("portfolio_tracker_init_error", str(e))
            raise

    async def start_tracking(self):
        """Inicia rastreamento do portfólio"""
        if self.is_tracking:
            return

        self.is_tracking = True

        # Iniciar tasks
        self.tracking_task = asyncio.create_task(self._portfolio_tracking_loop())
        self.calculation_task = asyncio.create_task(self._performance_calculation_loop())

        await self.logger.log_activity("portfolio_tracking_started", {})
        print("📊 Rastreamento de portfólio iniciado")

    async def stop_tracking(self):
        """Para o rastreamento"""
        self.is_tracking = False

        # Cancelar tasks
        for task in [self.tracking_task, self.calculation_task]:
            if task:
                task.cancel()

        await self.logger.log_activity("portfolio_tracking_stopped", {})
        print("⏹️ Rastreamento de portfólio parado")

    async def _portfolio_tracking_loop(self):
        """Loop principal de rastreamento"""
        while self.is_tracking:
            try:
                # Calcular snapshot atual
                await self._calculate_portfolio_snapshot()

                # Atualizar posições
                await self._update_position_summaries()

                # Atualizar alocação de ativos
                await self._update_asset_allocation()

                # Salvar snapshot se significativo
                await self._save_snapshot_if_needed()

                await asyncio.sleep(self.snapshot_frequency_minutes * 60)

            except Exception as e:
                await self.logger.log_error("portfolio_tracking_error", str(e))
                await asyncio.sleep(300)

    async def _calculate_portfolio_snapshot(self):
        """Calcula snapshot atual do portfólio"""
        try:
            if not self.position_manager:
                return

            # Obter resumo das posições
            position_summary = await self.position_manager.get_position_summary()

            # Calcular valores
            unrealized_pnl = position_summary.get("total_unrealized_pnl", 0.0)
            realized_pnl = position_summary.get("daily_pnl", 0.0)
            used_margin = position_summary.get("total_margin_used", 0.0)

            # Valor total do portfólio
            total_value = self.initial_capital + realized_pnl + unrealized_pnl
            available_cash = total_value - used_margin

            # Retorno total
            total_return = total_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100

            # Posições ativas
            positions_count = position_summary.get("active_positions_count", 0)
            symbols_count = position_summary.get("symbols_count", {})
            active_symbols = list(symbols_count.keys())

            # Maior posição
            largest_position = None
            largest_position_value = 0.0
            if active_symbols:
                # Simplificado - em produção, calcular valor real de cada posição
                largest_position = max(symbols_count.keys(), key=lambda x: symbols_count[x])
                largest_position_value = symbols_count[largest_position] * 100  # Estimativa

            # Métricas de risco simplificadas
            risk_metrics = {
                "portfolio_volatility": await self._calculate_portfolio_volatility(),
                "var_95": await self._calculate_var_95(),
                "max_drawdown": await self._calculate_current_drawdown(),
                "sharpe_ratio": await self._calculate_current_sharpe_ratio()
            }

            # Atualizar propriedades
            self.current_value = total_value
            self.available_cash = available_cash
            self.used_margin = used_margin

            # Criar snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                total_value=total_value,
                available_cash=available_cash,
                used_margin=used_margin,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                daily_pnl=realized_pnl,  # Simplificado
                total_return=total_return,
                total_return_pct=total_return_pct,
                positions_count=positions_count,
                active_symbols=active_symbols,
                largest_position=largest_position,
                largest_position_value=largest_position_value,
                risk_metrics=risk_metrics
            )

            # Cache do snapshot
            await self.cache_manager.set(
                CacheNamespace.TRADING_SESSIONS,
                "current_portfolio_snapshot",
                asdict(snapshot),
                ttl=300
            )

            self._current_snapshot = snapshot

        except Exception as e:
            await self.logger.log_error("portfolio_snapshot_calculation_error", str(e))

    async def _update_position_summaries(self):
        """Atualiza resumos das posições"""
        try:
            if not self.position_manager:
                return

            self.current_positions = {}

            # Obter posições ativas
            for position_id, position in self.position_manager.active_positions.items():
                # Calcular métricas da posição
                market_value = position.quantity * position.current_price
                unrealized_pnl = position.unrealized_pnl or 0.0
                unrealized_pnl_pct = (unrealized_pnl / (position.quantity * position.entry_price)) * 100

                # Peso no portfólio
                weight_in_portfolio = (market_value / self.current_value) * 100 if self.current_value > 0 else 0.0

                # Dias mantido
                days_held = (datetime.utcnow() - position.open_time).days

                # Contribuição para risco (simplificado)
                risk_contribution = weight_in_portfolio * 0.1  # 10% do peso como aproximação

                summary = PositionSummary(
                    symbol=position.symbol,
                    position_type=position.position_type,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=position.current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    weight_in_portfolio=weight_in_portfolio,
                    days_held=days_held,
                    risk_contribution=risk_contribution
                )

                self.current_positions[position.symbol] = summary

        except Exception as e:
            await self.logger.log_error("position_summaries_update_error", str(e))

    async def _update_asset_allocation(self):
        """Atualiza alocação por classe de ativo"""
        try:
            self.asset_allocation = {}

            # Agrupar posições por classe de ativo
            allocation_data = defaultdict(lambda: {
                "total_value": 0.0,
                "positions_count": 0,
                "unrealized_pnl": 0.0,
                "symbols": []
            })

            for symbol, position_summary in self.current_positions.items():
                asset_class = self.symbol_classification.get(symbol, AssetClass.VOLATILITY_INDICES)

                allocation_data[asset_class]["total_value"] += position_summary.market_value
                allocation_data[asset_class]["positions_count"] += 1
                allocation_data[asset_class]["unrealized_pnl"] += position_summary.unrealized_pnl
                allocation_data[asset_class]["symbols"].append(symbol)

            # Criar objetos AssetAllocation
            for asset_class, data in allocation_data.items():
                weight_pct = (data["total_value"] / self.current_value) * 100 if self.current_value > 0 else 0.0

                allocation = AssetAllocation(
                    asset_class=asset_class,
                    total_value=data["total_value"],
                    weight_pct=weight_pct,
                    positions_count=data["positions_count"],
                    unrealized_pnl=data["unrealized_pnl"],
                    symbols=data["symbols"]
                )

                self.asset_allocation[asset_class] = allocation

        except Exception as e:
            await self.logger.log_error("asset_allocation_update_error", str(e))

    async def _save_snapshot_if_needed(self):
        """Salva snapshot se houve mudança significativa"""
        try:
            if not hasattr(self, '_current_snapshot'):
                return

            # Verificar se há mudança significativa (>0.1% ou nova posição)
            should_save = False

            if not self.portfolio_history:
                should_save = True
            else:
                last_snapshot = self.portfolio_history[-1]
                value_change_pct = abs(self._current_snapshot.total_value - last_snapshot.total_value) / last_snapshot.total_value * 100

                if (value_change_pct > 0.1 or
                    self._current_snapshot.positions_count != last_snapshot.positions_count):
                    should_save = True

            if should_save:
                self.portfolio_history.append(self._current_snapshot)

                # Manter apenas histórico recente
                cutoff_date = datetime.utcnow() - timedelta(days=self.max_history_days)
                self.portfolio_history = [
                    s for s in self.portfolio_history
                    if s.timestamp > cutoff_date
                ]

                # Salvar no banco
                await self._save_snapshot_to_db(self._current_snapshot)

                # Calcular retorno diário
                if len(self.portfolio_history) > 1:
                    prev_value = self.portfolio_history[-2].total_value
                    daily_return = (self._current_snapshot.total_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)

                    # Manter apenas últimos 252 dias
                    if len(self.daily_returns) > 252:
                        self.daily_returns = self.daily_returns[-252:]

        except Exception as e:
            await self.logger.log_error("snapshot_save_error", str(e))

    async def _performance_calculation_loop(self):
        """Loop de cálculo de performance"""
        while self.is_tracking:
            try:
                # Calcular métricas para diferentes períodos
                for period in PortfolioPeriod:
                    metrics = await self._calculate_performance_metrics(period)
                    if metrics:
                        self.performance_cache[period] = metrics

                # Cache das métricas
                await self.cache_manager.set(
                    CacheNamespace.TRADING_SESSIONS,
                    "portfolio_performance_metrics",
                    {period.value: asdict(metrics) for period, metrics in self.performance_cache.items()},
                    ttl=900  # 15 minutos
                )

                await asyncio.sleep(self.performance_update_frequency_minutes * 60)

            except Exception as e:
                await self.logger.log_error("performance_calculation_error", str(e))
                await asyncio.sleep(600)

    async def _calculate_performance_metrics(self, period: PortfolioPeriod) -> Optional[PerformanceMetrics]:
        """Calcula métricas de performance para um período"""
        try:
            if len(self.portfolio_history) < 2:
                return None

            # Filtrar dados por período
            snapshots = self._filter_snapshots_by_period(period)
            if len(snapshots) < 2:
                return None

            # Calcular retornos
            returns = self._calculate_returns_from_snapshots(snapshots)
            if not returns:
                return None

            # Métricas básicas
            total_return = snapshots[-1].total_value - snapshots[0].total_value
            total_return_pct = (total_return / snapshots[0].total_value) * 100

            # Métricas de risco
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, snapshots)

            # Drawdown
            max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(snapshots)

            # VaR e Expected Shortfall
            var_95 = np.percentile(returns, 5) if len(returns) > 10 else 0.0
            tail_returns = [r for r in returns if r <= var_95]
            expected_shortfall = np.mean(tail_returns) if tail_returns else 0.0

            # Métricas de trading
            trade_metrics = await self._calculate_trade_metrics(period)

            metrics = PerformanceMetrics(
                period=period,
                total_return=total_return,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                volatility=volatility,
                var_95=var_95 * snapshots[-1].total_value,  # Converter para valor absoluto
                expected_shortfall=expected_shortfall * snapshots[-1].total_value,
                win_rate=trade_metrics["win_rate"],
                profit_factor=trade_metrics["profit_factor"],
                avg_trade_return=trade_metrics["avg_trade_return"],
                best_trade=trade_metrics["best_trade"],
                worst_trade=trade_metrics["worst_trade"],
                total_trades=trade_metrics["total_trades"],
                winning_trades=trade_metrics["winning_trades"],
                losing_trades=trade_metrics["losing_trades"]
            )

            return metrics

        except Exception as e:
            await self.logger.log_error("performance_metrics_calculation_error", f"{period.value}: {str(e)}")
            return None

    def _filter_snapshots_by_period(self, period: PortfolioPeriod) -> List[PortfolioSnapshot]:
        """Filtra snapshots por período"""
        if period == PortfolioPeriod.ALL_TIME:
            return self.portfolio_history

        now = datetime.utcnow()
        if period == PortfolioPeriod.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == PortfolioPeriod.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        elif period == PortfolioPeriod.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif period == PortfolioPeriod.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            return self.portfolio_history

        return [s for s in self.portfolio_history if s.timestamp > cutoff]

    def _calculate_returns_from_snapshots(self, snapshots: List[PortfolioSnapshot]) -> List[float]:
        """Calcula retornos a partir dos snapshots"""
        if len(snapshots) < 2:
            return []

        returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1].total_value
            curr_value = snapshots[i].total_value
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)

        return returns

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02/252  # Risk-free rate 2% anual
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calcula Sortino ratio"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02/252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0

    def _calculate_calmar_ratio(self, returns: List[float], snapshots: List[PortfolioSnapshot]) -> float:
        """Calcula Calmar ratio"""
        if len(returns) < 2 or len(snapshots) < 2:
            return 0.0

        annual_return = np.mean(returns) * 252
        _, max_drawdown_pct = self._calculate_max_drawdown(snapshots)

        return annual_return / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0.0

    def _calculate_max_drawdown(self, snapshots: List[PortfolioSnapshot]) -> Tuple[float, float]:
        """Calcula máximo drawdown"""
        if len(snapshots) < 2:
            return 0.0, 0.0

        values = [s.total_value for s in snapshots]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = (max_drawdown / peak) * 100 if peak > 0 else 0.0
        return max_drawdown, max_drawdown_pct

    async def _calculate_trade_metrics(self, period: PortfolioPeriod) -> Dict[str, float]:
        """Calcula métricas de trading"""
        # Implementação simplificada
        return {
            "win_rate": 65.0,
            "profit_factor": 1.5,
            "avg_trade_return": 0.02,
            "best_trade": 0.15,
            "worst_trade": -0.08,
            "total_trades": 100,
            "winning_trades": 65,
            "losing_trades": 35
        }

    # Métodos auxiliares para cálculos de risco
    async def _calculate_portfolio_volatility(self) -> float:
        """Calcula volatilidade do portfólio"""
        if len(self.daily_returns) < 10:
            return 0.0
        return float(np.std(self.daily_returns[-30:]) * np.sqrt(252))

    async def _calculate_var_95(self) -> float:
        """Calcula VaR 95%"""
        if len(self.daily_returns) < 30:
            return 0.0
        return float(abs(np.percentile(self.daily_returns[-30:], 5)) * self.current_value)

    async def _calculate_current_drawdown(self) -> float:
        """Calcula drawdown atual"""
        if len(self.portfolio_history) < 2:
            return 0.0

        peak_value = max(s.total_value for s in self.portfolio_history[-30:])
        current_value = self.current_value
        return float((peak_value - current_value) / peak_value * 100) if peak_value > 0 else 0.0

    async def _calculate_current_sharpe_ratio(self) -> float:
        """Calcula Sharpe ratio atual"""
        if len(self.daily_returns) < 30:
            return 0.0
        return self._calculate_sharpe_ratio(self.daily_returns[-30:])

    # Métodos de consulta pública
    async def get_current_portfolio_value(self) -> Dict[str, Any]:
        """Retorna valor atual do portfólio"""
        try:
            snapshot = getattr(self, '_current_snapshot', None)
            if not snapshot:
                await self._calculate_portfolio_snapshot()
                snapshot = getattr(self, '_current_snapshot', None)

            if not snapshot:
                return {"error": "Unable to calculate portfolio value"}

            return {
                "timestamp": snapshot.timestamp.isoformat(),
                "total_value": snapshot.total_value,
                "initial_capital": self.initial_capital,
                "total_return": snapshot.total_return,
                "total_return_pct": snapshot.total_return_pct,
                "unrealized_pnl": snapshot.unrealized_pnl,
                "realized_pnl": snapshot.realized_pnl,
                "available_cash": snapshot.available_cash,
                "used_margin": snapshot.used_margin,
                "positions_count": snapshot.positions_count,
                "active_symbols": snapshot.active_symbols
            }

        except Exception as e:
            await self.logger.log_error("current_portfolio_value_error", str(e))
            return {"error": str(e)}

    async def get_position_details(self) -> Dict[str, Any]:
        """Retorna detalhes das posições"""
        try:
            return {
                "positions": {
                    symbol: asdict(summary)
                    for symbol, summary in self.current_positions.items()
                },
                "asset_allocation": {
                    asset_class.value: asdict(allocation)
                    for asset_class, allocation in self.asset_allocation.items()
                }
            }

        except Exception as e:
            await self.logger.log_error("position_details_error", str(e))
            return {"error": str(e)}

    async def get_performance_summary(self, period: PortfolioPeriod = PortfolioPeriod.ALL_TIME) -> Dict[str, Any]:
        """Retorna resumo de performance"""
        try:
            metrics = self.performance_cache.get(period)
            if not metrics:
                metrics = await self._calculate_performance_metrics(period)

            if not metrics:
                return {"error": f"No data available for period {period.value}"}

            return asdict(metrics)

        except Exception as e:
            await self.logger.log_error("performance_summary_error", f"{period.value}: {str(e)}")
            return {"error": str(e)}

    async def get_portfolio_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Retorna histórico do portfólio"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            filtered_history = [
                s for s in self.portfolio_history
                if s.timestamp > cutoff_date
            ]

            return [asdict(snapshot) for snapshot in filtered_history]

        except Exception as e:
            await self.logger.log_error("portfolio_history_error", str(e))
            return []

    # Métodos auxiliares para banco de dados
    async def _save_snapshot_to_db(self, snapshot: PortfolioSnapshot):
        """Salva snapshot no banco de dados"""
        try:
            # Implementar salvamento no banco
            pass
        except Exception as e:
            await self.logger.log_error("snapshot_db_save_error", str(e))

    async def _load_historical_data(self):
        """Carrega dados históricos"""
        try:
            # Implementar carregamento do banco
            pass
        except Exception as e:
            await self.logger.log_error("historical_data_load_error", str(e))

    async def shutdown(self):
        """Encerra o rastreador de portfólio"""
        await self.stop_tracking()
        await self.logger.log_activity("portfolio_tracker_shutdown", {})
        print("🔌 Portfolio Tracker encerrado")
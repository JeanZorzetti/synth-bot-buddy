"""
Parameter Optimizer Module

Grid Search algorithm to find optimal trading parameters (SL/TP/Timeout)
for each symbol by testing different combinations and maximizing Sharpe Ratio.
"""
import logging
import itertools
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de uma combinação de parâmetros testada"""
    stop_loss_pct: float
    take_profit_pct: float
    timeout_minutes: int
    total_trades: int
    win_rate: float
    total_profit_loss: float
    profit_loss_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_duration_minutes: float
    timeout_rate: float
    score: float  # Métrica combinada para ranking


@dataclass
class TradeSimulation:
    """Trade simulado com parâmetros específicos"""
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    profit_loss: float
    profit_loss_pct: float
    is_winner: bool
    exit_reason: str


class ParameterOptimizer:
    """
    Otimizador de Parâmetros usando Grid Search

    Testa diferentes combinações de Stop Loss, Take Profit e Timeout
    para encontrar os parâmetros que maximizam o Sharpe Ratio.
    """

    def __init__(
        self,
        historical_trades: List[Dict[str, Any]],
        initial_capital: float = 10000.0
    ):
        """
        Args:
            historical_trades: Lista de trades históricos do Forward Testing
            initial_capital: Capital inicial para simulações
        """
        self.historical_trades = historical_trades
        self.initial_capital = initial_capital

        # Define o grid de parâmetros a serem testados
        self.stop_loss_grid = [0.5, 0.7, 1.0, 1.5, 2.0]  # %
        self.take_profit_grid = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # %
        self.timeout_grid = [3, 5, 10, 15, 30]  # minutes

        logger.info(f"🔍 ParameterOptimizer initialized with {len(historical_trades)} historical trades")

    def optimize(
        self,
        symbol: str = None,
        top_n: int = 10
    ) -> List[OptimizationResult]:
        """
        Executa Grid Search para encontrar os melhores parâmetros

        Args:
            symbol: Ativo específico (None = todos)
            top_n: Número de melhores resultados a retornar

        Returns:
            Lista dos top_n melhores resultados ordenados por score
        """
        logger.info(f"🚀 Starting Grid Search for symbol: {symbol or 'ALL'}")

        # Filtra trades por símbolo se especificado
        trades_to_use = self.historical_trades
        if symbol:
            trades_to_use = [t for t in self.historical_trades if t.get('symbol') == symbol]

        if not trades_to_use:
            logger.warning(f"⚠️ No trades found for symbol: {symbol}")
            return []

        logger.info(f"📊 Testing {len(trades_to_use)} trades with parameter grid:")
        logger.info(f"   SL: {self.stop_loss_grid}")
        logger.info(f"   TP: {self.take_profit_grid}")
        logger.info(f"   Timeout: {self.timeout_grid}")

        # Gera todas as combinações de parâmetros
        param_combinations = list(itertools.product(
            self.stop_loss_grid,
            self.take_profit_grid,
            self.timeout_grid
        ))

        total_combinations = len(param_combinations)
        logger.info(f"🔢 Total combinations to test: {total_combinations}")

        # Testa cada combinação
        results = []
        for idx, (sl, tp, timeout) in enumerate(param_combinations):
            if (idx + 1) % 10 == 0:
                logger.info(f"⏳ Progress: {idx + 1}/{total_combinations} ({(idx + 1)/total_combinations*100:.1f}%)")

            result = self._evaluate_parameters(
                trades_to_use, sl, tp, timeout
            )
            results.append(result)

        # Ordena por score (descendente)
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(f"✅ Grid Search completed! Best Sharpe Ratio: {results[0].sharpe_ratio:.3f}")
        logger.info(f"   Best params: SL={results[0].stop_loss_pct}% | TP={results[0].take_profit_pct}% | Timeout={results[0].timeout_minutes}min")

        return results[:top_n]

    def _evaluate_parameters(
        self,
        trades: List[Dict[str, Any]],
        stop_loss_pct: float,
        take_profit_pct: float,
        timeout_minutes: int
    ) -> OptimizationResult:
        """
        Avalia uma combinação específica de parâmetros

        Simula os trades com os novos parâmetros e calcula métricas.
        """
        simulated_trades = []

        for trade in trades:
            # Simula o trade com os novos parâmetros
            sim_trade = self._simulate_trade(
                trade, stop_loss_pct, take_profit_pct, timeout_minutes
            )
            simulated_trades.append(sim_trade)

        # Calcula métricas agregadas
        metrics = self._calculate_metrics(simulated_trades)

        # Calcula score combinado (peso maior para Sharpe)
        score = (
            metrics['sharpe_ratio'] * 0.5 +
            metrics['win_rate'] * 0.2 +
            metrics['profit_loss_pct'] * 0.2 -
            metrics['max_drawdown_pct'] * 0.1
        )

        return OptimizationResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            timeout_minutes=timeout_minutes,
            total_trades=len(simulated_trades),
            win_rate=metrics['win_rate'],
            total_profit_loss=metrics['total_profit_loss'],
            profit_loss_pct=metrics['profit_loss_pct'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            avg_trade_duration_minutes=metrics['avg_trade_duration_minutes'],
            timeout_rate=metrics['timeout_rate'],
            score=score
        )

    def _simulate_trade(
        self,
        original_trade: Dict[str, Any],
        stop_loss_pct: float,
        take_profit_pct: float,
        timeout_minutes: int
    ) -> TradeSimulation:
        """
        Simula um trade com novos parâmetros SL/TP/Timeout

        Baseado no preço de entrada, tipo de posição e movimento de preço,
        determina onde o trade teria fechado com os novos parâmetros.
        """
        entry_price = original_trade['entry_price']
        entry_time = datetime.fromisoformat(original_trade['entry_time'])
        position_type = original_trade['position_type']

        # Calcula níveis de SL e TP
        if position_type == 'LONG':
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            take_profit = entry_price * (1 + take_profit_pct / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            take_profit = entry_price * (1 - take_profit_pct / 100)

        # Calcula tempo máximo
        max_exit_time = entry_time + timedelta(minutes=timeout_minutes)

        # Determina o exit real baseado nos limites
        original_exit_price = original_trade['exit_price']
        original_exit_time = datetime.fromisoformat(original_trade['exit_time'])

        exit_price = original_exit_price
        exit_time = original_exit_time
        exit_reason = original_trade.get('exit_reason', 'MANUAL')

        # Verifica se atingiu SL ou TP primeiro
        if position_type == 'LONG':
            if original_exit_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
            elif original_exit_price >= take_profit:
                exit_price = take_profit
                exit_reason = 'TAKE_PROFIT'
        else:  # SHORT
            if original_exit_price >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
            elif original_exit_price <= take_profit:
                exit_price = take_profit
                exit_reason = 'TAKE_PROFIT'

        # Verifica timeout
        if original_exit_time >= max_exit_time:
            exit_time = max_exit_time
            exit_reason = 'TIMEOUT'
            # Usa o preço que teria no timeout (aproximação: preço original)
            exit_price = original_exit_price

        # Calcula P&L
        if position_type == 'LONG':
            profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100

        profit_loss = (self.initial_capital * profit_loss_pct) / 100
        is_winner = profit_loss > 0

        return TradeSimulation(
            entry_price=entry_price,
            entry_time=entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            is_winner=is_winner,
            exit_reason=exit_reason
        )

    def _calculate_metrics(self, trades: List[TradeSimulation]) -> Dict[str, float]:
        """Calcula métricas agregadas dos trades simulados"""
        if not trades:
            return {
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'profit_loss_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'avg_trade_duration_minutes': 0.0,
                'timeout_rate': 0.0
            }

        # Win Rate
        winners = sum(1 for t in trades if t.is_winner)
        win_rate = (winners / len(trades)) * 100

        # P&L Total
        total_profit_loss = sum(t.profit_loss for t in trades)
        profit_loss_pct = (total_profit_loss / self.initial_capital) * 100

        # Sharpe Ratio
        returns = [t.profit_loss_pct for t in trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max Drawdown
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0.0

        for trade in trades:
            capital += trade.profit_loss
            if capital > peak_capital:
                peak_capital = capital
            drawdown = ((peak_capital - capital) / peak_capital) * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Duração Média
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
        avg_duration = np.mean(durations) if durations else 0.0

        # Timeout Rate
        timeouts = sum(1 for t in trades if t.exit_reason == 'TIMEOUT')
        timeout_rate = (timeouts / len(trades)) * 100

        return {
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_trade_duration_minutes': avg_duration,
            'timeout_rate': timeout_rate
        }

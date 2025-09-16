"""
Real Position Manager - Gerenciamento Real de Posições
Sistema avançado para gerenciamento de posições reais com stop loss dinâmico, trailing stops e análise de risco.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from real_deriv_client import RealDerivWebSocketClient
from real_tick_processor import ProcessedTickData
from real_trading_executor import TradingPosition, TradeStatus, RealTradeResult

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    RISK_LIMIT = "risk_limit"
    AI_SIGNAL = "ai_signal"

class TrailingStopType(Enum):
    FIXED_DISTANCE = "fixed_distance"
    PERCENTAGE = "percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"

@dataclass
class PositionRiskMetrics:
    """Métricas de risco de posição"""
    var_1d: float  # Value at Risk 1 dia
    expected_shortfall: float  # Expected Shortfall
    max_drawdown: float  # Máximo drawdown
    sharpe_ratio: float  # Sharpe ratio
    calmar_ratio: float  # Calmar ratio
    volatility: float  # Volatilidade
    beta: float  # Beta relativo ao mercado
    correlation: float  # Correlação com índice de referência

@dataclass
class TrailingStopConfig:
    """Configuração de trailing stop"""
    enabled: bool = False
    stop_type: TrailingStopType = TrailingStopType.PERCENTAGE
    distance: float = 0.02  # 2% por padrão
    min_profit_threshold: float = 0.01  # 1% de lucro mínimo para ativar
    update_frequency_seconds: int = 5
    max_trail_distance: float = 0.10  # 10% máximo

@dataclass
class PositionExitConditions:
    """Condições de saída de posição"""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_time_minutes: Optional[int] = None
    max_drawdown_pct: Optional[float] = None
    min_profit_target: Optional[float] = None
    trailing_stop: Optional[TrailingStopConfig] = None

@dataclass
class EnhancedPosition(TradingPosition):
    """Posição aprimorada com recursos avançados"""
    risk_metrics: Optional[PositionRiskMetrics] = None
    exit_conditions: Optional[PositionExitConditions] = None
    trailing_stop_price: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    price_history: List[Tuple[datetime, float]] = None
    last_update: Optional[datetime] = None

    def __post_init__(self):
        super().__post_init__()
        if self.price_history is None:
            self.price_history = []
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price

class RealPositionManager:
    """Gerenciador avançado de posições reais"""

    def __init__(self, deriv_client: RealDerivWebSocketClient):
        self.deriv_client = deriv_client

        # Posições gerenciadas
        self.managed_positions: Dict[str, EnhancedPosition] = {}
        self.closed_positions: List[EnhancedPosition] = []

        # Configurações
        self.default_exit_conditions = PositionExitConditions(
            max_time_minutes=60,  # 1 hora máximo
            max_drawdown_pct=0.05,  # 5% drawdown máximo
            trailing_stop=TrailingStopConfig(enabled=True)
        )

        # Estado de monitoramento
        self.is_monitoring = False
        self.monitoring_task = None
        self.update_frequency = 1  # segundos

        # Métricas agregadas
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.max_concurrent_positions = 0
        self.average_holding_time = 0.0

        # Thread pool para cálculos
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        logger.info("RealPositionManager inicializado")

    async def add_position(self, position: TradingPosition,
                          exit_conditions: PositionExitConditions = None) -> bool:
        """Adiciona uma posição para gerenciamento"""
        try:
            # Criar posição aprimorada
            enhanced_position = EnhancedPosition(
                contract_id=position.contract_id,
                symbol=position.symbol,
                trade_type=position.trade_type,
                amount=position.amount,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                exit_conditions=exit_conditions or self.default_exit_conditions,
                last_update=datetime.now()
            )

            # Adicionar preço inicial ao histórico
            enhanced_position.price_history.append(
                (enhanced_position.entry_time, enhanced_position.entry_price)
            )

            # Calcular métricas de risco iniciais
            enhanced_position.risk_metrics = await self._calculate_risk_metrics(enhanced_position)

            self.managed_positions[position.contract_id] = enhanced_position

            # Atualizar estatísticas
            self.max_concurrent_positions = max(
                self.max_concurrent_positions,
                len(self.managed_positions)
            )

            logger.info(f"Posição {position.contract_id} adicionada ao gerenciamento")

            # Iniciar monitoramento se não estiver ativo
            if not self.is_monitoring:
                await self.start_monitoring()

            return True

        except Exception as e:
            logger.error(f"Erro ao adicionar posição {position.contract_id}: {e}")
            return False

    async def start_monitoring(self):
        """Inicia o monitoramento de posições"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoramento de posições iniciado")

    async def stop_monitoring(self):
        """Para o monitoramento de posições"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Monitoramento de posições parado")

    async def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.is_monitoring:
            try:
                # Atualizar todas as posições
                await self._update_all_positions()

                # Verificar condições de saída
                await self._check_exit_conditions()

                # Atualizar trailing stops
                await self._update_trailing_stops()

                # Calcular métricas agregadas
                await self._update_aggregate_metrics()

                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                await asyncio.sleep(5)

    async def _update_all_positions(self):
        """Atualiza preços de todas as posições"""
        for contract_id, position in self.managed_positions.items():
            try:
                # Obter preço atual
                current_price = await self._get_current_price(position.symbol)
                if current_price is None:
                    continue

                # Atualizar posição
                position.current_price = current_price
                position.last_update = datetime.now()

                # Atualizar histórico de preços
                position.price_history.append((datetime.now(), current_price))

                # Manter apenas últimos 1000 pontos
                if len(position.price_history) > 1000:
                    position.price_history = position.price_history[-1000:]

                # Atualizar extremos
                position.highest_price = max(position.highest_price or 0, current_price)
                position.lowest_price = min(position.lowest_price or float('inf'), current_price)

                # Calcular PnL não realizado
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)

                # Atualizar métricas de risco
                position.risk_metrics = await self._calculate_risk_metrics(position)

            except Exception as e:
                logger.error(f"Erro ao atualizar posição {contract_id}: {e}")

    async def _check_exit_conditions(self):
        """Verifica condições de saída para todas as posições"""
        positions_to_close = []

        for contract_id, position in self.managed_positions.items():
            try:
                exit_reason = await self._should_exit_position(position)
                if exit_reason:
                    positions_to_close.append((contract_id, exit_reason))

            except Exception as e:
                logger.error(f"Erro ao verificar condições de saída para {contract_id}: {e}")

        # Fechar posições que atingiram condições de saída
        for contract_id, exit_reason in positions_to_close:
            await self._close_position(contract_id, exit_reason)

    async def _should_exit_position(self, position: EnhancedPosition) -> Optional[PositionExitReason]:
        """Verifica se uma posição deve ser fechada"""
        if not position.exit_conditions or not position.current_price:
            return None

        conditions = position.exit_conditions
        current_price = position.current_price

        # Stop Loss
        if (conditions.stop_loss and
            ((position.trade_type.value == "buy" and current_price <= conditions.stop_loss) or
             (position.trade_type.value == "sell" and current_price >= conditions.stop_loss))):
            return PositionExitReason.STOP_LOSS

        # Take Profit
        if (conditions.take_profit and
            ((position.trade_type.value == "buy" and current_price >= conditions.take_profit) or
             (position.trade_type.value == "sell" and current_price <= conditions.take_profit))):
            return PositionExitReason.TAKE_PROFIT

        # Trailing Stop
        if (position.trailing_stop_price and
            ((position.trade_type.value == "buy" and current_price <= position.trailing_stop_price) or
             (position.trade_type.value == "sell" and current_price >= position.trailing_stop_price))):
            return PositionExitReason.TRAILING_STOP

        # Tempo máximo
        if conditions.max_time_minutes:
            time_elapsed = (datetime.now() - position.entry_time).total_seconds() / 60
            if time_elapsed >= conditions.max_time_minutes:
                return PositionExitReason.TIME_BASED

        # Drawdown máximo
        if conditions.max_drawdown_pct:
            if position.trade_type.value == "buy":
                drawdown = (position.highest_price - current_price) / position.highest_price
            else:
                drawdown = (current_price - position.lowest_price) / position.lowest_price

            if drawdown >= conditions.max_drawdown_pct:
                return PositionExitReason.RISK_LIMIT

        return None

    async def _update_trailing_stops(self):
        """Atualiza trailing stops para posições elegíveis"""
        for position in self.managed_positions.values():
            try:
                if (not position.exit_conditions or
                    not position.exit_conditions.trailing_stop or
                    not position.exit_conditions.trailing_stop.enabled or
                    not position.current_price):
                    continue

                await self._update_position_trailing_stop(position)

            except Exception as e:
                logger.error(f"Erro ao atualizar trailing stop para {position.contract_id}: {e}")

    async def _update_position_trailing_stop(self, position: EnhancedPosition):
        """Atualiza trailing stop de uma posição específica"""
        trailing_config = position.exit_conditions.trailing_stop
        current_price = position.current_price

        # Verificar se atingiu lucro mínimo para ativar trailing
        current_pnl_pct = abs(position.unrealized_pnl or 0) / position.amount
        if current_pnl_pct < trailing_config.min_profit_threshold:
            return

        # Calcular nova posição do trailing stop
        if trailing_config.stop_type == TrailingStopType.PERCENTAGE:
            distance = trailing_config.distance
        elif trailing_config.stop_type == TrailingStopType.ATR_BASED:
            distance = await self._calculate_atr_distance(position)
        elif trailing_config.stop_type == TrailingStopType.VOLATILITY_BASED:
            distance = await self._calculate_volatility_distance(position)
        else:
            distance = trailing_config.distance

        # Limitar distância máxima
        distance = min(distance, trailing_config.max_trail_distance)

        if position.trade_type.value == "buy":
            # Para posições compradas, trailing stop sobe com o preço
            new_trailing_stop = current_price * (1 - distance)
            if (position.trailing_stop_price is None or
                new_trailing_stop > position.trailing_stop_price):
                position.trailing_stop_price = new_trailing_stop
        else:
            # Para posições vendidas, trailing stop desce com o preço
            new_trailing_stop = current_price * (1 + distance)
            if (position.trailing_stop_price is None or
                new_trailing_stop < position.trailing_stop_price):
                position.trailing_stop_price = new_trailing_stop

    async def _calculate_atr_distance(self, position: EnhancedPosition) -> float:
        """Calcula distância baseada no ATR"""
        if len(position.price_history) < 14:
            return 0.02  # 2% padrão

        # Calcular True Range dos últimos 14 períodos
        prices = [price for _, price in position.price_history[-14:]]
        true_ranges = []

        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            true_ranges.append(high_low)

        if not true_ranges:
            return 0.02

        atr = np.mean(true_ranges)
        return min(atr / position.entry_price * 2, 0.05)  # Máximo 5%

    async def _calculate_volatility_distance(self, position: EnhancedPosition) -> float:
        """Calcula distância baseada na volatilidade"""
        if len(position.price_history) < 20:
            return 0.02

        # Calcular volatilidade dos retornos
        prices = [price for _, price in position.price_history[-20:]]
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]

        if not returns:
            return 0.02

        volatility = np.std(returns)
        return min(volatility * 2, 0.08)  # Máximo 8%

    async def _close_position(self, contract_id: str, exit_reason: PositionExitReason):
        """Fecha uma posição"""
        try:
            position = self.managed_positions.get(contract_id)
            if not position:
                return

            # Enviar comando de fechamento
            response = await self.deriv_client.send_request({
                "sell": contract_id,
                "price": 0  # Preço de mercado
            })

            if response and "sell" in response:
                # Atualizar posição com dados finais
                position.last_update = datetime.now()

                # Mover para posições fechadas
                self.closed_positions.append(position)
                del self.managed_positions[contract_id]

                # Atualizar PnL realizado
                if position.unrealized_pnl:
                    self.total_realized_pnl += position.unrealized_pnl

                logger.info(f"Posição {contract_id} fechada: {exit_reason.value}")

            else:
                logger.error(f"Falha ao fechar posição {contract_id}")

        except Exception as e:
            logger.error(f"Erro ao fechar posição {contract_id}: {e}")

    async def _calculate_risk_metrics(self, position: EnhancedPosition) -> PositionRiskMetrics:
        """Calcula métricas de risco para uma posição"""
        try:
            if len(position.price_history) < 10:
                # Métricas padrão para posições novas
                return PositionRiskMetrics(
                    var_1d=0.0,
                    expected_shortfall=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    calmar_ratio=0.0,
                    volatility=0.0,
                    beta=1.0,
                    correlation=0.0
                )

            # Executar cálculos em thread pool
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._calculate_risk_metrics_sync,
                position
            )

        except Exception as e:
            logger.error(f"Erro ao calcular métricas de risco: {e}")
            return PositionRiskMetrics(0, 0, 0, 0, 0, 0, 1.0, 0)

    def _calculate_risk_metrics_sync(self, position: EnhancedPosition) -> PositionRiskMetrics:
        """Cálculos síncronos de métricas de risco"""
        prices = np.array([price for _, price in position.price_history])
        returns = np.diff(prices) / prices[:-1]

        if len(returns) == 0:
            return PositionRiskMetrics(0, 0, 0, 0, 0, 0, 1.0, 0)

        # VaR 1 dia (95% confiança)
        var_1d = np.percentile(returns, 5) * position.amount

        # Expected Shortfall
        tail_returns = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = np.mean(tail_returns) * position.amount if len(tail_returns) > 0 else 0

        # Máximo drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe ratio (anualizado)
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Calmar ratio
        calmar_ratio = (np.mean(returns) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0

        # Volatilidade anualizada
        volatility = np.std(returns) * np.sqrt(252)

        return PositionRiskMetrics(
            var_1d=var_1d,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            beta=1.0,  # Simplificado
            correlation=0.0  # Simplificado
        )

    async def _update_aggregate_metrics(self):
        """Atualiza métricas agregadas do portfólio"""
        # PnL não realizado total
        self.total_unrealized_pnl = sum(
            pos.unrealized_pnl or 0
            for pos in self.managed_positions.values()
        )

        # Tempo médio de manutenção
        if self.closed_positions:
            holding_times = [
                (pos.last_update - pos.entry_time).total_seconds() / 60
                for pos in self.closed_positions
                if pos.last_update
            ]
            self.average_holding_time = np.mean(holding_times) if holding_times else 0

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual de um símbolo"""
        try:
            response = await self.deriv_client.send_request({
                "ticks": symbol
            })

            if response and "tick" in response:
                return response["tick"]["quote"]

            return None

        except Exception as e:
            logger.error(f"Erro ao obter preço de {symbol}: {e}")
            return None

    def _calculate_unrealized_pnl(self, position: EnhancedPosition) -> float:
        """Calcula PnL não realizado"""
        if not position.current_price:
            return 0.0

        if position.trade_type.value == "buy":
            return (position.current_price - position.entry_price) * position.amount
        else:
            return (position.entry_price - position.current_price) * position.amount

    async def force_close_position(self, contract_id: str) -> bool:
        """Força o fechamento de uma posição"""
        await self._close_position(contract_id, PositionExitReason.MANUAL)
        return contract_id not in self.managed_positions

    async def force_close_all_positions(self) -> int:
        """Força o fechamento de todas as posições"""
        positions_to_close = list(self.managed_positions.keys())

        for contract_id in positions_to_close:
            await self._close_position(contract_id, PositionExitReason.EMERGENCY)

        return len(positions_to_close)

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Obtém resumo do portfólio"""
        active_positions = len(self.managed_positions)
        total_exposure = sum(pos.amount for pos in self.managed_positions.values())

        return {
            "active_positions": active_positions,
            "total_exposure": total_exposure,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "max_concurrent_positions": self.max_concurrent_positions,
            "average_holding_time_minutes": self.average_holding_time,
            "closed_positions_count": len(self.closed_positions)
        }

    async def get_position_details(self, contract_id: str) -> Optional[Dict]:
        """Obtém detalhes de uma posição específica"""
        position = self.managed_positions.get(contract_id)
        if not position:
            return None

        return {
            "position": asdict(position),
            "risk_metrics": asdict(position.risk_metrics) if position.risk_metrics else None,
            "exit_conditions": asdict(position.exit_conditions) if position.exit_conditions else None
        }

    async def shutdown(self):
        """Encerra o gerenciador de posições"""
        await self.stop_monitoring()
        await self.force_close_all_positions()
        self.thread_pool.shutdown(wait=True)
        logger.info("RealPositionManager encerrado")
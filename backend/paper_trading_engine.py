"""
Paper Trading Engine para simulação realista de trading

Este módulo implementa um motor de paper trading que simula:
- Latência de execução realista
- Slippage de mercado
- Gerenciamento de posições
- Cálculo de métricas em tempo real
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Tipo de posição"""
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    """Status da posição"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Representa uma posição de trading"""
    id: str
    symbol: str
    position_type: PositionType
    entry_price: float
    size: float
    entry_time: datetime
    status: PositionStatus
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    # Trailing Stop Loss
    trailing_stop_enabled: bool = False
    trailing_stop_distance_pct: float = 0.0  # Distância do trailing em %
    highest_price: Optional[float] = None  # Maior preço atingido (LONG)
    lowest_price: Optional[float] = None   # Menor preço atingido (SHORT)

    def to_dict(self):
        """Converte para dicionário"""
        data = asdict(self)
        data['position_type'] = self.position_type.value
        data['status'] = self.status.value
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return data

    def calculate_pnl(self, current_price: float) -> tuple[float, float]:
        """
        Calcula P&L atual da posição

        Returns:
            (profit_loss, profit_loss_pct)
        """
        if self.position_type == PositionType.LONG:
            pnl = (current_price - self.entry_price) * self.size
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.size
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100

        return pnl, pnl_pct


@dataclass
class Trade:
    """Representa um trade completo (posição fechada)"""
    id: str
    symbol: str
    position_type: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    profit_loss: float
    profit_loss_pct: float
    is_winner: bool
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit', 'timeout', 'manual'

    def to_dict(self):
        """Converte para dicionário"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'position_type': self.position_type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'is_winner': self.is_winner,
            'exit_reason': self.exit_reason
        }


class PaperTradingEngine:
    """
    Motor de Paper Trading com simulação realista

    Simula:
    - Latência de execução (~100ms)
    - Slippage de mercado (0.1%)
    - Comissões (opcional)
    - Gestão de posições
    - Cálculo de métricas em tempo real
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        execution_latency_ms: float = 100.0,
        slippage_pct: float = 0.1,
        commission_pct: float = 0.0,
        max_positions: int = 5
    ):
        """
        Inicializa o motor de paper trading

        Args:
            initial_capital: Capital inicial em USD
            execution_latency_ms: Latência de execução em ms
            slippage_pct: Slippage percentual
            commission_pct: Comissão percentual por trade
            max_positions: Máximo de posições simultâneas
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.execution_latency_ms = execution_latency_ms
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.max_positions = max_positions

        # Estado
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Métricas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital

        # Controle
        self.is_running = False
        self.start_time: Optional[datetime] = None

        logger.info(f"PaperTradingEngine inicializado com capital: ${initial_capital:,.2f}")

    def start(self):
        """Inicia sessão de paper trading"""
        self.is_running = True
        self.start_time = datetime.now()
        logger.info("Paper trading iniciado")

    def stop(self):
        """Para sessão de paper trading"""
        self.is_running = False

        # Fechar todas as posições abertas
        for position_id in list(self.positions.keys()):
            # Usar último preço conhecido ou preço de entrada
            position = self.positions[position_id]
            logger.warning(f"Fechando posição {position_id} ao parar sessão")
            # Fechar com preço de entrada (sem lucro/prejuízo)
            self.close_position(position_id, position.entry_price)

        logger.info("Paper trading parado")

    def reset(self):
        """Reseta o motor para estado inicial"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.equity_curve.clear()

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital

        self.is_running = False
        self.start_time = None

        logger.info("Paper trading resetado")

    async def execute_order(
        self,
        symbol: str,
        position_type: PositionType,
        size: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_enabled: bool = False,
        trailing_stop_distance_pct: float = 0.5
    ) -> Optional[Position]:
        """
        Executa uma ordem de compra/venda

        Args:
            symbol: Símbolo do ativo
            position_type: LONG ou SHORT
            size: Tamanho da posição em USD
            current_price: Preço atual do mercado
            stop_loss: Preço de stop loss (opcional)
            take_profit: Preço de take profit (opcional)
            trailing_stop_enabled: Se trailing stop está ativado
            trailing_stop_distance_pct: Distância do trailing em % (padrão: 0.5%)

        Returns:
            Position criada ou None se falhou
        """
        if not self.is_running:
            logger.error("Paper trading não está rodando")
            return None

        # Verificar limite de posições
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Limite de {self.max_positions} posições atingido")
            return None

        # Verificar capital disponível
        if size > self.capital:
            logger.warning(f"Capital insuficiente: ${size:.2f} > ${self.capital:.2f}")
            return None

        # Simular latência de execução
        await asyncio.sleep(self.execution_latency_ms / 1000)

        # Aplicar slippage (pior preço para o trader)
        if position_type == PositionType.LONG:
            executed_price = current_price * (1 + self.slippage_pct / 100)
        else:  # SHORT
            executed_price = current_price * (1 - self.slippage_pct / 100)

        # Aplicar comissão
        commission = size * (self.commission_pct / 100)

        # Criar posição
        position_id = f"{symbol}_{int(time.time() * 1000)}"
        position = Position(
            id=position_id,
            symbol=symbol,
            position_type=position_type,
            entry_price=executed_price,
            size=size - commission,  # Deduzir comissão do tamanho
            entry_time=datetime.now(),
            status=PositionStatus.OPEN,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_enabled=trailing_stop_enabled,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            highest_price=executed_price if trailing_stop_enabled else None,
            lowest_price=executed_price if trailing_stop_enabled else None
        )

        # Atualizar capital
        self.capital -= size

        # Registrar posição
        self.positions[position_id] = position

        logger.info(
            f"Posição aberta: {position_id} | {position_type.value} | "
            f"${size:.2f} @ ${executed_price:.4f} | Slippage: {self.slippage_pct}%"
        )

        return position

    def close_position(self, position_id: str, current_price: float, exit_reason: str = 'manual') -> Optional[Trade]:
        """
        Fecha uma posição existente

        Args:
            position_id: ID da posição a fechar
            current_price: Preço atual do mercado
            exit_reason: Razão do fechamento ('stop_loss', 'take_profit', 'timeout', 'manual')

        Returns:
            Trade criado ou None se falhou
        """
        if position_id not in self.positions:
            logger.error(f"Posição {position_id} não encontrada")
            return None

        position = self.positions[position_id]

        # Aplicar slippage (pior preço para o trader)
        if position.position_type == PositionType.LONG:
            executed_price = current_price * (1 - self.slippage_pct / 100)
        else:  # SHORT
            executed_price = current_price * (1 + self.slippage_pct / 100)

        # Calcular P&L
        pnl, pnl_pct = position.calculate_pnl(executed_price)

        # Aplicar comissão de saída
        commission = position.size * (self.commission_pct / 100)
        pnl -= commission

        # Atualizar posição
        position.exit_price = executed_price
        position.exit_time = datetime.now()
        position.profit_loss = pnl
        position.profit_loss_pct = pnl_pct
        position.status = PositionStatus.CLOSED

        # Atualizar capital
        self.capital += position.size + pnl

        # Criar trade
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            position_type=position.position_type.value,
            entry_price=position.entry_price,
            exit_price=executed_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            profit_loss=pnl,
            profit_loss_pct=pnl_pct,
            is_winner=pnl > 0,
            exit_reason=exit_reason
        )

        # Atualizar métricas
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # Atualizar drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        current_dd = ((self.peak_capital - self.capital) / self.peak_capital) * 100
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        # Registrar trade
        self.trade_history.append(trade)

        # Registrar ponto na equity curve
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'capital': round(self.capital, 2),
            'profit_loss': round(pnl, 2),
            'trade_id': trade.id
        })

        # Remover posição
        del self.positions[position_id]

        logger.info(
            f"Posição fechada: {position_id} | "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Capital: ${self.capital:.2f}"
        )

        return trade

    def _update_trailing_stop(self, position: Position, current_price: float):
        """
        Atualiza trailing stop loss para uma posição

        Args:
            position: Posição a atualizar
            current_price: Preço atual do ativo
        """
        if position.position_type == PositionType.LONG:
            # Inicializar highest_price se não existir
            if position.highest_price is None:
                position.highest_price = position.entry_price

            # Atualizar highest_price se preço subiu
            if current_price > position.highest_price:
                old_highest = position.highest_price
                position.highest_price = current_price

                # Calcular novo stop loss baseado no highest_price
                new_stop_loss = position.highest_price * (1 - position.trailing_stop_distance_pct / 100)

                # Só mover SL para cima (nunca para baixo)
                if position.stop_loss is None or new_stop_loss > position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_stop_loss

                    logger.info(f"📈 Trailing SL movido (LONG): ${old_stop_loss:.5f if old_stop_loss else 0:.5f} → ${new_stop_loss:.5f}")
                    logger.info(f"   Highest: ${old_highest:.5f} → ${position.highest_price:.5f}")
                    logger.info(f"   Current: ${current_price:.5f}")

        else:  # SHORT
            # Inicializar lowest_price se não existir
            if position.lowest_price is None:
                position.lowest_price = position.entry_price

            # Atualizar lowest_price se preço caiu
            if current_price < position.lowest_price:
                old_lowest = position.lowest_price
                position.lowest_price = current_price

                # Calcular novo stop loss baseado no lowest_price
                new_stop_loss = position.lowest_price * (1 + position.trailing_stop_distance_pct / 100)

                # Só mover SL para baixo (nunca para cima)
                if position.stop_loss is None or new_stop_loss < position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_stop_loss

                    logger.info(f"📉 Trailing SL movido (SHORT): ${old_stop_loss:.5f if old_stop_loss else 0:.5f} → ${new_stop_loss:.5f}")
                    logger.info(f"   Lowest: ${old_lowest:.5f} → ${position.lowest_price:.5f}")
                    logger.info(f"   Current: ${current_price:.5f}")

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Atualiza todas as posições com preços atuais
        Verifica stop loss e take profit

        Args:
            current_prices: Dict com preços atuais por símbolo
        """
        for position_id, position in list(self.positions.items()):
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]

            # LOG DETALHADO: Mostrar posição e preços
            logger.info(f"🔍 Verificando posição {position_id[-8:]}:")
            logger.info(f"   Tipo: {position.position_type.value} | Entry: ${position.entry_price:.5f} | Current: ${current_price:.5f}")
            logger.info(f"   SL: ${position.stop_loss:.5f} | TP: ${position.take_profit:.5f}")

            # Atualizar trailing stop loss se habilitado
            if position.trailing_stop_enabled:
                self._update_trailing_stop(position, current_price)

            # Verificar stop loss
            if position.stop_loss:
                if position.position_type == PositionType.LONG and current_price <= position.stop_loss:
                    logger.info(f"🛑 Stop loss atingido para {position_id[-8:]}: {current_price:.5f} <= {position.stop_loss:.5f}")
                    self.close_position(position_id, current_price, exit_reason='stop_loss')
                    continue
                elif position.position_type == PositionType.SHORT and current_price >= position.stop_loss:
                    logger.info(f"🛑 Stop loss atingido para {position_id[-8:]}: {current_price:.5f} >= {position.stop_loss:.5f}")
                    self.close_position(position_id, current_price, exit_reason='stop_loss')
                    continue

            # Verificar take profit
            if position.take_profit:
                if position.position_type == PositionType.LONG and current_price >= position.take_profit:
                    logger.info(f"🎯 Take profit atingido para {position_id[-8:]}: {current_price:.5f} >= {position.take_profit:.5f}")
                    self.close_position(position_id, current_price, exit_reason='take_profit')
                    continue
                elif position.position_type == PositionType.SHORT and current_price <= position.take_profit:
                    logger.info(f"🎯 Take profit atingido para {position_id[-8:]}: {current_price:.5f} <= {position.take_profit:.5f}")
                    self.close_position(position_id, current_price, exit_reason='take_profit')
                    continue

    def get_metrics(self) -> Dict:
        """
        Retorna métricas atuais do paper trading

        Returns:
            Dict com métricas detalhadas
        """
        # Calcular métricas básicas
        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Calcular profit factor
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0

        # Calcular Sharpe Ratio (simplificado)
        if len(self.trade_history) > 1:
            returns = [t.profit_loss_pct for t in self.trade_history]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Tempo de operação
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime_seconds,
            'initial_capital': self.initial_capital,
            'current_capital': round(self.capital, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'peak_capital': round(self.peak_capital, 2),
            'max_drawdown_pct': round(self.max_drawdown, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'open_positions': len(self.positions),
            'avg_profit_per_trade': round(total_pnl / self.total_trades, 2) if self.total_trades > 0 else 0
        }

    def get_open_positions(self) -> List[Dict]:
        """Retorna lista de posições abertas"""
        return [pos.to_dict() for pos in self.positions.values()]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Retorna histórico de trades"""
        recent_trades = self.trade_history[-limit:] if limit else self.trade_history
        return [trade.to_dict() for trade in reversed(recent_trades)]

    def get_equity_curve(self) -> List[Dict]:
        """Retorna equity curve completa"""
        return self.equity_curve

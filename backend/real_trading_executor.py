"""
Real Trading Executor - Execução Real de Trades
Sistema completo para execução de trades reais na API Deriv com validação, monitoramento e controle de risco.
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
import threading

from real_deriv_client import RealDerivWebSocketClient, TickData as RealTickData
from real_tick_processor import RealTickProcessor, ProcessedTickData
from risk_management import RiskManager, RiskMetrics
from autonomous_trading_engine import TradingDecision, DecisionType

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"

class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"

class ContractType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    RISE_FALL = "RISEFALL"
    HIGHER_LOWER = "HIGHERLOWER"
    TOUCH_NO_TOUCH = "TOUCHNOTOUCH"
    ENDS_BETWEEN = "ENDSBETWEEN"
    STAYS_BETWEEN = "STAYSBETWEEN"

@dataclass
class RealTradeRequest:
    """Requisição de trade real"""
    symbol: str
    trade_type: TradeType
    contract_type: ContractType
    amount: float
    duration: int
    duration_unit: str = "s"  # s, m, h, d
    barrier: Optional[float] = None
    barrier2: Optional[float] = None
    currency: str = "USD"
    trading_decision_id: Optional[str] = None
    ai_confidence: Optional[float] = None
    expected_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class RealTradeResult:
    """Resultado de trade real"""
    trade_id: str
    contract_id: Optional[str]
    status: TradeStatus
    symbol: str
    trade_type: TradeType
    amount: float
    entry_price: Optional[float]
    exit_price: Optional[float]
    profit_loss: Optional[float]
    execution_time: datetime
    completion_time: Optional[datetime]
    error_message: Optional[str] = None
    commission: Optional[float] = None
    swap: Optional[float] = None

@dataclass
class TradingPosition:
    """Posição de trading ativa"""
    contract_id: str
    symbol: str
    trade_type: TradeType
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class TradingSession:
    """Sessão de trading"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_trades: int = 0
    successful_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    active_positions: List[TradingPosition] = None

    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = []

class RealTradingExecutor:
    """Executor de Trading Real com integração completa à API Deriv"""

    def __init__(self, app_id: str, api_token: str = None):
        self.app_id = app_id
        self.api_token = api_token

        # Clientes e processadores
        self.deriv_client = RealDerivWebSocketClient(app_id, api_token)
        self.tick_processor = RealTickProcessor()
        self.risk_manager = RiskManager()

        # Estado de trading
        self.is_trading_enabled = False
        self.trading_session: Optional[TradingSession] = None
        self.active_positions: Dict[str, TradingPosition] = {}
        self.trade_history: List[RealTradeResult] = []
        self.pending_orders: Dict[str, RealTradeRequest] = {}

        # Configurações de trading
        self.max_concurrent_trades = 5
        self.max_daily_loss = 1000.0  # USD
        self.max_position_size = 100.0  # USD
        self.min_confidence_threshold = 0.65

        # Controle de risco em tempo real
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.last_trade_time = None
        self.trade_cooldown_seconds = 30

        # Threading e async
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.monitoring_task = None
        self.position_update_task = None

        # Locks para thread safety
        self.trading_lock = threading.Lock()
        self.position_lock = threading.Lock()

        logger.info("RealTradingExecutor inicializado")

    async def initialize(self) -> bool:
        """Inicializa o executor de trading"""
        try:
            # Conectar ao cliente Deriv
            if not await self.deriv_client.connect():
                logger.error("Falha ao conectar ao cliente Deriv")
                return False

            # Inicializar processador de ticks
            await self.tick_processor.initialize()

            # Inicializar gerenciador de risco
            await self.risk_manager.initialize()

            # Verificar saldo da conta
            account_info = await self.get_account_info()
            if not account_info:
                logger.error("Falha ao obter informações da conta")
                return False

            logger.info(f"Conta conectada: {account_info}")

            # Iniciar tarefas de monitoramento
            await self._start_monitoring_tasks()

            logger.info("RealTradingExecutor inicializado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao inicializar RealTradingExecutor: {e}")
            return False

    async def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        try:
            response = await self.deriv_client.send_request({
                "balance": 1
            })

            if response and "balance" in response:
                return {
                    "balance": response["balance"]["balance"],
                    "currency": response["balance"]["currency"],
                    "login_id": response["balance"]["loginid"]
                }

            return None

        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {e}")
            return None

    async def start_trading_session(self, session_config: Dict = None) -> str:
        """Inicia uma nova sessão de trading"""
        session_id = f"session_{int(time.time())}"

        self.trading_session = TradingSession(
            session_id=session_id,
            start_time=datetime.now()
        )

        self.is_trading_enabled = True
        self.daily_pnl = 0.0
        self.session_pnl = 0.0

        logger.info(f"Sessão de trading iniciada: {session_id}")
        return session_id

    async def stop_trading_session(self) -> TradingSession:
        """Para a sessão de trading atual"""
        self.is_trading_enabled = False

        if self.trading_session:
            self.trading_session.end_time = datetime.now()

            # Calcular estatísticas finais
            if self.trading_session.total_trades > 0:
                self.trading_session.win_rate = (
                    self.trading_session.successful_trades /
                    self.trading_session.total_trades
                )

            # Fechar todas as posições abertas
            await self._close_all_positions()

            logger.info(f"Sessão de trading finalizada: {self.trading_session.session_id}")

        return self.trading_session

    async def execute_trade(self, trade_request: RealTradeRequest) -> RealTradeResult:
        """Executa um trade real"""
        async with asyncio.Lock():
            try:
                # Validações pré-trade
                validation_result = await self._validate_trade_request(trade_request)
                if not validation_result["valid"]:
                    return RealTradeResult(
                        trade_id=f"failed_{int(time.time())}",
                        contract_id=None,
                        status=TradeStatus.REJECTED,
                        symbol=trade_request.symbol,
                        trade_type=trade_request.trade_type,
                        amount=trade_request.amount,
                        entry_price=None,
                        exit_price=None,
                        profit_loss=None,
                        execution_time=datetime.now(),
                        completion_time=None,
                        error_message=validation_result["reason"]
                    )

                # Verificar cooldown entre trades
                if not self._check_trade_cooldown():
                    return RealTradeResult(
                        trade_id=f"cooldown_{int(time.time())}",
                        contract_id=None,
                        status=TradeStatus.REJECTED,
                        symbol=trade_request.symbol,
                        trade_type=trade_request.trade_type,
                        amount=trade_request.amount,
                        entry_price=None,
                        exit_price=None,
                        profit_loss=None,
                        execution_time=datetime.now(),
                        completion_time=None,
                        error_message="Trade cooldown ativo"
                    )

                # Preparar requisição para API Deriv
                deriv_request = self._prepare_deriv_request(trade_request)

                # Executar trade na API
                logger.info(f"Executando trade: {trade_request.symbol} {trade_request.trade_type.value}")
                response = await self.deriv_client.send_request(deriv_request)

                if not response or "error" in response:
                    error_msg = response.get("error", {}).get("message", "Erro desconhecido") if response else "Sem resposta da API"
                    return RealTradeResult(
                        trade_id=f"error_{int(time.time())}",
                        contract_id=None,
                        status=TradeStatus.FAILED,
                        symbol=trade_request.symbol,
                        trade_type=trade_request.trade_type,
                        amount=trade_request.amount,
                        entry_price=None,
                        exit_price=None,
                        profit_loss=None,
                        execution_time=datetime.now(),
                        completion_time=None,
                        error_message=error_msg
                    )

                # Processar resposta de sucesso
                buy_result = response.get("buy", {})
                contract_id = buy_result.get("contract_id")

                if not contract_id:
                    return RealTradeResult(
                        trade_id=f"no_contract_{int(time.time())}",
                        contract_id=None,
                        status=TradeStatus.FAILED,
                        symbol=trade_request.symbol,
                        trade_type=trade_request.trade_type,
                        amount=trade_request.amount,
                        entry_price=None,
                        exit_price=None,
                        profit_loss=None,
                        execution_time=datetime.now(),
                        completion_time=None,
                        error_message="Contract ID não recebido"
                    )

                # Criar resultado de trade
                trade_result = RealTradeResult(
                    trade_id=str(contract_id),
                    contract_id=contract_id,
                    status=TradeStatus.EXECUTED,
                    symbol=trade_request.symbol,
                    trade_type=trade_request.trade_type,
                    amount=trade_request.amount,
                    entry_price=buy_result.get("start_spot"),
                    exit_price=None,
                    profit_loss=None,
                    execution_time=datetime.now(),
                    completion_time=None
                )

                # Registrar posição ativa
                position = TradingPosition(
                    contract_id=contract_id,
                    symbol=trade_request.symbol,
                    trade_type=trade_request.trade_type,
                    amount=trade_request.amount,
                    entry_price=buy_result.get("start_spot", 0.0),
                    entry_time=datetime.now(),
                    stop_loss=trade_request.stop_loss,
                    take_profit=trade_request.take_profit
                )

                with self.position_lock:
                    self.active_positions[contract_id] = position

                # Atualizar estatísticas
                self._update_trading_statistics(trade_result)
                self.trade_history.append(trade_result)
                self.last_trade_time = time.time()

                logger.info(f"Trade executado com sucesso: {contract_id}")
                return trade_result

            except Exception as e:
                logger.error(f"Erro ao executar trade: {e}")
                return RealTradeResult(
                    trade_id=f"exception_{int(time.time())}",
                    contract_id=None,
                    status=TradeStatus.FAILED,
                    symbol=trade_request.symbol,
                    trade_type=trade_request.trade_type,
                    amount=trade_request.amount,
                    entry_price=None,
                    exit_price=None,
                    profit_loss=None,
                    execution_time=datetime.now(),
                    completion_time=None,
                    error_message=str(e)
                )

    async def _validate_trade_request(self, trade_request: RealTradeRequest) -> Dict[str, Any]:
        """Valida uma requisição de trade"""
        # Verificar se trading está habilitado
        if not self.is_trading_enabled:
            return {"valid": False, "reason": "Trading desabilitado"}

        # Verificar limites de posições
        if len(self.active_positions) >= self.max_concurrent_trades:
            return {"valid": False, "reason": "Máximo de posições atingido"}

        # Verificar tamanho da posição
        if trade_request.amount > self.max_position_size:
            return {"valid": False, "reason": "Tamanho da posição excede limite"}

        # Verificar perda diária
        if abs(self.daily_pnl) >= self.max_daily_loss:
            return {"valid": False, "reason": "Limite de perda diária atingido"}

        # Verificar confiança da IA
        if (trade_request.ai_confidence and
            trade_request.ai_confidence < self.min_confidence_threshold):
            return {"valid": False, "reason": "Confiança da IA insuficiente"}

        # Verificar saldo da conta
        account_info = await self.get_account_info()
        if account_info and account_info["balance"] < trade_request.amount:
            return {"valid": False, "reason": "Saldo insuficiente"}

        return {"valid": True, "reason": "Validação aprovada"}

    def _check_trade_cooldown(self) -> bool:
        """Verifica cooldown entre trades"""
        if self.last_trade_time is None:
            return True

        return (time.time() - self.last_trade_time) >= self.trade_cooldown_seconds

    def _prepare_deriv_request(self, trade_request: RealTradeRequest) -> Dict:
        """Prepara requisição para API Deriv"""
        request = {
            "buy": 1,
            "parameters": {
                "contract_type": trade_request.contract_type.value,
                "symbol": trade_request.symbol,
                "amount": trade_request.amount,
                "duration": trade_request.duration,
                "duration_unit": trade_request.duration_unit,
                "currency": trade_request.currency
            }
        }

        # Adicionar barreiras se necessário
        if trade_request.barrier:
            request["parameters"]["barrier"] = trade_request.barrier
        if trade_request.barrier2:
            request["parameters"]["barrier2"] = trade_request.barrier2

        return request

    def _update_trading_statistics(self, trade_result: RealTradeResult):
        """Atualiza estatísticas de trading"""
        if self.trading_session:
            self.trading_session.total_trades += 1

    async def _start_monitoring_tasks(self):
        """Inicia tarefas de monitoramento"""
        self.monitoring_task = asyncio.create_task(self._monitor_positions())
        self.position_update_task = asyncio.create_task(self._update_positions())

    async def _monitor_positions(self):
        """Monitora posições ativas"""
        while self.is_trading_enabled:
            try:
                await self._check_position_exits()
                await asyncio.sleep(5)  # Verificar a cada 5 segundos
            except Exception as e:
                logger.error(f"Erro no monitoramento de posições: {e}")
                await asyncio.sleep(10)

    async def _update_positions(self):
        """Atualiza preços das posições"""
        while self.is_trading_enabled:
            try:
                for contract_id, position in list(self.active_positions.items()):
                    # Obter preço atual
                    current_price = await self._get_current_price(position.symbol)
                    if current_price:
                        position.current_price = current_price
                        position.unrealized_pnl = self._calculate_unrealized_pnl(position)

                await asyncio.sleep(1)  # Atualizar a cada segundo
            except Exception as e:
                logger.error(f"Erro na atualização de posições: {e}")
                await asyncio.sleep(5)

    async def _check_position_exits(self):
        """Verifica condições de saída das posições"""
        for contract_id, position in list(self.active_positions.items()):
            try:
                # Verificar stop loss
                if (position.stop_loss and position.current_price and
                    ((position.trade_type == TradeType.BUY and position.current_price <= position.stop_loss) or
                     (position.trade_type == TradeType.SELL and position.current_price >= position.stop_loss))):
                    await self._close_position(contract_id, "Stop Loss")

                # Verificar take profit
                if (position.take_profit and position.current_price and
                    ((position.trade_type == TradeType.BUY and position.current_price >= position.take_profit) or
                     (position.trade_type == TradeType.SELL and position.current_price <= position.take_profit))):
                    await self._close_position(contract_id, "Take Profit")

            except Exception as e:
                logger.error(f"Erro ao verificar saída da posição {contract_id}: {e}")

    async def _close_position(self, contract_id: str, reason: str):
        """Fecha uma posição"""
        try:
            # Enviar comando de fechamento para API
            response = await self.deriv_client.send_request({
                "sell": contract_id,
                "price": 0  # Mercado
            })

            if response and "sell" in response:
                # Atualizar resultado do trade
                position = self.active_positions.get(contract_id)
                if position:
                    # Encontrar trade correspondente no histórico
                    for trade_result in self.trade_history:
                        if trade_result.contract_id == contract_id:
                            trade_result.exit_price = position.current_price
                            trade_result.profit_loss = position.unrealized_pnl
                            trade_result.completion_time = datetime.now()
                            trade_result.status = TradeStatus.EXECUTED
                            break

                    # Atualizar estatísticas
                    if position.unrealized_pnl and position.unrealized_pnl > 0:
                        if self.trading_session:
                            self.trading_session.successful_trades += 1

                    self.session_pnl += position.unrealized_pnl or 0
                    self.daily_pnl += position.unrealized_pnl or 0

                # Remover posição ativa
                with self.position_lock:
                    del self.active_positions[contract_id]

                logger.info(f"Posição {contract_id} fechada: {reason}")

        except Exception as e:
            logger.error(f"Erro ao fechar posição {contract_id}: {e}")

    async def _close_all_positions(self):
        """Fecha todas as posições ativas"""
        for contract_id in list(self.active_positions.keys()):
            await self._close_position(contract_id, "Sessão finalizada")

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
            logger.error(f"Erro ao obter preço atual de {symbol}: {e}")
            return None

    def _calculate_unrealized_pnl(self, position: TradingPosition) -> float:
        """Calcula PnL não realizado de uma posição"""
        if not position.current_price:
            return 0.0

        if position.trade_type == TradeType.BUY:
            return (position.current_price - position.entry_price) * position.amount
        else:
            return (position.entry_price - position.current_price) * position.amount

    async def get_trading_status(self) -> Dict[str, Any]:
        """Obtém status atual do trading"""
        return {
            "is_trading_enabled": self.is_trading_enabled,
            "session": asdict(self.trading_session) if self.trading_session else None,
            "active_positions": len(self.active_positions),
            "session_pnl": self.session_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": len(self.trade_history),
            "last_trade_time": self.last_trade_time
        }

    async def get_positions(self) -> List[Dict]:
        """Obtém posições ativas"""
        return [asdict(position) for position in self.active_positions.values()]

    async def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Obtém histórico de trades"""
        return [asdict(trade) for trade in self.trade_history[-limit:]]

    async def shutdown(self):
        """Encerra o executor de trading"""
        self.is_trading_enabled = False

        # Parar tarefas de monitoramento
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.position_update_task:
            self.position_update_task.cancel()

        # Fechar todas as posições
        await self._close_all_positions()

        # Desconectar cliente
        await self.deriv_client.disconnect()

        logger.info("RealTradingExecutor encerrado")

# Integração com Autonomous Trading Engine
async def execute_ai_decision(executor: RealTradingExecutor, decision: TradingDecision) -> RealTradeResult:
    """Executa uma decisão da IA no executor real"""
    if decision.action == DecisionType.BUY:
        trade_type = TradeType.BUY
        contract_type = ContractType.CALL
    elif decision.action == DecisionType.SELL:
        trade_type = TradeType.SELL
        contract_type = ContractType.PUT
    else:
        # Não executar trades para HOLD
        return None

    trade_request = RealTradeRequest(
        symbol=decision.symbol,
        trade_type=trade_type,
        contract_type=contract_type,
        amount=decision.position_size,
        duration=60,  # 1 minuto
        duration_unit="s",
        trading_decision_id=decision.decision_id,
        ai_confidence=decision.confidence,
        expected_profit=decision.expected_return
    )

    return await executor.execute_trade(trade_request)
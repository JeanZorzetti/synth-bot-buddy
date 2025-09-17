"""
Order Execution Monitor - Sistema de Execução e Monitoramento de Ordens
Sistema completo para execução, rastreamento e monitoramento de ordens em tempo real
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem
from real_deriv_websocket import RealDerivWebSocket
from real_trading_executor import TradeType, TradingMode
from real_risk_manager import RealRiskManager

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionType(Enum):
    NEW = "new"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    CANCELED = "canceled"
    REPLACED = "replaced"
    REJECTED = "rejected"

@dataclass
class OrderRequest:
    """Solicitação de ordem"""
    symbol: str
    order_type: OrderType
    trade_type: TradeType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    expire_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Order:
    """Ordem completa com todos os detalhes"""
    order_id: str
    client_order_id: str
    symbol: str
    order_type: OrderType
    trade_type: TradeType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    commission: float
    created_time: datetime
    updated_time: datetime
    filled_time: Optional[datetime]
    contract_id: Optional[str]
    error_message: Optional[str]
    executions: List['OrderExecution']
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.executions is None:
            self.executions = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OrderExecution:
    """Execução de ordem"""
    execution_id: str
    order_id: str
    execution_type: ExecutionType
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    liquidity_indicator: Optional[str] = None
    trade_id: Optional[str] = None

@dataclass
class OrderBook:
    """Livro de ordens simplificado"""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    timestamp: datetime

class OrderExecutionMonitor:
    """Monitor de execução de ordens"""

    def __init__(self):
        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()
        self.deriv_client = RealDerivWebSocket()
        self.risk_manager = None  # Será definido na inicialização

        # Estado das ordens
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.order_history: List[Order] = []

        # Configurações
        self.max_order_age_hours = 24
        self.execution_timeout_seconds = 30
        self.max_retries = 3
        self.retry_delay_seconds = 1

        # Estado do sistema
        self.is_monitoring = False
        self.monitoring_task = None
        self.execution_queue = asyncio.Queue()

        # Métricas de execução
        self.execution_metrics = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "cancelled_orders": 0,
            "avg_execution_time": 0.0,
            "fill_rate": 0.0,
            "rejection_rate": 0.0
        }

        # Order books para diferentes símbolos
        self.order_books: Dict[str, OrderBook] = {}

        # Tasks de monitoramento
        self.order_monitoring_task = None
        self.execution_processing_task = None
        self.metrics_update_task = None

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self, risk_manager: RealRiskManager):
        """Inicializa o monitor de execução"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.logger.initialize()
            await self.deriv_client.initialize()

            self.risk_manager = risk_manager

            # Carregar ordens ativas do banco
            await self._load_active_orders()

            await self.logger.log_activity("order_execution_monitor_initialized", {
                "active_orders": len(self.active_orders)
            })

            print("✅ Order Execution Monitor inicializado com sucesso")

        except Exception as e:
            await self.logger.log_error("order_monitor_init_error", str(e))
            raise

    async def start_monitoring(self):
        """Inicia monitoramento de ordens"""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Iniciar tasks de monitoramento
        self.order_monitoring_task = asyncio.create_task(self._order_monitoring_loop())
        self.execution_processing_task = asyncio.create_task(self._execution_processing_loop())
        self.metrics_update_task = asyncio.create_task(self._metrics_update_loop())

        await self.logger.log_activity("order_monitoring_started", {})
        print("📊 Monitoramento de ordens iniciado")

    async def stop_monitoring(self):
        """Para o monitoramento"""
        self.is_monitoring = False

        # Cancelar tasks
        for task in [self.order_monitoring_task, self.execution_processing_task, self.metrics_update_task]:
            if task:
                task.cancel()

        await self.logger.log_activity("order_monitoring_stopped", {})
        print("⏹️ Monitoramento de ordens parado")

    async def submit_order(self, order_request: OrderRequest) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submete uma nova ordem"""
        try:
            # Gerar IDs únicos
            order_id = str(uuid.uuid4())
            client_order_id = order_request.client_order_id or f"order_{int(datetime.utcnow().timestamp())}"

            # Validar ordem com risk manager
            if self.risk_manager:
                is_valid, error_msg, adjusted_quantity = await self.risk_manager.validate_new_position(
                    order_request.symbol,
                    order_request.trade_type,
                    order_request.quantity,
                    order_request.price or 0.0
                )

                if not is_valid:
                    await self.logger.log_error("order_risk_validation_failed", f"{order_request.symbol}: {error_msg}")
                    return False, error_msg, None

                # Usar quantidade ajustada pelo risk manager
                order_request.quantity = adjusted_quantity

            # Criar objeto Order
            order = Order(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=order_request.symbol,
                order_type=order_request.order_type,
                trade_type=order_request.trade_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                remaining_quantity=order_request.quantity,
                avg_fill_price=0.0,
                commission=0.0,
                created_time=datetime.utcnow(),
                updated_time=datetime.utcnow(),
                filled_time=None,
                contract_id=None,
                error_message=None,
                executions=[],
                metadata=order_request.metadata or {}
            )

            # Adicionar à fila de execução
            await self.execution_queue.put(order)

            # Adicionar às ordens ativas
            self.active_orders[order_id] = order

            # Salvar no banco
            await self._save_order_to_db(order)

            # Cache da ordem
            await self.cache_manager.set(
                CacheNamespace.TRADING_SESSIONS,
                f"order_{order_id}",
                asdict(order),
                ttl=3600
            )

            await self.logger.log_activity("order_submitted", {
                "order_id": order_id,
                "symbol": order_request.symbol,
                "type": order_request.order_type.value,
                "trade_type": order_request.trade_type.value,
                "quantity": order_request.quantity,
                "price": order_request.price
            })

            self.execution_metrics["total_orders"] += 1
            print(f"📤 Ordem submetida: {order_id} - {order_request.symbol} {order_request.trade_type.value}")

            return True, None, order_id

        except Exception as e:
            await self.logger.log_error("order_submission_error", f"{order_request.symbol}: {str(e)}")
            return False, str(e), None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return False

            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return False

            # Enviar cancelamento para Deriv
            if order.contract_id:
                response = await self.deriv_client.send_request({
                    "cancel": order.contract_id
                })

                if response and "cancel" in response:
                    success = response["cancel"]["status"] == "cancelled"
                else:
                    success = False
            else:
                success = True  # Ordem ainda não executada

            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_time = datetime.utcnow()

                # Mover para ordens completadas
                self.completed_orders.append(order)
                del self.active_orders[order_id]

                # Atualizar banco
                await self._update_order_in_db(order)

                await self.logger.log_activity("order_cancelled", {
                    "order_id": order_id,
                    "symbol": order.symbol
                })

                self.execution_metrics["cancelled_orders"] += 1
                print(f"❌ Ordem cancelada: {order_id}")

                return True
            else:
                await self.logger.log_error("order_cancellation_failed", f"{order_id}: Falha no cancelamento")
                return False

        except Exception as e:
            await self.logger.log_error("order_cancellation_error", f"{order_id}: {str(e)}")
            return False

    async def _execution_processing_loop(self):
        """Loop de processamento de execuções"""
        while self.is_monitoring:
            try:
                # Processar ordens da fila
                try:
                    order = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                    await self._execute_order(order)
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                await self.logger.log_error("execution_processing_error", str(e))
                await asyncio.sleep(1)

    async def _execute_order(self, order: Order):
        """Executa uma ordem específica"""
        try:
            order.status = OrderStatus.SUBMITTED
            order.updated_time = datetime.utcnow()

            execution_start_time = datetime.utcnow()

            # Preparar request para Deriv API
            deriv_request = await self._prepare_deriv_request(order)

            if not deriv_request:
                await self._handle_order_rejection(order, "Falha na preparação do request")
                return

            # Executar com retry
            success = False
            last_error = None

            for attempt in range(self.max_retries):
                try:
                    response = await self.deriv_client.send_request(deriv_request)

                    if response and self._is_successful_response(response):
                        await self._handle_successful_execution(order, response, execution_start_time)
                        success = True
                        break
                    else:
                        error_msg = self._extract_error_message(response)
                        last_error = error_msg
                        await asyncio.sleep(self.retry_delay_seconds * (attempt + 1))

                except Exception as e:
                    last_error = str(e)
                    await asyncio.sleep(self.retry_delay_seconds * (attempt + 1))

            if not success:
                await self._handle_order_rejection(order, last_error or "Execução falhou após múltiplas tentativas")

        except Exception as e:
            await self.logger.log_error("order_execution_error", f"{order.order_id}: {str(e)}")
            await self._handle_order_rejection(order, str(e))

    async def _prepare_deriv_request(self, order: Order) -> Optional[Dict[str, Any]]:
        """Prepara request para Deriv API"""
        try:
            base_request = {
                "contract_type": "CALL" if order.trade_type == TradeType.BUY else "PUT",
                "currency": "USD",
                "amount": order.quantity,
                "duration": 15,  # 15 minutos padrão
                "duration_unit": "m",
                "symbol": order.symbol,
                "basis": "stake"
            }

            # Ajustar baseado no tipo de ordem
            if order.order_type == OrderType.MARKET:
                return {"buy": 1, **base_request}

            elif order.order_type == OrderType.LIMIT:
                # Para limite, usar price como barrier
                if order.price:
                    base_request["barrier"] = str(order.price)
                return {"buy": 1, **base_request}

            else:
                # Outros tipos de ordem não implementados completamente
                return {"buy": 1, **base_request}

        except Exception as e:
            await self.logger.log_error("deriv_request_preparation_error", f"{order.order_id}: {str(e)}")
            return None

    def _is_successful_response(self, response: Dict[str, Any]) -> bool:
        """Verifica se resposta da API é bem-sucedida"""
        return "buy" in response or "proposal" in response

    def _extract_error_message(self, response: Optional[Dict[str, Any]]) -> str:
        """Extrai mensagem de erro da resposta"""
        if not response:
            return "Resposta vazia da API"

        if "error" in response:
            return response["error"].get("message", "Erro desconhecido")

        return "Falha na execução"

    async def _handle_successful_execution(self, order: Order, response: Dict[str, Any], start_time: datetime):
        """Trata execução bem-sucedida"""
        try:
            # Extrair dados da resposta
            if "buy" in response:
                buy_data = response["buy"]
                contract_id = buy_data.get("contract_id")
                price = float(buy_data.get("buy_price", 0))
            else:
                contract_id = response.get("proposal", {}).get("id")
                price = order.price or 0.0

            # Atualizar ordem
            order.status = OrderStatus.CONFIRMED
            order.contract_id = contract_id
            order.avg_fill_price = price
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0.0
            order.filled_time = datetime.utcnow()
            order.updated_time = datetime.utcnow()

            # Calcular tempo de execução
            execution_time = (order.filled_time - start_time).total_seconds()

            # Criar execução
            execution = OrderExecution(
                execution_id=str(uuid.uuid4()),
                order_id=order.order_id,
                execution_type=ExecutionType.FILL,
                quantity=order.quantity,
                price=price,
                commission=0.0,  # Deriv não cobra comissão
                timestamp=order.filled_time,
                trade_id=contract_id
            )

            order.executions.append(execution)

            # Mover para ordens completadas
            self.completed_orders.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

            # Atualizar banco
            await self._update_order_in_db(order)

            # Atualizar métricas
            self.execution_metrics["successful_orders"] += 1

            # Atualizar tempo médio de execução
            current_avg = self.execution_metrics["avg_execution_time"]
            total_successful = self.execution_metrics["successful_orders"]
            self.execution_metrics["avg_execution_time"] = (
                (current_avg * (total_successful - 1) + execution_time) / total_successful
            )

            await self.logger.log_activity("order_executed_successfully", {
                "order_id": order.order_id,
                "contract_id": contract_id,
                "symbol": order.symbol,
                "price": price,
                "execution_time": execution_time
            })

            print(f"✅ Ordem executada: {order.order_id} - Contrato: {contract_id}")

        except Exception as e:
            await self.logger.log_error("successful_execution_handling_error", f"{order.order_id}: {str(e)}")
            await self._handle_order_rejection(order, f"Erro no processamento da execução: {str(e)}")

    async def _handle_order_rejection(self, order: Order, error_message: str):
        """Trata rejeição de ordem"""
        try:
            order.status = OrderStatus.REJECTED
            order.error_message = error_message
            order.updated_time = datetime.utcnow()

            # Mover para ordens completadas
            self.completed_orders.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

            # Atualizar banco
            await self._update_order_in_db(order)

            # Atualizar métricas
            self.execution_metrics["failed_orders"] += 1

            await self.logger.log_activity("order_rejected", {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "error": error_message
            })

            print(f"❌ Ordem rejeitada: {order.order_id} - {error_message}")

        except Exception as e:
            await self.logger.log_error("order_rejection_handling_error", f"{order.order_id}: {str(e)}")

    async def _order_monitoring_loop(self):
        """Loop de monitoramento de ordens ativas"""
        while self.is_monitoring:
            try:
                current_time = datetime.utcnow()
                orders_to_expire = []

                for order_id, order in self.active_orders.items():
                    # Verificar expiração
                    age_hours = (current_time - order.created_time).total_seconds() / 3600
                    if age_hours >= self.max_order_age_hours:
                        orders_to_expire.append(order_id)

                    # Verificar status no Deriv se confirmada
                    if order.status == OrderStatus.CONFIRMED and order.contract_id:
                        await self._check_order_status(order)

                # Expirar ordens antigas
                for order_id in orders_to_expire:
                    await self._expire_order(order_id)

                await asyncio.sleep(5)  # Verificar a cada 5 segundos

            except Exception as e:
                await self.logger.log_error("order_monitoring_error", str(e))
                await asyncio.sleep(10)

    async def _check_order_status(self, order: Order):
        """Verifica status de uma ordem no Deriv"""
        try:
            if not order.contract_id:
                return

            response = await self.deriv_client.send_request({
                "proposal_open_contract": 1,
                "contract_id": order.contract_id
            })

            if response and "proposal_open_contract" in response:
                contract_data = response["proposal_open_contract"]

                # Verificar se contrato foi finalizado
                if contract_data.get("is_expired") or contract_data.get("is_sold"):
                    order.status = OrderStatus.FILLED
                    order.updated_time = datetime.utcnow()

                    # Mover para completadas se ainda não movida
                    if order.order_id in self.active_orders:
                        self.completed_orders.append(order)
                        del self.active_orders[order.order_id]

                    await self._update_order_in_db(order)

        except Exception as e:
            await self.logger.log_error("order_status_check_error", f"{order.order_id}: {str(e)}")

    async def _expire_order(self, order_id: str):
        """Expira uma ordem antiga"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return

            order.status = OrderStatus.EXPIRED
            order.updated_time = datetime.utcnow()

            self.completed_orders.append(order)
            del self.active_orders[order_id]

            await self._update_order_in_db(order)

            await self.logger.log_activity("order_expired", {
                "order_id": order_id,
                "symbol": order.symbol,
                "age_hours": (order.updated_time - order.created_time).total_seconds() / 3600
            })

        except Exception as e:
            await self.logger.log_error("order_expiration_error", f"{order_id}: {str(e)}")

    async def _metrics_update_loop(self):
        """Loop de atualização de métricas"""
        while self.is_monitoring:
            try:
                await self._update_execution_metrics()
                await asyncio.sleep(60)  # A cada minuto

            except Exception as e:
                await self.logger.log_error("metrics_update_error", str(e))
                await asyncio.sleep(300)

    async def _update_execution_metrics(self):
        """Atualiza métricas de execução"""
        try:
            total_orders = self.execution_metrics["total_orders"]

            if total_orders > 0:
                # Taxa de preenchimento
                filled_orders = self.execution_metrics["successful_orders"]
                self.execution_metrics["fill_rate"] = (filled_orders / total_orders) * 100

                # Taxa de rejeição
                failed_orders = self.execution_metrics["failed_orders"]
                self.execution_metrics["rejection_rate"] = (failed_orders / total_orders) * 100

            # Cache das métricas
            await self.cache_manager.set(
                CacheNamespace.TRADING_SESSIONS,
                "execution_metrics",
                self.execution_metrics,
                ttl=300
            )

        except Exception as e:
            await self.logger.log_error("execution_metrics_update_error", str(e))

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Obtém status de uma ordem"""
        try:
            # Verificar ordens ativas
            order = self.active_orders.get(order_id)

            # Se não estiver ativa, verificar completadas
            if not order:
                order = next((o for o in self.completed_orders if o.order_id == order_id), None)

            if not order:
                return None

            return {
                "order": asdict(order),
                "executions": [asdict(ex) for ex in order.executions]
            }

        except Exception as e:
            await self.logger.log_error("order_status_query_error", f"{order_id}: {str(e)}")
            return None

    async def get_execution_summary(self) -> Dict[str, Any]:
        """Retorna resumo de execução"""
        try:
            return {
                "metrics": self.execution_metrics,
                "active_orders": len(self.active_orders),
                "completed_orders": len(self.completed_orders),
                "active_order_details": [
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "status": order.status.value,
                        "type": order.order_type.value,
                        "quantity": order.quantity,
                        "created_time": order.created_time.isoformat()
                    }
                    for order in self.active_orders.values()
                ],
                "recent_completions": [
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "status": order.status.value,
                        "filled_quantity": order.filled_quantity,
                        "avg_fill_price": order.avg_fill_price,
                        "filled_time": order.filled_time.isoformat() if order.filled_time else None
                    }
                    for order in self.completed_orders[-10:]  # Últimas 10
                ]
            }

        except Exception as e:
            await self.logger.log_error("execution_summary_error", str(e))
            return {}

    # Métodos auxiliares para banco de dados
    async def _save_order_to_db(self, order: Order):
        """Salva ordem no banco de dados"""
        try:
            # Implementar salvamento no banco
            pass
        except Exception as e:
            await self.logger.log_error("order_db_save_error", f"{order.order_id}: {str(e)}")

    async def _update_order_in_db(self, order: Order):
        """Atualiza ordem no banco de dados"""
        try:
            # Implementar atualização no banco
            pass
        except Exception as e:
            await self.logger.log_error("order_db_update_error", f"{order.order_id}: {str(e)}")

    async def _load_active_orders(self):
        """Carrega ordens ativas do banco"""
        try:
            # Implementar carregamento do banco
            pass
        except Exception as e:
            await self.logger.log_error("active_orders_load_error", str(e))

    async def shutdown(self):
        """Encerra o monitor de execução"""
        await self.stop_monitoring()

        # Cancelar todas as ordens ativas
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id)

        await self.logger.log_activity("order_execution_monitor_shutdown", {})
        print("🔌 Order Execution Monitor encerrado")
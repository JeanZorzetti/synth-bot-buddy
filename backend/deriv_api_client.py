"""
🔌 DERIV API CLIENT
Integração direta com a API da Deriv para execução de trades
Executar operações de compra/venda de forma autônoma
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

from autonomous_trading_engine import TradingDecision, TradeAction


class ContractType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    DIGITRISE = "DIGITRISE"
    DIGITFALL = "DIGITFALL"
    DIGITDIFF = "DIGITDIFF"
    DIGITEVEN = "DIGITEVEN"
    DIGITODD = "DIGITODD"
    DIGITMATCH = "DIGITMATCH"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"
    CANCELLED = "CANCELLED"


@dataclass
class DerivContract:
    """Contrato executado na Deriv"""
    contract_id: str
    symbol: str
    contract_type: ContractType
    stake: float
    entry_spot: float
    duration: int
    duration_unit: str
    timestamp: datetime
    status: OrderStatus
    payout: float = 0.0
    profit_loss: float = 0.0
    exit_spot: float = 0.0
    exit_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'contract_type': self.contract_type.value,
            'stake': self.stake,
            'entry_spot': self.entry_spot,
            'duration': self.duration,
            'duration_unit': self.duration_unit,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'payout': self.payout,
            'profit_loss': self.profit_loss,
            'exit_spot': self.exit_spot,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None
        }


class DerivAPIClient:
    """
    🔌 Cliente da API Deriv para trading autônomo

    Funcionalidades:
    1. Conexão WebSocket com Deriv API
    2. Execução automática de contratos
    3. Monitoramento de posições em tempo real
    4. Gestão de saldo e histórico
    """

    def __init__(self, api_token: str, app_id: str = "1089", is_demo: bool = True):
        self.api_token = api_token
        self.app_id = app_id
        self.is_demo = is_demo

        # URLs da API
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        if is_demo:
            self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

        self.logger = logging.getLogger(__name__)

        # Estado da conexão
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_authorized = False

        # Estado da conta
        self.account_balance = 0.0
        self.account_currency = "USD"
        self.account_id = ""

        # Contratos ativos
        self.active_contracts: Dict[str, DerivContract] = {}
        self.contract_history: List[DerivContract] = []

        # Callbacks
        self.on_tick_callback: Optional[Callable] = None
        self.on_contract_update_callback: Optional[Callable] = None

        # Request tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> bool:
        """Conecta com a API da Deriv"""
        try:
            self.logger.info(f"🔌 Conectando com Deriv API... {'(DEMO)' if self.is_demo else '(REAL)'}")

            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True

            # Autorizar conexão
            await self._authorize()

            # Obter informações da conta
            await self._get_account_status()

            # Iniciar loop de recebimento de mensagens
            asyncio.create_task(self._message_loop())

            self.logger.info("✅ Conectado e autorizado na Deriv API")
            return True

        except Exception as e:
            self.logger.error(f"❌ Erro ao conectar com Deriv API: {e}")
            return False

    async def disconnect(self):
        """Desconecta da API"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.is_authorized = False
            self.logger.info("🔌 Desconectado da Deriv API")

    async def _authorize(self):
        """Autoriza a conexão com o token"""
        auth_request = {
            "authorize": self.api_token,
            "req_id": self._generate_req_id()
        }

        await self._send_request(auth_request)

    async def _get_account_status(self):
        """Obtém status da conta"""
        balance_request = {
            "balance": 1,
            "req_id": self._generate_req_id()
        }

        await self._send_request(balance_request)

    async def _send_request(self, request: Dict[str, Any]):
        """Envia requisição via WebSocket"""
        if not self.websocket or not self.is_connected:
            raise Exception("WebSocket não conectado")

        request_json = json.dumps(request)
        await self.websocket.send(request_json)

        self.logger.debug(f"📤 Enviado: {request_json}")

    async def _message_loop(self):
        """Loop principal de recebimento de mensagens"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("🔌 Conexão WebSocket fechada")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"❌ Erro no loop de mensagens: {e}")

    async def _handle_message(self, data: Dict[str, Any]):
        """Processa mensagens recebidas da API"""
        self.logger.debug(f"📥 Recebido: {json.dumps(data, indent=2)}")

        # Autorização
        if "authorize" in data:
            if data.get("error"):
                self.logger.error(f"❌ Erro de autorização: {data['error']}")
            else:
                self.is_authorized = True
                self.account_id = data["authorize"]["loginid"]
                self.logger.info(f"✅ Autorizado - Conta: {self.account_id}")

        # Saldo da conta
        elif "balance" in data:
            if not data.get("error"):
                self.account_balance = float(data["balance"]["balance"])
                self.account_currency = data["balance"]["currency"]
                self.logger.info(f"💰 Saldo: {self.account_balance} {self.account_currency}")

        # Ticks de preço
        elif "tick" in data:
            await self._handle_tick(data["tick"])

        # Proposta de contrato
        elif "proposal" in data:
            await self._handle_proposal(data)

        # Compra de contrato
        elif "buy" in data:
            await self._handle_buy_response(data)

        # Histórico de contratos
        elif "portfolio" in data:
            await self._handle_portfolio_update(data["portfolio"])

        # Atualizações de contratos
        elif "proposal_open_contract" in data:
            await self._handle_contract_update(data["proposal_open_contract"])

        # Erros
        elif "error" in data:
            self.logger.error(f"❌ Erro da API: {data['error']}")

    async def _handle_tick(self, tick_data: Dict[str, Any]):
        """Processa tick de preço recebido"""
        if self.on_tick_callback:
            await self.on_tick_callback(tick_data)

    async def _handle_proposal(self, proposal_data: Dict[str, Any]):
        """Processa resposta de proposta de contrato"""
        req_id = proposal_data.get("req_id")
        if req_id in self.pending_requests:
            proposal_info = proposal_data.get("proposal", {})
            self.pending_requests[req_id]["proposal"] = proposal_info

    async def _handle_buy_response(self, buy_data: Dict[str, Any]):
        """Processa resposta de compra de contrato"""
        if buy_data.get("error"):
            self.logger.error(f"❌ Erro ao comprar contrato: {buy_data['error']}")
            return

        buy_info = buy_data["buy"]
        contract_id = buy_info["contract_id"]

        # Criar contrato na nossa estrutura
        contract = DerivContract(
            contract_id=contract_id,
            symbol=buy_info.get("shortcode", "").split("_")[1] if "_" in buy_info.get("shortcode", "") else "",
            contract_type=ContractType.CALL,  # Seria determinado pelo shortcode
            stake=float(buy_info["buy_price"]),
            entry_spot=float(buy_info.get("start_spot", 0)),
            duration=0,  # Seria extraído do shortcode
            duration_unit="S",
            timestamp=datetime.now(),
            status=OrderStatus.OPEN,
            payout=float(buy_info.get("payout", 0))
        )

        self.active_contracts[contract_id] = contract

        self.logger.info(f"✅ Contrato comprado: {contract_id} | Stake: ${contract.stake}")

    async def _handle_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Processa atualização do portfólio"""
        contracts = portfolio_data.get("contracts", [])

        for contract_data in contracts:
            contract_id = str(contract_data["contract_id"])

            if contract_id in self.active_contracts:
                # Atualizar contrato existente
                contract = self.active_contracts[contract_id]
                contract.profit_loss = float(contract_data.get("profit", 0))

                # Verificar se foi finalizado
                if contract_data.get("is_expired") or contract_data.get("is_sold"):
                    contract.status = OrderStatus.WON if contract.profit_loss > 0 else OrderStatus.LOST
                    contract.exit_time = datetime.now()

                    # Mover para histórico
                    self.contract_history.append(contract)
                    del self.active_contracts[contract_id]

                    self.logger.info(f"📊 Contrato finalizado: {contract_id} | P&L: ${contract.profit_loss}")

    async def _handle_contract_update(self, contract_data: Dict[str, Any]):
        """Processa atualizações de contrato em tempo real"""
        contract_id = str(contract_data["contract_id"])

        if contract_id in self.active_contracts:
            contract = self.active_contracts[contract_id]
            contract.profit_loss = float(contract_data.get("profit", 0))
            contract.exit_spot = float(contract_data.get("current_spot", 0))

            if self.on_contract_update_callback:
                await self.on_contract_update_callback(contract)

    async def execute_trading_decision(self, decision: TradingDecision) -> Optional[str]:
        """
        🤖 Executa decisão de trading da IA automaticamente

        Converte decisão da IA em contrato Deriv real
        """
        if not self.is_connected or not self.is_authorized:
            self.logger.error("❌ API não conectada ou autorizada")
            return None

        try:
            # Verificar saldo suficiente
            if decision.position_size > self.account_balance:
                self.logger.warning(f"⚠️ Saldo insuficiente: ${self.account_balance} < ${decision.position_size}")
                return None

            # Mapear decisão da IA para contrato Deriv
            contract_type = self._map_decision_to_contract_type(decision)

            # Calcular duração baseada no holding period esperado
            duration = max(60, min(decision.expected_holding_period, 3600))  # 1min a 1h

            # Obter proposta primeiro
            proposal_id = await self._get_contract_proposal(
                symbol=decision.symbol,
                contract_type=contract_type,
                stake=decision.position_size,
                duration=duration
            )

            if not proposal_id:
                return None

            # Comprar contrato
            contract_id = await self._buy_contract(proposal_id)

            if contract_id:
                self.logger.info(f"🤖 IA executou: {decision.action.value} {decision.symbol} "
                               f"| ${decision.position_size} | Contrato: {contract_id}")

                # Subscrever para atualizações do contrato
                await self._subscribe_contract_updates(contract_id)

            return contract_id

        except Exception as e:
            self.logger.error(f"❌ Erro ao executar decisão de trading: {e}")
            return None

    def _map_decision_to_contract_type(self, decision: TradingDecision) -> ContractType:
        """Mapeia decisão da IA para tipo de contrato Deriv"""

        # Para trading autônomo, usar contratos Rise/Fall simples
        if decision.action == TradeAction.BUY:
            return ContractType.CALL  # Rise
        else:
            return ContractType.PUT   # Fall

    async def _get_contract_proposal(self,
                                   symbol: str,
                                   contract_type: ContractType,
                                   stake: float,
                                   duration: int) -> Optional[str]:
        """Obtém proposta de contrato"""

        req_id = self._generate_req_id()

        # Mapear símbolo para formato Deriv
        deriv_symbol = self._map_symbol_to_deriv(symbol)

        proposal_request = {
            "proposal": 1,
            "subscribe": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type.value,
            "currency": self.account_currency,
            "symbol": deriv_symbol,
            "duration": duration,
            "duration_unit": "s",
            "req_id": req_id
        }

        # Armazenar requisição pendente
        self.pending_requests[req_id] = {
            "type": "proposal",
            "proposal": None
        }

        await self._send_request(proposal_request)

        # Aguardar resposta (timeout de 5 segundos)
        for _ in range(50):  # 5 segundos
            await asyncio.sleep(0.1)
            if req_id in self.pending_requests and self.pending_requests[req_id]["proposal"]:
                proposal = self.pending_requests[req_id]["proposal"]
                proposal_id = proposal.get("id")
                del self.pending_requests[req_id]
                return proposal_id

        # Timeout
        if req_id in self.pending_requests:
            del self.pending_requests[req_id]

        self.logger.warning("⏰ Timeout ao obter proposta de contrato")
        return None

    async def _buy_contract(self, proposal_id: str) -> Optional[str]:
        """Compra contrato usando proposal_id"""

        buy_request = {
            "buy": proposal_id,
            "price": 0,  # Usar preço da proposta
            "req_id": self._generate_req_id()
        }

        await self._send_request(buy_request)

        # A resposta será processada em _handle_buy_response
        # Retornar None por agora - o contract_id será logado quando recebido
        return "pending"

    async def _subscribe_contract_updates(self, contract_id: str):
        """Subscreve para atualizações de contrato em tempo real"""

        subscribe_request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
            "req_id": self._generate_req_id()
        }

        await self._send_request(subscribe_request)

    def _map_symbol_to_deriv(self, symbol: str) -> str:
        """Mapeia símbolos para formato Deriv"""
        symbol_mapping = {
            "EURUSD": "frxEURUSD",
            "GBPUSD": "frxGBPUSD",
            "USDJPY": "frxUSDJPY",
            "AUDUSD": "frxAUDUSD",
            "USDCAD": "frxUSDCAD",
            "USDCHF": "frxUSDCHF",
            "EURJPY": "frxEURJPY",
            "EURGBP": "frxEURGBP",
            "Volatility 10 Index": "R_10",
            "Volatility 25 Index": "R_25",
            "Volatility 50 Index": "R_50",
            "Volatility 75 Index": "R_75",
            "Volatility 100 Index": "R_100"
        }

        return symbol_mapping.get(symbol, symbol)

    def _generate_req_id(self) -> str:
        """Gera ID único para requisições"""
        return str(uuid.uuid4())

    async def get_account_balance(self) -> float:
        """Retorna saldo atual da conta"""
        return self.account_balance

    def get_active_contracts(self) -> List[Dict[str, Any]]:
        """Retorna contratos ativos"""
        return [contract.to_dict() for contract in self.active_contracts.values()]

    def get_contract_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna histórico de contratos"""
        recent_history = self.contract_history[-limit:] if limit else self.contract_history
        return [contract.to_dict() for contract in recent_history]

    def get_trading_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de trading"""
        total_contracts = len(self.contract_history)
        if total_contracts == 0:
            return {
                'total_contracts': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'average_profit': 0.0
            }

        winning_contracts = sum(1 for c in self.contract_history if c.profit_loss > 0)
        total_profit = sum(c.profit_loss for c in self.contract_history)

        return {
            'total_contracts': total_contracts,
            'winning_contracts': winning_contracts,
            'losing_contracts': total_contracts - winning_contracts,
            'win_rate': winning_contracts / total_contracts * 100,
            'total_profit': total_profit,
            'average_profit': total_profit / total_contracts,
            'active_contracts': len(self.active_contracts),
            'account_balance': self.account_balance
        }

    async def emergency_close_all_contracts(self):
        """Fecha todos os contratos em caso de emergência"""
        self.logger.warning("🚨 FECHANDO TODOS OS CONTRATOS - EMERGÊNCIA!")

        for contract_id in list(self.active_contracts.keys()):
            try:
                # Tentar vender contrato
                sell_request = {
                    "sell": int(contract_id),
                    "price": 0,  # Vender pelo preço atual
                    "req_id": self._generate_req_id()
                }

                await self._send_request(sell_request)
                self.logger.info(f"🔄 Tentativa de fechar contrato: {contract_id}")

            except Exception as e:
                self.logger.error(f"❌ Erro ao fechar contrato {contract_id}: {e}")

    def set_tick_callback(self, callback: Callable):
        """Define callback para recebimento de ticks"""
        self.on_tick_callback = callback

    def set_contract_update_callback(self, callback: Callable):
        """Define callback para atualizações de contratos"""
        self.on_contract_update_callback = callback
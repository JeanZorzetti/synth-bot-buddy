"""
🔄 REAL DERIV CLIENT
Real integration with Deriv Binary API for live trading
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import base64
from collections import defaultdict, deque
import aiohttp
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Estados de conexão do WebSocket"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class RealTickData:
    """Estrutura de dados para tick real da Deriv"""
    symbol: str
    tick: float
    epoch: int
    quote: float
    pip_size: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class RealTradeResult:
    """Resultado de trade real executado"""
    contract_id: str
    buy_price: float
    payout: float
    profit: Optional[float] = None
    status: str = "open"
    start_time: datetime = None
    end_time: Optional[datetime] = None
    duration: int = 30  # seconds


@dataclass
class AccountInfo:
    """Informações da conta real"""
    balance: float
    currency: str
    loginid: str
    email: str
    is_virtual: bool
    country: str


class RateLimiter:
    """Rate limiter para requisições à API"""

    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    async def acquire(self):
        """Aguardar se necessário para respeitar rate limit"""
        now = time.time()

        # Remove requisições antigas
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()

        # Se atingiu o limite, aguarda
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self.requests.append(now)


class RealDerivWebSocketClient:
    """Cliente WebSocket real para Deriv Binary API"""

    def __init__(self, app_id: str, api_token: str = None, server_url: str = None):
        self.app_id = app_id
        self.api_token = api_token
        self.server_url = server_url or "wss://ws.binaryws.com/websockets/v3"

        # Connection management
        self.websocket = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = 10
        self.reconnection_delay = 5  # seconds

        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=50, time_window=60)

        # Callbacks
        self.tick_callbacks = []
        self.balance_callbacks = []
        self.trade_callbacks = []
        self.connection_callbacks = []

        # Data storage
        self.account_info = None
        self.active_subscriptions = set()
        self.request_id_counter = 0
        self.pending_requests = {}

        # Buffers para dados
        self.tick_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.active_contracts = {}

    def generate_request_id(self) -> str:
        """Gerar ID único para requisições"""
        self.request_id_counter += 1
        return f"req_{int(time.time())}_{self.request_id_counter}"

    async def connect(self) -> bool:
        """Conectar ao WebSocket da Deriv"""
        try:
            self.connection_state = ConnectionState.CONNECTING
            logger.info(f"Connecting to Deriv WebSocket: {self.server_url}")

            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                f"{self.server_url}?app_id={self.app_id}",
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )

            self.connection_state = ConnectionState.CONNECTED
            logger.info("Connected to Deriv WebSocket successfully")

            # Start message handler
            asyncio.create_task(self._message_handler())

            # Authenticate if token provided
            if self.api_token:
                await self._authenticate()

            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    await callback(True)
                except Exception as e:
                    logger.error(f"Error in connection callback: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Deriv WebSocket: {e}")
            self.connection_state = ConnectionState.ERROR
            return False

    async def _authenticate(self) -> bool:
        """Autenticar com token da API"""
        try:
            request_id = self.generate_request_id()
            auth_message = {
                "authorize": self.api_token,
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(auth_message))

            # Aguardar resposta de autenticação
            response = await self._wait_for_response(request_id, timeout=10)

            if response and "authorize" in response:
                self.account_info = AccountInfo(
                    balance=response["authorize"]["balance"],
                    currency=response["authorize"]["currency"],
                    loginid=response["authorize"]["loginid"],
                    email=response["authorize"]["email"],
                    is_virtual=response["authorize"]["is_virtual"] == 1,
                    country=response["authorize"]["country"]
                )

                self.connection_state = ConnectionState.AUTHENTICATED
                logger.info(f"Authenticated successfully. Account: {self.account_info.loginid}")
                return True
            else:
                logger.error("Authentication failed")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _message_handler(self):
        """Handler para mensagens recebidas do WebSocket"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connection_state = ConnectionState.DISCONNECTED
            await self._handle_disconnection()
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await self._handle_disconnection()

    async def _process_message(self, data: Dict):
        """Processar mensagem recebida"""
        # Handle tick data
        if "tick" in data:
            await self._handle_tick_data(data)

        # Handle balance updates
        elif "balance" in data:
            await self._handle_balance_update(data)

        # Handle trade results
        elif "buy" in data or "sell" in data:
            await self._handle_trade_result(data)

        # Handle portfolio updates
        elif "portfolio" in data:
            await self._handle_portfolio_update(data)

        # Handle request responses
        elif "req_id" in data:
            await self._handle_request_response(data)

        # Handle errors
        elif "error" in data:
            await self._handle_error(data)

    async def _handle_tick_data(self, data: Dict):
        """Processar dados de tick recebidos"""
        try:
            tick_info = data["tick"]

            tick_data = RealTickData(
                symbol=tick_info["symbol"],
                tick=float(tick_info["quote"]),
                epoch=int(tick_info["epoch"]),
                quote=float(tick_info["quote"]),
                pip_size=float(tick_info.get("pip_size", 0.01)),
                timestamp=datetime.fromtimestamp(tick_info["epoch"]),
                bid=tick_info.get("bid"),
                ask=tick_info.get("ask")
            )

            # Calculate spread if bid/ask available
            if tick_data.bid and tick_data.ask:
                tick_data.spread = tick_data.ask - tick_data.bid

            # Store in buffer
            self.tick_buffer[tick_data.symbol].append(tick_data)

            # Notify callbacks
            for callback in self.tick_callbacks:
                try:
                    await callback(tick_data)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

        except Exception as e:
            logger.error(f"Error handling tick data: {e}")

    async def _handle_balance_update(self, data: Dict):
        """Processar atualizações de saldo"""
        try:
            balance_info = data["balance"]
            new_balance = float(balance_info["balance"])

            if self.account_info:
                self.account_info.balance = new_balance

            # Notify callbacks
            for callback in self.balance_callbacks:
                try:
                    await callback(new_balance)
                except Exception as e:
                    logger.error(f"Error in balance callback: {e}")

        except Exception as e:
            logger.error(f"Error handling balance update: {e}")

    async def _handle_trade_result(self, data: Dict):
        """Processar resultado de trade"""
        try:
            if "buy" in data:
                trade_info = data["buy"]
            elif "sell" in data:
                trade_info = data["sell"]
            else:
                return

            trade_result = RealTradeResult(
                contract_id=trade_info["contract_id"],
                buy_price=float(trade_info["buy_price"]),
                payout=float(trade_info["payout"]),
                start_time=datetime.fromtimestamp(trade_info["start_time"]),
                status="open"
            )

            # Store active contract
            self.active_contracts[trade_result.contract_id] = trade_result

            # Notify callbacks
            for callback in self.trade_callbacks:
                try:
                    await callback(trade_result)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")

        except Exception as e:
            logger.error(f"Error handling trade result: {e}")

    async def _handle_portfolio_update(self, data: Dict):
        """Processar atualizações do portfolio"""
        try:
            portfolio = data["portfolio"]["contracts"]

            for contract in portfolio:
                contract_id = contract["contract_id"]

                if contract_id in self.active_contracts:
                    # Update existing contract
                    self.active_contracts[contract_id].profit = float(contract.get("profit", 0))

                    # Check if contract closed
                    if contract.get("is_sold") == 1:
                        self.active_contracts[contract_id].status = "closed"
                        self.active_contracts[contract_id].end_time = datetime.fromtimestamp(
                            contract.get("sell_time", time.time())
                        )

        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")

    async def _handle_request_response(self, data: Dict):
        """Processar resposta de requisição"""
        req_id = data["req_id"]
        if req_id in self.pending_requests:
            future = self.pending_requests[req_id]
            if not future.done():
                future.set_result(data)
            del self.pending_requests[req_id]

    async def _handle_error(self, data: Dict):
        """Processar erros da API"""
        error = data["error"]
        logger.error(f"Deriv API Error: {error['code']} - {error['message']}")

        # Handle specific errors
        if error["code"] == "InvalidToken":
            logger.error("Invalid API token - authentication failed")
            self.connection_state = ConnectionState.ERROR

    async def _wait_for_response(self, request_id: str, timeout: int = 30) -> Optional[Dict]:
        """Aguardar resposta de uma requisição específica"""
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return None

    async def _handle_disconnection(self):
        """Handle WebSocket disconnection"""
        self.connection_state = ConnectionState.DISCONNECTED

        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                await callback(False)
            except Exception as e:
                logger.error(f"Error in disconnection callback: {e}")

        # Attempt reconnection
        if self.reconnection_attempts < self.max_reconnection_attempts:
            self.reconnection_attempts += 1
            self.connection_state = ConnectionState.RECONNECTING

            logger.info(f"Attempting reconnection {self.reconnection_attempts}/{self.max_reconnection_attempts}")
            await asyncio.sleep(self.reconnection_delay)

            success = await self.connect()
            if success:
                self.reconnection_attempts = 0
                # Resubscribe to previous subscriptions
                await self._resubscribe()
        else:
            logger.error("Max reconnection attempts reached")
            self.connection_state = ConnectionState.ERROR

    async def _resubscribe(self):
        """Resubscrever aos ticks após reconexão"""
        for symbol in list(self.active_subscriptions):
            await self.subscribe_ticks(symbol)

    # Public API Methods

    async def subscribe_ticks(self, symbol: str) -> bool:
        """Subscrever aos ticks de um símbolo"""
        try:
            if self.connection_state not in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
                logger.error("Not connected to subscribe to ticks")
                return False

            request_id = self.generate_request_id()
            subscribe_message = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(subscribe_message))

            response = await self._wait_for_response(request_id)

            if response and "subscription" in response:
                self.active_subscriptions.add(symbol)
                logger.info(f"Successfully subscribed to {symbol} ticks")
                return True
            else:
                logger.error(f"Failed to subscribe to {symbol} ticks")
                return False

        except Exception as e:
            logger.error(f"Error subscribing to {symbol} ticks: {e}")
            return False

    async def unsubscribe_ticks(self, symbol: str) -> bool:
        """Cancelar subscrição aos ticks de um símbolo"""
        try:
            request_id = self.generate_request_id()
            unsubscribe_message = {
                "forget_all": "ticks",
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(unsubscribe_message))

            self.active_subscriptions.discard(symbol)
            logger.info(f"Unsubscribed from {symbol} ticks")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol} ticks: {e}")
            return False

    async def buy_contract(self, contract_type: str, symbol: str, amount: float,
                          duration: int, duration_unit: str = "s") -> Optional[RealTradeResult]:
        """Comprar contrato real"""
        try:
            if self.connection_state != ConnectionState.AUTHENTICATED:
                logger.error("Not authenticated to buy contracts")
                return None

            request_id = self.generate_request_id()
            buy_message = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "contract_type": contract_type,
                    "symbol": symbol,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "currency": self.account_info.currency if self.account_info else "USD"
                },
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(buy_message))

            response = await self._wait_for_response(request_id)

            if response and "buy" in response:
                trade_info = response["buy"]

                trade_result = RealTradeResult(
                    contract_id=trade_info["contract_id"],
                    buy_price=float(trade_info["buy_price"]),
                    payout=float(trade_info["payout"]),
                    start_time=datetime.fromtimestamp(trade_info["start_time"]),
                    duration=duration,
                    status="open"
                )

                self.active_contracts[trade_result.contract_id] = trade_result
                logger.info(f"Contract purchased: {trade_result.contract_id}")
                return trade_result
            else:
                logger.error("Failed to buy contract")
                return None

        except Exception as e:
            logger.error(f"Error buying contract: {e}")
            return None

    async def sell_contract(self, contract_id: str) -> Optional[float]:
        """Vender contrato antes do vencimento"""
        try:
            request_id = self.generate_request_id()
            sell_message = {
                "sell": contract_id,
                "price": 0,  # Sell at market price
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(sell_message))

            response = await self._wait_for_response(request_id)

            if response and "sell" in response:
                sell_price = float(response["sell"]["sold_for"])

                # Update contract status
                if contract_id in self.active_contracts:
                    self.active_contracts[contract_id].status = "closed"
                    self.active_contracts[contract_id].end_time = datetime.now()
                    self.active_contracts[contract_id].profit = sell_price - self.active_contracts[contract_id].buy_price

                logger.info(f"Contract sold: {contract_id} for {sell_price}")
                return sell_price
            else:
                logger.error(f"Failed to sell contract: {contract_id}")
                return None

        except Exception as e:
            logger.error(f"Error selling contract: {e}")
            return None

    async def get_balance(self) -> Optional[float]:
        """Obter saldo atual da conta"""
        try:
            if self.account_info:
                return self.account_info.balance

            request_id = self.generate_request_id()
            balance_message = {
                "balance": 1,
                "account": "all",
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(balance_message))

            response = await self._wait_for_response(request_id)

            if response and "balance" in response:
                balance = float(response["balance"]["balance"])
                if self.account_info:
                    self.account_info.balance = balance
                return balance
            else:
                logger.error("Failed to get balance")
                return None

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None

    async def get_portfolio(self) -> List[RealTradeResult]:
        """Obter portfolio atual"""
        try:
            request_id = self.generate_request_id()
            portfolio_message = {
                "portfolio": 1,
                "req_id": request_id
            }

            await self.rate_limiter.acquire()
            await self.websocket.send(json.dumps(portfolio_message))

            response = await self._wait_for_response(request_id)

            if response and "portfolio" in response:
                contracts = response["portfolio"]["contracts"]

                portfolio = []
                for contract in contracts:
                    contract_id = contract["contract_id"]

                    trade_result = RealTradeResult(
                        contract_id=contract_id,
                        buy_price=float(contract["buy_price"]),
                        payout=float(contract["payout"]),
                        profit=float(contract.get("profit", 0)),
                        status="closed" if contract.get("is_sold") == 1 else "open",
                        start_time=datetime.fromtimestamp(contract["purchase_time"])
                    )

                    portfolio.append(trade_result)
                    self.active_contracts[contract_id] = trade_result

                return portfolio
            else:
                logger.error("Failed to get portfolio")
                return []

        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return []

    def get_latest_tick(self, symbol: str) -> Optional[RealTickData]:
        """Obter último tick de um símbolo"""
        if symbol in self.tick_buffer and self.tick_buffer[symbol]:
            return self.tick_buffer[symbol][-1]
        return None

    def get_tick_history(self, symbol: str, count: int = 100) -> List[RealTickData]:
        """Obter histórico de ticks de um símbolo"""
        if symbol in self.tick_buffer:
            return list(self.tick_buffer[symbol])[-count:]
        return []

    def add_tick_callback(self, callback: Callable[[RealTickData], None]):
        """Adicionar callback para novos ticks"""
        self.tick_callbacks.append(callback)

    def add_balance_callback(self, callback: Callable[[float], None]):
        """Adicionar callback para mudanças de saldo"""
        self.balance_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable[[RealTradeResult], None]):
        """Adicionar callback para novos trades"""
        self.trade_callbacks.append(callback)

    def add_connection_callback(self, callback: Callable[[bool], None]):
        """Adicionar callback para mudanças de conexão"""
        self.connection_callbacks.append(callback)

    def is_connected(self) -> bool:
        """Verificar se está conectado"""
        return self.connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]

    def is_authenticated(self) -> bool:
        """Verificar se está autenticado"""
        return self.connection_state == ConnectionState.AUTHENTICATED

    async def disconnect(self):
        """Desconectar do WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connection_state = ConnectionState.DISCONNECTED
            logger.info("Disconnected from Deriv WebSocket")


# 🧪 Função de teste
async def test_real_deriv_client():
    """Testar cliente Deriv real"""
    # ATENÇÃO: Use apenas com conta demo para testes!
    APP_ID = "1089"  # Replace with your app ID
    API_TOKEN = None  # Replace with your demo token for testing

    client = RealDerivWebSocketClient(APP_ID, API_TOKEN)

    # Callback para ticks
    async def on_tick(tick_data: RealTickData):
        print(f"📊 Tick received: {tick_data.symbol} = {tick_data.tick} at {tick_data.timestamp}")

    # Callback para saldo
    async def on_balance(balance: float):
        print(f"💰 Balance updated: {balance}")

    # Callback para trades
    async def on_trade(trade_result: RealTradeResult):
        print(f"📈 Trade executed: {trade_result.contract_id} - Buy price: {trade_result.buy_price}")

    # Callback para conexão
    async def on_connection(connected: bool):
        print(f"🔌 Connection status: {'Connected' if connected else 'Disconnected'}")

    # Adicionar callbacks
    client.add_tick_callback(on_tick)
    client.add_balance_callback(on_balance)
    client.add_trade_callback(on_trade)
    client.add_connection_callback(on_connection)

    # Conectar
    connected = await client.connect()

    if connected:
        print("✅ Connected to Deriv successfully")

        # Subscrever aos ticks
        symbols = ["R_100", "R_50", "R_25"]
        for symbol in symbols:
            success = await client.subscribe_ticks(symbol)
            print(f"📊 Subscribed to {symbol}: {success}")

        # Manter conexão por 60 segundos para testar
        await asyncio.sleep(60)

        # Obter informações
        balance = await client.get_balance()
        print(f"💰 Current balance: {balance}")

        portfolio = await client.get_portfolio()
        print(f"📋 Portfolio contracts: {len(portfolio)}")

        # Desconectar
        await client.disconnect()
    else:
        print("❌ Failed to connect to Deriv")


if __name__ == "__main__":
    print("🔄 TESTING REAL DERIV CLIENT")
    print("=" * 40)
    print("⚠️  WARNING: This will connect to real Deriv API")
    print("   Make sure to use demo account for testing!")
    print("=" * 40)

    # Uncomment to run test (only with demo account!)
    # asyncio.run(test_real_deriv_client())
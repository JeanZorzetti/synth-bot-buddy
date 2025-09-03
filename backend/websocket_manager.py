import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from typing import Optional, Dict, Any, Callable
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"  
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

class DerivWebSocketManager:
    def __init__(self, app_id: str = "1089", api_token: Optional[str] = None):
        self.app_id = app_id
        self.api_token = api_token
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.state = ConnectionState.DISCONNECTED
        self.url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        # Event handlers
        self.on_tick: Optional[Callable] = None
        self.on_balance_update: Optional[Callable] = None
        self.on_trade_result: Optional[Callable] = None
        self.on_connection_status: Optional[Callable] = None
        
        # Connection management
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.heartbeat_interval = 30
        
        # Request management
        self.request_id_counter = 1
        self.pending_requests: Dict[int, Dict] = {}
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to Deriv API"""
        try:
            self.state = ConnectionState.CONNECTING
            self._notify_connection_status()
            
            logger.info(f"Connecting to Deriv WebSocket: {self.url}")
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.state = ConnectionState.CONNECTED
            self._notify_connection_status()
            logger.info("WebSocket connection established")
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            # Authenticate if token is provided
            if self.api_token:
                await self._authenticate()
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.state = ConnectionState.ERROR
            self._notify_connection_status()
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate using API token"""
        try:
            auth_request = {
                "authorize": self.api_token,
                "req_id": self._get_request_id()
            }
            
            await self._send_request(auth_request)
            logger.info("Authentication request sent")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def _message_listener(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.state = ConnectionState.DISCONNECTED
            self._notify_connection_status()
        except Exception as e:
            logger.error(f"Message listener error: {e}")
            self.state = ConnectionState.ERROR
            self._notify_connection_status()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("msg_type")
        req_id = data.get("req_id")
        
        logger.debug(f"Received message: {msg_type}")
        
        # Handle authentication response
        if msg_type == "authorize":
            if "error" in data:
                logger.error(f"Authentication error: {data['error']}")
                self.state = ConnectionState.ERROR
            else:
                logger.info("Successfully authenticated")
                self.state = ConnectionState.AUTHENTICATED
                # Get account balance after authentication
                await self.get_balance()
            self._notify_connection_status()
        
        # Handle tick data
        elif msg_type == "tick":
            if self.on_tick:
                tick_data = {
                    "symbol": data.get("tick", {}).get("symbol"),
                    "price": data.get("tick", {}).get("quote"),
                    "timestamp": data.get("tick", {}).get("epoch")
                }
                await self.on_tick(tick_data)
        
        # Handle balance updates  
        elif msg_type == "balance":
            if self.on_balance_update:
                balance_data = {
                    "balance": data.get("balance", {}).get("balance"),
                    "currency": data.get("balance", {}).get("currency")
                }
                await self.on_balance_update(balance_data)
        
        # Handle buy/sell responses
        elif msg_type in ["buy", "sell"]:
            if self.on_trade_result:
                trade_data = {
                    "contract_id": data.get("buy", {}).get("contract_id") or data.get("sell", {}).get("transaction_id"),
                    "payout": data.get("buy", {}).get("payout"),
                    "price": data.get("buy", {}).get("buy_price") or data.get("sell", {}).get("sold_for"),
                    "type": msg_type
                }
                await self.on_trade_result(trade_data)
        
        # Handle errors
        elif "error" in data:
            logger.error(f"API Error: {data['error']}")
        
        # Clean up pending requests
        if req_id and req_id in self.pending_requests:
            del self.pending_requests[req_id]
    
    async def subscribe_to_ticks(self, symbol: str = "R_75"):
        """Subscribe to real-time tick data for a symbol"""
        if self.state != ConnectionState.AUTHENTICATED:
            logger.error("Must be authenticated to subscribe to ticks")
            return False
        
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": self._get_request_id()
        }
        
        return await self._send_request(request)
    
    async def get_balance(self):
        """Get account balance"""
        if self.state != ConnectionState.AUTHENTICATED:
            logger.error("Must be authenticated to get balance")
            return False
        
        request = {
            "balance": 1,
            "subscribe": 1,
            "req_id": self._get_request_id()
        }
        
        return await self._send_request(request)
    
    async def buy_contract(self, contract_type: str, amount: float, duration: int, symbol: str = "R_75"):
        """Buy a contract"""
        if self.state != ConnectionState.AUTHENTICATED:
            logger.error("Must be authenticated to buy contracts")
            return False
        
        request = {
            "buy": 1,
            "parameters": {
                "contract_type": contract_type.upper(),
                "currency": "USD",
                "amount": amount,
                "duration": duration,
                "duration_unit": "t",  # ticks
                "symbol": symbol
            },
            "req_id": self._get_request_id()
        }
        
        return await self._send_request(request)
    
    async def _send_request(self, request: Dict[str, Any]) -> bool:
        """Send request via WebSocket"""
        if not self.websocket or self.websocket.closed:
            logger.error("WebSocket is not connected")
            return False
        
        try:
            message = json.dumps(request)
            await self.websocket.send(message)
            
            # Store pending request
            req_id = request.get("req_id")
            if req_id:
                self.pending_requests[req_id] = {
                    "request": request,
                    "timestamp": time.time()
                }
            
            logger.debug(f"Sent request: {request}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            return False
    
    async def _heartbeat(self):
        """Send periodic ping to keep connection alive"""
        while self.websocket and not self.websocket.closed:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self.websocket and not self.websocket.closed:
                    ping_request = {
                        "ping": 1,
                        "req_id": self._get_request_id()
                    }
                    await self._send_request(ping_request)
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break
    
    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        self.is_running = False
        
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        self.state = ConnectionState.DISCONNECTED
        self._notify_connection_status()
    
    def _get_request_id(self) -> int:
        """Get unique request ID"""
        request_id = self.request_id_counter
        self.request_id_counter += 1
        return request_id
    
    def _notify_connection_status(self):
        """Notify connection status change"""
        if self.on_connection_status:
            asyncio.create_task(self.on_connection_status(self.state))
    
    def set_tick_handler(self, handler: Callable):
        """Set tick data handler"""
        self.on_tick = handler
    
    def set_balance_handler(self, handler: Callable):
        """Set balance update handler"""
        self.on_balance_update = handler
    
    def set_trade_handler(self, handler: Callable):
        """Set trade result handler"""
        self.on_trade_result = handler
    
    def set_connection_handler(self, handler: Callable):
        """Set connection status handler"""
        self.on_connection_status = handler
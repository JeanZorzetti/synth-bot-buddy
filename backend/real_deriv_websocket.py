"""
Real Deriv WebSocket Client - Phase 13 Real-Time Data Pipeline
Cliente WebSocket real para Deriv Binary API com dados de mercado reais
"""

import os
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import hmac
import hashlib
import base64

from database_config import get_db_manager
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

class DerivAPIError(Exception):
    """Deriv API specific errors"""
    pass

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class TickData:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    price: float
    spread: float
    pip_size: float
    quote_id: str
    epoch: int

@dataclass
class CandleData:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    epoch: int

@dataclass
class MarketStatus:
    symbol: str
    is_trading_suspended: bool
    market_status: str
    exchange_is_open: bool
    timestamp: datetime

class RealDerivWebSocket:
    """Real-time WebSocket client for Deriv Binary API"""

    def __init__(self):
        # API Configuration
        self.api_url = os.getenv('DERIV_WS_URL', 'wss://ws.binaryws.com/websockets/v3')
        self.app_id = os.getenv('DERIV_APP_ID', '1089')  # Default app ID for testing
        self.api_token = os.getenv('DERIV_API_TOKEN', '')  # Optional for authorized calls

        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds

        # Subscriptions
        self.active_subscriptions: Dict[str, str] = {}  # symbol -> subscription_id
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.candle_callbacks: List[Callable[[CandleData], None]] = []
        self.market_status_callbacks: List[Callable[[MarketStatus], None]] = []

        # Message handling
        self.request_id_counter = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}

        # Supported symbols for real trading
        self.supported_symbols = {
            'R_10', 'R_25', 'R_50', 'R_75', 'R_100',  # Volatility indices
            'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY', 'frxAUDUSD', 'frxUSDCHF',  # Forex
            'frxEURGBP', 'frxEURJPY', 'frxGBPJPY', 'frxUSDCAD', 'frxNZDUSD',
            'cryBTCUSD', 'cryETHUSD', 'cryLTCUSD', 'cryBCHUSD', 'cryXRPUSD'  # Crypto
        }

        # Data storage
        self.db_manager = None
        self.cache_manager = None

        # Logging
        self.logger = logging_system.loggers.get('websocket', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize database and cache connections"""
        try:
            self.db_manager = await get_db_manager()
            self.cache_manager = await get_cache_manager()

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.INFO,
                "Real Deriv WebSocket client initialized"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'initialize'}
            )
            raise

    async def connect(self) -> bool:
        """Establish WebSocket connection to Deriv API"""
        if self.connection_status == ConnectionStatus.CONNECTED:
            return True

        try:
            self.connection_status = ConnectionStatus.CONNECTING

            # Connect to Deriv WebSocket API
            self.websocket = await websockets.connect(
                f"{self.api_url}?app_id={self.app_id}",
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )

            self.connection_status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.INFO,
                f"Connected to Deriv WebSocket API: {self.api_url}"
            )

            # Start message handler
            asyncio.create_task(self._message_handler())

            # Authorize if token is provided
            if self.api_token:
                await self._authorize()

            return True

        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'connect', 'url': self.api_url}
            )
            return False

    async def disconnect(self):
        """Disconnect from WebSocket"""
        try:
            if self.websocket and not self.websocket.closed:
                # Unsubscribe from all active subscriptions
                for symbol in list(self.active_subscriptions.keys()):
                    await self.unsubscribe_ticks(symbol)

                await self.websocket.close()

            self.connection_status = ConnectionStatus.DISCONNECTED
            self.websocket = None

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.INFO,
                "Disconnected from Deriv WebSocket API"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'disconnect'}
            )

    async def _authorize(self):
        """Authorize with API token"""
        try:
            request = {
                "authorize": self.api_token,
                "req_id": self._get_request_id()
            }

            response = await self._send_request(request)

            if 'error' in response:
                raise DerivAPIError(f"Authorization failed: {response['error']['message']}")

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.INFO,
                "Successfully authorized with Deriv API"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'authorize'}
            )
            raise

    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)

                except json.JSONDecodeError as e:
                    logging_system.log_error(
                        LogComponent.WEBSOCKET,
                        e,
                        {'message': message[:100], 'action': 'parse_message'}
                    )
                except Exception as e:
                    logging_system.log_error(
                        LogComponent.WEBSOCKET,
                        e,
                        {'action': 'process_message'}
                    )

        except websockets.exceptions.ConnectionClosed:
            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.WARNING,
                "WebSocket connection closed"
            )
            await self._handle_disconnect()

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'message_handler'}
            )
            await self._handle_disconnect()

    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming message"""
        req_id = data.get('req_id')
        msg_type = data.get('msg_type')

        # Handle pending requests
        if req_id and req_id in self.pending_requests:
            future = self.pending_requests.pop(req_id)
            if not future.cancelled():
                future.set_result(data)
            return

        # Handle subscription messages
        if msg_type == 'tick':
            await self._handle_tick_message(data)
        elif msg_type == 'candles':
            await self._handle_candle_message(data)
        elif msg_type == 'market_status':
            await self._handle_market_status_message(data)
        elif msg_type == 'error':
            await self._handle_error_message(data)

    async def _handle_tick_message(self, data: Dict[str, Any]):
        """Handle real-time tick data"""
        try:
            tick_data = data.get('tick', {})

            if not tick_data:
                return

            # Parse tick data
            tick = TickData(
                symbol=tick_data.get('symbol', ''),
                timestamp=datetime.fromtimestamp(tick_data.get('epoch', 0)),
                bid=float(tick_data.get('bid', 0)),
                ask=float(tick_data.get('ask', 0)),
                price=float(tick_data.get('quote', 0)),
                spread=float(tick_data.get('ask', 0)) - float(tick_data.get('bid', 0)),
                pip_size=float(tick_data.get('pip_size', 0.0001)),
                quote_id=tick_data.get('id', ''),
                epoch=tick_data.get('epoch', 0)
            )

            # Store in database
            await self._store_tick_data(tick)

            # Cache latest tick
            await self._cache_tick_data(tick)

            # Notify callbacks
            for callback in self.tick_callbacks:
                try:
                    await callback(tick)
                except Exception as e:
                    logging_system.log_error(
                        LogComponent.WEBSOCKET,
                        e,
                        {'callback': str(callback), 'symbol': tick.symbol}
                    )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'handle_tick_message', 'data': data}
            )

    async def _handle_candle_message(self, data: Dict[str, Any]):
        """Handle real-time candle data"""
        try:
            candles_data = data.get('candles', [])

            for candle_data in candles_data:
                candle = CandleData(
                    symbol=data.get('echo_req', {}).get('ticks_history', ''),
                    timestamp=datetime.fromtimestamp(candle_data.get('epoch', 0)),
                    open_price=float(candle_data.get('open', 0)),
                    high_price=float(candle_data.get('high', 0)),
                    low_price=float(candle_data.get('low', 0)),
                    close_price=float(candle_data.get('close', 0)),
                    volume=int(candle_data.get('volume', 0)),
                    epoch=candle_data.get('epoch', 0)
                )

                # Store in database
                await self._store_candle_data(candle)

                # Notify callbacks
                for callback in self.candle_callbacks:
                    try:
                        await callback(candle)
                    except Exception as e:
                        logging_system.log_error(
                            LogComponent.WEBSOCKET,
                            e,
                            {'callback': str(callback), 'symbol': candle.symbol}
                        )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'handle_candle_message', 'data': data}
            )

    async def _handle_market_status_message(self, data: Dict[str, Any]):
        """Handle market status updates"""
        try:
            market_data = data.get('market_status', {})

            if not market_data:
                return

            status = MarketStatus(
                symbol=market_data.get('symbol', ''),
                is_trading_suspended=market_data.get('is_trading_suspended', False),
                market_status=market_data.get('market_status', ''),
                exchange_is_open=market_data.get('exchange_is_open', False),
                timestamp=datetime.utcnow()
            )

            # Cache market status
            await self._cache_market_status(status)

            # Notify callbacks
            for callback in self.market_status_callbacks:
                try:
                    await callback(status)
                except Exception as e:
                    logging_system.log_error(
                        LogComponent.WEBSOCKET,
                        e,
                        {'callback': str(callback), 'symbol': status.symbol}
                    )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'handle_market_status_message', 'data': data}
            )

    async def _handle_error_message(self, data: Dict[str, Any]):
        """Handle error messages"""
        error = data.get('error', {})
        error_code = error.get('code')
        error_message = error.get('message', 'Unknown error')

        logging_system.log(
            LogComponent.WEBSOCKET,
            LogLevel.ERROR,
            f"Deriv API Error: {error_code} - {error_message}",
            {'error_data': error}
        )

    async def _handle_disconnect(self):
        """Handle connection disconnect"""
        self.connection_status = ConnectionStatus.DISCONNECTED

        # Attempt reconnection
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self.connection_status = ConnectionStatus.RECONNECTING

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.WARNING,
                f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}"
            )

            await asyncio.sleep(self.reconnect_delay)

            success = await self.connect()
            if success:
                # Resubscribe to active subscriptions
                await self._resubscribe_all()

    async def _resubscribe_all(self):
        """Resubscribe to all active subscriptions after reconnection"""
        try:
            symbols_to_resubscribe = list(self.active_subscriptions.keys())
            self.active_subscriptions.clear()

            for symbol in symbols_to_resubscribe:
                await self.subscribe_ticks(symbol)

            logging_system.log(
                LogComponent.WEBSOCKET,
                LogLevel.INFO,
                f"Resubscribed to {len(symbols_to_resubscribe)} symbols"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'resubscribe_all'}
            )

    async def subscribe_ticks(self, symbol: str) -> bool:
        """Subscribe to real-time tick data for symbol"""
        try:
            if symbol in self.active_subscriptions:
                return True

            if symbol not in self.supported_symbols:
                logging_system.log(
                    LogComponent.WEBSOCKET,
                    LogLevel.WARNING,
                    f"Symbol {symbol} not in supported symbols list"
                )

            request = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": self._get_request_id()
            }

            response = await self._send_request(request)

            if 'error' in response:
                raise DerivAPIError(f"Subscription failed: {response['error']['message']}")

            subscription_id = response.get('subscription', {}).get('id')
            if subscription_id:
                self.active_subscriptions[symbol] = subscription_id

                logging_system.log(
                    LogComponent.WEBSOCKET,
                    LogLevel.INFO,
                    f"Subscribed to tick data for {symbol}",
                    {'subscription_id': subscription_id}
                )
                return True

            return False

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'subscribe_ticks', 'symbol': symbol}
            )
            return False

    async def unsubscribe_ticks(self, symbol: str) -> bool:
        """Unsubscribe from tick data for symbol"""
        try:
            if symbol not in self.active_subscriptions:
                return True

            subscription_id = self.active_subscriptions[symbol]

            request = {
                "forget": subscription_id,
                "req_id": self._get_request_id()
            }

            response = await self._send_request(request)

            if 'error' not in response:
                del self.active_subscriptions[symbol]
                logging_system.log(
                    LogComponent.WEBSOCKET,
                    LogLevel.INFO,
                    f"Unsubscribed from tick data for {symbol}"
                )
                return True

            return False

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'unsubscribe_ticks', 'symbol': symbol}
            )
            return False

    async def get_active_symbols(self) -> Dict[str, Any]:
        """Get list of active trading symbols"""
        try:
            request = {
                "active_symbols": "brief",
                "product_type": "basic",
                "req_id": self._get_request_id()
            }

            response = await self._send_request(request)

            if 'error' in response:
                raise DerivAPIError(f"Failed to get active symbols: {response['error']['message']}")

            return response.get('active_symbols', [])

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'get_active_symbols'}
            )
            return []

    async def get_candles_history(self, symbol: str, granularity: int = 60, count: int = 1000) -> List[CandleData]:
        """Get historical candles data"""
        try:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles",
                "req_id": self._get_request_id()
            }

            response = await self._send_request(request)

            if 'error' in response:
                raise DerivAPIError(f"Failed to get candles: {response['error']['message']}")

            candles = []
            candles_data = response.get('candles', [])

            for candle_data in candles_data:
                candle = CandleData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(candle_data.get('epoch', 0)),
                    open_price=float(candle_data.get('open', 0)),
                    high_price=float(candle_data.get('high', 0)),
                    low_price=float(candle_data.get('low', 0)),
                    close_price=float(candle_data.get('close', 0)),
                    volume=int(candle_data.get('volume', 0)),
                    epoch=candle_data.get('epoch', 0)
                )
                candles.append(candle)

            return candles

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'get_candles_history', 'symbol': symbol}
            )
            return []

    async def _store_tick_data(self, tick: TickData):
        """Store tick data in database"""
        try:
            if not self.db_manager:
                return

            market_data = {
                'symbol': tick.symbol,
                'timestamp': tick.timestamp,
                'open_price': tick.price,
                'high_price': tick.price,
                'low_price': tick.price,
                'close_price': tick.price,
                'volume': 1,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.spread
            }

            await self.db_manager.store_market_data([market_data])

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'store_tick_data', 'symbol': tick.symbol}
            )

    async def _store_candle_data(self, candle: CandleData):
        """Store candle data in database"""
        try:
            if not self.db_manager:
                return

            market_data = {
                'symbol': candle.symbol,
                'timestamp': candle.timestamp,
                'open_price': candle.open_price,
                'high_price': candle.high_price,
                'low_price': candle.low_price,
                'close_price': candle.close_price,
                'volume': candle.volume
            }

            await self.db_manager.store_market_data([market_data])

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'store_candle_data', 'symbol': candle.symbol}
            )

    async def _cache_tick_data(self, tick: TickData):
        """Cache latest tick data"""
        try:
            if not self.cache_manager:
                return

            await self.cache_manager.cache_market_tick(
                tick.symbol,
                asdict(tick)
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'cache_tick_data', 'symbol': tick.symbol}
            )

    async def _cache_market_status(self, status: MarketStatus):
        """Cache market status"""
        try:
            if not self.cache_manager:
                return

            await self.cache_manager.set(
                CacheNamespace.MARKET_DATA,
                f"{status.symbol}:status",
                asdict(status),
                ttl=300  # 5 minutes
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.WEBSOCKET,
                e,
                {'action': 'cache_market_status', 'symbol': status.symbol}
            )

    async def _send_request(self, request: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """Send request and wait for response"""
        if not self.websocket or self.websocket.closed:
            raise DerivAPIError("WebSocket not connected")

        req_id = request.get('req_id')
        if not req_id:
            req_id = self._get_request_id()
            request['req_id'] = req_id

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[req_id] = future

        try:
            # Send request
            await self.websocket.send(json.dumps(request))

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            raise DerivAPIError(f"Request timeout: {request}")

        except Exception as e:
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            raise DerivAPIError(f"Request failed: {e}")

    def _get_request_id(self) -> str:
        """Generate unique request ID"""
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}_{int(datetime.utcnow().timestamp())}"

    # Callback management
    def add_tick_callback(self, callback: Callable[[TickData], None]):
        """Add callback for tick data"""
        self.tick_callbacks.append(callback)

    def remove_tick_callback(self, callback: Callable[[TickData], None]):
        """Remove tick callback"""
        if callback in self.tick_callbacks:
            self.tick_callbacks.remove(callback)

    def add_candle_callback(self, callback: Callable[[CandleData], None]):
        """Add callback for candle data"""
        self.candle_callbacks.append(callback)

    def remove_candle_callback(self, callback: Callable[[CandleData], None]):
        """Remove candle callback"""
        if callback in self.candle_callbacks:
            self.candle_callbacks.remove(callback)

    def add_market_status_callback(self, callback: Callable[[MarketStatus], None]):
        """Add callback for market status"""
        self.market_status_callbacks.append(callback)

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'status': self.connection_status.value,
            'reconnect_attempts': self.reconnect_attempts,
            'active_subscriptions': len(self.active_subscriptions),
            'subscribed_symbols': list(self.active_subscriptions.keys()),
            'pending_requests': len(self.pending_requests),
            'supported_symbols': list(self.supported_symbols),
            'callbacks': {
                'tick_callbacks': len(self.tick_callbacks),
                'candle_callbacks': len(self.candle_callbacks),
                'market_status_callbacks': len(self.market_status_callbacks)
            }
        }

# Global WebSocket client instance
deriv_websocket = RealDerivWebSocket()

async def get_deriv_websocket() -> RealDerivWebSocket:
    """Get initialized Deriv WebSocket client"""
    if deriv_websocket.connection_status == ConnectionStatus.DISCONNECTED:
        await deriv_websocket.initialize()
        await deriv_websocket.connect()

    return deriv_websocket
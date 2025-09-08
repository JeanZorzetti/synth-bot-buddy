import asyncio
import json
import logging
import websockets
import time
from websockets.exceptions import ConnectionClosed, WebSocketException
from typing import Optional, Dict, Any, Callable, List, Set
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"  
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    API_ERROR = "api_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TOKEN_EXPIRED = "token_expired"
    CONNECTION_TIMEOUT = "connection_timeout"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class SubscriptionRequest:
    symbol: str
    subscription_type: str  # 'ticks', 'candles', 'book'
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class PendingRequest:
    request_id: int
    request_data: Dict[str, Any]
    callback: Optional[Callable] = None
    timestamp: datetime = None
    timeout: int = 30
    retries: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > (self.timestamp + timedelta(seconds=self.timeout))

class MessageQueue:
    """Thread-safe message queue for reliability"""
    
    def __init__(self, maxsize: int = 1000):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._failed_messages = deque(maxlen=100)
        self._processed_count = 0
        self._failed_count = 0
        
    async def put(self, message: Dict[str, Any], priority: int = 0):
        """Add message to queue with priority"""
        try:
            await self._queue.put((priority, time.time(), message))
        except asyncio.QueueFull:
            logger.warning("Message queue full, discarding oldest message")
            try:
                await self._queue.get_nowait()
                await self._queue.put((priority, time.time(), message))
            except asyncio.QueueEmpty:
                pass
    
    async def get(self) -> Dict[str, Any]:
        """Get message from queue (highest priority first)"""
        priority, timestamp, message = await self._queue.get()
        return message
    
    def add_failed_message(self, message: Dict[str, Any]):
        """Add failed message to retry queue"""
        self._failed_messages.append({
            'message': message,
            'failed_at': datetime.utcnow(),
            'retry_count': 0
        })
        self._failed_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'queue_size': self._queue.qsize(),
            'processed_count': self._processed_count,
            'failed_count': self._failed_count,
            'failed_messages': len(self._failed_messages)
        }

class ErrorClassifier:
    """Classify and handle different types of errors"""
    
    ERROR_PATTERNS = {
        ErrorType.AUTHENTICATION_ERROR: [
            'InvalidToken', 'AuthorizationRequired', 'InvalidAppID',
            'TokenExpired', 'InvalidCredentials'
        ],
        ErrorType.RATE_LIMIT_ERROR: [
            'RateLimitExceeded', 'TooManyRequests', 'DisconnectionRate',
            'ConnectionLimitExceeded'
        ],
        ErrorType.API_ERROR: [
            'InvalidSymbol', 'MarketIsClosed', 'InvalidContract',
            'InsufficientBalance', 'InvalidAmount'
        ],
        ErrorType.NETWORK_ERROR: [
            'ConnectionError', 'NetworkTimeout', 'SocketError',
            'ConnectionLost', 'NetworkUnavailable'
        ]
    }
    
    @classmethod
    def classify_error(cls, error_message: str, error_code: str = None) -> ErrorType:
        """Classify error based on message and code"""
        error_text = f"{error_message} {error_code or ''}".lower()
        
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_text:
                    return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    @classmethod
    def get_recovery_strategy(cls, error_type: ErrorType) -> Dict[str, Any]:
        """Get recovery strategy for error type"""
        strategies = {
            ErrorType.AUTHENTICATION_ERROR: {
                'action': 'reauthenticate',
                'retry_delay': 5,
                'max_retries': 3
            },
            ErrorType.RATE_LIMIT_ERROR: {
                'action': 'throttle',
                'retry_delay': 30,
                'max_retries': 5
            },
            ErrorType.API_ERROR: {
                'action': 'validate_request',
                'retry_delay': 1,
                'max_retries': 1
            },
            ErrorType.NETWORK_ERROR: {
                'action': 'reconnect',
                'retry_delay': 10,
                'max_retries': 10
            },
            ErrorType.UNKNOWN_ERROR: {
                'action': 'reconnect',
                'retry_delay': 5,
                'max_retries': 3
            }
        }
        
        return strategies.get(error_type, strategies[ErrorType.UNKNOWN_ERROR])

class SubscriptionManager:
    """Manage WebSocket subscriptions"""
    
    def __init__(self):
        self.subscriptions: Dict[str, SubscriptionRequest] = {}
        self.active_symbols: Set[str] = set()
        self.subscription_callbacks: Dict[str, List[Callable]] = {}
        
    def add_subscription(self, subscription: SubscriptionRequest) -> str:
        """Add new subscription"""
        sub_id = f"{subscription.subscription_type}_{subscription.symbol}"
        self.subscriptions[sub_id] = subscription
        self.active_symbols.add(subscription.symbol)
        
        if subscription.callback:
            if sub_id not in self.subscription_callbacks:
                self.subscription_callbacks[sub_id] = []
            self.subscription_callbacks[sub_id].append(subscription.callback)
        
        logger.info(f"Added subscription: {sub_id}")
        return sub_id
    
    def remove_subscription(self, sub_id: str) -> bool:
        """Remove subscription"""
        if sub_id in self.subscriptions:
            subscription = self.subscriptions[sub_id]
            del self.subscriptions[sub_id]
            
            # Remove from active symbols if no other subscriptions
            symbol_subs = [s for s in self.subscriptions.values() 
                          if s.symbol == subscription.symbol]
            if not symbol_subs:
                self.active_symbols.discard(subscription.symbol)
            
            # Remove callbacks
            if sub_id in self.subscription_callbacks:
                del self.subscription_callbacks[sub_id]
            
            logger.info(f"Removed subscription: {sub_id}")
            return True
        
        return False
    
    def get_subscription_request(self, subscription: SubscriptionRequest) -> Dict[str, Any]:
        """Generate WebSocket request for subscription"""
        base_request = {
            "req_id": int(time.time() * 1000) % 100000
        }
        
        if subscription.subscription_type == "ticks":
            return {
                **base_request,
                "ticks": subscription.symbol,
                "subscribe": 1
            }
        elif subscription.subscription_type == "candles":
            return {
                **base_request,
                "ticks_history": subscription.symbol,
                "adjust_start_time": 1,
                "count": subscription.params.get('count', 1000),
                "end": "latest",
                "granularity": subscription.params.get('granularity', 60),
                "style": "candles",
                "subscribe": 1
            }
        elif subscription.subscription_type == "book":
            return {
                **base_request,
                "proposal_open_contract": 1,
                "contract_id": subscription.params.get('contract_id'),
                "subscribe": 1
            }
        
        return base_request
    
    def get_active_subscriptions(self) -> List[Dict[str, Any]]:
        """Get list of active subscriptions"""
        return [
            {
                'id': sub_id,
                'symbol': sub.symbol,
                'type': sub.subscription_type,
                'active': sub.active,
                'created_at': sub.created_at.isoformat()
            }
            for sub_id, sub in self.subscriptions.items()
        ]

class EnhancedDerivWebSocket:
    """Enhanced WebSocket manager with advanced features"""
    
    def __init__(self, app_id: str = "99188", api_token: Optional[str] = None):
        self.app_id = app_id
        self.api_token = api_token
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ConnectionState.DISCONNECTED
        self.url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        # Enhanced components
        self.subscription_manager = SubscriptionManager()
        self.message_queue = MessageQueue()
        self.error_classifier = ErrorClassifier()
        
        # Event handlers
        self.on_tick: Optional[Callable] = None
        self.on_balance_update: Optional[Callable] = None
        self.on_trade_result: Optional[Callable] = None
        self.on_connection_status: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Connection management
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.heartbeat_interval = 30
        self.connection_timeout = 30
        
        # Request management
        self.request_id_counter = 1
        self.pending_requests: Dict[int, PendingRequest] = {}
        self.request_rate_limiter = []
        self.max_requests_per_second = 10
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors_handled': 0,
            'reconnections': 0,
            'uptime_start': None
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
    async def connect(self) -> bool:
        """Enhanced connection with error handling and recovery"""
        try:
            self.state = ConnectionState.CONNECTING
            self._notify_connection_status()
            
            logger.info(f"Connecting to Deriv WebSocket: {self.url}")
            
            # Connection with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    compression=None  # Disable compression for better performance
                ),
                timeout=self.connection_timeout
            )
            
            self.state = ConnectionState.CONNECTED
            self._notify_connection_status()
            logger.info("Enhanced WebSocket connection established")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Authenticate if token is provided
            if self.api_token:
                await self._authenticate()
            
            # Record uptime start
            self.stats['uptime_start'] = datetime.utcnow()
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {self.connection_timeout} seconds")
            self.state = ConnectionState.ERROR
            self._notify_connection_status()
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.state = ConnectionState.ERROR
            self._notify_connection_status()
            await self._handle_error(str(e))
            return False
    
    async def _start_background_tasks(self):
        """Start all background tasks"""
        tasks = [
            self._message_listener(),
            self._message_processor(),
            self._heartbeat(),
            self._request_timeout_monitor(),
            self._subscription_monitor(),
            self._statistics_updater()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _authenticate(self) -> bool:
        """Enhanced authentication with error handling"""
        try:
            auth_request = {
                "authorize": self.api_token,
                "req_id": self._get_request_id()
            }
            
            await self._send_request(auth_request)
            logger.info("Authentication request sent")
            
            # Wait for authentication response (with timeout)
            auth_timeout = 15  # seconds
            start_time = time.time()
            
            while (self.state != ConnectionState.AUTHENTICATED and 
                   self.state != ConnectionState.ERROR and 
                   time.time() - start_time < auth_timeout):
                await asyncio.sleep(0.5)
            
            if self.state == ConnectionState.AUTHENTICATED:
                logger.info("Authentication successful")
                await self._restore_subscriptions()
                return True
            else:
                logger.error("Authentication failed or timed out")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._handle_error(str(e), ErrorType.AUTHENTICATION_ERROR)
            return False
    
    async def _message_listener(self):
        """Enhanced message listener with error handling"""
        try:
            while self.websocket and not self.websocket.closed:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=60  # 1 minute timeout
                    )
                    
                    self.stats['messages_received'] += 1
                    
                    # Add to message queue for processing
                    try:
                        data = json.loads(message)
                        await self.message_queue.put(data, priority=self._get_message_priority(data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        continue
                        
                except asyncio.TimeoutError:
                    logger.warning("Message receive timeout, checking connection...")
                    if self.websocket.closed:
                        break
                    continue
                except ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Message listener error: {e}")
                    await self._handle_error(str(e))
                    
        except Exception as e:
            logger.error(f"Message listener fatal error: {e}")
        finally:
            if not self.websocket.closed:
                await self.websocket.close()
    
    async def _message_processor(self):
        """Process messages from queue"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._process_message(message)
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, data: Dict[str, Any]):
        """Enhanced message processing with error detection"""
        try:
            # Check for errors first
            if 'error' in data:
                await self._handle_api_error(data['error'])
                return
            
            # Handle authentication response
            if 'authorize' in data:
                if data.get('authorize'):
                    self.state = ConnectionState.AUTHENTICATED
                    self._notify_connection_status()
                    logger.info(f"Authenticated successfully: {data['authorize'].get('email', 'Unknown')}")
                else:
                    await self._handle_error("Authentication failed", ErrorType.AUTHENTICATION_ERROR)
                return
            
            # Handle subscription data
            if 'tick' in data:
                await self._handle_tick_data(data['tick'])
            elif 'candles' in data:
                await self._handle_candles_data(data['candles'])
            elif 'proposal' in data:
                await self._handle_proposal_data(data['proposal'])
            elif 'buy' in data:
                await self._handle_buy_response(data['buy'])
            elif 'balance' in data:
                await self._handle_balance_update(data['balance'])
            
            # Handle pending request responses
            req_id = data.get('req_id')
            if req_id and req_id in self.pending_requests:
                pending = self.pending_requests[req_id]
                if pending.callback:
                    await pending.callback(data)
                del self.pending_requests[req_id]
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.message_queue.add_failed_message(data)
    
    async def _handle_api_error(self, error_data: Dict[str, Any]):
        """Handle API errors with classification and recovery"""
        error_code = error_data.get('code', '')
        error_message = error_data.get('message', 'Unknown error')
        
        error_type = self.error_classifier.classify_error(error_message, error_code)
        
        logger.error(f"API Error [{error_type.value}]: {error_message} (Code: {error_code})")
        self.stats['errors_handled'] += 1
        
        # Get recovery strategy
        strategy = self.error_classifier.get_recovery_strategy(error_type)
        
        # Execute recovery action
        if strategy['action'] == 'reauthenticate':
            await self._reauthenticate()
        elif strategy['action'] == 'throttle':
            await self._throttle_requests(strategy['retry_delay'])
        elif strategy['action'] == 'reconnect':
            await self._schedule_reconnection(strategy['retry_delay'])
        
        # Notify error handler
        if self.on_error:
            await self.on_error(error_type, error_message, error_code)
    
    async def _handle_error(self, error_message: str, error_type: ErrorType = None):
        """Enhanced error handling with automatic recovery"""
        if error_type is None:
            error_type = self.error_classifier.classify_error(error_message)
        
        logger.error(f"Error [{error_type.value}]: {error_message}")
        self.stats['errors_handled'] += 1
        
        self.state = ConnectionState.ERROR
        self._notify_connection_status()
        
        # Get and execute recovery strategy
        strategy = self.error_classifier.get_recovery_strategy(error_type)
        
        if strategy['action'] == 'reconnect' and self.reconnect_attempts < self.max_reconnect_attempts:
            await self._schedule_reconnection(strategy['retry_delay'])
        
        # Notify error handler
        if self.on_error:
            await self.on_error(error_type, error_message, None)
    
    async def _schedule_reconnection(self, delay: int):
        """Schedule automatic reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return
        
        logger.info(f"Scheduling reconnection in {delay} seconds (attempt {self.reconnect_attempts + 1})")
        
        await asyncio.sleep(delay)
        
        self.reconnect_attempts += 1
        self.stats['reconnections'] += 1
        
        await self.reconnect()
    
    async def reconnect(self) -> bool:
        """Enhanced reconnection with subscription restoration"""
        logger.info("Attempting to reconnect...")
        
        # Close existing connection
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.state = ConnectionState.RECONNECTING
        self._notify_connection_status()
        
        # Clean up background tasks
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
        
        # Attempt connection
        success = await self.connect()
        
        if success:
            logger.info("Reconnection successful")
            self.reconnect_attempts = 0
            return True
        else:
            logger.error("Reconnection failed")
            return False
    
    async def _restore_subscriptions(self):
        """Restore all active subscriptions after reconnection"""
        logger.info("Restoring subscriptions after reconnection...")
        
        for sub_id, subscription in self.subscription_manager.subscriptions.items():
            if subscription.active:
                try:
                    request = self.subscription_manager.get_subscription_request(subscription)
                    await self._send_request(request)
                    logger.info(f"Restored subscription: {sub_id}")
                except Exception as e:
                    logger.error(f"Failed to restore subscription {sub_id}: {e}")
    
    # Subscription methods
    async def subscribe_to_ticks(self, symbol: str, callback: Optional[Callable] = None) -> str:
        """Subscribe to tick data for a symbol"""
        subscription = SubscriptionRequest(
            symbol=symbol,
            subscription_type="ticks",
            params={},
            callback=callback
        )
        
        sub_id = self.subscription_manager.add_subscription(subscription)
        
        if self.state == ConnectionState.AUTHENTICATED:
            request = self.subscription_manager.get_subscription_request(subscription)
            await self._send_request(request)
        
        return sub_id
    
    async def subscribe_to_candles(self, symbol: str, granularity: int = 60, 
                                 callback: Optional[Callable] = None) -> str:
        """Subscribe to candle data for a symbol"""
        subscription = SubscriptionRequest(
            symbol=symbol,
            subscription_type="candles",
            params={'granularity': granularity, 'count': 1000},
            callback=callback
        )
        
        sub_id = self.subscription_manager.add_subscription(subscription)
        
        if self.state == ConnectionState.AUTHENTICATED:
            request = self.subscription_manager.get_subscription_request(subscription)
            await self._send_request(request)
        
        return sub_id
    
    async def unsubscribe(self, sub_id: str) -> bool:
        """Unsubscribe from a subscription"""
        if sub_id in self.subscription_manager.subscriptions:
            subscription = self.subscription_manager.subscriptions[sub_id]
            
            # Send unsubscribe request
            if self.state == ConnectionState.AUTHENTICATED:
                unsubscribe_request = {
                    "forget": sub_id,
                    "req_id": self._get_request_id()
                }
                await self._send_request(unsubscribe_request)
            
            # Remove from manager
            return self.subscription_manager.remove_subscription(sub_id)
        
        return False
    
    # Enhanced utility methods
    async def _send_request(self, request: Dict[str, Any]) -> bool:
        """Enhanced request sending with rate limiting and queue"""
        try:
            # Rate limiting
            current_time = time.time()
            self.request_rate_limiter = [
                timestamp for timestamp in self.request_rate_limiter 
                if current_time - timestamp < 1  # Keep last second only
            ]
            
            if len(self.request_rate_limiter) >= self.max_requests_per_second:
                logger.warning("Rate limit reached, queueing request")
                await asyncio.sleep(1)
                return await self._send_request(request)
            
            if not self.websocket or self.websocket.closed:
                raise ConnectionError("WebSocket not connected")
            
            message = json.dumps(request)
            await self.websocket.send(message)
            
            self.request_rate_limiter.append(current_time)
            self.stats['messages_sent'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            self.message_queue.add_failed_message(request)
            return False
    
    def _get_message_priority(self, message: Dict[str, Any]) -> int:
        """Determine message priority (0 = highest)"""
        if 'error' in message:
            return 0  # Errors get highest priority
        elif 'tick' in message:
            return 1  # Tick data is high priority
        elif 'authorize' in message:
            return 0  # Authentication is highest priority
        else:
            return 2  # Other messages are lower priority
    
    async def _request_timeout_monitor(self):
        """Monitor and clean up expired requests"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_requests = []
                
                for req_id, pending in self.pending_requests.items():
                    if pending.is_expired:
                        expired_requests.append(req_id)
                
                for req_id in expired_requests:
                    pending = self.pending_requests[req_id]
                    logger.warning(f"Request {req_id} timed out")
                    
                    # Try retry if possible
                    if pending.retries < pending.max_retries:
                        pending.retries += 1
                        pending.timestamp = current_time
                        await self._send_request(pending.request_data)
                    else:
                        del self.pending_requests[req_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Request timeout monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _subscription_monitor(self):
        """Monitor subscription health"""
        while True:
            try:
                # Check if subscriptions are still receiving data
                # Implementation would check last received data timestamps
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Subscription monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _statistics_updater(self):
        """Update statistics periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Log current statistics
                if self.stats['uptime_start']:
                    uptime = datetime.utcnow() - self.stats['uptime_start']
                    logger.info(f"WebSocket Stats - Uptime: {uptime}, "
                              f"Messages: {self.stats['messages_received']}, "
                              f"Errors: {self.stats['errors_handled']}, "
                              f"Reconnections: {self.stats['reconnections']}")
                
            except Exception as e:
                logger.error(f"Statistics updater error: {e}")
                await asyncio.sleep(30)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        uptime = None
        if self.stats['uptime_start']:
            uptime = str(datetime.utcnow() - self.stats['uptime_start'])
        
        return {
            'connection_state': self.state.value,
            'uptime': uptime,
            'stats': self.stats,
            'subscriptions': self.subscription_manager.get_active_subscriptions(),
            'message_queue': self.message_queue.get_stats(),
            'pending_requests': len(self.pending_requests),
            'reconnect_attempts': self.reconnect_attempts,
            'max_reconnect_attempts': self.max_reconnect_attempts
        }
    
    # Event handler setters (keep compatibility with original)
    def set_tick_handler(self, handler: Callable):
        self.on_tick = handler
    
    def set_balance_handler(self, handler: Callable):
        self.on_balance_update = handler
    
    def set_trade_handler(self, handler: Callable):
        self.on_trade_result = handler
    
    def set_connection_handler(self, handler: Callable):
        self.on_connection_status = handler
    
    def set_error_handler(self, handler: Callable):
        self.on_error = handler
    
    # Compatibility methods
    def _get_request_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter
    
    def _notify_connection_status(self):
        if self.on_connection_status:
            asyncio.create_task(self.on_connection_status(self.state))
    
    async def disconnect(self):
        """Clean disconnect"""
        self.is_running = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.state = ConnectionState.DISCONNECTED
        self._notify_connection_status()
        
        logger.info("WebSocket disconnected cleanly")
    
    # Additional handler methods for enhanced features
    async def _handle_tick_data(self, tick_data: Dict[str, Any]):
        if self.on_tick:
            await self.on_tick(tick_data)
    
    async def _handle_candles_data(self, candles_data: Dict[str, Any]):
        # Handle candles data (implement as needed)
        pass
    
    async def _handle_proposal_data(self, proposal_data: Dict[str, Any]):
        # Handle proposal data (implement as needed)
        pass
    
    async def _handle_buy_response(self, buy_data: Dict[str, Any]):
        if self.on_trade_result:
            await self.on_trade_result(buy_data)
    
    async def _handle_balance_update(self, balance_data: Dict[str, Any]):
        if self.on_balance_update:
            await self.on_balance_update(balance_data)
    
    async def _heartbeat(self):
        """Enhanced heartbeat with connection monitoring"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.websocket and not self.websocket.closed:
                    # Send ping request to keep connection alive
                    ping_request = {
                        "ping": 1,
                        "req_id": self._get_request_id()
                    }
                    await self._send_request(ping_request)
                else:
                    # Connection lost, attempt reconnection
                    logger.warning("Connection lost during heartbeat, attempting reconnection")
                    await self._handle_error("Connection lost", ErrorType.NETWORK_ERROR)
                    break
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await self._handle_error(str(e), ErrorType.NETWORK_ERROR)
                break
    
    async def _reauthenticate(self):
        """Reauthenticate with current token"""
        if self.api_token:
            logger.info("Attempting reauthentication...")
            await self._authenticate()
        else:
            logger.error("Cannot reauthenticate: No API token available")
    
    async def _throttle_requests(self, delay: int):
        """Throttle requests for specified delay"""
        logger.info(f"Throttling requests for {delay} seconds due to rate limiting")
        await asyncio.sleep(delay)
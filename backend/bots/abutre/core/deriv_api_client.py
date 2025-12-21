"""
ABUTRE BOT - Deriv API WebSocket Client

Features:
- Async WebSocket connection
- Auto-reconnect on disconnect
- Rate limiting
- Tick stream subscription
- Balance monitoring
- Order execution
"""
import asyncio
import json
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

from ..config import config
from ..utils.logger import default_logger as logger


class DerivAPIClient:
    """Async WebSocket client for Deriv API"""

    def __init__(
        self,
        api_token: str = None,
        ws_url: str = None,
        on_tick: Callable = None,
        on_balance: Callable = None,
        on_trade_result: Callable = None
    ):
        """
        Initialize Deriv API client

        Args:
            api_token: Deriv API token
            ws_url: WebSocket URL
            on_tick: Callback for tick updates (func(tick_data))
            on_balance: Callback for balance updates (func(balance))
            on_trade_result: Callback for trade results (func(result))
        """
        self.api_token = api_token or config.DERIV_API_TOKEN
        self.ws_url = ws_url or config.DERIV_WS_URL

        # Callbacks
        self.on_tick = on_tick
        self.on_balance = on_balance
        self.on_trade_result = on_trade_result

        # Connection state
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.reconnect_attempts = 0

        # Subscriptions
        self.subscriptions = {}

        # Rate limiting
        self.request_times = []
        self.max_requests_per_second = config.MAX_REQUESTS_PER_SECOND

        logger.info(f"DerivAPIClient initialized (URL: {self.ws_url})")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection

        Returns:
            True if connected successfully
        """
        try:
            logger.info("Connecting to Deriv API...")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Connected to Deriv API successfully")

            # Authorize
            await self.authorize()

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("Disconnected from Deriv API")

    async def reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        while self.reconnect_attempts < config.WS_MAX_RECONNECT_ATTEMPTS:
            self.reconnect_attempts += 1
            delay = min(config.WS_RECONNECT_DELAY * (2 ** self.reconnect_attempts), 60)

            logger.warning(
                f"Reconnect attempt {self.reconnect_attempts}/"
                f"{config.WS_MAX_RECONNECT_ATTEMPTS} in {delay}s..."
            )

            await asyncio.sleep(delay)

            if await self.connect():
                # Re-subscribe to previous subscriptions
                for sub_id, params in self.subscriptions.items():
                    await self.subscribe(**params)
                return True

        logger.critical("Max reconnect attempts reached. Giving up.")
        return False

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request and wait for response

        Args:
            request: Request payload

        Returns:
            Response data
        """
        # Rate limiting
        now = datetime.now().timestamp()
        self.request_times = [t for t in self.request_times if now - t < 1.0]

        if len(self.request_times) >= self.max_requests_per_second:
            wait_time = 1.0 - (now - self.request_times[0])
            logger.debug(f"Rate limit reached. Waiting {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)

        self.request_times.append(now)

        # Send request
        try:
            await self.ws.send(json.dumps(request))
            logger.debug(f"Sent request: {request.get('req_id', 'unknown')}")

            # Wait for response
            response_text = await self.ws.recv()
            response = json.loads(response_text)

            # Check for errors
            if 'error' in response:
                logger.error(f"API Error: {response['error']}")
                raise Exception(response['error']['message'])

            return response

        except ConnectionClosed:
            logger.error("Connection closed. Attempting reconnect...")
            self.is_connected = False
            await self.reconnect()
            raise

    async def authorize(self) -> Dict[str, Any]:
        """
        Authorize with API token

        Returns:
            Authorization response
        """
        logger.info("Authorizing with API token...")

        request = {
            "authorize": self.api_token,
            "req_id": f"auth_{datetime.now().timestamp()}"
        }

        response = await self._send_request(request)

        if 'authorize' in response:
            logger.info(
                f"Authorized successfully | "
                f"Currency: {response['authorize']['currency']} | "
                f"Balance: {response['authorize']['balance']}"
            )

            # Trigger balance callback
            if self.on_balance:
                await self.on_balance(float(response['authorize']['balance']))

        return response

    async def subscribe_ticks(self, symbol: str = None) -> str:
        """
        Subscribe to tick stream

        Args:
            symbol: Asset symbol (default: from config)

        Returns:
            Subscription ID
        """
        symbol = symbol or config.SYMBOL

        logger.info(f"Subscribing to ticks: {symbol}")

        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": f"ticks_{datetime.now().timestamp()}"
        }

        response = await self._send_request(request)

        if 'subscription' in response:
            sub_id = response['subscription']['id']
            self.subscriptions[sub_id] = {'symbol': symbol, 'type': 'ticks'}
            logger.info(f"Subscribed to ticks (ID: {sub_id})")
            return sub_id

        return None

    async def subscribe_balance(self) -> str:
        """
        Subscribe to balance updates

        Returns:
            Subscription ID
        """
        logger.info("Subscribing to balance updates")

        request = {
            "balance": 1,
            "subscribe": 1,
            "req_id": f"balance_{datetime.now().timestamp()}"
        }

        response = await self._send_request(request)

        if 'subscription' in response:
            sub_id = response['subscription']['id']
            self.subscriptions[sub_id] = {'type': 'balance'}
            logger.info(f"Subscribed to balance (ID: {sub_id})")
            return sub_id

        return None

    async def buy_contract(
        self,
        contract_type: str,
        symbol: str,
        amount: float,
        duration: int,
        duration_unit: str = 't',
        basis: str = 'stake'
    ) -> Dict[str, Any]:
        """
        Buy a contract

        Args:
            contract_type: 'CALL' or 'PUT'
            symbol: Asset symbol
            amount: Stake amount
            duration: Contract duration
            duration_unit: 't' (ticks) or 's' (seconds)
            basis: 'stake' or 'payout'

        Returns:
            Buy response
        """
        logger.info(
            f"Buying contract | Type: {contract_type} | "
            f"Symbol: {symbol} | Stake: ${amount:.2f} | "
            f"Duration: {duration}{duration_unit}"
        )

        # Get proposal first (to get contract_id)
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": basis,
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
            "req_id": f"proposal_{datetime.now().timestamp()}"
        }

        proposal_response = await self._send_request(proposal_request)

        if 'proposal' not in proposal_response:
            raise Exception("Failed to get proposal")

        # Buy contract
        buy_request = {
            "buy": proposal_response['proposal']['id'],
            "price": amount,
            "req_id": f"buy_{datetime.now().timestamp()}"
        }

        buy_response = await self._send_request(buy_request)

        if 'buy' in buy_response:
            logger.info(
                f"Contract purchased | ID: {buy_response['buy']['contract_id']} | "
                f"Payout: ${buy_response['buy']['payout']:.2f}"
            )

        return buy_response

    async def listen(self):
        """
        Main event loop - listen for incoming messages

        This should run in background asyncio task
        """
        logger.info("Starting message listener...")

        while self.is_connected:
            try:
                message_text = await self.ws.recv()
                message = json.loads(message_text)

                # Route message to appropriate handler
                if 'tick' in message:
                    await self._handle_tick(message['tick'])

                elif 'balance' in message:
                    await self._handle_balance(message['balance'])

                elif 'proposal_open_contract' in message:
                    await self._handle_trade_result(message['proposal_open_contract'])

                elif 'error' in message:
                    logger.error(f"API Error: {message['error']}")

            except ConnectionClosed:
                logger.error("Connection lost during listen()")
                self.is_connected = False
                await self.reconnect()
                break

            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)

    async def _handle_tick(self, tick_data: Dict[str, Any]):
        """Handle incoming tick data"""
        if self.on_tick:
            await self.on_tick(tick_data)

    async def _handle_balance(self, balance_data: Dict[str, Any]):
        """Handle balance updates"""
        balance = float(balance_data['balance'])
        logger.debug(f"Balance update: ${balance:.2f}")

        if self.on_balance:
            await self.on_balance(balance)

    async def _handle_trade_result(self, trade_data: Dict[str, Any]):
        """Handle trade result"""
        if self.on_trade_result:
            await self.on_trade_result(trade_data)


# Example usage
async def main():
    """Test the Deriv API client"""

    async def on_tick(tick):
        print(f"Tick: {tick['symbol']} = {tick['quote']}")

    async def on_balance(balance):
        print(f"Balance: ${balance:.2f}")

    client = DerivAPIClient(
        on_tick=on_tick,
        on_balance=on_balance
    )

    # Connect
    if await client.connect():
        # Subscribe to ticks and balance
        await client.subscribe_ticks()
        await client.subscribe_balance()

        # Start listening (this will run forever)
        await client.listen()


if __name__ == "__main__":
    asyncio.run(main())

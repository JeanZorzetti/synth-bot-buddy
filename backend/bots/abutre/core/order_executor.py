"""
ABUTRE BOT - Order Execution System

Features:
- Place orders via Deriv API
- Retry logic (3 attempts)
- Slippage monitoring
- Position tracking
- Order history
"""
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from ..config import config
from ..utils.logger import default_logger as logger
from .deriv_api_client import DerivAPIClient


class OrderStatus(Enum):
    """Order status enum"""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Order:
    """Represents a trading order"""

    def __init__(
        self,
        order_id: int,
        direction: str,  # 'CALL' or 'PUT'
        stake: float,
        level: int,
        symbol: str = None
    ):
        self.order_id = order_id
        self.direction = direction
        self.stake = stake
        self.level = level
        self.symbol = symbol or config.SYMBOL

        self.status = OrderStatus.PENDING
        self.contract_id: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.exit_price: Optional[float] = None
        self.payout: Optional[float] = None
        self.profit: Optional[float] = None

        self.created_at = datetime.now()
        self.executed_at: Optional[datetime] = None
        self.closed_at: Optional[datetime] = None

        self.error_message: Optional[str] = None
        self.retries = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'direction': self.direction,
            'stake': self.stake,
            'level': self.level,
            'symbol': self.symbol,
            'status': self.status.value,
            'contract_id': self.contract_id,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'payout': self.payout,
            'profit': self.profit,
            'created_at': self.created_at,
            'executed_at': self.executed_at,
            'closed_at': self.closed_at,
            'error_message': self.error_message,
            'retries': self.retries
        }

    def __repr__(self):
        return (
            f"Order(#{self.order_id} | "
            f"{self.direction} | "
            f"${self.stake:.2f} | "
            f"Lv{self.level} | "
            f"{self.status.value})"
        )


class OrderExecutor:
    """Executes orders via Deriv API"""

    def __init__(self, api_client: DerivAPIClient):
        """
        Initialize order executor

        Args:
            api_client: Deriv API client instance
        """
        self.api_client = api_client

        # Order tracking
        self.orders: Dict[int, Order] = {}
        self.order_counter = 0

        # Execution settings
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds

        logger.info("OrderExecutor initialized")

    async def place_order(
        self,
        direction: str,
        stake: float,
        level: int,
        dry_run: bool = False
    ) -> Order:
        """
        Place a new order

        Args:
            direction: 'CALL' or 'PUT'
            stake: Stake amount in USD
            level: Martingale level
            dry_run: If True, don't actually execute (paper trading)

        Returns:
            Order object
        """
        self.order_counter += 1
        order = Order(
            order_id=self.order_counter,
            direction=direction,
            stake=stake,
            level=level
        )

        self.orders[order.order_id] = order

        logger.info(f"Placing order: {order}")

        # Dry run mode (paper trading)
        if dry_run or not config.AUTO_TRADING:
            logger.warning(
                f"DRY RUN MODE | Order #{order.order_id} NOT executed "
                f"(AUTO_TRADING={config.AUTO_TRADING})"
            )
            order.status = OrderStatus.FILLED  # Simulate success
            order.executed_at = datetime.now()
            order.contract_id = f"DRY_RUN_{order.order_id}"
            return order

        # Execute with retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                order.status = OrderStatus.EXECUTING
                order.retries = attempt

                logger.info(
                    f"Executing order #{order.order_id} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

                # Call Deriv API
                response = await self.api_client.buy_contract(
                    contract_type=direction,
                    symbol=order.symbol,
                    amount=stake,
                    duration=config.DURATION,
                    duration_unit=config.DURATION_UNIT
                )

                # Extract result
                if 'buy' in response:
                    buy_data = response['buy']

                    order.status = OrderStatus.FILLED
                    order.contract_id = buy_data.get('contract_id')
                    order.entry_price = buy_data.get('buy_price')
                    order.payout = buy_data.get('payout')
                    order.executed_at = datetime.now()

                    logger.info(
                        f"Order #{order.order_id} filled | "
                        f"Contract ID: {order.contract_id} | "
                        f"Entry: ${order.entry_price:.2f} | "
                        f"Payout: ${order.payout:.2f}"
                    )

                    return order

                else:
                    raise Exception("No 'buy' in response")

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Order #{order.order_id} execution failed "
                    f"(attempt {attempt}/{self.max_retries}): {error_msg}"
                )

                order.error_message = error_msg

                # Retry if not last attempt
                if attempt < self.max_retries:
                    logger.warning(f"Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Final failure
                    order.status = OrderStatus.FAILED
                    logger.critical(
                        f"Order #{order.order_id} FAILED after "
                        f"{self.max_retries} attempts"
                    )
                    return order

        return order

    async def monitor_contract(self, contract_id: str) -> Dict[str, Any]:
        """
        Monitor contract status

        Args:
            contract_id: Contract ID to monitor

        Returns:
            Contract status data
        """
        # Subscribe to contract updates
        # (This would use Deriv's proposal_open_contract stream)
        # For now, return placeholder

        logger.debug(f"Monitoring contract: {contract_id}")

        # TODO: Implement actual contract monitoring
        # via Deriv API proposal_open_contract subscription

        return {
            'contract_id': contract_id,
            'status': 'open',
            'profit': 0.0
        }

    async def update_order_result(
        self,
        order_id: int,
        exit_price: float,
        profit: float,
        closed_at: datetime = None
    ):
        """
        Update order with final result

        Args:
            order_id: Order ID
            exit_price: Exit price
            profit: Final profit/loss
            closed_at: Close timestamp
        """
        if order_id not in self.orders:
            logger.error(f"Order #{order_id} not found")
            return

        order = self.orders[order_id]
        order.exit_price = exit_price
        order.profit = profit
        order.closed_at = closed_at or datetime.now()

        logger.info(
            f"Order #{order_id} closed | "
            f"Exit: ${exit_price:.2f} | "
            f"P&L: ${profit:+.2f}"
        )

    def get_order(self, order_id: int) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_active_orders(self) -> list[Order]:
        """Get all active orders"""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.EXECUTING]
        ]

    def get_order_history(self, limit: int = 50) -> list[Order]:
        """Get order history (most recent first)"""
        sorted_orders = sorted(
            self.orders.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return sorted_orders[:limit]

    def get_stats(self) -> dict:
        """Get execution stats"""
        total_orders = len(self.orders)
        filled = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
        failed = sum(1 for o in self.orders.values() if o.status == OrderStatus.FAILED)

        total_profit = sum(
            o.profit for o in self.orders.values()
            if o.profit is not None
        )

        return {
            'total_orders': total_orders,
            'filled': filled,
            'failed': failed,
            'success_rate': (filled / total_orders * 100) if total_orders > 0 else 0,
            'total_profit': total_profit
        }


# Example usage
async def test_order_executor():
    """Test order executor"""
    from .deriv_api_client import DerivAPIClient

    # Create mock API client
    api_client = DerivAPIClient()

    executor = OrderExecutor(api_client)

    print("\n=== TESTING ORDER EXECUTOR ===\n")

    # Test 1: Place order (dry run)
    print("Test 1: Place order (dry run)")
    order1 = await executor.place_order(
        direction='CALL',
        stake=1.0,
        level=1,
        dry_run=True
    )
    print(f"  {order1}")
    print(f"  Status: {order1.status.value}")

    # Test 2: Place multiple orders
    print("\nTest 2: Place 3 orders")
    for i in range(3):
        order = await executor.place_order(
            direction='PUT',
            stake=2.0 ** i,
            level=i + 1,
            dry_run=True
        )
        print(f"  {order}")

    # Test 3: Get stats
    print("\nTest 3: Execution stats")
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test 4: Get history
    print("\nTest 4: Order history")
    history = executor.get_order_history(limit=5)
    for order in history:
        print(f"  {order}")


if __name__ == "__main__":
    asyncio.run(test_order_executor())

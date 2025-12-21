"""
ABUTRE BOT - Market Data Handler

Features:
- Receive ticks in real-time
- Build M1 candles from ticks
- Detect candle color (green/red)
- Calculate streak count
- Maintain rolling buffer
"""
import asyncio
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime, timedelta
from collections import deque

from ..config import config
from ..utils.logger import default_logger as logger


class Candle:
    """Represents a 1-minute candle"""

    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.ticks: List[float] = []
        self.is_closed = False

    def add_tick(self, price: float):
        """Add a tick to the candle"""
        self.ticks.append(price)

        if self.open is None:
            self.open = price

        self.high = max(self.high, price) if self.high is not None else price
        self.low = min(self.low, price) if self.low is not None else price
        self.close = price

    def close_candle(self):
        """Mark candle as closed"""
        self.is_closed = True

    @property
    def color(self) -> int:
        """
        Get candle color
        Returns: 1 (green/bullish), -1 (red/bearish), 0 (doji)
        """
        if self.open is None or self.close is None:
            return 0

        if self.close > self.open:
            return 1  # Green (bullish)
        elif self.close < self.open:
            return -1  # Red (bearish)
        else:
            return 0  # Doji

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'color': self.color,
            'ticks_count': len(self.ticks),
            'is_closed': self.is_closed
        }

    def __repr__(self):
        color_str = {1: 'GREEN', -1: 'RED', 0: 'DOJI'}[self.color]
        return (
            f"Candle({self.timestamp.strftime('%H:%M:%S')} | "
            f"O:{self.open:.4f} H:{self.high:.4f} L:{self.low:.4f} C:{self.close:.4f} | "
            f"{color_str})"
        )


class MarketDataHandler:
    """Processes tick stream and builds M1 candles"""

    def __init__(
        self,
        buffer_size: int = 100,
        on_candle_closed: Callable = None,
        on_streak_detected: Callable = None
    ):
        """
        Initialize market data handler

        Args:
            buffer_size: Number of candles to keep in memory
            on_candle_closed: Callback when candle closes (func(candle))
            on_streak_detected: Callback when streak >= threshold (func(streak_count, direction))
        """
        self.buffer_size = buffer_size
        self.on_candle_closed = on_candle_closed
        self.on_streak_detected = on_streak_detected

        # Rolling buffer of closed candles
        self.candles: deque[Candle] = deque(maxlen=buffer_size)

        # Current (incomplete) candle
        self.current_candle: Optional[Candle] = None

        # Streak tracking
        self.current_streak_count = 0
        self.current_streak_direction = 0  # 1 or -1

        logger.info(f"MarketDataHandler initialized (buffer: {buffer_size} candles)")

    async def process_tick(self, tick_data: Dict[str, Any]):
        """
        Process incoming tick

        Args:
            tick_data: Tick data from Deriv API
                {
                    'quote': 123.45,
                    'epoch': 1234567890,
                    'symbol': '1HZ100V'
                }
        """
        price = float(tick_data['quote'])
        epoch = int(tick_data['epoch'])
        tick_time = datetime.fromtimestamp(epoch)

        # Round down to minute
        candle_time = tick_time.replace(second=0, microsecond=0)

        # Check if we need a new candle
        if self.current_candle is None or self.current_candle.timestamp != candle_time:
            # Close previous candle if exists
            if self.current_candle is not None:
                await self._close_candle(self.current_candle)

            # Start new candle
            self.current_candle = Candle(candle_time)
            logger.debug(f"New candle started: {candle_time.strftime('%H:%M:%S')}")

        # Add tick to current candle
        self.current_candle.add_tick(price)

    async def _close_candle(self, candle: Candle):
        """
        Close a candle and update streak tracking

        Args:
            candle: Candle to close
        """
        candle.close_candle()
        self.candles.append(candle)

        logger.info(f"Candle closed: {candle}")

        # Update streak tracking
        self._update_streak(candle)

        # Trigger callback
        if self.on_candle_closed:
            await self.on_candle_closed(candle)

    def _update_streak(self, candle: Candle):
        """Update current streak count"""
        color = candle.color

        # Skip dojis (they don't break streaks in our strategy)
        if color == 0:
            logger.debug("Doji detected - streak continues")
            return

        # Check if streak continues
        if color == self.current_streak_direction:
            self.current_streak_count += 1
        else:
            # New streak starts
            self.current_streak_direction = color
            self.current_streak_count = 1

        direction_str = "GREEN" if color == 1 else "RED"
        logger.info(
            f"Streak update: {self.current_streak_count} {direction_str} candles"
        )

        # Check if trigger threshold reached
        if self.current_streak_count >= config.DELAY_THRESHOLD:
            logger.warning(
                f"TRIGGER THRESHOLD REACHED: {self.current_streak_count} "
                f"{direction_str} candles!"
            )

            # Trigger callback
            if self.on_streak_detected:
                asyncio.create_task(
                    self.on_streak_detected(
                        self.current_streak_count,
                        self.current_streak_direction
                    )
                )

    def get_recent_candles(self, count: int = 10) -> List[Candle]:
        """
        Get N most recent closed candles

        Args:
            count: Number of candles to retrieve

        Returns:
            List of candles (newest last)
        """
        return list(self.candles)[-count:]

    def get_current_streak(self) -> tuple[int, int]:
        """
        Get current streak info

        Returns:
            (streak_count, direction)
        """
        return (self.current_streak_count, self.current_streak_direction)

    def reset_streak(self):
        """Reset streak counter (e.g., after entering a trade)"""
        logger.info("Streak counter reset")
        self.current_streak_count = 0
        self.current_streak_direction = 0


# Example usage
async def main():
    """Test the market data handler"""

    async def on_candle_closed(candle: Candle):
        print(f"Candle closed: {candle}")

    async def on_streak_detected(count: int, direction: int):
        dir_str = "GREEN" if direction == 1 else "RED"
        print(f"ALERT: {count} {dir_str} candles detected!")

    handler = MarketDataHandler(
        buffer_size=20,
        on_candle_closed=on_candle_closed,
        on_streak_detected=on_streak_detected
    )

    # Simulate tick stream
    base_time = datetime.now().replace(second=0, microsecond=0)

    # Simulate 10 minutes with random ticks
    import random
    price = 1000.0

    for minute in range(10):
        # 60 ticks per minute (1-second index)
        for second in range(60):
            tick_time = base_time + timedelta(minutes=minute, seconds=second)

            # Random walk
            price += random.uniform(-0.5, 0.5)

            tick_data = {
                'quote': price,
                'epoch': int(tick_time.timestamp()),
                'symbol': '1HZ100V'
            }

            await handler.process_tick(tick_data)

        # Simulate 1-second delay
        await asyncio.sleep(0.01)

    # Print recent candles
    print("\nRecent candles:")
    for candle in handler.get_recent_candles(10):
        print(f"  {candle}")

    print(f"\nCurrent streak: {handler.get_current_streak()}")


if __name__ == "__main__":
    asyncio.run(main())

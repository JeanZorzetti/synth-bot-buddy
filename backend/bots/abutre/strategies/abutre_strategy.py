"""
ABUTRE BOT - Delayed Martingale Strategy

Core Logic:
1. Monitor streak count (consecutive candles of same color)
2. When streak >= DELAY_THRESHOLD (8), prepare to enter
3. Enter AGAINST the trend (reversal bet)
4. Use Martingale up to MAX_LEVEL (10)
5. Reset on win

Validated Performance:
- ROI: +40.25% (180 days)
- Win Rate: 100% (1,018 trades)
- Max DD: 24.81%
"""
from typing import Optional, Tuple
from datetime import datetime

from ..config import config
from ..core.market_data_handler import Candle
from ..utils.logger import default_logger as logger, trade_logger


class TradingSignal:
    """Represents a trading signal"""

    def __init__(
        self,
        action: str,  # 'ENTER', 'LEVEL_UP', 'CLOSE', 'WAIT'
        direction: Optional[str] = None,  # 'CALL' or 'PUT'
        stake: Optional[float] = None,
        level: Optional[int] = None,
        reason: str = ""
    ):
        self.action = action
        self.direction = direction
        self.stake = stake
        self.level = level
        self.reason = reason
        self.timestamp = datetime.now()

    def __repr__(self):
        stake_str = f"${self.stake:.2f}" if self.stake is not None else "$0.00"
        level_str = str(self.level) if self.level is not None else "N/A"
        direction_str = self.direction if self.direction is not None else "N/A"

        return (
            f"TradingSignal({self.action} | "
            f"Direction: {direction_str} | "
            f"Stake: {stake_str} | "
            f"Level: {level_str} | "
            f"Reason: {self.reason})"
        )


class PositionState:
    """Tracks current position state"""

    def __init__(self):
        self.in_position = False
        self.entry_time: Optional[datetime] = None
        self.entry_candle_idx: int = 0
        self.entry_streak_size: int = 0
        self.direction: Optional[str] = None  # 'CALL' or 'PUT'
        self.current_level: int = 0
        self.current_stake: float = 0.0
        self.total_loss: float = 0.0
        self.contract_id: Optional[str] = None

    def open_position(
        self,
        direction: str,
        streak_size: int,
        stake: float,
        candle_idx: int
    ):
        """Open new position"""
        self.in_position = True
        self.entry_time = datetime.now()
        self.entry_candle_idx = candle_idx
        self.entry_streak_size = streak_size
        self.direction = direction
        self.current_level = 1
        self.current_stake = stake
        self.total_loss = 0.0

        trade_logger.trade_opened(
            trade_id=candle_idx,
            direction=direction,
            level=1,
            stake=stake
        )

    def level_up(self):
        """Increase Martingale level"""
        self.current_level += 1
        self.total_loss += self.current_stake
        self.current_stake *= config.MULTIPLIER

        trade_logger.level_up(
            trade_id=self.entry_candle_idx,
            new_level=self.current_level,
            new_stake=self.current_stake
        )

    def close_position(self, result: str, profit: float, balance: float):
        """Close position"""
        trade_logger.trade_closed(
            trade_id=self.entry_candle_idx,
            result=result,
            profit=profit,
            balance=balance,
            levels_used=self.current_level
        )

        # Reset state
        self.in_position = False
        self.entry_time = None
        self.entry_candle_idx = 0
        self.entry_streak_size = 0
        self.direction = None
        self.current_level = 0
        self.current_stake = 0.0
        self.total_loss = 0.0
        self.contract_id = None

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'in_position': self.in_position,
            'entry_time': self.entry_time,
            'entry_streak_size': self.entry_streak_size,
            'direction': self.direction,
            'current_level': self.current_level,
            'current_stake': self.current_stake,
            'total_loss': self.total_loss,
            'contract_id': self.contract_id
        }


class AbutreStrategy:
    """Delayed Martingale Strategy Implementation"""

    def __init__(self):
        """Initialize strategy"""
        self.position = PositionState()
        self.candle_count = 0

        logger.info("AbutreStrategy initialized")
        logger.info(f"  Delay Threshold: {config.DELAY_THRESHOLD} candles")
        logger.info(f"  Max Level: {config.MAX_LEVEL}")
        logger.info(f"  Initial Stake: ${config.INITIAL_STAKE}")
        logger.info(f"  Multiplier: {config.MULTIPLIER}x")

    def detect_trigger(self, streak_count: int, streak_direction: int) -> bool:
        """
        Detect if trigger condition is met

        Args:
            streak_count: Current streak count
            streak_direction: 1 (green) or -1 (red)

        Returns:
            True if trigger detected
        """
        # Already in position - no new triggers
        if self.position.in_position:
            return False

        # Check if streak >= threshold
        if streak_count >= config.DELAY_THRESHOLD:
            direction_str = "GREEN" if streak_direction == 1 else "RED"
            logger.warning(
                f"TRIGGER DETECTED | Streak: {streak_count} {direction_str} candles"
            )
            return True

        return False

    def get_bet_direction(self, streak_direction: int) -> str:
        """
        Get bet direction (AGAINST the trend)

        Args:
            streak_direction: 1 (green) or -1 (red)

        Returns:
            'CALL' (buy/green) or 'PUT' (sell/red)
        """
        # Bet AGAINST the trend (reversal)
        if streak_direction == 1:  # Green streak
            return 'PUT'  # Bet on red (reversal)
        else:  # Red streak
            return 'CALL'  # Bet on green (reversal)

    def calculate_position_size(self, level: int) -> float:
        """
        Calculate stake size for given level

        Args:
            level: Martingale level (1-based)

        Returns:
            Stake amount in USD
        """
        return config.INITIAL_STAKE * (config.MULTIPLIER ** (level - 1))

    def should_level_up(self, candle: Candle) -> bool:
        """
        Check if we should increase Martingale level (we lost)

        Args:
            candle: Latest closed candle

        Returns:
            True if should level up
        """
        if not self.position.in_position:
            return False

        candle_direction_str = 'CALL' if candle.color == 1 else 'PUT'

        # If candle direction matches our bet -> WIN
        if candle_direction_str == self.position.direction:
            return False  # WIN - don't level up

        # LOSS - should level up (if not at max)
        return True

    def should_stop_loss(self) -> bool:
        """
        Check if we should stop loss (max level reached)

        Returns:
            True if should stop
        """
        if self.position.current_level >= config.MAX_LEVEL:
            logger.critical(
                f"MAX LEVEL REACHED ({config.MAX_LEVEL}) | "
                f"Total Loss: ${self.position.total_loss:.2f}"
            )
            return True

        return False

    def analyze_candle(
        self,
        candle: Candle,
        streak_count: int,
        streak_direction: int
    ) -> TradingSignal:
        """
        Analyze candle and generate trading signal

        Args:
            candle: Latest closed candle
            streak_count: Current streak count
            streak_direction: 1 (green) or -1 (red)

        Returns:
            TradingSignal object
        """
        self.candle_count += 1

        # ==================== CASE 1: NOT IN POSITION ====================
        if not self.position.in_position:
            # Check for trigger
            if self.detect_trigger(streak_count, streak_direction):
                direction = self.get_bet_direction(streak_direction)
                stake = self.calculate_position_size(level=1)

                return TradingSignal(
                    action='ENTER',
                    direction=direction,
                    stake=stake,
                    level=1,
                    reason=f"Trigger: {streak_count} consecutive candles"
                )

            # No trigger - keep waiting
            return TradingSignal(
                action='WAIT',
                reason=f"Streak: {streak_count}/{config.DELAY_THRESHOLD}"
            )

        # ==================== CASE 2: IN POSITION ====================

        # Check result of this candle
        candle_direction_str = 'CALL' if candle.color == 1 else 'PUT'

        # WIN?
        if candle_direction_str == self.position.direction:
            # Calculate profit
            profit = self.position.current_stake * (1 - config.SPREAD_PCT)
            profit -= self.position.total_loss

            return TradingSignal(
                action='CLOSE',
                direction=self.position.direction,
                stake=profit,
                level=self.position.current_level,
                reason=f"WIN at Level {self.position.current_level}"
            )

        # LOSS - should we level up?

        # Check if max level reached
        if self.should_stop_loss():
            total_loss = self.position.total_loss + self.position.current_stake

            return TradingSignal(
                action='CLOSE',
                direction=self.position.direction,
                stake=-total_loss,
                level=self.position.current_level,
                reason=f"STOP LOSS at Level {config.MAX_LEVEL}"
            )

        # Level up
        next_level = self.position.current_level + 1
        next_stake = self.calculate_position_size(next_level)

        return TradingSignal(
            action='LEVEL_UP',
            direction=self.position.direction,
            stake=next_stake,
            level=next_level,
            reason=f"LOSS at Level {self.position.current_level}, continuing..."
        )

    def execute_signal(self, signal: TradingSignal) -> bool:
        """
        Execute trading signal (update internal state)

        Note: Actual order execution happens in OrderExecutor
        This just updates strategy state

        Args:
            signal: TradingSignal to execute

        Returns:
            True if state updated successfully
        """
        if signal.action == 'WAIT':
            # Nothing to do
            return True

        elif signal.action == 'ENTER':
            # Open new position
            self.position.open_position(
                direction=signal.direction,
                streak_size=self.candle_count,
                stake=signal.stake,
                candle_idx=self.candle_count
            )
            logger.info(f"Position opened: {signal}")
            return True

        elif signal.action == 'LEVEL_UP':
            # Increase level
            self.position.level_up()
            logger.warning(f"Level increased: {signal}")
            return True

        elif signal.action == 'CLOSE':
            # Close position
            result = "WIN" if signal.stake > 0 else "LOSS"
            self.position.close_position(
                result=result,
                profit=signal.stake,
                balance=0.0  # Will be updated by RiskManager
            )
            logger.info(f"Position closed: {signal}")
            return True

        else:
            logger.error(f"Unknown signal action: {signal.action}")
            return False

    def get_position_state(self) -> dict:
        """Get current position state"""
        return self.position.to_dict()

    def reset(self):
        """Reset strategy state"""
        logger.info("Resetting strategy state")
        self.position = PositionState()
        self.candle_count = 0


# Example usage
def test_strategy():
    """Test the strategy with simulated candles"""
    strategy = AbutreStrategy()

    # Simulate scenario: 10 green candles in a row
    print("\n=== SIMULATING 10 GREEN CANDLES ===\n")

    for i in range(1, 11):
        # Create mock green candle
        candle = Candle(datetime.now())
        candle.open = 100.0
        candle.close = 100.5  # Green
        candle.close_candle()

        # Analyze
        signal = strategy.analyze_candle(candle, streak_count=i, streak_direction=1)
        print(f"Candle {i}: {signal}")

        # Execute signal
        if signal.action != 'WAIT':
            strategy.execute_signal(signal)
            print(f"  Position State: {strategy.get_position_state()}")

    # Now simulate reversal (red candle)
    print("\n=== RED CANDLE (REVERSAL) ===\n")

    candle = Candle(datetime.now())
    candle.open = 100.5
    candle.close = 100.0  # Red
    candle.close_candle()

    signal = strategy.analyze_candle(candle, streak_count=1, streak_direction=-1)
    print(f"Candle 11 (RED): {signal}")

    if signal.action != 'WAIT':
        strategy.execute_signal(signal)
        print(f"  Position State: {strategy.get_position_state()}")


if __name__ == "__main__":
    test_strategy()

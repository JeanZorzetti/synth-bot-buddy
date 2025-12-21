"""
ABUTRE BOT - Risk Management System

Safety Critical Features:
1. Max Level Enforcement (Stop at Level 10)
2. Max Drawdown Killer (Emergency stop at 25%)
3. Balance Validation (before each order)
4. Emergency Shutdown Protocol
5. Daily Loss Limit (optional)
"""
from typing import Optional, Tuple
from datetime import datetime, timedelta

from ..config import config
from ..utils.logger import default_logger as logger, trade_logger


class RiskViolation(Exception):
    """Raised when risk limit is violated"""
    pass


class RiskManager:
    """Manages risk limits and emergency stops"""

    def __init__(self, initial_balance: float = None):
        """
        Initialize risk manager

        Args:
            initial_balance: Starting balance (default: from config)
        """
        self.initial_balance = initial_balance or config.BANKROLL
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance

        # Drawdown tracking
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0

        # Daily limits (optional)
        self.daily_loss_limit = config.BANKROLL * 0.10  # 10% per day
        self.daily_loss = 0.0
        self.today_date = datetime.now().date()

        # Emergency flags
        self.emergency_stop = False
        self.emergency_reason = ""

        # Trade tracking
        self.total_trades = 0
        self.wins = 0
        self.losses = 0

        logger.info("RiskManager initialized")
        logger.info(f"  Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"  Max Drawdown Limit: {config.MAX_DRAWDOWN_PCT*100:.1f}%")
        logger.info(f"  Min Balance: ${config.MIN_BALANCE:.2f}")
        logger.info(f"  Daily Loss Limit: ${self.daily_loss_limit:.2f}")

    def validate_balance(self, required_amount: float) -> bool:
        """
        Check if balance is sufficient for trade

        Args:
            required_amount: Amount needed for trade

        Returns:
            True if balance sufficient

        Raises:
            RiskViolation: If balance insufficient
        """
        if self.current_balance < required_amount:
            raise RiskViolation(
                f"Insufficient balance | "
                f"Required: ${required_amount:.2f} | "
                f"Available: ${self.current_balance:.2f}"
            )

        if self.current_balance < config.MIN_BALANCE:
            raise RiskViolation(
                f"Balance below minimum | "
                f"Current: ${self.current_balance:.2f} | "
                f"Minimum: ${config.MIN_BALANCE:.2f}"
            )

        return True

    def validate_level(self, level: int) -> bool:
        """
        Check if Martingale level is within limits

        Args:
            level: Proposed level

        Returns:
            True if level allowed

        Raises:
            RiskViolation: If level exceeds max
        """
        if level > config.MAX_LEVEL:
            raise RiskViolation(
                f"Max level exceeded | "
                f"Proposed: {level} | "
                f"Maximum: {config.MAX_LEVEL}"
            )

        return True

    def update_balance(self, new_balance: float):
        """
        Update current balance and check drawdown

        Args:
            new_balance: New balance amount
        """
        old_balance = self.current_balance
        self.current_balance = new_balance

        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            logger.info(f"New peak balance: ${self.peak_balance:.2f}")

        # Calculate drawdown
        self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance

        if self.current_drawdown > self.max_drawdown_reached:
            self.max_drawdown_reached = self.current_drawdown
            logger.warning(
                f"New max drawdown: {self.max_drawdown_reached*100:.2f}%"
            )

        # Check drawdown limit
        if self.current_drawdown >= config.MAX_DRAWDOWN_PCT:
            self.trigger_emergency_stop(
                f"Max drawdown exceeded | "
                f"Current: {self.current_drawdown*100:.2f}% | "
                f"Limit: {config.MAX_DRAWDOWN_PCT*100:.1f}%"
            )

        # Update daily loss
        if new_balance < old_balance:
            loss = old_balance - new_balance
            self.daily_loss += loss

            # Check daily loss limit
            if self.daily_loss >= self.daily_loss_limit:
                self.trigger_emergency_stop(
                    f"Daily loss limit exceeded | "
                    f"Current: ${self.daily_loss:.2f} | "
                    f"Limit: ${self.daily_loss_limit:.2f}"
                )

        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date != self.today_date:
            logger.info(
                f"New trading day | "
                f"Previous day loss: ${self.daily_loss:.2f}"
            )
            self.daily_loss = 0.0
            self.today_date = current_date

    def record_trade(self, profit: float, result: str):
        """
        Record trade result and update stats

        Args:
            profit: Profit/loss amount (negative for loss)
            result: 'WIN' or 'LOSS'
        """
        self.total_trades += 1

        if result == 'WIN':
            self.wins += 1
        else:
            self.losses += 1

        # Update balance
        new_balance = self.current_balance + profit
        self.update_balance(new_balance)

        # Log stats
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100)

        logger.info(
            f"Trade #{self.total_trades} | "
            f"Result: {result} | "
            f"P&L: ${profit:+.2f} | "
            f"Balance: ${self.current_balance:.2f} | "
            f"Win Rate: {win_rate:.1f}% | "
            f"ROI: {roi:+.2f}%"
        )

    def trigger_emergency_stop(self, reason: str):
        """
        Trigger emergency shutdown

        Args:
            reason: Reason for shutdown
        """
        self.emergency_stop = True
        self.emergency_reason = reason

        trade_logger.emergency_stop(reason, self.current_balance)

        logger.critical("="*70)
        logger.critical("EMERGENCY STOP TRIGGERED")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Final Balance: ${self.current_balance:.2f}")
        logger.critical(f"Total Drawdown: {self.current_drawdown*100:.2f}%")
        logger.critical("="*70)

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed

        Returns:
            (can_trade, reason)
        """
        if self.emergency_stop:
            return False, f"Emergency stop active: {self.emergency_reason}"

        if self.current_balance < config.MIN_BALANCE:
            return False, f"Balance below minimum (${config.MIN_BALANCE:.2f})"

        if self.current_drawdown >= config.MAX_DRAWDOWN_PCT:
            return False, f"Max drawdown exceeded ({config.MAX_DRAWDOWN_PCT*100:.1f}%)"

        if self.daily_loss >= self.daily_loss_limit:
            return False, f"Daily loss limit reached (${self.daily_loss_limit:.2f})"

        return True, "OK"

    def get_stats(self) -> dict:
        """Get current risk stats"""
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100)

        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown_pct': self.max_drawdown_reached * 100,
            'daily_loss': self.daily_loss,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate_pct': win_rate,
            'roi_pct': roi,
            'emergency_stop': self.emergency_stop,
            'emergency_reason': self.emergency_reason
        }

    def reset_daily_limits(self):
        """Reset daily loss counter (called at midnight)"""
        logger.info(f"Resetting daily limits | Previous loss: ${self.daily_loss:.2f}")
        self.daily_loss = 0.0
        self.today_date = datetime.now().date()

    def reset_emergency(self):
        """Reset emergency stop (use with caution!)"""
        logger.warning("Resetting emergency stop flag")
        self.emergency_stop = False
        self.emergency_reason = ""


# Example usage
def test_risk_manager():
    """Test risk manager"""
    rm = RiskManager(initial_balance=2000.0)

    print("\n=== TESTING RISK MANAGER ===\n")

    # Test 1: Validate balance
    print("Test 1: Validate balance")
    try:
        rm.validate_balance(50.0)
        print("  ✅ Balance sufficient for $50")
    except RiskViolation as e:
        print(f"  ❌ {e}")

    # Test 2: Validate level
    print("\nTest 2: Validate level")
    try:
        rm.validate_level(5)
        print("  ✅ Level 5 is allowed")
    except RiskViolation as e:
        print(f"  ❌ {e}")

    try:
        rm.validate_level(11)
        print("  ✅ Level 11 is allowed")
    except RiskViolation as e:
        print(f"  ❌ {e}")

    # Test 3: Record winning trades
    print("\nTest 3: Record 5 winning trades")
    for i in range(5):
        rm.record_trade(profit=10.0, result='WIN')

    print(f"\nStats after 5 wins:")
    stats = rm.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test 4: Record losing trades (trigger drawdown)
    print("\nTest 4: Simulate big loss (trigger drawdown)")
    rm.record_trade(profit=-600.0, result='LOSS')

    can_trade, reason = rm.can_trade()
    print(f"  Can trade? {can_trade} ({reason})")

    print(f"\nFinal Stats:")
    stats = rm.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_risk_manager()

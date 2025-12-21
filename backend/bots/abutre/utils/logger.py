"""
ABUTRE BOT - Structured Logging System

Features:
- Console + File output
- Rotating file handler (max 10MB per file)
- Color-coded levels
- Structured format with timestamps
- Separate error log
"""
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname:8}{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str = "abutre",
    log_level: str = "INFO",
    log_file: Path = None,
    error_log_file: Path = None
) -> logging.Logger:
    """
    Setup structured logger with console and file handlers

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to main log file
        error_log_file: Path to error-only log file

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers (avoid duplicates)
    logger.handlers.clear()

    # Format string
    log_format = (
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    date_format = '%Y-%m-%d %H:%M:%S'

    # ==================== CONSOLE HANDLER ====================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ==================== FILE HANDLER (ALL LOGS) ====================
    if log_file:
        # Create directory if not exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (max 10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # ==================== ERROR FILE HANDLER ====================
    if error_log_file:
        # Create directory if not exists
        error_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler for errors only
        error_file_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_formatter = logging.Formatter(log_format, datefmt=date_format)
        error_file_handler.setFormatter(error_file_formatter)
        logger.addHandler(error_file_handler)

    return logger


class TradeLogger:
    """Specialized logger for trade events"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def trigger_detected(self, streak_count: int, direction: str, candle_idx: int):
        """Log when trigger is detected"""
        self.logger.info(
            f"TRIGGER DETECTED | Streak: {streak_count} | Direction: {direction} | Candle: {candle_idx}"
        )

    def trade_opened(self, trade_id: int, direction: str, level: int, stake: float):
        """Log when trade is opened"""
        self.logger.info(
            f"TRADE OPENED | ID: {trade_id} | Direction: {direction} | "
            f"Level: {level} | Stake: ${stake:.2f}"
        )

    def level_up(self, trade_id: int, new_level: int, new_stake: float):
        """Log when Martingale level increases"""
        self.logger.warning(
            f"LEVEL UP | Trade: {trade_id} | New Level: {new_level} | "
            f"New Stake: ${new_stake:.2f}"
        )

    def trade_closed(
        self,
        trade_id: int,
        result: str,
        profit: float,
        balance: float,
        levels_used: int
    ):
        """Log when trade is closed"""
        emoji = "WIN" if result == "WIN" else "LOSS"
        level = logging.INFO if result == "WIN" else logging.ERROR

        self.logger.log(
            level,
            f"TRADE CLOSED {emoji} | ID: {trade_id} | P&L: ${profit:+.2f} | "
            f"Balance: ${balance:.2f} | Levels: {levels_used}"
        )

    def emergency_stop(self, reason: str, balance: float):
        """Log emergency shutdown"""
        self.logger.critical(
            f"EMERGENCY STOP | Reason: {reason} | Balance: ${balance:.2f}"
        )

    def system_alert(self, message: str):
        """Log system-level alert"""
        self.logger.warning(f"SYSTEM ALERT | {message}")


# Create default logger
default_logger = setup_logger(
    name="abutre",
    log_level="INFO",
    log_file=Path("backend/bots/abutre/logs/abutre.log"),
    error_log_file=Path("backend/bots/abutre/logs/errors.log")
)

# Create trade logger
trade_logger = TradeLogger(default_logger)


if __name__ == "__main__":
    # Test logging
    logger = setup_logger(
        name="test",
        log_level="DEBUG",
        log_file=Path("test.log"),
        error_log_file=Path("test_errors.log")
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test trade logger
    tl = TradeLogger(logger)
    tl.trigger_detected(8, "SELL", 1234)
    tl.trade_opened(1, "SELL", 1, 1.0)
    tl.level_up(1, 3, 4.0)
    tl.trade_closed(1, "WIN", 7.60, 2007.60, 4)
    tl.emergency_stop("Max Drawdown exceeded", 1500.0)

    print("\nLog files created:")
    print("  - test.log (all logs)")
    print("  - test_errors.log (errors only)")

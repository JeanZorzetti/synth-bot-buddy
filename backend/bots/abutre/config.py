"""
ABUTRE BOT - Configuration
Parametros validados em backtest (180 dias, +40.25% ROI)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AbutreConfig:
    """Configuration class for Abutre bot"""

    # ==================== STRATEGY PARAMETERS ====================
    # Validated in backtest: 180 days, $2000 -> $2805 (+40.25%)

    # Delay Threshold: Wait for N consecutive candles before entering
    # Historical max streak: 18 candles
    # Our capacity: 10 levels
    # Optimal delay: 18 - 10 = 8
    DELAY_THRESHOLD = int(os.getenv('DELAY_THRESHOLD', 8))

    # Max Martingale Level (safety limit)
    # Level 10 = $512 stake, $1023 accumulated loss
    MAX_LEVEL = int(os.getenv('MAX_LEVEL', 10))

    # Initial stake per trade (in USD)
    INITIAL_STAKE = float(os.getenv('INITIAL_STAKE', 1.0))

    # Martingale multiplier (classic = 2.0x)
    MULTIPLIER = float(os.getenv('MULTIPLIER', 2.0))

    # Starting bankroll
    BANKROLL = float(os.getenv('BANKROLL', 2000.0))

    # Max Drawdown tolerance (25% = emergency shutdown)
    MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT', 0.25))

    # ==================== DERIV API SETTINGS ====================

    # API Token (from Deriv dashboard)
    DERIV_API_TOKEN = os.getenv('DERIV_API_TOKEN', '')

    # App ID (REQUIRED - register at api.deriv.com)
    DERIV_APP_ID = os.getenv('DERIV_APP_ID', '99188')  # Default app_id

    # Deriv WebSocket URL
    DERIV_WS_URL = os.getenv('DERIV_WS_URL', 'wss://ws.derivws.com/websockets/v3')

    # Asset symbol to trade
    SYMBOL = os.getenv('SYMBOL', '1HZ100V')  # V100 1-second volatility index

    # Contract type
    CONTRACT_TYPE = os.getenv('CONTRACT_TYPE', 'CALL')  # or 'PUT'

    # Duration (in ticks or seconds)
    DURATION = int(os.getenv('DURATION', 1))  # 1 tick
    DURATION_UNIT = os.getenv('DURATION_UNIT', 't')  # 't' = ticks, 's' = seconds

    # ==================== RISK MANAGEMENT ====================

    # Spread/commission assumption (5% per trade)
    SPREAD_PCT = float(os.getenv('SPREAD_PCT', 0.05))

    # Emergency stop if balance drops below this
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', 500.0))

    # Auto-trading enabled
    AUTO_TRADING = os.getenv('AUTO_TRADING', 'false').lower() == 'true'

    # ==================== SYSTEM SETTINGS ====================

    # Database path
    DB_PATH = Path(os.getenv('DB_PATH', 'backend/bots/abutre/data/abutre.db'))

    # Log level (DEBUG, INFO, WARNING, ERROR)
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # Log file path
    LOG_FILE = Path(os.getenv('LOG_FILE', 'backend/bots/abutre/logs/abutre.log'))

    # WebSocket reconnect settings
    WS_RECONNECT_DELAY = int(os.getenv('WS_RECONNECT_DELAY', 5))  # seconds
    WS_MAX_RECONNECT_ATTEMPTS = int(os.getenv('WS_MAX_RECONNECT_ATTEMPTS', 10))

    # Rate limiting (avoid API ban)
    MAX_REQUESTS_PER_SECOND = int(os.getenv('MAX_REQUESTS_PER_SECOND', 5))

    # ==================== NOTIFICATIONS ====================

    # Telegram bot (optional)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

    # Email alerts (optional)
    EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '')
    SMTP_SERVER = os.getenv('SMTP_SERVER', '')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USER = os.getenv('SMTP_USER', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

    # ==================== CALCULATED VALUES ====================

    @classmethod
    def max_loss_at_level(cls, level: int) -> float:
        """Calculate accumulated loss at a given level"""
        return sum([cls.INITIAL_STAKE * (cls.MULTIPLIER ** i) for i in range(level)])

    @classmethod
    def stake_at_level(cls, level: int) -> float:
        """Calculate stake size at a given level"""
        return cls.INITIAL_STAKE * (cls.MULTIPLIER ** (level - 1))

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        if not cls.DERIV_API_TOKEN:
            errors.append("DERIV_API_TOKEN is required")

        if cls.DELAY_THRESHOLD < 5 or cls.DELAY_THRESHOLD > 15:
            errors.append(f"DELAY_THRESHOLD={cls.DELAY_THRESHOLD} out of safe range (5-15)")

        if cls.MAX_LEVEL < 8 or cls.MAX_LEVEL > 12:
            errors.append(f"MAX_LEVEL={cls.MAX_LEVEL} out of safe range (8-12)")

        max_capacity = cls.max_loss_at_level(cls.MAX_LEVEL)
        if max_capacity > cls.BANKROLL:
            errors.append(
                f"Bankroll ${cls.BANKROLL} insufficient for {cls.MAX_LEVEL} levels "
                f"(requires ${max_capacity:.2f})"
            )

        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("="*70)
        print("ABUTRE BOT - CONFIGURATION")
        print("="*70)
        print(f"\n[STRATEGY PARAMETERS]")
        print(f"  Delay Threshold:    {cls.DELAY_THRESHOLD} velas")
        print(f"  Max Level:          {cls.MAX_LEVEL}")
        print(f"  Initial Stake:      ${cls.INITIAL_STAKE:.2f}")
        print(f"  Multiplier:         {cls.MULTIPLIER}x")
        print(f"  Bankroll:           ${cls.BANKROLL:.2f}")
        print(f"  Max Drawdown:       {cls.MAX_DRAWDOWN_PCT*100:.1f}%")

        print(f"\n[ASSET]")
        print(f"  Symbol:             {cls.SYMBOL}")
        print(f"  Contract Type:      {cls.CONTRACT_TYPE}")
        print(f"  Duration:           {cls.DURATION} {cls.DURATION_UNIT}")

        print(f"\n[RISK LIMITS]")
        print(f"  Max Loss (Lv {cls.MAX_LEVEL}):    ${cls.max_loss_at_level(cls.MAX_LEVEL):.2f}")
        print(f"  Max Stake (Lv {cls.MAX_LEVEL}):   ${cls.stake_at_level(cls.MAX_LEVEL):.2f}")
        print(f"  Min Balance:        ${cls.MIN_BALANCE:.2f}")

        print(f"\n[SYSTEM]")
        print(f"  Auto-Trading:       {'ENABLED' if cls.AUTO_TRADING else 'DISABLED'}")
        print(f"  Log Level:          {cls.LOG_LEVEL}")
        print(f"  Database:           {cls.DB_PATH}")

        print(f"\n[NOTIFICATIONS]")
        print(f"  Telegram:           {'ENABLED' if cls.TELEGRAM_BOT_TOKEN else 'DISABLED'}")
        print(f"  Email:              {'ENABLED' if cls.EMAIL_ENABLED else 'DISABLED'}")

        print("\n" + "="*70 + "\n")


# Create singleton instance
config = AbutreConfig()


if __name__ == "__main__":
    # Test configuration
    if config.validate():
        config.print_summary()
    else:
        print("\nConfiguration validation FAILED!")
        exit(1)

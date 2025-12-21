"""
ABUTRE BOT - Database Management

Tables:
- trades: Complete trade history
- candles: M1 candle buffer
- balance_history: Equity curve
- system_events: Critical events log
"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ..config import config
from ..utils.logger import default_logger as logger

Base = declarative_base()


# ==================== MODELS ====================

class Trade(Base):
    """Trade history"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, unique=True, nullable=False)

    # Entry
    entry_time = Column(DateTime, nullable=False)
    entry_candle_idx = Column(Integer, nullable=False)
    entry_streak_size = Column(Integer, nullable=False)
    direction = Column(String(10), nullable=False)  # 'CALL' or 'PUT'

    # Execution
    initial_stake = Column(Float, nullable=False)
    max_level_reached = Column(Integer, nullable=False)
    contract_id = Column(String(100), nullable=True)

    # Exit
    exit_time = Column(DateTime, nullable=True)
    result = Column(String(20), nullable=True)  # 'WIN', 'LOSS', 'STOP_LOSS'
    profit = Column(Float, nullable=True)

    # Balance tracking
    balance_before = Column(Float, nullable=False)
    balance_after = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.now)


class Candle(Base):
    """M1 Candle data"""
    __tablename__ = 'candles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False)

    # OHLC
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)

    # Metadata
    color = Column(Integer, nullable=False)  # 1 (green), -1 (red), 0 (doji)
    ticks_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.now)


class BalanceHistory(Base):
    """Balance snapshots for equity curve"""
    __tablename__ = 'balance_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)

    balance = Column(Float, nullable=False)
    peak_balance = Column(Float, nullable=False)
    drawdown_pct = Column(Float, default=0.0)

    # Context
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.now)


class SystemEvent(Base):
    """System events log"""
    __tablename__ = 'system_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)

    event_type = Column(String(50), nullable=False)  # 'TRIGGER', 'TRADE_OPEN', 'TRADE_CLOSE', 'EMERGENCY_STOP', etc.
    severity = Column(String(20), nullable=False)  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message = Column(Text, nullable=False)

    # Optional context data (JSON-serialized)
    context = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.now)


# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Manages database operations"""

    def __init__(self, db_path: Path = None):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            echo=False  # Set True for SQL debugging
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Database initialized: {self.db_path}")

    def get_session(self) -> Session:
        """Get new database session"""
        return self.SessionLocal()

    # ==================== TRADE OPERATIONS ====================

    def insert_trade(
        self,
        trade_id: int,
        entry_time: datetime,
        entry_candle_idx: int,
        entry_streak_size: int,
        direction: str,
        initial_stake: float,
        balance_before: float
    ) -> Trade:
        """Insert new trade"""
        session = self.get_session()
        try:
            trade = Trade(
                trade_id=trade_id,
                entry_time=entry_time,
                entry_candle_idx=entry_candle_idx,
                entry_streak_size=entry_streak_size,
                direction=direction,
                initial_stake=initial_stake,
                max_level_reached=1,
                balance_before=balance_before
            )
            session.add(trade)
            session.commit()
            logger.debug(f"Trade #{trade_id} inserted into database")
            return trade
        finally:
            session.close()

    def update_trade(
        self,
        trade_id: int,
        exit_time: datetime = None,
        result: str = None,
        profit: float = None,
        balance_after: float = None,
        max_level_reached: int = None,
        contract_id: str = None
    ):
        """Update existing trade"""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if not trade:
                logger.error(f"Trade #{trade_id} not found in database")
                return

            if exit_time:
                trade.exit_time = exit_time
            if result:
                trade.result = result
            if profit is not None:
                trade.profit = profit
            if balance_after is not None:
                trade.balance_after = balance_after
            if max_level_reached is not None:
                trade.max_level_reached = max_level_reached
            if contract_id:
                trade.contract_id = contract_id

            session.commit()
            logger.debug(f"Trade #{trade_id} updated in database")
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 50) -> List[Trade]:
        """Get recent trades"""
        session = self.get_session()
        try:
            trades = session.query(Trade).order_by(Trade.entry_time.desc()).limit(limit).all()
            return trades
        finally:
            session.close()

    # ==================== CANDLE OPERATIONS ====================

    def insert_candle(
        self,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        color: int,
        ticks_count: int = 0
    ):
        """Insert candle"""
        session = self.get_session()
        try:
            candle = Candle(
                timestamp=timestamp,
                open=open,
                high=high,
                low=low,
                close=close,
                color=color,
                ticks_count=ticks_count
            )
            session.add(candle)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to insert candle: {e}")
        finally:
            session.close()

    def get_recent_candles(self, limit: int = 100) -> List[Candle]:
        """Get recent candles"""
        session = self.get_session()
        try:
            candles = session.query(Candle).order_by(Candle.timestamp.desc()).limit(limit).all()
            return candles
        finally:
            session.close()

    # ==================== BALANCE OPERATIONS ====================

    def insert_balance_snapshot(
        self,
        timestamp: datetime,
        balance: float,
        peak_balance: float,
        drawdown_pct: float,
        total_trades: int,
        wins: int,
        losses: int
    ):
        """Insert balance snapshot"""
        session = self.get_session()
        try:
            snapshot = BalanceHistory(
                timestamp=timestamp,
                balance=balance,
                peak_balance=peak_balance,
                drawdown_pct=drawdown_pct,
                total_trades=total_trades,
                wins=wins,
                losses=losses
            )
            session.add(snapshot)
            session.commit()
        finally:
            session.close()

    def get_equity_curve(self, limit: int = 1000) -> List[BalanceHistory]:
        """Get equity curve data"""
        session = self.get_session()
        try:
            snapshots = session.query(BalanceHistory).order_by(BalanceHistory.timestamp.asc()).limit(limit).all()
            return snapshots
        finally:
            session.close()

    # ==================== SYSTEM EVENTS ====================

    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        context: str = None
    ):
        """Log system event"""
        session = self.get_session()
        try:
            event = SystemEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                message=message,
                context=context
            )
            session.add(event)
            session.commit()
        finally:
            session.close()

    def get_recent_events(self, limit: int = 100) -> List[SystemEvent]:
        """Get recent events"""
        session = self.get_session()
        try:
            events = session.query(SystemEvent).order_by(SystemEvent.timestamp.desc()).limit(limit).all()
            return events
        finally:
            session.close()


# Singleton instance
db = DatabaseManager()


# Example usage
if __name__ == "__main__":
    # Test database
    print("Testing database...")

    # Insert test trade
    trade_id = 1
    db.insert_trade(
        trade_id=trade_id,
        entry_time=datetime.now(),
        entry_candle_idx=100,
        entry_streak_size=8,
        direction='CALL',
        initial_stake=1.0,
        balance_before=2000.0
    )

    # Update trade
    db.update_trade(
        trade_id=trade_id,
        exit_time=datetime.now(),
        result='WIN',
        profit=0.95,
        balance_after=2000.95,
        max_level_reached=1
    )

    # Insert candle
    db.insert_candle(
        timestamp=datetime.now(),
        open=100.0,
        high=100.5,
        low=99.5,
        close=100.2,
        color=1,
        ticks_count=60
    )

    # Insert balance snapshot
    db.insert_balance_snapshot(
        timestamp=datetime.now(),
        balance=2000.95,
        peak_balance=2000.95,
        drawdown_pct=0.0,
        total_trades=1,
        wins=1,
        losses=0
    )

    # Log event
    db.log_event(
        event_type='TRADE_OPEN',
        severity='INFO',
        message='Trade #1 opened',
        context='{"direction": "CALL", "stake": 1.0}'
    )

    # Query
    print("\nRecent trades:")
    for trade in db.get_recent_trades(limit=5):
        print(f"  Trade #{trade.trade_id} | {trade.direction} | {trade.result} | P&L: ${trade.profit}")

    print("\nRecent candles:")
    for candle in db.get_recent_candles(limit=5):
        color_str = {1: 'GREEN', -1: 'RED', 0: 'DOJI'}[candle.color]
        print(f"  {candle.timestamp} | {color_str} | O:{candle.open:.2f} C:{candle.close:.2f}")

    print("\nRecent events:")
    for event in db.get_recent_events(limit=5):
        print(f"  [{event.severity}] {event.event_type}: {event.message}")

    print("\nDatabase test complete!")

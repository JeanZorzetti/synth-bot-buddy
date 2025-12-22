"""
ABUTRE BOT - Main Entry Point

Delayed Martingale Strategy for V100/V75
Validated: +40.25% ROI in 180 days (backtest)

Usage:
    python main.py [--demo] [--paper-trading]

Arguments:
    --demo: Use demo account (recommended for testing)
    --paper-trading: Monitor only, don't execute trades
"""
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from .config import config
from .core.deriv_api_client import DerivAPIClient
from .core.market_data_handler import MarketDataHandler, Candle
from .core.order_executor import OrderExecutor
from .core.database import db
from .core.async_db_writer import AsyncDatabaseWriter, get_async_db_writer
from .core.websocket_server import WebSocketServer
from .strategies.abutre_strategy import AbutreStrategy
from .strategies.risk_manager import RiskManager, RiskViolation
from .utils.logger import default_logger as logger, trade_logger

# Import manager for WebSocket broadcasts
import sys
from pathlib import Path
backend_path = str(Path(__file__).parent.parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
from abutre_manager import get_abutre_manager


class AbutreBot:
    """Main bot orchestrator"""

    def __init__(self, demo_mode: bool = False, paper_trading: bool = False, ws_port: int = 8000, disable_ws: bool = False, on_market_data=None):
        """
        Initialize Abutre bot

        Args:
            demo_mode: Use demo account
            paper_trading: Monitor only, don't execute
            ws_port: WebSocket server port for dashboard
            disable_ws: Disable WebSocket server (use when managed by FastAPI)
            on_market_data: Callback para broadcast de market data (FastAPI integration)
        """
        self.demo_mode = demo_mode
        self.paper_trading = paper_trading or not config.AUTO_TRADING
        self.disable_ws = disable_ws
        self.on_market_data_callback = on_market_data

        # Components
        self.api_client: DerivAPIClient = None
        self.market_handler: MarketDataHandler = None
        self.strategy: AbutreStrategy = None
        self.risk_manager: RiskManager = None
        self.order_executor: OrderExecutor = None
        self.ws_server: WebSocketServer = None if disable_ws else WebSocketServer(port=ws_port)
        self.async_db: AsyncDatabaseWriter = None

        # State
        self.is_running = False
        self.start_time: datetime = None

        logger.info("="*70)
        logger.info("ABUTRE BOT INITIALIZING")
        logger.info("="*70)
        logger.info(f"  Mode: {'DEMO' if demo_mode else 'LIVE'}")
        logger.info(f"  Paper Trading: {self.paper_trading}")
        logger.info(f"  Auto Trading: {config.AUTO_TRADING}")
        logger.info("="*70)

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing components...")

        # 1. Validate configuration
        if not config.validate():
            logger.critical("Configuration validation FAILED!")
            return False

        config.print_summary()

        # 2. Initialize API client
        self.api_client = DerivAPIClient(
            on_tick=self.on_tick,
            on_balance=self.on_balance
        )

        # 3. Initialize market data handler
        self.market_handler = MarketDataHandler(
            buffer_size=100,
            on_candle_closed=self.on_candle_closed,
            on_streak_detected=self.on_streak_detected
        )

        # 4. Initialize strategy
        self.strategy = AbutreStrategy()

        # 5. Initialize risk manager
        self.risk_manager = RiskManager(initial_balance=config.BANKROLL)

        # 6. Initialize order executor
        self.order_executor = OrderExecutor(self.api_client)

        # 7. Initialize async database writer
        self.async_db = get_async_db_writer(db)
        await self.async_db.start()
        logger.info("✅ AsyncDatabaseWriter iniciado")

        # 8. Start WebSocket server (if enabled)
        if self.ws_server:
            self.ws_server.set_bot_reference(self)
            await self.ws_server.start()

        # 9. Connect to Deriv API
        if not await self.api_client.connect():
            logger.critical("Failed to connect to Deriv API!")
            return False

        # 10. Subscribe to data streams
        await self.api_client.subscribe_ticks(config.SYMBOL)
        await self.api_client.subscribe_balance()

        logger.info("Initialization complete!")
        return True

    # ==================== EVENT HANDLERS ====================

    async def on_tick(self, tick_data: dict):
        """Handle incoming tick"""
        # Process tick through market handler
        await self.market_handler.process_tick(tick_data)

    async def on_balance(self, balance: float):
        """Handle balance update"""
        self.risk_manager.update_balance(balance)

        # Broadcast via AbutreManager (FastAPI WebSocket)
        try:
            manager = get_abutre_manager()
            await manager.broadcast_risk_stats()
        except Exception as e:
            logger.debug(f"Could not broadcast balance update: {e}")

        # Log balance snapshot every 100 candles
        if self.market_handler.candle_count % 100 == 0:
            db.insert_balance_snapshot(
                timestamp=datetime.now(),
                balance=stats['current_balance'],
                peak_balance=stats['peak_balance'],
                drawdown_pct=stats['current_drawdown_pct'],
                total_trades=stats['total_trades'],
                wins=stats['wins'],
                losses=stats['losses']
            )

    async def on_candle_closed(self, candle: Candle):
        """Handle candle close"""
        logger.debug(f"Candle #{self.market_handler.candle_count}: {candle}")

        # Save to database (async, non-blocking)
        await self.async_db.insert_candle(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            color=candle.color,
            ticks_count=len(candle.ticks)
        )

        # Get current streak
        streak_count, streak_direction = self.market_handler.get_current_streak()

        # Broadcast market data via AbutreManager (FastAPI WebSocket)
        try:
            manager = get_abutre_manager()

            # Update manager's market data
            manager.market_data = {
                'symbol': config.SYMBOL,
                'current_price': candle.close,
                'current_streak_count': streak_count,
                'current_streak_direction': streak_direction
            }

            # Broadcast to all clients
            await manager.broadcast_market_data()

        except Exception as e:
            logger.debug(f"Could not broadcast market data: {e}")

        # Analyze with strategy
        signal = self.strategy.analyze_candle(candle, streak_count, streak_direction)

        # Log apenas sinais não-WAIT
        if signal.action != 'WAIT':
            logger.info(f"📊 Strategy signal: {signal}")
        else:
            logger.debug(f"Strategy signal: {signal}")

        # Execute signal
        if signal.action != 'WAIT':
            await self.execute_signal(signal)

    async def on_streak_detected(self, streak_count: int, direction: int):
        """Handle streak trigger detection"""
        direction_str = "GREEN" if direction == 1 else "RED"

        trade_logger.trigger_detected(
            streak_count=streak_count,
            direction=direction_str,
            candle_idx=self.market_handler.candle_count
        )

        # Emit to dashboard
        if self.ws_server:
            await self.ws_server.emit_trigger_detected(streak_count, direction_str)
            await self.ws_server.emit_system_alert('warning', f'Trigger detected: {streak_count} {direction_str} candles')

        # Log to database (async)
        await self.async_db.log_event(
            event_type='TRIGGER',
            severity='WARNING',
            message=f"Streak trigger: {streak_count} {direction_str} candles",
            context=f'{{"streak_count": {streak_count}, "direction": "{direction_str}"}}'
        )

    # ==================== SIGNAL EXECUTION ====================

    async def execute_signal(self, signal):
        """Execute trading signal"""
        # Check if trading allowed
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.error(f"Trading NOT allowed: {reason}")
            db.log_event(
                event_type='TRADING_BLOCKED',
                severity='ERROR',
                message=f"Trading blocked: {reason}"
            )
            return

        # Handle different signal types
        if signal.action == 'ENTER':
            await self.handle_entry(signal)

        elif signal.action == 'LEVEL_UP':
            await self.handle_level_up(signal)

        elif signal.action == 'CLOSE':
            await self.handle_close(signal)

    async def handle_entry(self, signal):
        """Handle trade entry"""
        logger.info(f"Executing ENTRY signal: {signal}")

        # Validate stake
        try:
            self.risk_manager.validate_balance(signal.stake)
            self.risk_manager.validate_level(signal.level)
        except RiskViolation as e:
            logger.error(f"Risk violation: {e}")
            if self.ws_server:
                await self.ws_server.emit_system_alert('error', f'Risk violation: {e}')
            return

        # Place order
        order = await self.order_executor.place_order(
            direction=signal.direction,
            stake=signal.stake,
            level=signal.level,
            dry_run=self.paper_trading
        )

        # Update strategy state
        self.strategy.execute_signal(signal)

        # Emit to dashboard
        if self.ws_server:
            trade_data = {
                'trade_id': order.order_id,
                'entry_time': datetime.now().isoformat(),
                'direction': signal.direction,
                'level': signal.level,
                'stake': signal.stake,
                'entry_streak_size': self.market_handler.current_streak_count
            }
            await self.ws_server.emit_trade_opened(trade_data)
            await self.ws_server.emit_position_update({
                'in_position': True,
                'direction': signal.direction,
                'entry_timestamp': datetime.now().isoformat(),
                'entry_streak_size': self.market_handler.current_streak_count,
                'current_level': signal.level,
                'current_stake': signal.stake,
                'total_loss': 0,
                'next_stake': signal.stake * self.strategy.multiplier
            })

        # Save to database
        db.insert_trade(
            trade_id=order.order_id,
            entry_time=datetime.now(),
            entry_candle_idx=self.market_handler.candle_count,
            entry_streak_size=self.market_handler.current_streak_count,
            direction=signal.direction,
            initial_stake=signal.stake,
            balance_before=self.risk_manager.current_balance
        )

    async def handle_level_up(self, signal):
        """Handle Martingale level up"""
        logger.warning(f"Executing LEVEL UP signal: {signal}")

        # Validate
        try:
            self.risk_manager.validate_balance(signal.stake)
            self.risk_manager.validate_level(signal.level)
        except RiskViolation as e:
            logger.error(f"Risk violation: {e}")
            # Force close position
            signal.action = 'CLOSE'
            signal.stake = -self.strategy.position.total_loss
            await self.handle_close(signal)
            return

        # Place order
        order = await self.order_executor.place_order(
            direction=signal.direction,
            stake=signal.stake,
            level=signal.level,
            dry_run=self.paper_trading
        )

        # Update strategy state
        self.strategy.execute_signal(signal)

    async def handle_close(self, signal):
        """Handle position close"""
        result = "WIN" if signal.stake > 0 else "LOSS"
        logger.info(f"Executing CLOSE signal: {signal} ({result})")

        # Update strategy state
        self.strategy.execute_signal(signal)

        # Record trade result
        self.risk_manager.record_trade(profit=signal.stake, result=result)

        # Emit to dashboard
        if self.ws_server:
            trade_data = {
                'trade_id': str(self.strategy.position.entry_candle_idx),
                'exit_time': datetime.now().isoformat(),
                'result': result,
                'profit': signal.stake,
                'final_level': signal.level,
                'balance': self.risk_manager.current_balance
            }
            await self.ws_server.emit_trade_closed(trade_data)
            await self.ws_server.emit_position_update({
                'in_position': False,
                'direction': None,
                'entry_timestamp': None,
                'entry_streak_size': 0,
                'current_level': 0,
                'current_stake': 0,
                'total_loss': 0,
                'next_stake': 0
            })

            # Emit risk stats
            stats = self.risk_manager.get_stats()
            await self.ws_server.emit_risk_stats({
                'total_trades': stats['total_trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate_pct'],
                'roi': stats['roi_pct']
            })

            alert_level = 'success' if result == 'WIN' else 'error'
            await self.ws_server.emit_system_alert(alert_level, f"Trade closed: {result} | P&L: ${signal.stake:+.2f}")

        # Update database
        db.update_trade(
            trade_id=self.strategy.position.entry_candle_idx,
            exit_time=datetime.now(),
            result=result,
            profit=signal.stake,
            balance_after=self.risk_manager.current_balance,
            max_level_reached=signal.level
        )

        # Log event
        db.log_event(
            event_type='TRADE_CLOSE',
            severity='INFO' if result == 'WIN' else 'WARNING',
            message=f"Trade closed: {result} | P&L: ${signal.stake:+.2f}",
            context=f'{{"result": "{result}", "profit": {signal.stake}, "level": {signal.level}}}'
        )

    # ==================== MAIN LOOP ====================

    async def run(self):
        """Main event loop"""
        self.is_running = True
        self.start_time = datetime.now()

        logger.info("="*70)
        logger.info("ABUTRE BOT STARTED")
        logger.info(f"Start Time: {self.start_time}")
        logger.info("="*70)

        # Start listening to WebSocket
        await self.api_client.listen()

    async def stop(self):
        """Stop bot (called by AbutreManager)"""
        await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")

        self.is_running = False

        # Stop async database writer (flush pending operations)
        if self.async_db:
            await self.async_db.stop()
            logger.info("✅ AsyncDatabaseWriter parado")

        # Emit shutdown status and stop WebSocket server
        if self.ws_server:
            await self.ws_server.emit_bot_status('STOPPED', 'Bot shutdown complete')
            await self.ws_server.stop()

        # Disconnect API
        if self.api_client:
            await self.api_client.disconnect()

        # Print final stats
        if self.risk_manager:
            stats = self.risk_manager.get_stats()
            logger.info("="*70)
            logger.info("FINAL STATISTICS")
            logger.info("="*70)
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            logger.info("="*70)

        logger.info("Shutdown complete.")


# ==================== CLI ====================

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Abutre Bot - Delayed Martingale Strategy')
    parser.add_argument('--demo', action='store_true', help='Use demo account')
    parser.add_argument('--paper-trading', action='store_true', help='Monitor only, no execution')

    args = parser.parse_args()

    # Create bot
    bot = AbutreBot(
        demo_mode=args.demo,
        paper_trading=args.paper_trading
    )

    # Initialize
    if not await bot.initialize():
        logger.critical("Initialization failed!")
        return

    # Run
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

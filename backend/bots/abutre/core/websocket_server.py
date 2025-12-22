"""
WebSocket Server for Abutre Dashboard

Exposes real-time events to frontend using Socket.IO
"""
import asyncio
from typing import Optional, Callable
from datetime import datetime
import socketio
from aiohttp import web

from ..utils.logger import default_logger as logger


class WebSocketServer:
    """Socket.IO server for real-time dashboard updates"""

    def __init__(self, port: int = 8000):
        """
        Initialize WebSocket server

        Args:
            port: Server port (default: 8000)
        """
        self.port = port
        self.sio = socketio.AsyncServer(
            async_mode='aiohttp',
            cors_allowed_origins='*',  # Allow all origins (dev mode)
            logger=False,
            engineio_logger=False
        )
        self.app = web.Application()
        self.sio.attach(self.app)
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Bot reference (set via set_bot_reference)
        self.bot: Optional['AbutreBot'] = None

        # Setup Socket.IO event handlers
        self._setup_handlers()

        logger.info(f"WebSocket server initialized on port {port}")

    def _setup_handlers(self):
        """Setup Socket.IO event handlers"""

        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"[WS] Client connected: {sid}")
            await self.sio.emit('connection_ack', {'status': 'connected'}, to=sid)

            # Send initial state
            if self.bot:
                await self._send_initial_state(sid)

        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"[WS] Client disconnected: {sid}")

        @self.sio.event
        async def bot_command(sid, data):
            """
            Handle bot control commands from frontend

            Commands:
                - start: Start trading
                - pause: Pause trading
                - stop: Stop trading
            """
            command = data.get('command')
            logger.info(f"[WS] Received command: {command} from {sid}")

            if not self.bot:
                await self.sio.emit('error', {'message': 'Bot not initialized'}, to=sid)
                return

            # Execute command
            if command == 'start':
                self.bot.paper_trading = False
                await self.emit_bot_status('RUNNING', 'Bot started')

            elif command == 'pause':
                self.bot.paper_trading = True
                await self.emit_bot_status('PAUSED', 'Bot paused (paper trading)')

            elif command == 'stop':
                await self.emit_bot_status('STOPPING', 'Bot shutdown initiated')
                asyncio.create_task(self.bot.shutdown())

            else:
                await self.sio.emit('error', {'message': f'Unknown command: {command}'}, to=sid)

        @self.sio.event
        async def update_settings(sid, data):
            """
            Handle settings update from frontend

            Settings:
                - delay_threshold: int (6-12)
                - max_level: int (8-12)
                - initial_stake: float ($0.50-$5.00)
                - multiplier: float (1.5x-3.0x)
                - max_drawdown: float (15%-35%)
                - auto_trading: bool
            """
            logger.info(f"[WS] Settings update from {sid}: {data}")

            if not self.bot:
                await self.sio.emit('error', {'message': 'Bot not initialized'}, to=sid)
                return

            # Update strategy params
            if self.bot.strategy:
                if 'delay_threshold' in data:
                    self.bot.strategy.delay_threshold = int(data['delay_threshold'])
                if 'max_level' in data:
                    self.bot.strategy.max_level = int(data['max_level'])
                if 'initial_stake' in data:
                    self.bot.strategy.initial_stake = float(data['initial_stake'])
                if 'multiplier' in data:
                    self.bot.strategy.multiplier = float(data['multiplier'])

            # Update risk manager params
            if self.bot.risk_manager:
                if 'max_drawdown' in data:
                    max_dd_pct = float(data['max_drawdown'])
                    self.bot.risk_manager.max_drawdown_pct = max_dd_pct

            # Update auto trading
            if 'auto_trading' in data:
                self.bot.paper_trading = not bool(data['auto_trading'])

            # Acknowledge
            await self.sio.emit('settings_updated', {'success': True}, to=sid)
            await self.emit_system_alert('success', 'Settings updated successfully')

    async def _send_initial_state(self, sid: str):
        """Send initial bot state to newly connected client"""
        if not self.bot:
            return

        # Bot status
        status = 'RUNNING' if self.bot.is_running else 'STOPPED'
        if self.bot.paper_trading:
            status = 'PAUSED'

        await self.sio.emit('bot_status', {
            'status': status,
            'message': f'Bot is {status.lower()}'
        }, to=sid)

        # Balance
        if self.bot.risk_manager:
            stats = self.bot.risk_manager.get_stats()
            await self.sio.emit('balance_update', {
                'balance': stats['current_balance'],
                'peak': stats['peak_balance'],
                'drawdown_pct': stats['current_drawdown_pct']
            }, to=sid)

            # Risk stats
            await self.sio.emit('risk_stats', {
                'total_trades': stats['total_trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate_pct'],
                'roi': stats['roi_pct'],
                'daily_loss': stats.get('daily_loss_pct', 0)
            }, to=sid)

        # Current position
        if self.bot.strategy and self.bot.strategy.in_position:
            pos = self.bot.strategy.position
            await self.sio.emit('position_update', {
                'in_position': True,
                'direction': pos.direction,
                'entry_timestamp': pos.entry_timestamp.isoformat(),
                'entry_streak_size': pos.entry_streak_size,
                'current_level': pos.current_level,
                'current_stake': pos.current_stake,
                'total_loss': pos.total_loss,
                'next_stake': pos.current_stake * self.bot.strategy.multiplier
            }, to=sid)

        # Market data
        if self.bot.market_handler:
            streak_count, streak_dir = self.bot.market_handler.get_current_streak()
            last_candle = self.bot.market_handler.get_last_candle()

            await self.sio.emit('market_data', {
                'symbol': 'V100',
                'price': last_candle.close if last_candle else 0,
                'streak_count': streak_count,
                'streak_direction': 'GREEN' if streak_dir == 1 else 'RED'
            }, to=sid)

    # ==================== PUBLIC EMIT METHODS ====================

    def set_bot_reference(self, bot: 'AbutreBot'):
        """Set reference to main bot instance"""
        self.bot = bot
        logger.info("Bot reference set on WebSocket server")

    async def emit_balance_update(self, balance: float, peak: float, drawdown_pct: float):
        """Emit balance update to all clients"""
        await self.sio.emit('balance_update', {
            'balance': balance,
            'peak': peak,
            'drawdown_pct': drawdown_pct,
            'timestamp': datetime.now().isoformat()
        })

    async def emit_new_candle(self, candle_data: dict):
        """Emit new candle event"""
        await self.sio.emit('new_candle', candle_data)

    async def emit_trigger_detected(self, streak_count: int, direction: str):
        """Emit streak trigger detection"""
        await self.sio.emit('trigger_detected', {
            'streak_count': streak_count,
            'direction': direction,
            'timestamp': datetime.now().isoformat()
        })

    async def emit_trade_opened(self, trade_data: dict):
        """Emit trade opened event"""
        await self.sio.emit('trade_opened', trade_data)

    async def emit_trade_closed(self, trade_data: dict):
        """Emit trade closed event"""
        await self.sio.emit('trade_closed', trade_data)

    async def emit_position_update(self, position_data: dict):
        """Emit position state update"""
        await self.sio.emit('position_update', position_data)

    async def emit_bot_status(self, status: str, message: str):
        """Emit bot status change"""
        await self.sio.emit('bot_status', {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    async def emit_system_alert(self, level: str, message: str):
        """
        Emit system alert

        Args:
            level: 'success', 'warning', 'error', 'info'
            message: Alert message
        """
        await self.sio.emit('system_alert', {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    async def emit_market_data(self, symbol: str, price: float, streak_count: int, streak_direction: str):
        """Emit market data update"""
        await self.sio.emit('market_data', {
            'symbol': symbol,
            'price': price,
            'streak_count': streak_count,
            'streak_direction': streak_direction,
            'timestamp': datetime.now().isoformat()
        })

    async def emit_risk_stats(self, stats: dict):
        """Emit risk statistics"""
        await self.sio.emit('risk_stats', stats)

    # ==================== SERVER LIFECYCLE ====================

    async def start(self):
        """Start the WebSocket server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        logger.info(f"[WS] Server started on http://localhost:{self.port}")

    async def stop(self):
        """Stop the WebSocket server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("[WS] Server stopped")

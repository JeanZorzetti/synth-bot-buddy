"""
ABUTRE BOT MANAGER

Gerencia o ciclo de vida do bot Abutre integrado ao FastAPI.
Permite iniciar/parar o bot via API e enviar dados para o dashboard.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Set
from datetime import datetime
import sys
from pathlib import Path
from fastapi import WebSocket

# Adicionar backend ao path para imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gerencia conexões WebSocket para broadcast de dados"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Adiciona nova conexão"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove conexão"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket desconectado. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Envia mensagem para todos os clientes conectados"""
        dead_connections = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Erro ao enviar para WebSocket: {e}")
                dead_connections.add(connection)

        # Remove conexões mortas
        for conn in dead_connections:
            self.active_connections.discard(conn)


# Singleton do WebSocket manager
_ws_manager: Optional[WebSocketManager] = None

def get_ws_manager() -> WebSocketManager:
    """Retorna instância singleton do WebSocket manager"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


class AbutreManager:
    """Singleton manager para o bot Abutre"""

    _instance: Optional['AbutreManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.bot = None
        self.bot_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.demo_mode = True
        self.paper_trading = True

        # Estatísticas
        self.stats = {
            'total_trades': 0,
            'current_balance': 2000.0,
            'roi_pct': 0.0,
            'win_rate_pct': 0.0,
            'max_drawdown_pct': 0.0,
        }

        logger.info("✅ AbutreManager initialized")

    async def start_bot(self, demo: bool = True, paper_trading: bool = True) -> Dict[str, Any]:
        """
        Inicia o bot Abutre

        Args:
            demo: Usar conta DEMO
            paper_trading: Apenas observar (não executar trades)

        Returns:
            Status da operação
        """
        if self.is_running:
            return {
                'status': 'error',
                'message': 'Bot já está rodando',
                'is_running': True,
            }

        try:
            logger.info("🚀 Iniciando Abutre Bot...")

            # Importar bot aqui para evitar problemas de import circular
            from bots.abutre.main import AbutreBot

            self.demo_mode = demo
            self.paper_trading = paper_trading
            self.start_time = datetime.now()

            # Criar instância do bot
            # Desabilitar WebSocket interno pois FastAPI já tem /ws/dashboard
            self.bot = AbutreBot(
                demo_mode=demo,
                paper_trading=paper_trading,
                disable_ws=True,  # Evita conflito de porta com FastAPI
                on_market_data=self._on_market_data_callback  # Hook para broadcast
            )

            # Iniciar bot em background
            self.bot_task = asyncio.create_task(self._run_bot())
            self.is_running = True

            logger.info("✅ Abutre Bot iniciado com sucesso")

            return {
                'status': 'success',
                'message': 'Bot iniciado com sucesso',
                'is_running': True,
                'start_time': self.start_time.isoformat(),
                'config': {
                    'demo_mode': demo,
                    'paper_trading': paper_trading,
                    'symbol': '1HZ100V',  # V100
                    'delay_threshold': 8,
                    'max_level': 10,
                    'initial_stake': 1.0,
                }
            }

        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.is_running = False
            return {
                'status': 'error',
                'message': f'Erro ao iniciar bot: {str(e)}',
                'is_running': False,
            }

    async def stop_bot(self) -> Dict[str, Any]:
        """
        Para o bot Abutre

        Returns:
            Status da operação
        """
        if not self.is_running:
            return {
                'status': 'error',
                'message': 'Bot não está rodando',
                'is_running': False,
            }

        try:
            logger.info("⏹️ Parando Abutre Bot...")

            # Parar bot
            if self.bot:
                await self.bot.stop()

            # Cancelar task
            if self.bot_task:
                self.bot_task.cancel()
                try:
                    await self.bot_task
                except asyncio.CancelledError:
                    pass

            # Calcular duração
            duration = None
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()

            self.is_running = False
            self.bot = None
            self.bot_task = None

            logger.info("✅ Abutre Bot parado com sucesso")

            return {
                'status': 'success',
                'message': 'Bot parado com sucesso',
                'is_running': False,
                'duration_seconds': duration,
                'final_stats': self.stats.copy(),
            }

        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return {
                'status': 'error',
                'message': f'Erro ao parar bot: {str(e)}',
                'is_running': self.is_running,
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do bot

        Returns:
            Dicionário com status completo
        """
        duration_seconds = 0
        if self.is_running and self.start_time:
            duration_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_seconds': duration_seconds,
            'demo_mode': self.demo_mode,
            'paper_trading': self.paper_trading,
            'config': {
                'symbol': '1HZ100V',  # V100
                'delay_threshold': 8,
                'max_level': 10,
                'initial_stake': 1.0,
                'bankroll': 2000.0,
            },
            'stats': self.stats.copy(),
        }

    def update_stats(self, stats: Dict[str, Any]):
        """
        Atualiza estatísticas do bot

        Args:
            stats: Novas estatísticas
        """
        self.stats.update(stats)

    async def broadcast_market_data(self, symbol: str, price: float, streak_count: int, streak_direction: str):
        """
        Envia dados de mercado para todos os clientes WebSocket conectados

        Args:
            symbol: Símbolo do ativo (ex: 1HZ100V)
            price: Preço atual
            streak_count: Contador de streak
            streak_direction: Direção do streak ('GREEN' ou 'RED')
        """
        ws_manager = get_ws_manager()

        # Converter direção para número: GREEN=1, RED=-1
        direction_num = 1 if streak_direction == 'GREEN' else -1

        message = {
            'event': 'market_data',
            'data': {
                'symbol': symbol,
                'current_price': price,
                'current_streak_count': streak_count,
                'current_streak_direction': direction_num,
            }
        }

        await ws_manager.broadcast(message)
        logger.debug(f"📊 Market data broadcast: {symbol} @ {price} | Streak: {streak_count} {streak_direction}")

    async def _on_market_data_callback(self, symbol: str, price: float, streak_count: int, streak_direction: str):
        """
        Callback chamado pelo bot quando há novos dados de mercado

        Args:
            symbol: Símbolo do ativo
            price: Preço atual
            streak_count: Contador de streak
            streak_direction: Direção ('GREEN' ou 'RED')
        """
        await self.broadcast_market_data(symbol, price, streak_count, streak_direction)

    async def _run_bot(self):
        """Task interna que roda o bot"""
        try:
            logger.info("🤖 Bot task iniciada")

            # Inicializar componentes
            if not await self.bot.initialize():
                logger.error("❌ Falha na inicialização do bot")
                self.is_running = False
                return

            # Rodar bot
            await self.bot.run()

        except asyncio.CancelledError:
            logger.info("⏹️ Bot task cancelada")
            raise
        except Exception as e:
            logger.error(f"❌ Erro no bot task: {e}")
            self.is_running = False
        finally:
            logger.info("🏁 Bot task finalizada")


# Singleton global
_abutre_manager: Optional[AbutreManager] = None


def get_abutre_manager() -> AbutreManager:
    """Retorna instância global do AbutreManager"""
    global _abutre_manager
    if _abutre_manager is None:
        _abutre_manager = AbutreManager()
    return _abutre_manager


def initialize_abutre_manager() -> AbutreManager:
    """Inicializa o manager (chamado no startup do FastAPI)"""
    logger.info("🔧 Inicializando AbutreManager...")
    manager = get_abutre_manager()
    logger.info("✅ AbutreManager pronto")
    return manager

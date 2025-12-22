"""
ASYNC DATABASE WRITER

Evita bloqueio do event loop fazendo writes em background com queue.
Flush automático a cada N operações ou intervalo de tempo.
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque

from .database import DatabaseManager
from ..utils.logger import default_logger as logger


class AsyncDatabaseWriter:
    """
    Wrapper assíncrono para DatabaseManager que usa queue para batch writes
    """

    def __init__(self, db_manager: DatabaseManager, flush_interval: float = 5.0, flush_size: int = 50):
        """
        Initialize async database writer

        Args:
            db_manager: DatabaseManager instance
            flush_interval: Flush queue a cada N segundos
            flush_size: Flush queue quando atingir N operações
        """
        self.db_manager = db_manager
        self.flush_interval = flush_interval
        self.flush_size = flush_size

        # Queue de operações pendentes
        self.operations: deque = deque()

        # Background task
        self.flush_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start(self):
        """Inicia background task de flush"""
        if self.is_running:
            return

        self.is_running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"AsyncDatabaseWriter started (flush_interval={self.flush_interval}s, flush_size={self.flush_size})")

    async def stop(self):
        """Para background task e faz flush final"""
        if not self.is_running:
            return

        self.is_running = False

        # Cancelar task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass

        # Flush final
        await self._flush()
        logger.info("AsyncDatabaseWriter stopped")

    async def insert_candle(self, timestamp: datetime, open: float, high: float, low: float, close: float, color: int, ticks_count: int):
        """
        Adiciona operação de insert_candle à queue

        Args:
            timestamp: Timestamp do candle
            open, high, low, close: OHLC
            color: 1 (green), -1 (red), 0 (doji)
            ticks_count: Número de ticks
        """
        operation = {
            'type': 'insert_candle',
            'args': {
                'timestamp': timestamp,
                'open': open,
                'high': high,
                'low': low,
                'close': close,
                'color': color,
                'ticks_count': ticks_count
            }
        }

        self.operations.append(operation)

        # Flush se atingiu tamanho máximo
        if len(self.operations) >= self.flush_size:
            await self._flush()

    async def log_event(self, event_type: str, severity: str, message: str, context: str = None):
        """
        Adiciona operação de log_event à queue

        Args:
            event_type: Tipo do evento (TRIGGER, TRADE, ERROR, etc)
            severity: INFO, WARNING, ERROR, CRITICAL
            message: Mensagem descritiva
            context: JSON string com dados adicionais
        """
        operation = {
            'type': 'log_event',
            'args': {
                'event_type': event_type,
                'severity': severity,
                'message': message,
                'context': context
            }
        }

        self.operations.append(operation)

        # Eventos críticos fazem flush imediato
        if severity in ('ERROR', 'CRITICAL'):
            await self._flush()

    async def insert_balance(self, timestamp: datetime, balance: float, peak_balance: float, drawdown_pct: float, total_trades: int, wins: int, losses: int):
        """Adiciona operação de insert_balance à queue"""
        operation = {
            'type': 'insert_balance',
            'args': {
                'timestamp': timestamp,
                'balance': balance,
                'peak_balance': peak_balance,
                'drawdown_pct': drawdown_pct,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses
            }
        }

        self.operations.append(operation)

        if len(self.operations) >= self.flush_size:
            await self._flush()

    async def _flush_loop(self):
        """Loop que faz flush periódico"""
        try:
            while self.is_running:
                await asyncio.sleep(self.flush_interval)
                if self.operations:
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self):
        """
        Executa todas as operações pendentes na queue

        Roda em executor para não bloquear event loop
        """
        if not self.operations:
            return

        # Copiar queue e limpar
        operations_to_flush = list(self.operations)
        self.operations.clear()

        # Executar em thread pool para não bloquear event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._execute_operations, operations_to_flush)

        logger.debug(f"Flushed {len(operations_to_flush)} database operations")

    def _execute_operations(self, operations: list):
        """
        Executa operações de forma síncrona (roda em executor)

        Args:
            operations: Lista de operações para executar
        """
        for op in operations:
            try:
                op_type = op['type']
                args = op['args']

                if op_type == 'insert_candle':
                    self.db_manager.insert_candle(**args)
                elif op_type == 'log_event':
                    self.db_manager.log_event(**args)
                elif op_type == 'insert_balance':
                    self.db_manager.insert_balance(**args)
                else:
                    logger.error(f"Unknown operation type: {op_type}")

            except Exception as e:
                logger.error(f"Error executing database operation: {e}")


# Singleton instance
_async_db_writer: Optional[AsyncDatabaseWriter] = None


def get_async_db_writer(db_manager: DatabaseManager = None) -> AsyncDatabaseWriter:
    """
    Retorna instância singleton do AsyncDatabaseWriter

    Args:
        db_manager: DatabaseManager instance (required on first call)

    Returns:
        AsyncDatabaseWriter instance
    """
    global _async_db_writer

    if _async_db_writer is None:
        if db_manager is None:
            raise ValueError("db_manager required on first call to get_async_db_writer()")
        _async_db_writer = AsyncDatabaseWriter(db_manager)

    return _async_db_writer

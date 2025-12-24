"""
ABUTRE EVENTS API

Endpoints para receber eventos do Deriv Bot XML
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from database.abutre_repository_async import get_async_repository
from api.schemas.abutre_events import (
    CandleEvent,
    TriggerEvent,
    TradeOpenedEvent,
    TradeClosedEvent,
    BalanceEvent,
    EventResponse
)

# Para broadcast via WebSocket
import sys
from pathlib import Path
backend_path = str(Path(__file__).parent.parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from abutre_manager import get_abutre_manager, get_ws_manager
except ImportError:
    # Fallback se não conseguir importar
    def get_abutre_manager():
        return None
    def get_ws_manager():
        return None


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/abutre/events", tags=["Abutre Events"])


# ==================== CANDLE ====================

@router.post("/candle", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def post_candle_event(event: CandleEvent):
    """
    Receive candle closed event from Deriv Bot XML

    Args:
        event: Candle data (timestamp, OHLC, color)

    Returns:
        EventResponse with event_id
    """
    try:
        repo = await get_async_repository()

        # Insert into database
        candle_id = await repo.insert_candle(
            timestamp=event.timestamp,
            symbol=event.symbol,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.close,
            color=event.color
        )

        # Broadcast via WebSocket
        try:
            ws_manager = get_ws_manager()
            if ws_manager:
                await ws_manager.broadcast({
                    'event': 'new_candle',
                    'data': {
                        'timestamp': event.timestamp.isoformat(),
                        'symbol': event.symbol,
                        'open': event.open,
                        'high': event.high,
                        'low': event.low,
                        'close': event.close,
                        'color': event.color
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast candle: {e}")

        return EventResponse(
            status="success",
            message=f"Candle event recorded successfully",
            event_id=candle_id
        )

    except Exception as e:
        logger.error(f"Error processing candle event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process candle event: {str(e)}"
        )


# ==================== TRIGGER ====================

@router.post("/trigger", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def post_trigger_event(event: TriggerEvent):
    """
    Receive trigger detected event (streak >= 8)

    Args:
        event: Trigger data (streak_count, direction)

    Returns:
        EventResponse with event_id
    """
    try:
        repo = await get_async_repository()

        # Insert into database
        trigger_id = await repo.insert_trigger(
            timestamp=event.timestamp,
            streak_count=event.streak_count,
            direction=event.direction
        )

        # Broadcast via WebSocket
        try:
            ws_manager = get_ws_manager()
            if ws_manager:
                await ws_manager.broadcast({
                    'event': 'trigger_detected',
                    'data': {
                        'timestamp': event.timestamp.isoformat(),
                        'streak_count': event.streak_count,
                        'direction': event.direction
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast trigger: {e}")

        return EventResponse(
            status="success",
            message=f"Trigger event recorded successfully",
            event_id=trigger_id
        )

    except Exception as e:
        logger.error(f"Error processing trigger event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process trigger event: {str(e)}"
        )


# ==================== TRADE OPENED ====================

@router.post("/trade_opened", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def post_trade_opened_event(event: TradeOpenedEvent):
    """
    Receive trade opened event

    Args:
        event: Trade opened data (trade_id, direction, stake, level)

    Returns:
        EventResponse with event_id
    """
    try:
        repo = await get_async_repository()

        # Insert into database
        trade_id = await repo.insert_trade_opened(
            trade_id=event.trade_id,
            timestamp=event.timestamp,
            direction=event.direction,
            stake=event.stake,
            level=event.level,
            contract_id=event.contract_id
        )

        # Broadcast via WebSocket
        try:
            ws_manager = get_ws_manager()
            if ws_manager:
                await ws_manager.broadcast({
                    'event': 'trade_opened',
                    'data': {
                        'timestamp': event.timestamp.isoformat(),
                        'trade_id': event.trade_id,
                        'direction': event.direction,
                        'stake': event.stake,
                        'level': event.level,
                        'contract_id': event.contract_id
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast trade_opened: {e}")

        return EventResponse(
            status="success",
            message=f"Trade opened event recorded successfully",
            event_id=trade_id
        )

    except Exception as e:
        logger.error(f"Error processing trade_opened event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process trade_opened event: {str(e)}"
        )


# ==================== TRADE CLOSED ====================

@router.post("/trade_closed", response_model=EventResponse, status_code=status.HTTP_200_OK)
async def post_trade_closed_event(event: TradeClosedEvent):
    """
    Receive trade closed event

    Args:
        event: Trade closed data (trade_id, result, profit, balance)

    Returns:
        EventResponse
    """
    try:
        repo = await get_async_repository()

        # Update trade in database
        success = await repo.update_trade_closed(
            trade_id=event.trade_id,
            exit_time=event.timestamp,
            result=event.result,
            profit=event.profit,
            balance=event.balance,
            max_level=event.max_level_reached
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trade {event.trade_id} not found"
            )

        # Calculate and store balance snapshot
        stats = await repo.get_trade_stats()

        # Get latest balance or use initial
        latest_balance = await repo.get_latest_balance() or 10000.0
        peak_balance = max(latest_balance, event.balance)
        drawdown_pct = ((peak_balance - event.balance) / peak_balance * 100) if peak_balance > 0 else 0.0
        roi_pct = ((event.balance - 10000.0) / 10000.0 * 100)

        await repo.insert_balance_snapshot(
            timestamp=event.timestamp,
            balance=event.balance,
            peak_balance=peak_balance,
            drawdown_pct=drawdown_pct,
            total_trades=stats['total_trades'],
            wins=stats['wins'],
            losses=stats['losses'],
            roi_pct=roi_pct
        )

        # Broadcast via WebSocket
        try:
            ws_manager = get_ws_manager()
            if ws_manager:
                # Broadcast trade closed
                await ws_manager.broadcast({
                    'event': 'trade_closed',
                    'data': {
                        'timestamp': event.timestamp.isoformat(),
                        'trade_id': event.trade_id,
                        'result': event.result,
                        'profit': event.profit,
                        'balance': event.balance,
                        'max_level_reached': event.max_level_reached
                    }
                })

                # Broadcast updated risk stats
                await ws_manager.broadcast({
                    'event': 'risk_stats',
                    'data': {
                        'current_balance': event.balance,
                        'peak_balance': peak_balance,
                        'current_drawdown_pct': drawdown_pct,
                        'roi_pct': roi_pct,
                        'total_trades': stats['total_trades'],
                        'wins': stats['wins'],
                        'losses': stats['losses'],
                        'win_rate_pct': stats['win_rate_pct']
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast trade_closed: {e}")

        return EventResponse(
            status="success",
            message=f"Trade closed event recorded successfully",
            event_id=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing trade_closed event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process trade_closed event: {str(e)}"
        )


# ==================== BALANCE ====================

@router.post("/balance", response_model=EventResponse, status_code=status.HTTP_200_OK)
async def post_balance_event(event: BalanceEvent):
    """
    Receive balance update event

    Args:
        event: Balance data (timestamp, balance)

    Returns:
        EventResponse
    """
    try:
        repo = await get_async_repository()

        # Get current stats
        stats = await repo.get_trade_stats()

        # Calculate metrics
        latest_balance = await repo.get_latest_balance() or 10000.0
        peak_balance = max(latest_balance, event.balance)
        drawdown_pct = ((peak_balance - event.balance) / peak_balance * 100) if peak_balance > 0 else 0.0
        roi_pct = ((event.balance - 10000.0) / 10000.0 * 100)

        # Insert balance snapshot
        snapshot_id = await repo.insert_balance_snapshot(
            timestamp=event.timestamp,
            balance=event.balance,
            peak_balance=peak_balance,
            drawdown_pct=drawdown_pct,
            total_trades=stats['total_trades'],
            wins=stats['wins'],
            losses=stats['losses'],
            roi_pct=roi_pct
        )

        # Broadcast via WebSocket
        try:
            ws_manager = get_ws_manager()
            if ws_manager:
                await ws_manager.broadcast({
                    'event': 'balance_update',
                    'data': {
                        'timestamp': event.timestamp.isoformat(),
                        'balance': event.balance
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast balance: {e}")

        return EventResponse(
            status="success",
            message=f"Balance event recorded successfully",
            event_id=snapshot_id
        )

    except Exception as e:
        logger.error(f"Error processing balance event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process balance event: {str(e)}"
        )


# ==================== GET ENDPOINTS (para o dashboard) ====================

@router.get("/stats")
async def get_stats():
    """Get aggregated statistics"""
    try:
        repo = await get_async_repository()
        stats = await repo.get_trade_stats()
        latest_balance = await repo.get_latest_balance() or 10000.0

        return {
            "status": "success",
            "data": {
                **stats,
                "current_balance": latest_balance,
                "roi_pct": ((latest_balance - 10000.0) / 10000.0 * 100)
            }
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    try:
        repo = await get_async_repository()
        trades = await repo.get_recent_trades(limit=limit)

        return {
            "status": "success",
            "data": trades
        }

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trades: {str(e)}"
        )


@router.get("/balance_history")
async def get_balance_history(limit: int = 1000):
    """Get balance history for equity curve"""
    try:
        repo = await get_async_repository()
        history = await repo.get_balance_history(limit=limit)

        return {
            "status": "success",
            "data": history
        }

    except Exception as e:
        logger.error(f"Error getting balance history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get balance history: {str(e)}"
        )

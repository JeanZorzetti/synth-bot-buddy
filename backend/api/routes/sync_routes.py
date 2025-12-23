"""
SYNC ROUTES - Endpoints para sincronização de histórico por período

Permite buscar e sincronizar trades da Deriv API por período específico
"""
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import logging
import asyncio

from database import get_abutre_repository

router = APIRouter(prefix="/api/abutre/sync", tags=["Sync"])
logger = logging.getLogger(__name__)


class SyncPeriodRequest(BaseModel):
    """Request para sincronização de período"""
    date_from: datetime = Field(..., description="Data inicial (ISO 8601)")
    date_to: datetime = Field(..., description="Data final (ISO 8601)")
    force: bool = Field(default=False, description="Forçar re-sincronização mesmo se trades já existem")


class SyncResponse(BaseModel):
    """Response de sincronização"""
    status: str
    message: str
    trades_synced: int
    trades_failed: int
    period_from: str
    period_to: str


@router.get("/trades", response_model=dict, status_code=status.HTTP_200_OK)
async def get_trades_by_period(
    date_from: Optional[str] = Query(None, description="Data inicial (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Data final (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Limite de trades")
):
    """
    Busca trades do banco de dados por período

    Se date_from e date_to não forem fornecidos, retorna os últimos `limit` trades
    """
    try:
        repo = get_abutre_repository()

        if date_from and date_to:
            # Parse dates
            try:
                dt_from = datetime.fromisoformat(date_from + "T00:00:00")
                dt_to = datetime.fromisoformat(date_to + "T23:59:59")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de data inválido. Use YYYY-MM-DD"
                )

            if dt_from > dt_to:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="date_from deve ser anterior a date_to"
                )

            # Buscar trades por período
            trades = repo.get_trades_by_period(dt_from, dt_to, limit)
        else:
            # Buscar trades recentes
            trades = repo.get_recent_trades(limit=limit)

        return {
            "status": "success",
            "data": {
                "trades": trades,
                "count": len(trades),
                "period_from": date_from,
                "period_to": date_to,
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"Error getting trades by period: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trades: {str(e)}"
        )


@router.post("/trigger", response_model=SyncResponse, status_code=status.HTTP_200_OK)
async def trigger_sync_period(request: SyncPeriodRequest):
    """
    Força sincronização de trades da Deriv para um período específico

    **NOTA**: A API da Deriv (profit_table) não suporta filtro por data diretamente.
    Este endpoint busca os últimos trades e filtra no backend.

    Para períodos muito antigos, pode ser necessário fazer múltiplas chamadas.
    """
    try:
        # Importar função de sync (evitar import circular)
        from auto_sync_deriv import sync_deriv_history_period

        # Validar período
        if request.date_from > request.date_to:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="date_from deve ser anterior a date_to"
            )

        # Verificar se período não é muito grande (máximo 90 dias)
        period_days = (request.date_to - request.date_from).days
        if period_days > 90:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Período máximo permitido: 90 dias"
            )

        logger.info(f"Triggering sync for period: {request.date_from} to {request.date_to}")

        # Executar sincronização em background
        result = await sync_deriv_history_period(
            date_from=request.date_from,
            date_to=request.date_to,
            force=request.force
        )

        if result["success"]:
            return SyncResponse(
                status="success",
                message=f"Sincronização concluída para período {request.date_from.date()} a {request.date_to.date()}",
                trades_synced=result["trades_synced"],
                trades_failed=result["trades_failed"],
                period_from=request.date_from.isoformat(),
                period_to=request.date_to.isoformat()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Falha na sincronização: {result.get('error', 'Unknown error')}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering sync: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger sync: {str(e)}"
        )


@router.get("/quick/{days}", response_model=SyncResponse, status_code=status.HTTP_200_OK)
async def quick_sync_last_days(days: int = Query(..., ge=1, le=90, description="Número de dias")):
    """
    Sincronização rápida dos últimos N dias

    Atalhos comuns:
    - /quick/7 - Última semana
    - /quick/30 - Último mês
    - /quick/90 - Últimos 3 meses
    """
    try:
        date_to = datetime.now()
        date_from = date_to - timedelta(days=days)

        request = SyncPeriodRequest(
            date_from=date_from,
            date_to=date_to,
            force=False
        )

        return await trigger_sync_period(request)

    except Exception as e:
        logger.error(f"Error in quick sync: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to quick sync: {str(e)}"
        )

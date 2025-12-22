"""
ABUTRE EVENTS - Pydantic Schemas

Schemas de validação para eventos enviados pelo Deriv Bot XML
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class CandleEvent(BaseModel):
    """Evento de candle fechado"""
    timestamp: datetime = Field(..., description="Timestamp do candle")
    symbol: str = Field(default="1HZ100V", description="Símbolo do ativo")
    open: float = Field(..., gt=0, description="Preço de abertura")
    high: float = Field(..., gt=0, description="Preço máximo")
    low: float = Field(..., gt=0, description="Preço mínimo")
    close: float = Field(..., gt=0, description="Preço de fechamento")
    color: Literal[1, -1] = Field(..., description="1 = green, -1 = red")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-22T18:30:00Z",
                "symbol": "1HZ100V",
                "open": 663.50,
                "high": 663.92,
                "low": 663.12,
                "close": 663.60,
                "color": 1
            }
        }


class TriggerEvent(BaseModel):
    """Evento de trigger (streak >= 8)"""
    timestamp: datetime = Field(..., description="Timestamp do trigger")
    streak_count: int = Field(..., ge=8, description="Tamanho do streak")
    direction: Literal["GREEN", "RED"] = Field(..., description="Direção do streak")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-22T18:30:05Z",
                "streak_count": 8,
                "direction": "GREEN"
            }
        }


class TradeOpenedEvent(BaseModel):
    """Evento de trade aberto"""
    timestamp: datetime = Field(..., description="Timestamp de abertura")
    trade_id: str = Field(..., description="ID único do trade")
    direction: Literal["CALL", "PUT"] = Field(..., description="Direção do trade")
    stake: float = Field(..., gt=0, description="Valor apostado")
    level: int = Field(..., ge=1, le=10, description="Nível do Martingale (1-10)")
    contract_id: Optional[str] = Field(None, description="ID do contrato Deriv")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-22T18:31:00Z",
                "trade_id": "trade_1703271060",
                "direction": "PUT",
                "stake": 1.0,
                "level": 1,
                "contract_id": "12345678"
            }
        }


class TradeClosedEvent(BaseModel):
    """Evento de trade fechado"""
    timestamp: datetime = Field(..., description="Timestamp de fechamento")
    trade_id: str = Field(..., description="ID único do trade")
    result: Literal["WIN", "LOSS", "STOP_LOSS"] = Field(..., description="Resultado do trade")
    profit: float = Field(..., description="Lucro/prejuízo (positivo = WIN, negativo = LOSS)")
    balance: float = Field(..., gt=0, description="Saldo após o trade")
    max_level_reached: int = Field(..., ge=1, le=10, description="Máximo nível atingido")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-22T18:32:00Z",
                "trade_id": "trade_1703271060",
                "result": "WIN",
                "profit": 0.95,
                "balance": 10001.95,
                "max_level_reached": 1
            }
        }


class BalanceEvent(BaseModel):
    """Evento de atualização de balance"""
    timestamp: datetime = Field(..., description="Timestamp da atualização")
    balance: float = Field(..., gt=0, description="Saldo atual")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-12-22T18:32:00Z",
                "balance": 10001.95
            }
        }


# Response schemas

class EventResponse(BaseModel):
    """Resposta padrão para eventos"""
    status: Literal["success", "error"] = Field(..., description="Status da operação")
    message: str = Field(..., description="Mensagem descritiva")
    event_id: Optional[int] = Field(None, description="ID do evento criado")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Candle event recorded successfully",
                "event_id": 12345
            }
        }

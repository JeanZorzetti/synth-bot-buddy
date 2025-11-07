"""
Pydantic models for order management
Objetivo 1 - Fase 2: Backend Models
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from datetime import datetime


class OrderRequest(BaseModel):
    """Request model for executing an order"""

    token: str = Field(
        ...,
        min_length=10,
        description="Deriv API token with Read and Trade scopes"
    )

    contract_type: Literal["CALL", "PUT"] = Field(
        ...,
        description="Contract type: CALL (Rise) or PUT (Fall)"
    )

    symbol: str = Field(
        default="R_75",
        description="Trading symbol (e.g., R_75, R_100, R_50)"
    )

    amount: float = Field(
        ...,
        gt=0,
        le=100,
        description="Stake amount in USD (must be between $0.35 and $100)"
    )

    duration: int = Field(
        ...,
        gt=0,
        le=60,
        description="Contract duration in specified unit (1-60)"
    )

    duration_unit: Literal["m", "h", "d"] = Field(
        default="m",
        description="Duration unit: m (minutes), h (hours), d (days)"
    )

    basis: Literal["stake", "payout"] = Field(
        default="stake",
        description="Basis of amount: stake or payout"
    )

    @validator('amount')
    def validate_amount(cls, v):
        """Validate amount is within Deriv's minimum"""
        if v < 0.35:
            raise ValueError("Amount must be at least $0.35 (Deriv minimum)")
        return round(v, 2)  # Round to 2 decimal places

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format"""
        v = v.upper().strip()
        # Common symbols validation
        valid_prefixes = ['R_', 'BOOM', 'CRASH', 'JD', 'FRX', 'WLD', 'OTC']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            # Allow but warn
            pass
        return v

    @validator('token')
    def validate_token(cls, v):
        """Validate token format (basic check)"""
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Token appears to be invalid (too short)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "token": "your_deriv_token_here",
                "contract_type": "CALL",
                "symbol": "R_75",
                "amount": 1.0,
                "duration": 5,
                "duration_unit": "m",
                "basis": "stake"
            }
        }


class OrderResponse(BaseModel):
    """Response model for order execution"""

    success: bool = Field(
        ...,
        description="Whether the order was executed successfully"
    )

    contract_id: Optional[int] = Field(
        None,
        description="Unique contract ID from Deriv (if successful)"
    )

    buy_price: Optional[float] = Field(
        None,
        description="Actual price paid for the contract"
    )

    payout: Optional[float] = Field(
        None,
        description="Potential payout if contract wins"
    )

    potential_profit: Optional[float] = Field(
        None,
        description="Potential profit (payout - buy_price)"
    )

    longcode: Optional[str] = Field(
        None,
        description="Human-readable description of the contract"
    )

    status: Optional[str] = Field(
        None,
        description="Current status of the contract"
    )

    contract_url: Optional[str] = Field(
        None,
        description="Direct URL to view contract on Deriv platform"
    )

    error: Optional[str] = Field(
        None,
        description="Error message if execution failed"
    )

    error_code: Optional[str] = Field(
        None,
        description="Error code for programmatic handling"
    )

    error_details: Optional[dict] = Field(
        None,
        description="Additional error details"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the response"
    )

    class Config:
        json_schema_extra = {
            "example_success": {
                "success": True,
                "contract_id": 123456789,
                "buy_price": 1.0,
                "payout": 1.85,
                "potential_profit": 0.85,
                "longcode": "Win payout if Volatility 75 Index is strictly higher...",
                "status": "active",
                "contract_url": "https://app.deriv.com/contract/123456789",
                "error": None,
                "timestamp": "2025-11-06T13:30:00"
            },
            "example_error": {
                "success": False,
                "contract_id": None,
                "error": "Insufficient balance",
                "error_code": "InsufficientBalance",
                "error_details": {"required": 1.0, "available": 0.5},
                "timestamp": "2025-11-06T13:30:00"
            }
        }


class ProposalData(BaseModel):
    """Model for proposal data from Deriv API"""

    id: str = Field(
        ...,
        description="Unique proposal ID"
    )

    ask_price: float = Field(
        ...,
        description="Price to buy the contract"
    )

    payout: float = Field(
        ...,
        description="Potential payout"
    )

    spot: Optional[float] = Field(
        None,
        description="Current spot price"
    )

    spot_time: Optional[int] = Field(
        None,
        description="Timestamp of spot price"
    )

    display_value: Optional[str] = Field(
        None,
        description="Formatted display value"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "f8d1234-abcd-1234-efgh-123456789abc",
                "ask_price": 1.0,
                "payout": 1.85,
                "spot": 1234.56,
                "spot_time": 1699284000,
                "display_value": "$1.00"
            }
        }


class OrderHistoryItem(BaseModel):
    """Model for order history tracking"""

    id: str = Field(
        ...,
        description="Unique identifier for this history item"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the order was placed"
    )

    contract_id: int = Field(
        ...,
        description="Deriv contract ID"
    )

    contract_type: Literal["CALL", "PUT"] = Field(
        ...,
        description="Type of contract"
    )

    symbol: str = Field(
        ...,
        description="Trading symbol"
    )

    amount: float = Field(
        ...,
        description="Stake amount"
    )

    payout: float = Field(
        ...,
        description="Potential payout"
    )

    result: Optional[Literal["win", "loss", "pending"]] = Field(
        None,
        description="Final result of the contract"
    )

    profit_loss: Optional[float] = Field(
        None,
        description="Actual profit or loss"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "hist_123456",
                "timestamp": "2025-11-06T13:30:00",
                "contract_id": 123456789,
                "contract_type": "CALL",
                "symbol": "R_75",
                "amount": 1.0,
                "payout": 1.85,
                "result": "win",
                "profit_loss": 0.85
            }
        }

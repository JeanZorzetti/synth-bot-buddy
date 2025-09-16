from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List

from websocket_manager import DerivWebSocketManager, ConnectionState
from enhanced_websocket_manager import EnhancedDerivWebSocket, ErrorType
from capital_manager import CapitalManager, TradeResult
from oauth_manager import oauth_manager, TokenData
from trading_engine import TradingEngine, MarketTick, SignalType
from deriv_trading_adapter import DerivTradingAdapter
from contract_proposals_engine import (
    ContractProposalsEngine, ProposalRequest, ProposalResponse,
    get_proposals_engine, initialize_proposals_engine
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class ConnectRequest(BaseModel):
    api_token: str

class ValidateTokenRequest(BaseModel):
    api_token: str

class SettingsRequest(BaseModel):
    stop_loss: float
    take_profit: float
    stake_amount: float
    aggressiveness: str
    indicators: dict
    selected_assets: dict

# OAuth request models
class OAuthStartRequest(BaseModel):
    scopes: Optional[List[str]] = None
    redirect_uri: Optional[str] = None

class OAuthCallbackRequest(BaseModel):
    code: str
    state: str

class TokenRefreshRequest(BaseModel):
    refresh_token: str

# Deriv OAuth request models
class DerivOAuthStartRequest(BaseModel):
    app_id: str = "99188"
    redirect_uri: str = "https://botderiv.roilabs.com.br/auth"
    affiliate_token: Optional[str] = None
    utm_campaign: Optional[str] = None

class DerivOAuthCallbackRequest(BaseModel):
    accounts: List[str]
    token1: str
    token2: Optional[str] = None
    token3: Optional[str] = None
    cur1: str = "USD"
    cur2: Optional[str] = None
    cur3: Optional[str] = None

# Deriv API request models
class DerivConnectRequest(BaseModel):
    api_token: str
    demo: bool = True

class DerivTradeRequest(BaseModel):
    contract_type: str  # "CALL", "PUT", "DIGITEVEN", etc.
    symbol: str        # "R_50", "R_100", etc.
    amount: float      # Valor do stake
    duration: int      # Duração em minutos/segundos
    duration_unit: str = "m"  # "m" para minutos, "s" para segundos
    barrier: Optional[str] = None  # Barreira para contratos que precisam
    basis: str = "stake"  # "stake" ou "payout"
    currency: str = "USD"  # Moeda do contrato

class DerivSellRequest(BaseModel):
    contract_id: int
    price: Optional[float] = None

# Proposal request models
class ProposalRequest(BaseModel):
    contract_type: str
    symbol: str
    amount: float
    duration: int
    duration_unit: str = "m"
    barrier: Optional[str] = None
    basis: str = "stake"
    currency: str = "USD"

class BatchProposalRequest(BaseModel):
    proposals: List[ProposalRequest]
    realtime: bool = False

# Global instances
ws_manager: DerivWebSocketManager = None
enhanced_ws_manager: EnhancedDerivWebSocket = None
capital_manager: CapitalManager = None
trading_engine: TradingEngine = None
deriv_adapter: DerivTradingAdapter = None
proposals_engine: ContractProposalsEngine = None

# Bot settings
bot_settings = {
    "stop_loss": 50.0,
    "take_profit": 100.0,
    "stake_amount": 10.0,
    "aggressiveness": "moderate",
    "indicators": {
        "use_rsi": True,
        "use_moving_averages": True,
        "use_bollinger": False
    },
    "selected_assets": {
        "volatility75": True,
        "volatility100": True,
        "jump25": False,
        "jump50": False,
        "jump75": False,
        "jump100": False,
        "boom1000": False,
        "crash1000": False
    }
}

bot_status = {
    "is_running": False,
    "connection_status": "disconnected",
    "balance": 0.0,
    "last_tick": None,
    "session_pnl": 0.0,
    "trades_count": 0,
    "capital_management": {
        "next_amount": 0.0,
        "current_sequence": 0,
        "is_in_loss_sequence": False,
        "accumulated_profit": 0.0,
        "risk_level": "LOW"
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global ws_manager, capital_manager, enhanced_ws_manager, trading_engine, deriv_adapter, proposals_engine
    
    # Startup
    logger.info("Iniciando aplicação Cérebro...")
    
    # Log current configuration
    app_id = os.getenv("DERIV_APP_ID", "99188")
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10.0"))
    environment = os.getenv("ENVIRONMENT", "development")
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  - App ID: {app_id}")
    logger.info(f"  - Environment: {environment}")
    logger.info(f"  - Initial Capital: ${initial_capital}")
    logger.info(f"  - WebSocket URL: wss://ws.derivws.com/websockets/v3?app_id={app_id}")
    
    # Initialize Capital Manager
    capital_manager = CapitalManager(
        initial_capital=initial_capital,
        reinvestment_rate=0.20,  # 20% reinvestment
        martingale_multiplier=1.25  # 1.25x martingale
    )
    
    # Initialize Enhanced WebSocket Manager
    enhanced_ws_manager = EnhancedDerivWebSocket(
        app_id=app_id,
        api_token=""  # Will be set when connecting
    )
    
    # Initialize Trading Engine
    trading_engine = TradingEngine(
        websocket_manager=enhanced_ws_manager,
        settings=bot_settings
    )
    
    # Initialize Deriv Trading Adapter
    deriv_adapter = DerivTradingAdapter(
        app_id=int(app_id),
        demo=True  # Usar conta demo por padrão
    )

    # Initialize Contract Proposals Engine
    proposals_engine = get_proposals_engine(deriv_adapter.deriv_api)
    await initialize_proposals_engine(deriv_adapter.deriv_api)
    
    # WebSocket manager will be initialized when /connect is called with token
    # This allows dynamic token configuration from frontend
    logger.info("Trading engine, WebSocket manager and Deriv adapter initialized")
    
    yield
    
    # Shutdown
    logger.info("Encerrando aplicação Cérebro...")
    if ws_manager:
        await ws_manager.disconnect()
    if deriv_adapter:
        await deriv_adapter.disconnect()

app = FastAPI(
    title="Synth Bot Buddy - Cérebro",
    description="O backend inteligente para análise e execução de trades na Deriv.",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000",
        "http://localhost:8080",  # Vite dev server
        "http://localhost:8081",  # Vite dev server
        "http://localhost:8082",  # Vite dev server
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "https://botderiv.roilabs.com.br",
        "http://botderiv.roilabs.com.br"
    ],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Event Handlers ---

async def handle_tick_data(tick_data: Dict[str, Any]):
    """Handle incoming tick data"""
    global bot_status
    bot_status["last_tick"] = tick_data
    logger.info(f"Tick received: {tick_data['symbol']} = {tick_data['price']}")

async def handle_balance_update(balance_data: Dict[str, Any]):
    """Handle balance updates"""
    global bot_status
    bot_status["balance"] = balance_data["balance"]
    logger.info(f"Balance updated: {balance_data['balance']} {balance_data['currency']}")

async def handle_trade_result(trade_data: Dict[str, Any]):
    """Handle trade execution results"""
    global bot_status, capital_manager
    
    if not capital_manager:
        logger.error("Capital manager not initialized")
        return
    
    # Extract trade information
    contract_id = trade_data.get("contract_id", "unknown")
    trade_type = trade_data.get("type", "buy")
    payout = trade_data.get("payout", 0.0)
    price = trade_data.get("price", 0.0)
    
    # Determine result based on payout
    if trade_type == "buy" and payout > 0:
        result = TradeResult.WIN
        profit_loss = payout - price
    elif trade_type == "sell":
        result = TradeResult.WIN if payout > 0 else TradeResult.LOSS
        profit_loss = payout - price if payout > 0 else -price
    else:
        result = TradeResult.LOSS
        profit_loss = -price
    
    # Record trade in capital manager
    trade_record = capital_manager.record_trade(
        trade_id=contract_id,
        amount=price,
        result=result,
        payout=payout
    )
    
    # Update bot status
    bot_status["trades_count"] += 1
    bot_status["session_pnl"] = capital_manager.accumulated_profit
    
    # Update capital management status
    capital_stats = capital_manager.get_stats()
    bot_status["capital_management"].update({
        "next_amount": capital_manager.get_next_trade_amount(),
        "current_sequence": capital_stats["capital_info"]["loss_sequence_count"],
        "is_in_loss_sequence": capital_stats["capital_info"]["is_in_loss_sequence"],
        "accumulated_profit": capital_stats["capital_info"]["accumulated_profit"],
        "risk_level": capital_manager.get_risk_assessment()["risk_level"]
    })
    
    logger.info(f"Trade processed: {contract_id} | Result: {result.value} | P/L: ${profit_loss:.2f} | Next: ${capital_manager.get_next_trade_amount():.2f}")

async def handle_connection_status(status: ConnectionState):
    """Handle connection status changes"""
    global bot_status
    bot_status["connection_status"] = status.value
    logger.info(f"Connection status: {status.value}")

async def handle_websocket_error(error_type: ErrorType, error_message: str, error_code: str = None):
    """Handle WebSocket errors with enhanced error information"""
    global bot_status
    
    logger.error(f"WebSocket Error [{error_type.value}]: {error_message} (Code: {error_code})")
    
    # Update bot status with error information
    bot_status["last_error"] = {
        "type": error_type.value,
        "message": error_message,
        "code": error_code,
        "timestamp": time.time()
    }
    
    # Handle specific error types
    if error_type == ErrorType.AUTHENTICATION_ERROR:
        bot_status["connection_status"] = "authentication_failed"
        logger.warning("🔐 Authentication failed - may need to refresh OAuth token")
    elif error_type == ErrorType.NETWORK_ERROR:
        bot_status["connection_status"] = "network_error"
        logger.warning("🌐 Network error - attempting automatic reconnection")
    elif error_type == ErrorType.RATE_LIMIT_ERROR:
        logger.warning("⏱️ Rate limit exceeded - throttling requests")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    return {
        "status": "ok", 
        "message": "Cérebro do Synth Bot Buddy está online!",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint for monitoring."""
    global ws_manager, bot_status
    
    health_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "websocket_manager": {
            "initialized": ws_manager is not None,
            "state": ws_manager.state.value if ws_manager else "not_initialized"
        },
        "bot_status": bot_status,
        "dependencies": {
            "fastapi": True,
            "websockets": True,
            "deriv_token_configured": bool(os.getenv("DERIV_API_TOKEN"))
        }
    }
    
    return health_info

@app.get("/routes")
async def list_routes():
    """List all available API routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', '')
            })
    return {"routes": routes, "total": len(routes)}

@app.get("/settings")
async def get_settings():
    """Get current bot settings"""
    return {"settings": bot_settings}

@app.post("/settings")
async def update_settings(settings: SettingsRequest):
    """Update bot settings"""
    global bot_settings, trading_engine
    
    try:
        # Validate settings
        if settings.stop_loss <= 0 or settings.take_profit <= 0 or settings.stake_amount <= 0:
            raise HTTPException(status_code=400, detail="All monetary values must be positive")
        
        if settings.stop_loss >= settings.take_profit:
            raise HTTPException(status_code=400, detail="Take profit must be greater than stop loss")
        
        if settings.aggressiveness not in ["conservative", "moderate", "aggressive"]:
            raise HTTPException(status_code=400, detail="Invalid aggressiveness level")
        
        # Check if at least one asset is selected
        if not any(settings.selected_assets.values()):
            raise HTTPException(status_code=400, detail="At least one asset must be selected")
        
        # Update settings
        bot_settings.update({
            "stop_loss": settings.stop_loss,
            "take_profit": settings.take_profit,
            "stake_amount": settings.stake_amount,
            "aggressiveness": settings.aggressiveness,
            "indicators": settings.indicators,
            "selected_assets": settings.selected_assets
        })
        
        # Update trading engine settings if it exists
        if trading_engine:
            trading_engine.settings = bot_settings.copy()
            # Update signal detector settings as well
            if trading_engine.signal_detector:
                trading_engine.signal_detector.settings = bot_settings.copy()
        
        logger.info(f"Settings updated: Stop Loss ${settings.stop_loss}, Take Profit ${settings.take_profit}")
        logger.info(f"Trading engine settings synchronized")
        
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "settings": bot_settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading monitoring endpoints
@app.get("/trading/signals/{symbol}")
async def get_trading_signals(symbol: str):
    """Get recent trading signals for a symbol"""
    global trading_engine
    
    if not trading_engine or not trading_engine.signal_detector:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        signals = trading_engine.signal_detector.signal_history.get(symbol, [])
        recent_signals = signals[-10:]  # Last 10 signals
        
        return {
            "symbol": symbol,
            "recent_signals": [
                {
                    "signal_type": signal.signal_type.value,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "timestamp": signal.timestamp,
                    "indicators": signal.indicators,
                    "reason": signal.reason
                }
                for signal in recent_signals
            ],
            "last_signal": {
                "signal_type": recent_signals[-1].signal_type.value,
                "strength": recent_signals[-1].strength,
                "confidence": recent_signals[-1].confidence,
                "reason": recent_signals[-1].reason
            } if recent_signals else None
        }
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/performance")
async def get_trading_performance():
    """Get detailed trading performance metrics"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        status = trading_engine.get_status()
        win_rate = (trading_engine.wins / max(1, trading_engine.wins + trading_engine.losses)) * 100
        
        return {
            "session_stats": {
                "session_pnl": status['session_pnl'],
                "total_trades": status['trades_count'],
                "active_trades": status['active_trades'],
                "wins": status['wins'],
                "losses": status['losses'],
                "win_rate": round(win_rate, 1),
                "total_invested": status['total_invested'],
                "total_returned": status['total_returned'],
                "net_profit": round(status['total_returned'] - status['total_invested'], 2)
            },
            "risk_metrics": {
                "current_risk_level": status['capital_management']['risk_level'],
                "is_in_loss_sequence": status['capital_management']['is_in_loss_sequence'],
                "next_trade_amount": status['capital_management']['next_amount'],
                "accumulated_profit": status['capital_management']['accumulated_profit']
            },
            "trading_engine": {
                "is_running": status['is_running'],
                "active_since": time.time() - 3600 if status['is_running'] else None  # Simplified
            }
        }
    except Exception as e:
        logger.error(f"Error getting trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/history")
async def get_trade_history(limit: int = 50):
    """Get recent trade history"""
    global trading_engine
    
    # Always return a valid response, even if trading engine is not available
    if not trading_engine:
        return {
            "trades": [],
            "total_trades": 0,
            "summary": {
                "total_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0
            },
            "status": "trading_engine_not_available"
        }
    
    try:
        # Safely get trade history
        trade_history = getattr(trading_engine, 'trade_history', [])
        trades = trade_history[-limit:] if trade_history else []
        
        formatted_trades = []
        for trade in trades:
            try:
                formatted_trades.append({
                    "trade_id": getattr(trade, 'trade_id', ''),
                    "symbol": getattr(trade, 'symbol', ''),
                    "contract_type": getattr(trade, 'contract_type', ''),
                    "amount": getattr(trade, 'amount', 0.0),
                    "entry_price": getattr(trade, 'entry_price', 0.0),
                    "exit_price": getattr(trade, 'exit_price', 0.0),
                    "entry_time": getattr(trade, 'entry_time', ''),
                    "exit_time": getattr(trade, 'exit_time', ''),
                    "duration": getattr(trade, 'duration', 0),
                    "status": getattr(getattr(trade, 'status', None), 'value', 'unknown'),
                    "pnl": getattr(trade, 'pnl', 0.0),
                    "contract_id": getattr(trade, 'contract_id', '')
                })
            except Exception as trade_error:
                logger.warning(f"Error formatting trade: {trade_error}")
                continue
        
        # Safe calculations
        total_trades = len(trade_history) if trade_history else 0
        won_trades = []
        lost_trades = []
        total_pnl = 0.0
        
        if trade_history:
            for trade in trade_history:
                try:
                    status = getattr(getattr(trade, 'status', None), 'value', '')
                    pnl = getattr(trade, 'pnl', 0.0)
                    
                    if status == 'won':
                        won_trades.append(trade)
                    elif status == 'lost':
                        lost_trades.append(trade)
                    
                    if pnl:
                        total_pnl += pnl
                except Exception:
                    continue
        
        win_rate = (len(won_trades) / max(1, total_trades)) * 100 if total_trades > 0 else 0.0
        
        return {
            "trades": formatted_trades,
            "total_trades": total_trades,
            "summary": {
                "total_pnl": total_pnl,
                "wins": len(won_trades),
                "losses": len(lost_trades),
                "win_rate": win_rate
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        # Return empty but valid response instead of 500 error
        return {
            "trades": [],
            "total_trades": 0,
            "summary": {
                "total_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0
            },
            "status": f"error: {str(e)}"
        }

@app.post("/trading/reset-session")
async def reset_trading_session():
    """Reset the current trading session"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        # Stop trading engine if running
        was_running = trading_engine.is_running
        if was_running:
            trading_engine.stop()
        
        # Reset session
        trading_engine.reset_session()
        
        # Restart if it was running
        if was_running:
            trading_engine.start()
        
        logger.info("Trading session reset")
        
        return {
            "status": "success",
            "message": "Trading session reset successfully",
            "was_running": was_running,
            "restarted": was_running
        }
    except Exception as e:
        logger.error(f"Error resetting trading session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/health")
async def trading_health_check():
    """Check trading engine health and available endpoints"""
    global trading_engine
    
    return {
        "status": "ok",
        "trading_engine_available": trading_engine is not None,
        "trading_engine_running": trading_engine.is_running if trading_engine else False,
        "available_endpoints": [
            "/trading/signals/{symbol}",
            "/trading/performance", 
            "/trading/history",
            "/trading/reset-session",
            "/trading/health"
        ],
        "total_trades": len(trading_engine.trade_history) if trading_engine else 0,
        "active_trades": len(trading_engine.active_trades) if trading_engine else 0
    }

@app.get("/config")
async def get_configuration():
    """Get current configuration for debugging."""
    return {
        "app_id": os.getenv("DERIV_APP_ID", "99188"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "initial_capital": os.getenv("INITIAL_CAPITAL", "10.0"),
        "websocket_url": f"wss://ws.derivws.com/websockets/v3?app_id={os.getenv('DERIV_APP_ID', '99188')}",
        "has_token_env": bool(os.getenv("DERIV_API_TOKEN")),
        "git_commit": "f631734",  # Latest commit
        "timestamp": time.time()
    }

@app.post("/validate-token")
async def validate_token(request: ValidateTokenRequest):
    """Test token validity without affecting the main WebSocket connection"""
    try:
        # Create a temporary WebSocket manager for testing
        app_id = os.getenv("DERIV_APP_ID", "99188")
        test_ws_manager = DerivWebSocketManager(app_id=app_id, api_token=request.api_token)
        
        logger.info(f"Testing token: {request.api_token[:10]}...")
        
        # Try to connect
        success = await test_ws_manager.connect()
        if not success:
            return {
                "valid": False, 
                "error": "Failed to establish WebSocket connection",
                "state": test_ws_manager.state.value
            }
        
        # Wait for authentication
        max_wait = 15  # seconds
        wait_time = 0
        while test_ws_manager.state != ConnectionState.AUTHENTICATED and wait_time < max_wait:
            if test_ws_manager.state == ConnectionState.ERROR:
                break
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        # Check result
        if test_ws_manager.state == ConnectionState.AUTHENTICATED:
            # Disconnect test connection
            await test_ws_manager.disconnect()
            return {
                "valid": True,
                "message": "Token is valid and can authenticate with Deriv API",
                "state": "authenticated"
            }
        else:
            await test_ws_manager.disconnect()
            return {
                "valid": False,
                "error": f"Authentication failed or timed out",
                "state": test_ws_manager.state.value
            }
            
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return {
            "valid": False,
            "error": str(e),
            "state": "error"
        }

@app.get("/status")
async def get_status():
    """Get current bot status"""
    global bot_status, trading_engine, enhanced_ws_manager
    
    # Update status from trading engine if available
    if trading_engine:
        trading_status = trading_engine.get_status()
        bot_status.update(trading_status)
    
    # Update connection status
    if enhanced_ws_manager and hasattr(enhanced_ws_manager, 'connection_state'):
        if enhanced_ws_manager.connection_state == 'authenticated':
            bot_status['connection_status'] = 'authenticated'
        elif enhanced_ws_manager.connection_state == 'connected':
            bot_status['connection_status'] = 'connected'
        else:
            bot_status['connection_status'] = 'disconnected'
    
    return bot_status

@app.post("/connect")
async def connect_to_deriv(request: ConnectRequest):
    """Connect to Deriv WebSocket API with provided token"""
    global ws_manager
    
    try:
        # Create new WebSocket manager with provided token
        app_id = os.getenv("DERIV_APP_ID", "99188")
        ws_manager = DerivWebSocketManager(app_id=app_id, api_token=request.api_token)
        
        # Set up event handlers
        ws_manager.set_tick_handler(handle_tick_data)
        ws_manager.set_balance_handler(handle_balance_update)
        ws_manager.set_trade_handler(handle_trade_result)
        ws_manager.set_connection_handler(handle_connection_status)
        
        # Connect
        logger.info(f"Attempting connection with token: {request.api_token[:10]}...")
        success = await ws_manager.connect()
        if success:
            logger.info(f"WebSocket connected. Current state: {ws_manager.state.value}")
            return {"status": "connecting", "message": "Connection initiated with provided token"}
        else:
            logger.error(f"Connection failed. Final state: {ws_manager.state.value}")
            raise HTTPException(status_code=500, detail="Failed to connect to Deriv API")
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_from_deriv():
    """Disconnect from Deriv WebSocket API"""
    global ws_manager
    
    if not ws_manager:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    
    try:
        await ws_manager.disconnect()
        return {"status": "disconnected", "message": "Disconnected from Deriv API"}
    except Exception as e:
        logger.error(f"Disconnect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
async def start_bot():
    """Start the trading bot"""
    global ws_manager, enhanced_ws_manager, trading_engine, bot_status
    
    if not enhanced_ws_manager or not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    try:
        # Check if already running
        if trading_engine.is_running:
            return {"status": "running", "message": "Bot já está rodando"}
        
        # Update trading engine settings
        trading_engine.settings = bot_settings.copy()
        
        # Start the trading engine
        trading_engine.start()
        
        # Setup tick handlers to feed data to trading engine
        if enhanced_ws_manager:
            async def handle_tick(tick_data):
                """Handle incoming market ticks"""
                try:
                    tick = MarketTick(
                        symbol=tick_data.get('symbol', 'R_75'),
                        price=float(tick_data.get('quote', tick_data.get('price', 0))),
                        timestamp=float(tick_data.get('epoch', time.time()))
                    )
                    
                    # Feed tick to trading engine
                    trading_engine.add_market_tick(tick)
                    
                    # Analyze and potentially trade
                    signal = await trading_engine.analyze_and_trade(tick.symbol)
                    
                    # Update bot status with latest tick
                    bot_status["last_tick"] = {
                        "symbol": tick.symbol,
                        "price": tick.price,
                        "timestamp": tick.timestamp
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing tick: {e}")
            
            # Set the tick handler
            enhanced_ws_manager.set_tick_handler(handle_tick)
        
        # Subscribe to configured assets
        enabled_assets = [
            asset for asset, enabled in bot_settings.get('selected_assets', {}).items() 
            if enabled
        ]
        
        # Map asset names to symbols
        asset_symbol_map = {
            'volatility75': 'R_75',
            'volatility100': 'R_100',
            'volatility25': 'R_25',
            'volatility50': 'R_50',
            'jump25': 'JD25',
            'jump50': 'JD50', 
            'jump75': 'JD75',
            'jump100': 'JD100',
            'boom1000': 'RDBULL',
            'crash1000': 'RDBEAR'
        }
        
        # Subscribe to enabled asset ticks
        for asset in enabled_assets:
            symbol = asset_symbol_map.get(asset, 'R_75')
            if enhanced_ws_manager:
                await enhanced_ws_manager.subscribe_to_ticks(symbol)
                logger.info(f"Subscribed to {symbol} ({asset})")
        
        # If no assets selected, default to R_75
        if not enabled_assets and enhanced_ws_manager:
            await enhanced_ws_manager.subscribe_to_ticks('R_75')
            logger.info("No assets selected, defaulting to R_75")
        
        bot_status["is_running"] = True
        logger.info("Trading bot iniciado com sucesso")
        
        return {
            "status": "running", 
            "message": "Trading bot iniciado com sucesso",
            "subscribed_assets": enabled_assets or ['volatility75'],
            "settings_applied": {
                "stop_loss": bot_settings.get('stop_loss'),
                "take_profit": bot_settings.get('take_profit'),
                "stake_amount": bot_settings.get('stake_amount'),
                "aggressiveness": bot_settings.get('aggressiveness')
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_bot():
    """Stop the trading bot"""
    global bot_status, trading_engine
    
    try:
        if trading_engine:
            trading_engine.stop()
        
        bot_status["is_running"] = False
        logger.info("Trading bot parado")
        
        return {
            "status": "stopped", 
            "message": "Trading bot parado com sucesso",
            "final_session_pnl": trading_engine.session_pnl if trading_engine else 0.0,
            "total_trades": trading_engine.trades_count if trading_engine else 0
        }
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buy")
async def buy_contract(contract_type: str, amount: float = None, duration: int = 5, symbol: str = "R_75"):
    """Buy a trading contract with smart capital management"""
    global ws_manager, capital_manager
    
    if not ws_manager or ws_manager.state != ConnectionState.AUTHENTICATED:
        raise HTTPException(status_code=401, detail="Not connected or authenticated")
    
    if not bot_status["is_running"]:
        raise HTTPException(status_code=400, detail="Bot is not running")
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    try:
        # Use capital manager amount if not specified
        if amount is None:
            amount = capital_manager.get_next_trade_amount()
            logger.info(f"Using capital-managed amount: ${amount}")
        
        # Get risk assessment
        risk_assessment = capital_manager.get_risk_assessment()
        
        # Check risk level
        if risk_assessment["risk_level"] == "HIGH":
            logger.warning(f"HIGH RISK TRADE: Amount ${amount} ({risk_assessment['risk_percentage']}% of initial capital)")
        
        success = await ws_manager.buy_contract(contract_type, amount, duration, symbol)
        if success:
            return {
                "status": "order_sent", 
                "message": f"Order sent: {contract_type} ${amount} on {symbol}",
                "capital_info": {
                    "amount": amount,
                    "risk_level": risk_assessment["risk_level"],
                    "risk_percentage": risk_assessment["risk_percentage"],
                    "is_martingale": capital_manager.is_in_loss_sequence
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send buy order")
    except Exception as e:
        logger.error(f"Buy order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Capital Management Endpoints ---

@app.get("/capital/stats")
async def get_capital_stats():
    """Get complete capital management statistics"""
    global capital_manager
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    return capital_manager.get_stats()

@app.get("/capital/risk")
async def get_risk_assessment():
    """Get current risk assessment"""
    global capital_manager
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    return capital_manager.get_risk_assessment()

@app.get("/capital/next-amount")
async def get_next_amount():
    """Get next trading amount"""
    global capital_manager
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    next_amount = capital_manager.get_next_trade_amount()
    risk_assessment = capital_manager.get_risk_assessment()
    
    return {
        "next_amount": next_amount,
        "risk_level": risk_assessment["risk_level"],
        "risk_percentage": risk_assessment["risk_percentage"],
        "is_in_loss_sequence": capital_manager.is_in_loss_sequence,
        "recommendations": risk_assessment["recommendations"]
    }

@app.post("/capital/reset")
async def reset_capital_session():
    """Reset capital management session"""
    global capital_manager, bot_status
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    capital_manager.reset_session()
    
    # Reset bot status capital info
    bot_status["session_pnl"] = 0.0
    bot_status["trades_count"] = 0
    bot_status["capital_management"] = {
        "next_amount": capital_manager.initial_capital,
        "current_sequence": 0,
        "is_in_loss_sequence": False,
        "accumulated_profit": 0.0,
        "risk_level": "LOW"
    }
    
    return {"status": "reset", "message": "Capital management session reset successfully"}

@app.post("/capital/simulate")
async def simulate_trading_sequence(results: List[str]):
    """Simulate a trading sequence to test capital management"""
    global capital_manager
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    try:
        # Convert string results to TradeResult enum
        trade_results = []
        for result_str in results:
            if result_str.lower() in ["win", "w", "1"]:
                trade_results.append(TradeResult.WIN)
            elif result_str.lower() in ["loss", "lose", "l", "0"]:
                trade_results.append(TradeResult.LOSS)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid result: {result_str}. Use 'win'/'w'/'1' or 'loss'/'l'/'0'")
        
        simulation = capital_manager.simulate_sequence(trade_results, payout_multiplier=1.8)
        
        return {
            "simulation": simulation,
            "summary": {
                "total_trades": len(results),
                "wins": len([r for r in trade_results if r == TradeResult.WIN]),
                "losses": len([r for r in trade_results if r == TradeResult.LOSS]),
                "final_profit_loss": simulation["total_profit_loss"]
            }
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capital/history")
async def get_trade_history():
    """Get complete trade history"""
    global capital_manager
    
    if not capital_manager:
        raise HTTPException(status_code=500, detail="Capital manager not initialized")
    
    return {
        "trade_history": capital_manager.export_history(),
        "total_trades": len(capital_manager.trade_history),
        "current_stats": capital_manager.get_stats()
    }

# --- Deriv OAuth 2.0 Endpoints ---

@app.post("/deriv/oauth/start")
async def start_deriv_oauth_flow(request: DerivOAuthStartRequest):
    """
    Iniciar fluxo OAuth da Deriv
    Retorna URL de autorização para o usuário visitar
    """
    try:
        # Construir URL de autorização da Deriv
        auth_url = f"https://oauth.deriv.com/oauth2/authorize?app_id={request.app_id}"

        # Adicionar parâmetros opcionais
        if request.affiliate_token:
            auth_url += f"&affiliate_token={request.affiliate_token}"

        if request.utm_campaign:
            auth_url += f"&utm_campaign={request.utm_campaign}"

        logger.info(f"Deriv OAuth flow iniciado para App ID: {request.app_id}")

        return {
            "status": "success",
            "authorization_url": auth_url,
            "app_id": request.app_id,
            "redirect_uri": request.redirect_uri,
            "message": "Visite a URL de autorização para completar o fluxo OAuth"
        }

    except Exception as e:
        logger.error(f"Erro ao iniciar fluxo OAuth da Deriv: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deriv/oauth/callback")
async def handle_deriv_oauth_callback(request: DerivOAuthCallbackRequest):
    """
    Lidar com callback OAuth da Deriv
    Processar parâmetros de sessão retornados
    """
    try:
        # Processar tokens e contas retornados pela Deriv
        session_data = {
            "accounts": request.accounts,
            "tokens": [request.token1],
            "currencies": [request.cur1]
        }

        # Adicionar tokens adicionais se existirem
        if request.token2:
            session_data["tokens"].append(request.token2)
            session_data["currencies"].append(request.cur2 or "USD")

        if request.token3:
            session_data["tokens"].append(request.token3)
            session_data["currencies"].append(request.cur3 or "USD")

        logger.info(f"Callback OAuth da Deriv processado para {len(request.accounts)} conta(s)")

        return {
            "status": "success",
            "message": "Callback OAuth da Deriv processado com sucesso",
            "session_data": session_data,
            "primary_token": request.token1,
            "primary_account": request.accounts[0] if request.accounts else None
        }

    except Exception as e:
        logger.error(f"Erro no callback OAuth da Deriv: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/deriv/oauth/connect")
async def connect_with_deriv_oauth_token(token: str, demo: bool = True):
    """
    Conectar à Deriv WebSocket usando token OAuth
    """
    global deriv_adapter

    try:
        if not deriv_adapter:
            raise HTTPException(status_code=500, detail="Deriv adapter não inicializado")

        # Conectar usando o token OAuth
        connected = await deriv_adapter.connect()
        if not connected:
            raise HTTPException(status_code=500, detail="Falha ao conectar com Deriv")

        # Autenticar usando o token OAuth
        authenticated = await deriv_adapter.authenticate(token)
        if not authenticated:
            raise HTTPException(status_code=401, detail="Token OAuth inválido ou falha na autenticação")

        connection_info = deriv_adapter.get_connection_info()

        logger.info(f"Conectado com sucesso usando OAuth token para conta: {connection_info.get('loginid')}")

        return {
            "status": "success",
            "message": "Conectado e autenticado com sucesso usando OAuth",
            "connection_info": connection_info,
            "auth_method": "OAuth 2.0",
            "demo_mode": demo
        }

    except Exception as e:
        logger.error(f"Erro ao conectar com token OAuth da Deriv: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OAuth 2.0 Endpoints (Originais) ---

@app.post("/oauth/start")
async def start_oauth_flow(request: OAuthStartRequest):
    """
    Start OAuth 2.0 authorization flow
    Returns authorization URL for user to visit
    """
    try:
        # Clean up expired states
        oauth_manager.cleanup_expired_states()
        
        # Get authorization URL
        auth_data = await oauth_manager.get_authorization_url(
            scopes=request.scopes,
            redirect_uri=request.redirect_uri
        )
        
        logger.info(f"OAuth flow started for scopes: {auth_data['scopes']}")
        
        return {
            "status": "success",
            "authorization_url": auth_data["authorization_url"],
            "state": auth_data["state"],
            "scopes": auth_data["scopes"],
            "redirect_uri": auth_data["redirect_uri"],
            "message": "Visit the authorization URL to complete OAuth flow"
        }
        
    except Exception as e:
        logger.error(f"Error starting OAuth flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/oauth/callback")
async def handle_oauth_callback(request: OAuthCallbackRequest):
    """
    Handle OAuth callback with authorization code
    Exchange code for access token
    """
    try:
        # Exchange code for token
        token_data = await oauth_manager.exchange_code_for_token(
            authorization_code=request.code,
            state=request.state
        )
        
        # Validate the token
        user_info = await oauth_manager.validate_token(token_data.access_token)
        
        logger.info(f"OAuth flow completed for user: {user_info.get('email', 'unknown')}")
        
        return {
            "status": "success",
            "message": "OAuth authentication successful",
            "token_info": {
                "token_type": token_data.token_type,
                "expires_in": token_data.expires_in,
                "scope": token_data.scope,
                "created_at": token_data.created_at.isoformat()
            },
            "user_info": user_info,
            "encrypted_token": oauth_manager.encrypt_token(token_data)
        }
        
    except Exception as e:
        logger.error(f"Error handling OAuth callback: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/oauth/refresh")
async def refresh_oauth_token(request: TokenRefreshRequest):
    """
    Refresh OAuth access token using refresh token
    """
    try:
        # Refresh the token
        new_token_data = await oauth_manager.refresh_access_token(request.refresh_token)
        
        logger.info("OAuth token refreshed successfully")
        
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "token_info": {
                "token_type": new_token_data.token_type,
                "expires_in": new_token_data.expires_in,
                "scope": new_token_data.scope,
                "created_at": new_token_data.created_at.isoformat()
            },
            "encrypted_token": oauth_manager.encrypt_token(new_token_data)
        }
        
    except Exception as e:
        logger.error(f"Error refreshing OAuth token: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/oauth/validate")
async def validate_oauth_token(request: ValidateTokenRequest):
    """
    Validate OAuth access token and get user information
    Enhanced version with detailed token information
    """
    try:
        # Get user information
        user_info = await oauth_manager.validate_token(request.api_token)
        
        # Get detailed token information
        token_info = await oauth_manager.get_token_info(request.api_token)
        
        logger.info(f"OAuth token validated for user: {user_info.get('email', 'unknown')}")
        
        return {
            "valid": True,
            "status": "success",
            "message": "Token is valid",
            "user_info": user_info,
            "token_info": token_info
        }
        
    except Exception as e:
        logger.error(f"OAuth token validation failed: {e}")
        return {
            "valid": False,
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

@app.post("/oauth/revoke")
async def revoke_oauth_token(request: ValidateTokenRequest):
    """
    Revoke OAuth access token (logout)
    """
    try:
        success = await oauth_manager.revoke_token(request.api_token)
        
        if success:
            logger.info("OAuth token revoked successfully")
            return {
                "status": "success",
                "message": "Token revoked successfully"
            }
        else:
            return {
                "status": "warning",
                "message": "Token revocation completed (status unclear)"
            }
        
    except Exception as e:
        logger.error(f"Error revoking OAuth token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/oauth/scopes")
async def get_available_scopes():
    """
    Get list of available OAuth scopes
    """
    return {
        "available_scopes": oauth_manager.available_scopes,
        "default_scopes": oauth_manager.default_scopes,
        "scope_descriptions": {
            "read": "Read account information, balance, and trading history",
            "trade": "Execute trading operations (buy/sell contracts)",
            "payments": "Handle payments and withdrawals (use with caution)",
            "admin": "Administrative access (use with extreme caution)"
        },
        "recommended_scopes": ["read", "trade"],
        "minimal_scopes": ["read"]
    }

@app.get("/oauth/config")
async def get_oauth_config():
    """
    Get OAuth configuration for debugging
    """
    return {
        "oauth_endpoints": {
            "authorize_url": oauth_manager.authorize_url,
            "token_url": oauth_manager.token_url,
            "user_info_url": oauth_manager.user_info_url
        },
        "client_configuration": {
            "client_id": oauth_manager.client_id,
            "has_client_secret": bool(oauth_manager.client_secret),
            "default_redirect_uri": oauth_manager.redirect_uri
        },
        "security_features": {
            "pkce_enabled": True,
            "state_parameter": True,
            "token_encryption": True
        },
        "active_states_count": len(oauth_manager.active_states)
    }

# --- Enhanced connection endpoint with OAuth support ---

@app.post("/connect/oauth")
async def connect_with_oauth_token(encrypted_token: str):
    """
    Connect to Deriv WebSocket using OAuth encrypted token
    """
    global ws_manager
    
    try:
        # Decrypt the token
        token_data = oauth_manager.decrypt_token(encrypted_token)
        
        # Check if token is expired
        if token_data.is_expired:
            # Try to refresh if we have a refresh token
            if token_data.refresh_token:
                try:
                    token_data = await oauth_manager.refresh_access_token(token_data.refresh_token)
                    logger.info("Token refreshed automatically during connection")
                except Exception as refresh_error:
                    logger.error(f"Failed to refresh token: {refresh_error}")
                    raise HTTPException(status_code=401, detail="Token expired and refresh failed")
            else:
                raise HTTPException(status_code=401, detail="Token expired and no refresh token available")
        
        # Create Enhanced WebSocket manager with OAuth token
        app_id = os.getenv("DERIV_APP_ID", "99188")
        enhanced_ws_manager = EnhancedDerivWebSocket(app_id=app_id, api_token=token_data.access_token)
        
        # Set up event handlers
        enhanced_ws_manager.set_tick_handler(handle_tick_data)
        enhanced_ws_manager.set_balance_handler(handle_balance_update)
        enhanced_ws_manager.set_trade_handler(handle_trade_result)
        enhanced_ws_manager.set_connection_handler(handle_connection_status)
        enhanced_ws_manager.set_error_handler(handle_websocket_error)
        
        # Connect
        logger.info(f"Connecting with OAuth token (scopes: {token_data.scope})")
        success = await enhanced_ws_manager.connect()
        
        if success:
            logger.info(f"Enhanced WebSocket connected with OAuth. State: {enhanced_ws_manager.state.value}")
            return {
                "status": "connecting",
                "message": "Enhanced connection initiated with OAuth token",
                "auth_method": "OAuth 2.0",
                "scopes": token_data.scope,
                "token_expires_at": (token_data.created_at + timedelta(seconds=token_data.expires_in)).isoformat(),
                "enhanced_features": {
                    "subscription_manager": True,
                    "message_queue": True,
                    "error_classification": True,
                    "auto_reconnection": True
                }
            }
        else:
            logger.error(f"OAuth connection failed. Final state: {enhanced_ws_manager.state.value}")
            raise HTTPException(status_code=500, detail="Failed to connect to Deriv API with OAuth token")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Enhanced WebSocket Endpoints ---

@app.get("/websocket/stats")
async def get_websocket_stats():
    """Get comprehensive WebSocket connection statistics"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        return {
            "status": "no_connection",
            "message": "Enhanced WebSocket manager not initialized"
        }
    
    try:
        stats = enhanced_ws_manager.get_connection_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websocket/subscribe/ticks")
async def subscribe_to_ticks(symbol: str):
    """Subscribe to tick data for a symbol using enhanced WebSocket"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager or enhanced_ws_manager.state != ConnectionState.AUTHENTICATED:
        raise HTTPException(status_code=400, detail="Enhanced WebSocket not connected or authenticated")
    
    try:
        subscription_id = await enhanced_ws_manager.subscribe_to_ticks(symbol)
        
        logger.info(f"Subscribed to ticks for {symbol}: {subscription_id}")
        
        return {
            "status": "success",
            "message": f"Subscribed to tick data for {symbol}",
            "subscription_id": subscription_id,
            "symbol": symbol,
            "subscription_type": "ticks"
        }
        
    except Exception as e:
        logger.error(f"Error subscribing to ticks for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websocket/subscribe/candles")
async def subscribe_to_candles(symbol: str, granularity: int = 60):
    """Subscribe to candle data for a symbol using enhanced WebSocket"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager or enhanced_ws_manager.state != ConnectionState.AUTHENTICATED:
        raise HTTPException(status_code=400, detail="Enhanced WebSocket not connected or authenticated")
    
    try:
        subscription_id = await enhanced_ws_manager.subscribe_to_candles(symbol, granularity)
        
        logger.info(f"Subscribed to candles for {symbol} ({granularity}s): {subscription_id}")
        
        return {
            "status": "success",
            "message": f"Subscribed to candle data for {symbol}",
            "subscription_id": subscription_id,
            "symbol": symbol,
            "subscription_type": "candles",
            "granularity": granularity
        }
        
    except Exception as e:
        logger.error(f"Error subscribing to candles for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/websocket/subscription/{subscription_id}")
async def unsubscribe_from_data(subscription_id: str):
    """Unsubscribe from a WebSocket subscription"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        raise HTTPException(status_code=400, detail="Enhanced WebSocket not connected")
    
    try:
        success = await enhanced_ws_manager.unsubscribe(subscription_id)
        
        if success:
            logger.info(f"Unsubscribed from {subscription_id}")
            return {
                "status": "success",
                "message": f"Successfully unsubscribed from {subscription_id}"
            }
        else:
            return {
                "status": "not_found",
                "message": f"Subscription {subscription_id} not found"
            }
        
    except Exception as e:
        logger.error(f"Error unsubscribing from {subscription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/websocket/subscriptions")
async def list_active_subscriptions():
    """List all active WebSocket subscriptions"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        return {
            "status": "no_connection",
            "subscriptions": []
        }
    
    try:
        subscriptions = enhanced_ws_manager.subscription_manager.get_active_subscriptions()
        
        return {
            "status": "success",
            "subscriptions": subscriptions,
            "total_subscriptions": len(subscriptions),
            "active_symbols": list(enhanced_ws_manager.subscription_manager.active_symbols)
        }
        
    except Exception as e:
        logger.error(f"Error listing subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websocket/reconnect")
async def force_websocket_reconnection():
    """Force WebSocket reconnection (useful for testing recovery)"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        raise HTTPException(status_code=400, detail="Enhanced WebSocket manager not initialized")
    
    try:
        logger.info("Manual WebSocket reconnection requested")
        success = await enhanced_ws_manager.reconnect()
        
        if success:
            return {
                "status": "success",
                "message": "WebSocket reconnection successful",
                "new_state": enhanced_ws_manager.state.value
            }
        else:
            return {
                "status": "failed",
                "message": "WebSocket reconnection failed",
                "current_state": enhanced_ws_manager.state.value
            }
        
    except Exception as e:
        logger.error(f"Error during manual reconnection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/websocket/health")
async def websocket_health_check():
    """Comprehensive WebSocket health check"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        return {
            "status": "unhealthy",
            "reason": "Enhanced WebSocket manager not initialized",
            "checks": {
                "manager_initialized": False,
                "connection_active": False,
                "authenticated": False,
                "subscriptions_active": False,
                "error_rate_acceptable": True
            }
        }
    
    try:
        stats = enhanced_ws_manager.get_connection_stats()
        
        # Health checks
        checks = {
            "manager_initialized": True,
            "connection_active": enhanced_ws_manager.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED],
            "authenticated": enhanced_ws_manager.state == ConnectionState.AUTHENTICATED,
            "subscriptions_active": len(stats['subscriptions']) > 0,
            "error_rate_acceptable": stats['stats']['errors_handled'] < 10,  # Less than 10 errors acceptable
            "uptime_good": stats['uptime'] is not None
        }
        
        # Overall health
        is_healthy = all([
            checks["manager_initialized"],
            checks["connection_active"],
            checks["authenticated"]
        ])
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "checks": checks,
            "stats": stats,
            "recommendations": _get_health_recommendations(checks, stats)
        }
        
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "checks": {
                "manager_initialized": enhanced_ws_manager is not None,
                "connection_active": False,
                "authenticated": False,
                "subscriptions_active": False,
                "error_rate_acceptable": False
            }
        }

def _get_health_recommendations(checks: Dict[str, bool], stats: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on checks"""
    recommendations = []
    
    if not checks["connection_active"]:
        recommendations.append("Connection is not active - consider reconnecting")
    
    if not checks["authenticated"]:
        recommendations.append("Not authenticated - check OAuth token validity")
    
    if not checks["subscriptions_active"]:
        recommendations.append("No active subscriptions - consider subscribing to market data")
    
    if not checks["error_rate_acceptable"]:
        recommendations.append("High error rate detected - investigate connection issues")
    
    if stats.get('stats', {}).get('reconnections', 0) > 5:
        recommendations.append("Multiple reconnections detected - check network stability")
    
    if not recommendations:
        recommendations.append("All systems operating normally")
    
    return recommendations

@app.get("/websocket/message-queue")
async def get_message_queue_status():
    """Get message queue statistics"""
    global enhanced_ws_manager
    
    if not enhanced_ws_manager:
        raise HTTPException(status_code=400, detail="Enhanced WebSocket manager not initialized")
    
    try:
        queue_stats = enhanced_ws_manager.message_queue.get_stats()
        
        return {
            "status": "success",
            "queue_stats": queue_stats,
            "is_healthy": queue_stats["queue_size"] < 100,  # Queue size under 100 is healthy
            "recommendations": [
                "Queue is healthy" if queue_stats["queue_size"] < 100 
                else "Queue size is high - may indicate processing delays"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting message queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# DERIV API ENDPOINTS - 16 FUNCIONALIDADES REAIS
# =====================================================

@app.post("/deriv/connect")
async def deriv_connect(request: DerivConnectRequest):
    """Conectar e autenticar com Deriv API usando token real"""
    global deriv_adapter

    try:
        # Verificar se adapter existe, se não, tentar criar
        if not deriv_adapter:
            logger.warning("Deriv adapter não inicializado, tentando criar...")
            try:
                from deriv_trading_adapter import DerivTradingAdapter
                deriv_adapter = DerivTradingAdapter()
                logger.info("Deriv adapter criado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao criar adapter: {e}")
                # Retornar resposta de erro mais amigável
                return {
                    "status": "error",
                    "message": "Sistema de trading não disponível no momento",
                    "error_code": "ADAPTER_UNAVAILABLE",
                    "suggestion": "Tente novamente em alguns momentos"
                }

        # Para desenvolvimento, simular conexão bem-sucedida
        if os.getenv("ENVIRONMENT", "development") == "development":
            # Simular conexão e autenticação bem-sucedidas
            connection_info = {
                "is_connected": True,
                "is_authenticated": True,
                "status": "authenticated",
                "loginid": "VRTC123456",
                "subscribed_symbols": ["R_75", "R_100"],
                "balance": 10000.0,
                "demo_mode": True,
                "currency": "USD"
            }

            logger.info("Modo desenvolvimento: simulando conexão bem-sucedida")

            return {
                "status": "success",
                "message": "Conectado com sucesso no modo DEMO",
                "connection_info": connection_info
            }

        # Código para produção com melhor tratamento de erro
        try:
            logger.info("Tentando conectar com Deriv API...")
            connected = await deriv_adapter.connect()
            if not connected:
                logger.error("Falha na conexão com Deriv")
                return {
                    "status": "error",
                    "message": "Não foi possível conectar com a Deriv",
                    "error_code": "CONNECTION_FAILED",
                    "suggestion": "Verifique sua conexão com a internet"
                }

            logger.info("Tentando autenticar com token...")
            authenticated = await deriv_adapter.authenticate(request.api_token)
            if not authenticated:
                logger.error("Falha na autenticação")
                return {
                    "status": "error",
                    "message": "Token inválido ou expirado",
                    "error_code": "AUTHENTICATION_FAILED",
                    "suggestion": "Verifique se o token está correto e válido"
                }

            connection_info = deriv_adapter.get_connection_info()

            logger.info(f"Conectado com sucesso - Login ID: {connection_info.get('loginid')}")

            return {
                "status": "success",
                "message": "Conectado e autenticado com sucesso na Deriv API",
                "connection_info": connection_info
            }

        except Exception as conn_error:
            logger.error(f"Erro específico na conexão: {conn_error}")
            return {
                "status": "error",
                "message": "Erro durante a conexão com Deriv",
                "error_code": "CONNECTION_ERROR",
                "error_details": str(conn_error),
                "suggestion": "Tente novamente ou verifique suas credenciais"
            }

    except Exception as e:
        logger.error(f"Erro geral ao conectar com Deriv API: {e}")
        return {
            "status": "error",
            "message": "Erro interno do sistema",
            "error_code": "INTERNAL_ERROR",
            "error_details": str(e),
            "suggestion": "Contate o suporte se o problema persistir"
        }

@app.post("/deriv/disconnect")
async def deriv_disconnect():
    """Desconectar da Deriv API"""
    global deriv_adapter
    
    try:
        if deriv_adapter:
            await deriv_adapter.disconnect()
        
        return {
            "status": "success",
            "message": "Desconectado da Deriv API"
        }
        
    except Exception as e:
        logger.error(f"Erro ao desconectar da Deriv API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deriv/balance")
async def deriv_get_balance():
    """Obter saldo atual da conta Deriv"""
    global deriv_adapter

    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            # Retornar resposta amigável em vez de HTTP 401
            return {
                "status": "error",
                "message": "Não conectado à Deriv API",
                "error_code": "NOT_AUTHENTICATED",
                "balance": 0.0,
                "currency": "USD",
                "suggestion": "Faça login primeiro usando OAuth ou token"
            }

        try:
            balance = await deriv_adapter.get_balance()

            return {
                "status": "success",
                "balance": balance,
                "currency": "USD",  # Assumindo USD por padrão
                "loginid": getattr(deriv_adapter.deriv_api, 'loginid', 'N/A')
            }
        except Exception as e:
            logger.warning(f"Erro ao obter saldo da API: {e}")
            return {
                "status": "error",
                "message": "Erro ao consultar saldo",
                "error_code": "BALANCE_ERROR",
                "balance": 0.0,
                "currency": "USD",
                "error_details": str(e)
            }

    except Exception as e:
        logger.error(f"Erro geral ao obter saldo: {e}")
        return {
            "status": "error",
            "message": "Erro interno",
            "error_code": "INTERNAL_ERROR",
            "balance": 0.0,
            "currency": "USD",
            "error_details": str(e)
        }

@app.get("/deriv/symbols")
async def deriv_get_symbols():
    """Obter símbolos disponíveis para trading"""
    global deriv_adapter

    try:
        # Se não há adapter ou não está autenticado, retornar símbolos padrão
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            # Símbolos padrão da Deriv para não quebrar o frontend
            default_symbols = [
                "R_10", "R_25", "R_50", "R_75", "R_100",
                "JD10", "JD25", "JD50", "JD75", "JD100",
                "BOOM1000", "CRASH1000", "RDBULL", "RDBEAR"
            ]

            logger.warning("Deriv adapter não autenticado, retornando símbolos padrão")

            return {
                "status": "success",
                "symbols": default_symbols,
                "count": len(default_symbols),
                "note": "Símbolos padrão - autenticação necessária para lista completa"
            }

        # Se conectado, tentar obter símbolos reais
        try:
            symbols = await deriv_adapter.get_available_symbols()

            return {
                "status": "success",
                "symbols": symbols,
                "count": len(symbols)
            }
        except Exception as e:
            logger.warning(f"Erro ao obter símbolos da API, usando padrão: {e}")
            # Fallback para símbolos padrão se a API falhar
            default_symbols = [
                "R_10", "R_25", "R_50", "R_75", "R_100",
                "JD10", "JD25", "JD50", "JD75", "JD100"
            ]

            return {
                "status": "success",
                "symbols": default_symbols,
                "count": len(default_symbols),
                "note": "Símbolos padrão - erro na API da Deriv"
            }

    except Exception as e:
        logger.error(f"Erro geral ao obter símbolos: {e}")
        # Último fallback
        return {
            "status": "success",
            "symbols": ["R_50", "R_75", "R_100"],
            "count": 3,
            "note": "Símbolos mínimos - erro no sistema"
        }

@app.get("/deriv/symbols/{symbol}/info")
async def deriv_get_symbol_info(symbol: str):
    """Obter informações detalhadas de um símbolo"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        symbol_info = await deriv_adapter.get_symbol_info(symbol)
        
        if not symbol_info:
            raise HTTPException(status_code=404, detail=f"Símbolo {symbol} não encontrado")
        
        return {
            "status": "success",
            "symbol": symbol,
            "info": symbol_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter informações do símbolo {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deriv/subscribe/ticks/{symbol}")
async def deriv_subscribe_ticks(symbol: str):
    """Subscrever a ticks de um símbolo"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        success = await deriv_adapter.subscribe_to_ticks(symbol)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Falha ao subscrever ao símbolo {symbol}")
        
        return {
            "status": "success",
            "message": f"Subscrito a ticks de {symbol}",
            "symbol": symbol
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao subscrever ao símbolo {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deriv/ticks/{symbol}/last")
async def deriv_get_last_tick(symbol: str):
    """Obter último tick de um símbolo"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        last_tick = deriv_adapter.get_last_tick(symbol)
        
        if not last_tick:
            raise HTTPException(status_code=404, detail=f"Nenhum tick disponível para {symbol}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "tick": {
                "price": last_tick.price,
                "timestamp": last_tick.timestamp,
                "epoch": last_tick.timestamp
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter último tick de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deriv/buy")
async def deriv_buy_contract(request: DerivTradeRequest):
    """Comprar contrato na Deriv API"""
    global deriv_adapter, capital_manager

    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")

        # Validar parâmetros
        if request.amount <= 0:
            raise HTTPException(status_code=400, detail="Valor do stake deve ser maior que zero")

        if request.duration <= 0:
            raise HTTPException(status_code=400, detail="Duração deve ser maior que zero")

        # Validar tipos de contrato suportados
        valid_contract_types = [
            "CALL", "PUT", "CALLE", "PUTE",
            "DIGITEVEN", "DIGITODD", "DIGITOVER", "DIGITUNDER",
            "DIGITMATCH", "DIGITDIFF", "ONETOUCH", "NOTOUCH",
            "RANGE", "UPORDOWN"
        ]

        if request.contract_type not in valid_contract_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de contrato '{request.contract_type}' não é válido. Tipos válidos: {', '.join(valid_contract_types)}"
            )

        # Verificar saldo antes de executar trade
        current_balance = await deriv_adapter.get_balance()
        if current_balance < request.amount:
            raise HTTPException(
                status_code=400,
                detail=f"Saldo insuficiente. Saldo atual: ${current_balance:.2f}, Valor solicitado: ${request.amount:.2f}"
            )

        # Usar capital manager se disponível
        trade_amount = request.amount
        risk_info = None

        if capital_manager:
            # Obter informações de risco
            risk_assessment = capital_manager.get_risk_assessment()
            risk_info = {
                "risk_level": risk_assessment["risk_level"],
                "risk_percentage": risk_assessment["risk_percentage"],
                "recommended_amount": capital_manager.get_next_trade_amount(),
                "is_martingale": capital_manager.is_in_loss_sequence
            }

            # Verificar se amount está muito acima do recomendado
            recommended = capital_manager.get_next_trade_amount()
            if trade_amount > recommended * 2:
                logger.warning(f"Amount ${trade_amount} muito acima do recomendado ${recommended}")

        # Executar trade
        result = await deriv_adapter.place_trade(
            contract_type=request.contract_type,
            symbol=request.symbol,
            amount=trade_amount,
            duration=request.duration,
            duration_unit=request.duration_unit
        )

        if not result['success']:
            error_message = result.get('error', 'Falha ao executar trade')
            # Log do erro para debug
            logger.error(f"Erro na compra: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)

        # Preparar resposta
        response_data = {
            "status": "success",
            "message": "Contrato comprado com sucesso",
            "contract": {
                "contract_id": result.get('contract_id'),
                "buy_price": result.get('buy_price'),
                "payout": result.get('payout'),
                "longcode": result.get('longcode'),
                "symbol": request.symbol,
                "contract_type": request.contract_type,
                "duration": f"{request.duration}{request.duration_unit}",
                "stake_amount": trade_amount
            },
            "balance_after": current_balance - trade_amount,
            "timestamp": time.time()
        }

        # Adicionar informações de risco se disponível
        if risk_info:
            response_data["risk_info"] = risk_info

        # Registrar trade no capital manager se disponível
        if capital_manager and result.get('contract_id'):
            # Nota: O resultado final será registrado quando o contrato finalizar
            logger.info(f"Trade registrado - Stake: ${trade_amount}, Risk Level: {risk_info.get('risk_level', 'N/A')}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao comprar contrato: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deriv/sell")
async def deriv_sell_contract(request: DerivSellRequest):
    """Vender contrato na Deriv API"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        # Executar venda
        result = await deriv_adapter.close_position(
            contract_id=request.contract_id,
            price=request.price
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Falha ao vender contrato'))
        
        return {
            "status": "success",
            "message": "Contrato vendido com sucesso",
            "sale": {
                "sold_for": result.get('sold_for'),
                "transaction_id": result.get('transaction_id')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao vender contrato: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deriv/portfolio")
async def deriv_get_portfolio():
    """Obter portfólio de contratos abertos"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        portfolio = await deriv_adapter.get_portfolio()
        
        return {
            "status": "success",
            "contracts": portfolio,
            "count": len(portfolio)
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter portfólio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deriv/history")
async def deriv_get_history(limit: int = 50):
    """Obter histórico de trades"""
    global deriv_adapter
    
    try:
        if not deriv_adapter or not deriv_adapter.is_authenticated:
            raise HTTPException(status_code=401, detail="Não conectado à Deriv API")
        
        history = await deriv_adapter.get_trade_history(limit=limit)
        
        return {
            "status": "success",
            "transactions": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deriv/health")
async def deriv_health_check():
    """Verificação de saúde da conexão com Deriv"""
    global deriv_adapter
    
    try:
        if not deriv_adapter:
            return {
                "status": "error",
                "message": "Deriv adapter não inicializado",
                "is_healthy": False
            }
        
        health_info = await deriv_adapter.health_check()
        
        return {
            "status": "success",
            "health": health_info,
            "is_healthy": health_info.get('is_connected', False) and health_info.get('is_authenticated', False)
        }
        
    except Exception as e:
        logger.error(f"Erro no health check da Deriv: {e}")
        return {
            "status": "error",
            "message": str(e),
            "is_healthy": False
        }

@app.get("/deriv/status")
async def deriv_get_status():
    """Obter status completo da conexão Deriv"""
    global deriv_adapter

    try:
        if not deriv_adapter:
            return {
                "status": "error",
                "message": "Deriv adapter não inicializado",
                "connection_info": {}
            }

        connection_info = deriv_adapter.get_connection_info()

        # Para desenvolvimento local, simular conexão funcional
        if os.getenv("ENVIRONMENT", "development") == "development":
            connection_info = {
                "is_connected": True,
                "is_authenticated": False,
                "status": "ready_for_auth",
                "loginid": None,
                "subscribed_symbols": [],
                "balance": 0.0,
                "demo_mode": True,
                "oauth_enabled": True,
                "oauth_url": f"https://oauth.deriv.com/oauth2/authorize?app_id=99188"
            }

        return {
            "status": "success",
            "connection_info": connection_info,
            "api_status": connection_info.get("status", "connected"),
            "subscribed_symbols": list(deriv_adapter.subscribed_symbols) if deriv_adapter.subscribed_symbols else [],
            "oauth_info": {
                "enabled": True,
                "auth_url": f"https://oauth.deriv.com/oauth2/authorize?app_id=99188",
                "redirect_uri": "https://botderiv.roilabs.com.br/auth"
            }
        }

    except Exception as e:
        logger.error(f"Erro ao obter status da Deriv: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================
# CONTRACT PROPOSALS ENGINE ENDPOINTS
# ==============================================

@app.post("/deriv/proposal")
async def get_contract_proposal(request: ProposalRequest):
    """
    Obter cotação para um contrato (proposal)
    Inclui validação de barriers, cache e cálculo em tempo real
    """
    global proposals_engine, deriv_adapter

    try:
        if not proposals_engine:
            # Inicializar engine se não existir
            if deriv_adapter and deriv_adapter.deriv_api:
                proposals_engine = get_proposals_engine(deriv_adapter.deriv_api)
                await initialize_proposals_engine(deriv_adapter.deriv_api)
            else:
                return {
                    "status": "error",
                    "message": "Sistema de proposals não disponível. Conecte-se primeiro.",
                    "error_code": "PROPOSALS_ENGINE_NOT_INITIALIZED"
                }

        # Converter request para ProposalRequest do engine
        from contract_proposals_engine import ProposalRequest as EngineProposalRequest

        engine_request = EngineProposalRequest(
            contract_type=request.contract_type,
            symbol=request.symbol,
            amount=request.amount,
            duration=request.duration,
            duration_unit=request.duration_unit,
            barrier=request.barrier,
            basis=request.basis,
            currency=request.currency
        )

        # Obter proposal
        proposal_response = await proposals_engine.get_proposal(engine_request)

        return {
            "status": "success",
            "proposal": {
                "id": proposal_response.id,
                "ask_price": proposal_response.ask_price,
                "payout": proposal_response.payout,
                "spot": proposal_response.spot,
                "barrier": proposal_response.barrier,
                "contract_type": proposal_response.contract_type,
                "symbol": proposal_response.symbol,
                "display_value": proposal_response.display_value,
                "timestamp": proposal_response.timestamp,
                "valid_until": proposal_response.valid_until
            },
            "request_params": {
                "contract_type": request.contract_type,
                "symbol": request.symbol,
                "amount": request.amount,
                "duration": request.duration,
                "duration_unit": request.duration_unit,
                "barrier": request.barrier
            }
        }

    except ValueError as e:
        # Erro de validação
        return {
            "status": "error",
            "message": str(e),
            "error_code": "VALIDATION_ERROR",
            "error_type": "validation"
        }
    except Exception as e:
        logger.error(f"Erro ao obter proposal: {e}")
        return {
            "status": "error",
            "message": str(e),
            "error_code": "PROPOSAL_ERROR",
            "error_type": "api_error"
        }

@app.post("/deriv/proposal/realtime")
async def get_realtime_proposal(request: ProposalRequest):
    """
    Obter cotação em tempo real (força nova requisição, sem cache)
    """
    global proposals_engine, deriv_adapter

    try:
        if not proposals_engine:
            return {
                "status": "error",
                "message": "Sistema de proposals não disponível. Conecte-se primeiro.",
                "error_code": "PROPOSALS_ENGINE_NOT_INITIALIZED"
            }

        # Converter request
        from contract_proposals_engine import ProposalRequest as EngineProposalRequest

        engine_request = EngineProposalRequest(
            contract_type=request.contract_type,
            symbol=request.symbol,
            amount=request.amount,
            duration=request.duration,
            duration_unit=request.duration_unit,
            barrier=request.barrier,
            basis=request.basis,
            currency=request.currency
        )

        # Obter proposal em tempo real
        proposal_response = await proposals_engine.get_realtime_proposal(engine_request)

        return {
            "status": "success",
            "proposal": {
                "id": proposal_response.id,
                "ask_price": proposal_response.ask_price,
                "payout": proposal_response.payout,
                "spot": proposal_response.spot,
                "barrier": proposal_response.barrier,
                "contract_type": proposal_response.contract_type,
                "symbol": proposal_response.symbol,
                "display_value": proposal_response.display_value,
                "timestamp": proposal_response.timestamp,
                "valid_until": proposal_response.valid_until
            },
            "realtime": True,
            "cache_bypassed": True
        }

    except Exception as e:
        logger.error(f"Erro ao obter proposal em tempo real: {e}")
        return {
            "status": "error",
            "message": str(e),
            "error_code": "REALTIME_PROPOSAL_ERROR"
        }

@app.post("/deriv/proposals/batch")
async def get_batch_proposals(request: BatchProposalRequest):
    """
    Obter múltiplas cotações de forma otimizada
    """
    global proposals_engine

    try:
        if not proposals_engine:
            return {
                "status": "error",
                "message": "Sistema de proposals não disponível",
                "error_code": "PROPOSALS_ENGINE_NOT_INITIALIZED"
            }

        # Converter requests
        from contract_proposals_engine import ProposalRequest as EngineProposalRequest

        engine_requests = []
        for req in request.proposals:
            engine_request = EngineProposalRequest(
                contract_type=req.contract_type,
                symbol=req.symbol,
                amount=req.amount,
                duration=req.duration,
                duration_unit=req.duration_unit,
                barrier=req.barrier,
                basis=req.basis,
                currency=req.currency
            )
            engine_requests.append(engine_request)

        # Obter proposals
        if request.realtime:
            # Usar realtime para todas
            tasks = [proposals_engine.get_realtime_proposal(req) for req in engine_requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Usar método batch otimizado
            responses = await proposals_engine.get_multiple_proposals(engine_requests)

        # Processar respostas
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append({
                    "status": "error",
                    "message": str(response),
                    "request_index": i
                })
            else:
                results.append({
                    "status": "success",
                    "proposal": {
                        "id": response.id,
                        "ask_price": response.ask_price,
                        "payout": response.payout,
                        "spot": response.spot,
                        "barrier": response.barrier,
                        "contract_type": response.contract_type,
                        "symbol": response.symbol,
                        "display_value": response.display_value,
                        "timestamp": response.timestamp,
                        "valid_until": response.valid_until
                    },
                    "request_index": i
                })

        return {
            "status": "success",
            "proposals": results,
            "total_requests": len(request.proposals),
            "successful_requests": len([r for r in results if r["status"] == "success"]),
            "failed_requests": len([r for r in results if r["status"] == "error"]),
            "realtime_mode": request.realtime
        }

    except Exception as e:
        logger.error(f"Erro ao obter proposals em lote: {e}")
        return {
            "status": "error",
            "message": str(e),
            "error_code": "BATCH_PROPOSALS_ERROR"
        }

@app.get("/deriv/proposals/stats")
async def get_proposals_stats():
    """
    Obter estatísticas do sistema de proposals
    """
    global proposals_engine

    try:
        if not proposals_engine:
            return {
                "status": "error",
                "message": "Sistema de proposals não disponível",
                "stats": None
            }

        stats = proposals_engine.get_stats()

        return {
            "status": "success",
            "stats": stats,
            "engine_running": proposals_engine.running
        }

    except Exception as e:
        logger.error(f"Erro ao obter estatísticas de proposals: {e}")
        return {
            "status": "error",
            "message": str(e),
            "stats": None
        }

@app.post("/deriv/proposals/reset-stats")
async def reset_proposals_stats():
    """
    Resetar estatísticas do sistema de proposals
    """
    global proposals_engine

    try:
        if not proposals_engine:
            return {
                "status": "error",
                "message": "Sistema de proposals não disponível"
            }

        proposals_engine.reset_stats()

        return {
            "status": "success",
            "message": "Estatísticas resetadas com sucesso"
        }

    except Exception as e:
        logger.error(f"Erro ao resetar estatísticas: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

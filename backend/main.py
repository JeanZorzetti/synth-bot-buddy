from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
import asyncio
import time
from typing import Dict, Any

from websocket_manager import DerivWebSocketManager, ConnectionState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global WebSocket manager instance
ws_manager: DerivWebSocketManager = None
bot_status = {
    "is_running": False,
    "connection_status": "disconnected",
    "balance": 0.0,
    "last_tick": None,
    "session_pnl": 0.0,
    "trades_count": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global ws_manager
    
    # Startup
    logger.info("Iniciando aplicação Cérebro...")
    
    # Initialize WebSocket manager
    api_token = os.getenv("DERIV_API_TOKEN")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    
    ws_manager = DerivWebSocketManager(app_id=app_id, api_token=api_token)
    
    # Set up event handlers
    ws_manager.set_tick_handler(handle_tick_data)
    ws_manager.set_balance_handler(handle_balance_update)
    ws_manager.set_trade_handler(handle_trade_result)
    ws_manager.set_connection_handler(handle_connection_status)
    
    yield
    
    # Shutdown
    logger.info("Encerrando aplicação Cérebro...")
    if ws_manager:
        await ws_manager.disconnect()

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
    global bot_status
    bot_status["trades_count"] += 1
    logger.info(f"Trade executed: {trade_data}")

async def handle_connection_status(status: ConnectionState):
    """Handle connection status changes"""
    global bot_status
    bot_status["connection_status"] = status.value
    logger.info(f"Connection status: {status.value}")

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

@app.get("/status")
async def get_status():
    """Get current bot status"""
    return bot_status

@app.post("/connect")
async def connect_to_deriv():
    """Connect to Deriv WebSocket API"""
    global ws_manager
    
    if not ws_manager:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    
    if ws_manager.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
        return {"status": "already_connected", "connection_status": ws_manager.state.value}
    
    try:
        success = await ws_manager.connect()
        if success:
            return {"status": "connecting", "message": "Connection initiated"}
        else:
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
    global ws_manager, bot_status
    
    if not ws_manager:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    
    try:
        # Connect if not connected
        if ws_manager.state == ConnectionState.DISCONNECTED:
            success = await ws_manager.connect()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to connect to Deriv API")
        
        # Wait for authentication
        max_wait = 10  # seconds
        wait_time = 0
        while ws_manager.state != ConnectionState.AUTHENTICATED and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        if ws_manager.state != ConnectionState.AUTHENTICATED:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Deriv API")
        
        # Subscribe to tick data
        await ws_manager.subscribe_to_ticks("R_75")  # Volatility 75 Index
        
        bot_status["is_running"] = True
        logger.info("Bot iniciado com sucesso")
        
        return {"status": "running", "message": "Bot iniciado com sucesso"}
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_bot():
    """Stop the trading bot"""
    global bot_status
    
    bot_status["is_running"] = False
    logger.info("Bot parado")
    
    return {"status": "stopped", "message": "Bot parado com sucesso"}

@app.post("/buy")
async def buy_contract(contract_type: str, amount: float, duration: int = 5, symbol: str = "R_75"):
    """Buy a trading contract"""
    global ws_manager
    
    if not ws_manager or ws_manager.state != ConnectionState.AUTHENTICATED:
        raise HTTPException(status_code=401, detail="Not connected or authenticated")
    
    if not bot_status["is_running"]:
        raise HTTPException(status_code=400, detail="Bot is not running")
    
    try:
        success = await ws_manager.buy_contract(contract_type, amount, duration, symbol)
        if success:
            return {"status": "order_sent", "message": f"Order sent: {contract_type} {amount} on {symbol}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send buy order")
    except Exception as e:
        logger.error(f"Buy order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

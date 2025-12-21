"""
FastAPI application - Trading Bot Backend with comprehensive logging
"""
import asyncio
import os
import time
import logging
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Bot controller
from bot_controller import bc, CONFIG
from app_state import bot_controller, price_cache, update_stats

# Load routers
from routes.accounts_routes import router as accounts_router
from routes.bot_routes import router as bot_router
from routes.config_routes import router as config_router
from routes.dashboard_routes import router as dashboard_router
from routes.history_routes import router as history_router

# Setup logging
logger = logging.getLogger("MainApp")

# Load environment
load_dotenv()

# Configuration
APP_PORT = int(os.getenv("PORT", "8000"))
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"

logger.info("=" * 60)
logger.info("🚀 STARTING TRADING BOT BACKEND")
logger.info("=" * 60)
logger.info(f"📋 Configuration:")
logger.info(f"   🌐 Port: {APP_PORT}")
logger.info(f"   ⚡ Live Mode: {LIVE_MODE}")
logger.info(f"   🧪 Dry Run: {CONFIG.get('dryRun', True)}")
logger.info(f"   🕐 Timeframe: {CONFIG.get('timeframe', '5')}m")
logger.info("=" * 60)

# Initialize FastAPI
app = FastAPI(
    title="AI Trading Bot",
    description="High-probability crypto trading with AI coin selection",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger("WebSocketManager")
        self.logger.info("✅ WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.debug(f"🔌 New WebSocket connection (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.debug(f"🔌 WebSocket disconnected (remaining: {len(self.active_connections)})")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to send to WebSocket: {e}")
                disconnected.append(connection)
        
        # Cleanup disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
        
        if message.get("type") == "price" and len(self.active_connections) > 0:
            self.logger.debug(f"📡 Broadcasted to {len(self.active_connections)} clients")

ws_manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    logger.debug(f"🌐 Dashboard requested from {request.client.host}")
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/accounts", response_class=HTMLResponse)
async def accounts_page(request: Request):
    from services.accounts_service import get_accounts
    accounts = get_accounts()
    logger.debug(f"🌐 Accounts page requested - {len(accounts)} accounts")
    return templates.TemplateResponse("accounts.html", {"request": request, "accounts": accounts})

@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    from services.config_service import get_config
    config = get_config()
    logger.debug(f"🌐 Config page requested")
    return templates.TemplateResponse("config.html", {"request": request, "config": config})

# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔌 New WebSocket connection established")
    await ws_manager.connect(websocket)
    try:
        # Send initial data
        await websocket.send_json({
            "type": "init",
            "bot_status": bc.is_running(),
            "trades_today": bc.trades_today,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug("📤 Sent initial data to WebSocket client")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                logger.debug("🏓 WebSocket ping received")
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)
        logger.info("🔌 WebSocket connection closed")

@app.websocket("/ws/prices")
async def price_websocket(websocket: WebSocket):
    logger.info("📈 New price WebSocket connection")
    await websocket.accept()
    try:
        while True:
            # Send price updates
            price_data = {
                "type": "prices",
                "prices": price_cache,
                "count": len(price_cache),
                "timestamp": int(time.time()),
                "ai_coins": bc.get_coins()
            }
            await websocket.send_json(price_data)
            logger.debug(f"📤 Sent prices for {len(price_cache)} symbols")
            await asyncio.sleep(3)
    except Exception as e:
        logger.error(f"❌ Price WebSocket error: {e}")
    logger.info("📈 Price WebSocket connection closed")

# API endpoints
@app.get("/api/status")
async def status():
    """Get system status with detailed logging"""
    logger.info("📊 Status check requested")
    
    stats = bc.get_stats()
    uptime = stats.get("stats", {}).get("uptime_seconds", 0)
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    
    response = {
        "bot_running": bc.is_running(),
        "trades_today": bc.trades_today,
        "max_daily_trades": bc.max_daily_trades,
        "ai_coins": bc.get_coins(),
        "dry_run": CONFIG.get("dryRun", True),
        "uptime": f"{hours}h {minutes}m",
        "server_time": datetime.now().isoformat(),
        "version": "2.0.0",
        "performance": {
            "qualified_rate": bc._calculate_qualified_rate(),
            "win_rate": bc._calculate_win_rate(),
            "scans_per_minute": stats.get("stats", {}).get("scans_per_minute", 0)
        }
    }
    
    logger.info(f"📊 Status: Bot running={bc.is_running()}, Trades today={bc.trades_today}")
    return response
# Add these routes to your main.py:

@app.get("/api/groq/status")
async def groq_status():
    """Get Groq API status and usage"""
    try:
        insights = bc.get_market_insights()
        groq_stats = insights.get("groq_stats", {})
        
        return {
            "groq_configured": groq_stats.get("api_key_configured", False),
            "total_requests": groq_stats.get("total_requests", 0),
            "success_rate": groq_stats.get("success_rate", 0),
            "total_tokens": groq_stats.get("total_tokens", 0),
            "current_model": groq_stats.get("current_model", "unknown"),
            "market_sentiment": insights.get("market_analysis", {}).get("market_sentiment", "unknown"),
            "coin_count": len(bc.get_coins()),
            "last_update": bc.ai_selector.last_update
        }
    except Exception as e:
        logger.error(f"❌ Groq status error: {e}")
        return {"error": str(e)}

@app.post("/api/groq/update")
async def update_groq_coins():
    """Force update Groq coin selection"""
    logger.info("🔄 Manual Groq update requested via API")
    
    try:
        result = bc.force_ai_update()
        
        if result["success"]:
            logger.info(f"✅ Manual Groq update successful: {result['message']}")
            return {
                "success": True,
                "message": result["message"],
                "coins": result["coins"],
                "market_insights": result.get("market_insights", {}),
                "update_time": result.get("update_time", 0)
            }
        else:
            logger.error(f"❌ Manual Groq update failed: {result.get('error')}")
            raise HTTPException(500, result.get("message", "Groq update failed"))
            
    except Exception as e:
        logger.error(f"❌ Groq update API error: {e}", exc_info=True)
        raise HTTPException(500, f"Groq update failed: {str(e)}")

@app.get("/api/groq/insights")
async def get_groq_insights():
    """Get current market insights from Groq"""
    try:
        insights = bc.get_market_insights()
        
        return {
            "market_analysis": insights.get("market_analysis", {}),
            "groq_stats": insights.get("groq_stats", {}),
            "current_coins": bc.get_coins(),
            "last_update": bc.ai_selector.last_update,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Groq insights error: {e}")
        return {"error": str(e)}

@app.get("/api/ai/coins")
async def get_ai_coins():
    """Get current AI coin selection"""
    coins = bc.get_coins()
    logger.info(f"🤖 AI coins requested: {len(coins)} symbols")
    return {
        "coins": coins,
        "count": len(coins),
        "last_update": bc.ai_selector.last_update,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ai/update")
async def update_ai_coins():
    """Update AI coin selection manually"""
    logger.info("🔄 Manual AI coin update requested")
    try:
        start_time = time.time()
        coins = bc.ai_selector.update_from_llm()
        elapsed = time.time() - start_time
        
        logger.info(f"✅ AI coins updated in {elapsed:.2f}s: {coins}")
        
        return {
            "success": True,
            "coins": coins,
            "count": len(coins),
            "update_time": elapsed,
            "message": f"AI coins updated successfully with {len(coins)} symbols"
        }
    except Exception as e:
        logger.error(f"❌ AI update failed: {e}", exc_info=True)
        raise HTTPException(500, f"AI update failed: {str(e)}")

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    logger.debug("🏓 Ping received")
    return {
        "status": "alive", 
        "timestamp": datetime.now().isoformat(),
        "bot_running": bc.is_running(),
        "version": "2.0.0"
    }

@app.get("/api/debug")
async def debug_info():
    """Debug endpoint for system inspection"""
    logger.info("🔍 Debug info requested")
    
    try:
        # Get file stats
        import os
        from pathlib import Path
        
        data_dir = Path("app/data")
        log_dir = Path("logs")
        
        file_stats = {}
        if data_dir.exists():
            for file in data_dir.glob("*.json"):
                file_stats[file.name] = {
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                }
        
        # Get recent logs
        recent_logs = []
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_log, "r") as f:
                        recent_logs = f.readlines()[-10:]  # Last 10 lines
                except:
                    recent_logs = ["Unable to read log file"]
        
        response = {
            "system": {
                "python_version": os.sys.version,
                "platform": os.sys.platform,
                "working_directory": os.getcwd()
            },
            "files": file_stats,
            "recent_logs": recent_logs,
            "memory_usage": "N/A",  # Could add psutil if installed
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug("🔍 Debug info generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Debug endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"Debug error: {str(e)}")

# Include routers
app.include_router(accounts_router, prefix="/api/accounts", tags=["Accounts"])
app.include_router(bot_router, prefix="/api/bot", tags=["Bot"])
app.include_router(config_router, prefix="/api/config", tags=["Config"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(history_router, prefix="/api/history", tags=["History"])

logger.info("✅ All routers registered successfully")

# Background tasks
async def price_update_task():
    """Update prices for AI coins with logging"""
    import requests
    
    logger.info("📈 Starting price update task...")
    
    update_count = 0
    error_count = 0
    
    while True:
        try:
            symbols = bc.get_coins()
            if not symbols:
                logger.warning("⚠️ No symbols to update prices for")
                await asyncio.sleep(10)
                continue
            
            start_time = time.time()
            updated = 0
            
            for symbol in symbols:
                try:
                    response = requests.get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "spot", "symbol": symbol},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("retCode") == 0:
                            ticker = data.get("result", {}).get("list", [{}])[0]
                            price = float(ticker.get("lastPrice", 0))
                            old_price = price_cache.get(symbol)
                            price_cache[symbol] = price
                            updated += 1
                            
                            # Broadcast update
                            await ws_manager.broadcast({
                                "type": "price",
                                "symbol": symbol,
                                "price": price,
                                "old_price": old_price,
                                "change": ((price - old_price) / old_price * 100) if old_price else 0,
                                "timestamp": int(time.time())
                            })
                            
                            update_count += 1
                except requests.exceptions.Timeout:
                    logger.warning(f"⏰ Price timeout for {symbol}")
                    error_count += 1
                except Exception as e:
                    logger.error(f"❌ Price error for {symbol}: {e}")
                    error_count += 1
            
            elapsed = time.time() - start_time
            
            # Periodic logging
            if update_count % 100 == 0:
                logger.info(
                    f"📊 Price updates: {update_count} total, "
                    f"{error_count} errors, "
                    f"last batch: {updated}/{len(symbols)} in {elapsed:.2f}s"
                )
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Price update task error: {e}", exc_info=True)
            await asyncio.sleep(10)

async def stats_update_task():
    """Periodically update statistics"""
    logger.info("📈 Starting stats update task...")
    
    while True:
        try:
            update_stats()
            await asyncio.sleep(60)  # Update every minute
            
            # Log stats periodically
            if int(time.time()) % 300 < 5:  # Every 5 minutes
                logger.info(
                    f"📊 Current stats: ${trading_stats.get('total_profit', 0):.2f} profit, "
                    f"{trading_stats.get('win_rate', 0):.1f}% win rate, "
                    f"{trading_stats.get('active_trades', 0)} active trades"
                )
                
        except Exception as e:
            logger.error(f"❌ Stats update error: {e}")
            await asyncio.sleep(30)

@app.on_event("startup")
async def startup():
    """Start background tasks on startup with detailed logging"""
    logger.info("=" * 60)
    logger.info("🚀 APPLICATION STARTUP")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info("📋 ACTIVE CONFIGURATION:")
    for key, value in CONFIG.items():
        if key not in ["api_key", "api_secret", "secretKey"]:  # Don't log secrets
            logger.info(f"   {key}: {value}")
    
    # Check environment
    required_vars = ["PORT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
    else:
        logger.info("✅ All required environment variables present")
    
    # Start background tasks
    try:
        asyncio.create_task(price_update_task())
        logger.info("✅ Price update task started")
        
        asyncio.create_task(stats_update_task())
        logger.info("✅ Stats update task started")
        
        # Start bot if not in dry run
        if not CONFIG.get("dryRun", True) and LIVE_MODE:
            bc.start()
            logger.info("✅ Bot started in LIVE mode")
        else:
            logger.info("✅ Bot ready (Dry Run mode)")
        
        logger.info("=" * 60)
        logger.info(f"🌐 Server running on http://0.0.0.0:{APP_PORT}")
        logger.info(f"📚 API Docs: http://0.0.0.0:{APP_PORT}/api/docs")
        logger.info(f"📖 ReDoc: http://0.0.0.0:{APP_PORT}/api/redoc")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown():
    """Clean shutdown with logging"""
    logger.info("=" * 60)
    logger.info("🛑 APPLICATION SHUTDOWN")
    logger.info("=" * 60)
    
    try:
        bc.stop()
        logger.info("✅ Bot stopped")
        
        # Close WebSocket connections
        for connection in list(ws_manager.active_connections):
            ws_manager.disconnect(connection)
        logger.info(f"✅ Closed {len(ws_manager.active_connections)} WebSocket connections")
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)
    
    logger.info("=" * 60)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"❌ HTTP Error {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled Exception: {exc} - Path: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if CONFIG.get("debug", False) else "Contact administrator",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    config = {
        "host": "0.0.0.0",
        "port": APP_PORT,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    logger.info(f"🚀 Starting uvicorn server with config: {config}")
    
    try:
        uvicorn.run(app, **config)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        raise

"""
Bot service with comprehensive logging
"""
import logging
from datetime import datetime
from bot_controller import bc

logger = logging.getLogger("BotService")

def get_bot_status() -> Dict:
    """Get bot status with detailed logging"""
    logger.debug("📊 Getting bot status")
    
    try:
        stats = bc.get_stats()
        uptime = stats.get("stats", {}).get("uptime_seconds", 0)
        
        # Format uptime
        days = int(uptime // 86400)
        hours = int((uptime % 86400) // 3600)
        minutes = int((uptime % 3600) // 60)
        
        status = {
            "running": bc.is_running(),
            "trades_today": bc.trades_today,
            "max_daily_trades": bc.max_daily_trades,
            "ai_coins": bc.get_coins(),
            "stats": {
                "uptime": f"{days}d {hours}h {minutes}m",
                "uptime_seconds": uptime,
                "total_scans": stats.get("stats", {}).get("total_scans", 0),
                "qualified_signals": stats.get("stats", {}).get("qualified_signals", 0),
                "successful_trades": stats.get("stats", {}).get("successful_trades", 0),
                "failed_trades": stats.get("stats", {}).get("failed_trades", 0),
                "total_pnl": stats.get("stats", {}).get("total_pnl", 0)
            },
            "performance": {
                "qualification_rate": bc._calculate_qualified_rate(),
                "win_rate": bc._calculate_win_rate(),
                "scans_per_minute": stats.get("stats", {}).get("scans_per_minute", 0)
            },
            "last_update": datetime.now().isoformat()
        }
        
        logger.info(
            f"📊 Bot Status: Running={bc.is_running()}, "
            f"Trades today={bc.trades_today}/{bc.max_daily_trades}, "
            f"Uptime={status['stats']['uptime']}"
        )
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to get bot status: {e}", exc_info=True)
        return {
            "running": False,
            "error": str(e),
            "last_update": datetime.now().isoformat()
        }

def start_bot() -> Dict:
    """Start bot with logging"""
    logger.info("🚀 Starting bot...")
    
    if bc.is_running():
        logger.warning("⚠️ Bot is already running")
        return {
            "success": False, 
            "message": "Bot is already running",
            "running": True
        }
    
    try:
        bc.start()
        logger.info("✅ Bot started successfully")
        
        return {
            "success": True, 
            "message": "Bot started successfully",
            "running": True,
            "start_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to start bot: {str(e)}",
            "error": str(e),
            "running": False
        }

def stop_bot() -> Dict:
    """Stop bot with logging"""
    logger.info("🛑 Stopping bot...")
    
    if not bc.is_running():
        logger.warning("⚠️ Bot is already stopped")
        return {
            "success": False, 
            "message": "Bot is already stopped",
            "running": False
        }
    
    try:
        bc.stop()
        logger.info("✅ Bot stopped successfully")
        
        return {
            "success": True, 
            "message": "Bot stopped successfully",
            "running": False,
            "stop_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to stop bot: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to stop bot: {str(e)}",
            "error": str(e),
            "running": True  # Assume still running if stop failed
        }

def restart_bot() -> Dict:
    """Restart bot with logging"""
    logger.info("🔄 Restarting bot...")
    
    try:
        # Stop if running
        if bc.is_running():
            stop_result = stop_bot()
            if not stop_result["success"]:
                logger.error("❌ Failed to stop bot during restart")
                return {
                    "success": False,
                    "message": "Failed to stop bot during restart",
                    "error": stop_result.get("error")
                }
        
        # Start bot
        start_result = start_bot()
        
        if start_result["success"]:
            logger.info("✅ Bot restarted successfully")
            return {
                "success": True,
                "message": "Bot restarted successfully",
                "running": True,
                "restart_time": datetime.now().isoformat()
            }
        else:
            logger.error("❌ Failed to start bot during restart")
            return {
                "success": False,
                "message": "Failed to start bot during restart",
                "error": start_result.get("error")
            }
            
    except Exception as e:
        logger.error(f"❌ Bot restart failed: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Bot restart failed: {str(e)}",
            "error": str(e)
        }

def get_bot_performance(days: int = 7) -> Dict:
    """Get bot performance metrics"""
    logger.info(f"📈 Getting bot performance for last {days} days")
    
    try:
        # This would query the performance logs
        # For now, return basic metrics
        stats = bc.get_stats()
        
        performance = {
            "period_days": days,
            "total_trades": stats.get("stats", {}).get("successful_trades", 0),
            "win_rate": bc._calculate_win_rate(),
            "total_pnl": stats.get("stats", {}).get("total_pnl", 0),
            "avg_trade_size": stats.get("stats", {}).get("total_volume_traded", 0) / 
                             max(stats.get("stats", {}).get("successful_trades", 1), 1),
            "qualification_rate": bc._calculate_qualified_rate(),
            "scans_per_day": stats.get("stats", {}).get("total_scans", 0) / 
                           max(stats.get("stats", {}).get("uptime_seconds", 1) / 86400, 1),
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(
            f"📈 Performance: {performance['total_trades']} trades, "
            f"{performance['win_rate']:.1f}% win rate, "
            f"${performance['total_pnl']:.2f} P&L"
        )
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Failed to get performance: {e}")
        return {
            "error": str(e),
            "period_days": days,
            "last_updated": datetime.now().isoformat()
        }

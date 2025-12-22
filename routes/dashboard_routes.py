"""
Dashboard API routes with enhanced analytics
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import Optional

from services.dashboard_service import (
    get_dashboard_data, 
    get_recent_trades, 
    get_performance,
    get_daily_performance,
    get_trade_analytics
)

router = APIRouter()
logger = logging.getLogger("DashboardRouter")

@router.get("/")
async def dashboard():
    """Get comprehensive dashboard data"""
    try:
        data = get_dashboard_data()
        return {
            "success": True,
            "data": data,
            "timestamp": data.get("last_update", "")
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dashboard data: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load dashboard data: {str(e)}")

@router.get("/recent-trades")
async def recent_trades(limit: int = 10):
    """Get recent trades"""
    try:
        trades = get_recent_trades(limit)
        return {
            "success": True,
            "trades": trades,
            "count": len(trades),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"❌ Failed to get recent trades: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load recent trades: {str(e)}")

@router.get("/performance")
async def performance():
    """Get performance metrics for last 7 days"""
    try:
        metrics = get_performance()
        return {
            "success": True,
            "metrics": metrics,
            "period": "last_7_days"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get performance metrics: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load performance metrics: {str(e)}")

@router.get("/daily-performance")
async def daily_performance(days: Optional[int] = 30):
    """Get daily performance metrics"""
    try:
        if days and days > 365:
            days = 365  # Limit to 1 year for performance reasons
        
        daily_data = get_daily_performance(days)
        
        return {
            "success": True,
            "daily_performance": daily_data,
            "days": days,
            "count": len(daily_data)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get daily performance: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load daily performance: {str(e)}")

@router.get("/analytics")
async def analytics():
    """Get detailed trade analytics"""
    try:
        analytics_data = get_trade_analytics()
        
        return {
            "success": True,
            "analytics": analytics_data,
            "summary": {
                "total_trades": analytics_data.get("total_trades", 0),
                "closed_trades": analytics_data.get("closed_trades", 0),
                "open_trades": analytics_data.get("open_trades", 0),
                "total_profit": analytics_data.get("total_profit", 0)
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get trade analytics: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load trade analytics: {str(e)}")

@router.get("/quick-stats")
async def quick_stats():
    """Get quick statistics summary"""
    try:
        # Get dashboard data for quick stats
        dashboard_data = get_dashboard_data()
        
        # Get performance metrics
        performance_data = get_performance()
        
        # Get account stats
        from services.accounts_service import get_account_stats
        account_stats = get_account_stats()
        
        return {
            "success": True,
            "quick_stats": {
                "bot_status": dashboard_data.get("bot_status", "stopped"),
                "trades_today": dashboard_data.get("trades_today", 0),
                "daily_limit": dashboard_data.get("daily_limit", 10),
                "open_trades": dashboard_data.get("open_trades", 0),
                "total_profit": dashboard_data.get("total_profit", 0),
                "total_balance": dashboard_data.get("total_balance", 0),
                "active_accounts": dashboard_data.get("active_accounts", 0),
                "win_rate": dashboard_data.get("win_rate", 0),
                "qualification_rate": dashboard_data.get("qualification_rate", 0),
                "market_sentiment": dashboard_data.get("market_sentiment", "neutral"),
                "profit_probability": dashboard_data.get("profit_probability", 0)
            },
            "performance": {
                "win_rate": performance_data.get("win_rate", 0),
                "total_profit": performance_data.get("total_profit", 0),
                "avg_profit": performance_data.get("avg_profit", 0),
                "success_rate": performance_data.get("success_rate", 0)
            },
            "accounts": {
                "total": account_stats.get("total_accounts", 0),
                "validated": account_stats.get("validated_accounts", 0),
                "validation_rate": account_stats.get("validation_rate", 0)
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get quick stats: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load quick stats: {str(e)}")

@router.get("/health")
async def dashboard_health():
    """Health check for dashboard service"""
    try:
        # Test all dashboard functions
        dashboard_data = get_dashboard_data()
        recent_trades_data = get_recent_trades(5)
        performance_data = get_performance()
        
        return {
            "status": "healthy",
            "checks": {
                "dashboard_data": "ok" if dashboard_data else "failed",
                "recent_trades": "ok" if recent_trades_data else "failed",
                "performance_metrics": "ok" if performance_data else "failed"
            },
            "metrics": {
                "trades_today": dashboard_data.get("trades_today", 0),
                "total_balance": dashboard_data.get("total_balance", 0),
                "bot_running": dashboard_data.get("bot_status") == "running"
            }
        }
    except Exception as e:
        logger.error(f"❌ Dashboard health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/market-insights")
async def market_insights():
    """Get market insights from Groq AI"""
    try:
        from bot_controller import bc
        
        # Get market insights from bot controller
        insights = bc.get_market_insights()
        
        # Get current AI coins
        ai_coins = bc.get_coins()
        
        return {
            "success": True,
            "market_insights": {
                "market_sentiment": insights.get("market_analysis", {}).get("market_sentiment", "unknown"),
                "profit_probability": insights.get("market_analysis", {}).get("profit_probability_today", 0),
                "price_range": insights.get("market_analysis", {}).get("price_range", "All under $1.00"),
                "risk_level": insights.get("market_analysis", {}).get("risk_level", "medium"),
                "key_insights": insights.get("market_analysis", {}).get("key_insights", []),
                "top_opportunities": insights.get("market_analysis", {}).get("top_opportunities", []),
                "recommended_allocation": insights.get("market_analysis", {}).get("recommended_allocation", 0.5)
            },
            "ai_selection": {
                "coins": ai_coins,
                "count": len(ai_coins),
                "last_update": bc.ai_selector.last_update,
                "all_under_1_dollar": True  # From your bot configuration
            },
            "groq_stats": insights.get("groq_stats", {}),
            "timestamp": insights.get("last_update", "")
        }
    except Exception as e:
        logger.error(f"❌ Failed to get market insights: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load market insights: {str(e)}")

@router.get("/bot-status")
async def bot_status():
    """Get detailed bot status"""
    try:
        from bot_controller import bc
        
        bot_stats = bc.get_stats()
        
        return {
            "success": True,
            "bot_status": {
                "running": bc.is_running(),
                "trades_today": bc.trades_today,
                "max_daily_trades": bc.max_daily_trades,
                "daily_reset_in": bot_stats.get("daily_reset_in", 0),
                "uptime": bot_stats.get("uptime", "0h 0m"),
                "dry_run": bot_stats.get("dry_run", True),
                "ai_coins_count": len(bc.get_coins()),
                "ai_probability": bot_stats.get("ai_probability", 80),
                "price_range": "All under $1.00"
            },
            "performance": bot_stats.get("stats", {}),
            "rate_limiter": bot_stats.get("stats", {}).get("rate_limiter_stats", {}),
            "timestamp": bot_stats.get("stats", {}).get("uptime_seconds", 0)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get bot status: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load bot status: {str(e)}")

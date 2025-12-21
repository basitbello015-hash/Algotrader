"""
Global application state with logging
"""
import logging
from bot_controller import bc

logger = logging.getLogger("AppState")

# Bot controller instance
bot_controller = bc

# Price cache for dashboard
price_cache = {}

# WebSocket connections
active_connections = []

# Trading statistics
trading_stats = {
    "total_profit": 0.0,
    "win_rate": 0.0,
    "active_trades": 0,
    "last_update": None
}

def update_stats():
    """Update trading statistics"""
    try:
        trades = bc._read_trades()
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in trades if t.get("status") in ["CLOSED", "STOPPED"]]
        
        # Calculate profit
        total_profit = 0.0
        for trade in closed_trades:
            if trade.get("exit_price") and trade.get("entry_price"):
                profit = (trade["exit_price"] - trade["entry_price"]) * trade.get("size", 0)
                total_profit += profit
        
        # Calculate win rate
        winning = 0
        for trade in closed_trades:
            if trade.get("pnl", 0) > 0:
                winning += 1
        
        win_rate = (winning / len(closed_trades)) * 100 if closed_trades else 0
        
        trading_stats.update({
            "total_profit": round(total_profit, 2),
            "win_rate": round(win_rate, 1),
            "active_trades": len(open_trades),
            "last_update": datetime.now().isoformat()
        })
        
        logger.debug(f"📊 Stats updated: ${total_profit:.2f} profit, {win_rate:.1f}% win rate")
        
    except Exception as e:
        logger.error(f"❌ Failed to update stats: {e}")

logger.info("✅ AppState initialized")

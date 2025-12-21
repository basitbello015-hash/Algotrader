import json
from datetime import datetime, timedelta
from typing import Dict, List
from bot_controller import bc, ACCOUNTS_FILE, TRADES_FILE

def get_dashboard_data() -> Dict:
    """Get dashboard data"""
    # Read trades
    trades = []
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except:
        pass
    
    # Read accounts
    accounts = []
    try:
        with open(ACCOUNTS_FILE, "r") as f:
            accounts = json.load(f)
    except:
        pass
    
    # Calculate metrics
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") in ["CLOSED", "STOPPED"]]
    
    # Today's trades
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("entry_time", "").startswith(today)]
    
    # Calculate profit
    total_profit = 0.0
    for trade in closed_trades:
        if trade.get("exit_price") and trade.get("entry_price"):
            profit = (trade["exit_price"] - trade["entry_price"]) * trade.get("size", 0)
            total_profit += profit
    
    # Account balances
    total_balance = sum(a.get("balance", 0) for a in accounts)
    active_accounts = len([a for a in accounts if a.get("validated", False)])
    
    return {
        "total_profit": round(total_profit, 2),
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "trades_today": len(today_trades),
        "total_balance": round(total_balance, 2),
        "active_accounts": active_accounts,
        "bot_status": "running" if bc.is_running() else "stopped",
        "daily_trades": bc.trades_today,
        "daily_limit": bc.max_daily_trades,
        "ai_coins": bc.get_coins(),
        "qualification_rate": round((bc.stats.get("qualified_signals", 0) / max(bc.stats.get("total_scans", 1), 1)) * 100, 1),
        "last_update": datetime.now().isoformat()
    }

def get_recent_trades(limit: int = 10) -> List[Dict]:
    """Get recent trades"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
        
        # Sort by entry time (newest first)
        trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
        return trades[:limit]
    except:
        return []

def get_performance() -> Dict:
    """Get performance metrics"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except:
        trades = []
    
    # Last 7 days
    week_ago = datetime.now() - timedelta(days=7)
    week_trades = [t for t in trades if datetime.fromisoformat(t.get("entry_time", "2000-01-01")) > week_ago]
    
    if not week_trades:
        return {"total_trades": 0, "win_rate": 0, "avg_profit": 0}
    
    winning = 0
    total_profit = 0.0
    
    for trade in week_trades:
        if trade.get("status") in ["CLOSED", "STOPPED"]:
            if trade.get("exit_price") and trade.get("entry_price"):
                profit = (trade["exit_price"] - trade["entry_price"]) * trade.get("size", 0)
                total_profit += profit
                if profit > 0:
                    winning += 1
    
    win_rate = (winning / len(week_trades)) * 100 if week_trades else 0
    
    return {
        "total_trades": len(week_trades),
        "winning_trades": winning,
        "win_rate": round(win_rate, 1),
        "total_profit": round(total_profit, 2),
        "avg_profit": round(total_profit / len(week_trades), 2) if week_trades else 0
    }

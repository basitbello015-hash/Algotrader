import json
from datetime import datetime, timedelta
from typing import Dict, List
from bot_controller import bc, TRADES_FILE
from services.accounts_service import get_accounts, get_account_stats

def get_dashboard_data() -> Dict:
    """Get dashboard data with environment variable accounts"""
    # Read trades
    trades = []
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except FileNotFoundError:
        # Create empty file if doesn't exist
        with open(TRADES_FILE, "w") as f:
            json.dump([], f)
        trades = []
    except Exception as e:
        import logging
        logger = logging.getLogger("DashboardService")
        logger.error(f"❌ Failed to load trades: {e}")
        trades = []
    
    # Get accounts from environment variables
    accounts = get_accounts()
    
    # Calculate metrics
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") in ["CLOSED", "STOPPED"]]
    
    # Today's trades
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("entry_time", "").startswith(today)]
    
    # Calculate profit
    total_profit = 0.0
    winning_trades = 0
    
    for trade in closed_trades:
        if trade.get("exit_price") and trade.get("entry_price"):
            profit = (trade["exit_price"] - trade["entry_price"]) * trade.get("size", 0)
            total_profit += profit
            if profit > 0:
                winning_trades += 1
    
    # Win rate calculation
    win_rate = 0
    if closed_trades:
        win_rate = (winning_trades / len(closed_trades)) * 100
    
    # Account statistics from environment variables
    account_stats = get_account_stats()
    
    # Bot performance stats
    bot_stats = bc.get_stats()
    
    # Get market insights
    insights = bc.get_market_insights()
    market_sentiment = insights.get("market_analysis", {}).get("market_sentiment", "neutral")
    profit_probability = insights.get("market_analysis", {}).get("profit_probability_today", 0)
    
    # Calculate qualification rate safely
    qualified_signals = bc.stats.get("qualified_signals", 0)
    total_scans = max(bc.stats.get("total_scans", 1), 1)  # Avoid division by zero
    qualification_rate = (qualified_signals / total_scans) * 100
    
    return {
        "total_profit": round(total_profit, 2),
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "trades_today": len(today_trades),
        "total_balance": round(account_stats.get("total_balance", 0), 2),
        "active_accounts": account_stats.get("validated_accounts", 0),
        "total_accounts": account_stats.get("total_accounts", 0),
        "bot_status": "running" if bc.is_running() else "stopped",
        "daily_trades": bc.trades_today,
        "daily_limit": bc.max_daily_trades,
        "ai_coins": bc.get_coins(),
        "ai_coin_count": len(bc.get_coins()),
        "qualification_rate": round(qualification_rate, 1),
        "win_rate": round(win_rate, 1),
        "market_sentiment": market_sentiment,
        "profit_probability": profit_probability,
        "price_range": "All under $1.00",
        "last_update": datetime.now().isoformat(),
        "configuration_method": account_stats.get("configuration_method", "environment_variables"),
        "uptime": bot_stats.get("uptime", "0h 0m"),
        "successful_trades": bc.stats.get("successful_trades", 0),
        "failed_trades": bc.stats.get("failed_trades", 0),
        "total_volume": round(bc.stats.get("total_volume_traded", 0), 2)
    }

def get_recent_trades(limit: int = 10) -> List[Dict]:
    """Get recent trades"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
        
        # Sort by entry time (newest first)
        trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
        
        # Add profit calculation to each trade
        for trade in trades[:limit]:
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            size = trade.get("size", 0)
            
            if entry_price > 0 and exit_price > 0:
                profit = (exit_price - entry_price) * size
                profit_percent = ((exit_price - entry_price) / entry_price) * 100
                trade["profit"] = round(profit, 4)
                trade["profit_percent"] = round(profit_percent, 2)
            else:
                trade["profit"] = 0
                trade["profit_percent"] = 0
        
        return trades[:limit]
    except FileNotFoundError:
        return []
    except Exception as e:
        import logging
        logger = logging.getLogger("DashboardService")
        logger.error(f"❌ Failed to get recent trades: {e}")
        return []

def get_performance() -> Dict:
    """Get performance metrics"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except FileNotFoundError:
        trades = []
    except Exception as e:
        import logging
        logger = logging.getLogger("DashboardService")
        logger.error(f"❌ Failed to load trades for performance metrics: {e}")
        trades = []
    
    # Last 7 days
    week_ago = datetime.now() - timedelta(days=7)
    
    week_trades = []
    for trade in trades:
        try:
            entry_time = trade.get("entry_time", "")
            if entry_time:
                trade_date = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                if trade_date > week_ago:
                    week_trades.append(trade)
        except (ValueError, KeyError):
            continue
    
    if not week_trades:
        return {
            "total_trades": 0, 
            "winning_trades": 0, 
            "win_rate": 0, 
            "total_profit": 0, 
            "avg_profit": 0,
            "total_volume": 0,
            "success_rate": 0
        }
    
    winning = 0
    total_profit = 0.0
    total_volume = 0.0
    
    for trade in week_trades:
        if trade.get("status") in ["CLOSED", "STOPPED"]:
            if trade.get("exit_price") and trade.get("entry_price"):
                profit = (trade["exit_price"] - trade["entry_price"]) * trade.get("size", 0)
                total_profit += profit
                if profit > 0:
                    winning += 1
        
        # Calculate trade volume
        entry_price = trade.get("entry_price", 0)
        size = trade.get("size", 0)
        if entry_price > 0 and size > 0:
            total_volume += entry_price * size
    
    total_trades = len(week_trades)
    win_rate = (winning / total_trades) * 100 if total_trades > 0 else 0
    
    # Get current account stats for volume context
    account_stats = get_account_stats()
    total_balance = account_stats.get("total_balance", 0)
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning,
        "win_rate": round(win_rate, 1),
        "total_profit": round(total_profit, 2),
        "avg_profit": round(total_profit / total_trades, 2) if total_trades > 0 else 0,
        "total_volume": round(total_volume, 2),
        "avg_volume": round(total_volume / total_trades, 2) if total_trades > 0 else 0,
        "volume_vs_balance": round((total_volume / total_balance) * 100, 1) if total_balance > 0 else 0,
        "success_rate": round(win_rate, 1)
    }

def get_daily_performance(days: int = 30) -> List[Dict]:
    """Get daily performance metrics for the last N days"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except FileNotFoundError:
        return []
    
    # Group trades by day
    daily_data = {}
    
    for trade in trades:
        try:
            entry_time = trade.get("entry_time", "")
            if not entry_time:
                continue
                
            trade_date = datetime.fromisoformat(entry_time.replace('Z', '+00:00')).date()
            
            # Skip if older than requested days
            if (datetime.now().date() - trade_date).days > days:
                continue
            
            if trade_date not in daily_data:
                daily_data[trade_date] = {
                    "date": trade_date.isoformat(),
                    "trades": 0,
                    "winning_trades": 0,
                    "total_profit": 0.0,
                    "total_volume": 0.0
                }
            
            daily_data[trade_date]["trades"] += 1
            
            # Calculate profit for closed trades
            if trade.get("status") in ["CLOSED", "STOPPED"]:
                entry_price = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                size = trade.get("size", 0)
                
                if entry_price > 0 and exit_price > 0:
                    profit = (exit_price - entry_price) * size
                    daily_data[trade_date]["total_profit"] += profit
                    
                    if profit > 0:
                        daily_data[trade_date]["winning_trades"] += 1
            
            # Calculate volume
            entry_price = trade.get("entry_price", 0)
            size = trade.get("size", 0)
            if entry_price > 0 and size > 0:
                daily_data[trade_date]["total_volume"] += entry_price * size
                
        except (ValueError, KeyError):
            continue
    
    # Convert to list and sort by date
    result = list(daily_data.values())
    result.sort(key=lambda x: x["date"], reverse=True)
    
    # Calculate win rates
    for day in result:
        day["win_rate"] = round((day["winning_trades"] / day["trades"]) * 100, 1) if day["trades"] > 0 else 0
        day["total_profit"] = round(day["total_profit"], 2)
        day["total_volume"] = round(day["total_volume"], 2)
        day["avg_profit_per_trade"] = round(day["total_profit"] / day["trades"], 2) if day["trades"] > 0 else 0
    
    return result

def get_trade_analytics() -> Dict:
    """Get detailed trade analytics"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
    except FileNotFoundError:
        return {"total_trades": 0, "analysis": {}}
    
    if not trades:
        return {"total_trades": 0, "analysis": {}}
    
    # Get bot performance data
    bot_stats = bc.get_stats()
    
    # Calculate various analytics
    closed_trades = [t for t in trades if t.get("status") in ["CLOSED", "STOPPED"]]
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    
    # Trade duration analysis
    durations = []
    for trade in closed_trades:
        try:
            entry_time = datetime.fromisoformat(trade.get("entry_time", "").replace('Z', '+00:00'))
            exit_time = datetime.fromisoformat(trade.get("exit_time", "").replace('Z', '+00:00'))
            duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
            durations.append(duration)
        except (ValueError, KeyError):
            continue
    
    # Profit analysis by symbol
    profit_by_symbol = {}
    for trade in closed_trades:
        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        size = trade.get("size", 0)
        
        if entry_price > 0 and exit_price > 0:
            profit = (exit_price - entry_price) * size
            
            if symbol not in profit_by_symbol:
                profit_by_symbol[symbol] = {
                    "trades": 0,
                    "winning_trades": 0,
                    "total_profit": 0.0,
                    "avg_profit": 0.0
                }
            
            profit_by_symbol[symbol]["trades"] += 1
            profit_by_symbol[symbol]["total_profit"] += profit
            if profit > 0:
                profit_by_symbol[symbol]["winning_trades"] += 1
    
    # Calculate averages
    for symbol in profit_by_symbol:
        trades_count = profit_by_symbol[symbol]["trades"]
        if trades_count > 0:
            profit_by_symbol[symbol]["avg_profit"] = profit_by_symbol[symbol]["total_profit"] / trades_count
            profit_by_symbol[symbol]["win_rate"] = (profit_by_symbol[symbol]["winning_trades"] / trades_count) * 100
    
    # Sort symbols by total profit
    sorted_symbols = sorted(profit_by_symbol.items(), key=lambda x: x[1]["total_profit"], reverse=True)
    
    # Get account information
    account_stats = get_account_stats()
    
    analytics = {
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "open_trades": len(open_trades),
        "avg_trade_duration": round(sum(durations) / len(durations), 1) if durations else 0,
        "total_profit": round(sum(t["total_profit"] for t in profit_by_symbol.values()), 2),
        "total_volume": round(bc.stats.get("total_volume_traded", 0), 2),
        "most_profitable_symbols": [
            {
                "symbol": symbol,
                "trades": data["trades"],
                "total_profit": round(data["total_profit"], 2),
                "avg_profit": round(data["avg_profit"], 2),
                "win_rate": round(data.get("win_rate", 0), 1)
            }
            for symbol, data in sorted_symbols[:5]  # Top 5
        ],
        "account_metrics": {
            "total_accounts": account_stats.get("total_accounts", 0),
            "validated_accounts": account_stats.get("validated_accounts", 0),
            "total_balance": round(account_stats.get("total_balance", 0), 2),
            "average_balance": round(account_stats.get("average_balance", 0), 2)
        },
        "bot_performance": {
            "qualified_signals": bc.stats.get("qualified_signals", 0),
            "total_scans": bc.stats.get("total_scans", 0),
            "qualification_rate": round((bc.stats.get("qualified_signals", 0) / max(bc.stats.get("total_scans", 1), 1)) * 100, 1),
            "scans_per_minute": bc.stats.get("scans_per_minute", 0)
        }
    }
    
    return analytics

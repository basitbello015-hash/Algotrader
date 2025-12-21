import json
from datetime import datetime
from typing import List, Dict

TRADES_FILE = "app/data/trades.json"

def get_trade_history(days: int = 30) -> List[Dict]:
    """Get trade history"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
        
        # Filter by date
        if days > 0:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [t for t in trades if datetime.fromisoformat(t.get("entry_time", "2000-01-01")) > cutoff]
        
        return trades
    except:
        return []

def get_trade_by_id(trade_id: str) -> Dict:
    """Get specific trade"""
    trades = get_trade_history(0)  # All trades
    for trade in trades:
        if trade.get("id") == trade_id:
            return trade
    return {}

def update_trade(trade_id: str, updates: Dict) -> bool:
    """Update trade"""
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
        
        updated = False
        for i, trade in enumerate(trades):
            if trade.get("id") == trade_id:
                trades[i].update(updates)
                updated = True
                break
        
        if updated:
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)
        
        return updated
    except:
        return False

from fastapi import APIRouter
from services.history_service import get_trade_history, get_trade_by_id

router = APIRouter()

@router.get("/")
async def history(days: int = 30):
    """Get trade history"""
    return get_trade_history(days)

@router.get("/{trade_id}")
async def get_trade(trade_id: str):
    """Get specific trade"""
    trade = get_trade_by_id(trade_id)
    if trade:
        return trade
    return {"error": "Trade not found"}

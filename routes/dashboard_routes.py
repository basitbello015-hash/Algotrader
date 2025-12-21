from fastapi import APIRouter
from services.dashboard_service import (
    get_dashboard_data, get_recent_trades, get_performance
)

router = APIRouter()

@router.get("/")
async def dashboard():
    """Get dashboard data"""
    return get_dashboard_data()

@router.get("/recent-trades")
async def recent_trades(limit: int = 10):
    """Get recent trades"""
    return get_recent_trades(limit)

@router.get("/performance")
async def performance():
    """Get performance metrics"""
    return get_performance()

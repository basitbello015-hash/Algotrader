from fastapi import APIRouter
from services.bot_service import get_bot_status, start_bot, stop_bot

router = APIRouter()

@router.get("/status")
async def status():
    """Get bot status"""
    return get_bot_status()

@router.post("/start")
async def start():
    """Start bot"""
    return start_bot()

@router.post("/stop")
async def stop():
    """Stop bot"""
    return stop_bot()

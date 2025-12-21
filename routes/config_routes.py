from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from services.config_service import get_config, update_config

router = APIRouter()

@router.get("/")
async def read_config():
    """Get current config"""
    return get_config()

@router.post("/")
async def save_config(config_data: Dict[str, Any]):
    """Save config"""
    result = update_config(config_data)
    if result["success"]:
        return {"status": "saved", "config": result["config"]}
    raise HTTPException(500, result.get("message", "Failed to save config"))

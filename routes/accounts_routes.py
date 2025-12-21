from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from services.accounts_service import (
    get_accounts, add_account, delete_account, test_account
)

router = APIRouter()

@router.get("/")
async def list_accounts():
    """Get all accounts"""
    return get_accounts()

@router.post("/")
async def create_account(account_data: Dict[str, Any]):
    """Add new account"""
    result = add_account(account_data)
    if result["success"]:
        return result
    raise HTTPException(400, result.get("message", "Failed to add account"))

@router.delete("/{account_id}")
async def remove_account(account_id: str):
    """Delete account"""
    result = delete_account(account_id)
    if result["success"]:
        return result
    raise HTTPException(404, result.get("message", "Account not found"))

@router.post("/{account_id}/test")
async def test_connection(account_id: str):
    """Test account connection"""
    result = test_account(account_id)
    return result

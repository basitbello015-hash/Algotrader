"""
Accounts API routes - Environment Variable Based
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from services.accounts_service import (
    get_accounts, 
    test_account, 
    get_account_stats,
    validate_all_accounts,
    get_environment_status,
    sync_accounts_with_bot,
    add_demo_account,
    # Note: add_account and delete_account are disabled for environment variables
)

router = APIRouter()
logger = logging.getLogger("AccountsRouter")

@router.get("/")
async def list_accounts():
    """Get all accounts from environment variables"""
    try:
        accounts = get_accounts()
        return {
            "success": True,
            "accounts": accounts,
            "count": len(accounts),
            "source": "environment_variables"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get accounts: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load accounts: {str(e)}")

@router.post("/")
async def create_account(account_data: Dict[str, Any]):
    """Add new account - Not supported with environment variables"""
    from services.accounts_service import add_account
    result = add_account(account_data)
    
    if result["success"]:
        return result
    elif result.get("error") == "Accounts managed via environment variables":
        # This is expected when using environment variables
        return result
    else:
        raise HTTPException(400, result.get("message", "Failed to add account"))

@router.delete("/{account_id}")
async def remove_account(account_id: str):
    """Delete account - Not supported with environment variables"""
    from services.accounts_service import delete_account
    result = delete_account(account_id)
    
    if result["success"]:
        return result
    elif "environment variables" in result.get("message", ""):
        # This is expected when using environment variables
        return result
    else:
        raise HTTPException(404, result.get("message", "Account not found"))

@router.post("/{account_id}/test")
async def test_connection(account_id: str):
    """Test account connection"""
    try:
        result = test_account(account_id)
        if result.get("success"):
            return {
                "success": True,
                "result": {
                    "connection": "success",
                    "balance": result.get("balance", 0),
                    "validation_time": result.get("validation_time", 0),
                    "message": result.get("message", "")
                }
            }
        else:
            return {
                "success": False,
                "result": {
                    "connection": "failed",
                    "reason": result.get("error") or result.get("message", "Unknown error"),
                    "validation_time": result.get("validation_time", 0)
                }
            }
    except Exception as e:
        logger.error(f"❌ Account test failed: {e}", exc_info=True)
        raise HTTPException(500, f"Account test failed: {str(e)}")

@router.get("/stats")
async def get_accounts_stats():
    """Get account statistics"""
    try:
        stats = get_account_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Failed to get account stats: {e}")
        raise HTTPException(500, f"Failed to get stats: {str(e)}")

@router.post("/validate-all")
async def validate_all_accounts_endpoint():
    """Validate all accounts from environment variables"""
    try:
        result = validate_all_accounts()
        return result
    except Exception as e:
        logger.error(f"❌ Failed to validate all accounts: {e}")
        raise HTTPException(500, f"Failed to validate accounts: {str(e)}")

@router.get("/env-status")
async def get_env_status():
    """Get environment variables status"""
    try:
        status = get_environment_status()
        return status
    except Exception as e:
        logger.error(f"❌ Failed to get environment status: {e}")
        raise HTTPException(500, f"Failed to get environment status: {str(e)}")

@router.post("/sync-bot")
async def sync_bot_accounts():
    """Sync accounts with bot controller"""
    try:
        result = sync_accounts_with_bot()
        return result
    except Exception as e:
        logger.error(f"❌ Failed to sync with bot: {e}")
        raise HTTPException(500, f"Failed to sync with bot: {str(e)}")

@router.post("/demo")
async def create_demo_account():
    """Add demo account for testing"""
    try:
        result = add_demo_account()
        if result["success"]:
            return result
        else:
            raise HTTPException(400, result.get("message", "Failed to add demo account"))
    except Exception as e:
        logger.error(f"❌ Failed to add demo account: {e}")
        raise HTTPException(500, f"Failed to add demo account: {str(e)}")

@router.get("/health")
async def accounts_health():
    """Health check for accounts service"""
    try:
        accounts = get_accounts()
        stats = get_account_stats()
        
        return {
            "status": "healthy",
            "accounts_count": len(accounts),
            "validated_accounts": stats.get("validated_accounts", 0),
            "total_balance": stats.get("total_balance", 0),
            "configuration_method": stats.get("configuration_method", "unknown")
        }
    except Exception as e:
        logger.error(f"❌ Accounts health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

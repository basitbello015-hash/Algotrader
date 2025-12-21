"""
Accounts service with comprehensive logging
"""
import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from bot_controller import bc

logger = logging.getLogger("AccountsService")
ACCOUNTS_FILE = "app/data/accounts.json"

def get_accounts() -> List[Dict]:
    """Get all accounts with logging"""
    logger.debug("📖 Getting all accounts")
    accounts = bc.load_accounts()
    logger.info(f"📊 Found {len(accounts)} accounts")
    
    # Log account status summary
    validated = len([a for a in accounts if a.get("validated", False)])
    total_balance = sum(a.get("balance", 0) for a in accounts)
    
    logger.debug(
        f"📊 Account summary: {validated}/{len(accounts)} validated, "
        f"Total balance: ${total_balance:.2f}"
    )
    
    return accounts

def add_account(data: Dict) -> Dict:
    """Add new account with detailed logging"""
    logger.info("➕ Adding new account")
    logger.debug(f"Account data received: {json.dumps(data, indent=2)}")
    
    # Generate ID
    account_id = str(uuid.uuid4())
    logger.debug(f"Generated account ID: {account_id}")
    
    # Create account object
    account = {
        "id": account_id,
        "name": data.get("name", "Unnamed Account"),
        "exchange": data.get("exchange", "bybit"),
        "api_key": data.get("apiKey"),
        "api_secret": data.get("secretKey"),
        "monitoring": True,
        "validated": False,
        "balance": 0.0,
        "created": datetime.now().isoformat(),
        "last_validation": None,
        "validation_error": None
    }
    
    logger.info(f"📝 Creating account: {account['name']} on {account['exchange']}")
    
    # Validate account
    logger.info("🔐 Validating account credentials...")
    start_time = datetime.now()
    
    validated, balance, error = bc.validate_account(account)
    
    validation_time = (datetime.now() - start_time).total_seconds()
    
    if validated:
        account["validated"] = True
        account["balance"] = balance
        account["last_validation"] = datetime.now().isoformat()
        logger.info(f"✅ Account validated successfully in {validation_time:.2f}s")
        logger.info(f"💰 Account balance: ${balance:.2f}")
    else:
        account["validated"] = False
        account["validation_error"] = error
        logger.error(f"❌ Account validation failed: {error}")
        logger.warning(f"⚠️ Account added but not validated")
    
    # Save account
    try:
        accounts = bc.load_accounts()
        accounts.append(account)
        bc.save_accounts(accounts)
        
        logger.info(f"💾 Account saved successfully")
        logger.debug(f"Total accounts: {len(accounts)}")
        
        return {
            "success": True,
            "account": account,
            "validated": validated,
            "balance": balance,
            "message": "Account added" + (" and validated" if validated else f" but validation failed: {error}")
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to save account: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to save account"
        }

def delete_account(account_id: str) -> Dict:
    """Delete account with logging"""
    logger.info(f"🗑️ Deleting account: {account_id}")
    
    accounts = bc.load_accounts()
    initial_count = len(accounts)
    
    # Find account to delete
    account_to_delete = next((a for a in accounts if a.get("id") == account_id), None)
    
    if not account_to_delete:
        logger.warning(f"⚠️ Account {account_id} not found for deletion")
        return {"success": False, "message": "Account not found"}
    
    logger.info(f"🗑️ Deleting account: {account_to_delete.get('name')}")
    
    # Filter out the account
    filtered = [a for a in accounts if a.get("id") != account_id]
    
    if len(filtered) < len(accounts):
        try:
            bc.save_accounts(filtered)
            logger.info(f"✅ Account deleted successfully")
            logger.debug(f"Accounts before: {initial_count}, after: {len(filtered)}")
            
            return {
                "success": True, 
                "message": "Account deleted",
                "deleted_account": account_to_delete.get("name"),
                "remaining_accounts": len(filtered)
            }
        except Exception as e:
            logger.error(f"❌ Failed to save after deletion: {e}")
            return {"success": False, "message": f"Deletion failed: {str(e)}"}
    
    logger.error(f"❌ Unexpected deletion failure")
    return {"success": False, "message": "Deletion failed"}

def test_account(account_id: str) -> Dict:
    """Test account connection with detailed logging"""
    logger.info(f"🔍 Testing account connection: {account_id}")
    
    accounts = bc.load_accounts()
    account = next((a for a in accounts if a.get("id") == account_id), None)
    
    if not account:
        logger.warning(f"⚠️ Account {account_id} not found")
        return {"success": False, "message": "Account not found"}
    
    logger.info(f"🔐 Testing connection for: {account.get('name')}")
    
    # Perform validation
    start_time = datetime.now()
    validated, balance, error = bc.validate_account(account)
    validation_time = (datetime.now() - start_time).total_seconds()
    
    if validated:
        # Update account with new balance
        for acc in accounts:
            if acc["id"] == account_id:
                old_balance = acc.get("balance", 0)
                acc["validated"] = True
                acc["balance"] = balance
                acc["last_validation"] = datetime.now().isoformat()
                acc["validation_error"] = None
                break
        
        try:
            bc.save_accounts(accounts)
            
            logger.info(f"✅ Connection test successful in {validation_time:.2f}s")
            logger.info(f"💰 Balance: ${balance:.2f} (was: ${old_balance:.2f})")
            
            return {
                "success": True,
                "validated": True,
                "balance": balance,
                "validation_time": validation_time,
                "balance_change": balance - old_balance,
                "message": f"Connection successful - Balance: ${balance:.2f}"
            }
        except Exception as e:
            logger.error(f"❌ Failed to save updated account: {e}")
            return {
                "success": False,
                "validated": False,
                "error": str(e),
                "message": "Connection successful but failed to update record"
            }
    else:
        logger.error(f"❌ Connection test failed: {error}")
        
        # Update error in account
        for acc in accounts:
            if acc["id"] == account_id:
                acc["validated"] = False
                acc["last_validation"] = datetime.now().isoformat()
                acc["validation_error"] = error
                break
        
        try:
            bc.save_accounts(accounts)
        except Exception as e:
            logger.error(f"❌ Failed to save error status: {e}")
        
        return {
            "success": False,
            "validated": False,
            "error": error,
            "validation_time": validation_time,
            "message": f"Connection failed: {error}"
        }

def get_account_stats() -> Dict:
    """Get account statistics"""
    logger.debug("📊 Getting account statistics")
    
    accounts = get_accounts()
    
    if not accounts:
        logger.debug("📭 No accounts found")
        return {"total_accounts": 0, "total_balance": 0}
    
    stats = {
        "total_accounts": len(accounts),
        "validated_accounts": len([a for a in accounts if a.get("validated", False)]),
        "invalid_accounts": len([a for a in accounts if not a.get("validated", True)]),
        "total_balance": sum(a.get("balance", 0) for a in accounts),
        "average_balance": sum(a.get("balance", 0) for a in accounts) / len(accounts),
        "exchanges": {}
    }
    
    # Count by exchange
    for account in accounts:
        exchange = account.get("exchange", "unknown")
        stats["exchanges"][exchange] = stats["exchanges"].get(exchange, 0) + 1
    
    logger.info(
        f"📊 Account stats: {stats['total_accounts']} total, "
        f"{stats['validated_accounts']} validated, "
        f"${stats['total_balance']:.2f} total balance"
    )
    
    return stats

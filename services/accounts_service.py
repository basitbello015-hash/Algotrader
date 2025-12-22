"""
Accounts service with comprehensive logging - ENVIRONMENT VARIABLE BASED
"""
import json
import os
import uuid
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from bot_controller import bc

logger = logging.getLogger("AccountsService")

# No more ACCOUNTS_FILE since we're using environment variables

def generate_account_id(api_key: str, api_secret: str) -> str:
    """Generate deterministic account ID from API credentials"""
    unique_string = f"{api_key}{api_secret}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:12]

def load_accounts_from_env() -> List[Dict]:
    """Load accounts from environment variables"""
    logger.info("🔐 Loading accounts from environment variables...")
    
    accounts = []
    account_counter = 1
    
    while True:
        # Look for environment variables in format:
        # BYBIT_API_KEY_1, BYBIT_API_SECRET_1
        # BYBIT_API_KEY_2, BYBIT_API_SECRET_2
        # etc.
        api_key = os.getenv(f"BYBIT_API_KEY_{account_counter}")
        api_secret = os.getenv(f"BYBIT_API_SECRET_{account_counter}")
        account_name = os.getenv(f"BYBIT_ACCOUNT_NAME_{account_counter}", f"Account {account_counter}")
        
        if not api_key or not api_secret:
            # Also check for non-numbered variables for single account
            if account_counter == 1:
                api_key = os.getenv("BYBIT_API_KEY")
                api_secret = os.getenv("BYBIT_API_SECRET")
                account_name = os.getenv("BYBIT_ACCOUNT_NAME", "Default Account")
                
                if api_key and api_secret:
                    account_id = generate_account_id(api_key, api_secret)
                    account = {
                        "id": account_id,
                        "name": account_name,
                        "exchange": "bybit",
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "monitoring": True,
                        "validated": False,
                        "balance": 0.0,
                        "created": datetime.now().isoformat(),
                        "last_validation": None,
                        "validation_error": None,
                        "source": "env_variable"
                    }
                    accounts.append(account)
                    logger.info(f"✅ Found single account: {account_name}")
                break
            else:
                break
        
        if api_key and api_secret:
            account_id = generate_account_id(api_key, api_secret)
            account = {
                "id": account_id,
                "name": account_name,
                "exchange": "bybit",
                "api_key": api_key,
                "api_secret": api_secret,
                "monitoring": True,
                "validated": False,
                "balance": 0.0,
                "created": datetime.now().isoformat(),
                "last_validation": None,
                "validation_error": None,
                "source": f"env_variable_{account_counter}"
            }
            accounts.append(account)
            logger.info(f"✅ Found account {account_counter}: {account_name}")
            account_counter += 1
        else:
            break
    
    if not accounts:
        logger.warning("⚠️ No accounts found in environment variables")
        logger.info("ℹ️ Expected environment variables:")
        logger.info("   Single account: BYBIT_API_KEY, BYBIT_API_SECRET")
        logger.info("   Multiple accounts: BYBIT_API_KEY_1, BYBIT_API_SECRET_1, BYBIT_API_KEY_2, etc.")
    
    return accounts

def get_accounts() -> List[Dict]:
    """Get all accounts from environment variables"""
    logger.debug("📖 Getting all accounts from environment variables")
    
    # Use the bot controller's load_accounts method
    env_accounts = bc.load_accounts()
    
    # Convert to our format
    accounts = []
    for env_acc in env_accounts:
        api_key = env_acc.get("api_key", "")
        api_secret = env_acc.get("api_secret", "")
        
        if api_key and api_secret:
            account_id = generate_account_id(api_key, api_secret)
            account = {
                "id": account_id,
                "name": env_acc.get("name", "Unknown Account"),
                "exchange": "bybit",
                "api_key": api_key,
                "api_secret": api_secret,  # Note: We store but NEVER log this
                "monitoring": True,
                "validated": env_acc.get("validated", False),
                "balance": env_acc.get("balance", 0.0),
                "created": env_acc.get("created", datetime.now().isoformat()),
                "last_validation": env_acc.get("last_validation"),
                "validation_error": env_acc.get("validation_error"),
                "source": "env_variable"
            }
            accounts.append(account)
    
    # Log account status summary
    validated = len([a for a in accounts if a.get("validated", False)])
    total_balance = sum(a.get("balance", 0) for a in accounts)
    
    logger.info(f"📊 Found {len(accounts)} accounts in environment variables")
    logger.debug(
        f"📊 Account summary: {validated}/{len(accounts)} validated, "
        f"Total balance: ${total_balance:.2f}"
    )
    
    return accounts

def add_account(data: Dict) -> Dict:
    """Add new account - NOT SUPPORTED with environment variables"""
    logger.warning("🚫 Account addition via UI is disabled when using environment variables")
    logger.info("ℹ️ To add accounts, set environment variables on your server:")
    logger.info("   Single account: BYBIT_API_KEY, BYBIT_API_SECRET")
    logger.info("   Multiple accounts: BYBIT_API_KEY_1, BYBIT_API_SECRET_1, etc.")
    
    return {
        "success": False,
        "error": "Accounts managed via environment variables",
        "message": "Accounts are managed through environment variables. Please set BYBIT_API_KEY and BYBIT_API_SECRET in your environment."
    }

def delete_account(account_id: str) -> Dict:
    """Delete account - NOT SUPPORTED with environment variables"""
    logger.warning("🚫 Account deletion via UI is disabled when using environment variables")
    logger.info("ℹ️ To remove accounts, delete the environment variables from your server")
    
    # Find the account to provide helpful info
    accounts = get_accounts()
    account_to_delete = next((a for a in accounts if a.get("id") == account_id), None)
    
    if account_to_delete:
        account_name = account_to_delete.get("name", "Unknown")
        logger.info(f"ℹ️ To delete account '{account_name}', remove its environment variables")
        
        return {
            "success": False,
            "message": f"Account '{account_name}' is managed via environment variables. Remove the environment variables to delete it."
        }
    
    return {
        "success": False,
        "message": "Account deletion is not supported with environment variable configuration"
    }

def test_account(account_id: str) -> Dict:
    """Test account connection with detailed logging"""
    logger.info(f"🔍 Testing account connection: {account_id}")
    
    accounts = get_accounts()
    account = next((a for a in accounts if a.get("id") == account_id), None)
    
    if not account:
        logger.warning(f"⚠️ Account {account_id} not found")
        return {"success": False, "message": "Account not found"}
    
    account_name = account.get('name', 'Unknown Account')
    logger.info(f"🔐 Testing connection for: {account_name}")
    
    # Prepare account dict for bot controller validation
    validation_account = {
        "name": account_name,
        "api_key": account.get("api_key"),
        "api_secret": account.get("api_secret"),
        "validated": account.get("validated", False),
        "balance": account.get("balance", 0.0)
    }
    
    # Perform validation using bot controller
    start_time = datetime.now()
    validated, balance, error = bc.validate_account(validation_account)
    validation_time = (datetime.now() - start_time).total_seconds()
    
    if validated:
        old_balance = account.get("balance", 0)
        
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
    else:
        logger.error(f"❌ Connection test failed: {error}")
        
        return {
            "success": False,
            "validated": False,
            "error": error,
            "validation_time": validation_time,
            "message": f"Connection failed: {error}"
        }

def validate_all_accounts() -> Dict:
    """Validate all accounts from environment variables"""
    logger.info("🔐 Validating all accounts from environment variables")
    
    accounts = get_accounts()
    results = []
    successful = 0
    failed = 0
    total_balance = 0.0
    
    for account in accounts:
        account_name = account.get('name', 'Unknown Account')
        logger.info(f"🔐 Validating: {account_name}")
        
        validation_account = {
            "name": account_name,
            "api_key": account.get("api_key"),
            "api_secret": account.get("api_secret")
        }
        
        validated, balance, error = bc.validate_account(validation_account)
        
        result = {
            "name": account_name,
            "validated": validated,
            "balance": balance,
            "error": error,
            "source": account.get("source", "env")
        }
        
        if validated:
            successful += 1
            total_balance += balance
            logger.info(f"  ✅ Validated - Balance: ${balance:.2f}")
        else:
            failed += 1
            logger.error(f"  ❌ Failed: {error}")
        
        results.append(result)
    
    logger.info(f"📊 Validation complete: {successful} successful, {failed} failed")
    logger.info(f"💰 Total validated balance: ${total_balance:.2f}")
    
    return {
        "success": True,
        "results": results,
        "summary": {
            "total_accounts": len(accounts),
            "successful_validations": successful,
            "failed_validations": failed,
            "total_balance": total_balance
        }
    }

def get_account_stats() -> Dict:
    """Get account statistics from environment variables"""
    logger.debug("📊 Getting account statistics from environment variables")
    
    accounts = get_accounts()
    
    if not accounts:
        logger.debug("📭 No accounts found in environment variables")
        return {
            "total_accounts": 0, 
            "total_balance": 0,
            "configuration_method": "environment_variables",
            "environment_variables_configured": False
        }
    
    # Count validated accounts
    validated_accounts = [a for a in accounts if a.get("validated", False)]
    invalid_accounts = [a for a in accounts if not a.get("validated", True)]
    
    # Calculate balances
    total_balance = sum(a.get("balance", 0) for a in accounts)
    avg_balance = total_balance / len(accounts) if accounts else 0
    
    stats = {
        "total_accounts": len(accounts),
        "validated_accounts": len(validated_accounts),
        "invalid_accounts": len(invalid_accounts),
        "total_balance": total_balance,
        "average_balance": avg_balance,
        "configuration_method": "environment_variables",
        "environment_variables_configured": True,
        "validation_rate": (len(validated_accounts) / len(accounts) * 100) if accounts else 0,
        "accounts": []
    }
    
    # Add account summaries
    for account in accounts:
        stats["accounts"].append({
            "name": account.get("name", "Unknown"),
            "validated": account.get("validated", False),
            "balance": account.get("balance", 0),
            "source": account.get("source", "env")
        })
    
    logger.info(
        f"📊 Account stats: {stats['total_accounts']} total, "
        f"{stats['validated_accounts']} validated ({stats['validation_rate']:.1f}%), "
        f"${stats['total_balance']:.2f} total balance"
    )
    
    return stats

def get_environment_status() -> Dict:
    """Check environment variables status"""
    logger.debug("🔍 Checking environment variables status")
    
    env_vars = {}
    
    # Check for single account
    has_single_key = bool(os.getenv("BYBIT_API_KEY"))
    has_single_secret = bool(os.getenv("BYBIT_API_SECRET"))
    
    env_vars["single_account"] = {
        "BYBIT_API_KEY": "✅ Configured" if has_single_key else "❌ Missing",
        "BYBIT_API_SECRET": "✅ Configured" if has_single_secret else "❌ Missing",
        "complete": has_single_key and has_single_secret
    }
    
    # Check for multiple accounts
    multiple_accounts = []
    for i in range(1, 10):  # Check up to 9 accounts
        key = os.getenv(f"BYBIT_API_KEY_{i}")
        secret = os.getenv(f"BYBIT_API_SECRET_{i}")
        if key or secret:  # At least one is configured
            multiple_accounts.append({
                "account_number": i,
                "BYBIT_API_KEY": "✅ Configured" if key else "❌ Missing",
                "BYBIT_API_SECRET": "✅ Configured" if secret else "❌ Missing",
                "complete": bool(key and secret)
            })
    
    env_vars["multiple_accounts"] = multiple_accounts
    
    # Groq API key check
    groq_key = os.getenv("GROQ_API_KEY")
    env_vars["groq"] = {
        "GROQ_API_KEY": "✅ Configured" if groq_key else "❌ Missing",
        "has_key": bool(groq_key)
    }
    
    # Summary
    total_accounts = len(load_accounts_from_env())
    env_vars["summary"] = {
        "total_accounts_found": total_accounts,
        "groq_configured": bool(groq_key),
        "has_valid_accounts": total_accounts > 0
    }
    
    logger.info(f"🔍 Environment status: {total_accounts} accounts found, Groq: {'✅' if groq_key else '❌'}")
    
    return env_vars

def add_demo_account() -> Dict:
    """Add a demo account for testing when no environment variables are set"""
    logger.info("🎮 Adding demo account for testing")
    
    # Check if we already have accounts
    existing_accounts = get_accounts()
    if existing_accounts:
        logger.warning("⚠️ Cannot add demo account - real accounts already configured")
        return {
            "success": False,
            "message": "Cannot add demo account when real accounts are configured"
        }
    
    # Check if we're in dry run mode
    from bot_controller import CONFIG
    if not CONFIG.get("dryRun", True):
        logger.error("❌ Cannot add demo account in live trading mode")
        return {
            "success": False,
            "message": "Demo accounts only available in dry run mode"
        }
    
    # Create demo account
    demo_account = {
        "id": "demo_account_123",
        "name": "Demo Account (Testnet)",
        "exchange": "bybit",
        "api_key": "demo_api_key",
        "api_secret": "demo_api_secret",
        "monitoring": True,
        "validated": True,
        "balance": 1000.00,
        "created": datetime.now().isoformat(),
        "last_validation": datetime.now().isoformat(),
        "validation_error": None,
        "source": "demo",
        "is_demo": True
    }
    
    logger.info(f"✅ Demo account created: {demo_account['name']} with ${demo_account['balance']:.2f}")
    logger.info("ℹ️ Demo account works in DRY RUN mode only")
    
    # Return the demo account
    return {
        "success": True,
        "account": demo_account,
        "validated": True,
        "balance": demo_account["balance"],
        "message": "Demo account created for testing (dry run mode only)"
    }

def sync_accounts_with_bot() -> Dict:
    """Sync accounts with bot controller and validate them"""
    logger.info("🔄 Syncing accounts with bot controller...")
    
    # Reload accounts from environment
    accounts = get_accounts()
    
    # Validate each account
    validation_results = []
    for account in accounts:
        account_name = account.get('name', 'Unknown Account')
        
        validation_account = {
            "name": account_name,
            "api_key": account.get("api_key"),
            "api_secret": account.get("api_secret")
        }
        
        validated, balance, error = bc.validate_account(validation_account)
        
        result = {
            "name": account_name,
            "validated": validated,
            "balance": balance,
            "error": error
        }
        validation_results.append(result)
        
        if validated:
            logger.info(f"  ✅ {account_name}: Validated, Balance: ${balance:.2f}")
        else:
            logger.error(f"  ❌ {account_name}: Validation failed: {error}")
    
    # Get bot status
    bot_stats = bc.get_stats() if hasattr(bc, 'get_stats') else {}
    
    return {
        "success": True,
        "accounts_count": len(accounts),
        "validation_results": validation_results,
        "bot_status": {
            "running": bc.is_running() if hasattr(bc, 'is_running') else False,
            "ai_coins_count": len(bc.get_coins()) if hasattr(bc, 'get_coins') else 0
        },
        "message": f"Synced {len(accounts)} accounts with bot controller"
    }

import json
import os
from typing import Dict, Any
from datetime import datetime 

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "exchange": "bybit",
    "strategy": "high-probability",
    "timeframe": "5",
    "stopLoss": 1.0,
    "takeProfit": 2.5,
    "maxHold": 5400,
    "rsiPeriod": 14,
    "rsiOversold": 35,
    "tradeAllocation": 100,
    "minTradeAmount": 5.0,
    "maxTradesPerDay": 10,
    "scanInterval": 60,
    "useMarketOrder": True,
    "testOnTestnet": False,
    "dryRun": True,
    "ai_coins": ["ADAUSDT", "DOGEUSDT", "TRXUSDT", "VETUSDT", "ALGOUSDT", "XLMUSDT", "ONEUSDT", "ANKRUSDT", "COTIUSDT", "SHIBUSDT"]
}

def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Merge with defaults to ensure all keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    except:
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Config save error: {e}")
        return False

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration"""
    config = get_config()
    config.update(updates)
    
    if save_config(config):
        return {"success": True, "config": config}
    return {"success": False, "message": "Failed to save config"}

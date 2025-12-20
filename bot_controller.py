"""
Enhanced Bot Controller v2.0 - HIGH PROBABILITY TRADING
GROQ API INTEGRATION FOR AI COIN SELECTION
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import schedule
import logging
import csv
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from pybit.unified_trading import HTTP
from groq import Groq  # Groq Python SDK

# -------------------- LOGGING SETUP --------------------
def setup_logging():
    """Setup comprehensive logging"""
    logger = logging.getLogger("AlgoTrader")
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        f"logs/algo_trader_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# -------------------- GROQ API CLIENT --------------------
class GroqAIClient:
    """Groq API client for fast AI coin selection"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.logger = logging.getLogger("GroqClient")
        
        if not self.api_key:
            self.logger.warning("⚠️ GROQ_API_KEY not found in environment")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                self.logger.info("✅ Groq client initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Groq client: {e}")
                self.client = None
        
        # Available Groq models
        self.models = [
            "llama3-70b-8192",      # Most capable
            "llama3-8b-8192",       # Fast and efficient
            "mixtral-8x7b-32768",   # Good balance
            "gemma-7b-it"           # Fastest
        ]
        
        self.current_model = self.models[0]
        
        # Cache for recent predictions
        self.cache = {}
        self.cache_duration = 3600
        
        # Usage tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
    
    def get_coins_analysis(self, prompt: str, model: str = None) -> Optional[Dict]:
        """Get coin analysis from Groq API"""
        if not self.client:
            self.logger.warning("⚠️ Groq client not available")
            return None
        
        # Check cache first
        cache_key = f"{model or self.current_model}:{hash(prompt)}"
        if cache_key in self.cache:
            cached_time, result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                self.logger.debug(f"📦 Using cached Groq result")
                return result
        
        model_to_use = model or self.current_model
        self.total_requests += 1
        
        self.logger.info(f"🤖 Calling Groq API (Model: {model_to_use})...")
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a cryptocurrency trading expert. 
                        Analyze market data and provide actionable trading insights.
                        Always return valid JSON arrays without markdown formatting."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            elapsed = time.time() - start_time
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            self.successful_requests += 1
            self.total_tokens += tokens_used
            
            self.logger.info(f"✅ Groq response in {elapsed:.2f}s, {tokens_used} tokens")
            
            # Parse JSON response
            try:
                result = json.loads(content)
                
                # Cache the result
                self.cache[cache_key] = (time.time(), result)
                
                # Clean cache if too large
                if len(self.cache) > 100:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                    del self.cache[oldest_key]
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse Groq JSON response: {e}")
                self.total_errors += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Groq API error: {e}")
            self.total_errors += 1
            
            # Try fallback model if first fails
            if model_to_use != self.models[-1]:
                self.logger.info(f"🔄 Trying fallback model: {self.models[-1]}")
                return self.get_coins_analysis(prompt, self.models[-1])
            
            return None
    
    def get_daily_coins(self) -> List[str]:
        """Get daily coin picks from Groq AI"""
        self.logger.info("🤖 Getting daily coin picks from Groq...")
        
        prompt = """Analyze the current cryptocurrency market and select the top 10 
        trading pairs (format: SYMBOLUSDT) for coins under $1 that have the highest 
        probability of 2-3% gains in the next 24 hours.
        
        CRITERIA:
        1. Price under $1.00
        2. High volume (150%+ of average)
        3. RSI below 40 (oversold)
        4. Positive momentum indicators
        5. Good liquidity on Binance/Bybit
        6. Recent consolidation or breakout patterns
        
        IMPORTANT: Return ONLY a JSON object with this exact structure:
        {
            "analysis_date": "YYYY-MM-DD",
            "selected_coins": ["ADAUSDT", "DOGEUSDT", ...],
            "reasoning": "Brief explanation of selection",
            "confidence_score": 0.85
        }
        
        Ensure exactly 10 coins in the array."""
        
        result = self.get_coins_analysis(prompt)
        
        if result and "selected_coins" in result:
            coins = result["selected_coins"]
            confidence = result.get("confidence_score", 0)
            reasoning = result.get("reasoning", "")
            
            self.logger.info(f"🎯 Groq selected {len(coins)} coins with {confidence:.0%} confidence")
            self.logger.info(f"📝 Reasoning: {reasoning[:100]}...")
            
            # Validate coins
            valid_coins = []
            for coin in coins:
                if isinstance(coin, str) and coin.endswith('USDT'):
                    valid_coins.append(coin)
                else:
                    self.logger.warning(f"⚠️ Invalid coin format: {coin}")
            
            if len(valid_coins) >= 5:
                return valid_coins[:10]
        
        # Fallback to default coins
        self.logger.warning("⚠️ Groq selection failed, using default coins")
        return DEFAULT_AI_COINS
    
    def get_market_analysis(self) -> Dict[str, Any]:
        """Get comprehensive market analysis from Groq"""
        self.logger.info("📊 Getting market analysis from Groq...")
        
        prompt = """Provide a concise cryptocurrency market analysis for today.
        
        Focus on:
        1. Overall market sentiment (bullish/bearish/neutral)
        2. Key support/resistance levels for major coins
        3. Volume trends
        4. Risk assessment for altcoins under $1
        
        Return JSON format:
        {
            "market_sentiment": "bullish",
            "sentiment_score": 0.75,
            "key_insights": ["insight1", "insight2"],
            "risk_level": "medium",
            "recommended_allocation": 0.8,
            "top_opportunities": ["ADAUSDT", "DOGEUSDT"]
        }"""
        
        result = self.get_coins_analysis(prompt)
        
        if result:
            self.logger.info(f"📊 Market analysis: {result.get('market_sentiment', 'unknown')} sentiment")
            return result
        
        return {
            "market_sentiment": "neutral",
            "sentiment_score": 0.5,
            "key_insights": ["No AI analysis available"],
            "risk_level": "medium",
            "recommended_allocation": 0.5,
            "top_opportunities": DEFAULT_AI_COINS[:5]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Groq API usage statistics"""
        return {
            "api_key_configured": self.client is not None,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "current_model": self.current_model,
            "cache_size": len(self.cache)
        }

# Default AI coins (fallback)
DEFAULT_AI_COINS = [
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "VETUSDT", "ALGOUSDT",
    "XLMUSDT", "ONEUSDT", "ANKRUSDT", "COTIUSDT", "SHIBUSDT"
]

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config() -> Dict[str, Any]:
    """Load configuration with debugging"""
    cfg_path = os.path.join(BASE_DIR, "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config = json.load(f)
                logger.info(f"✅ Config loaded from {cfg_path}")
                
                # Ensure required fields exist
                config.setdefault("tradeAllocation", 100)
                config.setdefault("minTradeAmount", 5.0)
                config.setdefault("maxTradesPerDay", 10)
                config.setdefault("dryRun", True)
                return config
        else:
            logger.warning(f"⚠️ Config file not found at {cfg_path}, using defaults")
            return {}
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}", exc_info=True)
        return {}

CONFIG = load_config()

# ==================== TRADING SETTINGS ====================
TIMEFRAME = CONFIG.get("timeframe", "5")
MIN_TRADE_AMOUNT = float(CONFIG.get("minTradeAmount", 5.0))
DRY_RUN = CONFIG.get("dryRun", True)

logger.info(f"📊 Trading Settings:")
logger.info(f"   Timeframe: {TIMEFRAME} minutes")
logger.info(f"   Min Trade: ${MIN_TRADE_AMOUNT}")
logger.info(f"   Dry Run: {DRY_RUN}")
logger.info(f"   Allocation: 100% per trade")
logger.info(f"   AI Provider: Groq API")

# File paths
ACCOUNTS_FILE = os.path.join(BASE_DIR, "app/data/accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "app/data/trades.json")
COINS_FILE = os.path.join(BASE_DIR, "app/data/coins_daily.json")
PERF_LOG_FILE = os.path.join(BASE_DIR, "logs/performance.csv")

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "app/data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# -------------------- RATE LIMITER --------------------
class RateLimiter:
    def __init__(self, max_per_second=3, max_per_minute=30, name="RateLimiter"):
        self.max_per_second = max_per_second
        self.max_per_minute = max_per_minute
        self.second_bucket = max_per_second
        self.minute_bucket = max_per_minute
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.name = name
        self.total_waits = 0
        self.total_acquires = 0
        logger.debug(f"🔧 {name} initialized: {max_per_second}/sec, {max_per_minute}/min")
    
    def acquire(self) -> bool:
        self.total_acquires += 1
        
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill buckets
            if elapsed >= 60:
                logger.debug(f"🔄 {self.name} refilling both buckets")
                self.second_bucket = self.max_per_second
                self.minute_bucket = self.max_per_minute
                self.last_refill = now
            elif elapsed >= 1:
                self.second_bucket = min(self.max_per_second, 
                                        self.second_bucket + int(elapsed))
                self.last_refill = now
            
            if self.second_bucket > 0 and self.minute_bucket > 0:
                self.second_bucket -= 1
                self.minute_bucket -= 1
                logger.debug(f"✅ {self.name} acquired token (remaining: {self.second_bucket}/{self.minute_bucket})")
                return True
            
            # Calculate wait time
            wait_for_second = 1 - (now - self.last_refill) if self.second_bucket == 0 else 0
            wait_for_minute = 60 - (now - self.last_refill) if self.minute_bucket == 0 else 0
            wait_time = max(wait_for_second, wait_for_minute)
            
            if wait_time > 0:
                self.total_waits += 1
                logger.warning(f"⏳ {self.name} rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                return self.acquire()
            
            logger.error(f"❌ {self.name} failed to acquire token")
            return False
    
    def get_stats(self) -> Dict:
        return {
            "total_acquires": self.total_acquires,
            "total_waits": self.total_waits,
            "remaining_second": self.second_bucket,
            "remaining_minute": self.minute_bucket
        }

# -------------------- AI COIN SELECTOR WITH GROQ --------------------
class GroqCoinSelector:
    """Coin selector using Groq API"""
    
    def __init__(self):
        self.coins_file = COINS_FILE
        self.last_update = None
        self.current_coins = []
        self.groq_client = GroqAIClient()
        self.logger = logging.getLogger("GroqCoinSelector")
        self.market_analysis = {}
        self.load_coins()
        self.logger.info("✅ Groq Coin Selector initialized")
    
    def load_coins(self):
        """Load AI-selected coins from file"""
        try:
            if os.path.exists(self.coins_file):
                with open(self.coins_file, "r") as f:
                    data = json.load(f)
                    self.current_coins = data.get("coins", DEFAULT_AI_COINS)
                    self.last_update = data.get("updated")
                    self.market_analysis = data.get("market_analysis", {})
                    
                    self.logger.info(f"📁 Loaded {len(self.current_coins)} coins from file")
                    
                    # Check if data is from today
                    if self.last_update and not self.last_update.startswith(datetime.now().strftime("%Y-%m-%d")):
                        self.logger.info("🔄 Data is from a different day, will update")
            else:
                self.current_coins = DEFAULT_AI_COINS
                self.save_coins()
                self.logger.warning("⚠️ No coin file found, using defaults")
        except Exception as e:
            self.logger.error(f"❌ Failed to load coins: {e}", exc_info=True)
            self.current_coins = DEFAULT_AI_COINS
    
    def save_coins(self, coins=None, market_analysis=None):
        """Save coins to file with Groq analysis"""
        if coins:
            self.current_coins = coins
            self.logger.info(f"💾 Saving {len(coins)} coins from Groq")
        
        if market_analysis:
            self.market_analysis = market_analysis
        
        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "coins": self.current_coins,
            "updated": datetime.now().isoformat(),
            "market_analysis": self.market_analysis,
            "groq_stats": self.groq_client.get_stats()
        }
        
        try:
            with open(self.coins_file, "w") as f:
                json.dump(data, f, indent=2)
            self.last_update = data["updated"]
            
            self.logger.info(f"✅ Groq coins saved: {len(self.current_coins)} symbols")
            self.logger.debug(f"Market sentiment: {self.market_analysis.get('market_sentiment', 'unknown')}")
            
            return self.current_coins
        except Exception as e:
            self.logger.error(f"❌ Failed to save coins: {e}")
            return self.current_coins
    
    def get_coins(self) -> List[str]:
        """Get current AI coins - update if new day"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update if new day or no coins
        if not self.last_update or not self.last_update.startswith(today) or not self.current_coins:
            self.logger.info("🔄 New day or missing coins, updating from Groq...")
            return self.update_from_groq()
        
        return self.current_coins
    
    def update_from_groq(self) -> List[str]:
        """Fetch coin picks from Groq API with market analysis"""
        start_time = time.time()
        self.logger.info("🤖 Starting Groq coin selection...")
        
        try:
            # Get market analysis first
            self.logger.info("📊 Getting market analysis...")
            market_analysis = self.groq_client.get_market_analysis()
            
            # Get coin picks
            self.logger.info("🎯 Getting coin picks...")
            coins = self.groq_client.get_daily_coins()
            
            elapsed = time.time() - start_time
            
            if coins and len(coins) >= 5:
                self.logger.info(f"✅ Groq selection completed in {elapsed:.2f}s")
                self.logger.info(f"📊 Market sentiment: {market_analysis.get('market_sentiment', 'unknown')}")
                self.logger.info(f"🎯 Selected coins: {', '.join(coins)}")
                
                return self.save_coins(coins, market_analysis)
            else:
                self.logger.warning(f"⚠️ Groq returned insufficient coins ({len(coins) if coins else 0}), using fallback")
                
                # Use market analysis to shuffle default coins
                sentiment = market_analysis.get("market_sentiment", "neutral")
                import random
                
                if sentiment == "bullish":
                    fallback_coins = DEFAULT_AI_COINS.copy()
                elif sentiment == "bearish":
                    fallback_coins = ["ADAUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT", "ONEUSDT",
                                     "ANKRUSDT", "COTIUSDT", "TRXUSDT", "DOGEUSDT", "SHIBUSDT"]
                else:
                    fallback_coins = DEFAULT_AI_COINS.copy()
                    random.shuffle(fallback_coins)
                
                return self.save_coins(fallback_coi

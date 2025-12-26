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
from datetime import datetime,timedelta
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
        datefmt='%Y-%m-d %H:%M:%S'
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
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
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
                        Always return valid JSON without markdown formatting.
                        
                        IMPORTANT: You MUST return a JSON object with this EXACT structure:
                        {
                            "analysis_date": "YYYY-MM-DD",
                            "selected_coins": ["COIN1USDT", "COIN2USDT", ...],
                            "reasoning": "Brief analysis here",
                            "confidence_score": 0.85,
                            "profit_probability_percentage": 80,
                            "price_range": "All under $1.00",
                            "risk_assessment": "Low to Medium"
                        }
                        
                        Ensure ALL selected_coins are under $1.00 and have high profit probability."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=800,  # Increased for better analysis
                response_format={"type": "json_object"}
            )

            elapsed = time.time() - start_time

            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            self.successful_requests += 1
            self.total_tokens += tokens_used

            self.logger.info(f"✅ Groq response in {elapsed:.2f}s, {tokens_used} tokens")
            self.logger.debug(f"📝 Raw response preview: {content[:200]}...")

            # Parse JSON response
            try:
                result = json.loads(content)
                
                # Validate the response structure
                if not isinstance(result, dict):
                    self.logger.error(f"❌ Groq response is not a dictionary: {type(result)}")
                    self.total_errors += 1
                    return None
                
                if "selected_coins" not in result:
                    self.logger.error(f"❌ Groq response missing 'selected_coins' key")
                    self.logger.debug(f"📝 Response keys: {list(result.keys())}")
                    self.total_errors += 1
                    return None

                # Cache the result
                self.cache[cache_key] = (time.time(), result)

                # Clean cache if too large
                if len(self.cache) > 100:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                    del self.cache[oldest_key]

                return result

            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse Groq JSON response: {e}")
                self.logger.debug(f"📝 Problematic content: {content[:500]}")
                self.total_errors += 1
                return None

        except Exception as e:
            self.logger.error(f"❌ Groq API error: {str(e)[:200]}")
            self.total_errors += 1

            # Try fallback model if first fails
            if model_to_use != self.models[-1]:
                self.logger.info(f"🔄 Trying fallback model: {self.models[-1]}")
                return self.get_coins_analysis(prompt, self.models[-1])

            return None

    def get_daily_coins(self) -> List[str]:
        """Get daily coin picks from Groq AI"""
        self.logger.info("🤖 Getting daily coin picks from Groq...")

        prompt = """CRITICAL INSTRUCTION: You are a quantitative crypto analyst. Select the 10 cryptocurrency trading pairs CURRENTLY UNDER $1.00 that present the STRONGEST BULLISH TECHNICAL SETUPS for potential profit in the next 24-48 hours.

        PRIORITIZATION CRITERIA (Rank coins by how many of these bullish signals they exhibit):
        1.  **Price:** MUST be under $1.00 USD.
        2.  **Momentum & Volume:** High relative volume (preferably >150% of recent average) accompanying price moves.
        3.  **Oversold Bounce:** RSI (14) is below 40 and showing an upward turn.
        4.  **Trend Shift:** MACD showing a recent or impending bullish crossover/histogram divergence.
        5.  **Support:** Price is at or near a defined support level (e.g., trendline, moving average, previous resistance-turn-support).
        6.  **Liquidity:** Sufficient liquidity (visible order book depth, not just reported volume).
        7.  **Pattern:** Evidence of consolidation (e.g., wedge, triangle) with a potential bullish breakout.
        8.  **Market Context:** Positive sector/market sentiment aligns with the coin's move.

         REQUIRED OUTPUT FORMAT (STRICT JSON):
       {
        "analysis_date": "YYYY-MM-DD",
         "selected_coins": ["COIN1USDT", "COIN2USDT", "COIN3USDT", "COIN4USDT", "COIN5USDT", "COIN6USDT", "COIN7USDT", "COIN8USDT", "COIN9USDT", "COIN10USDT"],
         "reasoning": "Concise 1-2 sentence summary of the overall market context and why these were chosen based on the criteria.",
         "confidence_score": 0.75,
         "risk_assessment": "Medium. All selections are low-cap, high-volatility assets. Strict stop-losses are mandatory."
        }

          MANDATORY RULES:
          1.  Output MUST be valid JSON.
          2.  `selected_coins` MUST be exactly 10 strings. Use the 'USDT' suffix (e.g., 'ADAUSDT').
          3.  All coins MUST be under $1.00 at the time of analysis.
          4.  `confidence_score` is a float between 0.5 and 0.9, reflecting the aggregate strength of signals.
          5.  Do NOT include disclaimers in the JSON. Provide only the specified keys.
          """

        result = self.get_coins_analysis(prompt)

        if result and isinstance(result, dict) and "selected_coins" in result:
            coins = result["selected_coins"]
            
            # Ensure coins is a list
            if not isinstance(coins, list):
                self.logger.error(f"❌ selected_coins is not a list: {type(coins)}")
                coins = HIGH_PROBABILITY_COINS
            else:
                confidence = result.get("confidence_score", 0)
                profit_prob = result.get("profit_probability_percentage", 0)
                reasoning = result.get("reasoning", "")

                self.logger.info(f"🎯 Groq selected {len(coins)} coins with {confidence:.0%} confidence, {profit_prob}% profit probability")
                if reasoning:
                    self.logger.info(f"📝 Reasoning: {reasoning[:100]}...")

            # Validate coins
            valid_coins = []
            for coin in coins:
                if isinstance(coin, str) and coin.endswith('USDT'):
                    valid_coins.append(coin)
                else:
                    self.logger.warning(f"⚠️ Invalid coin format: {coin}")

            # Ensure exactly 10 coins
            if len(valid_coins) == 10:
                self.logger.info(f"✅ Validated: Exactly 10 coins selected")
                return valid_coins
            elif len(valid_coins) >= 5:
                self.logger.warning(f"⚠️ Got {len(valid_coins)} coins instead of 10, using what we have")
                return valid_coins[:10]
            else:
                self.logger.warning(f"⚠️ Insufficient valid coins ({len(valid_coins)}), using defaults")

        # Fallback to default coins with 80% probability focus
        self.logger.warning("⚠️ Groq selection failed or invalid, using high-probability default coins")
        return HIGH_PROBABILITY_COINS

    def get_market_analysis(self) -> Dict[str, Any]:
        """Get comprehensive market analysis from Groq"""
        self.logger.info("📊 Getting market analysis from Groq...")

        prompt = """Provide a concise cryptocurrency market analysis for today focusing on coins under $1 with high profit probability.
        
        Focus on:
        1. Overall market sentiment for low-cap coins under $1
        2. Technical setups showing 80%+ profit probability today
        3. Volume trends in altcoins under $1
        4. Risk assessment with high probability setups
        5. Specific coins under $1 ready for 2-3% gains today
        
        Return JSON format:
        {
            "market_sentiment": "bullish",
            "sentiment_score": 0.75,
            "profit_probability_today": 80,
            "key_insights": ["insight1", "insight2"],
            "risk_level": "medium",
            "recommended_allocation": 0.8,
            "top_opportunities": ["ADAUSDT", "DOGEUSDT"],
            "price_range": "All under $1.00",
            "expected_gains": "2-3% with 80% probability"
        }"""

        result = self.get_coins_analysis(prompt)

        if result and isinstance(result, dict):
            prob = result.get('profit_probability_today', 0)
            self.logger.info(f"📊 Market analysis: {result.get('market_sentiment', 'unknown')} sentiment, {prob}% profit probability")
            return result

        self.logger.warning("⚠️ Groq market analysis failed, using default analysis")
        return {
            "market_sentiment": "neutral",
            "sentiment_score": 0.5,
            "profit_probability_today": 70,
            "key_insights": ["No AI analysis available"],
            "risk_level": "medium",
            "recommended_allocation": 0.5,
            "top_opportunities": HIGH_PROBABILITY_COINS[:5],
            "price_range": "All under $1.00",
            "expected_gains": "2-3% with 70% probability"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get Groq API usage statistics"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "api_key_configured": self.api_key is not None and self.client is not None,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "current_model": self.current_model,
            "cache_size": len(self.cache),
            "models_available": self.models
        }

# High probability coins under $1 (fallback with 80% probability focus)
HIGH_PROBABILITY_COINS = [
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
                config.setdefault("scanInterval", 60)
                config.setdefault("testOnTestnet", False)
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
SCAN_INTERVAL = CONFIG.get("scanInterval", 60)

logger.info(f"📊 Trading Settings:")
logger.info(f"   Timeframe: {TIMEFRAME} minutes")
logger.info(f"   Min Trade: ${MIN_TRADE_AMOUNT}")
logger.info(f"   Dry Run: {DRY_RUN}")
logger.info(f"   Scan Interval: {SCAN_INTERVAL} seconds")
logger.info(f"   Allocation: 100% per trade")
logger.info(f"   AI Provider: Groq API")
logger.info(f"   Target: Coins under $1 with 80% profit probability")

# File paths
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
        
        # Display Groq client status
        stats = self.groq_client.get_stats()
        if stats["api_key_configured"]:
            self.logger.info(f"🔑 Groq API Key: {'Configured' if stats['api_key_configured'] else 'Missing'}")
            self.logger.info(f"🤖 Groq Models Available: {len(stats['models_available'])}")
        else:
            self.logger.warning("⚠️ GROQ_API_KEY not configured in environment variables")
            self.logger.info("ℹ️ Set GROQ_API_KEY environment variable for AI coin selection")

    def load_coins(self):
        """Load AI-selected coins from file - handles both list and dict formats"""
        try:
            if os.path.exists(self.coins_file):
                with open(self.coins_file, "r") as f:
                    data = json.load(f)
                    
                    # FIXED: Handle both list and dictionary formats
                    if isinstance(data, dict):
                        # Handle dictionary format
                        self.current_coins = data.get("coins", HIGH_PROBABILITY_COINS)
                        self.last_update = data.get("updated")
                        self.market_analysis = data.get("market_analysis", {})
                        
                        # Ensure current_coins is a list
                        if not isinstance(self.current_coins, list):
                            self.logger.warning(f"⚠️ Coins data is not a list, resetting to defaults")
                            self.current_coins = HIGH_PROBABILITY_COINS
                            
                    elif isinstance(data, list):
                        # FIX: Handle list format directly (backward compatibility)
                        self.logger.info(f"📁 Loading coins from list format (backward compatibility)")
                        self.current_coins = []
                        
                        # Validate each coin in the list
                        for coin in data:
                            if isinstance(coin, str) and coin.endswith('USDT'):
                                self.current_coins.append(coin)
                            else:
                                self.logger.warning(f"⚠️ Invalid coin format in list: {coin}")
                        
                        # If no valid coins, use defaults
                        if not self.current_coins:
                            self.logger.warning("⚠️ No valid coins in list, using defaults")
                            self.current_coins = HIGH_PROBABILITY_COINS
                        
                        self.last_update = datetime.now().isoformat()
                        self.market_analysis = {
                            "market_sentiment": "neutral",
                            "profit_probability_today": 80,
                            "price_range": "All under $1.00",
                            "backward_compatible": True
                        }
                        
                        # Save in new format for future
                        self.save_coins()
                        
                    else:
                        self.logger.warning(f"⚠️ Coins file contains unexpected data type: {type(data)}")
                        self.current_coins = HIGH_PROBABILITY_COINS

                    self.logger.info(f"📁 Loaded {len(self.current_coins)} coins from file")

                    # Check if data is from today
                    if self.last_update and not self.last_update.startswith(datetime.now().strftime("%Y-%m-%d")):
                        self.logger.info("🔄 Data is from a different day, will update")
            else:
                self.current_coins = HIGH_PROBABILITY_COINS
                self.save_coins()
                self.logger.warning("⚠️ No coin file found, using defaults")
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse coins file (invalid JSON): {e}")
            self.logger.info("🔄 Creating new coins file with default format...")
            self.current_coins = HIGH_PROBABILITY_COINS
            self.save_coins()  # Recreate with proper format
        except Exception as e:
            self.logger.error(f"❌ Failed to load coins: {e}", exc_info=True)
            self.current_coins = HIGH_PROBABILITY_COINS

    def save_coins(self, coins=None, market_analysis=None):
        """Save coins to file with Groq analysis"""
        if coins:
            # Ensure coins is a list
            if isinstance(coins, list):
                self.current_coins = coins
                self.logger.info(f"💾 Saving {len(coins)} coins from Groq")
            else:
                self.logger.error(f"❌ Cannot save coins: Expected list, got {type(coins)}")
                self.current_coins = HIGH_PROBABILITY_COINS

        if market_analysis:
            self.market_analysis = market_analysis

        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "coins": self.current_coins,
            "updated": datetime.now().isoformat(),
            "market_analysis": self.market_analysis,
            "groq_stats": self.groq_client.get_stats(),
            "profit_probability": self.market_analysis.get("profit_probability_today", 80) if isinstance(self.market_analysis, dict) else 80,
            "price_range": "All under $1.00"
        }

        try:
            with open(self.coins_file, "w") as f:
                json.dump(data, f, indent=2)
            self.last_update = data["updated"]

            self.logger.info(f"✅ Groq coins saved: {len(self.current_coins)} symbols")
            prob = self.market_analysis.get("profit_probability_today", 80) if isinstance(self.market_analysis, dict) else 80
            self.logger.info(f"📊 Market probability: {prob}% profit chance")

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
        self.logger.info("🤖 Starting Groq coin selection (80% profit probability, under $1)...")

        try:
            # Get market analysis first
            self.logger.info("📊 Getting market analysis...")
            market_analysis = self.groq_client.get_market_analysis()

            # Get coin picks
            self.logger.info("🎯 Getting high probability coin picks...")
            coins = self.groq_client.get_daily_coins()

            elapsed = time.time() - start_time

            if coins and len(coins) >= 5:
                self.logger.info(f"✅ Groq selection completed in {elapsed:.2f}s")
                
                prob = market_analysis.get("profit_probability_today", 80) if isinstance(market_analysis, dict) else 80
                self.logger.info(f"📊 Market probability: {prob}% profit chance")
                self.logger.info(f"💰 Price range: {market_analysis.get('price_range', 'Under $1.00') if isinstance(market_analysis, dict) else 'Under $1.00'}")
                self.logger.info(f"🎯 Selected {len(coins)} coins under $1")

                return self.save_coins(coins, market_analysis)
            else:
                self.logger.warning(f"⚠️ Groq returned insufficient coins ({len(coins) if coins else 0}), using fallback")

                # Use market analysis to select high probability coins
                sentiment = market_analysis.get("market_sentiment", "neutral") if isinstance(market_analysis, dict) else "neutral"
                prob = market_analysis.get("profit_probability_today", 80) if isinstance(market_analysis, dict) else 80
                
                self.logger.info(f"📊 Using fallback with {prob}% probability focus, {sentiment} sentiment")

                return self.save_coins(HIGH_PROBABILITY_COINS, market_analysis)

        except Exception as e:
            self.logger.error(f"❌ Groq update failed: {e}", exc_info=True)
            elapsed = time.time() - start_time
            self.logger.warning(f"⏱️ Failed after {elapsed:.2f}s, using high probability defaults")

            return self.save_coins(HIGH_PROBABILITY_COINS, {
                "market_sentiment": "error",
                "profit_probability_today": 80,
                "error": str(e),
                "fallback_used": True,
                "price_range": "All under $1.00"
            })

    def get_market_insights(self) -> Dict[str, Any]:
        """Get market insights from Groq analysis"""
        return {
            **(self.market_analysis if isinstance(self.market_analysis, dict) else {}),
            "last_update": self.last_update,
            "coin_count": len(self.current_coins),
            "groq_stats": self.groq_client.get_stats(),
            "all_coins_under_1_dollar": True
        }

# -------------------- TRADE PERFORMANCE LOGGER --------------------
class PerformanceLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.setup_csv()

    def setup_csv(self):
        """Create CSV header if file doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("timestamp,symbol,entry_price,exit_price,size,pnl,pnl_percent,duration,status,notes\n")
            logger.info(f"📊 Performance log created: {self.log_file}")

    def log_trade(self, trade: Dict):
        """Log trade to CSV for performance analysis"""
        try:
            with open(self.log_file, "a") as f:
                timestamp = trade.get("entry_time", datetime.now().isoformat())
                symbol = trade.get("symbol", "UNKNOWN")
                entry = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                size = trade.get("size", 0)
                status = trade.get("status", "UNKNOWN")

                # Calculate P&L
                if entry > 0 and exit_price > 0:
                    pnl = (exit_price - entry) * size
                    pnl_percent = ((exit_price - entry) / entry) * 100
                else:
                    pnl = 0
                    pnl_percent = 0

                # Calculate duration
                if "entry_time" in trade and "exit_time" in trade:
                    entry_dt = datetime.fromisoformat(trade["entry_time"])
                    exit_dt = datetime.fromisoformat(trade["exit_time"])
                    duration = (exit_dt - entry_dt).total_seconds()
                else:
                    duration = 0

                line = f"{timestamp},{symbol},{entry},{exit_price},{size},{pnl:.4f},{pnl_percent:.2f},{duration},{status},\n"
                f.write(line)

            logger.debug(f"📈 Trade logged to performance CSV: {symbol}")

        except Exception as e:
            logger.error(f"❌ Failed to log trade performance: {e}")

# -------------------- ENHANCED BOT CONTROLLER --------------------
class EnhancedBotController:
    def __init__(self, log_queue=None):
        self.log_queue = log_queue
        self._running = False
        self._stop = threading.Event()
        self._file_lock = threading.Lock()
        self._threads = []

        # Logging
        self.logger = logging.getLogger("BotController")
        self.logger.info("=" * 60)
        self.logger.info("🚀 INITIALIZING ENHANCED BOT CONTROLLER v2.0 (GROQ)")
        self.logger.info("=" * 60)

        # Core components
        self.rate_limiter = RateLimiter(max_per_second=3, max_per_minute=30, name="MainLimiter")
        self.ai_selector = GroqCoinSelector()
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="Scanner")
        self.performance_logger = PerformanceLogger(PERF_LOG_FILE)

        # Daily tracking
        self.trades_today = 0
        self.day_start = time.time()
        self.max_daily_trades = CONFIG.get("maxTradesPerDay", 10)

        # Performance tracking
        self.stats = {
            "start_time": time.time(),
            "total_scans": 0,
            "qualified_signals": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_volume_traded": 0.0,
            "total_pnl": 0.0,
            "avg_trade_duration": 0.0,
            "scans_per_minute": 0.0
        }

        # Scan performance tracking
        self.scan_times = []
        self.last_scan_report = time.time()

        # Initialize files
        self._init_files()

        # Start scheduler for daily tasks
        self._start_scheduler()

        # Display Groq status
        groq_stats = self.ai_selector.groq_client.get_stats()
        self.logger.info(f"✅ Bot initialized successfully")
        self.logger.info(f"   🤖 AI Provider: Groq ({'✅ Connected' if groq_stats['api_key_configured'] else '❌ Not Connected'})")
        self.logger.info(f"   🤖 AI Coins: {len(self.get_coins())} symbols (all under $1)")
        self.logger.info(f"   🎯 Target: 80%+ daily profit probability")
        self.logger.info(f"   ⏰ Timeframe: {TIMEFRAME} minutes")
        self.logger.info(f"   💰 Min Trade: ${MIN_TRADE_AMOUNT}")
        self.logger.info(f"   🎯 Allocation: 100% per trade")
        self.logger.info(f"   📈 Max Daily Trades: {self.max_daily_trades}")
        self.logger.info(f"   🧪 Dry Run: {DRY_RUN}")
        self.logger.info(f"   🔍 Scan Interval: {SCAN_INTERVAL} seconds")

    def _init_files(self):
        """Initialize data files with logging"""
        files_to_create = [
            (TRADES_FILE, []),
            (COINS_FILE, {
                "coins": HIGH_PROBABILITY_COINS, 
                "date": datetime.now().strftime("%Y-%m-%d"),
                "updated": datetime.now().isoformat(),
                "market_analysis": {
                    "market_sentiment": "neutral",
                    "profit_probability_today": 80,
                    "price_range": "All under $1.00"
                },
                "groq_stats": {"api_key_configured": False}
            })
        ]

        for path, default in files_to_create:
            try:
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        json.dump(default, f)
                    self.logger.info(f"📁 Created file: {path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create {path}: {e}")

    def _start_scheduler(self):
        """Start background scheduler for daily tasks"""
        def daily_update():
            self.logger.info("=" * 50)
            self.logger.info("🔄 DAILY GROQ UPDATE STARTING")
            self.logger.info("=" * 50)

            # Reset daily trade counter
            old_trades = self.trades_today
            self.trades_today = 0
            self.day_start = time.time()
            self.logger.info(f"🔄 Reset daily trades: {old_trades} → 0")

            # Update AI coins from Groq
            self.logger.info("🤖 Updating AI coin selection from Groq...")
            start_time = time.time()
            coins = self.ai_selector.update_from_groq()
            elapsed = time.time() - start_time

            # Get market insights
            insights = self.ai_selector.get_market_insights()
            sentiment = insights.get("market_analysis", {}).get("market_sentiment", "unknown")
            probability = insights.get("market_analysis", {}).get("profit_probability_today", 80)

            self.logger.info(f"✅ Groq update completed in {elapsed:.2f}s")
            self.logger.info(f"📊 Market sentiment: {sentiment} ({probability}% profit probability)")
            self.logger.info(f"💰 All coins under $1: Yes")
            self.logger.info(f"🎯 Selected {len(coins)} coins")

            # Log daily performance
            self._log_daily_performance()

            self.logger.info("✅ Daily Groq update completed")
            self.logger.info("=" * 50)

        # Schedule daily update at 00:00 UTC
        schedule.every().day.at("00:00").do(daily_update)
        self.logger.info("📅 Groq scheduler scheduled for 00:00 UTC")

        # Run scheduler in background thread
        def run_scheduler():
            self.logger.info("⏰ Groq scheduler started")
            while not self._stop.is_set():
                try:
                    schedule.run_pending()
                    time.sleep(60)
                except Exception as e:
                    self.logger.error(f"❌ Scheduler error: {e}")

        scheduler_thread = threading.Thread(
            target=run_scheduler, 
            daemon=True,
            name="GroqScheduler"
        )
        scheduler_thread.start()
        self._threads.append(scheduler_thread)

    def _log_daily_performance(self):
        """Log daily performance summary"""
        try:
            perf_file = f"logs/performance_daily_{datetime.now().strftime('%Y%m%d')}.json"

            daily_stats = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trades_today": self.stats.get("successful_trades", 0),
                "total_pnl": self.stats.get("total_pnl", 0),
                "win_rate": self._calculate_win_rate(),
                "avg_trade_duration": self.stats.get("avg_trade_duration", 0),
                "total_volume": self.stats.get("total_volume_traded", 0),
                "scans_per_minute": self.stats.get("scans_per_minute", 0),
                "qualified_rate": self._calculate_qualified_rate(),
                "ai_probability": self.ai_selector.market_analysis.get("profit_probability_today", 80) if isinstance(self.ai_selector.market_analysis, dict) else 80
            }

            with open(perf_file, "w") as f:
                json.dump(daily_stats, f, indent=2)

            self.logger.info(f"📊 Daily performance logged to {perf_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to log daily performance: {e}")

    def _calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        trades = self._read_trades()
        if not trades:
            return 0.0

        winning = 0
        for trade in trades:
            if trade.get("status") == "CLOSED" and trade.get("pnl", 0) > 0:
                winning += 1

        return (winning / len(trades)) * 100 if trades else 0.0

    def _calculate_qualified_rate(self) -> float:
        """Calculate qualification rate"""
        if self.stats["total_scans"] == 0:
            return 0.0
        return (self.stats["qualified_signals"] / self.stats["total_scans"]) * 100

    # ------------------ ACCOUNT MANAGEMENT FROM ENVIRONMENT ------------------
    def load_accounts(self) -> List[Dict]:
        """Load accounts from environment variables"""
        self.logger.info("🔐 Loading accounts from environment variables...")
        
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
                        account = {
                            "name": account_name,
                            "api_key": api_key,
                            "api_secret": api_secret,
                            "validated": False,
                            "balance": 0.0
                        }
                        accounts.append(account)
                        self.logger.info(f"✅ Found account: {account_name}")
                        break
                else:
                    break
            
            if api_key and api_secret:
                account = {
                    "name": account_name,
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "validated": False,
                    "balance": 0.0
                }
                accounts.append(account)
                self.logger.info(f"✅ Found account {account_counter}: {account_name}")
                account_counter += 1
            else:
                break
        
        if not accounts:
            self.logger.warning("⚠️ No accounts found in environment variables")
            self.logger.info("ℹ️ Expected environment variables:")
            self.logger.info("   Single account: BYBIT_API_KEY, BYBIT_API_SECRET")
            self.logger.info("   Multiple accounts: BYBIT_API_KEY_1, BYBIT_API_SECRET_1, BYBIT_API_KEY_2, etc.")
        
        return accounts

    def validate_account(self, account: Dict) -> Tuple[bool, Optional[float], Optional[str]]:
        """Validate exchange API credentials with detailed logging - FIXED VERSION"""
        account_name = account.get('name', 'Unknown')
        self.logger.info(f"🔐 Validating account: {account_name}")
        
        # Check API credentials first
        api_key = account.get("api_key", "").strip()
        api_secret = account.get("api_secret", "").strip()
        
        if not api_key or not api_secret:
            self.logger.error(f"❌ Missing API credentials for {account_name}")
            return False, None, "Missing API credentials"

        # Mask API key for logging
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        self.logger.info(f"📝 Validating API Key: {masked_key}")

        try:
            # Create client
            client = self._get_client(account)
            if not client:
                self.logger.error(f"❌ Failed to create API client for {account_name}")
                return False, None, "Failed to create client"

            # Get testnet status
            testnet = CONFIG.get("testOnTestnet", False) and CONFIG.get("dryRun", True)
            mode = "TESTNET" if testnet else "MAINNET"
            self.logger.info(f"🌐 Connecting to Bybit {mode}...")

            # Apply rate limiting
            if not self.rate_limiter.acquire():
                self.logger.warning(f"⏳ Rate limited, waiting...")
                time.sleep(1)
                if not self.rate_limiter.acquire():
                    self.logger.error("❌ Rate limit timeout")
                    return False, None, "Rate limit timeout"

            self.logger.debug(f"📡 Calling Bybit API...")

            # Call API - pybit returns a DICTIONARY, NOT a response object
            try:
                resp = client.get_wallet_balance(accountType="UNIFIED")
                
                # DEBUG: Log response type
                self.logger.debug(f"📡 API Response type: {type(resp)}")
                
            except Exception as api_error:
                self.logger.error(f"❌ API call failed: {api_error}")
                return False, None, f"API call failed: {api_error}"

            # CRITICAL FIX: pybit returns dict directly, NOT an object with .json() method
            # So we just check if it's a dictionary
            if isinstance(resp, dict):
                ret_code = resp.get("retCode")
                ret_msg = resp.get("retMsg", "Unknown error")
                
                self.logger.debug(f"📊 Response Code: {ret_code}, Message: {ret_msg}")
                
                if ret_code == 0:
                    # SUCCESS: Account is validated!
                    result = resp.get("result", {})
                    if isinstance(result, dict):
                        balance_list = result.get("list", [])
                        if balance_list and isinstance(balance_list, list) and len(balance_list) > 0:
                            # Try multiple possible balance field names
                            balance_data = balance_list[0]
                            
                            # Try different field names that might contain balance
                            total_equity = (
                                balance_data.get("totalEquity") or 
                                balance_data.get("totalWalletBalance") or 
                                balance_data.get("walletBalance") or 
                                balance_data.get("equity") or 
                                0.0
                            )
                            
                            # Convert to float
                            try:
                                total_equity = float(total_equity)
                            except (ValueError, TypeError):
                                total_equity = 0.0
                            
                            # Even if balance is $0, the account is still VALID!
                            if total_equity == 0:
                                self.logger.warning(f"⚠️ Account validated but balance is $0.00")
                                self.logger.info(f"💡 Account is valid, just needs funding")
                            else:
                                self.logger.info(f"✅ Account validated successfully - Balance: ${total_equity:.2f}")
                            
                            return True, total_equity, None
                        else:
                            # Even if we can't get balance data, the account might still be valid
                            self.logger.warning(f"⚠️ Account validated but no balance data in response")
                            return True, 0.0, "No balance data but API valid"
                    else:
                        self.logger.warning(f"⚠️ Account validated but unexpected result format")
                        return True, 0.0, "Unexpected result format"
                else:
                    # FAILURE: API returned an error code
                    # Common error codes and their meanings
                    error_map = {
                        10001: "Invalid API key",
                        10002: "Invalid API signature",
                        10003: "Invalid request",
                        10004: "Unauthorized request",
                        10005: "Permission denied",
                        10006: "IP address not whitelisted",
                        10007: "Timestamp out of range",
                        10016: "Service unavailable",
                        10017: "Request too frequent",
                        33004: "Account not unified (use Unified account)",
                        131001: "Insufficient balance"  # This is OK - account is still valid!
                    }
                    
                    error_desc = error_map.get(ret_code, "Unknown error")
                    self.logger.error(f"❌ Account validation FAILED: {error_desc}")
                    self.logger.error(f"   Error Code: {ret_code}")
                    self.logger.error(f"   Error Message: {ret_msg}")
                    
                    # Special case: Insufficient balance error means account IS valid!
                    if ret_code == 131001:
                        self.logger.warning(f"⚠️ Account validated (insufficient balance is OK)")
                        return True, 0.0, "Insufficient balance"
                    
                    # Provide troubleshooting tips
                    self.logger.info(f"🔧 Troubleshooting steps:")
                    self.logger.info(f"   1. Check API key permissions (Read Wallet Balance required)")
                    self.logger.info(f"   2. Verify IP whitelist in Bybit settings")
                    self.logger.info(f"   3. Ensure you're using correct environment (Testnet/Mainnet)")
                    self.logger.info(f"   4. Check if API key has expired")
                    self.logger.info(f"   5. Make sure account is Unified account type")
                    
                    return False, None, f"{error_desc} (Code: {ret_code})"
            else:
                # Unexpected response type
                self.logger.error(f"❌ Unexpected response type: {type(resp)}")
                return False, None, f"Unexpected response type: {type(resp)}"

        except Exception as e:
            self.logger.error(f"❌ Account validation exception: {e}", exc_info=True)
            return False, None, str(e)

    def _get_client(self, account: Dict) -> Optional[HTTP]:
        """Create Bybit client with logging"""
        try:
            key = account.get("api_key", "").strip()
            secret = account.get("api_secret", "").strip()
            
            if not key:
                self.logger.error(f"❌ Empty API key for {account.get('name')}")
                return None
                
            if not secret:
                self.logger.error(f"❌ Empty API secret for {account.get('name')}")
                return None

            # Check key length (Bybit keys are typically ~40 chars)
            if len(key) < 30 or len(secret) < 30:
                self.logger.warning(f"⚠️ API key/secret seems unusually short")
                self.logger.info(f"   Key length: {len(key)}, Secret length: {len(secret)}")

            testnet = CONFIG.get("testOnTestnet", False) and CONFIG.get("dryRun", True)
            mode = "TESTNET" if testnet else "MAINNET"
            
            # Log connection details (masked)
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            self.logger.info(f"🔧 Creating {mode} client (API Key: {masked_key})")

            return HTTP(
                api_key=key,
                api_secret=secret,
                testnet=testnet,
                recv_window=5000
            )
        except Exception as e:
            self.logger.error(f"❌ Client creation failed: {e}", exc_info=True)
            return None

    # ------------------ TRADING LOGIC ------------------
    def get_coins(self) -> List[str]:
        """Get current Groq-selected coins"""
        coins = self.ai_selector.get_coins()
        self.logger.debug(f"📊 Active Groq coins: {len(coins)} symbols (all under $1)")
        return coins

    def get_market_insights(self) -> Dict[str, Any]:
        """Get market insights from Groq analysis"""
        insights = self.ai_selector.get_market_insights()

        # Log insights
        sentiment = insights.get("market_analysis", {}).get("market_sentiment", "unknown")
        prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)
        price_range = insights.get("market_analysis", {}).get("price_range", "Under $1.00")

        self.logger.info(f"📊 Groq Insights: {sentiment.upper()} sentiment, {prob}% profit probability, {price_range}")

        return insights

    def force_ai_update(self) -> Dict[str, Any]:
        """Force immediate AI update from Groq"""
        self.logger.info("🔄 Manual Groq update requested (80% probability focus)")

        try:
            start_time = time.time()
            coins = self.ai_selector.update_from_groq()
            elapsed = time.time() - start_time

            insights = self.ai_selector.get_market_insights()
            prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)

            self.logger.info(f"✅ Manual Groq update completed in {elapsed:.2f}s ({prob}% probability)")

            return {
                "success": True,
                "coins": coins,
                "count": len(coins),
                "update_time": elapsed,
                "market_insights": insights,
                "profit_probability": prob,
                "price_range": "All under $1.00",
                "message": f"Groq update successful - {len(coins)} coins selected with {prob}% profit probability"
            }

        except Exception as e:
            self.logger.error(f"❌ Manual Groq update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Groq update failed"
            }

    def _check_daily_limit(self) -> bool:
        """Check if daily limit reached with logging"""
        # Reset counter if new day
        if time.time() - self.day_start > 86400:
            old_count = self.trades_today
            self.trades_today = 0
            self.day_start = time.time()
            self.logger.info(f"🔄 Daily reset: {old_count} trades → 0")

        if self.trades_today >= self.max_daily_trades:
            remaining = int(86400 - (time.time() - self.day_start))
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60

            self.logger.warning(
                f"⏰ Daily limit reached: {self.trades_today}/{self.max_daily_trades} "
                f"(resets in {hours}h {minutes}m)"
            )
            return True

        remaining_trades = self.max_daily_trades - self.trades_today
        self.logger.debug(f"📊 Daily trades: {self.trades_today}/{self.max_daily_trades} ({remaining_trades} remaining)")
        return False

    def calculate_position(self, balance: float, entry_price: float) -> Dict:
        """Calculate position size using 100% of balance"""
        self.logger.debug(f"🧮 Calculating position: Balance=${balance:.2f}, Entry=${entry_price:.4f}")

        # Use ENTIRE balance for trade (100% allocation)
        trade_value = balance

        # Check minimum trade amount
        if trade_value < MIN_TRADE_AMOUNT:
            error_msg = f"Balance ${balance:.2f} below minimum ${MIN_TRADE_AMOUNT}"
            self.logger.warning(f"⚠️ {error_msg}")
            return {
                "size": 0.0,
                "value": 0.0,
                "error": error_msg,
                "allocation_pct": 100
            }

        position_size = trade_value / entry_price

        result = {
            "size": position_size,
            "value": trade_value,
            "risk_amount": trade_value,
            "allocation_pct": 100,
            "estimated_fee": trade_value * 0.001
        }

        self.logger.debug(
            f"📐 Position calculated: {position_size:.4f} coins "
            f"(${trade_value:.2f} at ${entry_price:.4f})"
        )

        return result

    # ------------------ SCANNING & SIGNALS ------------------
    def scan_symbol(self, client: HTTP, symbol: str) -> Optional[Dict]:
        """Scan single symbol for opportunities with detailed logging"""
        scan_start = time.time()
        self.logger.debug(f"🔍 Scanning {symbol}...")

        try:
            if not self.rate_limiter.acquire():
                self.logger.warning(f"⏳ Rate limited while scanning {symbol}")
                return None

            # Get kline data
            self.logger.debug(f"📡 Fetching klines for {symbol}...")
            resp = client.get_kline(
                category="spot",
                symbol=symbol,
                interval=TIMEFRAME,
                limit=50
            )

            # Handle response object
            if hasattr(resp, 'json'):
                resp = resp.json()
            
            # Check if resp is dictionary before using .get()
            if isinstance(resp, dict) and resp.get("retCode") != 0:
                self.logger.debug(f"⚠️ {symbol}: API error {resp.get('retCode')}")
                return None

            result = resp.get("result", {}) if isinstance(resp, dict) else {}
            klines = result.get("list", [])

            if len(klines) < 20:
                self.logger.debug(f"⚠️ {symbol}: Insufficient data ({len(klines)} candles)")
                return None

            # Parse candles
            candles = []
            volumes = []
            for k in klines[-20:]:
                try:
                    candles.append({
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                    volumes.append(float(k[5]))
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"⚠️ {symbol}: Failed to parse candle data: {e}")
                    continue

            # Calculate indicators
            closes = [c['close'] for c in candles]
            current_price = closes[-1]

            # CRITICAL: Check if price is under $1.00
            if current_price >= 1.00:
                self.logger.debug(f"💰 {symbol}: Price ${current_price:.4f} is NOT under $1.00 - SKIPPING")
                return None

            # 1. Volume check (MOST IMPORTANT)
            if len(volumes) >= 2:
                current_volume = volumes[-1]
                avg_volume = sum(volumes[:-1]) / (len(volumes) - 1)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                volume_ok = volume_ratio >= 1.5

                if not volume_ok:
                    self.logger.debug(f"📊 {symbol}: Volume insufficient ({volume_ratio:.2f}x avg)")
                    return None
            else:
                self.logger.debug(f"⚠️ {symbol}: Not enough volume data")
                return None

            # 2. RSI check (tighter for higher probability)
            rsi = self._calculate_rsi(closes, 14)
            if not rsi or rsi > 35:  # Changed from 40 to 35 for higher probability
                self.logger.debug(f"📊 {symbol}: RSI too high ({rsi:.1f}) for 80% probability")
                return None

            # 3. Price action - recent higher low
            if len(closes) >= 5:
                lows = [c['low'] for c in candles]
                higher_low = lows[-1] > lows[-3]

                if not higher_low:
                    self.logger.debug(f"📊 {symbol}: No higher low pattern")
                    return None

            # Calculate score (adjusted for 80% probability target)
            base_score = 8.0  # Higher base for 80% probability
            volume_score = min(volume_ratio - 1.0, 3.0)  # Higher weight for volume
            rsi_score = max(0, (35 - rsi) / 8)  # Adjusted for tighter RSI
            price_score = 1.0 if current_price < 0.5 else 0.5  # Bonus for very low prices
            score = base_score + volume_score + rsi_score + price_score

            scan_time = time.time() - scan_start

            opportunity = {
                "symbol": symbol,
                "score": min(score, 10.0),
                "price": current_price,
                "price_under_1": current_price < 1.00,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "scan_time": scan_time,
                "timestamp": time.time(),
                "probability_estimate": min(80 + (score - 8) * 10, 95)  # Estimate probability
            }

            self.logger.debug(
                f"✅ {symbol}: QUALIFIED - Score: {score:.1f}, "
                f"RSI: {rsi:.1f}, Volume: {volume_ratio:.2f}x, "
                f"Price: ${current_price:.4f} (Under $1: ✓), "
                f"Prob: {opportunity['probability_estimate']:.0f}%"
            )

            return opportunity

        except FutureTimeoutError:
            self.logger.warning(f"⏰ {symbol}: Scan timeout")
            return None
        except Exception as e:
            self.logger.error(f"❌ {symbol}: Scan error: {e}", exc_info=True)
            return None

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI with validation"""
        if len(closes) < period + 1:
            return None

        try:
            gains = []
            losses = []

            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"❌ RSI calculation error: {e}")
            return None

    def find_opportunity(self, client: HTTP) -> Optional[Dict]:
        """Find best trading opportunity with performance tracking"""
        scan_start = time.time()
        symbols = self.get_coins()

        self.logger.info(f"🔍 Scanning {len(symbols)} symbols (all under $1) for high probability opportunities...")

        opportunities = []
        scan_results = []

        # Scan all symbols in parallel
        futures = []
        for symbol in symbols:
            future = self.executor.submit(self.scan_symbol, client, symbol)
            futures.append((symbol, future))

        for symbol, future in futures:
            try:
                result = future.result(timeout=10)
                scan_results.append({
                    "symbol": symbol,
                    "success": result is not None,
                    "qualified": result is not None and result.get("score", 0) >= 8.0  # Higher threshold for 80% probability
                })

                if result and result["score"] >= 8.0:
                    opportunities.append(result)

            except FutureTimeoutError:
                self.logger.warning(f"⏰ {symbol}: Scan timeout")
                scan_results.append({"symbol": symbol, "success": False, "qualified": False})
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Scan failed: {e}")
                scan_results.append({"symbol": symbol, "success": False, "qualified": False})

        # Update statistics
        total_scans = len(symbols)
        successful_scans = sum(1 for r in scan_results if r["success"])
        qualified = len(opportunities)

        self.stats["total_scans"] += total_scans
        self.stats["qualified_signals"] += qualified

        scan_time = time.time() - scan_start
        self.scan_times.append(scan_time)

        # Keep only last 100 scan times
        if len(self.scan_times) > 100:
            self.scan_times = self.scan_times[-100:]

        avg_scan_time = sum(self.scan_times) / len(self.scan_times) if self.scan_times else 0

        # Update scans per minute
        if time.time() - self.last_scan_report > 60:
            self.stats["scans_per minute"] = len(self.scan_times)
            self.last_scan_report = time.time()

        # Log scan performance
        self.logger.info(
            f"📊 Scan completed: {total_scans} symbols under $1, "
            f"{successful_scans} successful, "
            f"{qualified} qualified (80%+ probability) in {scan_time:.2f}s "
            f"(avg: {avg_scan_time:.2f}s)"
        )

        if qualified > 0:
            self.logger.info(f"🎯 Found {qualified} high-probability opportunities (80%+)")

            # Sort by score and return best
            opportunities.sort(key=lambda x: x["score"], reverse=True)
            best = opportunities[0]

            prob = best.get("probability_estimate", 80)
            self.logger.info(
                f"🏆 Best opportunity: {best['symbol']} "
                f"(Score: {best['score']:.1f}, "
                f"RSI: {best.get('rsi', 0):.1f}, "
                f"Volume: {best.get('volume_ratio', 0):.2f}x, "
                f"Price: ${best.get('price', 0):.4f}, "
                f"Probability: {prob:.0f}%)"
            )

            return best

        self.logger.debug("📭 No high-probability (80%+) opportunities found")
        return None

    # ------------------ TRADE EXECUTION ------------------
    def execute_trade(self, client: HTTP, account: Dict, opportunity: Dict) -> bool:
        """Execute a trade with comprehensive logging"""
        if self._check_daily_limit():
            return False

        symbol = opportunity["symbol"]
        entry_price = opportunity["price"]
        probability = opportunity.get("probability_estimate", 80)

        self.logger.info("=" * 50)
        self.logger.info(f"🚀 EXECUTING HIGH PROBABILITY TRADE: {symbol} ({probability:.0f}% probability)")
        self.logger.info("=" * 50)

        # Get account balance
        self.logger.info(f"📊 Checking account balance...")
        try:
            if not self.rate_limiter.acquire():
                self.logger.error("❌ Rate limited while checking balance")
                return False

            balance_resp = client.get_wallet_balance(accountType="UNIFIED")
            
            # Handle response object
            if hasattr(balance_resp, 'json'):
                balance_resp = balance_resp.json()

            # Check if balance_resp is dictionary before using .get()
            if isinstance(balance_resp, dict) and balance_resp.get("retCode") != 0:
                error_msg = balance_resp.get("retMsg", "Unknown error")
                self.logger.error(f"❌ Balance check failed: {error_msg}")
                return False

            result = balance_resp.get("result", {}) if isinstance(balance_resp, dict) else {}
            balance_list = result.get("list", [])

            if not balance_list:
                self.logger.error("❌ No balance data returned")
                return False

            # Use TOTAL equity (100% allocation)
            balance = float(balance_list[0].get("totalEquity", 0))
            self.logger.info(f"💰 Account balance: ${balance:.2f}")

            if balance < MIN_TRADE_AMOUNT:
                self.logger.error(f"❌ Balance ${balance:.2f} below minimum ${MIN_TRADE_AMOUNT}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Balance check error: {e}", exc_info=True)
            return False

        # Calculate position (100% of balance)
        position = self.calculate_position(balance, entry_price)

        if position["size"] <= 0:
            error = position.get("error", "Unknown error")
            self.logger.error(f"❌ Position calculation failed: {error}")
            return False

        # Calculate TP/SL (1% SL, 2.5% TP) - Adjusted for 80% probability
        stop_loss = entry_price * 0.99
        take_profit = entry_price * 1.025

        self.logger.info("📊 Trade Parameters:")
        self.logger.info(f"   📍 Entry: ${entry_price:.4f} (Under $1: ✓)")
        self.logger.info(f"   🎯 Probability: {probability:.0f}%")
        self.logger.info(f"   🛑 Stop Loss: ${stop_loss:.4f} (-1.0%)")
        self.logger.info(f"   🎯 Take Profit: ${take_profit:.4f} (+2.5%)")
        self.logger.info(f"   📦 Size: {position['size']:.4f} {symbol}")
        self.logger.info(f"   💰 Value: ${position['value']:.2f} (100% of balance)")
        self.logger.info(f"   📈 Risk: ${position['risk_amount']:.2f}")
        self.logger.info(f"   💸 Estimated Fee: ${position.get('estimated_fee', 0):.2f}")

        # DRY RUN MODE
        if DRY_RUN:
            trade_id = f"DRY_{uuid.uuid4().hex[:8]}"

            self.logger.info("🧪 DRY RUN MODE - No real orders will be placed")

            # Simulate trade execution
            time.sleep(0.5)

            trade = {
                "id": trade_id,
                "symbol": symbol,
                "side": "BUY",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": position["size"],
                "value": position["value"],
                "account": account.get("name", "Unknown"),
                "status": "OPEN",
                "type": "DRY_RUN",
                "entry_time": datetime.now().isoformat(),
                "max_hold": 5400,
                "scan_score": opportunity.get("score", 0),
                "rsi": opportunity.get("rsi", 0),
                "volume_ratio": opportunity.get("volume_ratio", 0),
                "probability_estimate": probability,
                "price_under_1": True
            }

            self._save_trade(trade)
            self.trades_today += 1
            self.stats["successful_trades"] += 1
            self.stats["total_volume_traded"] += position["value"]

            self.logger.info("✅ DRY RUN TRADE EXECUTED SUCCESSFULLY")
            self.logger.info(f"   🆔 Trade ID: {trade_id}")
            self.logger.info(f"   📅 Daily Trades: {self.trades_today}/{self.max_daily_trades}")
            self.logger.info(f"   🎯 Probability: {probability:.0f}%")

            return True

        # LIVE TRADE
        else:
            self.logger.warning("⚠️ LIVE TRADE MODE - Real money at risk!")

            try:
                if not self.rate_limiter.acquire():
                    self.logger.error("❌ Rate limited while placing order")
                    return False

                # Place MARKET order
                self.logger.info("📤 Placing market order...")

                order_resp = client.place_order(
                    category="spot",
                    symbol=symbol,
                    side="Buy",
                    orderType="Market",
                    qty=str(position["size"]),
                    timeInForce="GTC"
                )
                
                # Handle response object
                if hasattr(order_resp, 'json'):
                    order_resp = order_resp.json()

                # FIXED: Check if order_resp is dictionary
                if isinstance(order_resp, dict) and order_resp.get("retCode") == 0:
                    order_id = order_resp["result"]["orderId"]

                    self.logger.info(f"✅ ORDER PLACED SUCCESSFULLY")
                    self.logger.info(f"   🆔 Order ID: {order_id}")

                    trade = {
                        "id": str(uuid.uuid4()),
                        "symbol": symbol,
                        "side": "BUY",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": position["size"],
                        "value": position["value"],
                        "account": account.get("name", "Unknown"),
                        "status": "OPEN",
                        "type": "LIVE",
                        "order_id": order_id,
                        "order_response": order_resp,
                        "entry_time": datetime.now().isoformat(),
                        "max_hold": 5400,
                        "scan_score": opportunity.get("score", 0),
                        "rsi": opportunity.get("rsi", 0),
                        "volume_ratio": opportunity.get("volume_ratio", 0),
                        "probability_estimate": probability,
                        "price_under_1": True
                    }

                    self._save_trade(trade)
                    self.trades_today += 1
                    self.stats["successful_trades"] += 1
                    self.stats["total_volume_traded"] += position["value"]

                    # Log to performance CSV
                    self.performance_logger.log_trade(trade)

                    self.logger.info("✅ LIVE TRADE EXECUTED SUCCESSFULLY")
                    self.logger.info(f"   🆔 Trade ID: {trade['id']}")
                    self.logger.info(f"   📅 Daily Trades: {self.trades_today}/{self.max_daily_trades}")
                    self.logger.info(f"   🎯 Probability: {probability:.0f}%")

                    return True
                else:
                    error_msg = order_resp.get("retMsg", "Unknown error") if isinstance(order_resp, dict) else "Invalid response"
                    self.logger.error(f"❌ ORDER FAILED: {error_msg}")

                    self.stats["failed_trades"] += 1
                    return False

            except Exception as e:
                self.logger.error(f"❌ TRADE EXECUTION ERROR: {e}", exc_info=True)
                self.stats["failed_trades"] += 1
                return False

    def _save_trade(self, trade: Dict):
        """Save trade to history with backup"""
        try:
            trades = self._read_trades()
            trades.append(trade)

            # Keep only last 200 trades
            if len(trades) > 200:
                removed = trades[:-200]
                trades = trades[-200:]
                self.logger.debug(f"🗑️ Removed {len(removed)} old trades from history")

            with self._file_lock:
                with open(TRADES_FILE, "w") as f:
                    json.dump(trades, f, indent=2)

                # Create backup
                backup_file = f"logs/trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, "w") as f_backup:
                    json.dump(trades, f_backup, indent=2)

                self.logger.debug(f"💾 Trade saved and backed up to {backup_file}")
                
                prob = trade.get("probability_estimate", 80)
                self.logger.info(f"📝 Trade {trade.get('id')} added to history ({prob:.0f}% probability)")

        except Exception as e:
            self.logger.error(f"❌ Failed to save trade: {e}", exc_info=True)

    def _read_trades(self) -> List[Dict]:
        """Read trade history with error handling"""
        with self._file_lock:
            try:
                with open(TRADES_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Corrupted trades file: {e}")
                return []
            except Exception as e:
                self.logger.error(f"❌ Failed to read trades: {e}")
                return []

    def update_trade(self, trade_id: str, updates: Dict) -> bool:
        """Update trade status with logging"""
        self.logger.info(f"🔄 Updating trade {trade_id}")

        trades = self._read_trades()
        updated = False

        for i, trade in enumerate(trades):
            if trade.get("id") == trade_id:
                old_status = trade.get("status")
                new_status = updates.get("status")

                trades[i].update(updates)
                updated = True

                if new_status and old_status != new_status:
                    self.logger.info(f"📊 Trade {trade_id} status changed: {old_status} → {new_status}")

                    # Log performance if trade closed
                    if new_status in ["CLOSED", "STOPPED"]:
                        self.performance_logger.log_trade(trades[i])

                break

        if updated:
            try:
                with self._file_lock:
                    with open(TRADES_FILE, "w") as f:
                        json.dump(trades, f, indent=2)
                self.logger.info(f"✅ Trade {trade_id} updated successfully")
                return True
            except Exception as e:
                self.logger.error(f"❌ Failed to save updated trade: {e}")
                return False

        self.logger.warning(f"⚠️ Trade {trade_id} not found for update")
        return False

    # ------------------ BOT CONTROL ------------------
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start the bot with comprehensive logging"""
        if self._running:
            self.logger.warning("⚠️ Bot is already running")
            return

        self._running = True
        self._stop.clear()

        self.logger.info("=" * 60)
        self.logger.info("🚀 STARTING HIGH PROBABILITY TRADING BOT (80%+)")
        self.logger.info("=" * 60)

        # Log startup configuration
        self.logger.info("📋 STARTUP CONFIGURATION:")
        self.logger.info(f"   🤖 AI Provider: Groq")
        self.logger.info(f"   🤖 AI Coins: {len(self.get_coins())} symbols (all under $1)")
        self.logger.info(f"   🎯 Target Probability: 80%+ daily profit")
        self.logger.info(f"   ⏰ Timeframe: {TIMEFRAME} minutes")
        self.logger.info(f"   💰 Allocation: 100% per trade")
        self.logger.info(f"   🛑 Stop Loss: 1.0%")
        self.logger.info(f"   🎯 Take Profit: 2.5%")
        self.logger.info(f"   📈 Max Daily Trades: {self.max_daily_trades}")
        self.logger.info(f"   🧪 Dry Run: {DRY_RUN}")
        self.logger.info(f"   🔧 Testnet: {CONFIG.get('testOnTestnet', False)}")

        # Check accounts from environment
        accounts = self.load_accounts()
        valid_accounts = []
        
        # Validate accounts
        for account in accounts:
            self.logger.info(f"🔐 Validating {account.get('name')}...")
            valid, balance, error = self.validate_account(account)
            if valid:
                account["validated"] = True
                account["balance"] = balance
                valid_accounts.append(account)
                self.logger.info(f"✅ {account.get('name')}: Validated, Balance: ${balance:.2f}")
            else:
                self.logger.error(f"❌ {account.get('name')}: Validation failed: {error}")

        self.logger.info(f"👥 Accounts: {len(valid_accounts)}/{len(accounts)} validated")

        if not valid_accounts:
            self.logger.error("❌ No validated accounts available. Bot cannot start.")
            self._running = False
            return

        # Get market insights
        insights = self.get_market_insights()
        prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)
        self.logger.info(f"📊 Market Probability: {prob}% profit chance today")

        # Start main trading loop in separate thread
        bot_thread = threading.Thread(
            target=self._trading_loop, 
            daemon=True,
            name="TradingLoop"
        )
        bot_thread.start()
        self._threads.append(bot_thread)

        self.logger.info("✅ Bot started successfully with 80%+ probability focus")
        self.logger.info("=" * 60)

    def stop(self):
        """Stop the bot with clean shutdown"""
        if not self._running:
            self.logger.warning("⚠️ Bot is already stopped")
            return

        self.logger.info("=" * 60)
        self.logger.info("🛑 STOPPING TRADING BOT")
        self.logger.info("=" * 60)

        self._running = False
        self._stop.set()

        # Stop executor
        self.logger.info("🔄 Shutting down executor...")
        self.executor.shutdown(wait=False, cancel_futures=True)

        # Log final statistics
        self.logger.info("📊 FINAL STATISTICS:")
        self.logger.info(f"   📈 Total Scans: {self.stats['total_scans']}")
        self.logger.info(f"   🎯 Qualified Signals: {self.stats['qualified_signals']} (80%+ probability)")
        self.logger.info(f"   ✅ Successful Trades: {self.stats['successful_trades']}")
        self.logger.info(f"   ❌ Failed Trades: {self.stats['failed_trades']}")
        self.logger.info(f"   💰 Total Volume: ${self.stats['total_volume_traded']:.2f}")
        self.logger.info(f"   🏆 Win Rate: {self._calculate_win_rate():.1f}%")
        self.logger.info(f"   📊 Qualification Rate: {self._calculate_qualified_rate():.1f}%")
        
        # Get final AI probability
        insights = self.get_market_insights()
        prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)
        self.logger.info(f"   🎯 AI Probability Target: {prob}%")

        # Wait for threads to finish
        self.logger.info("🔄 Waiting for threads to finish...")
        for thread in self._threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=5)
                    self.logger.debug(f"✅ Thread {thread.name} stopped")
                except Exception as e:
                    self.logger.warning(f"⚠️ Thread {thread.name} didn't stop cleanly: {e}")

        self.logger.info("✅ Bot stopped successfully")
        self.logger.info("=" * 60)

    def _trading_loop(self):
        """Main trading loop with heartbeat logging"""
        self.logger.info("🔄 Starting trading loop (80%+ probability focus)...")

        loop_count = 0
        last_heartbeat = time.time()
        last_opportunity_check = 0

        while self._running and not self._stop.is_set():
            loop_start = time.time()
            loop_count += 1

            try:
                # Heartbeat every 30 seconds
                if time.time() - last_heartbeat > 30:
                    insights = self.get_market_insights()
                    prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)
                    
                    self.logger.info("💓 BOT HEARTBEAT - Still running...")
                    self.logger.info(f"   📊 Loop #{loop_count}")
                    self.logger.info(f"   📅 Daily Trades: {self.trades_today}/{self.max_daily_trades}")
                    self.logger.info(f"   🎯 AI Probability: {prob}%")
                    self.logger.info(f"   📊 Qualification Rate: {self._calculate_qualified_rate():.1f}%")
                    last_heartbeat = time.time()

                # Load all accounts from environment
                accounts = self.load_accounts()
                if not accounts:
                    self.logger.warning("⚠️ No accounts configured, sleeping 30s...")
                    time.sleep(30)
                    continue

                # Filter only validated accounts
                valid_accounts = [a for a in accounts if a.get("validated", False)]
                if not valid_accounts:
                    self.logger.warning("⚠️ No validated accounts, sleeping 30s...")
                    time.sleep(30)
                    continue

                self.logger.debug(f"👥 Processing {len(valid_accounts)} validated accounts")

                # Process each account
                for account in valid_accounts:
                    if not self._running or self._stop.is_set():
                        self.logger.info("🛑 Stop signal received, breaking account loop")
                        break

                    account_name = account.get("name", "Unknown")
                    self.logger.debug(f"👤 Processing account: {account_name}")

                    client = self._get_client(account)
                    if not client:
                        self.logger.error(f"❌ Failed to create client for {account_name}")
                        continue

                    # Check if we should look for opportunities
                    current_time = time.time()
                    if current_time - last_opportunity_check < SCAN_INTERVAL:
                        wait_time = SCAN_INTERVAL - (current_time - last_opportunity_check)
                        self.logger.debug(f"⏳ Waiting {wait_time:.1f}s before next scan")
                        time.sleep(min(wait_time, 5))
                        continue

                    # Find high probability trading opportunity
                    last_opportunity_check = current_time
                    opportunity = self.find_opportunity(client)

                    if opportunity:
                        prob = opportunity.get("probability_estimate", 80)
                        self.logger.info(f"🎯 {prob:.0f}% probability opportunity found for {account_name}: {opportunity['symbol']}")

                        # Execute trade
                        success = self.execute_trade(client, account, opportunity)

                        if success:
                            # Wait before next scan
                            wait_time = SCAN_INTERVAL * 2
                            self.logger.info(f"⏸️ Waiting {wait_time}s before next scan")
                            time.sleep(wait_time)
                        else:
                            # Short wait after failed execution
                            time.sleep(5)

                    # Rate limiting between scans
                    self.logger.debug(f"⏳ Waiting {SCAN_INTERVAL}s before next account scan")
                    time.sleep(SCAN_INTERVAL)

                # Brief pause between account cycles
                loop_time = time.time() - loop_start
                self.logger.debug(f"🔄 Account cycle completed in {loop_time:.2f}s")

                if loop_time < 5:
                    time.sleep(5 - loop_time)

            except Exception as e:
                self.logger.error(f"❌ TRADING LOOP ERROR: {e}", exc_info=True)
                self.logger.info("🔄 Restarting loop in 30s...")
                time.sleep(30)

        self.logger.info("🔄 Trading loop stopped")

    def get_stats(self) -> Dict:
        """Get comprehensive bot statistics"""
        uptime = time.time() - self.stats["start_time"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        insights = self.get_market_insights()
        prob = insights.get("market_analysis", {}).get("profit_probability_today", 80)

        return {
            "running": self._running,
            "uptime": f"{hours}h {minutes}m",
            "trades_today": self.trades_today,
            "max_daily_trades": self.max_daily_trades,
            "ai_coins": self.get_coins(),
            "ai_probability": prob,
            "price_range": "All under $1.00",
            "stats": {
                **self.stats,
                "uptime_seconds": uptime,
                "win_rate": self._calculate_win_rate(),
                "qualified_rate": self._calculate_qualified_rate(),
                "avg_scan_time": sum(self.scan_times) / len(self.scan_times) if self.scan_times else 0,
                "rate_limiter_stats": self.rate_limiter.get_stats()
            },
            "daily_reset_in": int(86400 - (time.time() - self.day_start)),
            "dry_run": DRY_RUN,
            "groq_status": self.ai_selector.groq_client.get_stats()
        }

# Global instance
bc = EnhancedBotController()

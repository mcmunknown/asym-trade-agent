import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-5")
    
    # Web Research API Keys
    MESSARI_API_KEY = os.getenv("MESSARI_API_KEY")
    GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
    TOKENTERMINAL_API_KEY = os.getenv("TOKENTERMINAL_API_KEY")
    ARKHAM_API_KEY = os.getenv("ARKHAM_API_KEY")
    SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
    LOOKONCHAIN_API_KEY = os.getenv("LOOKONCHAIN_API_KEY")

    # Trading Configuration
    DEFAULT_TRADE_SIZE = float(os.getenv("DEFAULT_TRADE_SIZE", 3.0))
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", 75))
    MIN_LEVERAGE = int(os.getenv("MIN_LEVERAGE", 50))
    TARGET_ASSETS = os.getenv("TARGET_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,LTCUSDT").split(",")

    # Risk Management
    MAX_POSITION_SIZE_PERCENTAGE = float(os.getenv("MAX_POSITION_SIZE_PERCENTAGE", 2.0))
    STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 2.0))
    TAKE_PROFIT_MULTIPLIER = float(os.getenv("TAKE_PROFIT_MULTIPLIER", 1.5))

    # OpenRouter API Configuration - GPT-5, Claude 4.5, Gemini 2.5
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Bybit API Configuration
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    BYBIT_BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"

    # Web Research Configuration
    ENABLE_WEB_RESEARCH = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"
    RESEARCH_CACHE_TTL = int(os.getenv("RESEARCH_CACHE_TTL", 3600))  # seconds
    RESEARCH_UPDATE_INTERVAL = int(os.getenv("RESEARCH_UPDATE_INTERVAL", 1800))  # 30 minutes
    MESSARI_RATE_LIMIT = int(os.getenv("MESSARI_RATE_LIMIT", 3))  # calls per second
    GLASSNODE_RATE_LIMIT = int(os.getenv("GLASSNODE_RATE_LIMIT", 2))  # calls per second
    DEFIllAMA_RATE_LIMIT = int(os.getenv("DEFIllAMA_RATE_LIMIT", 5))  # calls per second
    TOKENTERMINAL_RATE_LIMIT = int(os.getenv("TOKENTERMINAL_RATE_LIMIT", 3))  # calls per second
    ARKHAM_RATE_LIMIT = int(os.getenv("ARKHAM_RATE_LIMIT", 3))  # calls per second

    # System Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DATA_COLLECTION_INTERVAL = int(os.getenv("DATA_COLLECTION_INTERVAL", 60))  # seconds
    SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", 300))  # seconds (5 minutes)

    # Multi-Model AI Configuration
    ENABLE_MULTI_MODEL = os.getenv("ENABLE_MULTI_MODEL", "true").lower() == "true"
    CONSENSUS_MECHANISM = os.getenv("CONSENSUS_MECHANISM", "majority_vote")  # majority_vote or unanimous
    CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", 2))  # Minimum votes required (2 out of 3)

    # AI Model Configuration
    AI_MODELS_ENABLED = os.getenv("AI_MODELS_ENABLED", "grok4fast,qwen3max,deepseekterminus").split(",")
    GROK4FAST_MODEL = os.getenv("GROK4FAST_MODEL", "x-ai/grok-4-fast")
    QWEN3MAX_MODEL = os.getenv("QWEN3MAX_MODEL", "alibaba/qwen-3-max")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-v3.1-terminus")

    # Model-Specific Settings
    GROK4FAST_TEMPERATURE = float(os.getenv("GROK4FAST_TEMPERATURE", 0.1))
    QWEN3MAX_TEMPERATURE = float(os.getenv("QWEN3MAX_TEMPERATURE", 0.6))
    DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0.3))

    # Performance Tracking
    TRACK_MODEL_PERFORMANCE = os.getenv("TRACK_MODEL_PERFORMANCE", "true").lower() == "true"
    MODEL_PERFORMANCE_LOG_INTERVAL = int(os.getenv("MODEL_PERFORMANCE_LOG_INTERVAL", 100))  # Log every N trades
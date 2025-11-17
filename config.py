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
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", 50))  # FIXED: was 5 (reversed!)
    MIN_LEVERAGE = int(os.getenv("MIN_LEVERAGE", 1))   # FIXED: was 50 (reversed!)

    # Validate leverage bounds
    if not (1 <= MAX_LEVERAGE <= 100):
        raise ValueError(f"MAX_LEVERAGE must be between 1 and 100, got {MAX_LEVERAGE}")
    if not (1 <= MIN_LEVERAGE <= MAX_LEVERAGE):
        raise ValueError(f"MIN_LEVERAGE must be between 1 and MAX_LEVERAGE ({MAX_LEVERAGE}), got {MIN_LEVERAGE}")

    # EXPANDED: 15 liquid crypto assets for more trading opportunities
    TARGET_ASSETS = os.getenv("TARGET_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ARBUSDT,OPUSDT,RENDERUSDT,INJUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,MATICUSDT").split(",")

    # Risk Management
    MAX_POSITION_SIZE_PERCENTAGE = float(os.getenv("MAX_POSITION_SIZE_PERCENTAGE", 2.0))
    STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 2.0))
    TAKE_PROFIT_MULTIPLIER = float(os.getenv("TAKE_PROFIT_MULTIPLIER", 1.5))

    # OpenRouter API Configuration - GPT-5, Claude 4.5, Gemini 2.5
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Bybit API Configuration - PRODUCTION LIVE TRADING
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    BYBIT_BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
    
    # Trading Control - PRODUCTION MODE
    DISABLE_TRADING = os.getenv("DISABLE_TRADING", "false").lower() == "true"

    # Micro-account turbo mode (high-risk, high-reward profile for tiny balances)
    # Enabled by default; can be disabled via MICRO_TURBO_MODE=false in env.
    MICRO_TURBO_MODE = os.getenv("MICRO_TURBO_MODE", "true").lower() == "true"

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
    SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", 60))  # seconds (1 minute - faster for crypto)
    MICRO_TURBO_SIGNAL_INTERVAL = int(os.getenv("MICRO_TURBO_SIGNAL_INTERVAL", 30))  # 30 seconds for balance <$100
    MIN_SIGNAL_INTERVAL = int(os.getenv("MIN_SIGNAL_INTERVAL", 60))  # seconds between signals

    # Trading Fees and Costs
    COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", 0.001))  # 0.1% default commission

    # Signal Quality Thresholds - FIXED: Lowered for aggressive micro trading
    SIGNAL_CONFIDENCE_THRESHOLD = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 0.50))  # was 0.65
    SNR_THRESHOLD = float(os.getenv("SNR_THRESHOLD", 1.0))  # was 1.5

    # Calculus Priority Configuration
    CALCULUS_PRIORITY_MODE = os.getenv("CALCULUS_PRIORITY_MODE", "true").lower() == "true"

    # Force Leverage Settings
    FORCE_LEVERAGE_ENABLED = os.getenv("FORCE_LEVERAGE_ENABLED", "false").lower() == "true"
    FORCE_LEVERAGE_VALUE = float(os.getenv("FORCE_LEVERAGE_VALUE", 50.0))
    FORCE_MARGIN_FRACTION = float(os.getenv("FORCE_MARGIN_FRACTION", 0.4))

    # Risk Management Parameters
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.02))  # 2% per trade
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", 0.10))  # 10% portfolio risk
    MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", 1.5))
    DEFAULT_MAX_POSITIONS = int(os.getenv("DEFAULT_MAX_POSITIONS", 5))
    DEFAULT_MAX_CORRELATION = float(os.getenv("DEFAULT_MAX_CORRELATION", 0.7))
    MIN_KELLY_FRACTION = float(os.getenv("MIN_KELLY_FRACTION", 0.02))
    MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", 0.60))
    B_TRADE_KELLY_FRACTION = float(os.getenv("B_TRADE_KELLY_FRACTION", 0.25))
    A_TRADE_KELLY_FRACTION = float(os.getenv("A_TRADE_KELLY_FRACTION", 0.50))
    CONSECUTIVE_LOSS_KELLY_MULTIPLIER = float(os.getenv("CONSECUTIVE_LOSS_KELLY_MULTIPLIER", 0.5))
    CONSECUTIVE_LOSS_LEVERAGE_MULTIPLIER = float(os.getenv("CONSECUTIVE_LOSS_LEVERAGE_MULTIPLIER", 0.7))
    MICRO_ACCOUNT_MAX_MARGIN_PCT = float(os.getenv("MICRO_ACCOUNT_MAX_MARGIN_PCT", 0.40))
    SMALL_ACCOUNT_MAX_MARGIN_PCT = float(os.getenv("SMALL_ACCOUNT_MAX_MARGIN_PCT", 0.50))
    LARGE_ACCOUNT_MAX_MARGIN_PCT = float(os.getenv("LARGE_ACCOUNT_MAX_MARGIN_PCT", 0.60))
    MARGIN_BUFFER = float(os.getenv("MARGIN_BUFFER", 1.10))
    B_TRADE_TP_MULTIPLIER = float(os.getenv("B_TRADE_TP_MULTIPLIER", 0.7))
    B_TRADE_SL_MULTIPLIER = float(os.getenv("B_TRADE_SL_MULTIPLIER", 0.8))
    MEAN_REVERSION_TP_SIGMA_MULTIPLIER = float(os.getenv("MEAN_REVERSION_TP_SIGMA_MULTIPLIER", 0.6))
    MEAN_REVERSION_MIN_TP_PCT = float(os.getenv("MEAN_REVERSION_MIN_TP_PCT", 0.002))  # FIXED: was 0.005 (0.5%)
    MEAN_REVERSION_SL_SIGMA_MULTIPLIER = float(os.getenv("MEAN_REVERSION_SL_SIGMA_MULTIPLIER", 0.4))
    DIRECTIONAL_TP_SIGMA_MULTIPLIER = float(os.getenv("DIRECTIONAL_TP_SIGMA_MULTIPLIER", 1.5))
    DIRECTIONAL_MIN_TP_PCT = float(os.getenv("DIRECTIONAL_MIN_TP_PCT", 0.004))  # FIXED: was 0.008 (0.8%)
    DIRECTIONAL_SL_SIGMA_MULTIPLIER = float(os.getenv("DIRECTIONAL_SL_SIGMA_MULTIPLIER", 0.75))
    MEAN_REVERSION_MIN_SL_PCT = float(os.getenv("MEAN_REVERSION_MIN_SL_PCT", 0.003))
    DIRECTIONAL_MIN_SL_PCT = float(os.getenv("DIRECTIONAL_MIN_SL_PCT", 0.005))
    RELAXED_MIN_RR = float(os.getenv("RELAXED_MIN_RR", 1.3))

    # Calculus Parameters
    LAMBDA_PARAM = float(os.getenv("LAMBDA_PARAM", 0.1))
    CURVATURE_EDGE_THRESHOLD = float(os.getenv("CURVATURE_EDGE_THRESHOLD", 0.02))

    # Symbol Minimums
    SYMBOL_MIN_ORDER_QTY = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 1.0,
        "ARBUSDT": 1.0,
        "XRPUSDT": 10.0,
        "OPUSDT": 1.0,
        "RENDERUSDT": 1.0,
        "INJUSDT": 1.0,
        "BNBUSDT": 0.01,
        "AVAXUSDT": 0.1,
        "ADAUSDT": 10.0,
        "LINKUSDT": 0.1,
        "DOGEUSDT": 100.0,
        "LTCUSDT": 0.1,
        "MATICUSDT": 10.0
    }

    SYMBOL_MIN_NOTIONALS = {
        "BTCUSDT": 5.0,
        "ETHUSDT": 5.0,
        "SOLUSDT": 5.0,
        "ARBUSDT": 5.0,
        "XRPUSDT": 5.0,
        "OPUSDT": 5.0,
        "RENDERUSDT": 5.0,
        "INJUSDT": 5.0,
        "BNBUSDT": 5.0,
        "AVAXUSDT": 5.0,
        "ADAUSDT": 5.0,
        "LINKUSDT": 5.0,
        "DOGEUSDT": 5.0,
        "LTCUSDT": 5.0,
        "MATICUSDT": 5.0
    }
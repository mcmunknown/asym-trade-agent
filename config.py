"""
Enhanced Configuration for Anne's Calculus Trading System
==========================================================

This configuration file contains all parameters for the complete calculus-based
trading system with real-time data processing, signal generation, and risk management.

Anne's Teaching Approach: Formula → Meaning → Worked Example
- Every parameter is documented with mathematical purpose
- Risk management follows institutional standards
- Performance optimization for high-frequency trading
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class Config:
    """
    Master configuration class for Anne's calculus trading system.
    All parameters are optimized for institutional-grade performance.
    """

    # ===========================================
    # API CONFIGURATION
    # ===========================================

    # Bybit API Keys (required for trading)
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

    # AI Model Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-5")

    # Research API Keys (optional, for fundamental analysis)
    MESSARI_API_KEY = os.getenv("MESSARI_API_KEY")
    GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
    TOKENTERMINAL_API_KEY = os.getenv("TOKENTERMINAL_API_KEY")
    ARKHAM_API_KEY = os.getenv("ARKHAM_API_KEY")
    SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
    LOOKONCHAIN_API_KEY = os.getenv("LOOKONCHAIN_API_KEY")

    # ===========================================
    # BYBIT TRADING CONFIGURATION
    # ===========================================

    # Trading Environment
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    BYBIT_BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
    BYBIT_TLD = os.getenv("BYBIT_TLD", "com")  # Default global endpoint

    # Trading Assets (high-liquidity perpetual futures)
    TARGET_ASSETS = os.getenv(
        "TARGET_ASSETS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,LTCUSDT"
    ).split(",")

    # High-Frequency Trading Configuration
    MAX_ORDERS_PER_SECOND = int(os.getenv("MAX_ORDERS_PER_SECOND", 10))
    ORDER_TIMEOUT_SECONDS = int(os.getenv("ORDER_TIMEOUT_SECONDS", 30))
    BATCH_ORDER_SIZE = int(os.getenv("BATCH_ORDER_SIZE", 5))  # Max orders per batch

    # ===========================================
    # ANNE'S CALCULUS TRADING PARAMETERS
    # ===========================================

    # Exponential Smoothing Parameters
    # Formula: P̂ₜ = λPₜ + (1-λ)P̂ₜ₋₁
    LAMBDA_PARAM = float(os.getenv("LAMBDA_PARAM", 0.75))  # Smoothing factor (0<λ<1)
    MIN_SMOOTHING_WINDOW = int(os.getenv("MIN_SMOOTHING_WINDOW", 20))  # Minimum data points

    # Signal-to-Noise Ratio (SNR) Parameters
    # Formula: SNRᵥ = |vₜ|/σᵥ
    SNR_THRESHOLD = float(os.getenv("SNR_THRESHOLD", 0.7))  # Minimum SNR for valid signals
    SNR_WINDOW_SIZE = int(os.getenv("SNR_WINDOW_SIZE", 14))  # Rolling window for variance calculation

    # Velocity and Acceleration Thresholds
    MIN_VELOCITY_THRESHOLD = float(os.getenv("MIN_VELOCITY_THRESHOLD", 0.0001))  # Minimum velocity for signals
    MIN_ACCELERATION_THRESHOLD = float(os.getenv("MIN_ACCELERATION_THRESHOLD", 0.00001))  # Minimum acceleration
    VELOCITY_SMOOTHING_FACTOR = float(os.getenv("VELOCITY_SMOOTHING_FACTOR", 0.8))  # Central difference smoothing

    # Signal Generation Parameters
    SIGNAL_CONFIDENCE_THRESHOLD = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 0.6))  # Minimum confidence for trading
    MIN_SIGNAL_INTERVAL = int(os.getenv("MIN_SIGNAL_INTERVAL", 30))  # Minimum seconds between signals
    MAX_SIGNAL_AGE = int(os.getenv("MAX_SIGNAL_AGE", 300))  # Maximum signal age in seconds

    # ===========================================
    # KALMAN FILTER PARAMETERS
    # ===========================================

    # State Space Model: sₜ = [P̂ₜ, vₜ, aₜ]ᵀ
    KALMAN_PROCESS_NOISE_PRICE = float(os.getenv("KALMAN_PROCESS_NOISE_PRICE", 1e-5))
    KALMAN_PROCESS_NOISE_VELOCITY = float(os.getenv("KALMAN_PROCESS_NOISE_VELOCITY", 1e-6))
    KALMAN_PROCESS_NOISE_ACCELERATION = float(os.getenv("KALMAN_PROCESS_NOISE_ACCELERATION", 1e-7))

    # Observation Noise (measurement uncertainty)
    KALMAN_OBSERVATION_NOISE = float(os.getenv("KALMAN_OBSERVATION_NOISE", 1e-4))

    # Initial State Uncertainty
    KALMAN_INITIAL_UNCERTAINTY_PRICE = float(os.getenv("KALMAN_INITIAL_UNCERTAINTY_PRICE", 1.0))
    KALMAN_INITIAL_UNCERTAINTY_VELOCITY = float(os.getenv("KALMAN_INITIAL_UNCERTAINTY_VELOCITY", 1.0))
    KALMAN_INITIAL_UNCERTAINTY_ACCELERATION = float(os.getenv("KALMAN_INITIAL_UNCERTAINTY_ACCELERATION", 1.0))

    # Adaptive Parameters
    KALMAN_ADAPTIVE_NOISE = os.getenv("KALMAN_ADAPTIVE_NOISE", "true").lower() == "true"
    KALMAN_INNOVATION_HISTORY_LENGTH = int(os.getenv("KALMAN_INNOVATION_HISTORY_LENGTH", 100))

    # ===========================================
    # RISK MANAGEMENT PARAMETERS
    # ===========================================

    # Position Sizing (based on signal strength and confidence)
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.02))  # 2% max risk per trade
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", 0.10))  # 10% total portfolio risk
    BASE_LEVERAGE = float(os.getenv("BASE_LEVERAGE", 10.0))  # Base leverage (conservative)
    MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", 75.0))  # Maximum allowed leverage
    MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", 1.5))  # Minimum risk/reward ratio

    # Position Limits
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))  # Maximum concurrent positions
    MAX_CORRELATION = float(os.getenv("MAX_CORRELATION", 0.7))  # Maximum correlation between positions
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 1000.0))  # Maximum position size in USD

    # Dynamic Stop Loss and Take Profit
    BASE_STOP_LOSS_PCT = float(os.getenv("BASE_STOP_LOSS_PCT", 0.02))  # 2% base stop loss
    BASE_TAKE_PROFIT_PCT = float(os.getenv("BASE_TAKE_PROFIT_PCT", 0.04))  # 4% base take profit
    VOLATILITY_ADJUSTMENT_FACTOR = float(os.getenv("VOLATILITY_ADJUSTMENT_FACTOR", 2.0))  # ATR multiplier for stops
    TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"

    # Portfolio Risk Controls
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", 0.10))  # 10% daily loss limit
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", 5))  # Stop trading after 5 consecutive losses
    MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", 0.20))  # 20% maximum drawdown

    # Emergency Controls
    EMERGENCY_STOP_ENABLED = os.getenv("EMERGENCY_STOP_ENABLED", "true").lower() == "true"
    CIRCUIT_BREAKER_THRESHOLD = float(os.getenv("CIRCUIT_BREAKER_THRESHOLD", 0.15))  # 15% loss triggers circuit breaker

    # ===========================================
    # WEBSOCKET DATA CONFIGURATION
    # ===========================================

    # Connection Parameters
    WEBSOCKET_HEARTBEAT_INTERVAL = int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", 20))  # 20 seconds
    MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", 10))
    RECONNECT_BACKOFF_FACTOR = float(os.getenv("RECONNECT_BACKOFF_FACTOR", 2.0))
    CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", 30))  # seconds

    # Data Quality Parameters
    DATA_QUALITY_THRESHOLD = float(os.getenv("DATA_QUALITY_THRESHOLD", 0.8))  # Minimum data quality score
    STALE_DATA_THRESHOLD = int(os.getenv("STALE_DATA_THRESHOLD", 60))  # Max seconds without data
    PRICE_DEVIATION_THRESHOLD = float(os.getenv("PRICE_DEVIATION_THRESHOLD", 0.05))  # 5% max price deviation

    # Subscription Parameters
    SUBSCRIBE_TRADES = os.getenv("SUBSCRIBE_TRADES", "true").lower() == "true"
    SUBSCRIBE_ORDERBOOK = os.getenv("SUBSCRIBE_ORDERBOOK", "true").lower() == "true"
    SUBSCRIBE_TICKERS = os.getenv("SUBSCRIBE_TICKERS", "false").lower() == "true"
    ORDERBOOK_DEPTH = int(os.getenv("ORDERBOOK_DEPTH", 25))  # Order book depth levels

    # ===========================================
    # SYSTEM PERFORMANCE PARAMETERS
    # ===========================================

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/calculus_trading.log")
    ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"

    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    PERFORMANCE_UPDATE_INTERVAL = int(os.getenv("PERFORMANCE_UPDATE_INTERVAL", 60))  # seconds
    METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", 30))

    # System Health Monitoring
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 30))  # seconds
    ERROR_RATE_THRESHOLD = float(os.getenv("ERROR_RATE_THRESHOLD", 0.05))  # 5% error rate threshold
    MEMORY_USAGE_THRESHOLD = float(os.getenv("MEMORY_USAGE_THRESHOLD", 0.80))  # 80% memory usage threshold

    # Data Processing
    DATA_PROCESSING_THREADS = int(os.getenv("DATA_PROCESSING_THREADS", 2))
    SIGNAL_PROCESSING_BATCH_SIZE = int(os.getenv("SIGNAL_PROCESSING_BATCH_SIZE", 100))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", 1000))  # Maximum cached items

    # ===========================================
    # BACKTESTING CONFIGURATION
    # ===========================================

    # Historical Data Parameters
    BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2023-01-01")
    BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2023-12-31")
    BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", 10000.0))

    # Trading Costs
    COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", 0.001))  # 0.1% commission
    SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", 0.0005))  # 0.05% slippage

    # Optimization Parameters
    ENABLE_PARAMETER_OPTIMIZATION = os.getenv("ENABLE_PARAMETER_OPTIMIZATION", "false").lower() == "true"
    OPTIMIZATION_METRIC = os.getenv("OPTIMIZATION_METRIC", "sharpe_ratio")  # Optimization target
    OPTIMIZATION_ITERATIONS = int(os.getenv("OPTIMIZATION_ITERATIONS", 100))

    # Monte Carlo Simulation
    ENABLE_MONTE_CARLO = os.getenv("ENABLE_MONTE_CARLO", "true").lower() == "true"
    MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", 1000))
    MONTE_CARLO_CONFIDENCE_LEVEL = float(os.getenv("MONTE_CARLO_CONFIDENCE_LEVEL", 0.95))

    # ===========================================
    # WEB RESEARCH CONFIGURATION (Optional)
    # ===========================================

    # Research Data Collection
    ENABLE_WEB_RESEARCH = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"
    RESEARCH_CACHE_TTL = int(os.getenv("RESEARCH_CACHE_TTL", 3600))  # seconds
    RESEARCH_UPDATE_INTERVAL = int(os.getenv("RESEARCH_UPDATE_INTERVAL", 1800))  # 30 minutes

    # API Rate Limits
    MESSARI_RATE_LIMIT = int(os.getenv("MESSARI_RATE_LIMIT", 3))  # calls per second
    GLASSNODE_RATE_LIMIT = int(os.getenv("GLASSNODE_RATE_LIMIT", 2))  # calls per second
    DEFIllAMA_RATE_LIMIT = int(os.getenv("DEFIllAMA_RATE_LIMIT", 5))  # calls per second
    TOKENTERMINAL_RATE_LIMIT = int(os.getenv("TOKENTERMINAL_RATE_LIMIT", 3))  # calls per second
    ARKHAM_RATE_LIMIT = int(os.getenv("ARKHAM_RATE_LIMIT", 3))  # calls per second

    # ===========================================
    # AI MODEL CONFIGURATION (Optional)
    # ===========================================

    # Multi-Model Consensus
    ENABLE_MULTI_MODEL = os.getenv("ENABLE_MULTI_MODEL", "false").lower() == "true"
    CONSENSUS_MECHANISM = os.getenv("CONSENSUS_MECHANISM", "majority_vote")
    CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", 2))  # Minimum votes required

    # Model Configuration
    AI_MODELS_ENABLED = os.getenv("AI_MODELS_ENABLED", "grok4fast,qwen3max,deepseekterminus").split(",")
    GROK4FAST_MODEL = os.getenv("GROK4FAST_MODEL", "x-ai/grok-4-fast")
    QWEN3MAX_MODEL = os.getenv("QWEN3MAX_MODEL", "qwen/qwen-2.5-max")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-v3.1-terminus")

    # Model Parameters
    GROK4FAST_TEMPERATURE = float(os.getenv("GROK4FAST_TEMPERATURE", 0.1))
    QWEN3MAX_TEMPERATURE = float(os.getenv("QWEN3MAX_TEMPERATURE", 0.6))
    DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0.3))

    # OpenRouter API
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # ===========================================
    # SAFETY AND EMERGENCY CONFIGURATION
    # ===========================================

    # Trading Safety Flags
    DISABLE_SHORT_SELLING = os.getenv("DISABLE_SHORT_SELLING", "false").lower() == "true"
    MAX_POSITION_DIRECTION = os.getenv("MAX_POSITION_DIRECTION", "BOTH")  # LONG_ONLY, SHORT_ONLY, or BOTH
    SHORT_SELLING_BYPASS_CODE = os.getenv("SHORT_SELLING_BYPASS_CODE", "DISABLED")

    # Emergency Settings
    EMERGENCY_MODE = os.getenv("EMERGENCY_MODE", "false").lower() == "true"
    EMERGENCY_CONTACT_EMAIL = os.getenv("EMERGENCY_CONTACT_EMAIL", "")
    EMERGENCY_WEBHOOK_URL = os.getenv("EMERGENCY_WEBHOOK_URL", "")

    # System Limits
    MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", 100))
    MAX_HOURLY_TRADES = int(os.getenv("MAX_HOURLY_TRADES", 20))
    TRADE_COOLDOWN_SECONDS = int(os.getenv("TRADE_COOLDOWN_SECONDS", 10))

    # Data Retention
    TRADE_HISTORY_RETENTION_DAYS = int(os.getenv("TRADE_HISTORY_RETENTION_DAYS", 90))
    LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", 30))
    METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", 30))

    # ===========================================
    # CLASS METHODS FOR CONFIGURATION MANAGEMENT
    # ===========================================

    @classmethod
    def get_trading_config(cls) -> Dict[str, Any]:
        """Get all trading-related configuration."""
        return {
            'bybit': {
                'testnet': cls.BYBIT_TESTNET,
                'base_url': cls.BYBIT_BASE_URL,
                'tld': cls.BYBIT_TLD,
                'max_orders_per_second': cls.MAX_ORDERS_PER_SECOND,
                'order_timeout': cls.ORDER_TIMEOUT_SECONDS,
                'batch_order_size': cls.BATCH_ORDER_SIZE
            },
            'assets': cls.TARGET_ASSETS,
            'leverage': {
                'base': cls.BASE_LEVERAGE,
                'max': cls.MAX_LEVERAGE
            },
            'position_limits': {
                'max_positions': cls.MAX_POSITIONS,
                'max_correlation': cls.MAX_CORRELATION,
                'max_position_size': cls.MAX_POSITION_SIZE
            }
        }

    @classmethod
    def get_calculus_config(cls) -> Dict[str, Any]:
        """Get calculus-based trading configuration."""
        return {
            'smoothing': {
                'lambda_param': cls.LAMBDA_PARAM,
                'min_window': cls.MIN_SMOOTHING_WINDOW
            },
            'snr': {
                'threshold': cls.SNR_THRESHOLD,
                'window_size': cls.SNR_WINDOW_SIZE
            },
            'derivatives': {
                'min_velocity_threshold': cls.MIN_VELOCITY_THRESHOLD,
                'min_acceleration_threshold': cls.MIN_ACCELERATION_THRESHOLD,
                'velocity_smoothing': cls.VELOCITY_SMOOTHING_FACTOR
            },
            'signals': {
                'confidence_threshold': cls.SIGNAL_CONFIDENCE_THRESHOLD,
                'min_interval': cls.MIN_SIGNAL_INTERVAL,
                'max_age': cls.MAX_SIGNAL_AGE
            }
        }

    @classmethod
    def get_risk_config(cls) -> Dict[str, Any]:
        """Get risk management configuration."""
        return {
            'position_sizing': {
                'max_risk_per_trade': cls.MAX_RISK_PER_TRADE,
                'max_portfolio_risk': cls.MAX_PORTFOLIO_RISK,
                'min_risk_reward_ratio': cls.MIN_RISK_REWARD_RATIO
            },
            'stops': {
                'base_stop_loss': cls.BASE_STOP_LOSS_PCT,
                'base_take_profit': cls.BASE_TAKE_PROFIT_PCT,
                'volatility_adjustment': cls.VOLATILITY_ADJUSTMENT_FACTOR,
                'trailing_stop': cls.TRAILING_STOP_ENABLED
            },
            'portfolio_controls': {
                'daily_loss_limit': cls.DAILY_LOSS_LIMIT,
                'max_consecutive_losses': cls.MAX_CONSECUTIVE_LOSSES,
                'max_drawdown': cls.MAX_DRAWDOWN_LIMIT,
                'circuit_breaker': cls.CIRCUIT_BREAKER_THRESHOLD
            }
        }

    @classmethod
    def get_kalman_config(cls) -> Dict[str, Any]:
        """Get Kalman filter configuration."""
        return {
            'process_noise': {
                'price': cls.KALMAN_PROCESS_NOISE_PRICE,
                'velocity': cls.KALMAN_PROCESS_NOISE_VELOCITY,
                'acceleration': cls.KALMAN_PROCESS_NOISE_ACCELERATION
            },
            'observation_noise': cls.KALMAN_OBSERVATION_NOISE,
            'initial_uncertainty': {
                'price': cls.KALMAN_INITIAL_UNCERTAINTY_PRICE,
                'velocity': cls.KALMAN_INITIAL_UNCERTAINTY_VELOCITY,
                'acceleration': cls.KALMAN_INITIAL_UNCERTAINTY_ACCELERATION
            },
            'adaptive': {
                'enabled': cls.KALMAN_ADAPTIVE_NOISE,
                'history_length': cls.KALMAN_INNOVATION_HISTORY_LENGTH
            }
        }

    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check required API keys
        if not cls.BYBIT_API_KEY:
            issues.append("BYBIT_API_KEY is required for trading")
        if not cls.BYBIT_API_SECRET:
            issues.append("BYBIT_API_SECRET is required for trading")

        # Check parameter ranges
        if not 0 < cls.LAMBDA_PARAM < 1:
            issues.append("LAMBDA_PARAM must be between 0 and 1")
        if cls.SNR_THRESHOLD <= 0:
            issues.append("SNR_THRESHOLD must be positive")
        if cls.MAX_LEVERAGE > 100:
            issues.append("MAX_LEVERAGE should not exceed 100x")
        if cls.MAX_RISK_PER_TRADE > 0.05:
            issues.append("MAX_RISK_PER_TRADE should not exceed 5%")

        # Check logical consistency
        if cls.BASE_STOP_LOSS_PCT >= cls.BASE_TAKE_PROFIT_PCT:
            issues.append("BASE_STOP_LOSS_PCT should be less than BASE_TAKE_PROFIT_PCT")
        if cls.MIN_RISK_REWARD_RATIO <= 1:
            issues.append("MIN_RISK_REWARD_RATIO should be greater than 1")

        return issues

    @classmethod
    def print_configuration_summary(cls):
        """Print a summary of key configuration parameters."""
        print("=" * 60)
        print("ANNE'S CALCULUS TRADING SYSTEM - CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Environment: {'TESTNET' if cls.BYBIT_TESTNET else 'LIVE'}")
        print(f"Target Assets: {', '.join(cls.TARGET_ASSETS)}")
        print(f"Max Leverage: {cls.MAX_LEVERAGE}x")
        print(f"Max Risk per Trade: {cls.MAX_RISK_PER_TRADE:.1%}")
        print(f"Min Risk/Reward Ratio: {cls.MIN_RISK_REWARD_RATIO:.1f}")
        print(f"SNR Threshold: {cls.SNR_THRESHOLD}")
        print(f"Signal Confidence Threshold: {cls.SIGNAL_CONFIDENCE_THRESHOLD:.1%}")
        print(f"Kalman Adaptive Noise: {cls.KALMAN_ADAPTIVE_NOISE}")
        print(f"Emergency Stop: {cls.EMERGENCY_STOP_ENABLED}")
        print("=" * 60)

# Configuration validation on import
validation_issues = Config.validate_configuration()
if validation_issues:
    print("⚠️  CONFIGURATION ISSUES FOUND:")
    for issue in validation_issues:
        print(f"  - {issue}")
    print("Please address these issues before running the trading system.")
else:
    print("✅ Configuration validation passed")

# Print configuration summary
Config.print_configuration_summary()

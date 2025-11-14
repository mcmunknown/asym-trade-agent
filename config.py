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

    # EXECUTION COST FIX: Focus on ultra-liquid symbols during optimization
    ULTRA_LIQUID_MODE = os.getenv("ULTRA_LIQUID_MODE", "true").lower() == "true"
    
    # FINAL BREAKTHROUGH: Emergency calculus mode - trust math, ignore statistics
    EMERGENCY_CALCULUS_MODE = os.getenv("EMERGENCY_CALCULUS_MODE", "true").lower() == "true"
    
    # Trading Assets (high-liquidity perpetual futures)
    if ULTRA_LIQUID_MODE:
        # Ultra-liquid symbols with tightest spreads for execution cost optimization
        TARGET_ASSETS = os.getenv(
            "TARGET_ASSETS",
            "BTCUSDT,ETHUSDT"  # Only BTC/ETH during spread optimization
        ).split(",")
    else:
        # Full symbol list
        TARGET_ASSETS = os.getenv(
            "TARGET_ASSETS",
            "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,ADAUSDT,LINKUSDT,LTCUSDT,XRPUSDT,DOGEUSDT,TRXUSDT,MATICUSDT,ATOMUSDT,APTUSDT,OPUSDT,ARBUSDT"
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
    LAMBDA_PARAM = float(os.getenv("LAMBDA_PARAM", 0.80))  # Increased smoothing for less SNR volatility
    MIN_SMOOTHING_WINDOW = int(os.getenv("MIN_SMOOTHING_WINDOW", 20))  # Minimum data points

    # Signal-to-Noise Ratio (SNR) Parameters - CRYPTO ADAPTED
    # Formula: SNRᵥ = |vₜ|/σᵥ
    # QUANTUM-OPTIMIZED: Ultra-low thresholds to capture micro-movements with 50x leverage
    SNR_THRESHOLD = float(os.getenv("SNR_THRESHOLD", 0.1))  # AGGRESSIVE: Almost any velocity matters
    SNR_WINDOW_SIZE = int(os.getenv("SNR_WINDOW_SIZE", 10))  # Shorter window for faster crypto response

    # Velocity and Acceleration Thresholds - CRYPTO ADAPTED  
    MIN_VELOCITY_THRESHOLD = float(os.getenv("MIN_VELOCITY_THRESHOLD", 0.0008))  # Increased for crypto volatility
    MIN_ACCELERATION_THRESHOLD = float(os.getenv("MIN_ACCELERATION_THRESHOLD", 0.00002))  # 2x sensitivity for crypto
    VELOCITY_SMOOTHING_FACTOR = float(os.getenv("VELOCITY_SMOOTHING_FACTOR", 0.80))  # Reduced smoothing for faster response

    # Signal Generation Parameters - QUANTUM UNLEASHED
    # QUANTUM-OPTIMIZED: Trust calculus derivatives over statistical confidence
    SIGNAL_CONFIDENCE_THRESHOLD = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 0.01))  # AGGRESSIVE: Trust calculus
    MIN_SIGNAL_INTERVAL = int(os.getenv("MIN_SIGNAL_INTERVAL", 1))  # Ultra-fast 1-second cycles
    MAX_SIGNAL_AGE = int(os.getenv("MAX_SIGNAL_AGE", 180))  # Shorter signals for crypto timeframes

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
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", 0.60))  # 60% total portfolio risk (AGGRESSIVE for compounding)
    BASE_LEVERAGE = float(os.getenv("BASE_LEVERAGE", 6.0))  # 6x base leverage for crypto volatility control
    MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", 50.0))  # Maximum allowed leverage (crypto-optimized)
    MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", 1.5))  # Minimum risk/reward ratio

    # Position Limits
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))  # Maximum concurrent positions
    MAX_CORRELATION = float(os.getenv("MAX_CORRELATION", 0.7))  # Maximum correlation between positions
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 1000.0))  # Maximum position size in USD

    CALCULUS_PRIORITY_MODE = os.getenv("CALCULUS_PRIORITY_MODE", "true").lower() == "true"
    FORCE_LEVERAGE_ENABLED = os.getenv("FORCE_LEVERAGE_ENABLED", "true").lower() == "true"
    FORCE_LEVERAGE_VALUE = float(os.getenv("FORCE_LEVERAGE_VALUE", 50.0))
    FORCE_MARGIN_FRACTION = float(os.getenv("FORCE_MARGIN_FRACTION", 0.80))  # AGGRESSIVE: 80% margin for micro tier
    CALCULUS_LOSS_BLOCK_THRESHOLD = int(os.getenv("CALCULUS_LOSS_BLOCK_THRESHOLD", 5))  # More tolerance
    CURVATURE_EDGE_THRESHOLD = float(os.getenv("CURVATURE_EDGE_THRESHOLD", 0.001))  # ULTRA-QUANTUM: 0.1% baseline
    CURVATURE_FORECAST_HORIZONS = os.getenv("CURVATURE_FORECAST_HORIZONS", "2,6,15")
    TP_SECONDARY_MULTIPLIER = float(os.getenv("TP_SECONDARY_MULTIPLIER", 1.8))
    TP_PRIMARY_FRACTION = float(os.getenv("TP_PRIMARY_FRACTION", 0.35))  # AGGRESSIVE: 35% TP1, 65% TP2 for bigger winners
    TP_TRAIL_BUFFER_MULTIPLIER = float(os.getenv("TP_TRAIL_BUFFER_MULTIPLIER", 0.5))
    CURVATURE_EDGE_MIN = float(os.getenv("CURVATURE_EDGE_MIN", 0.0005))  # ULTRA-QUANTUM: 0.05% floor
    CURVATURE_EDGE_MAX = float(os.getenv("CURVATURE_EDGE_MAX", 0.005))  # ULTRA-QUANTUM: Lower max
    TP_PRIMARY_PROB_BASE = float(os.getenv("TP_PRIMARY_PROB_BASE", 0.50))  # QUANTUM: Lower base
    TP_PRIMARY_PROB_MIN = float(os.getenv("TP_PRIMARY_PROB_MIN", 0.40))  # QUANTUM: Accept 40% probability  
    TP_PRIMARY_PROB_MAX = float(os.getenv("TP_PRIMARY_PROB_MAX", 0.65))
    TP_SECONDARY_PROB_MIN = float(os.getenv("TP_SECONDARY_PROB_MIN", 0.25))  # QUANTUM: Lower secondary too
    GOVERNOR_BLOCK_RELAX = int(os.getenv("GOVERNOR_BLOCK_RELAX", 30))  # QUANTUM: 30 blocks vs 120
    GOVERNOR_TIME_RELAX_SEC = int(os.getenv("GOVERNOR_TIME_RELAX_SEC", 300))  # QUANTUM: 5 min vs 30 min
    GOVERNOR_FEE_PRESSURE_HARD = float(os.getenv("GOVERNOR_FEE_PRESSURE_HARD", 0.8))  # More tolerance
    COMPOUNDING_LADDER = os.getenv(
        "COMPOUNDING_LADDER",
        # ULTIMATE AGGRESSIVE compounding ladder (balance:mode:leverage:margin_fraction)
        # 0–25   : force 50x, 80% margin (micro tier all-in)
        # 25–100 : force 60x, 75% margin (tier1 aggressive)
        # 100–500: force 60x, 70% margin (tier2 balanced)
        # 500–2k : force 50x, 65% margin (tier3 conservative)
        # >2k    : auto leverage, 50% margin (Sharpe/Kelly guided)
        "0:force:50:0.80;25:force:60:0.75;100:force:60:0.70;500:force:50:0.65;2000:auto:auto:0.50"
    )
    SCOUT_ENTRY_SCALE = float(os.getenv("SCOUT_ENTRY_SCALE", 0.70))  # QUANTUM: 70% scout vs 55%
    WEEKLY_VAR_CAP = float(os.getenv("WEEKLY_VAR_CAP", 0.25))

    # Base per-symbol notional targets (scaled by balance tiers inside risk manager)
    _default_symbol_bases = {
        "BTCUSDT": 8.0,
        "ETHUSDT": 12.0,
        "BNBUSDT": 10.0,
        "SOLUSDT": 8.0,
        "AVAXUSDT": 8.0,
        "LTCUSDT": 10.0,
        "LINKUSDT": 7.0,
        "ADAUSDT": 6.0,
        "XRPUSDT": 6.0,
        "DOGEUSDT": 6.0,
        "TRXUSDT": 6.0,
        "MATICUSDT": 6.0,
        "ATOMUSDT": 7.0,
        "APTUSDT": 7.0,
        "OPUSDT": 7.0,
        "ARBUSDT": 7.0,
    }

    # Optional explicit overrides via environment (format: "BTCUSDT:15,ETHUSDT:20")
    _caps_env = os.getenv("SYMBOL_MAX_NOTIONAL_CAPS")
    _cap_overrides = {}
    if _caps_env:
        try:
            for item in _caps_env.split(","):
                if ":" in item:
                    sym, val = item.split(":", 1)
                    _cap_overrides[sym.strip().upper()] = float(val)
        except ValueError:
            _cap_overrides = {}

    SYMBOL_BASE_NOTIONALS = _default_symbol_bases
    SYMBOL_MAX_NOTIONAL_CAPS = _cap_overrides

    FEE_BUFFER_MULTIPLIER = float(os.getenv("FEE_BUFFER_MULTIPLIER", 2.5))  # Crypto-optimized: reduced from 4.0
    MAKER_REBATE_PCT = float(os.getenv("MAKER_REBATE_PCT", 0.0001))  # Expected maker rebate for liquid symbols
    EV_DEBUG_LOGGING = os.getenv("EV_DEBUG_LOGGING", "false").lower() == "true"
    MIN_FINAL_TP_PROBABILITY = float(os.getenv("MIN_FINAL_TP_PROBABILITY", 0.52))
    POSTERIOR_CONFIDENCE_Z = float(os.getenv("POSTERIOR_CONFIDENCE_Z", 1.96))
    POSTERIOR_DECAY = float(os.getenv("POSTERIOR_DECAY", 0.02))

    # Derivative-aware horizon & barrier optimizer
    USE_BARRIER_TP_OPTIMIZER = os.getenv("USE_BARRIER_TP_OPTIMIZER", "true").lower() == "true"
    BARRIER_OPT_GRID_STEPS = int(os.getenv("BARRIER_OPT_GRID_STEPS", 5))
    BARRIER_OPT_TP_RANGE = float(os.getenv("BARRIER_OPT_TP_RANGE", 0.35))
    BARRIER_OPT_SL_RANGE = float(os.getenv("BARRIER_OPT_SL_RANGE", 0.3))
    BARRIER_OPT_MIN_RR = float(os.getenv("BARRIER_OPT_MIN_RR", 0.6))
    BARRIER_OPT_MAX_RR = float(os.getenv("BARRIER_OPT_MAX_RR", 3.0))
    BARRIER_OPT_MIN_TP_PROB = float(os.getenv("BARRIER_OPT_MIN_TP_PROB", 0.35))

    DERIVATIVE_TREND_V_WEIGHT = float(os.getenv("DERIVATIVE_TREND_V_WEIGHT", 1.0))
    DERIVATIVE_TREND_A_WEIGHT = float(os.getenv("DERIVATIVE_TREND_A_WEIGHT", 0.6))
    DERIVATIVE_TREND_J_WEIGHT = float(os.getenv("DERIVATIVE_TREND_J_WEIGHT", 0.2))
    DERIVATIVE_TREND_J_NORM = float(os.getenv("DERIVATIVE_TREND_J_NORM", 0.01))
    DERIVATIVE_TREND_BASE = float(os.getenv("DERIVATIVE_TREND_BASE", 1.0))
    DERIVATIVE_TREND_CONF_DIV = float(os.getenv("DERIVATIVE_TREND_CONF_DIV", 3.0))
    DERIVATIVE_HORIZON_K_GAIN = float(os.getenv("DERIVATIVE_HORIZON_K_GAIN", 0.35))
    DERIVATIVE_HORIZON_MIN = float(os.getenv("DERIVATIVE_HORIZON_MIN", 60.0))
    DERIVATIVE_HORIZON_MAX = float(os.getenv("DERIVATIVE_HORIZON_MAX", 1200.0))
    DERIVATIVE_DRIFT_MIN_WEIGHT = float(os.getenv("DERIVATIVE_DRIFT_MIN_WEIGHT", 0.35))
    DERIVATIVE_DRIFT_MAX_WEIGHT = float(os.getenv("DERIVATIVE_DRIFT_MAX_WEIGHT", 0.85))
    DEFAULT_OU_THETA = float(os.getenv("DEFAULT_OU_THETA", 0.05))

    MIN_PROFIT_BEFORE_FORECAST_EXIT_PCT = float(os.getenv("MIN_PROFIT_BEFORE_FORECAST_EXIT_PCT", 100.0))  # AGGRESSIVE: Never exit on forecast
    STRONG_TREND_HOLD_MULT = float(os.getenv("STRONG_TREND_HOLD_MULT", 2.5))
    STRONG_TREND_SCORE_THRESHOLD = float(os.getenv("STRONG_TREND_SCORE_THRESHOLD", 1.8))
    TRAIL_ACTIVATION_PROGRESS_PCT = float(os.getenv("TRAIL_ACTIVATION_PROGRESS_PCT", 0.5))  # AGGRESSIVE: Allow trailing from start
    TRAIL_PROGRESS_RELAX_MULT = float(os.getenv("TRAIL_PROGRESS_RELAX_MULT", 1.4))
    MICRO_MIN_TP_USDT = float(os.getenv("MICRO_MIN_TP_USDT", 0.35))
    MIN_EMERGENCY_EV_PCT = float(os.getenv("MIN_EMERGENCY_EV_PCT", 0.00015))  # +0.015% threshold
    EMERGENCY_EV_SKIP_PCT = float(os.getenv("EMERGENCY_EV_SKIP_PCT", -0.0008))
    EV_POSITION_SCALE_MIN = float(os.getenv("EV_POSITION_SCALE_MIN", 0.9))  # AGGRESSIVE: Minimal EV scaling
    EV_POSITION_SCALE_MAX = float(os.getenv("EV_POSITION_SCALE_MAX", 1.1))  # AGGRESSIVE: Minimal EV scaling
    EV_POSITION_REF_PCT = float(os.getenv("EV_POSITION_REF_PCT", 0.0015))
    MICRO_EV_HARD_ENTRY_PCT = float(os.getenv("MICRO_EV_HARD_ENTRY_PCT", 0.0008))
    MICRO_EV_GENTLE_ENTRY_PCT = float(os.getenv("MICRO_EV_GENTLE_ENTRY_PCT", 0.0004))
    MICRO_EV_RED_ZONE_SKIP_PCT = float(os.getenv("MICRO_EV_RED_ZONE_SKIP_PCT", -0.001))  # AGGRESSIVE: Hard floor only
    MICRO_EV_YELLOW_SIZE_SCALE = float(os.getenv("MICRO_EV_YELLOW_SIZE_SCALE", 0.55))
    MICRO_EV_GREEN_SIZE_SCALE = float(os.getenv("MICRO_EV_GREEN_SIZE_SCALE", 1.0))
    MICRO_SYMBOL_AVG_EV_FLOOR = float(os.getenv("MICRO_SYMBOL_AVG_EV_FLOOR", 0.0))
    MICRO_SYMBOL_EV_MIN_SAMPLES = int(os.getenv("MICRO_SYMBOL_EV_MIN_SAMPLES", 6))
    MICRO_SYMBOL_DRAWDOWN_FLOOR_PCT = float(os.getenv("MICRO_SYMBOL_DRAWDOWN_FLOOR_PCT", 0.05))
    MICRO_SYMBOL_BLOCK_DURATION = int(os.getenv("MICRO_SYMBOL_BLOCK_DURATION", 300))
    MICRO_MIN_SIZE_FORCE_EV_PCT = float(os.getenv("MICRO_MIN_SIZE_FORCE_EV_PCT", 0.001))

    # Renaissance-style Order Book Imbalance Gating
    USE_ORDERBOOK_IMBALANCE_GATE = os.getenv("USE_ORDERBOOK_IMBALANCE_GATE", "true").lower() == "true"
    ORDERBOOK_IMBALANCE_THRESHOLD = float(os.getenv("ORDERBOOK_IMBALANCE_THRESHOLD", 0.15))
    ORDERBOOK_IMBALANCE_MIN_SAMPLES = int(os.getenv("ORDERBOOK_IMBALANCE_MIN_SAMPLES", 10))
    ORDERBOOK_IMBALANCE_WINDOW_SIZE = int(os.getenv("ORDERBOOK_IMBALANCE_WINDOW_SIZE", 60))
    ORDERBOOK_GATE_ALLOW_WEAK = os.getenv("ORDERBOOK_GATE_ALLOW_WEAK", "true").lower() == "true"  # Allow trades with weak imbalance (just penalize confidence)
    ORDERBOOK_CONFIDENCE_BOOST_ENABLED = os.getenv("ORDERBOOK_CONFIDENCE_BOOST_ENABLED", "true").lower() == "true"

    # Execution Fix: Hard Limits for Position Management
    EXECUTION_SL_RETRY_ATTEMPTS = int(os.getenv("EXECUTION_SL_RETRY_ATTEMPTS", 3))
    EXECUTION_ENTRY_SPACING_SECONDS = float(os.getenv("EXECUTION_ENTRY_SPACING_SECONDS", 1.0))  # Minimum 1s between same-symbol entries
    EXECUTION_MAX_POSITION_AGE_SECONDS = float(os.getenv("EXECUTION_MAX_POSITION_AGE_SECONDS", 300))  # Max 5 min per position
    EXECUTION_PREVENT_MULTI_OPEN = os.getenv("EXECUTION_PREVENT_MULTI_OPEN", "true").lower() == "true"  # Never open 2+ per symbol

    SYMBOL_MIN_ORDER_QTY = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.1,
        "BNBUSDT": 0.01,
        "AVAXUSDT": 0.1,
        "ADAUSDT": 1.0,
        "LINKUSDT": 0.1,
        "LTCUSDT": 0.1,
        "XRPUSDT": 0.1,
        "DOGEUSDT": 1.0,
        "TRXUSDT": 1.0,
        "MATICUSDT": 1.0,
        "ATOMUSDT": 0.1,
        "APTUSDT": 0.01,
        "OPUSDT": 0.1,
        "ARBUSDT": 0.1,
    }

    SYMBOL_MIN_NOTIONALS = {
        symbol: 5.0 for symbol in _default_symbol_bases
    }

    # Balance tiers -> multiplier for symbol base notionals (upper bound inclusive)
    NOTIONAL_CAP_TIERS = [
        (50.0, 1.0),
        (500.0, 2.0),
        (5000.0, 4.0),
        (50000.0, 6.0),
        (float("inf"), 8.0),
    ]

    MICRO_TIER_BLOCKED_SYMBOLS = {"ETHUSDT", "SOLUSDT"}

    SIGNAL_TIER_CONFIG = [
        {
            "name": "micro",
            "max_equity": 25.0,
            "snr_threshold": 0.60,  # Crypto-optimized: lowered from 0.80
            "confidence_threshold": 0.35,  # Crypto-optimized: lowered from 0.45
            "min_signal_interval": 6,  # Faster cycle: reduced from 8
            "min_ou_hold_seconds": 120,  # Crypto faster: reduced from 240
            "max_ou_hold_seconds": 480,  # Crypto faster: reduced from 900
            "min_ev_pct": 0.0005,  # Crypto-optimized: lowered from 0.0008
            "min_tp_distance_pct": 0.030,  # EXECUTION COST FIX: 3% minimum (was 1.2%)
            "min_probability_samples": 6,  # Faster adaptation: reduced from 8
            "max_positions_per_symbol": 1,
            "max_positions_per_minute": 25  # Increased frequency for crypto
        },
        {
            "name": "tier1",
            "max_equity": 100.0,
            "snr_threshold": 0.45,  # Crypto-optimized: increased from 0.35
            "confidence_threshold": 0.25,  # Crypto-optimized: increased from 0.20
            "min_signal_interval": 8,  # Faster cycle: reduced from 10
            "min_ou_hold_seconds": 120,  # Crypto faster: reduced from 180
            "max_ou_hold_seconds": 300,  # Crypto faster: reduced from 360
            "min_ev_pct": 0.0008,  # Crypto-optimized: reduced from 0.0012
            "min_tp_distance_pct": 0.025,  # EXECUTION COST FIX: 2.5% minimum (was 1.0%)
            "min_probability_samples": 8,  # Faster adaptation: reduced from 12
            "max_positions_per_symbol": 1,
            "max_positions_per_minute": 35  # Increased frequency for crypto
        },
        {
            "name": "tier2",
            "max_equity": 1000.0,
            "snr_threshold": 0.50,  # Crypto-optimized: increased from 0.40
            "confidence_threshold": 0.28,  # Crypto-optimized: increased from 0.22
            "min_signal_interval": 10,  # Faster cycle: reduced from 12
            "min_ou_hold_seconds": 180,  # Crypto faster: reduced from 300
            "max_ou_hold_seconds": 600,  # Crypto faster: reduced from 900
            "min_ev_pct": 0.0007,  # Crypto-optimized: reduced from 0.0010
            "min_tp_distance_pct": 0.025,  # EXECUTION COST FIX: 2.5% minimum (was 0.9%)
            "min_probability_samples": 12,  # Faster adaptation: reduced from 16
            "max_positions_per_symbol": 1,
            "max_positions_per_minute": 45  # Increased frequency for crypto
        },
        {
            "name": "tier3",
            "max_equity": float("inf"),
            "snr_threshold": 0.55,  # Crypto-optimized: increased from 0.45
            "confidence_threshold": 0.30,  # Crypto-optimized: increased from 0.24
            "min_signal_interval": 12,  # Faster cycle: reduced from 15
            "min_ou_hold_seconds": 300,  # Crypto faster: reduced from 600
            "max_ou_hold_seconds": 900,  # Crypto faster: reduced from 1800
            "min_ev_pct": 0.0006,  # Crypto-optimized: reduced from 0.0008
            "min_tp_distance_pct": 0.020,  # EXECUTION COST FIX: 2% minimum (was 0.8%)
            "min_probability_samples": 18,  # Faster adaptation: reduced from 24
            "max_positions_per_symbol": 1,
            "max_positions_per_minute": 55  # Increased frequency for crypto
        }
    ]

    SYMBOL_TIER_WHITELIST = {
        "micro": [
            "LTCUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "TRXUSDT",
            "MATICUSDT",
            "APTUSDT",
            "OPUSDT",
            "ARBUSDT"
        ],
        "tier1": [
            "LTCUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "TRXUSDT",
            "MATICUSDT",
            "APTUSDT",
            "OPUSDT",
            "ARBUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "LINKUSDT"
        ],
        "tier2": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "ADAUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "TRXUSDT",
            "MATICUSDT",
            "ATOMUSDT",
            "APTUSDT",
            "OPUSDT",
            "ARBUSDT",
            "SUIUSDT",
            "FILUSDT",
            "NEARUSDT",
            "LDOUSDT"
        ],
        "tier3": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "ADAUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "TRXUSDT",
            "MATICUSDT",
            "ATOMUSDT",
            "APTUSDT",
            "OPUSDT",
            "ARBUSDT",
            "SUIUSDT",
            "FILUSDT",
            "NEARUSDT",
            "LDOUSDT",
            "INJUSDT",
            "AAVEUSDT",
            "SNXUSDT"
        ]
    }

    SYMBOL_CANDIDATE_POOL = {
        "micro": [
            "SUIUSDT",
            "FILUSDT",
            "NEARUSDT",
            "LDOUSDT",
            "INJUSDT",
            "AAVEUSDT",
            "SNXUSDT"
        ],
        "tier1": [
            "SUIUSDT",
            "FILUSDT",
            "NEARUSDT",
            "LDOUSDT",
            "INJUSDT",
            "AAVEUSDT",
            "SNXUSDT"
        ],
        "tier2": [
            "SUIUSDT",
            "FILUSDT",
            "NEARUSDT",
            "LDOUSDT",
            "INJUSDT",
            "AAVEUSDT",
            "SNXUSDT"
        ],
        "tier3": []
    }

    MICROSTRUCTURE_LIMITS = {
        "max_spread_pct": float(os.getenv("MICRO_MAX_SPREAD_PCT", 0.0008)),  # Crypto-optimized: increased for wider spreads
        "max_slippage_pct": float(os.getenv("MICRO_MAX_SLIPPAGE_PCT", 0.0010)),  # Crypto-optimized: increased for higher slip
        "min_samples": int(os.getenv("MICRO_MIN_SAMPLES", 6)),  # Faster adaptation: reduced from 8
        "candidate_ev_samples": int(os.getenv("MICRO_CANDIDATE_EV_SAMPLES", 8)),  # Faster promotion: reduced from 10
        "candidate_ev_buffer": float(os.getenv("MICRO_CANDIDATE_EV_BUFFER", 0.000015))  # Tighter buffer for crypto
    }

    # Dynamic Stop Loss and Take Profit
    BASE_STOP_LOSS_PCT = float(os.getenv("BASE_STOP_LOSS_PCT", 0.02))  # 2% base stop loss
    BASE_TAKE_PROFIT_PCT = float(os.getenv("BASE_TAKE_PROFIT_PCT", 0.04))  # 4% base take profit
    VOLATILITY_ADJUSTMENT_FACTOR = float(os.getenv("VOLATILITY_ADJUSTMENT_FACTOR", 2.0))  # ATR multiplier for stops
    TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"

    # Portfolio Risk Controls - CRYPTO ADAPTED
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", 0.08))  # 8% daily loss limit (crypto-optimized)
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", 4))  # Stop trading after 4 consecutive losses (crypto)
    MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", 0.15))  # 15% maximum drawdown (crypto-optimized)

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
            },
            'overrides': {
                'calculus_priority_mode': cls.CALCULUS_PRIORITY_MODE,
                'force_leverage_enabled': cls.FORCE_LEVERAGE_ENABLED,
                'force_leverage_value': cls.FORCE_LEVERAGE_VALUE,
                'force_margin_fraction': cls.FORCE_MARGIN_FRACTION,
                'calculus_loss_block_threshold': cls.CALCULUS_LOSS_BLOCK_THRESHOLD
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
        print(f"Calculus Priority Mode: {cls.CALCULUS_PRIORITY_MODE}")
        if cls.FORCE_LEVERAGE_ENABLED:
            print(f"Force Leverage: {cls.FORCE_LEVERAGE_VALUE}x @ {cls.FORCE_MARGIN_FRACTION:.0%} margin fraction")
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

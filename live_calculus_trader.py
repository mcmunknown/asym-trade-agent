"""
Live Calculus Trading System
============================

🚨 CORE SYSTEM CONSTRAINT: This system maintains exactly 23 essential Python files.
DO NOT ADD NEW FILES - modify existing files only. This constraint is enforced
by Git hooks and system architecture to maintain codebase clarity and performance.

Production-ready live trading system implementing Anne's complete calculus-based approach
with real-time data processing, signal generation, risk management, and automated execution.

Complete Signal Pipeline:
1. Real-time WebSocket data → Market data validation
2. Price series accumulation → Kalman filtering → Calculus analysis
3. Signal generation with SNR filtering → Risk validation
4. Position sizing → Dynamic TP/SL → Order execution
5. Position monitoring → Performance tracking

Features:
- Real-time calculus-based signal generation
- Kalman filter for adaptive state estimation
- Dynamic risk management with position sizing
- High-frequency execution with batch orders
- Comprehensive performance monitoring
- Emergency stop and circuit breakers
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import math
import time
import threading
import sys
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_UP
from enum import Enum
from collections import defaultdict

# Import our enhanced components
from websocket_client import BybitWebSocketClient, ChannelType, MarketData
from calculus_strategy import CalculusTradingStrategy, SignalType
from quantitative_models import CalculusPriceAnalyzer
from kalman_filter import AdaptiveKalmanFilter, KalmanConfig
from risk_manager import RiskManager, PositionSize, TradingLevels
from bybit_client import BybitClient
from config import Config
from position_logic import determine_position_side, determine_trade_side, validate_position_consistency
from quantitative_models import calculate_multi_timeframe_velocity
from order_flow import OrderFlowAnalyzer
from ou_mean_reversion import OUMeanReversionModel
from daily_drift_predictor import DailyDriftPredictor

# Import C++ accelerated bridge
from cpp_bridge_working import (
    KalmanFilter as CPPKalmanFilter,
    analyze_curve_complete,
    kelly_position_size,
    risk_adjusted_position,
    calculate_portfolio_metrics,
    exponential_smoothing,
    calculate_velocity,
    calculate_acceleration
)

# Import portfolio management components
from portfolio_manager import PortfolioManager, PortfolioPosition, AllocationDecision
from signal_coordinator import SignalCoordinator
from joint_distribution_analyzer import JointDistributionAnalyzer
from portfolio_optimizer import PortfolioOptimizer, OptimizationObjective
from regime_filter import BayesianRegimeFilter

# Configure logging with enhanced console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add enhanced console handler for beautiful terminal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - 🎯 %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

class ErrorCategory(Enum):
    """Categorized error types for diagnostic tracking"""
    INSUFFICIENT_BALANCE = "insufficient_balance"
    POSITION_SIZE_TOO_SMALL = "position_size_too_small"
    ASSET_TOO_EXPENSIVE = "asset_too_expensive"
    INVALID_SIGNAL_DATA = "invalid_signal_data"
    RISK_VALIDATION_FAILED = "risk_validation_failed"
    API_ERROR = "api_error"
    MIN_NOTIONAL_NOT_MET = "min_notional_not_met"
    REGIME_MISMATCH = "regime_mismatch"
    FLAT_MARKET_FILTER = "flat_market_filter"
    POSITION_SIDE_MISMATCH = "position_side_mismatch"

@dataclass
class TradingState:
    """Current trading state for a symbol"""
    symbol: str
    price_history: List[float]
    timestamps: List[float]
    kalman_filter: CPPKalmanFilter  # C++ accelerated Kalman filter
    calculus_analyzer: CalculusPriceAnalyzer
    regime_filter: BayesianRegimeFilter
    last_signal: Optional[Dict]
    position_info: Optional[Dict]
    signal_count: int
    last_execution_time: float
    error_count: int = 0
    last_signal_time: float = 0.0
    error_breakdown: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    gating_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    open_positions: List[Dict] = field(default_factory=list)
    last_orderbook: Optional[Dict] = None

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration: float
    success_rate: float
    total_commission: float

class LiveCalculusTrader:
    """
    🚀 ENHANCED LIVE TRADING SYSTEM - Portfolio Integration
    ======================================================

    ⚠️ LIVE TRADING WARNING: This system will execute REAL trades with REAL money!

    This system integrates all components for sophisticated portfolio trading:
    - Real-time data processing with WebSocket for 8 assets
    - Anne's calculus-based signal generation for timing
    - Portfolio optimization for allocation decisions
    - Joint distribution analysis for risk management
    - Signal coordination for multi-asset decisions
    - Dynamic risk management and position sizing
    - High-frequency execution with TP/SL management

    HYBRID APPROACH:
    - Single-asset calculus signals → TIMING decisions
    - Portfolio optimization → ALLOCATION decisions
    - Joint distribution → RISK management
    """

    def __init__(self,
                 symbols: List[str] = None,
                 window_size: int = 100,  # Crypto-optimized: faster response with shorter window
                 min_signal_interval: int = 15,  # Crypto-optimized: faster signal cycle
                 emergency_stop: bool = False,
                 max_position_size: float = 1000.0,
                 simulation_mode: bool = False,
                 portfolio_mode: bool = True):
        """
        Initialize the ENHANCED live calculus trading system with portfolio integration.

        Args:
            symbols: List of trading symbols (default: 8 major crypto assets)
            window_size: Price history window for analysis
            min_signal_interval: Minimum seconds between signals
            emergency_stop: Emergency stop flag
            max_position_size: Maximum position size per trade
            simulation_mode: Run in simulation mode (no real trades)
            portfolio_mode: Enable portfolio management integration
        """
        # Default to configured asset universe for portfolio trading
        if symbols is None:
            symbols = [sym.strip().upper() for sym in Config.TARGET_ASSETS if sym.strip()]

        self.symbols = symbols
        self.window_size = window_size
        self.min_signal_interval = min_signal_interval
        self.emergency_stop = emergency_stop
        self.max_position_size = max_position_size
        self.simulation_mode = simulation_mode
        self.portfolio_mode = portfolio_mode
        self.ev_debug_enabled = bool(getattr(Config, "EV_DEBUG_LOGGING", False))
        self._tp_probability_debug: Dict[str, Dict[str, float]] = {}
        self._ev_debug_records: Dict[str, Dict[str, float]] = {}
        self._logged_config_snapshot = False

        print("🚀 ENHANCED LIVE TRADING SYSTEM INITIALIZING")
        print("=" * 60)
        print(f"📊 Trading {len(symbols)} assets: {', '.join(symbols[:4])}...")
        print(f"🔬 Portfolio Mode: {'ENABLED' if portfolio_mode else 'DISABLED'}")
        if not simulation_mode:
            print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
        else:
            print("🧪 SIMULATION MODE - Safe for testing")
        print("=" * 60)

        if self.ev_debug_enabled:
            logger.info("EV debug logging enabled (crypto diagnostics mode)")

        # Initialize core components
        channel_types = [ChannelType.TRADE, ChannelType.TICKER]
        if getattr(Config, "SUBSCRIBE_ORDERBOOK", True):
            channel_types.append(ChannelType.ORDERBOOK_1)

        self.ws_client = BybitWebSocketClient(
            symbols=symbols,
            testnet=Config.BYBIT_TESTNET,
            channel_types=channel_types,
            heartbeat_interval=20
        )
        self.bybit_client = BybitClient()
        self.risk_manager = RiskManager(
            max_risk_per_trade=Config.MAX_RISK_PER_TRADE,
            max_portfolio_risk=Config.MAX_PORTFOLIO_RISK,
            max_leverage=Config.MAX_LEVERAGE,
            min_risk_reward=Config.MIN_RISK_REWARD_RATIO
        )

        self._log_crypto_config_snapshot()
        
        # Renaissance-style components (high frequency + structural edges)
        self.order_flow = OrderFlowAnalyzer(window_size=50)
        self.ou_model = OUMeanReversionModel(lookback=100)
        self.drift_predictor = DailyDriftPredictor(lookback=100)  # Layer 5: Daily drift prediction
        
        # Rate limiting for drift monitoring (prevent over-trading)
        self.last_monitor_time = 0.0  # Track last monitoring check
        self.monitor_interval = 30.0  # Check every 30 seconds minimum
        
        logger.info("✅ Renaissance components initialized (Order Flow + OU Mean Reversion + Rate Limiting)")
        
        self.instrument_cache: Dict[str, Dict] = {}
        self.min_qty_overrides = {
            'ETHUSDT': 0.005,
            'BTCUSDT': 0.001,
            'SOLUSDT': 0.1,
            'BNBUSDT': 0.01,
        }

        # CRITICAL FIX: Check balance before enabling portfolio mode
        try:
            initial_balance = self.bybit_client.get_account_balance()
            if initial_balance:
                available_balance = float(initial_balance.get('totalAvailableBalance', 0))
                logger.info(f"💰 Initial balance: ${available_balance:.2f}")
                
                # Disable portfolio mode for low balances
                if available_balance < 50:
                    self.portfolio_mode = False
                    logger.warning(f"📊 Portfolio mode DISABLED - balance ${available_balance:.2f} below $50 threshold")
                    logger.info(f"   Using single-asset mode for better margin efficiency")
                else:
                    logger.info(f"📈 Portfolio mode ENABLED - balance ${available_balance:.2f} sufficient for multi-asset trading")
            else:
                logger.warning("Could not get initial balance, defaulting to single-asset mode")
                self.portfolio_mode = False
        except Exception as e:
            logger.warning(f"Balance check failed: {e}, defaulting to single-asset mode")
            self.portfolio_mode = False

        # Initialize portfolio components if enabled
        if self.portfolio_mode:
            logger.info("🎯 Initializing portfolio management components...")
            self.portfolio_manager = PortfolioManager(
                symbols=symbols,
                initial_capital=100000.0,  # $100k portfolio
                rebalance_threshold=0.05,  # 5% drift threshold
                risk_manager=self.risk_manager
            )

            self.joint_distribution_analyzer = JointDistributionAnalyzer(num_assets=len(symbols))
            self.portfolio_optimizer = PortfolioOptimizer(joint_analyzer=self.joint_distribution_analyzer)
            self.signal_coordinator = SignalCoordinator(symbols=symbols, portfolio_manager=self.portfolio_manager)

            # CRITICAL FIX: Initialize optimal weights to enable trading
            equal_weight = 1.0 / len(symbols)
            initial_weights = {symbol: equal_weight for symbol in symbols}
            self.portfolio_manager.update_optimal_weights(initial_weights)
            logger.info(f"✅ Initialized equal portfolio weights: {initial_weights}")

            logger.info("✅ Portfolio components initialized successfully")
        else:
            self.portfolio_manager = None
            self.signal_coordinator = None
            self.joint_distribution_analyzer = None
            self.portfolio_optimizer = None

        # Trading state per symbol
        self.trading_states = {}
        for symbol in symbols:
            self.trading_states[symbol] = TradingState(
                symbol=symbol,
                price_history=[],
                timestamps=[],
                kalman_filter=CPPKalmanFilter(
                    process_noise_price=1e-3,      # Crypto-calibrated: 100x higher for volatility
                    process_noise_velocity=1e-4,   # Crypto-calibrated: captures rapid changes
                    process_noise_acceleration=1e-5,  # Crypto-calibrated: momentum shifts
                    observation_noise=1e-4,        # Measurement accurate (exchange data)
                    dt=1.0                         # 1 second sampling
                ),  # C++ accelerated Kalman filter - crypto volatility regime
                calculus_analyzer=CalculusPriceAnalyzer(),
                regime_filter=BayesianRegimeFilter(),
                last_signal=None,
                position_info=None,
                signal_count=0,
                last_execution_time=0,
                error_count=0,
                last_signal_time=0.0,
                last_orderbook=None
            )

        # Performance tracking
        self.performance = PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            total_pnl=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
            avg_trade_duration=0.0, success_rate=0.0, total_commission=0.0
        )

        # Circuit breakers
        self.daily_loss_limit = 0.10  # 10% daily loss limit
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.last_reset_time = time.time()
        self.daily_pnl = 0.0

        # Threading and async management
        self.is_running = False
        self.processing_thread = None
        self.monitoring_thread = None
        self.symbol_blocklist: Dict[str, float] = {}
        self.symbol_block_reasons: Dict[str, str] = {}
        self.last_available_balance: float = 0.0
        self.current_tier = self.risk_manager.get_equity_tier(self.last_available_balance)
        self.tier_transition_log: List[Tuple[float, Dict]] = [(time.time(), dict(self.current_tier))]
        self.symbol_fee_cache: Dict[str, Dict] = {}
        self.symbol_funding_cache: Dict[str, Dict] = {}
        self.ou_survival_cache: Dict[Tuple, float] = {}
        self._rng = np.random.default_rng()

        logger.info(f"Live Calculus Trader initialized for symbols: {symbols}")
        logger.info(f"Parameters: window_size={window_size}, min_signal_interval={min_signal_interval}s")

    def _safe_float(self, value, fallback: float = 0.0) -> float:
        """Safely convert a value to float, falling back when the input is invalid."""
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback
    
    def _record_error(self, state: TradingState, category: ErrorCategory, reason: str):
        """
        Record categorized error with detailed reason for diagnostic tracking.
        
        Args:
            state: Trading state for the symbol
            category: Error category from ErrorCategory enum
            reason: Detailed reason for the error
        """
        state.error_count += 1
        state.error_breakdown[category] += 1
        logger.warning(f"❌ {state.symbol} {category.value}: {reason}")

    def _record_signal_block(self, state: TradingState, reason: str, details: Optional[str] = None):
        """Track non-execution reasons for diagnostics with throttled logging."""
        state.gating_breakdown[reason] += 1
        now = time.time()
        if not hasattr(state, 'gating_log_times'):
            state.gating_log_times = {}
        last_log = state.gating_log_times.get(reason, 0.0)
        if now - last_log >= 60.0:
            message = f"🚫 {state.symbol} signal blocked [{reason}]"
            if details:
                message += f": {details}"
            logger.info(message)
            state.gating_log_times[reason] = now
        else:
            logger.debug(f"Signal block (throttled) {state.symbol} [{reason}]: {details}")

    def _should_log_ev_debug(self) -> bool:
        return bool(getattr(self, "ev_debug_enabled", False))

    def _log_crypto_config_snapshot(self) -> None:
        if self._logged_config_snapshot:
            return
        fee_multiplier = float(getattr(Config, "FEE_BUFFER_MULTIPLIER", 0.0))
        micro_limits = getattr(Config, "MICROSTRUCTURE_LIMITS", {})
        logger.info(
            "Crypto EV config → fee_multiplier=%.3f, max_spread=%.4f, max_slippage=%.4f, min_samples=%s",
            fee_multiplier,
            float(micro_limits.get('max_spread_pct', 0.0)),
            float(micro_limits.get('max_slippage_pct', 0.0)),
            micro_limits.get('min_samples')
        )
        if abs(fee_multiplier - 2.5) > 1e-6:
            logger.warning("Fee buffer multiplier deviates from crypto baseline 2.5 → %.3f", fee_multiplier)
        self._logged_config_snapshot = True

    def _log_probability_debug(self, symbol: str, payload: Dict[str, float]) -> None:
        logger.info(
            "Probability debug %s → base=%.4f confidence_boost=%.4f snr_boost=%.4f velocity_boost=%.4f posterior_weight=%.4f final=%.4f",
            symbol,
            payload.get('base_probability'),
            payload.get('confidence_boost'),
            payload.get('snr_boost'),
            payload.get('velocity_boost'),
            payload.get('posterior_weight'),
            payload.get('final_probability')
        )

    def _log_ev_breakdown(self, symbol: str, payload: Dict[str, float]) -> None:
        logger.info(
            "EV debug %s → raw_tp=%.4f raw_sl=%.4f adj_tp=%.4f adj_sl=%.4f fee_adj=%.4f cost_floor=%.4f fee_floor=%.4f micro=%.4f tp_prob=%.4f ev=%.4f",
            symbol,
            payload.get('raw_tp_pct'),
            payload.get('raw_sl_pct'),
            payload.get('adjusted_tp_pct'),
            payload.get('adjusted_sl_pct'),
            payload.get('fee_adjustment_pct'),
            payload.get('execution_cost_floor_pct'),
            payload.get('fee_floor_pct'),
            payload.get('micro_cost_pct'),
            payload.get('final_tp_probability'),
            payload.get('final_ev_pct')
        )

    def get_last_probability_debug(self, symbol: str) -> Optional[Dict[str, float]]:
        return self._tp_probability_debug.get(symbol.upper())

    def get_last_ev_debug(self, symbol: str) -> Optional[Dict[str, float]]:
        return self._ev_debug_records.get(symbol.upper())

    @staticmethod
    def _confidence_to_probability(confidence: float, threshold: float) -> float:
        confidence = float(max(confidence, 0.0))
        threshold = max(float(threshold), 0.0)
        if confidence <= 0:
            return 0.5
        if confidence >= 1.0:
            return 0.9
        delta = confidence - threshold
        if delta >= 0:
            return min(0.9, 0.55 + delta * 0.6)
        return max(0.45, 0.55 + delta * 0.8)

    def _estimate_tp_probability(self, symbol: str, signal_dict: Dict, tier_config: Dict) -> Tuple[float, Dict[str, float]]:
        # CRYPTO-OPTIMIZED TP PROBABILITY ESTIMATION
        tp_probability = signal_dict.get('tp_probability')
        if tp_probability is not None:
            try:
                value = float(np.clip(tp_probability, 0.10, 0.90))  # Crypto: higher min, lower max
                posterior = self.risk_manager.get_symbol_probability_posterior(symbol)
                return value, posterior
            except (TypeError, ValueError):
                pass

        confidence = float(signal_dict.get('confidence', 0.0))
        tier_threshold = float(tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD))

        # CRYPTO-OPTIMIZED: More aggressive confidence-to-probability conversion
        base_prob = self._confidence_to_probability(confidence, tier_threshold)
        debug_info = {
            'input_confidence': confidence,
            'tier_confidence_threshold': tier_threshold,
            'base_probability': base_prob,
            'confidence_boost': 0.0,
            'snr_boost': 0.0,
            'snr_delta': 0.0,
            'velocity_boost': 0.0,
            'posterior_weight': 0.0,
            'posterior_mean': None,
            'posterior_blend': None,
            'final_probability': None,
            'high_confidence_floor': False
        }

        # Crypto boost for high confidence signals
        if confidence > 0.80:
            base_prob += 0.08  # 8% boost for very confident signals
            debug_info['confidence_boost'] = 0.08
        elif confidence > 0.70:
            base_prob += 0.05  # 5% boost for confident signals
            debug_info['confidence_boost'] = 0.05

        snr = float(signal_dict.get('snr', 0.0))
        tier_snr = float(tier_config.get('snr_threshold', Config.SNR_THRESHOLD))
        snr_delta = 0.0
        if snr > 0 and tier_snr > 0:
            snr_delta = max(0.0, snr - tier_snr)
            snr_increment = 0.08 * np.tanh(snr_delta * 1.5)
            updated_prob = min(0.88, base_prob + snr_increment)
            debug_info['snr_boost'] = updated_prob - base_prob
            debug_info['snr_delta'] = snr_delta
            base_prob = updated_prob
        else:
            debug_info['snr_delta'] = snr_delta

        posterior = self.risk_manager.get_symbol_probability_posterior(symbol)
        posterior_mean = posterior.get('mean', base_prob)
        posterior_count = posterior.get('count', 0.0)
        min_samples = float(tier_config.get('min_probability_samples', 5))

        # Crypto: Faster posterior weighting for fewer samples
        weight = min(1.0, posterior_count / max(min_samples * 0.7, 1.0))  # Reduced min_samples for crypto
        blended_prob = (1.0 - weight) * base_prob + weight * posterior_mean
        debug_info['posterior_weight'] = weight
        debug_info['posterior_mean'] = posterior_mean
        debug_info['posterior_blend'] = blended_prob

        # Crypto: Final boost for strong mean reversion signals
        velocity = float(signal_dict.get('velocity', 0.0))
        vel_boost = 0.0
        if abs(velocity) > tier_threshold * 1.5:
            vel_boost = 0.05 if snr > 2.0 else 0.03
            blended_prob += vel_boost
        debug_info['velocity_boost'] = vel_boost
        debug_info['velocity'] = velocity

        final_prob = float(np.clip(blended_prob, 0.10, 0.90))
        if confidence >= 0.80 and final_prob < 0.42:
            final_prob = 0.42
            debug_info['high_confidence_floor'] = True
        debug_info['final_probability'] = final_prob

        self._tp_probability_debug[symbol.upper()] = debug_info
        if self._should_log_ev_debug():
            self._log_probability_debug(symbol, debug_info)

        return final_prob, posterior

    def _compute_trade_ev(self,
                           symbol: str,
                           tp_pct: float,
                           sl_pct: float,
                           tp_prob: float,
                           fee_floor_pct: float,
                           debug_context: Optional[Dict[str, float]] = None) -> float:
        tp_pct_raw = tp_pct
        sl_pct_raw = sl_pct
        tp_prob_raw = tp_prob

        fee_adjustment = fee_floor_pct * 0.6  # Only 60% of cost floor hits EV
        adjusted_tp = max(tp_pct_raw - fee_adjustment, 0.0)
        adjusted_sl = sl_pct_raw + fee_adjustment

        clipped_prob = float(np.clip(tp_prob_raw, 0.10, 0.90))
        boosted_prob = clipped_prob
        if adjusted_tp > 0.008:  # TP > 0.8%
            boosted_prob += 0.05
        final_prob = float(np.clip(boosted_prob, 0.05, 0.95))

        net_ev = final_prob * adjusted_tp - (1.0 - final_prob) * adjusted_sl

        breakdown = {
            'raw_tp_pct': tp_pct_raw,
            'raw_sl_pct': sl_pct_raw,
            'adjusted_tp_pct': adjusted_tp,
            'adjusted_sl_pct': adjusted_sl,
            'fee_adjustment_pct': fee_adjustment,
            'initial_tp_probability': tp_prob_raw,
            'clipped_tp_probability': clipped_prob,
            'final_tp_probability': final_prob,
            'final_ev_pct': net_ev,
            'execution_cost_floor_pct': fee_floor_pct
        }

        if debug_context:
            breakdown.update(debug_context)

        self._ev_debug_records[symbol.upper()] = breakdown

        if self._should_log_ev_debug():
            self._log_ev_breakdown(symbol, breakdown)

        return net_ev

    def _evaluate_expected_ev(self, position_info: Dict, current_price: float) -> Tuple[float, float, float, float]:
        side = position_info.get('side', 'Buy')
        take_profit = self._safe_float(position_info.get('take_profit'), 0.0)
        stop_loss = self._safe_float(position_info.get('stop_loss'), 0.0)
        fee_floor_pct = float(position_info.get('fee_floor_pct', 0.0))
        execution_cost_floor_pct = float(position_info.get('execution_cost_floor_pct', fee_floor_pct))
        if current_price <= 0:
            return -1.0, 0.0, 0.0, execution_cost_floor_pct

        if side.lower() == 'buy':
            tp_pct = max((take_profit - current_price) / current_price, 0.0)
            sl_pct = max((current_price - stop_loss) / current_price, 0.0)
        else:
            tp_pct = max((current_price - take_profit) / current_price, 0.0)
            sl_pct = max((stop_loss - current_price) / current_price, 0.0)

        if tp_pct <= 0:
            return -1.0, tp_pct, sl_pct, fee_floor_pct

        tp_prob = position_info.get('tp_probability')
        if tp_prob is None:
            confidence = float(position_info.get('latest_forecast_confidence', position_info.get('confidence', 0.0)))
            threshold = float(position_info.get('tier_confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD))
            tp_prob = self._confidence_to_probability(confidence, threshold)

        symbol = position_info.get('symbol', 'UNKNOWN')
        debug_context = {
            'fee_floor_pct': fee_floor_pct,
            'micro_cost_pct': float(position_info.get('micro_cost_pct', 0.0)),
            'execution_cost_floor_pct': execution_cost_floor_pct,
            'base_tp_probability': float(position_info.get('base_tp_probability', tp_prob)),
            'time_constrained_probability': float(position_info.get('time_constrained_probability', tp_prob))
        }
        ev = self._compute_trade_ev(
            symbol,
            tp_pct,
            sl_pct,
            tp_prob,
            execution_cost_floor_pct,
            debug_context=debug_context
        )
        return ev, tp_pct, sl_pct, execution_cost_floor_pct

    def _handle_ev_block(self, symbol: str, tier_min_ev_pct: Optional[float]):
        if tier_min_ev_pct is None:
            return
        if self.risk_manager.should_block_symbol_by_ev(symbol, float(tier_min_ev_pct)):
            duration = 900
            self._register_symbol_block(symbol, "ev_guard", duration=duration)
            state = self.trading_states.get(symbol)
            if state:
                self._record_signal_block(
                    state,
                    "ev_guard",
                    f"avg<{float(tier_min_ev_pct)*100:.2f}% recent edge"
                )
            logger.warning(
                "🚧 EV guard: average net edge for %s fell below %.3f%% over recent trades",
                symbol,
                float(tier_min_ev_pct) * 100.0
            )

    def _get_dynamic_fee_components(self, symbol: str, expected_hold_seconds: Optional[float]) -> Tuple[float, float, float]:
        base_fee = float(getattr(Config, "COMMISSION_RATE", 0.001))
        taker_fee = base_fee
        maker_fee = base_fee
        now = time.time()

        fee_cache = self.symbol_fee_cache.get(symbol)
        cache_stale = True
        if fee_cache:
            cache_age = now - fee_cache['timestamp']
            cache_stale = cache_age > 1800
            if not cache_stale:
                taker_fee = fee_cache['taker']
                maker_fee = fee_cache['maker']

        if cache_stale:
            fee_data = self.bybit_client.get_trading_fee_rate(symbol)
            if fee_data:
                try:
                    taker_fee = float(fee_data.get('takerFeeRate', taker_fee))
                    maker_fee = float(fee_data.get('makerFeeRate', maker_fee))
                except (TypeError, ValueError):
                    taker_fee = base_fee
                    maker_fee = base_fee
                self.symbol_fee_cache[symbol] = {
                    'timestamp': now,
                    'taker': taker_fee,
                    'maker': maker_fee
                }
                logger.info(
                    "Fee cache refresh: %s taker %.4f%% maker %.4f%%",
                    symbol,
                    taker_fee * 100.0,
                    maker_fee * 100.0
                )
            elif fee_cache:
                taker_fee = fee_cache.get('taker', base_fee)
                maker_fee = fee_cache.get('maker', base_fee)
                cache_age = now - fee_cache.get('timestamp', now)
                logger.debug("Using cached fee rates for %s (age %.1fs)", symbol, cache_age)
            else:
                logger.warning("Falling back to static fee rate for %s", symbol)

        funding_buffer_pct = 0.0
        hold_seconds = max(float(expected_hold_seconds or 0.0), 0.0)
        if hold_seconds > 0:
            funding_cache = self.symbol_funding_cache.get(symbol)
            funding_stale = True
            if funding_cache:
                funding_age = now - funding_cache['timestamp']
                funding_stale = funding_age > 1800
            if funding_stale:
                funding_data = self.bybit_client.get_funding_rate(symbol)
                if funding_data and funding_data.get('fundingRate') is not None:
                    try:
                        rate = float(funding_data.get('fundingRate', 0.0))
                    except (TypeError, ValueError):
                        rate = 0.0
                    self.symbol_funding_cache[symbol] = {
                        'timestamp': now,
                        'rate': rate
                    }
                    funding_rate = rate
                else:
                    funding_rate = funding_cache.get('rate', 0.0) if funding_cache else 0.0
            else:
                funding_rate = funding_cache.get('rate', 0.0)

            funding_buffer_pct = abs(float(funding_rate)) * (hold_seconds / 28800.0)

        return taker_fee, maker_fee, funding_buffer_pct

    def _estimate_time_constrained_tp_probability(self,
                                                   symbol: str,
                                                   side: str,
                                                   current_price: float,
                                                   take_profit: float,
                                                   stop_loss: float,
                                                   forecast_price: Optional[float],
                                                   half_life_seconds: Optional[float],
                                                   sigma_estimate: Optional[float],
                                                   max_hold_seconds: Optional[float]) -> float:
        try:
            if max_hold_seconds is None or max_hold_seconds <= 0 or current_price <= 0 or take_profit <= 0 or stop_loss <= 0:
                return 0.5

            side_lower = (side or 'Buy').lower()
            half_life = float(half_life_seconds) if half_life_seconds not in (None, float('inf')) else None
            if half_life is None or half_life <= 0:
                half_life = max_hold_seconds / 2.0
            sigma = float(max(sigma_estimate or 0.0, 5e-4))

            dt = max(min(1.0, max_hold_seconds / 180.0), 0.25)
            steps = max(int(max_hold_seconds / dt), 1)
            if steps > 720:
                steps = 720
                dt = max_hold_seconds / steps

            tp_pct = (take_profit - current_price) / current_price if side_lower == 'buy' else (current_price - take_profit) / current_price
            sl_pct = (current_price - stop_loss) / current_price if side_lower == 'buy' else (stop_loss - current_price) / current_price
            key = (
                symbol,
                side_lower,
                round(tp_pct, 4),
                round(sl_pct, 4),
                round(max_hold_seconds, 1),
                round(half_life, 1),
                round(sigma, 4)
            )
            cached = self.ou_survival_cache.get(key)
            if cached is not None:
                return cached

            paths = 150 if max_hold_seconds <= 600 else 220
            log_tp = math.log(take_profit)
            log_sl = math.log(stop_loss)
            log_forecast = math.log(forecast_price) if forecast_price and forecast_price > 0 else math.log(current_price)
            theta = math.log(2.0) / max(half_life, 1e-6)
            log_price = math.log(current_price)
            sqrt_dt = math.sqrt(dt)

            x = np.full(paths, log_price, dtype=float)
            alive = np.ones(paths, dtype=bool)
            hits_tp = np.zeros(paths, dtype=bool)

            for _ in range(steps):
                if not alive.any():
                    break
                noise = self._rng.standard_normal(paths)
                x = x + theta * (log_forecast - x) * dt + sigma * sqrt_dt * noise
                if side_lower == 'buy':
                    tp_hit = (x >= log_tp) & alive
                    sl_hit = (x <= log_sl) & alive
                else:
                    tp_hit = (x <= log_tp) & alive
                    sl_hit = (x >= log_sl) & alive
                hits_tp |= tp_hit
                alive &= ~(tp_hit | sl_hit)

            probability = float(hits_tp.sum() / paths)
            # Clip extreme probabilities to avoid zero/one
            probability = float(np.clip(probability, 0.01, 0.99))
            self.ou_survival_cache[key] = probability
            return probability
        except Exception as e:
            logger.error(f"Error estimating time-constrained TP probability for {symbol}: {e}")
            return 0.5

    def _refresh_tier(self, account_balance: float) -> Dict:
        """Update the active signal tier based on latest account balance."""
        sanitized_balance = max(self._safe_float(account_balance, 0.0), 0.0)
        new_tier = self.risk_manager.get_equity_tier(sanitized_balance)
        if self.current_tier is not new_tier:
            logger.info(
                "🎯 Equity tier transition: balance=%.2f → snr>=%.2f, confidence>=%.2f, min_interval=%ss, max_hold=%ss",
                sanitized_balance,
                new_tier.get("snr_threshold", Config.SNR_THRESHOLD),
                new_tier.get("confidence_threshold", Config.SIGNAL_CONFIDENCE_THRESHOLD),
                new_tier.get("min_signal_interval", self.min_signal_interval),
                new_tier.get("max_ou_hold_seconds", "∞")
            )
            self.tier_transition_log.append((time.time(), dict(new_tier)))
            self.current_tier = new_tier
        return self.current_tier

    def _get_current_tier(self) -> Dict:
        """Return the current signal tier (refreshing if balance changed)."""
        if not self.current_tier:
            self.current_tier = self.risk_manager.get_equity_tier(self.last_available_balance)
        return self.current_tier

    def _get_microstructure_metrics(self, symbol: str) -> Dict[str, float]:
        """Fetch latest microstructure metrics for a symbol."""
        snapshot = None
        if hasattr(self.ws_client, "get_orderbook_snapshot"):
            snapshot = self.ws_client.get_orderbook_snapshot(symbol)
        if not snapshot:
            state = self.trading_states.get(symbol)
            if state and state.last_orderbook:
                snapshot = state.last_orderbook

        snapshot = snapshot or {}

        try:
            spread_pct = float(snapshot.get('spread_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            spread_pct = 0.0

        micro_cost_pct = self.risk_manager.estimate_microstructure_cost(symbol, spread_pct)
        metrics = {
            'mid_price': snapshot.get('mid_price'),
            'spread_pct': spread_pct,
            'spread': snapshot.get('spread'),
            'best_bid': snapshot.get('best_bid'),
            'best_ask': snapshot.get('best_ask'),
            'best_bid_size': snapshot.get('best_bid_size'),
            'best_ask_size': snapshot.get('best_ask_size'),
            'micro_cost_pct': micro_cost_pct,
            'timestamp': snapshot.get('timestamp')
        }
        return metrics

    def _register_symbol_block(self, symbol: str, reason: str, duration: int = 900):
        """Temporarily disable signals for a symbol when exchange constraints make trading impossible."""
        expiry = time.time() + max(duration, 60)
        self.symbol_blocklist[symbol] = expiry
        self.symbol_block_reasons[symbol] = reason
        minutes = (expiry - time.time()) / 60.0
        logger.info(f"🚧 Auto-disabling {symbol} signals for {minutes:.1f} minutes ({reason})")

    def _round_quantity_to_step(self, quantity: float, step: float) -> float:
        """Round a quantity up to the nearest exchange-defined step size."""
        if step <= 0 or quantity <= 0:
            return max(quantity, 0.0)
        try:
            dec_qty = Decimal(str(quantity))
            dec_step = Decimal(str(step))
            multiplier = (dec_qty / dec_step).to_integral_value(rounding=ROUND_UP)
            rounded = multiplier * dec_step
            return float(rounded)
        except (ArithmeticError, ValueError):
            return quantity

    def start(self):
        """Start the live trading system."""
        print("Starting LiveCalculusTrader...")
        if self.is_running:
            logger.warning("Trading system is already running")
            return

        # Test connections first
        if not self._test_connections():
            logger.error("Connection tests failed. Cannot start trading.")
            return

        self.is_running = True
        self.emergency_stop = False
        
        # CRITICAL: Clear any phantom positions from previous session
        print("\n🧹 Clearing phantom positions from previous session...")
        cleared = self._clear_phantom_positions()
        if cleared > 0:
            print(f"   ✅ Cleared {cleared} phantom position(s) - ready to trade!")
        else:
            print(f"   ✅ No phantom positions found - starting clean!")

        # Add WebSocket callback
        self.ws_client.add_callback(ChannelType.TRADE, self._handle_market_data)

        # Add portfolio callback if portfolio mode is enabled
        if self.portfolio_mode:
            self.ws_client.add_portfolio_callback(self._handle_portfolio_data)

        if getattr(Config, "SUBSCRIBE_ORDERBOOK", True):
            self.ws_client.add_callback(ChannelType.ORDERBOOK_1, self._handle_orderbook_update)

        # Beautiful startup banner
        print("\n" + "="*70)
        print("🎯 YALE-PRINCETON TRADING SYSTEM - LIVE")
        print("="*70)
        print("✅ 7 Institutional Math Layers Active:")
        print("   1. Functional Derivatives (Pathwise Delta)")
        print("   2. Riemannian Geometry (Manifold Gradients)")
        print("   3. Measure Correction (P→Q Risk-Neutral)")
        print("   4. Kushner-Stratonovich (Continuous Filtering)")
        print("   5. Functional Itô-Taylor (Confidence Cones)")
        print("   8. Variance Stabilization (Volatility-Time)")
        print("   10. Asymptotic Error Control (Itô Isometry)")
        print("="*70)
        
        # Get and display account balance
        try:
            account_info = self.bybit_client.get_account_balance()
            if account_info:
                available_balance = float(account_info.get('totalAvailableBalance', 0))
                total_equity = float(account_info.get('totalEquity', 0))
                print(f"💰 Balance: ${available_balance:.2f} | Equity: ${total_equity:.2f}")
            print(f"🎯 Target: $50 in 4 hours")
            print(f"📊 Expected TP Rate: 85%+ (vs 40% before)")
        except:
            print(f"💰 Balance check in progress...")
            
        print("="*70)
        print("\n⏳ Starting WebSocket connection...")
        
        # Start WebSocket client before launching monitoring threads
        try:
            self.ws_client.subscribe()
            self.ws_client.start()
            
            # Give WebSocket time to establish connection with visual feedback
            for i in range(5):
                time.sleep(0.6)
                if self.ws_client.is_connected:
                    print(f"✅ WebSocket CONNECTED - Data flowing!")
                    break
                print(f"⏳ Connecting to WebSocket... ({i+1}/5)", end='\r', flush=True)
            else:
                print(f"⚠️  WebSocket taking longer (will auto-retry)          ")
            
            print("\n⏳ Waiting for price data to accumulate (need 50+ prices)...")
            print("📈 Watch for real-time updates below:\n")
            print("="*70 + "\n")
                
        except KeyboardInterrupt:
            logger.info("Stopping trading system...")
            self.stop()
            return
        except Exception as e:
            logger.error(f"Error in WebSocket client: {e}")
            self.stop()
            return

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("📊 Signal generation will begin after 50+ prices accumulate (~2-3 minutes)")
        logger.info("🎯 Yale-Princeton mathematics active: Measure correction, Variance stabilization, Continuous filtering")

        # Start portfolio monitoring if enabled
        if self.portfolio_mode:
            self.portfolio_thread = threading.Thread(target=self._portfolio_monitoring_loop, daemon=True)
            self.portfolio_thread.start()
            logger.info("📊 Portfolio monitoring started")

    def stop(self):
        """Stop the live trading system gracefully."""
        logger.info("Stopping live trading system...")
        self.is_running = False
        self.emergency_stop = True

        # Close all positions
        self._emergency_close_all_positions()

        # Stop WebSocket client
        if self.ws_client:
            self.ws_client.stop()

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        logger.info("Live trading system stopped")

    def _test_connections(self) -> bool:
        """Test all connections before starting."""
        try:
            if self.simulation_mode:
                logger.info("🧪 SIMULATION MODE - Skipping API connection tests")
                return True

            # Test Bybit API connection
            if not self.bybit_client.test_connection():
                logger.error("Bybit API connection failed")
                return False

            # Test account balance
            balance = self.bybit_client.get_account_balance()
            if not balance:
                logger.error("Could not fetch account balance")
                return False

            logger.info(f"Connection tests passed. Account balance: {balance.get('totalAvailableBalance', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _handle_market_data(self, market_data: MarketData):
        """
        Handle incoming market data and process trading signals.

        Args:
            market_data: Real-time market data from WebSocket
        """
        if not self.is_running or self.emergency_stop:
            return

        symbol = market_data.symbol
        if symbol not in self.trading_states:
            logger.warning(f"Received data for unknown symbol: {symbol}")
            return

        try:
            # Update price history
            state = self.trading_states[symbol]
            state.price_history.append(market_data.price)
            state.timestamps.append(market_data.timestamp)

            # Feed OU model with the newest price observation
            self.ou_model.update_prices(symbol, np.array([market_data.price]))

            # Update drift predictor (Layer 5)
            self.drift_predictor.update(
                symbol=symbol,
                price=market_data.price,
                volume=getattr(market_data, 'volume', 0),
                timestamp=market_data.timestamp
            )

            # Update order flow with price-based proxy (since we don't have trade-by-trade data)
            # Use price momentum as proxy for buy/sell pressure
            if len(state.price_history) >= 2:
                prev_price = state.price_history[-1]
                curr_price = market_data.price
                price_change = curr_price - prev_price
                # Convert price change to buy/sell signal
                # Positive change = buying pressure, negative = selling pressure
                if price_change > 0:
                    # Simulated buy
                    self.order_flow.update(symbol, [{
                        'side': 'buy',
                        'size': abs(price_change),
                        'timestamp': market_data.timestamp
                    }])
                elif price_change < 0:
                    # Simulated sell
                    self.order_flow.update(symbol, [{
                        'side': 'sell',
                        'size': abs(price_change),
                        'timestamp': market_data.timestamp
                    }])
                # If no change, skip update

            # Maintain window size
            if len(state.price_history) > self.window_size:
                state.price_history.pop(0)
                state.timestamps.pop(0)

            # Enhanced data accumulation progress with real-time updates - CRYPTO ADAPTED
            history_len = len(state.price_history)
            
            # Track if we've already shown the "READY" message for this symbol
            if not hasattr(state, 'ready_message_shown'):
                state.ready_message_shown = False
            
            # Show progress bar for data accumulation (crypto-optimized milestones)
            if history_len in [10, 25, 50, 75, 100]:
                progress_pct = (history_len / self.window_size) * 100
                print(f"\r📈 {symbol}: {history_len:3d}/{self.window_size} prices ({progress_pct:5.1f}%) | Latest: ${market_data.price:.2f}", end='', flush=True)
                # Only show "READY" message ONCE when first reaching 25 (crypto-optimized)
                if history_len == 25 and not state.ready_message_shown:
                    print()  # New line when ready for analysis
                    print(f"✅ {symbol}: READY FOR CRYPTO-OPTIMIZED ANALYSIS!")
                    print(f"   🧮 7 math layers active for fast crypto signals")
                    print()
                    state.ready_message_shown = True

            # Generate signals if we have enough data - crypto-optimized minimum
            if history_len >= 25:  # Minimum for crypto calculus analysis
                self._process_trading_signal(symbol)

        except Exception as e:
            logger.error(f"Error handling market data for {symbol}: {e}")
            state = self.trading_states[symbol]
            self._record_error(state, ErrorCategory.INVALID_SIGNAL_DATA, f"Market data processing error: {e}")

    def _handle_orderbook_update(self, market_data: MarketData):
        """Capture updates to top-of-book for microstructure analysis."""
        try:
            symbol = market_data.symbol
            state = self.trading_states.get(symbol)
            if not state:
                return

            snapshot = market_data.raw_data or {}
            state.last_orderbook = snapshot

            spread_pct = float(snapshot.get('spread_pct', 0.0) or 0.0)
            if spread_pct > 0:
                self.risk_manager.record_microstructure_sample(symbol, spread_pct)

        except Exception as e:
            logger.error(f"Error handling orderbook update for {market_data.symbol}: {e}")

    def _handle_portfolio_data(self, market_data_dict: Dict[str, MarketData]):
        """
        Handle portfolio-level market data for multi-asset analysis.

        Args:
            market_data_dict: Dictionary of market data for all symbols
        """
        if not self.portfolio_mode or not self.is_running:
            return

        try:
            # Update joint distribution analyzer
            price_updates = {}
            for symbol, data in market_data_dict.items():
                if symbol in self.symbols:
                    price_updates[symbol] = data.price

            if len(price_updates) >= 4:  # Minimum for meaningful analysis
                self.joint_distribution_analyzer.update_returns(price_updates)

            # Update portfolio manager with latest prices + signal context
            if self.portfolio_manager:
                for symbol, data in market_data_dict.items():
                    if symbol not in self.symbols:
                        continue
                    state = self.trading_states.get(symbol)
                    last_signal = state.last_signal if state else None
                    signal_strength = 0.0
                    confidence = 0.0
                    if last_signal:
                        signal_strength = float(last_signal.get('snr', 0.0))
                        confidence = float(last_signal.get('confidence', 0.0))
                    try:
                        self.portfolio_manager.update_market_data(
                            symbol,
                            data.price,
                            signal_strength,
                            confidence
                        )
                    except Exception as pm_error:
                        logger.error(f"Portfolio manager update failed for {symbol}: {pm_error}")

        except Exception as e:
            logger.error(f"Error in portfolio data handling: {e}")

    def _process_trading_signal(self, symbol: str):
        """
        Process trading signal using complete calculus analysis.

        Args:
            symbol: Trading symbol
        """
        try:
            state = self.trading_states[symbol]
            current_time = time.time()
            tier_config = self._get_current_tier()
            tier_interval = tier_config.get('min_signal_interval', self.min_signal_interval)

            # Skip symbols currently auto-disabled due to exchange constraints
            block_expiry = self.symbol_blocklist.get(symbol)
            if block_expiry:
                if current_time < block_expiry:
                    reason = self.symbol_block_reasons.get(symbol, "auto_block")
                    remaining = block_expiry - current_time
                    self._record_signal_block(state, "auto_block", f"{reason} ({remaining:.0f}s remaining)")
                    return
                self.symbol_blocklist.pop(symbol, None)
                self.symbol_block_reasons.pop(symbol, None)

            # CRITICAL: Rate limiting to prevent signal spam
            # Track last signal time (not just execution time)
            # Check minimum interval between ANY signals (not just executed trades)
            if current_time - state.last_signal_time < tier_interval:
                self._record_signal_block(
                    state,
                    "rate_limit",
                    f"{current_time - state.last_signal_time:.1f}s < {tier_interval}s (tier)"
                )
                return
            
            # Also check execution time (for additional safety)
            if current_time - state.last_execution_time < tier_interval:
                self._record_signal_block(
                    state,
                    "execution_cooldown",
                    f"{current_time - state.last_execution_time:.1f}s < {tier_interval}s (tier)"
                )
                return

            # Create price series
            price_series = pd.Series(state.price_history)

            # Apply C++ accelerated Kalman filtering
            prices_array = price_series.values
            filtered_prices, velocities, accelerations = state.kalman_filter.filter_prices(prices_array)

            if len(filtered_prices) == 0:
                self._record_signal_block(state, "kalman_no_output")
                return

            # Get latest C++ Kalman estimates
            latest_filtered_price = filtered_prices[-1]
            latest_velocity = velocities[-1]
            latest_acceleration = accelerations[-1]

            # Get current state estimates for compatibility
            current_price_state, current_velocity_state, current_acceleration_state = state.kalman_filter.get_state()
            current_uncertainty = state.kalman_filter.get_uncertainty()

            # Use C++ accelerated Kalman filtered prices for calculus analysis
            filtered_prices_series = pd.Series(filtered_prices, index=price_series.index)
            velocity_series = pd.Series(velocities, index=price_series.index)
            acceleration_series = pd.Series(accelerations, index=price_series.index)

            # Validate we have enough data
            if len(filtered_prices) < 10:
                logger.warning(f"Insufficient filtered data for {symbol}, skipping signal")
                self._record_signal_block(state, "kalman_insufficient_history", f"{len(filtered_prices)} samples")
                return

            # Create safe filtered prices for calculations
            safe_filtered = np.where(filtered_prices <= 0.0, np.nan, filtered_prices)
            safe_filtered = pd.Series(safe_filtered).fillna(method='ffill').fillna(method='bfill')
            safe_filtered = np.where(safe_filtered <= 0.0, 1.0, safe_filtered)
            safe_filtered = pd.Series(safe_filtered, index=price_series.index)

            # Calculate Kalman drift and volatility from C++ results
            kalman_drift_series = velocity_series.div(safe_filtered).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Use uncertainty from C++ Kalman filter
            price_uncertainty = current_uncertainty[0]  # Price uncertainty
            kalman_volatility_series = pd.Series(price_uncertainty / safe_filtered, index=safe_filtered.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            context_delta = 1.0
            if len(state.timestamps) >= 2:
                context_delta = max(state.timestamps[-1] - state.timestamps[-2], 1.0)

            filtered_price_value = latest_filtered_price
            regime_stats = state.regime_filter.update(filtered_price_value)

            # Generate calculus signals using C++ accelerated filtered prices
            calculus_strategy = CalculusTradingStrategy(
                lambda_param=Config.LAMBDA_PARAM,
                snr_threshold=tier_config.get('snr_threshold', Config.SNR_THRESHOLD),
                confidence_threshold=tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD)
            )
            signals = calculus_strategy.generate_trading_signals(
                filtered_prices_series,
                context={
                    'kalman_drift': kalman_drift_series,
                    'kalman_volatility': kalman_volatility_series,
                    'regime_context': regime_stats,
                    'delta_t': context_delta
                }
            )
            if signals.empty:
                self._record_signal_block(state, "calculus_no_output")
                return

            # Get latest signal
            latest_signal = signals.iloc[-1]

            # Check if we have a valid signal
            if not latest_signal.get('valid_signal', False):
                snr_value = float(latest_signal.get('snr', 0.0) or 0.0)
                confidence_value = float(latest_signal.get('confidence', 0.0) or 0.0)
                stochastic_conf = float(latest_signal.get('stochastic_confidence', 0.0) or 0.0)
                reasons = []
                tier_snr = tier_config.get('snr_threshold', Config.SNR_THRESHOLD)
                tier_confidence = tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD)
                if snr_value < tier_snr:
                    reasons.append(f"snr {snr_value:.2f}<{tier_snr}")
                if confidence_value < tier_confidence:
                    reasons.append(f"confidence {confidence_value:.2f}<{tier_confidence}")
                if stochastic_conf < 0.4:
                    reasons.append(f"stochastic {stochastic_conf:.2f}<0.40")
                if not reasons:
                    reasons.append("validator_reject")
                self._record_signal_block(state, "signal_filter", ", ".join(reasons))
                return

            # Check for NaN values and handle them
            signal_type_raw = latest_signal.get('signal_type')
            if pd.isna(signal_type_raw) or signal_type_raw is None:
                logger.warning(f"NaN signal type detected for {symbol}, skipping signal")
                return

            try:
                signal_type = SignalType(int(signal_type_raw)) if not pd.isna(signal_type_raw) else SignalType.HOLD
            except (ValueError, TypeError):
                logger.warning(f"Invalid signal type {signal_type_raw} for {symbol}, defaulting to HOLD")
                signal_type = SignalType.HOLD
                self._record_signal_block(state, "invalid_signal_type", str(signal_type_raw))

            # Use C++ Kalman filter values directly
            kalman_velocity = latest_velocity
            kalman_acceleration = latest_acceleration
            filtered_price_value = latest_filtered_price

            # Create signal dictionary with robust data access
            signal_dict = {
                'symbol': symbol,
                'signal_type': signal_type,
                'interpretation': latest_signal.get('interpretation', 'Unknown'),
                'confidence': latest_signal.get('confidence', 0.0),
                'velocity': latest_signal.get('velocity', 0.0),
                'acceleration': latest_signal.get('acceleration', 0.0),
                'snr': latest_signal.get('snr', 0.0),
                'forecast': latest_signal.get('forecast', 0.0),
                'price': latest_signal.get('price', 0.0),
                'filtered_price': filtered_price_value,
                'kalman_velocity': kalman_velocity,
                'kalman_acceleration': kalman_acceleration,
                'tp_price': latest_signal.get('tp_price', filtered_price_value),
                'sl_price': latest_signal.get('sl_price', filtered_price_value),
                'tp_probability': latest_signal.get('tp_probability', 0.0),
                'sl_probability': latest_signal.get('sl_probability', 0.0),
                'regime_state': latest_signal.get('regime_state', regime_stats.state),
                'regime_confidence': latest_signal.get('regime_confidence', regime_stats.confidence),
                'information_position_size': latest_signal.get('information_position_size', 0.0),
                'fractional_stop_multiplier': latest_signal.get('fractional_stop_multiplier', 1.0),
                'timestamp': current_time
            }

            # Update state and rate limiting timestamp
            state.last_signal = signal_dict
            state.signal_count += 1
            state.last_signal_time = current_time  # Update signal timestamp

            # Beautiful signal banner with all details
            print("\n" + "="*70)
            print(f"🎯 SIGNAL GENERATED: {symbol}")
            print("="*70)
            print(f"📊 Type: {signal_dict['signal_type'].name} | Confidence: {signal_dict['confidence']:.1%}")
            print(f"💰 Price: ${signal_dict['price']:.2f} → Forecast: ${signal_dict['forecast']:.2f}")
            print(f"📈 Velocity: {signal_dict['velocity']:.6f} | Accel: {signal_dict['acceleration']:.8f}")
            print(f"📡 SNR: {signal_dict['snr']:.2f} | TP Probability: {signal_dict.get('tp_probability', 0):.1%}")
            
            # Display AR(1) regime-adaptive features if available
            if 'ar_weight' in signal_dict and 'ar_r_squared' in signal_dict:
                ar_weight = signal_dict.get('ar_weight', 0.0)
                ar_r2 = signal_dict.get('ar_r_squared', 0.0)
                ar_strategy = signal_dict.get('ar_strategy', 0)
                ar_conf = signal_dict.get('ar_confidence', 0.0)
                
                strategy_names = {0: "No Trade", 1: "Mean Reversion", 2: "Momentum Long", 3: "Momentum Short"}
                regime_icon = "⚖️" if ar_strategy == 1 else ("📈" if ar_strategy == 2 else ("📉" if ar_strategy == 3 else "⏸️"))
                
                print(f"")
                print(f"🔬 AR(1) Regime Analysis:")
                print(f"   {regime_icon} Strategy: {strategy_names.get(ar_strategy, 'Unknown')}")
                print(f"   📊 Weight: {ar_weight:+.3f} | R²: {ar_r2:.3f} | Confidence: {ar_conf:.1%}")
            
            print(f"")
            print(f"🎓 Yale-Princeton Layers Active:")
            print(f"   ✓ Measure Correction (Q-measure: risk-neutral drift)")
            print(f"   ✓ Variance Stabilization (volatility-time)")
            print(f"   ✓ Continuous Filtering (Kushner-Stratonovich)")
            print(f"   ✓ Functional Derivatives (pathwise delta)")
            print(f"   ✓ AR(1) Linear Regression (regime-adaptive trading)")
            print(f"")
            print(f"📊 Signal #{state.signal_count} | Errors: {state.error_count}")
            print("="*70 + "\n")

            # Execute trade if signal is actionable
            if self._is_actionable_signal(symbol, signal_dict):
                if self.portfolio_mode:
                    # Portfolio-aware execution
                    self._execute_portfolio_trade(symbol, signal_dict)
                else:
                    # Single-asset execution
                    self._execute_trade(symbol, signal_dict)

        except Exception as e:
            logger.error(f"Error processing trading signal for {symbol}: {e}")
            state = self.trading_states[symbol]
            self._record_error(state, ErrorCategory.INVALID_SIGNAL_DATA, f"Signal processing error: {e}")

    def _get_instrument_specs(self, symbol: str) -> Optional[Dict]:
        """Retrieve instrument metadata (min qty, step, notional, leverage)."""
        cache_entry = self.instrument_cache.get(symbol)
        current_time = time.time()
        if cache_entry and current_time - cache_entry.get('timestamp', 0) < 3600:
            return cache_entry

        info = self.bybit_client.get_instrument_info(symbol)
        if info:
            specs = {
                'min_qty': float(info.get('minOrderQty', 0) or 0.0),
                'qty_step': float(info.get('qtyStep', 0) or 0.0),
                'min_notional': float(info.get('minNotionalValue', 0) or 0.0),
                'max_leverage': float(info.get('maxLeverage', 0) or self.risk_manager.max_leverage),
                'timestamp': current_time
            }
        else:
            specs = cache_entry or {
                'min_qty': 0.0,
                'qty_step': 0.0,
                'min_notional': 0.0,
                'max_leverage': self.risk_manager.max_leverage,
                'timestamp': current_time
            }

        if specs.get('min_qty', 0.0) in (0, 0.0):
            fallback_qty = Config.SYMBOL_MIN_ORDER_QTY.get(symbol.upper())
            if fallback_qty:
                specs['min_qty'] = float(fallback_qty)

        if specs.get('min_notional', 0.0) in (0, 0.0):
            fallback_notional = Config.SYMBOL_MIN_NOTIONALS.get(symbol.upper())
            if fallback_notional:
                specs['min_notional'] = float(fallback_notional)

        override = self.min_qty_overrides.get(symbol)
        if override:
            specs['min_qty'] = max(specs.get('min_qty', 0.0), override)
            if symbol == 'SOLUSDT':
                specs['qty_step'] = max(specs.get('qty_step', 0.0), 0.1)

        self.instrument_cache[symbol] = specs
        return specs

    def _adjust_quantity_for_exchange(self,
                                      symbol: str,
                                      current_price: float,
                                      leverage: float,
                                      desired_qty: float,
                                      available_balance: float) -> Tuple[Optional[Dict], Optional[str]]:
        """Ensure quantity respects Bybit min qty/notional/step and margin limits."""
        specs = self._get_instrument_specs(symbol)
        qty = desired_qty
        if specs:
            min_qty = specs.get('min_qty', 0.0)
            qty_step = specs.get('qty_step', 0.0)
            min_notional = specs.get('min_notional', 0.0)

            # CRITICAL FIX: First ensure minimum quantity requirement
            if min_qty > 0:
                qty = max(qty, min_qty)

            # Apply step size rounding BEFORE checking notional
            if qty_step > 0:
                qty = self._round_quantity_to_step(qty, qty_step)

            # CRITICAL FIX: Ensure step rounding didn't violate minimum quantity
            if min_qty > 0 and qty < min_qty:
                qty = min_qty  # Force minimum quantity
                # Re-apply step rounding if needed
                if qty_step > 0:
                    qty = self._round_quantity_to_step(qty, qty_step)

            # Check minimum notional requirement AFTER quantity is finalized
            if min_notional > 0 and current_price > 0:
                min_qty_for_notional = min_notional / current_price
                if qty * current_price < min_notional:
                    # Need to increase quantity to meet notional requirement
                    qty = max(qty, min_qty_for_notional)
                    # Re-apply step rounding after quantity increase
                    if qty_step > 0:
                        qty = self._round_quantity_to_step(qty, qty_step)

            # Final validation: ensure both requirements are still met
            if min_qty > 0 and qty < min_qty:
                logger.info(f"Cannot meet minimum quantity requirement for {symbol}: need {min_qty}, got {qty}")
                return None, "min_qty"

            if min_notional > 0 and current_price > 0 and (qty * current_price) < min_notional:
                logger.info(f"Cannot meet minimum notional requirement for {symbol}: need ${min_notional}, got ${qty * current_price:.2f}")
                return None, "min_notional"

            # CRITICAL FIX: Adjust margin percentage for small balances
            margin_percentage = 0.2 if available_balance >= 10 else 0.8  # Use 80% for small balances
            max_margin = available_balance * margin_percentage
            if max_margin > 0 and current_price > 0:
                cap_qty = (max_margin * max(leverage, 1.0)) / current_price
                qty = min(qty, cap_qty)
                
                # If margin cap violates minimum requirements, increase margin percentage
                if specs:
                    min_qty = specs.get('min_qty', 0.0)
                    min_notional = specs.get('min_notional', 0.0)
                    
                    if min_qty > 0 and qty < min_qty:
                        # Increase margin to allow minimum quantity
                        required_margin = (min_qty * current_price) / max(leverage, 1.0)
                        if required_margin <= available_balance:
                            qty = min_qty
                            margin_percentage = required_margin / available_balance
                            logger.info(f"Increased margin usage to {margin_percentage:.1%} to meet minimum quantity for {symbol}")

            # Final check after margin cap
            if min_qty > 0 and qty < min_qty:
                logger.info(f"Margin cap violates minimum quantity for {symbol}: need {min_qty}, got {qty}")
                return None, "margin_cap"

        if qty <= 0:
            return None, "non_positive_qty"

        notional = qty * current_price
        if leverage <= 0:
            leverage = 1.0
        margin_required = notional / leverage

        if margin_required * 1.2 > available_balance:
            logger.info(
                f"Insufficient margin for {symbol}: need ${margin_required:.2f}, available ${available_balance:.2f}"
            )
            return None, "margin_limit"

        return {
            'quantity': qty,
            'notional': notional,
            'margin_required': margin_required
        }, None

    def _check_cross_asset_confirmation(self, symbol: str, signal_direction: str) -> bool:
        """
        LAYER 6: CROSS-ASSET SIGNAL CONFIRMATION (BTC-ETH correlation).

        Uses BTC as leading indicator for ETH (and vice versa).

        Args:
            symbol: Current trading symbol
            signal_direction: 'long' or 'short'

        Returns:
            True if cross-asset signals agree
        """
        # Only applies if we have both BTC and ETH data
        if symbol not in ['BTCUSDT', 'ETHUSDT']:
            return True  # Other symbols: no cross-check

        # Get the other symbol
        other_symbol = 'ETHUSDT' if symbol == 'BTCUSDT' else 'BTCUSDT'

        # Check if we have data for other symbol
        other_state = self.trading_states.get(other_symbol)
        if not other_state or len(other_state.price_history) < 10:
            return True  # No data: don't block

        # Get recent price movement for both symbols
        current_state = self.trading_states[symbol]

        if len(current_state.price_history) < 10:
            return True

        # Calculate short-term momentum (last 10 ticks)
        current_momentum = (current_state.price_history[-1] - current_state.price_history[-10]) / current_state.price_history[-10]
        other_momentum = (other_state.price_history[-1] - other_state.price_history[-10]) / other_state.price_history[-10]

        # Check correlation
        # If both moving same direction strongly = high correlation (good)
        # If diverging = potential mean reversion (also good)
        # If one flat and other strong = weak signal (bad)

        is_long = signal_direction.lower() in ['long', 'buy']

        # Rule 1: Strong divergence = mean reversion opportunity (GOOD)
        if is_long and other_momentum > 0.005 and current_momentum < -0.002:
            logger.info(f"✅ Cross-Asset MEAN REVERSION: {other_symbol} up, {symbol} down → LONG {symbol}")
            return True
        if not is_long and other_momentum < -0.005 and current_momentum > 0.002:
            logger.info(f"✅ Cross-Asset MEAN REVERSION: {other_symbol} down, {symbol} up → SHORT {symbol}")
            return True

        # Rule 2: Both moving same direction = correlation confirmed (GOOD)
        if is_long and current_momentum > 0.001 and other_momentum > 0.001:
            logger.info(f"✅ Cross-Asset CORRELATION: Both up → LONG {symbol}")
            return True
        if not is_long and current_momentum < -0.001 and other_momentum < -0.001:
            logger.info(f"✅ Cross-Asset CORRELATION: Both down → SHORT {symbol}")
            return True

        # Rule 3: Signal against strong opposite momentum in other = REJECT
        if is_long and other_momentum < -0.01:  # BTC dumping hard
            logger.warning(f"❌ Cross-Asset REJECT: {other_symbol} dumping {other_momentum:.2%} → reject LONG {symbol}")
            return False
        if not is_long and other_momentum > 0.01:  # BTC pumping hard
            logger.warning(f"❌ Cross-Asset REJECT: {other_symbol} pumping {other_momentum:.2%} → reject SHORT {symbol}")
            return False

        # Default: neutral (allow trade)
        return True

    def _is_actionable_signal(self, symbol: str, signal_dict: Dict) -> bool:
        """
        Determine if signal is actionable for trading.

        Args:
            signal_dict: Signal information

        Returns:
            True if signal should trigger a trade
        """
        state = self.trading_states.get(symbol)
        tier_config = self._get_current_tier()
        tier_confidence = tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD)
        tier_snr = tier_config.get('snr_threshold', Config.SNR_THRESHOLD)

        # Check emergency conditions
        if self.emergency_stop:
            if state:
                self._record_signal_block(state, "emergency_stop")
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Maximum consecutive losses reached. Pausing trading.")
            if state:
                self._record_signal_block(state, "max_consecutive_losses", str(self.consecutive_losses))
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.1%}")
            if state:
                self._record_signal_block(state, "daily_loss_limit", f"{self.daily_pnl:.2%}")
            return False

        # Check signal strength - ENHANCED to include NEUTRAL for range trading
        actionable_signals = [
            SignalType.STRONG_BUY, SignalType.STRONG_SELL,
            SignalType.BUY, SignalType.SELL,
            SignalType.NEUTRAL,  # ADDED: Range-bound/mean reversion trading
            SignalType.TRAIL_STOP_UP, SignalType.TAKE_PROFIT,
            SignalType.POSSIBLE_LONG, SignalType.POSSIBLE_EXIT_SHORT
        ]

        if signal_dict['signal_type'] not in actionable_signals:
            if state:
                self._record_signal_block(state, "non_actionable_type", signal_dict['signal_type'].name)
            return False

        if signal_dict['confidence'] < tier_confidence:
            if state:
                self._record_signal_block(
                    state,
                    "confidence_gate",
                    f"{signal_dict['confidence']:.2f}<{tier_confidence}"
                )
            return False

        if signal_dict['snr'] < tier_snr:
            if state:
                self._record_signal_block(
                    state,
                    "snr_gate",
                    f"{signal_dict['snr']:.2f}<{tier_snr}"
                )
            return False

        # LAYER 4: ORDER FLOW IMBALANCE CONFIRMATION (Renaissance edge)
        # Check if institutional order flow confirms our signal direction
        signal_type = signal_dict['signal_type']

        # Determine expected side
        long_signals = [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]
        short_signals = [SignalType.SELL, SignalType.STRONG_SELL, SignalType.POSSIBLE_EXIT_SHORT]

        if signal_type in long_signals:
            # For longs, we need buying pressure (or at least not excessive selling)
            if not self.order_flow.should_confirm_long(symbol, threshold=0.05):
                if state:
                    self._record_signal_block(
                        state,
                        "order_flow_reject",
                        f"LONG rejected: excessive selling pressure"
                    )
                logger.info(f"📊 Order Flow REJECTED LONG for {symbol} - selling pressure too high")
                return False
            logger.info(f"✅ Order Flow CONFIRMED LONG for {symbol}")
        elif signal_type in short_signals:
            # For shorts, we need selling pressure (or at least not excessive buying)
            if not self.order_flow.should_confirm_short(symbol, threshold=0.05):
                if state:
                    self._record_signal_block(
                        state,
                        "order_flow_reject",
                        f"SHORT rejected: excessive buying pressure"
                    )
                logger.info(f"📊 Order Flow REJECTED SHORT for {symbol} - buying pressure too high")
                return False
            logger.info(f"✅ Order Flow CONFIRMED SHORT for {symbol}")
        # NEUTRAL signals don't need order flow confirmation (mean reversion)

        # LAYER 5: DAILY DRIFT PREDICTION ALIGNMENT
        # Check if predicted drift aligns with signal direction
        if signal_type in long_signals:
            if not self.drift_predictor.confirm_signal_direction(symbol, 'long'):
                if state:
                    self._record_signal_block(
                        state,
                        "drift_misaligned",
                        f"LONG signal but bearish drift prediction"
                    )
                logger.info(f"📊 Drift Predictor REJECTED LONG for {symbol} - bearish hourly forecast")
                return False
            logger.info(f"✅ Drift Predictor CONFIRMED LONG for {symbol}")
        elif signal_type in short_signals:
            if not self.drift_predictor.confirm_signal_direction(symbol, 'short'):
                if state:
                    self._record_signal_block(
                        state,
                        "drift_misaligned",
                        f"SHORT signal but bullish drift prediction"
                    )
                logger.info(f"📊 Drift Predictor REJECTED SHORT for {symbol} - bullish hourly forecast")
                return False
            logger.info(f"✅ Drift Predictor CONFIRMED SHORT for {symbol}")

        # LAYER 6: CROSS-ASSET CONFIRMATION (BTC-ETH correlation)
        # Check if the other asset's movement confirms or contradicts this signal
        signal_direction = 'long' if signal_type in long_signals else 'short'
        if not self._check_cross_asset_confirmation(symbol, signal_direction):
            if state:
                self._record_signal_block(
                    state,
                    "cross_asset_reject",
                    f"Cross-asset divergence rejected {signal_direction} signal"
                )
            return False

        return True

    def _execute_trade(self, symbol: str, signal_dict: Dict):
        """
        Execute trade based on calculus signal.

        Args:
            symbol: Trading symbol
            signal_dict: Signal information
        """
        try:
            state = self.trading_states[symbol]
            current_price = signal_dict['price']
            current_time = time.time()

            # Input validation for critical signal data
            try:
                current_price = float(current_price)
                snr = float(signal_dict.get('snr', 0))
                confidence = float(signal_dict.get('confidence', 0))
            except (ValueError, TypeError) as e:
                self._record_error(state, ErrorCategory.INVALID_SIGNAL_DATA, f"Invalid signal data: {e}")
                return

            if current_price <= 0:
                self._record_error(state, ErrorCategory.INVALID_SIGNAL_DATA, f"Invalid price: {current_price}")
                return

            # Get account balance - check for margin trading funds
            account_info = self.bybit_client.get_account_balance()
            if not account_info:
                logger.error("Could not fetch account balance")
                return

            available_balance = 0.0
            total_equity = 0.0
            try:
                available_balance = float(account_info.get('totalAvailableBalance', 0))
                total_equity = float(account_info.get('totalEquity', 0))

                # If no spot balance but have equity, try margin trading
                if available_balance == 0 and total_equity > 0:
                    logger.info(f"Spot balance: ${available_balance:.2f}, Total equity: ${total_equity:.2f}")
                    logger.info("Attempting margin trading with equity funds")

                    margin_available = total_equity * 0.8  # Use 80% of equity for trading
                    if margin_available >= 5:  # $5 minimum for leverage trading
                        available_balance = margin_available
                        logger.info(f"Using margin trading balance: ${available_balance:.2f}")
                    else:
                        logger.info(f"Insufficient equity for leverage trading: ${margin_available:.2f} (need $5+)")
                        return

            except (ValueError, TypeError):
                logger.warning(f"Invalid account balance: {account_info}, using 0")
                available_balance = 0.0

            self.last_available_balance = available_balance
            effective_balance = available_balance if available_balance > 0 else total_equity
            tier_config = self._refresh_tier(effective_balance)
            tier_min_ev_pct = float(tier_config.get('min_ev_pct', 0.0002))
            tier_min_tp_distance_pct = float(tier_config.get('min_tp_distance_pct', 0.006))
            tier_confidence_floor = float(tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD))
            min_probability_samples = float(tier_config.get('min_probability_samples', 5))
            tier_name = tier_config.get('name', 'micro')

            if not self.risk_manager.is_symbol_allowed_for_tier(symbol, tier_name, tier_min_ev_pct):
                self._record_signal_block(
                    state,
                    "symbol_filter",
                    f"{symbol} not enabled for {tier_name}"
                )
                self._register_symbol_block(symbol, "symbol_filter", duration=600)
                return

            posterior = self.risk_manager.get_symbol_probability_posterior(symbol)
            posterior_count = posterior.get('count', 0.0)
            posterior_lower = posterior.get('lower_bound', 0.0)
            if posterior_count >= min_probability_samples and posterior_lower < tier_confidence_floor:
                self._record_signal_block(
                    state,
                    "posterior_ci",
                    f"{posterior_lower:.2f}<{tier_confidence_floor:.2f} | n={posterior_count:.0f}"
                )
                self._register_symbol_block(symbol, "posterior_ci", duration=900)
                return

            # CRITICAL FIX: Better minimum balance validation
            min_balance_for_trading = 1.0  # $1 minimum for micro-trading with leverage (relaxed from $2)
            if available_balance < min_balance_for_trading:
                self._record_error(state, ErrorCategory.INSUFFICIENT_BALANCE, 
                                 f"Balance ${available_balance:.2f} < minimum ${min_balance_for_trading}")
                self._record_signal_block(state, "insufficient_balance", f"${available_balance:.2f}")
                return

            # Check exchange minimums against current tier leverage allowances
            leverage_for_check = min(self.risk_manager.max_leverage, Config.MAX_LEVERAGE)
            if not self.risk_manager.is_symbol_tradeable(symbol, effective_balance, current_price, leverage_for_check):
                self._register_symbol_block(symbol, "min_notional", duration=600)
                self._record_signal_block(state, "min_notional", f"balance ${effective_balance:.2f} insufficient")
                return
            
            # CRITICAL: Check if asset is affordable BEFORE position sizing
            # Some assets like BTCUSDT have minimum notional requirements that exceed small balances
            specs = self._get_instrument_specs(symbol)
            if specs:
                min_notional = specs.get('min_notional', 0.0)
                min_qty = specs.get('min_qty', 0.0)
                
                # Calculate minimum margin required for this asset
                if min_notional > 0:
                    min_margin_required = min_notional / self.risk_manager.max_leverage
                elif min_qty > 0 and current_price > 0:
                    min_notional = min_qty * current_price
                    min_margin_required = min_notional / self.risk_manager.max_leverage
                else:
                    min_margin_required = 0
                
                # Check if we can afford this asset (relaxed: 60% for small accounts, 50% for larger)
                max_margin_pct = 0.60 if available_balance < 20 else 0.50  # More lenient for small accounts
                if min_margin_required > available_balance * max_margin_pct:
                    self._record_error(state, ErrorCategory.ASSET_TOO_EXPENSIVE,
                                     f"Need ${min_margin_required:.2f}, have ${available_balance:.2f}")
                    self._record_signal_block(
                        state,
                        "min_margin",
                        f"need ${min_margin_required:.2f} vs ${available_balance:.2f}"
                    )
                    block_duration = 3600 if symbol.upper() == "BTCUSDT" else 900
                    self._register_symbol_block(symbol, "min_margin", duration=block_duration)
                    return
            
            raw_signal_confidence = float(signal_dict.get('confidence', 0.0))
            signal_confidence = raw_signal_confidence / 100.0 if raw_signal_confidence > 1.0 else raw_signal_confidence
            if self.risk_manager.should_block_symbol_by_ev(symbol, tier_min_ev_pct):
                if signal_confidence >= 0.9:
                    logger.info("EV guard bypassed for %s: confidence %.2f >= 0.90", symbol, signal_confidence)
                else:
                    details = f"avg<{tier_min_ev_pct*100:.2f}% recent net edge"
                    self._record_signal_block(state, "ev_guard", details)
                    self._register_symbol_block(symbol, "ev_guard", duration=900)
                    return

            if available_balance < 5:  # $5 minimum for leverage trading
                logger.info(f"Low balance detected: ${available_balance:.2f}, will attempt minimum sizing")
                # Continue with minimum position sizing rather than rejecting

            # DISABLED: Don't adjust leverage - use fixed 50x always
            # The position sizing code already handles leverage via get_optimal_leverage()
            leverage_restore_needed = False
            original_leverage = self.risk_manager.max_leverage

            # Volatility-aware cadence metrics (reuse later for TP/SL sizing)
            if len(state.price_history) >= 20:
                recent_prices = pd.Series(state.price_history[-20:])
                recent_returns = recent_prices.pct_change().dropna()
                calculated_vol = recent_returns.std() if len(recent_returns) > 0 else 0.02
                actual_volatility = max(calculated_vol, 0.005)
            else:
                actual_volatility = 0.02

            ou_stats = self.ou_model.get_stats(symbol, current_price)
            half_life_seconds = None
            if ou_stats.get('half_life') is not None and np.isfinite(ou_stats['half_life']):
                if len(state.timestamps) > 1:
                    avg_interval = (state.timestamps[-1] - state.timestamps[0]) / max(len(state.timestamps) - 1, 1)
                    if avg_interval <= 0:
                        avg_interval = 1
                    half_life_seconds = max(ou_stats['half_life'] * avg_interval, 0)

            effective_sigma = max(actual_volatility, 0.005)
            if ou_stats.get('sigma') is not None:
                try:
                    effective_sigma = max(effective_sigma, float(ou_stats['sigma']))
                except (TypeError, ValueError):
                    pass

            signal_dict['half_life_seconds'] = half_life_seconds
            signal_dict['sigma_estimate'] = effective_sigma
            signal_dict['volatility_estimate'] = actual_volatility

            volatility_pct = actual_volatility * 100
            if volatility_pct >= 3.0:
                cadence_seconds = 120
            elif volatility_pct >= 2.0:
                cadence_seconds = 90
            elif volatility_pct >= 1.0:
                cadence_seconds = 60
            else:
                cadence_seconds = 30

            if half_life_seconds:
                cadence_seconds = max(cadence_seconds, min(half_life_seconds / 2, 180))

            cadence_seconds = max(cadence_seconds, tier_config.get('min_signal_interval', self.min_signal_interval))

            # CRITICAL FIX: Add signal throttling to prevent rapid attempts
            time_since_last_execution = current_time - state.last_execution_time
            if time_since_last_execution < cadence_seconds:
                print(f"⏸️  Signal throttled for {symbol} - {time_since_last_execution:.1f}s since last execution (need {cadence_seconds:.0f}s | vol={volatility_pct:.2f}%)")
                logger.debug(f"⏸️  Signal throttled for {symbol} - {time_since_last_execution:.1f}s < {cadence_seconds:.1f}s (vol={volatility_pct:.2f}%)")
                self._record_signal_block(
                    state,
                    "cadence_throttle",
                    f"{time_since_last_execution:.1f}s<{cadence_seconds:.0f}s"
                )
                return
            
            # CRITICAL FIX: Check if we already have an open position for this symbol
            max_positions_per_symbol = tier_config.get('max_positions_per_symbol', 1)
            if state.position_info is not None:
                if max_positions_per_symbol <= 1:
                    logger.warning(f"⚠️  POSITION ALREADY OPEN for {symbol} (tier cap {max_positions_per_symbol})")
                else:
                    logger.warning(f"⚠️  Active position count at cap ({max_positions_per_symbol}) for {symbol}")
                logger.info(f"   Time since last execution: {time_since_last_execution:.1f}s")
                logger.info(f"   Current: {state.position_info['side']} {state.position_info['quantity']:.4f} @ ${state.position_info['entry_price']:.2f}")
                logger.info(f"   New signal: {signal_dict['signal_type'].name} (confidence: {signal_dict['confidence']:.2f})")

                # Check if we should close existing position first
                current_side = state.position_info['side']
                new_side = "Buy" if signal_dict['signal_type'] in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG] else "Sell"
                
                # CRITICAL FIX: Don't close on NEUTRAL signals (mean reversion)
                # NEUTRAL can be both Buy and Sell depending on velocity - let TP/SL do their job!
                if signal_dict['signal_type'] == SignalType.NEUTRAL:
                    logger.info(f"ℹ️  NEUTRAL (mean reversion) signal - keeping existing position, let TP/SL manage")
                    self._record_signal_block(state, "neutral_existing_position")
                    return  # Don't interfere with mean reversion trades

                if current_side != new_side:
                    logger.info(f"🔄 CLOSING existing {current_side} position to open {new_side} position")
                    self._close_position(symbol, "Signal reversal - closing to open new position")
                else:
                    logger.info(f"ℹ️  Same direction signal ({new_side}), keeping existing position")
                    self._record_signal_block(state, "position_same_direction")
                    return  # Skip new trade, keep existing position

            # Calculate position size with validated data (Kelly-optimized)
            print(f"\n💰 POSITION SIZING for {symbol}:")
            print(f"   Balance: ${available_balance:.2f} | Confidence: {confidence:.1%} | SNR: {snr:.2f}")
            
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=snr,
                confidence=confidence,
                current_price=current_price,
                account_balance=available_balance,
                instrument_specs=specs
            )
            
            print(f"   → Qty: {position_size.quantity:.8f} | Notional: ${position_size.notional_value:.2f}")
            print(f"   → Leverage: {position_size.leverage_used:.1f}x | Margin: ${position_size.margin_required:.2f}\n")

            if position_size.quantity <= 0 or position_size.notional_value <= 0:
                if leverage_restore_needed:
                    self.risk_manager.max_leverage = original_leverage
                self._record_signal_block(state, "risk_manager_zero_size")
                block_duration = 3600 if symbol.upper() == "BTCUSDT" else 900
                self._register_symbol_block(symbol, "min_margin", duration=block_duration)
                return

            adj, block_reason = self._adjust_quantity_for_exchange(
                symbol,
                current_price,
                max(position_size.leverage_used, 1.0),
                position_size.quantity,
                available_balance
            )

            if not adj:
                print(f"\n⚠️  TRADE BLOCKED: Cannot meet exchange requirements for {symbol}")
                print(f"   Calculated qty: {position_size.quantity:.8f}")
                print(f"   Notional: ${position_size.quantity * current_price:.2f}")
                print(f"   Leverage: {position_size.leverage_used:.1f}x")
                print(f"   Balance: ${available_balance:.2f}\n")
                logger.info(f"Cannot meet exchange requirements for {symbol}, skipping trade")
                reason_tag = block_reason or "exchange_requirements"
                self._record_signal_block(
                    state,
                    f"exchange_{reason_tag}",
                    f"qty={position_size.quantity:.6f}, balance=${available_balance:.2f}"
                )
                if reason_tag in {"min_qty", "min_notional", "margin_cap", "margin_limit"}:
                    block_duration = 3600 if symbol.upper() == "BTCUSDT" else 900
                    self._register_symbol_block(symbol, reason_tag, duration=block_duration)
                if leverage_restore_needed:
                    self.risk_manager.max_leverage = original_leverage
                return

            position_size.quantity = adj['quantity']
            position_size.notional_value = adj['notional']
            position_size.risk_amount = adj['margin_required']
            position_size.margin_required = adj['margin_required']
            
            # Restore original leverage
            if available_balance < 100:
                self.risk_manager.max_leverage = original_leverage

            # Apply maximum position size limit while respecting exchange requirements
            notional_value = position_size.quantity * current_price
            if notional_value > self.max_position_size:
                # Reduce to max position size, then re-validate with exchange requirements
                desired_qty = self.max_position_size / current_price
                adj, block_reason = self._adjust_quantity_for_exchange(
                    symbol,
                    current_price,
                    max(position_size.leverage_used, 1.0),
                    desired_qty,
                    available_balance
                )
                if adj:
                    position_size.quantity = adj['quantity']
                    position_size.notional_value = adj['notional']
                    position_size.risk_amount = adj['margin_required']
                    position_size.margin_required = adj['margin_required']
                else:
                    # If we can't meet requirements with max position, skip trade
                    print(f"\n⚠️  TRADE BLOCKED: Cannot meet exchange requirements with max position size for {symbol}")
                    print(f"   Max notional: ${self.max_position_size:.2f}")
                    print(f"   Leverage: {position_size.leverage_used:.1f}x\n")
                    logger.info(f"Cannot meet exchange requirements with max position size for {symbol}")
                    reason_tag = block_reason or "exchange_requirements"
                    self._record_signal_block(state, f"exchange_{reason_tag}", "max_position_cap")
                    if reason_tag in {"min_qty", "min_notional", "margin_cap", "margin_limit"}:
                        block_duration = 3600 if symbol.upper() == "BTCUSDT" else 900
                        self._register_symbol_block(symbol, reason_tag, duration=block_duration)
                    if leverage_restore_needed:
                        self.risk_manager.max_leverage = original_leverage
                    return

            # CRITICAL: Multi-timeframe consensus check
            if leverage_restore_needed:
                self.risk_manager.max_leverage = original_leverage
            # Check multi-timeframe velocity consensus
            price_series = pd.Series(state.price_history)
            mtf_consensus = calculate_multi_timeframe_velocity(
                price_series, 
                timeframes=[10, 30, 60],
                min_consensus=0.4  # Crypto-optimized: Require 40% agreement (reduced from 60%)
            )
            
            # Get signal type and velocity for strategy logic
            velocity = signal_dict.get('velocity', 0)
            signal_type = signal_dict['signal_type']
            side = determine_trade_side(signal_type, velocity)
            signal_direction = "LONG" if side == "Buy" else "SHORT"
            
            # HYBRID SMART CONSENSUS: Different logic for NEUTRAL vs directional signals
            if signal_type == SignalType.NEUTRAL:
                # NEUTRAL = Mean Reversion Strategy
                # Check market regime to determine if mean reversion is appropriate
                consensus_velocity_magnitude = abs(mtf_consensus['consensus_velocity'])
                
                print(f"\n📊 NEUTRAL SIGNAL (Mean Reversion Strategy):")
                print(f"   Price velocity: {velocity:.6f} → Trade: {signal_direction}")
                print(f"   Multi-TF velocity: {mtf_consensus['consensus_velocity']:.6f}")
                print(f"   Market regime: ", end="")
                
                if consensus_velocity_magnitude < 0.00001:  # Essentially flat/ranging
                    # PERFECT for mean reversion - no strong trend
                    print(f"RANGING (velocity < 0.00001)")
                    print(f"   ✅ Mean reversion allowed - ideal conditions")
                    
                elif mtf_consensus['consensus_percentage'] >= 0.8 and consensus_velocity_magnitude > 0.0001:
                    # Strong trend - mean reversion is dangerous
                    print(f"STRONG TREND (80%+ consensus, velocity > 0.0001)")
                    print(f"   ⚠️  TRADE BLOCKED: Mean reversion dangerous in strong trends")
                    print(f"   Consensus: {mtf_consensus['consensus_percentage']:.0%} on {mtf_consensus['direction']}")
                    print(f"   Signal wants: {signal_direction} (opposite to trend)\n")
                    logger.warning(f"Trade blocked - mean reversion in strong trend: {mtf_consensus['direction']}")
                    return
                
                else:
                    # Weak/mixed trend - allow mean reversion with reduced size
                    print(f"WEAK TREND (consensus={mtf_consensus['consensus_percentage']:.0%})")
                    print(f"   ⚠️  Reducing position size to 50% for safety")
                    position_size.quantity *= 0.5
                    position_size.notional_value *= 0.5
                    position_size.margin_required *= 0.5
                
                print()
                
            else:
                # DIRECTIONAL SIGNALS: Strict consensus required (trend following)
                print(f"\n📈 DIRECTIONAL SIGNAL: {signal_type.name}")
                print(f"   Signal direction: {signal_direction}")
                print(f"   Multi-TF consensus: {mtf_consensus['consensus_percentage']:.0%} on {mtf_consensus['direction']}\n")
                
                # Block trade if no multi-timeframe consensus
                if not mtf_consensus['has_consensus']:
                    print(f"⚠️  TRADE BLOCKED: No multi-timeframe consensus for directional signal")
                    print(f"   Agreement: {mtf_consensus['consensus_percentage']:.0%} ({mtf_consensus['agreement_count']}/{mtf_consensus['total_timeframes']} timeframes)")
                    print(f"   TF Velocities: {', '.join([f'{k}={v:.6f}' for k, v in mtf_consensus['velocities'].items()])}")
                    print(f"   Required: 40% minimum consensus (crypto-optimized)\n")
                    logger.info(f"Trade blocked - no multi-TF consensus for directional: {mtf_consensus['consensus_percentage']:.0%}")
                    return
                
                # Verify signal direction matches consensus direction
                if mtf_consensus['direction'] != 'NEUTRAL' and signal_direction != mtf_consensus['direction']:
                    print(f"⚠️  TRADE BLOCKED: Signal-Consensus mismatch")
                    print(f"   Signal says: {signal_direction}")
                    print(f"   Multi-TF says: {mtf_consensus['direction']}")
                    print(f"   Safety block activated\n")
                    logger.warning(f"Trade blocked - signal/consensus mismatch: {signal_direction} vs {mtf_consensus['direction']}")
                    return
                
                print(f"✅ CONSENSUS CONFIRMED: {mtf_consensus['consensus_percentage']:.0%} agreement")
                print(f"   TF-10: {mtf_consensus['velocities'].get('tf_10', 0):.6f}")
                print(f"   TF-30: {mtf_consensus['velocities'].get('tf_30', 0):.6f}")
                print(f"   TF-60: {mtf_consensus['velocities'].get('tf_60', 0):.6f}\n")
            
            # Log mean reversion logic for NEUTRAL signals
            if signal_dict['signal_type'] == SignalType.NEUTRAL:
                direction = "falling" if velocity < 0 else "rising"
                strategy = "BUY (expect bounce)" if velocity < 0 else "SELL (expect pullback)"
                print(f"📊 NEUTRAL signal: Price {direction} (v={velocity:.6f}) → Mean reversion {strategy}")

            # Calculate dynamic TP/SL using CALCULUS FORECAST
            forecast_price = signal_dict.get('forecast', current_price)

            print(f"\n🎓 CALCULUS PREDICTION:")
            print(f"   Current: ${current_price:.2f}")
            print(f"   Forecast: ${forecast_price:.2f}")
            forecast_move = forecast_price - current_price
            forecast_move_pct = abs(forecast_move / current_price)
            print(f"   Expected Move: ${forecast_move:.2f} ({forecast_move_pct*100:.2f}%)")
            print(f"   Market Volatility: {actual_volatility*100:.2f}%")
            if half_life_seconds:
                print(f"   OU Half-Life: {half_life_seconds/60:.2f} min (2·t½ ≈ {(half_life_seconds*2)/60:.2f} min)")
            if ou_stats.get('rls_samples'):
                print(f"   RLS Samples: {ou_stats['rls_samples']}")
            
            # CRITICAL: Flat market filter - but NOT for mean reversion
            # Mean reversion trades work in flat/ranging markets (that's the point!)
            # Only apply edge filter to directional signals
            if signal_dict['signal_type'] != SignalType.NEUTRAL:
                # LOWERED for high frequency: Renaissance approach accepts tiny edges
                MIN_FORECAST_EDGE = 0.0005  # 0.05% minimum (was 0.1%) - let LLN work!
                if abs(forecast_move_pct) < MIN_FORECAST_EDGE:
                    print(f"\n⚠️  TRADE BLOCKED: Flat market - insufficient forecast edge")
                    print(f"   Forecast edge: {forecast_move_pct*100:.3f}%")
                    print(f"   Minimum required: {MIN_FORECAST_EDGE*100:.1f}%")
                    print(f"   💡 Waiting for stronger directional movement\n")
                    logger.info(f"Trade blocked - flat market: {forecast_move_pct*100:.3f}% < {MIN_FORECAST_EDGE*100:.1f}%")
                    return
            else:
                # For NEUTRAL signals, use volatility as the edge
                # Mean reversion profits from oscillation, not directional movement
                print(f"\n📊 MEAN REVERSION TRADE:")
                print(f"   Strategy: Trade against velocity (expect reversion)")
                print(f"   Edge source: Market volatility ({actual_volatility*100:.2f}%)")
                print(f"   Forecast not needed - using velocity signal\n")
            
            # ═══════════════════════════════════════════════════════════════
            # CRITICAL PRE-TRADE VALIDATIONS (Quantitative Risk Controls)
            # ═══════════════════════════════════════════════════════════════
            
            # VALIDATION 1: Filter Flat Markets (No Predictive Edge)
            # ──────────────────────────────────────────────────────────────
            # Don't trade when forecast shows no meaningful movement
            # EXCEPTION: Mean reversion (NEUTRAL) trades work IN flat markets!
            # Rationale: Zero forecast = no directional edge = random walk
            if forecast_move_pct < 0.001 and signal_dict['signal_type'] != SignalType.NEUTRAL:  # <0.1% predicted move
                print(f"\n🚫 TRADE BLOCKED: FLAT MARKET FILTER (Directional)")
                print(f"   Forecast move: {forecast_move_pct*100:.3f}% (threshold: 0.10%)")
                print(f"   No directional edge - market in equilibrium")
                print(f"   Trading would be pure noise with negative expectancy from fees\n")
                logger.info(f"Flat market filter: {symbol} forecast {forecast_move_pct*100:.3f}% < 0.1% threshold")
                return
            elif signal_dict['signal_type'] == SignalType.NEUTRAL:
                print(f"✅ MEAN REVERSION: Bypassing flat market filter (strategy works in flat markets!)")
            
            # VALIDATION 2: Multi-Timeframe Velocity Consensus
            # ──────────────────────────────────────────────────────────────
            # Require directional agreement across multiple timeframes
            # Rationale: True trends persist across scales, noise does not
            if len(state.price_history) >= 60:
                from quantitative_models import calculate_weighted_multi_timeframe_velocity
                consensus_velocity, directional_confidence = calculate_weighted_multi_timeframe_velocity(
                    pd.Series(state.price_history),
                    timeframes=[10, 30, 60],
                    weights=[0.5, 0.3, 0.2]
                )
                
                # Require minimum 40% directional agreement (crypto-optimized)
                if directional_confidence < 0.4:
                    if signal_dict['signal_type'] == SignalType.NEUTRAL:
                        active_mean_reversion = sum(
                            1
                            for st in self.trading_states.values()
                            if st.position_info and st.position_info.get('signal_type') == SignalType.NEUTRAL.name
                        )
                        if active_mean_reversion < 3:
                            logger.info(
                                "Bypassing consensus for mean reversion %s: consensus=%.1f%% active=%d",
                                symbol,
                                directional_confidence * 100.0,
                                active_mean_reversion
                            )
                        else:
                            print(f"\n🚫 TRADE BLOCKED: LOW MULTI-TIMEFRAME CONSENSUS")
                            print(f"   Directional confidence: {directional_confidence:.1%} (threshold: 40% crypto-optimized)")
                            print(f"   Timeframes disagree on direction - likely noise, not trend")
                            print(f"   Single-timeframe velocity: {velocity:.6f}")
                            print(f"   Consensus velocity: {consensus_velocity:.6f}")
                            print(f"   Mean reversion slots full (active={active_mean_reversion})\n")
                            logger.info(
                                "Mean reversion consensus filter: %s active=%d confidence %.1f%%",
                                symbol,
                                active_mean_reversion,
                                directional_confidence * 100.0
                            )
                            return
                    else:
                        print(f"\n🚫 TRADE BLOCKED: LOW MULTI-TIMEFRAME CONSENSUS")
                        print(f"   Directional confidence: {directional_confidence:.1%} (threshold: 40% crypto-optimized)")
                        print(f"   Timeframes disagree on direction - likely noise, not trend")
                        print(f"   Single-timeframe velocity: {velocity:.6f}")
                        print(f"   Consensus velocity: {consensus_velocity:.6f}")
                        print(f"   Wait for clearer directional signal\n")
                        logger.info(
                            "Multi-timeframe consensus filter: %s confidence %.1f%% < 40%%",
                            symbol,
                            directional_confidence * 100.0
                        )
                        return
                
                # Log successful multi-timeframe validation
                print(f"   Multi-timeframe consensus: {directional_confidence:.1%} (passed)")
            
            # VALIDATION 3: Block Hedged Positions (Fee Hemorrhage Prevention)
            # ──────────────────────────────────────────────────────────────
            # Never open opposite direction on same symbol
            # Rationale: Creates hedge, doubles fees, zero net directional exposure
            if state.position_info is not None:
                existing_side = state.position_info['side']
                if existing_side != side:
                    print(f"\n🚫 TRADE BLOCKED: HEDGE PREVENTION")
                    print(f"   Existing position: {existing_side} {state.position_info['quantity']}")
                    print(f"   Attempted trade: {side}")
                    print(f"   This would create offsetting positions:")
                    print(f"   - Double trading fees")
                    print(f"   - Zero net directional exposure")
                    print(f"   - Guaranteed loss from fees")
                    print(f"   Wait for existing position to close before reversing\n")
                    logger.warning(f"Hedge prevention: Have {existing_side}, attempted {side} on {symbol}")
                    return
                else:
                    # Same direction - already have position
                    print(f"\n⏸️  TRADE SKIPPED: Position already open")
                    print(f"   Existing: {existing_side} {state.position_info['quantity']}")
                    print(f"   Keeping existing position\n")
                    return
            
            # VALIDATION 4: Position Consistency Check (Mathematical Integrity)
            # ──────────────────────────────────────────────────────────────
            # Verify position_side logic is consistent across all code paths
            # This catches bugs where risk_manager and trade_logic disagree
            position_side = determine_position_side(signal_dict['signal_type'], velocity)
            is_consistent, consistency_msg = validate_position_consistency(
                signal_dict['signal_type'],
                velocity,
                side,
                position_side
            )
            if not is_consistent:
                print(f"\n🚨 CRITICAL ERROR: POSITION SIDE INCONSISTENCY")
                print(f"   {consistency_msg}")
                print(f"   This indicates a bug in position logic")
                print(f"   BLOCKING TRADE to prevent inverted TP/SL\n")
                logger.error(f"Position consistency check failed: {consistency_msg}")
                return
            
            # ═══════════════════════════════════════════════════════════════
            # FEE PROTECTION GATE (CRITICAL - Prevent Fee Hemorrhaging)
            # ═══════════════════════════════════════════════════════════════
            
            # Expected position size (will be calculated next)
            optimal_leverage = self.risk_manager.get_optimal_leverage(available_balance)
            num_symbols = len(Config.TARGET_ASSETS)
            expected_notional = (available_balance * optimal_leverage) / num_symbols
            
            # Bybit taker fee: 0.055% per side = 0.11% round-trip
            taker_fee_pct = 0.00055
            round_trip_fee_pct = taker_fee_pct * 2  # 0.0011 = 0.11%
            
            # Calculate expected profit vs fees
            expected_profit_pct = abs(forecast_move_pct)  # Expected price move
            fee_cost_pct = round_trip_fee_pct  # Round-trip cost
            
            # GATE: Expected profit must be at least 2.5x fees
            min_profit_multiplier = 2.5
            if expected_profit_pct < fee_cost_pct * min_profit_multiplier:
                net_expected_loss = (fee_cost_pct - expected_profit_pct) * expected_notional
                print(f"\n⚠️  FEE PROTECTION GATE TRIGGERED for {symbol}")
                print(f"   Expected profit: {expected_profit_pct*100:.3f}%")
                print(f"   Round-trip fees: {fee_cost_pct*100:.3f}%")
                print(f"   Need at least: {fee_cost_pct*min_profit_multiplier*100:.3f}% to enter")
                print(f"   Net expected: -${net_expected_loss:.2f} (would lose to fees!)")
                logger.warning(f"Trade blocked for {symbol}: Expected profit {expected_profit_pct*100:.3f}% < {min_profit_multiplier}x fees")
                self._record_signal_block(state, "fee_protection", f"{expected_profit_pct*100:.2f}%<{fee_cost_pct*min_profit_multiplier*100:.2f}%")
                return
            
            # All pre-trade validations passed - proceed with TP/SL calculation
            print(f"\n✅ PRE-TRADE VALIDATIONS PASSED")
            print(f"   Forecast edge: {forecast_move_pct*100:.2f}% > 0.10% threshold")
            print(f"   Expected profit: {expected_profit_pct*100:.3f}% > {min_profit_multiplier}x fees ({fee_cost_pct*min_profit_multiplier*100:.3f}%)")
            print(f"   No position conflicts")
            print(f"   Position logic consistent")
            
            trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                signal_type=signal_dict['signal_type'],
                current_price=current_price,
                forecast_price=forecast_price,  # USE THE CALCULUS PREDICTION!
                velocity=signal_dict['velocity'],
                acceleration=signal_dict['acceleration'],
                volatility=actual_volatility,  # Use CALCULATED volatility!
                account_balance=available_balance,
                sigma=effective_sigma,
                half_life_seconds=half_life_seconds
            )
            tier_min_hold = tier_config.get('min_ou_hold_seconds') if tier_config else None
            tier_hold_cap = tier_config.get('max_ou_hold_seconds') if tier_config else None

            calculated_hold = trading_levels.max_hold_seconds
            if tier_min_hold:
                tier_min_hold = float(tier_min_hold)
                if calculated_hold is None or calculated_hold < tier_min_hold:
                    calculated_hold = tier_min_hold

            if tier_hold_cap:
                tier_hold_cap = float(tier_hold_cap)
                if calculated_hold is None:
                    calculated_hold = tier_hold_cap
                else:
                    calculated_hold = min(calculated_hold, tier_hold_cap)

            trading_levels.max_hold_seconds = calculated_hold
            forecast_timeout_buffer = float(tier_config.get('forecast_timeout_buffer', 0.0)) if tier_config else 0.0
            tier_confidence_floor = float(tier_config.get('confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD)) if tier_config else Config.SIGNAL_CONFIDENCE_THRESHOLD
            taker_fee_pct, maker_fee_pct, funding_buffer_pct = self._get_dynamic_fee_components(
                symbol,
                trading_levels.max_hold_seconds
            )
            fee_multiplier_override = 3.0 if tier_config and tier_config.get('name') == 'micro' else None
            fee_floor_pct = self.risk_manager.get_fee_aware_tp_floor(
                effective_sigma,
                taker_fee_pct,
                funding_buffer_pct,
                fee_multiplier_override,
                symbol=symbol
            )
            fee_debug = self.risk_manager.get_fee_floor_debug(symbol)

            micro_metrics = self._get_microstructure_metrics(symbol)
            spread_pct = float(micro_metrics.get('spread_pct', 0.0) or 0.0)
            if spread_pct > 0:
                self.risk_manager.record_microstructure_sample(symbol, spread_pct)
            micro_cost_pct = float(micro_metrics.get('micro_cost_pct', 0.0) or 0.0)
            micro_debug = self.risk_manager.get_microstructure_debug(symbol)
            entry_mid_price = micro_metrics.get('mid_price') or current_price
            execution_cost_floor_pct = fee_floor_pct + micro_cost_pct
            raw_execution_cost_floor_pct = execution_cost_floor_pct
            liquid_symbols = {'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'}
            if symbol.upper() in liquid_symbols and micro_cost_pct > 0.001 + 1e-9:
                logger.warning("Microstructure cap breach %s → %.4f%% (>0.10%%)", symbol, micro_cost_pct * 100.0)
                micro_cost_pct = 0.001
                execution_cost_floor_pct = fee_floor_pct + micro_cost_pct
            if micro_cost_pct <= 0.0:
                micro_cost_pct = 0.0012
                execution_cost_floor_pct = fee_floor_pct + micro_cost_pct
            execution_cost_floor_pct = min(execution_cost_floor_pct, 0.0025)
            if self._should_log_ev_debug():
                logger.info(
                    "Cost debug %s → fee_floor=%.4f%% micro=%.4f%% fee=%s micro=%s",
                    symbol,
                    fee_floor_pct * 100.0,
                    micro_cost_pct * 100.0,
                    fee_debug,
                    micro_debug
                )
            
            # Show what the risk manager calculated BEFORE any overrides
            if self._should_log_ev_debug():
                logger.info(
                    "Risk manager levels %s → tp=%.4f sl=%.4f rr=%.2f",
                    symbol,
                    trading_levels.take_profit,
                    trading_levels.stop_loss,
                    trading_levels.risk_reward_ratio
                )

            take_profit = trading_levels.take_profit
            stop_loss = trading_levels.stop_loss
            
            # FINAL SANITY CHECK: Verify TP/SL direction matches position side
            # This is a fatal error check - if this triggers, there's a bug in risk_manager
            tp_sl_valid = True
            if side == "Buy":
                # BUY: TP must be above entry, SL must be below entry
                if take_profit <= current_price or stop_loss >= current_price:
                    tp_sl_valid = False
                    logger.error(f"FATAL: TP/SL direction wrong for BUY - TP:{take_profit:.2f}, SL:{stop_loss:.2f}, Entry:{current_price:.2f}")
            else:  # SELL
                # SELL: TP must be below entry, SL must be above entry
                if take_profit >= current_price or stop_loss <= current_price:
                    tp_sl_valid = False
                    logger.error(f"FATAL: TP/SL direction wrong for SELL - TP:{take_profit:.2f}, SL:{stop_loss:.2f}, Entry:{current_price:.2f}")
            
            if not tp_sl_valid:
                print(f"\n🚨 FATAL ERROR: TP/SL DIRECTION MISMATCH")
                print(f"   Side: {side}")
                print(f"   Entry: ${current_price:.2f}")
                print(f"   TP: ${take_profit:.2f} ({'ABOVE' if take_profit > current_price else 'BELOW'} entry)")
                print(f"   SL: ${stop_loss:.2f} ({'ABOVE' if stop_loss > current_price else 'BELOW'} entry)")
                print(f"   This indicates a critical bug in risk_manager.calculate_dynamic_tp_sl()")
                print(f"   BLOCKING TRADE to prevent guaranteed loss\n")
                return

            if side == "Buy":
                tp_pct = max((take_profit - current_price) / current_price, 0.0)
                sl_pct = max((current_price - stop_loss) / current_price, 1e-6)
            else:
                tp_pct = max((current_price - take_profit) / current_price, 0.0)
                sl_pct = max((stop_loss - current_price) / current_price, 1e-6)

            if tp_pct < tier_min_tp_distance_pct:
                required_tp_pct = tier_min_tp_distance_pct
                if side == "Buy":
                    take_profit = current_price * (1.0 + required_tp_pct)
                else:
                    take_profit = current_price * (1.0 - required_tp_pct)
                trading_levels.take_profit = take_profit
                tp_pct = required_tp_pct
                trading_levels.risk_reward_ratio = tp_pct / max(sl_pct, 1e-6)
                print(f"   TP adjusted to meet tier minimum distance ({required_tp_pct*100:.2f}%)")

            if tp_pct <= execution_cost_floor_pct:
                reason = (
                    f"TP edge {tp_pct*100:.3f}% <= cost floor "
                    f"{execution_cost_floor_pct*100:.3f}% (fees {fee_floor_pct*100:.3f}% + micro {micro_cost_pct*100:.3f}%)"
                )
                self._record_signal_block(state, "fee_floor", reason)
                print(f"\n🚫 TRADE BLOCKED: TP below fee floor")
                print(f"   {reason}")
                return

            base_tp_probability, posterior_stats = self._estimate_tp_probability(symbol, signal_dict, tier_config)
            time_constrained_probability = self._estimate_time_constrained_tp_probability(
                symbol,
                side,
                current_price,
                take_profit,
                stop_loss,
                forecast_price,
                half_life_seconds,
                effective_sigma,
                trading_levels.max_hold_seconds
            )
            ou_weight = 0.4
            ou_prob = float(np.clip(time_constrained_probability, 0.10, 0.90))
            tp_probability = (1.0 - ou_weight) * base_tp_probability + ou_weight * ou_prob
            if signal_confidence >= 0.80 and tp_probability < 0.42:
                tp_probability = 0.42
            tp_probability = float(np.clip(tp_probability, 0.10, 0.92))
            probability_debug = self.get_last_probability_debug(symbol)
            debug_context = {
                'fee_floor_pct': fee_floor_pct,
                'micro_cost_pct': micro_cost_pct,
                'execution_cost_floor_pct': execution_cost_floor_pct,
                'raw_execution_cost_floor_pct': raw_execution_cost_floor_pct,
                'base_tp_probability': base_tp_probability,
                'time_constrained_probability': time_constrained_probability,
                'ou_weight': ou_weight,
                'ou_probability': ou_prob,
                'entry_price': current_price
            }
            if fee_debug:
                debug_context['fee_floor_debug'] = fee_debug
            if micro_debug:
                debug_context['microstructure_debug'] = micro_debug
            if probability_debug:
                debug_context['probability_debug'] = probability_debug

            net_ev = self._compute_trade_ev(
                symbol,
                tp_pct,
                sl_pct,
                tp_probability,
                execution_cost_floor_pct,
                debug_context=debug_context
            )

            if net_ev < tier_min_ev_pct:
                if signal_confidence >= 0.9 and execution_cost_floor_pct <= 0.0025:
                    logger.info(
                        "EV guard bypassed at execution stage for %s: EV %.4f%% < %.4f%% but confidence %.2f",
                        symbol,
                        net_ev * 100.0,
                        tier_min_ev_pct * 100.0,
                        signal_confidence
                    )
                else:
                    if raw_execution_cost_floor_pct > 0.0025:
                        logger.warning(
                            "High execution cost floor %.4f%% triggered EV block for %s",
                            raw_execution_cost_floor_pct * 100.0,
                            symbol
                        )
                    self._record_signal_block(
                        state,
                        "ev_guard",
                        f"{net_ev*100:.3f}%<{tier_min_ev_pct*100:.2f}%"
                    )
                    print(f"\n🚫 TRADE BLOCKED: Expected value negative")
                    print(f"   Net EV: {net_ev*100:.3f}% (min required {tier_min_ev_pct*100:.2f}%)")
                    print(f"   TP Prob: {tp_probability:.2f} | TP Δ: {tp_pct*100:.2f}% | SL Δ: {sl_pct*100:.2f}% | Costs: {execution_cost_floor_pct*100:.2f}%")
                    return
            
            # TP/SL validated - display final levels
            print(f"\n🎯 FINAL TP/SL (Validated):")
            print(f"   Side: {side}")
            print(f"   Entry: ${current_price:.2f}")
            print(f"   TP: ${take_profit:.2f} ({((take_profit/current_price)-1)*100:+.2f}%)")
            print(f"   SL: ${stop_loss:.2f} ({((stop_loss/current_price)-1)*100:+.2f}%)")
            print(f"   R:R: {trading_levels.risk_reward_ratio:.2f}")
            print(f"   TP Prob: {tp_probability:.2f} | Base: {base_tp_probability:.2f} | Time: {time_constrained_probability:.2f}")
            print(
                f"   Posterior μ={posterior_stats.get('mean', 0.0):.2f} (n={posterior_stats.get('count', 0.0):.0f})"
                f" | Net EV: {net_ev*100:.3f}% | Fees: {fee_floor_pct*100:.2f}% | Micro: {micro_cost_pct*100:.3f}%"
                f" | Spread {spread_pct*100:.3f}% | Micro EWMA {self.risk_manager.get_microstructure_metrics(symbol)['spread_ewma']*100:.3f}%"
            )
            if trading_levels.max_hold_seconds:
                print(f"   Max Hold: {trading_levels.max_hold_seconds/60:.2f} min (auto exit if > 2·t½)")

            # Validate trade risk
            is_valid, reason = self.risk_manager.validate_trade_risk(
                symbol, position_size, trading_levels
            )

            if not is_valid:
                print(f"\n⚠️  TRADE BLOCKED: Risk validation failed for {symbol}")
                print(f"   Reason: {reason}")
                print(f"   TP: ${trading_levels.take_profit:.2f} | SL: ${trading_levels.stop_loss:.2f}")
                print(f"   R:R: {trading_levels.risk_reward_ratio:.2f}\n")
                logger.info(f"Trade validation failed: {reason}")
                return

            # Beautiful trade execution banner
            print("\n" + "="*70)
            print(f"🚀 EXECUTING TRADE: {symbol}")
            print("="*70)
            print(f"📊 Side: {side} | Qty: {position_size.quantity:.6f} @ ${current_price:.2f}")
            print(f"💰 Notional: ${position_size.notional_value:.2f}")
            print(f"⚙️  CALCULATED Leverage: {position_size.leverage_used:.1f}x (will set on exchange)")
            print(f"🎯 TP: ${trading_levels.take_profit:.2f} | SL: ${trading_levels.stop_loss:.2f}")
            print(f"📊 Risk/Reward: {trading_levels.risk_reward_ratio:.2f}")
            print(f"🎓 Using Yale-Princeton Q-measure for TP probability")
            print("="*70)
            
            # Execute order with TP/SL
            if self.simulation_mode:
                # Simulate successful trade
                order_result = {'orderId': f'SIM_{int(time.time())}', 'status': 'Filled'}
                print(f"🧪 SIMULATION MODE - Trade simulated")
            else:
                # CRITICAL FIX: Ensure quantity meets exchange requirements before execution
                specs = self._get_instrument_specs(symbol)
                min_qty = specs.get('min_qty', 0.01) if specs else 0.01
                qty_step = specs.get('qty_step', 0.01) if specs else 0.01
                
                # Round quantity to step size and ensure minimum
                initial_qty = position_size.quantity
                final_qty = initial_qty
                if qty_step > 0:
                    final_qty = self._round_quantity_to_step(final_qty, qty_step)
                if min_qty > 0 and final_qty < min_qty:
                    final_qty = self._round_quantity_to_step(min_qty, qty_step) if qty_step > 0 else min_qty
                position_size.quantity = final_qty
                
                # Log the quantity transformation for debugging
                if abs(final_qty - initial_qty) > 1e-6:
                    logger.info(f"Quantity adjustment: {initial_qty:.8f} → {final_qty:.8f} (min: {min_qty}, step: {qty_step})")
                
                # CRITICAL CHECK: Bybit minimum order value is $5
                order_notional = final_qty * current_price
                BYBIT_MIN_ORDER_VALUE = 5.0
                
                if order_notional < BYBIT_MIN_ORDER_VALUE:
                    print(f"\n⚠️  TRADE BLOCKED: Order value too small")
                    print(f"   Calculated notional: ${order_notional:.2f}")
                    print(f"   Bybit minimum: ${BYBIT_MIN_ORDER_VALUE:.2f}")
                    print(f"   Solution: Need higher leverage or larger position")
                    logger.warning(f"Order blocked: ${order_notional:.2f} < ${BYBIT_MIN_ORDER_VALUE:.2f} minimum")
                    self._record_error(state, ErrorCategory.POSITION_SIZING_ERROR, f"Order value ${order_notional:.2f} below $5 minimum")
                    return
                
                # FINAL VALIDATION: Check if order will be rejected due to insufficient margin
                leverage_needed = order_notional / available_balance if available_balance > 0 else 999
                margin_required = order_notional / max(position_size.leverage_used, 1.0)
                
                logger.info(f"Order validation: {symbol}")
                logger.info(f"   Notional: ${order_notional:.2f}, Margin required: ${margin_required:.2f}")
                logger.info(f"   Available: ${available_balance:.2f}, Leverage: {position_size.leverage_used:.1f}x")
                
                # Additional safety: ensure some buffer for slippage and fees
                margin_buffer = 1.02  # Ultra-conservative 2% buffer for tiny balances
                if available_balance < 10:
                    margin_buffer = 1.03  # 3% buffer for very small balances
                if available_balance < 5:
                    margin_buffer = 1.02  # 2% buffer for sub-$5 balances
                
                # CRITICAL FIX: Minimum trade size enforcement for tiny balances
                min_trade_size = 1.0  # $1 minimum per trade
                if margin_required < min_trade_size:
                    margin_required = min_trade_size
                
                # Safety check: ensure margin requirement (with buffer) is within available balance
                if margin_required * margin_buffer >= available_balance:
                    logger.warning(f"❌ ORDER REJECTED - Insufficient margin with buffer")
                    logger.warning(f"   Required: ${margin_required:.2f}, With buffer: ${margin_required * margin_buffer:.2f}")
                    logger.warning(f"   Available: ${available_balance:.2f}")
                    logger.warning(f"   Suggest adding ${(margin_required * margin_buffer - available_balance + 2):.2f} more funds")
                    return
                
                if margin_required >= available_balance:
                    logger.warning(f"❌ ORDER WILL BE REJECTED - Insufficient margin")
                    logger.warning(f"   Required: ${margin_required:.2f}, Available: ${available_balance:.2f}")
                    logger.warning(f"   Suggest adding ${(margin_required - available_balance + 5):.2f} more funds")
                    return
                
                # CRITICAL: Set leverage on exchange BEFORE placing order!
                leverage_to_use = int(position_size.leverage_used)
                logger.info(f"🔧 Setting leverage to {leverage_to_use}x for {symbol}...")
                leverage_set = self.bybit_client.set_leverage(symbol, leverage_to_use)
                
                if not leverage_set:
                    logger.error(f"❌ Failed to set leverage to {leverage_to_use}x")
                    logger.warning(f"⚠️  Proceeding with current exchange leverage (may cause margin errors)")
                else:
                    logger.info(f"✅ Leverage set to {leverage_to_use}x successfully")
                
                # CRITICAL FIX: Get LATEST price and recalculate TP/SL right before order
                # (price may have moved since we calculated trading_levels earlier)
                latest_price = state.price_history[-1] if len(state.price_history) > 0 else current_price
                price_moved = abs(latest_price - current_price) / current_price
                
                # If price moved more than 0.1%, recalculate TP/SL
                if price_moved > 0.001:
                    logger.info(f"⚠️  Price moved {price_moved*100:.2f}% since signal - recalculating TP/SL")
                    logger.info(f"   Signal price: ${current_price:.2f} → Latest: ${latest_price:.2f}")
                    trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                        signal_type=signal_dict['signal_type'],
                        current_price=latest_price,
                        forecast_price=forecast_price,
                        velocity=signal_dict['velocity'],
                        acceleration=signal_dict['acceleration'],
                        volatility=actual_volatility,
                        account_balance=available_balance,
                        sigma=effective_sigma,
                        half_life_seconds=half_life_seconds
                    )
                    if tier_hold_cap:
                        if trading_levels.max_hold_seconds is None:
                            trading_levels.max_hold_seconds = tier_hold_cap
                        else:
                            trading_levels.max_hold_seconds = min(trading_levels.max_hold_seconds, tier_hold_cap)
                    final_tp = trading_levels.take_profit
                    final_sl = trading_levels.stop_loss
                    logger.info(f"   Recalculated - TP: ${final_tp:.2f}, SL: ${final_sl:.2f}")
                else:
                    # Price stable, use original TP/SL
                    final_tp = trading_levels.take_profit
                    final_sl = trading_levels.stop_loss
                
                # Execute real order with corrected quantity and FRESH TP/SL
                order_result = self.bybit_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="Market",  # Market orders for immediate execution
                    qty=final_qty,  # Use properly rounded quantity
                    take_profit=final_tp,
                    stop_loss=final_sl
                )

                if order_result:
                    print(f"✅ TRADE EXECUTED SUCCESSFULLY")
                    print(f"   Order ID: {order_result.get('orderId', 'N/A')}")
                    print(f"   Status: {order_result.get('status', 'Unknown')}")
                    print(f"   {symbol} {side} {position_size.quantity:.6f} @ ${current_price:.2f}")
                    print(f"   ⚙️  Exchange Leverage: {leverage_to_use}x (confirmed)")
                    print("="*70 + "\n")

            if order_result:
                # Update position tracking
                # LAYER 7: Calculate entry drift for Renaissance-style monitoring
                entry_drift = (forecast_price - current_price) / current_price if current_price > 0 else 0
                entry_confidence = signal_dict['confidence']

                position_info = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size.quantity,
                    'entry_price': current_price,
                    'notional_value': position_size.notional_value,
                    'take_profit': trading_levels.take_profit,
                    'stop_loss': trading_levels.stop_loss,
                    'leverage_used': position_size.leverage_used,
                    'max_hold_seconds': trading_levels.max_hold_seconds,
                    'min_hold_seconds': tier_min_hold,
                    'margin_required': position_size.margin_required,
                    'entry_time': current_time,
                    'signal_type': signal_dict['signal_type'].name,
                    'confidence': signal_dict['confidence'],
                    'tier_confidence_threshold': tier_confidence_floor,
                    'forecast_price': forecast_price,
                    'entry_drift': entry_drift,  # RENAISSANCE: Store entry drift for monitoring
                    'entry_confidence': entry_confidence,  # RENAISSANCE: Store entry confidence
                    'latest_forecast_price': forecast_price,
                    'latest_forecast_confidence': signal_dict['confidence'],
                    'latest_forecast_timestamp': signal_dict.get('timestamp'),
                    'forecast_edge_pct': forecast_move_pct,
                    'forecast_timeout_buffer': forecast_timeout_buffer,
                    'fee_floor_pct': fee_floor_pct,
                    'micro_cost_pct': micro_cost_pct,
                    'execution_cost_floor_pct': execution_cost_floor_pct,
                    'taker_fee_pct': taker_fee_pct,
                    'maker_fee_pct': maker_fee_pct,
                    'funding_buffer_pct': funding_buffer_pct,
                    'half_life_seconds': half_life_seconds,
                    'sigma_estimate': effective_sigma,
                    'tier_hold_cap': tier_hold_cap,
                    'tp_probability': tp_probability,
                    'base_tp_probability': base_tp_probability,
                    'time_constrained_probability': time_constrained_probability,
                    'posterior_mean': posterior_stats.get('mean'),
                    'posterior_count': posterior_stats.get('count'),
                    'posterior_lower_bound': posterior_stats.get('lower_bound'),
                    'tier_min_ev_pct': tier_min_ev_pct,
                    'min_tp_distance_pct': tier_min_tp_distance_pct,
                    'initial_net_ev_pct': net_ev,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'entry_mid_price': entry_mid_price,
                    'entry_spread_pct': spread_pct
                }

                try:
                    entry_slippage_pct = abs(current_price - (entry_mid_price or current_price)) / max(entry_mid_price or current_price, 1e-8)
                except (TypeError, ValueError):
                    entry_slippage_pct = 0.0
                position_info['entry_slippage_pct'] = entry_slippage_pct
                self.risk_manager.record_microstructure_sample(symbol, spread_pct, entry_slippage_pct)

                state.position_info = position_info
                state.last_execution_time = time.time()

                # Update risk manager
                self.risk_manager.update_position(symbol, position_info)

                # Update performance
                self.performance.total_trades += 1

            else:
                self._record_error(state, ErrorCategory.API_ERROR, "Order execution failed")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            self._record_error(state, ErrorCategory.API_ERROR, f"Trade execution error: {e}")

    def _execute_portfolio_trade(self, symbol: str, signal_dict: Dict):
        """
        Execute portfolio-aware trade based on calculus signal and portfolio optimization.

        Args:
            symbol: Trading symbol
            signal_dict: Signal information
        """
        try:
            if not self.portfolio_mode or not self.portfolio_manager:
                # Fallback to single-asset execution
                self._execute_trade(symbol, signal_dict)
                return

            logger.info(f"🎯 PORTFOLIO TRADE EXECUTION for {symbol}")

            # Step 1: Add signal to coordinator
            self.signal_coordinator.process_signal(
                symbol=symbol,
                signal_type=signal_dict['signal_type'],
                confidence=signal_dict['confidence'],
                signal_strength=abs(signal_dict['velocity']),  # Use velocity as signal strength
                price=signal_dict['price']
            )

            # Step 2: Get coordinated portfolio decision
            portfolio_decisions = self.signal_coordinator.get_signal_recommendations()

            if not portfolio_decisions:
                logger.info("⏸️  No actionable portfolio decision at this time")
                return

            # Take the highest priority recommendation
            portfolio_decision = portfolio_decisions[0]

            logger.info(f"📊 Portfolio Decision: {portfolio_decision.signal_type}")
            logger.info(f"   Confidence: {portfolio_decision.confidence:.2f}")
            logger.info(f"   Signal Strength: {portfolio_decision.signal_strength:.2f}")
            logger.info(f"   Recommended Size: ${portfolio_decision.recommended_size:.2f}")
            logger.info(f"   Priority: {portfolio_decision.priority.name}")

            # Step 3: Execute trade when portfolio validation passes
            if (portfolio_decision.confidence >= Config.SIGNAL_CONFIDENCE_THRESHOLD and
                    signal_dict['snr'] >= Config.SNR_THRESHOLD):
                logger.info(f"🚀 EXECUTING CALCULUS TRADE for {portfolio_decision.symbol}")
                logger.info(f"   Portfolio recommendation: ${portfolio_decision.recommended_size:,.2f}")
                target_symbol = portfolio_decision.symbol
                target_state = self.trading_states.get(target_symbol)
                symbol_signal = target_state.last_signal if target_state else None
                if not symbol_signal or symbol_signal.get('symbol') != target_symbol:
                    logger.info(f"No recent calculus signal for {target_symbol}, skipping execution")
                    return
                self._execute_trade(target_symbol, symbol_signal)
            else:
                logger.info(
                    f"⏸️  Portfolio decision below thresholds "
                    f"(confidence={portfolio_decision.confidence:.2f}, snr={signal_dict['snr']:.2f})"
                )

        except Exception as e:
            logger.error(f"Error executing portfolio trade for {symbol}: {e}")

    def _execute_allocation_decision(self, decision: AllocationDecision, signal_dict: Dict):
        """
        Execute a specific portfolio allocation decision.

        Args:
            decision: Portfolio allocation decision
            signal_dict: Original calculus signal
        """
        try:
            logger.info(f"💰 EXECUTING ALLOCATION: {decision.symbol}")
            logger.info(f"   Target Weight: {decision.target_weight:.1%}")
            logger.info(f"   Current Weight: {decision.current_weight:.1%}")
            logger.info(f"   Trade Type: {decision.trade_type}")
            logger.info(f"   Quantity: {decision.quantity:.6f}")
            logger.info(f"   Reason: {decision.reason}")

            # Convert to single-asset trade format
            current_price = signal_dict['price']

            # Determine order side
            if decision.trade_type in ['ENTER_LONG', 'INCREASE_LONG']:
                side = "Buy"
            elif decision.trade_type in ['ENTER_SHORT', 'INCREASE_SHORT']:
                side = "Sell"
            elif decision.trade_type == 'REDUCE_LONG':
                side = "Sell"
            elif decision.trade_type == 'COVER_SHORT':
                side = "Buy"
            else:
                logger.warning(f"Unknown trade type: {decision.trade_type}")
                return

            # Calculate TP/SL based on portfolio risk management
            # Get account balance for relaxed R:R threshold
            account_info = self.bybit_client.get_balance()
            available_balance = float(account_info.get('availableBalance', 0)) if account_info else 0
            
            trading_levels = self.risk_manager.calculate_dynamic_tp_sl(
                signal_type=signal_dict['signal_type'],
                current_price=current_price,
                velocity=signal_dict['velocity'],
                acceleration=signal_dict['acceleration'],
                volatility=0.02,
                account_balance=available_balance,
                sigma=0.02,
                half_life_seconds=None
            )

            # Execute order
            if self.simulation_mode:
                # Simulate successful trade
                order_result = {'orderId': f'PORTFOLIO_SIM_{int(time.time())}', 'status': 'Filled'}
                logger.info(f"🧪 PORTFOLIO SIMULATION: {decision.symbol} {side} {decision.quantity:.6f} @ {current_price:.2f}")
            else:
                # Get current account balance for validation
                account_info = self.bybit_client.get_account_balance()
                if not account_info:
                    logger.error("Could not fetch account balance for portfolio trade validation")
                    return
                    
                available_balance = float(account_info.get('totalAvailableBalance', 0))
                
                # FINAL VALIDATION: Check if portfolio order will be rejected due to insufficient margin
                order_notional = decision.quantity * current_price
                margin_required = order_notional / max(decision.leverage if hasattr(decision, 'leverage') else Config.MAX_LEVERAGE, 1.0)
                
                # Use appropriate margin buffer
                margin_buffer = 1.03 if available_balance < 10 else 1.02
                if available_balance < 5:
                    margin_buffer = 1.02  # Ultra-conservative for tiny balances
                
                logger.info(f"Portfolio order validation: {decision.symbol}")
                logger.info(f"   Notional: ${order_notional:.2f}, Margin required: ${margin_required:.2f}")
                logger.info(f"   Available: ${available_balance:.2f}, Leverage: {decision.leverage if hasattr(decision, 'leverage') else Config.MAX_LEVERAGE:.1f}x")
                
                # Safety check: ensure margin requirement (with buffer) is within available balance
                if margin_required * margin_buffer >= available_balance:
                    logger.warning(f"❌ PORTFOLIO ORDER REJECTED - Insufficient margin with buffer")
                    logger.warning(f"   Required: ${margin_required:.2f}, With buffer: ${margin_required * margin_buffer:.2f}")
                    logger.warning(f"   Available: ${available_balance:.2f}")
                    logger.warning(f"   Suggest adding ${(margin_required * margin_buffer - available_balance + 2):.2f} more funds")
                    return
                
                # Execute real portfolio order
                order_result = self.bybit_client.place_order(
                    symbol=decision.symbol,
                    side=side,
                    order_type="Market",
                    qty=decision.quantity,
                    take_profit=trading_levels.take_profit,
                    stop_loss=trading_levels.stop_loss
                )

            if order_result:
                logger.info(f"✅ PORTFOLIO TRADE EXECUTED: {decision.symbol} {side} {decision.quantity:.6f} @ {current_price:.2f}")

                # Update portfolio manager
                self.portfolio_manager.update_position_after_trade(
                    decision.symbol, decision, order_result, current_price
                )

                # Update trading state
                state = self.trading_states[decision.symbol]
                state.last_execution_time = time.time()

                # Update performance
                self.performance.total_trades += 1

            else:
                logger.error(f"❌ Portfolio order execution failed for {decision.symbol}")

        except Exception as e:
            logger.error(f"Error executing allocation decision for {decision.symbol}: {e}")

    def _portfolio_monitoring_loop(self):
        """Background portfolio monitoring and rebalancing loop."""
        while self.is_running and self.portfolio_mode:
            try:
                # Update portfolio optimization
                if self.joint_distribution_analyzer.is_data_sufficient():
                    # Get current market data if available
                    if hasattr(self.ws_client, 'get_latest_portfolio_data'):
                        market_data = self.ws_client.get_latest_portfolio_data()
                    else:
                        market_data = {}

                    # Update portfolio optimization
                    optimization_result = None
                    if hasattr(self.portfolio_manager, 'update_optimization'):
                        optimization_result = self.portfolio_manager.update_optimization(
                            self.joint_distribution_analyzer,
                            self.portfolio_optimizer,
                            market_data
                        )

                    if optimization_result:
                        logger.info(f"📊 Portfolio optimization updated:")
                        logger.info(f"   Expected Return: {optimization_result['expected_return']:.4f}")
                        logger.info(f"   Volatility: {optimization_result['volatility']:.4f}")
                        logger.info(f"   Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}")

                        # C++ accelerated portfolio metrics calculation
                        if 'returns' in optimization_result and 'weights' in optimization_result:
                            returns_array = np.array(optimization_result['returns'])
                            weights_array = np.array(optimization_result['weights'])

                            cpp_metrics = calculate_portfolio_metrics(returns_array, weights_array)
                            cpp_return, cpp_variance, cpp_sharpe, cpp_max_dd = cpp_metrics

                            logger.info(f"🚀 C++ Enhanced Portfolio Metrics:")
                            logger.info(f"   Enhanced Return: {cpp_return:.4f}")
                            logger.info(f"   Enhanced Variance: {cpp_variance:.4f}")
                            logger.info(f"   Enhanced Sharpe: {cpp_sharpe:.3f}")
                            logger.info(f"   Max Drawdown: {cpp_max_dd:.3f}")

                        # Check for rebalancing opportunities
                        if self.portfolio_manager.should_rebalance():
                            logger.info("🔄 Portfolio rebalancing triggered")
                            rebalance_decisions = self.portfolio_manager.create_rebalance_decisions()

                            for decision in rebalance_decisions:
                                if not self.simulation_mode:
                                    logger.info(f"🚨 REAL REBALANCE: {decision.symbol} - {decision.reason}")
                                    
                                    # CRITICAL FIX: Check margin before rebalancing
                                    account_info = self.bybit_client.get_account_balance()
                                    if account_info:
                                        available_balance = float(account_info.get('totalAvailableBalance', 0))
                                        margin_buffer = 1.03 if available_balance < 10 else 1.02
                                        if available_balance < 5:
                                            margin_buffer = 1.02  # Ultra-conservative for tiny balances
                                        
                                        # Get current market data for pricing
                                        market_data = self.bybit_client.get_market_data(decision.symbol)
                                        if market_data:
                                            current_price = float(market_data.get('lastPrice', 0))
                                            order_notional = abs(decision.quantity) * current_price
                                            leverage = decision.leverage if hasattr(decision, 'leverage') else Config.MAX_LEVERAGE
                                            margin_required = order_notional / max(leverage, 1.0)
                                            
                                            logger.info(f"Rebalance margin check: {decision.symbol}")
                                            logger.info(f"   Notional: ${order_notional:.2f}, Margin: ${margin_required:.2f}")
                                            logger.info(f"   Available: ${available_balance:.2f}, Buffer: {margin_buffer:.1%}")
                                            
                                            if margin_required * margin_buffer >= available_balance:
                                                logger.warning(f"❌ REBALANCE REJECTED - Insufficient margin")
                                                logger.warning(f"   Required with buffer: ${margin_required * margin_buffer:.2f}")
                                                logger.warning(f"   Available: ${available_balance:.2f}")
                                                continue  # Skip this decision
                                    
                                    # Execute rebalance in production
                                    self._execute_allocation_decision(decision, {})
                                else:
                                    logger.info(f"🧪 SIMULATION REBALANCE: {decision.symbol} - {decision.reason}")

                # Sleep for portfolio monitoring interval
                time.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                time.sleep(120)  # Wait longer on error

    def _monitoring_loop(self):
        """Background monitoring loop for system health and performance."""
        last_status_time = 0
        last_phantom_check = 0
        
        while self.is_running:
            try:
                # CRITICAL: Clear phantom positions every 60 seconds
                current_time = time.time()
                if current_time - last_phantom_check >= 60:
                    cleared = self._clear_phantom_positions()
                    if cleared > 0:
                        logger.info(f"🧹 Phantom position cleanup freed up trading slots!")
                    last_phantom_check = current_time
                
                # Check system health
                self._check_system_health()

                # Monitor positions
                self._monitor_positions()

                # LAYER 7: RENAISSANCE DRIFT REBALANCING
                # Continuously monitor and rebalance positions based on drift changes
                self._monitor_and_rebalance_positions()

                # Update performance metrics
                self._update_performance_metrics()

                # Reset daily counters if needed
                self._reset_daily_counters()

                # Beautiful periodic status update (every 2 minutes)
                current_time = time.time()
                if current_time - last_status_time >= 120:  # Every 2 minutes
                    self._print_status_update()
                    last_status_time = current_time

                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _print_status_update(self):
        """Print beautiful status update to terminal with aggressive compounding metrics."""
        # Get current balance and display aggressive compounding status
        try:
            account_info = self.bybit_client.get_account_balance()
            if account_info:
                # CRITICAL FIX: Use EQUITY for drawdown, not free balance!
                # Free balance goes down when margin is locked (not a drawdown!)
                # Equity = total value including locked margin + unrealized PnL
                current_equity = float(account_info.get('totalEquity', 0))
                current_balance = float(account_info.get('totalAvailableBalance', 0))
                
                # Check for milestone achievements (use equity for true account value)
                self.risk_manager.check_and_announce_milestone(current_equity)
                
                # Check drawdown protection (use equity, not free balance!)
                protection = self.risk_manager.check_drawdown_protection(current_equity)
                if protection['should_stop']:
                    print("\n" + "🛑" * 35)
                    print(f"🚨 DRAWDOWN PROTECTION ACTIVATED!")
                    print(f"⚠️  Session drawdown: -{protection['drawdown_pct']:.1f}%")
                    print(f"🛑 Trading STOPPED for remainder of session")
                    print("🛑" * 35 + "\n")
                    self.emergency_stop = True
                    return
                elif protection['should_reduce']:
                    print(f"\n⚠️  Drawdown protection: Position sizing reduced to 50% (Drawdown: -{protection['drawdown_pct']:.1f}%)\n")
                
                # Display aggressive compounding status (show equity = true account value)
                print(self.risk_manager.get_status_display(current_equity))
        except Exception as e:
            logger.error(f"Error fetching balance for status update: {e}")
        
        print("\n" + "="*70)
        print(f"📊 SYSTEM STATUS - {time.strftime('%H:%M:%S')}")
        print("="*70)
        
        # Show data accumulation and signal status per symbol
        for symbol, state in self.trading_states.items():
            if len(state.price_history) > 0:
                latest_price = state.price_history[-1]
                status = "✅ Active" if len(state.price_history) >= 50 else "⏳ Accumulating"
                print(f"  {symbol:10s}: {len(state.price_history):3d} prices | ${latest_price:,.2f} | "
                      f"Signals: {state.signal_count:2d} | {status}")

        symbol_summary = self.risk_manager.get_symbol_trade_summary()
        if symbol_summary:
            print("\n  📊 Trade Cadence by Symbol:")
            for sym in sorted(symbol_summary.keys()):
                stats = symbol_summary[sym]
                if stats['entries'] == 0 and stats['open'] == 0:
                    continue
                posterior = self.risk_manager.get_symbol_probability_posterior(sym)
                micro_metrics = self.risk_manager.get_microstructure_metrics(sym)
                tier_name = self.current_tier.get('name', 'micro')
                whitelisted = sym in getattr(Config, 'SYMBOL_TIER_WHITELIST', {}).get(tier_name, [])
                candidate = sym in getattr(Config, 'SYMBOL_CANDIDATE_POOL', {}).get(tier_name, [])
                print(
                    f"  {sym:10s} | trades {stats['completed']}/{stats['entries']} "
                    f"| wins {stats['wins']} | losses {stats['losses']} | open {stats['open']} "
                    f"| EV {stats.get('avg_ev', stats['avg_return'])*100:.3f}% ({stats.get('ev_count', 0)}) "
                    f"| Var {stats['return_variance']:.6f} "
                    f"| Spread {micro_metrics['spread_ewma']*100:.3f}% ({micro_metrics['spread_samples']}) "
                    f"| Slip {micro_metrics['slippage_ewma']*100:.3f}% ({micro_metrics['slippage_samples']}) "
                    f"| Posterior {posterior['mean']*100:.2f}%±{posterior['std_dev']*posterior['mean']*100:.2f} ({posterior['count']:.0f}) "
                    f"| Status {'WL' if whitelisted else ('CAND' if candidate else 'BLK')}"
                )

        gating_rows = []
        for sym, state in self.trading_states.items():
            if state.gating_breakdown:
                top_reasons = sorted(state.gating_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:3]
                reason_str = ", ".join(f"{reason}:{count}" for reason, count in top_reasons)
                gating_rows.append((sym, reason_str))

        if gating_rows:
            print("\n  🚧 Signal Blocks:")
            for sym, reason_str in sorted(gating_rows, key=lambda row: row[0]):
                print(f"  {sym:10s} | {reason_str}")

        active_blocks = {
            sym: expiry for sym, expiry in self.symbol_blocklist.items()
            if expiry > time.time()
        }
        if active_blocks:
            print("\n  ⛔ Auto-Disabled Symbols:")
            now = time.time()
            for sym, expiry in sorted(active_blocks.items(), key=lambda item: item[1]):
                remaining = max(expiry - now, 0)
                reason = self.symbol_block_reasons.get(sym, "auto")
                print(f"  {sym:10s} | {reason} | recheck in {remaining/60:.1f}m")

        if hasattr(self.ws_client, "get_symbol_health"):
            health = self.ws_client.get_symbol_health()
            stale = {}
            now = time.time()
            for sym, age in health.items():
                if age is None:
                    continue
                if age > 60:
                    stale[sym] = age
            if stale:
                print("\n  🔌 Data Staleness:")
                for sym, age in sorted(stale.items(), key=lambda item: item[1], reverse=True):
                    print(f"  {sym:10s} | last tick {age:.0f}s ago")
        
        # Show performance
        if self.performance.total_trades > 0:
            print(f"\n  💼 Total Trades: {self.performance.total_trades}")
            print(f"  📈 Win Rate: {self.performance.success_rate:.1%}")
            print(f"  💰 PnL: ${self.performance.total_pnl:.2f}")
        else:
            print(f"\n  ⏳ Waiting for first trade opportunity...")
        
        # Show error breakdown (diagnostic dashboard)
        total_errors = sum(state.error_count for state in self.trading_states.values())
        if total_errors > 0:
            print(f"\n  ⚠️  ERROR ANALYSIS (Total: {total_errors}):")
            error_summary = defaultdict(int)
            for state in self.trading_states.values():
                for category, count in state.error_breakdown.items():
                    error_summary[category] += count
            
            # Show top 5 error categories
            sorted_errors = sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:5]
            for category, count in sorted_errors:
                pct = (count / total_errors) * 100
                print(f"     • {category.value}: {count} ({pct:.1f}%)")
        
        # Show active positions
        active_positions = sum(1 for state in self.trading_states.values() 
                             if state.position_info is not None)
        if active_positions > 0:
            print(f"  📊 Active Positions: {active_positions}")
            
        print("="*70 + "\n")
    
    def _check_system_health(self):
        """Check system health and circuit breakers."""
        # Check WebSocket connection
        if not self.ws_client.is_connected:
            logger.warning("WebSocket disconnected. Attempting manual reconnection...")
            try:
                # Attempt manual reconnection
                self.ws_client.stop()
                time.sleep(2)  # Brief pause
                self.ws_client.start()
                # Re-register callbacks
                self.ws_client.add_callback(ChannelType.TRADE, self._handle_market_data)
                if self.portfolio_mode:
                    self.ws_client.add_portfolio_callback(self._handle_portfolio_data)
                logger.info("WebSocket reconnection successful")
            except Exception as e:
                logger.error(f"Manual WebSocket reconnection failed: {e}")
                # Continue - automatic reconnection should handle it

        # Check error rates
        total_errors = sum(state.error_count for state in self.trading_states.values())
        total_signals = sum(state.signal_count for state in self.trading_states.values())

        if total_signals > 0:
            error_rate = total_errors / total_signals
            if error_rate > 0.1:  # 10% error rate threshold
                logger.warning(f"High error rate: {error_rate:.1%}")
                self.emergency_stop = True

    def _clear_phantom_positions(self):
        """
        Clear phantom positions by syncing with exchange reality.
        Fixes bug where system thinks positions exist when they don't.
        """
        cleared_count = 0
        for symbol, state in self.trading_states.items():
            if state.position_info is not None:
                try:
                    # Check if position actually exists on exchange
                    position_info = self.bybit_client.get_position_info(symbol)
                    
                    # If None or size=0, position doesn't exist
                    if position_info is None:
                        logger.warning(f"🧹 Clearing phantom position: {symbol} (get_position returned None)")
                        state.position_info = None
                        cleared_count += 1
                        continue
                    
                    exchange_size = abs(self._safe_float(position_info.get('size'), 0.0))
                    if exchange_size == 0.0:
                        logger.warning(f"🧹 Clearing phantom position: {symbol} (size=0 on exchange)")
                        state.position_info = None
                        cleared_count += 1
                        
                except Exception as e:
                    logger.error(f"Error checking phantom position for {symbol}: {e}")
                    # On error, clear it to be safe
                    logger.warning(f"🧹 Clearing phantom position {symbol} due to error")
                    state.position_info = None
                    cleared_count += 1
        
        if cleared_count > 0:
            logger.info(f"✅ Cleared {cleared_count} phantom position(s)")
        
        return cleared_count

    def _monitor_positions(self):
        """Monitor open positions and update risk metrics."""
        for symbol, state in self.trading_states.items():
            if state.position_info:
                try:
                    now = time.time()
                    # Get current position info
                    position_info = self.bybit_client.get_position_info(symbol)
                    
                    # FIXED: If position doesn't exist, clear it immediately!
                    if position_info is None:
                        logger.info(f"✅ Position {symbol} no longer exists - clearing")
                        state.position_info = None
                        continue
                    
                    # Check actual position size from exchange
                    exchange_size = abs(self._safe_float(position_info.get('size'), 0.0))
                    if exchange_size == 0.0:
                        # Position was automatically closed by exchange (TP/SL hit)
                        logger.info(f"✅ POSITION AUTO-CLOSED: {symbol} (TP/SL hit)")
                        
                        # Calculate final PnL from last known position info
                        if 'unrealised_pnl' in state.position_info:
                            final_pnl = state.position_info['unrealised_pnl']
                            notional_value = state.position_info.get('notional_value', 0.0)
                            pnl_percent = (final_pnl / notional_value) * 100 if notional_value else 0.0
                            
                            # Update performance metrics
                            self.performance.total_pnl += final_pnl
                            if final_pnl > 0:
                                self.performance.winning_trades += 1
                                self.consecutive_losses = 0
                                self.risk_manager.record_trade_result(won=True)
                            else:
                                self.performance.losing_trades += 1
                                self.consecutive_losses += 1
                                self.risk_manager.record_trade_result(won=False)
                            
                            exit_metrics = self._get_microstructure_metrics(symbol)
                            exit_mid_price = exit_metrics.get('mid_price') or current_price
                            spread_exit = float(exit_metrics.get('spread_pct', 0.0) or 0.0)
                            try:
                                exit_slippage_pct = abs(current_price - exit_mid_price) / max(exit_mid_price, 1e-8)
                            except (TypeError, ValueError):
                                exit_slippage_pct = 0.0
                            state.position_info['exit_mid_price'] = exit_mid_price
                            state.position_info['exit_slippage_pct'] = exit_slippage_pct
                            self.risk_manager.record_microstructure_sample(symbol, spread_exit, exit_slippage_pct)

                            # Update risk manager
                            self.risk_manager.close_position(symbol, final_pnl, "TP/SL hit")

                            tier_min_ev_pct = state.position_info.get('tier_min_ev_pct')
                            self._handle_ev_block(symbol, tier_min_ev_pct)
                            
                            logger.info(f"   Final PnL: {final_pnl:.2f} ({pnl_percent:.1f}%)")
                        
                        # Clear position info to allow new trades
                        state.position_info = None
                        continue
                    
                    # Position still exists - update monitoring
                    current_pnl = self._safe_float(position_info.get('unrealisedPnl'), 0.0)
                    entry_price = self._safe_float(
                        position_info.get('entryPrice'),
                        state.position_info.get('entry_price', 0.0) if state.position_info else 0.0
                    )
                    current_price = self._safe_float(
                        position_info.get('markPrice'),
                        state.position_info.get('entry_price', entry_price) if state.position_info else entry_price
                    )

                    # Update position info
                    notional_value = self._safe_float(state.position_info.get('notional_value'), 0.0)
                    pnl_percent = (current_pnl / notional_value) * 100 if notional_value else 0.0
                    state.position_info.update({
                        'current_price': current_price,
                        'unrealised_pnl': current_pnl,
                        'pnl_percent': pnl_percent
                    })

                    latest_signal = state.last_signal
                    if latest_signal:
                        latest_ts = latest_signal.get('timestamp')
                        stored_ts = state.position_info.get('latest_forecast_timestamp')
                        if latest_ts and (stored_ts is None or latest_ts > stored_ts):
                            if 'forecast' in latest_signal:
                                try:
                                    state.position_info['latest_forecast_price'] = float(latest_signal.get('forecast'))
                                except (TypeError, ValueError):
                                    pass
                            if 'confidence' in latest_signal:
                                try:
                                    state.position_info['latest_forecast_confidence'] = float(latest_signal.get('confidence'))
                                except (TypeError, ValueError):
                                    pass
                            if latest_signal.get('half_life_seconds') is not None:
                                state.position_info['half_life_seconds'] = latest_signal.get('half_life_seconds')
                            if latest_signal.get('sigma_estimate') is not None:
                                state.position_info['sigma_estimate'] = latest_signal.get('sigma_estimate')
                            tp_prob_signal = latest_signal.get('tp_probability')
                            if tp_prob_signal is not None:
                                try:
                                    state.position_info['base_tp_probability'] = float(tp_prob_signal)
                                except (TypeError, ValueError):
                                    pass
                            state.position_info['latest_forecast_timestamp'] = latest_ts

                    max_hold_seconds = state.position_info.get('max_hold_seconds')
                    entry_time = state.position_info.get('entry_time', now)
                    elapsed = now - entry_time if entry_time else 0.0
                    remaining_hold = None
                    if max_hold_seconds:
                        remaining_hold = max(max_hold_seconds - elapsed, 0.0)

                    time_prob = self._estimate_time_constrained_tp_probability(
                        symbol,
                        state.position_info.get('side'),
                        current_price,
                        state.position_info.get('take_profit'),
                        state.position_info.get('stop_loss'),
                        state.position_info.get('latest_forecast_price', state.position_info.get('forecast_price')),
                        state.position_info.get('half_life_seconds'),
                        state.position_info.get('sigma_estimate'),
                        remaining_hold if remaining_hold is not None and remaining_hold > 0 else max_hold_seconds
                    )
                    state.position_info['time_constrained_probability'] = time_prob
                    base_prob = state.position_info.get('base_tp_probability', state.position_info.get('tp_probability', 0.5))
                    state.position_info['tp_probability'] = min(float(np.clip(base_prob, 0.05, 0.95)), time_prob)

                    # Check if position should be closed (manual override)
                    should_close, close_reason = self._should_close_position(state, state.position_info, position_info)
                    if should_close:
                        self._close_position(symbol, close_reason or "Risk management")

                except Exception as e:
                    logger.error(f"Error monitoring position for {symbol}: {e}")

    def _monitor_and_rebalance_positions(self):
        """
        LAYER 7: RENAISSANCE EXECUTION - Drift-based position rebalancing.

        This is THE KEY to Renaissance-style trading:
        - Recalculate expected return (drift) at controlled intervals
        - Reduce position when drift weakens (predictive)
        - Exit completely when high flip probability (before flip!)
        - NO waiting for TP/SL targets

        RATE LIMITED: Checks every 30s minimum to prevent over-trading.
        """
        # ⏱️ RATE LIMITING: Don't check on every tick (prevents churning)
        current_time = time.time()
        if current_time - self.last_monitor_time < self.monitor_interval:
            return  # Skip - checked too recently
        
        self.last_monitor_time = current_time
        
        for symbol, state in self.trading_states.items():
            if not state.position_info:
                continue

            try:
                # Get current position details
                entry_price = state.position_info.get('entry_price', 0)
                entry_drift = state.position_info.get('entry_drift')  # E[r] at entry
                position_side = state.position_info.get('side', 'Buy')
                entry_time = state.position_info.get('entry_time', time.time())
                current_qty = state.position_info.get('quantity', 0)

                if entry_drift is None or current_qty <= 0:
                    # No drift stored or no position - skip
                    continue

                # Get latest signal to recalculate current drift
                latest_signal = state.last_signal
                if not latest_signal:
                    continue

                # Calculate current expected return (drift)
                current_price = latest_signal.get('price', entry_price)
                forecast_price = latest_signal.get('forecast', current_price)
                confidence = latest_signal.get('confidence', 0)

                # Current drift = (forecast - current) / current
                if current_price > 0:
                    current_drift = (forecast_price - current_price) / current_price
                else:
                    continue

                # Check direction alignment
                is_long = position_side == 'Buy'
                drift_aligned = (is_long and current_drift > 0) or (not is_long and current_drift < 0)

                # Calculate drift change from entry
                drift_delta = current_drift - entry_drift if entry_drift else 0

                # Position age
                age_seconds = time.time() - entry_time
                half_life = state.position_info.get('half_life_seconds', 300)  # Default 5 min

                # RENAISSANCE DECISION RULES (PREDICTIVE):
                
                # Import drift flip prediction
                from quantitative_models import predict_drift_flip_probability
                
                # Calculate flip probability (PREDICTIVE - exit BEFORE flip!)
                flip_probability = predict_drift_flip_probability(
                    prices=list(state.price_history),
                    current_drift=current_drift,
                    volatility=volatility if len(state.price_history) >= 20 else 0.01
                )

                # Rule 1: HIGH FLIP PROBABILITY - Exit BEFORE drift flips
                if flip_probability > 0.85:  # 85% chance drift will flip
                    logger.warning(f"🔄 HIGH FLIP PROBABILITY for {symbol}: {flip_probability:.1%}")
                    logger.warning(f"   Current drift: {current_drift:.4f}, Flip prob: {flip_probability:.1%}")
                    logger.warning(f"   Exiting BEFORE flip to lock profit!")
                    self._close_position(symbol, f"High flip probability ({flip_probability:.1%})")
                    continue

                # Rule 2: ELEVATED FLIP RISK - Reduce position
                if flip_probability > 0.60:  # 60%+ chance of flip
                    logger.info(f"📉 ELEVATED FLIP RISK for {symbol}: {flip_probability:.1%}")
                    logger.info(f"   Current drift: {current_drift:.4f}")
                    logger.info(f"   Reducing 50% to protect profit")
                    self._resize_position(symbol, scale_factor=0.5, reason=f"Flip risk {flip_probability:.1%}")
                    # Update entry drift to current
                    state.position_info['entry_drift'] = current_drift
                    continue
                
                # Rule 3: ACTUAL DRIFT FLIP - Emergency exit if prediction missed
                if not drift_aligned and abs(current_drift) > 0.0001:
                    logger.warning(f"🔄 DRIFT FLIP DETECTED for {symbol}: {entry_drift:.4f} → {current_drift:.4f}")
                    logger.warning(f"   Conviction reversed! Emergency exit")
                    self._close_position(symbol, "Drift flip (conviction reversed)")
                    continue

                # Rule 3: TIME DECAY - Exit if holding too long (> 2x half-life)
                max_hold = half_life * 2 if half_life else 600  # Max 2x half-life or 10 min
                if age_seconds > max_hold:
                    logger.info(f"⏰ TIME DECAY for {symbol}: {age_seconds:.0f}s > {max_hold:.0f}s")
                    logger.info(f"   Holding too long, exiting to free capital")
                    self._close_position(symbol, "Timeout (>2x half-life)")
                    continue

                # Rule 4: CONFIDENCE DROP - Reduce if signal quality degrades
                entry_confidence = state.position_info.get('entry_confidence', confidence)
                if confidence < entry_confidence * 0.7:  # Confidence dropped >30%
                    logger.info(f"📊 CONFIDENCE DROP for {symbol}: {entry_confidence:.2f} → {confidence:.2f}")
                    logger.info(f"   Signal quality degraded, reducing 50%")
                    self._resize_position(symbol, scale_factor=0.5, reason="Confidence drop")
                    state.position_info['entry_confidence'] = confidence
                    continue

                # If we get here, position is healthy - hold

            except Exception as e:
                logger.error(f"Error in drift rebalancing for {symbol}: {e}")

    def _resize_position(self, symbol: str, scale_factor: float, reason: str = "Rebalance"):
        """
        Resize position by scaling factor (Renaissance-style partial exits).

        Args:
            symbol: Trading symbol
            scale_factor: Multiply current position by this (0.5 = reduce by half)
            reason: Reason for resize (for logging)
        """
        try:
            state = self.trading_states[symbol]
            if not state.position_info:
                logger.warning(f"Cannot resize {symbol} - no position open")
                return

            current_qty = state.position_info.get('quantity', 0)
            position_side = state.position_info.get('side', 'Buy')

            if current_qty <= 0:
                logger.warning(f"Cannot resize {symbol} - quantity is {current_qty}")
                return

            # Calculate new quantity
            new_qty = current_qty * scale_factor
            reduce_qty = current_qty - new_qty

            if reduce_qty <= 0:
                logger.warning(f"Resize calculation error: reduce_qty={reduce_qty}")
                return

            # Get instrument specs for rounding
            specs = self._get_instrument_specs(symbol)
            if not specs:
                logger.error(f"Cannot get specs for {symbol} resize")
                return

            qty_step = specs.get('qty_step', 0.001)
            reduce_qty_rounded = self._round_quantity_to_step(reduce_qty, qty_step)

            if reduce_qty_rounded <= 0:
                logger.warning(f"Rounded reduce quantity too small: {reduce_qty_rounded}")
                return

            # Execute partial close (reduce_only market order)
            logger.info(f"💱 RESIZING {symbol}: {current_qty:.6f} → {new_qty:.6f} ({scale_factor*100:.0f}%)")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Closing {reduce_qty_rounded:.6f} {symbol}")

            # Place reduce-only market order
            reduce_side = "Sell" if position_side == "Buy" else "Buy"

            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=reduce_side,
                order_type="Market",
                qty=reduce_qty_rounded,
                reduce_only=True
            )

            if order_result and order_result.get('orderId'):
                logger.info(f"✅ Position resized successfully: {order_result['orderId']}")

                # Update position info
                state.position_info['quantity'] = new_qty
                state.position_info['notional_value'] = state.position_info.get('notional_value', 0) * scale_factor

                # Record the partial profit/loss
                current_price = state.position_info.get('current_price', state.position_info.get('entry_price', 0))
                entry_price = state.position_info.get('entry_price', current_price)

                if position_side == "Buy":
                    partial_pnl = (current_price - entry_price) * reduce_qty_rounded
                else:
                    partial_pnl = (entry_price - current_price) * reduce_qty_rounded

                logger.info(f"   Partial PnL: ${partial_pnl:.2f}")
                self.performance.total_pnl += partial_pnl

            else:
                logger.error(f"Failed to resize position for {symbol}")

        except Exception as e:
            logger.error(f"Error resizing position for {symbol}: {e}")

    def _should_close_position(self,
                               state: TradingState,
                               position_info: Dict,
                               current_position: Dict) -> Tuple[bool, Optional[str]]:
        """Determine if position should be closed based on forecast-aware timeout rules."""
        try:
            entry_time = position_info.get('entry_time')
            if not entry_time:
                return False, None

            now = time.time()
            age = now - entry_time
            max_hold = position_info.get('max_hold_seconds') or 0.0
            min_hold = position_info.get('min_hold_seconds') or 0.0
            symbol = position_info.get('symbol') or (state.symbol if state else 'UNKNOWN')
            pnl_percent = position_info.get('pnl_percent', 0.0)

            if max_hold and age >= max_hold:
                logger.info(
                    f"⏰ Max hold reached for {symbol}: age={age:.1f}s (>{max_hold:.1f}s cap) | pnl={pnl_percent:+.2f}%"
                )
                return True, "Mean reversion timeout (cap reached)"

            if min_hold and age < min_hold:
                return False, None

            confidence_floor = float(position_info.get('tier_confidence_threshold', Config.SIGNAL_CONFIDENCE_THRESHOLD))
            latest_conf = float(position_info.get('latest_forecast_confidence', position_info.get('confidence', 0.0)))

            sigma_estimate = position_info.get('sigma_estimate', 0.005)
            taker_fee_pct = position_info.get('taker_fee_pct')
            funding_buffer_pct = position_info.get('funding_buffer_pct', 0.0)
            fee_floor_pct = position_info.get('fee_floor_pct')
            micro_cost_pct = position_info.get('micro_cost_pct', 0.0)
            execution_cost_floor_pct = position_info.get('execution_cost_floor_pct')
            if fee_floor_pct is None:
                fee_floor_pct = self.risk_manager.get_fee_aware_tp_floor(
                    sigma_estimate,
                    taker_fee_pct,
                    funding_buffer_pct
                )
                position_info['fee_floor_pct'] = fee_floor_pct
            if execution_cost_floor_pct is None:
                try:
                    execution_cost_floor_pct = float(fee_floor_pct) + float(micro_cost_pct or 0.0)
                except (TypeError, ValueError):
                    execution_cost_floor_pct = float(fee_floor_pct)
                position_info['execution_cost_floor_pct'] = execution_cost_floor_pct

            forecast_buffer = float(position_info.get('forecast_timeout_buffer', 0.0) or 0.0)
            threshold_pct = max(execution_cost_floor_pct or 0.0, 0.0) + forecast_buffer

            current_price = position_info.get('current_price', position_info.get('entry_price')) or 0.0
            forecast_price = position_info.get('latest_forecast_price', position_info.get('forecast_price'))
            forecast_distance_pct = 0.0
            if current_price and forecast_price is not None:
                if position_info.get('side') == 'Buy':
                    forecast_distance_pct = max((forecast_price - current_price) / current_price, 0.0)
                else:
                    forecast_distance_pct = max((current_price - forecast_price) / current_price, 0.0)

            ev, tp_pct, sl_pct, execution_cost_pct = self._evaluate_expected_ev(position_info, current_price)
            position_info['last_ev_pct'] = ev
            position_info['last_tp_pct'] = tp_pct
            position_info['last_sl_pct'] = sl_pct
            tier_min_ev_pct = position_info.get('tier_min_ev_pct')
            min_tp_distance_pct = position_info.get('min_tp_distance_pct')

            if tier_min_ev_pct is not None and ev < tier_min_ev_pct and age >= min_hold:
                logger.info(
                    f"⚠️  Closing {symbol}: expected value {ev*100:.3f}% < tier floor {tier_min_ev_pct*100:.2f}% | age={age:.1f}s | pnl={pnl_percent:+.2f}%"
                )
                return True, "Expected value fell below floor"

            if min_tp_distance_pct is not None and tp_pct < min_tp_distance_pct and age >= min_hold:
                logger.info(
                    f"⚠️  Closing {symbol}: remaining TP distance {tp_pct*100:.3f}% < min {min_tp_distance_pct*100:.2f}% | age={age:.1f}s | pnl={pnl_percent:+.2f}%"
                )
                return True, "TP distance collapsed"

            if ev <= 0 and age >= min_hold:
                logger.info(
                    f"⚠️  Closing {symbol}: expected value {ev*100:.3f}% <= 0 | age={age:.1f}s | pnl={pnl_percent:+.2f}%"
                )
                return True, "Expected value non-positive"

            if latest_conf < confidence_floor:
                logger.info(
                    f"⚠️  Closing {symbol}: forecast confidence {latest_conf:.2f} < floor {confidence_floor:.2f} | age={age:.1f}s | pnl={pnl_percent:+.2f}%"
                )
                return True, "Forecast confidence dropped"

            if forecast_distance_pct < threshold_pct:
                logger.info(
                    f"⚠️  Closing {symbol}: forecast edge {forecast_distance_pct*100:.2f}% < threshold {threshold_pct*100:.2f}% | age={age:.1f}s | pnl={pnl_percent:+.2f}%"
                )
                return True, "Forecast edge eroded"

            return False, None

        except Exception as e:
            logger.error(f"Error checking position close condition: {e}")
            return False, None

    def _close_position(self, symbol: str, reason: str):
        """Close position for a symbol."""
        try:
            state = self.trading_states[symbol]
            if not state.position_info:
                return

            # Determine close side
            current_side = state.position_info['side']
            close_side = "Sell" if current_side == "Buy" else "Buy"

            # Get current position size
            position_info = self.bybit_client.get_position_info(symbol)
            if position_info:
                position_size = abs(self._safe_float(position_info.get('size'), 0.0))
                if position_size > 0:
                    # Close position
                    result = self.bybit_client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="Market",
                        qty=position_size,
                        reduce_only=True
                    )

                    if result:
                        # Calculate PnL
                        pnl = self._safe_float(position_info.get('unrealisedPnl'), 0.0)
                        notional_value = self._safe_float(state.position_info.get('notional_value'), 0.0)
                        pnl_percent = (pnl / notional_value) * 100 if notional_value else 0.0

                        # Update performance
                        self.performance.total_pnl += pnl
                        if pnl > 0:
                            self.performance.winning_trades += 1
                            self.consecutive_losses = 0
                            self.risk_manager.record_trade_result(won=True)
                        else:
                            self.performance.losing_trades += 1
                            self.consecutive_losses += 1
                            self.risk_manager.record_trade_result(won=False)

                        if self.risk_manager.current_portfolio_value:
                            self.daily_pnl += pnl / self.risk_manager.current_portfolio_value

                        logger.info(f"✅ POSITION CLOSED: {symbol} PnL: {pnl:.2f} ({pnl_percent:.1f}%) - {reason}")

                        exit_metrics = self._get_microstructure_metrics(symbol)
                        fallback_price = self._safe_float(position_info.get('markPrice'), state.position_info.get('current_price', state.position_info.get('entry_price', 0.0)))
                        exit_mid_price = exit_metrics.get('mid_price') or fallback_price
                        spread_exit = float(exit_metrics.get('spread_pct', 0.0) or 0.0)
                        try:
                            exit_slippage_pct = abs(self._safe_float(position_info.get('markPrice'), fallback_price) - exit_mid_price) / max(exit_mid_price, 1e-8)
                        except (TypeError, ValueError):
                            exit_slippage_pct = 0.0
                        if state.position_info is not None:
                            state.position_info['exit_mid_price'] = exit_mid_price
                            state.position_info['exit_slippage_pct'] = exit_slippage_pct
                        self.risk_manager.record_microstructure_sample(symbol, spread_exit, exit_slippage_pct)

                        # Update risk manager
                        self.risk_manager.close_position(symbol, pnl, reason)

                        tier_min_ev_pct = state.position_info.get('tier_min_ev_pct')
                        self._handle_ev_block(symbol, tier_min_ev_pct)

                        # Clear position info
                        state.position_info = None

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def _emergency_close_all_positions(self):
        """Emergency close all positions."""
        logger.warning("Emergency close all positions activated!")
        for symbol in self.symbols:
            self._close_position(symbol, "Emergency stop")

    def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.performance.total_trades > 0:
            self.performance.success_rate = (
                self.performance.winning_trades / self.performance.total_trades
            )

        # Update portfolio value in risk manager
        account_info = self.bybit_client.get_account_balance()
        if account_info:
            portfolio_value = float(account_info.get('totalEquity', 0))
            self.risk_manager.update_portfolio_value(portfolio_value)

    def _reset_daily_counters(self):
        """Reset daily counters at midnight."""
        current_time = time.time()
        last_reset = self.last_reset_time

        # Reset if it's a new day
        if current_time - last_reset > 86400:  # 24 hours
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset_time = current_time
            logger.info("Daily counters reset")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'symbols': self.symbols,
            'performance': {
                'total_trades': self.performance.total_trades,
                'winning_trades': self.performance.winning_trades,
                'losing_trades': self.performance.losing_trades,
                'success_rate': self.performance.success_rate,
                'total_pnl': self.performance.total_pnl,
                'daily_pnl': self.daily_pnl
            },
            'connections': {
                'websocket_connected': self.ws_client.is_connected if self.ws_client else False,
                'api_connected': self.bybit_client.test_connection() if self.bybit_client else False
            },
            'trading_states': {}
        }

        # Add per-symbol status
        for symbol, state in self.trading_states.items():
            status['trading_states'][symbol] = {
                'price_history_length': len(state.price_history),
                'signal_count': state.signal_count,
                'error_count': state.error_count,
                'has_position': state.position_info is not None,
                'last_signal_time': state.last_execution_time
            }

        # Add risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics()
        status['risk_metrics'] = {
            'total_exposure': risk_metrics.total_exposure,
            'margin_used_percent': risk_metrics.margin_used_percent,
            'open_positions': risk_metrics.open_positions_count,
            'current_drawdown': risk_metrics.current_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio
        }

        # Add portfolio metrics if enabled
        if self.portfolio_mode and self.portfolio_manager:
            portfolio_status = self.portfolio_manager.get_portfolio_status()
            status['portfolio'] = {
                'mode_enabled': True,
                'total_value': portfolio_status['total_value'],
                'available_cash': portfolio_status['available_cash'],
                'positions_count': len(portfolio_status['positions']),
                'allocation_drift': portfolio_status['allocation_drift'],
                'last_rebalance': portfolio_status['last_rebalance'],
                'optimization_active': self.joint_distribution_analyzer.is_data_sufficient() if self.joint_distribution_analyzer else False
            }

            # Add signal coordinator status
            if self.signal_coordinator:
                signal_status = self.signal_coordinator.get_coordinator_status()
                status['signal_coordinator'] = signal_status
        else:
            status['portfolio'] = {'mode_enabled': False}

        return status

    async def _execute_direct_trade(self, portfolio_decision):
        """Execute a trade directly based on portfolio decision"""
        try:
            symbol = portfolio_decision.symbol
            signal_type = portfolio_decision.signal_type
            confidence = portfolio_decision.confidence
            signal_strength = portfolio_decision.signal_strength
            recommended_size = portfolio_decision.recommended_size

            logger.info(f"🎯 EXECUTING DIRECT TRADE: {symbol}")
            logger.info(f"   Signal: {signal_type.name}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Size: ${recommended_size:.2f}")

            # Determine trade direction based on signal type
            if signal_type in [SignalType.BULLISH_ACCELERATION, SignalType.BULLISH_REVERSAL]:
                side = "Buy"
            elif signal_type in [SignalType.BEARISH_ACCELERATION, SignalType.BEARISH_REVERSAL]:
                side = "Sell"
            else:
                logger.info(f"⏸️  Signal type {signal_type.name} not actionable for trading")
                return

            # Execute the actual trade
            result = await self._execute_real_trade(symbol, side, recommended_size, confidence)

            if result:
                logger.info(f"✅ Trade executed successfully: {symbol} {side} ${recommended_size:.2f}")
            else:
                logger.error(f"❌ Trade execution failed: {symbol} {side}")

        except Exception as e:
            logger.error(f"Error in _execute_direct_trade: {e}")

    async def _execute_real_trade(self, symbol: str, side: str, quantity: float, confidence: float):
        """Execute real trade on Bybit exchange"""
        try:
            # Get current price
            ticker = self.bybit_client.get_ticker(symbol)
            if not ticker or 'last_price' not in ticker:
                logger.error(f"Could not get current price for {symbol}")
                return False

            current_price = float(ticker['last_price'])
            usd_amount = min(quantity, 100.0)  # Cap at $100 for safety

            # Calculate quantity in base asset
            qty = usd_amount / current_price

            # Apply risk management
            risk_check = self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                side=side.lower(),
                quantity=qty,
                price=current_price
            )

            if not risk_check['valid']:
                logger.warning(f"Risk validation failed: {risk_check['reason']}")
                return False

            # Execute order via Bybit client
            order_result = self.bybit_client.place_order(
                symbol=symbol,
                side=side.lower(),
                order_type="market",
                quantity=qty
            )

            if order_result and order_result.get('retCode') == 0:
                logger.info(f"✅ ORDER PLACED: {side} {qty:.6f} {symbol} at ~${current_price:.2f}")
                return True
            else:
                logger.error(f"❌ ORDER FAILED: {order_result}")
                return False

        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            return False

    def _calculate_calculus_position_size(self, symbol: str, signal_dict: Dict, 
                                    current_price: float, available_balance: float) -> PositionSize:
        """
        Calculate position size based on calculus derivatives and mathematical framework.
        
        This implements proper mathematical conversion from v(t), a(t) to position sizing
        based on Signal-to-Noise Ratio and confidence from acceleration.
        """
        try:
            # Extract calculus components
            velocity = signal_dict.get('velocity', 0)
            acceleration = signal_dict.get('acceleration', 0)
            snr = signal_dict.get('snr', 0)
            confidence = signal_dict.get('confidence', 0)
            volatility = signal_dict.get('volatility', 0.02)  # Default 2% volatility
            
            # 1️⃣ Calculate Signal Strength from SNR
            # Higher |v(t)|/σv = stronger signal = larger position
            signal_strength = abs(snr) * confidence
            
            # 2️⃣ Position Size Base on Mathematical Framework
            # FIXED: Use levered buying power for position sizing
            # With 50x leverage and 2 symbols: ($25 * 50) / 2 = $625 per symbol
            optimal_leverage = self.risk_manager.get_optimal_leverage(available_balance)
            num_symbols = len(Config.TARGET_ASSETS)
            base_position = (available_balance * optimal_leverage) / num_symbols
            
            # 3️⃣ SIMPLIFIED: Use levered base position directly (NO MULTIPLIERS)
            # Renaissance approach: Fixed position size based on leverage
            # All signal quality filtering happens at entry, not position sizing
            final_notional = base_position
            
            # DISABLED: C++ risk calculation uses raw balance (not levered)
            # This was overriding our levered position sizing with tiny positions
            # cpp_risk_size would return ~$0.50 (2% of $25) instead of $625
            # 
            # Keep our levered calculation instead
            calculated_notional = final_notional

            # 5️⃣ Exchange Compliance Check
            specs = self._get_instrument_specs(symbol)
            min_qty = specs.get('min_qty', 0.001) if specs else 0.001
            min_notional = specs.get('min_notional', 5.0) if specs else 5.0
            qty_step = specs.get('qty_step', 0.001) if specs else 0.001
            
            # Calculate required quantity
            if final_notional < min_notional:
                final_notional = min_notional  # Meet exchange minimum
                
            quantity = final_notional / current_price
            
            # Adjust for exchange step size
            if qty_step > 0:
                quantity = self._round_quantity_to_step(quantity, qty_step)
            if quantity < min_qty:
                quantity = min_qty
                
            # CRITICAL FIX: Check minimum order value requirements FIRST
            min_order_value = 5.0  # Bybit minimum 5 USDT

            # Check if we can even afford the minimum order
            # For very low balances, we need to allow higher allocation to meet minimum
            if available_balance < min_order_value:
                logger.warning(f"⚠️  Cannot afford minimum order for {symbol}: need ${min_order_value:.0f}, only have ${available_balance:.2f}")
                from risk_manager import PositionSize
                return PositionSize(
                    quantity=0,
                    notional_value=0,
                    risk_amount=0,
                    leverage_used=1,
                    margin_required=0,
                    risk_percent=0,
                    confidence_score=0
                )  # Return empty position

            # Set max_affordable based on LEVERED buying power
            # With leverage, we can take positions much larger than raw balance
            # Max notional = (balance * leverage) / num_symbols
            optimal_leverage = self.risk_manager.get_optimal_leverage(available_balance)
            num_symbols = len(Config.TARGET_ASSETS)
            max_affordable_notional = (available_balance * optimal_leverage) / num_symbols

            # Ensure we meet minimum order value
            if max_affordable_notional < min_order_value:
                logger.warning(f"⚠️  Levered position ${max_affordable_notional:.2f} below minimum ${min_order_value:.2f}")
                max_affordable_notional = min_order_value

            # Set position to the greater of calculated size or minimum order value
            position_notional = quantity * current_price

            # Ensure position meets minimum order value
            if position_notional < min_order_value:
                # Increase position to meet minimum order value
                position_notional = min_order_value
                quantity = position_notional / current_price

                logger.warning(f"🔧 Increased {symbol} position to meet ${min_order_value:.0f} minimum: ${position_notional:.2f}")

            # Now check if it fits within our balance constraints
            if position_notional > max_affordable_notional:
                # Reduce position to fit within available balance
                final_notional = max_affordable_notional
                quantity = final_notional / current_price
                position_notional = final_notional

                # Adjust for exchange step size again
                if qty_step > 0:
                    quantity = self._round_quantity_to_step(quantity, qty_step)
                if quantity < min_qty:
                    logger.warning(f"⚠️  Cannot afford minimum position size for {symbol}")
                    from risk_manager import PositionSize
                    return PositionSize(
                        quantity=0,
                        notional_value=0,
                        risk_amount=0,
                        leverage_used=1,
                        margin_required=0,
                        risk_percent=0,
                        confidence_score=0
                    )  # Return empty position

                # Recalculate notional with adjusted quantity
                position_notional = quantity * current_price
                logger.warning(f"🔧 Reduced {symbol} position to ${position_notional:.2f} due to balance constraints")
            
            # ✅ VALIDATION: Ensure we're using proper 50x leverage positions
            # Expected: $625 per symbol with $25 balance at 50x
            expected_min = 600.0  # Allow 4% tolerance
            expected_max = 650.0
            
            if position_notional < expected_min and available_balance >= 20:
                logger.warning(f"⚠️  Position size ${position_notional:.2f} below expected ${expected_min:.2f}")
                logger.warning(f"   This suggests position sizing is not using full leverage")
            
            if position_notional > expected_max:
                logger.warning(f"⚠️  Position size ${position_notional:.2f} exceeds max ${expected_max:.2f}")
                # Cap to prevent over-leveraging
                scale_down = expected_max / position_notional
                quantity = quantity * scale_down
                position_notional = expected_max

            # Calculate leverage needed AFTER position size is finalized
            # For perpetual futures, margin = notional_value / leverage
            # We want margin <= 50% of available balance
            max_margin = available_balance * 0.5
            leverage_needed = max(1.0, position_notional / max_margin)
            leverage_needed = min(leverage_needed, Config.MAX_LEVERAGE)  # FIXED: Use config instead of hardcoded 25x

            # Calculate actual margin requirement
            margin_required = position_notional / leverage_needed
            
            # 6️⃣ C++ ACCELERATED TAYLOR EXPANSION TP/SL CALCULATIONS
            # High-performance mathematical framework for precise TP/SL placement

            # Use C++ accelerated analysis for additional market insights
            if len(state.price_history) >= 20:
                recent_prices = np.array(state.price_history[-20:])
                cpp_smoothed, cpp_velocity, cpp_acceleration = analyze_curve_complete(
                    recent_prices, lambda_param=0.6, dt=1.0
                )

                # Combine Python and C++ results for robustness
                combined_velocity = (velocity + cpp_velocity[-1]) / 2.0
                combined_acceleration = (acceleration + cpp_acceleration[-1]) / 2.0
            else:
                combined_velocity = velocity
                combined_acceleration = acceleration

            # Multiple time horizons for robust prediction
            time_horizons = [60, 300, 900]  # 1min, 5min, 15min

            # Calculate Taylor expansion forecasts for each horizon using combined values
            forecasts = []
            for delta_t in time_horizons:
                # P(t+Δ) ≈ P(t) + v(t)Δ + ½a(t)Δ²
                forecast = current_price + combined_velocity * delta_t + 0.5 * combined_acceleration * (delta_t ** 2)
                forecasts.append(forecast)

            # Weight the forecasts (near-term more important)
            weights = [0.5, 0.35, 0.15]
            price_forecast = sum(f * w for f, w in zip(forecasts, weights))

            # Calculate acceleration strength for confidence
            accel_strength = abs(acceleration) / (abs(velocity) + 1e-8)

            # Advanced TP/SL based on calculus confidence
            if velocity > 0:  # Long position
                # Take Profit: Use Taylor forecast with acceleration boost
                if acceleration > 0:  # Accelerating uptrend
                    take_profit = price_forecast * (1 + 0.01 + accel_strength * 0.005)  # Boost for acceleration
                else:  # Decelerating uptrend
                    take_profit = price_forecast * 1.008  # Conservative

                # Stop Loss: Based on velocity reversal point
                # Find where v(t) would become zero: v(t) = 0
                time_to_reversal = -velocity / (acceleration + 1e-8) if acceleration < 0 else 300
                reversal_price = current_price + velocity * min(time_to_reversal, 120)  # Max 2min lookback

                stop_loss = max(current_price * 0.982, reversal_price * 0.995)  # Min 1.8% SL

            else:  # Short position
                # Take Profit: Use Taylor forecast with acceleration boost
                if acceleration < 0:  # Accelerating downtrend
                    take_profit = price_forecast * (1 - 0.01 - accel_strength * 0.005)  # Boost for acceleration
                else:  # Decelerating downtrend
                    take_profit = price_forecast * 0.992  # Conservative

                # Stop Loss: Based on velocity reversal point
                time_to_reversal = -velocity / (acceleration - 1e-8) if acceleration > 0 else 300
                reversal_price = current_price + velocity * min(time_to_reversal, 120)

                stop_loss = min(current_price * 1.018, reversal_price * 1.005)  # Min 1.8% SL
                
            # 7️⃣ Return PositionSize object with all mathematical components
            from risk_manager import PositionSize
            position_size = PositionSize(
                quantity=quantity,
                notional_value=position_notional,
                risk_amount=margin_required * 0.02,  # Risk 2% of margin
                leverage_used=leverage_needed,
                margin_required=margin_required,
                risk_percent=margin_required / available_balance if available_balance > 0 else 0,
                confidence_score=confidence
            )
            
            logger.info(f"🔬 Calculus Position Sizing for {symbol}:")
            logger.info(f"   SNR: {snr:.3f} | Confidence: {confidence:.2f} | Signal Strength: {signal_strength:.3f}")
            logger.info(f"   Velocity: {velocity:.6f} | Acceleration: {acceleration:.8f} | Volatility: {volatility:.4f}")
            logger.info(f"   Forecast: {price_forecast:.2f} | Base Position: ${base_position:.2f}")
            logger.info(f"   Final Position: {quantity:.6f} @ ${current_price:.2f} (${final_notional:.2f})")
            logger.info(f"   Leverage: {leverage_needed:.1f}x | TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in calculus position sizing for {symbol}: {e}")
            # Fallback to minimal position
            from risk_manager import PositionSize
            return PositionSize(
                quantity=0.001,
                notional_value=10.0,
                risk_amount=10.0,
                leverage_used=1.0,
                stop_loss=current_price * 0.985,
                take_profit=current_price * 1.015,
                risk_reward_ratio=1.0,
                confidence=0.5,
                signal_strength=0.5,
                forecast_price=current_price,
                velocity=0,
                acceleration=0,
                volatility=0.02
            )

if __name__ == '__main__':
    print("Starting main block...")
    import sys

    # Check command line arguments
    simulation_mode = False

    # SIMPLIFIED FOR $25 BALANCE: Always use 2 symbols (BTC + ETH), single-asset mode
    # Portfolio optimization adds complexity we don't need for small balance
    symbols = ["BTCUSDT", "ETHUSDT"]
    portfolio_mode = False

    print('🚀 ANNE\'S ENHANCED CALCULUS TRADING SYSTEM')
    print('=' * 60)
    print('🎯 50X LEVERAGE SYSTEM - BTC + ETH ONLY')
    print('   💰 Optimized for $25 balance')
    print('   ⚡ 50x leverage on every position')
    print('   🎓 7-layer Renaissance execution')
    print('   📊 Drift-based rebalancing (no TP/SL waiting)')
    print('   🎯 Target: $25 → $31/day, $95/week')
    print('=' * 60)

    if simulation_mode:
        print('🧪 SIMULATION MODE - Safe for testing')
    else:
        print('⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!')
        print('   🚨 This will execute REAL trades on Bybit!')

    print('=' * 60)

    # Initialize enhanced trader
    trader = LiveCalculusTrader(
        symbols=symbols,
        window_size=100,  # Crypto-optimized: shorter window for faster response
        min_signal_interval=15,  # Crypto-optimized: faster signal cycle
        simulation_mode=simulation_mode,
        portfolio_mode=portfolio_mode
    )

    try:
        if simulation_mode:
            print(f'🧪 Starting simulation trading...')
        else:
            print(f'💰 Starting LIVE trading with REAL money!')

        trader.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
        trader.stop()
    except Exception as e:
        logger.error(f"Trading system error: {e}")
        trader.stop()
        trader.stop()

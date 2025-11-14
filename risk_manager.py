"""
Risk Manager for Anne's Calculus Trading System
===============================================

This module implements comprehensive risk management for the calculus-based trading system,
including position sizing, dynamic TP/SL calculation, and portfolio risk controls.

Features:
- Signal strength-based position sizing
- Dynamic TP/SL using calculus indicators
- Volatility-adjusted risk parameters
- Portfolio-level risk controls
- Maximum drawdown protection
- Correlation management
- Risk-reward optimization
"""

import numpy as np
import pandas as pd
import logging
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
from config import Config
from utils.math_forecasting_utils import (
    ForecastInputs,
    compute_derivative_trend_score,
    compute_derivative_horizon,
    compute_unified_log_drift,
    optimize_tp_sl
)
from calculus_strategy import SignalType
from position_logic import determine_position_side, validate_position_consistency
import time

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for position sizing"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class PositionSize:
    """Position size calculation result"""
    quantity: float
    notional_value: float
    risk_amount: float
    leverage_used: float
    margin_required: float
    risk_percent: float
    confidence_score: float

@dataclass
class TradingLevels:
    """Dynamic TP/SL levels based on calculus indicators"""
    entry_price: float
    stop_loss: float
    take_profit: float
    trail_stop: Optional[float]
    risk_reward_ratio: float
    position_side: str
    confidence_level: float
    max_hold_seconds: Optional[float] = None
    secondary_take_profit: Optional[float] = None
    primary_tp_fraction: float = 1.0
    secondary_tp_fraction: float = 0.0
    secondary_tp_probability: Optional[float] = None
    forecast_deltas: Optional[Dict[str, float]] = None
    f_score: Optional[float] = None
    curvature_metrics: Optional[Dict[str, float]] = None
    trailing_buffer_pct: Optional[float] = None

@dataclass
class RiskMetrics:
    """Current portfolio risk metrics"""
    total_exposure: float
    available_balance: float
    margin_used_percent: float
    open_positions_count: int
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    correlation_score: float

class RiskManager:
    """
    Comprehensive risk management system for Anne's calculus trading system.

    Implements position sizing based on Signal-to-Noise Ratio (SNR) and signal confidence,
with dynamic TP/SL levels calculated using calculus indicators.
    """

    def __init__(self,
                 max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.10,
                 max_leverage: float = 75.0,
                 min_risk_reward: float = 1.5,
                 max_positions: int = 5,
                 max_correlation: float = 0.7):
        """
        Initialize risk manager with safety parameters.

        Args:
            max_risk_per_trade: Maximum risk per trade as percentage of portfolio
            max_portfolio_risk: Maximum total portfolio risk
            max_leverage: Maximum leverage allowed
            min_risk_reward: Minimum risk-reward ratio
            max_positions: Maximum number of concurrent positions
            max_correlation: Maximum correlation between positions
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_leverage = max_leverage
        self.min_risk_reward = min_risk_reward
        self.max_positions = max_positions
        self.max_correlation = max_correlation

        # Track portfolio state
        self.open_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.max_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.probability_debug: Dict[str, Dict[str, float]] = {}
        self.cadence_debug: Dict[str, Dict[str, float]] = {}
        self.fee_mode_tracker: Dict[str, Dict[str, float]] = {}
        self.symbol_base_notional = {
            sym.upper(): float(cap)
            for sym, cap in getattr(Config, "SYMBOL_BASE_NOTIONALS", {}).items()
            if cap is not None
        }
        self.symbol_notional_overrides = {
            sym.upper(): float(cap)
            for sym, cap in getattr(Config, "SYMBOL_MAX_NOTIONAL_CAPS", {}).items()
            if cap is not None
        }
        self.notional_cap_tiers = sorted(getattr(Config, "NOTIONAL_CAP_TIERS", []), key=lambda item: item[0])
        self.signal_tiers = getattr(Config, "SIGNAL_TIER_CONFIG", [])
        self.symbol_trade_stats: Dict[str, Dict] = {}
        self.symbol_session_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'session_trades': 0,
            'session_pnl': 0.0,
            'session_ev_sum': 0.0,
            'session_ev_count': 0,
            'session_start_balance': 0.0,
            'blocked_until': 0.0
        })

        # Volatility tracking
        self.volatility_window = 20
        self.price_history = {}

        logger.info(f"Risk manager initialized: max_risk_per_trade={max_risk_per_trade:.1%}, "
                   f"max_leverage={max_leverage}x, min_risk_reward={min_risk_reward:.1f}")
        
        # Aggressive compounding settings
        self.aggressive_mode = True  # Enable aggressive compounding
        self.session_high = 0.0
        
        # Phase 4: Sharpe-based leverage (Python fallback)
        self.sharpe_tracker = None  # Will be initialized with Python fallback
        self.leverage_bootstrap = None
        self.use_sharpe_leverage = True
        self.trade_returns = []  # Track for Sharpe calculation
        self.expectancy_metrics = None
        self.ev_window = 50
        self.min_kelly_fraction = 0.02
        self.max_kelly_fraction = 0.60
        self.microstructure_debug: Dict[str, Dict[str, float]] = {}
        self.fee_floor_debug: Dict[str, Dict[str, float]] = {}
        self._warned_fee_multiplier = False
        self._init_sharpe_tracker()
        self.consecutive_losses = 0
        self.milestones = [10, 20, 50, 100, 200, 500, 1000]
        self.reached_milestones = set()
        self.session_start_balance = 0.0
        self.session_start_time = time.time()
        self.calculus_priority_mode = bool(getattr(Config, "CALCULUS_PRIORITY_MODE", False))
        self.force_leverage_enabled = bool(getattr(Config, "FORCE_LEVERAGE_ENABLED", False))
        self.force_leverage_value = float(getattr(Config, "FORCE_LEVERAGE_VALUE", max_leverage))
        self.force_margin_fraction = float(getattr(Config, "FORCE_MARGIN_FRACTION", 0.35))
        self.force_margin_fraction = max(0.0, min(self.force_margin_fraction, 1.0))
        self.calculus_loss_block_threshold = int(getattr(Config, "CALCULUS_LOSS_BLOCK_THRESHOLD", 3))
        if self.force_leverage_enabled:
            self.max_leverage = max(self.max_leverage, self.force_leverage_value)
        self.curvature_failures = defaultdict(int)
        self.curvature_success = defaultdict(int)
        self.fee_recovery_balance = 0.0
        self.total_fees_paid = 0.0
        self.total_realized_pnl = 0.0
        self.var_pnl_window = deque(maxlen=200)
        self.weekly_var_cap = float(getattr(Config, "WEEKLY_VAR_CAP", 0.25))
        self.var_guard_active = False
        self.compounding_ladder = self._parse_compounding_ladder(getattr(Config, "COMPOUNDING_LADDER", ""))
    
    def get_equity_tier(self, account_balance: float) -> Dict:
        """Return configuration tier for a given account balance."""
        if not self.signal_tiers:
            return {
                "snr_threshold": Config.SNR_THRESHOLD,
                "confidence_threshold": Config.SIGNAL_CONFIDENCE_THRESHOLD,
                "min_signal_interval": Config.MIN_SIGNAL_INTERVAL,
                "max_ou_hold_seconds": None,
                "max_positions_per_symbol": 1,
                "max_positions_per_minute": 20
            }

        fallback_name = "tier"
        for tier in self.signal_tiers:
            max_equity = tier.get("max_equity", float("inf"))
            if account_balance < max_equity:
                tier.setdefault("name", tier.get("name") or fallback_name)
                return tier
        self.signal_tiers[-1].setdefault("name", self.signal_tiers[-1].get("name") or fallback_name)
        return self.signal_tiers[-1]

    def get_symbol_min_margin(self, symbol: str, leverage: float) -> float:
        """Return minimum margin required to place the smallest exchange-allowed order."""
        sym = symbol.upper()
        min_qty = Config.SYMBOL_MIN_ORDER_QTY.get(sym, 0.0)
        min_notional = Config.SYMBOL_MIN_NOTIONALS.get(sym, 0.0)
        price_placeholder = 1.0  # Caller should override with live price for precision
        min_order_notional = max(min_notional, min_qty * price_placeholder)
        if leverage <= 0:
            leverage = max(self.max_leverage, 1.0)
        return min_order_notional / leverage

    def _parse_compounding_ladder(self, spec: str) -> List[Dict[str, Optional[float]]]:
        ladder: List[Dict[str, Optional[float]]] = []
        if not spec:
            return ladder
        for segment in spec.split(';'):
            segment = segment.strip()
            if not segment:
                continue
            parts = [p.strip() for p in segment.split(':')]
            if len(parts) < 4:
                logger.warning("Invalid compounding ladder segment skipped: %s", segment)
                continue
            try:
                balance = float(parts[0])
            except (TypeError, ValueError):
                logger.warning("Invalid balance in compounding ladder segment: %s", segment)
                continue
            mode = parts[1].lower()
            leverage_part = parts[2].lower()
            margin_part = parts[3].lower()
            leverage_value: Optional[float] = None
            margin_fraction: Optional[float] = None
            if leverage_part not in {"", "auto", "none"}:
                try:
                    leverage_value = float(leverage_part)
                except (TypeError, ValueError):
                    logger.warning("Invalid leverage value in compounding ladder: %s", segment)
            if margin_part not in {"", "auto", "none"}:
                try:
                    margin_fraction = float(margin_part)
                except (TypeError, ValueError):
                    logger.warning("Invalid margin fraction in compounding ladder: %s", segment)
            ladder.append({
                'balance': balance,
                'mode': mode,
                'leverage': leverage_value,
                'margin_fraction': margin_fraction
            })
        ladder.sort(key=lambda entry: entry['balance'])
        return ladder

    def _resolve_compounding_band(self, balance: float) -> Optional[Dict[str, Optional[float]]]:
        if not self.compounding_ladder:
            return None
        band: Optional[Dict[str, Optional[float]]] = None
        for entry in self.compounding_ladder:
            if balance >= entry['balance']:
                band = entry
            else:
                break
        return band

    def _record_fee_outflow(self, notional_value: float, fee_pct: Optional[float]) -> None:
        if notional_value is None or notional_value <= 0:
            return
        fee_pct = fee_pct if fee_pct is not None else getattr(Config, "COMMISSION_RATE", 0.001)
        try:
            fee_value = max(float(notional_value) * float(fee_pct), 0.0)
        except (TypeError, ValueError):
            fee_value = 0.0
        if fee_value <= 0:
            return
        self.fee_recovery_balance += fee_value
        self.total_fees_paid += fee_value

    def _reconcile_fee_recovery(self, pnl: float) -> None:
        self.total_realized_pnl += pnl
        if pnl > 0:
            self.fee_recovery_balance = max(self.fee_recovery_balance - pnl, 0.0)
        elif pnl < 0:
            self.fee_recovery_balance += abs(pnl)

    def _evaluate_var_guard(self) -> None:
        if not self.var_pnl_window:
            self.var_guard_active = False
            return
        balance = max(self.current_portfolio_value, 1.0)
        cumulative_loss = sum(p for p in self.var_pnl_window if p < 0)
        loss_ratio = abs(cumulative_loss) / balance if balance > 0 else 0.0
        if loss_ratio >= self.weekly_var_cap:
            if not self.var_guard_active:
                logger.warning("VAR guard activated: weekly loss ratio %.2f%% >= %.2f%%", loss_ratio * 100, self.weekly_var_cap * 100)
            self.var_guard_active = True
        elif self.var_guard_active and loss_ratio < self.weekly_var_cap * 0.5:
            logger.info("VAR guard relaxed: loss ratio %.2f%% below half cap", loss_ratio * 100)
            self.var_guard_active = False

    def get_fee_recovery_pressure(self, account_balance: float) -> float:
        denom = max(account_balance, 1.0)
        pressure = self.fee_recovery_balance / denom
        return float(max(0.0, min(pressure, 2.0)))

    def get_ev_percentile(self, symbol: str, percentile: float = 0.5) -> float:
        stats = self._get_symbol_stats(symbol)
        if not stats['ev_history']:
            return 0.0
        percentile = float(np.clip(percentile, 0.0, 1.0))
        arr = np.array(stats['ev_history'])
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, percentile * 100.0))

    def is_var_guard_active(self) -> bool:
        return bool(self.var_guard_active)

    def is_symbol_tradeable(self, symbol: str, account_balance: float, current_price: float, leverage: float) -> bool:
        """Determine if symbol can meet exchange minimums with current balance."""
        sym = symbol.upper()
        blocked_micro = getattr(Config, "MICRO_TIER_BLOCKED_SYMBOLS", set())
        emergency_mode = bool(getattr(Config, "EMERGENCY_CALCULUS_MODE", False))
        if account_balance < 25 and sym in blocked_micro and not emergency_mode:
            logger.info(f"🚫 {symbol} blocked for micro tier balance ${account_balance:.2f}")
            return False
        min_qty = Config.SYMBOL_MIN_ORDER_QTY.get(sym, 0.0)
        min_notional = Config.SYMBOL_MIN_NOTIONALS.get(sym, 0.0)
        min_notional = max(min_notional, min_qty * current_price)
        if min_notional <= 0:
            return True
        if leverage <= 0:
            leverage = max(self.max_leverage, 1.0)
        required_margin = min_notional / leverage
        if account_balance <= 0:
            return False
        allowable_pct = 0.6 if account_balance < 20 else (0.55 if account_balance < 50 else 0.5)
        return required_margin <= account_balance * allowable_pct

    def _init_sharpe_tracker(self):
        """Initialize Sharpe tracker with Python fallback."""
        try:
            # Try C++ version first
            from cpp_bridge_working import mathcore
            self.sharpe_tracker = mathcore.SharpeTracker(window_size=100, risk_free_rate=0.04)
            self.leverage_bootstrap = mathcore.LeverageBootstrap()
            logger.info("✅ Using C++ Sharpe tracker (high performance)")
        except Exception as e:
            # Python fallback
            logger.info(f"Using Python Sharpe tracker fallback: {e}")
            self.sharpe_tracker = self._PythonSharpeTracker()
            self.leverage_bootstrap = self._PythonLeverageBootstrap()
    
    class _PythonSharpeTracker:
        """Python fallback for Sharpe tracker."""
        def __init__(self, window_size=100):
            self.returns = []
            self.window_size = window_size
        
        def add_return(self, trade_return):
            self.returns.append(trade_return)
            if len(self.returns) > self.window_size:
                self.returns.pop(0)
        
        def calculate_sharpe(self):
            if len(self.returns) < 2:
                return 0.0
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            if std_return < 1e-10:
                return 0.0
            # Annualized (assume 365 trading periods)
            return (mean_return - 0.04/365) / std_return * np.sqrt(365)
        
        def get_recommended_leverage(self, max_leverage=10.0):
            if len(self.returns) < 20:
                return 1.0
            sharpe = self.calculate_sharpe()
            if sharpe > 0.5:
                leverage = 1.0 + (sharpe / 2.0)
                return min(leverage, max_leverage)
            return 1.0
        
        def has_sufficient_data(self):
            return len(self.returns) >= 20
        
        def get_trade_count(self):
            return len(self.returns)
    
    class _PythonLeverageBootstrap:
        """Python fallback for leverage bootstrap."""
        def get_bootstrap_leverage(self, trade_count, account_balance=0):
            # MICRO-TIER PROTECTION: cap leverage at 8x for balances below $20
            if 0 < account_balance < 20:
                return 8.0
            else:
                # Normal bootstrap for larger accounts
                if trade_count <= 20:
                    return 1.0
                elif trade_count <= 50:
                    return 1.5
                elif trade_count <= 100:
                    return 2.0
            return 0.0  # Use dynamic
        
        def is_bootstrap_complete(self, trade_count):
            return trade_count > 100
    
    def get_optimal_leverage(self, account_balance: float) -> float:
        """
        Calculate optimal leverage with Sharpe-based intelligence and bootstrap mode.
        
        SMALL ACCOUNTS (<$20):
        Phase 1 (Trades 1-20): 5.0x - Enable $5+ notional
        Phase 2 (Trades 21-50): 8.0x - Aggressive growth
        Phase 3 (Trades 51-100): 10.0x - Maximum bootstrap
        Phase 4 (100+): Dynamic Sharpe-based
        
        LARGER ACCOUNTS ($20+):
        Phase 1 (Trades 1-20): 1.0x - Establish baseline
        Phase 2 (Trades 21-50): 1.5x - Gradual increase
        Phase 3 (Trades 51-100): 2.0x - Moderate leverage
        Phase 4 (100+): Dynamic Sharpe-based
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Optimal leverage for current balance
        """
        if self.force_leverage_enabled:
            return min(self.force_leverage_value, self.max_leverage)

        total_trades = len(self.trade_history)
        
        # BOOTSTRAP MODE (Trades 1-100)
        if not self.leverage_bootstrap.is_bootstrap_complete(total_trades):
            bootstrap_lev = self.leverage_bootstrap.get_bootstrap_leverage(total_trades, account_balance)
            logger.debug(f"Bootstrap mode: Trade #{total_trades}, Balance: ${account_balance:.2f}, Leverage: {bootstrap_lev}x")
            return bootstrap_lev
        
        # SHARPE-BASED DYNAMIC LEVERAGE (100+ trades)
        if self.use_sharpe_leverage and self.sharpe_tracker.has_sufficient_data():
            sharpe_lev = self.sharpe_tracker.get_recommended_leverage(self.max_leverage)
            logger.debug(f"Sharpe-based leverage: {sharpe_lev:.2f}x (Sharpe: {self.sharpe_tracker.calculate_sharpe():.2f})")
            return sharpe_lev
        
        # FALLBACK: Tiered leverage based on account size
        if account_balance < 20:
            return 8.0   # Micro-tier cap for diversification room
        elif account_balance < 50:
            return 7.0   # Moderate aggression
        elif account_balance < 100:
            return 6.0   # Gradual reduction
        elif account_balance < 200:
            return 6.0   # Hold steady before further scaling
        elif account_balance < 500:
            return 5.0   # Consolidation phase
        else:
            return 4.0   # Capital preservation mode
    
    def get_kelly_position_fraction(self, confidence: float, win_rate: float = 0.75) -> float:
        """Calculate capital fraction using rolling EV audit when available."""
        if self.expectancy_metrics and self.expectancy_metrics['variance'] > 1e-9:
            kelly_base = self.expectancy_metrics['kelly_base']
            # Guard rails for tiny or explosive Kelly values
            kelly_base = max(kelly_base, self.min_kelly_fraction)
            kelly_base = min(kelly_base, self.max_kelly_fraction)

            if confidence >= 0.85:
                confidence_scale = 0.9
            elif confidence >= 0.75:
                confidence_scale = 0.7
            else:
                confidence_scale = 0.5

            kelly_fraction = kelly_base * confidence_scale
            return min(max(kelly_fraction, self.min_kelly_fraction), self.max_kelly_fraction)

        # Fallback to heuristic Kelly if insufficient data
        b = 1.5  # Minimum risk:reward ratio assumption
        p = win_rate
        q = 1 - p
        kelly_fraction = (p * b - q) / b

        if confidence >= 0.85:
            return min(max(kelly_fraction * 0.60, self.min_kelly_fraction), self.max_kelly_fraction)
        elif confidence >= 0.75:
            return min(max(kelly_fraction * 0.50, self.min_kelly_fraction), self.max_kelly_fraction)
        else:
            return min(max(kelly_fraction * 0.40, self.min_kelly_fraction), self.max_kelly_fraction)

    def calculate_position_size(self,
                              symbol: str,
                              signal_strength: float,
                              confidence: float,
                              current_price: float,
                              account_balance: float,
                              volatility: float = None,
                              instrument_specs: Optional[Dict] = None,
                              scout_scale: float = 1.0,
                              leverage_hint: Optional[float] = None,
                              governor_mode: Optional[str] = None,
                              net_ev_pct: Optional[float] = None,
                              min_size_force_ev_pct: Optional[float] = None,
                              net_ev_zone: Optional[str] = None,
                              drift_scale_factor: float = 1.0) -> PositionSize:
        """
        Calculate optimal position size based on calculus signal strength and portfolio risk.

        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (SNR value)
            confidence: Signal confidence (0-1)
            current_price: Current market price
            account_balance: Available account balance
            volatility: Current volatility (optional)
            instrument_specs: Optional exchange requirements (min qty/notional)
            scout_scale: Fractional multiplier applied when entering scout mode
            leverage_hint: External leverage hint (from governor)
            governor_mode: Optional string descriptor for logging

        Returns:
            PositionSize calculation result
        """
        try:
            scout_scale = float(max(0.1, min(scout_scale or 1.0, 1.0)))
            compounding_band = self._resolve_compounding_band(account_balance)
            sizing_label = "FORCED" if self.force_leverage_enabled else "AGGRESSIVE"

            # AGGRESSIVE COMPOUNDING MODE
            # Use Kelly Criterion for optimal position sizing
            if self.force_leverage_enabled:
                optimal_leverage = min(self.force_leverage_value, self.max_leverage)
                kelly_fraction = self.force_margin_fraction
                if compounding_band:
                    band_mode = compounding_band.get('mode')
                    band_leverage = compounding_band.get('leverage')
                    band_margin = compounding_band.get('margin_fraction')

                    # Clamp ladder margin fractions to a sane range (0–70%)
                    if band_margin is not None:
                        try:
                            band_margin = max(0.0, min(float(band_margin), 0.70))
                        except (TypeError, ValueError):
                            band_margin = None

                    if band_mode == 'force':
                        if band_leverage is not None:
                            optimal_leverage = min(band_leverage, self.max_leverage)
                        if band_margin is not None:
                            kelly_fraction = band_margin
                        sizing_label = "FORCED/LADDER"
                    elif band_mode == 'auto':
                        if band_leverage is not None:
                            optimal_leverage = min(band_leverage, self.max_leverage)
                        else:
                            optimal_leverage = min(self.get_optimal_leverage(account_balance), self.max_leverage)
                        if band_margin is not None:
                            kelly_fraction = band_margin
                        else:
                            kelly_fraction = self.get_kelly_position_fraction(confidence)
                        sizing_label = "FORCED→AUTO"
            else:
                # Get optimal leverage for current balance
                optimal_leverage = self.get_optimal_leverage(account_balance)

                # Get Kelly position fraction (40-60% of capital based on confidence)
                kelly_fraction = self.get_kelly_position_fraction(confidence)

                # Apply consecutive loss protection
                if self.consecutive_losses >= 3:
                    kelly_fraction *= 0.5  # Cut position size by 50% after 3 losses
                    optimal_leverage *= 0.7  # Reduce leverage by 30%
                    logger.warning(f"⚠️  {self.consecutive_losses} consecutive losses - reducing position size & leverage")
                if compounding_band:
                    if compounding_band['mode'] == 'force':
                        if compounding_band.get('leverage') is not None:
                            optimal_leverage = min(compounding_band['leverage'], self.max_leverage)
                        if compounding_band.get('margin_fraction') is not None:
                            kelly_fraction = compounding_band['margin_fraction']
                        sizing_label = "AGGRESSIVE/LADDER"
                    elif compounding_band['mode'] == 'auto':
                        if compounding_band.get('leverage') is not None:
                            optimal_leverage = min(compounding_band['leverage'], self.max_leverage)
                        if compounding_band.get('margin_fraction') is not None:
                            kelly_fraction = compounding_band['margin_fraction']
                        sizing_label = "AGGRESSIVE/LADDER"

            if leverage_hint is not None:
                try:
                    optimal_leverage = min(max(float(leverage_hint), 1.0), self.max_leverage)
                except (TypeError, ValueError):
                    pass

            kelly_fraction = max(self.min_kelly_fraction, min(kelly_fraction, self.max_kelly_fraction))
            kelly_fraction *= scout_scale
            kelly_fraction = max(self.min_kelly_fraction * 0.5, min(kelly_fraction, self.max_kelly_fraction))

            # EV-AWARE SCALING: reduce/increase margin fraction based on expected value
            ev_scale = 1.0
            ev_floor = float(getattr(Config, "MIN_EMERGENCY_EV_PCT", 0.0003))
            ev_ref = float(getattr(Config, "EV_POSITION_REF_PCT", 0.0015))
            ev_min_scale = float(getattr(Config, "EV_POSITION_SCALE_MIN", 0.35))
            ev_max_scale = float(getattr(Config, "EV_POSITION_SCALE_MAX", 1.15))
            if net_ev_pct is not None and np.isfinite(net_ev_pct):
                # Linear scale clamped between min/max
                scaled = 0.0
                if ev_ref > 1e-9:
                    scaled = net_ev_pct / ev_ref
                ev_scale = float(np.clip(scaled, ev_min_scale, ev_max_scale))
                if net_ev_pct < 0.0:
                    ev_scale = max(ev_scale, ev_min_scale)
            micro_zone = net_ev_zone or 'green'
            if account_balance < 25 and net_ev_zone:
                if net_ev_zone == 'yellow':
                    ev_scale *= float(getattr(Config, 'MICRO_EV_YELLOW_SIZE_SCALE', 0.55))
                elif net_ev_zone == 'green':
                    ev_scale *= float(getattr(Config, 'MICRO_EV_GREEN_SIZE_SCALE', 1.0))
            if account_balance < 25 and self.force_leverage_enabled:
                # Micro emergency under forced leverage still respects EV scaling
                kelly_fraction *= ev_scale
            elif account_balance < 25:
                kelly_fraction *= ev_scale
            else:
                kelly_fraction *= max(ev_scale, 0.6)
            kelly_fraction = float(np.clip(kelly_fraction, self.min_kelly_fraction * 0.5, self.max_kelly_fraction))

            # Exchange feasibility check before sizing
            if instrument_specs and current_price > 0:
                try:
                    min_qty = float(instrument_specs.get('min_qty', 0.0) or 0.0)
                    min_notional_spec = float(instrument_specs.get('min_notional', 0.0) or 0.0)
                except (TypeError, ValueError):
                    min_qty = 0.0
                    min_notional_spec = 0.0

                min_exchange_notional = 0.0
                if min_qty > 0:
                    min_exchange_notional = max(min_exchange_notional, min_qty * current_price)
                if min_notional_spec > 0:
                    min_exchange_notional = max(min_exchange_notional, min_notional_spec)

                if min_exchange_notional > 0:
                    min_margin_required = min_exchange_notional / max(optimal_leverage, 1.0)
                    if account_balance < 20:
                        allowable_pct = 0.6
                    elif account_balance < 50:
                        allowable_pct = 0.55
                    else:
                        allowable_pct = 0.5
                    allowed_margin = account_balance * allowable_pct
                    if min_margin_required > allowed_margin:
                        logger.info(
                            f"🚧 {symbol} exchange minimums require ${min_margin_required:.2f} margin; "
                            f"allowed ${allowed_margin:.2f} with balance ${account_balance:.2f}"
                        )
                        return PositionSize(0, 0, 0, optimal_leverage, 0, 0, confidence)
            
            # Calculate position notional (Kelly fraction of capital × leverage)
            position_notional = account_balance * kelly_fraction * optimal_leverage

            # Apply per-symbol notional caps (dynamic by balance tier)
            symbol_cap = None if self.force_leverage_enabled else self._resolve_symbol_notional_cap(symbol, account_balance)
            if symbol_cap is not None and symbol_cap > 0 and position_notional > symbol_cap:
                logger.info(
                    f"📉 Notional capped for {symbol}: {position_notional:.2f} → {symbol_cap:.2f}"
                )
                position_notional = symbol_cap
                denom = max(account_balance * max(optimal_leverage, 1e-9), 1e-9)
                kelly_fraction = position_notional / denom

            # Enforce exchange minimum order value with symbol-aware floor
            sym = symbol.upper()
            global_min_notional = max(getattr(Config, "MIN_ORDER_NOTIONAL", 5.05), 5.0)
            symbol_min_notional = float(getattr(Config, "SYMBOL_MIN_NOTIONALS", {}).get(sym, global_min_notional))
            min_notional = max(global_min_notional, symbol_min_notional)

            # MICRO BOOST: for tiny balances, target a bit above exchange min (e.g. 1.2×)
            if account_balance < 25:
                target_notional = symbol_min_notional * 1.2
                min_notional = max(min_notional, target_notional)

            if position_notional < min_notional:
                logger.info(
                    f"📈 Raising notional for {symbol} to meet minimum: {position_notional:.2f} → {min_notional:.2f}"
                )
                position_notional = min_notional
                denom = max(account_balance * max(optimal_leverage, 1e-9), 1e-9)
                kelly_fraction = position_notional / denom
            
            # Volatility adjustment (higher volatility = smaller position)
            if (not self.force_leverage_enabled) and volatility is not None and volatility > 0.03:  # >3% volatility
                volatility_adjustment = min(0.03 / volatility, 1.0)
                position_notional *= volatility_adjustment
            
            # RENAISSANCE: Apply drift-based position scaling
            # When drift aligns with signal, amplify position. When misaligned, reduce.
            if drift_scale_factor != 1.0:
                position_notional *= drift_scale_factor
                logger.info(f"🎯 Drift scaling: {drift_scale_factor:.2f}x applied to {symbol} position")
            
            # Calculate quantity
            quantity = position_notional / current_price
            
            # Calculate margin requirement
            margin_required = position_notional / optimal_leverage

            min_force_ev = min_size_force_ev_pct if min_size_force_ev_pct is not None else getattr(Config, 'MICRO_MIN_SIZE_FORCE_EV_PCT', 0.001)
            if account_balance < 25 and net_ev_pct is not None and instrument_specs:
                try:
                    min_notional_spec = float(instrument_specs.get('min_notional', getattr(Config, 'MIN_ORDER_NOTIONAL', 5.0)))
                except (TypeError, ValueError):
                    min_notional_spec = getattr(Config, 'MIN_ORDER_NOTIONAL', 5.0)
                if position_notional <= min_notional_spec * 1.01 and net_ev_pct < min_force_ev:
                    logger.info(
                        "Skipping forced min ticket for %s: EV %.4f%% < %.4f%%",
                        symbol,
                        net_ev_pct * 100.0,
                        min_force_ev * 100.0
                    )
                    return PositionSize(0, 0, 0, optimal_leverage, 0, 0, confidence)
            
            # CRITICAL SAFETY: Cap margin per trade as a fraction of balance
            # Micro/early accounts are allowed to be aggressive, but not fully all-in.
            if account_balance < 25:
                max_margin_pct = 0.55  # Up to 55% of balance per trade in micro band
            elif account_balance < 100:
                max_margin_pct = 0.60  # Up to 60% for early growth band
            else:
                max_margin_pct = 0.60  # 60% cap for larger balances (compounded but controlled)
            
            max_allowed_margin = account_balance * max_margin_pct
            
            # Safety check: ensure margin + buffer doesn't exceed limit
            margin_buffer = 1.10  # 10% buffer for fees, slippage, and market moves
            if margin_required > max_allowed_margin:
                # Scale down to fit within safe margin limits
                scale_factor = max_allowed_margin / margin_required
                quantity *= scale_factor
                position_notional *= scale_factor
                margin_required *= scale_factor
                logger.info(f"Position scaled down by {scale_factor:.2f}x to {margin_required/account_balance:.1%} of balance")
            
            # Calculate risk amount (for tracking, not limiting)
            risk_amount = margin_required * 0.02  # 2% of margin at risk
            
            # Calculate risk percent of total capital
            risk_percent = margin_required / account_balance if account_balance > 0 else 0
            
            ev_info = ""
            if self.expectancy_metrics:
                ev_info = (f", EV={self.expectancy_metrics['expectancy']:.3f}, "
                           f"p_win={self.expectancy_metrics['p_win']:.2f}, "
                           f"Var={self.expectancy_metrics['variance']:.4f}")

            if scout_scale < 0.999:
                sizing_label += "*SCOUT"
            if governor_mode:
                sizing_label += f"[{governor_mode}]"
            logger.info(
                "💰 %s SIZING: Balance=$%.2f, Kelly=%.1f%%%s, Leverage=%.1fx, Scout=%.2f, Notional=$%.2f, Margin=$%.2f",
                sizing_label,
                account_balance,
                kelly_fraction * 100.0,
                ev_info,
                optimal_leverage,
                scout_scale,
                position_notional,
                margin_required
            )

            return PositionSize(
                quantity=quantity,
                notional_value=position_notional,
                risk_amount=risk_amount,
                leverage_used=optimal_leverage,
                margin_required=margin_required,
                risk_percent=risk_percent,
                confidence_score=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return PositionSize(0, 0, 0, 1, 0, 0, 0)
    
    def check_and_announce_milestone(self, current_balance: float) -> bool:
        """Check if we've reached a new milestone and announce it."""
        for milestone in self.milestones:
            if current_balance >= milestone and milestone not in self.reached_milestones:
                self.reached_milestones.add(milestone)
                
                if self.session_start_balance > 0:
                    growth_pct = ((current_balance - self.session_start_balance) / self.session_start_balance) * 100
                    elapsed_hours = (time.time() - self.session_start_time) / 3600
                    
                    if current_balance > self.session_start_balance and elapsed_hours > 0.1:
                        growth_rate = (current_balance / self.session_start_balance) ** (1 / elapsed_hours)
                        if growth_rate > 1 and current_balance < 1000:
                            hours_to_1000 = np.log(1000 / current_balance) / np.log(growth_rate)
                            eta_str = f"{hours_to_1000:.1f} hours" if hours_to_1000 < 72 else f"{hours_to_1000/24:.1f} days"
                        else:
                            eta_str = "ACHIEVED!" if current_balance >= 1000 else "calculating..."
                    else:
                        eta_str = "calculating..."
                    
                    print("\n" + "🎉" * 35)
                    print(f"🏆 MILESTONE REACHED: ${milestone}!")
                    print("🎉" * 35)
                    print(f"💰 Current Balance: ${current_balance:.2f}")
                    print(f"📈 Session Growth: +${current_balance - self.session_start_balance:.2f} ({growth_pct:+.1f}%)")
                    print(f"⏱️  Time Elapsed: {elapsed_hours:.1f} hours")
                    next_m = [m for m in self.milestones if m > milestone]
                    print(f"🎯 Next Milestone: ${next_m[0]}" if next_m else "🎯 Target ACHIEVED!")
                    print(f"⏰ ETA to $1,000: {eta_str}")
                    print("🎉" * 35 + "\n")
                
                return True
        return False
    
    def calculate_growth_metrics(self, current_balance: float) -> Dict:
        """Calculate growth metrics and ETA to $1,000."""
        if self.session_start_balance == 0:
            self.session_start_balance = current_balance
            self.session_start_time = time.time()
        
        elapsed_hours = (time.time() - self.session_start_time) / 3600
        
        if elapsed_hours < 0.01:
            return {'growth_pct': 0, 'growth_rate_hourly': 0, 'eta_to_1000_hours': 999, 'eta_to_1000_str': 'calculating...'}
        
        growth_pct = ((current_balance - self.session_start_balance) / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
        
        if current_balance > self.session_start_balance:
            growth_rate = (current_balance / self.session_start_balance) ** (1 / elapsed_hours)
            hourly_return = (growth_rate - 1) * 100
            
            if growth_rate > 1 and current_balance < 1000:
                hours_to_1000 = np.log(1000 / current_balance) / np.log(growth_rate)
                eta_str = f"{hours_to_1000:.1f} hours" if hours_to_1000 < 72 else f"{hours_to_1000/24:.1f} days"
            else:
                hours_to_1000 = 999
                eta_str = "at current rate..." if growth_rate <= 1 else "ACHIEVED!"
        else:
            hourly_return = 0
            hours_to_1000 = 999
            eta_str = "need positive growth..."
        
        return {'growth_pct': growth_pct, 'growth_rate_hourly': hourly_return, 'eta_to_1000_hours': hours_to_1000, 'eta_to_1000_str': eta_str}
    
    def check_drawdown_protection(self, current_balance: float) -> Dict:
        """Check if drawdown protection should activate."""
        # Initialize session high if not set
        if self.session_high == 0:
            self.session_high = current_balance
        
        # Update session high
        if current_balance > self.session_high:
            self.session_high = current_balance
        
        # Calculate drawdown from session high
        if self.session_high > 0:
            drawdown_pct = ((self.session_high - current_balance) / self.session_high) * 100
        else:
            drawdown_pct = 0
        
        # More lenient for aggressive compounding - allow 30% drawdown before stopping
        should_stop = drawdown_pct >= 30
        should_reduce = drawdown_pct >= 15 and drawdown_pct < 30
        reduction_factor = 0 if should_stop else (0.5 if should_reduce else 1.0)
        
        return {'should_stop': should_stop, 'should_reduce': should_reduce, 'reduction_factor': reduction_factor, 'drawdown_pct': drawdown_pct, 'session_high': self.session_high}
    
    def record_trade_result(self, won: bool):
        """Record trade result for consecutive loss tracking."""
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                logger.warning(f"⚠️  {self.consecutive_losses} consecutive losses - risk reduction active")

    def _update_expectancy_metrics(self):
        """Recompute rolling expectancy statistics for EV-driven sizing."""
        if not self.trade_returns:
            self.expectancy_metrics = None
            return

        returns_window = np.array(self.trade_returns[-self.ev_window:])
        sample_size = len(returns_window)
        if sample_size < 5:
            self.expectancy_metrics = None
            return

        wins = returns_window[returns_window > 0]
        losses = returns_window[returns_window < 0]
        p_win = len(wins) / sample_size if sample_size > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        variance = returns_window.var(ddof=1) if sample_size > 1 else 0.0
        expectancy = p_win * avg_win - (1 - p_win) * avg_loss
        kelly_base = expectancy / variance if variance > 1e-9 else 0.0

        self.expectancy_metrics = {
            'sample_size': sample_size,
            'p_win': p_win,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'variance': variance,
            'expectancy': expectancy,
            'kelly_base': max(kelly_base, 0.0)
        }

        # Periodic logging to trace expectancy drift without spamming
        if sample_size % 10 == 0:
            logger.info(
                f"📈 EV audit: n={sample_size}, p_win={p_win:.2f}, avg_win={avg_win:.3f}, "
                f"avg_loss={avg_loss:.3f}, EV={expectancy:.3f}, Var={variance:.4f}, "
                f"Kelly*={self.expectancy_metrics['kelly_base']:.3f}"
            )
    
    def get_status_display(self, current_balance: float) -> str:
        """Get formatted status display for aggressive compounding mode."""
        metrics = self.calculate_growth_metrics(current_balance)
        protection = self.check_drawdown_protection(current_balance)
        
        status = "\n" + "="*70 + "\n💰 AGGRESSIVE COMPOUNDING STATUS\n" + "="*70 + "\n"
        status += f"💵 Current Balance: ${current_balance:.2f}\n"
        status += f"📊 Session Start: ${self.session_start_balance:.2f}\n"
        status += f"📈 Growth: +${current_balance - self.session_start_balance:.2f} ({metrics['growth_pct']:+.1f}%)\n"
        status += f"⚡ Hourly Rate: {metrics['growth_rate_hourly']:+.1f}%/hr\n"
        status += f"🎯 ETA to $1,000: {metrics['eta_to_1000_str']}\n"
        
        if protection['drawdown_pct'] > 0:
            status += f"⚠️  Drawdown: -{protection['drawdown_pct']:.1f}% (High: ${protection['session_high']:.2f})\n"
        
        if self.consecutive_losses > 0:
            status += f"🔴 Consecutive Losses: {self.consecutive_losses}\n"
        
        next_milestone = next((m for m in self.milestones if current_balance < m), None)
        if next_milestone:
            progress_to_next = (current_balance / next_milestone) * 100
            status += f"🏁 Next Milestone: ${next_milestone} ({progress_to_next:.1f}% there)\n"
        
        status += "="*70 + "\n"
        return status

    def calculate_dynamic_tp_sl(self,
                               signal_type: SignalType,
                               current_price: float,
                               velocity: float,
                               acceleration: float,
                               volatility: float,
                               forecast_price: float = None,
                               atr: float = None,
                               account_balance: float = 0.0,
                               sigma: Optional[float] = None,
                               half_life_seconds: Optional[float] = None,
                               f_score: Optional[float] = None,
                               forecast_deltas: Optional[Dict[str, float]] = None,
                               jerk: Optional[float] = None,
                               jounce: Optional[float] = None,
                               funding_bias: Optional[float] = None,
                               ou_params: Optional[Dict[str, float]] = None,
                               leverage_used: Optional[float] = None,
                               fee_cost_pct: Optional[float] = None) -> TradingLevels:
        """
        Calculate dynamic TP/SL levels using volatility-proportional bands.

        Primary objective: enforce TP = 1.5σ and SL = 0.75σ while
        deriving an OU-informed maximum holding window (~2 · t½).

        Args:
            signal_type: Type of trading signal
            current_price: Current market price
            velocity: Current price velocity
            acceleration: Current price acceleration
            volatility: Current market volatility
            forecast_price: Optional calculus forecast (unused for TP, kept for future use)
            atr: Average True Range (optional, unused)
            account_balance: Current account balance (unused, reserved for compatibility)
            sigma: Explicit volatility estimate (fallback to `volatility` when None)
            half_life_seconds: OU half-life in seconds (for expiry window)

        Returns:
            TradingLevels with volatility-based TP/SL and expiry guidance
        """
        try:
            # Use canonical position_side determination (single source of truth)
            position_side = determine_position_side(signal_type, velocity)
            logger.debug(f"Position side: {position_side} (signal={signal_type.name}, v={velocity:.6f})")
            sigma_pct = sigma if sigma is not None else volatility
            sigma_pct = float(max(sigma_pct, 5e-4))  # Minimum 0.05%

            # MICRO EMERGENCY: detect tiny-balance emergency calculus mode for TP/SL tuning
            emergency_mode = bool(getattr(Config, "EMERGENCY_CALCULUS_MODE", False))
            micro_emergency = emergency_mode and (0.0 < float(account_balance) < 25.0)

            fee_pct = float(getattr(Config, "COMMISSION_RATE", 0.001))
            fee_buffer = max(fee_pct * 4.0, 0.0)
            volatility_floor_pct = max(1.5 * sigma_pct, 0.006, fee_buffer)

            curve_tp_pct = 0.0
            momentum_score = 0.0
            scale = 1.0
            tp_pct_candidate = 0.0
            if f_score is not None and np.isfinite(f_score):
                curve_tp_pct = max(float(f_score), 0.0)
            elif forecast_price is not None and forecast_price > 0 and current_price > 0:
                if position_side == "long":
                    curve_delta = max(forecast_price - current_price, 0.0)
                else:
                    curve_delta = max(current_price - forecast_price, 0.0)
                curve_tp_pct = curve_delta / current_price if current_price > 0 else 0.0

            def _kappa_scale(score: float) -> float:
                if score < 0.009:
                    return 1.0
                if score < 0.012:
                    return 1.2
                if score < 0.018:
                    return 1.45
                if score < 0.024:
                    return 1.8
                return 2.1

            if curve_tp_pct > 0:
                scale = _kappa_scale(curve_tp_pct)
            if forecast_deltas and current_price > 0:
                try:
                    nearest_horizon = min(forecast_deltas.keys())
                    nearest_delta = forecast_deltas.get(nearest_horizon, 0.0)
                    momentum_score = float(nearest_delta) / current_price
                except (ValueError, TypeError):
                    momentum_score = 0.0

            direction = 1 if position_side == "long" else -1
            directional_momentum = momentum_score * direction
            if directional_momentum > 0:
                scale *= (1.0 + min(directional_momentum * 3.0, 0.12))
            elif directional_momentum < 0:
                scale *= (1.0 + max(directional_momentum * 2.0, -0.15))

            if funding_bias is not None:
                try:
                    funding_adjust = float(np.clip(-funding_bias * 200.0, -0.1, 0.1))
                    scale *= (1.0 + funding_adjust)
                except (TypeError, ValueError):
                    pass

            scale = float(np.clip(scale, 0.8, 2.5))
            # CRITICAL FIX: NEUTRAL signals need special handling for positive EV
            # When velocity ≈ 0 (flat market), use mean-reversion optimized TP/SL
            is_neutral_flat = (
                signal_type == SignalType.NEUTRAL and 
                abs(velocity) < 0.0008  # Essentially zero velocity
            )
            
            if is_neutral_flat:
                # For mean reversion in flat markets: TP=0.5%, SL=0.1% gives 0.2% EV at 50% WR
                # This overcomes fees (0.04% entry + 0.04% exit = 0.08%)
                tp_pct = max(0.005, volatility_floor_pct * 2.0)  # Min 0.5% for NEUTRAL flat
                tp_pct_candidate = tp_pct
            else:
                if curve_tp_pct > 0:
                    tp_pct_candidate = curve_tp_pct * scale
                tp_pct = max(tp_pct_candidate, volatility_floor_pct)

            # MICRO EMERGENCY: slightly larger TP distance so strong micro-moves pay more
            if micro_emergency:
                tp_pct = max(tp_pct, volatility_floor_pct * 1.1)
            tp_offset = current_price * tp_pct

            # For NEUTRAL flat signals: use tighter SL to maximize R:R
            if is_neutral_flat:
                sl_floor_pct = max(sigma_pct * 0.35, 0.001)  # Min 0.1% for NEUTRAL
                sl_pct = max(tp_pct / 5.0, sl_floor_pct)  # TP/SL = 5:1 for mean reversion
            else:
                sl_floor_pct = max(sigma_pct * 0.35, 0.0035)
                sl_pct = max(tp_pct / 1.8, sl_floor_pct)
            if jerk is not None and np.isfinite(jerk):
                jerk_adjust = max(min(abs(jerk) * 2.0, 0.4), 0.0)
                sl_pct = max(sl_pct * (1.0 - jerk_adjust * 0.1), sl_floor_pct)
            sl_offset = current_price * sl_pct

            secondary_multiplier = float(getattr(Config, "TP_SECONDARY_MULTIPLIER", 1.8))
            # MICRO EMERGENCY: push TP2 further out so big winners pay more
            if micro_emergency:
                secondary_multiplier = max(secondary_multiplier, 1.9)
            secondary_tp_pct = tp_pct * secondary_multiplier
            secondary_take_profit = current_price + tp_offset * secondary_multiplier if position_side == "long" else current_price - tp_offset * secondary_multiplier

            if position_side == "long":
                take_profit = current_price + tp_offset
                stop_loss = current_price - sl_offset
            else:
                take_profit = current_price - tp_offset
                stop_loss = current_price + sl_offset  # SL uses same crypto-optimized offset

            micro_tp_floor = float(getattr(Config, "MICRO_MIN_TP_USDT", 0.0))
            if micro_emergency and micro_tp_floor > 0:
                notional = current_price * max(1e-9, tp_offset / tp_pct)
                target_pct_floor = micro_tp_floor / max(notional, 1e-9)
                if tp_pct < target_pct_floor:
                    tp_pct = target_pct_floor
                    tp_offset = current_price * tp_pct
                    if position_side == "long":
                        take_profit = current_price + tp_offset
                    else:
                        take_profit = current_price - tp_offset
                    secondary_tp_pct = tp_pct * secondary_multiplier
                    secondary_take_profit = current_price + tp_offset * secondary_multiplier if position_side == "long" else current_price - tp_offset * secondary_multiplier

            risk_reward_ratio = (tp_offset / sl_offset) if sl_offset > 0 else 0.0

            # Derivative trend metrics for horizon & optimizer
            trend_score = compute_derivative_trend_score(
                velocity, acceleration, jerk or 0.0, sigma_pct, Config
            )
            dynamic_hold = compute_derivative_horizon(half_life_seconds, trend_score, Config)
            if dynamic_hold is not None:
                max_hold_seconds = dynamic_hold
            elif half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                hold_mult = 2.5 if micro_emergency else 2.0
                max_hold_seconds = max(half_life_seconds * hold_mult, 60.0)
            else:
                max_hold_seconds = None

            leverage_for_optimizer = leverage_used if leverage_used is not None else float(getattr(Config, "FORCE_LEVERAGE_VALUE", 20.0))
            fee_floor_pct = self.get_fee_aware_tp_floor(sigma_pct, fee_buffer_multiplier=None)

            ou_inputs = ou_params or {}
            barrier_ready = (
                getattr(Config, "USE_BARRIER_TP_OPTIMIZER", False) and
                bool(ou_inputs)
            )
            if barrier_ready:
                horizon = dynamic_hold or max_hold_seconds or 180.0
                horizon = max(horizon, 60.0)
                forecast_inputs = ForecastInputs(
                    price=current_price,
                    velocity=velocity,
                    acceleration=acceleration,
                    jerk=jerk or 0.0,
                    snap=jounce or 0.0,
                    sigma_pct=sigma_pct,
                    half_life_seconds=half_life_seconds,
                    ou_mu=ou_inputs.get('mu'),
                    ou_theta=ou_inputs.get('theta'),
                    ou_sigma=ou_inputs.get('sigma')
                )
                mu_H, sigma_H = compute_unified_log_drift(forecast_inputs, Config, horizon)
                opt_tp_pct, opt_sl_pct, opt_meta = optimize_tp_sl(
                    mu_H,
                    sigma_H,
                    tp_pct,
                    sl_pct,
                    fee_floor_pct,
                    leverage_for_optimizer,
                    Config
                )
                tp_pct = opt_tp_pct
                sl_pct = opt_sl_pct
                tp_offset = current_price * tp_pct
                sl_offset = current_price * sl_pct
                if position_side == "long":
                    take_profit = current_price + tp_offset
                    stop_loss = current_price - sl_offset
                    secondary_take_profit = current_price + tp_offset * secondary_multiplier
                else:
                    take_profit = current_price - tp_offset
                    stop_loss = current_price + sl_offset
                    secondary_take_profit = current_price - tp_offset * secondary_multiplier
                risk_reward_ratio = (tp_offset / sl_offset) if sl_offset > 0 else risk_reward_ratio
            elif max_hold_seconds is None and half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                hold_mult = 2.5 if micro_emergency else 2.0
                max_hold_seconds = max(half_life_seconds * hold_mult, 60.0)

            # Confidence blends velocity/acceleration relative to volatility scale
            volatility_floor = max(sigma_pct, 1e-6)
            velocity_strength = abs(velocity)
            acceleration_strength = abs(acceleration)
            normalized_velocity = min(velocity_strength / volatility_floor, 2.0)
            normalized_acceleration = min(acceleration_strength / volatility_floor, 2.0)
            confidence_level = min(1.0, 0.5 * normalized_velocity + 0.5 * normalized_acceleration)

            trail_stop = None
            if signal_type == SignalType.TRAIL_STOP_UP:
                trail_stop = stop_loss
            elif signal_type == SignalType.HOLD_SHORT:
                trail_stop = stop_loss

            trail_mult = float(getattr(Config, "TP_TRAIL_BUFFER_MULTIPLIER", 0.5))
            # MICRO EMERGENCY: slightly looser trail so strong moves are not choked early
            if micro_emergency:
                trail_mult = max(trail_mult, 0.60)
            trailing_buffer_pct = trail_mult * tp_pct

            # TP FRACTIONS: allow micro-emergency to keep more size for TP2
            primary_fraction = float(getattr(Config, "TP_PRIMARY_FRACTION", 0.4))
            if micro_emergency:
                primary_fraction = float(getattr(Config, "MICRO_PRIMARY_FRACTION", 0.35))
            secondary_fraction = max(1.0 - primary_fraction, 0.0)

            return TradingLevels(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trail_stop=trail_stop,
                risk_reward_ratio=risk_reward_ratio,
                position_side=position_side,
                confidence_level=confidence_level,
                max_hold_seconds=max_hold_seconds,
                secondary_take_profit=secondary_take_profit,
                primary_tp_fraction=primary_fraction,
                secondary_tp_fraction=secondary_fraction,
                secondary_tp_probability=None,
                forecast_deltas=forecast_deltas,
                f_score=f_score,
                curvature_metrics={
                    'jerk': float(jerk) if jerk is not None else None,
                    'jounce': float(jounce) if jounce is not None else None,
                    'volatility_floor_pct': volatility_floor_pct,
                    'tp_pct_candidate': tp_pct_candidate,
                    'secondary_multiplier': secondary_multiplier,
                    'kappa_scale': scale,
                    'curve_tp_pct': curve_tp_pct,
                    'tp_pct_final': tp_pct,
                    'secondary_tp_pct': secondary_tp_pct,
                    'momentum_score': momentum_score,
                    'directional_momentum': directional_momentum,
                    'funding_bias': funding_bias
                },
                trailing_buffer_pct=trailing_buffer_pct
            )

        except Exception as e:
            logger.error(f"Error calculating TP/SL levels: {e}")
            return TradingLevels(current_price, current_price, current_price, None, 1.0, "long", 0.0, None)

    def get_fee_aware_tp_floor(self,
                               sigma_pct: float,
                               taker_fee_pct: Optional[float] = None,
                               funding_buffer_pct: float = 0.0,
                               fee_buffer_multiplier: Optional[float] = None,
                               symbol: Optional[str] = None) -> float:
        """Return the minimum TP percentage accounting for volatility, fees, and funding."""
        sigma_pct = float(max(sigma_pct, 5e-4))
        base_fee_pct = float(taker_fee_pct if taker_fee_pct is not None else getattr(Config, "COMMISSION_RATE", 0.001))
        config_multiplier = float(getattr(Config, "FEE_BUFFER_MULTIPLIER", 4.0))
        liquid_symbols = {'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'}
        maker_rebate = float(getattr(Config, "MAKER_REBATE_PCT", 0.0))
        effective_fee_pct = base_fee_pct
        if symbol and symbol.upper() in liquid_symbols and maker_rebate > 0:
            effective_fee_pct = max(base_fee_pct - maker_rebate, 0.0)

        if base_fee_pct <= 0.0006 and config_multiplier > 2.0:
            config_multiplier = 2.0

        applied_multiplier = fee_buffer_multiplier if fee_buffer_multiplier is not None else config_multiplier
        if symbol and symbol.upper() in liquid_symbols and applied_multiplier > 2.2:
            applied_multiplier = 2.2

        if fee_buffer_multiplier is None:
            applied_multiplier = min(applied_multiplier, config_multiplier)

        if base_fee_pct <= 0.0006 and fee_buffer_multiplier is not None and applied_multiplier > 2.2:
            applied_multiplier = 2.2

        crypto_baseline = 2.5
        if not self._warned_fee_multiplier and abs(config_multiplier - crypto_baseline) > 1e-6:
            logger.warning(
                "FEE_BUFFER_MULTIPLIER=%.3f deviates from crypto baseline %.3f",
                config_multiplier,
                crypto_baseline
            )
            self._warned_fee_multiplier = True

        fee_buffer_pct = max(effective_fee_pct * applied_multiplier, 0.0)
        funding_component = max(funding_buffer_pct, 0.0)
        sigma_component = 1.5 * sigma_pct
        static_floor = 0.006
        total_fee_floor = max(sigma_component, static_floor, fee_buffer_pct + funding_component)

        details = {
            'sigma_pct': sigma_pct,
            'sigma_component': sigma_component,
            'static_min_tp_pct': static_floor,
            'base_fee_pct': base_fee_pct,
            'effective_fee_pct': effective_fee_pct,
            'config_multiplier': config_multiplier,
            'applied_multiplier': applied_multiplier,
            'fee_buffer_pct': fee_buffer_pct,
            'funding_buffer_pct': funding_component,
            'total_fee_floor_pct': total_fee_floor
        }

        if symbol:
            self.fee_floor_debug[symbol.upper()] = details

        if getattr(Config, "EV_DEBUG_LOGGING", False):
            logger.info("Fee floor debug %s: %s", symbol or "<unknown>", details)

        return total_fee_floor

    def validate_trade_risk(self,
                           symbol: str,
                           position_size: PositionSize,
                           trading_levels: TradingLevels) -> Tuple[bool, str]:
        """
        Validate trade against risk management rules.

        Args:
            symbol: Trading symbol
            position_size: Calculated position size
            trading_levels: TP/SL levels

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check risk-reward ratio (allow slight floating point tolerance, relaxed for small accounts)
            min_rr = 1.3 if trading_levels.confidence_level > 0 else self.min_risk_reward  # Relaxed threshold
            if trading_levels.risk_reward_ratio < (min_rr - 0.01):
                return False, f"Risk-reward ratio {trading_levels.risk_reward_ratio:.2f} below minimum {min_rr}"

            # Check maximum position count
            if len(self.open_positions) >= self.max_positions:
                return False, f"Maximum positions ({self.max_positions}) already open"

            # Check position correlation
            if not self._check_correlation(symbol, trading_levels.position_side):
                return False, f"High correlation with existing positions"

            # Check leverage limits
            if position_size.leverage_used > self.max_leverage:
                return False, f"Leverage {position_size.leverage_used}x exceeds maximum {self.max_leverage}x"

            # Check portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(position_size)
            if portfolio_risk > self.max_portfolio_risk:
                return False, f"Portfolio risk {portfolio_risk:.1%} exceeds maximum {self.max_portfolio_risk:.1%}"

            # Check final TP probability threshold (unless calculus priority overrides)
            if not (self.calculus_priority_mode or self.force_leverage_enabled):
                min_final_tp = float(getattr(Config, "MIN_FINAL_TP_PROBABILITY", 0.52))
                if position_size.confidence_score < min_final_tp:
                    return False, (
                        f"Final TP probability {position_size.confidence_score:.2f} < "
                        f"required {min_final_tp:.2f}"
                    )

            return True, "Trade validation passed"

        except Exception as e:
            logger.error(f"Error validating trade risk: {e}")
            return False, f"Validation error: {str(e)}"

    def _check_correlation(self, symbol: str, position_side: str) -> bool:
        """
        Check correlation with existing positions.

        Args:
            symbol: New trading symbol
            position_side: Position side (long/short)

        Returns:
            True if correlation is acceptable
        """
        if not self.open_positions:
            return True

        # Simple correlation check (can be enhanced with actual correlation data)
        same_side_positions = sum(
            1 for pos in self.open_positions.values()
            if pos['side'] == position_side
        )

        # Limit positions on same side
        max_same_side = self.max_positions // 2
        return same_side_positions < max_same_side

    def _calculate_portfolio_risk(self, new_position: PositionSize) -> float:
        """
        Calculate total portfolio risk with new position.

        Args:
            new_position: New position size

        Returns:
            Total portfolio risk as percentage
        """
        total_risk = new_position.risk_percent

        # Add risk from existing positions (simplified)
        for pos_info in self.open_positions.values():
            total_risk += pos_info.get('risk_percent', 0.02)

        return total_risk

    def _resolve_symbol_notional_cap(self, symbol: str, account_balance: float) -> Optional[float]:
        sym = symbol.upper()
        override = self.symbol_notional_overrides.get(sym)
        if override is not None:
            return override

        base = self.symbol_base_notional.get(sym)
        if base is None or account_balance <= 0:
            return base

        factor = 1.0
        if self.notional_cap_tiers:
            for threshold, tier_factor in self.notional_cap_tiers:
                if account_balance < threshold:
                    factor = tier_factor
                    break

        return base * factor

    def _get_symbol_stats(self, symbol: str) -> Dict:
        """Fetch mutable trade stats bucket for a symbol."""
        key = symbol.upper()
        stats = self.symbol_trade_stats.get(key)
        if stats is None:
            stats = {
                'entries': 0,
                'completed': 0,
                'wins': 0,
                'losses': 0,
                'open': 0,
                'net_pnl': 0.0,
                'notional_executed': 0.0,
                'margin_committed': 0.0,
                'active_margin': 0.0,
                'holding_seconds_total': 0.0,
                'returns': deque(maxlen=200),
                'ev_history': deque(maxlen=200),
                'fee_history': deque(maxlen=200),
                'spread_history': deque(maxlen=200),
                'slippage_history': deque(maxlen=200),
                'ewma_spread': 0.0,
                'ewma_slippage': 0.0,
                'last_micro_update': 0.0,
                'probability_history': deque(maxlen=200),
                'last_liquidity_mode': None,
                'beta_alpha': 1.0,
                'beta_beta': 1.0,
                'beta_last_update': time.time(),
                'last_entry': None,
                'last_exit': None,
            }
            self.symbol_trade_stats[key] = stats
        return stats

    def update_position(self, symbol: str, position_info: Dict):
        """
        Update position tracking.

        Args:
            symbol: Trading symbol
            position_info: Position information
        """
        self.open_positions[symbol] = position_info
        stats = self._get_symbol_stats(symbol)
        stats['entries'] += 1
        stats['open'] += 1
        margin_used = position_info.get('margin_required', 0.0)
        stats['active_margin'] += margin_used
        stats['margin_committed'] += margin_used
        stats['notional_executed'] += position_info.get('notional_value', 0.0)
        stats['last_entry'] = time.time()

        tp_prob = position_info.get('tp_probability')
        try:
            if tp_prob is not None:
                stats['probability_history'].append(float(tp_prob))
        except (TypeError, ValueError):
            pass
        logger.debug(f"Position updated: {symbol} - {position_info}")
        self._record_fee_outflow(position_info.get('notional_value', 0.0), position_info.get('taker_fee_pct'))

    def close_position(self, symbol: str, pnl: float, exit_reason: str, ev_snapshot: Optional[float] = None):
        """
        Close position and update statistics.

        Args:
            symbol: Trading symbol
            pnl: Profit/loss from position
            exit_reason: Reason for closing position
        """
        if symbol in self.open_positions:
            position_info = self.open_positions.pop(symbol)

            # Record trade
            trade_record = {
                'symbol': symbol,
                'entry_time': position_info.get('entry_time'),
                'exit_time': time.time(),
                'pnl': pnl,
                'pnl_percent': pnl / position_info.get('notional_value', 1) * 100,
                'exit_reason': exit_reason,
                'holding_period': time.time() - position_info.get('entry_time', time.time()),
                'max_leverage': position_info.get('leverage_used', 1),
                'taker_fee_pct': position_info.get('taker_fee_pct'),
                'funding_cost_pct': position_info.get('funding_buffer_pct'),
                'tp_probability': position_info.get('tp_probability'),
                'time_constrained_probability': position_info.get('time_constrained_probability'),
                'success': pnl > 0,
                'entry_spread_pct': position_info.get('entry_spread_pct'),
                'entry_slippage_pct': position_info.get('entry_slippage_pct'),
                'exit_slippage_pct': position_info.get('exit_slippage_pct'),
                'execution_cost_floor_pct': position_info.get('execution_cost_floor_pct'),
                'micro_cost_pct': position_info.get('micro_cost_pct'),
                'fee_recovery_balance': self.fee_recovery_balance
            }

            self.trade_history.append(trade_record)
            self.daily_pnl += pnl
            self._record_fee_outflow(position_info.get('notional_value', 0.0), position_info.get('taker_fee_pct'))
            self._reconcile_fee_recovery(pnl)
            self.var_pnl_window.append(pnl)
            self._evaluate_var_guard()
            
            # Record return for Sharpe tracker (Phase 4)
            margin_basis = position_info.get('margin_required') or position_info.get('notional_value', 0)
            trade_return = pnl / margin_basis if margin_basis else 0.0
            self.sharpe_tracker.add_return(trade_return)
            self.trade_returns.append(trade_return)
            if len(self.trade_returns) > 1000:
                self.trade_returns = self.trade_returns[-1000:]
            self._update_expectancy_metrics()

            stats = self._get_symbol_stats(symbol)
            stats['open'] = max(stats['open'] - 1, 0)
            stats['active_margin'] = max(stats['active_margin'] - margin_basis, 0.0)
            stats['net_pnl'] += pnl
            stats['completed'] += 1
            stats['holding_seconds_total'] += trade_record['holding_period']
            stats['returns'].append(trade_return)
            if ev_snapshot is not None and np.isfinite(ev_snapshot):
                stats['ev_history'].append(float(ev_snapshot))
                net_edge = float(ev_snapshot)
            else:
                net_edge = trade_record['pnl_percent'] / 100.0 if trade_record.get('pnl_percent') is not None else trade_return
            stats['ev_history'].append(net_edge)
            taker_fee_pct = trade_record.get('taker_fee_pct')
            if taker_fee_pct is not None:
                stats['fee_history'].append(float(taker_fee_pct))

            # Bayesian posterior update for TP probability
            decay = float(getattr(Config, "POSTERIOR_DECAY", 0.0))
            if decay > 0:
                stats['beta_alpha'] = 1.0 + (stats['beta_alpha'] - 1.0) * (1.0 - decay)
                stats['beta_beta'] = 1.0 + (stats['beta_beta'] - 1.0) * (1.0 - decay)

            if trade_record['success']:
                stats['beta_alpha'] += 1.0
            else:
                stats['beta_beta'] += 1.0

            stats['beta_last_update'] = time.time()
            if pnl >= 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            stats['last_exit'] = time.time()

            session_stats = self.symbol_session_stats[symbol.upper()]
            session_stats['session_trades'] += 1
            session_stats['session_pnl'] += pnl
            if ev_snapshot is not None and np.isfinite(ev_snapshot):
                session_stats['session_ev_sum'] += float(ev_snapshot)
                session_stats['session_ev_count'] += 1
            else:
                session_stats['session_ev_sum'] += net_edge
                session_stats['session_ev_count'] += 1
            if session_stats['session_start_balance'] <= 0 and self.current_portfolio_value > 0:
                session_stats['session_start_balance'] = self.current_portfolio_value

            logger.info(f"Position closed: {symbol} PnL: {pnl:.2f} ({trade_record['pnl_percent']:.1f}%) "
                       f"Reason: {exit_reason}")

            f_score = position_info.get('f_score') if position_info else None
            if f_score is not None:
                key = symbol.upper()
                exit_reason_lower = (exit_reason or '').lower()
                if pnl > 0 or exit_reason_lower.startswith('primary tp') or exit_reason_lower.startswith('secondary tp'):
                    self.curvature_failures[key] = 0
                    self.curvature_success[key] += 1
                else:
                    self.curvature_failures[key] += 1

    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value for drawdown tracking.

        Args:
            new_value: Current portfolio value
        """
        self.current_portfolio_value = new_value
        self.max_portfolio_value = max(self.max_portfolio_value, new_value)

    def get_symbol_trade_summary(self) -> Dict[str, Dict[str, float]]:
        """Return lightweight per-symbol trade metrics for monitoring."""
        summary = {}
        for symbol, stats in self.symbol_trade_stats.items():
            completed = stats['completed']
            returns_arr = np.array(stats['returns']) if stats['returns'] else None
            avg_return = float(np.mean(returns_arr)) if returns_arr is not None and returns_arr.size > 0 else 0.0
            return_variance = float(np.var(returns_arr, ddof=1)) if returns_arr is not None and returns_arr.size > 1 else 0.0
            summary[symbol] = {
                'entries': stats['entries'],
                'completed': completed,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'open': stats['open'],
                'net_pnl': stats['net_pnl'],
                'avg_return': avg_return,
                'return_variance': return_variance,
                'avg_hold_minutes': (stats['holding_seconds_total'] / completed / 60.0) if completed else 0.0,
                'avg_ev': float(np.mean(stats['ev_history'])) if stats['ev_history'] else 0.0,
                'ev_count': len(stats['ev_history']),
                'spread_ewma': float(stats.get('ewma_spread', 0.0) or 0.0),
                'slippage_ewma': float(stats.get('ewma_slippage', 0.0) or 0.0)
            }
        now = time.time()
        for symbol, sess in self.symbol_session_stats.items():
            session_trades = sess.get('session_trades', 0)
            avg_ev = (sess['session_ev_sum'] / sess['session_ev_count']) if sess['session_ev_count'] else 0.0
            summary.setdefault(symbol, {})
            summary[symbol].update({
                'session_trades': session_trades,
                'session_avg_ev': avg_ev,
                'session_pnl': sess.get('session_pnl', 0.0),
                'blocked_until': sess.get('blocked_until', 0.0) - now
            })
        return summary

    def should_block_symbol_micro(self, symbol: str, account_balance: float) -> bool:
        if account_balance >= 25 or not getattr(Config, 'EMERGENCY_CALCULUS_MODE', False):
            return False
        sess = self.symbol_session_stats[symbol.upper()]
        now = time.time()
        blocked_until = sess.get('blocked_until', 0.0)
        if blocked_until and now < blocked_until:
            return True
        trades = sess.get('session_trades', 0)
        ev_count = sess.get('session_ev_count', 0)
        min_samples = getattr(Config, 'MICRO_SYMBOL_EV_MIN_SAMPLES', 6)
        if trades < min_samples or ev_count < 1:
            return False
        avg_ev = sess['session_ev_sum'] / max(sess['session_ev_count'], 1)
        ev_floor = getattr(Config, 'MICRO_SYMBOL_AVG_EV_FLOOR', 0.0)
        start_balance = sess.get('session_start_balance', account_balance)
        drawdown_floor = getattr(Config, 'MICRO_SYMBOL_DRAWDOWN_FLOOR_PCT', 0.05)
        pnl = sess.get('session_pnl', 0.0)
        drawdown_hit = False
        if start_balance > 0:
            drawdown_hit = (-pnl) >= start_balance * drawdown_floor
        if avg_ev < ev_floor or drawdown_hit:
            sess['blocked_until'] = now + getattr(Config, 'MICRO_SYMBOL_BLOCK_DURATION', 300)
            return True
        return False

    def get_symbol_probability_posterior(self, symbol: str) -> Dict[str, float]:
        stats = self._get_symbol_stats(symbol)
        alpha = max(float(stats.get('beta_alpha', 1.0)), 1.0)
        beta = max(float(stats.get('beta_beta', 1.0)), 1.0)
        total = alpha + beta
        mean = alpha / total
        variance = (alpha * beta) / (total ** 2 * (total + 1.0))
        std_dev = math.sqrt(max(variance, 1e-12))
        z_score = float(getattr(Config, "POSTERIOR_CONFIDENCE_Z", 1.96))
        lower = max(0.0, mean - z_score * std_dev)
        upper = min(1.0, mean + z_score * std_dev)
        sample_count = max((alpha + beta) - 2.0, 0.0)
        return {
            'alpha': alpha,
            'beta': beta,
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'lower_bound': lower,
            'upper_bound': upper,
            'count': sample_count
        }

    def record_microstructure_sample(self,
                                     symbol: str,
                                     spread_pct: float,
                                     slippage_pct: Optional[float] = None):
        stats = self._get_symbol_stats(symbol)
        try:
            spread_value = max(float(spread_pct), 0.0)
        except (TypeError, ValueError):
            spread_value = 0.0

        stats['spread_history'].append(spread_value)
        spread_alpha = 0.2
        prev_spread = float(stats.get('ewma_spread', 0.0) or 0.0)
        stats['ewma_spread'] = prev_spread + spread_alpha * (spread_value - prev_spread)

        if slippage_pct is not None:
            try:
                slippage_value = max(float(slippage_pct), 0.0)
            except (TypeError, ValueError):
                slippage_value = 0.0
            stats['slippage_history'].append(slippage_value)
            slip_alpha = 0.2
            prev_slip = float(stats.get('ewma_slippage', 0.0) or 0.0)
            stats['ewma_slippage'] = prev_slip + slip_alpha * (slippage_value - prev_slip)

        stats['last_micro_update'] = time.time()

    def track_probability_snapshot(self, symbol: str, payload: Dict[str, float]) -> None:
        key = symbol.upper()
        self.probability_debug[key] = dict(payload)
        stats = self._get_symbol_stats(symbol)
        final_prob = payload.get('final_probability')
        try:
            if final_prob is not None:
                stats['probability_history'].append(float(final_prob))
        except (TypeError, ValueError):
            pass
        liquidity_mode = payload.get('liquidity_mode')
        if liquidity_mode is not None:
            stats['last_liquidity_mode'] = liquidity_mode

    def _seed_microstructure_stats(self, symbol: str, fallback_spread: float, fallback_slip: float = 0.0) -> None:
        stats = self._get_symbol_stats(symbol)
        stats['ewma_spread'] = fallback_spread
        stats['ewma_slippage'] = fallback_slip
        stats['last_micro_update'] = time.time()
        stats['spread_history'].append(fallback_spread)
        if fallback_slip > 0:
            stats['slippage_history'].append(fallback_slip)

    def estimate_microstructure_cost(self,
                                     symbol: str,
                                     spread_pct: Optional[float] = None) -> float:
        stats = self._get_symbol_stats(symbol)
        spread_estimate = spread_pct if spread_pct is not None else stats.get('ewma_spread', 0.0)
        try:
            spread_estimate = max(float(spread_estimate), 0.0)
        except (TypeError, ValueError):
            spread_estimate = 0.0

        liquid_symbols = {'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'}
        fallback_spread = 0.0004 if symbol.upper() in liquid_symbols else 0.0006
        fallback_slip = 0.0002 if symbol.upper() in liquid_symbols else 0.0003
        now = time.time()
        last_update = float(stats.get('last_micro_update', 0.0) or 0.0)
        stale = (now - last_update) > 300 if last_update > 0 else True

        if stale or (not stats['spread_history'] and spread_estimate <= 0):
            self._seed_microstructure_stats(symbol, fallback_spread, fallback_slip)
            spread_estimate = max(spread_estimate, fallback_spread)

        ewma_spread = float(stats.get('ewma_spread', 0.0) or 0.0)
        ewma_slippage = float(stats.get('ewma_slippage', 0.0) or 0.0)

        spread_component = max(spread_estimate, ewma_spread)
        slippage_component = max(ewma_slippage, 0.0)

        # CRYPTO-OPTIMIZED: Assume entry and exit both cross the spread; include slippage buffer
        micro_cost_pct = spread_component + slippage_component

        debug_details = {
            'raw_spread_pct': spread_estimate,
            'ewma_spread_pct': ewma_spread,
            'ewma_slippage_pct': ewma_slippage,
            'spread_component_pct': spread_component,
            'slippage_component_pct': slippage_component,
            'pre_cap_micro_pct': micro_cost_pct
        }

        cap_reason = None
        micro_cap_pct = 0.002  # Maximum 0.2% total microstructure cost for crypto
        if micro_cost_pct > micro_cap_pct:
            micro_cost_pct = micro_cap_pct
            cap_reason = 'global_cap_0.20%'

        if symbol.upper() in liquid_symbols and micro_cost_pct > 0.001:
            micro_cost_pct = 0.001
            cap_reason = 'liquid_cap_0.10%'

        if ewma_spread < 0.0001 and micro_cost_pct > 0.0001:
            micro_cost_pct = 0.0001
            cap_reason = 'narrow_spread_cap_0.01%'

        debug_details['cap_reason'] = cap_reason
        debug_details['final_micro_pct'] = micro_cost_pct

        key = symbol.upper()
        self.microstructure_debug[key] = debug_details

        if getattr(Config, "EV_DEBUG_LOGGING", False):
            logger.info("Microstructure cost debug %s: %s", key, debug_details)

        return max(micro_cost_pct, 0.0)

    def get_microstructure_metrics(self, symbol: str) -> Dict[str, float]:
        stats = self._get_symbol_stats(symbol)
        now = time.time()
        last_update = float(stats.get('last_micro_update', 0.0) or 0.0)
        liquid_symbols = {'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'}
        fallback_spread = 0.0004 if symbol.upper() in liquid_symbols else 0.0006
        fallback_slip = 0.0002 if symbol.upper() in liquid_symbols else 0.0003
        if last_update == 0 or (now - last_update) > 300:
            self._seed_microstructure_stats(symbol, fallback_spread, fallback_slip)

        return {
            'spread_ewma': float(stats.get('ewma_spread', 0.0) or 0.0),
            'slippage_ewma': float(stats.get('ewma_slippage', 0.0) or 0.0),
            'spread_samples': len(stats.get('spread_history', []) or []),
            'slippage_samples': len(stats.get('slippage_history', []) or []),
            'avg_ev': float(np.mean(stats['ev_history'])) if stats['ev_history'] else 0.0,
            'ev_samples': len(stats['ev_history']),
            'micro_cost_pct': self.estimate_microstructure_cost(symbol)
        }

    def get_microstructure_debug(self, symbol: str) -> Dict[str, float]:
        return self.microstructure_debug.get(symbol.upper(), {})

    def get_fee_floor_debug(self, symbol: str) -> Dict[str, float]:
        return self.fee_floor_debug.get(symbol.upper(), {})

    def get_probability_debug(self, symbol: str) -> Dict[str, float]:
        return self.probability_debug.get(symbol.upper(), {})

    def is_symbol_allowed_for_tier(self,
                                   symbol: str,
                                   tier_name: str,
                                   tier_min_ev_pct: float) -> bool:
        symbol = symbol.upper()
        if self.calculus_priority_mode:
            return True
        if self.should_block_symbol_micro(symbol, self.current_portfolio_value or 0.0):
            return False
        whitelist = getattr(Config, "SYMBOL_TIER_WHITELIST", {})
        candidate_pool = getattr(Config, "SYMBOL_CANDIDATE_POOL", {})
        if whitelist and symbol in whitelist.get(tier_name, []):
            return True

        if symbol in whitelist.get("*", []):
            return True

        if symbol in candidate_pool.get(tier_name, []):
            limits = getattr(Config, "MICROSTRUCTURE_LIMITS", {})
            max_spread = float(limits.get('max_spread_pct', 0.0006))
            max_slippage = float(limits.get('max_slippage_pct', 0.0008))
            min_samples = int(limits.get('min_samples', 10))
            ev_samples_required = int(limits.get('candidate_ev_samples', 15))
            ev_buffer = float(limits.get('candidate_ev_buffer', 0.0))

            stats = self._get_symbol_stats(symbol)
            spread_samples = len(stats.get('spread_history', []) or [])
            slippage_samples = len(stats.get('slippage_history', []) or [])
            spread_ok = float(stats.get('ewma_spread', 0.0) or 0.0) <= max_spread
            slippage_ok = float(stats.get('ewma_slippage', 0.0) or 0.0) <= max_slippage
            sample_ok = spread_samples >= min_samples and slippage_samples >= min_samples

            ev_hist = stats.get('ev_history')
            ev_sample_ok = ev_hist and len(ev_hist) >= ev_samples_required
            avg_ev = (sum(ev_hist) / len(ev_hist)) if ev_hist else 0.0
            ev_ok = avg_ev >= (tier_min_ev_pct + ev_buffer)

            return sample_ok and spread_ok and slippage_ok and ev_sample_ok and ev_ok

        return False

    def get_symbol_ev_metrics(self, symbol: str) -> Dict[str, float]:
        stats = self._get_symbol_stats(symbol)
        ev_hist = stats.get('ev_history') or []
        if not ev_hist:
            return {
                'avg_ev': 0.0,
                'count': 0,
                'min_ev': 0.0,
                'max_ev': 0.0
            }
        ev_array = np.array(ev_hist, dtype=float)
        return {
            'avg_ev': float(ev_array.mean()),
            'count': int(ev_array.size),
            'min_ev': float(ev_array.min()),
            'max_ev': float(ev_array.max())
        }

    def should_block_symbol_by_ev(self, symbol: str, min_ev_pct: float, window: int = 20) -> bool:
        if self.calculus_priority_mode:
            return False
        stats = self._get_symbol_stats(symbol)
        ev_hist: deque = stats.get('ev_history') or deque()
        if not ev_hist:
            return False
        if len(ev_hist) < max(window // 2, 5):
            return False
        recent = list(ev_hist)[-window:]
        avg_ev = sum(recent) / len(recent)
        stats['avg_ev_recent'] = avg_ev
        stats['avg_ev_window'] = len(recent)
        return avg_ev < float(min_ev_pct)

    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current portfolio risk metrics.

        Returns:
            RiskMetrics with current portfolio state
        """
        total_exposure = sum(pos.get('notional_value', 0) for pos in self.open_positions.values())
        margin_used = sum(pos.get('margin_required', 0) for pos in self.open_positions.values())

        # Calculate drawdown
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
        else:
            current_drawdown = 0

        # Calculate Sharpe ratio (simplified)
        if self.trade_history:
            returns = [trade['pnl_percent'] for trade in self.trade_history]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        return RiskMetrics(
            total_exposure=total_exposure,
            available_balance=self.current_portfolio_value - margin_used,
            margin_used_percent=margin_used / self.current_portfolio_value if self.current_portfolio_value > 0 else 0,
            open_positions_count=len(self.open_positions),
            max_drawdown=current_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_score=self._calculate_portfolio_correlation()
        )

    def _calculate_portfolio_correlation(self) -> float:
        """
        Calculate overall portfolio correlation score.

        Returns:
            Correlation score (0-1, higher = more correlated)
        """
        if len(self.open_positions) <= 1:
            return 0

        # Simplified correlation calculation
        # In practice, this would use actual price correlation data
        long_positions = sum(1 for pos in self.open_positions.values() if pos.get('side') == 'long')
        short_positions = len(self.open_positions) - long_positions

        # Balance score (more balanced = lower correlation)
        balance_score = 1 - abs(long_positions - short_positions) / len(self.open_positions)

        return 1 - balance_score  # Convert to correlation score

    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.

        Returns:
            Risk report with all relevant metrics
        """
        risk_metrics = self.calculate_risk_metrics()

        return {
            'risk_metrics': risk_metrics,
            'open_positions': self.open_positions,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'daily_pnl': self.daily_pnl,
            'risk_limits': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_leverage': self.max_leverage,
                'max_positions': self.max_positions
            },
            'recommendations': self._generate_risk_recommendations(risk_metrics)
        }

    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics) -> List[str]:
        """
        Generate risk management recommendations.

        Args:
            risk_metrics: Current risk metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        if risk_metrics.margin_used_percent > 0.8:
            recommendations.append("High margin usage - consider reducing position sizes")

        if risk_metrics.current_drawdown > 0.1:
            recommendations.append("High drawdown - consider reducing risk exposure")


# ==============================================================================
# RENAISSANCE MEDALLION: DAILY DRIFT PREDICTOR
# ==============================================================================

class DailyDriftPredictor:
    """
    Predicts tomorrow's expected return using multi-factor linear model.
    
    This is the CORE of Renaissance's edge - they predict the daily distribution
    and then execute thousands of intraday trades aligned to that prediction.
    
    Formula: E[r_t+1] = w0 + w1*order_flow + w2*vol_regime + w3*funding + w4*dow
    """
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.lookback_samples = lookback_hours * 60  # 1-min resolution
        
        # Per-symbol tracking
        self.order_flow_history = {}      # symbol -> deque of (buy - sell)
        self.volatility_history = {}      # symbol -> deque of realized vol
        self.funding_history = {}         # symbol -> deque of funding rates
        self.spread_history = {}          # symbol -> deque of bid-ask spreads (pct)
        self.depth_history = {}           # symbol -> deque of cumulative depth
        self.timestamps = {}              # symbol -> deque of timestamps
        
        # Model weights (empirically derived from historical data)
        self.model_weights = {
            'bias': 0.00005,                  # +0.05% base daily drift
            'order_flow': 0.0001,             # Order flow coefficient
            'volatility_regime': -0.0001,     # High vol = mean revert
            'funding_bias': 0.00002,          # Funding pressure
            'liquidity_delta': -0.00008,      # Liquidity factor (high spread → expect revert)
            'day_of_week': {}                 # Day effects
        }
        
        for day in range(5):
            self.model_weights['day_of_week'][day] = 0.00001
        self.model_weights['day_of_week'][0] = 0.00015  # Monday boost
    
    def update_orderflow(self, symbol: str, buy_volume: float, sell_volume: float, timestamp: float):
        """Update order flow for symbol."""
        if symbol not in self.order_flow_history:
            self.order_flow_history[symbol] = deque(maxlen=self.lookback_samples)
            self.timestamps[symbol] = deque(maxlen=self.lookback_samples)
        
        imbalance = buy_volume - sell_volume
        self.order_flow_history[symbol].append(imbalance)
        self.timestamps[symbol].append(timestamp)
    
    def update_volatility(self, symbol: str, current_volatility: float):
        """Update volatility for symbol."""
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=self.lookback_samples)
        self.volatility_history[symbol].append(current_volatility)
    
    def update_funding(self, symbol: str, funding_rate: float):
        """Update funding rate for symbol."""
        if symbol not in self.funding_history:
            self.funding_history[symbol] = deque(maxlen=self.lookback_samples)
        self.funding_history[symbol].append(funding_rate)
    
    def update_spread(self, symbol: str, spread_pct: float):
        """Update bid-ask spread for symbol (as percentage)."""
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=self.lookback_samples)
        self.spread_history[symbol].append(spread_pct)
    
    def update_depth(self, symbol: str, cumulative_depth: float):
        """Update order book depth for symbol (cumulative $)."""
        if symbol not in self.depth_history:
            self.depth_history[symbol] = deque(maxlen=self.lookback_samples)
        self.depth_history[symbol].append(cumulative_depth)
    
    def predict_drift(self, symbol: str) -> Dict:
        """Predict E[tomorrow's return]."""
        if symbol not in self.order_flow_history or len(self.order_flow_history[symbol]) < 60:
            return {
                'expected_return_pct': 0.0,
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
        
        # Calculate factors
        bias = self.model_weights['bias']
        
        # Order flow factor
        flow_values = list(self.order_flow_history[symbol])
        mean_flow = np.mean(flow_values[-60:])
        std_flow = np.std(flow_values[-60:]) or 1
        normalized_flow = (mean_flow / abs(std_flow) if std_flow > 0 else 0) / 10
        order_flow_contrib = self.model_weights['order_flow'] * np.tanh(normalized_flow)
        
        # Volatility factor
        vol_contrib = 0.0
        if symbol in self.volatility_history and len(self.volatility_history[symbol]) > 0:
            current_vol = list(self.volatility_history[symbol])[-1]
            mean_vol = np.mean(list(self.volatility_history[symbol])[-60:]) or 1
            vol_ratio = current_vol / mean_vol if mean_vol > 0 else 1
            vol_contrib = self.model_weights['volatility_regime'] * (vol_ratio - 1.0)
        
        # Funding factor
        funding_contrib = 0.0
        if symbol in self.funding_history and len(self.funding_history[symbol]) > 0:
            current_funding = list(self.funding_history[symbol])[-1]
            funding_contrib = self.model_weights['funding_bias'] * np.tanh(current_funding / 0.0001)
        
        # Liquidity factor (spread-based): high spread indicates low liquidity, expect mean reversion
        liquidity_contrib = 0.0
        if symbol in self.spread_history and len(self.spread_history[symbol]) > 10:
            spreads = list(self.spread_history[symbol])[-60:]
            current_spread = spreads[-1]
            mean_spread = np.mean(spreads)
            if mean_spread > 1e-6:
                spread_ratio = current_spread / mean_spread
                # High spread relative to mean → negative return (expect revert tighter)
                liquidity_contrib = self.model_weights['liquidity_delta'] * (spread_ratio - 1.0)
        
        # Day-of-week factor
        dow = datetime.now().weekday()
        dow_contrib = self.model_weights['day_of_week'].get(dow, 0.0)
        
        # Combine (all 6 factors)
        expected_return = bias + order_flow_contrib + vol_contrib + funding_contrib + liquidity_contrib + dow_contrib
        
        # Confidence
        signal_strength = abs(normalized_flow) + abs(vol_ratio - 1.0) if 'vol_ratio' in locals() else 0
        confidence = float(np.clip(signal_strength / 5.0, 0.0, 1.0))
        
        direction = 'BULLISH' if expected_return > 0.00001 else 'BEARISH' if expected_return < -0.00001 else 'NEUTRAL'
        
        return {
            'expected_return_pct': float(expected_return),
            'confidence': float(confidence),
            'direction': direction
        }
    
    def is_aligned(self, symbol: str, micro_direction: str) -> bool:
        """Check if micro signal aligns with daily drift."""
        drift = self.predict_drift(symbol)
        daily_dir = 'BUY' if drift['expected_return_pct'] > 0.00001 else 'SELL' if drift['expected_return_pct'] < -0.00001 else 'NEUTRAL'
        return micro_direction == daily_dir if daily_dir != 'NEUTRAL' else True
    
    def get_alignment_boost(self, symbol: str, micro_direction: str) -> float:
        """Get confidence multiplier from alignment."""
        drift = self.predict_drift(symbol)
        if drift['confidence'] < 0.3:
            return 1.0
        if self.is_aligned(symbol, micro_direction):
            return 1.0 + (drift['confidence'] * 0.3)
        return max(0.7, 1.0 - (drift['confidence'] * 0.2))
    
    def predict_drift_adaptive(self, symbol: str, velocity_magnitude: float = 0.0, 
                               acceleration_magnitude: float = 0.0) -> Dict:
        """
        Adaptive drift prediction based on signal timescale.
        
        Higher derivatives (acceleration, jerk) indicate faster/shorter-horizon signals.
        Adjust drift confidence weights accordingly:
        - High velocity/accel → shorter horizon → lower daily drift weight
        - Low velocity/accel → longer horizon → higher daily drift weight
        """
        base_drift = self.predict_drift(symbol)
        
        # Infer signal horizon from derivative magnitudes
        # velocity >= 0.001 suggests fast signal
        # acceleration >= 0.0001 suggests very fast signal
        if acceleration_magnitude >= 0.0001:
            # Very fast signal (1-5 min horizon) - daily drift less relevant
            horizon_scale = 0.5
        elif velocity_magnitude >= 0.001:
            # Fast signal (5-15 min horizon) - moderate drift relevance
            horizon_scale = 0.75
        else:
            # Slower signal (15+ min horizon) - full drift relevance
            horizon_scale = 1.0
        
        # Adjust confidence based on horizon alignment with drift
        adjusted_confidence = base_drift['confidence'] * horizon_scale
        
        return {
            'expected_return_pct': float(base_drift['expected_return_pct']),
            'confidence': float(adjusted_confidence),
            'direction': base_drift['direction'],
            'horizon_scale': float(horizon_scale)
        }
    
    def predict_drift_cross_asset(self, symbol: str, cross_contributions: Dict[str, float]) -> Dict:
        """
        Enhance drift prediction using cross-asset signals.
        
        Signature: E[r_target] = base_drift + cross_asset_boost
        
        Where:
        cross_asset_boost = sum(correlation[i] * momentum[i] * influence[i])
        
        for all correlated symbols
        """
        base_drift = self.predict_drift(symbol)
        
        # Compute cross-asset enhancement
        cross_boost = 0.0
        for other_symbol, correlation in cross_contributions.items():
            try:
                other_drift = self.predict_drift(other_symbol)
                # Weight by correlation strength and other's drift direction
                contribution = correlation * other_drift['expected_return_pct']
                cross_boost += contribution * 0.2  # 20% weight to cross signals
            except Exception as e:
                logger.warning(f"Error computing cross-asset contribution for {other_symbol}: {e}")
        
        enhanced_return = base_drift['expected_return_pct'] + cross_boost
        enhanced_confidence = base_drift['confidence'] * (1.0 + min(abs(cross_boost) / 0.001, 0.5))
        
        direction = 'BULLISH' if enhanced_return > 0.00001 else 'BEARISH' if enhanced_return < -0.00001 else 'NEUTRAL'
        
        return {
            'expected_return_pct': float(enhanced_return),
            'confidence': float(np.clip(enhanced_confidence, 0, 1)),
            'direction': direction,
            'cross_asset_boost': float(cross_boost),
            'components': cross_contributions
        }

        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy parameters")

        if risk_metrics.correlation_score > 0.7:
            recommendations.append("High position correlation - diversify positions")

        if len(self.open_positions) >= self.max_positions * 0.8:
            recommendations.append("Approaching maximum position limit")

        return recommendations or ["Risk parameters within acceptable limits"]


class CrossAssetReturnMatrix:
    """
    Renaissance-style cross-asset return prediction layer.
    
    Tracks returns across multiple crypto assets simultaneously to exploit:
    - Momentum bleed (BTC → ETH → SOL)
    - Correlation structures
    - Lead-lag relationships
    - Funding rate contagion
    
    This amplifies return predictability beyond single-asset analysis.
    """
    
    def __init__(self, symbols: List[str], lookback_periods: int = 120):
        self.symbols = [s.upper() for s in symbols]  # Normalize to uppercase
        self.lookback = lookback_periods  # 120 1-min candles
        
        # Track returns: symbol -> deque of log returns
        self.returns_history = {}
        for symbol in self.symbols:
            self.returns_history[symbol] = deque(maxlen=lookback_periods)
        
        # Track prices for return calculation
        self.last_prices = {}
        
        # Cache correlation matrix (recompute every 60 ticks)
        self.correlation_matrix = None
        self.correlation_cache_age = 0
        self.correlation_refresh_interval = 60
    
    def update_price(self, symbol: str, current_price: float) -> bool:
        """
        Update price for symbol, compute log-return.
        
        Returns:
            True if return was recorded, False if insufficient history
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.symbols:
            return False
        
        if symbol_upper not in self.last_prices:
            self.last_prices[symbol_upper] = current_price
            return False
        
        # Compute log-return
        prev_price = self.last_prices[symbol_upper]
        if prev_price <= 0:
            self.last_prices[symbol_upper] = current_price
            return False
        
        log_return = float(np.log(current_price / prev_price))
        self.returns_history[symbol_upper].append(log_return)
        self.last_prices[symbol_upper] = current_price
        
        # Invalidate correlation cache
        self.correlation_cache_age += 1
        
        return len(self.returns_history[symbol_upper]) > 10
    
    def get_correlation_matrix(self, force_recompute: bool = False) -> Optional[np.ndarray]:
        """
        Compute or retrieve cached correlation matrix (NxN).
        
        Recomputes every N ticks to adapt to changing market dynamics.
        """
        if (self.correlation_matrix is None or 
            self.correlation_cache_age >= self.correlation_refresh_interval or
            force_recompute):
            
            # Collect returns for all symbols that have data
            return_arrays = []
            valid_symbols = []
            for symbol in self.symbols:
                if len(self.returns_history[symbol]) >= 20:
                    return_arrays.append(list(self.returns_history[symbol]))
                    valid_symbols.append(symbol)
            
            if len(return_arrays) < 2:
                return None
            
            # Pad to same length
            max_len = max(len(r) for r in return_arrays)
            padded = []
            for r in return_arrays:
                if len(r) < max_len:
                    r = [0.0] * (max_len - len(r)) + r
                padded.append(r)
            
            # Compute correlation
            returns_df = pd.DataFrame(padded, index=valid_symbols)
            try:
                self.correlation_matrix = returns_df.T.corr().fillna(0).values
                self.correlation_cache_age = 0
            except Exception as e:
                logger.error(f"Error computing correlation matrix: {e}")
                return None
        
        return self.correlation_matrix
    
    def get_cross_asset_contributions(self, target_symbol: str) -> Dict[str, float]:
        """
        Return correlation weights for how other symbols influence target.
        
        Returns:
            Dict: {'BTC': 0.82, 'ETH': 0.71, 'SOL': 0.45, ...}
        """
        target_upper = target_symbol.upper()
        if target_upper not in self.symbols:
            return {}
        
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None or len(corr_matrix) == 0:
            return {}
        
        # Find index of target symbol
        valid_symbols = []
        for symbol in self.symbols:
            if len(self.returns_history[symbol]) >= 20:
                valid_symbols.append(symbol)
        
        if target_upper not in valid_symbols:
            return {}
        
        target_idx = valid_symbols.index(target_upper)
        
        # Get correlations for target (absolute value, so both positive/negative matter)
        if target_idx >= len(corr_matrix):
            return {}
        
        target_correlations = corr_matrix[target_idx]
        
        # Build contributions dict
        contributions = {}
        for i, symbol in enumerate(valid_symbols):
            if symbol != target_upper and i < len(target_correlations):
                corr = float(abs(target_correlations[i]))  # Use absolute value
                if corr > 0.1:  # Only significant correlations
                    contributions[symbol] = corr
        
        return contributions
    
    def get_momentum_direction(self, symbol: str) -> float:
        """
        Get recent momentum for symbol (-1 to +1).
        
        Positive: recent returns trending up
        Negative: recent returns trending down
        Zero: neutral
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.returns_history:
            return 0.0
        
        recent_returns = list(self.returns_history[symbol_upper])[-10:]
        if not recent_returns:
            return 0.0
        
        avg_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) or 1.0
        
        # Normalize to [-1, 1]
        momentum = np.tanh(avg_return / (std_return * 0.01))
        return float(momentum)


# Global risk manager instance
risk_manager = RiskManager()
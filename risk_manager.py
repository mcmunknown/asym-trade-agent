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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque
from config import Config
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
        self.symbol_trade_stats: Dict[str, Dict] = {}

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
        self._init_sharpe_tracker()
        self.consecutive_losses = 0
        self.milestones = [10, 20, 50, 100, 200, 500, 1000]
        self.reached_milestones = set()
        self.session_start_balance = 0.0
        self.session_start_time = time.time()
    
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
            # AGGRESSIVE MODE for small accounts (<$20) - need to hit $5 minimum notional
            if account_balance > 0 and account_balance < 20:
                if trade_count <= 20:
                    return 10.0  # Small account: 10x to meet $5 minimum (was 5x, too low!)
                elif trade_count <= 50:
                    return 12.0  # Ramp up
                elif trade_count <= 100:
                    return 15.0  # Pre-dynamic
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
        if account_balance < 10:
            return 15.0  # Maximum aggression for tiny balances
        elif account_balance < 20:
            return 12.0  # High aggression for acceleration phase
        elif account_balance < 50:
            return 10.0  # Moderate aggression
        elif account_balance < 100:
            return 8.0   # Reducing as balance grows
        elif account_balance < 200:
            return 7.0   # More conservative
        elif account_balance < 500:
            return 6.0   # Consolidation phase
        else:
            return 5.0   # Capital preservation mode
    
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
                              volatility: float = None) -> PositionSize:
        """
        Calculate optimal position size based on calculus signal strength and portfolio risk.

        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (SNR value)
            confidence: Signal confidence (0-1)
            current_price: Current market price
            account_balance: Available account balance
            volatility: Current volatility (optional)

        Returns:
            PositionSize calculation result
        """
        try:
            # AGGRESSIVE COMPOUNDING MODE
            # Use Kelly Criterion for optimal position sizing
            
            # Get optimal leverage for current balance
            optimal_leverage = self.get_optimal_leverage(account_balance)
            
            # Get Kelly position fraction (40-60% of capital based on confidence)
            kelly_fraction = self.get_kelly_position_fraction(confidence)
            
            # Apply consecutive loss protection
            if self.consecutive_losses >= 3:
                kelly_fraction *= 0.5  # Cut position size by 50% after 3 losses
                optimal_leverage *= 0.7  # Reduce leverage by 30%
                logger.warning(f"⚠️  {self.consecutive_losses} consecutive losses - reducing position size & leverage")
            
            # Calculate position notional (Kelly fraction of capital × leverage)
            position_notional = account_balance * kelly_fraction * optimal_leverage

            # Apply per-symbol notional caps (dynamic by balance tier)
            symbol_cap = self._resolve_symbol_notional_cap(symbol, account_balance)
            if symbol_cap is not None and symbol_cap > 0 and position_notional > symbol_cap:
                logger.info(
                    f"📉 Notional capped for {symbol}: {position_notional:.2f} → {symbol_cap:.2f}"
                )
                position_notional = symbol_cap
                denom = max(account_balance * max(optimal_leverage, 1e-9), 1e-9)
                kelly_fraction = position_notional / denom

            # Enforce exchange minimum order value with tiny buffer
            min_notional = max(getattr(Config, "MIN_ORDER_NOTIONAL", 5.05), 5.0)
            if position_notional < min_notional:
                logger.info(
                    f"📈 Raising notional for {symbol} to meet $5 minimum: {position_notional:.2f} → {min_notional:.2f}"
                )
                position_notional = min_notional
                denom = max(account_balance * max(optimal_leverage, 1e-9), 1e-9)
                kelly_fraction = position_notional / denom
            
            # Volatility adjustment (higher volatility = smaller position)
            if volatility is not None and volatility > 0.03:  # >3% volatility
                volatility_adjustment = min(0.03 / volatility, 1.0)
                position_notional *= volatility_adjustment
            
            # Calculate quantity
            quantity = position_notional / current_price
            
            # Calculate margin requirement
            margin_required = position_notional / optimal_leverage
            
            # CRITICAL SAFETY: For small balances (<$20), NEVER use more than 40% per trade
            # This prevents "all-in" trades that leave no room for other opportunities
            if account_balance < 20:
                max_margin_pct = 0.40  # Max 40% of balance per trade
            elif account_balance < 50:
                max_margin_pct = 0.50  # Max 50% for growing accounts
            else:
                max_margin_pct = 0.60  # Max 60% for larger accounts
            
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

            logger.info(f"💰 AGGRESSIVE SIZING: Balance=${account_balance:.2f}, "
                       f"Kelly={kelly_fraction:.1%}{ev_info}, Leverage={optimal_leverage:.1f}x, "
                       f"Notional=${position_notional:.2f}, Margin=${margin_required:.2f}")

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
                               half_life_seconds: Optional[float] = None) -> TradingLevels:
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

            tp_offset = current_price * sigma_pct * 1.5
            sl_offset = current_price * sigma_pct * 0.75

            if position_side == "long":
                take_profit = current_price + tp_offset
                stop_loss = current_price - sl_offset
            else:
                take_profit = current_price - tp_offset
                stop_loss = current_price + sl_offset

            risk_reward_ratio = (tp_offset / sl_offset) if sl_offset > 0 else 0.0

            # Confidence blends velocity/acceleration relative to volatility scale
            volatility_floor = max(sigma_pct, 1e-6)
            velocity_strength = abs(velocity)
            acceleration_strength = abs(acceleration)
            normalized_velocity = min(velocity_strength / volatility_floor, 2.0)
            normalized_acceleration = min(acceleration_strength / volatility_floor, 2.0)
            confidence_level = min(1.0, 0.5 * normalized_velocity + 0.5 * normalized_acceleration)

            max_hold_seconds = None
            if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                max_hold_seconds = max(half_life_seconds * 2.0, 60.0)

            trail_stop = None
            if signal_type == SignalType.TRAIL_STOP_UP:
                trail_stop = stop_loss
            elif signal_type == SignalType.HOLD_SHORT:
                trail_stop = stop_loss

            return TradingLevels(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trail_stop=trail_stop,
                risk_reward_ratio=risk_reward_ratio,
                position_side=position_side,
                confidence_level=confidence_level,
                max_hold_seconds=max_hold_seconds
            )

        except Exception as e:
            logger.error(f"Error calculating TP/SL levels: {e}")
            return TradingLevels(current_price, current_price, current_price, None, 1.0, "long", 0.0, None)

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

            # Check confidence level
            if position_size.confidence_score < 0.5:
                return False, f"Low confidence score: {position_size.confidence_score:.2f}"

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
        logger.debug(f"Position updated: {symbol} - {position_info}")

    def close_position(self, symbol: str, pnl: float, exit_reason: str):
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
                'max_leverage': position_info.get('leverage_used', 1)
            }

            self.trade_history.append(trade_record)
            self.daily_pnl += pnl
            
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
            if pnl >= 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            stats['last_exit'] = time.time()

            logger.info(f"Position closed: {symbol} PnL: {pnl:.2f} ({trade_record['pnl_percent']:.1f}%) "
                       f"Reason: {exit_reason}")

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
            }
        return summary

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

        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy parameters")

        if risk_metrics.correlation_score > 0.7:
            recommendations.append("High position correlation - diversify positions")

        if len(self.open_positions) >= self.max_positions * 0.8:
            recommendations.append("Approaching maximum position limit")

        return recommendations or ["Risk parameters within acceptable limits"]

# Global risk manager instance
risk_manager = RiskManager()
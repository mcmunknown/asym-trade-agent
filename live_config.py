"""
🚀 LIVE TRADING CONFIGURATION
===========================

⚠️ CRITICAL WARNING: This configuration is for REAL live trading with REAL money!
   - This will execute REAL trades on Bybit exchange
   - REAL money will be at risk
   - REAL leverage will be used
   - REAL profits and losses will occur

DO NOT USE THIS CONFIGURATION unless:
1. You have thoroughly tested the system in simulation mode
2. You understand all the risks involved
3. You have sufficient risk capital you can afford to lose
4. You have appropriate risk management controls in place

This configuration implements Anne's complete calculus-based portfolio trading system
with institutional-grade risk management for live cryptocurrency trading.
"""

import os
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class TradingMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class LiveTradingConfig:
    """Complete configuration for live trading system."""

    # =============================================================================
    # 🚨 CRITICAL LIVE TRADING SETTINGS
    # =============================================================================
    trading_mode: TradingMode = TradingMode.LIVE  # ⚠️ ALWAYS LIVE FOR REAL TRADING
    enable_real_trading: bool = True  # ⚠️ MUST BE True FOR LIVE TRADING
    require_manual_confirmation: bool = False  # Set to True for additional safety

    # =============================================================================
    # 📊 TRADING SYMBOLS & PORTFOLIO
    # =============================================================================
    # Default 8 major crypto assets for diversification
    symbols: List[str] = None

    # Portfolio allocation strategy
    portfolio_mode: bool = True  # Enable portfolio management
    initial_capital: float = 100000.0  # $100k starting capital
    max_portfolio_risk: float = 0.15  # 15% max portfolio risk
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalancing

    # =============================================================================
    # 🔬 ANNE'S CALCULUS STRATEGY SETTINGS
    # =============================================================================
    # Calculus analysis parameters
    lambda_param: float = 0.75  # Exponential smoothing parameter
    snr_threshold: float = 0.7  # Signal-to-noise ratio threshold
    min_signal_interval: int = 30  # Minimum seconds between signals

    # Kalman filter settings
    kalman_process_variance: float = 1e-5
    kalman_measurement_variance: float = 1e-3
    kalman_initial_state_variance: float = 1.0

    # =============================================================================
    # 🛡️ RISK MANAGEMENT CRITICAL SETTINGS
    # =============================================================================
    # Position sizing
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 25000.0  # $25k max per position
    max_positions: int = 8  # Maximum concurrent positions

    # Leverage controls
    max_leverage: float = 3.0  # Maximum 3x leverage
    default_leverage: float = 2.0  # Default 2x leverage

    # Stop loss and take profit
    max_stop_loss_percent: float = 0.05  # 5% maximum stop loss
    default_stop_loss_percent: float = 0.03  # 3% default stop loss
    default_take_profit_percent: float = 0.06  # 6% default take profit
    trailing_stop_enabled: bool = True  # Enable trailing stops

    # Daily and overall limits
    daily_loss_limit: float = 0.10  # 10% daily loss limit
    max_consecutive_losses: int = 5  # Stop after 5 consecutive losses
    max_drawdown: float = 0.20  # 20% maximum drawdown

    # =============================================================================
    # 💰 CAPITAL & ACCOUNT SETTINGS
    # =============================================================================
    # Account configuration
    account_type: str = "UNIFIED"  # Bybit account type
    base_currency: str = "USDT"  # Base currency for trading

    # Capital allocation
    trading_capital_percent: float = 0.80  # 80% of available capital for trading
    reserve_capital_percent: float = 0.20  # 20% reserve for margin/costs

    # =============================================================================
    # 📈 PORTFOLIO OPTIMIZATION SETTINGS
    # =============================================================================
    # Joint distribution analysis
    lookback_period: int = 100  # Days for covariance calculation
    min_observations: int = 50  # Minimum observations for analysis

    # Portfolio optimization
    optimization_objective: str = "sharpe"  # sharpe, min_variance, risk_parity
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # =============================================================================
    # 🔄 EXECUTION & ORDER SETTINGS
    # =============================================================================
    # Order execution
    order_type: str = "Market"  # Market, Limit
    execution_timeout: int = 30  # Seconds before order cancellation
    slippage_tolerance: float = 0.001  # 0.1% slippage tolerance

    # Position management
    position_update_interval: int = 10  # Seconds between position updates
    profit_check_interval: int = 30  # Seconds between profit checks

    # =============================================================================
    # 📡 DATA & WEBSOCKET SETTINGS
    # =============================================================================
    # WebSocket configuration
    websocket_symbols: List[str] = None
    channel_types: List[str] = None
    heartbeat_interval: int = 20
    reconnect_attempts: int = 5
    reconnect_delay: int = 10

    # Data quality
    price_validation_enabled: bool = True
    max_price_change_percent: float = 0.20  # 20% max price change validation

    # =============================================================================
    # 📊 MONITORING & LOGGING
    # =============================================================================
    # Performance monitoring
    enable_performance_tracking: bool = True
    performance_update_interval: int = 60  # Seconds
    export_performance_data: bool = True
    performance_export_path: str = "performance_data.json"

    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "live_trading.log"
    max_log_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_log_count: int = 5

    # =============================================================================
    # 🚨 EMERGENCY & SAFETY CONTROLS
    # =============================================================================
    # Emergency controls
    emergency_stop_enabled: bool = True
    emergency_stop_conditions: List[str] = None

    # Circuit breakers
    circuit_breaker_enabled: bool = True
    error_rate_threshold: float = 0.10  # 10% error rate triggers circuit breaker
    max_api_errors_per_minute: int = 10

    # Position monitoring
    position_monitoring_enabled: bool = True
    auto_close_positions_on_stop: bool = True

    # =============================================================================
    # 🔧 SYSTEM & OPERATIONAL SETTINGS
    # =============================================================================
    # Threading and performance
    max_concurrent_signals: int = 5
    signal_processing_timeout: int = 10  # Seconds
    portfolio_update_timeout: int = 30  # Seconds

    # Backup and recovery
    enable_state_backup: bool = True
    state_backup_interval: int = 300  # Seconds
    state_backup_path: str = "system_state.json"

    # =============================================================================
    # 📈 TRADING HOURS & MARKET CONDITIONS
    # =============================================================================
    # Trading schedule
    trading_hours_enabled: bool = False  # 24/7 for crypto
    trading_start_hour: int = 0  # UTC
    trading_end_hour: int = 23  # UTC

    # Market conditions
    volatility_threshold: float = 0.05  # 5% volatility threshold
    suspend_trading_high_volatility: bool = True

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        # Set default symbols if not provided
        if self.symbols is None:
            self.symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
                'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT'
            ]

        # Set default websocket symbols
        if self.websocket_symbols is None:
            self.websocket_symbols = self.symbols.copy()

        # Set default channel types
        if self.channel_types is None:
            self.channel_types = ['trade', 'orderbook_1', 'ticker']

        # Set default emergency stop conditions
        if self.emergency_stop_conditions is None:
            self.emergency_stop_conditions = [
                'daily_loss_limit',
                'max_drawdown_exceeded',
                'consecutive_losses',
                'high_error_rate',
                'api_connection_lost',
                'manual_stop'
            ]

        # Validate critical settings for live trading
        self._validate_live_config()

    def _validate_live_config(self):
        """Validate configuration for live trading."""
        if self.trading_mode == TradingMode.LIVE:
            # Critical safety checks
            if self.max_risk_per_trade > 0.05:
                raise ValueError("⚠️ LIVE TRADING: max_risk_per_trade too high! Must be ≤ 5%")

            if self.max_leverage > 5.0:
                raise ValueError("⚠️ LIVE TRADING: max_leverage too high! Must be ≤ 5x")

            if self.daily_loss_limit > 0.20:
                raise ValueError("⚠️ LIVE TRADING: daily_loss_limit too high! Must be ≤ 20%")

            if self.max_drawdown > 0.30:
                raise ValueError("⚠️ LIVE TRADING: max_drawdown too high! Must be ≤ 30%")

            # Portfolio validation
            if self.portfolio_mode and len(self.symbols) < 4:
                raise ValueError("⚠️ LIVE TRADING: Need at least 4 symbols for portfolio mode")

            # Capital validation
            if self.initial_capital < 10000:
                raise ValueError("⚠️ LIVE TRADING: initial_capital too low! Must be ≥ $10,000")

            print("✅ Live trading configuration validation passed")
            print("⚠️  REMINDER: This is REAL live trading with REAL money!")

# =============================================================================
# 🚀 CONFIGURATION INSTANCES
# =============================================================================

def get_conservative_config() -> LiveTradingConfig:
    """Get conservative risk configuration for live trading."""
    config = LiveTradingConfig()

    # Conservative risk settings
    config.max_risk_per_trade = 0.01  # 1% per trade
    config.max_leverage = 2.0  # 2x max leverage
    config.default_leverage = 1.5  # 1.5x default
    config.daily_loss_limit = 0.05  # 5% daily loss limit
    config.max_drawdown = 0.15  # 15% max drawdown
    config.default_stop_loss_percent = 0.02  # 2% stop loss
    config.default_take_profit_percent = 0.04  # 4% take profit

    # Conservative allocation
    config.initial_capital = 50000.0  # $50k starting capital
    config.max_position_size = 10000.0  # $10k max per position

    return config

def get_moderate_config() -> LiveTradingConfig:
    """Get moderate risk configuration for live trading."""
    config = LiveTradingConfig()

    # Use default moderate settings
    return config

def get_aggressive_config() -> LiveTradingConfig:
    """Get aggressive risk configuration for live trading."""
    config = LiveTradingConfig()

    # Aggressive risk settings (within safe bounds)
    config.max_risk_per_trade = 0.03  # 3% per trade
    config.max_leverage = 4.0  # 4x max leverage
    config.default_leverage = 3.0  # 3x default
    config.daily_loss_limit = 0.15  # 15% daily loss limit
    config.max_drawdown = 0.25  # 25% max drawdown
    config.default_stop_loss_percent = 0.04  # 4% stop loss
    config.default_take_profit_percent = 0.08  # 8% take profit

    # Aggressive allocation
    config.initial_capital = 200000.0  # $200k starting capital
    config.max_position_size = 50000.0  # $50k max per position

    return config

def get_simulation_config() -> LiveTradingConfig:
    """Get simulation configuration for testing."""
    config = LiveTradingConfig()

    # Simulation settings
    config.trading_mode = TradingMode.SIMULATION
    config.enable_real_trading = False
    config.require_manual_confirmation = False

    # Relaxed settings for testing
    config.initial_capital = 100000.0  # $100k for testing
    config.max_risk_per_trade = 0.05  # 5% for testing
    config.daily_loss_limit = 0.30  # 30% for testing

    return config

# =============================================================================
# 🎯 DEFAULT CONFIGURATION FOR LIVE TRADING
# =============================================================================

# Default to moderate risk for live trading
DEFAULT_LIVE_CONFIG = get_moderate_config()

# =============================================================================
# ⚠️ LIVE TRADING SAFETY CHECK
# =============================================================================

def confirm_live_trading():
    """
    Interactive confirmation for live trading.

    Returns:
        bool: True if user confirms live trading, False otherwise
    """
    print("\n" + "="*80)
    print("⚠️  LIVE TRADING CONFIRMATION REQUIRED")
    print("="*80)
    print("🚨 YOU ARE ABOUT TO START LIVE TRADING WITH REAL MONEY!")
    print("🚨 THIS WILL EXECUTE REAL TRADES ON BYBIT EXCHANGE!")
    print("🚨 REAL PROFITS AND LOSSES WILL OCCUR!")
    print("\nRisks involved:")
    print("• You can lose your entire trading capital")
    print("• Market volatility can cause rapid losses")
    print("• Technical issues may affect trading")
    print("• System errors may occur")
    print("\nSafety measures in place:")
    print("• Risk limits and stop losses")
    print("• Daily loss limits")
    print("• Maximum drawdown protection")
    print("• Emergency stop functionality")
    print("="*80)

    try:
        confirmation = input("\nType 'CONFIRM LIVE TRADING' to proceed: ").strip()
        return confirmation == "CONFIRM LIVE TRADING"
    except KeyboardInterrupt:
        print("\n❌ Live trading cancelled by user")
        return False

# =============================================================================
# 📋 CONFIGURATION VALIDATION SUMMARY
# =============================================================================

def print_config_summary(config: LiveTradingConfig):
    """Print configuration summary for review."""
    print("\n📋 LIVE TRADING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"🎯 Trading Mode: {config.trading_mode.value.upper()}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"📊 Symbols: {len(config.symbols)} assets")
    print(f"🔬 Portfolio Mode: {'Enabled' if config.portfolio_mode else 'Disabled'}")
    print(f"🛡️ Max Risk per Trade: {config.max_risk_per_trade:.1%}")
    print(f"⚡ Max Leverage: {config.max_leverage}x")
    print(f"📉 Daily Loss Limit: {config.daily_loss_limit:.1%}")
    print(f"📊 Max Drawdown: {config.max_drawdown:.1%}")
    print(f"🔄 Rebalance Threshold: {config.rebalance_threshold:.1%}")
    print("="*60)

    if config.trading_mode == TradingMode.LIVE:
        print("⚠️  THIS CONFIGURATION IS FOR LIVE TRADING")
        print("⚠️  REAL MONEY WILL BE AT RISK")
        print("⚠️  ENSURE ALL SETTINGS ARE REVIEWED CAREFULLY")

    print("="*60)

if __name__ == "__main__":
    # Example usage
    config = DEFAULT_LIVE_CONFIG
    print_config_summary(config)

    # For live trading, always require confirmation
    if config.trading_mode == TradingMode.LIVE:
        if confirm_live_trading():
            print("✅ Live trading confirmed!")
        else:
            print("❌ Live trading cancelled!")
            exit(1)

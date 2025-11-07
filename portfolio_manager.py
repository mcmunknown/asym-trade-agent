"""
Portfolio Manager for Anne's Joint Distribution Calculus Trading System
====================================================================

This module implements the central portfolio coordination layer that manages
all 8 cryptocurrency assets together, providing portfolio-level oversight
while preserving Anne's single-asset calculus analysis for timing decisions.

Key Features:
- Real-time allocation tracking vs. optimal weights
- Automatic rebalancing when drift > 5%
- Portfolio-level position sizing based on optimal weights
- Multi-asset signal coordination
- Risk budget allocation across assets
- Performance attribution and monitoring

Hybrid Approach: Single-asset calculus signals for entry/exit, portfolio optimization for sizing
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from joint_distribution_analyzer import JointDistributionAnalyzer, JointDistributionStats
from portfolio_optimizer import PortfolioOptimizer, OptimizationResult
from risk_manager import RiskManager
from calculus_strategy import CalculusTradingStrategy, SignalType
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class AllocationDecision:
    """Decision for portfolio allocation"""
    symbol: str
    target_weight: float
    current_weight: float
    trade_type: str  # ENTER_LONG, INCREASE_LONG, REDUCE_LONG, etc.
    quantity: float
    reason: str
    confidence: float = 0.0
    timestamp: float = 0.0

class RebalanceType(Enum):
    """Types of portfolio rebalancing triggers"""
    DRIFT_BASED = "drift_based"  # Allocation drift > threshold
    SIGNAL_BASED = "signal_based"  # Strong multi-asset signals
    RISK_BASED = "risk_based"  # Risk regime changes
    SCHEDULED = "scheduled"  # Time-based rebalancing
    MANUAL = "manual"  # Manual rebalancing trigger

@dataclass
class PortfolioPosition:
    """Position information for a single asset in the portfolio"""
    symbol: str
    current_weight: float  # Current allocation weight
    optimal_weight: float  # Target optimal weight
    quantity: float  # Position quantity
    notional_value: float  # Position value in USD
    unrealized_pnl: float  # Unrealized P&L
    unrealized_pnl_pct: float  # Unrealized P&L percentage
    last_update: float  # Last update timestamp
    signal_strength: float  # Current calculus signal strength
    confidence: float  # Signal confidence level

@dataclass
class PortfolioMetrics:
    """Current portfolio metrics and performance"""
    total_value: float  # Total portfolio value
    invested_value: float  # Total invested amount
    cash_balance: float  # Available cash
    unrealized_pnl: float  # Total unrealized P&L
    unrealized_pnl_pct: float  # Portfolio P&L percentage
    daily_pnl: float  # Daily P&L
    allocation_drift: float  # Current drift from optimal allocation
    portfolio_volatility: float  # Portfolio volatility
    sharpe_ratio: float  # Portfolio Sharpe ratio
    max_drawdown: float  # Maximum drawdown
    risk_budget_utilization: float  # Risk budget usage
    concentration_ratio: float  # Top asset concentration
    last_rebalance: float  # Last rebalance timestamp
    rebalance_count: int  # Number of rebalances

class PortfolioManager:
    """
    Central portfolio management system that coordinates all 8 assets.

    This manager implements the hybrid approach:
    - Uses Anne's single-asset calculus analysis for entry/exit timing
    - Uses portfolio optimization for position sizing and allocation
    - Maintains portfolio-level risk management
    - Handles automatic rebalancing when allocation drifts > 5%
    """

    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 target_allocation: float = 0.95,  # 95% invested, 5% cash
                 rebalance_threshold: float = 0.05,  # 5% drift threshold
                 min_position_size: float = 0.01,  # 1% minimum position
                 max_position_size: float = 0.30,  # 30% maximum position
                 joint_analyzer: Optional[JointDistributionAnalyzer] = None,
                 portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                 risk_manager: Optional[RiskManager] = None):
        """
        Initialize portfolio manager.

        Args:
            symbols: List of trading symbols (8 crypto assets)
            initial_capital: Initial portfolio capital
            target_allocation: Target allocation percentage (rest is cash)
            rebalance_threshold: Allocation drift threshold for rebalancing
            min_position_size: Minimum position size as percentage
            max_position_size: Maximum position size as percentage
            joint_analyzer: Joint distribution analyzer instance
            portfolio_optimizer: Portfolio optimizer instance
            risk_manager: Risk manager instance
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.target_allocation = target_allocation
        self.rebalance_threshold = rebalance_threshold
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size

        # Component instances
        self.joint_analyzer = joint_analyzer or JointDistributionAnalyzer(num_assets=len(symbols))
        self.portfolio_optimizer = portfolio_optimizer or PortfolioOptimizer(self.joint_analyzer)
        self.risk_manager = risk_manager or RiskManager()

        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash_balance = initial_capital
        self.total_value = initial_capital
        self.invested_value = 0.0

        # Optimization state
        self.optimal_weights: Dict[str, float] = {}
        self.last_optimization_time = 0
        self.last_rebalance_time = 0
        self.rebalance_count = 0

        # Performance tracking
        self.daily_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_history = []
        self.portfolio_value_history = []
        self.rebalance_history = []

        # Initialize positions
        self._initialize_positions()

        logger.info(f"Portfolio Manager initialized: {len(symbols)} assets, "
                   f"initial_capital=${initial_capital:,.0f}, "
                   f"rebalance_threshold={rebalance_threshold:.1%}")

    def _initialize_positions(self):
        """Initialize portfolio positions for all symbols."""
        for symbol in self.symbols:
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                current_weight=0.0,
                optimal_weight=0.0,
                quantity=0.0,
                notional_value=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                last_update=0,
                signal_strength=0.0,
                confidence=0.0
            )

        logger.info(f"Initialized positions for {len(self.positions)} symbols")

    def update_market_data(self, symbol: str, price: float, signal_strength: float, confidence: float):
        """
        Update market data and calculus signals for a symbol.

        Args:
            symbol: Asset symbol
            price: Current market price
            signal_strength: Calculus signal strength
            confidence: Signal confidence level
        """
        if symbol not in self.positions:
            logger.warning(f"Symbol {symbol} not in portfolio positions")
            return

        position = self.positions[symbol]
        old_value = position.notional_value

        # Update position values
        position.notional_value = position.quantity * price
        position.signal_strength = signal_strength
        position.confidence = confidence
        position.last_update = time.time()

        # Calculate unrealized P&L
        if position.quantity > 0:
            avg_cost = self.invested_value / sum(pos.quantity for pos in self.positions.values() if pos.quantity > 0) if self.invested_value > 0 else price
            position.unrealized_pnl = position.notional_value - (position.quantity * avg_cost)
            position.unrealized_pnl_pct = position.unrealized_pnl / (position.quantity * avg_cost) if position.quantity * avg_cost > 0 else 0

        # Update portfolio totals
        self._update_portfolio_totals()

        # Log significant changes
        if abs(position.unrealized_pnl) > 1000:  # > $1000 change
            logger.info(f"{symbol} P&L: ${position.unrealized_pnl:,.0f} ({position.unrealized_pnl_pct:.1%})")

    def _update_portfolio_totals(self):
        """Update total portfolio values and metrics."""
        total_invested = sum(pos.notional_value for pos in self.positions.values())
        self.invested_value = total_invested
        self.total_value = self.cash_balance + total_invested
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        # Update position weights
        for symbol, position in self.positions.items():
            position.current_weight = position.notional_value / self.total_value if self.total_value > 0 else 0

        # Track portfolio value history
        self.portfolio_value_history.append({
            'timestamp': time.time(),
            'value': self.total_value,
            'invested': self.invested_value,
            'cash': self.cash_balance,
            'unrealized_pnl': self.unrealized_pnl
        })

        # Keep last 1000 records
        if len(self.portfolio_value_history) > 1000:
            self.portfolio_value_history = self.portfolio_value_history[-1000:]

    def update_optimal_weights(self, optimal_weights: Dict[str, float]):
        """
        Update optimal portfolio weights from portfolio optimizer.

        Args:
            optimal_weights: Dictionary of optimal weights for each symbol
        """
        # Normalize weights
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}

        # Apply position size limits
        for symbol in self.symbols:
            weight = optimal_weights.get(symbol, 0.0)
            weight = np.clip(weight, self.min_position_size, self.max_position_size)
            optimal_weights[symbol] = weight

        # Re-normalize after applying limits
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}

        self.optimal_weights = optimal_weights
        self.last_optimization_time = time.time()

        # Update position optimal weights
        for symbol, position in self.positions.items():
            position.optimal_weight = optimal_weights.get(symbol, 0.0)

        logger.info(f"Updated optimal weights: {dict(list(optimal_weights.items())[:5])}")

    def calculate_allocation_drift(self) -> float:
        """
        Calculate current allocation drift from optimal weights.

        Returns:
            Allocation drift as percentage (0-1)
        """
        if not self.optimal_weights:
            return 0.0

        drift = 0.0
        for symbol, position in self.positions.items():
            if symbol in self.optimal_weights:
                weight_diff = abs(position.current_weight - position.optimal_weight)
                drift += weight_diff

        return drift / 2  # Divide by 2 since we're measuring absolute differences

    def should_rebalance(self) -> Tuple[bool, RebalanceType, str]:
        """
        Check if portfolio should be rebalanced.

        Returns:
            Tuple of (should_rebalance, rebalance_type, reason)
        """
        # Check allocation drift
        allocation_drift = self.calculate_allocation_drift()
        if allocation_drift > self.rebalance_threshold:
            return True, RebalanceType.DRIFT_BASED, f"Allocation drift {allocation_drift:.1%} > threshold {self.rebalance_threshold:.1%}"

        # Check for strong multi-asset signals
        strong_signals = 0
        for position in self.positions.values():
            if position.signal_strength > 1.5 and position.confidence > 0.8:
                strong_signals += 1

        if strong_signals >= 3:  # 3+ strong signals
            return True, RebalanceType.SIGNAL_BASED, f"Strong signals in {strong_signals} assets"

        # Check time-based rebalancing (every 4 hours)
        time_since_rebalance = time.time() - self.last_rebalance_time
        if time_since_rebalance > 14400:  # 4 hours
            return True, RebalanceType.SCHEDULED, "Scheduled rebalancing (4 hours)"

        return False, RebalanceType.SCHEDULED, "No rebalancing needed"

    def calculate_rebalance_trades(self) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to optimal weights.

        Returns:
            Dictionary of symbol -> target weight change
        """
        target_value = self.total_value * self.target_allocation
        target_cash = self.total_value * (1 - self.target_allocation)

        rebalance_trades = {}

        for symbol, position in self.positions.items():
            if symbol in self.optimal_weights:
                target_weight = self.optimal_weights[symbol]
                target_value_for_symbol = target_value * target_weight
                current_value = position.notional_value

                value_change = target_value_for_symbol - current_value

                # Apply minimum trade size
                if abs(value_change) < self.total_value * self.min_position_size:
                    rebalance_trades[symbol] = 0.0
                else:
                    rebalance_trades[symbol] = value_change

        # Check cash balance changes
        total_value_change = sum(rebalance_trades.values())
        cash_change = -total_value_change

        if abs(cash_change) > self.total_value * 0.01:  # Only if cash change > 1%
            rebalance_trades['CASH'] = cash_change

        return rebalance_trades

    def calculate_position_size(self, symbol: str, signal_strength: float, confidence: float) -> float:
        """
        Calculate optimal position size for a symbol using portfolio optimization.

        Args:
            symbol: Asset symbol
            signal_strength: Calculus signal strength
            confidence: Signal confidence level

        Returns:
            Optimal position size in USD
        """
        if symbol not in self.optimal_weights:
            return 0.0

        # Base allocation from optimal weights
        target_value = self.total_value * self.target_allocation * self.optimal_weights[symbol]

        # Adjust for signal strength and confidence
        signal_multiplier = min(2.0, max(0.5, signal_strength))
        confidence_multiplier = confidence

        adjusted_value = target_value * signal_multiplier * confidence_multiplier

        # Apply position size limits
        min_value = self.total_value * self.min_position_size
        max_value = self.total_value * self.max_position_size

        final_value = np.clip(adjusted_value, min_value, max_value)

        # Check available cash
        available_for_investment = min(final_value, self.cash_balance * 0.9)  # Keep 10% cash buffer

        return available_for_investment

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get comprehensive portfolio metrics.

        Returns:
            PortfolioMetrics object with current metrics
        """
        # Calculate allocation drift
        allocation_drift = self.calculate_allocation_drift()

        # Calculate concentration ratio (top 3 assets)
        sorted_weights = sorted([pos.current_weight for pos in self.positions.values()], reverse=True)
        concentration_ratio = sum(sorted_weights[:3]) if sorted_weights else 0

        # Calculate daily P&L
        daily_pnl = 0.0
        if len(self.portfolio_value_history) >= 2:
            current_value = self.portfolio_value_history[-1]['value']
            previous_value = self.portfolio_value_history[-2]['value']
            daily_pnl = current_value - previous_value

        # Calculate portfolio volatility (simplified)
        portfolio_volatility = 0.02  # 2% daily (placeholder)
        if len(self.portfolio_value_history) > 30:
            returns = []
            for i in range(1, min(30, len(self.portfolio_value_history))):
                prev_value = self.portfolio_value_history[-i-1]['value']
                curr_value = self.portfolio_value_history[-i]['value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            if returns:
                portfolio_volatility = np.std(returns)

        # Calculate Sharpe ratio
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_return = daily_pnl / self.total_value if self.total_value > 0 else 0
        sharpe_ratio = excess_return / (portfolio_volatility + 1e-8) if portfolio_volatility > 0 else 0

        # Calculate max drawdown
        max_drawdown = 0.0
        if len(self.portfolio_value_history) > 10:
            peak = max(entry['value'] for entry in self.portfolio_value_history)
            current = self.total_value
            if peak > 0:
                max_drawdown = (peak - current) / peak

        # Calculate risk budget utilization (simplified)
        risk_budget_utilization = self.invested_value / self.total_value if self.total_value > 0 else 0

        return PortfolioMetrics(
            total_value=self.total_value,
            invested_value=self.invested_value,
            cash_balance=self.cash_balance,
            unrealized_pnl=self.unrealized_pnl,
            unrealized_pnl_pct=self.unrealized_pnl / self.invested_value if self.invested_value > 0 else 0,
            daily_pnl=daily_pnl,
            allocation_drift=allocation_drift,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            risk_budget_utilization=risk_budget_utilization,
            concentration_ratio=concentration_ratio,
            last_rebalance=self.last_rebalance_time,
            rebalance_count=self.rebalance_count
        )

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio summary
        """
        metrics = self.get_portfolio_metrics()

        summary = {
            'portfolio_metrics': {
                'total_value': metrics.total_value,
                'invested_value': metrics.invested_value,
                'cash_balance': metrics.cash_balance,
                'unrealized_pnl': metrics.unrealized_pnl,
                'unrealized_pnl_pct': metrics.unrealized_pnl_pct,
                'daily_pnl': metrics.daily_pnl,
                'allocation_drift': metrics.allocation_drift,
                'portfolio_volatility': metrics.portfolio_volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            },
            'current_positions': {},
            'optimal_weights': self.optimal_weights.copy(),
            'rebalance_status': {
                'should_rebalance': self.should_rebalance()[0],
                'last_rebalance': self.last_rebalance_time,
                'rebalance_count': self.rebalance_count,
                'optimization_frequency': 1800,  # 30 minutes
                'drift_threshold': self.rebalance_threshold
            }
        }

        # Add current positions
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                summary['current_positions'][symbol] = {
                    'weight': position.current_weight,
                    'optimal_weight': position.optimal_weight,
                    'notional_value': position.notional_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'signal_strength': position.signal_strength,
                    'confidence': position.confidence
                }

        return summary

    def log_portfolio_status(self):
        """Log current portfolio status."""
        metrics = self.get_portfolio_metrics()

        logger.info(f"📊 Portfolio Status:")
        logger.info(f"   Total Value: ${metrics.total_value:,.0f}")
        logger.info(f"   Invested: ${metrics.invested_value:,.0f} ({metrics.invested_value/metrics.total_value:.1%})")
        logger.info(f"   Cash: ${metrics.cash_balance:,.0f}")
        logger.info(f"   P&L: ${metrics.unrealized_pnl:,.0f} ({metrics.unrealized_pnl_pct:.1%})")
        logger.info(f"   Allocation Drift: {metrics.allocation_drift:.1%}")
        logger.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        logger.info(f"   Concentration: {metrics.concentration_ratio:.1%}")

        should_rebalance, rebalance_type, reason = self.should_rebalance()
        if should_rebalance:
            logger.info(f"   🔄 REBALANCE NEEDED: {rebalance_type.value} - {reason}")
        else:
            logger.info(f"   ✅ No rebalancing needed: {reason}")

# Example usage
if __name__ == "__main__":
    # Create portfolio manager
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT']
    portfolio_manager = PortfolioManager(symbols, initial_capital=100000)

    # Simulate market updates
    for symbol in symbols:
        portfolio_manager.update_market_data(symbol, 50000, 1.2, 0.8)

    # Get portfolio summary
    summary = portfolio_manager.get_portfolio_summary()
    print("Portfolio Summary:")
    print(f"Total Value: ${summary['portfolio_metrics']['total_value']:,.0f}")
    print(f"Invested: ${summary['portfolio_metrics']['invested_value']:,.0f}")
    print(f"Cash: ${summary['portfolio_metrics']['cash_balance']:,.0f}")
    print(f"P&L: ${summary['portfolio_metrics']['unrealized_pnl']:,.0f}")
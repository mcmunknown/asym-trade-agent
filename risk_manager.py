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
from config import Config
from calculus_strategy import SignalType
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

        # Volatility tracking
        self.volatility_window = 20
        self.price_history = {}

        logger.info(f"Risk manager initialized: max_risk_per_trade={max_risk_per_trade:.1%}, "
                   f"max_leverage={max_leverage}x, min_risk_reward={min_risk_reward:.1f}")

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
            # Base position size on signal strength and confidence
            combined_strength = signal_strength * confidence

            # Risk adjustment based on signal strength
            risk_adjustment = min(combined_strength / 2.0, 1.0)  # Cap at 100% of max risk
            base_risk_percent = self.max_risk_per_trade * risk_adjustment

            # Volatility adjustment (higher volatility = smaller position)
            if volatility is not None:
                volatility_adjustment = 1.0 / (1.0 + volatility * 10)
                base_risk_percent *= volatility_adjustment

            # Calculate risk amount
            risk_amount = account_balance * base_risk_percent

            # Determine position size (using 2% risk as standard SL distance)
            standard_stop_distance = 0.02  # 2% standard stop loss
            position_value = risk_amount / standard_stop_distance

            # Calculate quantity
            quantity = position_value / current_price

            # Apply leverage (conservative by default, increase with signal strength)
            leverage = min(5.0 + combined_strength * 10, self.max_leverage)
            leveraged_quantity = quantity * leverage

            # Calculate margin requirement
            margin_required = position_value / leverage

            return PositionSize(
                quantity=leveraged_quantity,
                notional_value=position_value,
                risk_amount=risk_amount,
                leverage_used=leverage,
                margin_required=margin_required,
                risk_percent=base_risk_percent,
                confidence_score=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return PositionSize(0, 0, 0, 1, 0, 0, 0)

    def calculate_dynamic_tp_sl(self,
                               signal_type: SignalType,
                               current_price: float,
                               velocity: float,
                               acceleration: float,
                               volatility: float,
                               atr: float = None) -> TradingLevels:
        """
        Calculate dynamic TP/SL levels using calculus indicators.

        Following Anne's approach:
        - Use velocity and acceleration to determine momentum
        - Adjust levels based on signal strength and market volatility
        - Implement risk-reward optimization

        Args:
            signal_type: Type of trading signal
            current_price: Current market price
            velocity: Current price velocity
            acceleration: Current price acceleration
            volatility: Current market volatility
            atr: Average True Range (optional)

        Returns:
            TradingLevels with TP/SL calculations
        """
        try:
            # Determine position side
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]:
                position_side = "long"
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                position_side = "short"
            else:
                # Default to based on velocity
                position_side = "long" if velocity > 0 else "short"

            # Calculate base stop loss using ATR or volatility
            if atr is not None:
                base_stop_distance = atr * 2.0  # 2x ATR for stop loss
            else:
                base_stop_distance = current_price * volatility * 2.0

            # Adjust stop distance based on signal confidence
            confidence_adjustment = 1.0  # Will be adjusted below

            # Dynamic adjustments based on calculus indicators
            velocity_strength = abs(velocity)
            acceleration_strength = abs(acceleration)

            if position_side == "long":
                # Long position calculations
                stop_loss = current_price - base_stop_distance

                # Adjust stop loss based on acceleration
                if acceleration > 0:  # Accelerating uptrend
                    stop_loss = current_price - base_stop_distance * 0.8  # Tighter stop
                    confidence_adjustment = 1.2
                elif acceleration < 0:  # Decelerating uptrend
                    stop_loss = current_price - base_stop_distance * 1.2  # Wider stop
                    confidence_adjustment = 0.8

                # Calculate take profit based on risk-reward ratio
                risk_amount = current_price - stop_loss
                take_profit = current_price + risk_amount * self.min_risk_reward

                # Adjust take profit based on velocity
                if velocity_strength > volatility:  # Strong momentum
                    take_profit = current_price + risk_amount * (self.min_risk_reward + 0.5)

                # Trail stop for accelerating trends
                trail_stop = None
                if signal_type == SignalType.TRAIL_STOP_UP:
                    trail_stop = stop_loss

            else:  # short position
                # Short position calculations
                stop_loss = current_price + base_stop_distance

                # Adjust stop loss based on acceleration
                if acceleration < 0:  # Accelerating downtrend
                    stop_loss = current_price + base_stop_distance * 0.8  # Tighter stop
                    confidence_adjustment = 1.2
                elif acceleration > 0:  # Decelerating downtrend
                    stop_loss = current_price + base_stop_distance * 1.2  # Wider stop
                    confidence_adjustment = 0.8

                # Calculate take profit based on risk-reward ratio
                risk_amount = stop_loss - current_price
                take_profit = current_price - risk_amount * self.min_risk_reward

                # Adjust take profit based on velocity
                if velocity_strength > volatility:  # Strong momentum
                    take_profit = current_price - risk_amount * (self.min_risk_reward + 0.5)

                # Trail stop for accelerating trends
                trail_stop = None
                if signal_type == SignalType.HOLD_SHORT:
                    trail_stop = stop_loss

            # Calculate risk-reward ratio
            if position_side == "long":
                risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)
            else:
                risk_reward_ratio = (current_price - take_profit) / (stop_loss - current_price)

            # Confidence level based on signal strength and indicators
            confidence_level = min(1.0, (velocity_strength + acceleration_strength) / (2 * volatility))

            return TradingLevels(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trail_stop=trail_stop,
                risk_reward_ratio=risk_reward_ratio,
                position_side=position_side,
                confidence_level=confidence_level * confidence_adjustment
            )

        except Exception as e:
            logger.error(f"Error calculating TP/SL levels: {e}")
            return TradingLevels(current_price, current_price, current_price, None, 1.0, "long", 0.0)

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
            # Check risk-reward ratio
            if trading_levels.risk_reward_ratio < self.min_risk_reward:
                return False, f"Risk-reward ratio {trading_levels.risk_reward_ratio:.2f} below minimum {self.min_risk_reward}"

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

    def update_position(self, symbol: str, position_info: Dict):
        """
        Update position tracking.

        Args:
            symbol: Trading symbol
            position_info: Position information
        """
        self.open_positions[symbol] = position_info
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
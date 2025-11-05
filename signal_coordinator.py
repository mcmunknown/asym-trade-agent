"""
Signal Coordinator for Anne's Multi-Asset Calculus Trading System
=============================================================

This module coordinates signals from all 8 assets in Anne's calculus trading system
for live portfolio management, ensuring portfolio-level consistency and optimal
decision making across multiple cryptocurrency assets.

Key Features:
- Collects single-asset calculus signals from all assets
- Ranks signals by confidence and portfolio weight importance
- Validates portfolio-level signal consistency
- Coordinates execution timing across assets
- Prevents over-concentration and maintains diversification
- Integrates with portfolio optimization for sizing decisions

Hybrid Approach: Uses Anne's single-asset calculus analysis for timing,
portfolio optimization for allocation and risk management.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import time

from calculus_strategy import CalculusTradingStrategy, SignalType
from portfolio_manager import PortfolioManager
from config import Config

logger = logging.getLogger(__name__)

class SignalPriority(Enum):
    """Signal priority levels for portfolio coordination"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SignalValidation(Enum):
    """Signal validation results"""
    VALID = "valid"
    INVALID_RISK = "invalid_risk"
    INVALID_CORRELATION = "invalid_correlation"
    INVALID_POSITION_SIZE = "invalid_position_size"
    INVALID_TIMING = "invalid_timing"

@dataclass
class CoordinatedSignal:
    """Coordinated signal for a single asset in portfolio context"""
    symbol: str
    signal_type: SignalType
    priority: SignalPriority
    confidence: float
    signal_strength: float
    optimal_weight: float  # From portfolio optimizer
    recommended_size: float  # Portfolio-optimal position size
    validation_result: SignalValidation
    validation_reason: str
    timestamp: float
    portfolio_impact: float  # Expected impact on portfolio

class SignalCoordinator:
    """
    Coordinates signals across all 8 assets for portfolio-level trading decisions.

    This coordinator implements the hybrid approach:
    - Uses Anne's single-asset calculus signals for entry/exit timing
    - Applies portfolio-level validation and risk management
    - Coordinates execution to maintain optimal allocation
    - Ensures diversification and concentration controls
    """

    def __init__(self,
                 symbols: List[str],
                 portfolio_manager: PortfolioManager,
                 min_signal_interval: int = 30,  # Minimum seconds between signals per asset
                 max_concurrent_signals: int = 3,  # Max assets to trade simultaneously
                 correlation_threshold: float = 0.7,  # Max correlation between concurrent trades
                 concentration_limit: float = 0.4):  # Max allocation to single asset
        """
        Initialize signal coordinator.

        Args:
            symbols: List of trading symbols (8 crypto assets)
            portfolio_manager: Portfolio manager instance
            min_signal_interval: Minimum seconds between signals for same asset
            max_concurrent_signals: Maximum simultaneous asset trades
            correlation_threshold: Maximum correlation between concurrent trades
            concentration_limit: Maximum allocation to single asset
        """
        self.symbols = symbols
        self.portfolio_manager = portfolio_manager
        self.min_signal_interval = min_signal_interval
        self.max_concurrent_signals = max_concurrent_signals
        self.correlation_threshold = correlation_threshold
        self.concentration_limit = concentration_limit

        # Signal storage and tracking
        self.recent_signals: Dict[str, float] = {}  # symbol -> last signal time
        self.active_signals: Dict[str, CoordinatedSignal] = {}
        self.signal_history: List[CoordinatedSignal] = []

        # Signal analysis
        self.correlation_matrix = np.eye(len(symbols))  # Will be updated
        self.last_correlation_update = 0
        self.correlation_update_frequency = 300  # Update every 5 minutes

        # Signal statistics
        self.stats = {
            'total_signals_generated': 0,
            'signals_executed': 0,
            'signals_rejected': 0,
            'concurrent_trades': 0,
            'average_confidence': 0.0,
            'portfolio_optimization_saves': 0
        }

        logger.info(f"Signal Coordinator initialized: {len(symbols)} assets, "
                   f"max_concurrent={max_concurrent_signals}, "
                   f"correlation_threshold={correlation_threshold:.1f}")

    def process_signal(self,
                      symbol: str,
                      signal_type: SignalType,
                      confidence: float,
                      signal_strength: float,
                      price: float,
                      volatility: float = None) -> Optional[CoordinatedSignal]:
        """
        Process a single-asset calculus signal and coordinate it at portfolio level.

        Args:
            symbol: Asset symbol
            signal_type: Calculus signal type
            confidence: Signal confidence (0-1)
            signal_strength: Signal strength from calculus analysis
            price: Current market price
            volatility: Current volatility (optional)

        Returns:
            CoordinatedSignal if signal is valid for portfolio execution, None otherwise
        """
        current_time = time.time()

        # Check minimum signal interval
        if symbol in self.recent_signals:
            time_since_last = current_time - self.recent_signals[symbol]
            if time_since_last < self.min_signal_interval:
                logger.debug(f"Signal for {symbol} rejected: too frequent ({time_since_last:.1f}s < {self.min_signal_interval}s)")
                return None

        # Get portfolio context
        optimal_weights = self.portfolio_manager.optimal_weights
        if not optimal_weights:
            logger.warning(f"No optimal weights available for {symbol}")
            return None

        optimal_weight = optimal_weights.get(symbol, 0.0)
        if optimal_weight < 0.01:  # Less than 1% allocation
            logger.debug(f"Signal for {symbol} rejected: minimal optimal weight ({optimal_weight:.1%})")
            return None

        # Calculate signal priority
        priority = self._calculate_signal_priority(signal_type, confidence, signal_strength)

        # Calculate recommended position size
        recommended_size = self.portfolio_manager.calculate_position_size(symbol, signal_strength, confidence)

        # Validate signal at portfolio level
        validation_result, validation_reason = self._validate_signal_portfolio_level(
            symbol, signal_type, optimal_weight, recommended_size
        )

        if validation_result != SignalValidation.VALID:
            logger.info(f"Signal for {symbol} rejected: {validation_reason}")
            self.stats['signals_rejected'] += 1
            return None

        # Check for concurrent signal conflicts
        if not self._check_concurrent_signal_compatibility(symbol, optimal_weight):
            logger.info(f"Signal for {symbol} rejected: concurrent signal conflicts")
            self.stats['signals_rejected'] += 1
            return None

        # Create coordinated signal
        coordinated_signal = CoordinatedSignal(
            symbol=symbol,
            signal_type=signal_type,
            priority=priority,
            confidence=confidence,
            signal_strength=signal_strength,
            optimal_weight=optimal_weight,
            recommended_size=recommended_size,
            validation_result=validation_result,
            validation_reason=validation_reason,
            timestamp=current_time,
            portfolio_impact=self._calculate_portfolio_impact(symbol, optimal_weight, recommended_size)
        )

        # Store signal
        self.recent_signals[symbol] = current_time
        self.active_signals[symbol] = coordinated_signal
        self.signal_history.append(coordinated_signal)

        # Update statistics
        self.stats['total_signals_generated'] += 1
        self._update_signal_statistics()

        logger.info(f"✅ Coordinated signal generated for {symbol}: {signal_type.name} "
                   f"(priority={priority.name}, confidence={confidence:.2f}, "
                   f"size=${recommended_size:,.0f})")

        return coordinated_signal

    def _calculate_signal_priority(self, signal_type: SignalType, confidence: float, signal_strength: float) -> SignalPriority:
        """
        Calculate signal priority based on signal type and metrics.

        Args:
            signal_type: Type of calculus signal
            confidence: Signal confidence level
            signal_strength: Signal strength from calculus analysis

        Returns:
            Signal priority level
        """
        # Base priority by signal type
        type_priority = {
            SignalType.STRONG_BUY: SignalPriority.CRITICAL,
            SignalType.STRONG_SELL: SignalPriority.CRITICAL,
            SignalType.BUY: SignalPriority.HIGH,
            SignalType.SELL: SignalPriority.HIGH,
            SignalType.TRAIL_STOP_UP: SignalPriority.MEDIUM,
            SignalType.TAKE_PROFIT: SignalPriority.MEDIUM,
            SignalType.HOLD_SHORT: SignalPriority.MEDIUM,
            SignalType.LOOK_FOR_REVERSAL: SignalPriority.LOW,
            SignalType.POSSIBLE_LONG: SignalPriority.LOW,
            SignalType.POSSIBLE_EXIT_SHORT: SignalPriority.LOW,
            SignalType.NEUTRAL: SignalPriority.LOW,
        }.get(signal_type, SignalPriority.LOW)

        # Adjust based on confidence and strength
        confidence_multiplier = confidence
        strength_multiplier = min(2.0, signal_strength)

        # Combined priority score
        priority_score = (type_priority.value * confidence_multiplier * strength_multiplier)

        # Convert to priority level
        if priority_score >= 6:
            return SignalPriority.CRITICAL
        elif priority_score >= 3:
            return SignalPriority.HIGH
        elif priority_score >= 1.5:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW

    def _validate_signal_portfolio_level(self,
                                        symbol: str,
                                        signal_type: SignalType,
                                        optimal_weight: float,
                                        recommended_size: float) -> Tuple[SignalValidation, str]:
        """
        Validate signal at portfolio level.

        Args:
            symbol: Asset symbol
            signal_type: Signal type
            optimal_weight: Optimal portfolio weight
            recommended_size: Recommended position size

        Returns:
            Tuple of (validation_result, reason)
        """
        # Check concentration limit
        if optimal_weight > self.concentration_limit:
            return SignalValidation.INVALID_POSITION_SIZE, f"Weight {optimal_weight:.1%} exceeds concentration limit {self.concentration_limit:.1%}"

        # Check minimum position size
        portfolio_value = self.portfolio_manager.total_value
        min_position_value = portfolio_value * 0.005  # 0.5% minimum

        if recommended_size < min_position_value:
            return SignalValidation.INVALID_POSITION_SIZE, f"Size ${recommended_size:,.0f} below minimum ${min_position_value:,.0f}"

        # Check maximum position size
        max_position_value = portfolio_value * 0.2  # 20% maximum per trade
        if recommended_size > max_position_value:
            return SignalValidation.INVALID_POSITION_SIZE, f"Size ${recommended_size:,.0f} exceeds maximum ${max_position_value:,.0f}"

        # Risk-based validation for certain signal types
        if signal_type in [SignalType.SELL, SignalType.HOLD_SHORT]:
            # Additional checks for short/hold signals
            current_position = self.portfolio_manager.positions.get(symbol)
            if not current_position or current_position.quantity <= 0:
                return SignalValidation.INVALID_RISK, f"Short signal {signal_type.name} without long position"

        return SignalValidation.VALID, "Signal passes portfolio validation"

    def _check_concurrent_signal_compatibility(self, symbol: str, optimal_weight: float) -> bool:
        """
        Check if signal is compatible with current active signals.

        Args:
            symbol: Asset symbol
            optimal_weight: Optimal portfolio weight

        Returns:
            True if signal is compatible with active signals
        """
        # Remove expired signals
        current_time = time.time()
        expired_signals = [s for s in self.active_signals.keys()
                           if current_time - self.active_signals[s].timestamp > 300]  # 5 minutes
        for s in expired_signals:
            del self.active_signals[s]

        # Check concurrent signal limit
        if len(self.active_signals) >= self.max_concurrent_signals:
            logger.info(f"Maximum concurrent signals ({self.max_concurrent_signals}) reached")
            return False

        # Check correlation with active signals
        for active_symbol, active_signal in self.active_signals.items():
            if active_symbol != symbol:
                # Simplified correlation check - in practice would use real correlation matrix
                # High correlation assets (BTC/ETH, similar categories)
                correlation_groups = {
                    'BTCUSDT': ['ETHUSDT'],
                    'ETHUSDT': ['BTCUSDT'],
                    'SOLUSDT': ['AVAXUSDT'],
                    'AVAXUSDT': ['SOLUSDT'],
                    'BNBUSDT': [],
                    'ADAUSDT': [],
                    'LINKUSDT': [],
                    'LTCUSDT': []
                }

                if active_symbol in correlation_groups.get(symbol, []):
                    # Check if combined allocation exceeds threshold
                    combined_weight = optimal_weight + active_signal.optimal_weight
                    if combined_weight > 0.4:  # 40% concentration for correlated assets
                        return False

        return True

    def _calculate_portfolio_impact(self, symbol: str, optimal_weight: float, recommended_size: float) -> float:
        """
        Calculate expected impact of signal on portfolio.

        Args:
            symbol: Asset symbol
            optimal_weight: Optimal portfolio weight
            recommended_size: Recommended position size

        Returns:
            Expected portfolio impact score (0-1)
        """
        portfolio_value = self.portfolio_manager.total_value
        if portfolio_value <= 0:
            return 0.0

        # Impact based on deviation from optimal allocation
        current_position = self.portfolio_manager.positions.get(symbol)
        current_value = current_position.notional_value if current_position else 0.0
        target_value = portfolio_value * self.portfolio_manager.target_allocation * optimal_weight

        # Allocation deviation
        allocation_deviation = abs(recommended_size - (target_value - current_value)) / portfolio_value

        # Normalize to 0-1 scale (10% deviation = 1.0 impact)
        impact_score = min(1.0, allocation_deviation / 0.10)

        return impact_score

    def get_active_signals(self) -> Dict[str, CoordinatedSignal]:
        """
        Get current active signals.

        Returns:
            Dictionary of symbol -> CoordinatedSignal
        """
        # Clean expired signals
        current_time = time.time()
        self.active_signals = {
            symbol: signal for symbol, signal in self.active_signals.items()
            if current_time - signal.timestamp <= 300  # 5 minutes
        }

        return self.active_signals.copy()

    def get_signal_recommendations(self) -> List[CoordinatedSignal]:
        """
        Get prioritized signal recommendations for execution.

        Returns:
            List of coordinated signals sorted by priority and impact
        """
        active_signals = self.get_active_signals()

        if not active_signals:
            return []

        # Sort by priority and portfolio impact
        sorted_signals = sorted(
            active_signals.values(),
            key=lambda x: (x.priority.value, x.portfolio_impact),
            reverse=True
        )

        # Apply concurrent signal limit
        return sorted_signals[:self.max_concurrent_signals]

    def update_correlation_matrix(self, correlation_matrix: np.ndarray):
        """
        Update correlation matrix for signal validation.

        Args:
            correlation_matrix: N x N correlation matrix
        """
        self.correlation_matrix = correlation_matrix
        self.last_correlation_update = time.time()
        logger.debug("Updated correlation matrix for signal validation")

    def _update_signal_statistics(self):
        """Update signal statistics."""
        if self.signal_history:
            recent_signals = self.signal_history[-100:]  # Last 100 signals
            if recent_signals:
                avg_confidence = np.mean([s.confidence for s in recent_signals])
                self.stats['average_confidence'] = avg_confidence

    def get_signal_statistics(self) -> Dict:
        """
        Get signal coordinator statistics.

        Returns:
            Dictionary with signal statistics
        """
        active_count = len(self.active_signals)
        recent_signals = self.signal_history[-50:]  # Last 50 signals

        signal_type_counts = {}
        for signal in recent_signals:
            signal_type = signal.signal_type.name
            signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1

        return {
            'total_signals': self.stats['total_signals_generated'],
            'signals_executed': self.stats['signals_executed'],
            'signals_rejected': self.stats['signals_rejected'],
            'active_signals': active_count,
            'average_confidence': self.stats['average_confidence'],
            'concurrent_capacity': f"{active_count}/{self.max_concurrent_signals}",
            'signal_type_distribution': signal_type_counts,
            'last_correlation_update': self.last_correlation_update,
            'signals_per_hour': len(recent_signals) / max(1, (time.time() - max([s.timestamp for s in recent_signals], time.time())) / 3600)
        }

    def log_signal_status(self):
        """Log current signal status."""
        active_signals = self.get_active_signals()
        recommendations = self.get_signal_recommendations()

        logger.info(f"🎯 Signal Coordinator Status:")
        logger.info(f"   Active Signals: {len(active_signals)}")
        logger.info(f"   Recommendations: {len(recommendations)}")

        if recommendations:
            logger.info(f"   Top Recommendations:")
            for i, signal in enumerate(recommendations[:3]):
                logger.info(f"     {i+1}. {signal.symbol}: {signal.signal_type.name} "
                           f"(priority={signal.priority.name}, "
                           f"confidence={signal.confidence:.2f}, "
                           f"size=${signal.recommended_size:,.0f})")

        stats = self.get_signal_statistics()
        logger.info(f"   Statistics: {stats['signals_executed']} executed, "
                   f"{stats['signals_rejected']} rejected, "
                   f"avg_confidence={stats['average_confidence']:.2f}")

# Example usage
if __name__ == "__main__":
    # Create signal coordinator
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT']

    # Mock portfolio manager (would be real instance)
    class MockPortfolioManager:
        def __init__(self):
            self.total_value = 100000
            self.optimal_weights = {s: 0.125 for s in symbols}
            self.positions = {s: None for s in symbols}
        def calculate_position_size(self, symbol, strength, confidence):
            return self.total_value * 0.125 * strength * confidence

    portfolio_manager = MockPortfolioManager()
    coordinator = SignalCoordinator(symbols, portfolio_manager)

    # Process some example signals
    for symbol in symbols[:3]:
        signal = coordinator.process_signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=0.8,
            signal_strength=1.2,
            price=50000
        )
        if signal:
            print(f"Signal for {symbol}: {signal.signal_type.name} - ${signal.recommended_size:,.0f}")

    print(f"\nSignal Statistics: {coordinator.get_signal_statistics()}")
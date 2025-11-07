"""
Portfolio Optimizer for Anne's Calculus Trading System
======================================================

This module implements sophisticated portfolio optimization strategies that integrate
with Anne's calculus-based signals and the joint distribution analysis.

Mathematical Framework:
- Mean-variance optimization using joint distribution estimates
- Risk parity and equal risk contribution strategies
- Dynamic allocation based on market regimes and calculus signals
- Multi-objective optimization balancing return, risk, and liquidity

Formula → Meaning → Worked Example → Optimal Portfolio Allocation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

from stochastic_control import MultiAssetCovarianceController, LQGController

# Try to import cvxpy, make it optional
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("cvxpy not available, using numpy-based optimization only")

from joint_distribution_analyzer import JointDistributionAnalyzer, JointDistributionStats
from calculus_strategy import CalculusTradingStrategy, SignalType
from config import Config

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    CALCULUS_ENHANCED = "calculus_enhanced"
    REGIME_ADAPTIVE = "regime_adaptive"

@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization"""
    min_weight: float = 0.0  # Minimum weight per asset (long-only)
    max_weight: float = 0.3  # Maximum weight per asset (30% concentration limit)
    max_leverage: float = 1.0  # Maximum portfolio leverage
    target_return: Optional[float] = None  # Target portfolio return
    risk_budget: Optional[float] = None  # Maximum portfolio risk
    sector_limits: Optional[Dict[str, float]] = None  # Sector/asset class limits
    liquidity_constraint: float = 0.1  # Minimum liquidity requirement

@dataclass
class OptimizationResult:
    """Results from portfolio optimization"""
    optimal_weights: np.ndarray  # Optimal asset weights
    expected_return: float  # Expected portfolio return
    portfolio_risk: float  # Portfolio volatility
    sharpe_ratio: float  # Risk-adjusted return
    risk_contributions: np.ndarray  # Individual risk contributions
    optimization_status: str  # Solver status
    objective_value: float  # Final objective value
    constraints_satisfied: bool  # Whether all constraints are met
    calculation_time: float  # Time taken for optimization
    control_metadata: Optional[Dict[str, float]] = None

class PortfolioOptimizer:
    """
    Advanced portfolio optimizer integrating calculus-based signals with joint distribution analysis.

    This optimizer transforms the mathematical framework of Anne's system into
    actionable portfolio allocations across multiple cryptocurrency assets.
    """

    def __init__(self,
                 joint_analyzer: JointDistributionAnalyzer,
                 constraints: Optional[OptimizationConstraints] = None,
                 objective: OptimizationObjective = OptimizationObjective.CALCULUS_ENHANCED):
        """
        Initialize the portfolio optimizer.

        Args:
            joint_analyzer: Joint distribution analyzer for covariance estimation
            constraints: Optimization constraints
            objective: Primary optimization objective
        """
        self.joint_analyzer = joint_analyzer
        self.constraints = constraints or OptimizationConstraints()
        self.objective = objective

        # Strategy integration
        self.calculus_strategies = {}
        self.signal_weights = {}

        # Optimization cache
        self.last_optimization_time = 0
        self.optimization_frequency = 300  # Optimize every 5 minutes
        self.cached_result = None

        self.covariance_controller = MultiAssetCovarianceController(
            min_weight=self.constraints.min_weight,
            max_weight=self.constraints.max_weight
        )

        A = np.array([[0.95, 0.0],
                      [0.0, 0.90]])
        B = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2) * 0.5
        self.lqg_controller = LQGController(A, B, Q, R)

        logger.info(f"Portfolio Optimizer initialized: objective={objective.value}")

    def register_calculus_strategy(self, symbol: str, strategy: CalculusTradingStrategy):
        """
        Register a calculus trading strategy for an asset.

        Args:
            symbol: Asset symbol
            strategy: Calculus trading strategy instance
        """
        self.calculus_strategies[symbol] = strategy
        logger.info(f"Registered calculus strategy for {symbol}")

    def _calculate_signal_enhanced_returns(self,
                                         base_returns: np.ndarray,
                                         symbols: List[str]) -> np.ndarray:
        """
        Enhance expected returns with calculus-based signals.

        Formula: μ_enhanced = μ_base + α × signal_strength

        Args:
            base_returns: Base expected returns from historical data
            symbols: Asset symbols

        Returns:
            Signal-enhanced expected returns
        """
        enhanced_returns = base_returns.copy()

        for i, symbol in enumerate(symbols):
            if symbol in self.calculus_strategies:
                strategy = self.calculus_strategies[symbol]

                # Get latest signal (would use real-time data in practice)
                # For now, use a simplified signal based on recent performance
                signal_adjustment = 0

                # Enhance returns based on signal type and strength
                # This would integrate with real-time calculus signals
                if symbol in self.signal_weights:
                    signal_strength = self.signal_weights[symbol]
                    signal_adjustment = signal_strength * 0.01  # Scale signal impact

                enhanced_returns[i] += signal_adjustment

        logger.info(f"Signal enhancement applied to {len(symbols)} assets")
        return enhanced_returns

    def _minimize_variance(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray) -> OptimizationResult:
        """
        Closed-form minimum variance allocation using Σ⁻¹1 / (1ᵀΣ⁻¹1).
        """
        import time
        start_time = time.time()

        n = len(expected_returns)
        regularized_cov = covariance_matrix + np.eye(n) * 1e-8
        cov_inv = np.linalg.pinv(regularized_cov)
        ones = np.ones(n)
        denominator = ones @ cov_inv @ ones

        if abs(denominator) < 1e-12:
            weights = np.ones(n) / n
        else:
            weights = cov_inv @ ones
            weights = weights / denominator

        weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-12:
            weights = np.ones(n) / n
        else:
            weights = weights / weight_sum

        portfolio_variance = float(weights @ covariance_matrix @ weights)
        portfolio_risk = float(np.sqrt(max(portfolio_variance, 0.0)))
        expected_return = float(np.sum(weights * expected_returns))
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0.0

        calculation_time = time.time() - start_time

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=self._calculate_risk_contributions(weights, covariance_matrix),
            optimization_status='success',
            objective_value=portfolio_variance,
            constraints_satisfied=True,
            calculation_time=calculation_time
        )

    def _maximize_sharpe_ratio(self,
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray,
                              risk_free_rate: float = 0.02) -> OptimizationResult:
        """
        Tangency portfolio solution using Σ⁻¹(μ - r_f) normalised to unity.
        """
        import time
        start_time = time.time()

        n = len(expected_returns)
        regularized_cov = covariance_matrix + np.eye(n) * 1e-8
        cov_inv = np.linalg.pinv(regularized_cov)

        excess_returns = expected_returns - risk_free_rate / 252
        if not np.any(excess_returns > 0):
            logger.info("No positive excess returns, falling back to minimum variance")
            return self._minimize_variance(expected_returns, covariance_matrix)

        weights = cov_inv @ excess_returns
        weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-12:
            weights = np.ones(n) / n
        else:
            weights = weights / weight_sum

        portfolio_risk = float(np.sqrt(max(weights @ covariance_matrix @ weights, 0.0)))
        portfolio_return = float(weights @ excess_returns) + risk_free_rate / 252
        sharpe_ratio = ((portfolio_return - risk_free_rate / 252) / portfolio_risk
                        if portfolio_risk > 0 else 0.0)

        calculation_time = time.time() - start_time

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=portfolio_return,
            portfolio_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=self._calculate_risk_contributions(weights, covariance_matrix),
            optimization_status='success',
            objective_value=-sharpe_ratio,
            constraints_satisfied=True,
            calculation_time=calculation_time
        )

    def _risk_parity_optimization(self,
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray) -> OptimizationResult:
        """
        Risk parity solved via gradient descent equalising risk contributions.
        """
        import time
        start_time = time.time()

        n = len(expected_returns)
        weights = np.ones(n) / n
        learning_rate = 0.05

        for _ in range(200):
            cov_w = covariance_matrix @ weights
            portfolio_var = weights @ cov_w
            if portfolio_var <= 0:
                break
            contributions = weights * cov_w
            target = portfolio_var / n
            gradient = 2 * (contributions - target)
            weights -= learning_rate * gradient
            weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
            weight_sum = weights.sum()
            if abs(weight_sum) < 1e-12:
                weights = np.ones(n) / n
            else:
                weights = weights / weight_sum

        portfolio_risk = float(np.sqrt(max(weights @ covariance_matrix @ weights, 0.0)))
        expected_return = float(np.sum(weights * expected_returns))
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0.0

        calculation_time = time.time() - start_time

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            risk_contributions=self._calculate_risk_contributions(weights, covariance_matrix),
            optimization_status='success',
            objective_value=portfolio_risk,
            constraints_satisfied=True,
            calculation_time=calculation_time
        )

    def _calculate_risk_contributions(self,
                                     weights: np.ndarray,
                                     covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions for portfolio weights.

        Formula: RCᵢ = wᵢ(Σw)ᵢ / √(wᵀΣw)

        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix

        Returns:
            Risk contributions for each asset
        """
        portfolio_risk = np.sqrt(weights @ covariance_matrix @ weights)

        if portfolio_risk == 0:
            return np.zeros(len(weights))

        marginal_contributions = covariance_matrix @ weights
        risk_contributions = weights * marginal_contributions / portfolio_risk

        return risk_contributions

    def optimize_portfolio(self,
                          joint_stats: JointDistributionStats,
                          symbols: List[str],
                          current_timestamp: float) -> OptimizationResult:
        """
        Perform portfolio optimization using joint distribution analysis.

        Args:
            joint_stats: Joint distribution statistics
            symbols: Asset symbols
            current_timestamp: Current timestamp

        Returns:
            Optimization result
        """
        # Check if optimization is needed
        if current_timestamp - self.last_optimization_time < self.optimization_frequency:
            if self.cached_result:
                return self.cached_result

        logger.info(f"Starting portfolio optimization: {self.objective.value}")

        try:
            # Extract expected returns and covariance matrix
            expected_returns = joint_stats.expected_return * np.ones(len(symbols))  # Simplified
            covariance_matrix = joint_stats.covariance_matrix

            # Enhance returns with calculus signals
            if self.objective == OptimizationObjective.CALCULUS_ENHANCED:
                expected_returns = self._calculate_signal_enhanced_returns(expected_returns, symbols)

            # Perform optimization based on objective
            if self.objective == OptimizationObjective.MIN_VARIANCE:
                result = self._minimize_variance(expected_returns, covariance_matrix)
            elif self.objective == OptimizationObjective.MAX_SHARPE:
                result = self._maximize_sharpe_ratio(expected_returns, covariance_matrix)
            elif self.objective == OptimizationObjective.RISK_PARITY:
                result = self._risk_parity_optimization(expected_returns, covariance_matrix)
            elif self.objective == OptimizationObjective.EQUAL_WEIGHT:
                # Simple equal weight portfolio
                n = len(symbols)
                weights = np.ones(n) / n
                portfolio_risk = np.sqrt(weights @ covariance_matrix @ weights)
                expected_return = np.sum(weights * expected_returns)
                result = OptimizationResult(
                    optimal_weights=weights,
                    expected_return=expected_return,
                    portfolio_risk=portfolio_risk,
                    sharpe_ratio=expected_return / portfolio_risk if portfolio_risk > 0 else 0,
                    risk_contributions=self._calculate_risk_contributions(weights, covariance_matrix),
                    optimization_status='success',
                    objective_value=0,
                    constraints_satisfied=True,
                    calculation_time=0.001
                )
            else:
                # Default to minimum variance
                result = self._minimize_variance(expected_returns, covariance_matrix)

            control_weights, control_meta = self._apply_covariance_control(expected_returns, covariance_matrix)
            blended_weights = 0.5 * (result.optimal_weights + control_weights)
            weight_sum = blended_weights.sum()
            if abs(weight_sum) < 1e-8:
                blended_weights = np.ones_like(blended_weights) / len(blended_weights)
            else:
                blended_weights = blended_weights / weight_sum

            result.optimal_weights = blended_weights
            result.expected_return = float(np.sum(blended_weights * expected_returns))
            portfolio_variance = blended_weights @ covariance_matrix @ blended_weights
            result.portfolio_risk = float(np.sqrt(max(portfolio_variance, 0.0)))
            result.sharpe_ratio = (result.expected_return / result.portfolio_risk
                                   if result.portfolio_risk > 0 else 0.0)
            result.risk_contributions = self._calculate_risk_contributions(blended_weights, covariance_matrix)
            result.control_metadata = control_meta

            # Apply additional constraints
            result = self._apply_additional_constraints(result, symbols, joint_stats)

            # Cache result
            self.last_optimization_time = current_timestamp
            self.cached_result = result

            logger.info(f"Portfolio optimization completed: {result.optimization_status}")
            logger.info(f"Expected return: {result.expected_return:.4f}, Risk: {result.portfolio_risk:.4f}")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.3f}")

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return equal weight fallback
            n = len(symbols)
            weights = np.ones(n) / n
            return OptimizationResult(
                optimal_weights=weights,
                expected_return=0,
                portfolio_risk=0.1,
                sharpe_ratio=0,
                risk_contributions=weights * 0.1,
                optimization_status='failed',
                objective_value=float('inf'),
                constraints_satisfied=False,
                calculation_time=0,
                control_metadata=None
            )

    def _apply_covariance_control(self,
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Run the covariance controller + LQG overlay to keep risk within stochastic targets.
        """
        weights = self.covariance_controller.optimise(
            expected_returns,
            covariance_matrix,
            target_return=self.constraints.target_return
        )

        target_return = self.constraints.target_return or float(np.mean(expected_returns))
        diag_variance = np.clip(np.diag(covariance_matrix), 0.0, None)
        avg_vol = float(np.sqrt(np.mean(diag_variance)))
        target_risk = self.constraints.risk_budget or avg_vol

        state = np.array([
            expected_returns.mean() - target_return,
            avg_vol - target_risk
        ])
        control = self.lqg_controller.optimal_control(state)
        adjustment = float(control[0]) if control.ndim > 0 else float(control)

        weights = weights + adjustment / max(len(weights), 1)
        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-8:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weight_sum

        metadata = {
            "state_deviation_return": float(state[0]),
            "state_deviation_risk": float(state[1]),
            "control_adjustment": float(adjustment)
        }

        return weights, metadata

    def _apply_additional_constraints(self,
                                    result: OptimizationResult,
                                    symbols: List[str],
                                    joint_stats: JointDistributionStats) -> OptimizationResult:
        """
        Apply additional constraints based on market conditions and risk limits.

        Args:
            result: Initial optimization result
            symbols: Asset symbols
            joint_stats: Joint distribution statistics

        Returns:
            Constrained optimization result
        """
        weights = result.optimal_weights.copy()

        # Apply regime-based adjustments
        if joint_stats.regime_state == 'VOLATILE':
            # Reduce exposure in volatile markets
            weights *= 0.7
            # Allocate remaining weight to cash (represented as reduced positions)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

        elif joint_stats.regime_state == 'BEAR':
            # Defensive positioning in bear markets
            # Reduce weights overall and focus on least volatile assets
            volatilities = np.sqrt(np.diag(joint_stats.covariance_matrix))
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            weights = 0.5 * weights + 0.5 * inv_vol_weights

        # Apply position limits
        weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

        # Recalculate portfolio metrics with new weights
        portfolio_risk = np.sqrt(weights @ joint_stats.covariance_matrix @ weights)
        expected_return = np.sum(weights * result.expected_return * np.ones(len(weights)))
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0

        # Update result
        result.optimal_weights = weights
        result.portfolio_risk = portfolio_risk
        result.expected_return = expected_return
        result.sharpe_ratio = sharpe_ratio
        result.risk_contributions = self._calculate_risk_contributions(weights, joint_stats.covariance_matrix)

        return result

    def get_optimal_allocations(self,
                              result: OptimizationResult,
                              symbols: List[str],
                              total_capital: float) -> Dict[str, Dict]:
        """
        Get optimal dollar allocations for each asset.

        Args:
            result: Optimization result
            symbols: Asset symbols
            total_capital: Total capital to allocate

        Returns:
            Dictionary with allocation details for each asset
        """
        allocations = {}

        for i, symbol in enumerate(symbols):
            weight = result.optimal_weights[i]
            allocation_amount = weight * total_capital
            risk_contribution = result.risk_contributions[i]

            allocations[symbol] = {
                'weight': weight,
                'allocation_amount': allocation_amount,
                'risk_contribution': risk_contribution,
                'expected_return': result.expected_return * weight,
                'risk_attribution': result.portfolio_risk * abs(risk_contribution)
            }

        return allocations

    def get_optimization_summary(self,
                               result: OptimizationResult,
                               symbols: List[str]) -> Dict:
        """
        Get a comprehensive summary of the optimization results.

        Args:
            result: Optimization result
            symbols: Asset symbols

        Returns:
            Dictionary with optimization summary
        """
        # Get top allocations
        top_assets = []
        for i, (symbol, weight) in enumerate(zip(symbols, result.optimal_weights)):
            if weight > 0.01:  # Only include allocations > 1%
                top_assets.append({
                    'symbol': symbol,
                    'weight': weight,
                    'rank': i + 1
                })

        top_assets.sort(key=lambda x: x['weight'], reverse=True)

        return {
            'optimization_status': result.optimization_status,
            'objective_function': self.objective.value,
            'portfolio_metrics': {
                'expected_return': result.expected_return,
                'portfolio_risk': result.portfolio_risk,
                'sharpe_ratio': result.sharpe_ratio,
                'objective_value': result.objective_value
            },
            'top_allocations': top_assets[:5],  # Top 5 allocations
            'risk_analysis': {
                'total_risk': result.portfolio_risk,
                'risk_contributions': {
                    symbol: float(rc)
                    for symbol, rc in zip(symbols, result.risk_contributions)
                },
                'concentration_ratio': np.sum(np.sort(result.optimal_weights)[-3:])  # Top 3 concentration
            },
            'constraints_check': {
                'all_satisfied': result.constraints_satisfied,
                'max_weight': np.max(result.optimal_weights),
                'min_weight': np.min(result.optimal_weights[result.optimal_weights > 0]),
                'weights_sum': np.sum(result.optimal_weights)
            },
            'calculation_time': result.calculation_time
        }

# Example usage
if __name__ == "__main__":
    # Create sample joint distribution analyzer
    analyzer = JointDistributionAnalyzer(num_assets=4)

    # Create sample covariance matrix and returns
    np.random.seed(42)
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    covariance_matrix = np.random.rand(4, 4)
    covariance_matrix = covariance_matrix @ covariance_matrix.T  # Make positive definite
    np.fill_diagonal(covariance_matrix, np.diag(covariance_matrix) + 0.01)  # Add variance

    # Create sample joint stats
    from joint_distribution_analyzer import JointDistributionStats
    sample_stats = JointDistributionStats(
        covariance_matrix=covariance_matrix,
        correlation_matrix=np.corrcoef(covariance_matrix),
        eigenvalues=np.linalg.eigvals(covariance_matrix),
        eigenvectors=np.linalg.eig(covariance_matrix)[1],
        risk_contributions=np.ones(4) / 4,
        optimal_weights=np.ones(4) / 4,
        expected_return=0.001,
        portfolio_variance=0.0004,
        sharpe_ratio=1.5,
        regime_state='BULL',
        timestamp=1640995200
    )

    # Create optimizer and run optimization
    optimizer = PortfolioOptimizer(analyzer, objective=OptimizationObjective.MAX_SHARPE)
    result = optimizer.optimize_portfolio(sample_stats, symbols, 1640995200)

    print("Portfolio Optimization Results:")
    print(f"Status: {result.optimization_status}")
    print(f"Expected Return: {result.expected_return:.4f}")
    print(f"Portfolio Risk: {result.portfolio_risk:.4f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Optimal Weights: {result.optimal_weights}")

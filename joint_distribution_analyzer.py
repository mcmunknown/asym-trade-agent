"""
Joint Distribution Analyzer for Portfolio Optimization
====================================================

This module implements the multivariate joint distribution analysis for Anne's
calculus-based trading system, enabling sophisticated portfolio optimization
and risk management across multiple cryptocurrency assets.

Mathematical Foundation:
- Multivariate normal/t-copula modeling for asset returns
- Exponential weighted covariance matrix estimation
- Eigenvalue decomposition for principal risk factor analysis
- Dynamic correlation tracking with regime detection

Formula → Meaning → Worked Example → Portfolio Optimization
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings

# Suppress numerical warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

@dataclass
class JointDistributionStats:
    """Statistics from joint distribution analysis"""
    covariance_matrix: np.ndarray  # Σ - Covariance matrix
    correlation_matrix: np.ndarray  # Correlation matrix
    eigenvalues: np.ndarray  # λᵢ - Principal risk factors
    eigenvectors: np.ndarray  # vᵢ - Factor loadings
    risk_contributions: np.ndarray  # Risk budgeting analysis
    optimal_weights: np.ndarray  # Mean-variance optimal weights
    expected_return: float  # Portfolio expected return
    portfolio_variance: float  # wᵀΣw - Portfolio variance
    sharpe_ratio: float  # Risk-adjusted return measure
    regime_state: str  # Current market regime
    timestamp: float  # Analysis timestamp

class JointDistributionAnalyzer:
    """
    Implements Anne's joint distribution analysis for multivariate portfolio optimization.

    The "true joint distribution" for 8 crypto assets is the evolving, estimated
    multivariate probability model that captures how those eight returns behave together.
    """

    def __init__(self,
                 num_assets: int = 8,
                 decay_factor: float = 0.94,
                 min_observations: int = 30,
                 regularization_strength: float = 1e-4,
                 regime_threshold: float = 0.3):
        """
        Initialize the joint distribution analyzer.

        Args:
            num_assets: Number of crypto assets (default 8)
            decay_factor: Exponential decay factor λ for covariance estimation (0 < λ < 1)
            min_observations: Minimum observations required for analysis
            regularization_strength: L2 regularization for covariance matrix
            regime_threshold: Threshold for regime detection
        """
        self.num_assets = num_assets
        self.decay_factor = decay_factor
        self.min_observations = min_observations
        self.regularization_strength = regularization_strength
        self.regime_threshold = regime_threshold

        # Asset tracking
        self.asset_symbols = []
        self.price_history = {}
        self.return_history = {}
        self.asset_returns = {}
        self.asset_timestamps = {}

        # Analysis state
        self.last_analysis_time = 0
        self.analysis_frequency = 300  # Analyze every 5 minutes

        logger.info(f"Joint Distribution Analyzer initialized: {num_assets} assets, "
                   f"decay_factor={decay_factor}, regularization={regularization_strength}")

    def add_asset_data(self, symbol: str, price: float, timestamp: float):
        """
        Add new price data for an asset.

        Args:
            symbol: Asset symbol (e.g., 'BTCUSDT')
            price: Current price
            timestamp: Data timestamp
        """
        if symbol not in self.asset_symbols:
            self.asset_symbols.append(symbol)
            self.price_history[symbol] = []
            self.return_history[symbol] = []
            self.asset_returns[symbol] = []
            self.asset_timestamps[symbol] = []
            logger.info(f"Added new asset: {symbol}")

        # Store price data
        self.price_history[symbol].append((timestamp, price))

        # Calculate returns if we have previous price
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2][1]
            if prev_price > 0:
                returns = np.log(price / prev_price)  # Log returns
                self.return_history[symbol].append((timestamp, returns))
                self.asset_returns[symbol].append(returns)
                self.asset_timestamps[symbol].append(timestamp)

        # Maintain rolling window (keep last 1000 observations)
        max_history = 1000
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        if len(self.return_history[symbol]) > max_history:
            self.return_history[symbol] = self.return_history[symbol][-max_history:]
        if len(self.asset_returns[symbol]) > max_history:
            self.asset_returns[symbol] = self.asset_returns[symbol][-max_history:]
        if len(self.asset_timestamps[symbol]) > max_history:
            self.asset_timestamps[symbol] = self.asset_timestamps[symbol][-max_history:]

    def update_returns(self, price_updates: Dict[str, float], timestamp: Optional[float] = None):
        """
        Bulk-update asset return series from a snapshot of latest prices.

        Args:
            price_updates: Mapping of symbol -> latest price
            timestamp: Optional timestamp; defaults to current time
        """
        if not price_updates:
            return

        ts = timestamp or time.time()
        for symbol, price in price_updates.items():
            try:
                price_val = float(price)
            except (TypeError, ValueError):
                logger.debug(f"Skipping invalid price for {symbol}: {price}")
                continue

            if price_val <= 0:
                logger.debug(f"Skipping non-positive price for {symbol}: {price_val}")
                continue

            self.add_asset_data(symbol, price_val, ts)

    def _prepare_return_matrix(self) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Prepare the return matrix for analysis.

        Returns:
            Tuple of (return_matrix, valid_symbols, timestamps)
        """
        valid_symbols = []
        return_series = {}

        # Find symbols with sufficient data
        min_returns = 20  # Minimum returns for analysis
        for symbol in self.asset_symbols:
            if len(self.return_history[symbol]) >= min_returns:
                valid_symbols.append(symbol)
                timestamps, returns = zip(*self.return_history[symbol])
                return_series[symbol] = pd.Series(returns, index=timestamps)

        if len(valid_symbols) < 2:
            logger.warning(f"Insufficient assets with data: {len(valid_symbols)} < 2")
            return np.array([], dtype=float).reshape(0, 0), [], []

        # Create aligned return matrix
        # Use intersection of timestamps for all assets
        common_timestamps = None
        for symbol in valid_symbols:
            if common_timestamps is None:
                common_timestamps = set(return_series[symbol].index)
            else:
                common_timestamps &= set(return_series[symbol].index)

        common_timestamps = sorted(list(common_timestamps))
        if len(common_timestamps) < self.min_observations:
            logger.warning(f"Insufficient aligned observations: {len(common_timestamps)} < {self.min_observations}")
            return np.array([], dtype=float).reshape(0, 0), [], []

        # Build return matrix
        return_matrix = np.zeros((len(common_timestamps), len(valid_symbols)))
        for i, symbol in enumerate(valid_symbols):
            for j, timestamp in enumerate(common_timestamps):
                return_matrix[j, i] = return_series[symbol].loc[timestamp]

        logger.info(f"Prepared return matrix: {return_matrix.shape[0]} observations × {return_matrix.shape[1]} assets")
        return return_matrix, valid_symbols, common_timestamps

    def _estimate_covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate the covariance matrix with exponential weighting and regularization.

        Formula: Σₜ = (1-λ)Σₜ₋₁ + λ(Rₜ-μ)(Rₜ-μ)ᵀ

        Args:
            returns: Return matrix (T × N)

        Returns:
            Regularized covariance matrix (N × N)
        """
        T, N = returns.shape

        # Calculate mean returns
        mean_returns = np.nanmean(returns, axis=0)

        # Handle NaN values in returns
        returns_clean = returns.copy()
        for i in range(N):
            nan_mask = np.isnan(returns[:, i])
            if nan_mask.any():
                returns_clean[nan_mask, i] = 0
                logger.debug(f"Filled {nan_mask.sum()} NaN values for asset {i}")

        # Exponential weighting
        weights = np.array([self.decay_factor ** (T - t - 1) for t in range(T)])
        weights = weights / weights.sum()  # Normalize weights

        # Calculate weighted covariance matrix
        weighted_returns = returns_clean - mean_returns
        covariance_matrix = np.zeros((N, N))

        for t in range(T):
            deviation = weighted_returns[t:t+1, :]  # Shape (1, N)
            covariance_matrix += weights[t] * (deviation.T @ deviation)

        # Apply L2 regularization for numerical stability
        regularization_matrix = self.regularization_strength * np.eye(N)
        covariance_matrix += regularization_matrix

        # Ensure positive definiteness
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)  # Minimum eigenvalue
        covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        logger.info(f"Covariance matrix estimated: condition number = {np.linalg.cond(covariance_matrix):.2f}")
        return covariance_matrix

    def _perform_eigenvalue_analysis(self, covariance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition for principal risk factor analysis.

        Formula: Σ = QΛQᵀ where Λ = diag(λ₁, ..., λ₈)

        Args:
            covariance_matrix: Covariance matrix Σ

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Ensure matrix is symmetric for eigenvalue decomposition
        symmetric_cov = (covariance_matrix + covariance_matrix.T) / 2

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(symmetric_cov)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Calculate explained variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance

        logger.info(f"Eigenvalue analysis completed:")
        logger.info(f"  Total variance: {total_variance:.6f}")
        logger.info(f"  Top 3 eigenvalues: {eigenvalues[:3]}")
        logger.info(f"  Explained variance (top 3): {explained_variance_ratio[:3].sum():.1%}")

        return eigenvalues, eigenvectors

    def _detect_market_regime(self, returns: np.ndarray, correlation_matrix: np.ndarray) -> str:
        """
        Detect current market regime based on correlation patterns and volatility.

        Args:
            returns: Return matrix
            correlation_matrix: Current correlation matrix

        Returns:
            Regime state: 'BULL', 'BEAR', 'SIDEWAYS', or 'VOLATILE'
        """
        # Calculate average correlation
        avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))

        # Calculate recent volatility
        recent_returns = returns[-min(20, len(returns)):]  # Last 20 observations
        recent_volatility = np.nanstd(recent_returns, axis=0).mean()

        # Calculate recent trend
        recent_mean = np.nanmean(recent_returns, axis=0).mean()

        # Regime classification
        if avg_correlation > 0.7:
            regime = 'VOLATILE'  # High correlation = stress/volatility
        elif recent_mean > 0.002:
            regime = 'BULL'  # Strong positive returns
        elif recent_mean < -0.002:
            regime = 'BEAR'  # Strong negative returns
        else:
            regime = 'SIDEWAYS'  # Low correlation, modest returns

        logger.info(f"Market regime detected: {regime} (avg_corr={avg_correlation:.3f}, vol={recent_volatility:.4f})")
        return regime

    def _optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          risk_free_rate: float = 0.02) -> np.ndarray:
        """
        Calculate mean-variance optimal portfolio weights.

        Formula: w* = argmin wᵀΣw subject to wᵀμ = r* and wᵀ1 = 1

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix Σ
            risk_free_rate: Risk-free rate for Sharpe ratio optimization

        Returns:
            Optimal portfolio weights
        """
        N = len(expected_returns)

        try:
            # Calculate inverse of covariance matrix
            cov_inv = np.linalg.inv(covariance_matrix)

            # Minimum variance portfolio (no target return constraint)
            ones = np.ones((N, 1))
            min_var_weights = cov_inv @ ones / (ones.T @ cov_inv @ ones)
            min_var_weights = min_var_weights.flatten()

            # Maximum Sharpe ratio portfolio (tangency portfolio)
            excess_returns = expected_returns - risk_free_rate / 252  # Daily risk-free rate

            # Check if any excess returns are positive
            if np.any(excess_returns > 0):
                sharpe_weights = cov_inv @ excess_returns
                sharpe_weights = sharpe_weights / np.sum(np.abs(sharpe_weights))  # Normalize
            else:
                # Use minimum variance weights if no positive excess returns
                sharpe_weights = min_var_weights

            # Apply realistic constraints
            # Long-only constraint for crypto trading
            sharpe_weights = np.maximum(sharpe_weights, 0)

            # Normalize weights to sum to 1
            if sharpe_weights.sum() > 0:
                optimal_weights = sharpe_weights / sharpe_weights.sum()
            else:
                optimal_weights = np.ones(N) / N  # Equal weight fallback

            # Apply position limits (max 30% in any single asset)
            max_weight = 0.3
            optimal_weights = np.minimum(optimal_weights, max_weight)

            # Renormalize after applying limits
            optimal_weights = optimal_weights / optimal_weights.sum()

            logger.info(f"Portfolio optimization completed:")
            logger.info(f"  Expected portfolio return: {np.sum(optimal_weights * expected_returns):.4f}")
            logger.info(f"  Portfolio weights: {optimal_weights}")

            return optimal_weights

        except np.linalg.LinAlgError as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            return np.ones(N) / N

    def _calculate_risk_contributions(self,
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions for risk budgeting.

        Formula: RCᵢ = wᵢ(Σw)ᵢ / √(wᵀΣw)

        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix

        Returns:
            Risk contributions for each asset
        """
        # Calculate portfolio variance
        portfolio_variance = weights @ covariance_matrix @ weights

        # Calculate marginal risk contributions
        marginal_contributions = covariance_matrix @ weights

        # Calculate risk contributions
        risk_contributions = weights * marginal_contributions / np.sqrt(portfolio_variance)

        # Normalize to percentages
        risk_contributions = risk_contributions / np.sum(np.abs(risk_contributions))

        return risk_contributions

    def analyze_joint_distribution(self, timestamp: float) -> Optional[JointDistributionStats]:
        """
        Perform comprehensive joint distribution analysis.

        Args:
            timestamp: Current timestamp

        Returns:
            JointDistributionStats with analysis results or None if insufficient data
        """
        # Check if analysis is needed
        if timestamp - self.last_analysis_time < self.analysis_frequency:
            return None

        logger.info("Starting joint distribution analysis")

        # Prepare return matrix
        returns_matrix, valid_symbols, timestamps = self._prepare_return_matrix()
        if returns_matrix.size == 0:
            logger.warning("Insufficient data for joint distribution analysis")
            return None

        try:
            # Step 1: Estimate covariance matrix
            covariance_matrix = self._estimate_covariance_matrix(returns_matrix)

            # Step 2: Calculate correlation matrix
            std_devs = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)  # Handle any NaN

            # Step 3: Eigenvalue analysis
            eigenvalues, eigenvectors = self._perform_eigenvalue_analysis(covariance_matrix)

            # Step 4: Detect market regime
            regime_state = self._detect_market_regime(returns_matrix, correlation_matrix)

            # Step 5: Portfolio optimization
            expected_returns = np.nanmean(returns_matrix, axis=0)
            optimal_weights = self._optimize_portfolio(expected_returns, covariance_matrix)

            # Step 6: Calculate portfolio metrics
            portfolio_variance = optimal_weights @ covariance_matrix @ optimal_weights
            expected_return = np.sum(optimal_weights * expected_returns)
            sharpe_ratio = expected_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0

            # Step 7: Risk budgeting
            risk_contributions = self._calculate_risk_contributions(optimal_weights, covariance_matrix)

            # Update analysis timestamp
            self.last_analysis_time = timestamp

            # Create results
            stats = JointDistributionStats(
                covariance_matrix=covariance_matrix,
                correlation_matrix=correlation_matrix,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                risk_contributions=risk_contributions,
                optimal_weights=optimal_weights,
                expected_return=expected_return,
                portfolio_variance=portfolio_variance,
                sharpe_ratio=sharpe_ratio,
                regime_state=regime_state,
                timestamp=timestamp
            )

            logger.info(f"Joint distribution analysis completed successfully:")
            logger.info(f"  Assets analyzed: {len(valid_symbols)}")
            logger.info(f"  Portfolio Sharpe ratio: {sharpe_ratio:.3f}")
            logger.info(f"  Market regime: {regime_state}")

            return stats

        except Exception as e:
            logger.error(f"Joint distribution analysis failed: {e}")
            return None

    def get_hedge_ratios(self, base_asset_index: int, stats: JointDistributionStats) -> np.ndarray:
        """
        Calculate optimal hedge ratios for a base asset.

        Formula: h = -Σ₋₁,₋₁⁻¹ Σ₁,₋₁ where 1 is the base asset

        Args:
            base_asset_index: Index of the asset to hedge
            stats: Joint distribution statistics

        Returns:
            Hedge ratios for other assets
        """
        covariance_matrix = stats.covariance_matrix
        N = len(covariance_matrix)

        if base_asset_index >= N:
            logger.error(f"Invalid base asset index: {base_asset_index}")
            return np.zeros(N)

        try:
            # Partition covariance matrix
            mask = np.ones(N, dtype=bool)
            mask[base_asset_index] = False

            Sigma_11 = covariance_matrix[base_asset_index, base_asset_index]  # Scalar
            Sigma_12 = covariance_matrix[base_asset_index, mask]  # 1 × (N-1)
            Sigma_22_inv = np.linalg.inv(covariance_matrix[mask][:, mask])  # (N-1) × (N-1)

            # Calculate optimal hedge ratios
            hedge_ratios = -Sigma_22_inv @ Sigma_12.T

            logger.info(f"Hedge ratios calculated for asset {base_asset_index}: {hedge_ratios}")
            return hedge_ratios

        except np.linalg.LinAlgError as e:
            logger.error(f"Hedge ratio calculation failed: {e}")
            return np.zeros(N-1)

    def get_portfolio_summary(self, stats: JointDistributionStats) -> Dict:
        """
        Get a comprehensive summary of the joint distribution analysis.

        Args:
            stats: Joint distribution statistics

        Returns:
            Dictionary with analysis summary
        """
        return {
            'analysis_timestamp': stats.timestamp,
            'assets_analyzed': len(stats.optimal_weights),
            'market_regime': stats.regime_state,
            'portfolio_metrics': {
                'expected_return': stats.expected_return,
                'portfolio_variance': stats.portfolio_variance,
                'portfolio_volatility': np.sqrt(stats.portfolio_variance),
                'sharpe_ratio': stats.sharpe_ratio
            },
            'risk_analysis': {
                'top_eigenvalues': stats.eigenvalues[:3].tolist(),
                'explained_variance_ratio': (stats.eigenvalues / stats.eigenvalues.sum())[:3].tolist(),
                'average_correlation': np.mean(np.abs(stats.correlation_matrix[np.triu_indices_from(stats.correlation_matrix, k=1)]))
            },
            'optimal_weights': {
                symbol: float(weight)
                for symbol, weight in zip(self.asset_symbols, stats.optimal_weights)
                if weight > 0.01  # Only include weights > 1%
            },
            'risk_contributions': {
                symbol: float(rc)
                for symbol, rc in zip(self.asset_symbols, stats.risk_contributions)
            }
        }

    def is_data_sufficient(self) -> bool:
        """
        Check if we have sufficient data for meaningful analysis.

        Returns:
            bool: True if we have enough data for joint distribution analysis
        """
        # Check if we have data for enough assets
        if len(self.asset_symbols) < 4:
            return False

        # Check if we have enough observations
        total_observations = sum(len(self.asset_returns[symbol]) for symbol in self.asset_symbols)
        if total_observations < self.min_observations * len(self.asset_symbols):
            return False

        # Check if we have recent data (last 5 minutes)
        current_time = time.time()
        recent_threshold = 300  # 5 minutes

        for symbol in self.asset_symbols:
            if symbol in self.asset_returns and self.asset_returns[symbol]:
                latest_timestamp = self.asset_timestamps[symbol][-1]
                if current_time - latest_timestamp > recent_threshold:
                    return False

        return True

# Example usage and testing
if __name__ == "__main__":
    # Test the joint distribution analyzer
    analyzer = JointDistributionAnalyzer(num_assets=8)

    # Generate sample data
    np.random.seed(42)
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT']

    # Simulate 100 days of price data
    for day in range(100):
        timestamp = day * 86400  # Daily timestamps
        for i, symbol in enumerate(symbols):
            # Generate correlated price movements
            base_return = np.random.normal(0, 0.02)  # 2% daily volatility
            asset_return = base_return + np.random.normal(0, 0.01)  # Asset-specific noise
            price = 100 * (1 + asset_return)  # Simple price simulation
            analyzer.add_asset_data(symbol, price, timestamp)

    # Perform analysis
    stats = analyzer.analyze_joint_distribution(timestamp)
    if stats:
        summary = analyzer.get_portfolio_summary(stats)
        print("Joint Distribution Analysis Summary:")
        print(f"Market Regime: {summary['market_regime']}")
        print(f"Portfolio Sharpe Ratio: {summary['portfolio_metrics']['sharpe_ratio']:.3f}")
        print(f"Optimal Weights: {summary['optimal_weights']}")

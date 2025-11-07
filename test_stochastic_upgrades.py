"""
Tests for the stochastic calculus + optimal control upgrades.
"""

import time

import numpy as np
import pandas as pd

from quantitative_models import CalculusPriceAnalyzer
from calculus_strategy import CalculusTradingStrategy
from portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints, OptimizationObjective
from joint_distribution_analyzer import JointDistributionAnalyzer, JointDistributionStats


def _build_joint_stats(covariance: np.ndarray) -> JointDistributionStats:
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    weights = np.ones(covariance.shape[0]) / covariance.shape[0]
    risk_contributions = weights * (covariance @ weights)
    return JointDistributionStats(
        covariance_matrix=covariance,
        correlation_matrix=correlation,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        risk_contributions=risk_contributions,
        optimal_weights=weights,
        expected_return=0.01,
        portfolio_variance=float(weights @ covariance @ weights),
        sharpe_ratio=1.0,
        regime_state="STABLE",
        timestamp=time.time()
    )


def test_stochastic_layer_outputs():
    np.random.seed(0)
    prices = pd.Series(
        np.linspace(100, 110, 120) + np.random.normal(0, 0.5, 120),
        index=pd.date_range("2024-01-01", periods=120, freq="1min")
    )

    analyzer = CalculusPriceAnalyzer()
    analysis = analyzer.analyze_price_curve(prices)

    required_columns = [
        "estimated_drift", "estimated_diffusion", "optimal_delta",
        "hjb_action", "stochastic_volatility", "ito_correction"
    ]

    for column in required_columns:
        assert column in analysis, f"Missing stochastic column: {column}"

    latest = analysis.iloc[-1]
    assert np.isfinite(latest["optimal_delta"])
    assert latest["stochastic_volatility"] > 0


def test_strategy_stochastic_confidence():
    np.random.seed(1)
    prices = pd.Series(
        100 + np.cumsum(np.sin(np.linspace(0, 6, 180)) + np.random.normal(0, 0.2, 180)),
        index=pd.date_range("2024-01-01", periods=180, freq="30s")
    )

    strategy = CalculusTradingStrategy()
    signals = strategy.generate_trading_signals(prices)

    assert "stochastic_confidence" in signals
    sc = signals["stochastic_confidence"].dropna()
    assert (sc >= 0).all() and (sc <= 1).all()
    assert signals["hedge_directive"].str.contains("Δ=").any()


def test_portfolio_covariance_control():
    covariance = np.array([[0.02, 0.01, 0.0],
                           [0.01, 0.03, 0.015],
                           [0.0, 0.015, 0.025]])
    joint_stats = _build_joint_stats(covariance)
    joint_analyzer = JointDistributionAnalyzer(num_assets=3)
    constraints = OptimizationConstraints(max_weight=0.6, target_return=0.01, risk_budget=0.15)

    optimizer = PortfolioOptimizer(
        joint_analyzer=joint_analyzer,
        constraints=constraints,
        objective=OptimizationObjective.CALCULUS_ENHANCED
    )

    symbols = ["A", "B", "C"]
    result = optimizer.optimize_portfolio(joint_stats, symbols, current_timestamp=time.time())

    assert abs(result.optimal_weights.sum() - 1.0) < 1e-6
    assert result.control_metadata is not None
    assert "control_adjustment" in result.control_metadata

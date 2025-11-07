"""
Stochastic Calculus and Optimal Control Layer
============================================

This module provides the missing bridge between the deterministic calculus layer
and full stochastic calculus + optimal control that hedge funds rely on.

It implements:
    * Itô process estimation with drift/volatility inference
    * Dynamic hedging optimizer that minimises portfolio variance
    * Hamilton–Jacobi–Bellman (HJB) control solver for optimal actions
    * Stochastic volatility estimation via an unscented-style filter
    * Multi-asset covariance controller for dynamic hedge rebalancing
    * Linear-Quadratic-Gaussian (LQG) controller for continuous hedging

Formula → Meaning → Worked Example stays intact, but now extended
to stochastic differential equations and optimal control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1️⃣  Itô Process Estimation and Itô's Lemma utilities
# ---------------------------------------------------------------------------

@dataclass
class ItoProcessState:
    """Container describing the current Itô process parameters."""

    price: float
    drift: float
    diffusion: float
    dt: float


class ItoProcessModel:
    """
    Estimate and manipulate stochastic differential equations of the form:

        dP_t = μ_t P_t dt + σ_t P_t dW_t

    The estimators work on log-returns to maintain numerical stability.
    """

    def __init__(self,
                 window: int = 50,
                 min_vol: float = 1e-6,
                 max_vol: float = 5.0):
        self.window = window
        self.min_vol = min_vol
        self.max_vol = max_vol

    def estimate_state(self, prices: pd.Series, dt: float = 1.0) -> ItoProcessState:
        """
        Estimate drift (μ) and diffusion (σ) using rolling log-returns.

        Args:
            prices: price series
            dt: time increment between samples (in seconds or minutes)
        """
        if len(prices) < 2:
            raise ValueError("Need at least two prices to estimate SDE parameters")

        log_returns = np.log(prices / prices.shift(1)).dropna()
        recent = log_returns.iloc[-min(len(log_returns), self.window):]

        mu = np.clip(recent.mean() / dt, -5.0, 5.0)
        sigma = np.clip(recent.std(ddof=1) / np.sqrt(dt), self.min_vol, self.max_vol)

        return ItoProcessState(price=float(prices.iloc[-1]), drift=float(mu),
                               diffusion=float(sigma), dt=dt)

    @staticmethod
    def apply_ito_lemma(value_gradient: float,
                        value_gamma: float,
                        process_state: ItoProcessState) -> float:
        """
        Apply Itô's lemma for a value function V(P, t):

            dV = V_t dt + V_P dP + 0.5 V_PP (σ P)^2 dt

        Here we approximate V_t via the deterministic calculus layer, hence this
        method focuses on the stochastic correction terms.
        """
        sigma_term = 0.5 * value_gamma * (process_state.diffusion * process_state.price) ** 2
        stochastic_increment = value_gradient * process_state.diffusion * process_state.price
        deterministic_increment = process_state.drift * process_state.price * value_gradient

        return deterministic_increment + stochastic_increment + sigma_term

    def simulate_paths(self,
                       initial_price: float,
                       steps: int,
                       mu: float,
                       sigma: float,
                       dt: float,
                       n_paths: int = 1000,
                       random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Monte-Carlo simulation of the estimated SDE to stress-test hedges.
        """
        rng = random_state or np.random.default_rng()
        prices = np.zeros((n_paths, steps), dtype=float)
        prices[:, 0] = initial_price

        for t in range(1, steps):
            z = rng.standard_normal(size=n_paths)
            prices[:, t] = prices[:, t - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )

        return prices


# ---------------------------------------------------------------------------
# 2️⃣  Dynamic Hedging Optimisation
# ---------------------------------------------------------------------------

class DynamicHedgingOptimizer:
    """
    Minimises E[(dW_t)^2] with respect to the hedge ratio Δ_t.

    In the frictionless case the solution collapses to Δ_t = V_P (delta hedging),
    but here we explicitly include variance penalties and transaction costs so the
    optimiser can reason about practical constraints.
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 transaction_cost: float = 0.0,
                 max_leverage: float = 5.0):
        self.risk_aversion = max(risk_aversion, 1e-6)
        self.transaction_cost = max(transaction_cost, 0.0)
        self.max_leverage = max_leverage

    def optimal_delta(self,
                      value_gradient: float,
                      sigma: float,
                      price: float) -> Tuple[float, float]:
        """
        Compute the hedge ratio that minimises quadratic variation:

            min_Δ  (V_P - Δ)^2 σ^2 P^2 + λ Δ^2
        """
        variance_term = (sigma * price) ** 2
        penalty = self.transaction_cost + self.risk_aversion * 1e-3
        denominator = variance_term + penalty

        if denominator <= 0:
            return 0.0, 0.0

        unconstrained_delta = value_gradient * variance_term / denominator
        clipped_delta = float(np.clip(unconstrained_delta, -self.max_leverage, self.max_leverage))

        residual_variance = (value_gradient - clipped_delta) ** 2 * variance_term

        return clipped_delta, residual_variance


# ---------------------------------------------------------------------------
# 3️⃣  Hamilton–Jacobi–Bellman Optimal Control
# ---------------------------------------------------------------------------

class HJBSolver:
    """
    Lightweight HJB solver for a single-asset wealth process with quadratic utility:

        0 = V_t + max_a [ μ P a V_W + 0.5 σ^2 P^2 a^2 V_WW - r V ]
    """

    def __init__(self,
                 discount_rate: float = 0.0,
                 risk_aversion: float = 1.0,
                 action_grid: Optional[Iterable[float]] = None):
        self.discount_rate = discount_rate
        self.risk_aversion = max(risk_aversion, 1e-6)
        self.action_grid = list(action_grid) if action_grid is not None else np.linspace(-3, 3, 25)

    def optimal_action(self,
                       wealth_gradient: float,
                       wealth_gamma: float,
                       mu: float,
                       sigma: float,
                       price: float) -> Tuple[float, float]:
        """
        Return the action that maximises the HJB Hamiltonian on the discrete grid.
        """
        best_action = 0.0
        best_value = -np.inf

        for a in self.action_grid:
            drift_term = mu * price * a * wealth_gradient
            diff_term = 0.5 * (sigma * price * a) ** 2 * wealth_gamma
            running_cost = -self.discount_rate * self.risk_aversion * a ** 2
            value = drift_term + diff_term + running_cost

            if value > best_value:
                best_value = value
                best_action = a

        return float(best_action), float(best_value)


# ---------------------------------------------------------------------------
# 4️⃣  Stochastic Volatility Filtering (Unscented-style)
# ---------------------------------------------------------------------------

@dataclass
class VolatilityState:
    mean: float
    variance: float


class StochasticVolatilityFilter:
    """
    Simple unscented Kalman-style filter for Heston-type volatility:

        dσ_t = κ(θ - σ_t) dt + ξ σ_t dZ_t
    """

    def __init__(self,
                 kappa: float = 2.0,
                 theta: float = 0.5,
                 xi: float = 0.3,
                 rho: float = -0.2):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.state = VolatilityState(mean=theta, variance=0.1)

    def predict(self, dt: float) -> None:
        mean = self.state.mean + self.kappa * (self.theta - self.state.mean) * dt
        variance = self.state.variance + (self.xi * self.state.mean) ** 2 * dt
        self.state = VolatilityState(mean=float(mean), variance=float(max(variance, 1e-6)))

    def update(self, observed_return: float, dt: float) -> VolatilityState:
        self.predict(dt)
        measurement_variance = max(observed_return ** 2, 1e-8)

        kalman_gain = self.state.variance / (self.state.variance + measurement_variance)
        mean = self.state.mean + kalman_gain * (abs(observed_return) - self.state.mean)
        variance = (1 - kalman_gain) * self.state.variance

        self.state = VolatilityState(mean=float(max(mean, 1e-6)),
                                     variance=float(max(variance, 1e-8)))
        return self.state


# ---------------------------------------------------------------------------
# 5️⃣  Multi-Asset Covariance Controller
# ---------------------------------------------------------------------------

class MultiAssetCovarianceController:
    """
    Continuously re-solves:

        min_w  wᵀΣw
        s.t.   wᵀ1 = 1,  wᵀμ ≥ μ_target (optional)
    """

    def __init__(self, min_weight: float = -1.0, max_weight: float = 1.0):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimise(self,
                 expected_returns: np.ndarray,
                 covariance_matrix: np.ndarray,
                 target_return: Optional[float] = None) -> np.ndarray:
        n = len(expected_returns)
        cov_inv = np.linalg.pinv(covariance_matrix)

        ones = np.ones(n)
        base_weights = cov_inv @ ones
        base_weights /= ones @ base_weights

        if target_return is not None:
            excess_returns = expected_returns - target_return
            tilt = cov_inv @ excess_returns
            tilt -= np.mean(tilt)
            weights = base_weights + tilt
        else:
            weights = base_weights

        return np.clip(weights, self.min_weight, self.max_weight)


# ---------------------------------------------------------------------------
# 6️⃣  Linear-Quadratic-Gaussian (LQG) Controller
# ---------------------------------------------------------------------------

class LQGController:
    """
    LQG solves the continuous-time control problem:

        ẋ = A x + B u + w_t
        J = E ∫ (xᵀ Q x + uᵀ R u) dt

    The optimal control is u = -K x with K from the Riccati equation.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self._K = self._solve_riccati()

    def _solve_riccati(self) -> np.ndarray:
        P = np.copy(self.Q)
        for _ in range(50):
            BRB = self.B.T @ P @ self.B + self.R
            BRB_inv = np.linalg.pinv(BRB)
            P_dot = self.A.T @ P + P @ self.A - P @ self.B @ BRB_inv @ self.B.T @ P + self.Q
            P = P + 0.01 * P_dot  # simple Euler integration
        K = np.linalg.pinv(self.R) @ self.B.T @ P
        return K

    def optimal_control(self, state: np.ndarray) -> np.ndarray:
        return -self._K @ state

    @property
    def gain(self) -> np.ndarray:
        return self._K


__all__ = [
    "ItoProcessModel",
    "ItoProcessState",
    "DynamicHedgingOptimizer",
    "HJBSolver",
    "StochasticVolatilityFilter",
    "VolatilityState",
    "MultiAssetCovarianceController",
    "LQGController",
]

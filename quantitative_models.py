"""
Anne's Calculus-Based Trading Models
===================================

This module implements the calculus-based approach to trading exactly as Anne would teach it:
Formula → Meaning → Worked Example

Each function follows mathematical rigor with proper derivative calculations,
signal-to-noise ratios, and Taylor expansion forecasting.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from scipy import signal
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CalculusPriceAnalyzer:
    """
    Implements Anne's calculus-based price analysis with exact mathematical formulas.
    """

    def __init__(self, lambda_param: float = 0.6, snr_threshold: float = 1.0):
        """
        Initialize the calculus analyzer with Anne's parameters.

        Args:
            lambda_param: Smoothing parameter (0 < λ < 1) for exponential smoothing
            snr_threshold: Signal-to-noise ratio threshold for valid signals
        """
        self.lambda_param = lambda_param
        self.snr_threshold = snr_threshold
        self.price_history = []

    def exponential_smoothing(self, prices: pd.Series) -> pd.Series:
        """
        1️⃣ Exponential smoothing – making a continuous curve

        Formula (definition):
        P̂ₜ = λPₜ + (1-λ)P̂ₜ₋₁  where (0<λ<1)

        Derivative rule (from calculus):
        dP̂/dt = λ(P - P̂)

        Explanation:
        * This solves a first-order differential equation
        * The term e^(-λt) describes exponential decay of old data
        * Newer prices get heavier weight

        Purpose: creates a smooth, differentiable curve that follows price but removes market noise
        """
        logger.info(f"Applying exponential smoothing with λ = {self.lambda_param}")

        # Implement exact formula: P̂ₜ = λPₜ + (1-λ)P̂ₜ₋₁
        smoothed = pd.Series(index=prices.index, dtype=float)
        smoothed.iloc[0] = prices.iloc[0]  # Initial condition

        for i in range(1, len(prices)):
            smoothed.iloc[i] = (
                self.lambda_param * prices.iloc[i] +
                (1 - self.lambda_param) * smoothed.iloc[i-1]
            )

        logger.info(f"Exponential smoothing completed. Smoothed {len(smoothed)} price points")
        return smoothed

    def calculate_velocity(self, smoothed_prices: pd.Series, delta_t: float = 1.0) -> pd.Series:
        """
        2️⃣ First derivative – instantaneous velocity

        Derivative rule:
        v(t) = dP̂(t)/dt

        Numerical form (finite difference):
        vₜ ≈ (P̂ₜ - P̂ₜ₋Δ)/Δt  or  vₜ ≈ (P̂ₜ₊Δ - P̂ₜ₋Δ)/(2Δt)

        Meaning:
        * v(t) > 0: curve rising → buying pressure
        * v(t) < 0: curve falling → selling pressure
        * v(t) = 0: slope = 0 → potential turning point

        Purpose: measures direction and speed of change — the "gradient" of price
        """
        logger.info("Calculating velocity (first derivative)")

        # Use central finite difference for better accuracy: vₜ ≈ (P̂ₜ₊Δ - P̂ₜ₋Δ)/(2Δt)
        velocity = smoothed_prices.diff() / delta_t

        # Alternative: central difference for interior points
        velocity_central = pd.Series(index=smoothed_prices.index, dtype=float)
        for i in range(1, len(smoothed_prices) - 1):
            velocity_central.iloc[i] = (
                (smoothed_prices.iloc[i+1] - smoothed_prices.iloc[i-1]) / (2 * delta_t)
            )

        # Handle boundaries
        velocity_central.iloc[0] = velocity.iloc[1] if len(velocity) > 1 else 0
        velocity_central.iloc[-1] = velocity.iloc[-1]

        logger.info(f"Velocity calculation completed. Range: [{velocity_central.min():.6f}, {velocity_central.max():.6f}]")
        return velocity_central

    def calculate_acceleration(self, velocity: pd.Series, delta_t: float = 1.0) -> pd.Series:
        """
        3️⃣ Second derivative – acceleration / curvature

        Derivative rule:
        a(t) = d²P̂(t)/dt²

        Numerical form:
        aₜ ≈ (P̂ₜ₊Δ - 2P̂ₜ + P̂ₜ₋Δ)/Δ²

        Interpretation:
        * a(t) > 0: concave up   → buying momentum accelerating
        * a(t) < 0: concave down → selling momentum accelerating
        * crosses 0: inflection   → momentum reversal likely

        Purpose: detects when momentum is speeding up or slowing down
        """
        logger.info("Calculating acceleration (second derivative)")

        # Use central difference for second derivative: aₜ ≈ (vₜ₊Δ - vₜ₋Δ)/(2Δt)
        acceleration = velocity.diff() / delta_t

        # More accurate central difference for interior points
        acceleration_central = pd.Series(index=velocity.index, dtype=float)
        for i in range(1, len(velocity) - 1):
            acceleration_central.iloc[i] = (
                (velocity.iloc[i+1] - velocity.iloc[i-1]) / (2 * delta_t)
            )

        # Handle boundaries
        acceleration_central.iloc[0] = acceleration.iloc[1] if len(acceleration) > 1 else 0
        acceleration_central.iloc[-1] = acceleration.iloc[-1]

        logger.info(f"Acceleration calculation completed. Range: [{acceleration_central.min():.6f}, {acceleration_central.max():.6f}]")
        return acceleration_central

    def calculate_signal_to_noise_ratio(self, velocity: pd.Series, window: int = 14) -> pd.Series:
        """
        4️⃣ Variance – measuring noise

        Statistical rule:
        σᵥ² = Var(vₜ),  σₐ² = Var(aₜ)

        Signal-to-noise ratio (SNR):
        SNRᵥ = |vₜ|/σᵥ

        Interpretation:
        * SNRᵥ > 1: slope change stronger than random variation → statistically valid move
        * SNRᵥ < 1: move is mostly noise

        Purpose: confirms which slopes are real edges, not random wiggles
        """
        logger.info(f"Calculating Signal-to-Noise Ratio with window={window}")

        # Calculate rolling variance of velocity
        velocity_variance = velocity.rolling(window=window, min_periods=1).var()
        velocity_std = np.sqrt(velocity_variance)

        # Calculate SNR: SNRᵥ = |vₜ|/σᵥ
        snr = np.abs(velocity) / velocity_std.replace(0, np.nan)  # Avoid division by zero

        logger.info(f"SNR calculation completed. Mean SNR: {snr.mean():.2f}, SNR > 1: {(snr > self.snr_threshold).sum()}/{len(snr)}")
        return snr, velocity_variance

    def taylor_expansion_forecast(self, smoothed_prices: pd.Series, velocity: pd.Series,
                                 acceleration: pd.Series, delta: float = 1.0) -> pd.Series:
        """
        5️⃣ Taylor expansion – short-term forecast

        Formula (from calculus):
        P̂(t+Δ) = P̂(t) + v(t)Δ + ½a(t)Δ² + higher terms

        Practical version:
        P_pred = Pₜ + vₜ·dt + 0.5·aₜ·dt²

        Meaning: projects the next point of the curve assuming present slope and curvature continue

        Purpose: gives a mathematical "look-ahead" of price for one small interval
        """
        logger.info(f"Calculating Taylor expansion forecast with Δ={delta}")

        # Apply Taylor expansion: P̂(t+Δ) = P̂(t) + v(t)Δ + ½a(t)Δ²
        forecast = smoothed_prices + velocity * delta + 0.5 * acceleration * (delta ** 2)

        logger.info(f"Taylor expansion completed. Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")
        return forecast

    def analyze_price_curve(self, prices: pd.Series) -> pd.DataFrame:
        """
        Complete analysis following Anne's step-by-step approach:
        1. Exponential smoothing → continuous curve
        2. First derivative → velocity
        3. Second derivative → acceleration
        4. Variance → signal-to-noise ratio
        5. Taylor expansion → forecast

        Args:
            prices: Original price series

        Returns:
            DataFrame with all calculus-based indicators
        """
        logger.info("Starting Anne's complete calculus-based price analysis")

        if len(prices) < 20:
            logger.warning(f"Insufficient data points: {len(prices)} < 20")
            return pd.DataFrame()

        # Step 1: Exponential smoothing
        smoothed_prices = self.exponential_smoothing(prices)

        # Step 2: First derivative (velocity)
        velocity = self.calculate_velocity(smoothed_prices)

        # Step 3: Second derivative (acceleration)
        acceleration = self.calculate_acceleration(velocity)

        # Step 4: Signal-to-noise ratio
        snr, velocity_variance = self.calculate_signal_to_noise_ratio(velocity)

        # Step 5: Taylor expansion forecast
        forecast = self.taylor_expansion_forecast(smoothed_prices, velocity, acceleration)

        # Combine all results
        results = pd.DataFrame({
            'price': prices,
            'smoothed_price': smoothed_prices,
            'velocity': velocity,
            'acceleration': acceleration,
            'velocity_variance': velocity_variance,
            'snr': snr,
            'forecast': forecast,
            'valid_signal': snr > self.snr_threshold
        }, index=prices.index)

        logger.info("Complete calculus analysis completed successfully")
        return results

# Legacy functions for backward compatibility
def smooth_price_series(prices: pd.Series, lambda_param: float = 0.6) -> pd.Series:
    """Legacy wrapper for exponential smoothing"""
    analyzer = CalculusPriceAnalyzer(lambda_param=lambda_param)
    return analyzer.exponential_smoothing(prices)

def calculate_velocity(smoothed_prices: pd.Series) -> pd.Series:
    """Legacy wrapper for velocity calculation"""
    analyzer = CalculusPriceAnalyzer()
    return analyzer.calculate_velocity(smoothed_prices)

def calculate_acceleration(velocity: pd.Series) -> pd.Series:
    """Legacy wrapper for acceleration calculation"""
    analyzer = CalculusPriceAnalyzer()
    return analyzer.calculate_acceleration(velocity)

def calculate_velocity_variance(velocity: pd.Series, window: int = 14) -> pd.Series:
    """Legacy wrapper for velocity variance calculation"""
    analyzer = CalculusPriceAnalyzer()
    snr, variance = analyzer.calculate_signal_to_noise_ratio(velocity, window)
    return variance

def taylor_expansion_forecast(p_hat: float, v: float, a: float, delta: int = 1) -> float:
    """Legacy wrapper for single-point Taylor expansion"""
    return p_hat + v * delta + 0.5 * a * (delta ** 2)
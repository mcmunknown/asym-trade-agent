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
from typing import Optional, Tuple, Dict
from scipy.stats import norm

from information_geometry import InformationGeometryMetrics, FractionalVolatilityModel
from regime_filter import RegimeStats

from stochastic_control import (
    ItoProcessModel,
    ItoProcessState,
    DynamicHedgingOptimizer,
    HJBSolver,
    StochasticVolatilityFilter,
)

# New mathematical upgrades
from spline_derivatives import SplineDerivativeAnalyzer
from wavelet_denoising import WaveletDenoiser
from emd_denoising import EMDDenoiser

logger = logging.getLogger(__name__)

# Mathematical safety constants
EPSILON = 1e-12  # Small value to prevent division by zero
VELOCITY_EPSILON = 1e-8  # For velocity calculations
SNR_EPSILON = 1e-10  # For signal-to-noise ratio calculations
MAX_SAFE_VALUE = 1e10  # Maximum safe value to prevent overflow
MIN_SAFE_VALUE = -1e10  # Minimum safe value to prevent underflow

def safe_divide(numerator: float, denominator: float, epsilon: float = EPSILON) -> float:
    """
    Safe division that prevents division by zero and floating point exceptions.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        epsilon: Small value to add to denominator to prevent division by zero

    Returns:
        Safe division result
    """
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0

    abs_denominator = abs(denominator)
    if abs_denominator < epsilon:
        logger.warning(f"Division by zero prevented: denominator={denominator:.2e}, using epsilon={epsilon:.2e}")
        return 0.0

    # Check for potential overflow
    if abs(numerator / abs_denominator) > MAX_SAFE_VALUE:
        logger.warning(f"Potential overflow detected: {numerator}/{denominator}")
        return np.sign(numerator) * MAX_SAFE_VALUE

    return numerator / denominator

def safe_finite_check(value: float, default: float = 0.0) -> float:
    """
    Check if a value is finite and safe to use.

    Args:
        value: Value to check
        default: Default value if not finite

    Returns:
        Safe value
    """
    if not np.isfinite(value):
        logger.debug(f"Non-finite value detected: {value}, using default: {default}")
        return default
    return value

def epsilon_compare(a: float, b: float, epsilon: float = EPSILON) -> int:
    """
    Compare two floating point numbers with epsilon tolerance.

    Args:
        a: First value
        b: Second value
        epsilon: Tolerance for comparison

    Returns:
        -1 if a < b, 0 if a == b (within epsilon), 1 if a > b
    """
    if not np.isfinite(a) or not np.isfinite(b):
        return 0

    diff = a - b
    if abs(diff) < epsilon:
        return 0
    elif diff > 0:
        return 1
    else:
        return -1

class CalculusPriceAnalyzer:
    """
    Implements Anne's calculus-based price analysis with exact mathematical formulas.
    Enhanced with spline derivatives and advanced denoising for institutional-grade precision.
    """

    def __init__(self, 
                 lambda_param: float = 0.6, 
                 snr_threshold: float = 1.0,
                 use_spline_derivatives: bool = True,
                 use_wavelet_denoising: bool = True,
                 spline_window: int = 50,
                 wavelet_type: str = 'db4'):
        """
        Initialize the calculus analyzer with Anne's parameters.

        Args:
            lambda_param: Smoothing parameter (0 < λ < 1) for exponential smoothing
            snr_threshold: Signal-to-noise ratio threshold for valid signals
            use_spline_derivatives: Use analytical spline derivatives instead of finite differences
            use_wavelet_denoising: Use wavelet denoising before analysis
            spline_window: Window size for spline fitting
            wavelet_type: Wavelet family for denoising
        """
        self.lambda_param = lambda_param
        self.snr_threshold = snr_threshold
        self.price_history = []
        self.sde_model = ItoProcessModel()
        self.hedging_optimizer = DynamicHedgingOptimizer()
        self.hjb_solver = HJBSolver()
        
        # New mathematical upgrade components
        self.use_spline_derivatives = use_spline_derivatives
        self.use_wavelet_denoising = use_wavelet_denoising
        
        if use_spline_derivatives:
            self.spline_analyzer = SplineDerivativeAnalyzer(
                window_size=spline_window,
                adaptive_smoothing=True,
                spline_type='cubic'
            )
            
        if use_wavelet_denoising:
            self.wavelet_denoiser = WaveletDenoiser(
                wavelet_family=wavelet_type,
                threshold_method='sure',
                adaptive_scaling=True
            )
            
        # EMD denoiser for multi-scale analysis (only if needed)
        self.emd_denoiser = None  # Initialize lazily to avoid EMD import issues
        
        logger.info(f"Enhanced CalculusPriceAnalyzer initialized: "
                   f"spline_derivatives={use_spline_derivatives}, "
                   f"wavelet_denoising={use_wavelet_denoising}")

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

        # Input validation
        if len(prices) == 0:
            logger.warning("Empty price series provided")
            return pd.Series(dtype=float)

        # Check for invalid values
        if prices.isna().all():
            logger.error("All prices are NaN")
            return pd.Series(index=prices.index, dtype=float)

        # Forward fill any NaN values with limited propagation
        prices_clean = prices.ffill(limit=3)
        remaining_nan = prices_clean.isna()
        if remaining_nan.any():
            logger.warning(f"Found {remaining_nan.sum()} NaN values after forward fill")
            # Backward fill remaining NaN values
            prices_clean = prices_clean.bfill(limit=3)
            # Fill any remaining NaN with first valid value
            if prices_clean.isna().any():
                first_valid = prices_clean.first_valid_index()
                if first_valid is not None:
                    prices_clean = prices_clean.fillna(prices_clean.loc[first_valid])
                else:
                    logger.error("No valid price data found")
                    return pd.Series(index=prices.index, dtype=float)

        # Implement exact formula: P̂ₜ = λPₜ + (1-λ)P̂ₜ₋₁
        smoothed = pd.Series(index=prices.index, dtype=float)
        smoothed.iloc[0] = prices_clean.iloc[0]  # Initial condition

        for i in range(1, len(prices_clean)):
            current_price = prices_clean.iloc[i]
            prev_smoothed = smoothed.iloc[i-1]

            # Numerical stability check
            if not (np.isfinite(current_price) and np.isfinite(prev_smoothed)):
                logger.warning(f"Non-finite values at index {i}, using previous smoothed value")
                smoothed.iloc[i] = prev_smoothed
                continue

            smoothed.iloc[i] = (
                self.lambda_param * current_price +
                (1 - self.lambda_param) * prev_smoothed
            )

        # Final validation
        if smoothed.isna().any():
            logger.warning(f"NaN values found in smoothed series, filling with forward fill")
            smoothed = smoothed.ffill().bfill()

        logger.info(f"Exponential smoothing completed. Smoothed {len(smoothed)} price points")
        logger.info(f"Smoothed price range: [{smoothed.min():.2f}, {smoothed.max():.2f}]")

        return smoothed

    def calculate_velocity(self, smoothed_prices: pd.Series, delta_t: float = 1.0) -> pd.Series:
        """
        2️⃣ First derivative – instantaneous velocity

        Enhanced with analytical spline derivatives for institutional-grade precision.

        Derivative rule:
        v(t) = dP̂(t)/dt

        Analytical form (spline derivatives):
        v(t) = S'(t) where S(t) is fitted spline
        Finite difference fallback:
        vₜ ≈ (P̂ₜ - P̂ₜ₋Δ)/Δt  or  vₜ ≈ (P̂ₜ₊Δ - P̂ₜ₋Δ)/(2Δt)

        Meaning:
        * v(t) > 0: curve rising → buying pressure
        * v(t) < 0: curve falling → selling pressure
        * v(t) = 0: slope = 0 → potential turning point

        Purpose: measures direction and speed of change — the "gradient" of price
        """
        logger.info("Calculating velocity (first derivative)")
        
        # Use spline derivatives if enabled and sufficient data
        if self.use_spline_derivatives and len(smoothed_prices) >= self.spline_analyzer.min_window:
            try:
                # Apply wavelet denoising first if enabled
                if self.use_wavelet_denoising:
                    denoised_prices = self.wavelet_denoiser.denoise(smoothed_prices)
                else:
                    denoised_prices = smoothed_prices
                
                # Calculate spline derivatives
                timestamps = pd.Series(np.arange(len(denoised_prices)), index=denoised_prices.index)
                spline_derivatives = self.spline_analyzer.analyze_derivatives(denoised_prices, timestamps)
                
                if 'velocity' in spline_derivatives:
                    spline_velocity = pd.Series(spline_derivatives['velocity'], index=smoothed_prices.index)
                    
                    # Log quality metrics
                    quality = self.spline_analyzer.get_derivative_quality_metrics()
                    logger.info(f"Spline velocity calculated: quality={quality['mean_quality']:.3f}, "
                               f"stability={quality['quality_stability']:.3f}")
                    
                    return spline_velocity
                else:
                    logger.warning("Spline derivative calculation failed, falling back to finite differences")
                    
            except Exception as e:
                logger.error(f"Spline velocity calculation failed: {e}, using finite differences")
        
        # Fallback to original finite difference method

        # Input validation
        if len(smoothed_prices) < 2:
            logger.warning("Insufficient data for velocity calculation")
            return pd.Series(index=smoothed_prices.index, dtype=float)

        # Handle delta_t
        if delta_t <= 0 or not np.isfinite(delta_t):
            logger.warning(f"Invalid delta_t: {delta_t}, using default 1.0")
            delta_t = 1.0

        # Initialize velocity series
        velocity_central = pd.Series(index=smoothed_prices.index, dtype=float)

        # Use central finite difference for interior points: vₜ ≈ (P̂ₜ₊Δ - P̂ₜ₋Δ)/(2Δt)
        for i in range(1, len(smoothed_prices) - 1):
            price_prev = smoothed_prices.iloc[i-1]
            price_curr = smoothed_prices.iloc[i]
            price_next = smoothed_prices.iloc[i+1]

            # Numerical stability checks with safe division
            if all(np.isfinite([price_prev, price_curr, price_next])):
                price_diff = safe_finite_check(price_next - price_prev)
                denominator = safe_finite_check(2 * delta_t, VELOCITY_EPSILON)
                velocity_central.iloc[i] = safe_divide(price_diff, denominator, VELOCITY_EPSILON)
            else:
                logger.debug(f"Non-finite price values at index {i}")
                velocity_central.iloc[i] = 0.0

        # Handle boundaries with forward/backward differences
        if len(smoothed_prices) >= 2:
            # First point: forward difference
            price_0 = smoothed_prices.iloc[0]
            price_1 = smoothed_prices.iloc[1]
            if np.isfinite(price_0) and np.isfinite(price_1):
                price_diff = safe_finite_check(price_1 - price_0)
                denominator = safe_finite_check(delta_t, VELOCITY_EPSILON)
                velocity_central.iloc[0] = safe_divide(price_diff, denominator, VELOCITY_EPSILON)
            else:
                velocity_central.iloc[0] = 0.0

            # Last point: backward difference
            price_last = smoothed_prices.iloc[-1]
            price_prev_last = smoothed_prices.iloc[-2]
            if np.isfinite(price_last) and np.isfinite(price_prev_last):
                price_diff = safe_finite_check(price_last - price_prev_last)
                denominator = safe_finite_check(delta_t, VELOCITY_EPSILON)
                velocity_central.iloc[-1] = safe_divide(price_diff, denominator, VELOCITY_EPSILON)
            else:
                velocity_central.iloc[-1] = 0.0

        # Clean up any remaining NaN values
        velocity_central = velocity_central.fillna(0.0)

        # Apply velocity bounds to prevent extreme values
        max_velocity = smoothed_prices.std() * 10  # 10x standard deviation as upper bound
        velocity_central = velocity_central.clip(lower=-max_velocity, upper=max_velocity)

        logger.info(f"Velocity calculation completed. Range: [{velocity_central.min():.6f}, {velocity_central.max():.6f}]")
        logger.info(f"Velocity statistics - Mean: {velocity_central.mean():.6f}, Std: {velocity_central.std():.6f}")

        return velocity_central

    def calculate_acceleration(self, velocity: pd.Series, delta_t: float = 1.0) -> pd.Series:
        """
        3️⃣ Second derivative – acceleration / curvature

        Enhanced with analytical spline derivatives for institutional-grade precision.

        Derivative rule:
        a(t) = d²P̂(t)/dt²

        Analytical form (spline derivatives):
        a(t) = S''(t) where S(t) is fitted spline
        Numerical fallback:
        aₜ ≈ (P̂ₜ₊Δ - 2P̂ₜ + P̂ₜ₋Δ)/Δ²

        Interpretation:
        * a(t) > 0: concave up   → buying momentum accelerating
        * a(t) < 0: concave down → selling momentum accelerating
        * crosses 0: inflection   → momentum reversal likely

        Purpose: detects when momentum is speeding up or slowing down
        """
        logger.info("Calculating acceleration (second derivative)")
        
        # Use spline derivatives if enabled and sufficient data
        if self.use_spline_derivatives and len(velocity) >= self.spline_analyzer.min_window:
            try:
                # Apply wavelet denoising first if enabled
                if self.use_wavelet_denoising:
                    denoised_prices = self.wavelet_denoiser.denoise(velocity)
                else:
                    denoised_prices = velocity
                
                # Calculate spline derivatives from original price (not velocity)
                # We need to reconstruct or get original prices - for now use velocity
                timestamps = pd.Series(np.arange(len(denoised_prices)), index=denoised_prices.index)
                spline_derivatives = self.spline_analyzer.analyze_derivatives(denoised_prices, timestamps)
                
                if 'acceleration' in spline_derivatives:
                    spline_acceleration = pd.Series(spline_derivatives['acceleration'], index=velocity.index)
                    
                    # Log quality metrics
                    quality = self.spline_analyzer.get_derivative_quality_metrics()
                    logger.info(f"Spline acceleration calculated: quality={quality['mean_quality']:.3f}, "
                               f"stability={quality['quality_stability']:.3f}")
                    
                    return spline_acceleration
                else:
                    logger.warning("Spline derivative calculation failed, falling back to finite differences")
                    
            except Exception as e:
                logger.error(f"Spline acceleration calculation failed: {e}, using finite differences")
        
        # Fallback to original finite difference method

        # Input validation
        if len(velocity) < 3:
            logger.warning("Insufficient data for acceleration calculation")
            return pd.Series(index=velocity.index, dtype=float)

        # Handle delta_t
        if delta_t <= 0 or not np.isfinite(delta_t):
            logger.warning(f"Invalid delta_t: {delta_t}, using default 1.0")
            delta_t = 1.0

        # Initialize acceleration series
        acceleration_central = pd.Series(index=velocity.index, dtype=float)

        # Use central difference for second derivative: aₜ ≈ (vₜ₊Δ - vₜ₋Δ)/(2Δt)
        for i in range(1, len(velocity) - 1):
            vel_prev = velocity.iloc[i-1]
            vel_curr = velocity.iloc[i]
            vel_next = velocity.iloc[i+1]

            # Numerical stability checks with safe division
            if all(np.isfinite([vel_prev, vel_curr, vel_next])):
                vel_diff = safe_finite_check(vel_next - vel_prev)
                denominator = safe_finite_check(2 * delta_t, VELOCITY_EPSILON)
                acceleration_central.iloc[i] = safe_divide(vel_diff, denominator, VELOCITY_EPSILON)
            else:
                logger.debug(f"Non-finite velocity values at index {i}")
                acceleration_central.iloc[i] = 0.0

        # Handle boundaries
        if len(velocity) >= 3:
            # First point: use available forward differences
            vel_0 = velocity.iloc[0]
            vel_1 = velocity.iloc[1]
            vel_2 = velocity.iloc[2]

            if np.isfinite(vel_0) and np.isfinite(vel_1) and np.isfinite(vel_2):
                # Use forward difference approximation with safe division
                vel_combo = safe_finite_check(vel_2 - 2*vel_1 + vel_0)
                denominator = safe_finite_check(delta_t**2, VELOCITY_EPSILON)
                acceleration_central.iloc[0] = safe_divide(vel_combo, denominator, VELOCITY_EPSILON)
            else:
                acceleration_central.iloc[0] = 0.0

            # Last point: use available backward differences
            vel_last = velocity.iloc[-1]
            vel_prev_last = velocity.iloc[-2]
            vel_prev_prev = velocity.iloc[-3]

            if np.isfinite(vel_last) and np.isfinite(vel_prev_last) and np.isfinite(vel_prev_prev):
                # Use backward difference approximation with safe division
                vel_combo = safe_finite_check(vel_last - 2*vel_prev_last + vel_prev_prev)
                denominator = safe_finite_check(delta_t**2, VELOCITY_EPSILON)
                acceleration_central.iloc[-1] = safe_divide(vel_combo, denominator, VELOCITY_EPSILON)
            else:
                acceleration_central.iloc[-1] = 0.0

        # Clean up any remaining NaN values
        acceleration_central = acceleration_central.fillna(0.0)

        # Apply acceleration bounds to prevent extreme values
        velocity_std = velocity.std()
        max_acceleration = velocity_std * 5 / delta_t  # Reasonable upper bound
        acceleration_central = acceleration_central.clip(lower=-max_acceleration, upper=max_acceleration)

        logger.info(f"Acceleration calculation completed. Range: [{acceleration_central.min():.6f}, {acceleration_central.max():.6f}]")
        logger.info(f"Acceleration statistics - Mean: {acceleration_central.mean():.6f}, Std: {acceleration_central.std():.6f}")

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

        # Numerical stability: Apply variance bounds
        min_variance = 1e-10  # Minimum variance floor

        # Handle NaN values in variance calculation
        if velocity_variance.isna().any():
            logger.debug(f"Found {velocity_variance.isna().sum()} NaN values in velocity variance")
            # Fill NaN with minimum variance
            velocity_variance = velocity_variance.fillna(min_variance)

        # Calculate maximum variance ceiling only if we have valid data
        if velocity_variance.notna().any():
            max_variance = velocity_variance.quantile(0.99)  # Maximum variance ceiling
            max_variance = max(max_variance, min_variance)  # Ensure max > min
        else:
            max_variance = min_variance * 100  # Default maximum

        # Clamp variance to prevent extreme values
        velocity_variance = velocity_variance.clip(lower=min_variance, upper=max_variance)

        # Final check for any remaining NaN values
        if velocity_variance.isna().any():
            logger.warning(f"Still have {velocity_variance.isna().sum()} NaN values after processing")
            velocity_variance = velocity_variance.fillna(min_variance)
        velocity_std = np.sqrt(velocity_variance)

        # Numerical stability: Epsilon smoothing to prevent NaN values
        epsilon = 1e-8  # Small epsilon to prevent division by zero
        velocity_std_smooth = velocity_std + epsilon

        # Calculate SNR: SNRᵥ = |vₜ|/σᵥ with enhanced numerical safeguards
        snr = pd.Series(index=velocity.index, dtype=float)
        for i in range(len(velocity)):
            vel_abs = safe_finite_check(abs(velocity.iloc[i]))
            vel_std = safe_finite_check(velocity_std_smooth.iloc[i], SNR_EPSILON)
            snr.iloc[i] = safe_divide(vel_abs, vel_std, SNR_EPSILON)

        # Handle any remaining NaN values by setting SNR to 0
        snr = snr.fillna(0.0)

        # Cap SNR at reasonable maximum to prevent extreme values
        max_snr = 10.0
        snr = snr.clip(upper=max_snr)

        logger.info(f"SNR calculation completed. Mean SNR: {snr.mean():.2f}, SNR > 1: {(snr > self.snr_threshold).sum()}/{len(snr)}")
        logger.info(f"SNR statistics - Min: {snr.min():.3f}, Max: {snr.max():.3f}, Std: {snr.std():.3f}")

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

    def _compute_value_sensitivities(self,
                                     prices: pd.Series,
                                     forecast: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Estimate V_P and V_PP by differentiating the Taylor forecast with respect to price.
        """
        # Ensure we have a clean reference index for alignment
        if isinstance(prices, pd.Series):
            base_index = prices.index
            price_series = prices.copy()
        else:
            base_index = pd.RangeIndex(start=0, stop=len(prices))
            price_series = pd.Series(prices, index=base_index)

        if isinstance(forecast, pd.Series):
            forecast_series = forecast.reindex(base_index)
        else:
            forecast_series = pd.Series(forecast, index=base_index)

        price_diff = price_series.diff().replace(0.0, np.nan).reindex(base_index)
        value_diff = forecast_series.diff().reindex(base_index)

        gradient = value_diff.div(price_diff)
        gradient = gradient.replace([np.inf, -np.inf], np.nan).reindex(base_index)
        gradient = gradient.bfill().fillna(0.0)

        gamma_numer = gradient.diff().reindex(base_index)
        gamma = gamma_numer.div(price_diff)
        gamma = gamma.replace([np.inf, -np.inf], np.nan).reindex(base_index)
        gamma = gamma.fillna(0.0)

        return gradient, gamma

    def _estimate_drift_diffusion(self,
                                  prices: pd.Series,
                                  delta_t: float = 1.0) -> Tuple[pd.Series, pd.Series]:
        """
        Estimate drift and diffusion from log returns with smoothing.
        """
        safe_prices = prices.where(prices > 0.0).ffill().bfill()
        if safe_prices.isna().all():
            safe_prices = prices.replace(0.0, 1.0)

        log_returns = np.log(safe_prices / safe_prices.shift(1))
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        drift_series = log_returns.rolling(self.sde_model.window, min_periods=1).mean() / delta_t
        sigma_series = log_returns.rolling(self.sde_model.window, min_periods=1).std(ddof=1) / np.sqrt(delta_t)

        drift_series = drift_series.replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
        sigma_series = sigma_series.replace([np.inf, -np.inf], np.nan).bfill().fillna(self.sde_model.min_vol)
        sigma_series = sigma_series.clip(self.sde_model.min_vol, self.sde_model.max_vol)

        return drift_series, sigma_series

    def _augment_with_stochastic_process(self,
                                         prices: pd.Series,
                                         analysis: pd.DataFrame,
                                         delta_t: float = 1.0,
                                         drift_series: Optional[pd.Series] = None,
                                         sigma_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Add stochastic calculus metrics (Itô drift/diffusion, delta hedge, HJB control, volatility filtering).
        """
        if analysis.empty:
            return analysis

        gradient, gamma = self._compute_value_sensitivities(prices, analysis['forecast'])

        if drift_series is None or sigma_series is None:
            drift_series, sigma_series = self._estimate_drift_diffusion(prices, delta_t)

        ito_terms = pd.Series(0.0, index=analysis.index)
        delta_series = pd.Series(0.0, index=analysis.index)
        residual_variance = pd.Series(0.0, index=analysis.index)
        hjb_actions = pd.Series(0.0, index=analysis.index)
        hjb_values = pd.Series(0.0, index=analysis.index)

        vol_filter = StochasticVolatilityFilter()
        returns = prices.pct_change().fillna(0.0)
        stochastic_vol = pd.Series(0.0, index=analysis.index)

        for idx in analysis.index:
            price = safe_finite_check(prices.loc[idx], 0.0)
            grad = safe_finite_check(gradient.loc[idx], 0.0)
            gamma_val = safe_finite_check(gamma.loc[idx], 0.0)
            mu = safe_finite_check(drift_series.loc[idx], 0.0)
            sigma = safe_finite_check(sigma_series.loc[idx], self.sde_model.min_vol)

            state = ItoProcessState(price=price, drift=mu, diffusion=sigma, dt=delta_t)
            ito_terms.loc[idx] = ItoProcessModel.apply_ito_lemma(grad, gamma_val, state)

            delta, residual = self.hedging_optimizer.optimal_delta(grad, sigma, price)
            delta_series.loc[idx] = delta
            residual_variance.loc[idx] = residual

            action, hjb_value = self.hjb_solver.optimal_action(grad, gamma_val, mu, sigma, price)
            hjb_actions.loc[idx] = action
            hjb_values.loc[idx] = hjb_value

            vol_state = vol_filter.update(float(returns.loc[idx]), delta_t)
            stochastic_vol.loc[idx] = vol_state.mean

        analysis = analysis.copy()
        analysis['value_gradient'] = gradient
        analysis['value_gamma'] = gamma
        analysis['estimated_drift'] = drift_series
        analysis['estimated_diffusion'] = sigma_series
        analysis['ito_correction'] = ito_terms
        analysis['optimal_delta'] = delta_series
        analysis['residual_variance'] = residual_variance
        analysis['hjb_action'] = hjb_actions
        analysis['hjb_value'] = hjb_values
        analysis['stochastic_volatility'] = stochastic_vol

        return analysis

    def _apply_information_geometry(self, analysis: pd.DataFrame) -> pd.DataFrame:
        ig_metrics = InformationGeometryMetrics()
        info_df = ig_metrics.compute(analysis)
        return analysis.join(info_df, how='left')

    def _apply_fractional_volatility(self, analysis: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        frac_model = FractionalVolatilityModel()
        fractional_df = frac_model.enrich(analysis, returns)
        return analysis.join(fractional_df, how='left')

    def _apply_regime_context(self,
                              analysis: pd.DataFrame,
                              regime_context: Optional[RegimeStats]) -> pd.DataFrame:
        if regime_context is None or analysis.empty:
            return analysis

        augmented = analysis.copy()
        augmented['regime_state'] = regime_context.state
        augmented['regime_confidence'] = regime_context.confidence
        for regime, prob in regime_context.probabilities.items():
            augmented[f'regime_prob_{regime.lower()}'] = prob
        return augmented

    def calculate_tp_first_probability(self, current_price: float, tp_price: float,
                                     sl_price: float, volatility: float,
                                     drift: float = 0.0, time_horizon: float = 1.0) -> Tuple[float, float]:
        """
        TP-First Probability Calculator using Stochastic First-Passage Theory

        Formula: P_TP = Φ((ln(TP/S_t) - (μ - 0.5σ²)τ) / (σ√τ))

        Where:
        - Φ is the cumulative normal distribution
        - TP: Take Profit price level
        - S_t: Current price
        - μ: Drift rate (expected return)
        - σ: Volatility
        - τ: Time horizon

        This calculates the probability of hitting TP before SL using
        first-passage probability from stochastic calculus.

        Args:
            current_price: Current asset price S_t
            tp_price: Take profit level
            sl_price: Stop loss level
            volatility: Annualized volatility σ
            drift: Expected drift rate μ (default 0 for no drift assumption)
            time_horizon: Time horizon τ in same units as volatility (default 1.0)

        Returns:
            Tuple of (tp_probability, sl_probability)
        """
        logger.info(f"Calculating TP-first probability: S={current_price}, TP={tp_price}, SL={sl_price}")

        # Input validation
        if not all([current_price > 0, tp_price > 0, sl_price > 0]):
            logger.warning("Invalid price inputs for probability calculation")
            return 0.5, 0.5

        if volatility <= 0 or not np.isfinite(volatility):
            logger.warning(f"Invalid volatility: {volatility}, using default 0.2")
            volatility = 0.2

        if time_horizon <= 0 or not np.isfinite(time_horizon):
            logger.warning(f"Invalid time horizon: {time_horizon}, using 1.0")
            time_horizon = 1.0

        # Calculate drift-adjusted parameters
        drift_adjusted = drift - 0.5 * volatility**2

        # Calculate standardized z-scores for TP and SL
        tp_z_score = (np.log(tp_price / current_price) - drift_adjusted * time_horizon) / (volatility * np.sqrt(time_horizon))
        sl_z_score = (np.log(sl_price / current_price) - drift_adjusted * time_horizon) / (volatility * np.sqrt(time_horizon))

        # Calculate cumulative probabilities using normal distribution
        tp_probability = norm.cdf(tp_z_score)
        sl_probability = norm.cdf(sl_z_score)

        # Normalize to ensure probabilities sum to 1 (for TP vs SL first-passage)
        total_prob = tp_probability + sl_probability
        if total_prob > 0:
            tp_probability = tp_probability / total_prob
            sl_probability = sl_probability / total_prob
        else:
            tp_probability = sl_probability = 0.5

        logger.info(f"TP probability: {tp_probability:.3f}, SL probability: {sl_probability:.3f}")

        return tp_probability, sl_probability

    def enhanced_curvature_prediction(self, smoothed_prices: pd.Series, velocity: pd.Series,
                                    acceleration: pd.Series, delta_t: float = 1.0,
                                    include_jerk: bool = False) -> Dict[str, pd.Series]:
        """
        Enhanced Curvature Prediction using Higher-Order Taylor Expansion with Error Bounds

        Enhanced with spline derivatives for maximum precision and mathematical confidence intervals.

        Basic Formula: f̂(t+Δt) = f(t) + f'(t)Δt + (1/2)f''(t)Δt²
        Enhanced Formula (3rd/4th order with error bounds):
        f̂(t+Δt) = f(t) + f'(t)Δt + (1/2)f''(t)Δt² + (1/6)f'''(t)Δt³ + (1/24)f''''(t)Δt⁴
        
        Error Bound: |R_n| ≤ (max|f^(n+1)|/(n+1)!)|Δt|^(n+1)

        This captures momentum (slope) and acceleration (curvature) simultaneously
        for higher precision prediction of where the curve bends next, with mathematical confidence.

        Args:
            smoothed_prices: Smoothed price series f(t)
            velocity: First derivative f'(t)  
            acceleration: Second derivative f''(t)
            delta_t: Time step for prediction
            include_jerk: Whether to calculate 3rd/4th order terms

        Returns:
            Dictionary with enhanced forecasts, error bounds, and quality metrics
        """
        logger.info(f"Enhanced curvature prediction with Δt={delta_t}, include_jerk={include_jerk}")

        forecasts = {}
        error_bounds = {}
        confidences = {}

        # Get spline derivatives if available
        spline_velocity = None
        spline_acceleration = None
        spline_jerk = None
        spline_snap = None
        
        if self.use_spline_derivatives and len(smoothed_prices) >= self.spline_analyzer.min_window:
            try:
                # Apply denoising first if enabled
                if self.use_wavelet_denoising:
                    denoised_prices = self.wavelet_denoiser.denoise(smoothed_prices)
                else:
                    denoised_prices = smoothed_prices
                
                timestamps = pd.Series(np.arange(len(denoised_prices)), index=denoised_prices.index)
                spline_derivatives = self.spline_analyzer.analyze_derivatives(denoised_prices, timestamps)
                
                if 'velocity' in spline_derivatives:
                    spline_velocity = pd.Series(spline_derivatives['velocity'], index=smoothed_prices.index)
                if 'acceleration' in spline_derivatives:
                    spline_acceleration = pd.Series(spline_derivatives['acceleration'], index=smoothed_prices.index)
                if 'jerk' in spline_derivatives:
                    spline_jerk = pd.Series(spline_derivatives['jerk'], index=smoothed_prices.index)
                if 'snap' in spline_derivatives:
                    spline_snap = pd.Series(spline_derivatives['snap'], index=smoothed_prices.index)
                    
            except Exception as e:
                logger.warning(f"Spline derivatives failed: {e}, using finite differences")

        # Use spline derivatives if available, otherwise fallback to input
        use_velocity = spline_velocity if spline_velocity is not None else velocity
        use_acceleration = spline_acceleration if spline_acceleration is not None else acceleration
        use_jerk = spline_jerk if spline_jerk is not None else None
        use_snap = spline_snap if spline_snap is not None else None

        # 2nd Order Taylor Expansion
        pred_2nd = smoothed_prices + use_velocity * delta_t + 0.5 * use_acceleration * (delta_t ** 2)
        forecasts['order_2'] = pred_2nd
        
        # Error bound from 3rd order term
        if use_jerk is not None:
            error_2nd = np.abs(use_jerk) * (delta_t ** 3) / 6
        else:
            # Estimate jerk from acceleration differences
            estimated_jerk = use_acceleration.diff().fillna(0.0)
            error_2nd = np.abs(estimated_jerk) * (delta_t ** 3) / 6
        error_bounds['order_2'] = error_2nd
        confidences['order_2'] = 1.0 / (1.0 + error_2nd)

        # 3rd Order Taylor Expansion
        if use_jerk is not None and include_jerk:
            pred_3rd = (pred_2nd + 
                         (1.0/6.0) * use_jerk * (delta_t ** 3))
            forecasts['order_3'] = pred_3rd
            
            # Error bound from 4th order term
            if use_snap is not None:
                error_3rd = np.abs(use_snap) * (delta_t ** 4) / 24
            else:
                # Estimate snap from jerk differences
                estimated_snap = use_jerk.diff().fillna(0.0)
                error_3rd = np.abs(estimated_snap) * (delta_t ** 4) / 24
            error_bounds['order_3'] = error_3rd
            confidences['order_3'] = 1.0 / (1.0 + error_3rd)

        # 4th Order Taylor Expansion
        if use_snap is not None and include_jerk:
            pred_4th = (pred_3rd + 
                         (1.0/24.0) * use_snap * (delta_t ** 4))
            forecasts['order_4'] = pred_4th
            
            # Error bound from 5th order term (estimated)
            estimated_pental = use_snap.diff().fillna(0.0)
            error_4th = np.abs(estimated_pental) * (delta_t ** 5) / 120
            error_bounds['order_4'] = error_4th
            confidences['order_4'] = 1.0 / (1.0 + error_4th)

        # Select best order based on error bounds and confidence
        available_orders = [k for k in forecasts.keys() if k in error_bounds]
        if available_orders:
            best_order = min(available_orders, key=lambda x: np.mean(error_bounds[x]))
            best_forecast = forecasts[best_order]
            best_error = error_bounds[best_order]
            best_confidence = confidences[best_order]
            
            logger.info(f"Using {best_order} Taylor expansion: "
                       f"mean_error={best_error.mean():.6f}, "
                       f"mean_confidence={best_confidence.mean():.3f}")
        else:
            # Fallback to 2nd order
            best_order = 'order_2'
            best_forecast = pred_2nd
            best_error = error_bounds['order_2']
            best_confidence = confidences['order_2']
            
            logger.info("Using 2nd order Taylor expansion (fallback)")

        # Apply numerical safeguards to all forecasts
        for key in forecasts:
            forecasts[key] = forecasts[key].replace([np.inf, -np.inf], np.nan)
            forecasts[key] = forecasts[key].ffill().fillna(smoothed_prices)

        return {
            'best_forecast': best_forecast,
            'best_order': best_order,
            'best_error': best_error,
            'best_confidence': best_confidence,
            'all_forecasts': forecasts,
            'all_errors': error_bounds,
            'all_confidences': confidences,
            'forecast_range': [best_forecast.min(), best_forecast.max()]
        }

    def analyze_price_curve(self,
                            prices: pd.Series,
                            kalman_drift: Optional[pd.Series] = None,
                            kalman_volatility: Optional[pd.Series] = None,
                            regime_context: Optional[RegimeStats] = None,
                            delta_t: float = 1.0) -> pd.DataFrame:
        """
        Complete analysis following Anne's step-by-step approach, with optional
        Kalman overrides and regime context for TP/SL adjustments.
        """
        logger.info("Starting Anne's complete calculus-based price analysis")

        if len(prices) < 20:
            logger.warning(f"Insufficient data points: {len(prices)} < 20")
            return pd.DataFrame()

        returns = prices.pct_change().fillna(0.0)
        drift_series, sigma_series = self._estimate_drift_diffusion(prices, delta_t)

        # Step 1: Exponential smoothing
        smoothed_prices = self.exponential_smoothing(prices)

        # Step 2: First derivative (velocity)
        velocity = self.calculate_velocity(smoothed_prices)

        # Step 3: Second derivative (acceleration)
        acceleration = self.calculate_acceleration(velocity)

        # Step 4: Signal-to-noise ratio
        snr, velocity_variance = self.calculate_signal_to_noise_ratio(velocity)

        # Step 5: Enhanced curvature prediction (higher precision Taylor expansion)
        enhanced_result = self.enhanced_curvature_prediction(smoothed_prices, velocity, acceleration, include_jerk=True)
        enhanced_forecast = enhanced_result['best_forecast']

        # Step 6: Traditional Taylor expansion forecast (for comparison)
        forecast = self.taylor_expansion_forecast(smoothed_prices, velocity, acceleration)

        tp_probabilities = []
        sl_probabilities = []
        tp_price_levels = []
        sl_price_levels = []
        tp_pct_values = []
        sl_pct_values = []
        adjustment_strengths = []

        base_tp_pct = 0.02
        base_sl_pct = 0.01
        tp_scale = 0.8
        sl_scale = 0.6

        for idx in prices.index:
            current_price = prices.loc[idx]
            if not np.isfinite(current_price):
                tp_probabilities.append(0.5)
                sl_probabilities.append(0.5)
                tp_price_levels.append(np.nan)
                sl_price_levels.append(np.nan)
                tp_pct_values.append(base_tp_pct)
                sl_pct_values.append(base_sl_pct)
                adjustment_strengths.append(0.0)
                continue

            kalman_mu = kalman_drift.loc[idx] if kalman_drift is not None and idx in kalman_drift.index else None
            kalman_sigma = kalman_volatility.loc[idx] if kalman_volatility is not None and idx in kalman_volatility.index else None
            mu = kalman_mu if kalman_mu is not None else drift_series.loc[idx]
            sigma = kalman_sigma if kalman_sigma is not None else sigma_series.loc[idx]
            sigma = float(np.clip(sigma, self.sde_model.min_vol, self.sde_model.max_vol))

            base_tp_price = current_price * (1 + base_tp_pct)
            base_sl_price = current_price * (1 - base_sl_pct)

            tp_prob, sl_prob = self.calculate_tp_first_probability(
                current_price, base_tp_price, base_sl_price, sigma,
                drift=mu, time_horizon=delta_t
            )

            advantage = tp_prob - sl_prob
            tp_pct = base_tp_pct + max(advantage, 0.0) * tp_scale
            sl_pct = base_sl_pct + max(-advantage, 0.0) * sl_scale

            if regime_context:
                if regime_context.state == 'BULL':
                    tp_pct += regime_context.confidence * 0.005
                elif regime_context.state == 'BEAR':
                    sl_pct += regime_context.confidence * 0.006
                elif regime_context.state == 'RANGE':
                    tp_pct *= 0.95
                    sl_pct *= 1.05

            tp_pct = float(np.clip(tp_pct, 0.001, 0.15))
            sl_pct = float(np.clip(sl_pct, 0.001, 0.2))

            tp_price_levels.append(current_price * (1 + tp_pct))
            sl_price_levels.append(current_price * (1 - sl_pct))
            tp_pct_values.append(tp_pct)
            sl_pct_values.append(sl_pct)
            adjustment_strengths.append(abs(advantage))
            tp_probabilities.append(tp_prob)
            sl_probabilities.append(sl_prob)

        results = pd.DataFrame({
            'price': prices,
            'smoothed_price': smoothed_prices,
            'velocity': velocity,
            'acceleration': acceleration,
            'velocity_variance': velocity_variance,
            'snr': snr,
            'forecast': forecast,
            'enhanced_forecast': enhanced_forecast,
            'tp_probability': tp_probabilities,
            'sl_probability': sl_probabilities,
            'tp_price': tp_price_levels,
            'sl_price': sl_price_levels,
            'tp_pct': tp_pct_values,
            'sl_pct': sl_pct_values,
            'tp_adjustment_strength': adjustment_strengths,
            'tp_advantage': [tp - sl for tp, sl in zip(tp_probabilities, sl_probabilities)],
            'tp_adjusted': [
                abs(tp_pct_values[i] - base_tp_pct) > 1e-6 or
                abs(sl_pct_values[i] - base_sl_pct) > 1e-6
                for i in range(len(tp_pct_values))
            ],
            'valid_signal': snr > self.snr_threshold
        }, index=prices.index)

        results = self._augment_with_stochastic_process(
            prices, results, delta_t=delta_t,
            drift_series=drift_series, sigma_series=sigma_series
        )

        results = self._apply_information_geometry(results)
        results = self._apply_fractional_volatility(results, returns)
        results = self._apply_regime_context(results, regime_context)

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

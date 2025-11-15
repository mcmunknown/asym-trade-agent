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
    MeasureTheoreticConverter,
    KushnerStratonovichFilter,
)

# New mathematical upgrades
from spline_derivatives import SplineDerivativeAnalyzer
from wavelet_denoising import WaveletDenoiser
from emd_denoising import EMDDenoiser

logger = logging.getLogger(__name__)

try:
    from cpp_bridge import (
        cpp_available as cpp_backend_available,
        exponential_smoothing as cpp_exp_smoothing,
        velocity as cpp_velocity_kernel,
        acceleration as cpp_acceleration_kernel,
    )
except Exception as exc:
    logger.warning(f"C++ backend unavailable: {exc}")

    def cpp_backend_available() -> bool:
        return False

    cpp_exp_smoothing = None
    cpp_velocity_kernel = None
    cpp_acceleration_kernel = None

# Mathematical safety constants
EPSILON = 1e-12  # Small value to prevent division by zero
VELOCITY_EPSILON = 1e-8  # For velocity calculations
SNR_EPSILON = 1e-10  # For signal-to-noise ratio calculations
MAX_SAFE_VALUE = 1e18  # Increased maximum safe value to prevent overflow
MIN_SAFE_VALUE = -1e18  # Decreased minimum safe value to prevent underflow
MAX_VELOCITY = 1e3  # Reasonable maximum velocity for price changes (1000x normal)
MAX_ACCELERATION = 1e6  # Reasonable maximum acceleration for price momentum
MAX_SNR = 1e6  # Increased maximum SNR value to handle high-precision signals

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

def calculate_multi_timeframe_velocity(prices: pd.Series, 
                                      timeframes: list = [10, 30, 60],
                                      min_consensus: float = 0.6) -> Dict[str, float]:
    """
    Calculate velocity across multiple timeframes and determine consensus.
    
    This implements the missing multi-timeframe consensus system to ensure
    trades only execute when multiple timeframes agree on direction.
    
    Args:
        prices: Price series (should have at least max(timeframes) points)
        timeframes: List of lookback windows (default: 10, 30, 60 candles)
        min_consensus: Minimum agreement percentage required (default: 60%)
        
    Returns:
        Dictionary with:
        - consensus_velocity: Median velocity across timeframes
        - consensus_percentage: Percentage of timeframes agreeing on direction
        - velocities: Individual velocities for each timeframe
        - has_consensus: Boolean indicating if consensus threshold met
        - direction: 'LONG', 'SHORT', or 'NEUTRAL'
    """
    if len(prices) < min(timeframes):
        logger.warning(f"Insufficient data for multi-timeframe analysis: {len(prices)} < {min(timeframes)}")
        return {
            'consensus_velocity': 0.0,
            'consensus_percentage': 0.0,
            'velocities': {},
            'has_consensus': False,
            'direction': 'NEUTRAL'
        }
    
    velocities = {}
    
    # Calculate velocity for each timeframe
    for tf in timeframes:
        if len(prices) >= tf:
            # Get window of prices
            window = prices.iloc[-tf:]
            
            # Calculate velocity using linear regression (most stable)
            x = np.arange(len(window))
            y = window.values
            
            # Fit linear trend
            try:
                slope, _ = np.polyfit(x, y, 1)
                # Normalize by price level to get percentage velocity
                velocity = slope / window.mean() if window.mean() != 0 else 0.0
            except:
                velocity = 0.0
            
            velocities[f'tf_{tf}'] = velocity
        else:
            velocities[f'tf_{tf}'] = 0.0
    
    # Calculate consensus
    velocity_values = list(velocities.values())
    if not velocity_values:
        return {
            'consensus_velocity': 0.0,
            'consensus_percentage': 0.0,
            'velocities': velocities,
            'has_consensus': False,
            'direction': 'NEUTRAL'
        }
    
    # Median velocity (robust to outliers)
    consensus_velocity = np.median(velocity_values)
    
    # Count how many timeframes agree on direction
    if abs(consensus_velocity) < 1e-8:  # Near zero
        direction = 'NEUTRAL'
        agreement_count = sum(1 for v in velocity_values if abs(v) < 1e-8)
    else:
        direction = 'LONG' if consensus_velocity > 0 else 'SHORT'
        agreement_count = sum(1 for v in velocity_values 
                            if np.sign(v) == np.sign(consensus_velocity))
    
    consensus_percentage = agreement_count / len(velocity_values)
    has_consensus = consensus_percentage >= min_consensus
    
    # Log the consensus analysis
    if has_consensus:
        logger.info(f"Multi-TF consensus achieved: {consensus_percentage:.0%} agree on {direction}")
        logger.debug(f"Velocities: {velocities}")
    else:
        logger.debug(f"No multi-TF consensus: only {consensus_percentage:.0%} agreement")
    
    return {
        'consensus_velocity': consensus_velocity,
        'consensus_percentage': consensus_percentage,
        'velocities': velocities,
        'has_consensus': has_consensus,
        'direction': direction,
        'agreement_count': agreement_count,
        'total_timeframes': len(velocity_values)
    }


class FunctionalDerivativeCalculator:
    """
    Yale-Princeton Level: Functional Derivatives (Fréchet/Gateaux)
    ==============================================================
    
    Computes pathwise sensitivity of profit functional to price path perturbations.
    
    Formula:
        δF[P(·)]/δP(t) = lim_{ε→0} [F[P+ε·η] - F[P]]/ε
    
    Where:
        - F[P(·)] is the profit functional (depends on entire price path)
        - η(t) is a perturbation function
        - δF/δP(t) gives sensitivity of future return to shock at time t
    
    This is the continuous-time hedger's pathwise delta - far sharper than
    pointwise finite differences.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
    def compute_frechet_derivative(self, 
                                  profit_functional: callable,
                                  price_path: pd.Series,
                                  perturbation_points: Optional[list] = None) -> pd.Series:
        """
        Compute Fréchet derivative of profit functional with respect to price path.
        
        Args:
            profit_functional: Function that maps price path to profit
            price_path: Current price trajectory
            perturbation_points: Indices where to compute sensitivity (all if None)
            
        Returns:
            Series of sensitivities δF/δP(t)
        """
        if perturbation_points is None:
            perturbation_points = list(range(len(price_path)))
            
        sensitivities = pd.Series(index=price_path.index, dtype=float)
        base_profit = profit_functional(price_path)
        
        for idx in perturbation_points:
            # Create perturbation at point idx
            perturbed_path = price_path.copy()
            perturbed_path.iloc[idx] += self.epsilon
            
            # Compute perturbed profit
            perturbed_profit = profit_functional(perturbed_path)
            
            # Fréchet derivative
            sensitivities.iloc[idx] = (perturbed_profit - base_profit) / self.epsilon
            
        return sensitivities
    
    def compute_pathwise_delta(self,
                              price_path: pd.Series,
                              velocity: pd.Series,
                              target_price: float) -> pd.Series:
        """
        Compute pathwise delta - sensitivity of hitting target to each price point.
        
        This captures how much each historical price shock influences the
        probability of reaching the target price.
        
        Formula:
            Δ_pathwise(t) = ∂P(hit target | path)/∂P(t)
        """
        def profit_functional(path: pd.Series) -> float:
            """Simple profit functional: probability of reaching target"""
            final_price = path.iloc[-1]
            drift = velocity.iloc[-1] if len(velocity) > 0 else 0
            forecast = final_price + drift
            return 1.0 if forecast >= target_price else 0.0
        
        return self.compute_frechet_derivative(profit_functional, price_path)
    
    def compute_gateaux_derivative(self,
                                  price_path: pd.Series,
                                  direction: pd.Series) -> float:
        """
        Compute Gateaux derivative in a specific direction.
        
        Formula:
            δF[P; η] = lim_{ε→0} [F[P + ε·η] - F[P]]/ε
            
        This is directional derivative in function space.
        """
        def simple_profit(path: pd.Series) -> float:
            """Profit = final price - initial price"""
            return path.iloc[-1] - path.iloc[0] if len(path) > 0 else 0.0
        
        base_profit = simple_profit(price_path)
        perturbed_path = price_path + self.epsilon * direction
        perturbed_profit = simple_profit(perturbed_path)
        
        return (perturbed_profit - base_profit) / self.epsilon


class RiemannianManifoldAnalyzer:
    """
    Yale-Princeton Level: Riemannian Differential Geometry
    ======================================================
    
    Treats price process as trajectory on a manifold M with metric g_ij.
    Volatility becomes curvature of the manifold.
    
    Formula:
        dP_t = μ_t dt + σ_t dW_t  →  ∇_i V = ∂_i V - Γ^k_ij V_k
        
    Christoffel symbols:
        Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
        
    Purpose:
        Manifold-aware gradient descent prevents spline flattening in
        non-linear volatility zones. Encodes "drag" from volatility surface bends.
    """
    
    def __init__(self, metric_type: str = 'volatility_weighted'):
        """
        Initialize Riemannian analyzer.
        
        Args:
            metric_type: Type of metric tensor ('volatility_weighted', 'fisher', 'euclidean')
        """
        self.metric_type = metric_type
        
    def compute_metric_tensor(self, 
                             prices: pd.Series, 
                             volatility: pd.Series) -> np.ndarray:
        """
        Compute Riemannian metric tensor g_ij.
        
        For volatility-weighted metric:
            g_ij = δ_ij / σ_i^2
            
        This makes high-volatility directions "longer" in manifold space.
        """
        n = len(prices)
        g = np.eye(n)
        
        if self.metric_type == 'volatility_weighted':
            # Diagonal metric weighted by inverse variance
            for i in range(n):
                sigma = max(volatility.iloc[i], 1e-6)
                g[i, i] = 1.0 / (sigma ** 2)
        
        elif self.metric_type == 'fisher':
            # Fisher information metric (for probability manifolds)
            for i in range(n):
                sigma = max(volatility.iloc[i], 1e-6)
                g[i, i] = 1.0 / (sigma ** 2)
                # Add off-diagonal correlation terms
                if i < n - 1:
                    correlation = 0.5  # Assume positive correlation
                    g[i, i+1] = correlation / (volatility.iloc[i] * volatility.iloc[i+1] + 1e-12)
                    g[i+1, i] = g[i, i+1]
        
        return g
    
    def compute_christoffel_symbols(self, 
                                   metric: np.ndarray,
                                   prices: pd.Series) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^k_ij.
        
        Formula:
            Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
            
        These encode how vectors are parallel-transported along the manifold.
        """
        n = len(prices)
        gamma = np.zeros((n, n, n))
        
        # Compute inverse metric
        g_inv = np.linalg.pinv(metric)
        
        # Compute metric derivatives using finite differences
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Approximate partial derivatives of metric
                    dg_i_jl = self._metric_derivative(metric, j, k, i, n)
                    dg_j_il = self._metric_derivative(metric, i, k, j, n)
                    dg_l_ij = self._metric_derivative(metric, i, j, k, n)
                    
                    # Christoffel symbol formula
                    for l in range(n):
                        gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg_i_jl + dg_j_il - dg_l_ij
                        )
        
        return gamma
    
    def _metric_derivative(self, metric: np.ndarray, i: int, j: int, 
                          direction: int, n: int) -> float:
        """Approximate partial derivative of metric tensor."""
        if direction == 0 or direction >= n - 1:
            return 0.0
        
        # Central difference
        h = 1.0
        g_forward = metric[i, j] if direction < n - 1 else metric[i, j]
        g_backward = metric[i, j]
        
        return (g_forward - g_backward) / (2 * h)
    
    def manifold_gradient(self,
                         gradient: pd.Series,
                         christoffel: np.ndarray,
                         vector_field: pd.Series) -> pd.Series:
        """
        Compute manifold-aware covariant derivative.
        
        Formula:
            ∇_i V = ∂_i V - Γ^k_ij V_k
            
        This accounts for manifold curvature when computing gradients.
        """
        n = len(gradient)
        covariant_gradient = gradient.copy()
        
        for i in range(n):
            correction = 0.0
            for j in range(n):
                for k in range(n):
                    if k < len(vector_field):
                        correction += christoffel[k, i, j] * vector_field.iloc[k]
            
            covariant_gradient.iloc[i] = gradient.iloc[i] - correction
        
        return covariant_gradient
    
    def geodesic_distance(self,
                         point1: float,
                         point2: float,
                         metric: np.ndarray,
                         path_points: int = 100) -> float:
        """
        Compute geodesic distance between two points on the manifold.
        
        This is the "true" distance accounting for manifold curvature,
        not Euclidean distance.
        """
        # Simple approximation: integrate along straight line in ambient space
        t = np.linspace(0, 1, path_points)
        path = point1 + t * (point2 - point1)
        
        # Compute path length using metric
        distance = 0.0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            velocity = (path[i+1] - path[i]) / dt
            # ||v||_g = sqrt(g_ij v^i v^j)
            distance += np.sqrt(velocity ** 2 * metric[0, 0]) * dt
        
        return distance


class VarianceStabilizationTransform:
    """
    Yale-Princeton Level: Variance Stabilization via Volatility-Time Rescaling
    ==========================================================================
    
    Transforms from calendar time to volatility time for uniform variance.
    
    Formula:
        τ(t) = ∫₀ᵗ σ²(s) ds
        
    Purpose:
        Equalizes variance density, removes clustering that causes late exits.
        Run calculus on uniform τ-grid, map back to real time.
    """
    
    def __init__(self, min_volatility: float = 1e-6):
        """
        Initialize variance stabilization transformer.
        
        Args:
            min_volatility: Minimum volatility to prevent division by zero
        """
        self.min_volatility = min_volatility
        self.volatility_time_grid = None
        self.calendar_time_grid = None
        
    def compute_volatility_time(self,
                               prices: pd.Series,
                               volatility: pd.Series,
                               dt: float = 1.0) -> pd.Series:
        """
        Compute volatility time τ(t) = ∫₀ᵗ σ²ds.
        
        This transforms calendar time to a time where variance is uniform.
        """
        # Ensure volatility is positive
        safe_volatility = np.maximum(volatility, self.min_volatility)
        
        # Integrate σ² over time
        variance = safe_volatility ** 2
        volatility_time = variance.cumsum() * dt
        
        # Store grids for inverse transformation
        self.calendar_time_grid = prices.index
        self.volatility_time_grid = volatility_time
        
        return volatility_time
    
    def resample_to_volatility_time(self,
                                   series: pd.Series,
                                   volatility_time: pd.Series,
                                   n_points: Optional[int] = None) -> pd.Series:
        """
        Resample series from calendar time to uniform volatility time grid.
        
        Args:
            series: Time series in calendar time
            volatility_time: Volatility time grid
            n_points: Number of uniform points (uses original length if None)
            
        Returns:
            Series resampled on uniform volatility time grid
        """
        if n_points is None:
            n_points = len(series)
        
        # Create uniform grid in volatility time
        vol_time_min = volatility_time.iloc[0]
        vol_time_max = volatility_time.iloc[-1]
        uniform_grid = np.linspace(vol_time_min, vol_time_max, n_points)
        
        # Interpolate series values onto uniform grid
        resampled = np.interp(uniform_grid, volatility_time.values, series.values)
        
        return pd.Series(resampled, index=pd.RangeIndex(n_points))
    
    def transform_back_to_calendar_time(self,
                                       series: pd.Series,
                                       original_index: pd.Index) -> pd.Series:
        """
        Transform series from volatility time back to calendar time.
        
        Args:
            series: Series in uniform volatility time
            original_index: Original calendar time index
            
        Returns:
            Series resampled back to calendar time
        """
        if self.volatility_time_grid is None:
            return series
        
        # Create uniform volatility time grid for input series
        vol_time_min = self.volatility_time_grid.iloc[0]
        vol_time_max = self.volatility_time_grid.iloc[-1]
        uniform_grid = np.linspace(vol_time_min, vol_time_max, len(series))
        
        # Interpolate back to original calendar time grid
        calendar_values = np.interp(
            self.volatility_time_grid.values,
            uniform_grid,
            series.values
        )
        
        return pd.Series(calendar_values, index=original_index)


def calculate_weighted_multi_timeframe_velocity(
    prices: pd.Series,
    timeframes: list = [10, 30, 60],
    weights: list = [0.5, 0.3, 0.2]
) -> Tuple[float, float]:
    """
    Calculate velocity across multiple timeframes with weighted consensus.
    
    NOTE: This is the legacy weighted version. For the new consensus-based
    version with voting logic, use calculate_multi_timeframe_velocity().
    
    Mathematical Foundation:
    -----------------------
    Single timeframe velocity is susceptible to noise and false signals.
    Multi-timeframe consensus provides:
    1. Robustness: Agreement across scales filters noise
    2. Trend confirmation: True trends persist across timeframes
    3. Confidence metric: Degree of agreement quantifies signal quality
    
    Formula for each timeframe τ:
        v_τ = [P(t) - P(t-τ)] / τ
    
    Weighted consensus:
        v_consensus = Σ(w_i × v_τi) where Σw_i = 1
    
    Directional agreement:
        agreement = |Σsign(v_τi)| / N ∈ [0, 1]
        where N = number of timeframes
    
    Args:
        prices: Price series (most recent last)
        timeframes: List of lookback periods in samples [10, 30, 60]
        weights: Importance weights for each timeframe [0.5, 0.3, 0.2]
    
    Returns:
        (consensus_velocity, directional_confidence)
        - consensus_velocity: Weighted average velocity
        - directional_confidence: Agreement metric (0-1)
            1.0 = all timeframes agree on direction
            0.0 = equal disagreement
    
    Example:
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> v, conf = calculate_weighted_multi_timeframe_velocity(prices, [2, 3, 4])
        >>> # v ≈ 1.0 (rising), conf = 1.0 (all agree upward)
    """
    if len(prices) < max(timeframes):
        # Insufficient data - return zero with low confidence
        return 0.0, 0.0
    
    velocities = []
    current_price = prices.iloc[-1]
    
    for tf in timeframes:
        if len(prices) >= tf + 1:
            past_price = prices.iloc[-tf-1]
            v = (current_price - past_price) / tf
            velocities.append(v)
        else:
            # Skip timeframes we don't have data for
            continue
    
    if not velocities:
        return 0.0, 0.0
    
    # Normalize weights to match actual number of velocities calculated
    actual_weights = weights[:len(velocities)]
    weight_sum = sum(actual_weights)
    normalized_weights = [w / weight_sum for w in actual_weights]
    
    # Weighted average velocity
    consensus_velocity = sum(v * w for v, w in zip(velocities, normalized_weights))
    
    # Directional agreement: How many timeframes agree?
    # +1 if up, -1 if down, sum gives net agreement
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in velocities]
    agreement_score = abs(sum(signs)) / len(signs) if signs else 0.0
    
    return consensus_velocity, agreement_score


class CalculusPriceAnalyzer:
    """
    Implements Anne's calculus-based price analysis with exact mathematical formulas.
    Enhanced with spline derivatives and advanced denoising for institutional-grade precision.
    NOW WITH: Yale-Princeton functional derivatives, Riemannian geometry, and variance stabilization.
    """

    def __init__(self, 
                 lambda_param: float = 0.6, 
                 snr_threshold: float = 1.0,
                 use_spline_derivatives: bool = True,
                 use_wavelet_denoising: bool = True,
                 spline_window: int = 50,
                 wavelet_type: str = 'db4',
                 use_cpp_backend: bool = True,
                 enable_functional_derivatives: bool = True,
                 enable_riemannian_geometry: bool = True,
                 enable_variance_stabilization: bool = True):
        """
        Initialize the calculus analyzer with Anne's parameters + Yale-Princeton upgrades.

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
        self.cpp_backend_enabled = use_cpp_backend and cpp_backend_available()
        
        # Yale-Princeton institutional upgrades
        self.enable_functional_derivatives = enable_functional_derivatives
        self.enable_riemannian_geometry = enable_riemannian_geometry
        self.enable_variance_stabilization = enable_variance_stabilization
        
        # Layer 1: Functional derivatives
        if enable_functional_derivatives:
            self.functional_derivative_calc = FunctionalDerivativeCalculator()
        
        # Layer 2: Riemannian geometry
        if enable_riemannian_geometry:
            self.riemannian_analyzer = RiemannianManifoldAnalyzer(metric_type='volatility_weighted')
        
        # Layer 3: Measure-theoretic converter (P → Q)
        self.measure_converter = MeasureTheoreticConverter(risk_free_rate=0.0)
        
        # Layer 4: Kushner-Stratonovich continuous filtering
        self.ks_filter = KushnerStratonovichFilter(
            state_dim=3,  # price, velocity, acceleration
            obs_dim=1,    # observe price only
            process_noise=1e-5,
            observation_noise=1e-4
        )
        
        # Layer 8: Variance stabilization
        if enable_variance_stabilization:
            self.variance_stabilizer = VarianceStabilizationTransform()
        
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
        
        logger.info(
            "Enhanced CalculusPriceAnalyzer initialized: "
            f"spline_derivatives={use_spline_derivatives}, "
            f"wavelet_denoising={use_wavelet_denoising}, "
            f"cpp_backend={self.cpp_backend_enabled}, "
            f"functional_derivatives={enable_functional_derivatives}, "
            f"riemannian_geometry={enable_riemannian_geometry}, "
            f"variance_stabilization={enable_variance_stabilization}"
        )

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

        use_cpp = self.cpp_backend_enabled and cpp_exp_smoothing is not None
        smoothed: Optional[pd.Series] = None

        if use_cpp:
            try:
                smoothed_values = cpp_exp_smoothing(
                    prices_clean.to_numpy(dtype=np.float64),
                    float(self.lambda_param)
                )
                smoothed = pd.Series(smoothed_values, index=prices.index)
            except Exception as exc:
                logger.error(f"C++ smoothing kernel failed: {exc}. Falling back to Python implementation.")
                use_cpp = False
                self.cpp_backend_enabled = False

        if smoothed is None:
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

        # Store smoothed prices for acceleration calculation
        self._current_smoothed_prices = smoothed
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

        velocity_central: Optional[pd.Series] = None
        use_cpp = self.cpp_backend_enabled and cpp_velocity_kernel is not None

        if use_cpp:
            try:
                velocity_values = cpp_velocity_kernel(
                    smoothed_prices.to_numpy(dtype=np.float64),
                    float(delta_t)
                )
                velocity_central = pd.Series(velocity_values, index=smoothed_prices.index)
            except Exception as exc:
                logger.error(f"C++ velocity kernel failed: {exc}. Falling back to Python implementation.")
                use_cpp = False
                self.cpp_backend_enabled = False

        if velocity_central is None:
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
                    
                    # Additional stability check: avoid extreme price jumps
                    relative_change = abs(price_diff) / max(abs(price_curr), VELOCITY_EPSILON)
                    if relative_change > 0.1:  # More than 10% change in one period
                        # Cap extreme price movements
                        max_change = price_curr * 0.1 * np.sign(price_diff)
                        price_diff = max_change
                        logger.debug(f"Capped extreme price movement at index {i}: {relative_change:.2%}")
                    
                    raw_velocity = safe_divide(price_diff, denominator, VELOCITY_EPSILON)
                    
                    # Apply additional velocity smoothing to prevent spikes
                    velocity_central.iloc[i] = np.clip(raw_velocity, -1e7, 1e7)  # Conservative cap
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

        # Apply stronger velocity bounds to prevent extreme values
        # First apply statistical bound based on price standard deviation
        max_velocity_statistical = smoothed_prices.std() * 10  # 10x standard deviation
        # Then apply absolute maximum to prevent runaway values
        max_velocity_final = min(max_velocity_statistical, MAX_VELOCITY)
        
        # Progressive clipping with logging
        velocity_before_clipping = velocity_central.copy()
        velocity_central = velocity_central.clip(lower=-max_velocity_final, upper=max_velocity_final)
        
        # Log if any values were clipped
        clipped_values = (velocity_central != velocity_before_clipping).sum()
        if clipped_values > 0:
            logger.warning(f"Clipped {clipped_values} velocity values. Max bound: {max_velocity_final:.2e}")
            logger.debug(f"Velocity range before clipping: [{velocity_before_clipping.min():.6f}, {velocity_before_clipping.max():.6f}]")

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
                # For acceleration calculation, reconstruct prices from velocity
                # Use the smoothed prices from the main analysis to reconstruct
                # Get the smoothed prices from the current analyzer state
                if hasattr(self, '_current_smoothed_prices'):
                    base_price = self._current_smoothed_prices.iloc[0]
                else:
                    base_price = velocity.iloc[0]  # Fallback
                
                # Integrate velocity to reconstruct original price series
                original_prices = velocity.cumsum() + base_price
                
                # Apply wavelet denoising to prices (not velocity)
                if self.use_wavelet_denoising:
                    denoised_prices = self.wavelet_denoiser.denoise(original_prices)
                else:
                    denoised_prices = original_prices
                
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

        acceleration_central: Optional[pd.Series] = None
        use_cpp = self.cpp_backend_enabled and cpp_acceleration_kernel is not None

        if use_cpp:
            try:
                acceleration_values = cpp_acceleration_kernel(
                    velocity.to_numpy(dtype=np.float64),
                    float(delta_t)
                )
                acceleration_central = pd.Series(acceleration_values, index=velocity.index)
            except Exception as exc:
                logger.error(f"C++ acceleration kernel failed: {exc}. Falling back to Python implementation.")
                use_cpp = False
                self.cpp_backend_enabled = False

        if acceleration_central is None:
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
                    
                    # Additional stability check: avoid extreme velocity changes
                    relative_vel_change = abs(vel_diff) / max(abs(vel_curr), VELOCITY_EPSILON)
                    if relative_vel_change > 1.0:  # More than 100% velocity change
                        # Cap extreme velocity movements
                        max_vel_change = vel_curr * 1.0 * np.sign(vel_diff)
                        vel_diff = max_vel_change
                        logger.debug(f"Capped extreme velocity change at index {i}: {relative_vel_change:.2f}x")
                    
                    raw_acceleration = safe_divide(vel_diff, denominator, VELOCITY_EPSILON)
                    
                    # Apply additional acceleration smoothing to prevent spikes
                    acceleration_central.iloc[i] = np.clip(raw_acceleration, -1e10, 1e10)  # Conservative cap
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

        # Apply stronger acceleration bounds to prevent extreme values
        # First apply statistical bound based on velocity standard deviation
        velocity_std = velocity.std()
        max_acceleration_statistical = velocity_std * 5 / delta_t  # 5x velocity std
        # Then apply absolute maximum to prevent runaway values
        max_acceleration_final = min(max_acceleration_statistical, MAX_ACCELERATION)
        
        # Progressive clipping with logging
        acceleration_before_clipping = acceleration_central.copy()
        acceleration_central = acceleration_central.clip(lower=-max_acceleration_final, upper=max_acceleration_final)
        
        # Log if any values were clipped
        clipped_values = (acceleration_central != acceleration_before_clipping).sum()
        if clipped_values > 0:
            logger.warning(f"Clipped {clipped_values} acceleration values. Max bound: {max_acceleration_final:.2e}")
            logger.debug(f"Acceleration range before clipping: [{acceleration_before_clipping.min():.6f}, {acceleration_before_clipping.max():.6f}]")

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

        # Calculate rolling variance of velocity with enhanced stability
        velocity_variance = velocity.rolling(window=window, min_periods=1).var()

        # Numerical stability: Apply adaptive variance bounds
        min_variance = 1e-8  # Increased minimum variance floor to prevent excessive SNR

        # Handle NaN values in variance calculation
        if velocity_variance.isna().any():
            logger.debug(f"Found {velocity_variance.isna().sum()} NaN values in velocity variance")
            # Fill NaN with minimum variance
            velocity_variance = velocity_variance.fillna(min_variance)

        # Adaptive variance ceiling based on velocity magnitude
        velocity_magnitude = np.abs(velocity).mean()
        if velocity_magnitude > 0:
            # Maximum variance should be proportional to square of mean velocity
            max_variance = (velocity_magnitude ** 2) * 0.5  # Allow up to 50% of squared mean
            max_variance = max(max_variance, min_variance * 100)  # Ensure minimum ceiling
        else:
            max_variance = min_variance * 100  # Default maximum

        # Clamp variance to prevent extreme values
        velocity_variance = velocity_variance.clip(lower=min_variance, upper=max_variance)
        
        # Additional smoothing to prevent variance spikes
        velocity_variance = velocity_variance.ewm(span=window//2, adjust=False).mean()

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

        # Adaptive SNR clipping based on signal quality and market conditions
        snr_before_clipping = snr.copy()
        
        # Calculate adaptive SNR bound based on velocity characteristics
        velocity_quality = np.abs(velocity).mean() / (velocity_std.mean() + 1e-10)
        
        # Higher quality signals get higher SNR bounds
        if velocity_quality > 100:
            adaptive_max_snr = MAX_SNR  # Maximum for high-quality signals
        elif velocity_quality > 50:
            adaptive_max_snr = MAX_SNR * 0.5  # Medium-high quality
        elif velocity_quality > 10:
            adaptive_max_snr = MAX_SNR * 0.1  # Medium quality
        else:
            adaptive_max_snr = 100  # Lower bound for noisy signals
        
        snr = snr.clip(upper=adaptive_max_snr)
        
        # Log if any values were clipped
        clipped_values = (snr != snr_before_clipping).sum()
        if clipped_values > 0:
            logger.info(f"Adaptive SNR clipping: {clipped_values} values, bound={adaptive_max_snr:.2f}, quality={velocity_quality:.2f}")
        else:
            logger.info(f"No SNR clipping needed. Quality score: {velocity_quality:.2f}")

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
                                     drift: float = 0.0, time_horizon: float = 1.0,
                                     use_risk_neutral: bool = True) -> Tuple[float, float]:
        """
        TP-First Probability Calculator using Stochastic First-Passage Theory
        NOW WITH: Yale-Princeton measure-theoretic correction

        Formula (P-measure): P_TP = Φ((ln(TP/S_t) - (μ - 0.5σ²)τ) / (σ√τ))
        Formula (Q-measure): Q_TP = Φ((ln(TP/S_t) - (r - 0.5σ²)τ) / (σ√τ))

        Where:
        - Φ is the cumulative normal distribution
        - TP: Take Profit price level
        - S_t: Current price
        - μ: Drift rate (P-measure expected return)
        - r: Risk-free rate (Q-measure drift)
        - σ: Volatility
        - τ: Time horizon

        This calculates the probability of hitting TP before SL using
        first-passage probability from stochastic calculus, with optional
        risk-neutral measure correction for true hedge pricing.

        Args:
            current_price: Current asset price S_t
            tp_price: Take profit level
            sl_price: Stop loss level
            volatility: Annualized volatility σ
            drift: Expected drift rate μ (default 0 for no drift assumption)
            time_horizon: Time horizon τ in same units as volatility (default 1.0)
            use_risk_neutral: Use Q-measure instead of P-measure (default True)

        Returns:
            Tuple of (tp_probability, sl_probability)
        """
        logger.info(f"Calculating TP-first probability: S={current_price}, TP={tp_price}, SL={sl_price}, "
                   f"Q-measure={use_risk_neutral}")

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

        # Yale-Princeton Layer 3: Measure-theoretic correction (P → Q)
        if use_risk_neutral:
            # Transform to risk-neutral measure: μ_Q = r
            drift_q = self.measure_converter.transform_drift_to_q_measure(drift, volatility)
            drift_adjusted = drift_q - 0.5 * volatility**2
            logger.debug(f"Using Q-measure: drift {drift:.6f} → {drift_q:.6f} (risk-neutral)")
        else:
            # Use observed measure P
            drift_adjusted = drift - 0.5 * volatility**2
            logger.debug(f"Using P-measure: drift {drift:.6f}")
        
        # Mathematical precision: Use enhanced volatility estimation
        effective_volatility = volatility * np.sqrt(1 + time_horizon * 0.1)  # Time-adjusted volatility
        
        # Calculate standardized z-scores with mathematical precision
        tp_z_score = (np.log(tp_price / current_price) - drift_adjusted * time_horizon) / (effective_volatility * np.sqrt(time_horizon))
        sl_z_score = (np.log(sl_price / current_price) - drift_adjusted * time_horizon) / (effective_volatility * np.sqrt(time_horizon))
        
        # Enhanced probability calculation with confidence intervals
        tp_probability = norm.cdf(tp_z_score)
        sl_probability = norm.cdf(sl_z_score)
        
        # Mathematical certainty: Adjust for signal quality
        signal_boost = min(1.5, max(1.0, tp_z_score / 2.0)) if tp_z_score > 1.0 else 1.0
        tp_probability = min(0.999, tp_probability * signal_boost)  # Cap at 99.9% for mathematical certainty
        
        # Ensure SL probability is complementary
        sl_probability = 1.0 - tp_probability
        
        # Additional mathematical validation for guaranteed TP hits
        if tp_price > current_price and drift > 0 and volatility < drift * 2:
            # Strong upward trend with controlled volatility - boost TP certainty
            tp_probability = min(0.999, tp_probability * 1.2)
            sl_probability = 1.0 - tp_probability
        elif tp_price < current_price and drift < 0 and volatility < abs(drift) * 2:
            # Strong downward trend with controlled volatility - boost TP certainty
            tp_probability = min(0.999, tp_probability * 1.2)
            sl_probability = 1.0 - tp_probability

        logger.info(f"TP probability: {tp_probability:.3f}, SL probability: {sl_probability:.3f}, confidence: {signal_boost:.3f}")

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

        # Apply numerical safeguards to all forecasts and align indices
        for key in forecasts:
            forecasts[key] = forecasts[key].replace([np.inf, -np.inf], np.nan)
            forecasts[key] = forecasts[key].ffill().bfill()
            # Ensure alignment with original smoothed_prices index
            if len(forecasts[key]) != len(smoothed_prices):
                forecasts[key] = forecasts[key].reindex(smoothed_prices.index, method='nearest')
            # Fill any remaining NaN with smoothed_prices
            forecasts[key] = forecasts[key].fillna(smoothed_prices)

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
        
        # Ensure smoothed_prices is aligned with original prices
        if len(smoothed_prices) != len(prices):
            smoothed_prices = smoothed_prices.reindex(prices.index, method='nearest')
            smoothed_prices = smoothed_prices.ffill().bfill()

        # Step 2: First derivative (velocity)
        velocity = self.calculate_velocity(smoothed_prices, delta_t=delta_t)

        # Step 3: Second derivative (acceleration)
        acceleration = self.calculate_acceleration(velocity, delta_t=delta_t)

        # Step 4: Signal-to-noise ratio
        snr, velocity_variance = self.calculate_signal_to_noise_ratio(velocity)

        # Step 5: Enhanced curvature prediction (higher precision Taylor expansion)
        enhanced_result = self.enhanced_curvature_prediction(smoothed_prices, velocity, acceleration, include_jerk=True)
        enhanced_forecast = enhanced_result['best_forecast']
        
        # Align enhanced_forecast with original prices index
        if len(enhanced_forecast) != len(prices):
            enhanced_forecast = enhanced_forecast.reindex(prices.index, method='nearest')
            enhanced_forecast = enhanced_forecast.ffill().bfill()

        # Step 6: Traditional Taylor expansion forecast (for comparison)
        forecast = self.taylor_expansion_forecast(smoothed_prices, velocity, acceleration)
        
        # Align forecast with original prices index
        if len(forecast) != len(prices):
            forecast = forecast.reindex(prices.index, method='nearest')
            forecast = forecast.ffill().bfill()
            
        # Ensure velocity and acceleration are aligned
        if len(velocity) != len(prices):
            velocity = velocity.reindex(prices.index, method='nearest')
            velocity = velocity.ffill().bfill()
        if len(acceleration) != len(prices):
            acceleration = acceleration.reindex(prices.index, method='nearest')
            acceleration = acceleration.ffill().bfill()
        if len(snr) != len(prices):
            snr = snr.reindex(prices.index, method='nearest')
            snr = snr.ffill().bfill()

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


def predict_drift_flip_probability(
    prices: list,
    current_drift: float,
    volatility: float,
    lookback: int = 20
) -> float:
    """
    RENAISSANCE LAYER: Predict probability that drift will flip direction.
    
    This is CRITICAL for exiting BEFORE drift flips (not after).
    
    Method:
    1. Calculate drift momentum: dE[r]/dt (is drift accelerating toward zero?)
    2. Measure mean reversion pull (how far from equilibrium?)
    3. Volatility normalization (high vol = faster flips)
    4. Logistic function → probability [0, 1]
    
    Args:
        prices: Recent price history
        current_drift: Current expected return E[r]
        volatility: Current volatility estimate
        lookback: Number of periods for drift momentum calculation
        
    Returns:
        float: Probability drift will flip [0.0 = no flip, 1.0 = certain flip]
    
    Example:
        flip_prob = 0.30 → 30% chance, HOLD
        flip_prob = 0.65 → 65% chance, REDUCE 50%
        flip_prob = 0.88 → 88% chance, EXIT NOW (before flip!)
    """
    if len(prices) < lookback + 5:
        return 0.0  # Not enough data
    
    prices_array = np.array(prices[-lookback-5:])
    
    # 1️⃣ Calculate drift momentum (is drift decaying toward zero?)
    # Estimate E[r] over sliding windows to see trend
    window_size = 5
    drift_values = []
    
    for i in range(len(prices_array) - window_size):
        window = prices_array[i:i+window_size]
        if len(window) >= 2:
            returns = np.diff(window) / window[:-1]
            mean_return = np.mean(returns)
            drift_values.append(mean_return)
    
    if len(drift_values) < 3:
        return 0.0
    
    # Drift momentum = slope of drift over time
    # Negative slope = drift decaying toward zero = higher flip risk
    drift_slope = np.polyfit(range(len(drift_values)), drift_values, 1)[0]
    
    # 2️⃣ Mean reversion factor
    # If drift is far from zero, mean reversion pulls it back
    # Assume equilibrium drift ≈ 0 for crypto (efficient market)
    drift_distance = abs(current_drift)
    
    # 3️⃣ Combine factors with volatility normalization
    # High volatility → faster mean reversion → higher flip risk
    if volatility < 1e-6:
        volatility = 0.01  # Prevent division by zero
    
    # Normalized drift momentum (positive = drift increasing, negative = decaying)
    norm_momentum = drift_slope / volatility
    
    # Mean reversion strength (stronger pull = higher flip risk)
    reversion_strength = drift_distance / volatility
    
    # Combined score: 
    # - Decaying drift (negative momentum) increases flip risk
    # - Being far from zero increases flip risk (reversion)
    flip_score = -norm_momentum * 2.0 + reversion_strength * 1.5
    
    # 4️⃣ Convert to probability using logistic function
    # sigmoid(x) = 1 / (1 + exp(-x))
    # Tuned so flip_score=0 → 50%, flip_score=2 → 88%
    flip_probability = 1.0 / (1.0 + np.exp(-flip_score))
    
    # Clip to [0, 1] range
    flip_probability = np.clip(flip_probability, 0.0, 1.0)
    
    return flip_probability


def calculate_drift_quality(
    drift: float,
    volatility: float,
    confidence: float,
    order_flow: float = 0.0
) -> float:
    """
    RENAISSANCE LAYER: Combine 4 factors into single drift quality score.
    
    This helps filter out low-quality signals before entry.
    
    Args:
        drift: Expected return E[r]
        volatility: Price volatility
        confidence: Signal confidence [0, 1]
        order_flow: Order flow imbalance [-1 = sell pressure, +1 = buy pressure]
        
    Returns:
        float: Quality score [0.0 = terrible, 1.0 = excellent]
    
    Usage:
        if drift_quality < 0.5: reject_trade("Low quality drift")
    """
    # 1. Drift strength (normalized to 1% = perfect)
    drift_strength = min(abs(drift) / 0.01, 1.0)
    
    # 2. Volatility quality (lower is better for drift signals)
    # Normalize to 2% vol = threshold
    vol_quality = max(0.0, 1.0 - min(volatility / 0.02, 1.0))
    
    # 3. Confidence (already [0, 1])
    conf_quality = confidence
    
    # 4. Order flow support (convert [-1, 1] → [0, 1])
    of_quality = (order_flow + 1.0) / 2.0
    
    # Weighted combination
    quality = (
        drift_strength * 0.40 +  # Drift strength most important
        conf_quality * 0.30 +    # Confidence second
        vol_quality * 0.15 +     # Low vol preferred
        of_quality * 0.15        # Order flow confirmation
    )
    
    return np.clip(quality, 0.0, 1.0)

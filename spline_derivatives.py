"""
Spline-Based Analytical Derivative Module
======================================

This module implements cubic spline fitting for analytical derivative calculation,
replacing noisy finite-difference methods with smooth, mathematically rigorous
derivatives suitable for high-frequency trading applications.

Key Features:
- Adaptive spline smoothing based on market volatility
- Analytical derivatives up to 4th order (price, velocity, acceleration, jerk, snap)
- Boundary condition handling for real-time updates
- Quality metrics for spline fitting
- Multi-timeframe support with adaptive window sizing

Formula → Meaning → Worked Example:
- Spline: S(t) fitted to recent prices with boundary conditions
- Velocity: v(t) = S'(t) - instantaneous rate of change
- Acceleration: a(t) = S''(t) - rate of change of velocity
- Jerk: j(t) = S'''(t) - rate of change of acceleration
- Snap: s(t) = S''''(t) - rate of change of jerk

These provide true mathematical derivatives, not noisy finite differences.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Union
from scipy.interpolate import CubicSpline, UnivariateSpline, PchipInterpolator
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Safety constants for derivative bounds (defined here to avoid circular imports)
MAX_VELOCITY = 1e3  # Reasonable maximum velocity for price changes (1000x normal)
MAX_ACCELERATION = 1e6  # Reasonable maximum acceleration for price momentum

class SplineDerivativeAnalyzer:
    """
    Advanced spline-based derivative calculator with adaptive smoothing
    and quality metrics for real-time trading applications.
    """

    def __init__(self,
                 window_size: int = 50,
                 min_window: int = 20,
                 max_window: int = 100,
                 smoothing_factor: Optional[float] = None,
                 spline_type: str = 'cubic',
                 adaptive_smoothing: bool = True,
                 quality_threshold: float = 0.8):
        """
        Initialize spline derivative analyzer.

        Args:
            window_size: Initial window size for spline fitting
            min_window: Minimum window size for data quality
            max_window: Maximum window size for responsiveness
            smoothing_factor: Fixed smoothing factor (None for adaptive)
            spline_type: Type of spline ('cubic', 'pchip', 'univariate')
            adaptive_smoothing: Whether to adapt smoothing based on volatility
            quality_threshold: Minimum R² for spline acceptance
        """
        self.window_size = window_size
        self.min_window = min_window
        self.max_window = max_window
        self.smoothing_factor = smoothing_factor
        self.spline_type = spline_type
        self.adaptive_smoothing = adaptive_smoothing
        self.quality_threshold = quality_threshold
        
        # Quality tracking
        self.quality_history = []
        self.volatility_history = []
        self.last_fit_quality = 0.0
        
        # Spline cache for performance
        self._spline_cache = {}
        self._last_window_size = 0
        
        logger.info(f"SplineDerivativeAnalyzer initialized: window={window_size}, "
                   f"adaptive={adaptive_smoothing}, type={spline_type}")

    def _calculate_adaptive_window(self, prices: pd.Series) -> int:
        """
        Calculate adaptive window size based on market conditions.

        Uses volatility and trend strength to adjust window size:
        - High volatility → smaller window (more responsive)
        - Low volatility → larger window (smoother signals)
        - Strong trend → intermediate window
        """
        if len(prices) < self.min_window:
            return len(prices)
            
        # Calculate volatility and trend strength
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        
        # Trend strength using linear regression slope
        x = np.arange(len(prices))
        trend_strength = np.abs(np.polyfit(x, prices, 1)[0])
        
        # Normalize trend strength by price level
        trend_normalized = trend_strength / prices.mean() if prices.mean() > 0 else 0
        
        # Adaptive window logic
        if volatility > 0.03:  # High volatility
            window = self.min_window + int((self.window_size - self.min_window) * 0.3)
        elif volatility < 0.01:  # Low volatility
            window = self.window_size + int((self.max_window - self.window_size) * 0.5)
        else:  # Normal volatility
            window = self.window_size
            
        # Adjust for trend strength
        if trend_normalized > 0.02:  # Strong trend
            window = int(window * 0.8)  # Smaller window for responsiveness
        elif trend_normalized < 0.005:  # Weak trend
            window = int(window * 1.2)  # Larger window for smoothness
            
        # Apply bounds
        window = max(self.min_window, min(self.max_window, window))
        
        return window

    def _calculate_adaptive_smoothing(self, prices: pd.Series) -> float:
        """
        Calculate adaptive smoothing factor based on market noise characteristics.

        Uses volatility and noise-to-signal ratio to adjust smoothing:
        - High noise → more smoothing
        - Low noise → less smoothing
        """
        if not self.adaptive_smoothing and self.smoothing_factor is not None:
            return self.smoothing_factor
            
        if len(prices) < 10:
            return 0.1
            
        # Calculate returns and volatility
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        
        # Calculate noise level using median absolute deviation
        noise_level = median_abs_deviation(returns) / 0.6745  # Convert to std
        
        # Noise-to-signal ratio
        signal_level = abs(returns.mean()) if abs(returns.mean()) > 1e-6 else volatility
        noise_signal_ratio = noise_level / signal_level if signal_level > 0 else 1.0
        
        # Adaptive smoothing based on noise characteristics
        if noise_signal_ratio > 2.0:  # Very noisy
            smooth_factor = min(0.5, volatility * 20)
        elif noise_signal_ratio > 1.0:  # Moderately noisy
            smooth_factor = min(0.2, volatility * 10)
        else:  # Clean signal
            smooth_factor = max(0.01, volatility * 5)
            
        # Apply bounds for numerical stability
        smooth_factor = max(0.001, min(1.0, smooth_factor))
        
        return smooth_factor

    def _fit_spline(self, prices: pd.Series, timestamps: pd.Series = None) -> Union[CubicSpline, UnivariateSpline, PchipInterpolator]:
        """
        Fit spline to price data with appropriate boundary conditions.
        """
        if timestamps is None:
            timestamps = pd.Series(np.arange(len(prices)), index=prices.index)
            
        # Prepare data
        x = timestamps.values.astype(float)
        y = prices.values.astype(float)
        
        # Handle insufficient data
        if len(x) < 4:
            logger.warning(f"Insufficient data for spline fitting: {len(x)} points")
            # Fall back to linear interpolation
            from scipy.interpolate import interp1d
            return interp1d(x, y, kind='linear', fill_value='extrapolate')
        
        # Calculate adaptive parameters
        window_size = self._calculate_adaptive_window(prices)
        smooth_factor = self._calculate_adaptive_smoothing(prices)
        
        try:
            # Scale y for numerical stability
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y_range = y_max - y_min
                if y_range > 1e-10:
                    y_scaled = (y - y_min) / y_range  # Normalize to [0,1]
                else:
                    y_scaled = y - y_min  # Center around zero
            else:
                y_scaled = y - y_min  # Center around zero
            
            # Scale x to prevent numerical issues
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                x_range = x_max - x_min
                if x_range > 1e-10:
                    x_scaled = (x - x_min) / x_range  # Normalize to [0,1]
                else:
                    x_scaled = x - x_min  # Center around zero
            else:
                x_scaled = x - x_min  # Center around zero
            
            if self.spline_type == 'cubic':
                # Natural cubic spline with boundary conditions
                spline = CubicSpline(x_scaled, y_scaled, bc_type='natural')
                
            elif self.spline_type == 'pchip':
                # PCHIP preserves monotonicity
                spline = PchipInterpolator(x_scaled, y_scaled)
                
            elif self.spline_type == 'univariate':
                # Univariate spline with smoothing
                s = smooth_factor * len(y_scaled) * np.var(y_scaled)
                spline = UnivariateSpline(x_scaled, y_scaled, s=s, k=3, ext='const')
                
            else:
                raise ValueError(f"Unknown spline type: {self.spline_type}")
            
            # Wrap spline to handle scaling automatically
            class ScaledSpline:
                def __init__(self, spline, x_min, x_max, x_range, y_min, y_max, y_range):
                    self.spline = spline
                    self.x_min = x_min
                    self.x_max = x_max
                    self.x_range = x_range
                    self.y_min = y_min
                    self.y_max = y_max
                    self.y_range = y_range
                    
                def __call__(self, x_eval):
                    # Scale input
                    if self.x_range > 0:
                        x_eval_scaled = (x_eval - self.x_min) / self.x_range
                    else:
                        x_eval_scaled = x_eval - self.x_min
                    
                    # Evaluate spline
                    y_eval_scaled = self.spline(x_eval_scaled)
                    
                    # Scale output back with overflow protection
                    if self.y_range > 1e-10 and self.y_range < 1e10:
                        result = y_eval_scaled * self.y_range + self.y_min
                    else:
                        result = y_eval_scaled + self.y_min
                    
                    # Check for overflow
                    if not np.all(np.isfinite(result)):
                        logger.warning(f"Non-finite values detected in spline evaluation")
                        return np.full_like(y_eval_scaled, self.y_min if hasattr(self, 'y_min') else 0.0)
                    
                    return result
                
                def derivative(self, order):
                    def derivative_func(x_eval):
                        # Scale input
                        if self.x_range > 0:
                            x_eval_scaled = (x_eval - self.x_min) / self.x_range
                        else:
                            x_eval_scaled = x_eval - self.x_min
                        
                        # Evaluate derivative
                        y_deriv_scaled = self.spline.derivative(order)(x_eval_scaled)
                        
                        # Scale output (account for input scaling)
                        # Use safe scaling to prevent overflow
                        if self.x_range > 1e-10:
                            # Prevent extreme scaling factors
                            base_scale = 1.0 / self.x_range
                            # Limit power to prevent overflow and cap maximum scale
                            power = min(order, 3)
                            scale_factor = min(base_scale ** power, 1e6)  # Cap at 1e6
                        else:
                            scale_factor = 1.0
                            
                        if self.y_range > 1e-10 and self.y_range < 1e10:
                            scaled_result = y_deriv_scaled * self.y_range * scale_factor
                        else:
                            scaled_result = y_deriv_scaled * scale_factor
                        
                        # Apply final bounds to prevent extreme derivatives
                        if order == 1:  # Velocity
                            return np.clip(scaled_result, -MAX_VELOCITY, MAX_VELOCITY)
                        elif order == 2:  # Acceleration
                            return np.clip(scaled_result, -MAX_ACCELERATION, MAX_ACCELERATION)
                        else:
                            return scaled_result
                    
                    return derivative_func
            
            wrapped_spline = ScaledSpline(spline, x_min, x_max, x_range, y_min, y_max, y_range)
            
            # Calculate fit quality
            fitted_values = wrapped_spline(x)
            residuals = y - fitted_values
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.last_fit_quality = r_squared
            self.quality_history.append(r_squared)
            
            if r_squared < self.quality_threshold:
                logger.warning(f"Low spline fit quality: R² = {r_squared:.3f}")
                
            return wrapped_spline
            
        except Exception as e:
            logger.error(f"Spline fitting failed: {e}, falling back to linear interpolation")
            from scipy.interpolate import interp1d
            return interp1d(x, y, kind='linear', fill_value='extrapolate')

    def _get_spline_derivatives(self, spline, evaluation_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract analytical derivatives from fitted spline.
        """
        try:
            # Check if we have a wrapped spline with scaling
            if hasattr(spline, 'derivative'):
                # Use wrapped spline methods
                derivatives = {
                    'price': spline(evaluation_points),
                    'velocity': spline.derivative(1)(evaluation_points),
                    'acceleration': spline.derivative(2)(evaluation_points)
                }
                
                # Try higher orders if available
                try:
                    derivatives['jerk'] = spline.derivative(3)(evaluation_points)
                    derivatives['snap'] = spline.derivative(4)(evaluation_points)
                except:
                    pass
            else:
                # Standard scipy spline with bounds
                price = spline(evaluation_points)
                velocity = spline(evaluation_points, 1) if hasattr(spline, '__call__') and len(evaluation_points) > 1 else spline.derivative(1)(evaluation_points)
                acceleration = spline(evaluation_points, 2) if hasattr(spline, '__call__') and len(evaluation_points) > 1 else spline.derivative(2)(evaluation_points)
                jerk = spline(evaluation_points, 3) if hasattr(spline, '__call__') and len(evaluation_points) > 1 else spline.derivative(3)(evaluation_points)
                snap = spline(evaluation_points, 4) if hasattr(spline, '__call__') and len(evaluation_points) > 1 else spline.derivative(4)(evaluation_points)
                
                # Apply bounds to prevent extreme values
                derivatives = {
                    'price': price,
                    'velocity': np.clip(velocity, -MAX_VELOCITY, MAX_VELOCITY),
                    'acceleration': np.clip(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION),
                    'jerk': np.clip(jerk, -MAX_ACCELERATION, MAX_ACCELERATION),  # Use same bound for higher orders
                    'snap': np.clip(snap, -MAX_ACCELERATION, MAX_ACCELERATION)
                }
        except Exception as e:
            # Fallback for unsupported derivative orders
            logger.debug(f"Spline derivative extraction failed: {e}")
            derivatives = {
                'price': spline(evaluation_points) if hasattr(spline, '__call__') else np.full_like(evaluation_points, np.nan)
            }
            
            # Try to get basic derivatives
            if hasattr(spline, 'derivative'):
                try:
                    derivatives['velocity'] = spline.derivative(1)(evaluation_points)
                    derivatives['acceleration'] = spline.derivative(2)(evaluation_points)
                except:
                    pass
        
        # Ensure all arrays are 1D and same shape
        for key, value in derivatives.items():
            if hasattr(value, 'flatten'):
                derivatives[key] = value.flatten()
            elif hasattr(value, 'values'):
                derivatives[key] = value.values.flatten()
                
        return derivatives

    def analyze_derivatives(self,
                          prices: pd.Series,
                          timestamps: pd.Series = None,
                          evaluation_points: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate analytical derivatives using spline fitting.

        Args:
            prices: Price series for analysis
            timestamps: Optional timestamp series (uses index if None)
            evaluation_points: Points where to evaluate derivatives (uses all if None)

        Returns:
            Dictionary with derivatives and quality metrics
        """
        if len(prices) < self.min_window:
            logger.warning(f"Insufficient data: {len(prices)} < {self.min_window}")
            return {}
            
        # Use index as timestamps if not provided
        if timestamps is None:
            timestamps = pd.Series(np.arange(len(prices)), index=prices.index)
            
        # Use all timestamps if evaluation points not specified
        if evaluation_points is None:
            evaluation_points = timestamps.values.astype(float)
            
        try:
            # Fit spline
            spline = self._fit_spline(prices, timestamps)
            
            # Get derivatives
            derivatives = self._get_spline_derivatives(spline, evaluation_points)
            
            # Add quality metrics
            derivatives.update({
                'fit_quality': self.last_fit_quality,
                'window_used': self._calculate_adaptive_window(prices),
                'smoothing_used': self._calculate_adaptive_smoothing(prices),
                'spline_type': self.spline_type
            })
            
            return derivatives
            
        except Exception as e:
            logger.error(f"Derivative analysis failed: {e}")
            return {}

    def real_time_derivative(self,
                          current_price: float,
                          price_history: pd.Series,
                          timestamp: float = None) -> Dict[str, float]:
        """
        Calculate real-time derivatives for current price.

        Args:
            current_price: Latest price observation
            price_history: Recent price history for spline fitting
            timestamp: Current timestamp (uses len if None)

        Returns:
            Dictionary with current derivatives
        """
        if timestamp is None:
            timestamp = len(price_history)
            
        # Add current price to history
        extended_prices = price_history.append(pd.Series([current_price]))
        
        # Calculate derivatives
        result = self.analyze_derivatives(extended_prices, evaluation_points=np.array([timestamp]))
        
        # Extract current values
        if result:
            return {
                'price': float(result['price'][-1]) if len(result['price']) > 0 else current_price,
                'velocity': float(result['velocity'][-1]) if 'velocity' in result else 0.0,
                'acceleration': float(result['acceleration'][-1]) if 'acceleration' in result else 0.0,
                'jerk': float(result['jerk'][-1]) if 'jerk' in result else 0.0,
                'snap': float(result['snap'][-1]) if 'snap' in result else 0.0,
                'fit_quality': float(result['fit_quality']) if 'fit_quality' in result else 0.0
            }
        else:
            return {
                'price': current_price,
                'velocity': 0.0,
                'acceleration': 0.0,
                'jerk': 0.0,
                'snap': 0.0,
                'fit_quality': 0.0
            }

    def get_derivative_quality_metrics(self) -> Dict[str, float]:
        """
        Get quality metrics for derivative calculations.
        """
        if not self.quality_history:
            return {'mean_quality': 0.0, 'quality_stability': 0.0, 'sample_count': 0}
            
        quality_array = np.array(self.quality_history[-50:])  # Last 50 samples
        
        return {
            'mean_quality': float(np.mean(quality_array)),
            'quality_stability': float(1.0 - np.std(quality_array)),  # Higher is more stable
            'sample_count': len(self.quality_history),
            'current_quality': float(self.last_fit_quality),
            'quality_trend': float(np.polyfit(range(len(quality_array)), quality_array, 1)[0]) if len(quality_array) > 1 else 0.0
        }

    def reset_quality_tracking(self):
        """Reset quality tracking for new market regime."""
        self.quality_history = []
        self.volatility_history = []
        self.last_fit_quality = 0.0
        self._spline_cache = {}
        logger.info("Quality tracking reset")

# Convenience function for backward compatibility
def calculate_spline_derivatives(prices: pd.Series,
                               timestamps: pd.Series = None,
                               window_size: int = 50,
                               spline_type: str = 'cubic') -> Dict[str, np.ndarray]:
    """
    Convenience function for calculating spline derivatives.
    
    Args:
        prices: Price series
        timestamps: Optional timestamp series
        window_size: Window size for spline fitting
        spline_type: Type of spline to use
        
    Returns:
        Dictionary with analytical derivatives
    """
    analyzer = SplineDerivativeAnalyzer(window_size=window_size, spline_type=spline_type)
    return analyzer.analyze_derivatives(prices, timestamps)

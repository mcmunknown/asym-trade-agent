"""
Kalman Filter for Anne's Calculus Trading System
===============================================

This module implements the adaptive Kalman filter for state-space modeling
following Anne's calculus approach:

7️⃣ Kalman filter – adaptive update

State-space model:
sₜ = [P̂ₜ, vₜ, aₜ]ᵀ
sₜ₊₁ = A·sₜ + wₜ
P̂ₜ^obs = [1, 0, 0]·sₜ + vₜ^obs

Interpretation:
* A encodes the relationship between level, slope, and curvature
* wₜ and vₜ^obs are process & observation noise
* The Kalman equations update sₜ as new prices arrive

Purpose: keeps a live, statistically-weighted estimate of price, slope, and curvature
— smooth yet responsive
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
logger = logging.getLogger(__name__)

@dataclass
class KalmanState:
    """Kalman filter state containing estimates and covariances"""
    state_estimate: np.ndarray  # [price, velocity, acceleration]
    covariance_matrix: np.ndarray  # 3x3 covariance matrix
    timestamp: float
    innovation: Optional[float] = None  # Measurement residual
    innovation_covariance: Optional[float] = None  # Innovation covariance

@dataclass
class KalmanConfig:
    """Configuration parameters for the Kalman filter"""
    # Process noise covariance (how much we expect the state to change)
    process_noise_price: float = 1e-5
    process_noise_velocity: float = 1e-6
    process_noise_acceleration: float = 1e-7

    # Observation noise covariance (measurement noise)
    observation_noise: float = 1e-4

    # Initial state uncertainty
    initial_uncertainty_price: float = 1.0
    initial_uncertainty_velocity: float = 1.0
    initial_uncertainty_acceleration: float = 1.0

    # Time step (Δt)
    dt: float = 1.0

class AdaptiveKalmanFilter:
    """
    Adaptive Kalman filter for real-time price analysis with dynamic noise estimation.

    This filter implements the state-space model:
    sₜ = [P̂ₜ, vₜ, aₜ]ᵀ where:
    - P̂ₜ: smoothed price estimate
    - vₜ: velocity (first derivative)
    - aₜ: acceleration (second derivative)
    """

    def __init__(self, config: KalmanConfig = None):
        """
        Initialize the adaptive Kalman filter.

        Args:
            config: Configuration parameters for the filter
        """
        self.config = config or KalmanConfig()
        self.state = None
        self.initialized = False
        self.innovation_history = []
        self.max_history_length = 100

        # Initialize state transition matrix A
        self._initialize_state_transition_matrix()

        # Initialize observation matrix H = [1, 0, 0] (we observe price directly)
        self.H = np.array([[1.0, 0.0, 0.0]])

        # Initialize process noise covariance matrix Q
        self._initialize_process_noise_matrix()

        # Initialize observation noise covariance matrix R
        self.R = np.array([[self.config.observation_noise]])

        logger.info("Adaptive Kalman filter initialized")

    def _initialize_state_transition_matrix(self):
        """
        Initialize state transition matrix A for constant acceleration model:

        sₜ₊₁ = A·sₜ + wₜ

        where:
        P̂ₜ₊₁ = P̂ₜ + vₜ·Δt + 0.5·aₜ·Δt²
        vₜ₊₁ = vₜ + aₜ·Δt
        aₜ₊₁ = aₜ  (assuming constant acceleration)

        State vector: sₜ = [P̂ₜ, vₜ, aₜ]ᵀ
        """
        dt = self.config.dt
        self.A = np.array([
            [1.0, dt, 0.5 * dt**2],  # Price update
            [0.0, 1.0, dt],         # Velocity update
            [0.0, 0.0, 1.0]          # Acceleration update (constant)
        ])

    def _initialize_process_noise_matrix(self):
        """
        Initialize process noise covariance matrix Q.

        Process noise represents uncertainty in the state transition model.
        We expect more uncertainty in price than velocity, and more in velocity than acceleration.
        """
        q_p = self.config.process_noise_price
        q_v = self.config.process_noise_velocity
        q_a = self.config.process_noise_acceleration

        # Simplified diagonal Q matrix (can be extended with cross-correlations)
        self.Q = np.diag([q_p, q_v, q_a])

    def _initialize_state(self, initial_price: float):
        """
        Initialize the filter state with first observation.

        Args:
            initial_price: First price observation
        """
        # Initial state estimate: [price, velocity=0, acceleration=0]
        self.state = KalmanState(
            state_estimate=np.array([initial_price, 0.0, 0.0]),
            covariance_matrix=np.diag([
                self.config.initial_uncertainty_price,
                self.config.initial_uncertainty_velocity,
                self.config.initial_uncertainty_acceleration
            ]),
            timestamp=0.0
        )
        self.initialized = True
        logger.info(f"Kalman filter initialized with price: {initial_price}")

    def predict(self, dt: float = None) -> KalmanState:
        """
        Prediction step: sₜ₊₁|ₜ = A·sₜ|ₜ

        Args:
            dt: Time step since last update (uses config.dt if None)

        Returns:
            Predicted state
        """
        if not self.initialized:
            raise ValueError("Filter not initialized. Call update() first.")

        if dt is not None:
            # Update state transition matrix for new time step
            old_dt = self.config.dt
            self.config.dt = dt
            self._initialize_state_transition_matrix()
            # Update process noise for new time step
            self._initialize_process_noise_matrix()

        # Predict state
        predicted_state = self.A @ self.state.state_estimate

        # Predict covariance
        predicted_covariance = self.A @ self.state.covariance_matrix @ self.A.T + self.Q

        # Create new state with prediction
        predicted = KalmanState(
            state_estimate=predicted_state,
            covariance_matrix=predicted_covariance,
            timestamp=self.state.timestamp + (dt or self.config.dt)
        )

        # Restore original dt if it was changed
        if dt is not None:
            self.config.dt = old_dt
            self._initialize_state_transition_matrix()
            self._initialize_process_noise_matrix()

        return predicted

    def update(self, measurement: float, dt: float = None) -> KalmanState:
        """
        Update step: sₜ₊₁|ₜ₊₁ = sₜ₊₁|ₜ + K·(zₜ₊₁ - H·sₜ₊₁|ₜ)

        Args:
            measurement: New price observation
            dt: Time step since last update (uses config.dt if None)

        Returns:
            Updated state
        """
        if not self.initialized:
            self._initialize_state(measurement)
            return self.state

        # Prediction step
        predicted = self.predict(dt)

        # Calculate innovation (measurement residual) - ensure scalar result
        measurement_matrix_result = self.H @ predicted.state_estimate
        if measurement_matrix_result.ndim == 1:
            predicted_measurement = float(measurement_matrix_result[0])
        else:
            predicted_measurement = float(measurement_matrix_result[0, 0])
        innovation = measurement - predicted_measurement

        # Calculate innovation covariance - ensure scalar result
        innovation_covariance_matrix = self.H @ predicted.covariance_matrix @ self.H.T + self.R
        if innovation_covariance_matrix.ndim == 1:
            innovation_covariance = float(innovation_covariance_matrix[0])
        else:
            innovation_covariance = float(innovation_covariance_matrix[0, 0])

        # Calculate Kalman gain
        kalman_gain = predicted.covariance_matrix @ self.H.T / innovation_covariance

        # Update state estimate
        updated_state = predicted.state_estimate + kalman_gain * innovation

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(3)  # Identity matrix
        updated_covariance = (
            (I - kalman_gain @ self.H) @ predicted.covariance_matrix @
            (I - kalman_gain @ self.H).T + kalman_gain @ self.R @ kalman_gain.T
        )

        # Create updated state
        self.state = KalmanState(
            state_estimate=updated_state,
            covariance_matrix=updated_covariance,
            timestamp=predicted.timestamp,
            innovation=innovation,
            innovation_covariance=innovation_covariance
        )

        # Store innovation for adaptive noise estimation
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.max_history_length:
            self.innovation_history.pop(0)

        # Adapt noise parameters based on innovation statistics
        self._adapt_noise_parameters()

        return self.state

    def _adapt_noise_parameters(self):
        """
        Adaptively adjust noise parameters based on innovation statistics.

        This makes the filter responsive to changing market conditions:
        - Increase process noise during high volatility
        - Decrease process noise during stable periods
        - Adjust observation noise based on innovation variance
        """
        if len(self.innovation_history) < 10:
            return  # Need enough history for reliable statistics

        # Calculate innovation statistics
        innovations = np.array(self.innovation_history[-20:])  # Use last 20 innovations
        innovation_variance = np.var(innovations)
        innovation_mean = np.mean(innovations)

        # Expected innovation variance (trace of innovation covariance)
        expected_variance = np.trace(self.R + self.H @ self.state.covariance_matrix @ self.H.T)

        # Adapt observation noise
        if innovation_variance > 0:
            # Scale observation noise based on actual vs expected innovation variance
            adaptation_factor = innovation_variance / expected_variance
            adaptation_factor = np.clip(adaptation_factor, 0.5, 2.0)  # Limit adaptation

            # Smooth adaptation to avoid sudden changes
            self.R[0, 0] = 0.9 * self.R[0, 0] + 0.1 * (self.config.observation_noise * adaptation_factor)

        # Adapt process noise based on innovation magnitude
        innovation_magnitude = np.abs(innovation_mean)
        if innovation_magnitude > 0:
            # Increase process noise during large deviations
            process_noise_factor = 1.0 + min(innovation_magnitude / 100.0, 1.0)
            self.Q *= process_noise_factor

            # Renormalize to maintain reasonable values
            self.Q *= (self.config.process_noise_price / np.diag(self.Q)[0])

    def get_current_estimates(self) -> Dict:
        """
        Get current state estimates with confidence intervals.

        Returns:
            Dictionary with current estimates and uncertainties
        """
        if not self.initialized or self.state is None:
            return {}

        state = self.state
        if state.state_estimate is None:
            return {}

        # Safely extract standard deviations with proper numpy array handling
        if state.covariance_matrix is not None:
            std_devs = np.sqrt(np.diag(state.covariance_matrix))
        else:
            std_devs = np.array([1.0, 1.0, 1.0])

        # Safely extract state estimates with proper numpy array handling
        if state.state_estimate is not None and len(state.state_estimate) > 0:
            price_estimate = float(state.state_estimate.item(0))
        else:
            price_estimate = 0.0

        if state.state_estimate is not None and len(state.state_estimate) > 1:
            velocity_estimate = float(state.state_estimate.item(1))
        else:
            velocity_estimate = 0.0

        if state.state_estimate is not None and len(state.state_estimate) > 2:
            acceleration_estimate = float(state.state_estimate.item(2))
        else:
            acceleration_estimate = 0.0

        # Safely extract uncertainty values with proper numpy array handling
        price_uncertainty = float(std_devs.item(0)) if len(std_devs) > 0 else 1.0
        velocity_uncertainty = float(std_devs.item(1)) if len(std_devs) > 1 else 1.0
        acceleration_uncertainty = float(std_devs.item(2)) if len(std_devs) > 2 else 1.0

        # Innovation values are now stored as scalars after conversion in update()
        innovation_value = float(state.innovation) if state.innovation is not None else 0.0
        innovation_covariance_value = float(state.innovation_covariance) if state.innovation_covariance is not None else 1.0

        return {
            'price_estimate': price_estimate,
            'price_uncertainty': price_uncertainty,
            'velocity_estimate': velocity_estimate,
            'velocity_uncertainty': velocity_uncertainty,
            'acceleration_estimate': acceleration_estimate,
            'acceleration_uncertainty': acceleration_uncertainty,
            'timestamp': float(state.timestamp) if state.timestamp is not None else 0.0,
            'innovation': innovation_value,
            'innovation_covariance': innovation_covariance_value
        }

    def filter_price_series(self, prices: pd.Series, timestamps: pd.Series = None) -> pd.DataFrame:
        """
        Apply Kalman filter to a price series.

        Args:
            prices: Price series to filter
            timestamps: Optional timestamp series (uses index if None)

        Returns:
            DataFrame with filtered estimates and uncertainties
        """
        if len(prices) < 2:
            logger.error("Need at least 2 price points for Kalman filtering")
            return pd.DataFrame()

        logger.info(f"Applying Kalman filter to {len(prices)} price points")

        results = []
        self.reset()  # Reset filter state

        for i, price in enumerate(prices):
            if timestamps is not None:
                dt = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds() if i > 0 else 1.0
            else:
                dt = 1.0  # Assume unit time steps

            state = self.update(float(price), dt)
            estimates = self.get_current_estimates()

            results.append({
                'timestamp': timestamps.iloc[i] if timestamps is not None else prices.index[i],
                'raw_price': float(price),
                'filtered_price': estimates['price_estimate'],
                'price_uncertainty': estimates['price_uncertainty'],
                'velocity': estimates['velocity_estimate'],
                'velocity_uncertainty': estimates['velocity_uncertainty'],
                'acceleration': estimates['acceleration_estimate'],
                'acceleration_uncertainty': estimates['acceleration_uncertainty'],
                'innovation': estimates['innovation'],
                'innovation_covariance': estimates['innovation_covariance']
            })

        df = pd.DataFrame(results)
        df.set_index('timestamp', inplace=True)

        logger.info("Kalman filtering completed successfully")
        return df

    def get_forecast(self, steps_ahead: int = 1) -> Dict:
        """
        Generate multi-step ahead forecast using Kalman filter.

        Args:
            steps_ahead: Number of steps to forecast ahead

        Returns:
            Dictionary with forecasted values and uncertainties
        """
        if not self.initialized:
            return {}

        forecasts = []
        current_state = self.state

        for step in range(1, steps_ahead + 1):
            # Predict multiple steps ahead
            predicted = self.predict(self.config.dt)

            std_devs = np.sqrt(np.diag(predicted.covariance_matrix))

            # Safe extraction of forecast values with numpy item() method
            forecast_price = float(predicted.state_estimate.item(0)) if len(predicted.state_estimate) > 0 else 0.0
            forecast_velocity = float(predicted.state_estimate.item(1)) if len(predicted.state_estimate) > 1 else 0.0
            forecast_acceleration = float(predicted.state_estimate.item(2)) if len(predicted.state_estimate) > 2 else 0.0

            price_uncertainty = float(std_devs.item(0)) if len(std_devs) > 0 else 1.0
            velocity_uncertainty = float(std_devs.item(1)) if len(std_devs) > 1 else 1.0
            acceleration_uncertainty = float(std_devs.item(2)) if len(std_devs) > 2 else 1.0

            forecasts.append({
                'step': step,
                'forecast_price': forecast_price,
                'price_uncertainty': price_uncertainty,
                'forecast_velocity': forecast_velocity,
                'velocity_uncertainty': velocity_uncertainty,
                'forecast_acceleration': forecast_acceleration,
                'acceleration_uncertainty': acceleration_uncertainty
            })

            # Update current state for next prediction
            self.state = predicted

        return {
            'forecasts': forecasts,
            'base_timestamp': current_state.timestamp
        }

    def reset(self):
        """Reset filter to uninitialized state"""
        self.state = None
        self.initialized = False
        self.innovation_history = []
        logger.info("Kalman filter reset")

    def get_filter_statistics(self) -> Dict:
        """
        Get filter performance statistics.

        Returns:
            Dictionary with filter statistics
        """
        if not self.initialized or len(self.innovation_history) == 0:
            return {}

        innovations = np.array(self.innovation_history)

        # Safe extraction of statistics with numpy item() method
        innovation_mean = float(np.mean(innovations).item()) if len(innovations) > 0 else 0.0
        innovation_std = float(np.std(innovations).item()) if len(innovations) > 0 else 0.0
        innovation_variance = float(np.var(innovations).item()) if len(innovations) > 0 else 0.0
        max_innovation = float(np.max(np.abs(innovations)).item()) if len(innovations) > 0 else 0.0

        current_observation_noise = float(self.R[0, 0].item()) if self.R is not None and self.R.size > 0 else 0.0
        current_process_noise_diagonal = list(np.diag(self.Q).flatten()) if self.Q is not None else []

        if self.state is not None and self.state.covariance_matrix is not None:
            state_covariance_trace = float(np.trace(self.state.covariance_matrix).item())
            current_price_uncertainty = float(np.sqrt(self.state.covariance_matrix[0, 0]).item())
        else:
            state_covariance_trace = 0.0
            current_price_uncertainty = 0.0

        return {
            'total_updates': len(innovations),
            'innovation_mean': innovation_mean,
            'innovation_std': innovation_std,
            'innovation_variance': innovation_variance,
            'max_innovation': max_innovation,
            'current_observation_noise': current_observation_noise,
            'current_process_noise_diagonal': current_process_noise_diagonal,
            'state_covariance_trace': state_covariance_trace,
            'current_price_uncertainty': current_price_uncertainty
        }

# Legacy compatibility function
def apply_kalman_filter(prices: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.

    Args:
        prices: Price series to filter
        **kwargs: Additional parameters for KalmanConfig

    Returns:
        DataFrame with filtered results
    """
    config = KalmanConfig(**kwargs)
    filter_instance = AdaptiveKalmanFilter(config)
    return filter_instance.filter_price_series(prices)

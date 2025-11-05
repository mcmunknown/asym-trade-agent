"""
Anne's Calculus-Based Trading Strategy
======================================

This module implements the complete 6-case decision logic matrix
following Anne's calculus approach:

Formula → Meaning → Worked Example → Trading Decision

Each signal is validated using Signal-to-Noise Ratio (SNR) and
includes confidence scoring for risk management.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from enum import Enum
from quantitative_models import CalculusPriceAnalyzer, safe_finite_check, epsilon_compare, EPSILON, MAX_SAFE_VALUE

logger = logging.getLogger(__name__)

# Additional safety constants for strategy logic
VELOCITY_THRESHOLD = 1e-6  # Threshold for considering velocity "zero"

class SignalType(Enum):
    """Trading signal types following Anne's calculus decision matrix"""
    NEUTRAL = 0
    BUY = 1
    SELL = -1
    STRONG_BUY = 2
    STRONG_SELL = -2
    TRAIL_STOP_UP = 3
    TAKE_PROFIT = 4
    HOLD_SHORT = 5
    LOOK_FOR_REVERSAL = 6
    POSSIBLE_LONG = 7
    POSSIBLE_EXIT_SHORT = 8

class CalculusTradingStrategy:
    """
    Implements Anne's complete calculus-based trading decision matrix.

    6️⃣ Decision logic – turning geometry into trading rules:

    | Condition  | Interpretation         | Generic action      |
    | ---------- | ---------------------- | ------------------- |
    | (v>0, a>0) | uptrend accelerating   | trail stop upward   |
    | (v>0, a<0) | uptrend slowing        | take profit         |
    | (v<0, a<0) | downtrend accelerating | hold short          |
    | (v<0, a>0) | downtrend weakening    | look for reversal   |
    | (v=0,a>0)  | curvature bottom       | possible long entry |
    | (v=0,a<0)  | curvature top          | possible exit/short |
    """

    def __init__(self, lambda_param: float = 0.6, snr_threshold: float = 1.0,
                 confidence_threshold: float = 0.7):
        """
        Initialize the calculus trading strategy.

        Args:
            lambda_param: Smoothing parameter for exponential smoothing
            snr_threshold: Minimum SNR for valid signals
            confidence_threshold: Minimum confidence for trading decisions
        """
        self.analyzer = CalculusPriceAnalyzer(
            lambda_param=lambda_param,
            snr_threshold=snr_threshold
        )
        self.confidence_threshold = confidence_threshold
        self.signal_history = []

    def analyze_curve_geometry(self, velocity: float, acceleration: float,
                              snr: float) -> Tuple[SignalType, str, float]:
        """
        6️⃣ Decision logic – turning geometry into trading rules

        Args:
            velocity: Current velocity (first derivative)
            acceleration: Current acceleration (second derivative)
            snr: Signal-to-noise ratio

        Returns:
            Tuple of (signal_type, interpretation, confidence)
        """
        # Safety checks for input values
        velocity = safe_finite_check(velocity)
        acceleration = safe_finite_check(acceleration)
        snr = safe_finite_check(snr)

        # Define confidence based on SNR
        confidence = min(1.0, snr / 2.0)  # Normalize SNR to confidence

        # Check SNR threshold first
        if epsilon_compare(snr, self.analyzer.snr_threshold) < 0:
            return SignalType.NEUTRAL, f"Low SNR ({snr:.2f} < {self.analyzer.snr_threshold})", 0.0

        # Anne's 6-case decision matrix with epsilon-based comparisons
        if epsilon_compare(velocity, 0.0) > 0 and epsilon_compare(acceleration, 0.0) > 0:
            # (v>0, a>0): uptrend accelerating → trail stop upward
            return SignalType.TRAIL_STOP_UP, "Uptrend accelerating", confidence

        elif epsilon_compare(velocity, 0.0) > 0 and epsilon_compare(acceleration, 0.0) < 0:
            # (v>0, a<0): uptrend slowing → take profit
            return SignalType.TAKE_PROFIT, "Uptrend slowing", confidence

        elif epsilon_compare(velocity, 0.0) < 0 and epsilon_compare(acceleration, 0.0) < 0:
            # (v<0, a<0): downtrend accelerating → hold short
            return SignalType.HOLD_SHORT, "Downtrend accelerating", confidence

        elif epsilon_compare(velocity, 0.0) < 0 and epsilon_compare(acceleration, 0.0) > 0:
            # (v<0, a>0): downtrend weakening → look for reversal
            return SignalType.LOOK_FOR_REVERSAL, "Downtrend weakening", confidence

        elif epsilon_compare(abs(velocity), VELOCITY_THRESHOLD) < 1 and epsilon_compare(acceleration, 0.0) > 0:
            # (v≈0, a>0): curvature bottom → possible long entry
            return SignalType.POSSIBLE_LONG, "Curvature bottom forming", confidence

        elif epsilon_compare(abs(velocity), VELOCITY_THRESHOLD) < 1 and epsilon_compare(acceleration, 0.0) < 0:
            # (v≈0, a<0): curvature top → possible exit/short
            return SignalType.POSSIBLE_EXIT_SHORT, "Curvature top forming", confidence

        else:
            return SignalType.NEUTRAL, "No clear pattern", confidence

    def detect_crossovers(self, velocity_series: pd.Series, acceleration_series: pd.Series) -> pd.Series:
        """
        Detect velocity and acceleration crossovers for strong signals.

        Strong Buy: velocity crossing from negative to positive with positive acceleration
        Strong Sell: velocity crossing from positive to negative with negative acceleration
        """
        # Default to neutral (0) to avoid NaN values when no crossover is detected
        signals = pd.Series(0, index=velocity_series.index, dtype=int)

        for i in range(1, len(velocity_series)):
            # Strong Buy detection
            if (velocity_series.iloc[i-1] < 0 and velocity_series.iloc[i] > 0 and
                acceleration_series.iloc[i] > 0):
                signals.iloc[i] = SignalType.STRONG_BUY.value

            # Strong Sell detection
            elif (velocity_series.iloc[i-1] > 0 and velocity_series.iloc[i] < 0 and
                  acceleration_series.iloc[i] < 0):
                signals.iloc[i] = SignalType.STRONG_SELL.value

        return signals

    def generate_trading_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate complete trading signals using Anne's calculus approach.

        Args:
            prices: Price series for analysis

        Returns:
            DataFrame with all calculus indicators and trading signals
        """
        logger.info(f"Generating calculus-based trading signals for {len(prices)} price points")

        if len(prices) < 20:
            logger.error("Insufficient data for signal generation")
            return pd.DataFrame()

        # Complete calculus analysis
        analysis = self.analyzer.analyze_price_curve(prices)
        if analysis.empty:
            return pd.DataFrame()

        # Initialize signal dataframe
        signals = analysis.copy()

        # Add signal interpretation
        signals['signal_type'] = SignalType.NEUTRAL.value
        signals['interpretation'] = "Neutral"
        signals['confidence'] = 0.0

        # Analyze each point with enhanced safety checks
        for i in range(len(signals)):
            if pd.isna(signals['velocity'].iloc[i]) or pd.isna(signals['acceleration'].iloc[i]):
                continue

            velocity = safe_finite_check(signals['velocity'].iloc[i])
            acceleration = safe_finite_check(signals['acceleration'].iloc[i])
            snr = safe_finite_check(signals['snr'].iloc[i])

            # Additional safety checks for extreme values
            if abs(velocity) > MAX_SAFE_VALUE or abs(acceleration) > MAX_SAFE_VALUE or snr > MAX_SAFE_VALUE:
                logger.warning(f"Extreme values detected at index {i}: v={velocity:.2e}, a={acceleration:.2e}, snr={snr:.2e}")
                continue

            signal_type, interpretation, confidence = self.analyze_curve_geometry(
                velocity, acceleration, snr
            )

            signals.at[signals.index[i], 'signal_type'] = signal_type.value
            signals.at[signals.index[i], 'interpretation'] = interpretation
            signals.at[signals.index[i], 'confidence'] = confidence

        # Detect crossovers for strong signals
        crossover_signals = self.detect_crossovers(
            signals['velocity'], signals['acceleration']
        )

        # Override with strong signals where detected
        for i, strong_signal in crossover_signals.items():
            if pd.notna(strong_signal) and strong_signal != 0:
                signals.at[i, 'signal_type'] = strong_signal
                if strong_signal == SignalType.STRONG_BUY.value:
                    signals.at[i, 'interpretation'] = "Strong Buy - velocity cross + with positive acceleration"
                    signals.at[i, 'confidence'] = 1.0
                elif strong_signal == SignalType.STRONG_SELL.value:
                    signals.at[i, 'interpretation'] = "Strong Sell - velocity cross - with negative acceleration"
                    signals.at[i, 'confidence'] = 1.0

        # Add valid signal flag
        signals['valid_signal'] = (
            (signals['confidence'] >= self.confidence_threshold) &
            (signals['valid_signal'])  # SNR validation
        )

        # Add traditional signal for compatibility
        signals['signal'] = signals['signal_type'].map({
            SignalType.NEUTRAL.value: 0,
            SignalType.BUY.value: 1,
            SignalType.SELL.value: -1,
            SignalType.STRONG_BUY.value: 2,
            SignalType.STRONG_SELL.value: -2,
            SignalType.TRAIL_STOP_UP.value: 3,
            SignalType.TAKE_PROFIT.value: 4,
            SignalType.HOLD_SHORT.value: 5,
            SignalType.LOOK_FOR_REVERSAL.value: 6,
            SignalType.POSSIBLE_LONG.value: 7,
            SignalType.POSSIBLE_EXIT_SHORT.value: 8,
        })

        # Log signal statistics
        signal_counts = signals['signal_type'].value_counts()
        logger.info(f"Signal generation completed:")
        for signal_type, count in signal_counts.items():
            logger.info(f"  {SignalType(signal_type).name}: {count}")

        valid_signals = signals['valid_signal'].sum()
        logger.info(f"Valid trading signals: {valid_signals}/{len(signals)} "
                   f"({valid_signals/len(signals)*100:.1f}%)")

        return signals

    def get_latest_signal(self, prices: pd.Series) -> Dict:
        """
        Get the latest trading signal with all details.

        Args:
            prices: Recent price data

        Returns:
            Dictionary with signal details
        """
        signals = self.generate_trading_signals(prices)
        if signals.empty:
            return {}

        latest = signals.iloc[-1]
        return {
            'signal_type': SignalType(latest['signal_type']),
            'interpretation': latest['interpretation'],
            'confidence': latest['confidence'],
            'velocity': latest['velocity'],
            'acceleration': latest['acceleration'],
            'snr': latest['snr'],
            'forecast': latest['forecast'],
            'valid_signal': latest['valid_signal'],
            'timestamp': latest.name
        }

# Legacy function for backward compatibility
def generate_trading_signals(prices: pd.Series) -> pd.DataFrame:
    """Legacy wrapper for signal generation"""
    strategy = CalculusTradingStrategy()
    return strategy.generate_trading_signals(prices)

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
from quantitative_models import (
    CalculusPriceAnalyzer, safe_finite_check, epsilon_compare, EPSILON, MAX_SAFE_VALUE,
    MAX_VELOCITY, MAX_ACCELERATION, MAX_SNR
)

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

    def generate_trading_signals(self, prices: pd.Series, context: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
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

        context = context or {}
        kalman_drift = context.get('kalman_drift')
        kalman_volatility = context.get('kalman_volatility')
        regime_context = context.get('regime_context')
        delta_t = context.get('delta_t', 1.0)

        # Complete calculus analysis
        analysis = self.analyzer.analyze_price_curve(
            prices,
            kalman_drift=kalman_drift,
            kalman_volatility=kalman_volatility,
            regime_context=regime_context,
            delta_t=delta_t
        )
        if analysis.empty:
            return pd.DataFrame()

        # Initialize signal dataframe
        signals = analysis.copy()

        # Add signal interpretation
        signals['signal_type'] = SignalType.NEUTRAL.value
        signals['interpretation'] = "Neutral"
        signals['confidence'] = 0.0
        signals['stochastic_confidence'] = 0.0
        signals['hedge_directive'] = "Neutral"
        signals['tp_enhanced_signal'] = SignalType.NEUTRAL.value
        signals['tp_risk_adjusted'] = False

        delta_series = signals.get('optimal_delta', pd.Series(0.0, index=signals.index))
        hjb_series = signals.get('hjb_action', pd.Series(0.0, index=signals.index))
        residual_series = signals.get('residual_variance', pd.Series(0.0, index=signals.index))
        vol_series = signals.get('stochastic_volatility', pd.Series(0.0, index=signals.index))

        # Analyze each point with enhanced safety checks
        for i in range(len(signals)):
            if pd.isna(signals['velocity'].iloc[i]) or pd.isna(signals['acceleration'].iloc[i]):
                continue

            velocity = safe_finite_check(signals['velocity'].iloc[i])
            acceleration = safe_finite_check(signals['acceleration'].iloc[i])
            snr = safe_finite_check(signals['snr'].iloc[i])
            delta = safe_finite_check(delta_series.iloc[i])
            hjb_action = safe_finite_check(hjb_series.iloc[i])
            residual = abs(safe_finite_check(residual_series.iloc[i], 0.0))
            stochastic_vol = abs(safe_finite_check(vol_series.iloc[i], 0.0))

            # Apply early clipping to prevent extreme values
            velocity_clipped = np.clip(velocity, -MAX_VELOCITY, MAX_VELOCITY)
            acceleration_clipped = np.clip(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION)
            snr_clipped = np.clip(snr, 0, MAX_SNR)
            
            # Log if values were clipped
            if velocity_clipped != velocity or acceleration_clipped != acceleration or snr_clipped != snr:
                logger.debug(f"Clipped values at index {i}: "
                           f"v {velocity:.2e}→{velocity_clipped:.2e}, "
                           f"a {acceleration:.2e}→{acceleration_clipped:.2e}, "
                           f"snr {snr:.2f}→{snr_clipped:.2f}")
                
                # Update values with clipped versions
                velocity = velocity_clipped
                acceleration = acceleration_clipped
                snr = snr_clipped
            
            # Final safety check for truly extreme values (should rarely trigger)
            if abs(velocity) > MAX_VELOCITY or abs(acceleration) > MAX_ACCELERATION or snr > MAX_SNR:
                logger.warning(f"Extreme values still detected at index {i}: v={velocity:.2e}, a={acceleration:.2e}, snr={snr:.2f}")
                continue

            hedge_quality = 1.0 / (1.0 + residual)
            control_alignment = np.tanh(abs(hjb_action))
            vol_penalty = 1.0 / (1.0 + stochastic_vol)
            stochastic_confidence = np.clip(0.5 * (hedge_quality + control_alignment) * vol_penalty, 0.0, 1.0)

            signal_type, interpretation, base_confidence = self.analyze_curve_geometry(
                velocity, acceleration, snr
            )

            combined_confidence = min(1.0, base_confidence * (0.5 + 0.5 * stochastic_confidence))

            signals.at[signals.index[i], 'signal_type'] = signal_type.value
            signals.at[signals.index[i], 'interpretation'] = f"{interpretation} | Δ={delta:+.3f}, a*={hjb_action:+.3f}"
            signals.at[signals.index[i], 'confidence'] = combined_confidence
            signals.at[signals.index[i], 'stochastic_confidence'] = stochastic_confidence
            signals.at[signals.index[i], 'hedge_directive'] = f"Δ={delta:+.3f}, residual={residual:.4f}"

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

        snr_valid = signals['valid_signal'].copy()
        signals['valid_signal'] = (
            snr_valid &
            (signals['confidence'] >= self.confidence_threshold) &
            (signals['stochastic_confidence'] >= 0.4)
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

        # Enhanced TP-Probability Signal Adjustment
        tp_advantage_threshold = 0.1  # Need at least 10% TP advantage

        for i in range(len(signals)):
            if 'tp_advantage' in signals.columns and pd.notna(signals['tp_advantage'].iloc[i]):
                tp_advantage = signals['tp_advantage'].iloc[i]
                base_signal = signals['signal_type'].iloc[i]

                # Enhance signals with TP probability advantage
                if tp_advantage > tp_advantage_threshold:
                    # TP is more likely - upgrade bullish signals
                    if base_signal in [SignalType.BUY.value, SignalType.POSSIBLE_LONG.value]:
                        signals.at[i, 'tp_enhanced_signal'] = SignalType.STRONG_BUY.value
                        signals.at[i, 'tp_risk_adjusted'] = True
                        signals.at[i, 'confidence'] = min(signals.at[i, 'confidence'] + 0.2, 1.0)
                    elif base_signal == SignalType.NEUTRAL.value and tp_advantage > 0.2:
                        signals.at[i, 'tp_enhanced_signal'] = SignalType.BUY.value
                        signals.at[i, 'tp_risk_adjusted'] = True

                elif tp_advantage < -tp_advantage_threshold:
                    # SL is more likely - upgrade bearish signals
                    if base_signal in [SignalType.SELL.value, SignalType.POSSIBLE_EXIT_SHORT.value]:
                        signals.at[i, 'tp_enhanced_signal'] = SignalType.STRONG_SELL.value
                        signals.at[i, 'tp_risk_adjusted'] = True
                        signals.at[i, 'confidence'] = min(signals.at[i, 'confidence'] + 0.2, 1.0)
                    elif base_signal == SignalType.NEUTRAL.value and tp_advantage < -0.2:
                        signals.at[i, 'tp_enhanced_signal'] = SignalType.SELL.value
                        signals.at[i, 'tp_risk_adjusted'] = True

        # Final valid signal check with TP probability enhancement
        signals['tp_enhanced_valid'] = (
            signals['valid_signal'] &
            signals['tp_risk_adjusted'] &
            (signals['confidence'] >= self.confidence_threshold)
        )

        valid_signals = signals['valid_signal'].sum()
        tp_enhanced_valid = signals['tp_enhanced_valid'].sum()

        logger.info(f"Valid trading signals: {valid_signals}/{len(signals)} "
                   f"({valid_signals/len(signals)*100:.1f}%)")
        logger.info(f"TP-enhanced signals: {tp_enhanced_valid}/{len(signals)} "
                   f"({tp_enhanced_valid/len(signals)*100:.1f}%)")

        if 'tp_advantage' in signals.columns:
            avg_tp_advantage = signals['tp_advantage'].mean()
            logger.info(f"Average TP advantage: {avg_tp_advantage:.3f}")

        logger.info("Enhanced calculus-based trading signal generation completed")
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
        signal_value = latest.get('signal_type', SignalType.NEUTRAL.value)
        try:
            signal_type_value = SignalType(signal_value) if not pd.isna(signal_value) else SignalType.NEUTRAL
        except (ValueError, TypeError):
            signal_type_value = SignalType.NEUTRAL

        return {
            'signal_type': signal_type_value,
            'interpretation': latest['interpretation'],
            'confidence': latest['confidence'],
            'stochastic_confidence': latest.get('stochastic_confidence', 0.0),
            'velocity': latest['velocity'],
            'acceleration': latest['acceleration'],
            'snr': latest['snr'],
            'forecast': latest['forecast'],
            'optimal_delta': latest.get('optimal_delta', 0.0),
            'hjb_action': latest.get('hjb_action', 0.0),
            'stochastic_volatility': latest.get('stochastic_volatility', 0.0),
            'ito_correction': latest.get('ito_correction', 0.0),
            'hedge_directive': latest.get('hedge_directive', "Neutral"),
            'valid_signal': latest['valid_signal'],
            'timestamp': latest.name
        }

# Legacy function for backward compatibility
def generate_trading_signals(prices: pd.Series) -> pd.DataFrame:
    """Legacy wrapper for signal generation"""
    strategy = CalculusTradingStrategy()
    return strategy.generate_trading_signals(prices)

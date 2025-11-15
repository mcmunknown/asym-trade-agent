"""
Daily Drift Prediction Model (Renaissance Layer 5)
==================================================

Predicts expected return for the next hour using 6 market factors.
This is a lightweight implementation optimized for small accounts.

6 Factors:
1. Price momentum (last 1h return)
2. Volatility regime (recent ATR)
3. Volume trend (increasing/decreasing)
4. Time of day (trading session)
5. Day of week (Mon-Fri patterns)
6. Mean reversion signal (distance from mean)

Formula: E[r] = β₀ + β₁·momentum + β₂·vol + β₃·volume + β₄·time + β₅·day + β₆·reversion
"""

import numpy as np
from datetime import datetime, time as dt_time
from typing import Dict, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DailyDriftPredictor:
    """
    Lightweight drift predictor using 6 market factors.

    Predicts hourly expected return to align with calculus signals.
    """

    def __init__(self, lookback: int = 100):
        """Initialize drift predictor."""
        self.lookback = lookback
        self.price_history = {}  # symbol -> deque of prices
        self.volume_history = {}  # symbol -> deque of volumes
        self.timestamps = {}  # symbol -> deque of timestamps

        # Simple learned coefficients (would be trained on historical data)
        # For now, using reasonable defaults
        self.beta = {
            'intercept': 0.0001,  # Slight positive drift (crypto generally trends up)
            'momentum': 0.3,  # Momentum continuation
            'volatility': -0.1,  # High vol → mean reversion
            'volume': 0.05,  # Volume confirms direction
            'time_of_day': 0.02,  # Session effects
            'day_of_week': 0.01,  # Weekly patterns
            'mean_reversion': -0.15  # Distance from mean
        }

    def update(self, symbol: str, price: float, volume: float = 0, timestamp: float = None):
        """
        Update with latest market data.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trade volume (optional)
            timestamp: Unix timestamp (optional)
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
            self.volume_history[symbol] = deque(maxlen=self.lookback)
            self.timestamps[symbol] = deque(maxlen=self.lookback)

        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.timestamps[symbol].append(timestamp or datetime.now().timestamp())

    def predict_drift(self, symbol: str) -> Dict:
        """
        Predict expected return for next hour.

        Returns:
            Dict with:
                - drift: Expected return (e.g., 0.001 = +0.1%)
                - confidence: Prediction confidence (0-1)
                - factors: Breakdown of 6 factors
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return {
                'drift': 0.0,
                'confidence': 0.0,
                'factors': {}
            }

        prices = np.array(list(self.price_history[symbol]))
        volumes = np.array(list(self.volume_history[symbol]))

        # Factor 1: Momentum (last 1h return)
        if len(prices) >= 60:  # 60 ticks ≈ 1 hour
            recent_return = (prices[-1] - prices[-60]) / prices[-60]
        else:
            recent_return = (prices[-1] - prices[0]) / prices[0]
        momentum_signal = np.clip(recent_return, -0.05, 0.05)  # Cap at ±5%

        # Factor 2: Volatility regime
        if len(prices) >= 20:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            vol_signal = np.clip(volatility, 0, 0.05)  # Cap at 5%
        else:
            vol_signal = 0.01

        # Factor 3: Volume trend
        if len(volumes) >= 10 and np.sum(volumes) > 0:
            recent_vol = np.mean(volumes[-5:])
            past_vol = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes)
            volume_trend = (recent_vol - past_vol) / max(past_vol, 1e-8)
            volume_signal = np.clip(volume_trend, -1, 1)
        else:
            volume_signal = 0

        # Factor 4: Time of day
        current_hour = datetime.now().hour
        # US market hours (UTC): 13:30-20:00 are high activity
        if 13 <= current_hour <= 20:
            time_signal = 0.5  # Active trading
        elif 0 <= current_hour <= 6:
            time_signal = -0.3  # Low liquidity
        else:
            time_signal = 0.0

        # Factor 5: Day of week
        current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
        if current_day == 0:  # Monday
            day_signal = 0.3  # Positive Monday effect in crypto
        elif current_day == 4:  # Friday
            day_signal = -0.2  # Weekend uncertainty
        else:
            day_signal = 0.0

        # Factor 6: Mean reversion
        mean_price = np.mean(prices)
        current_price = prices[-1]
        deviation = (current_price - mean_price) / mean_price
        reversion_signal = -deviation  # Negative = expect reversion

        # Combine factors using learned coefficients
        predicted_drift = (
            self.beta['intercept'] +
            self.beta['momentum'] * momentum_signal +
            self.beta['volatility'] * vol_signal +
            self.beta['volume'] * volume_signal +
            self.beta['time_of_day'] * time_signal +
            self.beta['day_of_week'] * day_signal +
            self.beta['mean_reversion'] * reversion_signal
        )

        # Calculate confidence based on signal strength
        signal_strength = abs(momentum_signal) + abs(volume_signal) + abs(deviation)
        confidence = np.clip(signal_strength / 0.1, 0.3, 0.95)  # 30-95% confidence

        factors = {
            'momentum': momentum_signal,
            'volatility': vol_signal,
            'volume': volume_signal,
            'time_of_day': time_signal,
            'day_of_week': day_signal,
            'mean_reversion': reversion_signal
        }

        return {
            'drift': predicted_drift,
            'confidence': confidence,
            'factors': factors
        }

    def confirm_signal_direction(self, symbol: str, signal_direction: str) -> bool:
        """
        Check if predicted drift aligns with signal direction.

        Args:
            symbol: Trading symbol
            signal_direction: 'long' or 'short'

        Returns:
            True if drift prediction agrees with signal
        """
        prediction = self.predict_drift(symbol)
        drift = prediction['drift']

        if signal_direction.lower() in ['long', 'buy']:
            return drift > 0.0001  # Positive drift
        elif signal_direction.lower() in ['short', 'sell']:
            return drift < -0.0001  # Negative drift
        else:
            return True  # Neutral signals don't need confirmation

    def get_drift_boost(self, symbol: str, signal_direction: str) -> float:
        """
        Get confidence boost if drift aligns with signal.

        Returns:
            Multiplier: 1.0 (no boost) to 1.2 (strong alignment)
        """
        prediction = self.predict_drift(symbol)
        drift = prediction['drift']
        confidence = prediction['confidence']

        # Check alignment
        if signal_direction.lower() in ['long', 'buy'] and drift > 0:
            # Bullish signal + bullish drift = boost
            return 1.0 + (confidence * 0.2)
        elif signal_direction.lower() in ['short', 'sell'] and drift < 0:
            # Bearish signal + bearish drift = boost
            return 1.0 + (confidence * 0.2)
        elif abs(drift) < 0.0001:
            # Neutral drift = no penalty
            return 1.0
        else:
            # Misaligned = slight penalty
            return 0.95

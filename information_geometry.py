"""
Information-Geometry and Fractional Volatility Utilities
========================================================

Provides helpers for computing Fisher-information-based signal strength,
entropy-weighted rewards, and fractional volatility memory adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

EPSILON = 1e-12


class InformationGeometryMetrics:
    """
    Computes Fisher information, entropy, and derived reward metrics for
    signal strength and adaptive position sizing.
    """

    def __init__(self,
                 lambda_entropy: float = 0.25,
                 base_position: float = 0.02,
                 max_position: float = 0.30,
                 scaling: float = 0.15):
        self.lambda_entropy = lambda_entropy
        self.base_position = base_position
        self.max_position = max_position
        self.scaling = scaling

    def compute(self, analysis: pd.DataFrame) -> pd.DataFrame:
        velocity = analysis.get('velocity', pd.Series(0.0, index=analysis.index))
        acceleration = analysis.get('acceleration', pd.Series(0.0, index=analysis.index))
        snr = analysis.get('snr', pd.Series(0.0, index=analysis.index))
        price = analysis.get('price', pd.Series(1.0, index=analysis.index))
        forecast = analysis.get('forecast', pd.Series(price, index=analysis.index))
        velocity_variance = analysis.get('velocity_variance', pd.Series(EPSILON, index=analysis.index))

        # Normalize magnitudes to build a probability distribution over features
        magnitudes = pd.DataFrame({
            'velocity': np.abs(velocity),
            'acceleration': np.abs(acceleration),
            'snr': np.abs(snr)
        }, index=analysis.index)
        mass = magnitudes.sum(axis=1).replace(0.0, EPSILON)
        probabilities = magnitudes.div(mass, axis=0).fillna(1.0 / 3.0)

        entropy = -(probabilities * np.log(probabilities + EPSILON)).sum(axis=1).fillna(0.0)

        expected_return = (forecast - price) / price.replace(0.0, EPSILON)
        variance_return = velocity_variance.replace(0.0, EPSILON)

        fisher_info = (velocity ** 2 + acceleration ** 2) / (variance_return + EPSILON)

        reward = expected_return / np.sqrt(variance_return + self.lambda_entropy * entropy + EPSILON)

        position_size = np.clip(
            self.base_position + self.scaling * np.tanh(reward),
            self.base_position,
            self.max_position
        )

        information_flow = fisher_info / (1.0 + fisher_info)

        return pd.DataFrame({
            'fisher_information': fisher_info,
            'information_entropy': entropy,
            'information_reward': reward,
            'information_position_size': position_size,
            'information_flow': information_flow
        }, index=analysis.index)


class FractionalVolatilityModel:
    """
    Tracks fractional volatility memory via Hurst exponent estimation and
    generates smoother volatility forecasts for improved stop placement.
    """

    def __init__(self,
                 hurst_windows: Iterable[int] = (20, 40, 80),
                 smoothing_window: int = 32,
                 memory_window: int = 50):
        self.hurst_windows = list(hurst_windows)
        self.smoothing_window = max(2, smoothing_window)
        self.memory_window = max(self.smoothing_window, memory_window)

    def _rescaled_range(self, series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) < 2:
            return np.nan
        centered = clean - clean.mean()
        cumulative = centered.cumsum()
        R = cumulative.max() - cumulative.min()
        S = clean.std(ddof=0)
        if S <= 0 or np.isnan(S):
            return np.nan
        return R / S

    def estimate_hurst(self, returns: pd.Series) -> float:
        log_pairs = []
        for window in self.hurst_windows:
            if len(returns) < window:
                continue
            subset = returns.iloc[-window:]
            rs = self._rescaled_range(subset)
            if not np.isnan(rs) and rs > 0:
                log_pairs.append((np.log(window), np.log(rs)))

        if len(log_pairs) < 2:
            return 0.5

        xs, ys = zip(*log_pairs)
        slope, _ = np.polyfit(xs, ys, 1)
        return float(np.clip(slope, 0.1, 0.9))

    def _rolling_clustering(self, volatility: pd.Series) -> pd.Series:
        clustering = []
        for idx in range(len(volatility)):
            start = max(0, idx - self.memory_window + 1)
            window = volatility.iloc[start:idx + 1].dropna()
            if len(window) < 2:
                clustering.append(0.0)
                continue
            corr = window.autocorr(lag=1)
            clustering.append(float(0.0 if np.isnan(corr) else corr))
        return pd.Series(clustering, index=volatility.index)

    def enrich(self, analysis: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        volatility = returns.rolling(self.smoothing_window, min_periods=1).std().bfill().fillna(0.0)
        clustering = self._rolling_clustering(volatility)
        hurst = self.estimate_hurst(returns)

        fbm_volatility = volatility * (1.0 + (hurst - 0.5))
        stop_multiplier = 1.0 + max(hurst - 0.5, 0.0) * 0.65

        return pd.DataFrame({
            'hurst_exponent': hurst,
            'fbm_volatility': fbm_volatility,
            'volatility_clustering': clustering,
            'fractional_stop_multiplier': stop_multiplier
        }, index=analysis.index)

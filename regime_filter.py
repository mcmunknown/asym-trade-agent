"""
Bayesian Particle Filter for Market Regime Detection
====================================================

This module provides a lightweight particle-based Bayesian updater to
track bull/bear/range regimes and expose probabilities/confidence for
regime-aware strategy adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

EPSILON = 1e-12


@dataclass
class RegimeStats:
    """Helper structure describing the current regime belief."""
    state: str
    probabilities: Dict[str, float]
    confidence: float


class BayesianRegimeFilter:
    """
    Simple particle filter for bull/bear/range regime tracking.

    Each particle represents a regime hypothesis and is weighted using the
    likelihood of observed returns under that regime assumption.
    """

    REGIMES = ['BULL', 'BEAR', 'RANGE']

    def __init__(self,
                 n_particles: int = 300,
                 resample_threshold: float = 0.6):
        self.n_particles = max(100, n_particles)
        self.resample_threshold = np.clip(resample_threshold, 0.1, 0.9)
        self._init_particles()
        self._last_price = None

        # Define regime-specific transition biases
        self.transition_matrix = {
            'BULL': {'BULL': 0.82, 'RANGE': 0.14, 'BEAR': 0.04},
            'BEAR': {'BEAR': 0.78, 'RANGE': 0.16, 'BULL': 0.06},
            'RANGE': {'RANGE': 0.70, 'BULL': 0.15, 'BEAR': 0.15}
        }

        # Assume soft target means/vols
        self._drifts = {'BULL': 0.0012, 'BEAR': -0.0012, 'RANGE': 0.0002}
        self._vols = {'BULL': 0.0020, 'BEAR': 0.0025, 'RANGE': 0.0035}

    def _init_particles(self):
        self.particles = [{
            'regime': np.random.choice(self.REGIMES),
            'weight': 1.0 / self.n_particles
        } for _ in range(self.n_particles)]

    def _transition_regime(self, current: str) -> str:
        probs = self.transition_matrix.get(current, {})
        choices = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / (weights.sum() + EPSILON)
        return str(np.random.choice(choices, p=weights))

    def _likelihood(self, return_pct: float, regime: str) -> float:
        mu = self._drifts.get(regime, 0.0)
        sigma = max(self._vols.get(regime, 0.002), EPSILON)
        exponent = -0.5 * ((return_pct - mu) / sigma) ** 2
        denom = sigma * np.sqrt(2 * np.pi)
        return float(np.exp(exponent) / max(denom, EPSILON))

    def _normalize_weights(self):
        total = sum(p['weight'] for p in self.particles)
        if total <= 0:
            uniform = 1.0 / self.n_particles
            for p in self.particles:
                p['weight'] = uniform
            return
        for p in self.particles:
            p['weight'] /= total

    def _effective_sample_size(self) -> float:
        weights = np.array([p['weight'] for p in self.particles])
        return 1.0 / np.sum(weights ** 2 + EPSILON)

    def _resample(self):
        weights = np.array([p['weight'] for p in self.particles])
        regimes = [p['regime'] for p in self.particles]
        chosen = np.random.choice(regimes, size=self.n_particles, replace=True, p=weights)
        self.particles = [{
            'regime': str(reg),
            'weight': 1.0 / self.n_particles
        } for reg in chosen]

    def update(self, price: float) -> RegimeStats:
        """
        Update the regime belief based on a new price observation.
        """
        if self._last_price is None or price <= 0:
            self._last_price = price
            probabilities = {reg: 1.0 / len(self.REGIMES) for reg in self.REGIMES}
            return RegimeStats(state='RANGE', probabilities=probabilities, confidence=0.33)

        return_pct = (price - self._last_price) / max(self._last_price, EPSILON)
        self._last_price = price

        for particle in self.particles:
            particle['regime'] = self._transition_regime(particle['regime'])
            lik = self._likelihood(return_pct, particle['regime'])
            particle['weight'] *= lik

        self._normalize_weights()

        if self._effective_sample_size() < self.resample_threshold * self.n_particles:
            self._resample()

        regime_probs = {reg: 0.0 for reg in self.REGIMES}
        for particle in self.particles:
            regime_probs[particle['regime']] += particle['weight']

        total_prob = sum(regime_probs.values())
        if total_prob <= 0:
            regime_probs = {reg: 1.0 / len(self.REGIMES) for reg in self.REGIMES}
        else:
            regime_probs = {reg: prob / total_prob for reg, prob in regime_probs.items()}

        dominant = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[dominant]

        return RegimeStats(state=dominant, probabilities=regime_probs, confidence=confidence)

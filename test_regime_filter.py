"""
Tests for the Bayesian regime filter used by the calculus trading strategy.
"""

import pytest

from regime_filter import BayesianRegimeFilter


def test_bayesian_regime_filter_updates():
    regime_filter = BayesianRegimeFilter(n_particles=200)
    stats_initial = regime_filter.update(100.0)
    assert stats_initial.state in regime_filter.REGIMES
    assert abs(sum(stats_initial.probabilities.values()) - 1.0) < 1e-6

    stats_next = regime_filter.update(101.0)
    assert stats_next.state in regime_filter.REGIMES
    assert 0.0 <= stats_next.confidence <= 1.0
    assert stats_next.probabilities.get(stats_next.state, 0.0) == pytest.approx(stats_next.confidence, rel=1e-3, abs=1e-3)

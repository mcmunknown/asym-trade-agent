import math

from utils.math_forecasting_utils import (
    ForecastInputs,
    compute_unified_log_drift,
    compute_barrier_hit_probability,
    optimize_tp_sl,
)
from config import Config


def test_barrier_probability_zero_drift_symmetry():
    prob = compute_barrier_hit_probability(0.0, 0.02, 0.02, 0.02)
    assert abs(prob - 0.5) < 1e-6


def test_barrier_probability_positive_drift_prefers_tp():
    prob = compute_barrier_hit_probability(0.01, 0.02, 0.02, 0.02)
    assert prob > 0.5


def test_unified_drift_matches_inputs():
    inputs = ForecastInputs(
        price=2000.0,
        velocity=0.5,
        acceleration=0.1,
        jerk=0.01,
        snap=0.0,
        sigma_pct=0.01,
        half_life_seconds=120.0,
        ou_mu=math.log(2000.0),
        ou_theta=0.1,
        ou_sigma=0.01
    )
    mu, sigma = compute_unified_log_drift(inputs, Config, 120.0)
    assert math.isfinite(mu) and math.isfinite(sigma)
    assert sigma > 0


def test_optimize_tp_sl_respects_fee_floor():
    tp, sl, meta = optimize_tp_sl(
        mu=0.0,
        sigma=0.02,
        base_tp_pct=0.01,
        base_sl_pct=0.01,
        fee_floor_pct=0.005,
        leverage=20.0,
        config=Config
    )
    assert tp >= 0.005
    assert sl > 0
    assert 'tp_prob' in meta and 0 <= meta['tp_prob'] <= 1

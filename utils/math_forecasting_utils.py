"""Mathematical forecasting helpers for unified drift/sigma estimation and TP optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict


@dataclass
class ForecastInputs:
    price: float
    velocity: float
    acceleration: float
    jerk: float
    snap: float
    sigma_pct: float
    half_life_seconds: Optional[float] = None
    ou_mu: Optional[float] = None
    ou_theta: Optional[float] = None
    ou_sigma: Optional[float] = None


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                return float(value)
    except Exception:
        pass
    return default


def compute_derivative_trend_score(velocity: float,
                                   acceleration: float,
                                   jerk: float,
                                   sigma_pct: float,
                                   config: object) -> float:
    """Return a dimensionless trend intensity score based on derivatives."""
    sigma = max(abs(sigma_pct), 1e-6)
    v_weight = float(getattr(config, "DERIVATIVE_TREND_V_WEIGHT", 1.0))
    a_weight = float(getattr(config, "DERIVATIVE_TREND_A_WEIGHT", 0.6))
    j_weight = float(getattr(config, "DERIVATIVE_TREND_J_WEIGHT", 0.2))
    jerk_norm = max(float(getattr(config, "DERIVATIVE_TREND_J_NORM", 0.01)), 1e-6)

    score = (
        v_weight * abs(velocity) / (sigma + 1e-12) +
        a_weight * abs(acceleration) / (sigma + 1e-12) +
        j_weight * abs(jerk) / jerk_norm
    )
    return float(score)


def compute_derivative_horizon(half_life_seconds: Optional[float],
                               trend_score: float,
                               config: object) -> Optional[float]:
    """Adjust OU half-life derived horizon using derivative trend score."""
    if half_life_seconds is None or not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        return None

    base_multiplier = 2.5 if getattr(config, "EMERGENCY_CALCULUS_MODE", False) else 2.0
    base_horizon = half_life_seconds * base_multiplier

    k_gain = float(getattr(config, "DERIVATIVE_HORIZON_K_GAIN", 0.35))
    trend_mid = float(getattr(config, "DERIVATIVE_TREND_BASE", 1.0))
    horizon_min = float(getattr(config, "DERIVATIVE_HORIZON_MIN", 60.0))
    horizon_max = float(getattr(config, "DERIVATIVE_HORIZON_MAX", 1200.0))

    adjustment = 1.0 + k_gain * (trend_score - trend_mid)
    adjusted = base_horizon * max(min(adjustment, 2.5), 0.4)
    adjusted = max(min(adjusted, horizon_max), horizon_min)
    return adjusted


def blend_drift_components(taylor_drift: float,
                           ou_drift: float,
                           trend_confidence: float,
                           config: object) -> float:
    """Blend Taylor (local) and OU (global) drift into a single forecast drift."""
    min_weight = float(getattr(config, "DERIVATIVE_DRIFT_MIN_WEIGHT", 0.35))
    max_weight = float(getattr(config, "DERIVATIVE_DRIFT_MAX_WEIGHT", 0.85))
    weight = min(max(trend_confidence, 0.0), 1.0)
    weight = min_weight + (max_weight - min_weight) * weight
    return weight * taylor_drift + (1.0 - weight) * ou_drift


def compute_unified_log_drift(inputs: ForecastInputs,
                              config: object,
                              horizon_seconds: float) -> Tuple[float, float]:
    """Return (mu_H, sigma_H) for log-price over horizon H seconds."""
    price = max(inputs.price, 1e-8)
    v = _safe_float(inputs.velocity)
    a = _safe_float(inputs.acceleration)
    j = _safe_float(inputs.jerk)
    s = _safe_float(inputs.snap)
    sigma_pct = max(_safe_float(inputs.sigma_pct, 5e-4), 5e-4)

    delta = max(horizon_seconds, 1.0)
    taylor_delta = (v * delta / price) + 0.5 * (a / price) * (delta ** 2)
    taylor_delta += (j / price) * (delta ** 3) / 6.0
    taylor_delta += (s / price) * (delta ** 4) / 24.0
    taylor_drift = taylor_delta / delta

    ou_mu = _safe_float(inputs.ou_mu, None)
    ou_theta = _safe_float(inputs.ou_theta, None)
    ou_sigma = _safe_float(inputs.ou_sigma, sigma_pct)
    if ou_mu is None:
        ou_mu = math.log(price)
    if ou_theta is None or ou_theta <= 0:
        ou_theta = float(getattr(config, "DEFAULT_OU_THETA", 0.05))

    log_price = math.log(price)
    ou_expected = ou_mu + (log_price - ou_mu) * math.exp(-ou_theta * delta)
    ou_drift = (ou_expected - log_price) / delta

    trend_score = compute_derivative_trend_score(v / price, a / price, j / price,
                                                 sigma_pct, config)
    confidence = min(trend_score / float(getattr(config, "DERIVATIVE_TREND_CONF_DIV", 3.0)), 1.0)
    mu_H = blend_drift_components(taylor_drift, ou_drift, confidence, config)

    sigma_base = max(ou_sigma, sigma_pct)
    sigma_H = sigma_base * math.sqrt(delta)
    return mu_H, sigma_H


def _log_barrier(tp_pct: float, sl_pct: float) -> Tuple[float, float]:
    tp = max(tp_pct, 1e-6)
    sl = max(sl_pct, 1e-6)
    return math.log1p(tp), -math.log1p(sl)


def compute_barrier_hit_probability(mu: float,
                                    sigma: float,
                                    tp_pct: float,
                                    sl_pct: float) -> float:
    """Probability of hitting TP before SL for drifted Brownian motion."""
    if sigma <= 0:
        return 0.5 if mu == 0 else (1.0 if mu > 0 else 0.0)

    upper, lower = _log_barrier(tp_pct, sl_pct)
    width = upper - lower
    if width <= 0:
        return 0.5

    mu_term = (2.0 * mu) / (sigma ** 2)
    try:
        numerator = 1.0 - math.exp(-mu_term * (-lower))
        denominator = 1.0 - math.exp(-mu_term * width)
    except OverflowError:
        numerator = 1.0 if mu_term * (-lower) > 0 else 0.0
        denominator = 1.0 if mu_term * width > 0 else 0.0

    if abs(mu_term) < 1e-6:
        return max(min(-lower / width, 1.0), 0.0)

    if denominator == 0:
        return 1.0 if numerator > 0 else 0.0

    return max(min(numerator / denominator, 1.0), 0.0)


def evaluate_tp_sl_candidate(mu: float,
                             sigma: float,
                             tp_pct: float,
                             sl_pct: float,
                             leverage: float,
                             fee_cost_pct: float) -> Dict[str, float]:
    tp_prob = compute_barrier_hit_probability(mu, sigma, tp_pct, sl_pct)
    tp_gain = tp_pct * leverage
    sl_loss = sl_pct * leverage
    expected_value = tp_prob * tp_gain - (1.0 - tp_prob) * sl_loss - fee_cost_pct
    return {
        'tp_prob': tp_prob,
        'ev': expected_value,
        'tp_gain': tp_gain,
        'sl_loss': sl_loss,
    }


def optimize_tp_sl(mu: float,
                   sigma: float,
                   base_tp_pct: float,
                   base_sl_pct: float,
                   fee_floor_pct: float,
                   leverage: float,
                   config: object) -> Tuple[float, float, Dict[str, float]]:
    """Grid search refinement for TP/SL maximizing EV under constraints."""
    grid_steps = int(getattr(config, "BARRIER_OPT_GRID_STEPS", 5))
    tp_range = float(getattr(config, "BARRIER_OPT_TP_RANGE", 0.35))
    sl_range = float(getattr(config, "BARRIER_OPT_SL_RANGE", 0.3))
    min_rr = float(getattr(config, "BARRIER_OPT_MIN_RR", 0.6))
    max_rr = float(getattr(config, "BARRIER_OPT_MAX_RR", 3.0))
    min_tp_prob = float(getattr(config, "BARRIER_OPT_MIN_TP_PROB", 0.35))

    best_tp = base_tp_pct
    best_sl = base_sl_pct
    best_meta = {'tp_prob': 0.0, 'ev': -math.inf}

    tp_candidates = [max(base_tp_pct * (1 + tp_range * (i - grid_steps // 2) / max(grid_steps // 2, 1)), fee_floor_pct)
                     for i in range(grid_steps)]
    sl_candidates = [max(base_sl_pct * (1 + sl_range * (j - grid_steps // 2) / max(grid_steps // 2, 1)), fee_floor_pct / 2)
                     for j in range(grid_steps)]

    for tp_pct in tp_candidates:
        for sl_pct in sl_candidates:
            if sl_pct <= 0:
                continue
            rr = tp_pct / sl_pct
            if rr < min_rr or rr > max_rr:
                continue
            meta = evaluate_tp_sl_candidate(mu, sigma, tp_pct, sl_pct, leverage, fee_floor_pct)
            if meta['tp_prob'] < min_tp_prob:
                continue
            if meta['ev'] > best_meta['ev']:
                best_tp, best_sl, best_meta = tp_pct, sl_pct, meta

    return best_tp, best_sl, best_meta


__all__ = [
    'ForecastInputs',
    'compute_derivative_trend_score',
    'compute_derivative_horizon',
    'compute_unified_log_drift',
    'compute_barrier_hit_probability',
    'optimize_tp_sl'
]

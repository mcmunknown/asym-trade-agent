import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from live_calculus_trader import LiveCalculusTrader
from risk_manager import RiskManager
from calculus_strategy import SignalType


def _legacy_ev(tp_pct: float, sl_pct: float, tp_prob: float, fee_floor_pct: float) -> float:
    """Previous EV logic (full fee impact, conservative probability clip)."""
    tp_pct = max(tp_pct - fee_floor_pct, 0.0)
    sl_pct = sl_pct + fee_floor_pct
    tp_prob = float(np.clip(tp_prob, 0.05, 0.95))
    return tp_prob * tp_pct - (1.0 - tp_prob) * sl_pct


def test_crypto_ev_adjustment_improves_expected_value():
    tp_pct = 0.0120  # 1.20%
    sl_pct = 0.0038  # 0.38%
    tp_prob = 0.35
    fee_floor_pct = 0.0040  # 0.40%

    old_ev = _legacy_ev(tp_pct, sl_pct, tp_prob, fee_floor_pct)
    trader = object.__new__(LiveCalculusTrader)
    trader.ev_debug_enabled = False
    trader._ev_debug_records = {}
    trader._tp_probability_debug = {}
    new_ev = trader._compute_trade_ev("TEST", tp_pct, sl_pct, tp_prob, fee_floor_pct)

    assert old_ev < 0  # Previously blocked trades
    assert new_ev > 0  # Crypto adjustments flip EV positive
    assert new_ev > old_ev


class _DummyRiskManager:
    def __init__(self, posterior):
        self.posterior = posterior

    def get_symbol_probability_posterior(self, symbol):
        return self.posterior


def test_tp_probability_crypto_boosts_produce_aggressive_estimate():
    trader = object.__new__(LiveCalculusTrader)
    dummy_posterior = {"mean": 0.50, "count": 3}
    trader.risk_manager = _DummyRiskManager(dummy_posterior)
    trader.ev_debug_enabled = False
    trader._tp_probability_debug = {}
    trader._ev_debug_records = {}

    signal_payload = {
        "confidence": 0.85,
        "snr": 6.0,
        "velocity": 0.012,
    }
    tier_config = {
        "confidence_threshold": 0.60,
        "snr_threshold": 2.0,
        "min_probability_samples": 6,
    }

    tp_prob, posterior = trader._estimate_tp_probability("BTCUSDT", signal_payload, tier_config)

    assert posterior is dummy_posterior
    assert 0.55 < tp_prob < 0.70  # Boosted yet still within crypto clip range
    debug_payload = trader.get_last_probability_debug("BTCUSDT")
    assert debug_payload is not None
    assert pytest.approx(debug_payload["final_probability"], rel=1e-9) == tp_prob
    assert debug_payload["velocity_boost"] >= 0.0


def test_tp_probability_high_confidence_floor_applied():
    trader = object.__new__(LiveCalculusTrader)
    dummy_posterior = {"mean": 0.20, "count": 10}
    trader.risk_manager = _DummyRiskManager(dummy_posterior)
    trader.ev_debug_enabled = False
    trader._tp_probability_debug = {}
    trader._ev_debug_records = {}

    signal_payload = {
        "confidence": 0.85,
        "snr": 0.0,
        "velocity": 0.0,
    }
    tier_config = {
        "confidence_threshold": 0.80,
        "snr_threshold": 1.5,
        "min_probability_samples": 5,
    }

    tp_prob, _ = trader._estimate_tp_probability("ADAUSDT", signal_payload, tier_config)
    assert tp_prob >= 0.42
    debug_payload = trader.get_last_probability_debug("ADAUSDT")
    assert debug_payload["high_confidence_floor"] is True


def test_crypto_sl_offset_uses_half_sigma_with_floor():
    risk_manager = RiskManager()
    current_price = 100.0

    levels = risk_manager.calculate_dynamic_tp_sl(
        signal_type=SignalType.BUY,
        current_price=current_price,
        velocity=0.02,
        acceleration=0.0,
        volatility=0.01,
    )

    sl_offset = current_price - levels.stop_loss
    expected_sl_offset = current_price * 0.01 * 0.5
    assert sl_offset == pytest.approx(expected_sl_offset, rel=1e-6)

    low_vol_levels = risk_manager.calculate_dynamic_tp_sl(
        signal_type=SignalType.BUY,
        current_price=current_price,
        velocity=0.0,
        acceleration=0.0,
        volatility=0.001,
    )

    low_vol_sl_offset = current_price - low_vol_levels.stop_loss
    assert low_vol_sl_offset == pytest.approx(current_price * 0.004, rel=1e-6)
    rr = (low_vol_levels.take_profit - current_price) / low_vol_sl_offset
    assert rr == pytest.approx(1.5, rel=1e-6)  # Maintains healthy R:R after floor application


def test_microstructure_costs_capped_for_liquid_symbols():
    risk_manager = RiskManager()
    stats = risk_manager._get_symbol_stats("BTCUSDT")
    stats["ewma_spread"] = 0.004
    stats["ewma_slippage"] = 0.003

    cost = risk_manager.estimate_microstructure_cost("BTCUSDT")
    assert cost == pytest.approx(0.001, rel=1e-9)  # 0.1% cap for liquid symbols
    debug = risk_manager.get_microstructure_debug("BTCUSDT")
    assert debug["final_micro_pct"] == pytest.approx(cost, rel=1e-9)
    assert debug["cap_reason"] == "liquid_cap_0.10%"


def test_microstructure_costs_limited_by_crypto_cap():
    risk_manager = RiskManager()
    stats = risk_manager._get_symbol_stats("ADAUSDT")
    stats["ewma_spread"] = 0.003
    stats["ewma_slippage"] = 0.002

    cost = risk_manager.estimate_microstructure_cost("ADAUSDT")
    assert cost == pytest.approx(0.002, rel=1e-9)  # Global 0.2% cap
    debug = risk_manager.get_microstructure_debug("ADAUSDT")
    assert debug["final_micro_pct"] == pytest.approx(cost, rel=1e-9)
    assert debug["cap_reason"] == "global_cap_0.20%"


def test_microstructure_fallback_seeds_when_stale():
    risk_manager = RiskManager()
    stats = risk_manager._get_symbol_stats("SOLUSDT")
    stats["ewma_spread"] = 0.0
    stats["ewma_slippage"] = 0.0
    stats["last_micro_update"] = 0.0

    cost = risk_manager.estimate_microstructure_cost("SOLUSDT")
    assert cost > 0.0
    debug = risk_manager.get_microstructure_debug("SOLUSDT")
    assert debug["final_micro_pct"] == pytest.approx(cost, rel=1e-9)
    assert debug["cap_reason"] in {None, "global_cap_0.20%", "liquid_cap_0.10%"}


def test_fee_floor_debug_tracks_components():
    risk_manager = RiskManager()
    floor = risk_manager.get_fee_aware_tp_floor(
        sigma_pct=0.01,
        taker_fee_pct=0.0006,
        funding_buffer_pct=0.0001,
        symbol="ADAUSDT"
    )
    debug = risk_manager.get_fee_floor_debug("ADAUSDT")
    assert debug["total_fee_floor_pct"] == pytest.approx(floor, rel=1e-9)
    assert debug["applied_multiplier"] <= debug["config_multiplier"]
    assert debug["effective_fee_pct"] <= debug["base_fee_pct"]


def test_candidate_pool_respects_relaxed_micro_limits():
    risk_manager = RiskManager()
    stats = risk_manager._get_symbol_stats("SUIUSDT")
    stats["ewma_spread"] = 0.00075  # Between old (0.0006) and new (0.0008) caps
    stats["ewma_slippage"] = 0.00095
    stats["spread_history"].extend([0.0007] * 6)
    stats["slippage_history"].extend([0.0009] * 6)
    stats["ev_history"].extend([0.002] * 8)

    allowed = risk_manager.is_symbol_allowed_for_tier("SUIUSDT", "micro", tier_min_ev_pct=0.001)
    assert allowed  # Should pass under relaxed crypto thresholds


def test_candidate_pool_blocks_when_spread_exceeds_crypto_cap():
    risk_manager = RiskManager()
    stats = risk_manager._get_symbol_stats("NEARUSDT")
    stats["ewma_spread"] = 0.00085  # Above new 0.0008 cap
    stats["ewma_slippage"] = 0.00095
    stats["spread_history"].extend([0.00085] * 6)
    stats["slippage_history"].extend([0.0009] * 6)
    stats["ev_history"].extend([0.002] * 8)

    allowed = risk_manager.is_symbol_allowed_for_tier("NEARUSDT", "micro", tier_min_ev_pct=0.001)
    assert not allowed


def test_ev_debug_record_captures_breakdown():
    trader = object.__new__(LiveCalculusTrader)
    trader.ev_debug_enabled = False
    trader._ev_debug_records = {}
    trader._tp_probability_debug = {}
    debug_context = {
        "fee_floor_pct": 0.003,
        "micro_cost_pct": 0.001,
        "execution_cost_floor_pct": 0.004,
        "base_tp_probability": 0.35,
        "time_constrained_probability": 0.33,
        "ou_weight": 0.4,
        "ou_probability": 0.36
    }
    ev = trader._compute_trade_ev("DEBUG", 0.012, 0.004, 0.34, 0.004, debug_context)
    breakdown = trader.get_last_ev_debug("DEBUG")
    assert breakdown is not None
    assert pytest.approx(breakdown["final_ev_pct"], rel=1e-9) == ev
    assert breakdown["fee_adjustment_pct"] == pytest.approx(0.004 * 0.6, rel=1e-9)
    assert breakdown["ou_weight"] == pytest.approx(0.4, rel=1e-9)

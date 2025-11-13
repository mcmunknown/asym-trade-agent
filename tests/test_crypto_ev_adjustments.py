import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from live_calculus_trader import LiveCalculusTrader
from risk_manager import RiskManager
from calculus_strategy import SignalType
from config import Config


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
    trader.calculus_priority_mode = False
    new_ev = trader._compute_trade_ev("TEST", tp_pct, sl_pct, tp_prob, fee_floor_pct)

    assert old_ev < 0  # Previously blocked trades
    assert new_ev > 0  # Crypto adjustments flip EV positive
    assert new_ev > old_ev


class _DummyRiskManager:
    def __init__(self, posterior):
        self.posterior = posterior
        self.snapshot = None

    def get_symbol_probability_posterior(self, symbol):
        return self.posterior

    def track_probability_snapshot(self, symbol, payload):
        self.snapshot = payload


def test_tp_probability_crypto_boosts_produce_aggressive_estimate():
    trader = object.__new__(LiveCalculusTrader)
    dummy_posterior = {"mean": 0.50, "count": 3}
    trader.risk_manager = _DummyRiskManager(dummy_posterior)
    trader.ev_debug_enabled = False
    trader._tp_probability_debug = {}
    trader._ev_debug_records = {}
    trader.calculus_priority_mode = False

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
    assert 0.55 < tp_prob < 0.60  # Boosted yet still within crypto clip range
    debug_payload = trader.get_last_probability_debug("BTCUSDT")
    assert debug_payload is not None
    assert pytest.approx(debug_payload["final_probability"], rel=1e-9) == tp_prob
    assert debug_payload["velocity_boost"] >= 0.0
    assert trader.risk_manager.snapshot is not None
    assert pytest.approx(trader.risk_manager.snapshot["final_probability"], rel=1e-9) == tp_prob


def test_tp_probability_high_confidence_floor_applied():
    trader = object.__new__(LiveCalculusTrader)
    dummy_posterior = {"mean": 0.20, "count": 10}
    trader.risk_manager = _DummyRiskManager(dummy_posterior)
    trader.ev_debug_enabled = False
    trader._tp_probability_debug = {}
    trader._ev_debug_records = {}
    trader.calculus_priority_mode = False

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
    assert tp_prob >= 0.45
    debug_payload = trader.get_last_probability_debug("ADAUSDT")
    assert debug_payload["high_confidence_floor"] is True
    assert trader.risk_manager.snapshot is not None
    assert pytest.approx(trader.risk_manager.snapshot["final_probability"], rel=1e-9) == tp_prob


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
    expected_tp_pct = max(1.5 * 0.01, 0.006, Config.COMMISSION_RATE * 4.0)
    expected_sl_pct = max(expected_tp_pct / 1.8, max(0.01 * 0.35, 0.0035))
    expected_sl_offset = current_price * expected_sl_pct
    assert sl_offset == pytest.approx(expected_sl_offset, rel=1e-6)

    low_vol_levels = risk_manager.calculate_dynamic_tp_sl(
        signal_type=SignalType.BUY,
        current_price=current_price,
        velocity=0.0,
        acceleration=0.0,
        volatility=0.001,
    )

    low_vol_sl_offset = current_price - low_vol_levels.stop_loss
    expected_low_tp_pct = max(1.5 * 0.001, 0.006, Config.COMMISSION_RATE * 4.0)
    expected_low_sl_pct = max(expected_low_tp_pct / 1.8, max(0.001 * 0.35, 0.0035))
    assert low_vol_sl_offset == pytest.approx(current_price * expected_low_sl_pct, rel=1e-6)
    rr = (low_vol_levels.take_profit - current_price) / low_vol_sl_offset
    assert rr == pytest.approx(expected_low_tp_pct / expected_low_sl_pct, rel=1e-6)


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


def test_probability_snapshot_records_history():
    risk_manager = RiskManager()
    risk_manager.track_probability_snapshot("BNBUSDT", {"final_probability": 0.56, "confidence": 0.8})
    stats = risk_manager._get_symbol_stats("BNBUSDT")
    assert stats["probability_history"]
    assert pytest.approx(stats["probability_history"][-1], rel=1e-9) == 0.56
    debug = risk_manager.get_probability_debug("BNBUSDT")
    assert debug["final_probability"] == 0.56


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
    original_mode = Config.CALCULUS_PRIORITY_MODE
    try:
        Config.CALCULUS_PRIORITY_MODE = False
        risk_manager = RiskManager()
        stats = risk_manager._get_symbol_stats("NEARUSDT")
        stats["ewma_spread"] = 0.00085  # Above new 0.0008 cap
        stats["ewma_slippage"] = 0.00095
        stats["spread_history"].extend([0.00085] * 6)
        stats["slippage_history"].extend([0.0009] * 6)
        stats["ev_history"].extend([0.002] * 8)

        allowed = risk_manager.is_symbol_allowed_for_tier("NEARUSDT", "micro", tier_min_ev_pct=0.001)
        assert not allowed
    finally:
        Config.CALCULUS_PRIORITY_MODE = original_mode


def test_ev_debug_record_captures_breakdown():
    trader = object.__new__(LiveCalculusTrader)
    trader.ev_debug_enabled = False
    trader._ev_debug_records = {}
    trader._tp_probability_debug = {}
    trader.calculus_priority_mode = False
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


def test_calculus_priority_probability_uses_confidence():
    original_mode = Config.CALCULUS_PRIORITY_MODE
    try:
        Config.CALCULUS_PRIORITY_MODE = True
        trader = object.__new__(LiveCalculusTrader)
        trader.calculus_priority_mode = True
        trader.ev_debug_enabled = False
        trader._tp_probability_debug = {}
        trader._ev_debug_records = {}
        dummy_posterior = {"mean": 0.55, "count": 4}
        trader.risk_manager = _DummyRiskManager(dummy_posterior)
        trader._should_log_ev_debug = lambda: False

        signal_payload = {"confidence": 0.80, "snr": 1.2}
        tier_config = {"confidence_threshold": 0.60}

        prob, posterior = trader._estimate_tp_probability("BTCUSDT", signal_payload, tier_config)

        assert posterior is dummy_posterior
        assert prob == pytest.approx(0.60, rel=1e-9)
        debug_payload = trader.get_last_probability_debug("BTCUSDT")
        assert debug_payload is not None
        assert debug_payload["mode"] == "calculus_priority"
        assert pytest.approx(debug_payload["final_probability"], rel=1e-9) == prob
    finally:
        Config.CALCULUS_PRIORITY_MODE = original_mode


def test_calculus_tp_sl_tracks_forecast_delta():
    original_force = Config.FORCE_LEVERAGE_ENABLED
    try:
        Config.FORCE_LEVERAGE_ENABLED = False
        risk_manager = RiskManager()
        current_price = 100.0
        forecast_price = 102.0

        levels = risk_manager.calculate_dynamic_tp_sl(
            signal_type=SignalType.BUY,
            current_price=current_price,
            velocity=0.01,
            acceleration=0.0,
            volatility=0.005,
            forecast_price=forecast_price,
            sigma=0.005
        )

        assert levels.take_profit >= forecast_price
        assert levels.risk_reward_ratio > 1.0
    finally:
        Config.FORCE_LEVERAGE_ENABLED = original_force


def test_force_leverage_overrides_kelly():
    original_force_enabled = Config.FORCE_LEVERAGE_ENABLED
    original_force_value = Config.FORCE_LEVERAGE_VALUE
    original_force_fraction = Config.FORCE_MARGIN_FRACTION
    try:
        Config.FORCE_LEVERAGE_ENABLED = True
        Config.FORCE_LEVERAGE_VALUE = 50.0
        Config.FORCE_MARGIN_FRACTION = 0.4
        risk_manager = RiskManager()
        account_balance = 20.0
        current_price = 100.0

        position_size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            signal_strength=1.0,
            confidence=0.8,
            current_price=current_price,
            account_balance=account_balance
        )

        assert position_size.leverage_used == pytest.approx(50.0, rel=1e-9)
        comp_band = risk_manager._resolve_compounding_band(account_balance)
        margin_fraction = Config.FORCE_MARGIN_FRACTION
        if comp_band and comp_band.get('margin_fraction') is not None:
            margin_fraction = comp_band['margin_fraction']
        expected_notional = account_balance * margin_fraction * position_size.leverage_used
        assert position_size.notional_value == pytest.approx(expected_notional, rel=1e-6)
    finally:
        Config.FORCE_LEVERAGE_ENABLED = original_force_enabled
        Config.FORCE_LEVERAGE_VALUE = original_force_value
        Config.FORCE_MARGIN_FRACTION = original_force_fraction


def test_dynamic_tp_scales_with_f_score_and_secondary_target():
    risk_manager = RiskManager()
    current_price = 100.0

    baseline = risk_manager.calculate_dynamic_tp_sl(
        signal_type=SignalType.BUY,
        current_price=current_price,
        velocity=0.01,
        acceleration=0.0,
        volatility=0.005,
        sigma=0.005,
        f_score=0.009
    )

    strong = risk_manager.calculate_dynamic_tp_sl(
        signal_type=SignalType.BUY,
        current_price=current_price,
        velocity=0.01,
        acceleration=0.0,
        volatility=0.005,
        sigma=0.005,
        f_score=0.02
    )

    baseline_tp_pct = (baseline.take_profit - current_price) / current_price
    strong_tp_pct = (strong.take_profit - current_price) / current_price

    assert strong_tp_pct > baseline_tp_pct
    assert strong.secondary_take_profit is not None
    assert strong.secondary_take_profit == pytest.approx(
        strong.take_profit + (strong.take_profit - current_price) * (Config.TP_SECONDARY_MULTIPLIER - 1.0), rel=1e-6
    )
    assert strong.secondary_tp_fraction == pytest.approx(1.0 - Config.TP_PRIMARY_FRACTION, rel=1e-9)


def _stub_trader_for_governor() -> LiveCalculusTrader:
    trader = object.__new__(LiveCalculusTrader)
    trader.curvature_edge_threshold = Config.CURVATURE_EDGE_THRESHOLD
    trader.curvature_edge_min = Config.CURVATURE_EDGE_MIN
    trader.curvature_edge_max = Config.CURVATURE_EDGE_MAX
    trader.base_primary_probability = Config.TP_PRIMARY_PROB_BASE
    trader.primary_prob_min = Config.TP_PRIMARY_PROB_MIN
    trader.primary_prob_max = Config.TP_PRIMARY_PROB_MAX
    trader.secondary_prob_min = Config.TP_SECONDARY_PROB_MIN
    trader.governor_block_relax = Config.GOVERNOR_BLOCK_RELAX
    trader.governor_time_relax = Config.GOVERNOR_TIME_RELAX_SEC
    trader.governor_fee_hard_cap = Config.GOVERNOR_FEE_PRESSURE_HARD
    trader.scout_entry_scale = Config.SCOUT_ENTRY_SCALE
    trader.governor_stats = {'blocks_since_trade': 0, 'total_blocks': 0, 'last_trade_time': None}
    trader.last_available_balance = 0.0
    trader.risk_manager = RiskManager()
    trader.risk_manager.current_portfolio_value = 50.0
    trader.risk_manager.curvature_failures.clear()
    trader.risk_manager.curvature_success.clear()
    return trader


def test_governor_relaxes_threshold_when_scarcity_high():
    trader = _stub_trader_for_governor()
    trader.governor_stats = {
        'blocks_since_trade': trader.governor_block_relax * 2,
        'total_blocks': trader.governor_block_relax * 2,
        'last_trade_time': time.time() - (trader.governor_time_relax + 600)
    }
    thresholds = trader._compute_governor_thresholds("BTCUSDT", 7.0)
    assert thresholds['f_score'] < trader.curvature_edge_threshold
    assert thresholds['scout_scale'] < 1.0


def test_governor_tightens_with_fee_pressure():
    trader = _stub_trader_for_governor()
    trader.risk_manager.fee_recovery_balance = 20.0
    trader.risk_manager.current_portfolio_value = 10.0
    trader.governor_stats = {
        'blocks_since_trade': 0,
        'total_blocks': 0,
        'last_trade_time': time.time()
    }
    thresholds = trader._compute_governor_thresholds("BTCUSDT", 10.0)
    assert thresholds['f_score'] >= trader.curvature_edge_threshold
    assert thresholds['primary_prob'] >= trader.base_primary_probability


def test_compounding_ladder_scout_scaling():
    original_force = Config.FORCE_LEVERAGE_ENABLED
    original_ladder = Config.COMPOUNDING_LADDER
    try:
        Config.FORCE_LEVERAGE_ENABLED = False
        Config.COMPOUNDING_LADDER = "0:auto:auto:0.30"
        risk_manager = RiskManager()
        normal = risk_manager.calculate_position_size(
            symbol="TESTUSDT",
            signal_strength=1.0,
            confidence=0.8,
            current_price=100.0,
            account_balance=200.0,
            scout_scale=1.0
        )
        scout = risk_manager.calculate_position_size(
            symbol="TESTUSDT",
            signal_strength=1.0,
            confidence=0.8,
            current_price=100.0,
            account_balance=200.0,
            scout_scale=0.5
        )
        assert scout.notional_value < normal.notional_value
        assert scout.margin_required < normal.margin_required
    finally:
        Config.FORCE_LEVERAGE_ENABLED = original_force
        Config.COMPOUNDING_LADDER = original_ladder

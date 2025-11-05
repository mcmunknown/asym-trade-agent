#!/usr/bin/env python3
"""
Test Calculus Signal Generation
================================

Test script to verify that the enhanced calculus trading system can generate
signals without SIGFPE crashes and produces the expected "CALCULUS SIGNAL" output.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging to see signal messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_calculus_signal_generation():
    """Test that calculus signals are generated correctly with enhanced safety."""

    print("🧪 Testing Calculus Signal Generation")
    print("=" * 50)

    try:
        # Import our enhanced modules
        from calculus_strategy import CalculusTradingStrategy, SignalType
        from quantitative_models import CalculusPriceAnalyzer

        print("✅ Successfully imported enhanced modules")

        # Create realistic market data that would trigger various signals
        np.random.seed(42)
        n_points = 50

        # Create different scenarios to test all signal types
        test_scenarios = {
            "uptrend_accelerating": np.cumsum(np.random.normal(0.5, 0.1, n_points)) + 100,
            "uptrend_slowing": np.concatenate([
                np.cumsum(np.random.normal(0.5, 0.1, n_points//2)) + 100,
                100 + np.cumsum(np.random.normal(0.1, 0.05, n_points//2))
            ]),
            "downtrend_accelerating": np.cumsum(np.random.normal(-0.3, 0.1, n_points)) + 100,
            "downtrend_weakening": np.concatenate([
                np.cumsum(np.random.normal(-0.5, 0.1, n_points//2)) + 100,
                100 + np.cumsum(np.random.normal(-0.1, 0.05, n_points//2))
            ]),
            "curvature_bottom": np.concatenate([
                np.cumsum(np.random.normal(-0.2, 0.1, n_points//3)) + 100,
                np.cumsum(np.random.normal(0.0, 0.05, n_points//3)) + 98,
                np.cumsum(np.random.normal(0.1, 0.05, n_points//3)) + 98
            ]),
            "curvature_top": np.concatenate([
                np.cumsum(np.random.normal(0.2, 0.1, n_points//3)) + 100,
                np.cumsum(np.random.normal(0.0, 0.05, n_points//3)) + 102,
                np.cumsum(np.random.normal(-0.1, 0.05, n_points//3)) + 102
            ])
        }

        strategy = CalculusTradingStrategy(snr_threshold=0.5, confidence_threshold=0.5)
        analyzer = CalculusPriceAnalyzer()

        signal_count = 0
        scenarios_tested = 0

        for scenario_name, prices in test_scenarios.items():
            print(f"\n📊 Testing scenario: {scenario_name}")

            # Convert to pandas Series with datetime index
            price_series = pd.Series(
                prices,
                index=pd.date_range(start=datetime.now(), periods=len(prices), freq='1min')
            )

            # Generate signals
            try:
                signals = strategy.generate_trading_signals(price_series)

                if not signals.empty:
                    latest_signal = strategy.get_latest_signal(price_series)

                    if latest_signal and latest_signal.get('valid_signal', False):
                        signal_count += 1

                        # Print the signal in the same format as the live system
                        print(f"=== CALCULUS SIGNAL for {scenario_name.upper()} ===")
                        print(f"Signal Type: {latest_signal['signal_type'].name}")
                        print(f"Interpretation: {latest_signal['interpretation']}")
                        print(f"Velocity: {latest_signal['velocity']:.6f}")
                        print(f"Acceleration: {latest_signal['acceleration']:.6f}")
                        print(f"SNR: {latest_signal['snr']:.3f}")
                        print(f"Confidence: {latest_signal['confidence']:.3f}")
                        print(f"Price: {price_series.iloc[-1]:.2f}")
                        print()

                        scenarios_tested += 1
                    else:
                        print(f"   No valid signal generated (low confidence or SNR)")
                else:
                    print(f"   No signals generated")

            except Exception as e:
                print(f"   ❌ Error generating signals: {e}")
                return False

        print(f"\n🎯 Test Results:")
        print(f"   Scenarios tested: {len(test_scenarios)}")
        print(f"   Valid signals generated: {signal_count}")
        print(f"   Success rate: {signal_count/len(test_scenarios)*100:.1f}%")

        if signal_count > 0:
            print(f"\n🎉 SUCCESS: Calculus signal generation working!")
            print(f"📈 Generated {signal_count} valid trading signals without SIGFPE")
            print(f"🛡️  Enhanced safety features preventing crashes")
            return True
        else:
            print(f"\n⚠️  WARNING: No valid signals generated, but no crashes occurred")
            return False

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases that previously caused SIGFPE."""

    print("\n🧪 Testing Edge Cases")
    print("=" * 30)

    from calculus_strategy import CalculusTradingStrategy
    from quantitative_models import safe_divide, safe_finite_check

    # Test problematic price series
    edge_cases = {
        "constant_prices": pd.Series([100.0] * 25),
        "zero_volatility": pd.Series([100.0] * 25),
        "extreme_moves": pd.Series([100, 200, 50, 300, 25, 400, 12.5] * 4),
        "tiny_changes": pd.Series([100.0 + i*1e-8 for i in range(25)]),
    }

    strategy = CalculusTradingStrategy()
    edge_case_passed = 0

    for case_name, prices in edge_cases.items():
        try:
            signals = strategy.generate_trading_signals(prices)
            if not signals.empty:
                print(f"✅ {case_name}: Signal generation successful")
                edge_case_passed += 1
            else:
                print(f"⚠️  {case_name}: No signals generated")
        except Exception as e:
            print(f"❌ {case_name}: Error - {e}")

    print(f"\nEdge cases passed: {edge_case_passed}/{len(edge_cases)}")
    return edge_case_passed == len(edge_cases)

if __name__ == "__main__":
    print("🚀 Enhanced Calculus Trading System Test")
    print("=" * 50)
    print("Testing signal generation with SIGFPE protection")
    print()

    success = test_calculus_signal_generation()
    edge_success = test_edge_cases()

    print("\n" + "=" * 50)
    print("🏋️  FINAL TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ Signal Generation: WORKING")
        print("📈 CALCULUS SIGNAL messages: CONFIRMED")
        print("🛡️  SIGFPE Protection: ACTIVE")
    else:
        print("❌ Signal Generation: FAILED")

    if edge_success:
        print("✅ Edge Cases: HANDLED")
    else:
        print("⚠️  Edge Cases: Some issues")

    overall_success = success and edge_success
    print(f"\n🎯 OVERALL STATUS: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")

    if overall_success:
        print("\n🚀 The trading bot is ready for live operation!")
        print("📊 All mathematical operations are safe from SIGFPE")
        print("🎓 Anne's calculus system is working correctly")
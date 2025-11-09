#!/usr/bin/env python3
"""
Test Calculus Signal Generation
"""

import pandas as pd
import numpy as np
from calculus_strategy import CalculusTradingStrategy
from quantitative_models import CalculusPriceAnalyzer
from kalman_filter import AdaptiveKalmanFilter

def test_calculus_signals():
    """Test the calculus signal generation framework"""

    # Generate sample price data that mimics real crypto price movement
    np.random.seed(42)
    prices = []
    base_price = 50000

    for i in range(200):
        # Simulate realistic price movement with trend, volatility, and mean reversion
        noise = np.random.normal(0, 0.002)  # 0.2% std deviation
        trend = 0.0001 * np.sin(i * 0.05)  # Cyclical trend
        momentum = 0.00005 * i if i < 100 else -0.00005 * (i - 100)  # Momentum change

        base_price *= (1 + noise + trend + momentum)
        prices.append(base_price)

    price_series = pd.Series(prices)

    print(f"🔬 Testing Calculus Signal Generation")
    print(f"Generated {len(price_series)} price points")
    print(f"Price range: ${price_series.min():.2f} - ${price_series.max():.2f}")
    print(f"Price change: {((price_series.iloc[-1] / price_series.iloc[0]) - 1) * 100:.2f}%")
    print()

    # Test Kalman filtering
    print("📊 Testing Kalman Filter...")
    kalman_filter = AdaptiveKalmanFilter()
    kalman_results = kalman_filter.filter_price_series(price_series)

    if kalman_results.empty:
        print("❌ Kalman filter failed to generate results")
        return False

    print(f"✅ Kalman filter generated {len(kalman_results)} filtered points")
    print(f"Latest velocity: {kalman_results.iloc[-1].get('velocity', 0):.6f}")
    print(f"Latest acceleration: {kalman_results.iloc[-1].get('acceleration', 0):.8f}")
    print()

    # Test calculus strategy
    print("🎯 Testing Calculus Strategy...")
    strategy = CalculusTradingStrategy()

    # Use filtered prices for signal generation
    if 'filtered_price' in kalman_results.columns:
        filtered_prices = kalman_results['filtered_price']
    elif 'price_estimate' in kalman_results.columns:
        filtered_prices = kalman_results['price_estimate']
    else:
        print("❌ No filtered prices available from Kalman filter")
        return False

    signals = strategy.generate_trading_signals(filtered_prices)

    if signals.empty:
        print("❌ No signals generated")
        return False

    print(f"✅ Generated {len(signals)} signals")
    print(f"Valid signals: {signals['valid_signal'].sum()}")

    # Get latest valid signal
    valid_signals = signals[signals['valid_signal'] == True]
    if not valid_signals.empty:
        latest_signal = valid_signals.iloc[-1]
        print()
        print("📈 Latest Valid Signal:")
        print(f"   Signal Type: {latest_signal.get('signal_type', 'Unknown')}")
        print(f"   Interpretation: {latest_signal.get('interpretation', 'Unknown')}")
        print(f"   Price: ${latest_signal.get('price', 0):.2f}")
        print(f"   Velocity: {latest_signal.get('velocity', 0):.6f}")
        print(f"   Acceleration: {latest_signal.get('acceleration', 0):.8f}")
        print(f"   SNR: {latest_signal.get('snr', 0):.3f}")
        print(f"   Confidence: {latest_signal.get('confidence', 0):.2f}")
        print(f"   Forecast: ${latest_signal.get('forecast', 0):.2f}")

        # Check if signal meets trading criteria
        signal_type = latest_signal.get('signal_type', 0)
        confidence = latest_signal.get('confidence', 0)
        snr = latest_signal.get('snr', 0)

        print()
        print("🎯 Trading Assessment:")
        print(f"   Signal meets confidence threshold (≥0.7): {confidence >= 0.7}")
        print(f"   Signal meets SNR threshold (≥0.8): {snr >= 0.8}")
        print(f"   Signal is actionable: {signal_type in [1, 2, 3, 4] and confidence >= 0.7 and snr >= 0.8}")

        return True
    else:
        print("❌ No valid signals found")
        return False

if __name__ == "__main__":
    success = test_calculus_signals()
    if success:
        print("\n✅ Calculus signal generation test PASSED")
    else:
        print("\n❌ Calculus signal generation test FAILED")

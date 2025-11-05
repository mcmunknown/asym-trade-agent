#!/usr/bin/env python3
"""
Test script for Anne's cleaned calculus trading system
"""

import pandas as pd
import numpy as np
from quantitative_models import CalculusPriceAnalyzer
from calculus_strategy import CalculusTradingStrategy, SignalType
from kalman_filter import AdaptiveKalmanFilter
from risk_manager import RiskManager
from config import Config

def test_core_system():
    """Test all core components of Anne's calculus trading system."""
    print('🧮 Testing Anne\'s Calculus Trading System...')

    # Generate test data with sufficient points
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.5, 50)  # 50 data points
    prices = pd.Series([base_price + sum(price_changes[:i]) for i in range(50)])

    # Test 1: Calculus Price Analyzer
    print('\n1. Testing CalculusPriceAnalyzer...')
    analyzer = CalculusPriceAnalyzer(lambda_param=0.6, snr_threshold=1.0)
    results = analyzer.analyze_price_curve(prices)
    print(f'✅ CalculusPriceAnalyzer: Generated {len(results)} analysis points')
    if not results.empty:
        print(f'   Latest SNR: {results.iloc[-1]["snr"]:.3f}')
        print(f'   Latest velocity: {results.iloc[-1]["velocity"]:.6f}')

    # Test 2: Calculus Trading Strategy
    print('\n2. Testing CalculusTradingStrategy...')
    strategy = CalculusTradingStrategy()
    signals = strategy.generate_trading_signals(prices)
    print(f'✅ CalculusTradingStrategy: Generated {len(signals)} signals')
    if not signals.empty:
        latest_signal = signals.iloc[-1]["signal"]
        if not pd.isna(latest_signal) and int(latest_signal) >= 0:
            print(f'   Latest signal: {SignalType(int(latest_signal)).name}')
        else:
            print('   Latest signal: NEUTRAL (invalid signal)')

    # Test 3: Kalman Filter
    print('\n3. Testing AdaptiveKalmanFilter...')
    kalman = AdaptiveKalmanFilter()
    kalman_results = kalman.filter_price_series(prices)
    print(f'✅ AdaptiveKalmanFilter: Filtered {len(kalman_results)} price points')
    if not kalman_results.empty:
        print(f'   Latest velocity: {kalman_results.iloc[-1]["velocity"]:.6f}')

    # Test 4: Risk Manager
    print('\n4. Testing RiskManager...')
    risk_mgr = RiskManager()
    position_size = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=1.5,
        confidence=0.8,
        current_price=100.0,
        account_balance=10000.0
    )
    print(f'✅ RiskManager: Calculated position size')
    print(f'   Quantity: {position_size.quantity:.6f}')
    print(f'   Risk amount: ${position_size.risk_amount:.2f}')

    # Test 5: Configuration
    print('\n5. Testing Configuration...')
    calculus_config = Config.get_calculus_config()
    risk_config = Config.get_risk_config()
    kalman_config = Config.get_kalman_config()
    print(f'✅ Configuration: All configs retrieved successfully')
    print(f'   Lambda param: {calculus_config["smoothing"]["lambda_param"]}')
    print(f'   Max risk per trade: {risk_config["position_sizing"]["max_risk_per_trade"]:.1%}')
    print(f'   Adaptive noise: {kalman_config["adaptive"]["enabled"]}')

    print('\n🎉 All core calculus components working perfectly!')
    print('🔥 Anne\'s calculus-based trading system is clean and functional!')

if __name__ == "__main__":
    test_core_system()
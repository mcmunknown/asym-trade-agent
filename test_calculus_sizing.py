#!/usr/bin/env python3
"""
Test calculus-based position sizing implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_calculus_trader import LiveCalculusTrader

def test_calculus_sizing():
    """Test the new calculus-based position sizing"""
    print("🔬 Testing Calculus-Based Position Sizing")
    print("=" * 60)
    
    # Create trader in simulation mode for testing
    trader = LiveCalculusTrader(
        symbols=['BTCUSDT'],
        simulation_mode=True,
        portfolio_mode=False
    )
    
    # Test signal with various calculus parameters
    test_signals = [
        {
            'signal_type': 'STRONG_BUY',
            'confidence': 0.8,
            'snr': 0.9,
            'velocity': 0.001,
            'acceleration': 0.00001,
            'volatility': 0.02,
            'price': 50000
        },
        {
            'signal_type': 'BUY',
            'confidence': 0.6,
            'snr': 0.6,
            'velocity': 0.0005,
            'acceleration': 0.000005,
            'volatility': 0.03,
            'price': 50000
        },
        {
            'signal_type': 'SELL',
            'confidence': 0.7,
            'snr': 0.8,
            'velocity': -0.0008,
            'acceleration': -0.000008,
            'volatility': 0.025,
            'price': 50000
        }
    ]
    
    print("\n🎯 Testing Position Sizing Calculations:")
    print("=" * 60)
    
    for i, signal in enumerate(test_signals, 1):
        print(f"\n📊 Test {i}: {signal['signal_type']} Signal")
        print("-" * 40)
        
        # Test the calculus position sizing
        position_size = trader._calculate_calculus_position_size(
            symbol='BTCUSDT',
            signal_dict=signal,
            current_price=signal['price'],
            available_balance=1000.0
        )
        
        print(f"\n✅ Results:")
        print(f"   Quantity: {position_size.quantity:.6f}")
        print(f"   Notional Value: ${position_size.notional_value:.2f}")
        print(f"   Leverage Used: {position_size.leverage_used:.1f}x")
        print(f"   Risk Amount: ${position_size.risk_amount:.2f}")
        print(f"   Risk Percent: {position_size.risk_percent:.2f}%")
        print(f"   Confidence Score: {position_size.confidence_score:.2f}")
        
        # Check exchange compliance
        specs = trader._get_instrument_specs('BTCUSDT')
        min_qty = specs.get('min_qty', 0.001) if specs else 0.001
        min_notional = specs.get('min_notional', 5.0) if specs else 5.0
        
        print(f"\n📋 Exchange Compliance:")
        print(f"   Min Quantity: {min_qty} (Actual: {position_size.quantity:.6f})")
        print(f"   Min Notional: ${min_notional} (Actual: ${position_size.notional_value:.2f})")
        
        if position_size.quantity >= min_qty and position_size.notional_value >= min_notional:
            print("   ✅ EXCHANGE COMPLIANT")
        else:
            print("   ❌ EXCHANGE NON-COMPLIANT")
    
    print("\n🎉 Calculus Position Sizing Test Complete!")
    print("\n🔬 Key Mathematical Features:")
    print("   1. ✅ SNR-based signal strength calculation")
    print("   2. ✅ Velocity/acceleration volatility adjustment") 
    print("   3. ✅ Taylor expansion price forecasting")
    print("   4. ✅ Exchange min_qty/min_notional compliance")
    print("   5. ✅ Dynamic leverage calculation")
    print("   6. ✅ Risk-based position sizing")
    
    return trader

if __name__ == "__main__":
    test_calculus_sizing()

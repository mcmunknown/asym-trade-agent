#!/usr/bin/env python3
"""
Test script to verify the position sizing fix works with low balance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_calculus_trader import LiveCalculusTrader
from risk_manager import PositionSize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_position_sizing():
    """Test position sizing with low balance scenarios"""
    print("=" * 60)
    print("🧪 TESTING POSITION SIZING FIX")
    print("=" * 60)

    try:
        # Initialize trader
        trader = LiveCalculusTrader()

        # Test scenarios
        test_scenarios = [
            {
                'symbol': 'LINKUSDT',
                'price': 15.14,
                'available_balance': 6.05,
                'signal': {
                    'velocity': 0.0001,
                    'acceleration': -0.000001,
                    'snr': 0.6,
                    'confidence': 0.8,
                    'volatility': 0.02
                }
            },
            {
                'symbol': 'LTCUSDT',
                'price': 98.44,
                'available_balance': 6.05,
                'signal': {
                    'velocity': -0.0002,
                    'acceleration': 0.000002,
                    'snr': 0.7,
                    'confidence': 0.75,
                    'volatility': 0.025
                }
            },
            {
                'symbol': 'BTCUSDT',
                'price': 101559.50,
                'available_balance': 6.05,
                'signal': {
                    'velocity': 0.00005,
                    'acceleration': 0.0000001,
                    'snr': 0.8,
                    'confidence': 0.9,
                    'volatility': 0.015
                }
            }
        ]

        print("\n📊 Position Sizing Test Results:")
        print("-" * 60)

        for i, scenario in enumerate(test_scenarios, 1):
            symbol = scenario['symbol']
            price = scenario['price']
            balance = scenario['available_balance']
            signal = scenario['signal']

            print(f"\n{i}. Testing {symbol} @ ${price:.2f} with ${balance:.2f} available:")

            # Test position sizing
            position_size = trader._calculate_calculus_position_size(
                symbol=symbol,
                signal_dict=signal,
                current_price=price,
                available_balance=balance
            )

            if isinstance(position_size, PositionSize):
                print(f"   ✅ Position Size Calculation:")
                print(f"      Quantity: {position_size.quantity:.6f}")
                print(f"      Notional Value: ${position_size.notional_value:.2f}")
                print(f"      Leverage Used: {position_size.leverage_used:.1f}x")
                print(f"      Margin Required: ${position_size.margin_required:.2f}")
                print(f"      Risk Amount: ${position_size.risk_amount:.4f}")
                print(f"      Risk Percent: {position_size.risk_percent:.2%}")

                # Check if position is affordable
                if position_size.margin_required <= balance * 0.8:
                    print(f"      ✅ AFFORDABLE - fits within balance constraints")
                else:
                    print(f"      ❌ NOT AFFORDABLE - exceeds balance")

                if position_size.quantity > 0:
                    print(f"      ✅ VALID POSITION - can be traded")
                else:
                    print(f"      ❌ INVALID POSITION - quantity is zero")
            else:
                print(f"   ❌ Position sizing failed")

        print(f"\n🎯 Test Summary:")
        print(f"   The position sizing fix should ensure that:")
        print(f"   • Positions are reduced to fit within 80% of available balance")
        print(f"   • Zero quantity positions are returned if minimum can't be afforded")
        print(f"   • Leverage is properly calculated based on reduced position size")
        print(f"   • Risk amounts are scaled appropriately")

    except Exception as e:
        logger.error(f"Error in test_position_sizing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_position_sizing()
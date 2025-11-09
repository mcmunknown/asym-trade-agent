#!/usr/bin/env python3
"""
Test margin validation without executing real trades
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bybit_client import BybitClient
from risk_manager import RiskManager, PositionSize
from live_calculus_trader import LiveCalculusTrader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_margin_validation():
    """Test margin validation with various balance scenarios"""
    print("=" * 60)
    print("🧪 MARGIN VALIDATION TEST")
    print("=" * 60)

    try:
        # Initialize client
        client = BybitClient()
        if not client.client:
            print("❌ Failed to initialize Bybit client")
            return

        # Get actual account balance
        account_info = client.get_account_balance()
        if not account_info:
            print("❌ Could not fetch account balance")
            return

        available_balance = float(account_info.get('totalAvailableBalance', 0))
        total_equity = float(account_info.get('totalEquity', 0))
        
        print(f"💰 Account Balance:")
        print(f"   Total Equity: ${total_equity:.2f}")
        print(f"   Available: ${available_balance:.2f}")

        # Test scenarios
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'LINKUSDT']
        leverage = 10.0
        
        # Initialize risk manager
        risk_manager = RiskManager(
            max_risk_per_trade=0.02,
            max_leverage=leverage,
            min_risk_reward=1.5
        )

        print(f"\n📈 Testing Position Sizes with {leverage}x leverage:")
        print("-" * 60)

        for symbol in test_symbols:
            # Get market data
            market_data = client.get_market_data(symbol)
            if not market_data:
                continue
                
            price = float(market_data.get('lastPrice', 0))
            if price <= 0:
                continue
                
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=0.8,
                confidence=0.8,
                current_price=price,
                account_balance=available_balance
            )
            
            # Test margin validation
            order_notional = position_size.quantity * price
            margin_required = order_notional / max(position_size.leverage_used, 1.0)
            
            # Check buffers
            margin_buffer = 1.15
            if available_balance < 10:
                margin_buffer = 1.25
                
            print(f"\n{symbol}:")
            print(f"   Price: ${price:.2f}")
            print(f"   Quantity: {position_size.quantity:.6f}")
            print(f"   Notional: ${order_notional:.2f}")
            print(f"   Margin Required: ${margin_required:.2f}")
            print(f"   With Buffer: ${margin_required * margin_buffer:.2f}")
            
            # Validation results
            if margin_required * margin_buffer >= available_balance:
                print(f"   ❌ REJECTED - Insufficient margin with buffer")
                print(f"      Need ${(margin_required * margin_buffer - available_balance + 2):.2f} more")
            elif margin_required >= available_balance:
                print(f"   ❌ REJECTED - Insufficient margin")
                print(f"      Need ${(margin_required - available_balance + 5):.2f} more")
            else:
                margin_usage_percent = (margin_required / available_balance) * 100
                print(f"   ✅ VALID - Margin usage: {margin_usage_percent:.1f}%")
                
                if margin_usage_percent > 80:
                    print(f"   ⚠️  HIGH MARGIN USAGE")

        # Test with simulated low balances
        print(f"\n🧪 Simulated Low Balance Scenarios:")
        print("-" * 60)
        
        low_balances = [1.0, 2.0, 5.0, 8.0]
        
        for test_balance in low_balances:
            print(f"\nTesting with ${test_balance:.2f} balance:")
            
            for symbol in ['BTCUSDT', 'LTCUSDT']:
                market_data = client.get_market_data(symbol)
                if not market_data:
                    continue
                    
                price = float(market_data.get('lastPrice', 0))
                if price <= 0:
                    continue
                    
                position_size = risk_manager.calculate_position_size(
                    symbol=symbol,
                    signal_strength=0.8,
                    confidence=0.8,
                    current_price=price,
                    account_balance=test_balance
                )
                
                order_notional = position_size.quantity * price
                margin_required = order_notional / max(position_size.leverage_used, 1.0)
                
                margin_buffer = 1.25  # 25% for low balances
                
                if margin_required * margin_buffer >= test_balance:
                    print(f"   {symbol}: ❌ REJECTED (need ${margin_required * margin_buffer:.2f}, have ${test_balance:.2f})")
                else:
                    margin_usage = (margin_required / test_balance) * 100
                    print(f"   {symbol}: ✅ VALID (${margin_required:.2f} margin, {margin_usage:.1f}% usage)")

    except Exception as e:
        logger.error(f"Error in test_margin_validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_margin_validation()

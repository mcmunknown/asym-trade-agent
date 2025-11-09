#!/usr/bin/env python3
"""
Debug script to check account balance and margin issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bybit_client import BybitClient
from config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_balance():
    """Debug account balance and margin issues"""
    print("=" * 60)
    print("🔍 BALANCE DEBUG - Bybit Account Information")
    print("=" * 60)

    try:
        # Initialize client
        client = BybitClient()
        if not client.client:
            print("❌ Failed to initialize Bybit client")
            return

        # Get account balance
        print("\n📊 Getting account balance...")
        account_info = client.get_account_balance()

        if not account_info:
            print("❌ Could not fetch account balance")
            return

        print(f"✅ Account info received:")
        for key, value in account_info.items():
            if isinstance(value, str) and value.replace('.', '').isdigit():
                print(f"   {key}: ${float(value):.2f}")
            else:
                print(f"   {key}: {value}")

        # Extract key values
        total_equity = float(account_info.get('totalEquity', 0))
        available_balance = float(account_info.get('totalAvailableBalance', 0))
        wallet_balance = float(account_info.get('totalWalletBalance', 0))
        imr = float(account_info.get('accountIMR', 0))  # Initial Margin Requirement
        mmr = float(account_info.get('accountMMR', 0))  # Maintenance Margin Requirement

        print(f"\n💰 Key Balance Metrics:")
        print(f"   Total Equity: ${total_equity:.2f}")
        print(f"   Available Balance: ${available_balance:.2f}")
        print(f"   Wallet Balance: ${wallet_balance:.2f}")
        print(f"   Initial Margin Req: ${imr:.2f}")
        print(f"   Maintenance Margin Req: ${mmr:.2f}")

        # Calculate margin usage
        if total_equity > 0:
            margin_usage_percent = (imr / total_equity) * 100
            print(f"   Margin Usage: {margin_usage_percent:.1f}%")

        # Check for issue
        print(f"\n🚨 Issue Analysis:")
        if available_balance == 0 and total_equity > 0:
            print("   ⚠️  Available balance is 0 but total equity > 0")
            print("   This suggests margin is being used for open positions")
            print(f"   Available for trading: ${total_equity - imr:.2f}")

        if available_balance < 10:
            print("   ⚠️  Very low available balance - position sizing will be minimal")

        # Test position sizing for common symbols
        print(f"\n📈 Testing Position Sizes:")
        symbols = ['BTCUSDT', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT']

        for symbol in symbols:
            market_data = client.get_market_data(symbol)
            if market_data:
                price = float(market_data.get('lastPrice', 0))
                if price > 0:
                    # Calculate min position size with 5x leverage
                    min_position_value = 5  # $5 minimum
                    min_quantity = min_position_value / price
                    required_margin = min_position_value / 5  # With 5x leverage

                    print(f"   {symbol}:")
                    print(f"     Price: ${price:.2f}")
                    print(f"     Min quantity: {min_quantity:.6f}")
                    print(f"     Required margin (5x): ${required_margin:.2f}")

                    if required_margin > available_balance:
                        print(f"     ❌ Insufficient margin!")
                    else:
                        print(f"     ✅ Sufficient margin")

    except Exception as e:
        logger.error(f"Error in debug_balance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_balance()
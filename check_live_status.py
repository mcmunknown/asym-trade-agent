#!/usr/bin/env python3
"""
Check Live Trading Status and Account Balance
"""

import os
from bybit_client import BybitClient
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_live_status():
    """Check account status and readiness for live trading"""
    print('🔍 CHECKING LIVE TRADING STATUS')
    print('=' * 50)

    # Check environment variables
    print('\n📋 ENVIRONMENT CONFIGURATION:')
    print(f'   BYBIT_TESTNET: {os.getenv("BYBIT_TESTNET", "false")}')
    print(f'   LIVE_TRADING_ENABLED: {os.getenv("LIVE_TRADING_ENABLED", "false")}')
    print(f'   BYBIT_API_KEY: {"✅ Configured" if os.getenv("BYBIT_API_KEY") else "❌ Missing"}')
    print(f'   BYBIT_API_SECRET: {"✅ Configured" if os.getenv("BYBIT_API_SECRET") else "❌ Missing"}')

    try:
        # Initialize Bybit client
        print('\n🔌 CONNECTING TO BYBIT API...')
        client = BybitClient()

        # Get account balance
        print('\n💰 ACCOUNT BALANCE:')
        balance_info = client.get_account_balance()

        if balance_info:
            total_balance = float(balance_info['totalEquity'])
            available_balance = float(balance_info['totalAvailableBalance'])

            print(f'   Total Balance: ${total_balance:.2f}')
            print(f'   Available Balance: ${available_balance:.2f}')

            if total_balance > 0:
                print(f'   ✅ Account has funds for trading')
            else:
                print(f'   ⚠️  Account has no funds')
        else:
            print(f'   ❌ Could not retrieve balance information')

        # Get trading status
        print('\n📊 TRADING STATUS:')
        try:
            positions = client.get_positions()
            if 'result' in positions and positions['result']:
                active_positions = [p for p in positions['result']['list'] if float(p['size']) > 0]
                print(f'   Active Positions: {len(active_positions)}')

                for pos in active_positions[:3]:  # Show first 3 positions
                    symbol = pos['symbol']
                    side = pos['side']
                    size = float(pos['size'])
                    entry_price = float(pos['avgPrice'])
                    mark_price = float(pos['markPrice'])
                    pnl = float(pos['unrealisedPnl'])

                    print(f'   - {symbol}: {side} {size:.6f} @ ${entry_price:.2f} | PnL: ${pnl:.2f}')
            else:
                print(f'   No open positions')
        except Exception as e:
            print(f'   ⚠️  Could not fetch positions: {e}')

        print('\n🎯 LIVE TRADING READINESS:')

        live_trading_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
        testnet_mode = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

        if not testnet_mode and live_trading_enabled and 'total_balance' in locals() and total_balance > 0:
            print(f'   ✅ READY FOR LIVE TRADING!')
            print(f'      - Live mode: ✅')
            print(f'      - Trading enabled: ✅')
            print(f'      - Account funded: ✅')
            print(f'      - Balance: ${total_balance:.2f}')
        elif testnet_mode:
            print(f'   🧪 TESTNET MODE')
            print(f'      Set BYBIT_TESTNET=false for live trading')
        elif not live_trading_enabled:
            print(f'   🔒 TRADING DISABLED')
            print(f'      Set LIVE_TRADING_ENABLED=true to enable')
        else:
            print(f'   ⚠️  ACCOUNT NOT FUNDED')
            print(f'      Add funds to account before live trading')

    except Exception as e:
        print(f'❌ ERROR: {e}')
        logger.exception("Failed to check live status")

if __name__ == "__main__":
    check_live_status()
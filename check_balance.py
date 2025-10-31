#!/usr/bin/env python3
"""
Balance Check Script - Check current trading account balance and losses
"""

import json
from bybit_client import BybitClient
from trading_engine import TradingEngine

def main():
    print("🔍 BALANCE CHECK - Asymmetric Trading Agent")
    print("=" * 50)

    # Initialize clients
    bybit = BybitClient()
    engine = TradingEngine()

    # Test connection
    print("\n1. TESTING API CONNECTION...")
    if bybit.test_connection():
        print("✅ Bybit API connection successful")
    else:
        print("❌ Bybit API connection failed")
        return

    # Get account balance
    print("\n2. ACCOUNT BALANCE STATUS...")
    balance = bybit.get_account_balance()
    if balance:
        # Helper function to safely convert string to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value and str(value).strip() else default
            except (ValueError, TypeError):
                return default

        print("✅ Account Balance Retrieved:")
        total_equity = safe_float(balance.get('totalEquity'))
        available_balance = safe_float(balance.get('totalAvailableBalance'))
        wallet_balance = safe_float(balance.get('totalWalletBalance'))
        unrealized_pnl = safe_float(balance.get('totalPerpUPL'))
        initial_margin = safe_float(balance.get('accountIMR'))
        maintenance_margin = safe_float(balance.get('accountMMR'))

        print(f"   💰 Total Equity: ${total_equity:.2f}")
        print(f"   💵 Available Balance: ${available_balance:.2f}")
        print(f"   💼 Wallet Balance: ${wallet_balance:.2f}")
        print(f"   📊 Unrealized P&L: ${unrealized_pnl:.2f}")
        print(f"   🛡️ Initial Margin: ${initial_margin:.2f}")
        print(f"   ⚠️ Maintenance Margin: ${maintenance_margin:.2f}")

        # Calculate balance loss
        if wallet_balance > 0:
            balance_change = ((total_equity - wallet_balance) / wallet_balance) * 100
            print(f"\n   📈 BALANCE CHANGE: {balance_change:+.2f}%")

            if balance_change < 0:
                print(f"   📉 BALANCE LOSS: ${abs(total_equity - wallet_balance):.2f}")
            else:
                print(f"   📈 BALANCE GAIN: ${total_equity - wallet_balance:.2f}")
    else:
        print("❌ Failed to retrieve account balance")
        return

    # Get portfolio summary
    print("\n3. PORTFOLIO SUMMARY...")
    portfolio = engine.get_portfolio_summary()
    if portfolio:
        print("✅ Portfolio Summary:")
        print(f"   💰 Total Balance: ${portfolio.get('total_balance', 0):.2f}")
        print(f"   💵 Available: ${portfolio.get('available_balance', 0):.2f}")
        print(f"   📊 Active Positions: {portfolio.get('active_positions', 0)}")
        print(f"   📈 Total Trades: {portfolio.get('total_trades', 0)}")
        print(f"   💼 Total Invested: ${portfolio.get('total_invested', 0):.2f}")
        print(f"   📊 Unrealized P&L: ${portfolio.get('unrealized_pnl', 0):.2f}")

        # Calculate P&L percentage
        total_invested = portfolio.get('total_invested', 0)
        unrealized_pnl = portfolio.get('unrealized_pnl', 0)
        if total_invested > 0:
            pnl_percentage = (unrealized_pnl / total_invested) * 100
            print(f"   📈 P&L Percentage: {pnl_percentage:+.2f}%")
    else:
        print("❌ Failed to get portfolio summary")

    # Get active positions with P&L details
    print("\n4. ACTIVE POSITIONS & P&L...")
    active_positions = engine.get_active_positions()

    if active_positions:
        print(f"✅ Found {len(active_positions)} active positions:")
        total_position_pnl = 0

        for symbol, position_data in active_positions.items():
            print(f"\n   📊 {symbol}:")
            print(f"      Status: {position_data.get('status', 'Unknown')}")
            print(f"      Signal: {position_data.get('signal', {}).get('signal', 'Unknown')}")

            # Get current position info from exchange
            position_info = bybit.get_position_info(symbol)
            if position_info:
                unrealized_pnl = float(position_info.get('unrealisedPnl', 0))
                mark_price = float(position_info.get('markPrice', 0))
                entry_price = float(position_info.get('entryPrice', 0))
                position_size = float(position_info.get('size', 0))
                leverage = float(position_info.get('leverage', 0))

                print(f"      💰 Position Size: {position_size}")
                print(f"      🎯 Entry Price: ${entry_price:.4f}")
                print(f"      📊 Mark Price: ${mark_price:.4f}")
                print(f"      ⚡ Leverage: {leverage}x")
                print(f"      📈 Unrealized P&L: ${unrealized_pnl:.4f}")

                if entry_price > 0:
                    price_change_pct = ((mark_price - entry_price) / entry_price) * 100
                    print(f"      📊 Price Change: {price_change_pct:+.2f}%")

                total_position_pnl += unrealized_pnl
            else:
                print(f"      ❌ Could not retrieve position info")

        print(f"\n   💰 TOTAL POSITION P&L: ${total_position_pnl:.4f}")

        # Calculate loss analysis
        if total_position_pnl < 0:
            print(f"   📉 TOTAL POSITION LOSS: ${abs(total_position_pnl):.4f}")
            loss_pct = (abs(total_position_pnl) / portfolio.get('total_invested', 1)) * 100
            print(f"   📊 LOSS PERCENTAGE: {loss_pct:.2f}% of invested capital")
    else:
        print("✅ No active positions found")

    print("\n" + "=" * 50)
    print("🔍 BALANCE CHECK COMPLETE")

    # Summary
    if balance and portfolio:
        total_equity = float(balance.get('totalEquity', 0))
        unrealized_pnl = portfolio.get('unrealized_pnl', 0)

        if total_equity < 100:  # Less than $100
            print("⚠️  WARNING: Low balance detected!")
        if unrealized_pnl < -10:  # More than $10 loss
            print("⚠️  WARNING: Significant unrealized loss detected!")
        if not active_positions and total_equity < 50:
            print("⚠️  WARNING: Low balance with no active positions!")

if __name__ == "__main__":
    main()
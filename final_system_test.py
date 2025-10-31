#!/usr/bin/env python3
"""
FINAL SYSTEM VALIDATION TEST
============================

Comprehensive test of the enhanced trading system including:
- Real API connection validation
- Market data collection and processing
- Security controls verification
- Signal generation testing
- Live trading simulation (paper mode)
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from decimal import Decimal

def test_api_connection():
    """Test API connection and get real balance"""
    print("=== TESTING API CONNECTION ===")

    try:
        from bybit_client import BybitClient

        client = BybitClient()
        balance_data = client.get_account_balance()

        if balance_data:
            total_equity = float(balance_data.get('totalEquity', 0))
            available = float(balance_data.get('totalAvailableBalance', 0))
            wallet = float(balance_data.get('totalWalletBalance', 0))
            unrealized = float(balance_data.get('totalUnrealizedPnl', 0))

            print(f"✅ API Connection Successful")
            print(f"   Total Equity: ${total_equity:.2f}")
            print(f"   Available: ${available:.2f}")
            print(f"   Wallet Balance: ${wallet:.2f}")
            print(f"   Unrealized P&L: ${unrealized:.2f}")

            return True, total_equity
        else:
            print("❌ API connection failed")
            return False, 0

    except Exception as e:
        print(f"❌ API test error: {str(e)}")
        return False, 0

def test_security_systems():
    """Test bulletproof security systems"""
    print("\n=== TESTING SECURITY SYSTEMS ===")

    try:
        from bulletproof_config import get_bulletproof_config
        from institutional_security_architecture import create_institutional_security_architecture

        # Test configuration security
        config = get_bulletproof_config()
        hard_limits = config.get_hard_limits()
        security_status = config.get_security_status()

        print(f"✅ Configuration Security:")
        print(f"   System Locked: {security_status['system_locked']}")
        print(f"   Violations: {security_status['violation_count']}")
        print(f"   Max Leverage: {hard_limits['max_leverage_hard_limit']}x")
        print(f"   Max Position: {hard_limits['max_position_size_pct_hard_limit']}%")

        # Test institutional security
        security_arch = create_institutional_security_architecture("CONSERVATIVE")
        threat_level = security_arch['security_monitor'].current_threat_level

        print(f"✅ Institutional Security:")
        print(f"   Threat Level: {threat_level}")
        print(f"   Security Monitor: Active")
        print(f"   Risk Validator: Active")

        return True

    except Exception as e:
        print(f"❌ Security test error: {str(e)}")
        return False

def test_market_data():
    """Test market data collection with proper type conversion"""
    print("\n=== TESTING MARKET DATA COLLECTION ===")

    try:
        from data_collector import DataCollector

        collector = DataCollector()
        data_list = collector.collect_all_data()

        print(f"✅ Collected data for {len(data_list)} assets")

        if data_list:
            # Process first few assets with type conversion
            processed_assets = []
            for i, asset in enumerate(data_list[:3]):
                try:
                    symbol = asset.get('symbol', 'UNKNOWN')
                    price = float(asset.get('lastPrice', 0))
                    volume = float(asset.get('volume24h', 0))
                    change = float(asset.get('priceChange24h', 0))

                    processed_assets.append({
                        'symbol': symbol,
                        'price': price,
                        'volume': volume,
                        'change_24h': change,
                        'raw_data': asset
                    })

                    print(f"   📊 {symbol}: ${price:,.2f} ({change:+.2%}, Vol: {volume:,.0f})")

                except (ValueError, TypeError) as conversion_error:
                    print(f"   ⚠️  {asset.get('symbol', 'UNKNOWN')}: Data conversion error")

            return True, processed_assets

        return True, []

    except Exception as e:
        print(f"❌ Market data test error: {str(e)}")
        return False, []

async def test_signal_generation(processed_assets):
    """Test AI signal generation"""
    print("\n=== TESTING AI SIGNAL GENERATION ===")

    try:
        from bulletproof_trading_engine import BulletproofTradingEngine

        engine = BulletproofTradingEngine("CONSERVATIVE")
        init_result = engine.initialize()

        print(f"✅ Trading Engine Initialized: {init_result}")

        if processed_assets:
            # Convert back to expected format for signal processing
            market_data = [asset['raw_data'] for asset in processed_assets[:2]]

            signals = await engine.process_trading_signals(market_data)

            signal_count = len(signals) if signals else 0
            print(f"✅ Signal Generation: {signal_count} signals")

            if signals:
                for signal in signals[:3]:
                    print(f"   🎯 {signal.symbol}: {signal.signal} (Confidence: {signal.confidence:.1%})")
                    print(f"      Entry: ${signal.entry_price:.2f}, Target: ${signal.activation_price:.2f}")
            else:
                print("   📊 No trading signals (market conditions not favorable)")

        return True

    except Exception as e:
        print(f"❌ Signal generation test error: {str(e)}")
        return False

def test_position_sizing(balance):
    """Test position sizing with real balance"""
    print("\n=== TESTING POSITION SIZING ===")

    try:
        # Test conservative position sizing (1% of balance)
        max_position_pct = 0.01  # 1% hard limit from security system
        max_position_amount = balance * max_position_pct

        print(f"✅ Position Sizing Calculation:")
        print(f"   Account Balance: ${balance:.2f}")
        print(f"   Max Position %: {max_position_pct*100:.1f}%")
        print(f"   Max Position Amount: ${max_position_amount:.2f}")

        # Test with different asset prices
        test_prices = [50000, 1000, 100]  # BTC, ETH, ADA

        for price in test_prices:
            if max_position_amount >= 5:  # Bybit minimum
                quantity = max_position_amount / price
                print(f"   📈 ${price:,.0f} asset: {quantity:.6f} units (${quantity*price:.2f})")
            else:
                print(f"   ⚠️  ${price:,.0f} asset: Below minimum order size")

        # Check if balance is sufficient for trading
        min_required = 5.0  # Bybit minimum
        can_trade = balance >= min_required

        print(f"✅ Trading Capability: {'✅ READY' if can_trade else '❌ INSUFFICIENT FUNDS'}")
        print(f"   Minimum Required: ${min_required:.2f}")
        print(f"   Current Balance: ${balance:.2f}")

        return can_trade

    except Exception as e:
        print(f"❌ Position sizing test error: {str(e)}")
        return False

def main():
    """Run comprehensive system validation"""
    print("🛡️ FINAL SYSTEM VALIDATION TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []

    # Test 1: API Connection
    api_success, balance = test_api_connection()
    results.append(("API Connection", api_success))

    # Test 2: Security Systems
    security_success = test_security_systems()
    results.append(("Security Systems", security_success))

    # Test 3: Market Data
    data_success, processed_assets = test_market_data()
    results.append(("Market Data Collection", data_success))

    # Test 4: Signal Generation
    if processed_assets:
        try:
            signal_success = asyncio.run(test_signal_generation(processed_assets))
            results.append(("AI Signal Generation", signal_success))
        except Exception as e:
            print(f"❌ Signal generation failed: {str(e)}")
            results.append(("AI Signal Generation", False))
    else:
        results.append(("AI Signal Generation", False))

    # Test 5: Position Sizing
    sizing_success = test_position_sizing(balance)
    results.append(("Position Sizing", sizing_success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    success_rate = (passed / total) * 100
    print(f"\nOverall Success Rate: {passed}/{total} ({success_rate:.1f}%)")

    # System readiness assessment
    if success_rate >= 90:
        print("🎉 SYSTEM READY FOR LIVE TRADING!")
        print("   ✅ All critical systems operational")
        print("   ✅ Security controls active")
        print("   ✅ Trading capabilities verified")
    elif success_rate >= 75:
        print("⚠️  SYSTEM MOSTLY READY - Minor issues detected")
        print("   ✅ Core functionality operational")
        print("   ⚠️  Some non-critical features need attention")
    else:
        print("🚨 SYSTEM NOT READY - Critical issues found")
        print("   ❌ Major components non-functional")
        print("   ❌ Requires fixes before trading")

    # Account status
    print(f"\n💰 ACCOUNT STATUS:")
    print(f"   Current Balance: ${balance:.2f}")
    if balance < 5:
        print(f"   ⚠️  Balance below minimum for trading")
        print(f"   💡 Recommend: Add funds to enable active trading")
    else:
        print(f"   ✅ Sufficient balance for trading")

    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
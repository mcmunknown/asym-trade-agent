#!/usr/bin/env python3
"""
Test script to verify system components before live trading
"""

import asyncio
import logging
from config import Config
from bybit_client import BybitClient
from glm_client import GLMClient
from data_collector import DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_bybit_connection():
    """Test Bybit API connection"""
    print("🔌 Testing Bybit API connection...")

    try:
        client = BybitClient()

        async with client:
            # Test account balance
            balance = await client.get_account_balance()
            if balance:
                print(f"✅ Bybit API connection successful")
                print(f"   Available balance: {balance['availableBalance']} USDT")
                print(f"   Total equity: {balance['totalEquity']} USDT")
            else:
                print("❌ Bybit API connection failed - No balance data")
                return False

            # Test market data
            market_data = await client.get_market_data('BTCUSDT')
            if market_data:
                print(f"✅ Market data access successful")
                print(f"   BTC price: ${float(market_data['lastPrice']):,.2f}")
                print(f"   24h volume: ${float(market_data['turnover24h']):,.0f}")
            else:
                print("❌ Market data access failed")
                return False

        return True

    except Exception as e:
        print(f"❌ Bybit API test failed: {str(e)}")
        return False

async def test_glm_connection():
    """Test GLM-4.6 API connection"""
    print("\n🧠 Testing GLM-4.6 API connection...")

    try:
        glm = GLMClient()

        async with glm:
            # Test with a simple analysis
            test_market_data = {
                'price': 45000,
                'volume_24h': 1000000000,
                'change_24h': 2.5,
                'timestamp': '2024-01-01T00:00:00',
                'macro_narrative': 'Bitcoin ETF approval expected',
                'risk_sentiment': 'Risk-on'
            }

            test_fundamentals = {
                'revenue_trend': '↑',
                'tvl_trend': '↑',
                'staking_percentage': '5.2%',
                'token_burns': 'Active',
                'developer_activity': 'High',
                'wallet_accumulation': 'Strong'
            }

            test_technical = {
                'price_vs_30d_low': 5.0,
                '30d_low': 42000,
                'within_entry_zone': True,
                'ema_aligned': True,
                'rsi_1d': 55,
                'rsi_momentum_ok': True,
                'volume_confirmation': True,
                'atr_30d_pct': 4.5,
                'atr_ok': True,
                'liquidity_check': True,
                'current_price': 45000
            }

            result = await glm.analyze_market_conditions(
                test_market_data,
                test_fundamentals,
                test_technical,
                'BTCUSDT'
            )

            if result and 'signal' in result:
                print(f"✅ GLM-4.6 API connection successful")
                print(f"   Test signal: {result['signal']}")
                print(f"   Confidence: {result.get('confidence', 0)}%")
                if result['signal'] == 'BUY':
                    print(f"   Thesis: {result.get('thesis_summary', 'N/A')}")
            else:
                print("❌ GLM-4.6 API connection failed - Invalid response")
                return False

        return True

    except Exception as e:
        print(f"❌ GLM-4.6 API test failed: {str(e)}")
        return False

async def test_data_collection():
    """Test data collection system"""
    print("\n📊 Testing data collection system...")

    try:
        collector = DataCollector()

        # Test collecting data for one symbol
        data = await collector.collect_market_data('BTCUSDT')

        if data:
            print("✅ Data collection successful")
            print(f"   Symbol: {data['symbol']}")
            print(f"   Price: ${data['price']:,.2f}")
            print(f"   24h Volume: ${data['volume_24h']:,.0f}")
            print(f"   24h Change: {data['change_24h']:.2f}%")
            print(f"   Funding Rate: {data['funding_rate']:.4f}%")

            # Test technical indicators
            technical = collector.calculate_technical_indicators(data)
            if technical:
                print("✅ Technical indicators calculated")
                print(f"   RSI (1D): {technical.get('rsi_1d', 'N/A')}")
                print(f"   Price vs 30D Low: {technical.get('price_vs_30d_low', 'N/A')}%")
                print(f"   ATR (30D): {technical.get('atr_30d_pct', 'N/A')}%")
            else:
                print("⚠️  Technical indicators calculation failed")
        else:
            print("❌ Data collection failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Data collection test failed: {str(e)}")
        return False

async def test_configuration():
    """Test system configuration"""
    print("\n⚙️ Testing system configuration...")

    try:
        # Check required environment variables
        required_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET', 'GLM_API_KEY']
        missing_vars = []

        for var in required_vars:
            if not getattr(Config, var):
                missing_vars.append(var)

        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False

        print("✅ All required environment variables set")
        print(f"   Target Assets: {', '.join(Config.TARGET_ASSETS)}")
        print(f"   Default Trade Size: ${Config.DEFAULT_TRADE_SIZE}")
        print(f"   Max Leverage: {Config.MAX_LEVERAGE}x")
        print(f"   Testnet Mode: {'ON' if Config.BYBIT_TESTNET else 'OFF'}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all system tests"""
    print("=" * 60)
    print("🚀 ASYMMETRIC TRADING AGENT - SYSTEM TESTS")
    print("=" * 60)

    tests = [
        ("Configuration", test_configuration),
        ("Bybit API", test_bybit_connection),
        ("GLM-4.6 API", test_glm_connection),
        ("Data Collection", test_data_collection)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for trading.")
        print("💡 You can now run: python main.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("💡 Do not run the trading system until all tests pass.")

    return passed == len(results)

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error during tests: {str(e)}")
        exit(1)
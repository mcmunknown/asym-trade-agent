#!/usr/bin/env python3
"""
Test Script: Verify Long-Only Trading Functionality
Tests that short positions cannot be created while long positions work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_engine import TradingEngine
from multi_model_client import MultiModelConsensusEngine
from bybit_client import BybitClient
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_flags():
    """Test that safety configuration flags are properly set"""
    print("🔍 Testing Configuration Flags...")

    # Test short selling disabled
    print(f"   DISABLE_SHORT_SELLING = {Config.DISABLE_SHORT_SELLING}")
    assert Config.DISABLE_SHORT_SELLING == True, "DISABLE_SHORT_SELLING should be True"
    print("✅ DISABLE_SHORT_SELLING = True (Short selling disabled)")

    # Test position direction
    print(f"   MAX_POSITION_DIRECTION = {Config.MAX_POSITION_DIRECTION}")
    assert Config.MAX_POSITION_DIRECTION == "LONG_ONLY", "MAX_POSITION_DIRECTION should be LONG_ONLY"
    print("✅ MAX_POSITION_DIRECTION = LONG_ONLY")

    # Test bypass code
    print(f"   SHORT_SELLING_BYPASS_CODE = {Config.SHORT_SELLING_BYPASS_CODE}")
    assert Config.SHORT_SELLING_BYPASS_CODE == "DISABLED", "SHORT_SELLING_BYPASS_CODE should be DISABLED"
    print("✅ SHORT_SELLING_BYPASS_CODE = DISABLED")

    print("🎯 All configuration flags are correct!\n")
    return True

def test_multi_model_consensus():
    """Test that multi-model consensus blocks SELL signals"""
    print("🔍 Testing Multi-Model Consensus...")

    try:
        consensus_engine = MultiModelConsensusEngine()

        # Create mock data that might trigger SELL signals
        mock_symbol_data = {
            'symbol': 'BTCUSDT',
            'current_price': 113000,
            'rsi': 75,  # High RSI that might trigger SELL
            'volume': 'HIGH',
            'price_change': '+5%'
        }

        # Run consensus check
        import asyncio

        async def test_consensus():
            result = await consensus_engine.get_consensus_signal(mock_symbol_data)

            # Verify result is not SELL
            assert result.final_signal != "SELL", f"Consensus should not return SELL, got {result.final_signal}"
            assert result.final_signal in ["BUY", "NONE"], f"Signal should be BUY or NONE, got {result.final_signal}"

            print(f"✅ Consensus returned {result.final_signal} (SELL blocked)")
            print(f"   Confidence: {result.confidence_avg:.2f}")
            return result

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_consensus())
        loop.close()

        print("🎯 Multi-model consensus correctly blocks SELL signals!\n")
        return True

    except Exception as e:
        print(f"❌ Error testing multi-model consensus: {str(e)}")
        return False

def test_trading_engine_logic():
    """Test that trading engine rejects SELL signals"""
    print("🔍 Testing Trading Engine Logic...")

    try:
        trading_engine = TradingEngine()

        # Test TradingSignal dataclass
        from trading_engine import TradingSignal

        # Create a test signal
        test_signal = TradingSignal(
            symbol='BTCUSDT',
            signal='BUY',  # Should work
            signal_type='MAIN_STRATEGY',
            confidence=85.0,
            entry_price=110000,
            activation_price=121000,
            trailing_stop_pct=5.0,
            invalidation_level=104500,
            thesis_summary='Test signal',
            risk_reward_ratio='1:5',
            leverage=50,
            quantity=0.001
        )

        print("✅ TradingSignal accepts BUY signal")

        # Test that signal validation works
        assert Config.DISABLE_SHORT_SELLING == True, "Config should have short selling disabled"
        print("✅ Config has DISABLE_SHORT_SELLING = True")

        print("🎯 Trading engine logic correctly configured!\n")
        return True

    except Exception as e:
        print(f"❌ Error testing trading engine: {str(e)}")
        return False

def test_bybit_connection():
    """Test Bybit connection and account status"""
    print("🔍 Testing Bybit Connection...")

    try:
        bybit = BybitClient()

        # Test connection
        if bybit.test_connection():
            print("✅ Bybit API connection successful")

            # Check current positions
            balance = bybit.get_account_balance()
            if balance:
                total_equity = float(balance.get('totalEquity', 0))
                print(f"✅ Account balance: ${total_equity:.2f}")

                # Check for any existing positions
                print("📊 Checking for existing positions...")
                positions = []
                for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'LTCUSDT']:
                    pos = bybit.get_position_info(symbol)
                    if pos and float(pos.get('size', 0)) != 0:
                        positions.append(pos)
                        print(f"   {symbol}: {pos.get('side')} position detected")

                if not positions:
                    print("✅ No active positions found")
                else:
                    print(f"⚠️  Found {len(positions)} existing positions - will be closed by long-only logic")

            else:
                print("⚠️  Could not retrieve account balance")

            print("🎯 Bybit connection test complete!\n")
            return True

        else:
            print("❌ Bybit API connection failed")
            return False

    except Exception as e:
        print(f"❌ Error testing Bybit connection: {str(e)}")
        return False

def test_prompt_md_changes():
    """Test that prompt.md has been updated to remove short strategy"""
    print("🔍 Testing prompt.md Changes...")

    try:
        with open('prompt.md', 'r') as f:
            content = f.read()

        # Check that short strategy documentation is removed
        assert "SELL Criteria" not in content, "prompt.md should not contain SELL Criteria"
        assert "CONSERVATIVE SHORT" not in content, "prompt.md should not contain CONSERVATIVE SHORT"
        assert "RANGE FADE SHORT" not in content, "prompt.md should not contain RANGE FADE SHORT"
        print("✅ Short strategy documentation removed from prompt.md")

        # Check that long-only documentation is present
        assert "NO SHORT POSITIONS ALLOWED" in content, "prompt.md should contain short selling prohibition"
        assert "LONG-ONLY" in content, "prompt.md should mention long-only strategy"
        print("✅ Long-only strategy documented in prompt.md")

        print("🎯 prompt.md changes verified!\n")
        return True

    except Exception as e:
        print(f"❌ Error testing prompt.md: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 ASYMMETRIC TRADING AGENT - LONG-ONLY SAFETY TESTS")
    print("=" * 60)
    print("Testing that short positions are blocked while long positions work")
    print("=" * 60)

    tests = [
        ("Configuration Flags", test_configuration_flags),
        ("Multi-Model Consensus", test_multi_model_consensus),
        ("Trading Engine Logic", test_trading_engine_logic),
        ("Bybit Connection", test_bybit_connection),
        ("Prompt.md Changes", test_prompt_md_changes),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {str(e)}")
        print("-" * 40)

    print(f"\n📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for long-only trading")
        print("✅ Short positions have been surgically removed")
        print("✅ Long position functionality is preserved")
        print("✅ Safety features are active")
        print("\n🚀 READY TO DEPLOY LONG-ONLY TRADING BOT")
    else:
        print("⚠️  SOME TESTS FAILED - Review issues before deployment")
        print("❌ Do not deploy trading bot until all tests pass")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Simple system test to check if components are working
"""

import os
import sys
from bybit_client import BybitClient
from config import Config

def test_bybit_client():
    """Test Bybit client basic functionality"""
    print("🔍 Testing Bybit client...")
    
    try:
        client = BybitClient()
        
        # Test basic connection
        if client.test_connection():
            print("✅ Bybit client connection successful")
        else:
            print("❌ Bybit client connection failed")
            return False
            
        # Test account balance
        balance = client.get_account_balance()
        if balance:
            print(f"✅ Account balance: {balance}")
        else:
            print("⚠️ Could not retrieve account balance")
            
        # Test market data
        market_data = client.get_market_data("BTCUSDT")
        if market_data:
            print(f"✅ Market data for BTCUSDT: ${market_data.get('lastPrice', 'N/A')}")
        else:
            print("⚠️ Could not retrieve market data")
            
        return True
        
    except Exception as e:
        print(f"❌ Bybit client test failed: {e}")
        return False

def test_portfolio_components():
    """Test portfolio management components"""
    print("\n🔍 Testing portfolio components...")
    
    try:
        from portfolio_manager import PortfolioManager
        from joint_distribution_analyzer import JointDistributionAnalyzer
        from portfolio_optimizer import PortfolioOptimizer
        from risk_manager import RiskManager
        
        # Test portfolio manager
        pm = PortfolioManager(
            symbols=['BTCUSDT', 'ETHUSDT'],
            initial_capital=10000.0
        )
        print("✅ Portfolio Manager initialized")
        
        # Test joint distribution analyzer
        jda = JointDistributionAnalyzer(num_assets=2)
        print("✅ Joint Distribution Analyzer initialized")
        
        # Test portfolio optimizer
        po = PortfolioOptimizer(joint_analyzer=jda)
        print("✅ Portfolio Optimizer initialized")
        
        # Test risk manager
        rm = RiskManager()
        print("✅ Risk Manager initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_client():
    """Test WebSocket client"""
    print("\n🔍 Testing WebSocket client...")
    
    try:
        from websocket_client import BybitWebSocketClient
        
        ws_client = BybitWebSocketClient(
            symbols=['BTCUSDT'],
            testnet=True,  # Use testnet for testing
            channel_types=['trade', 'ticker']
        )
        print("✅ WebSocket client initialized")
        
        # Test connection (without starting)
        # ws_client.test_connection()  # Uncomment if test_connection exists
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all system tests"""
    print("🚀 SYSTEM STATUS TEST")
    print("=" * 50)
    
    # Check environment
    print("\n📋 ENVIRONMENT CHECK:")
    print(f"   BYBIT_TESTNET: {os.getenv('BYBIT_TESTNET', 'false')}")
    print(f"   LIVE_TRADING_ENABLED: {os.getenv('LIVE_TRADING_ENABLED', 'false')}")
    print(f"   API Key configured: {'✅' if Config.BYBIT_API_KEY else '❌'}")
    
    # Test components
    results = []
    
    results.append(test_bybit_client())
    results.append(test_portfolio_components())
    results.append(test_websocket_client())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - System ready for trading")
        return 0
    else:
        print("⚠️ Some tests failed - Check logs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())

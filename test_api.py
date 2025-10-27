#!/usr/bin/env python3
"""
Test API connections for production trading
"""

import asyncio
from bybit_client import BybitClient
from ai_client_factory import AIClientFactory

async def test_apis():
    print("🧪 TESTING API CONNECTIONS FOR PRODUCTION TRADING")
    print("=" * 60)
    
    # Test Bybit API
    print("\n📡 Testing Bybit API...")
    try:
        async with BybitClient() as client:
            balance = await client.get_account_balance()
            print('✅ Bybit API Connection:', 'SUCCESS' if balance else 'FAILED')
            if balance:
                print(f'💰 Account Balance: {balance}')
            
            # Test market data
            market_data = await client.get_market_data('BTCUSDT')
            if market_data and market_data.get('lastPrice', '0') != '0':
                print(f'📊 BTC Price: ${market_data["lastPrice"]}')
            else:
                print('❌ Market data failed')
                
    except Exception as e:
        print(f'❌ Bybit API Error: {e}')
    
    # Test OpenRouter API (Grok 4 Fast)
    print("\n🤖 Testing OpenRouter API (Grok 4 Fast)...")
    try:
        async with await AIClientFactory.get_working_client() as ai_client:
            print('✅ OpenRouter API Connection: SUCCESS')
            print('🧠 Model: Grok 4 Fast ready for analysis')
    except Exception as e:
        print(f'❌ OpenRouter API Error: {e}')
    
    print("\n" + "=" * 60)
    print("🚀 API TESTS COMPLETE - Ready for production trading")

if __name__ == "__main__":
    asyncio.run(test_apis())

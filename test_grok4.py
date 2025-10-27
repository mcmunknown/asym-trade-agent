

#!/usr/bin/env python3
"""
Test Grok 4 Fast with tool calling capabilities
"""

import asyncio
from grok4_client import Grok4FastClient
from config import Config

async def test_grok4_tools():
    print("🤖 TESTING GROK 4 FAST WITH TOOL CALLING")
    print("=" * 60)
    
    try:
        # Initialize Grok 4 Fast client
        grok_client = Grok4FastClient(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)
        
        # Test with sample market data
        sample_market_data = {
            'symbol': 'BTCUSDT',
            'market_data': {
                'lastPrice': '95000',
                'price24hPcnt': '2.5',
                'volume24h': '2500000000'
            },
            'technical_indicators': {
                'rsi': 55,
                'ema_20': 94000,
                'ema_50': 92000,
                'atr': 5000
            }
        }
        
        print(f"📊 Testing analysis for BTCUSDT...")
        print(f"🧠 Model: {Config.OPENROUTER_MODEL}")
        print(f"🔧 Using native tool calling capabilities...")
        
        # Run analysis
        analysis = await grok_client.analyze_with_tools('BTCUSDT', sample_market_data)
        
        print(f"\n✅ ANALYSIS COMPLETE:")
        print(f"🎯 Signal: {analysis.get('signal', 'NONE')}")
        print(f"📊 Confidence: {analysis.get('confidence', 0)}%")
        print(f"🤖 Model: {analysis.get('model_used', 'unknown')}")
        print(f"🛠️ Tool Calls: {analysis.get('tool_calls_made', 0)}")
        
        if analysis.get('signal') == 'BUY':
            print(f"💰 Activation Price: ${analysis.get('activation_price', 'N/A')}")
            print(f"📈 Thesis: {analysis.get('thesis_summary', 'N/A')[:100]}...")
        
        print(f"\n🚀 Grok 4 Fast tool calling working successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Grok 4 Fast: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_grok4_tools())


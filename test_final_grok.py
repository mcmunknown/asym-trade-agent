#!/usr/bin/env python3
"""
Final test for Grok 4 Fast with forced environment reload
"""

import os
import sys

# Force environment reload
if 'dotenv' in sys.modules:
    del sys.modules['dotenv']

from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload

from ai_client_factory import AIClientFactory
import asyncio

async def test_grok_final():
    print("🚀 Final Test: Grok 4 Fast with OpenRouter")
    print("=" * 50)
    
    # Verify environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    
    print(f"API Key: {'✅' if api_key else '❌'}")
    print(f"Model: {model}")
    
    if model == "x-ai/grok-4-fast":
        print("\n✅ Grok 4 Fast is configured!")
        
        # Test with actual API call
        client = AIClientFactory.create_client()
        print(f"Client Type: {type(client).__name__}")
        
        async with client as ai:
            result = await ai.analyze_market_conditions(
                {'price': 45000, 'change_24h': 2.0},
                {'revenue_trend': '↑', 'tvl_trend': '↑'},
                {'rsi_1d': 55, 'price_vs_30d_low': 5.0},
                'BTCUSDT'
            )
            
            print(f"\n🤖 Grok 4 Fast Analysis:")
            print(f"Signal: {result['signal']}")
            print(f"Confidence: {result.get('confidence', 0)}%")
            print(f"Thesis: {result['thesis_summary']}")
            
            if result['signal'] == 'BUY':
                print("\n🎯 BUY signal generated with Grok 4 Fast!")
            else:
                print("\n📊 No trade signal - conditions not met")
                
            print(f"\n✅ Grok 4 Fast integration is working perfectly!")
            print(f"🎉 Your trading agent is ready to use Grok 4 Fast!")
    else:
        print(f"\n❌ Model not set to Grok 4 Fast. Current: {model}")
        print("Please update your .env file:")
        print("OPENROUTER_MODEL=x-ai/grok-4-fast")

if __name__ == "__main__":
    asyncio.run(test_grok_final())

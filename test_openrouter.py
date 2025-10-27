#!/usr/bin/env python3
"""
Test OpenRouter GPT-5 Integration
"""

import asyncio
from glm_client import GLMClient

async def test_openrouter_system():
    print('🚀 Testing OpenRouter GPT-5 Integration...')
    print('=' * 60)

    async with GLMClient() as ai:
        print('✅ AI Client initialized')

        # Test bullish BTC scenario
        test_data = {
            'market_data': {'price': 43500, 'change_24h': 2.8, 'timestamp': '2024-01-01'},
            'fundamentals': {'revenue_trend': '↑', 'tvl_trend': '↑', 'wallet_accumulation': 'Strong'},
            'technical_indicators': {'rsi_1d': 58, 'price_vs_30d_low': 7.5, 'ema_aligned': True}
        }

        print('📊 Analyzing BTCUSDT with GPT-5...')
        result = await ai.analyze_market_conditions(
            test_data['market_data'],
            test_data['fundamentals'],
            test_data['technical_indicators'],
            'BTCUSDT'
        )

        print(f'🤖 AI Analysis Complete!')
        print(f'Signal: {result["signal"]}')
        print(f'Confidence: {result.get("confidence", 0)}%')
        print(f'Macro Analysis: {result.get("macro_tailwind", "N/A")}')
        print(f'Technical Setup: {result.get("technical_setup", "N/A")}')
        print(f'Thesis: {result["thesis_summary"]}')

        if result['signal'] == 'BUY':
            print('')
            print('🎯 BUY SIGNAL DETECTED!')
            price = test_data['market_data']['price']
            print(f'   📍 Entry: ${price:,.2f}')
            print(f'   🎯 Target: {result.get("activation_price", "N/A")}')
            print(f'   🛑 Stop Loss: {result.get("invalidation_level", "N/A")}')
            print(f'   📊 Confidence: {result.get("confidence", 0)}%')
            print(f'   🔄 Trailing Stop: {result.get("trailing_stop_pct", "N/A")}%')
            print('')
            print('🚀 READY TO EXECUTE LIVE TRADE!')
        else:
            print('')
            print('📊 No trade signal - conditions not met for entry')

if __name__ == "__main__":
    asyncio.run(test_openrouter_system())
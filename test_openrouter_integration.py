#!/usr/bin/env python3
"""
Test OpenRouter GPT-5 Integration with prompt.md framework
"""

import asyncio
import logging
from ai_client_factory import AIClientFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openrouter_integration():
    """Test the new OpenRouter Grok 4 Fast integration"""
    print('🚀 Testing OpenRouter Grok 4 Fast Integration with prompt.md...')
    print('=' * 60)

    try:
        ai_client = await AIClientFactory.get_working_client()
        async with ai_client as ai:
            print('✅ Trading AI Client initialized')

            # Test with enhanced market data structure
            test_data = {
                'market_data': {
                    'price': 43500.00,
                    'change_24h': 2.8,
                    'bybit_volume': 250000000,  # $250M
                    'bybit_funding_rate': -0.01,
                    'bybit_open_interest': 15000000,
                    'liquidation_level': 41000,
                    'spread_percentage': 0.05
                },
                'fundamentals': {
                    'treasury_accumulation': 'Strong',
                    'revenue_trend': '↑',
                    'tvl_trend': '↑',
                    'developer_activity': 'High',
                    'tokenomics_changes': 'Burn mechanism active',
                    'upcoming_events': 'None',
                    'wallet_accumulation': 'Strong'
                },
                'technical_indicators': {
                    'rsi_4h': 58,
                    'rsi_1d': 62,
                    'rsi_1w': 55,
                    'price_vs_30d_low': 7.5,
                    'atr_30d': 2800,
                    'ema_20_4h': True,
                    'ema_20_1d': True,
                    'ema_20_1w': True,
                    'ema_50_4h': True,
                    'ema_50_1d': True,
                    'ema_50_1w': True,
                    'volume_3d_anomaly': True,
                    'volume_7d_anomaly': False
                },
                'symbol': 'BTCUSDT'
            }

            print('📊 Analyzing BTCUSDT with Grok 4 Fast + prompt.md framework...')
            result = await ai.analyze_market_conditions(
                test_data['market_data'],
                test_data['fundamentals'],
                test_data['technical_indicators'],
                test_data['symbol']
            )

            print(f'🤖 OpenRouter Grok 4 Fast Analysis Complete!')
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
                print(f'   💰 Risk/Reward: {result.get("risk_reward_ratio", "N/A")}')
                print('')
                print('🚀 READY TO EXECUTE LIVE TRADE!')
            else:
                print('')
                print('📊 No trade signal - Conditions not met per prompt.md criteria')
                print(f'Reason: {result.get("thesis_summary", "N/A")}')

            return result

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')
        return None

if __name__ == "__main__":
    asyncio.run(test_openrouter_integration())
#!/usr/bin/env python3
"""
Quick test for OpenRouter Grok 4 Fast connection
"""

import asyncio
import os
from dotenv import load_dotenv
import aiohttp
import json

load_dotenv()

async def test_grok_connection():
    """Test basic OpenRouter Grok 4 Fast API connection"""
    print('🚀 Testing OpenRouter Grok 4 Fast Connection...')
    print('=' * 50)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/asym-trade-agent',
        'X-Title': 'Asymmetric Trading Agent'
    }

    payload = {
        'model': 'x-ai/grok-4-fast',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a trading analysis AI.'
            },
            {
                'role': 'user',
                'content': 'Analyze BTCUSDT for a long position. Return JSON with signal (BUY/NONE) and confidence (0-100).'
            }
        ],
        'max_tokens': 200,
        'temperature': 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        print('✅ Grok 4 Fast connection successful!')
                        print(f'📝 Response: {content[:200]}...')
                        return True
                    else:
                        print('❌ Invalid response format')
                        return False
                else:
                    error_text = await response.text()
                    print(f'❌ API Error ({response.status}): {error_text}')
                    return False

    except asyncio.TimeoutError:
        print('❌ Connection timeout')
        return False
    except Exception as e:
        print(f'❌ Connection error: {str(e)}')
        return False

if __name__ == "__main__":
    result = asyncio.run(test_grok_connection())
    if result:
        print('\n🎉 Grok 4 Fast is ready for trading analysis!')
    else:
        print('\n❌ Grok 4 Fast connection failed - check API key or model availability')

#!/usr/bin/env python3
"""
Direct OpenRouter API Test - Prove GPT-5 Integration Works
"""

import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

async def test_openrouter_direct():
    """Test OpenRouter API directly to prove it works"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = "openai/gpt-5"
    
    if not api_key:
        print("❌ No OpenRouter API key found!")
        return False
        
    print("🧪 Testing OpenRouter GPT-5 Direct API Call...")
    print(f"📝 API Key: {api_key[:20]}...")
    print(f"🤖 Model: {model}")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/asym-trade-agent',
        'X-Title': 'Asymmetric Trading Agent Test'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': 'You are a crypto trading analyst. Respond with exactly this JSON: {"signal": "BUY", "confidence": 85}'
            },
            {
                'role': 'user', 
                'content': 'Analyze BTC price at $43,500. Should we buy?'
            }
        ],
        'max_tokens': 100,
        'temperature': 0.1
    }
    
    try:
        print("🚀 Making API call to OpenRouter...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions", 
                headers=headers, 
                json=payload
            ) as response:
                
                print(f"📡 Response Status: {response.status}")
                
                data = await response.json()
                
                if 'error' in data:
                    print(f"❌ API Error: {data['error']['message']}")
                    return False
                    
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    print(f"✅ SUCCESS! GPT-5 Response:")
                    print(f"📄 Content: {content}")
                    
                    # Try to parse as JSON
                    try:
                        result = json.loads(content)
                        print(f"📊 Parsed JSON: {result}")
                        print(f"🎯 Signal: {result.get('signal', 'N/A')}")
                        print(f"📈 Confidence: {result.get('confidence', 'N/A')}")
                        return True
                    except:
                        print(f"⚠️  Could not parse as JSON, but API worked!")
                        return True
                else:
                    print(f"❌ No choices in response: {data}")
                    return False
                    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openrouter_direct())
    if success:
        print("\n🎉 PROOF: OpenRouter GPT-5 integration IS WORKING!")
    else:
        print("\n❌ OpenRouter GPT-5 integration FAILED!")
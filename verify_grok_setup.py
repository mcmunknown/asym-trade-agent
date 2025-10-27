#!/usr/bin/env python3
"""
Verify Grok 4 Fast setup with OpenRouter
"""

import os
from dotenv import load_dotenv
from ai_client_factory import AIClientFactory
from config import Config

def main():
    print("🔍 Verifying Grok 4 Fast Setup...")
    print("=" * 50)
    
    # Reload environment
    load_dotenv()
    
    # Check environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    
    print(f"API Key: {'✅ Configured' if api_key else '❌ Missing'}")
    print(f"Model: {model}")
    
    # Check Config class
    print(f"Config OPENROUTER_MODEL: {Config.OPENROUTER_MODEL}")
    
    # Create client to verify
    if Config.OPENROUTER_API_KEY:
        print("\n🤖 Creating AI Client...")
        client = AIClientFactory.create_client()
        print(f"Client Type: {type(client).__name__}")
        if hasattr(client, 'model'):
            print(f"Client Model: {client.model}")
        
        if model == "x-ai/grok-4-fast":
            print("\n✅ Grok 4 Fast is correctly configured!")
            print("🎉 Your asymmetric trading agent is ready with Grok 4 Fast!")
        else:
            print(f"\n⚠️  Model is set to: {model}")
            print("To use Grok 4 Fast, update your .env file:")
            print("OPENROUTER_MODEL=x-ai/grok-4-fast")
    else:
        print("\n❌ OpenRouter API key not configured")

if __name__ == "__main__":
    main()

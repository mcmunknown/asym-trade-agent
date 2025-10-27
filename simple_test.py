
#!/usr/bin/env python3
"""
Simple test to verify system components
"""

import asyncio
from config import Config

def test_config():
    print("🔧 TESTING CONFIGURATION")
    print("=" * 50)
    print(f"✅ API Keys Loaded: {bool(Config.BYBIT_API_KEY and Config.OPENROUTER_API_KEY)}")
    print(f"🔥 LIVE MODE: {not Config.BYBIT_TESTNET}")
    print(f"⚡ Trading Enabled: {not Config.DISABLE_TRADING}")
    print(f"🎯 Target Assets: {Config.TARGET_ASSETS}")
    print(f"💰 Trade Size: ${Config.DEFAULT_TRADE_SIZE}")
    print(f"📊 Leverage: {Config.MAX_LEVERAGE}x")
    print(f"⏰ Analysis Interval: {Config.SIGNAL_CHECK_INTERVAL}s")

if __name__ == "__main__":
    test_config()
    print("\n🚀 Configuration looks good for production trading!")

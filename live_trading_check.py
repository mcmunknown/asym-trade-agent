#!/usr/bin/env python3
"""
Live Trading System Status Check
"""

import os
from config import Config

def check_live_trading_status():
    print("=" * 80)
    print("🚀 LIVE TRADING SYSTEM STATUS CHECK")
    print("=" * 80)

    print(f"📊 Bybit Configuration:")
    print(f"   Testnet Mode: {Config.BYBIT_TESTNET}")
    print(f"   Base URL: {Config.BYBIT_BASE_URL}")
    print(f"   API Key: {Config.BYBIT_API_KEY[:10]}..." if Config.BYBIT_API_KEY else "   ❌ MISSING")

    print(f"\n🧠 GLM Configuration:")
    print(f"   API Key: {Config.GLM_API_KEY[:10]}..." if Config.GLM_API_KEY else "   ❌ MISSING")
    print(f"   Model: {Config.GLM_MODEL}")

    print(f"\n⚙️ Trading Parameters:")
    print(f"   Trade Size: ${Config.DEFAULT_TRADE_SIZE}")
    print(f"   Max Leverage: {Config.MAX_LEVERAGE}x")
    print(f"   Target Assets: {len(Config.TARGET_ASSETS)} pairs")
    print(f"   Data Interval: {Config.DATA_COLLECTION_INTERVAL}s")

    print(f"\n⚠️  WARNINGS:")

    if not Config.BYBIT_TESTNET:
        print(f"   🔴 LIVE TRADING ENABLED - REAL MONEY AT RISK!")
    else:
        print(f"   🟢 Testnet Mode - Safe for testing")

    if not Config.GLM_API_KEY:
        print(f"   ❌ GLM API Key missing")
    else:
        print(f"   ⚠️  GLM API may need balance funding")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Fund GLM API at https://z.ai")
    print(f"   2. Test with: python test_system.py")
    print(f"   3. Start with: python main.py")

    print("=" * 80)

if __name__ == "__main__":
    check_live_trading_status()
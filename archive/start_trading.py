#!/usr/bin/env python3
"""
🚀 START LIVE TRADING - 8 ASSETS
================================

Quick start script for Anne's calculus trading system
with 8 cryptocurrency assets and C++ acceleration.
"""

import os
import sys

def main():
    print("🚀 ANNE'S CALCULUS TRADING SYSTEM")
    print("================================")
    print("📊 Starting 8-Asset Live Trading")
    print()
    print("⚠️ WARNING: This will execute REAL trades!")
    print("⚠️ Ensure you understand the risks!")
    print()
    
    # Check virtual environment
    if not os.path.exists("venv"):
        print("❌ Virtual environment not found")
        print("Please run: source venv/bin/activate")
        sys.exit(1)
    
    # Safety confirmation
    confirm = input("Type 'LIVE' to start trading: ").strip()
    if confirm.upper() != 'LIVE':
        print("❌ Trading not started - confirmation failed")
        sys.exit(0)
    
    print("🚀 Starting live trading...")
    print("Press Ctrl+C to stop")
    print()
    
    # Start live trading
    os.system("source venv/bin/activate && python3 live_calculus_trader.py")

if __name__ == "__main__":
    main()

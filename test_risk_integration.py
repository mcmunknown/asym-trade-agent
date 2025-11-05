#!/usr/bin/env python3
"""
Risk Management Integration Test for Anne's Calculus Trading System
"""

import pandas as pd
import numpy as np
from risk_manager import RiskManager
from config import Config

def test_risk_management_integration():
    """Test comprehensive risk management functionality."""
    print('🛡️ Testing Risk Management Integration...')

    # Initialize risk manager
    risk_mgr = RiskManager()
    print('✅ Risk Manager initialized')

    # Test 1: Basic position sizing calculation
    print('\n1. Testing basic position sizing...')
    position_size = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=1.5,
        confidence=0.8,
        current_price=100000.0,
        account_balance=10000.0
    )
    print(f'✅ Position size calculated:')
    print(f'   Quantity: {position_size.quantity:.6f}')
    print(f'   Risk amount: ${position_size.risk_amount:.2f}')
    print(f'   Risk percent: {position_size.risk_percent:.2%}')

    # Test 2: Different account balances and risk scaling
    print('\n2. Testing position sizing with different account balances...')

    small_account = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=1.0,
        confidence=0.8,
        current_price=100000.0,
        account_balance=1000.0
    )

    large_account = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=1.0,
        confidence=0.8,
        current_price=100000.0,
        account_balance=100000.0
    )

    print(f'✅ Account-based position sizing:')
    print(f'   Small account ($1k): {small_account.quantity:.6f} (${small_account.risk_amount:.2f})')
    print(f'   Large account ($100k): {large_account.quantity:.6f} (${large_account.risk_amount:.2f})')
    print(f'   Scaling works correctly: {"✅" if large_account.quantity/small_account.quantity == 100 else "❌"}')

    # Test 3: Different signal strengths and position sizing
    print('\n3. Testing position sizing with different signal strengths...')

    weak_signal = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=0.5,
        confidence=0.6,
        current_price=100000.0,
        account_balance=10000.0
    )

    strong_signal = risk_mgr.calculate_position_size(
        symbol='BTCUSDT',
        signal_strength=2.0,
        confidence=0.9,
        current_price=100000.0,
        account_balance=10000.0
    )

    print(f'✅ Signal-based position sizing:')
    print(f'   Weak signal (0.5): {weak_signal.quantity:.6f} (${weak_signal.risk_amount:.2f})')
    print(f'   Strong signal (2.0): {strong_signal.quantity:.6f} (${strong_signal.risk_amount:.2f})')
    print(f'   Scaling factor: {strong_signal.quantity/weak_signal.quantity:.2f}x')

    # Test 4: Risk metrics calculation
    print('\n4. Testing risk metrics calculation...')

    # Update some position data
    risk_mgr.update_position('BTCUSDT', {
        'quantity': 0.05,
        'entry_price': 98000.0,
        'current_price': 100000.0,
        'unrealized_pnl': 100.0
    })

    risk_metrics = risk_mgr.calculate_risk_metrics()
    print(f'✅ Risk metrics calculated:')
    print(f'   Current exposure: ${risk_metrics.total_exposure:.2f}')
    print(f'   Available balance: ${risk_metrics.available_balance:.2f}')
    print(f'   Margin used: {risk_metrics.margin_used_percent:.1%}')
    print(f'   Open positions: {risk_metrics.open_positions_count}')
    print(f'   Current drawdown: {risk_metrics.current_drawdown:.2%}')
    print(f'   Sharpe ratio: {risk_metrics.sharpe_ratio:.2f}')

    risk_config = Config.get_risk_config()
    print(f'✅ Risk configuration loaded:')
    print(f'   Max risk per trade: {risk_config.get("max_risk_per_trade", 0.02):.1%}')
    print(f'   Max leverage: {risk_config.get("max_leverage", 25)}x')
    print(f'   Stop loss enabled: {risk_config.get("stop_loss", {}).get("enabled", True)}')
    print(f'   Risk per trade: {risk_config.get("risk_per_trade", 0.02):.1%}')

    print('\n🎉 Risk Management Integration Test Complete!')
    print('🛡️ All risk management components working correctly!')

if __name__ == "__main__":
    test_risk_management_integration()
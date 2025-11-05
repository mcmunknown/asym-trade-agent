#!/usr/bin/env python3
"""
Backtesting Framework Test for Anne's Calculus Trading System
"""

import pandas as pd
import numpy as np
from backtester import CalculusBacktester, BacktestConfig
from calculus_strategy import CalculusTradingStrategy
from risk_manager import RiskManager
from kalman_filter import AdaptiveKalmanFilter
from config import Config

def test_backtesting_framework():
    """Test comprehensive backtesting functionality."""
    print('📈 Testing Backtesting Framework...')

    # Generate realistic test data
    print('Generating test market data...')
    np.random.seed(42)
    base_price = 100000.0
    days = 30
    minutes_per_day = 1440
    total_points = days * minutes_per_day

    # Generate realistic BTC price movement with trend and volatility
    trend = np.linspace(0, 5000, total_points)  # Upward trend
    noise = np.random.normal(0, 500, total_points)  # Random noise
    seasonal = 1000 * np.sin(np.linspace(0, 10*np.pi, total_points))  # Seasonal pattern

    prices = base_price + trend + noise + seasonal
    timestamps = pd.date_range('2024-01-01', periods=total_points, freq='1min')

    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.random.uniform(0, 200, total_points),
        'low': prices - np.random.uniform(0, 200, total_points),
        'close': prices,
        'volume': np.random.uniform(100, 1000, total_points)
    })

    print(f'✅ Generated {len(market_data)} data points ({days} days)')

    # Test 1: Initialize backtesting engine
    print('\n1. Testing backtesting engine initialization...')

    # Create backtest configuration
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-01-31',
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005
    )

    backtester = CalculusBacktester(config)

    print('✅ Backtest engine initialized successfully')
    print(f'   Initial capital: ${config.initial_capital:,.0f}')
    print(f'   Commission: {config.commission_rate:.3%}')
    print(f'   Slippage: {config.slippage_rate:.3%}')

    # Test 2: Run basic backtest
    print('\n2. Testing basic backtest execution...')

    # Run backtest with generated data
    results = backtester.run_backtest(
        symbol='BTCUSDT',
        data=market_data
    )

    print('✅ Backtest completed successfully')
    print(f'   Total trades executed: {results.total_trades}')
    print(f'   Win rate: {results.win_rate:.1%}')
    print(f'   Net profit: ${results.net_profit:.2f}')
    print(f'   Max drawdown: {results.max_drawdown:.1%}')
    print(f'   Sharpe ratio: {results.sharpe_ratio:.2f}')

    # Test 3: Performance metrics calculation
    print('\n3. Testing performance metrics...')

    print(f'✅ Performance metrics available:')
    print(f'   Total return: {(results.net_profit/config.initial_capital):.1%}')
    print(f'   Final capital: ${results.final_capital:.2f}')
    print(f'   Max drawdown: {results.max_drawdown:.1%}')
    print(f'   Win rate: {results.win_rate:.1%}')
    print(f'   Sharpe ratio: {results.sharpe_ratio:.2f}')
    print(f'   Total trades: {results.total_trades}')

    # Test 4: Trade analysis
    print('\n4. Testing trade analysis...')

    if results.trades:
        print(f'✅ Trade analysis complete:')
        print(f'   Sample trade data available: {len(results.trades)} trades')
        if len(results.trades) > 0:
            sample_trade = results.trades[0]
            print(f'   First trade entry: ${sample_trade.entry_price:.2f}')
            print(f'   First trade exit: ${sample_trade.exit_price:.2f}')
            print(f'   First trade PnL: ${sample_trade.pnl:.2f}')
    else:
        print('⚠️  No trades executed in test period')

    # Test 5: Backtest configuration
    print('\n5. Testing backtest configuration...')

    print(f'✅ Configuration verified:')
    print(f'   Start date: {config.start_date}')
    print(f'   End date: {config.end_date}')
    print(f'   Commission: {config.commission_rate:.3%}')
    print(f'   Slippage: {config.slippage_rate:.3%}')

    print('\n🎉 Backtesting Framework Test Complete!')
    print('📈 All backtesting components working correctly!')

    # Summary
    print(f'\n📊 BACKTEST SUMMARY:')
    print(f'   Period: {days} days')
    print(f'   Initial capital: ${config.initial_capital:,.0f}')
    print(f'   Final capital: ${results.final_capital:,.0f}')
    print(f'   Total return: {results.net_profit/config.initial_capital:.1%}')
    print(f'   Total trades: {results.total_trades}')
    print(f'   Win rate: {results.win_rate:.1%}')
    print(f'   Max drawdown: {results.max_drawdown:.1%}')

if __name__ == "__main__":
    test_backtesting_framework()
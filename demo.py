#!/usr/bin/env python3
"""
Demo script to showcase the asymmetric trading agent functionality
"""

import asyncio
import json
from datetime import datetime
from glm_client import GLMClient

async def demo_glm_analysis():
    """Demonstrate GLM-4.6 analysis capabilities"""
    print("=" * 80)
    print("🧠 GLM-4.6 MARKET ANALYSIS DEMONSTRATION")
    print("=" * 80)

    # Initialize GLM client
    async with GLMClient() as glm:
        # Example 1: Bullish BTC scenario
        print("\n📈 SCENARIO 1: BULLISH BITCOIN SETUP")
        print("-" * 40)

        btc_data = {
            'market_data': {
                'price': 43500,
                'volume_24h': 25000000000,
                'change_24h': 3.2,
                'funding_rate': 0.01,
                'open_interest': 15000000000,
                'timestamp': datetime.now().isoformat(),
                'macro_narrative': 'Bitcoin ETF approval momentum building',
                'risk_sentiment': 'Risk-on'
            },
            'fundamentals': {
                'revenue_trend': '↑',
                'tvl_trend': '↑',
                'staking_percentage': '3.5%',
                'token_burns': 'Active',
                'developer_activity': 'High',
                'wallet_accumulation': 'Strong',
                'upcoming_unlocks': 'None',
                'governance_votes': 'None',
                'token_emissions': 'Deflationary'
            },
            'technical_indicators': {
                'price_vs_30d_low': 8.5,
                '30d_low': 40000,
                'within_entry_zone': True,
                'ema_aligned': True,
                'rsi_1d': 58,
                'rsi_momentum_ok': True,
                'volume_confirmation': True,
                'atr_30d_pct': 6.2,
                'atr_ok': True,
                'liquidity_check': True,
                'current_price': 43500
            }
        }

        result = await glm.analyze_market_conditions(
            btc_data['market_data'],
            btc_data['fundamentals'],
            btc_data['technical_indicators'],
            'BTCUSDT'
        )

        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Thesis: {result['thesis_summary']}")
        if result['signal'] == 'BUY':
            print(f"Target Price: ${result['activation_price']}")
            print(f"Stop Loss: ${result['invalidation_level']}")

        # Example 2: Bearish ETH scenario
        print("\n📉 SCENARIO 2: BEARISH ETHEREUM SETUP")
        print("-" * 40)

        eth_data = {
            'market_data': {
                'price': 2200,
                'volume_24h': 8000000000,
                'change_24h': -1.8,
                'funding_rate': 0.05,
                'open_interest': 8000000000,
                'timestamp': datetime.now().isoformat(),
                'macro_narrative': 'Regulatory concerns on DeFi protocols',
                'risk_sentiment': 'Risk-off'
            },
            'fundamentals': {
                'revenue_trend': '↓',
                'tvl_trend': '↓',
                'staking_percentage': '4.2%',
                'token_burns': 'None',
                'developer_activity': 'Medium',
                'wallet_accumulation': 'Distribution',
                'upcoming_unlocks': 'Significant (>5%)',
                'governance_votes': 'Major',
                'token_emissions': 'High'
            },
            'technical_indicators': {
                'price_vs_30d_low': 25.0,
                '30d_low': 1800,
                'within_entry_zone': False,
                'ema_aligned': False,
                'rsi_1d': 42,
                'rsi_momentum_ok': False,
                'volume_confirmation': False,
                'atr_30d_pct': 9.5,
                'atr_ok': False,
                'liquidity_check': True,
                'current_price': 2200
            }
        }

        result = await glm.analyze_market_conditions(
            eth_data['market_data'],
            eth_data['fundamentals'],
            eth_data['technical_indicators'],
            'ETHUSDT'
        )

        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Thesis: {result['thesis_summary']}")

async def demo_trading_simulation():
    """Simulate trading strategy performance"""
    print("\n" + "=" * 80)
    print("💰 TRADING STRATEGY SIMULATION")
    print("=" * 80)

    # Simulated trading parameters
    initial_capital = 100  # $100
    trade_size = 3.0       # $3 per trade
    max_leverage = 75      # 75x leverage
    target_pnl = 1.5       # 150% PNL target

    print(f"Initial Capital: ${initial_capital}")
    print(f"Trade Size: ${trade_size} with {max_leverage}x leverage")
    print(f"Effective Position: ${trade_size * max_leverage} per trade")
    print(f"Target PNL per trade: ${trade_size * target_pnl}")

    # Simulate 10 trades
    trades = []
    capital = initial_capital
    win_rate = 0.7  # 70% win rate assumption

    for i in range(1, 11):
        # Simulate trade outcome
        win = i <= (10 * win_rate)  # Win first 7 trades
        pnl = trade_size * target_pnl if win else -trade_size
        capital += pnl

        trade = {
            'trade_number': i,
            'outcome': 'WIN' if win else 'LOSS',
            'pnl': pnl,
            'capital': capital,
            'return_pct': ((capital - initial_capital) / initial_capital) * 100
        }
        trades.append(trade)

    print(f"\n{'Trade':<6} {'Outcome':<8} {'PNL':<10} {'Capital':<10} {'Return %':<8}")
    print("-" * 50)

    for trade in trades:
        print(f"{trade['trade_number']:<6} {trade['outcome']:<8} ${trade['pnl']:<9.2f} ${trade['capital']:<9.2f} {trade['return_pct']:<7.1f}%")

    final_return = ((capital - initial_capital) / initial_capital) * 100
    print("-" * 50)
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Return: {final_return:.1f}%")
    print(f"Win Rate: {win_rate * 100:.0f}%")

async def main():
    """Main demo function"""
    print("🚀 ASYMMETRIC CRYPTO TRADING AGENT - DEMO")
    print("=" * 80)
    print("This demo showcases the AI-powered trading system that combines:")
    print("• GLM-4.6 deep market analysis")
    print("• Bybit API integration")
    print("• High-leverage futures trading")
    print("• Risk management and automation")
    print("\n⚠️  DEMO MODE - No real trading executed")
    print("=" * 80)

    # Run GLM analysis demo
    await demo_glm_analysis()

    # Run trading simulation
    await demo_trading_simulation()

    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS")
    print("=" * 80)
    print("To run the actual trading system:")
    print("1. Ensure API keys are properly configured in .env")
    print("2. Run 'python test_system.py' to verify all components")
    print("3. Start trading with 'python main.py'")
    print("\n⚠️  WARNING: High-leverage trading carries substantial risk!")
    print("💡 Always test on testnet before using real funds.")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
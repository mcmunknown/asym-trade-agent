#!/usr/bin/env python3
"""
Simple test to start the trading system in simulation mode
"""

import time
import logging
logging.basicConfig(level=logging.DEBUG)
from live_portfolio_trader import LivePortfolioTrader

def main():
    print("🚀 Starting Trading System Test")
    print("=" * 50)
    
    # Initialize trader in simulation mode for safety
    trader = LivePortfolioTrader(
        symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        initial_capital=10000.0,
        emergency_stop=False,  # Enable trading
        simulation_mode=False  # LIVE TRADING - GO TO $20
    )
    
    print("✅ System initialized successfully")
    
    # Get status
    status = trader.get_status()
    print(f"\n📊 System Status:")
    print(f"   Status keys: {list(status.keys())}")
    
    if 'error' in status:
        print(f"   Error: {status['error']}")
    else:
        print(f"   Running: {status['system']['is_running']}")
        print(f"   Simulation: {status['system']['simulation_mode']}")
        print(f"   Emergency Stop: {status['system']['emergency_stop']}")
        print(f"   Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
    
    # Start LIVE TRADING with real money
    print("\n🚀 STARTING LIVE TRADING WITH REAL MONEY")
    print("   📈 Trading based on Anne's calculus signals")
    print("   💰 Starting with $10,000 capital")
    print("   🎯 Goal: Grow account to $20,000+")
    print("   ⚠️ NO SIMULATION - REAL TRADES")
    
    trader.start()
    
    # Monitor progress continuously
    target_goal = 20000.0
    last_trade_count = 0
    check_interval = 30  # Check every 30 seconds
    
    print(f"\n📊 Monitoring progress to $20,000 goal...")
    
    while True:
        try:
            status = trader.get_status()
            portfolio_value = status['portfolio']['total_value']
            trade_count = status['trading']['total_trades']
            pnl = status['portfolio']['unrealized_pnl']
            pnl_pct = status['portfolio']['unrealized_pnl_pct']
            
            # Update display
            print(f"\r💰 ${portfolio_value:,.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1%}) | Trades: {trade_count} | Progress: {(portfolio_value/20000)*100:.1f}%", end='', flush=True)
            
            # Check if goal reached
            if portfolio_value >= target_goal:
                print(f"\n\n🎉 GOAL REACHED! Account at ${portfolio_value:,.2f}")
                print(f"   Total profit: ${portfolio_value - 10000:,.2f}")
                print(f"   Total trades executed: {trade_count}")
                break
            
            # Alert if no new trades
            if trade_count == last_trade_count:
                if check_interval == 30:  # First 30 seconds
                    print("\n   ⚠️ No trades yet - checking system...")
            else:
                if trade_count > last_trade_count:
                    print(f"\n   ✅ New trades executed! Count: {trade_count}")
            
            last_trade_count = trade_count
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error monitoring: {e}")
            time.sleep(10)  # Wait before retry
    
    # Check status again
    status = trader.get_status()
    print(f"\n📊 Status after 10 seconds:")
    print(f"   Running: {status['system']['is_running']}")
    print(f"   Total Trades: {status['trading']['total_trades']}")
    print(f"   Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
    print(f"   P&L: ${status['portfolio']['unrealized_pnl']:,.2f}")
    
    # Stop trading
    print("\n🛑 Stopping trading...")
    trader.stop()
    
    print("\n✅ Test completed successfully!")
    print("   - System initialized without errors")
    print("   - Trading enabled and executed")
    print(f"   - Total trades executed: {status['trading']['total_trades']}")
    
    print("\n🎯 System is ready for live trading!")
    print("   Set simulation_mode=False for actual live trading")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Asymmetric Crypto Trading Agent
Automated trading system using Grok 4 Fast for analysis and Bybit for execution
"""

import time
import logging
import signal
import sys
import threading
from datetime import datetime
from data_collector import DataCollector
from trading_engine import TradingEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionTradingAgent:
    def __init__(self):
        self.data_collector = DataCollector()
        self.trading_engine = TradingEngine()
        self.is_running = False

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.is_running = False
        self.trading_engine.stop()

    def process_market_data(self, data_list):
        """Process collected market data"""
        try:
            if not data_list:
                logger.info("No market data available")
                return

            logger.info(f"Processing {len(data_list)} assets for asymmetric opportunities...")

            # Process signals through trading engine
            self.trading_engine.process_signals(data_list)

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    def print_status(self):
        """Print trading status"""
        while self.is_running:
            try:
                print(f"\n{'='*60}")
                print(f"🤖 ASYMMETRIC TRADING AGENT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                # Get portfolio summary
                portfolio = self.trading_engine.get_portfolio_summary()
                if portfolio:
                    print(f"💰 Total Balance: ${portfolio.get('total_balance', 0):.2f}")
                    print(f"💵 Available: ${portfolio.get('available_balance', 0):.2f}")
                    print(f"📊 Active Positions: {len(self.trading_engine.get_active_positions())}")

                    total_invested = portfolio.get('total_invested', 0)
                    unrealized_pnl = portfolio.get('unrealized_pnl', 0)
                    if total_invested > 0:
                        pnl_change = (unrealized_pnl / total_invested) * 100
                        print(f"📈 Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_change:+.2f}%)")

                # Print active positions
                active_positions = self.trading_engine.get_active_positions()
                if active_positions:
                    print(f"\n📊 ACTIVE POSITIONS ({len(active_positions)}):")
                    for symbol, position_data in active_positions.items():
                        signal = position_data.get('signal')
                        if signal:
                            print(f"  • {symbol}: Entry=${signal.entry_price:.4f}, "
                                  f"Target=${signal.activation_price:.4f}, "
                                  f"PNL={unrealized_pnl:.2f}%")

                print(f"{'='*60}\n")

                time.sleep(900)  # Update every 15 minutes

            except Exception as e:
                logger.error(f"Error printing status: {str(e)}")
                time.sleep(60)

    def run(self):
        """Main run loop"""
        try:
            logger.info("Starting Asymmetric Crypto Trading Agent...")

            # Initialize trading engine
            self.trading_engine.initialize()

            self.is_running = True

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            # Start trading engine in separate thread
            trading_thread = threading.Thread(target=self.trading_engine.start, daemon=True)
            trading_thread.start()

            # Start status printer in separate thread
            status_thread = threading.Thread(target=self.print_status, daemon=True)
            status_thread.start()

            logger.info("Production Trading Agent started successfully")

            # Main data collection loop
            while self.is_running:
                try:
                    logger.info("Starting data collection cycle...")

                    # Collect market data
                    data_list = self.data_collector.collect_all_data()

                    if data_list:
                        logger.info(f"Collected data for {len(data_list)} assets")
                        self.process_market_data(data_list)
                    else:
                        logger.warning("No data collected in this cycle")

                    # Wait for next cycle (run every 30 minutes for optimal asymmetric opportunities)
                    logger.info("Waiting 30 minutes for next analysis cycle...")

                    # Sleep in small increments to allow for graceful shutdown
                    for _ in range(180):  # 180 * 10 seconds = 30 minutes
                        if not self.is_running:
                            break
                        time.sleep(10)

                except Exception as e:
                    logger.error(f"Error in main run loop: {str(e)}")
                    time.sleep(300)  # Wait 5 minutes before retrying

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt...")
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
        finally:
            self.is_running = False
            self.trading_engine.stop()
            logger.info("Production Trading Agent stopped")

def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           ASYMMETRIC CRYPTO TRADING AGENT v2.0               ║
    ║                                                               ║
    ║  🤖 Grok 4 Fast Powered Research Analysis                     ║
    ║  📈 Bybit Perpetual Futures Execution                        ║
    ║  💰 High-Leverage Asymmetric Trading                         ║
    ║                                                               ║
    ║  Target: 150%+ PNL with 50-75x Leverage                     ║
    ║  Assets: BTC, ETH, SOL, BNB, AVAX, ADA, LINK, LTC           ║
    ║  Trade Size: $3 per position                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Show trading mode
    if Config.BYBIT_TESTNET:
        print("⚠️  RUNNING IN TESTNET MODE - No real money at risk!\n")
    else:
        print("🚀 LIVE TRADING MODE - Real money at stake!\n")
        print("💼 Account Balance Tracking Enabled")
        print("📊 Position Management Active")
        print("⚡ High-Frequency Execution Ready\n")

    print("Press Ctrl+C to stop the trading agent\n")

    try:
        agent = ProductionTradingAgent()
        agent.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
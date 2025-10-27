#!/usr/bin/env python3
"""
Asymmetric Crypto Trading Agent
Automated trading system using Grok 4 Fast for analysis and Bybit for execution
"""

import asyncio
import logging
import signal
import sys
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

class AsymmetricTradeAgent:
    def __init__(self):
        self.data_collector = DataCollector()
        self.trading_engine = TradingEngine()
        self.is_running = False

    async def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping trading agent...")
        self.is_running = False
        await self.trading_engine.stop()
        sys.exit(0)

    async def process_market_data(self, data_list):
        """Callback function to process collected market data"""
        try:
            logger.info(f"Processing market data for {len(data_list)} assets...")

            # Generate and execute trading signals
            await self.trading_engine.process_signals(data_list)

            # Log portfolio status
            portfolio = await self.trading_engine.get_portfolio_summary()
            if portfolio:
                logger.info(f"Portfolio Status - Balance: ${portfolio['total_balance']:.2f}, "
                          f"Active Positions: {portfolio['active_positions']}, "
                          f"Unrealized PNL: ${portfolio['unrealized_pnl']:.2f}")

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    async def print_status(self):
        """Print system status"""
        while self.is_running:
            try:
                # Get portfolio summary
                portfolio = await self.trading_engine.get_portfolio_summary()
                if portfolio:
                    print(f"\n{'='*60}")
                    print(f"ASYMMETRIC TRADING AGENT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")
                    print(f"Total Balance: ${portfolio['total_balance']:.2f}")
                    print(f"Available Balance: ${portfolio['available_balance']:.2f}")
                    print(f"Total Invested: ${portfolio['total_invested']:.2f}")
                    print(f"Unrealized PNL: ${portfolio['unrealized_pnl']:.2f}")
                    print(f"Active Positions: {portfolio['active_positions']}")
                    print(f"Total Trades: {portfolio['total_trades']}")

                    # Show active positions
                    active_positions = self.trading_engine.get_active_positions()
                    if active_positions:
                        print(f"\nActive Positions:")
                        for symbol, pos_data in active_positions.items():
                            signal = pos_data['signal']
                            try:
                                # Get current market price through trading engine to avoid session conflicts
                                current_price = await self.trading_engine.get_current_price(symbol)
                                # Avoid division by zero
                                if signal.entry_price > 0:
                                    pnl_change = ((current_price - signal.entry_price) / signal.entry_price) * 100
                                else:
                                    pnl_change = 0.0
                                    logger.warning(f"Invalid entry price for {symbol}: {signal.entry_price}")
                                print(f"  {symbol}: Entry=${signal.entry_price:.4f}, "
                                      f"Current=${current_price:.4f}, "
                                      f"Target=${signal.activation_price:.4f}, "
                                      f"PNL={pnl_change:.2f}%")
                            except Exception as e:
                                logger.error(f"Error getting current price for {symbol}: {e}")
                                # Fallback to entry price if API call fails
                                pnl_change = 0.0
                                print(f"  {symbol}: Entry=${signal.entry_price:.4f}, "
                                      f"Current=N/A, "
                                      f"Target=${signal.activation_price:.4f}, "
                                      f"PNL={pnl_change:.2f}%")

                    print(f"{'='*60}\n")

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error printing status: {str(e)}")
                await asyncio.sleep(60)

    async def run(self):
        """Main run loop"""
        try:
            logger.info("Starting Asymmetric Crypto Trading Agent...")

            # Initialize trading engine
            await self.trading_engine.initialize()

            self.is_running = True

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.signal_handler(s, f)))
            signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.signal_handler(s, f)))

            # Start trading engine
            trading_task = asyncio.create_task(self.trading_engine.start())

            # Start status printer
            status_task = asyncio.create_task(self.print_status())

            # Start data collection (this will run continuously)
            data_task = asyncio.create_task(
                self.data_collector.continuous_data_collection(self.process_market_data)
            )

            logger.info("All systems started. Trading agent is now running...")
            logger.info(f"Target Assets: {', '.join(Config.TARGET_ASSETS)}")
            logger.info(f"Trade Size: ${Config.DEFAULT_TRADE_SIZE} with {Config.MAX_LEVERAGE}x leverage")
            logger.info(f"Data Collection Interval: {Config.DATA_COLLECTION_INTERVAL}s")
            logger.info(f"Signal Check Interval: {Config.SIGNAL_CHECK_INTERVAL}s")

            # Wait for all tasks
            await asyncio.gather(trading_task, status_task, data_task, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main run loop: {str(e)}")
        finally:
            self.is_running = False
            await self.trading_engine.stop()
            logger.info("Trading agent stopped")

async def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           ASYMMETRIC CRYPTO TRADING AGENT v1.0               ║
    ║                                                               ║
    ║  🤖 Grok 4 Fast Powered Research Analysis                     ║
    ║  📈 Bybit Perpetual Futures Execution                        ║
    ║  💰 High-Leverage Asymmetric Trading                         ║
    ║                                                               ║
    ║  Target: 150%+ PNL with 50-75x Leverage                     ║
    ║  Assets: BTC, ETH, SOL, ARB, XRP, OP, RENDER, INJ           ║
    ║  Trade Size: $3 per position                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    if Config.BYBIT_TESTNET:
        print("⚠️  RUNNING IN TESTNET MODE - No real money at risk!")
    else:
        print("🚀 RUNNING IN LIVE MODE - Real trading enabled!")

    print("\nPress Ctrl+C to stop the trading agent\n")

    agent = AsymmetricTradeAgent()
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTrading agent stopped by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
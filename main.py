#!/usr/bin/env python3
"""
PRODUCTION Asymmetric Crypto Trading Agent
Hybrid Architecture: Data → LLM → Execution → Scheduler
LIVE TRADING - NO TESTNET - REAL MONEY
"""

import asyncio
import logging
import signal
import sys
import json
from datetime import datetime, timedelta
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
        self.last_analysis_time = None
        
        # HYBRID ARCHITECTURE COMPONENTS
        self.data_layer = DataCollector()  # Fetches fresh data
        self.llm_layer = None  # Will be initialized for reasoning
        self.execution_layer = TradingEngine()  # Places orders
        self.scheduler_interval = Config.SIGNAL_CHECK_INTERVAL  # Control frequency

    async def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 SHUTDOWN SIGNAL RECEIVED - Closing positions...")
        self.is_running = False
        await self.trading_engine.stop()
        sys.exit(0)

    async def execute_hybrid_trading_cycle(self):
        """
        PRODUCTION HYBRID CYCLE:
        1. Data Layer: Fetch fresh market data
        2. LLM Layer: Reason on data + prompt.md
        3. Execution Layer: Place orders based on LLM decisions
        4. Logger: Track everything
        """
        try:
            cycle_start = datetime.now()
            logger.info("🚀 STARTING HYBRID TRADING CYCLE")
            
            # STEP 1: DATA LAYER - Fetch fresh data snapshot
            logger.info("📊 DATA LAYER: Fetching fresh market data...")
            market_data_snapshot = await self.data_layer.fetch_all_assets_data()
            
            if not market_data_snapshot:
                logger.error("❌ DATA LAYER: Failed to fetch market data")
                return False
                
            logger.info(f"✅ DATA LAYER: Fetched data for {len(market_data_snapshot)} assets")
            
            # STEP 2: LLM LAYER - Reason on the data
            logger.info("🧠 LLM LAYER: Running asymmetric analysis...")
            trading_signals = await self.trading_engine.process_signals(market_data_snapshot)
            
            # STEP 3: EXECUTION LAYER - Execute decisions
            logger.info("⚡ EXECUTION LAYER: Processing signals...")
            # (This happens inside process_signals)
            
            # STEP 4: LOG RESULTS
            portfolio = await self.trading_engine.get_portfolio_summary()
            if portfolio:
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                logger.info(f"✅ CYCLE COMPLETE in {cycle_time:.1f}s")
                logger.info(f"💰 Portfolio: ${portfolio['total_balance']:.2f} | "
                          f"Positions: {portfolio['active_positions']} | "
                          f"PNL: ${portfolio['unrealized_pnl']:.2f}")
            
            self.last_analysis_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ HYBRID CYCLE ERROR: {str(e)}")
            return False

    async def production_scheduler(self):
        """
        PRODUCTION SCHEDULER:
        Runs hybrid trading cycles at configured intervals
        This replaces the broken continuous data collection
        """
        while self.is_running:
            try:
                # Execute one complete hybrid cycle
                success = await self.execute_hybrid_trading_cycle()
                
                if success:
                    # Calculate next run time
                    next_run = datetime.now() + timedelta(seconds=self.scheduler_interval)
                    wait_time = self.scheduler_interval
                    
                    logger.info(f"⏰ Next analysis at: {next_run.strftime('%H:%M:%S')} (waiting {wait_time}s)")
                    await asyncio.sleep(wait_time)
                else:
                    # If cycle failed, wait shorter time and retry
                    logger.warning("⚠️ Cycle failed, retrying in 60 seconds...")
                    await asyncio.sleep(60)
                    
            except Exception as e:
                logger.error(f"❌ SCHEDULER ERROR: {str(e)}")
                await asyncio.sleep(60)

    async def print_production_status(self):
        """Print production trading status"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                portfolio = await self.trading_engine.get_portfolio_summary()
                if portfolio:
                    print(f"\n{'='*80}")
                    print(f"🚀 PRODUCTION TRADING AGENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}")
                    print(f"💰 Total Balance: ${portfolio['total_balance']:.2f}")
                    print(f"💵 Available: ${portfolio['available_balance']:.2f}")
                    print(f"📊 Invested: ${portfolio['total_invested']:.2f}")
                    print(f"📈 Unrealized PNL: ${portfolio['unrealized_pnl']:.2f}")
                    print(f"🎯 Active Positions: {portfolio['active_positions']}")
                    print(f"📋 Total Trades: {portfolio['total_trades']}")
                    
                    if self.last_analysis_time:
                        last_run = self.last_analysis_time.strftime('%H:%M:%S')
                        print(f"🕐 Last Analysis: {last_run}")

                    # Show active positions with PNL
                    active_positions = self.trading_engine.get_active_positions()
                    if active_positions:
                        print(f"\n🎯 ACTIVE POSITIONS:")
                        for symbol, pos_data in active_positions.items():
                            signal = pos_data['signal']
                            try:
                                current_price = await self.trading_engine.get_current_price(symbol)
                                if signal.entry_price > 0:
                                    pnl_change = ((current_price - signal.entry_price) / signal.entry_price) * 100
                                    pnl_dollars = (current_price - signal.entry_price) * signal.quantity
                                else:
                                    pnl_change = 0.0
                                    pnl_dollars = 0.0
                                    
                                print(f"  📈 {symbol}:")
                                print(f"     Entry: ${signal.entry_price:.4f} → Current: ${current_price:.4f}")
                                print(f"     Target: ${signal.activation_price:.4f} | PNL: {pnl_change:.2f}% (${pnl_dollars:.2f})")
                                print(f"     Leverage: {signal.leverage}x | Size: ${Config.DEFAULT_TRADE_SIZE * signal.leverage:.0f}")
                            except Exception as e:
                                logger.error(f"Error getting price for {symbol}: {e}")

                    print(f"{'='*80}\n")

            except Exception as e:
                logger.error(f"Status display error: {str(e)}")

    async def run_production(self):
        """PRODUCTION MAIN RUN LOOP - Hybrid Architecture"""
        try:
            print("🚀 INITIALIZING PRODUCTION TRADING SYSTEM...")
            
            # Initialize execution layer
            await self.trading_engine.initialize()
            
            # Verify LIVE TRADING mode
            if Config.BYBIT_TESTNET:
                print("❌ ERROR: TESTNET MODE DETECTED - SWITCHING TO LIVE")
                Config.BYBIT_TESTNET = False
                
            if Config.DISABLE_TRADING:
                print("❌ ERROR: TRADING DISABLED - ENABLING LIVE TRADING")
                Config.DISABLE_TRADING = False

            self.is_running = True

            # Set up signal handlers
            signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.signal_handler(s, f)))
            signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.signal_handler(s, f)))

            # Start production tasks
            scheduler_task = asyncio.create_task(self.production_scheduler())
            status_task = asyncio.create_task(self.print_production_status())

            print("✅ PRODUCTION SYSTEM ONLINE")
            print(f"🎯 Target Assets: {', '.join(Config.TARGET_ASSETS)}")
            print(f"💰 Trade Size: ${Config.DEFAULT_TRADE_SIZE} with {Config.MAX_LEVERAGE}x leverage")
            print(f"⏰ Analysis Interval: {self.scheduler_interval}s")
            print(f"🔥 MODE: LIVE TRADING - REAL MONEY")

            # Run until stopped
            await asyncio.gather(scheduler_task, status_task, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt - shutting down...")
        except Exception as e:
            logger.error(f"❌ PRODUCTION ERROR: {str(e)}")
        finally:
            self.is_running = False
            await self.trading_engine.stop()
            logger.info("🛑 Production system stopped")

async def main():
    """PRODUCTION MAIN ENTRY POINT"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       🚀 PRODUCTION ASYMMETRIC TRADING AGENT v2.0            ║
    ║                                                               ║
    ║  🏗️  Hybrid Architecture: Data → LLM → Execution → Scheduler  ║
    ║  🧠 AI-Powered Institutional Analysis                        ║
    ║  ⚡ Bybit Perpetual Futures Execution                        ║
    ║  💰 150%+ PNL Targets with 50-75x Leverage                   ║
    ║                                                               ║
    ║  🎯 Assets: BTC, ETH, SOL, ARB, XRP, OP, RENDER, INJ       ║
    ║  💵 Position Size: $3 per trade (2% risk)                    ║
    ║  ⏰ Analysis: Every 60 minutes                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    print("🔥 MODE: LIVE TRADING - REAL MONEY AT RISK")
    print("🎯 GOAL: 3x Account Growth Through Asymmetric Trading")
    print("\nPress Ctrl+C to stop the production agent\n")

    agent = ProductionTradingAgent()
    await agent.run_production()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Production agent stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)
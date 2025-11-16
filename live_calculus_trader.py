import asyncio
from trading_orchestrator import TradingOrchestrator

async def main():
    """
    Main entry point for the trading bot.
    """
    orchestrator = TradingOrchestrator()
    await orchestrator.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")

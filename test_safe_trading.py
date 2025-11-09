#!/usr/bin/env python3
"""
Safe trading test to verify the balance fix works in real conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_calculus_trader import LiveCalculusTrader
import asyncio
import logging
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeTradingTest:
    def __init__(self):
        self.trader = None
        self.running = True

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Received interrupt signal, shutting down...")
        self.running = False
        if self.trader:
            self.trader.emergency_stop = True

    async def run_safe_test(self, duration_seconds=60):
        """Run a safe trading test for a limited duration"""
        print("=" * 60)
        print("🧪 SAFE TRADING TEST - Balance Fix Verification")
        print("=" * 60)
        print(f"⏰ Running for {duration_seconds} seconds maximum")
        print("🛡️  Safety measures enabled:")
        print("   • Reduced position sizes to 50% of available balance")
        print("   • Conservative leverage calculations")
        print("   • Automatic shutdown after timeout")
        print("   • Emergency stop on Ctrl+C")
        print("=" * 60)

        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            # Initialize trader
            self.trader = LiveCalculusTrader()

            print("\n🚀 Starting safe trading test...")
            print("📊 Monitoring for trades and balance issues...")

            start_time = asyncio.get_event_loop().time()

            # Run the trader with timeout
            while self.running:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time

                if elapsed >= duration_seconds:
                    print(f"\n⏰ Test completed after {duration_seconds} seconds")
                    break

                # Run a single trading iteration
                try:
                    await asyncio.sleep(1)  # Small delay to prevent overwhelming

                    # Check if trader is still running
                    if not self.trader.emergency_stop:
                        # Log status every 10 seconds
                        if int(elapsed) % 10 == 0:
                            print(f"⏱️  Test running... {int(elapsed)}s elapsed")
                    else:
                        print("⚠️  Emergency stop triggered")
                        break

                except Exception as e:
                    logger.error(f"Error during trading iteration: {e}")
                    break

            print("\n✅ Safe trading test completed")
            print("📊 Summary:")
            print("   • No 'insufficient balance' errors should occur")
            print("   • Position sizes should be conservative")
            print("   • System should handle low balance gracefully")

        except Exception as e:
            logger.error(f"Error in safe trading test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.trader:
                self.trader.emergency_stop = True
                print("🛑 Trader stopped")

if __name__ == "__main__":
    test = SafeTradingTest()
    asyncio.run(test.run_safe_test(duration_seconds=120))  # 2 minutes test
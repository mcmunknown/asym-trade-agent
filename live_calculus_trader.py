import pandas as pd
import logging
from websocket_client import BybitWebsocketClient
from calculus_strategy import generate_trading_signals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveCalculusTrader:
    def __init__(self, symbols: list, window_size: int = 100):
        self.ws_client = BybitWebsocketClient(symbols)
        self.window_size = window_size
        self.prices = {symbol: [] for symbol in symbols}

    def run(self):
        """Starts the live trading bot."""
        self.ws_client.subscribe()
        logger.info("Live calculus trader started. Waiting for data...")

        while True:
            message = self.ws_client.fetch_message()
            if message:
                self.process_message(message)

    def process_message(self, message: dict):
        """Processes a new message from the WebSocket."""
        try:
            data = message.get("data", [])
            for trade in data:
                symbol = trade.get("s")
                price = float(trade.get("p"))

                if symbol in self.prices:
                    self.prices[symbol].append(price)

                    # Keep the price list at the desired window size
                    if len(self.prices[symbol]) > self.window_size:
                        self.prices[symbol].pop(0)

                    # Generate signals if we have enough data
                    if len(self.prices[symbol]) == self.window_size:
                        price_series = pd.Series(self.prices[symbol])
                        signals = generate_trading_signals(price_series)
                        latest_signal = signals.iloc[-1]

                        logger.info(f"--- New Signal for {symbol} ---")
                        logger.info(f"Price: {latest_signal['price']:.2f}")
                        logger.info(f"Smoothed Price: {latest_signal['smoothed_price']:.2f}")
                        logger.info(f"Velocity: {latest_signal['velocity']:.4f}")
                        logger.info(f"Acceleration: {latest_signal['acceleration']:.6f}")
                        logger.info(f"Signal: {latest_signal['signal']}")
                        logger.info(f"Forecast: {latest_signal['forecast']:.2f}")
                        logger.info("--------------------")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

if __name__ == '__main__':
    trader = LiveCalculusTrader(symbols=["BTCUSDT", "ETHUSDT"])
    trader.run()

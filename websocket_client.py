from pybit.unified_trading import WebSocket
import logging

logger = logging.getLogger(__name__)

class BybitWebsocketClient:
    def __init__(self, symbols: list):
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
        )
        self.symbols = symbols

    def subscribe(self):
        """Subscribes to the public trade channel for the specified symbols."""
        topics = [f"publicTrade.{symbol}" for symbol in self.symbols]
        self.ws.subscribe(topics)
        logger.info(f"Subscribed to public trade channel for {self.symbols}")

    def fetch_message(self):
        """Fetches the latest message from the WebSocket."""
        return self.ws.fetch_message()

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    ws_client = BybitWebsocketClient(["BTCUSDT", "ETHUSDT"])
    ws_client.subscribe()
    while True:
        message = ws_client.fetch_message()
        if message:
            logger.info(message)

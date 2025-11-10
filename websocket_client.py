"""
Robust Bybit WebSocket Client - Direct Implementation
====================================================

Direct WebSocket implementation for Bybit V5 API without pybit dependency.
Provides reliable real-time data with automatic reconnection and fallback.

Features:
- Direct WebSocket connection to Bybit
- Automatic reconnection with exponential backoff
- Real-time data validation and quality control
- REST API fallback when WebSocket fails
- Connection state monitoring
"""

import asyncio
import json
import logging
import time
import threading
import websockets
from typing import List, Dict, Callable, Optional, Any
from enum import Enum
from dataclasses import dataclass
import queue
from datetime import datetime
import requests
import ssl
from direct_rest_client import DirectRESTClient

logger = logging.getLogger(__name__)

class ChannelType(Enum):
    """Supported Bybit WebSocket channel types"""
    TRADE = "publicTrade"
    ORDERBOOK_1 = "orderbook.1"
    ORDERBOOK_25 = "orderbook.25"
    ORDERBOOK_500 = "orderbook.500"
    KLINE_1 = "kline.1"
    KLINE_5 = "kline.5"
    KLINE_15 = "kline.15"
    KLINE_60 = "kline.60"
    TICKER = "tickers"

@dataclass
class MarketData:
    """Structured market data from Bybit WebSocket"""
    symbol: str
    timestamp: float
    price: float
    volume: float
    side: str  # 'Buy' or 'Sell'
    channel_type: ChannelType
    raw_data: Dict

class RobustBybitWebSocketClient:
    """
    Robust WebSocket client with direct implementation and REST API fallback.
    """

    def __init__(self,
                 symbols: List[str],
                 testnet: bool = False,
                 channel_types: List[ChannelType] = None,
                 heartbeat_interval: int = 20,
                 max_reconnect_attempts: int = 10,
                 reconnect_backoff: float = 2.0):
        """
        Initialize the robust WebSocket client.

        Args:
            symbols: List of trading symbols to subscribe to
            testnet: Use testnet environment
            channel_types: List of channel types to subscribe to
            heartbeat_interval: Heartbeat interval in seconds (default: 20s)
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_backoff: Backoff multiplier for reconnection
        """
        self.symbols = symbols
        self.testnet = testnet
        self.channel_types = channel_types or [ChannelType.TRADE]
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff = reconnect_backoff

        # WebSocket endpoints
        if testnet:
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.rest_url = "https://api-testnet.bybit.com"
        else:
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
            self.rest_url = "https://api.bybit.com"

        # Connection state
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        self.last_heartbeat = time.time()
        self.last_data_time = time.time()
        self.connection_thread = None
        self.heartbeat_thread = None
        
        # REST API fallback client
        self.rest_client = None
        self.rest_fallback_active = False

        # Data management
        self.data_queue = queue.Queue()
        self.callbacks = {}
        self.portfolio_callbacks: List[Callable[[Dict[str, MarketData]], None]] = []
        self.latest_market_data: Dict[str, MarketData] = {}
        self.rest_fallback_enabled = True
        self.rest_polling_thread = None

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'reconnections': 0,
            'last_message_time': None,
            'connection_uptime': 0,
            'data_quality_score': 1.0,
            'rest_fallback_count': 0,
            'websocket_failures': 0
        }

        logger.info(f"Robust Bybit WebSocket client initialized for {symbols}")

    def add_callback(self, channel_type: ChannelType, callback: Callable[[MarketData], None]):
        """Add callback function for specific channel type."""
        if channel_type not in self.callbacks:
            self.callbacks[channel_type] = []
        self.callbacks[channel_type].append(callback)
        logger.info(f"Added callback for {channel_type.value}")

    def add_portfolio_callback(self, callback: Callable[[Dict[str, MarketData]], None]):
        """Register portfolio-level market data callback."""
        self.portfolio_callbacks.append(callback)
        logger.info("Added portfolio-level market data callback")

    def _build_subscribe_message(self) -> Dict:
        """Build WebSocket subscription message."""
        topics = []
        for symbol in self.symbols:
            for channel_type in self.channel_types:
                topics.append(f"{channel_type.value}.{symbol}")
        
        return {
            "op": "subscribe",
            "args": topics
        }

    async def _connect_websocket(self):
        """Establish WebSocket connection."""
        try:
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ),
                timeout=30
            )

            # Send subscription message
            subscribe_msg = self._build_subscribe_message()
            await self.websocket.send(json.dumps(subscribe_msg))
            logger.info(f"WebSocket connected and subscribed to: {subscribe_msg['args']}")

            self.is_connected = True
            self.reconnect_count = 0
            self.stats['connection_start_time'] = time.time()

            # Start message processing
            await self._process_messages()

        except Exception as e:
            self.is_connected = False
            self.stats['websocket_failures'] += 1
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def _process_messages(self):
        """Process incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.is_connected = False

    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket message."""
        try:
            # Update statistics
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = datetime.now()
            self.last_data_time = time.time()

            # Handle different message types
            if 'topic' in data and 'data' in data:
                topic = data['topic']
                message_data = data['data']

                # Parse based on topic
                if 'publicTrade' in topic:
                    await self._parse_trade_data(topic, message_data)
                elif 'tickers' in topic:
                    await self._parse_ticker_data(topic, message_data)
                elif 'orderbook' in topic:
                    await self._parse_orderbook_data(topic, message_data)

            elif 'success' in data:
                # Subscription confirmation
                if data.get('success'):
                    logger.info(f"Subscription confirmed: {data.get('retMsg', 'Unknown')}")
                else:
                    logger.warning(f"Subscription failed: {data.get('retMsg', 'Unknown')}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _parse_trade_data(self, topic: str, data: List[Dict]):
        """Parse trade data and trigger callbacks."""
        try:
            symbol = topic.split('.')[-1]
            timestamp = time.time()

            for trade in data:
                # Validate trade data
                if not all(key in trade for key in ['p', 'v', 's', 'T']):
                    continue

                market_data = MarketData(
                    symbol=symbol,
                    price=float(trade['p']),
                    volume=float(trade['v']),
                    side=trade['s'],
                    timestamp=timestamp,
                    channel_type=ChannelType.TRADE,
                    raw_data=trade
                )

                # Update latest data
                self.latest_market_data[symbol] = market_data
                self.stats['messages_processed'] += 1

                # Trigger callbacks
                for callback in self.callbacks.get(ChannelType.TRADE, []):
                    try:
                        callback(market_data)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")

                # Trigger portfolio callbacks
                if self.portfolio_callbacks:
                    snapshot = dict(self.latest_market_data)
                    for portfolio_callback in self.portfolio_callbacks:
                        try:
                            portfolio_callback(snapshot)
                        except Exception as e:
                            logger.error(f"Portfolio callback error: {e}")

        except Exception as e:
            logger.error(f"Error parsing trade data: {e}")

    async def _parse_ticker_data(self, topic: str, data: List[Dict]):
        """Parse ticker data and trigger callbacks."""
        try:
            symbol = topic.split('.')[-1]
            timestamp = time.time()

            for ticker in data:
                market_data = MarketData(
                    symbol=symbol,
                    price=float(ticker.get('lastPrice', 0)),
                    volume=float(ticker.get('volume24h', 0)),
                    side='',
                    timestamp=timestamp,
                    channel_type=ChannelType.TICKER,
                    raw_data=ticker
                )

                # Update latest data
                self.latest_market_data[symbol] = market_data
                self.stats['messages_processed'] += 1

                # Trigger callbacks
                for callback in self.callbacks.get(ChannelType.TICKER, []):
                    try:
                        callback(market_data)
                    except Exception as e:
                        logger.error(f"Ticker callback error: {e}")

        except Exception as e:
            logger.error(f"Error parsing ticker data: {e}")

    async def _parse_orderbook_data(self, topic: str, data: List[Dict]):
        """Parse orderbook data (for future use)."""
        # Orderbook parsing can be implemented as needed
        pass

def _initialize_rest_client(self):
        """Initialize REST API fallback client."""
        try:
            from config import Config
            self.rest_client = DirectRESTClient(
                api_key=Config.BYBIT_API_KEY,
                api_secret=Config.BYBIT_API_SECRET,
                testnet=self.testnet
            )
            logger.info("REST API fallback client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize REST client: {e}")

    def _rest_fallback_polling(self):
        """REST API fallback polling when WebSocket fails."""
        if not self.rest_client:
            self._initialize_rest_client()
        
        if not self.rest_client:
            logger.error("Cannot start REST fallback - client initialization failed")
            return
            
        logger.info("Starting REST API fallback polling")
        self.rest_fallback_active = True
        
        while self.is_running and not self.is_connected and self.rest_fallback_enabled:
            try:
                # Get market data for all symbols
                market_data_dict = self.rest_client.get_market_tickers(self.symbols)
                
                if market_data_dict:
                    for symbol, data in market_data_dict.items():
                        market_data = MarketData(
                            symbol=symbol,
                            price=data['last_price'],
                            volume=data['volume_24h'],
                            side='',
                            timestamp=data['timestamp'],
                            channel_type=ChannelType.TICKER,
                            raw_data=data
                        )
                        
                        # Update and trigger callbacks
                        self.latest_market_data[symbol] = market_data
                        self.stats['rest_fallback_count'] += 1
                        
                        # Trigger callbacks
                        for callback in self.callbacks.get(ChannelType.TICKER, []):
                            try:
                                callback(market_data)
                            except Exception as e:
                                logger.error(f"REST fallback callback error: {e}")
                        
                        # Trigger portfolio callbacks
                        if self.portfolio_callbacks and len(self.latest_market_data) >= len(self.symbols) * 0.5:
                            snapshot = dict(self.latest_market_data)
                            for portfolio_callback in self.portfolio_callbacks:
                                try:
                                    portfolio_callback(snapshot)
                                except Exception as e:
                                    logger.error(f"REST portfolio callback error: {e}")

                # Wait before next poll
                time.sleep(3)  # Poll every 3 seconds for more responsive data
                
            except Exception as e:
                logger.error(f"REST fallback polling error: {e}")
                time.sleep(5)  # Wait longer on error
        
        self.rest_fallback_active = False
        logger.info("REST fallback polling stopped")

    def _connection_manager(self):
        """Manage WebSocket connection with reconnection logic."""
        while self.is_running:
            try:
                if not self.is_connected:
                    logger.info(f"Attempting to connect WebSocket (attempt {self.reconnect_count + 1})")
                    
                    # Run WebSocket connection in asyncio event loop
                    asyncio.run(self._connect_websocket())
                    
                # Connection established, wait for issues
                time.sleep(self.heartbeat_interval)
                
                # Check connection health
                if self.is_connected:
                    time_since_last_data = time.time() - self.last_data_time
                    if time_since_last_data > self.heartbeat_interval * 3:
                        logger.warning(f"Connection stale: {time_since_last_data:.1f}s since last data")
                        self.is_connected = False
                        self.stats['websocket_failures'] += 1

            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                self.is_connected = False
                
                # Exponential backoff for reconnection
                if self.reconnect_count < self.max_reconnect_attempts:
                    backoff_delay = min(self.reconnect_backoff ** self.reconnect_count, 60)
                    logger.info(f"Reconnecting in {backoff_delay:.1f} seconds...")
                    time.sleep(backoff_delay)
                    self.reconnect_count += 1
                    self.stats['reconnections'] += 1
                else:
                    logger.error(f"Max reconnection attempts reached. Enabling REST fallback.")
                    self.rest_fallback_enabled = True

    def _heartbeat_loop(self):
        """Send periodic heartbeat to maintain connection."""
        while self.is_running:
            try:
                if self.is_connected and self.websocket:
                    # Send ping
                    try:
                        asyncio.run(self.websocket.ping())
                        self.last_heartbeat = time.time()
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                        self.is_connected = False
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5)

    def subscribe(self):
        """Prepare subscription (actual subscription happens in start)."""
        topics = []
        for symbol in self.symbols:
            for channel_type in self.channel_types:
                topics.append(f"{channel_type.value}.{symbol}")
        logger.info(f"Will subscribe to topics: {topics}")
        return True

    def start(self):
        """Start the WebSocket client with robust connection management."""
        if self.is_running:
            logger.warning("WebSocket client is already running")
            return

        try:
            self.is_running = True
            logger.info("Starting robust WebSocket client...")

            # Start connection manager thread
            self.connection_thread = threading.Thread(target=self._connection_manager, daemon=True)
            self.connection_thread.start()

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

            # Start REST fallback thread
            self.rest_polling_thread = threading.Thread(target=self._rest_fallback_polling, daemon=True)
            self.rest_polling_thread.start()

            logger.info("Robust WebSocket client started with REST fallback")

            # Keep main thread alive
            while self.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("WebSocket client stopped by user")
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the WebSocket client gracefully."""
        logger.info("Stopping robust WebSocket client...")
        self.is_running = False
        self.is_connected = False
        
        # Close WebSocket connection
        if self.websocket:
            try:
                asyncio.run(self.websocket.close())
            except Exception:
                pass
        
        # Update statistics
        if 'connection_start_time' in self.stats:
            self.stats['connection_uptime'] = time.time() - self.stats['connection_start_time']
        
        logger.info("Robust WebSocket client stopped")

    def get_statistics(self) -> Dict:
        """Get connection and data statistics."""
        uptime = time.time() - self.stats.get('connection_start_time', time.time())
        return {
            **self.stats,
            'connection_uptime': uptime,
            'messages_per_second': self.stats['messages_received'] / max(uptime, 1),
            'data_quality_score': self.stats['data_quality_score'],
            'queue_size': self.data_queue.qsize(),
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'reconnect_count': self.reconnect_count,
            'rest_fallback_active': not self.is_connected and self.rest_fallback_enabled
        }

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    def trade_callback(market_data: MarketData):
        """Example callback for trade data"""
        logger.info(f"Trade: {market_data.symbol} {market_data.side} "
                   f"{market_data.volume}@{market_data.price}")

    # Initialize client
    ws_client = RobustBybitWebSocketClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        testnet=False,
        channel_types=[ChannelType.TRADE, ChannelType.TICKER],
        heartbeat_interval=20
    )

    # Add callback
    ws_client.add_callback(ChannelType.TRADE, trade_callback)

    try:
        # Subscribe and start
        ws_client.subscribe()
        ws_client.start()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        ws_client.stop()
        stats = ws_client.get_statistics()
        logger.info(f"Final statistics: {stats}")

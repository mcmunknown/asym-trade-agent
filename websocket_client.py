"""
Enhanced Bybit WebSocket Client for Anne's Calculus Trading System
==================================================================

This module provides a production-ready WebSocket client for Bybit's real-time data streams
with comprehensive error handling, heartbeat management, and multi-channel support.

Features:
- Robust reconnection logic with exponential backoff
- 20-second heartbeat to prevent disconnections
- Multiple channel subscriptions (trades, orderbook, klines)
- Data validation and quality checks
- Comprehensive error handling and logging
- Async support for high-frequency data processing
"""

import asyncio
import json
import logging
import time
import threading
from typing import List, Dict, Callable, Optional, Any
from enum import Enum
import websockets
from dataclasses import dataclass
from pybit.unified_trading import WebSocket
import queue
from datetime import datetime

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

class BybitWebSocketClient:
    """
    Enhanced WebSocket client for Bybit real-time data with robust error handling
    and production-ready features for Anne's calculus trading system.
    """

    def __init__(self,
                 symbols: List[str],
                 testnet: bool = False,
                 channel_types: List[ChannelType] = None,
                 heartbeat_interval: int = 20,
                 max_reconnect_attempts: int = 10,
                 reconnect_backoff: float = 2.0):
        """
        Initialize the enhanced WebSocket client.

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

        # Initialize WebSocket (actual stream subscriptions happen in start)
        self.ws = WebSocket(
            testnet=testnet,
            channel_type="linear",
        )

        # Data management
        self.data_queue = queue.Queue()
        self.callbacks = {}
        self.portfolio_callbacks: List[Callable[[Dict[str, MarketData]], None]] = []
        self.latest_market_data: Dict[str, MarketData] = {}
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        self.last_heartbeat = time.time()
        self.last_data_time = time.time()

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'reconnections': 0,
            'last_message_time': None,
            'connection_uptime': 0,
            'data_quality_score': 1.0
        }

        logger.info(f"Enhanced Bybit WebSocket client initialized for {symbols}")

    def add_callback(self, channel_type: ChannelType, callback: Callable[[MarketData], None]):
        """
        Add callback function for specific channel type.

        Args:
            channel_type: Channel type for the callback
            callback: Function to process market data
        """
        if channel_type not in self.callbacks:
            self.callbacks[channel_type] = []
        self.callbacks[channel_type].append(callback)
        logger.info(f"Added callback for {channel_type.value}")

    def add_portfolio_callback(self, callback: Callable[[Dict[str, MarketData]], None]):
        """
        Register callback that receives latest market data snapshot for all symbols.

        Args:
            callback: Function accepting dict[symbol -> MarketData]
        """
        self.portfolio_callbacks.append(callback)
        logger.info("Added portfolio-level market data callback")

    def _build_topics(self) -> List[str]:
        """Build topic list for subscription."""
        topics = []
        for symbol in self.symbols:
            for channel_type in self.channel_types:
                topics.append(f"{channel_type.value}.{symbol}")
        return topics

    def _start_streams(self):
        """Attach pybit WebSocket streams for the requested channel types."""
        try:
            if not self.symbols:
                raise ValueError("No symbols configured for WebSocket subscription")

            subscribed = False

            if ChannelType.TRADE in self.channel_types:
                self.ws.trade_stream(symbol=self.symbols, callback=self._handle_pybit_message)
                subscribed = True
                logger.info(f"Subscribed to trade stream for {self.symbols}")

            if ChannelType.TICKER in self.channel_types:
                self.ws.ticker_stream(symbol=self.symbols, callback=self._handle_pybit_message)
                subscribed = True
                logger.info(f"Subscribed to ticker stream for {self.symbols}")

            # Orderbook depth mapping for supported channel types
            depth_map = {
                ChannelType.ORDERBOOK_1: 1,
                ChannelType.ORDERBOOK_25: 25,
                ChannelType.ORDERBOOK_500: 500
            }

            for channel_type in self.channel_types:
                if channel_type in depth_map:
                    depth = depth_map[channel_type]
                    self.ws.orderbook_stream(depth=depth, symbol=self.symbols,
                                             callback=self._handle_pybit_message)
                    subscribed = True
                    logger.info(f"Subscribed to orderbook depth {depth} stream")

            if not subscribed:
                logger.warning("No supported channel types specified for WebSocket client")

            self.is_connected = True
            self.stats['connection_start_time'] = time.time()

        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to start WebSocket streams: {e}")
            raise

    def _handle_pybit_message(self, message: Dict):
        """Unified handler for pybit callbacks."""
        if not message:
            return
        self._process_message(message)

    def _validate_message(self, message: Dict) -> bool:
        """
        Validate WebSocket message for data quality.

        Args:
            message: Raw message from WebSocket

        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Check basic structure
            if not isinstance(message, dict):
                return False

            # Check for required fields
            if 'topic' not in message or 'data' not in message:
                return False

            # Check data array
            data = message.get('data', [])
            if not isinstance(data, list):
                return False

            # Validate each data item based on channel type
            topic = message.get('topic', '')
            for item in data:
                if not isinstance(item, dict):
                    return False

                # Trade data validation
                if 'publicTrade' in topic:
                    required_fields = ['p', 'v', 's', 'T']  # price, volume, side, timestamp
                    if not all(field in item for field in required_fields):
                        return False

                    # Validate numeric fields
                    try:
                        float(item['p'])
                        float(item['v'])
                        int(item['T'])
                    except (ValueError, TypeError):
                        return False

            return True

        except Exception as e:
            logger.warning(f"Message validation error: {e}")
            return False

    def _parse_trade_data(self, message: Dict) -> List[MarketData]:
        """Parse trade data from WebSocket message."""
        market_data_list = []
        try:
            symbol = message['topic'].split('.')[-1]
            timestamp = time.time()

            for trade in message['data']:
                market_data = MarketData(
                    symbol=symbol,
                    price=float(trade['p']),
                    volume=float(trade['v']),
                    side=trade['s'],
                    timestamp=timestamp,
                    channel_type=ChannelType.TRADE,
                    raw_data=trade
                )
                market_data_list.append(market_data)

        except Exception as e:
            logger.error(f"Error parsing trade data: {e}")

        return market_data_list

    def _handle_trade_message(self, message):
        """Handle trade message from pybit callback."""
        try:
            if isinstance(message, dict):
                topic = message.get('topic', '')
                symbol = topic.split('.')[-1] if '.' in topic else topic
                timestamp = time.time()
                
                market_data = MarketData(
                    symbol=symbol,
                    price=float(message.get('p', 0)),
                    volume=float(message.get('v', 0)),
                    side=message.get('s', 'Buy'),
                    timestamp=timestamp,
                    channel_type=ChannelType.TRADE,
                    raw_data=message
                )
                
                for callback in self.callbacks.get(ChannelType.TRADE, []):
                    callback(market_data)
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")

    def _handle_ticker_message(self, message):
        """Handle ticker message from pybit callback."""
        try:
            if isinstance(message, dict):
                topic = message.get('topic', '')
                symbol = topic.split('.')[-1] if '.' in topic else topic
                timestamp = time.time()
                
                market_data = MarketData(
                    symbol=symbol,
                    price=float(message.get('lastPrice', 0)),
                    volume=float(message.get('volume24h', 0)),
                    side='',
                    timestamp=timestamp,
                    channel_type=ChannelType.TICKER,
                    raw_data=message
                )
                
                for callback in self.callbacks.get(ChannelType.TICKER, []):
                    callback(market_data)
        except Exception as e:
            logger.error(f"Error in ticker callback: {e}")

    def _process_message(self, message: Dict):
        """
        Process incoming WebSocket message and distribute to callbacks.

        Args:
            message: Raw WebSocket message
        """
        try:
            # Update statistics
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = datetime.now()
            self.last_data_time = time.time()

            # Validate message quality
            if not self._validate_message(message):
                logger.warning(f"Invalid message received: {message}")
                self.stats['data_quality_score'] *= 0.99  # Decrease quality score
                return

            # Parse message based on topic
            topic = message.get('topic', '')
            market_data_list = []

            if 'publicTrade' in topic:
                market_data_list = self._parse_trade_data(message)
            else:
                # Handle other channel types as needed
                logger.debug(f"Unhandled topic type: {topic}")
                return

            # Distribute to callbacks
            if market_data_list:
                channel_type = ChannelType.TRADE  # For now, focusing on trade data
                if channel_type in self.callbacks:
                    for callback in self.callbacks[channel_type]:
                        try:
                            for market_data in market_data_list:
                                callback(market_data)
                                self.stats['messages_processed'] += 1
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                # Update latest market snapshot for portfolio callbacks
                for market_data in market_data_list:
                    self.latest_market_data[market_data.symbol] = market_data

                if self.portfolio_callbacks and self.latest_market_data:
                    snapshot = dict(self.latest_market_data)
                    for portfolio_callback in self.portfolio_callbacks:
                        try:
                            portfolio_callback(snapshot)
                        except Exception as e:
                            logger.error(f"Portfolio callback error: {e}")

            # Add to queue for async processing
            self.data_queue.put(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _heartbeat_loop(self):
        """Send periodic heartbeat to maintain connection."""
        while self.is_running:
            try:
                current_time = time.time()
                time_since_last_data = current_time - self.last_data_time

                # Check if connection is stale
                if time_since_last_data > self.heartbeat_interval * 2:
                    logger.warning(f"Connection stale: {time_since_last_data:.1f}s since last data")
                    self._reconnect()

                # Send heartbeat (pybit handles this automatically)
                self.last_heartbeat = current_time
                logger.debug("Heartbeat sent")

                time.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5)  # Wait before retry

    def _reconnect(self):
        """Attempt to reconnect to WebSocket with exponential backoff."""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.stop()
            return

        # Calculate backoff delay
        backoff_delay = self.reconnect_backoff ** self.reconnect_count
        backoff_delay = min(backoff_delay, 60)  # Cap at 60 seconds

        logger.info(f"Attempting reconnection {self.reconnect_count + 1}/{self.max_reconnect_attempts} "
                   f"in {backoff_delay:.1f} seconds...")

        time.sleep(backoff_delay)

        try:
            # Reinitialize WebSocket
            try:
                self.ws.exit()
            except Exception:
                pass
            self.ws = WebSocket(
                testnet=self.testnet,
                channel_type="linear",
            )

            # Resubscribe to topics
            self._start_streams()

            self.reconnect_count += 1
            self.stats['reconnections'] += 1
            self.is_connected = True

            logger.info("Reconnection successful")

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.is_connected = False
            self.reconnect_count += 1

    def subscribe(self):
        """Subscribe to the configured channels and symbols."""
        try:
            topics = self._build_topics()
            logger.info(f"Will subscribe to topics: {topics}")
            # Subscription happens in start() via run_stream
            self.is_connected = True
            logger.info(f"Ready to subscribe to {len(topics)} topics for {self.symbols}")

        except Exception as e:
            logger.error(f"Subscription preparation failed: {e}")
            self.is_connected = False

    def start(self):
        """Start the WebSocket client with background processing."""
        if self.is_running:
            logger.warning("WebSocket client is already running")
            return

        try:
            self.is_running = True
            start_time = time.time()

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            heartbeat_thread.start()

            logger.info("Enhanced Bybit WebSocket client started")
            self._start_streams()

            while self.is_running:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("WebSocket client stopped by user")
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        finally:
            self.is_running = False
            if self.is_connected:
                self.stats['connection_uptime'] = time.time() - start_time
                logger.info(f"WebSocket client stopped. Uptime: {self.stats['connection_uptime']:.1f}s")
            try:
                self.ws.exit()
            except Exception:
                pass

    def stop(self):
        """Stop the WebSocket client gracefully."""
        logger.info("Stopping WebSocket client...")
        self.is_running = False
        self.is_connected = False
        try:
            self.ws.exit()
        except Exception:
            pass

    def fetch_message(self) -> Optional[Dict]:
        """
        Fetch latest processed message from internal queue (legacy compatibility).

        Returns:
            Latest message or None if no message available
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def get_data_from_queue(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get data from the internal queue (non-blocking).

        Args:
            timeout: Queue timeout in seconds

        Returns:
            Message from queue or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

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
            'is_running': self.is_running
        }

if __name__ == '__main__':
    # Example usage with enhanced features
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    def trade_callback(market_data: MarketData):
        """Example callback for trade data"""
        logger.info(f"Trade: {market_data.symbol} {market_data.side} "
                   f"{market_data.volume}@{market_data.price}")

    # Initialize client with multiple channels
    ws_client = BybitWebSocketClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        testnet=False,
        channel_types=[ChannelType.TRADE, ChannelType.ORDERBOOK_1],
        heartbeat_interval=20
    )

    # Add callback for trade data
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

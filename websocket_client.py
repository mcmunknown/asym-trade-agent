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


class DirectRESTClient:
    """Minimal REST client enabling ticker polling when websockets fail."""

    PUBLIC_TIMEOUT = 8

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        category: str = "linear"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.category = category
        self.session = requests.Session()

        if not self.api_key or not self.api_secret:
            logger.warning("DirectRESTClient initialized without API credentials; public endpoints only.")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = timeout or self.PUBLIC_TIMEOUT

        try:
            response = self.session.request(method=method, url=url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error {payload.get('retCode')}: {payload.get('retMsg')}")
            return payload
        except requests.RequestException as exc:
            logger.error(f"REST request failed for {url}: {exc}")
            raise

    def get_market_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch latest ticker snapshot for each requested symbol."""
        if not symbols:
            return {}

        tickers: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            params = {"category": self.category, "symbol": symbol}
            try:
                payload = self._request("GET", "/v5/market/tickers", params=params)
            except Exception as exc:
                logger.error(f"Failed to fetch ticker for {symbol}: {exc}")
                continue

            result = payload.get("result", {})
            ticker_list = result.get("list") or []
            if not ticker_list:
                logger.warning(f"No ticker data returned for {symbol}")
                continue

            data = ticker_list[0]
            server_time = payload.get("time")
            timestamp = server_time / 1000.0 if isinstance(server_time, (int, float)) else time.time()

            try:
                tickers[symbol] = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "last_price": float(data.get("lastPrice", 0.0)),
                    "volume_24h": float(data.get("volume24h", 0.0)),
                    "price_change_pct": float(data.get("price24hPcnt", 0.0)),
                    "turnover_24h": float(data.get("turnover24h", 0.0)),
                    "bid_price": float(data.get("bid1Price", 0.0)),
                    "ask_price": float(data.get("ask1Price", 0.0)),
                    "raw": data,
                }
            except (TypeError, ValueError) as exc:
                logger.error(f"Malformed ticker data for {symbol}: {exc}")

        return tickers

    def close(self):
        """Close the underlying HTTP session."""
        self.session.close()

class BybitWebSocketClient:
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
        self.websocket_loop = None
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
        self.latest_orderbook: Dict[str, Dict[str, Any]] = {}
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

        self.symbol_last_update: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.symbol_warning_counts: Dict[str, int] = {symbol: 0 for symbol in symbols}

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

            # Connect with extended timeouts for better reliability
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ssl=ssl_context,
                    ping_interval=30,  # Less frequent pings to avoid conflicts
                    ping_timeout=20,   # Longer timeout for network congestion
                    close_timeout=15,
                    max_size=10**7,    # 10MB max message size
                    compression=None   # Disable compression for stability
                ),
                timeout=45  # Longer connection timeout
            )
            self.websocket_loop = asyncio.get_running_loop()

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
        """Process incoming WebSocket messages with enhanced diagnostics."""
        message_count = 0
        last_ping_time = time.time()
        
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                message_count += 1
                
                # Log message receipt for diagnostics
                if message_count % 50 == 0:  # Log every 50 messages
                    logger.info(f"WebSocket message processing healthy: {message_count} messages received")
                
                try:
                    data = json.loads(message)
                    
                    # Check for ping/pong messages
                    if isinstance(data, dict) and 'ping' in data:
                        pong_msg = {'pong': data['ping']}
                        await self.websocket.send(json.dumps(pong_msg))
                        last_ping_time = time.time()
                        logger.debug(f"Responded to ping: {data['ping']}")
                        continue
                    
                    # Handle regular messages
                    await self._handle_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                    logger.debug(f"Raw message: {message[:200]}...")  # Log first 200 chars
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    if message_count == 1:  # Log first message on error
                        logger.debug(f"First message content: {message}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.is_connected = False
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.is_connected = False
        finally:
            logger.info(f"WebSocket message processing ended. Total messages: {message_count}")
            self.websocket_loop = None
            self.websocket = None

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

            elif 'success' in data or 'retMsg' in data:
                # Subscription confirmation or response message
                success = data.get('success', data.get('retCode') == 0)
                msg = data.get('retMsg', data.get('message', 'Unknown'))
                
                if success:
                    logger.info(f"✓ WebSocket subscription successful: {msg}")
                    # Reset last data time on successful subscription
                    self.last_data_time = time.time()
                else:
                    logger.error(f"✗ WebSocket operation failed: {msg}")
                    
            elif 'auth' in data:
                # Auth response (for private streams)
                if data.get('auth') == 'success':
                    logger.info("✓ WebSocket authentication successful")
                else:
                    logger.error("✗ WebSocket authentication failed")

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
                self.symbol_last_update[symbol] = timestamp
                self.symbol_warning_counts[symbol] = 0
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

    async def _parse_ticker_data(self, topic: str, data):
        """Parse ticker data and trigger callbacks."""
        try:
            symbol = topic.split('.')[-1]
            timestamp = time.time()

            # The data is already the ticker dict from Bybit
            ticker = data
            
            if not ticker or not isinstance(ticker, dict):
                logger.warning(f"Invalid ticker data for {symbol}: {data}")
                return

            # Extract price with proper field fallback for Bybit v5 ticker data
            price = 0.0
            if 'lastPrice' in ticker and ticker['lastPrice']:
                price = float(ticker['lastPrice'])
            elif 'p1' in ticker and ticker['p1']:  # Current price in Bybit v5
                price = float(ticker['p1']) / 100  # Convert from price scale
            elif 'pc' in ticker and ticker['pc']:  # Previous close as fallback
                price = float(ticker['pc'])
            elif 'markI' in ticker and ticker['markI']:  # Mark price as fallback
                price = float(ticker['markI'])
            elif 'bid1Price' in ticker and ticker['bid1Price'] and 'ask1Price' in ticker and ticker['ask1Price']:
                price = (float(ticker['bid1Price']) + float(ticker['ask1Price'])) / 2
            elif 'markPrice' in ticker and ticker['markPrice']:
                price = float(ticker['markPrice'])
            
            # Extract volume
            volume = 0.0
            if 'volume24h' in ticker and ticker['volume24h']:
                volume = float(ticker['volume24h'])
            elif 'turnover24h' in ticker and ticker['turnover24h']:
                volume = float(ticker['turnover24h'])

            if price > 0:  # Only create MarketData if we have a valid price
                market_data = MarketData(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    side='',
                    timestamp=timestamp,
                    channel_type=ChannelType.TICKER,
                    raw_data=ticker
                )

                # Update latest data
                self.latest_market_data[symbol] = market_data
                self.symbol_last_update[symbol] = timestamp
                self.symbol_warning_counts[symbol] = 0
                self.stats['messages_processed'] += 1

                # Trigger callbacks
                for callback in self.callbacks.get(ChannelType.TICKER, []):
                    try:
                        callback(market_data)
                    except Exception as e:
                        logger.error(f"Ticker callback error: {e}")
                        
                # Trigger portfolio callbacks with current data
                if self.portfolio_callbacks:
                    snapshot = dict(self.latest_market_data)
                    for portfolio_callback in self.portfolio_callbacks:
                        try:
                            portfolio_callback(snapshot)
                        except Exception as e:
                            logger.error(f"Portfolio callback error: {e}")
            else:
                logger.debug(f"Skipping ticker for {symbol} - no valid price found")

        except Exception as e:
            logger.error(f"Error parsing ticker data for {topic}: {e}")
            logger.debug(f"Raw ticker data: {data}")

    async def _parse_orderbook_data(self, topic: str, data):
        """Parse orderbook data and track top-of-book microstructure."""
        try:
            symbol = topic.split('.')[-1]
            timestamp = time.time()

            entries = data
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                return

            best_bid = None
            best_ask = None
            best_bid_size = 0.0
            best_ask_size = 0.0

            previous_snapshot = self.latest_orderbook.get(symbol, {})

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                # Bybit orderbook.1 may supply bp/bv/ap/av fields
                if entry.get('bp') is not None:
                    try:
                        best_bid = float(entry.get('bp'))
                        best_bid_size = float(entry.get('bs', 0) or 0.0)
                    except (TypeError, ValueError):
                        pass
                if entry.get('ap') is not None:
                    try:
                        best_ask = float(entry.get('ap'))
                        best_ask_size = float(entry.get('as', 0) or 0.0)
                    except (TypeError, ValueError):
                        pass

                # Snapshot/delta format with arrays
                bids = entry.get('b') or entry.get('B')
                asks = entry.get('a') or entry.get('A')

                if bids and best_bid is None:
                    try:
                        price_str, size_str = bids[0][0], bids[0][1] if len(bids[0]) > 1 else 0
                        best_bid = float(price_str)
                        best_bid_size = float(size_str)
                    except (TypeError, ValueError, IndexError):
                        pass

                if asks and best_ask is None:
                    try:
                        price_str, size_str = asks[0][0], asks[0][1] if len(asks[0]) > 1 else 0
                        best_ask = float(price_str)
                        best_ask_size = float(size_str)
                    except (TypeError, ValueError, IndexError):
                        pass

            if best_bid is None:
                best_bid = previous_snapshot.get('best_bid')
                best_bid_size = previous_snapshot.get('best_bid_size', 0.0)

            if best_ask is None:
                best_ask = previous_snapshot.get('best_ask')
                best_ask_size = previous_snapshot.get('best_ask_size', 0.0)

            if best_bid is None and best_ask is None:
                return

            if best_bid is None and best_ask is not None:
                best_bid = best_ask
            if best_ask is None and best_bid is not None:
                best_ask = best_bid

            spread = max(best_ask - best_bid, 0.0) if best_ask is not None and best_bid is not None else 0.0
            mid_price = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else best_bid or best_ask
            spread_pct = (spread / mid_price) if mid_price and mid_price > 0 else 0.0

            snapshot = {
                'symbol': symbol,
                'timestamp': timestamp,
                'best_bid': best_bid,
                'best_bid_size': best_bid_size,
                'best_ask': best_ask,
                'best_ask_size': best_ask_size,
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'raw': data
            }

            self.latest_orderbook[symbol] = snapshot
            self.symbol_last_update[symbol] = timestamp
            self.symbol_warning_counts[symbol] = 0
            self.stats['messages_processed'] += 1

            # Fire callbacks for orderbook channel
            market_data = MarketData(
                symbol=symbol,
                price=mid_price or 0.0,
                volume=0.0,
                side='',
                timestamp=timestamp,
                channel_type=ChannelType.ORDERBOOK_1,
                raw_data=snapshot
            )

            for callback in self.callbacks.get(ChannelType.ORDERBOOK_1, []):
                try:
                    callback(market_data)
                except Exception as e:
                    logger.error(f"Orderbook callback error: {e}")

        except Exception as e:
            logger.error(f"Error parsing orderbook data for {topic}: {e}")

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

    def get_orderbook_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest top-of-book snapshot for a symbol."""
        return self.latest_orderbook.get(symbol)

    def _rest_fallback_polling(self):
        """Enhanced REST API fallback with adaptive polling and data validation."""
        if not self.rest_client:
            self._initialize_rest_client()
        
        if not self.rest_client:
            logger.error("Cannot start REST fallback - client initialization failed")
            return
            
        logger.info("Starting enhanced REST API fallback polling")
        self.rest_fallback_active = True
        
        # Adaptive polling parameters
        base_poll_interval = 2.0  # Start with 2 seconds
        max_poll_interval = 15.0  # Max 15 seconds
        current_interval = base_poll_interval
        consecutive_errors = 0
        last_successful_fetch = time.time()
        
        while self.is_running and not self.is_connected and self.rest_fallback_enabled:
            try:
                start_time = time.time()
                
                # Get market data for all symbols
                market_data_dict = self.rest_client.get_market_tickers(self.symbols)
                
                if market_data_dict:
                    # Update successful fetch tracking
                    last_successful_fetch = time.time()
                    consecutive_errors = 0
                    
                    # Reset polling interval to base on success
                    if current_interval > base_poll_interval:
                        current_interval = max(current_interval * 0.8, base_poll_interval)
                        logger.debug(f"Reduced REST poll interval to {current_interval:.1f}s")
                    
                    # Process data with validation
                    valid_symbols = 0
                    for symbol, data in market_data_dict.items():
                        # Validate data quality
                        if data['last_price'] <= 0 or data['timestamp'] <= 0:
                            logger.warning(f"Invalid REST data for {symbol}: price={data['last_price']}, timestamp={data['timestamp']}")
                            continue
                            
                        # Check data freshness (should be within last 30 seconds)
                        data_age = time.time() - data['timestamp']
                        if data_age > 30:
                            logger.warning(f"Stale REST data for {symbol}: {data_age:.1f}s old")
                            continue
                            
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
                        self.symbol_last_update[symbol] = data['timestamp']
                        self.symbol_warning_counts[symbol] = 0
                        self.stats['rest_fallback_count'] += 1
                        valid_symbols += 1
                        
                        # Trigger individual symbol callbacks
                        for callback in self.callbacks.get(ChannelType.TICKER, []):
                            try:
                                callback(market_data)
                            except Exception as e:
                                logger.error(f"REST fallback callback error for {symbol}: {e}")
                        
                        # Trigger portfolio callbacks only with sufficient data
                        if self.portfolio_callbacks and valid_symbols >= len(self.symbols) * 0.5:
                            snapshot = dict(self.latest_market_data)
                            for portfolio_callback in self.portfolio_callbacks:
                                try:
                                    portfolio_callback(snapshot)
                                except Exception as e:
                                    logger.error(f"REST portfolio callback error: {e}")
                    
                    # Log data quality metrics
                    fetch_duration = time.time() - start_time
                    logger.debug(f"REST fetch: {valid_symbols}/{len(self.symbols)} symbols in {fetch_duration:.2f}s")
                    
                else:
                    logger.warning("No market data received from REST API")
                    consecutive_errors += 1
                
                # Adaptive polling interval adjustment
                if consecutive_errors > 0:
                    # Increase interval on errors
                    current_interval = min(current_interval * 1.5, max_poll_interval)
                    logger.debug(f"Increased REST poll interval to {current_interval:.1f}s due to errors")
                
                # Ensure minimum polling interval
                poll_time = max(current_interval - (time.time() - start_time), 0.5)
                time.sleep(poll_time)
                
            except Exception as e:
                logger.error(f"REST fallback polling error: {e}")
                consecutive_errors += 1
                current_interval = min(current_interval * 2, max_poll_interval)
                
                # Extended backoff for repeated errors
                if consecutive_errors > 5:
                    time.sleep(10)
                else:
                    time.sleep(5)
        
        self.rest_fallback_active = False
        logger.info("Enhanced REST fallback polling stopped")

    def _fetch_single_ticker(self, symbol: str) -> bool:
        """Fetch a single symbol snapshot to refresh stale data."""
        try:
            if not self.rest_client:
                self._initialize_rest_client()
            if not self.rest_client:
                return False
            payload = self.rest_client.get_market_tickers([symbol])
            data = payload.get(symbol)
            if not data:
                return False
            timestamp = data.get('timestamp', time.time())
            last_price = data.get('last_price', 0.0)
            if last_price <= 0:
                return False
            market_data = MarketData(
                symbol=symbol,
                price=last_price,
                volume=data.get('volume_24h', 0.0),
                side='',
                timestamp=timestamp,
                channel_type=ChannelType.TICKER,
                raw_data=data
            )
            self.latest_market_data[symbol] = market_data
            self.symbol_last_update[symbol] = timestamp
            self.symbol_warning_counts[symbol] = 0
            return True
        except Exception as exc:
            logger.warning(f"Failed single-ticker refresh for {symbol}: {exc}")
            return False

    def _connection_manager(self):
        """Enhanced connection manager with network resilience and jitter."""
        import random
        
        while self.is_running:
            try:
                if not self.is_connected:
                    # Add jitter to prevent thundering herd on reconnections
                    jitter = random.uniform(0.1, 0.5)
                    base_delay = min(self.reconnect_backoff ** self.reconnect_count, 120)  # Max 2 minutes
                    jittered_delay = base_delay + jitter
                    
                    logger.info(f"Attempting WebSocket connection (attempt {self.reconnect_count + 1}/{self.max_reconnect_attempts})")
                    
                    try:
                        # Run WebSocket connection in asyncio event loop
                        asyncio.run(self._connect_websocket())
                    except Exception as conn_error:
                        logger.error(f"WebSocket connection attempt failed: {conn_error}")
                        self.is_connected = False
                        
                        # Only increment retry count if we haven't exceeded max attempts
                        if self.reconnect_count < self.max_reconnect_attempts:
                            logger.info(f"Reconnecting in {jittered_delay:.1f} seconds...")
                            time.sleep(jittered_delay)
                            self.reconnect_count += 1
                            self.stats['reconnections'] += 1
                        else:
                            logger.error("Max reconnection attempts reached. Switching to REST fallback mode.")
                            self.rest_fallback_enabled = True
                            # Reset count after extended period to allow retry
                            time.sleep(300)  # Wait 5 minutes before retry cycle
                            self.reconnect_count = 0
                        continue
                    
                # Connection established - monitor with adaptive checking
                if self.is_connected:
                    # Adaptive monitoring based on connection quality
                    if self.stats.get('websocket_failures', 0) == 0:
                        check_interval = self.heartbeat_interval * 2  # Less frequent when stable
                    else:
                        check_interval = self.heartbeat_interval  # More frequent when unstable
                    
                    time.sleep(check_interval)
                    
                    # Proactive connection quality check
                    time_since_last_data = time.time() - self.last_data_time
                    quality_threshold = check_interval * 2.5  # Adaptive threshold
                    
                    if time_since_last_data > quality_threshold:
                        logger.warning(f"Proactive connection check: {time_since_last_data:.1f}s since last data (threshold: {quality_threshold:.1f}s)")
                        self.is_connected = False
                        self.stats['websocket_failures'] += 1
                        # Don't immediately reconnect - let the connection manager loop handle it

            except KeyboardInterrupt:
                logger.info("Connection manager interrupted by user")
                break
            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                self.is_connected = False
                time.sleep(5)  # Brief pause before retrying manager loop

    def _heartbeat_loop(self):
        """Adaptive connection health monitoring with quality assessment."""
        consecutive_failures = 0
        adaptive_interval = self.heartbeat_interval
        
        while self.is_running:
            try:
                if self.is_connected:
                    time_since_last = time.time() - self.last_data_time
                    
                    # Adaptive timeout based on connection quality
                    if consecutive_failures == 0:
                        timeout_threshold = adaptive_interval * 4  # Normal threshold
                    elif consecutive_failures <= 2:
                        timeout_threshold = adaptive_interval * 2  # More sensitive
                    else:
                        timeout_threshold = adaptive_interval * 1.5  # Very sensitive
                    
                    if time_since_last > timeout_threshold:
                        logger.warning(f"Connection stale: {time_since_last:.1f}s without updates (threshold: {timeout_threshold:.1f}s)")
                        self.is_connected = False
                        self.stats['websocket_failures'] += 1
                        consecutive_failures += 1
                    else:
                        # Reset failure count on successful data receipt
                        if time_since_last < adaptive_interval:
                            consecutive_failures = 0
                            # Gradually restore normal interval
                            if adaptive_interval < self.heartbeat_interval:
                                adaptive_interval = min(adaptive_interval + 1, self.heartbeat_interval)
                        
                        # Adaptive interval adjustment based on data flow
                        if time_since_last > adaptive_interval * 0.8:
                            # Increase monitoring frequency when data is sparse
                            adaptive_interval = max(adaptive_interval - 0.5, 5)

                    # Per-symbol health checks
                    now = time.time()
                    symbol_timeout = max(min(timeout_threshold, 60.0), 45.0)
                    for symbol in self.symbols:
                        last_update = self.symbol_last_update.get(symbol, 0.0)
                        if last_update <= 0:
                            continue
                        age = now - last_update
                        if age > symbol_timeout:
                            warn_count = self.symbol_warning_counts.get(symbol, 0)
                            if warn_count % 3 == 0:
                                logger.warning(
                                    f"Symbol {symbol} stale: {age:.1f}s without updates"
                                )
                            self.symbol_warning_counts[symbol] = warn_count + 1
                            if not self._fetch_single_ticker(symbol):
                                if age > symbol_timeout * 2:
                                    logger.error(f"Symbol {symbol} unresponsive; forcing connection reset")
                                    self.is_connected = False
                                    self.stats['websocket_failures'] += 1
                                    break
                            else:
                                logger.info(f"Refreshed {symbol} via REST after {age:.1f}s gap")
                                break
                
                time.sleep(adaptive_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                consecutive_failures += 1
                adaptive_interval = max(adaptive_interval - 1, 3)  # Increase frequency on errors
                time.sleep(3)

    def test_connectivity(self):
        """Test network connectivity to Bybit endpoints."""
        import socket
        import urllib.request
        import urllib.error
        
        logger.info("Testing network connectivity to Bybit endpoints...")
        
        # Test basic network connectivity
        try:
            host = "api.bybit.com" if not self.testnet else "api-testnet.bybit.com"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((host, 443))
            sock.close()
            
            if result == 0:
                logger.info(f"✓ TCP connectivity to {host}:443 successful")
            else:
                logger.error(f"✗ TCP connectivity to {host}:443 failed (error code: {result})")
                return False
                
        except Exception as e:
            logger.error(f"✗ Network connectivity test failed: {e}")
            return False
        
        # Test HTTP connectivity
        try:
            url = "https://api.bybit.com/v5/market/time" if not self.testnet else "https://api-testnet.bybit.com/v5/market/time"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info("✓ HTTP connectivity to Bybit API successful")
                else:
                    logger.error(f"✗ HTTP connectivity failed with status: {response.status}")
                    return False
                    
        except urllib.error.URLError as e:
            logger.error(f"✗ HTTP connectivity test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error during HTTP test: {e}")
            return False
        
        # Test WebSocket endpoint accessibility
        try:
            ws_host = "stream.bybit.com" if not self.testnet else "stream-testnet.bybit.com"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((ws_host, 443))
            sock.close()
            
            if result == 0:
                logger.info(f"✓ WebSocket endpoint {ws_host}:443 accessible")
            else:
                logger.error(f"✗ WebSocket endpoint {ws_host}:443 not accessible (error code: {result})")
                return False
                
        except Exception as e:
            logger.error(f"✗ WebSocket endpoint test failed: {e}")
            return False
        
        logger.info("All connectivity tests passed!")
        return True

    def get_symbol_health(self) -> Dict[str, Optional[float]]:
        """Expose time since last update per symbol for diagnostics."""
        now = time.time()
        health: Dict[str, Optional[float]] = {}
        for symbol in self.symbols:
            last = self.symbol_last_update.get(symbol, 0.0)
            health[symbol] = None if last <= 0 else max(now - last, 0.0)
        return health

    def subscribe(self):
        """Prepare subscription (actual subscription happens in start)."""
        topics = []
        for symbol in self.symbols:
            for channel_type in self.channel_types:
                topics.append(f"{channel_type.value}.{symbol}")
        logger.info(f"Will subscribe to topics: {topics}")
        return True

    def start(self):
        """Start the WebSocket client with connectivity testing and robust connection management."""
        if self.is_running:
            logger.warning("WebSocket client is already running")
            return

        try:
            self.is_running = True
            logger.info("Starting enhanced WebSocket client with connectivity testing...")
            
            # Test network connectivity first
            if not self.test_connectivity():
                logger.error("Network connectivity tests failed. Switching to REST-only mode.")
                self.rest_fallback_enabled = True
                self.is_connected = False
            else:
                logger.info("Network connectivity verified. Attempting WebSocket connection...")

            # Initialize REST client for fallback
            self._initialize_rest_client()

            # Start connection manager thread
            self.connection_thread = threading.Thread(target=self._connection_manager, daemon=True)
            self.connection_thread.start()

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

            # Start REST fallback thread
            self.rest_polling_thread = threading.Thread(target=self._rest_fallback_polling, daemon=True)
            self.rest_polling_thread.start()

            logger.info("Enhanced WebSocket client started with connectivity monitoring and REST fallback")

        except KeyboardInterrupt:
            logger.info("WebSocket client stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
            self.stop()

    def stop(self):
        """Stop the WebSocket client gracefully."""
        logger.info("Stopping robust WebSocket client...")
        self.is_running = False
        self.is_connected = False
        
        # Close WebSocket connection
        if self.websocket:
            try:
                if self.websocket_loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self.websocket.close(),
                        self.websocket_loop
                    )
                    future.result(timeout=5)
                else:
                    asyncio.run(self.websocket.close())
            except Exception:
                pass
            finally:
                self.websocket = None
                self.websocket_loop = None
        
        # Update statistics
        if 'connection_start_time' in self.stats:
            self.stats['connection_uptime'] = time.time() - self.stats['connection_start_time']

        # Join threads to ensure clean shutdown
        for thread in [self.connection_thread, self.heartbeat_thread, self.rest_polling_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        self.connection_thread = None
        self.heartbeat_thread = None
        self.rest_polling_thread = None
        
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
    ws_client = BybitWebSocketClient(
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

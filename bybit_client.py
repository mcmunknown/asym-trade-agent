"""
Bybit API v5 Client - Unified Account Perpetual Futures Trading
Production-ready client for live trading with 50-75x leverage
"""

from custom_http_manager import CustomV5HTTPManager as HTTP
import logging
import time
from typing import Dict, List, Optional
from config import Config

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self):
        """Initialize Bybit client for unified account perpetual futures"""
        try:
            # Use pybit.unified_trading HTTP client for production
            self.client = HTTP(
                testnet=Config.BYBIT_TESTNET,
                api_key=Config.BYBIT_API_KEY,
                api_secret=Config.BYBIT_API_SECRET,
                log_requests=False,
                tld=Config.BYBIT_TLD,
            )
            logger.info(f"✅ Bybit client initialized - {'TESTNET' if Config.BYBIT_TESTNET else 'LIVE'}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bybit client: {str(e)}")
            self.client = None

    def get_account_balance(self) -> Optional[Dict]:
        """Get unified account balance"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            # Get wallet balance for unified account
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"  # Get USDT balance for trading
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    account_info = result['list'][0]
                    return {
                        'totalEquity': account_info.get('totalEquity', '0'),
                        'totalAvailableBalance': account_info.get('totalAvailableBalance', '0'),
                        'totalWalletBalance': account_info.get('totalWalletBalance', '0'),
                        'totalPerpUPL': account_info.get('totalPerpUPL', '0'),
                        'accountIMR': account_info.get('accountIMR', '0'),
                        'accountMMR': account_info.get('accountMMR', '0')
                    }
                else:
                    logger.warning("No account data found")
                    return None
            else:
                logger.error(f"Failed to get account balance: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return None

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for a symbol"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            # Get 24hr ticker data
            response = self.client.get_tickers(
                category="linear",
                symbol=symbol
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    ticker_data = result['list'][0]
                    return {
                        'symbol': ticker_data.get('symbol'),
                        'lastPrice': ticker_data.get('lastPrice'),
                        'bidPrice': ticker_data.get('bid1Price'),
                        'askPrice': ticker_data.get('ask1Price'),
                        'priceChange24h': ticker_data.get('price24hPcnt'),
                        'volume24h': ticker_data.get('turnover24h'),
                        'high24h': ticker_data.get('highPrice24h'),
                        'low24h': ticker_data.get('lowPrice24h'),
                        'openInterest': ticker_data.get('openInterest'),
                        'fundingRate': ticker_data.get('fundingRate')
                    }
                else:
                    logger.warning(f"No ticker data found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get market data for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Get current funding rate for a symbol"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            response = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=1
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    funding_data = result['list'][0]
                    return {
                        'symbol': symbol,
                        'fundingRate': funding_data.get('fundingRate'),
                        'fundingRateTimestamp': funding_data.get('fundingRateTimestamp')
                    }
                else:
                    logger.warning(f"No funding data found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get funding rate for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {str(e)}")
            return None

    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """Get open interest data for a symbol"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            response = self.client.get_open_interest(
                category="linear",
                symbol=symbol,
                interval="60",
                limit=1
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    oi_data = result['list'][0]
                    return {
                        'symbol': symbol,
                        'openInterest': oi_data.get('openInterest'),
                        'openInterestValue': oi_data.get('openInterestValue'),
                        'timestamp': oi_data.get('timestamp')
                    }
                else:
                    logger.warning(f"No OI data found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get open interest for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting open interest for {symbol}: {str(e)}")
            return None

    def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 200) -> List[Dict]:
        """Get kline/candlestick data for technical analysis"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    klines = []
                    for kline in result['list']:
                        # Official Bybit API v5 format: [timestamp, open, high, low, close, volume, turnover, openInterest]
                        kline_data = {
                            'timestamp': kline[0],
                            'open': kline[1],
                            'high': kline[2],
                            'low': kline[3],
                            'close': kline[4],
                            'volume': kline[5],
                            'turnover': kline[6]
                        }
                        # Add open interest if available (8th element)
                        if len(kline) > 7:
                            kline_data['openInterest'] = kline[7]
                        klines.append(kline_data)
                    return klines
                else:
                    logger.warning(f"No kline data found for {symbol}")
                    return []
            else:
                logger.error(f"Failed to get kline data for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting kline data for {symbol}: {str(e)}")
            return []

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, **kwargs) -> Optional[Dict]:
        """Place an order on the perpetual futures market with enhanced TP/SL support"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            order_params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'positionIdx': 0,  # One-way mode
                'timeInForce': 'GTC'
            }

            # Add optional parameters
            if 'price' in kwargs:
                order_params['price'] = str(kwargs['price'])
            if 'reduce_only' in kwargs:
                order_params['reduceOnly'] = kwargs['reduce_only']
            if 'close_on_trigger' in kwargs:
                order_params['closeOnTrigger'] = kwargs['close_on_trigger']
            if 'take_profit' in kwargs:
                order_params['takeProfit'] = str(kwargs['take_profit'])
            if 'stop_loss' in kwargs:
                order_params['stopLoss'] = str(kwargs['stop_loss'])
            if 'tp_trigger_by' in kwargs:
                order_params['tpTriggerBy'] = kwargs['tp_trigger_by']
            if 'sl_trigger_by' in kwargs:
                order_params['slTriggerBy'] = kwargs['sl_trigger_by']
            if 'tp_limit_price' in kwargs:
                order_params['tpLimitPrice'] = str(kwargs['tp_limit_price'])
            if 'sl_limit_price' in kwargs:
                order_params['slLimitPrice'] = str(kwargs['sl_limit_price'])

            response = self.client.place_order(**order_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Order placed successfully: {symbol} {side} {qty} @ {order_type}")
                if 'take_profit' in kwargs:
                    logger.info(f"  📈 TP: {kwargs['take_profit']}")
                if 'stop_loss' in kwargs:
                    logger.info(f"  📉 SL: {kwargs['stop_loss']}")
                return result
            else:
                logger.error(f"❌ Failed to place order: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def place_batch_orders(self, orders: List[Dict]) -> List[Dict]:
        """
        Place multiple orders in a single request for high-frequency trading.

        Args:
            orders: List of order dictionaries with order parameters

        Returns:
            List of order results
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            if len(orders) > 20:
                logger.error(f"Too many orders ({len(orders)}). Maximum is 20 per batch request")
                return []

            # Prepare batch request
            batch_request = []
            for order in orders:
                order_params = {
                    'category': 'linear',
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'orderType': order['order_type'],
                    'qty': str(order['qty']),
                    'positionIdx': 0,
                    'timeInForce': order.get('time_in_force', 'GTC')
                }

                # Add optional parameters
                if 'price' in order:
                    order_params['price'] = str(order['price'])
                if 'take_profit' in order:
                    order_params['takeProfit'] = str(order['take_profit'])
                if 'stop_loss' in order:
                    order_params['stopLoss'] = str(order['stop_loss'])
                if 'orderLinkId' in order:
                    order_params['orderLinkId'] = order['orderLinkId']

                batch_request.append(order_params)

            logger.info(f"Placing batch order with {len(orders)} orders")

            response = self.client.place_batch_order(category="linear", request=batch_request)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Batch order placed successfully")

                # Log individual order results
                if 'list' in result:
                    for order_result in result['list']:
                        orderLinkId = order_result.get('orderLinkId', 'N/A')
                        orderId = order_result.get('orderId', 'N/A')
                        status = order_result.get('orderStatus', 'N/A')
                        logger.info(f"  Order {orderLinkId}: {orderId} - {status}")

                return result.get('list', [])
            else:
                logger.error(f"❌ Failed to place batch order: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error placing batch order: {str(e)}")
            return []

    def place_conditional_order(self, symbol: str, side: str, order_type: str, qty: float,
                               trigger_price: float, **kwargs) -> Optional[Dict]:
        """
        Place a conditional order (stop order) for advanced risk management.

        Args:
            symbol: Trading symbol
            side: Order side (Buy/Sell)
            order_type: Order type (Market/Limit)
            qty: Order quantity
            trigger_price: Trigger price for conditional order
            **kwargs: Additional parameters (price, tp, sl, etc.)

        Returns:
            Order result or None if failed
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            order_params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'triggerPrice': str(trigger_price),
                'positionIdx': 0,
                'triggerBy': 'MarkPrice',  # Use mark price for trigger
                'timeInForce': 'GTC'
            }

            # Add optional parameters
            if 'price' in order_type.lower() and 'price' in kwargs:
                order_params['orderPrice'] = str(kwargs['price'])
            if 'tp_trigger_price' in kwargs:
                order_params['tpTriggerPrice'] = str(kwargs['tp_trigger_price'])
            if 'sl_trigger_price' in kwargs:
                order_params['slTriggerPrice'] = str(kwargs['sl_trigger_price'])

            response = self.client.place_order(**order_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Conditional order placed: {symbol} {side} {qty} @ trigger {trigger_price}")
                return result
            else:
                logger.error(f"❌ Failed to place conditional order: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error placing conditional order: {str(e)}")
            return None

    def set_trading_stop(self, symbol: str, **kwargs) -> bool:
        """
        Set trading stop (take profit/stop loss) for existing position.

        Args:
            symbol: Trading symbol
            **kwargs: TP/SL parameters
                - take_profit: Take profit price
                - stop_loss: Stop loss price
                - trailing_stop: Trailing stop amount
                - tp_trigger_by: TP trigger type (MarkPrice/LastPrice)
                - sl_trigger_by: SL trigger type (MarkPrice/LastPrice)
                - active_type: Stop activation type

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return False

            stop_params = {
                'category': 'linear',
                'symbol': symbol,
                'positionIdx': 0  # One-way mode
            }

            # Add TP/SL parameters
            if 'take_profit' in kwargs:
                stop_params['takeProfit'] = str(kwargs['take_profit'])
            if 'stop_loss' in kwargs:
                stop_params['stopLoss'] = str(kwargs['stop_loss'])
            if 'trailing_stop' in kwargs:
                stop_params['trailingStop'] = str(kwargs['trailing_stop'])
            if 'tp_trigger_by' in kwargs:
                stop_params['tpTriggerBy'] = kwargs['tp_trigger_by']
            if 'sl_trigger_by' in kwargs:
                stop_params['slTriggerBy'] = kwargs['sl_trigger_by']
            if 'active_type' in kwargs:
                stop_params['activeType'] = kwargs['active_type']

            response = self.client.set_trading_stop(**stop_params)

            if response and response.get('retCode') == 0:
                logger.info(f"✅ Trading stop set successfully for {symbol}")
                if 'take_profit' in kwargs:
                    logger.info(f"  📈 TP: {kwargs['take_profit']}")
                if 'stop_loss' in kwargs:
                    logger.info(f"  📉 SL: {kwargs['stop_loss']}")
                if 'trailing_stop' in kwargs:
                    logger.info(f"  🔄 Trailing Stop: {kwargs['trailing_stop']}")
                return True
            else:
                logger.error(f"❌ Failed to set trading stop for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Error setting trading stop: {str(e)}")
            return False

    def cancel_order(self, symbol: str, order_id: str = None, order_link_id: str = None) -> Optional[Dict]:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID (if provided)
            order_link_id: Custom order link ID (if provided)

        Returns:
            Cancellation result or None if failed
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            cancel_params = {
                'category': 'linear',
                'symbol': symbol
            }

            if order_id:
                cancel_params['orderId'] = order_id
            elif order_link_id:
                cancel_params['orderLinkId'] = order_link_id
            else:
                logger.error("Either order_id or order_link_id must be provided")
                return None

            response = self.client.cancel_order(**cancel_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Order cancelled successfully: {order_id or order_link_id}")
                return result
            else:
                logger.error(f"❌ Failed to cancel order: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return None

    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return False

            response = self.client.cancel_all_orders(
                category='linear',
                symbol=symbol
            )

            if response and response.get('retCode') == 0:
                logger.info(f"✅ All orders cancelled for {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to cancel all orders for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling all orders: {str(e)}")
            return False

    def get_active_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all active orders.

        Args:
            symbol: Trading symbol (optional)

        Returns:
            List of active orders
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            order_params = {
                'category': 'linear'
            }

            if symbol:
                order_params['symbol'] = symbol

            response = self.client.get_active_orders(**order_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    orders = []
                    for order in result['list']:
                        order_info = {
                            'orderId': order.get('orderId'),
                            'orderLinkId': order.get('orderLinkId'),
                            'symbol': order.get('symbol'),
                            'price': order.get('price'),
                            'qty': order.get('qty'),
                            'side': order.get('side'),
                            'orderType': order.get('orderType'),
                            'orderStatus': order.get('orderStatus'),
                            'createTime': order.get('createdTime'),
                            'takeProfit': order.get('takeProfit'),
                            'stopLoss': order.get('stopLoss')
                        }
                        orders.append(order_info)

                    logger.info(f"Retrieved {len(orders)} active orders")
                    return orders
                else:
                    logger.info("No active orders found")
                    return []
            else:
                logger.error(f"Failed to get active orders: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting active orders: {str(e)}")
            return []

    def amend_order(self, symbol: str, order_id: str = None, order_link_id: str = None, **kwargs) -> Optional[Dict]:
        """
        Amend an existing order.

        Args:
            symbol: Trading symbol
            order_id: Order ID (if provided)
            order_link_id: Custom order link ID (if provided)
            **kwargs: Parameters to amend (qty, price, etc.)

        Returns:
            Amendment result or None if failed
        """
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            amend_params = {
                'category': 'linear',
                'symbol': symbol
            }

            if order_id:
                amend_params['orderId'] = order_id
            elif order_link_id:
                amend_params['orderLinkId'] = order_link_id
            else:
                logger.error("Either order_id or order_link_id must be provided")
                return None

            # Add amendable parameters
            if 'qty' in kwargs:
                amend_params['qty'] = str(kwargs['qty'])
            if 'price' in kwargs:
                amend_params['price'] = str(kwargs['price'])
            if 'trigger_price' in kwargs:
                amend_params['triggerPrice'] = str(kwargs['trigger_price'])
            if 'take_profit' in kwargs:
                amend_params['takeProfit'] = str(kwargs['take_profit'])
            if 'stop_loss' in kwargs:
                amend_params['stopLoss'] = str(kwargs['stop_loss'])

            response = self.client.amend_order(**amend_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Order amended successfully: {order_id or order_link_id}")
                return result
            else:
                logger.error(f"❌ Failed to amend order: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error amending order: {str(e)}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (50-75x for asymmetric trading)"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return False

            # Check if leverage is in allowed range
            if leverage < 1 or leverage > 100:
                logger.error(f"Leverage {leverage} is outside allowed range (1-100)")
                return False

            response = self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )

            if response and response.get('retCode') == 0:
                logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
                return True
            elif response and response.get('retCode') == 110043:
                # Error code 110043 means leverage not modified (already set to this value)
                logger.info(f"✅ Leverage already set to {leverage}x for {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to set leverage for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return False

        except Exception as e:
            error_msg = str(e)
            # Check if error is about leverage already being set (error code 110043)
            if "110043" in error_msg or "leverage not modified" in error_msg.lower():
                logger.info(f"✅ Leverage already set to {leverage}x for {symbol} (exception handled)")
                return True
            else:
                logger.error(f"Error setting leverage: {error_msg}")
                return False

    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get current position information for a symbol"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            response = self.client.get_positions(
                category="linear",
                symbol=symbol
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    # Find active position (size > 0)
                    for position in result['list']:
                        if float(position.get('size', 0)) != 0:
                            return {
                                'symbol': position.get('symbol'),
                                'side': position.get('side'),
                                'size': position.get('size'),
                                'entryPrice': position.get('entryPrice'),
                                'markPrice': position.get('markPrice'),
                                'unrealisedPnl': position.get('unrealisedPnl'),
                                'percentage': position.get('percentage'),
                                'leverage': position.get('leverage'),
                                'positionValue': position.get('positionValue'),
                                'createdTime': position.get('createdTime')
                            }

                    # No active position found
                    logger.info(f"No active position found for {symbol}")
                    return None
                else:
                    logger.warning(f"No position data found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get position info for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting position info for {symbol}: {str(e)}")
            return None

    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Get instrument information for position sizing calculations"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    instrument = result['list'][0]
                    return {
                        'symbol': instrument.get('symbol'),
                        'maxLeverage': instrument.get('leverageFilter', {}).get('maxLeverage', '50'),
                        'minOrderQty': instrument.get('lotSizeFilter', {}).get('minOrderQty', '0.001'),
                        'maxOrderQty': instrument.get('lotSizeFilter', {}).get('maxOrderQty', '1000000'),
                        'qtyStep': instrument.get('lotSizeFilter', {}).get('qtyStep', '0.001'),
                        'minNotionalValue': instrument.get('lotSizeFilter', {}).get('minNotionalValue', '5.0'),
                        'pricePrecision': instrument.get('priceScale', '4'),
                        'status': instrument.get('status', 'Trading')
                    }
                else:
                    logger.warning(f"No instrument info found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get instrument info for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting instrument info for {symbol}: {str(e)}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Bybit API"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return False

            # Try to get server time
            response = self.client.get_server_time()

            if response and response.get('retCode') == 0:
                server_time = response.get('result', {}).get('timeSecond')
                logger.info(f"✅ Bybit API connection successful - Server time: {server_time}")
                return True
            else:
                logger.error(f"❌ Bybit API connection failed: {response.get('retMsg', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"❌ Bybit API connection test failed: {str(e)}")
            return False

    # ========================================
    # ENHANCED DATA COLLECTION METHODS (Phase 2)
    # ========================================

    def get_order_book_data(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Get order book depth data for liquidity analysis"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return None

            response = self.client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=limit
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result:
                    # Calculate liquidity metrics
                    bids = result.get('b', [])[:limit]  # Top bids
                    asks = result.get('a', [])[:limit]  # Top asks

                    # Calculate spread and liquidity score
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_ask * 100) if best_ask > 0 else 0

                    # Calculate liquidity score (sum of top 10 levels)
                    bid_liquidity = sum(float(bid[1]) * float(bid[0]) for bid in bids[:10])
                    ask_liquidity = sum(float(ask[1]) * float(ask[0]) for ask in asks[:10])
                    total_liquidity = bid_liquidity + ask_liquidity

                    return {
                        'symbol': symbol,
                        'timestamp': result.get('ts'),
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'bids': bids,
                        'asks': asks,
                        'bid_liquidity': bid_liquidity,
                        'ask_liquidity': ask_liquidity,
                        'total_liquidity': total_liquidity,
                        'liquidity_score': min(total_liquidity / 1000000, 1.0),  # Normalized 0-1
                        'order_book_depth': len(bids) + len(asks)
                    }
                else:
                    logger.warning(f"No order book data found for {symbol}")
                    return None
            else:
                logger.error(f"Failed to get order book for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting order book data for {symbol}: {str(e)}")
            return None

    def get_liquidation_data(self, symbol: str = None, start_time: int = None) -> List[Dict]:
        """Get recent liquidation data for risk analysis"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            # Since liquidation records API is not available in pybit HTTP client,
            # we'll simulate this to avoid errors
            logger.debug(f"Liquidation data requested for {symbol} - API not available, returning empty data")
            return []

        except Exception as e:
            logger.error(f"Error getting liquidation data: {str(e)}")
            return []

    def get_funding_rate_history(self, symbol: str, limit: int = 30) -> List[Dict]:
        """Get funding rate history for sentiment analysis"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            response = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=limit
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    funding_history = []
                    for funding_data in result['list']:
                        funding_record = {
                            'symbol': funding_data.get('symbol'),
                            'funding_rate': float(funding_data.get('fundingRate', 0)),
                            'funding_rate_timestamp': int(funding_data.get('fundingRateTimestamp', 0)),
                            'settle_price': float(funding_data.get('settlePrice', 0))
                        }
                        funding_history.append(funding_record)

                    # Calculate trend
                    if len(funding_history) >= 2:
                        recent_avg = sum(f['funding_rate'] for f in funding_history[:3]) / 3
                        older_avg = sum(f['funding_rate'] for f in funding_history[-3:]) / 3
                        trend = "increasing" if recent_avg > older_avg else "decreasing"

                        for record in funding_history:
                            record['trend'] = trend
                            record['recent_avg'] = recent_avg
                            record['older_avg'] = older_avg

                    logger.info(f"Retrieved {len(funding_history)} funding rate records for {symbol}")
                    return funding_history
                else:
                    logger.warning(f"No funding rate history found for {symbol}")
                    return []
            else:
                logger.error(f"Failed to get funding rate history for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting funding rate history for {symbol}: {str(e)}")
            return []

    def get_open_interest_history(self, symbol: str, interval: str = '1h', limit: int = 200) -> List[Dict]:
        """Get open interest history for market sentiment analysis"""
        try:
            if not self.client:
                logger.error("Bybit client not initialized")
                return []

            response = self.client.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime=interval
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                if result and result.get('list'):
                    oi_history = []
                    for oi_data in result['list']:
                        oi_record = {
                            'symbol': oi_data.get('symbol'),
                            'timestamp': int(oi_data.get('timestamp', 0)),
                            'open_interest': float(oi_data.get('openInterest', 0)),
                            'interval': interval
                        }
                        oi_history.append(oi_record)

                    # Calculate OI trend
                    if len(oi_history) >= 2:
                        recent_oi = oi_history[0]['open_interest']
                        previous_oi = oi_history[1]['open_interest']
                        oi_change = (recent_oi - previous_oi) / previous_oi * 100 if previous_oi > 0 else 0

                        for record in oi_history:
                            record['oi_change_pct'] = oi_change
                            record['oi_trend'] = "increasing" if oi_change > 0 else "decreasing"

                    logger.info(f"Retrieved {len(oi_history)} open interest records for {symbol}")
                    return oi_history
                else:
                    logger.warning(f"No open interest history found for {symbol}")
                    return []
            else:
                logger.error(f"Failed to get open interest history for {symbol}: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting open interest history for {symbol}: {str(e)}")
            return []

    def get_enhanced_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data with all enhanced features"""
        try:
            # Get basic market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                logger.error(f"Failed to get basic market data for {symbol}")
                return {}

            # Get enhanced data
            order_book = self.get_order_book_data(symbol)
            liquidations = self.get_liquidation_data(symbol)
            funding_history = self.get_funding_rate_history(symbol, 10)  # Last 10 funding periods
            oi_history = self.get_open_interest_history(symbol, '1h', 24)  # Last 24 hours

            # Analyze liquidation proximity
            liquidation_risk = "LOW"
            nearby_liquidations = 0
            current_price = float(market_data.get('lastPrice', 0))

            for liquidation in liquidations:
                liquidation_price = liquidation['price']
                price_diff_pct = abs(current_price - liquidation_price) / current_price * 100
                if price_diff_pct < 2.0:  # Within 2% of current price
                    nearby_liquidations += 1

            if nearby_liquidations > 5:
                liquidation_risk = "HIGH"
            elif nearby_liquidations > 2:
                liquidation_risk = "MEDIUM"

            # Analyze funding sentiment
            funding_sentiment = "NEUTRAL"
            if funding_history:
                latest_funding = funding_history[0]['funding_rate']
                if latest_funding > 0.01:  # High positive funding
                    funding_sentiment = "BULLISH"
                elif latest_funding < -0.01:  # High negative funding
                    funding_sentiment = "BEARISH"

            # Analyze OI trend
            oi_sentiment = "NEUTRAL"
            if oi_history:
                latest_oi = oi_history[0].get('oi_change_pct', 0)
                if latest_oi > 5:
                    oi_sentiment = "BULLISH"
                elif latest_oi < -5:
                    oi_sentiment = "BEARISH"

            enhanced_data = {
                **market_data,  # Original market data

                # Enhanced liquidity analysis
                'order_book_depth': order_book,
                'liquidity_score': order_book.get('liquidity_score', 0) if order_book else 0,
                'spread_pct': order_book.get('spread_pct', 0) if order_book else 0,

                # Liquidation analysis
                'liquidation_risk': liquidation_risk,
                'nearby_liquidations': nearby_liquidations,
                'recent_liquidations': liquidations[:5],  # Last 5 liquidations

                # Sentiment analysis
                'funding_sentiment': funding_sentiment,
                'funding_rate': funding_history[0]['funding_rate'] if funding_history else 0,
                'oi_sentiment': oi_sentiment,
                'oi_change_pct': oi_history[0].get('oi_change_pct', 0) if oi_history else 0,

                # Risk metrics
                'market_risk_score': self._calculate_market_risk(
                    liquidation_risk, order_book.get('liquidity_score', 0) if order_book else 0
                ),

                # Timestamps
                'enhanced_data_timestamp': int(time.time() * 1000)
            }

            logger.info(f"✅ Enhanced market data collected for {symbol} (risk: {liquidation_risk}, liquidity: {enhanced_data['liquidity_score']:.2f})")
            return enhanced_data

        except Exception as e:
            logger.error(f"Error getting enhanced market data for {symbol}: {str(e)}")
            return {}

    def _calculate_market_risk(self, liquidation_risk: str, liquidity_score: float) -> float:
        """Calculate overall market risk score (0-1, higher = riskier)"""
        try:
            risk_score = 0.5  # Base risk

            # Liquidation risk adjustment
            if liquidation_risk == "HIGH":
                risk_score += 0.3
            elif liquidation_risk == "MEDIUM":
                risk_score += 0.15

            # Liquidity risk adjustment (lower liquidity = higher risk)
            liquidity_risk = (1.0 - liquidity_score) * 0.3
            risk_score += liquidity_risk

            return min(risk_score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating market risk: {str(e)}")
            return 0.5
"""
Bybit API v5 Client - Unified Account Perpetual Futures Trading
Production-ready client for live trading with 50-75x leverage
"""

from pybit.unified_trading import HTTP
import logging
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
                log_requests=False
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
        """Place an order on the perpetual futures market"""
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

            response = self.client.place_order(**order_params)

            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Order placed successfully: {symbol} {side} {qty} @ {order_type}")
                return result
            else:
                logger.error(f"❌ Failed to place order: {response.get('retMsg', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
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
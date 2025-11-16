import logging
from typing import Dict, List, Optional
from pybit.unified_trading import HTTP
from config import Config

logger = logging.getLogger(__name__)

class BybitAPIException(Exception):
    """Custom exception for Bybit API errors"""
    pass

class BybitClient:
    def __init__(self):
        self.config = Config()
        self.client = None
        
    def _get_client(self):
        """Initialize or return the Bybit HTTP client"""
        if self.client is None:
            self.client = HTTP(
                testnet=False,  # LIVE TRADING - no testnet
                api_key=self.config.BYBIT_API_KEY,
                api_secret=self.config.BYBIT_API_SECRET
            )
        return self.client

    def get_account_balance(self) -> Dict:
        """Get unified account balance - LIVE TRADING"""
        try:
            client = self._get_client()
            balance = client.get_wallet_balance(accountType="UNIFIED")
            
            if balance and balance.get('retCode') == 0:
                result = balance.get('result', {}).get('list', [])
                if result:
                    account_data = result[0]
                    logger.info(f"✅ LIVE TRADING CONNECTED - Total Equity: ${account_data.get('totalEquity', '0')}")
                    logger.info(f"✅ Available Balance: ${account_data.get('totalAvailableBalance', '0')}")
                    return account_data
                else:
                    logger.error("No account data returned")
                    raise BybitAPIException("No account data returned")
            else:
                error_msg = balance.get('retMsg', 'Unknown error') if balance else 'No response'
                logger.error(f"❌ API Error: {error_msg}")
                raise BybitAPIException(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Exception getting account balance: {str(e)}")
            raise BybitAPIException(f"Exception getting account balance: {str(e)}") from e

    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data for perpetual futures - LIVE TRADING"""
        try:
            client = self._get_client()
            # Get tickers for linear perpetual markets
            response = client.get_tickers(category="linear", symbol=symbol)
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {}).get('list', [])
                if result:
                    market_data = result[0]
                    logger.info(f"✅ Got live market data for {symbol}: ${market_data.get('lastPrice', '0')}")
                    return market_data
                else:
                    logger.warning(f"No market data for {symbol}")
                    raise BybitAPIException(f"No market data for {symbol}")
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Market data API Error: {error_msg}")
                raise BybitAPIException(f"Market data API Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception getting market data for {symbol}: {str(e)}")
            raise BybitAPIException(f"Exception getting market data for {symbol}: {str(e)}") from e

    def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for perpetual futures - LIVE TRADING"""
        try:
            client = self._get_client()
            response = client.get_funding_rate_history(category="linear", symbol=symbol, limit=1)
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {}).get('list', [])
                if result:
                    funding_data = result[0]
                    logger.info(f"✅ Funding rate for {symbol}: {funding_data.get('fundingRate', '0')}")
                    return funding_data
                else:
                    logger.warning(f"No funding data for {symbol}")
                    raise BybitAPIException(f"No funding data for {symbol}")
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Funding rate API Error: {error_msg}")
                raise BybitAPIException(f"Funding rate API Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception getting funding rate for {symbol}: {str(e)}")
            raise BybitAPIException(f"Exception getting funding rate for {symbol}: {str(e)}") from e

    def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 200) -> List[Dict]:
        """Get kline/candlestick data for technical analysis - LIVE TRADING"""
        try:
            client = self._get_client()
            
            # Map intervals to Bybit format
            interval_map = {
                '1h': '60',
                '4h': '240', 
                '1D': 'D',
                '1W': 'W'
            }
            bybit_interval = interval_map.get(interval, interval)
            
            response = client.get_kline(
                category="linear",
                symbol=symbol,
                interval=bybit_interval,
                limit=limit
            )
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {}).get('list', [])
                logger.info(f"✅ Got {len(result)} klines for {symbol} {interval}")
                return result
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Kline data API Error: {error_msg}")
                raise BybitAPIException(f"Kline data API Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception getting kline data for {symbol}: {str(e)}")
            raise BybitAPIException(f"Exception getting kline data for {symbol}: {str(e)}") from e

    def place_order(self, symbol: str, side: str, order_type: str, qty: float,
                    price: float = None, time_in_force: str = "GTC",
                    reduce_only: bool = False, position_idx: int = 0) -> Dict:
        """Place order on perpetual futures - LIVE TRADING"""
        try:
            client = self._get_client()
            
            order_params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'timeInForce': time_in_force,
                'positionIdx': position_idx,  # 0 for one-way mode
                'reduceOnly': reduce_only
            }
            
            if price and order_type == 'Limit':
                order_params['price'] = str(price)
                
            response = client.place_order(**order_params)
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                logger.info(f"✅ Order placed successfully: {result.get('orderId', 'Unknown')}")
                logger.info(f"✅ {side} {qty} {symbol} @ {price or 'Market'}")
                return result
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Order placement failed: {error_msg}")
                raise BybitAPIException(f"Order placement failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception placing order: {str(e)}")
            raise BybitAPIException(f"Exception placing order: {str(e)}") from e

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for perpetual futures - LIVE TRADING"""
        try:
            client = self._get_client()
            
            response = client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response and response.get('retCode') == 0:
                logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
                return True
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Failed to set leverage: {error_msg}")
                raise BybitAPIException(f"Failed to set leverage: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception setting leverage: {str(e)}")
            raise BybitAPIException(f"Exception setting leverage: {str(e)}") from e

    def get_position_info(self, symbol: str) -> Dict:
        """Get current position information - LIVE TRADING"""
        try:
            client = self._get_client()
            
            response = client.get_positions(category="linear", symbol=symbol)
            
            if response and response.get('retCode') == 0:
                positions = response.get('result', {}).get('list', [])
                # Filter for active positions (size != 0)
                active_positions = [pos for pos in positions if float(pos.get('size', '0')) != 0]
                
                if active_positions:
                    position = active_positions[0]
                    logger.info(f"✅ Position found: {position.get('side', 'Unknown')} {position.get('size', '0')} {symbol}")
                    return position
                else:
                    logger.info(f"No active position for {symbol}")
                    return None
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Position info API Error: {error_msg}")
                raise BybitAPIException(f"Position info API Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception getting position info: {str(e)}")
            raise BybitAPIException(f"Exception getting position info: {str(e)}") from e

    def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest for perpetual futures - LIVE TRADING"""
        try:
            client = self._get_client()
            
            response = client.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="15min",
                limit=1
            )
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {}).get('list', [])
                if result:
                    oi_data = result[0]
                    logger.info(f"✅ Open interest for {symbol}: {oi_data.get('openInterest', '0')}")
                    return oi_data
                else:
                    logger.warning(f"No open interest data for {symbol}")
                    raise BybitAPIException(f"No open interest data for {symbol}")
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ Open interest API Error: {error_msg}")
                raise BybitAPIException(f"Open interest API Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Exception getting open interest for {symbol}: {str(e)}")
            raise BybitAPIException(f"Exception getting open interest for {symbol}: {str(e)}") from e

    def _get_default_balance(self) -> Dict:
        """Return default balance structure"""
        return {
            'accountType': 'UNIFIED',
            'totalWalletBalance': '0',
            'totalEquity': '0',
            'totalMarginBalance': '0', 
            'totalAvailableBalance': '0',
            'coin': []
        }

    def _get_default_market_data(self, symbol: str) -> Dict:
        """Return default market data structure"""
        return {
            'symbol': symbol,
            'lastPrice': '0',
            'markPrice': '0',
            'price24hPcnt': '0',
            'fundingRate': '0',
            'volume24h': '0',
            'turnover24h': '0',
            'highPrice24h': '0',
            'lowPrice24h': '0'
        }

    def test_connection(self) -> bool:
        """Test API connection - returns True if connected to LIVE TRADING"""
        try:
            client = self._get_client()
            response = client.get_account_info()
            
            if response and response.get('retCode') == 0:
                logger.info("✅ LIVE TRADING API CONNECTION SUCCESSFUL")
                return True
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logger.error(f"❌ API CONNECTION FAILED: {error_msg}")
                raise BybitAPIException(f"API CONNECTION FAILED: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ API CONNECTION EXCEPTION: {str(e)}")
            raise BybitAPIException(f"API CONNECTION EXCEPTION: {str(e)}") from e

    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook depth for maker order placement"""
        try:
            client = self._get_client()
            
            response = client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                bids = result.get('b', [])  # Bybit uses 'b' for bids
                asks = result.get('a', [])  # Bybit uses 'a' for asks
                
                return {
                    'bids': [[float(b[0]), float(b[1])] for b in bids],
                    'asks': [[float(a[0]), float(a[1])] for a in asks]
                }
            else:
                logger.error(f"Failed to get orderbook: {response.get('retMsg') if response else 'No response'}")
                raise BybitAPIException(f"Failed to get orderbook: {response.get('retMsg') if response else 'No response'}")
                
        except Exception as e:
            logger.error(f"Exception getting orderbook: {str(e)}")
            raise BybitAPIException(f"Exception getting orderbook: {str(e)}") from e

    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Check order status for maker fill confirmation"""
        try:
            client = self._get_client()
            
            response = client.get_open_orders(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                orders = result.get('list', [])
                if orders:
                    return orders[0]  # Return first matching order
            
            # If not in open orders, check history
            response = client.get_order_history(
                category="linear",
                symbol=symbol,
                orderId=order_id,
                limit=1
            )
            
            if response and response.get('retCode') == 0:
                result = response.get('result', {})
                orders = result.get('list', [])
                if orders:
                    return orders[0]
            
            return {}
                
        except Exception as e:
            logger.error(f"Exception getting order status: {str(e)}")
            raise BybitAPIException(f"Exception getting order status: {str(e)}") from e

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel unfilled limit order"""
        try:
            client = self._get_client()
            
            response = client.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            
            if response and response.get('retCode') == 0:
                logger.info(f"✅ Order {order_id} cancelled")
                return True
            else:
                logger.warning(f"Failed to cancel order: {response.get('retMsg') if response else 'No response'}")
                raise BybitAPIException(f"Failed to cancel order: {response.get('retMsg') if response else 'No response'}")
                
        except Exception as e:
            logger.error(f"Exception canceling order: {str(e)}")
            raise BybitAPIException(f"Exception canceling order: {str(e)}") from e

    def get_instrument_info(self, symbol: str) -> Dict:
        """Get instrument metadata (min qty, step, notional, leverage) for linear perps."""
        try:
            client = self._get_client()

            response = client.get_instruments_info(
                category="linear",
                symbol=symbol
            )

            if response and response.get('retCode') == 0:
                result = response.get('result', {}).get('list', [])
                if not result:
                    logger.warning(f"No instrument info for {symbol}")
                    raise BybitAPIException(f"No instrument info for {symbol}")

                info = result[0]
                lot = info.get('lotSizeFilter', {}) or {}
                lev = info.get('leverageFilter', {}) or {}

                # Flatten into the keys expected by _get_instrument_specs
                return {
                    'minOrderQty': lot.get('minOrderQty', '0'),
                    'qtyStep': lot.get('qtyStep', '0'),
                    'minNotionalValue': info.get('minNotionalValue', '0'),
                    'maxLeverage': lev.get('maxLeverage', str(Config.MAX_LEVERAGE)),
                }

            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            logger.error(f"❌ Instrument info API Error for {symbol}: {error_msg}")
            raise BybitAPIException(f"Instrument info API Error for {symbol}: {error_msg}")

        except Exception as e:
            logger.error(f"❌ Exception getting instrument info for {symbol}: {str(e)}")
            raise BybitAPIException(f"Exception getting instrument info for {symbol}: {str(e)}") from e
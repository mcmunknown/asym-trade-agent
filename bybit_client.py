import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import logging
from typing import Dict, List, Optional
from config import Config

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self):
        self.api_key = Config.BYBIT_API_KEY
        self.api_secret = Config.BYBIT_API_SECRET
        self.base_url = Config.BYBIT_BASE_URL
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, payload: str) -> str:
        return hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self) -> Dict[str, str]:
        timestamp = int(time.time() * 1000)
        recv_window = 5000
        # Bybit v5 API signature format: timestamp + apiKey + recvWindow
        payload = f"{timestamp}{self.api_key}{recv_window}"
        signature = self._generate_signature(payload)

        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': str(timestamp),
            'X-BAPI-SIGN': signature,
            'X-BAPI-RECV-WINDOW': str(recv_window),
            'Content-Type': 'application/json'
        }

    async def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data for a symbol"""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'limit': 1
            }

            async with self.session.get(url, params=params) as response:
                data = await response.json()
                if data['retCode'] == 0 and 'result' in data and 'list' in data['result']:
                    if data['result']['list']:
                        return data['result']['list'][0]
                logger.warning(f"No market data available for {symbol}")
                # Return default structure to prevent downstream errors
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
        except Exception as e:
            logger.error(f"Exception getting market data for {symbol}: {str(e)}")
            # Return default structure to prevent downstream errors
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

    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for a symbol"""
        try:
            url = f"{self.base_url}/v5/market/funding/history"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'limit': 1
            }

            async with self.session.get(url, params=params) as response:
                data = await response.json()
                if data['retCode'] == 0:
                    return data['result']['list'][0]
                else:
                    logger.error(f"Error getting funding rate for {symbol}: {data['retMsg']}")
                    return None
        except Exception as e:
            logger.error(f"Exception getting funding rate for {symbol}: {str(e)}")
            return None

    async def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest data for a symbol"""
        try:
            url = f"{self.base_url}/v5/market/open-interest"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'intervalTime': '15min',  # Required parameter (Fixed: intervalTime Is Required error)
                'limit': 1
            }

            async with self.session.get(url, params=params) as response:
                data = await response.json()
                if data['retCode'] == 0 and 'result' in data and 'list' in data['result']:
                    if data['result']['list']:
                        return data['result']['list'][0]
                logger.warning(f"No open interest data for {symbol}")
                return {'openInterest': '0'}
        except Exception as e:
            logger.error(f"Exception getting open interest for {symbol}: {str(e)}")
            return {'openInterest': '0'}

    async def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 200) -> List[Dict]:
        """Get kline/candlestick data for technical analysis - FIXED VERSION"""
        try:
            url = f"{self.base_url}/v5/market/kline"

            # Map intervals to Bybit format
            interval_map = {
                '1h': '60',
                '4h': '240',
                '1D': 'D',
                '1W': 'W'
            }
            bybit_interval = interval_map.get(interval, interval)

            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'limit': limit
            }

            logger.debug(f"Fetching kline data for {symbol} {interval} (mapped to {bybit_interval}), limit={limit}")
            logger.debug(f"API URL: {url}, params: {params}")

            async with self.session.get(url, params=params) as response:
                data = await response.json()
                logger.debug(f"Kline API response for {symbol}: {data}")

                if data['retCode'] == 0:
                    result_list = data.get('result', {}).get('list', [])
                    logger.info(f"Got {len(result_list)} klines for {symbol} {interval}")

                    # Convert kline data format if needed
                    if result_list:
                        # Bybit returns [timestamp, open, high, low, close, volume, turnover]
                        # Make sure the data is in the right format
                        formatted_list = []
                        for kline in result_list:
                            if isinstance(kline, list) and len(kline) >= 6:
                                formatted_list.append(kline)
                        return formatted_list
                    return result_list
                else:
                    logger.error(f"Error getting kline data for {symbol}: {data['retMsg']} (code: {data['retCode']})")
                    return []
        except Exception as e:
            logger.error(f"Exception getting kline data for {symbol}: {str(e)}")
            return []

    async def place_order(self, symbol: str, side: str, order_type: str, qty: float,
                        price: float = None, time_in_force: str = "GoodTillCancel",
                        reduce_only: bool = False, close_on_trigger: bool = False) -> Dict:
        """Place an order"""
        try:
            url = f"{self.base_url}/v5/order/create"
            headers = self._get_headers()

            payload = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'timeInForce': time_in_force,
                'reduceOnly': reduce_only,
                'closeOnTrigger': close_on_trigger
            }

            if price:
                payload['price'] = str(price)

            async with self.session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
                if data['retCode'] == 0:
                    logger.info(f"Order placed successfully: {data}")
                    return data['result']
                else:
                    logger.error(f"Error placing order: {data['retMsg']}")
                    return None
        except Exception as e:
            logger.error(f"Exception placing order: {str(e)}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            url = f"{self.base_url}/v5/position/set-leverage"
            headers = self._get_headers()

            payload = {
                'category': 'linear',
                'symbol': symbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
                if data['retCode'] == 0:
                    logger.info(f"Leverage set to {leverage}x for {symbol}")
                    return True
                else:
                    logger.error(f"Error setting leverage: {data['retMsg']}")
                    return False
        except Exception as e:
            logger.error(f"Exception setting leverage: {str(e)}")
            return False

    async def get_position_info(self, symbol: str) -> Dict:
        """Get current position information"""
        try:
            url = f"{self.base_url}/v5/position/list"
            headers = self._get_headers()
            params = {
                'category': 'linear',
                'symbol': symbol
            }

            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                if data['retCode'] == 0:
                    positions = data['result']['list']
                    if positions:
                        return positions[0]  # Return first position if exists
                    else:
                        return None
                else:
                    logger.error(f"Error getting position info: {data['retMsg']}")
                    return None
        except Exception as e:
            logger.error(f"Exception getting position info: {str(e)}")
            return None

    async def get_account_balance(self) -> Dict:
        """Get account balance information"""
        try:
            url = f"{self.base_url}/v5/account/wallet-balance"
            headers = self._get_headers()
            params = {
                'accountType': 'UNIFIED'
            }

            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                if data['retCode'] == 0 and 'result' in data and 'list' in data['result']:
                    if data['result']['list']:
                        return data['result']['list'][0]
                logger.warning("No account balance data available")
                # Return default balance structure
                return {
                    'accountType': 'UNIFIED',
                    'totalWalletBalance': '0',
                    'totalEquity': '0',
                    'totalMarginBalance': '0',
                    'totalAvailableBalance': '0',
                    'coin': []
                }
        except Exception as e:
            logger.error(f"Exception getting account balance: {str(e)}")
            # Return default balance structure
            return {
                'accountType': 'UNIFIED',
                'totalWalletBalance': '0',
                'totalEquity': '0',
                'totalMarginBalance': '0',
                'totalAvailableBalance': '0',
                'coin': []
            }
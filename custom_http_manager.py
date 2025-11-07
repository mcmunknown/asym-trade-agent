from pybit._http_manager import _V5HTTPManager
from pybit import _helpers
import time
import logging

logger = logging.getLogger(__name__)

class CustomV5HTTPManager(_V5HTTPManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_offset = 0
        self.sync_time()

    def sync_time(self):
        """Fetches the server time and calculates the time offset."""
        try:
            response = self._submit_request(
                method="GET",
                path=f"{self.endpoint}/v5/market/time",
                auth=False
            )
            server_time = int(response["result"]["timeSecond"]) * 1000
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            logger.info(f"Time offset calculated: {self.time_offset}ms")
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")

    def get_server_time(self):
        """Fetches the server time from Bybit."""
        try:
            return self._submit_request(
                method="GET",
                path=f"{self.endpoint}/v5/market/time",
                auth=False
            )
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return None

    def get_wallet_balance(self, **params):
        """Fetches the wallet balance from Bybit."""
        try:
            response = self._submit_request(
                method="GET",
                path=f"{self.endpoint}/v5/account/wallet-balance",
                query=params,
                auth=True
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get wallet balance: {e}")
            return None

    def _prepare_headers(self, payload, recv_window):
        """Prepare headers for authenticated request."""
        timestamp = _helpers.generate_timestamp() + self.time_offset
        signature = self._auth(payload=payload, recv_window=recv_window, timestamp=timestamp)
        return {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(recv_window),
        }

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
            response = self.client.get(f"{self.endpoint}/v5/market/time")
            response.raise_for_status()
            server_time = int(response.json()["result"]["timeSecond"]) * 1000
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            logger.info(f"Time offset calculated: {self.time_offset}ms")
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")

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

"""
Custom HTTP manager that extends pybit's unified trading HTTP client while
tracking server time offset for logging/diagnostics.
"""

import time
import logging
from pybit.unified_trading import HTTP as PybitHTTP

logger = logging.getLogger(__name__)


class CustomV5HTTPManager(PybitHTTP):
    """
    Thin wrapper around pybit.unified_trading.HTTP that logs clock skew
    between the local machine and Bybit so we can troubleshoot auth errors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_offset_ms = 0
        self.sync_time()

    def sync_time(self):
        """Fetch Bybit server time and compute local offset for monitoring."""
        try:
            response = super().get_server_time()
            if response and response.get("retCode") == 0:
                server_sec = int(response["result"]["timeSecond"])
                server_ms = server_sec * 1000
                local_ms = int(time.time() * 1000)
                self.time_offset_ms = server_ms - local_ms
                logger.info(f"Time offset calculated: {self.time_offset_ms}ms")
            else:
                logger.warning(
                    "Failed to sync time: %s",
                    (response or {}).get("retMsg", "Unknown error"),
                )
        except Exception as exc:
            logger.error(f"Failed to sync time: {exc}")

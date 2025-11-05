#!/usr/bin/env python3
"""
WebSocket connectivity test for Anne's calculus trading system using official pybit API
"""

import logging
from time import sleep
from pybit.unified_trading import WebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_websocket_connection():
    """Test basic WebSocket connectivity and data reception using official pybit API."""
    print('🔌 Testing WebSocket Connectivity with Official pybit API...')

    # Track data reception
    data_received = []

    def handle_message(message):
        """Handle incoming WebSocket messages."""
        data_received.append(message)
        print(f'✅ Received data message #{len(data_received)}')

        # Show sample trade data if available
        if 'data' in message and message['data']:
            trade = message['data'][0]  # First trade in the batch
            print(f'   Sample trade: Price={trade.get("p", "N/A")}, Size={trade.get("s", "N/A")}, Side={trade.get("S", "N/A")}')

    try:
        print('Initializing Bybit WebSocket connection...')

        # Initialize WebSocket with official pybit API
        ws = WebSocket(
            testnet=False,  # Use mainnet for real data
            channel_type="linear"  # Linear perpetual contracts
        )

        print('Subscribing to BTCUSDT trades...')

        # Subscribe to trade stream for BTCUSDT
        ws.trade_stream(
            symbol="BTCUSDT",
            callback=handle_message
        )

        print('✅ WebSocket connected and subscribed successfully')
        print('Waiting for trade data...')

        # Run for 15 seconds to collect data
        for i in range(15):
            sleep(1)
            if i % 5 == 0:  # Progress indicator every 5 seconds
                print(f'   Listening... {i+1}/15 seconds')

        # Test results
        if data_received:
            print(f'🎉 WebSocket test successful!')
            print(f'   Total messages received: {len(data_received)}')

            # Calculate statistics
            total_trades = sum(len(msg.get('data', [])) for msg in data_received)
            print(f'   Total trades processed: {total_trades}')

            # Show sample of first message
            print(f'   First message structure: {list(data_received[0].keys())}')
        else:
            print('⚠️  No data received in 15 seconds')
            print('   This could be normal if market is closed or no recent trades')

        print('✅ WebSocket connectivity test completed')

    except Exception as e:
        print(f'❌ WebSocket test failed: {e}')
        logger.exception('WebSocket test error')

if __name__ == "__main__":
    test_websocket_connection()
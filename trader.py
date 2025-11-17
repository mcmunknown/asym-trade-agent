#!/usr/bin/env python3
"""
🔥 TURBO MONEY MAKER - Clean rebuild from scratch
$18 → $28 in 10 minutes

NO gates. NO filters. NO academic bullshit.
Just: Signal → Execute → Profit.
"""

import time
import os
from collections import deque
from datetime import datetime
from bybit_client import BybitClient
from websocket_client import WebSocketDataClient

# ============================================================================
# CONFIGURATION
# ============================================================================

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LEVERAGE = 50
POSITION_PCT = 0.40  # Use 40% of balance per trade
TP_PCT = 0.01  # 1% take profit
SL_PCT = 0.005  # 0.5% stop loss
MIN_VELOCITY = 0.002  # 0.2% minimum move
MAX_HOLD_SECONDS = 1800  # 30 min max hold
CHECK_INTERVAL = 10  # Check every 10 seconds
PRICE_HISTORY_SIZE = 50

# ============================================================================
# TURBO TRADER
# ============================================================================

class TurboTrader:
    def __init__(self):
        print("🔥 TURBO MONEY MAKER - INITIALIZING\n")

        # Bybit client
        self.bybit = BybitClient(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=TESTNET
        )

        # WebSocket for real-time prices
        self.ws = WebSocketDataClient(
            symbols=SYMBOLS,
            callback=self.on_price_update
        )

        # Price tracking
        self.prices = {symbol: deque(maxlen=PRICE_HISTORY_SIZE) for symbol in SYMBOLS}

        # Positions
        self.positions = {}

        # Stats
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.start_balance = 0.0
        self.start_time = time.time()

        print(f"⚡ Leverage: {LEVERAGE}x FIXED")
        print(f"💰 Position Size: {POSITION_PCT*100}% of balance")
        print(f"🎯 TP: {TP_PCT*100}% | SL: {SL_PCT*100}%")
        print(f"📊 Symbols: {', '.join(SYMBOLS)}")
        print(f"🚀 NO GATES. NO FILTERS. INSTANT EXECUTION.\n")

    def on_price_update(self, symbol, price):
        """Store real-time price updates."""
        self.prices[symbol].append(float(price))

    def get_balance(self):
        """Get available balance."""
        try:
            result = self.bybit.get_account_balance()
            return float(result.get('totalAvailableBalance', 0))
        except:
            return 0

    def calculate_velocity(self, symbol):
        """
        Velocity = % change over last 10 prices
        Simple and effective.
        """
        prices = self.prices[symbol]
        if len(prices) < 10:
            return 0

        return (prices[-1] - prices[-10]) / prices[-10]

    def generate_signal(self, symbol):
        """
        SIMPLE SIGNAL:
        - velocity > 0.2% = BUY
        - velocity < -0.2% = SELL
        - else = NOTHING
        """
        if symbol in self.positions:
            return None  # Already have position

        velocity = self.calculate_velocity(symbol)

        if abs(velocity) < MIN_VELOCITY:
            return None

        price = self.prices[symbol][-1]
        direction = "Buy" if velocity > 0 else "Sell"

        return {
            "symbol": symbol,
            "direction": direction,
            "velocity": velocity,
            "price": price
        }

    def calculate_position_size(self, symbol, price):
        """Fixed sizing: 40% balance, 50x leverage."""
        balance = self.get_balance()

        # Margin = 40% of balance
        margin = balance * POSITION_PCT

        # Notional = margin * leverage
        notional = margin * LEVERAGE

        # Quantity
        qty = notional / price

        # Round to exchange requirements
        if symbol == "BTCUSDT":
            qty = round(qty, 3)
        elif symbol == "ETHUSDT":
            qty = round(qty, 2)
        else:
            qty = round(qty, 1)

        return qty, notional

    def calculate_tp_sl(self, direction, price):
        """Fixed TP/SL."""
        if direction == "Buy":
            tp = price * (1 + TP_PCT)
            sl = price * (1 - SL_PCT)
        else:
            tp = price * (1 - TP_PCT)
            sl = price * (1 + SL_PCT)

        return tp, sl

    def execute_trade(self, signal):
        """
        EXECUTE IMMEDIATELY.
        No gates, no filters, no checks.
        """
        symbol = signal["symbol"]
        direction = signal["direction"]
        price = signal["price"]
        velocity = signal["velocity"]

        # Position sizing
        qty, notional = self.calculate_position_size(symbol, price)
        tp, sl = self.calculate_tp_sl(direction, price)

        # Validate minimum ($5 Bybit requirement)
        if notional < 5.0:
            return

        print(f"\n{'='*70}")
        print(f"🚀 EXECUTING: {symbol} {direction}")
        print(f"{'='*70}")
        print(f"📊 Qty: {qty:.6f} @ ${price:.2f} | Velocity: {velocity*100:+.2f}%")
        print(f"💰 Notional: ${notional:.2f} | Leverage: {LEVERAGE}x")
        print(f"🎯 TP: ${tp:.2f} | SL: ${sl:.2f}")
        print(f"{'='*70}")

        # Set leverage
        try:
            self.bybit.set_leverage(symbol, LEVERAGE)
        except Exception as e:
            print(f"⚠️  Leverage set failed: {e}")

        # Place order
        try:
            order = self.bybit.place_order(
                symbol=symbol,
                side=direction,
                order_type="Market",
                qty=qty,
                take_profit=tp,
                stop_loss=sl
            )

            if order:
                self.positions[symbol] = {
                    "direction": direction,
                    "entry_price": price,
                    "qty": qty,
                    "tp": tp,
                    "sl": sl,
                    "entry_time": time.time(),
                    "order_id": order.get('orderId')
                }

                self.trades += 1
                print(f"✅ EXECUTED - Trade #{self.trades}")
                print(f"   Order ID: {order.get('orderId')}\n")
            else:
                print(f"❌ ORDER FAILED\n")

        except Exception as e:
            print(f"❌ EXECUTION ERROR: {e}\n")

    def monitor_positions(self):
        """Close positions at max hold time."""
        current_time = time.time()

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            elapsed = current_time - pos["entry_time"]

            if elapsed > MAX_HOLD_SECONDS:
                print(f"⏰ MAX HOLD TIME - Closing {symbol}")
                self.close_position(symbol, "max_hold")

    def close_position(self, symbol, reason):
        """Manually close position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        close_side = "Sell" if pos["direction"] == "Buy" else "Buy"

        try:
            order = self.bybit.place_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=pos["qty"]
            )

            if order:
                print(f"🔄 CLOSED {symbol} - Reason: {reason}\n")
                del self.positions[symbol]
        except Exception as e:
            print(f"❌ CLOSE ERROR {symbol}: {e}\n")

    def print_status(self):
        """Beautiful terminal UI."""
        balance = self.get_balance()
        session_time = (time.time() - self.start_time) / 60
        hourly_rate = ((balance - self.start_balance) / self.start_balance * 100) / (session_time / 60) if session_time > 0 else 0

        win_rate = (self.wins / max(self.trades, 1)) * 100
        pnl = balance - self.start_balance
        pnl_pct = (pnl / self.start_balance * 100) if self.start_balance > 0 else 0

        target = 28.0
        to_target = ((target - balance) / (target - self.start_balance) * 100) if self.start_balance > 0 else 0

        print(f"\n╔{'='*66}╗")
        print(f"║  🔥 TURBO MONEY MAKER - ${self.start_balance:.2f} → ${target:.2f} CHALLENGE{' '*14}║")
        print(f"╠{'='*66}╣")
        print(f"║  💰 Balance:      ${balance:.2f} → ${balance + pnl:.2f} ({pnl:+.2f} | {pnl_pct:+.1f}%){' '*(20 - len(f'{pnl:.2f}'))}║")
        print(f"║  🎯 Target:       ${target:.2f} ({to_target:.1f}% to go){' '*(31 - len(f'{to_target:.1f}'))}║")
        print(f"║  ⚡ Leverage:     {LEVERAGE}x FIXED{' '*46}║")
        print(f"║{' '*66}║")
        print(f"║  📊 PERFORMANCE:{' '*50}║")
        print(f"║     Trades:       {self.trades} ({self.wins}W / {self.losses}L) | Win Rate: {win_rate:.1f}%{' '*(20 - len(str(self.trades)))}║")
        print(f"║{' '*66}║")

        if self.positions:
            print(f"║  🔥 OPEN POSITIONS ({len(self.positions)}):{' '*43}║")
            for sym, pos in list(self.positions.items())[:3]:
                elapsed = (time.time() - pos["entry_time"]) / 60
                current_price = self.prices[sym][-1] if self.prices[sym] else pos["entry_price"]
                pnl_pos = ((current_price - pos["entry_price"]) / pos["entry_price"] * 100) if pos["direction"] == "Buy" else ((pos["entry_price"] - current_price) / pos["entry_price"] * 100)
                print(f"║     {sym:8s}  {pos['direction']:4s} {pos['qty']:.4f} @ ${pos['entry_price']:.0f} | {pnl_pos:+.1f}% | {elapsed:.0f}min{' '*(15 - len(f'{pos['entry_price']:.0f}'))}║")
        else:
            print(f"║  💤 No open positions{' '*43}║")

        print(f"║{' '*66}║")
        print(f"║  ⏱️  Session: {session_time:.1f} min | Hourly Rate: {hourly_rate:+.1f}%/hr{' '*(25 - len(f'{session_time:.1f}'))}║")
        print(f"╚{'='*66}╝\n")

    def run(self):
        """Main trading loop."""
        print("🚀 STARTING TURBO MONEY MAKER\n")

        # Connect WebSocket
        self.ws.connect()

        # Wait for price data
        print("⏳ Waiting for price data...")
        while len(self.prices["BTCUSDT"]) < 20:
            time.sleep(1)

        print("✅ Price data ready - STARTING TRADING\n")

        # Get starting balance
        self.start_balance = self.get_balance()
        print(f"💰 Starting balance: ${self.start_balance:.2f}\n")

        last_status = time.time()

        while True:
            try:
                # Generate and execute signals
                for symbol in SYMBOLS:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.execute_trade(signal)

                # Monitor positions
                self.monitor_positions()

                # Print status every 60 seconds
                if time.time() - last_status > 60:
                    self.print_status()
                    last_status = time.time()

                # Sleep
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 STOPPING...")
                self.print_status()
                break
            except Exception as e:
                print(f"❌ ERROR: {e}")
                time.sleep(5)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    trader = TurboTrader()
    trader.run()

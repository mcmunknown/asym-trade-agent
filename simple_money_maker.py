#!/usr/bin/env python3
"""
🔥 SIMPLE MONEY MAKER - $18 → $28 in 10 minutes

NO academic bullshit. NO 40 gates. NO filters.
Just: Signal → Execute → Profit.

Target: 10-15 trades/hour @ 0.5-1% avg = $10/hour minimum
"""

import time
import os
from collections import deque
from bybit_client import BybitClient
from websocket_client import WebSocketDataClient
from config import Config

class SimpleMoneyMaker:
    def __init__(self):
        self.bybit = BybitClient(
            api_key=Config.BYBIT_API_KEY,
            api_secret=Config.BYBIT_API_SECRET,
            testnet=Config.BYBIT_TESTNET
        )

        self.ws = WebSocketDataClient(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            callback=self.on_price_update
        )

        # Price tracking (last 50 prices per symbol)
        self.prices = {
            "BTCUSDT": deque(maxlen=50),
            "ETHUSDT": deque(maxlen=50),
            "SOLUSDT": deque(maxlen=50)
        }

        # Open positions
        self.positions = {}

        # Trading params
        self.leverage = 50
        self.position_pct = 0.40  # Use 40% of balance per trade
        self.tp_pct = 0.01  # 1% take profit
        self.sl_pct = 0.005  # 0.5% stop loss
        self.max_hold_seconds = 1800  # 30 min max
        self.min_velocity = 0.002  # 0.2% move minimum

        # Stats
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0

        print("🔥 SIMPLE MONEY MAKER INITIALIZED")
        print(f"   Leverage: {self.leverage}x FIXED")
        print(f"   Position size: {self.position_pct*100}% of balance")
        print(f"   TP: {self.tp_pct*100}% | SL: {self.sl_pct*100}%")
        print(f"   Symbols: {list(self.prices.keys())}")
        print(f"   NO FILTERS. NO GATES. JUST EXECUTION.\n")

    def on_price_update(self, symbol, price):
        """Store price updates."""
        self.prices[symbol].append(float(price))

    def calculate_velocity(self, symbol):
        """Calculate velocity = % change over last 10 prices."""
        prices = self.prices[symbol]
        if len(prices) < 10:
            return 0

        velocity = (prices[-1] - prices[-10]) / prices[-10]
        return velocity

    def generate_signal(self, symbol):
        """
        SIMPLE SIGNAL:
        - velocity > 0.2% = BUY
        - velocity < -0.2% = SELL
        - else = NOTHING

        NO filters. NO gates. NO academic bullshit.
        """
        velocity = self.calculate_velocity(symbol)

        if abs(velocity) < self.min_velocity:
            return None

        direction = "Buy" if velocity > 0 else "Sell"

        return {
            "symbol": symbol,
            "direction": direction,
            "velocity": velocity,
            "price": self.prices[symbol][-1]
        }

    def calculate_position_size(self, symbol, price):
        """Fixed leverage, fixed position size %."""
        balance = self.get_balance()

        # 40% of balance as margin
        margin = balance * self.position_pct

        # Notional = margin * leverage
        notional = margin * self.leverage

        # Quantity
        qty = notional / price

        # Round to exchange requirements
        if symbol == "BTCUSDT":
            qty = round(qty, 3)
        elif symbol == "ETHUSDT":
            qty = round(qty, 2)
        else:
            qty = round(qty, 1)

        return qty

    def calculate_tp_sl(self, direction, entry_price):
        """Fixed 1% TP, 0.5% SL."""
        if direction == "Buy":
            tp = entry_price * (1 + self.tp_pct)
            sl = entry_price * (1 - self.sl_pct)
        else:
            tp = entry_price * (1 - self.tp_pct)
            sl = entry_price * (1 + self.sl_pct)

        return tp, sl

    def get_balance(self):
        """Get available balance."""
        try:
            result = self.bybit.get_account_balance()
            return float(result.get('totalAvailableBalance', 0))
        except:
            return 0

    def execute_trade(self, signal):
        """
        EXECUTE IMMEDIATELY.
        No gates. No filters. No checks.

        If it fails, log it and move on.
        """
        symbol = signal["symbol"]
        direction = signal["direction"]
        price = signal["price"]
        velocity = signal["velocity"]

        # Check if position already open
        if symbol in self.positions:
            return

        # Calculate position
        qty = self.calculate_position_size(symbol, price)
        tp, sl = self.calculate_tp_sl(direction, price)

        # Validate minimum
        notional = qty * price
        if notional < 5.0:  # Bybit minimum
            print(f"⚠️  {symbol} notional ${notional:.2f} < $5 minimum - SKIPPED")
            return

        print(f"\n🚀 EXECUTING: {symbol} {direction}")
        print(f"   Price: ${price:.2f} | Velocity: {velocity*100:+.2f}%")
        print(f"   Qty: {qty:.4f} | Notional: ${notional:.2f}")
        print(f"   TP: ${tp:.2f} | SL: ${sl:.2f}")

        # Set leverage
        self.bybit.set_leverage(symbol, self.leverage)

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

                self.trades_executed += 1
                print(f"✅ EXECUTED - Trade #{self.trades_executed}")
                print(f"   Order ID: {order.get('orderId')}\n")
            else:
                print(f"❌ ORDER FAILED\n")

        except Exception as e:
            print(f"❌ EXECUTION ERROR: {e}\n")

    def monitor_positions(self):
        """Close positions that hit max hold time."""
        current_time = time.time()

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            elapsed = current_time - pos["entry_time"]

            # Max hold time reached
            if elapsed > self.max_hold_seconds:
                print(f"⏰ MAX HOLD TIME - Closing {symbol}")
                self.close_position(symbol, "max_hold")

    def close_position(self, symbol, reason):
        """Manually close position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Reverse direction
        close_side = "Sell" if pos["direction"] == "Buy" else "Buy"

        try:
            order = self.bybit.place_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=pos["qty"]
            )

            if order:
                print(f"🔄 CLOSED {symbol} - Reason: {reason}")
                del self.positions[symbol]
        except Exception as e:
            print(f"❌ CLOSE ERROR {symbol}: {e}")

    def print_status(self):
        """Print trading status."""
        balance = self.get_balance()
        win_rate = (self.wins / max(self.trades_executed, 1)) * 100

        print(f"\n{'='*60}")
        print(f"💰 Balance: ${balance:.2f}")
        print(f"📊 Trades: {self.trades_executed} | Wins: {self.wins} | Losses: {self.losses} | WR: {win_rate:.1f}%")
        print(f"🔥 Open positions: {len(self.positions)}")
        if self.positions:
            for sym, pos in self.positions.items():
                elapsed = time.time() - pos["entry_time"]
                print(f"   {sym}: {pos['direction']} {pos['qty']:.4f} @ ${pos['entry_price']:.2f} ({elapsed/60:.1f}min)")
        print(f"{'='*60}\n")

    def run(self):
        """
        MAIN LOOP:
        1. Check for signals
        2. Execute immediately
        3. Monitor positions
        4. Repeat every 10 seconds
        """
        print("🚀 STARTING SIMPLE MONEY MAKER\n")

        # Connect WebSocket
        self.ws.connect()

        # Wait for price data
        print("⏳ Waiting for price data...")
        while len(self.prices["BTCUSDT"]) < 20:
            time.sleep(1)

        print("✅ Price data ready - STARTING TRADING\n")

        last_status = time.time()

        while True:
            try:
                # Generate and execute signals
                for symbol in self.prices.keys():
                    signal = self.generate_signal(symbol)

                    if signal:
                        self.execute_trade(signal)

                # Monitor positions
                self.monitor_positions()

                # Print status every 60 seconds
                if time.time() - last_status > 60:
                    self.print_status()
                    last_status = time.time()

                # Sleep 10 seconds
                time.sleep(10)

            except KeyboardInterrupt:
                print("\n🛑 STOPPING...")
                self.print_status()
                break
            except Exception as e:
                print(f"❌ ERROR: {e}")
                time.sleep(5)


if __name__ == "__main__":
    trader = SimpleMoneyMaker()
    trader.run()

# 🔥 TURBO NO GATES - BYPASS PLAN

This will be inserted at line 2657 in live_calculus_trader.py, right after tier_config is set.

```python
            # 🔥🔥🔥 TURBO NO GATES MODE - BYPASS ALL 40 EXECUTION GATES 🔥🔥🔥
            # For micro accounts <$50, skip ALL safety checks and execute immediately
            if Config.TURBO_NO_GATES_MODE and available_balance < 50:
                print(f"\n🔥 TURBO NO GATES MODE - BYPASSING ALL 40 EXECUTION GATES")
                print(f"   Balance: ${available_balance:.2f} < $50")
                print(f"   ⚠️  ALL SAFETY CHECKS DISABLED - MAXIMUM AGGRESSION\n")

                # Skip: cooldown, adverse selection, 5-signal filter, tier checks, posterior,
                # balance checks, tradeable checks, margin checks, cadence throttle,
                # position conflicts, concurrent limits, VWAP, acceleration, multi-TF,
                # hedge prevention, consistency, fee protection, TP floor, EV guards, etc.

                # DIRECT TO EXECUTION:
                velocity = signal_dict.get('velocity', 0)
                signal_type = signal_dict['signal_type']
                side = "Buy" if signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG] else "Sell"

                # Position sizing: Fixed 40% of balance, 50x leverage
                margin_pct = 0.40
                leverage = 50

                margin_required = available_balance * margin_pct
                notional = margin_required * leverage
                qty = notional / current_price

                # Round to exchange requirements
                specs = self._get_instrument_specs(symbol)
                if specs:
                    qty_step = specs.get('qty_step', 0.01)
                    min_qty = specs.get('min_qty', 0.01)
                    if qty_step > 0:
                        qty = self._round_quantity_to_step(qty, qty_step)
                    if qty < min_qty:
                        qty = min_qty

                # TP/SL: Fixed 1% TP, 0.5% SL
                if side == "Buy":
                    tp = current_price * 1.01
                    sl = current_price * 0.995
                else:
                    tp = current_price * 0.99
                    sl = current_price * 1.005

                # Check minimum order value ($5 Bybit minimum)
                order_notional = qty * current_price
                if order_notional < 5.0:
                    print(f"⚠️  Order ${order_notional:.2f} < $5 minimum - SKIPPING {symbol}\n")
                    return

                print(f"\n{'='*70}")
                print(f"🚀 TURBO EXECUTION: {symbol}")
                print(f"{'='*70}")
                print(f"📊 Side: {side} | Qty: {qty:.6f} @ ${current_price:.2f}")
                print(f"💰 Notional: ${order_notional:.2f} | Leverage: {leverage}x")
                print(f"🎯 TP: ${tp:.2f} (+{((tp/current_price)-1)*100:.2f}%) | SL: ${sl:.2f} ({((sl/current_price)-1)*100:.2f}%)")
                print(f"{'='*70}\n")

                # Set leverage
                try:
                    self.bybit_client.set_leverage(symbol, leverage)
                    logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to set leverage: {e}")

                # Execute order
                try:
                    order_result = self.bybit_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="Market",
                        qty=qty,
                        take_profit=tp,
                        stop_loss=sl
                    )

                    if order_result:
                        print(f"✅ TURBO TRADE EXECUTED")
                        print(f"   Order ID: {order_result.get('orderId', 'N/A')}")
                        print(f"   Status: {order_result.get('status', 'Unknown')}\n")

                        # Track position
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'side': side,
                            'quantity': qty,
                            'entry_price': current_price,
                            'take_profit': tp,
                            'stop_loss': sl,
                            'leverage_used': leverage,
                            'entry_time': current_time,
                            'signal_type': signal_type.name
                        }

                        # Update state
                        state.position_info = self.positions[symbol]
                        state.last_execution_time = current_time
                        self.last_trade_time[symbol] = current_time

                        logger.info(f"🚀 TURBO: {symbol} {side} {qty:.6f} @ ${current_price:.2f}")
                    else:
                        print(f"❌ ORDER FAILED\n")
                        logger.error(f"Order placement failed for {symbol}")

                except Exception as e:
                    print(f"❌ EXECUTION ERROR: {e}\n")
                    logger.error(f"Turbo execution error for {symbol}: {e}")

                return  # Exit function - bypass all gates below
            # 🔥🔥🔥 END TURBO NO GATES MODE 🔥🔥🔥
```

This will be inserted right after line 2656 (tier_name assignment).
If TURBO_NO_GATES_MODE=true and balance < $50, it will:
1. Calculate position (40% of balance, 50x leverage)
2. Calculate TP/SL (fixed 1% / 0.5%)
3. Execute immediately
4. Return (bypass all 40 gates)

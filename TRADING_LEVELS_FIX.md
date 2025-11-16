# Trading Levels Variable Error - FIXED ✅

## Problem

```
ERROR: Error executing trade for ETHUSDT: cannot access local variable 'trading_levels' where it is not associated with a value
```

## Root Cause

Code was trying to use `trading_levels` variable **BEFORE** it was defined:

**Line 2297** (OLD):
```python
tp_price = trading_levels.take_profit  # ❌ trading_levels not defined yet!
sl_price = trading_levels.stop_loss
```

**Line 2517** (later in code):
```python
trading_levels = self.risk_manager.calculate_dynamic_tp_sl(...)  # ✅ Defined here
```

The liquidation buffer check code was inserted in the wrong place in the execution flow.

## Solution

### 1. **Removed Misplaced Code** (Lines 2284-2315)

Removed the liquidation buffer and SL enforcement code from BEFORE `trading_levels` is defined.

### 2. **Added Code in Correct Location** (Lines 2566-2597)

Moved the same code to AFTER `trading_levels` is created, right before it's used:

```python
# trading_levels is now defined (line 2517)

# Show what the risk manager calculated
if self._should_log_ev_debug():
    logger.info(...)

# ✅ NOW we can use trading_levels safely
# Liquidation buffer check + tight SL enforcement for high leverage
liq_component = (1.0 / max(position_size.leverage_used, 1e-6)) + float(getattr(Config, "MAINTENANCE_MARGIN_RATE", 0.005))
liq_buffer_floor = float(getattr(Config, "MIN_LIQUIDATION_BUFFER_PCT", 0.012))

if liq_distance_pct < liq_buffer_floor:
    self._record_signal_block(state, "liq_buffer", ...)
    return

# Enforce SL <= half liq distance and <=1%
tp_price = trading_levels.take_profit  # ✅ Works now!
sl_price = trading_levels.stop_loss
# ... rest of validation ...
trading_levels.stop_loss = sl_price
trading_levels.risk_reward_ratio = tp_offset / sl_offset

# Continue with normal flow
take_profit = trading_levels.take_profit
stop_loss = trading_levels.stop_loss
```

## What the Code Does

This code:
1. **Checks liquidation buffer** - Ensures stop loss won't trigger liquidation
2. **Enforces tight SL** - For high leverage (25-50x), keeps SL within safe limits
3. **Validates TP/SL** - Makes sure risk levels are appropriate

It needs to access `trading_levels` which contains the calculated TP/SL from risk manager.

## Files Modified

**`live_calculus_trader.py`**:
- Lines 2284-2315: ❌ Removed (misplaced code)
- Lines 2566-2597: ✅ Added (correct location)

## Why It Happened

The liquidation buffer check was probably copy-pasted or inserted during a refactoring and ended up in the wrong place in the execution flow.

## How to Test

Run the trading bot - it should now:
1. ✅ Generate signals without crashing
2. ✅ Calculate trading levels properly
3. ✅ Apply liquidation buffer checks correctly
4. ✅ Execute trades successfully

## Related Fixes

This is part of the complete error elimination:

1. ✅ **Kalman filter NoneType** - Fixed in `cpp_bridge_working.py`
2. ✅ **Quantitative models NoneType** - Fixed in `quantitative_models.py`
3. ✅ **Safe unpacking** - Fixed in `live_calculus_trader.py`
4. ✅ **Trading levels variable** - Fixed in `live_calculus_trader.py` (THIS FIX)

## Status

**FIXED** ✅

The error will not occur again because:
- `trading_levels` is now only accessed AFTER it's defined
- Code execution flow is correct
- Liquidation checks happen at the right time
